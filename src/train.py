"""CPU DenseNet and ViT training. Run from the repository with python -m src.train."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.dataset import ChestXrayDataset
from src.model import build_model, set_classifier_training
from src.preprocessing import ApplyCLAHE, build_transform

ROOT = Path(__file__).resolve().parents[1]


def run_epoch(model, loader, criterion, optimizer=None, set_training=set_classifier_training):
    if optimizer is None:
        model.eval()
    else:
        set_training(model)
    loss_sum, targets_all, scores_all = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for step, (images, targets) in enumerate(loader, 1):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            device = next(model.parameters()).device
            images, targets = images.to(device), targets.to(device)
            logits = model(images).squeeze(1)
            loss = criterion(logits, targets)
            if logits.shape != targets.shape or not torch.isfinite(loss):
                raise RuntimeError('Invalid model output or non-finite loss')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(targets)
            targets_all.extend(targets.detach().cpu().tolist())
            scores_all.extend(logits.detach().sigmoid().cpu().tolist())
            if step % 100 == 0:
                print(f'  batch {step}/{len(loader)}', flush=True)
    return {
        'loss': loss_sum / len(targets_all),
        'auroc': float(roc_auc_score(targets_all, scores_all)) if len(set(targets_all)) == 2 else None,
        'auprc': float(average_precision_score(targets_all, scores_all)) if len(set(targets_all)) == 2 else None,
        'images': len(targets_all),
    }


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('densenet', 'vit'), default='densenet')
    parser.add_argument('--run-name')
    parser.add_argument('--vit-last-block', action='store_true', help='Fine-tune the last ViT block and final LayerNorm')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--optimizer', choices=('adam','adamw'), default='adam')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--pos-weight', type=float, help='Positive BCE class weight; omit for unweighted BCE')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--selection-metric', choices=('auroc', 'auprc'), default='auroc',
                        help='Validation metric used to select best.pt')
    parser.add_argument('--smoke', action='store_true', help='Use only 32 training images for execution checks')
    parser.add_argument('--output-root', help='Checkpoint parent directory; defaults to outputs/models')
    parser.add_argument('--clahe', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--resume', action='store_true', help='Resume this run from last.pt without changing its conditions')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA GPU is unavailable')
    device = torch.device(args.device)
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    is_vit = args.model == 'vit'
    if args.vit_last_block and not is_vit:
        parser.error('--vit-last-block requires --model vit')
    if args.lr is None:
        args.lr = 1e-4 if args.vit_last_block else 1e-3
    if args.run_name is None:
        args.run_name = f'ViT_baseline_seed{args.seed}' if is_vit else f'B0_baseline_seed{args.seed}'
    if min(args.epochs, args.batch_size, args.threads) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('Epochs, batch size, threads and learning rate must be positive')
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('weight-decay must be finite and nonnegative')
    if args.pos_weight is not None and (not np.isfinite(args.pos_weight) or args.pos_weight <= 0):
        parser.error('pos-weight must be finite and positive')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('Run name must be a single directory name')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    manifest_dir = ROOT / 'data/processed/reduced_train_full_eval_qc'
    set_training = set_classifier_training
    model_config = {}
    if is_vit:
        from src.vit_model import MODEL_ID, MODEL_REVISION, load_vit

        model, processor = load_vit(ROOT / 'outputs/models/hf_cache', args.vit_last_block)

        clahe = ApplyCLAHE()
        def transform(image):
            if args.clahe:
                image = clahe(image)
            return processor(image.convert('RGB'), return_tensors='pt')['pixel_values'][0]

        set_training = type(model).train_selected_layers
        model_config = dict(pretrained_model=MODEL_ID, revision=MODEL_REVISION)

    else:
        transform = build_transform(use_clahe=args.clahe)

    train = ChestXrayDataset(manifest_dir / 'train.csv', ROOT / 'data/raw/images', transform)
    hashes = {'train': hashlib.sha256((manifest_dir / 'train.csv').read_bytes()).hexdigest()}
    if args.smoke:
        # Execution checks never open validation or test images.
        indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed))[:32].tolist()
        train = Subset(train, indices)
        val = train
    else:
        val = ChestXrayDataset(manifest_dir / 'val.csv', ROOT / 'data/raw/images', transform)
        if set(train.table['Patient ID']) & set(val.table['Patient ID']):
            raise ValueError('Training and validation patients overlap')
        hashes['val'] = hashlib.sha256((manifest_dir / 'val.csv').read_bytes()).hexdigest()
    output_root = Path(args.output_root) if args.output_root else ROOT / 'outputs/models'
    output = output_root / args.run_name
    output.mkdir(parents=True, exist_ok=args.resume)
    config = dict(vars(args), manifest_sha256=hashes, augmentation=False, clahe=args.clahe,
                  clahe_parameters={'clip_limit': 2.0, 'tile_grid_size': [8, 8]} if args.clahe else None,
                  selection=f'validation {args.selection_metric.upper()}, earliest epoch on ties', device=args.device,
                  torch_version=str(torch.__version__), train_images=len(train), val_images=len(val),
                  smoke_metrics_are_not_validation=args.smoke)
    config.update(model_config)
    previous = None
    if args.resume:
        previous = torch.load(output / 'last.pt', weights_only=True, map_location='cpu')
        critical = ('model', 'batch_size', 'lr', 'seed', 'smoke', 'manifest_sha256', 'clahe',
                    'augmentation', 'vit_last_block', 'device', 'torch_version', 'pretrained_model', 'revision')
        for key in critical:
            if previous['config'].get(key) != config.get(key):
                parser.error(f'Cannot resume with a changed setting: {key}')
        for key, default in {'optimizer':'adam', 'weight_decay':0.0, 'pos_weight':None}.items():
            if previous['config'].get(key, default) != config[key]:
                parser.error(f'Cannot resume with a changed setting: {key}')
        if previous['config'].get('selection_metric', 'auroc') != args.selection_metric:
            parser.error('Cannot change checkpoint selection while resuming')
        if previous['epoch'] > args.epochs:
            parser.error('Target epochs precede the checkpoint')
    if is_vit:
        processor.save_pretrained(output)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if not is_vit:
        model = build_model()
    model.to(device)
    optimizer_class = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    optimizer = optimizer_class((p for p in model.parameters() if p.requires_grad),
                                lr=args.lr, weight_decay=args.weight_decay)
    positive_weight = torch.tensor([args.pos_weight], device=device) if args.pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    history, best, start_epoch = [], -float('inf'), 1
    if previous is not None:
        model.load_state_dict(previous['model_state_dict'])
        optimizer.load_state_dict(previous['optimizer_state_dict'])
        generator.set_state(previous['loader_rng_state'])
        torch.set_rng_state(previous['torch_rng_state'])
        if device.type == 'cuda' and 'cuda_rng_states' in previous:
            torch.cuda.set_rng_state_all(previous['cuda_rng_states'])
        history_frame = pd.read_csv(output / 'history.csv')
        history = history_frame[history_frame.epoch <= previous['epoch']].to_dict('records')
        if [int(row['epoch']) for row in history] != list(range(1, previous['epoch'] + 1)):
            parser.error('Checkpoint and history do not agree')
        best_path = output / 'best.pt'
        if best_path.exists():
            best_checkpoint = torch.load(best_path, weights_only=True, map_location='cpu')
            if best_checkpoint['epoch'] > previous['epoch']:
                parser.error('Best checkpoint is newer than the resume checkpoint')
            best = best_checkpoint['metrics'][f'val_{args.selection_metric}']
            del best_checkpoint
        # Recover a save interrupted between writing last.pt and best.pt.
        if previous['metrics'][f'val_{args.selection_metric}'] > best:
            best = previous['metrics'][f'val_{args.selection_metric}']
            save_checkpoint(best_path, previous)
        start_epoch = previous['epoch'] + 1
        print(f'Resuming {output.name} at epoch {start_epoch}/{args.epochs}', flush=True)
        del previous
    if args.smoke:
        trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
        frozen = {name: value.clone() for name, value in model.state_dict().items()
                  if name not in trainable_names}
        if args.vit_last_block:
            block_parameter = model.network.vit.encoder.layer[-1].output.dense.weight
            block_before = block_parameter.detach().clone()
    classifier_before = model.classifier.weight.detach().clone()
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        print(f'Epoch {epoch}/{args.epochs}: training {len(train)} images', flush=True)
        training = run_epoch(model, train_loader, criterion, optimizer, set_training)
        print(f'Epoch {epoch}: evaluating {len(val)} images', flush=True)
        validation = run_epoch(model, val_loader, criterion)
        row = {'epoch': epoch, 'seconds': time.perf_counter() - started,
               **{f'train_{k}': v for k, v in training.items()},
               **{f'val_{k}': v for k, v in validation.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch, 'config': config, 'metrics': row,
                      'torch_rng_state': torch.get_rng_state(), 'loader_rng_state': generator.get_state()}
        if device.type == 'cuda':
            checkpoint['cuda_rng_states'] = torch.cuda.get_rng_state_all()
        save_checkpoint(output / 'last.pt', checkpoint)
        if validation[args.selection_metric] is not None and validation[args.selection_metric] > best:
            best = validation[args.selection_metric]
            save_checkpoint(output / 'best.pt', checkpoint)
        print(json.dumps(row), flush=True)
    if args.smoke and start_epoch <= args.epochs:
        state = model.state_dict()
        assert all(torch.equal(value, state[key]) for key, value in frozen.items())
        if args.vit_last_block:
            assert not torch.equal(block_before, block_parameter)
        del frozen, state
        assert not torch.equal(classifier_before, model.classifier.weight)
        if is_vit:
            from src.vit_model import ViTBinaryClassifier, ViTForImageClassification

            restored = ViTBinaryClassifier(
                ViTForImageClassification(model.network.config), args.vit_last_block
            )
        else:
            restored = build_model(pretrained=False)
        restored.load_state_dict(torch.load(output / 'last.pt', weights_only=True, map_location='cpu')['model_state_dict'])
        restored.to(device)
        restored.eval()
        images, _ = next(iter(val_loader))
        with torch.inference_mode():
            images = images.to(device)
            torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)
        print('SMOKE PASSED: frozen weights/buffers, trainable weights updated, exact checkpoint reload.', flush=True)
    (output / 'completed.json').write_text(json.dumps({
        'epochs': args.epochs, 'selection_metric': args.selection_metric,
        f'best_{args.selection_metric}': best if np.isfinite(best) else None,
    }), encoding='utf-8')
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    main()
