"""Extend a ViT run from its last completed epoch into a separate output directory."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.dataset import ChestXrayDataset
from src.preprocessing import ApplyCLAHE
from src.train import ROOT, run_epoch, save_checkpoint
from src.vit_model import load_vit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        parser.error('Use a distinct output directory.')
    # Retry an interrupted extension from its own last completed epoch.
    origin = output if (output / 'last.pt').is_file() else source
    checkpoint = torch.load(origin / 'last.pt', weights_only=True, map_location='cpu')
    config = checkpoint['config'].copy()
    if config.get('model') != 'vit':
        parser.error('Only ViT extensions are supported.')
    if args.epochs <= checkpoint['epoch']:
        parser.error('Target epochs must exceed the saved epoch.')
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('Source run requires a CUDA GPU.')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    manifest_dir = ROOT / 'data/processed/reduced_train_full_eval_qc'
    for split, digest in config['manifest_sha256'].items():
        if hashlib.sha256((manifest_dir / f'{split}.csv').read_bytes()).hexdigest() != digest:
            raise ValueError('Manifest changed.')
    model, processor = load_vit(ROOT / 'outputs/models/hf_cache', config['vit_last_block'])
    clahe = ApplyCLAHE()

    def transform(image):
        if config['clahe']:
            image = clahe(image)
        return processor(image.convert('RGB'), return_tensors='pt')['pixel_values'][0]

    train = ChestXrayDataset(manifest_dir / 'train.csv', ROOT / 'data/raw/images', transform)
    if config['smoke']:
        indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(config['seed']))[:32].tolist()
        train = Subset(train, indices)
        val = train
    else:
        val = ChestXrayDataset(manifest_dir / 'val.csv', ROOT / 'data/raw/images', transform)
        assert not set(train.table['Patient ID']) & set(val.table['Patient ID'])
    assert len(train) == config['train_images'] and len(val) == config['val_images']
    generator = torch.Generator()
    generator.set_state(checkpoint['loader_rng_state'])
    train_loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config['lr'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    torch.set_rng_state(checkpoint['torch_rng_state'])
    if device.type == 'cuda' and 'cuda_rng_states' in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_states'])
    history = pd.read_csv(origin / 'history.csv')
    history = history[history.epoch <= checkpoint['epoch']].to_dict('records')
    if [int(row['epoch']) for row in history] != list(range(1, checkpoint['epoch'] + 1)):
        raise ValueError('Incomplete source history.')
    best_checkpoint = torch.load(origin / 'best.pt', weights_only=True, map_location='cpu')
    best = best_checkpoint['metrics']['val_auroc']
    start_epoch = checkpoint['epoch'] + 1
    if output.exists() and origin != output:
        parser.error('Output already exists without a resumable checkpoint.')
    output.mkdir(parents=True, exist_ok=True)
    config.update(epochs=args.epochs, run_name=output.name, resumed_from=str(source),
                  continuation_start_epoch=start_epoch)
    processor.save_pretrained(output)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    if origin != output:
        save_checkpoint(output / 'best.pt', best_checkpoint)
    del checkpoint, best_checkpoint
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        print(f'Epoch {epoch}/{args.epochs}, CLAHE={config["clahe"]}', flush=True)
        training = run_epoch(model, train_loader, criterion, optimizer, type(model).train_selected_layers)
        validation = run_epoch(model, val_loader, criterion)
        row = dict(epoch=epoch, seconds=time.perf_counter() - started,
                   **{f'train_{key}': value for key, value in training.items()},
                   **{f'val_{key}': value for key, value in validation.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        checkpoint = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                          epoch=epoch, config=config, metrics=row,
                          torch_rng_state=torch.get_rng_state(), loader_rng_state=generator.get_state())
        if device.type == 'cuda':
            checkpoint['cuda_rng_states'] = torch.cuda.get_rng_state_all()
        if row['val_auroc'] is not None and row['val_auroc'] > best:
            best = row['val_auroc']
            save_checkpoint(output / 'best.pt', checkpoint)
        save_checkpoint(output / 'last.pt', checkpoint)
        print(json.dumps(row), flush=True)
    (output / 'completed.json').write_text(json.dumps(dict(epochs=args.epochs, best_auroc=best)), encoding='utf-8')
    print(f'Completed {output}', flush=True)


if __name__ == '__main__':
    main()
