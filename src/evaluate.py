"""Evaluate a completed ViT checkpoint on a fixed cohort."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

from src.dataset import ChestXrayDataset
from src.preprocessing import ApplyCLAHE
from src.train import ROOT
from src.vit_model import ViTBinaryClassifier
from src.vit_model import load_vit, ViTInputTransform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA GPU unavailable')
    device = torch.device(args.device)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if args.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    run = args.run_dir
    if not (run / 'completed.json').is_file():
        parser.error('Choose a completed training run.')
    output = run / f'{args.split}_evaluation'
    if output.exists():
        parser.error(f'Evaluation already exists: {output}')

    checkpoint = torch.load(run / 'best.pt', weights_only=True, map_location='cpu')
    config = checkpoint['config']
    if config['smoke'] or config['model'] != 'vit':
        parser.error('Choose a full ViT training checkpoint.')
    torch.set_num_threads(config['threads'])

    manifests = ROOT / 'data/processed/reduced_train_full_eval_qc'
    cohort_path = manifests / f'{args.split}.csv'
    for split in ('train', 'val'):
        actual = hashlib.sha256((manifests / f'{split}.csv').read_bytes()).hexdigest()
        if actual != config['manifest_sha256'][split]:
            raise ValueError(f'{split} manifest differs from the training run.')

    processor = ViTImageProcessor.from_pretrained(run, local_files_only=True)
    model, _ = load_vit(ROOT / 'outputs/models/hf_cache',
                        config.get('vit_last_block', False), config.get('vit_training'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    transform = ViTInputTransform(processor, config.get('clahe', False))

    cohort = ChestXrayDataset(cohort_path, ROOT / 'data/raw/images', transform)
    for split in ('train', 'val'):
        if split != args.split:
            patients = pd.read_csv(manifests / f'{split}.csv', dtype={'Patient ID': str})['Patient ID']
            if set(patients) & set(cohort.table['Patient ID']):
                raise ValueError(f'{split} and {args.split} patients overlap.')

    loader = DataLoader(cohort, batch_size=config['batch_size'], shuffle=False)
    scores = []
    started = time.perf_counter()
    print(f'Evaluating {len(cohort)} {args.split} images, checkpoint epoch {checkpoint["epoch"]}', flush=True)
    with torch.set_grad_enabled(False):
        for batch, (images, _) in enumerate(loader, 1):
            logits = model(images.to(device)).squeeze(1)
            if not torch.isfinite(logits).all():
                raise ValueError(f'Non-finite scores at batch {batch}.')
            scores.extend(logits.sigmoid().cpu().tolist())
            if batch % 100 == 0:
                print(f'Batch {batch}/{len(loader)}', flush=True)

    rows = cohort.table[['Image Index', 'Patient ID', 'pneumonia']].copy()
    rows['score'] = scores
    targets = rows['pneumonia']
    metrics = {
        'split': args.split,
        'checkpoint_epoch': checkpoint['epoch'],
        'manifest_sha256': hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        'images': len(rows),
        'patients': int(rows['Patient ID'].nunique()),
        'positive_images': int(targets.sum()),
        'auroc': roc_auc_score(targets, scores),
        'average_precision': average_precision_score(targets, scores),
        'seconds': time.perf_counter() - started,
    }
    output.mkdir()
    rows.to_csv(output / 'predictions.csv', index=False)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
