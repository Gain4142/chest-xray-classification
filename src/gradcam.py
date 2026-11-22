"""Grad-CAM for the selected ViT, sampled from each test confusion group."""
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

from src.train import ROOT
from src.vit_model import ViTBinaryClassifier


def main():
    torch.set_num_threads(4)
    run = ROOT / 'outputs/models/ViT_baseline_seed42_20261115'
    checkpoint_path = run / 'best.pt'
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    config = checkpoint['config']
    cutoff = json.loads((run / 'threshold_analysis/metrics.json').read_text())['threshold']
    processor = ViTImageProcessor.from_pretrained(run, local_files_only=True)
    architecture = ViTConfig.from_pretrained(
        config['pretrained_model'], revision=config['revision'],
        cache_dir=ROOT / 'outputs/models/hf_cache', local_files_only=True, num_labels=1,
    )
    model = ViTBinaryClassifier(ViTForImageClassification(architecture), False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.requires_grad_(False)
    model.eval()
    table = pd.read_csv(run / 'test_evaluation/predictions.csv', dtype={'Patient ID': str})
    table['prediction'] = (table.score >= cutoff).astype(int)
    table['group'] = np.select([
        (table.pneumonia == 1) & (table.prediction == 1),
        (table.pneumonia == 0) & (table.prediction == 0),
        (table.pneumonia == 0) & (table.prediction == 1),
    ], ['TP', 'TN', 'FP'], default='FN')
    samples = []
    seen = set()
    for group in ('TP', 'TN', 'FP', 'FN'):
        candidates = table[table.group == group].sample(frac=1, random_state=42)
        candidates = candidates[~candidates['Patient ID'].isin(seen)]
        chosen = candidates.drop_duplicates('Patient ID').head(2)
        if len(chosen) != 2:
            raise ValueError(f'Insufficient unique patients for {group}')
        seen.update(chosen['Patient ID'])
        samples.append(chosen)
    cases = pd.concat(samples).reset_index(drop=True)
    directory = ROOT / 'outputs/figures/gradcam'
    metrics = ROOT / 'outputs/metrics/gradcam'
    directory.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    records = []
    captured = {}

    def capture(module, inputs, output):
        captured['activation'] = output

    # Patch tokens must influence the CLS token through the final attention block.
    handle = model.network.vit.encoder.layer[-1].layernorm_before.register_forward_hook(capture)
    try:
        for group in ('TP', 'TN', 'FP', 'FN'):
            selected = cases[cases.group == group]
            fig, axes = plt.subplots(2, 3, figsize=(10, 7), layout='constrained')
            for index, (_, row) in enumerate(selected.iterrows()):
                with Image.open(ROOT / 'data/raw/images' / row['Image Index']) as source:
                    image = source.convert('RGB')
                    inputs = processor(image, return_tensors='pt')['pixel_values']
                    display = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR)) / 255
                with torch.no_grad():
                    reference = model(inputs)
                captured.clear()
                logits = model(inputs.requires_grad_(True))
                activation = captured['activation']
                gradient = torch.autograd.grad(logits.sum(), activation)[0]
                torch.testing.assert_close(reference, logits)
                score = float(logits.detach().sigmoid().item())
                if abs(score - row.score) > 1e-5:
                    raise ValueError('Score differs from saved test inference')
                tokens = activation[:, 1:, :]
                gradients = gradient[:, 1:, :]
                assert tokens.shape[1] == 196 and gradients.abs().sum() > 0
                weights = gradients.mean(dim=1, keepdim=True)
                raw = F.relu((weights * tokens).sum(dim=2)).reshape(1, 1, 14, 14)
                heat = F.interpolate(raw, (224, 224), mode='bilinear', align_corners=False)[0, 0].detach().numpy()
                maximum = float(heat.max())
                if maximum > 0:
                    heat /= maximum
                assert np.isfinite(heat).all()
                axes[index, 0].imshow(display)
                axes[index, 1].imshow(heat, cmap='inferno', vmin=0, vmax=1)
                axes[index, 2].imshow(display)
                overlay = axes[index, 2].imshow(heat, cmap='inferno', vmin=0, vmax=1, alpha=.45)
                titles = [f"Label {row.pneumonia} | predicted {row.prediction}",
                          'Positive-score Grad-CAM', f'Score {score:.4f}']
                for ax, title in zip(axes[index], titles):
                    ax.set_title(title, fontsize=10)
                    ax.axis('off')
                axes[index, 0].set_xlabel(row['Image Index'])
                records.append({**row.to_dict(), 'recomputed_score': score,
                                'cam_max_before_normalization': maximum})
                print(f'{group}: {row["Image Index"]}, score={score:.6f}, CAM max={maximum:.6g}', flush=True)
                del activation, gradient, tokens, gradients, logits
                captured.clear()
            fig.suptitle(f'ViT test examples | {group} | threshold {cutoff:.4f}')
            fig.colorbar(overlay, ax=axes[:, 2].tolist(), shrink=.6, label='Within-image relative contribution')
            fig.savefig(directory / f'vit_test_{group}.png', dpi=150)
            plt.close(fig)
    finally:
        handle.remove()
    pd.DataFrame(records).to_csv(metrics / 'vit_test_cases.csv', index=False)
    (metrics / 'vit_method.json').write_text(json.dumps({
        'model': config['pretrained_model'], 'checkpoint_epoch': checkpoint['epoch'],
        'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'threshold': cutoff, 'cohort': 'test',
        'selection': 'seed 42, 2 per TP/TN/FP/FN, 8 distinct patients; qualitative examples',
        'layer': 'vit.encoder.layer[-1].layernorm_before',
        'target': 'positive logit for all groups',
        'method': 'drop CLS token; spatial mean gradient per channel; weighted activations; ReLU; 14x14 to 224x224 bilinear',
        'normalization': 'per image maximum; zero maps retained',
        'checks': 'logits match no-grad forward; scores match saved test within 1e-5; nonzero patch gradients; finite maps',
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
