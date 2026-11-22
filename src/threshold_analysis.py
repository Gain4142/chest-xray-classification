"""Select a ViT score cutoff on validation and apply it once to test."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve


def read_predictions(run, split):
    directory = run / f'{split}_evaluation'
    summary = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    rows = pd.read_csv(directory / 'predictions.csv', dtype={'Patient ID': str})
    if (summary['split'] != split or len(rows) != summary['images']
            or rows['pneumonia'].sum() != summary['positive_images']
            or set(rows['pneumonia']) != {0, 1}
            or not rows['score'].between(0, 1).all()):
        raise ValueError(f'Invalid {split} predictions.')
    return rows, summary


def at_threshold(rows, threshold):
    tn, fp, fn, tp = map(int, confusion_matrix(
        rows['pneumonia'], rows['score'] >= threshold, labels=[0, 1]
    ).ravel())
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'precision': tp / (tp + fp) if tp + fp else None,
        'f1': 2 * tp / (2 * tp + fp + fn),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()

    validation, val_summary = read_predictions(args.run_dir, 'val')
    fpr, tpr, thresholds = roc_curve(
        validation['pneumonia'], validation['score'], drop_intermediate=False
    )
    finite = np.isfinite(thresholds)
    curve = pd.DataFrame({
        'threshold': thresholds[finite], 'sensitivity': tpr[finite],
        'specificity': 1 - fpr[finite], 'youden_j': (tpr - fpr)[finite],
    })
    # ROC thresholds descend, so take the highest threshold among numerical ties.
    best = curve[np.isclose(curve['youden_j'], curve['youden_j'].max(),
                            atol=1e-12, rtol=0)].iloc[0]
    threshold = float(best['threshold'])
    result = {
        'selection': 'Maximum validation Youden J; highest threshold on ties',
        'checkpoint_epoch': val_summary['checkpoint_epoch'],
        'threshold': threshold,
        'validation': at_threshold(validation, threshold),
    }

    test, test_summary = read_predictions(args.run_dir, 'test')
    if val_summary['checkpoint_epoch'] != test_summary['checkpoint_epoch']:
        raise ValueError('Validation and test checkpoint epochs differ.')
    if set(validation['Patient ID']) & set(test['Patient ID']):
        raise ValueError('Validation and test patient IDs overlap.')
    result['test'] = at_threshold(test, threshold)
    output = args.run_dir / 'threshold_analysis'
    output.mkdir(exist_ok=False)
    curve.to_csv(output / 'validation_thresholds.csv', index=False)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
