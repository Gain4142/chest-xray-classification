"""Inspect NIH ChestX-ray14 metadata without modifying the source CSV."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('data/raw/Data_Entry_2017.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/metrics/dataset_qc'))
    args = parser.parse_args()
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    required = ['Image Index', 'Finding Labels', 'Patient ID']
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if df.empty:
        raise ValueError('The metadata CSV has no rows.')

    # Match a complete label token, including images with multiple findings.
    labels = df['Finding Labels'].str.split('|', regex=False)
    pneumonia = labels.map(lambda values: 'Pneumonia' in values).astype(int)
    blank = df[required].apply(lambda column: column.str.strip().eq(''))
    patients = df.loc[~blank['Patient ID'], 'Patient ID']
    duplicate_name = df['Image Index'].duplicated(keep=False) & ~blank['Image Index']
    conflicting = df.groupby('Image Index')['Finding Labels'].transform('nunique').gt(1)

    summary = {
        'source_file': args.csv.name,
        'source_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        'metadata_rows': len(df),
        'unique_image_names': int(df.loc[~blank['Image Index'], 'Image Index'].nunique()),
        'unique_patients': int(patients.nunique()),
        'pneumonia_label_positive': int(pneumonia.sum()),
        'pneumonia_label_negative': int((pneumonia == 0).sum()),
        'positive_percent': round(float(pneumonia.mean() * 100), 4),
        'missing_required_values': {key: int(value) for key, value in blank.sum().items()},
        'duplicate_image_name_rows': int(duplicate_name.sum()),
        'duplicate_full_rows_beyond_first': int(df.duplicated().sum()),
        'image_name_rows_with_conflicting_labels': int(conflicting.sum()),
        'scope': 'Metadata only; image files have not been inspected.',
        'target_definition': '1 = Pneumonia label present; 0 = label absent, not necessarily normal.',
        'split_requirement': 'Split by Patient ID; this metadata inspection does not create splits.',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'metadata_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame({
        'target': [0, 1],
        'definition': ['Pneumonia label absent', 'Pneumonia label present'],
        'count': [int((pneumonia == 0).sum()), int(pneumonia.sum())],
    }).to_csv(args.output_dir / 'class_distribution.csv', index=False)
    labels.explode().value_counts().rename_axis('finding').reset_index(name='image_count').to_csv(
        args.output_dir / 'finding_counts.csv', index=False)
    flags = blank.any(axis=1) | duplicate_name | conflicting
    issues = df.loc[flags, required].copy()
    issues['missing_required_value'] = blank.any(axis=1)[flags]
    issues['duplicate_image_name'] = duplicate_name[flags]
    issues['conflicting_labels_for_image_name'] = conflicting[flags]
    issues.to_csv(args.output_dir / 'metadata_issues.csv', index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Reports saved to: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
