"""Create reproducible custom patient-disjoint splits, not the NIH official split."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'data/raw/Data_Entry_2017.csv'
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = ['Image Index', 'Finding Labels', 'Patient ID']
    if any(column not in df for column in required):
        raise ValueError('Required metadata columns are missing.')
    if df[required].apply(lambda col: col.str.strip().eq('')).any().any():
        raise ValueError('Resolve missing required metadata before splitting.')
    if df['Image Index'].duplicated().any():
        raise ValueError('Resolve duplicate image names before splitting.')
    df['pneumonia'] = df['Finding Labels'].str.split('|', regex=False).map(lambda x: int('Pneumonia' in x))
    patients = df.groupby('Patient ID', as_index=False).agg(has_pneumonia=('pneumonia', 'max'))
    train, heldout = train_test_split(patients, test_size=0.30, random_state=42, stratify=patients.has_pneumonia)
    val, test = train_test_split(heldout, test_size=0.50, random_state=42, stratify=heldout.has_pneumonia)
    cohorts = {'train': train, 'val': val, 'test': test}
    ids = {name: set(cohort['Patient ID']) for name, cohort in cohorts.items()}
    overlaps = {f'{a}_{b}': len(ids[a] & ids[b]) for a, b in [('train','val'), ('train','test'), ('val','test')]}
    if any(overlaps.values()) or set.union(*ids.values()) != set(patients['Patient ID']):
        raise RuntimeError('Patient disjointness or coverage verification failed.')
    out = root / 'data/processed/splits'
    if out.exists():
        raise FileExistsError('Split directory already exists. Review existing cohorts before changing them.')
    rows = []
    manifests = {}
    for name, cohort in cohorts.items():
        manifest = df.loc[df['Patient ID'].isin(ids[name]), required + ['pneumonia']].sort_values('Image Index')
        if set(manifest.pneumonia.unique()) != {0, 1}:
            raise RuntimeError(f'{name} does not contain both target classes.')
        manifests[name] = manifest
        rows.append({'split':name, 'patients':len(cohort), 'patients_with_pneumonia_label':int(cohort.has_pneumonia.sum()), 'images':len(manifest), 'positive_images':int(manifest.pneumonia.sum()), 'positive_image_percent':round(manifest.pneumonia.mean()*100,4)})
    if sum(len(x) for x in manifests.values()) != len(df):
        raise RuntimeError('Image coverage check failed.')
    out.mkdir(parents=True)
    for name, manifest in manifests.items():
        manifest.to_csv(out / f'{name}.csv', index=False)
    metrics = root / 'outputs/metrics/dataset_qc'
    pd.DataFrame(rows).to_csv(metrics / 'split_summary.csv', index=False)
    checks = {'method':'custom patient-level 70/15/15 split; not NIH official split', 'seed':42, 'stratification':'Patient has at least one Pneumonia-labeled image', 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'patient_overlap':overlaps, 'all_metadata_rows_assigned_once':True, 'both_image_classes_in_each_split':True, 'image_files_checked':False, 'note':'Do not rerun or rebalance held-out cohorts based on model results. Image duplicate checks remain pending.'}
    (metrics / 'split_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
