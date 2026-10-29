"""Describe NIH metadata before image acquisition or patient splitting."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / 'data/raw/Data_Entry_2017.csv', dtype=str, keep_default_na=False)
    metrics = root / 'outputs/metrics/dataset_qc'
    figures = root / 'outputs/figures'
    metrics.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tokens = df['Finding Labels'].str.split('|', regex=False)
    positive = tokens.map(lambda x: 'Pneumonia' in x)
    per_patient = df.assign(pneumonia=positive.astype(int)).groupby('Patient ID').agg(
        image_count=('Image Index', 'size'), pneumonia_label_images=('pneumonia', 'sum'))
    patient_stats = {
        'patients_with_pneumonia_label': int(per_patient.pneumonia_label_images.gt(0).sum()),
        'patients_with_multiple_images': int(per_patient.image_count.gt(1).sum()),
        'median_images_per_patient': float(per_patient.image_count.median()),
        'max_images_per_patient': int(per_patient.image_count.max()),
        'pneumonia_only_images': int(df['Finding Labels'].eq('Pneumonia').sum()),
        'pneumonia_with_other_findings': int((positive & df['Finding Labels'].ne('Pneumonia')).sum()),
    }
    (metrics / 'patient_summary.json').write_text(json.dumps(patient_stats, indent=2), encoding='utf-8')
    per_patient.image_count.value_counts().sort_index().rename_axis('images_per_patient').reset_index(name='patient_count').to_csv(metrics / 'patient_image_count_distribution.csv', index=False)
    view = pd.crosstab(df['View Position'].replace('', 'MISSING'), positive).reindex(columns=[False, True], fill_value=0)
    view.columns = ['pneumonia_label_absent', 'pneumonia_label_present']
    view.to_csv(metrics / 'view_position_distribution.csv')
    findings = tokens.explode().value_counts().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
    bars = axes[0].bar(['Label absent', 'Label present'], [(~positive).sum(), positive.sum()], color=['#64748b', '#e87924'])
    axes[0].bar_label(bars, fmt='{:,.0f}', padding=4)
    axes[0].set_ylim(0, len(df)*1.15)
    axes[0].set_title('Pneumonia label distribution')
    axes[0].set_ylabel('Metadata rows')
    axes[1].barh(findings.index, findings.values, color=['#e87924' if x == 'Pneumonia' else '#64748b' for x in findings.index])
    axes[1].set_title('Finding counts (labels can co-occur)')
    axes[1].set_xlabel('Metadata rows containing each label')
    fig.suptitle('NIH ChestX-ray14: metadata inspection')
    fig.savefig(figures / 'metadata_distribution.png', dpi=180)
    plt.close(fig)
    report = f'''# NIH ChestX-ray14: dataset selection and initial metadata QC

Date: 2026-10-29

## Decision
Use NIH ChestX-ray14 to stay aligned with the course and inspect patient-level leakage. This is a portfolio research task predicting the presence of the Pneumonia metadata label. Source labels are report-derived and are not confirmed clinical diagnoses.

Sources: https://www.kaggle.com/datasets/nih-chest-xrays/data ; https://nihcc.app.box.com/v/ChestXray-NIHCC ; https://arxiv.org/abs/1705.02315
Before image acquisition, record the selected release and its applicable usage terms. Do not commit raw data or redistribute images.

## Observed results
- Metadata rows: {len(df):,}; patients: {len(per_patient):,}.
- Pneumonia label present: {int(positive.sum()):,} ({positive.mean()*100:.4f}%).
- Pneumonia label absent: {int((~positive).sum()):,}. This group includes other findings and is not a normal-only group.
- Patients with at least one Pneumonia-labeled image: {patient_stats['patients_with_pneumonia_label']:,}.
- Patients with multiple images: {patient_stats['patients_with_multiple_images']:,}.
- Median / maximum images per patient: {patient_stats['median_images_per_patient']} / {patient_stats['max_images_per_patient']}.
- Pneumonia alone / with other findings: {patient_stats['pneumonia_only_images']} / {patient_stats['pneumonia_with_other_findings']}.
- Basic missingness, duplicate-row and filename-conflict results: see metadata_summary.json.

![Metadata distributions](../outputs/figures/metadata_distribution.png)

## Interpretation and next steps
1. Label imbalance makes accuracy insufficient; include AUPRC, sensitivity, specificity and AUROC. Choose thresholds using validation data only.
2. Split by Patient ID before any class balancing or image sampling. Verify disjoint patient IDs and ensure both classes occur in every split. This metadata-only report does not create or inspect splits; see active_dataset_plan.json for the current cohort.
3. Freeze the held-out evaluation cohort. Do not use test images to select preprocessing, exclusions or thresholds.
4. Plan CPU-feasible image acquisition after inspecting archive organization. A convenience subset from one archive may alter patient and label distributions; compare the acquired subset with this metadata baseline.
5. After acquisition, inspect file existence, decoding, dimensions, formats, exact duplicates and cross-split duplicates. Visual review flags should record quality concerns, not replace source labels with unverified diagnoses.

## Reproduce
Run from the repository root:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\inspect_dataset.py
.\\.venv\\Scripts\\python.exe scripts\\summarize_metadata.py
```

Metadata inspection is complete for the checks implemented. This report covers metadata only; current image/split checks are recorded separately in the QC outputs.
'''
    (metrics / 'metadata_report.md').write_text(report, encoding='utf-8')
    print(json.dumps(patient_stats, indent=2))
    print('Saved metadata figure, summary tables and outputs/metrics/dataset_qc/metadata_report.md')


if __name__ == '__main__':
    main()
