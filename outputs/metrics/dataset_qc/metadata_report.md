# NIH ChestX-ray14: dataset selection and initial metadata QC

Date: 2026-10-29

## Decision
Use NIH ChestX-ray14 to stay aligned with the course and inspect patient-level leakage. This is a portfolio research task predicting the presence of the Pneumonia metadata label. Source labels are report-derived and are not confirmed clinical diagnoses.

Sources: https://www.kaggle.com/datasets/nih-chest-xrays/data ; https://nihcc.app.box.com/v/ChestXray-NIHCC ; https://arxiv.org/abs/1705.02315
Before image acquisition, record the selected release and its applicable usage terms. Do not commit raw data or redistribute images.

## Observed results
- Metadata rows: 112,120; patients: 30,805.
- Pneumonia label present: 1,431 (1.2763%).
- Pneumonia label absent: 110,689. This group includes other findings and is not a normal-only group.
- Patients with at least one Pneumonia-labeled image: 1,008.
- Patients with multiple images: 13,302.
- Median / maximum images per patient: 1.0 / 184.
- Pneumonia alone / with other findings: 322 / 1109.
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
.\.venv\Scripts\python.exe scripts\inspect_dataset.py
.\.venv\Scripts\python.exe scripts\summarize_metadata.py
```

Metadata inspection is complete for the checks implemented. This report covers metadata only; current image/split checks are recorded separately in the QC outputs.
