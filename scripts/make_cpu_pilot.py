"""Prepare a CPU pilot manifest; original patient splits remain unchanged."""
from pathlib import Path
import json
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parents[1]
source = root/'data/processed/splits'
out = root/'data/processed/cpu_pilot'
if out.exists():
    raise FileExistsError('Pilot manifests already exist; review before changing cohorts.')
frames = {name:pd.read_csv(source/f'{name}.csv', dtype={'Patient ID':str}) for name in ['train','val','test']}
selected = {}
positive = frames['train'].query('pneumonia == 1')
negative = frames['train'].query('pneumonia == 0').sample(n=min(3*len(positive), int((frames['train'].pneumonia == 0).sum())), random_state=42)
selected['train'] = pd.concat([positive,negative]).sort_values('Image Index')
for name in ['val','test']:
    df = frames[name]
    patients = df.groupby('Patient ID', as_index=False).agg(has_pneumonia=('pneumonia','max'))
    cohort, _ = train_test_split(patients, train_size=1000, random_state=42, stratify=patients.has_pneumonia)
    selected[name] = df.loc[df['Patient ID'].isin(cohort['Patient ID'])].sort_values('Image Index')
ids = {name:set(df['Patient ID']) for name,df in selected.items()}
checks = {f'{a}_{b}':len(ids[a]&ids[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
if any(checks.values()):
    raise RuntimeError('Patient overlap found.')
rows = []
for name,df in selected.items():
    if set(df.pneumonia.unique()) != {0,1} or df['Image Index'].duplicated().any():
        raise RuntimeError('Class coverage or filename uniqueness failed.')
    if not set(df['Image Index']).issubset(set(frames[name]['Image Index'])):
        raise RuntimeError('Pilot does not match parent split.')
    rows.append({'split':name, 'patients':df['Patient ID'].nunique(), 'images':len(df), 'positive_images':int(df.pneumonia.sum()), 'negative_images':int((df.pneumonia==0).sum()), 'positive_percent':round(df.pneumonia.mean()*100,3)})
out.mkdir(parents=True)
for name,df in selected.items():
    df.to_csv(out/f'{name}.csv', index=False)
metrics = root/'outputs/metrics/dataset_qc'
pd.DataFrame(rows).to_csv(metrics/'cpu_pilot_summary.csv', index=False)
plan = {'seed':42, 'purpose':'Initial CPU feasibility experiment, not final benchmark', 'training':'All train positive images; random 3:1 negative-to-positive image sample', 'validation_and_test':'1000 patients each, stratified by any Pneumonia label; all images for selected patients; no image class balancing', 'patient_overlaps':checks, 'parent_manifest_sha256':{name:hashlib.sha256((source/f'{name}.csv').read_bytes()).hexdigest() for name in frames}, 'limitations':['Training prevalence is deliberately changed; raw probabilities are not population risks.', 'Patient stratification does not guarantee equal image prevalence.', 'Small held-out positive counts give uncertain sensitivity and AUPRC estimates.', 'Do not select a different seed/cohort after inspecting model performance.', 'No model evaluation or full pilot image acquisition has occurred.']}
(metrics/'cpu_pilot_plan.json').write_text(json.dumps(plan,indent=2), encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
print('Patient overlaps:', checks)
