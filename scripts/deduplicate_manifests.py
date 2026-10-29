"""Remove exact pixel duplicates from manifests, preserving downloaded originals."""
from pathlib import Path
import json
import pandas as pd
root=Path(__file__).resolve().parents[1]
metrics=root/'outputs/metrics/dataset_qc'
df=pd.read_csv(metrics/'image_inventory.csv',dtype={'Patient ID':str})
# Fixed rule independent of model performance: retain test, then val, then train.
df['priority']=df['split'].map({'test':0,'val':1,'train':2})
ordered=df.sort_values(['priority','Image Index'])
remove=ordered.duplicated('pixel_sha256',keep='first')
excluded=ordered.loc[remove].copy()
kept=ordered.loc[~remove].copy()
representatives=kept.set_index('pixel_sha256')['Image Index']
excluded['retained_image']=excluded['pixel_sha256'].map(representatives)
excluded['reason']='Exact decoded-pixel duplicate; prefer test > val > train, then filename order'
out=root/'data/processed/reduced_train_full_eval_qc'
if out.exists():
    raise FileExistsError('Clean manifests already exist; do not overwrite silently.')
ids={name:set(kept.loc[kept.split==name,'Patient ID']) for name in ['train','val','test']}
assert all(not ids[a]&ids[b] for a,b in [('train','val'),('train','test'),('val','test')])
assert not kept.pixel_sha256.duplicated().any()
rows=[]
for name in ['train','val','test']:
    group=kept.loc[kept.split==name]
    assert set(group.pneumonia.unique())=={0,1}
    rows.append({'split':name,'patients':group['Patient ID'].nunique(),'images':len(group),'positive_images':int(group.pneumonia.sum()),'excluded_duplicate_images':int((excluded.split==name).sum())})
out.mkdir()
for name in ['train','val','test']:
    kept.loc[kept.split==name,['Image Index','Finding Labels','Patient ID','pneumonia']].sort_values('Image Index').to_csv(out/f'{name}.csv',index=False)
excluded[['Image Index','Patient ID','Finding Labels','pneumonia','split','pixel_sha256','retained_image','reason']].to_csv(metrics/'duplicate_exclusions.csv',index=False)
pd.DataFrame(rows).to_csv(metrics/'active_dataset_summary.csv',index=False)
plan={'active_manifest_directory':'data/processed/reduced_train_full_eval_qc','input_images':len(df),'retained_images':len(kept),'excluded_images':len(excluded),'rule':'Exact pixel deduplication; prefer test > val > train then filename order; applied before model training/evaluation','raw_files_deleted':False,'remaining_exact_pixel_duplicates':0,'patient_overlap_all_pairs':0,'near_duplicate_and_visual_review':'pending'}
(metrics/'active_dataset_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
print(json.dumps(plan,indent=2))
