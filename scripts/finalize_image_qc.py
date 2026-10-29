"""Check downloaded file coverage and decoded-pixel duplicates across cohorts."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from PIL import Image

root=Path(__file__).resolve().parents[1]
metrics=root/'outputs/metrics/dataset_qc'
frames=[]
for name in ['train','val','test']:
    df=pd.read_csv(root/f'data/processed/reduced_train_full_eval/{name}.csv',dtype={'Patient ID':str})
    df['split']=name
    frames.append(df)
manifest=pd.concat(frames,ignore_index=True)
qc=pd.read_csv(metrics/'downloaded_image_qc.csv')
if qc.image.duplicated().any():
    raise ValueError('Duplicate QC rows.')
missing=set(manifest['Image Index'])-set(qc.image)
if missing:
    raise ValueError(f'{len(missing)} selected images are missing from download QC.')
joined=manifest.merge(qc,left_on='Image Index',right_on='image',suffixes=('', '_qc'),validate='one_to_one')
if not joined['split'].eq(joined.split_qc).all():
    raise ValueError('QC split mismatch.')
print('Checking decoded-pixel hashes...',flush=True)
from concurrent.futures import ThreadPoolExecutor

def pixel_hash(name):
    with Image.open(root/'data/raw/images'/name) as im:
        return hashlib.sha256(f'{im.mode}:{im.size}:'.encode()+im.tobytes()).hexdigest()

pixels=[]
with ThreadPoolExecutor(max_workers=8) as pool:
    for i,h in enumerate(pool.map(pixel_hash,joined.image)):
        pixels.append(h)
        if (i+1)%5000==0:
            print(f'Pixel hashes: {i+1:,}/{len(joined):,}',flush=True)
joined['pixel_sha256']=pixels
joined.to_csv(metrics/'image_inventory.csv',index=False)
flags=joined.duplicated('pixel_sha256',keep=False)
dup=joined.loc[flags].sort_values('pixel_sha256')
dup.to_csv(metrics/'pixel_duplicate_images.csv',index=False)
cross=dup.groupby('pixel_sha256')['split'].nunique()
summary={'expected_images':len(manifest),'verified_images':len(joined),'missing_images':len(missing),'file_hash_duplicate_groups':int((joined.groupby('sha256').size()>1).sum()),'pixel_duplicate_groups':int(dup.pixel_sha256.nunique()),'cross_split_pixel_duplicate_groups':int((cross>1).sum()),'minimum_width':int(joined.width.min()),'minimum_height':int(joined.height.min()),'original_png_bytes':int(joined.bytes.sum()),'status':'review_duplicates_before_training' if len(dup) else 'basic_image_qc_passed','scope':'Coverage, decoding, size, file hashes and exact pixel duplicates; near-duplicates and visual quality are not covered.'}
(metrics/'image_qc_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
