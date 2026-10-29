"""Download a tiny train-only convenience sample to verify image QC workflow."""
from pathlib import Path
import hashlib
import io
import zipfile
import pandas as pd
import requests
from PIL import Image

root = Path(__file__).resolve().parents[1]
train = set(pd.read_csv(root/'data/processed/splits/train.csv')['Image Index'])
response = requests.get('https://www.kaggle.com/api/v1/datasets/list/nih-chest-xrays/data', timeout=30)
response.raise_for_status()
candidates = [x['name'] for x in response.json()['datasetFiles'] if x['name'].endswith('.png') and Path(x['name']).name in train][:6]
if not candidates:
    raise RuntimeError('No train images found in the returned listing.')
out = root/'data/raw/qc_pilot'
out.mkdir(exist_ok=True)
rows = []
for remote in candidates:
    name = Path(remote).name
    target = out/name
    if not target.exists():
        url = 'https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data'
        r = requests.get(url, params={'fileName': remote}, timeout=60)
        r.raise_for_status()
        payload = r.content
        if zipfile.is_zipfile(io.BytesIO(payload)):
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                matches = [n for n in archive.namelist() if Path(n).name == name]
                if len(matches) != 1:
                    raise ValueError('Unexpected archive contents.')
                payload = archive.read(matches[0])
        with Image.open(io.BytesIO(payload)) as im:
            im.verify()
        target.write_bytes(payload)
    with Image.open(target) as im:
        im.load()
        rows.append({'image':name, 'split':'train', 'width':im.width, 'height':im.height, 'format':im.format, 'mode':im.mode, 'bytes':target.stat().st_size, 'sha256':hashlib.sha256(target.read_bytes()).hexdigest(), 'decode_ok':True})
    print(f'Checked {name}', flush=True)
report = pd.DataFrame(rows)
report.to_csv(root/'outputs/metrics/dataset_qc/pilot_image_qc.csv', index=False)
print(report[['image','width','height','format','mode','decode_ok']].to_string(index=False))
print(f'Total bytes: {report.bytes.sum()}')
