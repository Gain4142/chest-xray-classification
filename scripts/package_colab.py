"""Package train/validation originals and runnable code for Colab."""
from pathlib import Path
import hashlib
import json
import zipfile
import pandas as pd

root = Path.cwd()
out = root / 'outputs/colab'
out.mkdir(parents=True, exist_ok=True)
manifests = root / 'data/processed/reduced_train_full_eval_qc'
names = set()
for split in ('train', 'val'):
    names.update(pd.read_csv(manifests / f'{split}.csv')['Image Index'])
paths = [(p, p.relative_to(root).as_posix()) for p in sorted((root / 'src').glob('*.py'))]
for split in ('train', 'val'):
    p = manifests / f'{split}.csv'
    paths.append((p, p.relative_to(root).as_posix()))
for name in sorted(names):
    p = root / 'data/raw/images' / name
    paths.append((p, p.relative_to(root).as_posix()))
baseline = root / 'outputs/models/ViT_baseline_seed42_20261115'
for name in ('history.csv', 'config.json'):
    paths.append((baseline / name, f'reference_vit/{name}'))
target = out / 'vit_clahe_colab.zip'
with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
    for index, (source, destination) in enumerate(paths):
        bundle.write(source, destination)
        if index % 5000 == 0:
            print(f'Packed {index}/{len(paths)} files', flush=True)
print(f'{target}: {target.stat().st_size / 1e9:.2f} GB, {len(names)} images', flush=True)
