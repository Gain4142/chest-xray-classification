"""Install the versioned train/validation split and extract only selected images."""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(config_path, archive=None):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    tables = {}
    for split in ('train', 'val'):
        spec = config['data'][split]
        source = ROOT / spec['source']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == spec['sha256'], f'Changed versioned split: {split}'
        target = ROOT / spec['path']
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest() == spec['sha256'], f'Existing {split} split differs; it was not overwritten.'
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        with target.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == spec['images']
        assert sum(int(row['pneumonia']) for row in rows) == spec['positive_images']
        assert len({row['Image Index'] for row in rows}) == len(rows)
        tables[split] = rows
    assert not {r['Patient ID'] for r in tables['train']} & {r['Patient ID'] for r in tables['val']}
    names = {row['Image Index'] for rows in tables.values() for row in rows}
    assert all(Path(name).name == name and '/' not in name and '\\' not in name for name in names)
    image_dir = ROOT / 'data/raw/images'
    if archive:
        image_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as package:
            for number, name in enumerate(sorted(names), 1):
                entry = package.getinfo('data/raw/images/' + name)
                target = image_dir / name
                if not target.exists() or target.stat().st_size != entry.file_size:
                    temporary = target.with_suffix('.png.tmp')
                    with package.open(entry) as source, temporary.open('wb') as destination:
                        shutil.copyfileobj(source, destination)
                    temporary.replace(target)
                if number % 5000 == 0:
                    print(f'Prepared {number}/{len(names)} images', flush=True)
    missing = [name for name in sorted(names) if not (image_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} images missing. Supply --archive <vit_clahe_colab.zip> or copy images into {image_dir}. Examples: {missing[:3]}')
    print(f'Checked fixed splits and {len(names)} image paths. No test data used.', flush=True)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/baseline_comparison.json')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    prepare(args.config, args.archive)


if __name__ == '__main__':
    main()
