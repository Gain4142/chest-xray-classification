"""Train-only image/label review using the exact ViT training transform."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from transformers import ViTImageProcessor
from src.dataset import ChestXrayDataset
from src.experiment_config import ROOT, load_config, validate_training
from src.vit_model import MODEL_ID, MODEL_REVISION, ViTInputTransform


def preview(config_path=ROOT / 'params.yaml'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    config = validate_training(load_config(config_path))
    spec = config['data']['train']
    manifest = ROOT / spec['path']
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == spec['sha256']
    processor = ViTImageProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=ROOT / 'outputs/models/hf_cache',
        local_files_only=True)
    table = pd.read_csv(manifest, dtype={'Patient ID':str})
    indices = []
    for label in (0, 1):
        indices.extend(table.index[table.pneumonia.eq(label)].to_series().sample(n=4, random_state=42).tolist())
    out = ROOT / 'outputs/figures/vit_input_review'
    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        for clahe in (False, True):
            transform = ViTInputTransform(processor, clahe)
            dataset = ChestXrayDataset(manifest, ROOT / 'data/raw/images', transform)
            loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False,
                                num_workers=0, generator=torch.Generator().manual_seed(42))
            images, labels = next(iter(loader))
            assert tuple(images.shape) == (8,3,224,224) and torch.isfinite(images).all()
            assert labels.tolist() == table.iloc[indices].pneumonia.astype(float).tolist()
            for j,i in enumerate(indices):
                with Image.open(ROOT / 'data/raw/images' / table.iloc[i]['Image Index']) as source:
                    assert torch.equal(images[j], transform(source))
            tensors[clahe] = images
    assert torch.equal(before, torch.get_rng_state())
    mean = np.array(processor.image_mean); std = np.array(processor.image_std)
    fig, axes = plt.subplots(8,3,figsize=(10,23),layout='constrained')
    files = []
    for j,i in enumerate(indices):
        row = table.iloc[i]
        image_path = ROOT / 'data/raw/images' / row['Image Index']
        with Image.open(image_path) as source:
            axes[j,0].imshow(source.convert('RGB'))
        for col,clahe in ((1,False),(2,True)):
            array = tensors[clahe][j].permute(1,2,0).numpy()*std+mean
            axes[j,col].imshow(np.clip(array,0,1))
        axes[j,0].set_title(f"{row['Image Index']} | source label={row['pneumonia']}",fontsize=8)
        for col,title in ((1,'ViT input: CLAHE off'),(2,'ViT input: CLAHE on')):
            axes[j,col].set_title(title,fontsize=9)
        for ax in axes[j]: ax.axis('off')
        files.append({'image':row['Image Index'],'patient':str(row['Patient ID']),
                      'source_label':int(row['pneumonia']),
                      'sha256':hashlib.sha256(image_path.read_bytes()).hexdigest()})
    fig.suptitle('Training data only | 0 = no pneumonia label, not confirmed normal\n'
                 'Original / inverse-normalized DataLoader input / CLAHE input',fontsize=12)
    figure = out / 'batch.png';fig.savefig(figure,dpi=100);plt.close(fig)
    report = {'train_manifest_sha256':spec['sha256'],'processor':MODEL_ID,'revision':MODEL_REVISION,
              'batch_shape':[8,3,224,224],'both_clahe_conditions_checked':True,
              'actual_loader_labels_match_manifest':True,'caller_torch_rng_unchanged':True,
              'scope':'Input/label wiring only, not clinical diagnosis. No validation/test images.',
              'samples':files}
    (out/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return figure


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'params.yaml')
    preview(parser.parse_args().config)
