"""Create train-only QC samples and smoke-check the baseline data loader."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
from src.dataset import ChestXrayDataset
from src.preprocessing import build_transform


def main():
    manifest=ROOT/'data/processed/reduced_train_full_eval_qc/train.csv'
    train=pd.read_csv(manifest)
    inventory=pd.read_csv(ROOT/'outputs/metrics/dataset_qc/image_inventory.csv')
    train=train.merge(inventory[['image','mode']],left_on='Image Index',right_on='image',validate='one_to_one')
    selected=pd.concat([train.query('pneumonia == 1').sample(n=6,random_state=42),train.query('pneumonia == 0 and mode == "L"').sample(n=4,random_state=42),train.query('mode == "RGBA"').sample(n=2,random_state=42)]).drop_duplicates('Image Index')
    selected.to_csv(ROOT/'outputs/metrics/dataset_qc/train_visual_review_samples.csv',index=False)
    fig,axes=plt.subplots(3,4,figsize=(12,10),layout='constrained')
    for ax,(_,row) in zip(axes.flat,selected.iterrows()):
        with Image.open(ROOT/'data/raw/images'/row['Image Index']) as im:
            ax.imshow(im.convert('RGB'))
        ax.set_title(f"{row['Image Index']}\nPneumonia label={row['pneumonia']} | {row['mode']}",fontsize=8)
        ax.axis('off')
    for ax in list(axes.flat)[len(selected):]:
        ax.axis('off')
    fig.suptitle('Train-only visual QC samples (source labels, not diagnostic review)',fontsize=12)
    fig.savefig(ROOT/'outputs/figures/train_visual_qc.png',dpi=150)
    plt.close(fig)
    transform=build_transform()
    dataset=ChestXrayDataset(manifest,ROOT/'data/raw/images',transform)
    loader=DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0)
    images,targets=next(iter(loader))
    assert images.shape==(8,3,224,224)
    assert targets.shape==(8,) and torch.isfinite(images).all()
    # Exercise both L and RGBA, plus both labels, using the selected source files.
    for name in selected['Image Index']:
        with Image.open(ROOT/'data/raw/images'/name) as im:
            value=transform(im)
            assert value.shape==(3,224,224) and torch.isfinite(value).all()
    summary={'training_images':len(dataset),'sampled_images':len(selected),'batch_shape':list(images.shape),'target_shape':list(targets.shape),'finite_values':True,'transform':'RGB -> 224x224 bilinear -> float tensor -> ImageNet normalization','augmentation':'none for baseline','scope':'Train-only visual samples and loader smoke check; not full visual/clinical validation.'}
    (ROOT/'outputs/metrics/dataset_qc/preprocessing_check.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
