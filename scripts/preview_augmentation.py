"""Visualize weak train-only augmentation and verify evaluation determinism."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from src.dataset import ChestXrayDataset
from src.preprocessing import build_transform, build_train_transform, build_eval_transform

MEAN=np.array([0.485,0.456,0.406])
STD=np.array([0.229,0.224,0.225])


def display_array(tensor):
    return np.clip(tensor.permute(1,2,0).numpy()*STD+MEAN,0,1)


def run_review():
    metrics=ROOT/'outputs/metrics/augmentation'
    figures=ROOT/'outputs/figures/augmentation'
    metrics.mkdir(parents=True,exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    samples=pd.read_csv(ROOT/'outputs/metrics/preprocessing/comparison_samples.csv')
    train_path=ROOT/'data/processed/reduced_train_full_eval_qc/train.csv'
    train=pd.read_csv(train_path)
    assert set(samples['Image Index']).issubset(set(train['Image Index']))
    train_tf=build_train_transform()
    eval_tf=build_eval_transform()
    outputs=[]
    for _,row in samples.iterrows():
        filename=row['Image Index']
        path=ROOT/'data/raw/images'/filename
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            baseline=eval_tf(image)
            assert torch.equal(baseline,eval_tf(image))
            assert torch.equal(baseline,build_transform()(image))
            assert torch.equal(baseline,build_train_transform(augment=False)(image))
            variants=[]
            for seed in range(42,50):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    value=train_tf(image)
                assert value.shape==(3,224,224) and torch.isfinite(value).all()
                variants.append(value)
            assert len({hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in variants})>1
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                repeat=train_tf(image)
            assert torch.equal(variants[0],repeat)
        assert before==hashlib.sha256(path.read_bytes()).hexdigest()
        outputs.append((filename,int(row.pneumonia),baseline,variants))
    fig,axes=plt.subplots(3,3,figsize=(9,9),layout='constrained')
    filename,label,baseline,variants=outputs[0]
    for ax,value,title in zip(axes.flat,[baseline]+variants,['Evaluation preprocessing']+[f'Train augmentation | seed={s}' for s in range(42,50)]):
        ax.imshow(display_array(value)); ax.axis('off'); ax.set_title(title,fontsize=9)
    fig.suptitle(f'Train-only augmentation: {filename} | source label={label}\nCLAHE off; rotation +/-5 deg; brightness/contrast +/-10%',fontsize=11)
    repeated=figures/'same_image_augmentation.png'
    fig.savefig(repeated,dpi=140)
    plt.close(fig)
    lookup={name:i for i,name in enumerate(train['Image Index'])}
    chosen=[lookup[name] for name in samples['Image Index']]
    dataset=ChestXrayDataset(train_path,ROOT/'data/raw/images',train_tf)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        images,targets=next(iter(DataLoader(Subset(dataset,chosen),batch_size=4,num_workers=0,shuffle=False)))
    assert images.shape==(4,3,224,224) and targets.shape==(4,) and torch.isfinite(images).all()
    assert targets.tolist()==samples.pneumonia.astype(float).tolist()
    fig,axes=plt.subplots(1,4,figsize=(12,3.5),layout='constrained')
    for ax,value,target,name in zip(axes,images,targets,samples['Image Index']):
        ax.imshow(display_array(value)); ax.axis('off'); ax.set_title(f'{name}\nsource label={int(target)}',fontsize=9)
    fig.suptitle('Actual augmented DataLoader batch | training manifest only',fontsize=11)
    batch_figure=figures/'augmented_batch.png'
    fig.savefig(batch_figure,dpi=140)
    plt.close(fig)
    result={'training_samples':samples['Image Index'].tolist(),'samples':4,'repeated_variants_per_image':8,'visualization_seeds':list(range(42,50)),'clahe_enabled':False,'rotation':{'probability':0.5,'degrees':[-5,5],'expand':False,'fill':0},'brightness_contrast':{'probability':0.5,'brightness_factor':[0.9,1.1],'contrast_factor':[0.9,1.1]},'checks':{'sample_membership_in_training':True,'original_files_unchanged':True,'evaluation_deterministic':True,'evaluation_matches_previous_baseline':True,'augmentation_off_matches_evaluation':True,'seed_reproducible':True,'training_output_varies':True,'finite_3x224x224_output':True,'actual_loader_labels_match_manifest':True},'actual_batch_shape':list(images.shape),'normalized_batch_mean':float(images.mean()),'normalized_batch_std':float(images.std(unbiased=False)),'default_benchmark_training_changed':False,'full_model_training_started':False,'limitations':'Four convenience samples, not clinical validation. Rotation can add black corners and remove small edge regions. Generalization improvement is not established.'}
    (metrics/'augmentation_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    samples.to_csv(metrics/'augmentation_samples.csv',index=False)
    return {'repeated_figure':repeated,'batch_figure':batch_figure,'checks':result}


if __name__=='__main__':
    result=run_review()
    print(json.dumps(result['checks'],indent=2))
