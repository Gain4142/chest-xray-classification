"""Compare deterministic preprocessing on fixed, train-only image samples."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import functional as TF, InterpolationMode
from src.preprocessing import ApplyCLAHE, build_transform


def run_comparison():
    import cv2
    cv2.setNumThreads(1)
    metrics=ROOT/'outputs/metrics/preprocessing'
    figures=ROOT/'outputs/figures/preprocessing'
    metrics.mkdir(parents=True,exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    candidates=pd.read_csv(ROOT/'outputs/metrics/dataset_qc/train_visual_review_samples.csv',dtype={'Patient ID':str})
    samples=candidates.iloc[[0,1,6,10]].copy()
    train=pd.read_csv(ROOT/'data/processed/reduced_train_full_eval_qc/train.csv')
    assert set(samples['Image Index']).issubset(set(train['Image Index']))
    samples.to_csv(metrics/'comparison_samples.csv',index=False)
    clahe=ApplyCLAHE(clip_limit=2.0,tile_grid_size=(8,8))
    transforms={'resize_only':build_transform(),'clahe_then_resize':build_transform(use_clahe=True)}
    batches={name:[] for name in transforms}
    hashes={}
    rows=[]
    fig,axes=plt.subplots(4,3,figsize=(10,14),layout='constrained')
    for i,(_,row) in enumerate(samples.iterrows()):
        filename=row['Image Index']
        path=ROOT/'data/raw/images'/filename
        hashes[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as source:
            original=source.convert('L')
            resized=TF.resize(source.convert('RGB'),[224,224],interpolation=InterpolationMode.BILINEAR,antialias=True).convert('L')
            adjusted=TF.resize(clahe(source),[224,224],interpolation=InterpolationMode.BILINEAR,antialias=True).convert('L')
            variants=[('Original 1024 x 1024',original),('Resize only: 224 x 224',resized),('CLAHE -> resize: 224 x 224',adjusted)]
            for j,(name,image) in enumerate(variants):
                axes[i,j].imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[i,j].axis('off')
                axes[i,j].set_title(name if i==0 else '',fontsize=10)
                arr=np.asarray(image)
                rows.append({'image':filename,'pneumonia_label':int(row.pneumonia),'source_mode':source.mode,'variant':name,'width':image.width,'height':image.height,'mean_intensity':float(arr.mean()),'std_intensity':float(arr.std()),'p01':float(np.percentile(arr,1)),'p99':float(np.percentile(arr,99)),'zero_pixel_percent':float((arr==0).mean()*100),'max_pixel_percent':float((arr==255).mean()*100)})
            axes[i,0].text(0,-0.04,f'{filename} | label={int(row.pneumonia)} | source={source.mode}',transform=axes[i,0].transAxes,fontsize=8)
            for name,transform in transforms.items():
                first=transform(source)
                second=transform(source)
                assert first.shape==(3,224,224) and torch.isfinite(first).all()
                assert torch.equal(first,second), 'Evaluation preprocessing is not deterministic.'
                batches[name].append(first)
        assert hashes[filename]==hashlib.sha256(path.read_bytes()).hexdigest()
    fig.suptitle('Train-only preprocessing comparison | CLAHE clip=2.0, grid=8 x 8\nFixed display scale: 0-255; no cropping; source labels only',fontsize=11)
    figure_path=figures/'clahe_comparison.png'
    fig.savefig(figure_path,dpi=150)
    plt.close(fig)
    table=pd.DataFrame(rows)
    table.to_csv(metrics/'comparison_intensity_statistics.csv',index=False)
    stats=[]
    for name,values in batches.items():
        batch=torch.stack(values)
        stats.append({'variant':name,'batch_shape':list(batch.shape),'normalized_batch_mean':float(batch.mean()),'normalized_batch_std':float(batch.std(unbiased=False)),'channel_means':[float(x) for x in batch.mean(dim=(0,2,3))],'channel_stds':[float(x) for x in batch.std(dim=(0,2,3),unbiased=False)]})
    result={'sample_images':len(samples),'sample_source':'Existing fixed train-only visual QC list, rows 0/1/6/10','clahe':{'implementation':'OpenCV','version':cv2.__version__,'clip_limit':2.0,'tile_grid_size':[8,8],'order':'8-bit grayscale at original resolution -> CLAHE -> RGB -> 224x224 bilinear -> tensor -> ImageNet normalization'},'baseline_default':'resize_only; unchanged','checks':{'training_membership':True,'finite_3x224x224_tensors':True,'repeat_evaluation_deterministic':True,'source_file_hashes_unchanged':True},'normalized_batch_statistics':stats,'quality_or_model_improvement_established':False,'limitations':'Four convenience samples; intensity statistics are descriptive and do not measure diagnostic information or classification performance.'}
    (metrics/'comparison_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return {'figure':figure_path,'table':table,'checks':result}


if __name__=='__main__':
    result=run_comparison()
    print(json.dumps(result['checks'],indent=2))
