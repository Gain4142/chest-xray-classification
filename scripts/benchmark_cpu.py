"""Time a few training batches; not a full training run or performance evaluation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import math
import statistics
import time
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from src.dataset import ChestXrayDataset
from src.preprocessing import build_transform
from src.model import build_model,set_classifier_training


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    dataset=ChestXrayDataset(ROOT/'data/processed/reduced_train_full_eval_qc/train.csv',ROOT/'data/raw/images',build_transform())
    indices=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(42))[:32].tolist()
    loader=DataLoader(Subset(dataset,indices),batch_size=8,num_workers=0,shuffle=False)
    print('Preparing ImageNet-pretrained DenseNet-121 (weights cached in project)...',flush=True)
    model=build_model()
    set_classifier_training(model)
    optimizer=torch.optim.Adam(model.classifier.parameters(),lr=1e-3)
    criterion=nn.BCEWithLogitsLoss()
    backbone_before=model.features.conv0.weight.detach().clone()
    running_before=model.features.norm0.running_mean.detach().clone()
    classifier_before=model.classifier.weight.detach().clone()
    train_times=[]
    load_times=[]
    losses=[]
    positive=0
    iterator=iter(loader)
    batch=None
    for i in range(4):
        start=time.perf_counter()
        images,targets=next(iterator)
        load_elapsed=time.perf_counter()-start
        batch=images
        positive+=int(targets.sum())
        start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits=model(images).squeeze(1)
        if logits.shape!=targets.shape or not torch.isfinite(logits).all():
            raise RuntimeError('Invalid binary model output.')
        loss=criterion(logits,targets)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss.')
        loss.backward()
        optimizer.step()
        elapsed=time.perf_counter()-start
        losses.append(float(loss.detach()))
        if i>0:
            train_times.append(elapsed)
            load_times.append(load_elapsed)
        print(f'Batch {i+1}/4: compute {elapsed:.3f}s, loading {load_elapsed:.3f}s',flush=True)
    assert torch.equal(backbone_before,model.features.conv0.weight)
    assert torch.equal(running_before,model.features.norm0.running_mean)
    assert not any(p.requires_grad for p in model.features.parameters())
    assert not torch.equal(classifier_before,model.classifier.weight)
    model.eval()
    inference=[]
    with torch.inference_mode():
        for _ in range(3):
            start=time.perf_counter()
            model(batch)
            inference.append(time.perf_counter()-start)
    train_step=statistics.median(train_times)
    load_step=statistics.median(load_times)
    inference_step=statistics.median(inference)
    nval=len(pd.read_csv(ROOT/'data/processed/reduced_train_full_eval_qc/val.csv'))
    ntest=len(pd.read_csv(ROOT/'data/processed/reduced_train_full_eval_qc/test.csv'))
    result={'model':'DenseNet-121','weights':'IMAGENET1K_V1','device':'cpu','torch_threads':4,'batch_size':8,'loader_workers':0,'sample_images':32,'sample_positive_images':positive,'warmup_batches':1,'timed_training_batches':3,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_parameters':sum(p.numel() for p in model.parameters()),'median_train_compute_seconds_per_batch':round(train_step,4),'median_load_seconds_per_batch':round(load_step,4),'median_inference_compute_seconds_per_batch':round(inference_step,4),'estimated_training_minutes_per_epoch':round(math.ceil(len(dataset)/8)*(train_step+load_step)/60,2),'estimated_full_validation_minutes':round(math.ceil(nval/8)*(inference_step+load_step)/60,2),'estimated_full_test_minutes':round(math.ceil(ntest/8)*(inference_step+load_step)/60,2),'frozen_backbone_parameter_check':True,'frozen_batchnorm_statistics_check':True,'classifier_update_check':True,'finite_loss_check':True,'limitations':'Tiny train-only timing sample; thermal throttling, disk/cache effects and background work may change duration. No accuracy estimate; transient benchmark model discarded.','full_training_started':False,'validation_or_test_inference_performed':False}
    out=ROOT/'outputs/metrics/cpu_benchmark.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
