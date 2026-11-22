from pathlib import Path
import shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(r'C:\Projects\chest-xray-classification')
run=root/'outputs/models/B0_baseline_seed42_20261030'
h=pd.read_csv(run/'history.csv')
v=pd.read_csv(root/'data/processed/reduced_train_full_eval_qc/val.csv')
prevalence=v.pneumonia.mean()
assert len(h)==3 and h.epoch.tolist()==[1,2,3] and h.notna().all().all()
fig,axs=plt.subplots(1,3,figsize=(14,4.5))
for ax,metric,title in zip(axs,['loss','auroc','auprc'],['Binary cross-entropy','AUROC','Average precision (AUPRC)']):
    ax.plot(h.epoch,h['train_'+metric],'o-',label='Train (during updates)',color='#2878b5')
    ax.plot(h.epoch,h['val_'+metric],'o-',label='Validation',color='#e07826')
    if metric=='auprc': ax.axhline(prevalence,color='#777777',linestyle='--',label=f'Val prevalence: {prevalence:.4f}')
    ax.set(title=title,xlabel='Epoch',xticks=[1,2,3])
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('B0 DenseNet-121: frozen backbone, no augmentation / CLAHE',fontsize=14)
fig.text(.5,.015,'Train prevalence: 25%; validation prevalence: 1.29%. Train/validation loss and AP are not directly comparable.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.055,1,.94])
out=root/'outputs/figures/training';out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'B0_learning_curves.png',dpi=180)
metrics=root/'outputs/metrics/training';metrics.mkdir(parents=True,exist_ok=True)
shutil.copyfile(run/'history.csv',metrics/'B0_history.csv')
print('validation prevalence:',prevalence,'total minutes:',h.seconds.sum()/60)
print(out/'B0_learning_curves.png')
