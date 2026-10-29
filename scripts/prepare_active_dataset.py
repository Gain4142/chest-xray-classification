"""Reproduce reduced-train/full-evaluation manifests without silent replacement."""
from pathlib import Path
import hashlib
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'data/processed/splits'
    tables={name:pd.read_csv(source/f'{name}.csv',dtype={'Patient ID':str}) for name in ['train','val','test']}
    train=tables['train']
    positive=train.loc[train.pneumonia.eq(1)]
    negative=train.loc[train.pneumonia.eq(0)].sample(n=3*len(positive),random_state=42)
    tables['train']=pd.concat([positive,negative]).sort_values('Image Index').reset_index(drop=True)
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if set(tables[a]['Patient ID']) & set(tables[b]['Patient ID']):
            raise ValueError('Patient overlap found.')
    out=ROOT/'data/processed/reduced_train_full_eval'
    for name,table in tables.items():
        target=out/f'{name}.csv'
        if target.exists():
            saved=pd.read_csv(target,dtype=str).reset_index(drop=True)
            expected=table.astype(str).reset_index(drop=True)
            if not saved.equals(expected):
                raise FileExistsError(f'Existing cohort differs; no files changed: {target}')
    out.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():
        target=out/f'{name}.csv'
        if not target.exists(): table.to_csv(target,index=False)
    report={'seed':42,'training':'All positive train images plus 3x random negative images','evaluation':'Original full validation/test cohorts, before pixel deduplication','manifest_directory':str(out.relative_to(ROOT)),'counts':{name:len(table) for name,table in tables.items()},'parent_sha256':{name:hashlib.sha256((source/f'{name}.csv').read_bytes()).hexdigest() for name in tables}}
    (ROOT/'outputs/metrics/dataset_qc/pre_qc_dataset_plan.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
