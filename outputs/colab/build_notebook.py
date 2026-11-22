from pathlib import Path
import nbformat
root=Path.cwd()
out=root/'outputs/colab'
cells=[]
def md(s): cells.append(nbformat.v4.new_markdown_cell(s))
def code(s): cells.append(nbformat.v4.new_code_cell(s))
md("""# ViT CLAHE 비교 실험 — Colab GPU
Colab에서 런타임 → 런타임 유형 변경 → GPU를 선택하세요. Google Drive의 MyDrive/CXR_CLAHE/ 폴더에 함께 제공한 vit_clahe_colab.zip을 업로드하세요. 약 8GB 이상 업로드 공간과 시간이 필요합니다. 아래 셀을 위에서부터 실행합니다.

같은 GPU에서 기본 ViT와 CLAHE ViT를 각각 3에포크 학습합니다. 동일 seed 42, batch 8, Adam 0.001, BCE, 고정 특징 추출부, 동일 학습·검증 목록을 사용합니다. CLAHE는 clip limit 2.0, tile grid 8×8, 원본 해상도에서 적용하고 RGB/resize/정규화를 이어서 수행합니다. 모델별 최적 체크포인트는 검증 AUROC로 선택합니다. 결과와 체크포인트는 Drive에 저장합니다. GPU와 기존 CPU 실행 사이에는 라이브러리와 수치 연산 차이가 있을 수 있습니다.
""")
code("""import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
assert torch.cuda.is_available(), '런타임 유형을 GPU로 바꿔주세요.'
print(torch.__version__, torch.cuda.get_device_name(0))
%pip -q install transformers==4.57.6 opencv-python-headless pandas scikit-learn matplotlib
""")
code("""from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import zipfile
import os
import shutil
workspace = Path('/content/cxr_clahe')
archive = Path('/content/drive/MyDrive/CXR_CLAHE/vit_clahe_colab.zip')
assert archive.is_file(), f'{archive}에 ZIP을 업로드하세요.'
workspace.mkdir(exist_ok=True)
marker = workspace / '.extracted'
if not marker.exists():
    assert shutil.disk_usage('/content').free > archive.stat().st_size + 2_000_000_000
    with zipfile.ZipFile(archive) as package:
        package.extractall(workspace)
    marker.touch()
os.chdir(workspace)
output_root = Path('/content/drive/MyDrive/CXR_CLAHE/results')
output_root.mkdir(parents=True, exist_ok=True)
""")
sources={p.name:p.read_text(encoding='utf-8-sig') for p in (root/'src').glob('*.py')}
code("""import json
sources = json.loads(SOURCES_LITERAL)
for name, content in sources.items():
    (workspace / 'src' / name).write_text(content, encoding='utf-8')
""".replace("SOURCES_LITERAL", repr(__import__('json').dumps(sources))))
code("""import hashlib
import pandas as pd
import json
import subprocess
import sys
import time
baseline_config = json.loads((workspace / 'reference_vit/config.json').read_text())
manifest_root = workspace / 'data/processed/reduced_train_full_eval_qc'
for split in ('train', 'val'):
    manifest = manifest_root / f'{split}.csv'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == baseline_config['manifest_sha256'][split]
    rows = pd.read_csv(manifest)
    missing = [name for name in rows['Image Index'] if not (workspace / 'data/raw/images' / name).is_file()]
    assert not missing, missing[:5]
subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(output_root / 'environment.txt').open('w'), check=True)
print('学習 4,076枚・検証 16,872枚のファイルと一覧ハッシュを確認しました。')
""".replace('学習 4,076枚・検証 16,872枚のファイルと一覧ハッシュを確認しました。','학습 4,076장·검증 16,872장 파일과 목록 해시 확인 완료'))
code("""smoke_name = 'ViT_CLAHE_GPU_smoke'
smoke_dir = output_root / smoke_name
if not (smoke_dir / 'completed.json').exists():
    assert not smoke_dir.exists(), '중단된 smoke 폴더가 있습니다. 새 run-name으로 실행하세요.'
    subprocess.run([sys.executable, '-u', '-m', 'src.train',
                    '--model', 'vit', '--clahe', '--device', 'cuda',
                    '--smoke', '--epochs', '1', '--run-name', smoke_name,
                    '--output-root', str(output_root)], check=True)
""")
md("""## 같은 GPU에서 두 조건 실행
각 실행은 같은 사전학습 가중치와 동일한 새 분류층 초기화에서 출발합니다. 기존 학습 결과에서 이어서 학습하지 않습니다. GPU 소량 실행 시간은 전체 검증 포함 소요 시간과 다를 수 있습니다. 중단된 실행 폴더를 발견하면 자동 덮어쓰기를 멈추며, 새 run-name으로 재실행할 수 있습니다.
""")
code("""run_names = [('ViT_GPU_B0_seed42', False), ('ViT_GPU_CLAHE_seed42', True)]
for run_name, clahe in run_names:
    run = output_root / run_name
    if (run / 'completed.json').exists():
        print('완료된 실행 재사용', run_name)
        continue
    assert not run.exists(), f'중단된 실행 {run}; 새 run-name을 지정하세요.'
    command = [sys.executable, '-u', '-m', 'src.train', '--model', 'vit',
               '--device', 'cuda', '--epochs', '3', '--batch-size', '8',
               '--lr', '0.001', '--seed', '42', '--run-name', run_name,
               '--output-root', str(output_root)]
    if clahe:
        command.append('--clahe')
    started = time.time()
    subprocess.run(command, check=True)
    print(run_name, '총 경과 분', (time.time()-started)/60)
""")
code("""# 선택된 각 체크포인트의 검증 점수를 저장합니다.
for run_name, _ in run_names:
    run = output_root / run_name
    if not (run / 'val_evaluation/metrics.json').exists():
        subprocess.run([sys.executable, '-u', '-m', 'src.evaluate',
                        str(run), '--split', 'val', '--device', 'cuda'], check=True)

summary = []
for run_name, clahe in run_names:
    run = output_root / run_name
    h = pd.read_csv(run / 'history.csv')
    metrics = json.loads((run / 'val_evaluation/metrics.json').read_text())
    summary.append(dict(run=run_name, clahe=clahe,
                        best_epoch=metrics['checkpoint_epoch'],
                        val_auroc=metrics['auroc'],
                        val_average_precision=metrics['average_precision'],
                        training_and_validation_minutes=h.seconds.sum()/60))
comparison = pd.DataFrame(summary)
comparison.to_csv(output_root / 'clahe_comparison.csv', index=False)
display(comparison)
print('Drive에 저장한 clahe_comparison.csv와 두 history.csv를 가져오면 함께 해석할 수 있습니다.')
""")
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':'ViT_CLAHE_Colab.ipynb'},'accelerator':'GPU'})
nbformat.validate(nb)
nbformat.write(nb,out/'ViT_CLAHE_Colab.ipynb')
for i,c in enumerate(cells):
    if c.cell_type=='code' and '%' not in c.source:
        compile(c.source,f'cell_{i}','exec')
print(out/'ViT_CLAHE_Colab.ipynb')
