"""Build a small Jupyter GPU comparison bundle without data images or CPU packages."""
import ast
import json
import zipfile
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    existing = nbformat.read(ROOT / 'outputs/colab/Model_Comparison_Colab.ipynb', as_version=4)
    runner = existing.cells[-1].source
    cells = [nbformat.v4.new_markdown_cell(
        '# DenseNet–ViT GPU 비교\n'
        '친구의 GPU 컴퓨터에서 이 노트북을 실행합니다. 압축을 푼 프로젝트 폴더에서 Jupyter를 시작하세요. '
        '두 모델을 각각 8에포크, seed 42, 분류층만 학습하고 검증 AUPRC로 최적 모델을 선택합니다. '
        'CLAHE와 증강은 끕니다. 기존 데이터 분할을 재사용하며 테스트 데이터는 사용하지 않습니다. '
        '새 가상환경에 GPU용 torch·torchvision을 먼저 설치한 뒤 requirements-gpu-common.txt를 설치하세요.')]
    def code(source):
        cells.append(nbformat.v4.new_code_cell(source))

    code("""import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import sys, json, hashlib, subprocess, platform, datetime
import torch
ROOT = Path.cwd().resolve()
assert (ROOT / 'src/train.py').is_file(), '압축을 푼 프로젝트 폴더에서 Jupyter를 시작하거나 ROOT를 수정하세요.'
assert torch.cuda.is_available(), '현재 Jupyter 커널에서 GPU를 사용할 수 없습니다. GPU용 PyTorch와 커널을 확인하세요.'
print('Python:', sys.executable)
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
probe = torch.nn.Conv2d(3, 4, kernel_size=3).cuda()
probe_input = torch.randn(2, 3, 16, 16, device='cuda', requires_grad=True)
probe(probe_input).square().mean().backward()
torch.cuda.synchronize()
del probe, probe_input
print('GPU convolution and backward check passed.')
protocol = json.loads((ROOT / 'comparison_protocol.json').read_text(encoding='utf-8'))
assert protocol['epochs'] == 8
output_root = ROOT / 'outputs/models/model_comparison_8epochs_ap_local_v1'
output_root.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)""")
    cells.append(nbformat.v4.new_markdown_cell(
        '## 영상 준비\n기존 Drive의 vit_clahe_colab.zip을 이 프로젝트 폴더에 다운로드해 두거나, '
        '원본 영상을 data/raw/images/로 복사하세요. 다음 셀은 ZIP에서 영상만 가져오므로 최신 코드를 덮어쓰지 않습니다.'))
    code("""import zipfile, pandas as pd
archive = ROOT / 'vit_clahe_colab.zip'
if archive.is_file():
    with zipfile.ZipFile(archive) as package:
        images = [entry for entry in package.infolist() if entry.filename.startswith('data/raw/images/') and not entry.is_dir()]
        assert len(images) == 20948, '영상 ZIP 구성을 확인하세요.'
        for entry in images:
            target = (ROOT / entry.filename).resolve()
            assert target.is_relative_to(ROOT / 'data/raw/images')
            if not target.exists() or target.stat().st_size != entry.file_size:
                package.extract(entry, ROOT)
tables = {}
for split in ('train', 'val'):
    manifest = ROOT / 'data/processed/reduced_train_full_eval_qc' / (split + '.csv')
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == protocol['data'][split]['sha256']
    table = pd.read_csv(manifest, dtype={'Patient ID':str})
    assert len(table) == protocol['data'][split]['images']
    assert int(table.pneumonia.sum()) == protocol['data'][split]['positive_images']
    missing = [name for name in table['Image Index'] if not (ROOT / 'data/raw/images' / name).is_file()]
    assert not missing, f'{split}: missing images {missing[:5]}'
    tables[split] = table
    print(split, 'images:', len(table), 'positive:', int(table.pneumonia.sum()))
assert not set(tables['train']['Patient ID']) & set(tables['val']['Patient ID'])
protocol_path = output_root / 'protocol.json'
if protocol_path.exists():
    assert json.loads(protocol_path.read_text(encoding='utf-8')) == protocol
else:
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
runtime = {'recorded_at':datetime.datetime.now().astimezone().isoformat(), 'python':sys.version,
           'platform':platform.platform(), 'torch':str(torch.__version__), 'cuda':torch.version.cuda,
           'gpu':torch.cuda.get_device_name(0)}
runtime_path = output_root / 'runtime.json'
if runtime_path.exists():
    prior = json.loads(runtime_path.read_text())
    for key in ('python','platform','torch','cuda','gpu'):
        assert prior[key] == runtime[key], f'Runtime changed: {key}; use a separate output folder.'
else:
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding='utf-8')
with (output_root / 'environment.txt').open('w', encoding='utf-8') as env:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=env, check=True)
print('Data and environment checked. Results:', output_root)""")
    cells.append(nbformat.v4.new_markdown_cell(
        '## 학습\n첫 실행은 seed 42 한 쌍입니다. DenseNet 다음 ViT가 실행됩니다. '
        '첫 모델 다운로드에는 인터넷 연결이 필요합니다. 완료된 실행은 확인 후 재사용하고, '
        '중단된 폴더는 덮어쓰지 않습니다. GPU 컴퓨터를 절전·종료하지 마세요.'))
    code(runner)
    cells.append(nbformat.v4.new_markdown_cell(
        '## 결과 보관\n학습 후 outputs/models/model_comparison_8epochs_ap_local_v1 폴더 전체를 '
        '내 컴퓨터나 저장소로 복사하세요. best.pt, last.pt, history.csv, config.json, completed.json, '
        'comparison.csv와 환경 기록이 포함됩니다. 가중치 파일과 원본 영상은 Git에 올리지 않습니다.'))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec':{'display_name':'CXR GPU', 'language':'python', 'name':'cxr-gpu'}})
    nbformat.validate(notebook)
    for cell in cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
    return notebook


def main():
    output = ROOT / 'outputs/transfer'
    output.mkdir(parents=True, exist_ok=True)
    notebook_path = output / 'GPU_Jupyter_Comparison.ipynb'
    nbformat.write(build_notebook(), notebook_path)
    protocol = json.loads((ROOT / 'outputs/metrics/baseline_comparison/protocol.json').read_text())
    protocol['execution'] = 'Local Jupyter CUDA; run each paired comparison in the same environment and record hardware/versions.'
    for cohort in protocol['data'].values():
        cohort['path'] = cohort['path'].replace('\\', '/')
    protocol['next_run_commands'] = [command.replace('<NEW_DRIVE_COMPARISON_DIRECTORY>', 'outputs/models/model_comparison_8epochs_ap_local_v1') for command in protocol['next_run_commands']]
    requirements = '\n'.join(['# Install GPU-compatible torch and torchvision separately first.',
                              'transformers==4.57.6', 'numpy', 'pandas', 'scikit-learn', 'Pillow',
                              'opencv-python-headless', 'matplotlib', 'jupyterlab', 'ipykernel', ''])
    archive = output / 'cxr_gpu_comparison.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(notebook_path, notebook_path.name)
        bundle.writestr('requirements-gpu-common.txt', requirements)
        bundle.writestr('setup_windows_rtx50.cmd', '''@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\\Scripts\\python.exe (
    py -3.12 -m venv .venv
    if errorlevel 1 goto :failed
)
.venv\\Scripts\\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :failed
.venv\\Scripts\\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto :failed
.venv\\Scripts\\python.exe -m pip install -r requirements-gpu-common.txt
if errorlevel 1 goto :failed
.venv\\Scripts\\python.exe -m pip check
if errorlevel 1 goto :failed
.venv\\Scripts\\python.exe -m ipykernel install --user --name cxr-gpu --display-name "CXR GPU"
if errorlevel 1 goto :failed
echo Setup completed. To start: .venv\\Scripts\\python.exe -m jupyterlab
pause
exit /b 0
:failed
echo Setup failed. Save the error message above. Python 3.12 and a compatible NVIDIA driver are required.
pause
exit /b 1
''')
        bundle.writestr('comparison_protocol.json', json.dumps(protocol, ensure_ascii=False, indent=2))
        for name in ('__init__.py','dataset.py','model.py','preprocessing.py','train.py','vit_model.py'):
            bundle.write(ROOT / 'src' / name, 'src/' + name)
        for split in ('train','val'):
            relative = f'data/processed/reduced_train_full_eval_qc/{split}.csv'
            bundle.write(ROOT / relative, relative)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert not any(name.startswith('data/raw') or '.venv' in name for name in bundle.namelist())
    print(f'Created {archive} ({archive.stat().st_size:,} bytes); original images and GPU PyTorch are separate.')


if __name__ == '__main__':
    main()
