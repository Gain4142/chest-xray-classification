# Windows GPU 컴퓨터에서 실험 실행하기

이 저장소를 GPU 컴퓨터에 복제해 Jupyter에서 실행한다. 이후에는 `git pull`로 코드와 설정을 갱신한다. 원본 영상과 모델 가중치는 Git에 포함되지 않는다.

## 1. 저장소 받기

친구의 Windows 11 컴퓨터에서 Git을 설치하고 터미널에 입력한다.

```powershell
git clone https://github.com/Gain4142/chest-xray-classification.git
cd chest-xray-classification
```

이전에 받은 ZIP 폴더와 별도의 폴더를 사용한다. 기존 ZIP의 `src`를 새 저장소 위에 덮어쓰지 않는다.

## 2. 실행 환경 설치

Windows Python 3.12와 Python launcher가 필요하다. `py -3.12 --version`으로 확인한다. RTX 5080 Laptop GPU용 호환 NVIDIA 드라이버도 필요하다.

저장소 안의 `setup_windows_rtx50.cmd`를 더블클릭한다. 프로젝트 전용 `.venv`에 PyTorch 2.14.1, torchvision 0.29.1의 CUDA 13.0 빌드와 나머지 패키지를 설치한다. GPU의 합성곱·행렬곱·역전파까지 확인한 뒤 Jupyter의 `CXR GPU` 커널을 등록한다.

실패하면 **`setup_gpu.log` 파일**을 전달한다. 실패했다는 마지막 문구만으로 원인을 판단할 수 없다. 이 로그는 Git에서 제외된다. 기존 ZIP용 설치 실패 원인은 아직 확인되지 않았다.

수동 설치가 필요하면 프로젝트 폴더에서 다음을 실행한다. PowerShell의 스크립트 실행 정책을 변경하거나 가상환경을 활성화할 필요는 없다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --upgrade torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-common.txt
.\.venv\Scripts\python.exe -m scripts.check_gpu
.\.venv\Scripts\python.exe -m ipykernel install --user --name cxr-gpu --display-name "CXR GPU"
```

GPU 설치 조합: [PyTorch 공식 설치 명령](https://pytorch.org/get-started/previous-versions/). 기존 `requirements.txt`는 과거 CPU 환경 기록이므로 GPU 설치에는 `requirements-gpu-common.txt`를 사용한다.

## 3. 영상 준비

별도로 전달받은 `vit_clahe_colab.zip`을 저장소 최상위에 둔다. 이 ZIP은 학습·검증 영상 20,948장과 이전 코드가 들어 있는 전송 파일이다. **전체 ZIP을 수동으로 저장소에 풀지 않는다.** 아래 명령은 필요한 영상만 추출하고 Git에 저장된 분할 목록을 설치한다. 이전 코드로 덮어쓰지 않는다.

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_gpu_data --archive vit_clahe_colab.zip
```

이미 원본 영상을 `data/raw/images/`에 복사했다면 `--archive`를 생략한다. 분할은 `experiments/splits/train.csv`, `val.csv`에 고정돼 있다. 해시·영상 수·양성 수·환자 중복·영상 경로를 확인하며, 다른 분할이 이미 있으면 덮어쓰지 않고 중단한다. ZIP과 추출 영상을 함께 보관하면 영상만 약 17GB가 필요하며, Python 환경·모델·결과용 추가 공간도 필요하다.

## 4. Jupyter에서 실행

`start_jupyter.cmd`를 더블클릭하고 `notebooks/03_GPU_Comparison.ipynb`을 연다. 커널은 **CXR GPU**를 선택한다. 셀을 위에서부터 실행한다.

노트북은 저장소의 `src`와 `scripts`를 직접 실행한다. 코드 사본을 노트북 안에 넣지 않으므로 다음 Git 갱신이 그대로 반영된다. 첫 모델 다운로드에는 인터넷이 필요하다. 실행 중에는 GPU 컴퓨터를 절전하거나 종료하지 않는다.

현재 설정은 저장소 최상위의 **`params.yaml`**이다. Jupyter에 하이퍼파라미터를 따로 적지 않는다. 고정된 데이터 분할 정보는 `experiments/data_split.json`에 분리돼 있다.

| 항목 | 설정 |
|---|---|
| 모델 | DenseNet-121, ViT-B/16 |
| 학습 범위 | 본체 고정, 분류층만 학습 |
| 에포크 | 모델별 8 |
| 첫 seed | 42 |
| Batch size / optimizer / LR | 8 / Adam / 0.001 |
| Loss | 가중치 없는 BCEWithLogitsLoss |
| CLAHE / 증강 | 모두 끔 |
| Best checkpoint | 검증 Average Precision(AUPRC), 동점이면 먼저 나온 에포크 |
| 보조 지표 | 선택된 체크포인트의 AUROC |

추가 seed 123, 2026은 첫 비교의 소요 시간과 GPU 사용 가능 시간을 확인한 뒤 실행한다. 테스트 데이터는 이번 모델 선택에 사용하지 않는다.

터미널에서도 같은 코드를 실행할 수 있다.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_comparison
```

소량 실행 점검은 `--smoke`를 붙인다. 이 점검은 학습 영상 32장만 사용하며 지표는 검증 성능이 아니다. 추가 seed 실행 예시는 다음과 같다.

```powershell
.\.venv\Scripts\python.exe -m scripts.run_comparison --seeds 123 2026
```

## 5. 중단과 결과 보관

같은 명령을 다시 실행하면 완료된 모델은 재사용하고, 중단된 모델은 마지막 완료 에포크의 `last.pt`에서 이어간다. 가중치·optimizer·난수·배치 순서 상태를 복원한다. 첫 에포크 저장 이전에 끊기면 체크포인트가 없으므로, 해당 불완전 실행 폴더를 다른 이름으로 보관한 뒤 다시 실행한다.

기본 결과는 `outputs/models/baseline_frozen_8ep_ap_v2/`에 저장된다. 실제 위치는 `params.yaml`의 `output_root`와 `experiment_id`를 따른다.

- 모델별 `best.pt`, `last.pt`, `history.csv`, `config.json`, `completed.json`
- `comparison.csv`: 완료된 seed별 비교 결과
- `runtime.json`, `environment.txt`: GPU와 실행 환경
- `execution_manifest.json`, `source_snapshot/`, `invocation.json`: 코드·설정·Git 커밋 기록
- `params.resolved.yaml`, `invocations/`: CLI 덮어쓰기와 실행한 모델·seed까지 반영한 실제 설정 기록
- 모델별 `.log`: 진행 상황과 오류

이 폴더 전체를 원래 작업 컴퓨터나 별도 저장소로 복사한다. 모델 가중치·원본 영상은 Git에 올리지 않는다. 한 seed만 실행한 결과를 반복 실험 평균으로 표현하지 않는다.

## 6. 코드 갱신과 다른 실험

실행 중인 학습이 끝난 뒤 저장소 폴더에서 실행한다.

```powershell
git pull --ff-only
```

패키지 목록이 바뀌면 새 목록을 설치하고 Jupyter 커널을 재시작한다. 이후 노트북을 다시 실행하면 최신 코드가 사용된다. 환경을 처음부터 매번 만들 필요는 없다.

학습 중에는 설정을 수정하지 않는다. 실험이 끝나면 `params.yaml`에서 바꿀 값과 `experiment_id`를 수정하고, 노트북을 처음부터 다시 실행한다. 예를 들어 ViT만 더 작은 학습률로 학습하려면 다음과 같이 바꾼다. 나머지 값은 유지한다.

```yaml
experiment_id: vit_frozen_lr_1e4_v1
models: [vit]
lr: 0.0001
```

모든 새 실험은 저장소 최상위 `params.yaml` 하나에서 설정한다. 노트북의 `CONFIG` 경로는 바꾸지 않는다. 실행 당시 설정은 결과 폴더의 `params.resolved.yaml`과 `invocations/`에 자동 보관된다. 기존 실험별 YAML은 과거 기록으로 유지하며 새 실행 입력으로 사용하지 않는다.

| YAML 항목 | 바꿀 수 있는 내용 |
|---|---|
| `experiment_id` | 결과 폴더 이름. 학습 조건을 바꾸면 새 이름 사용 |
| `models`, `seeds` | 실행할 모델과 난수 seed 목록 |
| `epochs`, `batch_size`, `lr` | 학습 횟수, 배치 크기, 학습률 |
| `optimizer`, `weight_decay` | `adam` 또는 `adamw`, 가중치 감쇠 |
| `pos_weight` | `null`이면 일반 BCE, 양수이면 양성 클래스 손실 가중치 |
| `clahe` | CLAHE 적용 여부 |
| `vit_last_block` | ViT 마지막 블록·최종 LayerNorm 학습 여부 |
| `selection_metric` | Best checkpoint 기준 `auprc` 또는 `auroc` |
| `device`, `threads`, `output_root` | 실행 장치, CPU 스레드 수, 저장 위치 |

`augmentation`은 아직 학습 실행기에 연결하지 않았으므로 `false`로 둔다. `true`를 지정하면 무시하지 않고 오류로 중단한다. 본체를 학습할 때는 해당 실험에 맞게 학습률도 정한다. JSON 설정 파일도 하위 호환으로 지원하지만 새 실험의 기본 설정은 YAML이다.

seed나 실행할 모델 목록만 바꾸면 같은 학습 조건의 결과 폴더에 추가할 수 있다. 학습률 등 학습 조건이나 코드가 바뀌면 새 `experiment_id`를 사용한다. `invocations/`에는 실행할 때마다 실제 설정이 별도 파일로 남는다.

학습 조건·학습 코드·실행 환경이 달라졌는데 같은 결과 폴더를 사용하면 실행을 거부한다. 기존 실험을 덮어쓰지 않도록 새 `experiment_id` 또는 별도의 `--output-root`를 사용한다. 최종 모델 평가와 README 결과 정리는 실험 완료 후 진행한다.

## 추가 ViT 실험 설정

설정 입력은 params.yaml 하나다. vit_training은 frozen / last_block / llrd, scheduler는 constant / warmup_cosine이다. warmup_ratio=0.1은 전체 optimizer 업데이트의 10%를 뜻하며 검증 배치는 제외한다. LLRD는 embeddings부터 마지막 블록까지 모두 학습하며 층별 최대 학습률에 llrd_decay=0.75를 적용한다. Adam, weight_decay=0을 유지하므로 강의의 AdamW+weight decay 레시피 전체를 복제한 실험은 아니다.

추가 조건(각 seeds=[42,123,2026], 8ep, batch8, CLAHE/증강 off, BCE):
- vit_frozen_warmcos_8ep_ap_v1: frozen, lr=0.001, warmup_cosine
- vit_lastblock_warmcos_8ep_ap_v1: last_block, lr=0.00003, warmup_cosine
- vit_llrd_constant_8ep_ap_v1: llrd, lr=0.00003, constant
- vit_llrd_warmcos_8ep_ap_v1: llrd, lr=0.00003, warmup_cosine

각 조건을 실행할 때 params.yaml만 수정한다. 소량 점검은 별도 실행 결과이며 전체 GPU 본 학습으로 간주하지 않는다. parameter_groups.json은 학습 대상 이름과 층별 최대 LR, history.csv는 에포크 시작/다음 업데이트 LR을 기록한다. scheduler는 optimizer.step 다음 매 학습 배치마다 전진하며 scheduler 상태를 체크포인트에 함께 저장한다. scheduled 실행의 epochs를 바꾸어 재개하면 원래 학습률 곡선이 달라지므로 거부한다.

실행 전 커널을 재시작하고 실제 입력 배치 셀을 확인한다. 동일한 8개 학습 영상에서 원본/CLAHE 전후를 표시하며 라벨 매칭과 난수 상태 보존을 검사한다. 라벨 0은 정상 확진이 아니라 Pneumonia 라벨 없음이다. 친구 GPU에서 새 코드의 각 학습 범위·scheduler 조합 소량 점검은 별도로 필요하다.

기존 실행의 소스 지문이 달라지므로 기존 실험 ID로 새 학습을 시작/재개하지 않는다. 기존 Best 평가와 결과 기록은 보존한다. 추가 실험이 끝날 때까지 테스트 결과를 모델·설정 선택에 사용하지 않는다.
