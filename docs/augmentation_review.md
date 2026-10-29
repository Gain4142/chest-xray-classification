# 학습 증강 및 평가 입력 검증

## 범위
최종 학습 목록에 포함된 고정 표본 4장으로 증강 후보를 검토하였다. L·RGBA 및 두 이진 라벨을 포함한다. 검증·테스트 영상이나 모델 예측은 표본 선택에 사용하지 않았다.

## 설정
RGB 변환 및 224 x 224 축소 후 회전을 확률 0.5로 적용한다. 각도 범위는 -5~+5도이며 bilinear 보간과 검은색 채움을 사용한다. 별도로 확률 0.5로 밝기·대비 조정 묶음을 적용하며 각 계수 범위는 0.9~1.1이다. 이후 텐서 변환과 ImageNet 정규화를 적용한다.

CLAHE는 이번 비교에서 꺼져 있다. 좌우·상하 반전, 무작위 Crop, 노이즈와 가림은 넣지 않았다. 초기 설정이며 최적 조건이나 임상 정보 보존을 검증한 설정이 아니다.

## 확인 결과
- 같은 영상의 8개 난수 시드 조건(42~49)에서 변환 결과 변화 및 고정 시드 재현성을 확인했다.
- 평가용 변환은 반복 적용 결과가 일치했다.
- 증강을 끈 학습용 변환은 평가용 및 기존 기준 전처리와 일치했다.
- 원본 파일 해시는 유지되었다.
- 실제 DataLoader의 4장 묶음은 (4,3,224,224)이며 라벨은 학습 목록과 일치했다.
- 정규화 후 표본 평균 0.4086, 표준편차 1.2713을 기록했다. 이 값이 정확히 0과 1이어야 한다고 가정하지 않는다.

## 관찰 및 한계
약한 자세·밝기 변화가 보이고 회전에 따른 검은 테두리와 일부 가장자리 손실이 발생한다. 증강이 모든 영상에 적합하거나 성능을 높인다는 결론은 내리지 않는다. 문자·기기 표식과 라벨 노이즈 문제도 해결되지 않는다.

## 구현과 사용 상태
- `src/preprocessing.py`: `build_train_transform()`, `build_eval_transform()` 추가. `augment=False`로 학습 증강을 끌 수 있다.
- `scripts/preview_augmentation.py`: 반복 증강 시각화, 실제 배치 및 결정성 검사.
- `notebooks/02_Preprocessing.ipynb`: 전처리 비교에 증강 방법·결과·한계 추가.
- `outputs/figures/augmentation/`: 동일 영상 반복 증강 및 실제 배치 그림.
- `outputs/metrics/augmentation/`: 표본 목록 및 검사 기록.

기존 `build_transform()` 및 모델 속도 측정의 증강 없는 입력은 유지한다. 본 학습은 아직 시작하지 않았다. 원본 파일과 학습 목록 수는 증가하지 않으며 실제 학습 시 선택된 변환은 입력을 읽을 때 메모리에서 수행한다.

## 출처
데이터 제공: NIH Clinical Center. https://nihcc.app.box.com/v/ChestXray-NIHCC
API: https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomRotation.html ; https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html
