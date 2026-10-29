# 전처리 비교 기록: Resize 및 CLAHE

## 목적
고정된 학습 영상 4장에서 크기 축소와 국소 대비 조정의 영향을 비교한다. 영상 표현의 변화와 모델 성능 개선 여부를 구분한다.

## 비교 조건
| 조건 | 순서 |
|---|---|
| 원본 | 1024 x 1024 영상, 흑백 표시 |
| Resize | RGB 변환 후 224 x 224 bilinear 축소 |
| CLAHE + Resize | 원본 해상도에서 8비트 흑백 변환, CLAHE 적용, RGB 변환 후 같은 축소 |

CLAHE 설정: OpenCV 4.13.0, clipLimit=2.0, tileGridSize=(8,8). 이는 사전 지정한 비교 조건이며 최적 파라미터로 추정하지 않았다. 격자는 가로·세로 타일 수를 뜻한다. 동일한 0~255 밝기 범위로 그림을 표시하며 잘라내기는 적용하지 않았다.

데이터: 기존 학습 샘플 목록의 행 0/1/6/10. 양성 라벨 2장, 음성 라벨 2장이며 L과 RGBA 입력을 포함한다. 표본의 대표성을 주장하지 않는다.

## 관찰 및 검증
- 국소 명암과 경계 표현이 달라졌으며 문자·표식 등 비폐 영역도 함께 변한다.
- 크기 축소가 촬영 범위 차이, 문자나 기기를 제거하지 않는다.
- 모든 표본이 3 x 224 x 224 출력 크기와 유한 수치 검사를 통과했다.
- 반복 적용 결과의 일치와 원본 파일 해시 보존을 확인했다.
- 기존 무증강 DataLoader의 8장 입력 검사도 통과했다.
- 정규화 후 표본 평균·표준편차는 기록만 하며, ImageNet 통계 적용 결과가 정확히 0·1이어야 한다고 가정하지 않는다.

## 적용 상태 및 한계
`src/preprocessing.py`의 `build_transform(use_clahe=True)`로 선택 가능하게 구현했다. 기본값은 False이며 기존 기준 모델 입력은 유지한다. 채택할 경우 학습·검증·테스트에서 동일 조건을 사용한다. 모델 성능 비교와 CLAHE 최종 채택은 아직 수행하지 않았다. 이 네 표본만으로 진단 정보 보존이나 전체 데이터 품질 향상을 결론 내리지 않는다.

다음 단계는 약한 학습 증강의 시각화와 학습·평가 변환 구분이다.

## 산출물
- `notebooks/02_Preprocessing.ipynb`: 방법·비교 그림·결과·한계. 코드 셀 3개 실행 완료.
- `scripts/compare_preprocessing.py`: 비교 재현 코드.
- `outputs/figures/preprocessing/clahe_comparison.png`: 비교 그림.
- `outputs/metrics/preprocessing/`: 표본, 픽셀 통계, 검증 기록.
- `requirements.txt`: opencv-python-headless==4.13.0.92 추가.

## 출처
데이터 제공: NIH Clinical Center. https://nihcc.app.box.com/v/ChestXray-NIHCC
배포본: https://www.kaggle.com/datasets/nih-chest-xrays/data
방법 및 API: https://docs.opencv.org/4.13.0/d5/daf/tutorial_py_histogram_equalization.html ; https://docs.opencv.org/4.13.0/d6/db6/classcv_1_1CLAHE.html
