# 기존 3에포크 ViT 모델의 Grad-CAM

이 문서는 초기 실험 모델에 대한 분석 기록이다. 새 8에포크 모델 비교와 파인튜닝 이후의 최종 모델 평가는 별도로 수행한다.

## 대상과 사례 선택
최종 ViT의 best.pt(3에포크), 테스트 예측과 검증에서 확정한 임계값 0.0968017429를 사용했다. TP, TN, FP, FN 그룹별로 seed 42로 섞은 뒤 각각 2장을 추출했다. 총 8명으로 환자 중복을 제외했다. 이 선택은 정성적 예시이며 전체 오류 분포를 대표한다고 가정하지 않는다.

## 지도 계산
마지막 Transformer 블록의 self-attention 직전 LayerNorm 출력인 vit.encoder.layer[-1].layernorm_before를 사용했다. CLS 토큰을 제외한 196개 패치 토큰을 14×14 공간으로 배치했다. 양성 logit의 패치별 기울기를 채널별로 평균 내고, 활성값과 가중합한 뒤 ReLU를 적용했다. 224×224로 보간하고 영상마다 자체 최댓값으로 정규화했다. 모든 그룹에서 같은 양성 점수에 대한 기여를 표시한다. TN과 FN에서도 양성 점수에 대한 기여를 읽는다.

고정된 모델도 입력에 대한 기울기를 계산할 수 있도록 입력 텐서에 requires_grad를 설정했다. 계산은 모델 가중치를 갱신하지 않는다. 재계산 점수와 저장된 테스트 점수가 1e-5 이내인지, 일반 순전파 출력과 같은지, 패치 기울기가 0이 아닌지와 히트맵이 유한한지 확인한다.

## 해석 범위
각 영상은 원본, 히트맵, 중첩 그림 순서다. Label은 데이터셋 라벨이고 predicted는 확정한 임계값에 따른 예측이다. 히트맵은 영상 안에서 상대 정규화하므로 다른 영상의 색 강도를 절대 기여량으로 비교하기 어렵다. 14×14 지도를 확대해 표시하므로 세밀한 병변 경계를 제공하지 않는다. 강조 위치와 병변 일치 여부 및 지름길 학습에 대한 인과적 검증은 별도 분석이 필요하다.

## 저장 위치
- ../outputs/figures/gradcam/vit_test_TP.png
- ../outputs/figures/gradcam/vit_test_TN.png
- ../outputs/figures/gradcam/vit_test_FP.png
- ../outputs/figures/gradcam/vit_test_FN.png
- ../outputs/metrics/gradcam/vit_test_cases.csv
- ../outputs/metrics/gradcam/vit_method.json

프로젝트 루트에서 python -m src.gradcam으로 재생성한다.

ViT 대상 층과 CLS 토큰 처리 참고
https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md
