"""Summarize fixed-threshold ViT errors by view and record qualitative examples."""
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from src.train import ROOT

run = ROOT / 'outputs/models/ViT_baseline_seed42_20261115'
threshold = json.loads((run / 'threshold_analysis/metrics.json').read_text())['threshold']
predictions = pd.read_csv(run / 'test_evaluation/predictions.csv')
metadata = pd.read_csv(ROOT / 'data/raw/Data_Entry_2017.csv')
columns = ['Image Index', 'View Position', 'Finding Labels']
table = predictions.merge(metadata[columns], on='Image Index', validate='one_to_one', how='left')
assert len(table) == 16725 and table['View Position'].notna().all()
rows = []
for view, subset in [('All', table), *list(table.groupby('View Position'))]:
    y = subset.pneumonia
    tn, fp, fn, tp = map(int, confusion_matrix(y, subset.score >= threshold, labels=[0, 1]).ravel())
    rows.append(dict(view=view, images=len(subset), positives=int(y.sum()),
                     tp=tp, tn=tn, fp=fp, fn=fn,
                     sensitivity=tp/(tp+fn), specificity=tn/(tn+fp),
                     precision=tp/(tp+fp), auroc=roc_auc_score(y, subset.score),
                     average_precision=average_precision_score(y, subset.score)))
summary = pd.DataFrame(rows)
out = ROOT / 'outputs/metrics/error_analysis'
out.mkdir(parents=True, exist_ok=True)
summary.to_csv(out / 'vit_view_metrics.csv', index=False)
cases = pd.read_csv(ROOT / 'outputs/metrics/gradcam/vit_test_cases.csv')
cases = cases[cases.group.isin(['FP', 'FN'])].merge(metadata[columns], on='Image Index', validate='one_to_one')
# Korean observations describe visible overlays, without assigning causation.
observations = {
    '00012010_037.png': '여러 선 모양 기구가 보임. 흉부 내부와 영상 주변 모두 강조됨.',
    '00005850_003.png': '측면 문자 표식이 보임. 상단 경계·목·어깨 주변과 흉부 내부가 강조됨.',
    '00021216_005.png': '강조가 주로 횡격막 주변과 상복부 쪽에 분포함.',
    '00025769_024.png': '강한 강조가 영상 상단 경계와 목·어깨 주변에 분포함.',
}
cases['visual_observation'] = cases['Image Index'].map(observations)
cases['review_scope'] = '224-pixel overlay review; clinical lesion correspondence and original-resolution image quality unverified'
cases.to_csv(out / 'vit_fp_fn_review.csv', index=False)
print(summary.to_string(index=False))
print(cases[['Image Index','group','View Position','Finding Labels']].to_string(index=False))
