"""Verify both ten-epoch runs and package their reports for local archiving."""
import argparse
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--archive', type=Path, default=Path('/content/clahe_10epochs_verified_results.zip'))
    args = parser.parse_args()
    directory = args.results / 'verified_10epochs'
    directory.mkdir(exist_ok=True)
    histories = {}
    checks = []
    archive_files = []
    for name, label in [('ViT_GPU_B0_seed42_10epochs', 'Baseline'),
                        ('ViT_GPU_CLAHE_seed42_10epochs', 'CLAHE')]:
        run = args.results / name
        complete = json.loads((run / 'completed.json').read_text())
        assert complete['epochs'] == 10, name
        history = pd.read_csv(run / 'history.csv')
        assert history.epoch.tolist() == list(range(1, 11)), name
        assert np.isfinite(history[['val_auroc', 'val_auprc', 'val_loss']].to_numpy()).all()
        best_row = history.loc[history.val_auroc.idxmax()]
        for checkpoint_name in ('last.pt', 'best.pt'):
            checkpoint_path = run / checkpoint_name
            assert checkpoint_path.stat().st_size > 300_000_000
            checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
            assert checkpoint['epoch'] == (10 if checkpoint_name == 'last.pt' else int(best_row.epoch))
            assert checkpoint['config']['clahe'] == (label == 'CLAHE')
            del checkpoint
        evaluation = run / 'val_evaluation'
        metrics = json.loads((evaluation / 'metrics.json').read_text())
        predictions = pd.read_csv(evaluation / 'predictions.csv')
        assert metrics['checkpoint_epoch'] == int(best_row.epoch)
        assert metrics['images'] == len(predictions) == 16872
        assert predictions.pneumonia.sum() == 218 and predictions.score.between(0, 1).all()
        assert abs(metrics['auroc'] - best_row.val_auroc) < 1e-6
        checks.append(dict(model=label, epochs=10, best_epoch=int(best_row.epoch),
                           val_auroc=metrics['auroc'], val_average_precision=metrics['average_precision'],
                           training_and_validation_minutes=history.seconds.sum()/60,
                           best_checkpoint_bytes=(run/'best.pt').stat().st_size,
                           last_checkpoint_bytes=(run/'last.pt').stat().st_size))
        histories[label] = history
        for relative in ('history.csv', 'config.json', 'completed.json',
                         'val_evaluation/metrics.json', 'val_evaluation/predictions.csv'):
            archive_files.append((run / relative, name + '/' + relative))
    comparison = pd.DataFrame(checks)
    comparison.to_csv(directory/'comparison.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, column, title in zip(axes, ['val_loss', 'val_auroc', 'val_auprc'],
                                 ['Validation BCE', 'Validation AUROC', 'Validation AP']):
        for label, history in histories.items():
            ax.plot(history.epoch, history[column], 'o-', label=label)
        ax.set(title=title, xlabel='Epoch', xticks=list(range(1, 11)))
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(directory/'learning_curves.png', dpi=170)
    verification = dict(status='VERIFIED', checks=checks,
                        checkpoint_location='Drive CXR_CLAHE/results/<run>/best.pt and last.pt',
                        weights_in_archive=False)
    (directory/'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    notes = ['# ViT CLAHE 10에포크 결과',
             '동일 GPU, 동일 데이터 목록과 학습 설정에서 CLAHE 유무를 비교했다.',
             '두 조건 모두 1~10에포크의 기록, last.pt의 epoch 10, 검증 AUROC 최고 에포크의 best.pt를 확인했다.',
             '검증 예측 16,872장과 양성 라벨 218장, 점수 범위와 선택 에포크의 AUROC 일치를 확인했다.',
             '모델 가중치는 Drive에 보존하며 이 보고서 ZIP에는 기록과 검증 예측을 포함한다.',
             'CPU 실행과 GPU 실행의 시간은 환경 차이를 고려해 별도로 해석한다.',
             '재부팅·중단 대기 시간은 history의 에포크 소요 시간 합계에 모두 포함되지 않을 수 있다.']
    for row in checks:
        notes.append(f"- {row['model']}: best epoch {row['best_epoch']}, AUROC {row['val_auroc']:.6f}, AP {row['val_average_precision']:.6f}, 학습·검증 합계 {row['training_and_validation_minutes']:.1f}분")
    (directory/'report.md').write_text(chr(10).join(notes)+chr(10),encoding='utf-8')
    for path in directory.iterdir():
        if path.is_file():
            archive_files.append((path, 'verified_10epochs/'+path.name))
    with zipfile.ZipFile(args.archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for source, target in archive_files:
            bundle.write(source, target)
    print(json.dumps(verification, indent=2), flush=True)
    print('Verified report archive:', args.archive, flush=True)


if __name__ == '__main__':
    main()
