"""Check and archive the reports downloaded from the completed Colab runs."""
import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / 'outputs/metrics/clahe_comparison_10epochs'
    with zipfile.ZipFile(args.archive) as bundle:
        assert bundle.testzip() is None, 'Corrupt archive'
        verification = json.loads(bundle.read('verified_10epochs/verification.json'))
        assert verification['status'] == 'VERIFIED'
        assert {row['model'] for row in verification['checks']} == {'Baseline', 'CLAHE'}
        for name, label in [('ViT_GPU_B0_seed42_10epochs', 'Baseline'),
                            ('ViT_GPU_CLAHE_seed42_10epochs', 'CLAHE')]:
            def read_json(relative):
                return json.loads(bundle.read(f'{name}/{relative}'))

            def read_csv(relative):
                return list(csv.DictReader(io.StringIO(bundle.read(f'{name}/{relative}').decode('utf-8-sig'))))

            assert read_json('completed.json')['epochs'] == 10
            history = read_csv('history.csv')
            assert [int(row['epoch']) for row in history] == list(range(1, 11))
            best = max(history, key=lambda row: float(row['val_auroc']))
            metrics = read_json('val_evaluation/metrics.json')
            assert metrics['checkpoint_epoch'] == int(best['epoch'])
            predictions = read_csv('val_evaluation/predictions.csv')
            labels = [int(row['pneumonia']) for row in predictions]
            scores = [float(row['score']) for row in predictions]
            assert len(labels) == metrics['images'] == 16872 and sum(labels) == 218
            assert all(math.isfinite(x) and 0 <= x <= 1 for x in scores)
            assert abs(roc_auc_score(labels, scores) - metrics['auroc']) < 1e-9
            assert abs(average_precision_score(labels, scores) - metrics['average_precision']) < 1e-9
            check = next(row for row in verification['checks'] if row['model'] == label)
            assert check['epochs'] == 10 and check['best_epoch'] == int(best['epoch'])
            assert check['best_checkpoint_bytes'] > 300_000_000
            assert check['last_checkpoint_bytes'] > 300_000_000
        # Only extract known result members; never use arbitrary archive paths.
        for member in bundle.namelist():
            relative = Path(member)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert relative.parts[0] in {'verified_10epochs', 'ViT_GPU_B0_seed42_10epochs',
                                        'ViT_GPU_CLAHE_seed42_10epochs'}
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(member))
    figures = root / 'outputs/figures/training'
    figures.mkdir(parents=True, exist_ok=True)
    shutil.copy2(destination/'verified_10epochs/learning_curves.png', figures/'vit_clahe_10epochs.png')
    shutil.copy2(destination/'verified_10epochs/report.md', root/'docs/vit_clahe_10epochs.md')
    result = {'status': 'VERIFIED', 'verified_at': datetime.now().astimezone().isoformat(),
              'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
              'saved_reports': str(destination), 'checks': verification['checks'],
              'checkpoint_location': verification['checkpoint_location']}
    record = root/'outputs/colab/completion_and_shutdown.md'
    record.write_text('# Colab completion and shutdown\n\n'
                      'Both ten-epoch experiments completed. Drive checkpoint loading and epochs were verified remotely. '
                      'Downloaded histories and prediction metrics were independently verified locally.\n\n'
                      'The user authorized laptop shutdown after these checks. No forced application closure is used.\n\n'
                      '```json\n'+json.dumps(result, indent=2, ensure_ascii=False)+'\n```\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
