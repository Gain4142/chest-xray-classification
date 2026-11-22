"""Run versioned CXR comparisons from Jupyter or a terminal using repository code."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from scripts.prepare_gpu_data import ROOT, prepare


def fingerprint(config):
    return {'config':config, 'training_sources':{
        path.name:hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / 'src').glob('*.py'))}}


def summarize(config, output):
    import pandas as pd
    import torch
    rows = []
    for complete_path in sorted(output.glob('*/completed.json')):
        run = complete_path.parent
        complete = json.loads(complete_path.read_text())
        saved = json.loads((run / 'config.json').read_text())
        assert complete['epochs'] == config['epochs']
        assert saved['selection_metric'] == config['selection_metric']
        history = pd.read_csv(run / 'history.csv')
        assert history.epoch.tolist() == list(range(1, config['epochs'] + 1))
        selected = history.loc[history[f"val_{config['selection_metric']}"].idxmax()]
        checkpoint = torch.load(run / 'best.pt', weights_only=True, map_location='cpu')
        assert checkpoint['epoch'] == int(selected.epoch)
        for metric in ('auroc','auprc'):
            assert abs(checkpoint['metrics'][f'val_{metric}'] - selected[f'val_{metric}']) < 1e-12
        del checkpoint
        last = torch.load(run / 'last.pt', weights_only=True, map_location='cpu')
        assert last['epoch'] == config['epochs']
        del last
        rows.append({'model':saved['model'], 'seed':saved['seed'], 'best_epoch':int(selected.epoch),
                     'val_auroc':float(selected.val_auroc), 'val_average_precision':float(selected.val_auprc),
                     'training_and_validation_minutes':float(history.seconds.sum()/60),
                     'smoke_metrics_are_not_validation':saved['smoke']})
    if rows:
        frame = pd.DataFrame(rows).sort_values(['model','seed'])
        temporary = output / 'comparison.csv.tmp'
        frame.to_csv(temporary, index=False)
        temporary.replace(output / 'comparison.csv')
        print(frame.to_string(index=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/baseline_comparison.json')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--models', choices=('densenet','vit'), nargs='+')
    parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/models')
    parser.add_argument('--smoke', action='store_true', help='32 train images, one epoch; never validation/test performance')
    args = parser.parse_args(argv)
    config = prepare(args.config.resolve())
    if config.get('augmentation', False):
        parser.error('This runner does not implement augmentation yet; use an explicitly implemented experiment.')
    seeds = args.seeds or config['seeds']
    models = args.models or config['models']
    if not seeds or len(set(seeds)) != len(seeds) or not models or len(set(models)) != len(models):
        parser.error('Seeds and models must be nonempty and unique')
    experiment_id = config['experiment_id']
    if Path(experiment_id).name != experiment_id or experiment_id in ('.','..'):
        parser.error('experiment_id must be a directory name')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.chdir(ROOT)
    import torch
    if args.device == 'cuda':
        from scripts.check_gpu import check_gpu
        runtime = check_gpu()
    else:
        runtime = {'python':sys.version, 'platform':platform.platform(), 'torch':str(torch.__version__),
                   'torchvision':version('torchvision'), 'transformers':version('transformers'),
                   'numpy':version('numpy'), 'pandas':version('pandas'), 'sklearn':version('scikit-learn'),
                   'pillow':version('Pillow'), 'gpu':None}
    runtime['device'] = args.device
    output = args.output_root.resolve() / (experiment_id + ('_smoke' if args.smoke else ''))
    output.mkdir(parents=True, exist_ok=True)
    guard = fingerprint(config)
    guard['smoke'] = args.smoke
    guard_path = output / 'execution_manifest.json'
    if guard_path.exists():
        if json.loads(guard_path.read_text()) != guard:
            parser.error('Code/config changed. Create a new experiment_id or output root; existing results were not overwritten.')
    else:
        guard_path.write_text(json.dumps(guard, indent=2), encoding='utf-8')
    snapshot = output / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for source in (ROOT / 'src').glob('*.py'):
        if not (snapshot / source.name).exists():
            shutil.copyfile(source, snapshot / source.name)
    runtime_path = output / 'runtime.json'
    if runtime_path.exists():
        if json.loads(runtime_path.read_text()) != runtime:
            parser.error('Runtime/hardware changed. Use a new experiment_id or output root.')
    else:
        runtime_path.write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    with (output / 'environment.txt').open('w', encoding='utf-8') as env:
        subprocess.run([sys.executable,'-m','pip','freeze'], stdout=env, check=True)
    epochs = 1 if args.smoke else config['epochs']
    run_config = {**config, 'epochs':epochs}
    (output / 'invocation.json').write_text(json.dumps({'started_at':datetime.now().astimezone().isoformat(),
        'git_commit':commit, 'seeds':seeds, 'models':models}, indent=2), encoding='utf-8')
    for seed in seeds:
        for model in models:
            if {**fingerprint(config), 'smoke':args.smoke} != guard:
                raise RuntimeError('Source changed during execution. Stop updating code until the experiment finishes.')
            run_name = f'{model}_seed{seed}'
            run = output / run_name
            if (run / 'completed.json').exists():
                print(f'Already completed: {run_name}', flush=True)
                summarize(run_config, output)
                continue
            command = [sys.executable,'-u','-m','src.train','--model',model,
                       '--epochs',str(epochs),'--batch-size',str(config['batch_size']),
                       '--lr',str(config['lr']),'--threads',str(config['threads']), '--seed',str(seed),
                       '--selection-metric',config['selection_metric'],'--device',args.device,
                       '--run-name',run_name,'--output-root',str(output)]
            if args.smoke:
                command.append('--smoke')
            if config.get('clahe', False):
                command.append('--clahe')
            if model == 'vit' and config.get('vit_last_block', False):
                command.append('--vit-last-block')
            if run.exists():
                if not (run / 'last.pt').exists():
                    parser.error(f'Interrupted before first checkpoint: {run}. Move that incomplete run aside before retrying.')
                command.append('--resume')
            print('RUN', ' '.join(command), flush=True)
            with (output / f'{run_name}.log').open('a', encoding='utf-8') as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding='utf-8', errors='replace', bufsize=1,
                                           env={**os.environ, 'PYTHONIOENCODING':'utf-8'})
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    if process.wait() != 0:
                        raise RuntimeError(f'Training failed; see {output / (run_name + ".log")}')
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
                    raise
            summarize(run_config, output)
    print(f'Saved and verified results: {output}', flush=True)


if __name__ == '__main__':
    main()
