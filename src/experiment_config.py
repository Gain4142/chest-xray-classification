"""Load experiment YAML/JSON safely and reject misspelled or unsupported settings."""
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {'optimizer':'adam', 'weight_decay':0.0, 'pos_weight':None,
            'device':'cuda', 'output_root':'outputs/models'}
KEYS = {'experiment_id','models','epochs','batch_size','lr','threads','seeds','planned_seeds',
        'selection_metric','clahe','vit_last_block','augmentation','data','data_config', *DEFAULTS}


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f'Duplicate YAML key: {key}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def load_config(path, root=ROOT):
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() in ('.yaml','.yml'):
        config = yaml.load(text, Loader=UniqueLoader)
    elif path.suffix.lower() == '.json':
        config = json.loads(text)
    else:
        raise ValueError('Use a .yaml, .yml, or .json configuration file')
    if not isinstance(config, dict):
        raise ValueError('Experiment configuration must be a mapping')
    unknown = set(config) - KEYS
    if unknown:
        raise ValueError(f'Unknown settings: {sorted(unknown)}')
    if 'data_config' in config:
        if 'data' in config:
            raise ValueError('Use data or data_config, not both')
        config['data'] = json.loads((Path(root) / config.pop('data_config')).read_text(encoding='utf-8-sig'))
    if 'data' not in config:
        raise ValueError('Data split settings are required')
    return {**DEFAULTS, **config}


def validate_training(config):
    required = {'experiment_id','models','epochs','batch_size','lr','threads','seeds','selection_metric'}
    missing = required - set(config)
    if missing:
        raise ValueError(f'Missing training settings: {sorted(missing)}')
    for key in ('epochs','batch_size','threads'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('lr','weight_decay','pos_weight'):
        value = config[key]
        if key == 'pos_weight' and value is None:
            continue
        try:
            number = float(value)
        except (ValueError, TypeError):
            raise ValueError(f'{key} must be a finite number') from None
        if isinstance(value, bool) or not math.isfinite(number) or (number < 0 if key == 'weight_decay' else number <= 0):
            raise ValueError(f'Invalid {key}: {value}')
        config[key] = number
    for key in ('clahe','vit_last_block','augmentation'):
        if type(config.get(key, False)) is not bool:
            raise ValueError(f'{key} must be true or false')
    if config.get('augmentation', False):
        raise ValueError('Training augmentation is not implemented in this runner yet')
    if config['optimizer'] not in ('adam','adamw'):
        raise ValueError('optimizer must be adam or adamw')
    if config['selection_metric'] not in ('auroc','auprc'):
        raise ValueError('selection_metric must be auroc or auprc')
    if config['device'] not in ('cuda','cpu'):
        raise ValueError('device must be cuda or cpu')
    if not isinstance(config['output_root'], str) or not config['output_root']:
        raise ValueError('output_root must be a nonempty path string')
    models = config['models']
    seeds = config['seeds']
    if not isinstance(models, list) or not models or any(m not in ('densenet','vit') for m in models) or len(set(models)) != len(models):
        raise ValueError('models must be a unique list of densenet/vit')
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be unique integers in [0, 2**32)')
    name = config['experiment_id']
    if not isinstance(name, str) or not name or name in ('.','..') or any(c in name for c in '/\\:*?"<>|'):
        raise ValueError('experiment_id must be a valid directory name')
    return config
