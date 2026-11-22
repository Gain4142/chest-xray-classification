import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.run_comparison import fingerprint
from src.experiment_config import ROOT, load_config, validate_training


class ExperimentConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'params.yaml'
        self.config = validate_training(load_config(ROOT/'params.yaml'))

    def write(self, config):
        self.path.write_text(yaml.safe_dump(config), encoding='utf-8')
        return validate_training(load_config(self.path))

    def test_shipped_configs_load_and_keep_fixed_cohort(self):
        example = validate_training(load_config(ROOT/'experiments/vit_lr_example.yaml'))
        self.assertEqual(self.config['epochs'], 8)
        self.assertEqual(example['lr'], 0.0001)
        self.assertEqual(example['data'], self.config['data'])

    def test_common_scientific_notation_is_accepted(self):
        self.config['lr'] = '1e-4'
        self.assertEqual(self.write(self.config)['lr'], 0.0001)

    def test_unknown_or_duplicate_keys_are_rejected(self):
        self.path.write_text('epochs: 8\nepochs: 10\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_config(self.path)
        self.config['learning_rtae'] = 0.1
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            self.write(self.config)

    def test_unsupported_augmentation_and_invalid_numbers_are_rejected(self):
        for field,value in [('augmentation',True), ('lr',float('nan')), ('weight_decay',-1),
                            ('pos_weight',0), ('batch_size',True), ('clahe','false')]:
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config[field] = value
                with self.assertRaises(ValueError):
                    self.write(config)

    def test_recipe_guard_allows_additional_seeds_but_not_changed_learning_rate(self):
        original = fingerprint(self.config)
        selected = {**self.config,'seeds':[123,2026],'models':['vit']}
        self.assertEqual(fingerprint(selected), original)
        self.assertNotEqual(fingerprint({**self.config,'lr':0.0001}), original)

    def test_json_configs_remain_supported(self):
        path = self.path.with_suffix('.json')
        path.write_text(json.dumps(self.config), encoding='utf-8')
        self.assertEqual(validate_training(load_config(path)), self.config)
        legacy = validate_training(load_config(ROOT/'experiments/baseline_comparison.json'))
        self.assertEqual(legacy['optimizer'], 'adam')
        self.assertIsNone(legacy['pos_weight'])


if __name__ == '__main__':
    unittest.main()
