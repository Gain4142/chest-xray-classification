import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_gpu_data


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        data = {}
        for split, name, patient in [('train','train.png','1'), ('val','val.png','2')]:
            content = f'Image Index,Patient ID,pneumonia\n{name},{patient},1\n'.encode()
            path = self.root / f'experiments/splits/{split}.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            data[split] = {'source':f'experiments/splits/{split}.csv',
                           'path':f'data/processed/reduced_train_full_eval_qc/{split}.csv',
                           'images':1, 'positive_images':1, 'sha256':hashlib.sha256(content).hexdigest()}
        self.config = self.root / 'experiment.json'
        self.config.write_text(json.dumps({'data':data}))
        self.archive = self.root / 'images.zip'
        with zipfile.ZipFile(self.archive, 'w') as archive:
            archive.writestr('data/raw/images/train.png', b'train image')
            archive.writestr('data/raw/images/val.png', b'validation image')
            archive.writestr('src/train.py', 'obsolete code')
            archive.writestr('../outside.txt', 'unexpected file')

    def prepare(self, archive=None):
        with patch.object(prepare_gpu_data, 'ROOT', self.root), contextlib.redirect_stdout(io.StringIO()):
            return prepare_gpu_data.prepare(self.config, archive)

    def test_extracts_images_without_overwriting_source_or_extracting_other_entries(self):
        source = self.root / 'src/train.py'
        source.parent.mkdir()
        source.write_text('current code')
        self.prepare(self.archive)
        self.assertEqual(source.read_text(), 'current code')
        self.assertEqual((self.root/'data/raw/images/val.png').read_bytes(), b'validation image')
        self.assertFalse((self.root.parent/'outside.txt').exists())

    def test_conflicting_existing_split_is_preserved(self):
        path = self.root/'data/processed/reduced_train_full_eval_qc/train.csv'
        path.parent.mkdir(parents=True)
        path.write_text('another cohort')
        with self.assertRaisesRegex(AssertionError, 'not overwritten'):
            self.prepare(self.archive)
        self.assertEqual(path.read_text(), 'another cohort')

    def test_missing_images_do_not_silently_pass(self):
        with self.assertRaisesRegex(FileNotFoundError, '2 images missing'):
            self.prepare()


if __name__ == '__main__':
    unittest.main()
