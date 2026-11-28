import copy
import io
import unittest
import tempfile
import contextlib
from pathlib import Path
from unittest.mock import patch
import pandas as pd

import torch
from transformers import ViTConfig, ViTForImageClassification

from src.optimization import parameter_groups, build_scheduler
from src.vit_model import ViTBinaryClassifier
from src import train


def tiny(scope):
    network = ViTForImageClassification(ViTConfig(
        image_size=16, patch_size=8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_labels=1))
    return ViTBinaryClassifier(network, training_scope=scope)


class ViTOptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_scopes_and_llrd_cover_each_parameter_once(self):
        for scope in ('frozen','last_block','llrd'):
            model = tiny(scope)
            names = [n for n,p in model.named_parameters() if p.requires_grad]
            if scope == 'frozen':
                self.assertTrue(all('classifier' in n for n in names))
            elif scope == 'last_block':
                self.assertFalse(any('embeddings' in n or '.layer.0.' in n for n in names))
            else:
                self.assertEqual(len(names), len(list(model.named_parameters())))
                groups = parameter_groups(model, 0.001, 0, 0.75)
                self.assertEqual([g['lr'] for g in groups],
                                 [0.001*0.75**3,0.001*0.75**2,0.001*0.75,0.001])

    def test_warmup_peak_decay_and_group_ratios(self):
        params = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
        opt = torch.optim.Adam([{'params':[params[0]],'lr':0.001},
                                {'params':[params[1]],'lr':0.0001}])
        scheduler = build_scheduler(opt, 'warmup_cosine', 20, 0.1)
        rates = []
        for _ in range(20):
            rates.append([g['lr'] for g in opt.param_groups])
            sum(p.sum() for p in params).backward();opt.step();scheduler.step();opt.zero_grad()
        self.assertEqual(rates[0][0], 0)
        self.assertEqual(rates[2][0], 0.001)
        self.assertGreater(rates[2][0],rates[19][0])
        self.assertEqual(scheduler.get_last_lr(), [0,0])
        for a,b in rates[1:]: self.assertAlmostEqual(a/b,10)

    def test_scheduled_resume_exactly_matches_uninterrupted(self):
        model = tiny('llrd')
        opt = torch.optim.Adam(parameter_groups(model,.001,0,.75))
        sched = build_scheduler(opt,'warmup_cosine',12,.25)
        x=torch.randn(2,3,16,16);y=torch.tensor([[0.],[1.]])
        def update(m,o,s):
            m.train_selected_layers();o.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(m(x),y).backward()
            o.step();s.step()
        for _ in range(5):update(model,opt,sched)
        stream=io.BytesIO()
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict()},stream)
        for _ in range(7):update(model,opt,sched)
        stream.seek(0);state=torch.load(stream,weights_only=True)
        restored=tiny('llrd');restored.load_state_dict(state['model'])
        newopt=torch.optim.Adam(parameter_groups(restored,.001,0,.75))
        newsched=build_scheduler(newopt,'warmup_cosine',12,.25)
        newopt.load_state_dict(state['optimizer']);newsched.load_state_dict(state['scheduler'])
        for _ in range(7):update(restored,newopt,newsched)
        for name,p in model.state_dict().items():self.assertTrue(torch.equal(p,restored.state_dict()[name]),name)
        self.assertEqual(sched.state_dict(),newsched.state_dict())

    def test_updates_and_frozen_parameters_for_each_scope(self):
        x=torch.randn(2,3,16,16)
        for scope in ('frozen','last_block','llrd'):
            model=tiny(scope);before=copy.deepcopy(model.state_dict())
            opt=torch.optim.Adam(parameter_groups(model,.001,0,.75))
            model.train_selected_layers();model(x).sum().backward();opt.step()
            for name,p in model.named_parameters():
                if not p.requires_grad:self.assertTrue(torch.equal(before[name],p),name)
            self.assertFalse(torch.equal(before['network.classifier.weight'],model.classifier.weight))
            if scope=='llrd':
                name='network.vit.embeddings.patch_embeddings.projection.weight'
                self.assertFalse(torch.equal(before[name],model.state_dict()[name]))

    def test_training_entrypoint_checkpoint_resume_with_schedule(self):
        class Processor:
            def save_pretrained(self, output): pass
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, *args):
                self.images=torch.randn(32,3,16,16)
                self.table=pd.DataFrame({'Patient ID':list(range(32))})
            def __len__(self):return 32
            def __getitem__(self,i):return self.images[i],torch.tensor(float(i%2))
        def load(*args,**kwargs):return tiny(kwargs['training_scope']),Processor()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'data/processed/reduced_train_full_eval_qc/train.csv'
            manifest.parent.mkdir(parents=True);manifest.write_text('fixed training manifest')
            flags=['train','--model','vit','--vit-training','llrd','--scheduler','warmup_cosine',
                   '--warmup-ratio','0.25','--epochs','2','--lr','0.001','--threads','1',
                   '--smoke','--output-root',str(root/'runs')]
            original=train.save_checkpoint
            def interrupt(path,checkpoint):
                original(path,checkpoint)
                if path.name=='last.pt' and checkpoint['epoch']==1:raise RuntimeError('simulated interruption')
            with patch.object(train,'ROOT',root),patch.object(train,'ChestXrayDataset',Dataset), \
                 patch('src.vit_model.load_vit',load),contextlib.redirect_stdout(io.StringIO()):
                with patch('sys.argv',flags+['--run-name','full']):train.main()
                with patch('sys.argv',flags+['--run-name','resumed']),patch.object(train,'save_checkpoint',interrupt):
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'):train.main()
                with patch('sys.argv',flags+['--run-name','resumed','--resume']):train.main()
            full=torch.load(root/'runs/full/last.pt',weights_only=True)
            resumed=torch.load(root/'runs/resumed/last.pt',weights_only=True)
            for name,value in full['model_state_dict'].items():
                self.assertTrue(torch.equal(value,resumed['model_state_dict'][name]),name)
            self.assertEqual(full['scheduler_state_dict'],resumed['scheduler_state_dict'])


if __name__ == '__main__': unittest.main()
