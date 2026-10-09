import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from multitaskrec.model import NewTask
from universal_experiment import run as C


class BudgetTests(unittest.TestCase):
    def test_explicit_settings_preserve_original_training(self):
        torch.manual_seed(42)
        head = NewTask(6,4,[4],.01,torch.device('cpu'))
        cache = {k:torch.randn(8,d) for k,d in [('x',6),('g',4),('s0',4),('s1',4)]}
        cache['y'] = torch.tensor([0.,1.]*4)
        caches = {s:cache for s in ('train','val','test')}
        envs = [torch.randn(4),torch.randn(4)]
        with tempfile.TemporaryDirectory() as folder:
            a,b = Path(folder)/'a',Path(folder)/'b'
            a.mkdir(); b.mkdir()
            first = C.train_head(copy.deepcopy(head),caches,envs,'B',torch.device('cpu'),2,a)
            second = C.train_head(copy.deepcopy(head),caches,envs,'B',torch.device('cpu'),2,b,
                                  lr=C.P.LR,batch_size=C.P.BATCH_SIZE,patience=C.P.PATIENCE)
            np.testing.assert_array_equal(np.load(a/'B_predictions.npz')['test_p'],np.load(b/'B_predictions.npz')['test_p'])
            self.assertEqual(first['best_epoch'],second['best_epoch'])
            self.assertTrue(all(e['train_seconds']>=0 and e['val_seconds']>=0 for e in second['epochs']))

    def test_patience_three_stops_after_three_stale_epochs(self):
        torch.manual_seed(42)
        head = NewTask(6,4,[4],.01,torch.device('cpu'))
        cache = {k:torch.randn(8,d) for k,d in [('x',6),('g',4),('s0',4),('s1',4)]}
        cache['y'] = torch.tensor([0.,1.]*4)
        envs = [torch.randn(4),torch.randn(4)]
        with tempfile.TemporaryDirectory() as folder, patch.object(C,'evaluate',return_value=(.5,np.full(8,.5))):
            result = C.train_head(head,{s:cache for s in ('train','val','test')},envs,'B',torch.device('cpu'),20,
                                  Path(folder),lr=.001,batch_size=4,patience=3)
        self.assertEqual(len(result['epochs']),4)
        self.assertEqual(result['best_epoch'],1)
        self.assertEqual(result['epochs'][0]['steps'],2)
