import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from multitaskrec.model import MPTRec, NewTask
from universal_experiment.model import UniversalExpert, ResidualHead
from universal_experiment.paired_run import train_frozen_stage2


def state_hash(module):
    h = hashlib.sha256()
    for value in module.state_dict().values():
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Stage2TrainingTests(unittest.TestCase):
    def test_live_u_updates_and_base_stays_byte_identical(self):
        torch.manual_seed(34)
        base = MPTRec(2, {'c': 3}, 2, 3, [8, 4], [4], device=torch.device('cpu'))
        u = UniversalExpert([1, 2], rep_dim=4, hidden=8, seed=41)
        head = ResidualHead(NewTask(3, 4, [4], .0, torch.device('cpu')), 4)
        batch = (torch.arange(16) % 2, torch.arange(16) % 2,
                 torch.arange(16) % 2,
                 {'d': torch.randn(16), 'c': torch.randint(3, (16,))})
        spec = SimpleNamespace(dataset='census', loaders={k: [batch] for k in ('train', 'val', 'test')},
                               lr=.01, batch_size=16, P=SimpleNamespace(backbone_sha256=state_hash))
        before_base, before_u = state_hash(base), state_hash(u)
        with tempfile.TemporaryDirectory() as folder:
            result = train_frozen_stage2(spec, 'U', base, u, head, folder, 2, 5,
                                        torch.device('cpu'), trainable_u=True)
            self.assertEqual(before_base, state_hash(base))
            self.assertNotEqual(before_u, state_hash(u))
            self.assertTrue(Path(folder, 'U_stage2_universal.pt').exists())
            self.assertEqual(result['base_hash_before'], result['base_hash_after'])
            self.assertEqual(result['universal_hash_after'], state_hash(u))
            saved = torch.load(Path(folder, 'U_stage2_universal.pt'), weights_only=True)
            self.assertTrue(all(torch.equal(saved[k], u.state_dict()[k]) for k in saved))

    def test_cached_trainable_prediction_recomputes_u(self):
        from universal_experiment.paired_run import cached_prediction
        head = ResidualHead(NewTask(3, 4, [4], .0, torch.device('cpu')), 4)
        u = UniversalExpert([1, 2], rep_dim=4, hidden=8)
        batch = dict(x=torch.randn(16, 3), g=torch.randn(16, 4),
                     s0=torch.randn(16, 4), s1=torch.randn(16, 4))
        cached_prediction(head, batch, [torch.randn(4), torch.randn(4)], 'U', u).sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in u.encoder.parameters()))
