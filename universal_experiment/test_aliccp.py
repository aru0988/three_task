import copy
import importlib.util
import unittest
import torch
from multitaskrec.model import MPTRec
from universal_experiment.model import UniversalExpert, UniversalStage1


class AliTransferTests(unittest.TestCase):
    def test_runner_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('universal_experiment.aliccp'))

    def test_auxiliary_preserves_dropout_rng(self):
        base = MPTRec(2, {'c':3}, 2, 3, [8,4], [4], dropout=[.1,.3], device=torch.device('cpu'))
        wrapped = UniversalStage1(copy.deepcopy(base), UniversalExpert([1,2],rep_dim=4,hidden=8))
        x = {'d':torch.randn(16), 'c':torch.randint(3,(16,))}
        torch.manual_seed(10); base(x); reference = torch.get_rng_state()
        torch.manual_seed(10); wrapped(x)
        self.assertTrue(torch.equal(reference,torch.get_rng_state()))

    def test_aliccp_width(self):
        u = UniversalExpert([5]*16,rep_dim=64)
        self.assertEqual(tuple(u(torch.randn(4,80)).shape),(4,64))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA needed')
    def test_u_init_preserves_cuda_rng(self):
        torch.cuda.init(); torch.cuda.manual_seed_all(100)
        before=torch.cuda.get_rng_state()
        UniversalExpert([5]*16,rep_dim=64)
        self.assertTrue(torch.equal(before,torch.cuda.get_rng_state()))


if __name__=='__main__':
    unittest.main()
