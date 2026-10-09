import unittest
import torch
from multitaskrec.model import NewTask


class UniversalTests(unittest.TestCase):
    def module(self):
        from universal_experiment.model import UniversalExpert
        return UniversalExpert([1, 2, 3], rep_dim=4, hidden=8, seed=41)

    def test_implementation_exists(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('universal_experiment.model'))

    def test_initialization_preserves_rng(self):
        state = torch.random.get_rng_state().clone()
        self.module()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_whole_field_mask_and_detached_gradients(self):
        u = self.module()
        x = torch.randn(16, 6, requires_grad=True)
        g = torch.randn(16, 4, requires_grad=True)
        mask = u.sample_mask(16)
        expanded = u.expand_mask(mask)
        self.assertTrue(torch.equal(expanded[:, 1], expanded[:, 2]))
        self.assertTrue(torch.equal(expanded[:, 3], expanded[:, 5]))
        self.assertTrue((mask.sum(1) == 1).all())
        losses = u.losses(x, g, mask)
        losses['total'].backward()
        self.assertIsNone(x.grad)
        self.assertIsNone(g.grad)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in u.parameters() if p.grad is not None), 0)

    def test_normalizer_frozen_and_masked_values_hidden(self):
        u = self.module()
        x = torch.randn(16, 6)
        u.set_normalizer(x.mean(0), x.std(0, unbiased=False))
        before = u.mean.clone()
        mask = torch.zeros(16, 3, dtype=torch.bool)
        mask[:, 1] = True
        alt = x.clone(); alt[:, 1:3] += 100
        self.assertTrue(torch.equal(u.masked_input(x, mask), u.masked_input(alt, mask)))
        u.losses(alt, torch.randn(16, 4), mask)
        self.assertTrue(torch.equal(before, u.mean))

    def test_residual_zero_matches_original_and_single_row(self):
        from universal_experiment.model import ResidualHead
        head = NewTask(6, 4, [4], .01, torch.device('cpu'))
        r = ResidualHead(head, 4)
        args = (torch.randn(8,6), torch.randn(8,4), [torch.randn(8,4) for _ in range(2)], [torch.randn(4) for _ in range(2)])
        self.assertTrue(torch.equal(head(*args), r(*args, extra=torch.zeros(8,4))))
        out = r(args[0][:1],args[1][:1],[s[:1] for s in args[2]],args[3],extra=torch.zeros(1,4))
        self.assertEqual(out.shape, torch.Size([1]))

    def test_old_task_gradients_unchanged_with_ssl(self):
        import copy
        from multitaskrec.model import MPTRec
        from universal_experiment.model import UniversalExpert, UniversalStage1
        torch.manual_seed(12)
        base = MPTRec(2, {'c': 3}, 2, 3, [8, 4], [4], device=torch.device('cpu'))
        plain = copy.deepcopy(base)
        wrapped = UniversalStage1(base, UniversalExpert([1, 2], rep_dim=4, hidden=8))
        x = {'d': torch.randn(16), 'c': torch.randint(3, (16,))}
        y = torch.randint(2, (16,)).float()
        for m in (plain, wrapped):
            out = m(x)
            loss = sum(torch.nn.functional.binary_cross_entropy(p, y)
                       for k in ('gen_preds', 'fused_preds') for p in out[k]) + m.get_l2_reg()
            loss.backward()
        for a, b in zip(plain.parameters(), base.parameters()):
            if a.grad is None:
                self.assertIsNone(b.grad)
            else:
                self.assertTrue(torch.equal(a.grad, b.grad))


if __name__ == '__main__':
    unittest.main()
