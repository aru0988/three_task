"""Stage-2 NewTask 的 env_prior 契约（exp/stage2-attn-env-prior）。

- 默认 "learned"：与 master (fd75198) 的 NewTask.forward **逐位一致**（输出与梯度都一致）；
- "attn"：new_env_emb(x) = Σ_k W_k(x)·E_k —— W 复用既有 attention 权重，E_k 为 Stage-1 env_embs；
  该模式下 forward 不再引用 env_embedding_network 参数（保留该模块只为让两种模式的初始化随机流
  逐位对齐：同一 model seed 下共享子模块初始化完全相同，A/B 对照只差 env 向量来源）。
"""
import unittest

import torch
import torch.nn.functional as F

from multitaskrec.model import NewTask

INPUT_SIZE, REP_DIM, NUM_ENVS, BATCH = 8, 4, 2, 6
ENV_EMB_KEY = "env_embedding_network.weight"


def make_inputs(seed=7):
    """确定性输入：同 seed 两次调用逐位一致。"""
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(BATCH, INPUT_SIZE, generator=gen),
            torch.randn(BATCH, REP_DIM, generator=gen),
            [torch.randn(BATCH, REP_DIM, generator=gen) for _ in range(NUM_ENVS)],
            [torch.randn(REP_DIM, generator=gen) for _ in range(NUM_ENVS)])


def make_newtask(env_prior="learned", seed=11):
    """构造前定种 → 同 seed 两次构建逐位一致。"""
    torch.manual_seed(seed)
    return NewTask(input_size=INPUT_SIZE, rep_dim=REP_DIM, tower_dnn_hidden_units=(4, 2),
                   reg_dnn=3e-5, device=torch.device("cpu"), env_prior=env_prior)


def legacy_forward(newtask, dnn_input, gen_rep, spec_reps, env_embs):
    """master (fd75198) NewTask.forward 的逐行冻结副本 —— 默认路径的比对基准。"""
    exist_env_embs = torch.stack(env_embs, dim=1)
    new_env_emb = newtask.env_embedding_network(newtask.new_env_idx).squeeze(0)

    H_out = newtask.projection_network(dnn_input)
    W = torch.mm(H_out, exist_env_embs) / newtask.temperature
    W = F.softmax(W, dim=-1).unsqueeze(2)

    gate_out = newtask.gate_network(dnn_input).unsqueeze(dim=2)
    new_spec_rep = torch.matmul(torch.stack(spec_reps, dim=2), W).squeeze()
    env_aware_rep = new_spec_rep * new_env_emb
    all_reps = torch.stack([env_aware_rep, gen_rep], dim=2)
    fused_rep = torch.matmul(all_reps, gate_out).squeeze()

    output = newtask.tower_network(fused_rep)
    return output.squeeze()


def attn_reference(newtask, dnn_input, gen_rep, spec_reps, env_embs):
    """目标语义：new_env_emb(x) = Σ_k W_k(x)·E_k（W 与 spec 加权用的同一份 attention 权重）。"""
    exist_env_embs = torch.stack(env_embs, dim=1)                       # (rep_dim, K)
    W = torch.mm(newtask.projection_network(dnn_input), exist_env_embs) / newtask.temperature
    W = F.softmax(W, dim=-1)                                            # (B, K)
    new_env_emb = torch.mm(W, exist_env_embs.transpose(0, 1))           # (B, rep_dim)

    new_spec_rep = torch.matmul(torch.stack(spec_reps, dim=2), W.unsqueeze(2)).squeeze()
    gate_out = newtask.gate_network(dnn_input).unsqueeze(dim=2)
    all_reps = torch.stack([new_spec_rep * new_env_emb, gen_rep], dim=2)
    return newtask.tower_network(torch.matmul(all_reps, gate_out).squeeze()).squeeze()


def grads_of(module, output):
    """对 output.sum() 反传后的全参数梯度（先清零，避免上一轮累积）。"""
    module.zero_grad(set_to_none=True)
    output.sum().backward()
    return {name: param.grad.clone() for name, param in module.named_parameters()
            if param.grad is not None}


class TestLearnedDefaultUnchanged(unittest.TestCase):
    """默认路径 = master 行为：不传 env_prior 也要能构造，且输出/梯度逐位一致。"""

    def test_default_arg_keeps_learned_env_embedding(self):
        torch.manual_seed(11)
        newtask = NewTask(input_size=INPUT_SIZE, rep_dim=REP_DIM, tower_dnn_hidden_units=(4, 2),
                          reg_dnn=3e-5, device=torch.device("cpu"))      # 不传 env_prior
        self.assertEqual(newtask.env_prior, "learned")
        params = dict(newtask.named_parameters())
        self.assertEqual(tuple(params[ENV_EMB_KEY].shape), (1, REP_DIM))
        self.assertIn("new_env_idx", dict(newtask.named_buffers()))

    def test_learned_forward_bit_identical_to_master_reference(self):
        newtask = make_newtask("learned")
        inputs = make_inputs()
        torch.testing.assert_close(newtask(*inputs), legacy_forward(newtask, *inputs), rtol=0, atol=0)

    def test_learned_backward_bit_identical_to_master_reference(self):
        newtask = make_newtask("learned")
        inputs = make_inputs()
        ours = grads_of(newtask, newtask(*inputs))
        reference = grads_of(newtask, legacy_forward(newtask, *inputs))
        self.assertEqual(set(ours), set(reference))
        self.assertIn(ENV_EMB_KEY, ours)                                  # learned 模式该参数必须吃梯度
        for name in ours:
            torch.testing.assert_close(ours[name], reference[name], rtol=0, atol=0)


class TestAttnEnvPrior(unittest.TestCase):
    """attn 路径：无新参数依赖、公式正确、梯度/形状正确。"""

    def test_attn_forward_equals_attention_weighted_env_sum(self):
        newtask = make_newtask("attn")
        inputs = make_inputs()
        out = newtask(*inputs)
        self.assertEqual(tuple(out.shape), (BATCH,))
        torch.testing.assert_close(out, attn_reference(newtask, *inputs), rtol=1e-6, atol=1e-7)

    def test_attn_output_independent_of_env_embedding_params(self):
        newtask = make_newtask("attn")
        inputs = make_inputs()
        before = newtask(*inputs)
        with torch.no_grad():                                             # 大幅扰动仍不得影响输出
            newtask.env_embedding_network.weight.add_(1.0)
        torch.testing.assert_close(newtask(*inputs), before, rtol=0, atol=0)

    def test_attn_backward_grads_shapes_and_env_dependence(self):
        newtask = make_newtask("attn")
        dnn_input, gen_rep, spec_reps, env_embs = make_inputs()
        env_embs = [emb.clone().requires_grad_(True) for emb in env_embs]
        out = newtask(dnn_input, gen_rep, spec_reps, env_embs)
        newtask.zero_grad(set_to_none=True)
        F.binary_cross_entropy(out, torch.ones(BATCH)).backward()
        for name, param in newtask.named_parameters():
            if name == ENV_EMB_KEY:
                self.assertIsNone(param.grad, "attn 模式不引用该参数，梯度必须为 None")
                continue
            self.assertIsNotNone(param.grad, f"{name} 应收到梯度")
            self.assertEqual(tuple(param.grad.shape), tuple(param.shape))
            self.assertTrue(bool(torch.isfinite(param.grad).all()))
        for i, emb in enumerate(env_embs):                                # E_k 直接进入 new_env_emb
            self.assertIsNotNone(emb.grad, f"E_{i} 应有梯度（attn 模式输出依赖 Stage-1 env_embs）")
            self.assertEqual(tuple(emb.grad.shape), (REP_DIM,))
            self.assertGreater(float(emb.grad.abs().sum()), 0.0)

    def test_attn_shares_identical_init_with_learned(self):
        """同 model seed 下两种模式的全部参数与 buffer 逐位一致 → A/B 只差 env 向量来源。"""
        learned, attn = make_newtask("learned"), make_newtask("attn")
        learned_params = dict(learned.named_parameters())
        attn_params = dict(attn.named_parameters())
        self.assertEqual(set(learned_params), set(attn_params))
        for name in learned_params:
            torch.testing.assert_close(attn_params[name], learned_params[name], rtol=0, atol=0)
        self.assertEqual(set(dict(learned.named_buffers())), set(dict(attn.named_buffers())))

    def test_attn_output_differs_from_learned_on_identical_weights(self):
        learned, attn = make_newtask("learned"), make_newtask("attn")
        inputs = make_inputs()
        self.assertFalse(torch.allclose(learned(*inputs), attn(*inputs)))

    def test_unknown_env_prior_rejected(self):
        with self.assertRaises(ValueError):
            NewTask(input_size=INPUT_SIZE, rep_dim=REP_DIM, tower_dnn_hidden_units=(4, 2),
                    reg_dnn=3e-5, device=torch.device("cpu"), env_prior="bogus")


if __name__ == "__main__":
    unittest.main()
