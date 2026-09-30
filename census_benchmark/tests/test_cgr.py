"""忠实复现历史 CGR（Confidence-Gated Routing）的门控诊断测试。

唯一事实来源：docs/superpowers/experiments/2026-09-30-stage2-cgr-revalidation.md
历史实现（审计对象）：`archive/exploration` 分支的
  * `run_newtask_from_ckpt.py`（mode="cgr"、warmup_epochs=5、alpha=min(1, epoch/5)）
  * `multitaskrec/model.py` 的 `NewTask(fusion_mode="cgr")`（常量初始化 + 批级统计量门控输入）
  * `analysis/cgr_diagnosis.py` / `analysis/cgr_gradient_diagnosis.py`（历史诊断读数的口径来源）

本文件在实现之前写成（TDD：先失败），锁定七件事：
  1. **忠实公式**：W_attn / W_fw / g / W 与历史逐项同式（用手写复算逐位比对）
  2. **忠实构造**：历史构造顺序（含 RNG 混入——confidence_mlp 建在 projection 与 gate 之间）
  3. **忠实 warmup**：alpha = min(1, epoch/5)，逐 epoch 由 runner 写入（不进 state_dict）
  4. **结构性缺陷的证据**：g 是**批级常量**（批内唯一、跨批变化、同一样本换批即换 g）
  5. **指标口径**：批内 std / 唯一值占比 / |W−W_attn| L1 / 前 sigmoid 饱和 / 隐层行对称
  6. **预注册判据**：退化主导（unique_frac<=0.01 或 weight_l1<1e-6 或 grad_norm==0 ⇒ STOP），
     效应阈值 0.8521 仅在**非退化**时才解释
  7. **接线**：run_id 后缀 `-cgr`、config / metrics 记录；基线臂不受影响（逐位同 master NewTask）

定位：这不是新方法（置信门控 / 条件化路由是已有大类，历史 CGR 也从未超越 prompt），
而是一次**忠实复现 + 缺陷留痕**；失败结果按用户要求保留在本分支、永不合并 master。
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import DataLoader, Dataset, Subset

import run_census_benchmark
from census_benchmark import cgr as CGR
from census_benchmark import protocol as P
from config import CensusIncome_Vocabulary_Size
from multitaskrec.model import NewTask

# 与 test_protocol / test_smoke / test_cond_scale_bias 同口径：dnn_input = 2 特征 × embedding 4 = 8
TINY_VOCAB, TINY_INPUT_SIZE, TINY_EMBEDDING, TINY_REP_DIM = {"a": 3, "b": 2}, 8, 4, 4
SOURCE = Path(__file__).resolve().parents[1] / "cgr.py"


def tiny_backbone(device=None):
    """dnn_input 8 维、rep_dim 4、2 个 env（与其它测试文件同一夹具口径）。"""
    return run_census_benchmark.MPTRec(
        num_tasks=2, feature_vocabulary=dict(TINY_VOCAB), embedding_size=TINY_EMBEDDING,
        input_size=TINY_INPUT_SIZE, expert_dnn_hidden_units=(8, TINY_REP_DIM),
        tower_dnn_hidden_units=(4, 2), device=device)


def tiny_newtask(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM, device=None):
    """基线头（不得改动）。"""
    return NewTask(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=list(P.TOWER_HIDDEN),
                   reg_dnn=P.REG_DNN, device=device)


def tiny_variant(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM, device=None):
    """处理臂头：忠实复现的 CGR（子类，不动 model.py）。"""
    return CGR.CGRNewTask(input_size=input_size, rep_dim=rep_dim,
                          tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN, device=device)


def tiny_reps(device=None, n=8, seed=3):
    """抽一批表征：(dnn_input, gen_rep, spec_reps, env_embs)——与真实路径同款 no_grad 抽取。"""
    gen = torch.Generator().manual_seed(seed)
    features = {"a": torch.randint(0, TINY_VOCAB["a"], (n,), generator=gen),
                "b": torch.randint(0, TINY_VOCAB["b"], (n,), generator=gen)}
    with torch.no_grad():
        return tiny_backbone(device).get_infos(features)


def real_scale_reps(n=8, seed=5, device=None):
    """真实量级夹具：真 vocab、input_size=123、rep_dim=128（只用于饱和读数，不做逐位断言）。

    CensusIncome 的 123 维 = 28 个类目特征 × embedding 4 + 11 个 dense 特征——dense 列不在 vocab 里，
    由 `EmbeddingNetwork` 直接拼接（`x[name].unsqueeze(1)`），故这里按同一口径补上。
    """
    vocabulary = CensusIncome_Vocabulary_Size.copy()
    vocabulary.pop("education")
    backbone = run_census_benchmark.MPTRec(
        num_tasks=2, feature_vocabulary=vocabulary, embedding_size=P.EMBEDDING_SIZE,
        input_size=P.INPUT_SIZE, expert_dnn_hidden_units=list(P.EXPERT_HIDDEN),
        tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_embedding=P.REG_EMBEDDING,
        reg_dnn=P.REG_DNN, device=device)
    gen = torch.Generator().manual_seed(seed)
    features = {name: torch.randint(0, size, (n,), generator=gen) for name, size in vocabulary.items()}
    n_dense = P.INPUT_SIZE - len(vocabulary) * P.EMBEDDING_SIZE
    features.update({f"dense_{i}": torch.rand(n, generator=gen) for i in range(n_dense)})
    with torch.no_grad():
        return backbone.get_infos(features)


def live_confidence_mlp(head, seed=11):
    """把置信 MLP 置成**确定性的、可感的**门控，仅用于"门控确实随输入变化"的结构性断言。

    两点必要调整（不改历史实现本身，只改夹具）：
      * 常量初始化下 32 个隐层单元同进同出，可能整体落在 ReLU 死区 → 首层换成显式随机权重；
      * 历史 `bias = 2.0` 把 sigmoid 推到饱和（z≈2~7 时导数只剩 1e-2 量级）→ 这里取 0，
        否则"批内扰动 → g 变化"会被饱和压到浮点噪声量级，断言不可靠。
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        head.confidence_mlp[0].weight.copy_(
            torch.randn(head.confidence_mlp[0].weight.shape, generator=gen) * 0.5)
        head.confidence_mlp[0].bias.copy_(
            torch.randn(head.confidence_mlp[0].bias.shape, generator=gen) * 0.5)
        head.confidence_mlp[2].bias.zero_()
    return head


class TinyCensus(Dataset):
    """极小 CensusIncome 形状：(income, marital, new_task, features)。"""

    def __init__(self, n, seed):
        gen = torch.Generator().manual_seed(seed)
        self.features = {"a": torch.randint(0, 3, (n,), generator=gen),
                         "b": torch.randint(0, 2, (n,), generator=gen)}
        self.labels = [torch.randint(0, 2, (n,), generator=gen).float() for _ in range(3)]

    def __len__(self):
        return int(self.features["a"].shape[0])

    def __getitem__(self, i):
        return (self.labels[0][i], self.labels[1][i], self.labels[2][i],
                {"a": self.features["a"][i], "b": self.features["b"][i]})


def tiny_inputs(n_train=64, n_val=16, n_test=16, batch_size=16):
    train_ds, test_ds = TinyCensus(n_train, 1), TinyCensus(n_val + n_test, 2)
    val_idx, test_idx = P.make_split(len(test_ds), P.SPLIT_SEED)
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size),
               "val": DataLoader(Subset(test_ds, val_idx.tolist()), batch_size=batch_size),
               "test": DataLoader(Subset(test_ds, test_idx.tolist()), batch_size=batch_size)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=len(train_ds)), (val_idx, test_idx)


def run_tiny_stage1(root, device):
    loaders, stats, indices = tiny_inputs()
    stage1 = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_backbone(device),
                                             loaders=loaders, stats=stats, indices=indices, tag="short")
    return stage1, loaders, stats, indices


class TestModuleSourceCompiles(unittest.TestCase):
    """静态守卫：源码必须能被本解释器编译（最便宜的用例，先跑它；venv 是 3.10）。"""

    def test_source_compiles(self):
        self.assertTrue(SOURCE.is_file(), f"未找到被测源码: {SOURCE}")
        compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec")


class TestPreregisteredCriteria(unittest.TestCase):
    """预注册判据（只允许在看到结果之前修改）。"""

    def test_threshold_values(self):
        self.assertAlmostEqual(CGR.AUC_TEST_MIN, 0.8521, places=12)
        self.assertAlmostEqual(CGR.UNIQUE_FRAC_MAX, 0.01, places=12)
        self.assertAlmostEqual(CGR.WEIGHT_L1_MIN, 1e-6, places=18)
        self.assertAlmostEqual(CGR.GRAD_NORM_MIN, 0.0, places=12)
        self.assertAlmostEqual(CGR.PRESIGMOID_SATURATION_ABS, 8.0, places=12)
        self.assertEqual(CGR.WARMUP_EPOCHS, 5)                    # 历史常量
        self.assertEqual(CGR.TEMPERATURE, 150)                    # 历史常量（= NewTask.temperature）
        self.assertEqual(CGR.HIDDEN_UNITS, 32)                    # 历史常量（confidence_mlp 隐层）
        self.assertEqual(CGR.VARIANT, "cgr")
        self.assertEqual(CGR.BASELINE_VARIANT, "baseline")
        self.assertEqual(CGR.RUN_ID_SUFFIX, "-cgr")
        self.assertEqual(tuple(CGR.VARIANTS), (CGR.BASELINE_VARIANT, CGR.VARIANT))

    def test_degeneracy_rule_boundaries(self):
        fine = CGR.degeneracy_verdict(unique_frac=0.5, weight_l1=0.1, grad_norm=1e-3)
        self.assertEqual(fine["status"], "NOT_DEGENERATE")
        self.assertEqual(fine["triggered_rules"], [])

        at_bound = CGR.degeneracy_verdict(unique_frac=0.01, weight_l1=0.1, grad_norm=1e-3)
        self.assertEqual(at_bound["status"], "CONFIRMED_DEGENERATE")          # <= 含等号
        just_above = CGR.degeneracy_verdict(unique_frac=0.0100001, weight_l1=0.1, grad_norm=1e-3)
        self.assertEqual(just_above["status"], "NOT_DEGENERATE")

        self.assertEqual(CGR.degeneracy_verdict(unique_frac=0.5, weight_l1=1e-7,
                                                grad_norm=1e-3)["status"], "CONFIRMED_DEGENERATE")
        self.assertEqual(CGR.degeneracy_verdict(unique_frac=0.5, weight_l1=1e-6,
                                                grad_norm=1e-3)["status"], "NOT_DEGENERATE")  # < 严格
        self.assertEqual(CGR.degeneracy_verdict(unique_frac=0.5, weight_l1=0.1,
                                                grad_norm=0.0)["status"], "CONFIRMED_DEGENERATE")
        self.assertEqual(CGR.degeneracy_verdict(unique_frac=0.5, weight_l1=0.1,
                                                grad_norm=1e-12)["status"], "NOT_DEGENERATE")

    def test_degeneracy_reports_every_triggered_rule(self):
        verdict = CGR.degeneracy_verdict(unique_frac=0.001, weight_l1=0.0, grad_norm=0.0)
        self.assertEqual(verdict["triggered_rules"],
                         ["unique_frac <= 0.01", "weight_l1 < 1e-6", "grad_norm == 0"])
        self.assertEqual(sorted(verdict["checks"]),
                         ["grad_norm_eq_0", "unique_frac_le_0.01", "weight_l1_lt_1e-6"])
        self.assertAlmostEqual(verdict["checks"]["unique_frac_le_0.01"]["observed"], 0.001, places=12)
        self.assertAlmostEqual(verdict["checks"]["weight_l1_lt_1e-6"]["observed"], 0.0, places=12)
        self.assertAlmostEqual(verdict["checks"]["grad_norm_eq_0"]["observed"], 0.0, places=12)

    def test_arm_verdict_stops_when_degenerate(self):
        """退化是主导判据：即便 test AUC 再高也不解释效应阈值（预注册）。"""
        degeneracy = CGR.degeneracy_verdict(unique_frac=0.001, weight_l1=0.5, grad_norm=1e-3)
        arm = CGR.arm_verdict(0.99, degeneracy=degeneracy)
        self.assertEqual(arm["status"], "STOP")
        self.assertFalse(arm["pass"])
        self.assertFalse(arm["effect"]["evaluated"])
        self.assertIsNone(arm["effect"]["pass"])
        self.assertAlmostEqual(arm["effect"]["observed"], 0.99, places=12)

    def test_arm_verdict_effect_only_when_not_degenerate(self):
        ok = CGR.degeneracy_verdict(unique_frac=0.5, weight_l1=0.1, grad_norm=1e-3)
        self.assertEqual(CGR.arm_verdict(0.8521, degeneracy=ok)["status"], "EFFECT_CONFIRMED")   # 含等号
        self.assertTrue(CGR.arm_verdict(0.8521, degeneracy=ok)["pass"])
        self.assertEqual(CGR.arm_verdict(0.850069, degeneracy=ok)["status"], "EFFECT_NOT_CONFIRMED")
        self.assertFalse(CGR.arm_verdict(0.850069, degeneracy=ok)["pass"])
        self.assertTrue(CGR.arm_verdict(0.850069, degeneracy=ok)["effect"]["evaluated"])

    def test_arm_and_degeneracy_payloads_are_json_serialisable(self):
        degeneracy = CGR.degeneracy_verdict(unique_frac=0.001, weight_l1=0.0, grad_norm=0.0)
        arm = CGR.arm_verdict(0.86, degeneracy=degeneracy)
        self.assertEqual(arm["effect"]["rule"], "auc_test_education >= 0.8521")
        self.assertAlmostEqual(arm["effect"]["threshold"], 0.8521, places=12)
        json.dumps(arm)
        json.dumps(CGR.preregistered_criteria())
        json.dumps(CGR.provenance())


class TestFaithfulConstruction(unittest.TestCase):
    """忠实构造：历史顺序 + RNG 混入（这是历史对比里被审计出来的缺陷，必须原样复现）。"""

    def test_constant_init_values_match_history(self):
        head = tiny_variant()
        mlp = head.confidence_mlp
        self.assertIsInstance(mlp, torch.nn.Sequential)
        self.assertEqual(tuple(mlp[0].weight.shape), (CGR.HIDDEN_UNITS, 5 * TINY_REP_DIM))
        self.assertEqual(tuple(mlp[2].weight.shape), (1, CGR.HIDDEN_UNITS))
        self.assertTrue(torch.equal(mlp[0].weight, torch.full_like(mlp[0].weight, 0.01)))
        self.assertTrue(torch.equal(mlp[0].bias, torch.full_like(mlp[0].bias, 0.01)))
        self.assertTrue(torch.equal(mlp[2].weight, torch.full_like(mlp[2].weight, 0.1)))
        self.assertTrue(torch.equal(mlp[2].bias, torch.full_like(mlp[2].bias, 2.0)))
        self.assertAlmostEqual(head.temperature, 150.0, places=12)
        self.assertAlmostEqual(head.gate_warmup_alpha, 1.0, places=12)   # 历史默认值

    def test_rng_construction_order_confound_is_reproduced(self):
        """历史把 confidence_mlp 建在 projection 与 gate 之间 → 同 seed 下 gate/tower 初始化与基线**不同**。"""
        device = torch.device("cpu")
        P.seed_model(P.MODEL_SEED)
        baseline = tiny_newtask(device=device)
        rng_after_baseline = torch.get_rng_state().clone()

        P.seed_model(P.MODEL_SEED)
        variant = tiny_variant(device=device)

        self.assertFalse(torch.equal(torch.get_rng_state(), rng_after_baseline),
                         "历史实现确实多消耗了全局 RNG——构造顺序必须原样保留")
        got, want = variant.state_dict(), baseline.state_dict()
        for name in ("env_embedding_network.weight", "projection_network.0.weight",
                     "projection_network.2.weight"):
            self.assertTrue(torch.equal(got[name], want[name]), f"多抽之前的模块应与基线逐位相同: {name}")
        for name in ("gate_network.0.weight", "tower_network.mlp.linear0.weight",
                     "tower_network.mlp.linear2.weight"):
            self.assertFalse(torch.equal(got[name], want[name]),
                             f"多抽之后的模块必须体现 RNG 混入（历史缺陷留痕）: {name}")

    def test_shared_module_structure_matches_master_newtask(self):
        """防漂移：除 confidence_mlp 外，子模块名与形状必须与 master NewTask 完全一致。"""
        mine = {name: tuple(t.shape) for name, t in tiny_variant().state_dict().items()}
        master = {name: tuple(t.shape) for name, t in tiny_newtask().state_dict().items()}
        extra = sorted(set(mine) - set(master))
        self.assertEqual(extra, ["confidence_mlp.0.bias", "confidence_mlp.0.weight",
                                 "confidence_mlp.2.bias", "confidence_mlp.2.weight"])
        self.assertEqual(sorted(set(master) - set(mine)), [])
        for name in sorted(master):
            self.assertEqual(mine[name], master[name], f"共享张量形状漂移: {name}")

    def test_hidden_rows_are_tied_at_init(self):
        """历史常量初始化把 32 个隐层单元设成完全相同（对称性缺陷的起点）。"""
        symmetry = CGR.hidden_row_symmetry(tiny_variant())
        self.assertEqual(symmetry["hidden_row_symmetry_deviation"], 0.0)
        self.assertEqual(symmetry["hidden_row_symmetry_deviation_max"], 0.0)
        self.assertTrue(symmetry["hidden_rows_identical"])

    def test_hidden_row_symmetry_metric_detects_untied_rows(self):
        head = tiny_variant()
        with torch.no_grad():
            head.confidence_mlp[0].weight[0].add_(1.0)                    # 只撬动一行
        symmetry = CGR.hidden_row_symmetry(head)
        self.assertFalse(symmetry["hidden_rows_identical"])
        self.assertGreater(symmetry["hidden_row_symmetry_deviation"], 0.0)
        self.assertGreater(symmetry["hidden_row_symmetry_deviation_max"], 0.0)
        json.dumps(symmetry)

    def test_warmup_alpha_matches_historical_formula(self):
        self.assertEqual([CGR.warmup_alpha(e) for e in range(1, 8)],
                         [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(CGR.warmup_alpha(0), 0.0, places=12)
        head = tiny_variant()
        self.assertAlmostEqual(CGR.set_warmup_alpha(head, 3), 0.6, places=12)
        self.assertAlmostEqual(head.gate_warmup_alpha, 0.6, places=12)
        CGR.set_warmup_alpha(head, 5)
        self.assertAlmostEqual(head.gate_warmup_alpha, 1.0, places=12)


class TestFaithfulForward(unittest.TestCase):
    """忠实公式：W_attn / W_fw / g / W 与历史逐项同式。"""

    def test_gate_terms_match_hand_computation(self):
        device = torch.device("cpu")
        head = tiny_variant(device=device)
        CGR.set_warmup_alpha(head, 2)                                     # alpha = 0.4（非平凡）
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(device)

        with torch.no_grad():
            terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
            exist = torch.stack(env_embs, dim=1)
            new_emb = head.env_embedding_network(head.new_env_idx).squeeze(0)
            h_out = head.projection_network(dnn_input)
            want_w_attn = torch.nn.functional.softmax(
                torch.mm(h_out, exist) / head.temperature, dim=-1).unsqueeze(2)
            want_w_fw = (torch.ones(dnn_input.shape[0], len(spec_reps)) / len(spec_reps)).unsqueeze(2)
            gate_input = torch.cat([new_emb, exist.mean(dim=1), exist.var(dim=1),
                                    gen_rep.mean(dim=0), gen_rep.var(dim=0)], dim=-1)
            gate_input = gate_input.unsqueeze(0).expand(dnn_input.shape[0], -1)
            want_z = head.confidence_mlp[2](head.confidence_mlp[1](head.confidence_mlp[0](gate_input)))
            want_g_raw = torch.sigmoid(want_z)
            want_g = head.gate_warmup_alpha * want_g_raw + (1 - head.gate_warmup_alpha) * 0.5
            want_w = want_g.unsqueeze(-1) * want_w_attn + (1 - want_g.unsqueeze(-1)) * want_w_fw

        for got, want, name in ((terms["w_attn"], want_w_attn, "w_attn"),
                                (terms["w_fw"], want_w_fw, "w_fw"),
                                (terms["presigmoid"], want_z, "presigmoid"),
                                (terms["g_raw"], want_g_raw, "g_raw"),
                                (terms["g"], want_g, "g"),
                                (terms["w"], want_w, "w")):
            self.assertTrue(torch.equal(got, want), f"{name} 与历史公式不一致")

    def test_forward_matches_the_preserved_pipeline(self):
        """前向 = 历史 cgr 分支的整条管线（router 融合 → env scale → gate 融合 → tower）。"""
        device = torch.device("cpu")
        head = tiny_variant(device=device)
        CGR.set_warmup_alpha(head, 2)
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(device)

        with torch.no_grad():
            terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
            new_emb = head.env_embedding_network(head.new_env_idx).squeeze(0)
            new_spec_rep = torch.matmul(torch.stack(spec_reps, dim=2), terms["w"]).squeeze()
            env_aware = new_spec_rep * new_emb
            gate_out = head.gate_network(dnn_input).unsqueeze(dim=2)
            fused = torch.matmul(torch.stack([env_aware, gen_rep], dim=2), gate_out).squeeze()
            want = head.tower_network(fused).squeeze()
            got = head(dnn_input, gen_rep, spec_reps, env_embs)
        self.assertTrue(torch.equal(got, want), "前向不是历史 cgr 管线的组合")

    def test_warmup_alpha_zero_makes_g_exactly_half(self):
        """alpha=0 ⇒ g ≡ 0.5（历史 warmup 的"中性起点"），W 是 attention 与 FW 的对半混合。"""
        head = tiny_variant()
        CGR.set_warmup_alpha(head, 0)
        terms = head.gate_terms(*tiny_reps())
        self.assertTrue(torch.equal(terms["g"], torch.full_like(terms["g"], 0.5)))
        with torch.no_grad():
            want = 0.5 * terms["w_attn"] + 0.5 * terms["w_fw"]
        self.assertTrue(torch.equal(terms["w"], want))

    def test_gate_is_batch_global(self):
        """结构性缺陷 D1 的证据：g 是批级常量——批内唯一，且改动**任一样本**会改变整批的 g。

        夹具用 `P.seed_model` 钉死（与本文件其它用例的执行顺序无关）；扰动取大值（+50）是因为
        常量初始化的门控在小扰动下可能只动 ~1e-5（sigmoid 饱和 ⇒ 导数极小），
        这里要断的是**结构**（批级依赖），不是敏感度量级。
        """
        device = torch.device("cpu")
        P.seed_model(P.MODEL_SEED)
        head = live_confidence_mlp(tiny_variant(device=device))
        P.seed_model(P.MODEL_SEED)
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(device)
        with torch.no_grad():
            g = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)["g"]
            self.assertEqual(int(torch.unique(g).numel()), 1, "历史实现的 g 必须在批内逐位相同")
            perturbed = gen_rep.clone()
            perturbed[0] = perturbed[0] + 50.0                            # 只动第 0 个样本
            g_after = head.gate_terms(dnn_input, perturbed, spec_reps, env_embs)["g"]
        self.assertGreater(float((g_after - g).abs().max()), 1e-3, "整批 g 未随批内统计量变化")
        self.assertEqual(int(torch.unique(g_after).numel()), 1)

    def test_same_sample_changes_gate_when_batch_context_changes(self):
        """同一份样本换一个批上下文就换一个 g：门控依赖批组成而不是样本本身。

        对照组用"同一个样本复制两份"的 2 样本批（`var = 0`）——它与 8 样本批的批级统计量差别明确，
        避免把结论压到 sigmoid 饱和后的浮点噪声上；批大小取 2 而不是 1 是为了避开 D6（单样本批 NaN）。
        阈值 1e-5（实测 ~8.7e-5）：本用例断的是"依赖**存在且可复现**"（夹具由 `P.seed_model` 钉死），
        量级断言由 `test_gate_is_batch_global` 的 +50 扰动承担——常量初始化 + 常量第二层把门控的
        输入敏感度压得很低，硬要一个大量级只会写出靠夹具运气过的断言。
        """
        device = torch.device("cpu")
        P.seed_model(P.MODEL_SEED)
        head = live_confidence_mlp(tiny_variant(device=device))
        P.seed_model(P.MODEL_SEED)
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(device, n=8)
        with torch.no_grad():
            row = (dnn_input[:1].repeat(2, 1), gen_rep[:1].repeat(2, 1),
                   [s[:1].repeat(2, 1) for s in spec_reps])
            alone = head.gate_terms(row[0], row[1], row[2], env_embs)["g"]
            in_batch = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)["g"][:2]
        self.assertGreater(float((alone - in_batch).abs().max()), 1e-5)

    def test_single_sample_batch_gives_nan_gate(self):
        """审计发现的额外脆弱性（D6）：批大小 = 1 时 `gen_rep.var(dim=0)`（ddof=1）无定义 ⇒ g = NaN。

        历史实现的 `gate_input` 里 gen 统计量的自由度 = 批大小 − 1；CensusIncome 全量训练集
        （199523 = 779×256 + 99）与 val（49881 = 194×256 + 217）恰好都躲开了 1 样本批，
        所以历史 run 没有暴露这一点——但这仍是历史设计自带的爆点，照实锁住。
        """
        import warnings
        head = live_confidence_mlp(tiny_variant())
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(n=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                               # torch 的 var() 自由度告警即本现象
            with torch.no_grad():
                terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        self.assertTrue(bool(torch.isnan(terms["g"]).all()), "单样本批的 g 应为 NaN（历史行为）")
        self.assertTrue(bool(torch.isnan(terms["w"]).all()))

    def test_unnormalised_w_is_a_convex_combination(self):
        head = tiny_variant()
        terms = head.gate_terms(*tiny_reps())
        self.assertTrue(torch.allclose(terms["w"].sum(dim=1), torch.ones_like(terms["w"].sum(dim=1)),
                                       atol=1e-6))
        self.assertTrue(torch.all(terms["g"] >= 0.0) and torch.all(terms["g"] <= 1.0))

    def test_saturation_floor_at_historical_init_real_scale(self):
        """真实量级（rep_dim=128）下历史常量初始化把门控推向"信任 attention"一侧。

        只断言**地板**（≥0.85）与"尚未浮点饱和"（<1），不假设任意夹具下 sigmoid 恰好等于 1。
        """
        head = CGR.CGRNewTask(input_size=P.INPUT_SIZE, rep_dim=P.EXPERT_HIDDEN[-1],
                              tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN,
                              device=torch.device("cpu"))
        reps = real_scale_reps()
        self.assertEqual(reps[0].shape[1], P.INPUT_SIZE)                 # 123 = 28 类目 × 4 + 11 dense
        with torch.no_grad():
            terms = head.gate_terms(*reps)
        g_raw = float(terms["g_raw"].min())
        self.assertGreaterEqual(g_raw, 0.85, f"历史初始化未体现出饱和地板: g_raw={g_raw}")
        self.assertLess(float(terms["g_raw"].max()), 1.0)
        z, sig = terms["presigmoid"][0], terms["g_raw"][0]
        self.assertAlmostEqual(float(torch.logit(sig)), float(z), places=4)          # z 与 g_raw 同源
        stats = CGR.CGRStats()
        stats.update(terms)
        readings = stats.result()
        self.assertAlmostEqual(readings["presigmoid_saturation_abs"], 8.0, places=12)
        self.assertAlmostEqual(readings["presigmoid_abs_mean"], abs(float(z)), places=4)  # 读数与夹具同源
        self.assertAlmostEqual(readings["g_raw_mean"], float(sig), places=6)

    def test_hard_saturation_makes_cgr_identical_to_attention(self):
        """硬饱和夹具（sigmoid(30)=1.0，float32）⇒ W ≡ W_attn ⇒ |W−W_attn| L1 == 0 ⇒ 退化判据 2 触发。"""
        head = tiny_variant()
        with torch.no_grad():
            head.confidence_mlp[2].weight.zero_()
            head.confidence_mlp[2].bias.fill_(30.0)
        CGR.set_warmup_alpha(head, 5)                                     # alpha = 1.0（warmup 结束）
        terms = head.gate_terms(*tiny_reps())
        self.assertEqual(float(terms["g_raw"].max()), 1.0)
        self.assertTrue(torch.equal(terms["w"], terms["w_attn"]))
        stats = CGR.CGRStats()
        stats.update(terms)
        self.assertEqual(stats.result()["weight_l1_mean"], 0.0)
        verdict = CGR.degeneracy_verdict(unique_frac=stats.result()["g_within_batch_unique_frac_mean"],
                                         weight_l1=stats.result()["weight_l1_mean"], grad_norm=1e-3)
        self.assertEqual(verdict["status"], "CONFIRMED_DEGENERATE")
        self.assertIn("weight_l1 < 1e-6", verdict["triggered_rules"])


class TestCGRStats(unittest.TestCase):
    """指标口径：批内 std / 唯一值占比 / |W−W_attn| L1 / 前 sigmoid 饱和（流式累加，CPU float64）。"""

    @staticmethod
    def terms(g, z, w, w_attn, g_raw=None):
        g = torch.tensor(g, dtype=torch.float32).reshape(-1, 1)
        return {"g": g, "g_raw": g if g_raw is None else torch.tensor(g_raw, dtype=torch.float32).reshape(-1, 1),
                "presigmoid": torch.tensor(z, dtype=torch.float32).reshape(-1, 1),
                "w": torch.tensor(w, dtype=torch.float32).reshape(-1, 2, 1),
                "w_attn": torch.tensor(w_attn, dtype=torch.float32).reshape(-1, 2, 1)}

    def test_hand_computed_batch_metrics(self):
        stats = CGR.CGRStats()
        stats.update(self.terms(g=[0.6, 0.6], z=[1.0, 1.0],
                                w=[[[0.5], [0.5]], [[0.5], [0.5]]],
                                w_attn=[[[0.9], [0.1]], [[0.9], [0.1]]]))
        stats.update(self.terms(g=[0.9], z=[9.0], w=[[[0.2], [0.8]]], w_attn=[[[0.5], [0.5]]]))
        got = stats.result()
        self.assertEqual((got["n_samples"], got["n_batches"]), (3, 2))
        self.assertAlmostEqual(got["g_mean"], 0.7, places=6)
        self.assertAlmostEqual(got["g_min"], 0.6, places=6)
        self.assertAlmostEqual(got["g_max"], 0.9, places=6)
        self.assertAlmostEqual(got["g_within_batch_std_mean"], 0.0, places=6)
        self.assertAlmostEqual(got["g_within_batch_unique_frac_mean"], (0.5 + 1.0) / 2, places=6)
        self.assertAlmostEqual(got["g_within_batch_unique_frac_max"], 1.0, places=6)
        self.assertAlmostEqual(got["weight_l1_mean"], 0.7, places=6)      # (0.8 + 0.6) / 2
        self.assertAlmostEqual(got["weight_l1_max"], 0.8, places=6)
        self.assertAlmostEqual(got["presigmoid_abs_mean"], (1.0 + 1.0 + 9.0) / 3, places=6)
        self.assertAlmostEqual(got["presigmoid_abs_max"], 9.0, places=6)
        self.assertAlmostEqual(got["presigmoid_saturation_frac"], 1.0 / 3, places=6)
        self.assertAlmostEqual(got["presigmoid_saturation_abs"], 8.0, places=12)
        json.dumps(got)

    def test_streaming_preserves_sample_level_aggregates(self):
        """样本级聚合与是否分批无关；**批级**读数（批内 std / 唯一值占比 / 批均 L1）按定义依赖分批。"""
        first = self.terms(g=[0.6, 0.6], z=[1.0, 1.0], w=[[[0.5], [0.5]], [[0.5], [0.5]]],
                           w_attn=[[[0.9], [0.1]], [[0.9], [0.1]]])
        second = self.terms(g=[0.9], z=[9.0], w=[[[0.2], [0.8]]], w_attn=[[[0.5], [0.5]]])
        streamed = CGR.CGRStats()
        streamed.update(first)
        streamed.update(second)
        single = CGR.CGRStats()
        single.update({key: torch.cat([first[key], second[key]]) for key in first})
        streamed_result, single_result = streamed.result(), single.result()
        for key in ("g_mean", "g_min", "g_max", "g_raw_mean", "g_raw_min", "g_raw_max",
                    "presigmoid_abs_mean", "presigmoid_abs_max", "presigmoid_saturation_frac",
                    "n_samples"):
            self.assertAlmostEqual(streamed_result[key], single_result[key], places=9, msg=key)
        self.assertNotEqual(streamed_result["g_within_batch_unique_frac_mean"],
                            single_result["g_within_batch_unique_frac_mean"])   # 批级读数按批计

    def test_batch_constant_g_yields_zero_std_and_one_unique(self):
        """忠实实现的签名读数：批内 std 恒为 0、批内唯一值占比恒为 1/批大小。"""
        stats = CGR.CGRStats()
        stats.update(self.terms(g=[0.7] * 16, z=[2.0] * 16,
                                w=[[[0.3], [0.7]]] * 16, w_attn=[[[0.6], [0.4]]] * 16))
        got = stats.result()
        self.assertEqual(got["g_within_batch_std_mean"], 0.0)
        self.assertAlmostEqual(got["g_within_batch_unique_frac_mean"], 1.0 / 16, places=9)

    def test_empty_accumulator_is_zero_safe(self):
        empty = CGR.CGRStats().result()
        self.assertEqual((empty["n_samples"], empty["n_batches"]), (0, 0))
        for key, value in empty.items():
            if key not in ("n_samples", "n_batches", "presigmoid_saturation_abs"):
                self.assertEqual(value, 0.0, f"空累加器字段应为 0.0: {key}")
        self.assertAlmostEqual(empty["presigmoid_saturation_abs"], 8.0, places=12)   # 阈值恒回显
        json.dumps(empty)


class TestConfidenceGradTrace(unittest.TestCase):
    """训练期置信 MLP 梯度范数：backward 之后、step 之前读取（口径固定，见模块 docstring）。"""

    def test_confidence_grad_norm_matches_manual_computation(self):
        head = tiny_variant()
        head.train()
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps()
        pred = head(dnn_input, gen_rep, spec_reps, env_embs)
        torch.nn.BCELoss()(pred, torch.randint(0, 2, (dnn_input.shape[0],)).float()).backward()

        want = sum(float(p.grad.detach().double().norm(2)) ** 2
                   for p in head.confidence_mlp.parameters() if p.grad is not None) ** 0.5
        self.assertGreater(want, 0.0)
        self.assertAlmostEqual(CGR.confidence_grad_norm(head), want, places=9)

    def test_trace_records_per_epoch_grad_stats_and_gate_readings(self):
        head = tiny_variant()
        head.train()
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps()
        trace = CGR.TrainGateTrace()
        trace.start_epoch(1, CGR.warmup_alpha(1))
        head.zero_grad()
        head(dnn_input, gen_rep, spec_reps, env_embs).sum().backward()
        first = trace.record_step(head, dnn_input, gen_rep, spec_reps, env_embs)
        head.zero_grad()                                                 # 梯度归零 → 本步范数为 0
        second = trace.record_step(head, dnn_input, gen_rep, spec_reps, env_embs)
        record = trace.end_epoch()

        self.assertGreater(first, 0.0)
        self.assertEqual(second, 0.0)
        self.assertEqual(record["epoch"], 1)
        self.assertAlmostEqual(record["alpha"], 0.2, places=12)
        self.assertEqual(record["steps"], 2)
        self.assertEqual(record["grad_nonzero_steps"], 1)
        self.assertAlmostEqual(record["grad_norm_mean"], first / 2, places=9)
        self.assertAlmostEqual(record["grad_norm_max"], first, places=9)
        self.assertAlmostEqual(record["g_within_batch_unique_frac_mean"], 1.0 / 8, places=9)
        self.assertEqual(record["g_within_batch_std_mean"], 0.0)

        result = trace.result()
        self.assertEqual(result["epochs"], 1)
        self.assertEqual(result["steps"], 2)
        self.assertAlmostEqual(result["grad_norm_max"], first, places=9)
        self.assertEqual(len(result["per_epoch"]), 1)
        self.assertEqual(result["per_epoch"][0]["epoch"], 1)
        json.dumps(result)

    def test_zero_gradient_forever_fires_the_degeneracy_rule(self):
        head = tiny_variant()
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps()
        trace = CGR.TrainGateTrace()
        trace.start_epoch(1, 1.0)
        head.zero_grad(set_to_none=True)
        trace.record_step(head, dnn_input, gen_rep, spec_reps, env_embs)
        trace.end_epoch()
        result = trace.result()
        self.assertEqual(result["grad_norm_max"], 0.0)
        self.assertEqual(result["grad_nonzero_steps"], 0)
        verdict = CGR.degeneracy_verdict(unique_frac=0.5, weight_l1=0.1,
                                         grad_norm=result["grad_norm_max"])
        self.assertEqual(verdict["status"], "CONFIRMED_DEGENERATE")
        self.assertIn("grad_norm == 0", verdict["triggered_rules"])

    def test_empty_trace_reports_zero_grad_norm(self):
        result = CGR.TrainGateTrace().result()
        self.assertEqual((result["epochs"], result["steps"]), (0, 0))
        self.assertEqual(result["grad_norm_max"], 0.0)
        json.dumps(result)

    def test_epoch_bookkeeping_errors_are_explicit(self):
        trace = CGR.TrainGateTrace()
        with self.assertRaises(RuntimeError):
            trace.record_step(tiny_variant(), *tiny_reps())              # 未 start_epoch
        trace.start_epoch(1, 1.0)
        with self.assertRaises(RuntimeError):
            trace.start_epoch(2, 1.0)                                    # 上一个 epoch 未收尾
        with self.assertRaises(ValueError):
            trace.record_step(NewTask(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                      tower_dnn_hidden_units=list(P.TOWER_HIDDEN),
                                      reg_dnn=P.REG_DNN), *tiny_reps())  # 基线头不属于处理臂


class TestEvaluateGate(unittest.TestCase):
    """best_state 载入后的诊断：只用 val 前向一遍，不改参数、不训练、不参与选点。"""

    def test_matches_recomputation_and_has_no_side_effects(self):
        device = torch.device("cpu")
        head = tiny_variant(device=device)
        backbone = tiny_backbone(device)
        loaders, _, _ = tiny_inputs()
        before = {name: value.detach().clone() for name, value in head.state_dict().items()}
        sha_before = P.backbone_sha256(backbone)

        got = CGR.evaluate_gate(head, backbone, loaders["val"], device)

        stats = CGR.CGRStats()
        for _, _, _, features in loaders["val"]:
            dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
            stats.update(head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs))
        for key, value in stats.result().items():
            self.assertAlmostEqual(got[key], value, places=9, msg=f"诊断口径不一致: {key}")

        self.assertEqual(got["n_samples"], 16)
        self.assertAlmostEqual(got["warmup_alpha_at_eval"], 1.0, places=12)
        self.assertIn("hidden_row_symmetry_deviation", got)
        self.assertIn("hidden_rows_identical", got)
        for name, value in head.state_dict().items():                     # 参数逐位不变
            self.assertTrue(torch.equal(before[name], value), f"诊断改动了参数: {name}")
        self.assertEqual(P.backbone_sha256(backbone), sha_before)
        P.assert_no_grads(backbone)
        self.assertFalse(head.training)
        json.dumps(got)

    def test_records_the_warmup_alpha_in_effect(self):
        device = torch.device("cpu")
        head = tiny_variant(device=device)
        CGR.set_warmup_alpha(head, 3)                                     # 提前停时 alpha 停在 0.6
        loaders, _, _ = tiny_inputs()
        got = CGR.evaluate_gate(head, tiny_backbone(device), loaders["val"], device)
        self.assertAlmostEqual(got["warmup_alpha_at_eval"], 0.6, places=12)

    def test_baseline_newtask_is_rejected(self):
        device = torch.device("cpu")
        loaders, _, _ = tiny_inputs()
        with self.assertRaises(ValueError):
            CGR.evaluate_gate(tiny_newtask(device=device), tiny_backbone(device), loaders["val"], device)


class TestVariantFactory(unittest.TestCase):
    def test_baseline_variant_is_the_untouched_newtask(self):
        head = CGR.build_newtask(CGR.BASELINE_VARIANT, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                 tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)
        self.assertIs(type(head), NewTask)                                # 不是子类，就是基线类本身
        self.assertNotIn("confidence_mlp.0.weight", head.state_dict())

    def test_baseline_variant_consumes_no_extra_rng(self):
        """基线臂逐位不变：同 seed 下与直接构造的 master NewTask 完全一致（含 RNG 终点）。"""
        P.seed_model(P.MODEL_SEED)
        reference = tiny_newtask()
        rng_after_reference = torch.get_rng_state().clone()

        P.seed_model(P.MODEL_SEED)
        head = CGR.build_newtask(CGR.BASELINE_VARIANT, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                 tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_after_reference))
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(head.state_dict()[name], value), f"基线初始化漂移: {name}")

    def test_treatment_variant_is_the_faithful_subclass(self):
        head = CGR.build_newtask(CGR.VARIANT, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                 tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)
        self.assertIsInstance(head, CGR.CGRNewTask)
        self.assertIsInstance(head, NewTask)
        self.assertIn("confidence_mlp.0.weight", head.state_dict())

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            CGR.build_newtask("no_such_variant", input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                              tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)


class TestRunnerIntegration(unittest.TestCase):
    """接线：标准 short 命令 + `--variant cgr` / `--cgr`；基线臂不受影响。"""

    def test_treatment_arm_records_arm_block_trace_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            out = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=2, device=device,
                                                  model=tiny_backbone(device), loaders=loaders, stats=stats,
                                                  indices=indices, input_size=TINY_INPUT_SIZE,
                                                  rep_dim=TINY_REP_DIM, variant=CGR.VARIANT)
            run_path = Path(out["run_dir"])
            self.assertTrue(out["run_id"].endswith(CGR.RUN_ID_SUFFIX), out["run_id"])

            config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["variant"], CGR.VARIANT)
            self.assertEqual(config["warmup_epochs"], CGR.WARMUP_EPOCHS)

            payload = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["variant"], CGR.VARIANT)
            self.assertEqual(sorted(payload["mechanism"]),                      # M3/M4 键集不变
                             ["cos_gen_spec", "env_acc_stage1", "gate_mean", "gen_std"])
            arm = payload["cgr_arm"]
            self.assertEqual(arm["prereg"]["auc_test_min"], 0.8521)
            self.assertEqual(arm["prereg"]["warmup_epochs"], 5)
            self.assertIn(arm["degeneracy"]["status"], ("CONFIRMED_DEGENERATE", "NOT_DEGENERATE"))
            self.assertIn(arm["status"], ("STOP", "EFFECT_CONFIRMED", "EFFECT_NOT_CONFIRMED"))
            self.assertEqual(arm["effect"]["evaluated"], arm["degeneracy"]["status"] == "NOT_DEGENERATE")
            trace = arm["train_gate_trace"]
            self.assertEqual([record["epoch"] for record in trace], [1, 2])     # 逐 epoch 轨迹
            self.assertAlmostEqual(trace[0]["alpha"], 0.2, places=12)
            self.assertAlmostEqual(trace[1]["alpha"], 0.4, places=12)
            self.assertGreater(trace[1]["grad_norm_max"], 0.0)                  # 门控确实吃到梯度

            diagnostics = arm["diagnostics"]
            self.assertEqual(diagnostics["n_samples"], 16)                      # val 只跑一遍
            self.assertAlmostEqual(diagnostics["warmup_alpha_at_eval"], 0.4, places=12)
            self.assertAlmostEqual(diagnostics["g_within_batch_unique_frac_mean"], 1.0 / 16, places=9)
            self.assertEqual(diagnostics["g_within_batch_std_mean"], 0.0)       # 批级常量的签名读数
            self.assertIn("weight_l1_mean", diagnostics)
            self.assertIn("presigmoid_saturation_frac", diagnostics)
            self.assertIn("hidden_rows_identical", diagnostics)

            report = json.loads((run_path / "gate_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["A1"]["pass"])                               # 冻结三件套不受影响
            self.assertEqual(report["A3"]["status"], "on_demand")
            self.assertNotIn("cgr_arm", report)                                 # 臂级判定不混入 A/B 门禁

            state = torch.load(run_path / "newtask.pt", map_location="cpu")
            tiny_variant(device=device).load_state_dict(state)                   # strict：门控权重在 checkpoint 里
            self.assertIn("confidence_mlp.0.weight", state)

            summary = (root / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn(out["run_id"], summary)                                # 失败也必须留痕（协议 7.3）

    def test_baseline_arm_is_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            with mock.patch.object(CGR, "CGRNewTask", side_effect=AssertionError("基线臂不得构造 CGR 头")):
                out = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=2,
                                                      device=device, model=tiny_backbone(device),
                                                      loaders=loaders, stats=stats, indices=indices,
                                                      input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM)
            run_path = Path(out["run_dir"])
            self.assertFalse(out["run_id"].endswith(CGR.RUN_ID_SUFFIX))

            config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["variant"], CGR.BASELINE_VARIANT)
            self.assertNotIn("warmup_epochs", config)                            # 处理臂专属字段不得出现
            self.assertEqual(sorted(config),                                     # 键集 = 基线键集 + variant 标签
                             ["batch_size", "commit", "env_seed", "epochs", "frozen", "input_size", "lr",
                              "model_seed", "patience", "rep_dim", "run_id", "split_seed", "stage1_id",
                              "tag", "variant"])

            payload = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["variant"], CGR.BASELINE_VARIANT)
            self.assertNotIn("cgr_arm", payload)
            self.assertNotIn("confidence_mlp", json.dumps(payload))
            self.assertEqual(sorted(payload),
                             ["backbone_sha256_after", "backbone_sha256_before", "commit",
                              "env_ids_sha256", "mechanism", "run_id", "split_sha256", "stage1",
                              "stage1_id", "stage2", "variant"])
            self.assertEqual(sorted(payload["mechanism"]),
                             ["cos_gen_spec", "env_acc_stage1", "gate_mean", "gen_std"])
            state = torch.load(run_path / "newtask.pt", map_location="cpu")
            tiny_newtask(device=device).load_state_dict(state)                   # strict 载入基线头

    def test_cli_exposes_variant_and_cgr_alias(self):
        parser = run_census_benchmark.build_parser()
        args = parser.parse_args(["stage2", "--stage1-dir", "x"])
        self.assertEqual(args.variant, CGR.BASELINE_VARIANT)
        args = parser.parse_args(["stage2", "--stage1-dir", "x", "--variant", CGR.VARIANT])
        self.assertEqual(args.variant, CGR.VARIANT)
        args = parser.parse_args(["stage2", "--stage1-dir", "x", "--cgr"])
        self.assertEqual(args.variant, CGR.VARIANT)
        with self.assertRaises(SystemExit):
            parser.parse_args(["stage2", "--stage1-dir", "x", "--variant", "nope"])

    def test_static_guard_protocol_and_model_untouched(self):
        """本分支只加复现件：模型/配置不动，协议与指标文件在本分支上不动。"""
        import subprocess
        repo = Path(__file__).resolve().parents[2]
        for args, note in (
                (["master", "--", "multitaskrec", "config.py"], "模型/配置"),
                (["HEAD", "--", "census_benchmark/protocol.py", "census_benchmark/metrics.py"], "协议/指标")):
            diff = subprocess.run(["git", "diff", "--name-only", *args], cwd=repo,
                                  capture_output=True, text=True).stdout.strip()
            self.assertEqual(diff, "", f"{note}文件不得改动: {diff}")


if __name__ == "__main__":
    unittest.main()
