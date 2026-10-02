"""可学习标量衰减臂（恒等初始化）：对照说明
`docs/superpowers/specs/2026-10-03-aliccp-stage2-learnable-attenuation-design.md`。

- 默认关断（`learnable_spec_attenuation=False`）：NewTask 必须与分支基点
  `8133d32:multitaskrec/model.py`（= 基线 run 实际使用的模型代码）**逐位一致**（前向 + 反向 + 参数集合）。
- 启用（`learnable_spec_attenuation=True`）：新增**恰一个** 0 维标量 `spec_attenuation_raw`，
  θ_init = 1.0，`c = clamp(θ, 0.05, 1.0)`；初始化时 c = 1.0 ⇒ 前向与默认臂逐位一致（恒等保持）。
- kink 构造的数学前提（spec §1.2）：光滑参数化无法同时满足恒等 + 非零梯度（Fermat/反函数定理），
  故用 clamp；其边界梯度约定（max 边界通过、饱和区为零）由本文件微测试钉死。
- 本臂代码任何路径**不得使用** post-hoc 系数 0.697…（仅 metrics.py 保留 descriptive 参照常量）。

测试用 CPU 极小夹具，不构成任何性能证据，只验证语义、指标与接线。
"""
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aliccp_benchmark import bench, metrics, protocol as P
from multitaskrec.model import SPEC_ATTENUATION_MIN, NewTask

import run_aliccp_benchmark

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "superpowers" / "specs" / "2026-10-03-aliccp-stage2-learnable-attenuation-design.md"
BASE_COMMIT = "8133d32"          # 本分支基点（infra/aliccp-fair-benchmark 顶端 = 基线 run 的模型代码）
SCALAR = "spec_attenuation_raw"
BATCH, INPUT_SIZE, REP_DIM, NUM_SRC = 6, 8, 4, 2
WHITELIST = {
    "multitaskrec/model.py",
    "aliccp_benchmark/metrics.py",
    "aliccp_benchmark/bench.py",
    "run_aliccp_benchmark.py",
    "aliccp_benchmark/tests/test_learnable_attenuation.py",
    "docs/superpowers/specs/2026-10-03-aliccp-stage2-learnable-attenuation-design.md",
    "artifacts/aliccp_bench/SUMMARY.md",
}


def _git_show(path_in_repo: str) -> str:
    return subprocess.run(
        ["git", "show", f"{BASE_COMMIT}:{path_in_repo}"], cwd=REPO, check=True,
        capture_output=True, text=True, encoding="utf-8", errors="strict",
    ).stdout


def _base_newtask_class():
    """从 git 基点提交动态加载 NewTask 作为逐位对照的参考实现（不落地任何文件）。"""
    src = _git_show("multitaskrec/model.py")
    module = types.ModuleType("model_base")
    exec(compile(src, f"{BASE_COMMIT}:multitaskrec/model.py", "exec"), module.__dict__)
    return module.NewTask


def _ours(learnable: bool = False):
    return NewTask(input_size=INPUT_SIZE, rep_dim=REP_DIM, tower_dnn_hidden_units=(4, 2),
                   reg_dnn=P.REG_DNN, device=torch.device("cpu"),
                   learnable_spec_attenuation=learnable)


def _base():
    return _base_newtask_class()(input_size=INPUT_SIZE, rep_dim=REP_DIM, tower_dnn_hidden_units=(4, 2),
                                 reg_dnn=P.REG_DNN, device=torch.device("cpu"))


def _inputs(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(BATCH, INPUT_SIZE, generator=gen), torch.randn(BATCH, REP_DIM, generator=gen),
            [torch.randn(BATCH, REP_DIM, generator=gen) for _ in range(NUM_SRC)],
            [torch.randn(REP_DIM, generator=gen) for _ in range(NUM_SRC)],
            torch.randint(0, 2, (BATCH,), generator=gen).float())


def _forward_backward(module, inputs):
    dnn_input, gen_rep, spec_reps, env_embs, target = inputs
    module.zero_grad(set_to_none=True)
    pred = module(dnn_input, gen_rep, spec_reps, env_embs)
    F.binary_cross_entropy(pred, target).backward()
    return pred


class TestDefaultOffBitIdenticalToBase(unittest.TestCase):
    """默认关断臂必须是基点实现的逐位复制（spec §1.3 默认关断 / §3 L1/L2）。"""

    def test_default_state_dict_matches_base(self):
        ours = _ours()
        self.assertEqual(set(ours.state_dict()), set(_base().state_dict()))
        ours.load_state_dict(_base().state_dict())          # strict=True：参数集合必须一致
        self.assertNotIn(SCALAR, ours.state_dict())
        self.assertFalse(ours.learnable_spec_attenuation)
        self.assertIsNone(ours.spec_attenuation_value())

    def test_default_forward_bit_identical(self):
        base, ours, inputs = _base(), _ours(), _inputs(1)
        ours.load_state_dict(base.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(ours(*inputs[:4]), base(*inputs[:4])))

    def test_default_backward_bit_identical(self):
        torch.manual_seed(0)  # 固定构造 RNG：部分随机初始化下 tower 的 ReLU 全灭会零化整条 specific 路径梯度
        base, inputs = _base(), _inputs(2)
        torch.manual_seed(0)
        ours = _ours()
        pred_base, pred_ours = _forward_backward(base, inputs), _forward_backward(ours, inputs)
        self.assertTrue(torch.equal(pred_base, pred_ours))
        grads_ours = dict(ours.named_parameters())
        for name, param in base.named_parameters():
            self.assertIsNotNone(param.grad, f"基点该参数无梯度，反向对照无效: {name}")
            self.assertTrue(torch.equal(param.grad, grads_ours[name].grad), f"梯度非逐位一致: {name}")


class TestConstructionAndIdentity(unittest.TestCase):
    """启用臂构造（零 RNG 消耗、参数集合）与恒等保持（spec §3 L2/L3）。"""

    def test_enabled_construction_consumes_no_rng(self):
        torch.manual_seed(3)
        off = _ours(learnable=False)
        torch.manual_seed(3)
        on = _ours(learnable=True)
        off_params, on_params = dict(off.named_parameters()), dict(on.named_parameters())
        self.assertEqual(set(on_params) - set(off_params), {SCALAR})
        for name, param in off_params.items():
            self.assertTrue(torch.equal(param, on_params[name]), f"共享参数初始化漂移: {name}")
        self.assertEqual(float(on.spec_attenuation_raw.detach()), 1.0)

    def test_enabled_state_dict_adds_only_scalar(self):
        on = _ours(learnable=True)
        self.assertEqual(set(on.state_dict()) - set(_base().state_dict()), {SCALAR})
        with self.assertRaises(RuntimeError):               # strict 载入基点 state_dict 必须因缺键报错
            on.load_state_dict(_base().state_dict())
        base = _base()
        on.load_state_dict(base.state_dict(), strict=False)
        for name, param in base.named_parameters():
            self.assertTrue(torch.equal(param, dict(on.named_parameters())[name]))

    def test_enabled_forward_at_init_bit_identical_to_default(self):
        """恒等保持（核心性质 P1）：θ=1.0 时前向与默认臂逐位一致。"""
        torch.manual_seed(5)
        on = _ours(learnable=True)
        torch.manual_seed(5)
        off = _ours(learnable=False)
        inputs = _inputs(6)
        with torch.no_grad():
            self.assertEqual(float(on.spec_attenuation_value()), 1.0)
            self.assertTrue(torch.equal(on(*inputs[:4]), off(*inputs[:4])))


class TestGradientAtIdentity(unittest.TestCase):
    """kink 边界梯度约定（spec §1.2 可行性前提）+ 恒等点反向（spec §3 L4）。"""

    def test_clamp_boundary_gradient_convention(self):
        """PyTorch clamp 反向：max 边界（含等号）通过梯度、饱和区为零——恒等 + 非零梯度的实现前提。"""
        x = torch.tensor([2.0, -3.0])
        for raw0, expect in [(1.0, -1.0), (0.8, -1.0), (1.5, 0.0), (0.01, 0.0)]:
            raw = torch.tensor(raw0, requires_grad=True)
            c = torch.clamp(raw, min=SPEC_ATTENUATION_MIN, max=1.0)
            (x * c).sum().backward()
            self.assertEqual(float(raw.grad), expect, f"raw0={raw0} 边界梯度约定不符")

    def test_enabled_backward_scalar_grad_and_shared_grads_bit_identical(self):
        torch.manual_seed(0)
        off, inputs = _ours(learnable=False), _inputs(4)
        torch.manual_seed(0)
        on = _ours(learnable=True)
        pred_off, pred_on = _forward_backward(off, inputs), _forward_backward(on, inputs)
        self.assertTrue(torch.equal(pred_off, pred_on))
        self.assertIsNotNone(on.spec_attenuation_raw.grad)
        self.assertNotEqual(float(on.spec_attenuation_raw.grad), 0.0)   # 恒等点存在非零学习信号
        grads_off = dict(off.named_parameters())
        for name, param in grads_off.items():
            self.assertIsNotNone(param.grad, f"该参数无梯度，反向对照无效: {name}")
            self.assertTrue(
                torch.equal(param.grad, dict(on.named_parameters())[name].grad),
                f"乘 1.0 后共享参数梯度非逐位一致: {name}",
            )


class TestEnabledForwardFormula(unittest.TestCase):
    """启用臂前向 = 写死公式（spec §1.3 作用位置 / §3 L5/L6）。"""

    def _reference(self, module, inputs, v):
        dnn_input, gen_rep, spec_reps, env_embs, _ = inputs
        keys = torch.stack(env_embs, dim=1)
        logits = torch.mm(module.projection_network(dnn_input), keys) / module.temperature
        weights = F.softmax(logits, dim=-1).unsqueeze(2)
        c = torch.clamp(torch.tensor(v), min=SPEC_ATTENUATION_MIN, max=1.0)
        new_spec_rep = torch.matmul(torch.stack(spec_reps, dim=2), weights).squeeze() * c
        env_aware_rep = new_spec_rep * module.env_embedding_network(module.new_env_idx).squeeze(0)
        fused_rep = torch.matmul(torch.stack([env_aware_rep, gen_rep], dim=2),
                                 module.gate_network(dnn_input).unsqueeze(dim=2)).squeeze()
        return module.tower_network(fused_rep).squeeze()

    def test_enabled_forward_matches_reference_formula(self):
        on, inputs = _ours(learnable=True), _inputs(7)
        for v in (1.0, 0.8, 0.5, 0.02, 1.5):
            with torch.no_grad():
                on.spec_attenuation_raw.copy_(torch.tensor(v))
            expected = self._reference(on, inputs, v)
            with torch.no_grad():
                self.assertTrue(torch.equal(on(*inputs[:4]), expected), f"θ={v} 前向与公式不符")

    def test_coefficient_value_mapping(self):
        on = _ours(learnable=True)
        floor32 = float(torch.tensor(SPEC_ATTENUATION_MIN))  # clamp 下限的 float32 有效值
        for v, expected in [(1.0, 1.0), (0.99, float(torch.tensor(0.99))), (0.05, floor32),
                            (0.02, floor32), (0.0, floor32), (-1.0, floor32), (1.5, 1.0)]:
            with torch.no_grad():
                on.spec_attenuation_raw.copy_(torch.tensor(v))
            self.assertEqual(float(on.spec_attenuation_value()), expected, f"θ={v}")


class TestParameterBudget(unittest.TestCase):
    """恰一个标量、0 维、不进 L2（spec §3 L7）；真实配置下计数与预注册常量一致。"""

    def test_enabled_adds_exactly_one_zero_dim_param(self):
        off, on = _ours(), _ours(learnable=True)
        names_off = {n for n, _ in off.named_parameters()}
        names_on = {n for n, _ in on.named_parameters()}
        self.assertEqual(names_on - names_off, {SCALAR})
        self.assertEqual(on.spec_attenuation_raw.dim(), 0)
        self.assertEqual(on.spec_attenuation_raw.numel(), 1)

    def test_scalar_not_in_l2_reg(self):
        torch.manual_seed(9)
        off = _ours()
        torch.manual_seed(9)
        on = _ours(learnable=True)
        self.assertTrue(torch.equal(off.get_l2_reg(), on.get_l2_reg()))

    def test_real_config_param_counts_match_preregistered(self):
        real = NewTask(input_size=P.INPUT_SIZE, rep_dim=P.NEWTASK_REP_DIM,
                       tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN,
                       device=torch.device("cpu"), learnable_spec_attenuation=False)
        real_on = NewTask(input_size=P.INPUT_SIZE, rep_dim=P.NEWTASK_REP_DIM,
                          tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN,
                          device=torch.device("cpu"), learnable_spec_attenuation=True)
        count_off = sum(p.numel() for p in real.parameters())
        count_on = sum(p.numel() for p in real_on.parameters())
        self.assertEqual(count_off, metrics.DEFAULT_TRAINABLE_PARAMS)
        self.assertEqual(count_on, metrics.LEARNABLE_TRAINABLE_PARAMS)
        self.assertEqual(count_on, count_off + 1)


class TestMetricsFunctions(unittest.TestCase):
    """指标常量与纯函数（spec §3 L8）。"""

    def test_preregistered_constants_pinned(self):
        self.assertEqual(metrics.BASELINE_TEST_AUC, 0.5988392178311113)
        self.assertEqual(metrics.BASELINE_VAL_AUC, 0.5781533414372665)
        self.assertEqual(metrics.NULL_TEST_AUC, 0.6126857532817985)
        self.assertEqual(metrics.NULL_VAL_AUC, 0.5919398413138859)
        self.assertEqual(metrics.NULL_DELTA_TEST_AUC, 0.013846535450687258)
        self.assertEqual(metrics.NULL_DELTA_VAL_AUC, 0.013786499876619396)
        self.assertEqual(metrics.CONTROL_TEST_AUC, 0.6115453624066993)
        self.assertEqual(metrics.CONTROL_VAL_AUC, 0.5914326183692061)
        self.assertEqual(metrics.CONTROL_DELTA_TEST_AUC, 0.012706144575588052)
        self.assertEqual(metrics.CONTROL_DELTA_VAL_AUC, 0.013279276931939643)
        self.assertEqual(metrics.POSTHOC_COEF_F64, 0.6972233730330467)
        self.assertEqual(repr(metrics.POSTHOC_COEF_F32), repr(float(torch.tensor(0.6972233730330467))))
        self.assertEqual(metrics.LEARNABLE_RAW_INIT, 1.0)
        self.assertEqual(metrics.PI_DELTA_TEST_AUC, 0.0055)
        self.assertEqual(metrics.PI_TEST_AUC_MIN, 0.6043392178311113)
        # 预注册字面量与前臂一致；float64 下与 BASELINE + 0.0055 差 ≤ 1 ulp（前臂同款披露）
        self.assertAlmostEqual(metrics.PI_TEST_AUC_MIN,
                               metrics.BASELINE_TEST_AUC + metrics.PI_DELTA_TEST_AUC, places=12)
        self.assertEqual(metrics.PROTOCOL_DELTA_TEST_AUC, 0.005)
        self.assertEqual(metrics.DEFAULT_TRAINABLE_PARAMS, 8129)
        self.assertEqual(metrics.LEARNABLE_TRAINABLE_PARAMS, 8130)
        self.assertEqual(metrics.LEARNABLE_PARAM_NAME, SCALAR)
        self.assertEqual(SPEC_ATTENUATION_MIN, 0.05)
        self.assertEqual(metrics.LEARNABLE_COEF_MIN, SPEC_ATTENUATION_MIN)
        self.assertEqual(metrics.LEARNABLE_GRAD_MIN, 1e-12)
        self.assertEqual(metrics.LEARNABLE_GRAD_WINDOW, 10)
        self.assertEqual(metrics.LEARNABLE_CHANGE_MIN, 0.01)

    def _verdict(self, **overrides):
        kwargs = dict(
            test_auc=0.6100, val_auc_best=0.5900, c_final=0.70, grad_abs_max_first10=1e-6,
            trainable_params=8130, trainable_params_default=8129,
            trainable_param_names=["gate", SCALAR], buffer_keys=["new_env_idx"], a_class_ok=True,
        )
        kwargs.update(overrides)
        return metrics.learnable_attenuation_verdict(**kwargs)

    def test_verdict_classification_mechanics(self):
        self.assertEqual(self._verdict()["classification"], "LEARNED_ATTENUATION_SUPPORTED")
        self.assertTrue(self._verdict()["mechanism_pass"])
        self.assertTrue(self._verdict()["utility_pass"])
        # 效用未过（test 低于 PI）→ NO_UTILITY
        v = self._verdict(test_auc=metrics.PI_TEST_AUC_MIN - 1e-9)
        self.assertEqual(v["classification"], "LEARNED_ATTENUATION_NO_UTILITY")
        self.assertTrue(v["mechanism_pass"])
        # val 不同向（等于基线）→ U1 失败
        v = self._verdict(val_auc_best=metrics.BASELINE_VAL_AUC)
        self.assertFalse(v["checks"]["U1_pi_utility"])
        self.assertEqual(v["classification"], "LEARNED_ATTENUATION_NO_UTILITY")
        # 恒等驻留
        v = self._verdict(c_final=1.0)
        self.assertEqual(v["classification"], "IDENTITY_PRESERVED")
        self.assertFalse(v["checks"]["M1_change"])
        v = self._verdict(c_final=0.995)
        self.assertEqual(v["classification"], "IDENTITY_PRESERVED")
        # 触底
        v = self._verdict(c_final=SPEC_ATTENUATION_MIN)
        self.assertEqual(v["classification"], "DEGENERATE_FLOOR")
        v = self._verdict(c_final=SPEC_ATTENUATION_MIN + 0.005)
        self.assertEqual(v["classification"], "DEGENERATE_FLOOR")
        # 前 10 步无梯度信号（但 c 动了）→ 异常
        v = self._verdict(grad_abs_max_first10=1e-12)       # 边界：严格大于才过
        self.assertEqual(v["classification"], "GRADIENT_WINDOW_ANOMALY")
        self.assertFalse(v["checks"]["M2_gradient"])
        # 完整性失败优先于一切
        v = self._verdict(a_class_ok=False)
        self.assertEqual(v["classification"], "INTEGRITY_FAIL")
        v = self._verdict(trainable_params=8129)
        self.assertEqual(v["classification"], "INTEGRITY_FAIL")
        v = self._verdict(buffer_keys=["new_env_idx", "extra_buffer"])
        self.assertEqual(v["classification"], "INTEGRITY_FAIL")
        v = self._verdict(trainable_param_names=[SCALAR, SCALAR])
        self.assertEqual(v["classification"], "INTEGRITY_FAIL")

    def test_verdict_boundary_inclusivity(self):
        # M1：|c−1| == 0.01（含等号）通过；M3 上界 c == 0.99（含等号）通过
        v = self._verdict(c_final=0.99)
        self.assertTrue(v["checks"]["M1_change"])
        self.assertTrue(v["checks"]["M3_interior"])
        self.assertEqual(v["classification"], "LEARNED_ATTENUATION_SUPPORTED")
        # M3 下界 c == c_min + 0.01（含等号）通过
        v = self._verdict(c_final=SPEC_ATTENUATION_MIN + 0.01)
        self.assertTrue(v["checks"]["M3_interior"])
        # U1 test 边界：恰等于 PI 字面量 → 通过（闭区间）
        v = self._verdict(test_auc=metrics.PI_TEST_AUC_MIN)
        self.assertTrue(v["checks"]["U1_pi_utility"])
        # 协议级 Δ ≥ +0.005 单独记录
        self.assertTrue(self._verdict()["checks"]["protocol_utility"])
        self.assertFalse(self._verdict(test_auc=metrics.BASELINE_TEST_AUC + 0.004)["checks"]["protocol_utility"])
        # D1 方向门
        self.assertTrue(self._verdict(c_final=0.70)["checks"]["D1_direction"])
        self.assertFalse(self._verdict(c_final=0.995)["checks"]["D1_direction"])

    def test_verdict_descriptive_fields(self):
        v = self._verdict(test_auc=metrics.CONTROL_TEST_AUC, c_final=float(torch.tensor(0.6972233730330467)))
        d = v["descriptive"]
        self.assertAlmostEqual(d["reproduction_ratio_test_vs_null"],
                               metrics.CONTROL_DELTA_TEST_AUC / metrics.NULL_DELTA_TEST_AUC, places=12)
        self.assertAlmostEqual(d["reproduction_ratio_test_vs_control"],
                               metrics.CONTROL_DELTA_TEST_AUC / metrics.CONTROL_DELTA_TEST_AUC, places=12)
        c32 = float(torch.tensor(0.6972233730330467))
        self.assertAlmostEqual(d["posthoc_distance_f64"], abs(c32 - metrics.POSTHOC_COEF_F64), places=12)
        self.assertAlmostEqual(d["posthoc_distance_f32"], abs(c32 - metrics.POSTHOC_COEF_F32), places=12)

    def test_pred_dispersion_matches_numpy_f64(self):
        torch.manual_seed(11)
        values = torch.randn(101)
        out = metrics.pred_dispersion(values)
        arr = values.double().numpy()
        self.assertAlmostEqual(out["pred_mean"], float(arr.mean()), places=12)
        self.assertAlmostEqual(out["pred_std"], float(arr.std(ddof=0)), places=12)
        self.assertAlmostEqual(out["pred_var"], float(arr.var(ddof=0)), places=12)
        self.assertEqual(out["pred_min"], float(arr.min()))
        self.assertEqual(out["pred_max"], float(arr.max()))
        for q in (10, 25, 50, 75, 90):
            self.assertAlmostEqual(out[f"pred_q{q:02d}"], float(np.percentile(arr, q)), places=12)

    def test_source_gate_stats_streaming_and_values(self):
        """数据用二进制精确值（1/16 的倍数）→ float64 求和无舍入，流式与单批可逐位比较。"""
        stats = metrics.SourceGateStats(num_tasks=2)
        batch1 = [torch.tensor([[0.25, 0.75], [0.5, 0.5]]), torch.tensor([[0.125, 0.875], [0.375, 0.625]])]
        batch2 = [torch.tensor([[0.625, 0.375]]), torch.tensor([[0.5, 0.5]])]
        stats.update(batch1)
        stats.update(batch2)
        result = stats.result()
        self.assertAlmostEqual(result[0], (0.25 + 0.5 + 0.625) / 3, places=12)
        self.assertAlmostEqual(result[1], (0.125 + 0.375 + 0.5) / 3, places=12)
        single = metrics.SourceGateStats(num_tasks=2)
        single.update([torch.cat([batch1[0], batch2[0]]), torch.cat([batch1[1], batch2[1]])])
        for a, b in zip(result, single.result()):
            self.assertEqual(a, b)                          # 流式 == 单批（float64 逐位）


class TestNoPosthocCoefficientInArmCode(unittest.TestCase):
    """预注册禁令：本臂代码路径不得使用 post-hoc 系数（spec §1.3）。"""

    def test_model_and_bench_sources_free_of_posthoc_literal(self):
        for rel in ("multitaskrec/model.py", "aliccp_benchmark/bench.py", "run_aliccp_benchmark.py"):
            src = (REPO / rel).read_text(encoding="utf-8")
            self.assertNotIn("0.697", src, f"{rel} 不得出现 post-hoc 系数字面量")


# ---- L9：CPU 极小夹具端到端（复用 test_smoke 的微型数据格式，自包含）----
HEADER = "click,purchase,X,121,122,301"
TINY_VOCAB = {"121": 5, "122": 4}
TRAIN_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3], [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3], [1, 0, 7, 1, 1, 2], [0, 0, 9, 2, 2, 1], [1, 1, 9, 0, 1, 3],
    [0, 0, 8, 3, 0, 2], [1, 0, 8, 1, 1, 1], [0, 0, 7, 4, 2, 3], [1, 0, 7, 2, 1, 2],
]
VAL_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3],
    [1, 1, 8, 3, 1, 1], [0, 0, 7, 4, 0, 3], [1, 0, 7, 1, 1, 2],
]
TEST_ROWS = [
    [0, 0, 9, 2, 2, 1], [1, 0, 9, 0, 1, 3], [0, 0, 8, 3, 0, 2], [1, 1, 8, 1, 1, 1],
    [0, 0, 7, 4, 2, 3], [1, 0, 7, 2, 1, 2], [0, 0, 9, 1, 0, 3], [1, 0, 8, 4, 1, 2],
]


def _write_aliccp_file(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")


def _make_tiny_root(td: str):
    root = Path(td) / "artifacts"
    data_dir = Path(td) / "data"
    data_dir.mkdir()
    _write_aliccp_file(data_dir / "train.csv", TRAIN_ROWS)
    _write_aliccp_file(data_dir / "val.csv", VAL_ROWS)
    _write_aliccp_file(data_dir / "test.csv", TEST_ROWS)
    data_files = {split: str(data_dir / f"{split}.csv") for split in ("train", "val", "test")}
    budgets = {"train": len(TRAIN_ROWS), "val": len(VAL_ROWS), "test": len(TEST_ROWS)}
    return root, data_files, budgets


class TestRunnerLearnableArm(unittest.TestCase):
    """运行器接线（spec §3 L9）：默认关断字段不变；启用臂落盘 trace/探针/判定。"""

    def _run_stage2(self, root, data_files, budgets, stage1_id, learnable, run_id):
        return bench.run_stage2(
            root=root, stage1_id=stage1_id, data_files=data_files, budgets=budgets,
            prefix_tag="tiny", model_seed=123, epochs=1, patience=1, tag="smoke",
            device=torch.device("cpu"), vocab=TINY_VOCAB,
            expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
            enforce_b=False, run_id=run_id, learnable_attenuation=learnable,
            log=lambda *a, **k: None,
        )

    def test_default_run_has_marker_only_and_enabled_run_records_arm(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = _make_tiny_root(td)
            meta = bench.run_stage1(
                root=root, data_files=data_files, budgets=budgets, prefix_tag="tiny",
                model_seed=123, env_seed=456, epochs=1, patience=1,
                device=torch.device("cpu"), vocab=TINY_VOCAB,
                expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
                log=lambda *a, **k: None,
            )
            default = self._run_stage2(root, data_files, budgets, meta["stage1_id"], False, "tiny-default")
            dm = default["metrics"]
            self.assertFalse(dm["learnable_spec_attenuation"])
            for key in ("probe", "learnable_arm", "trainable_params", "learnable_trace"):
                self.assertNotIn(key, dm)
            self.assertFalse((P.run_dir(root, "tiny-default") / "attenuation_trace.json").exists())

            enabled = self._run_stage2(root, data_files, budgets, meta["stage1_id"], True, "tiny-enabled")
            em = enabled["metrics"]
            self.assertTrue(em["learnable_spec_attenuation"])
            self.assertEqual(em["trainable_params"], em["trainable_params_default"] + 1)  # 任意配置下恰多 1
            self.assertEqual(em["trainable_param_names"].count(SCALAR), 1)
            self.assertEqual(em["stage1_id"], meta["stage1_id"])
            self.assertEqual(em["buffer_keys"], ["new_env_idx"])
            for gate_id in ("A1", "A2", "A4", "A5", "A6"):
                self.assertEqual(enabled["gates"][gate_id]["verdict"], "PASS", gate_id)
            # 探针与轨迹
            self.assertIn("source_gate_mean", em["probe"])
            self.assertIn("pred_std", em["probe"])
            trace_path = P.run_dir(root, "tiny-enabled") / "attenuation_trace.json"
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(len(trace), em["learnable_trace"]["steps"])
            self.assertEqual(trace[0]["step"], 1)
            self.assertEqual(trace[0]["theta"], 1.0)          # 第一步记录的是恒等点状态
            self.assertEqual(trace[0]["c"], 1.0)
            self.assertIn("grad", trace[0])
            # 判定字段齐备且分类在预注册集合内
            arm = em["learnable_arm"]
            self.assertIn(arm["classification"], {
                "INTEGRITY_FAIL", "DEGENERATE_FLOOR", "IDENTITY_PRESERVED",
                "GRADIENT_WINDOW_ANOMALY", "LEARNED_ATTENUATION_SUPPORTED",
                "LEARNED_ATTENUATION_NO_UTILITY",
            })
            self.assertIn("checks", arm)
            self.assertIn("descriptive", arm)
            # run_id 后缀（由调用方给定，此处仅验证 CLI 侧拼接）
            self.assertEqual(em["run_id"], "tiny-enabled")

    def test_stage2_run_id_suffix(self):
        self.assertEqual(bench.stage2_run_id("base-id", True), "base-id-lattn")
        self.assertEqual(bench.stage2_run_id("base-id", False), "base-id")

    def test_cli_flag_exists(self):
        args = run_aliccp_benchmark.build_parser().parse_args(
            ["stage2", "--stage1-id", "s1-x", "--learnable-attenuation"]
        )
        self.assertTrue(args.learnable_attenuation)
        args = run_aliccp_benchmark.build_parser().parse_args(["stage2", "--stage1-id", "s1-x"])
        self.assertFalse(args.learnable_attenuation)


class TestStaticGuards(unittest.TestCase):
    """spec §3 L10：协议零改动、改动面 ⊆ 白名单、SUMMARY 列不变。"""

    def test_protocol_unchanged_since_base(self):
        diff = subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", "aliccp_benchmark/protocol.py"],
                              cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
        self.assertEqual(diff, "")

    def test_changed_files_subset_of_whitelist(self):
        changed = subprocess.run(["git", "diff", "--name-only", BASE_COMMIT], cwd=REPO, check=True,
                                 capture_output=True, text=True).stdout.split()
        self.assertTrue(changed, "应当存在本分支的改动")
        self.assertLessEqual(set(changed), WHITELIST, f"白名单外改动: {set(changed) - WHITELIST}")

    def test_summary_columns_unchanged(self):
        self.assertEqual(P.SUMMARY_COLUMNS, [
            "run_id", "commit", "tag", "auc_val_bsi_best", "auc_test_bsi",
            "A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "stage1_id",
        ])


class TestDocPreregistersSameNumbers(unittest.TestCase):
    """spec §3 L11：文档与代码字面量一致。"""

    def test_doc_preregisters_same_numbers(self):
        doc = DOC.read_text(encoding="utf-8")
        for literal in (
            "c_min = 0.05", "θ_init = 1.0", "0.6043392178311113", "0.5988392178311113",
            "0.5781533414372665", "0.6126857532817985", "0.5919398413138859",
            "0.6115453624066993", "0.6972233730330467", "0.6972233653068542",
            "1e-12", "8130", "8129", "s1-5c060b9c-m1688723512-e3-3a30e2c0",
            "LEARNED_ATTENUATION_SUPPORTED", "IDENTITY_PRESERVED",
        ):
            self.assertIn(literal, doc, f"文档缺少字面量: {literal}")
        self.assertIn(str(metrics.PI_TEST_AUC_MIN), doc)
        self.assertIn(str(metrics.BASELINE_TEST_AUC), doc)


if __name__ == "__main__":
    unittest.main()
