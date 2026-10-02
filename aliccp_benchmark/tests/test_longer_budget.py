"""更长 Stage-2 预算下的持久性检验（longer-budget）：对照说明
`docs/superpowers/specs/2026-10-03-aliccp-stage2-attenuation-longer-budget-design.md`（预注册，先于任何 run）。

- 默认 `spec_attenuation=1.0`：NewTask 必须与分支基点 `8133d32:multitaskrec/model.py` **逐位一致**
  （前向 + 反向 + 参数集合）。
- `spec_attenuation=c`（0 < c ≤ 1）：仅把 specific 混合 `new_spec_rep = Σ_k W_k·spec_rep_k` 乘以固定
  常数 c（本实验 c = 0.6972233730330467，与 seed-1/seed-2 处理臂同实现同系数）。
- 机制逐字守卫：`model.py` 全文件字节 == `f589e61` == seed-1 控制分支；`metrics.py`/`bench.py` 移植项
  AST 段字节相等；`run_stage2` 可执行结构（去 docstring）AST 等价。
- 判定：`persistence_verdict`（Δtest ≥ +0.0055 闭 × Δval > 0 严格 × A 类门禁）→ PERSISTS / NOT_PERSIST。

测试用 CPU 极小夹具，不构成任何性能证据，只验证语义、指标与接线。
"""
import ast
import inspect
import json
import math
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

from aliccp_benchmark import bench, longer_budget as LB, metrics, protocol as P
from multitaskrec.model import NewTask

import run_aliccp_benchmark

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "superpowers" / "specs" / "2026-10-03-aliccp-stage2-attenuation-longer-budget-design.md"
BASE_COMMIT = "8133d32"                                # 本分支基点（infra/aliccp-fair-benchmark 顶端）
MECHANISM_SOURCE = "f589e61"                           # 机制逐字来源（seed-2 复现分支）
SEED1_CONTROL_BRANCH = "exp/aliccp-stage2-specific-attenuation-control"   # 机制最终源头（seed-1 处理臂）
COEF = 0.6972233730330467                              # 预注册系数（§3；不动）
SEED2 = 1688723740                                     # 预注册第二 seed（§3）
SEED1 = 1688723512
STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"      # seed-2 固定 Stage-1 产物（§3）
BATCH, INPUT_SIZE, REP_DIM, NUM_SRC = 6, 8, 4, 2


# ---- 静态守卫工具 ----
def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="strict").stdout


def _show(rev_path):
    return _git("show", rev_path)


def _top_level_segments(source_text):
    """模块顶层 FunctionDef / ClassDef / 单目标 Assign 的源码段（ast.get_source_segment 原样切取）。"""
    segs = {}
    for node in ast.parse(source_text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            segs[node.name] = ast.get_source_segment(source_text, node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            segs[node.targets[0].id] = ast.get_source_segment(source_text, node)
    return segs


def _function_ast_dump(source_text, name, drop_docstring=False):
    tree = ast.parse(source_text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            if drop_docstring and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    node.body = node.body[1:]
            return ast.dump(node)
    raise AssertionError(f"源码中找不到函数: {name}")


def _base_newtask_class():
    """从基点提交动态加载 NewTask 作为逐位对照的参考实现（与 seed-2 复现测试同构）。"""
    src = _show(f"{BASE_COMMIT}:multitaskrec/model.py")
    module = types.ModuleType("model_base")
    exec(compile(src, f"{BASE_COMMIT}:multitaskrec/model.py", "exec"), module.__dict__)
    return module.NewTask


def _ours(spec_attenuation=1.0):
    return NewTask(input_size=INPUT_SIZE, rep_dim=REP_DIM, tower_dnn_hidden_units=(4, 2),
                   reg_dnn=P.REG_DNN, device=torch.device("cpu"), spec_attenuation=spec_attenuation)


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


class TestMechanismStaticGuards(unittest.TestCase):
    """I2/I7：机制逐字复用 + 跟踪改动白名单 + 协议文件零改动。"""

    WHITELIST = {
        "multitaskrec/model.py",
        "aliccp_benchmark/metrics.py",
        "aliccp_benchmark/bench.py",
        "aliccp_benchmark/longer_budget.py",
        "aliccp_benchmark/tests/test_longer_budget.py",
        "run_aliccp_benchmark.py",
        "docs/superpowers/specs/2026-10-03-aliccp-stage2-attenuation-longer-budget-design.md",
        "artifacts/aliccp_bench/SUMMARY.md",
    }

    def test_model_py_byte_identical_to_replication_and_seed1(self):
        """处理臂实现必须与 f589e61 及 seed-1 控制分支逐字相同（唯一变量是预算；实现漂移破坏可比性）。"""
        current = (REPO / "multitaskrec" / "model.py").read_text(encoding="utf-8")
        self.assertEqual(current, _show(f"{MECHANISM_SOURCE}:multitaskrec/model.py"),
                         "model.py 与 f589e61 逐字来源不一致")
        self.assertEqual(current, _show(f"{SEED1_CONTROL_BRANCH}:multitaskrec/model.py"),
                         "model.py 与 seed-1 控制分支逐字来源不一致")

    def test_metrics_ported_items_ast_segments_identical(self):
        """常量块 + SourceGateStats/_f64/pred_dispersion 与 f589e61 逐字相同（AST 段字节比较）。"""
        ours = _top_level_segments((REPO / "aliccp_benchmark" / "metrics.py").read_text(encoding="utf-8"))
        source = _top_level_segments(_show(f"{MECHANISM_SOURCE}:aliccp_benchmark/metrics.py"))
        for name in ("CANONICAL_ALICCP_SEED_LIST", "PROTOCOL_MODEL_SEED", "REPLICATION_MODEL_SEED",
                     "NULL_RUN_NULL_MEAN", "SPEC_ATTENUATION_COEF", "SPEC_ATTENUATION_COEF_F32",
                     "SourceGateStats", "_f64", "pred_dispersion"):
            self.assertIn(name, ours, f"metrics.py 缺少移植项: {name}")
            self.assertEqual(ours[name], source[name], f"移植项与来源不一致: {name}")

    def test_bench_ported_functions_ast_segments_identical(self):
        ours = _top_level_segments((REPO / "aliccp_benchmark" / "bench.py").read_text(encoding="utf-8"))
        source = _top_level_segments(_show(f"{MECHANISM_SOURCE}:aliccp_benchmark/bench.py"))
        for name in ("stage2_run_id", "newtask_attenuation_probe"):
            self.assertIn(name, ours, f"bench.py 缺少移植函数: {name}")
            self.assertEqual(ours[name], source[name], f"移植函数与来源不一致: {name}")

    def test_run_stage2_executable_structure_identical(self):
        """run_stage2 的可执行结构（AST，去 docstring）与 f589e61 等价；docstring 的文档引用允许适配。"""
        ours = _function_ast_dump((REPO / "aliccp_benchmark" / "bench.py").read_text(encoding="utf-8"),
                                  "run_stage2", drop_docstring=True)
        source = _function_ast_dump(_show(f"{MECHANISM_SOURCE}:aliccp_benchmark/bench.py"),
                                    "run_stage2", drop_docstring=True)
        self.assertEqual(ours, source, "run_stage2 可执行结构与来源不一致")

    def test_protocol_file_untouched_since_branch_base(self):
        diff = _git("diff", "--name-only", BASE_COMMIT, "--", "aliccp_benchmark/protocol.py").split()
        self.assertEqual(diff, [], f"协议文件不得修改: {diff}")

    def test_tracked_diff_within_whitelist(self):
        changed = _git("diff", "--name-only", BASE_COMMIT).split()
        extra = [p for p in changed if p not in self.WHITELIST]
        self.assertEqual(extra, [], f"白名单外的跟踪文件被改动: {extra}")


class TestDefaultArmBitIdenticalToBase(unittest.TestCase):
    """I1：默认臂（spec_attenuation=1.0）必须是基点实现的逐位复制。"""

    def test_default_state_dict_matches_base(self):
        ours = _ours()
        self.assertEqual(set(ours.state_dict()), set(_base().state_dict()))
        ours.load_state_dict(_base().state_dict())                        # strict=True：参数集合必须一致
        self.assertNotIn("spec_attenuation", ours.state_dict())           # 衰减不是参数、不进 state_dict

    def test_default_forward_bit_identical(self):
        base, ours, inputs = _base(), _ours(), _inputs(1)
        ours.load_state_dict(base.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(ours(*inputs[:4]), base(*inputs[:4])))

    def test_default_backward_bit_identical(self):
        torch.manual_seed(0)  # 固定构造 RNG：部分随机初始化下 tower 的 ReLU 全灭会零化整条 specific 路径梯度
        base, inputs = _base(), _inputs(2)
        torch.manual_seed(0)
        ours = _ours()                                                    # 构造顺序相同 → 初始化逐位一致
        pred_base, pred_ours = _forward_backward(base, inputs), _forward_backward(ours, inputs)
        self.assertTrue(torch.equal(pred_base, pred_ours))
        grads_ours = dict(ours.named_parameters())
        for name, param in base.named_parameters():
            self.assertIsNotNone(param.grad, f"基点该参数无梯度，反向对照无效: {name}")
            self.assertTrue(torch.equal(param.grad, grads_ours[name].grad), f"梯度非逐位一致: {name}")

    def test_attenuation_does_not_consume_rng(self):
        torch.manual_seed(3)
        off = _ours()
        torch.manual_seed(3)
        on = _ours(spec_attenuation=COEF)
        for name, param in off.named_parameters():
            self.assertTrue(torch.equal(param, dict(on.named_parameters())[name]), f"共享参数初始化漂移: {name}")
        self.assertEqual(on.spec_attenuation, COEF)

    def test_default_spec_attenuation_is_one(self):
        self.assertEqual(_ours().spec_attenuation, 1.0)


class TestAttenuationSemantics(unittest.TestCase):
    """I3：开启衰减后的公式、参数集合、梯度、合法性与系数钉死。"""

    def test_coefficient_literals(self):
        self.assertEqual(metrics.SPEC_ATTENUATION_COEF, 1.0 - 0.3027766269669533)
        self.assertEqual(metrics.SPEC_ATTENUATION_COEF, COEF)
        self.assertEqual(1.0 - metrics.NULL_RUN_NULL_MEAN, COEF)
        self.assertEqual(repr(metrics.SPEC_ATTENUATION_COEF_F32), repr(float(torch.tensor(COEF, dtype=torch.float32))))
        self.assertTrue(0.0 < metrics.SPEC_ATTENUATION_COEF <= 1.0)

    def test_seed_constants_pinned(self):
        self.assertEqual(metrics.CANONICAL_ALICCP_SEED_LIST,
                         (1688723512, 1688723740, 1688738016, 1688749593, 1688762746))
        self.assertEqual(metrics.REPLICATION_MODEL_SEED, metrics.CANONICAL_ALICCP_SEED_LIST[1])
        self.assertEqual(metrics.REPLICATION_MODEL_SEED, SEED2)
        self.assertEqual(metrics.PROTOCOL_MODEL_SEED, SEED1)
        self.assertEqual(metrics.PROTOCOL_MODEL_SEED, P.MODEL_SEED)      # 协议默认值未被本分支改动
        self.assertNotEqual(metrics.REPLICATION_MODEL_SEED, P.MODEL_SEED)

    def test_invalid_coefficients_raise(self):
        for bad in (0.0, -0.5, 1.0000001, 2.0):
            with self.assertRaises(ValueError, msg=f"c={bad} 应拒绝"):
                _ours(spec_attenuation=bad)
        _ours(spec_attenuation=1.0)          # 合法边界
        _ours(spec_attenuation=COEF)

    def test_enabled_forward_matches_reference_formula(self):
        """写死公式：new_spec_rep' = c × Σ_k W_k spec_rep_k（同温度 softmax，其余同基点）。"""
        ours, inputs = _ours(spec_attenuation=COEF), _inputs(7)
        dnn_input, gen_rep, spec_reps, env_embs, _ = inputs
        keys = torch.stack(env_embs, dim=1)
        logits = torch.mm(ours.projection_network(dnn_input), keys) / ours.temperature
        weights = F.softmax(logits, dim=-1).unsqueeze(2)
        new_spec_rep = torch.matmul(torch.stack(spec_reps, dim=2), weights).squeeze()
        new_spec_rep = new_spec_rep * ours.spec_attenuation
        env_aware_rep = new_spec_rep * ours.env_embedding_network(ours.new_env_idx).squeeze(0)
        fused_rep = torch.matmul(torch.stack([env_aware_rep, gen_rep], dim=2),
                                 ours.gate_network(dnn_input).unsqueeze(dim=2)).squeeze()
        expected = ours.tower_network(fused_rep).squeeze()
        with torch.no_grad():
            self.assertTrue(torch.equal(ours(*inputs[:4]), expected))

    def test_no_new_parameters_and_params_count_equal(self):
        base, ours = _base(), _ours(spec_attenuation=COEF)
        self.assertEqual(set(ours.state_dict()), set(base.state_dict()))
        self.assertEqual(set(dict(ours.named_parameters())), set(dict(base.named_parameters())))
        count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.assertEqual(count(ours), count(base))

    def test_get_l2_reg_unchanged(self):
        base, ours, inputs = _base(), _ours(spec_attenuation=COEF), _inputs(9)
        ours.load_state_dict(base.state_dict())
        self.assertTrue(torch.equal(ours.get_l2_reg(), base.get_l2_reg()))


class TestLongerBudgetPrereg(unittest.TestCase):
    """I5/I8：预注册常量与判定边界钉死；文档 token 一致。"""

    def test_pinned_constants(self):
        self.assertEqual(LB.LONGER_BUDGET_EPOCHS, 10)
        self.assertEqual(LB.LONGER_BUDGET_PATIENCE, 3)
        self.assertEqual(LB.LONGER_BUDGET_TAG, "long")
        self.assertEqual(LB.LONGER_BUDGET_DELTA_TEST_MIN, 0.0055)
        self.assertEqual(LB.SEED2_STAGE1_ID, STAGE1_ID)
        self.assertEqual(LB.SEED2_MODEL_SEED, SEED2)
        self.assertEqual(LB.SEED2_MODEL_SEED, metrics.REPLICATION_MODEL_SEED)
        self.assertEqual(LB.SPEC_ATTENUATION_COEF, metrics.SPEC_ATTENUATION_COEF)
        # 5-epoch context 钉死值（先例记录，逐位取自 on-disk run metrics.json）
        self.assertEqual(LB.SEED2_5EPOCH_BASELINE_RUN_ID, "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07")
        self.assertEqual(LB.SEED2_5EPOCH_ARM_RUN_ID, "20261003-0627-p2M-v500k-t1M-m1688723740-short-79b5e07-sattn")
        self.assertEqual(LB.SEED2_5EPOCH_BASELINE_TEST_AUC, 0.5974422649550507)
        self.assertEqual(LB.SEED2_5EPOCH_BASELINE_VAL_AUC, 0.5809347091990792)
        self.assertEqual(LB.SEED2_5EPOCH_ARM_TEST_AUC, 0.6051559662367624)
        self.assertEqual(LB.SEED2_5EPOCH_ARM_VAL_AUC, 0.5854178317514255)
        self.assertEqual(LB.SEED2_5EPOCH_DELTA_TEST, 0.007713701281711671)
        self.assertEqual(LB.SEED2_5EPOCH_DELTA_VAL, 0.004483122552346286)
        self.assertAlmostEqual(LB.SEED2_5EPOCH_ARM_TEST_AUC - LB.SEED2_5EPOCH_BASELINE_TEST_AUC,
                               LB.SEED2_5EPOCH_DELTA_TEST, places=15)
        self.assertAlmostEqual(LB.SEED2_5EPOCH_ARM_VAL_AUC - LB.SEED2_5EPOCH_BASELINE_VAL_AUC,
                               LB.SEED2_5EPOCH_DELTA_VAL, places=15)

    def test_analyze_defaults_match_prereg(self):
        sig = inspect.signature(LB.analyze_runs)
        self.assertEqual(sig.parameters["expected_stage1_id"].default, LB.SEED2_STAGE1_ID)
        self.assertEqual(sig.parameters["expected_model_seed"].default, LB.SEED2_MODEL_SEED)
        self.assertEqual(sig.parameters["expected_epochs"].default, LB.LONGER_BUDGET_EPOCHS)
        self.assertEqual(sig.parameters["expected_patience"].default, LB.LONGER_BUDGET_PATIENCE)

    def test_verdict_boundary_delta_test_min(self):
        # 用 0.0 基线使 Δ 在 float64 下精确可表示（0.60+0.0055−0.60 ≠ 0.0055，不能用作边界构造）
        ok = LB.persistence_verdict(baseline_test_auc=0.0, baseline_val_auc=0.0, arm_test_auc=0.0055,
                                    arm_val_auc=1e-6, a_class_pass=True)
        self.assertTrue(ok["checks"]["delta_test_ge_min"])                       # 恰好 0.0055 → 通过（闭）
        self.assertEqual(ok["classification"], "PERSISTS")
        below = LB.persistence_verdict(baseline_test_auc=0.0, baseline_val_auc=0.0, arm_test_auc=0.0055 - 1e-12,
                                       arm_val_auc=1e-6, a_class_pass=True)
        self.assertFalse(below["checks"]["delta_test_ge_min"])
        self.assertEqual(below["classification"], "NOT_PERSIST")

    def test_verdict_boundary_val_direction_is_strict(self):
        flat = LB.persistence_verdict(baseline_test_auc=0.0, baseline_val_auc=0.0, arm_test_auc=0.01,
                                      arm_val_auc=0.0, a_class_pass=True)
        self.assertFalse(flat["checks"]["delta_val_positive"])                   # 恰好 0 → 不通过（严格）
        self.assertEqual(flat["classification"], "NOT_PERSIST")
        above = LB.persistence_verdict(baseline_test_auc=0.0, baseline_val_auc=0.0, arm_test_auc=0.01,
                                       arm_val_auc=1e-12, a_class_pass=True)
        self.assertTrue(above["checks"]["delta_val_positive"])
        self.assertEqual(above["classification"], "PERSISTS")

    def test_verdict_requires_a_class_pass(self):
        v = LB.persistence_verdict(baseline_test_auc=0.0, baseline_val_auc=0.0, arm_test_auc=0.01,
                                   arm_val_auc=0.01, a_class_pass=False)
        self.assertTrue(v["checks"]["delta_test_ge_min"])
        self.assertTrue(v["checks"]["delta_val_positive"])
        self.assertFalse(v["checks"]["a_class_pass"])
        self.assertEqual(v["classification"], "NOT_PERSIST")
        self.assertEqual(v["delta_test_min"], 0.0055)

    def test_doc_preregisters_tokens(self):
        text = DOC.read_text(encoding="utf-8")
        for token in ("exp/aliccp-stage2-attenuation-longer-budget", "8133d32", "f589e61", "3314420",
                      "s1-5c060b9c-m1688723740-e3-4e1b5c6f", "1688723740", "20261003",
                      "0.6972233730330467", "0.6972233653068542", "0.3027766269669533",
                      "0.0055", "5→10", "2→3", "\"long\"", "PERSISTS", "NOT_PERSIST", "INVALID",
                      "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07",
                      "20261003-0627-p2M-v500k-t1M-m1688723740-short-79b5e07-sattn",
                      "0.007713701281711671", "0.004483122552346286",
                      "0.5974422649550507", "0.5809347091990792",
                      "0.6051559662367624", "0.5854178317514255",
                      "e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c",
                      "5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0",
                      "cd7b033423499e0d36eea988835cea4ac4e2a8e26427a2aea627333822101e0b",
                      "4660be5aaa3c59f53dd5b4394f77114a87db69064b32c45517db049a6b9157e7",
                      "61a66d81ce3dcf6bae64f4c2b37cf12e722943def646336a588746aba932931d",
                      "动机与事后对照", "不重训 Stage-1"):
            self.assertIn(token, text, f"预注册文档未写死: {token}")


class TestEpochPatienceOverride(unittest.TestCase):
    """I4：CLI 默认逐位不变；override 解析与 main() 线程化；夹具端到端记录。"""

    def test_cli_defaults_bit_identical(self):
        parser = run_aliccp_benchmark.build_parser()
        s2 = parser.parse_args(["stage2", "--stage1-id", "x"])
        self.assertEqual(s2.epochs, P.STAGE2_EPOCHS)
        self.assertEqual(s2.epochs, 5)
        self.assertEqual(s2.patience, P.STAGE2_PATIENCE)
        self.assertEqual(s2.patience, 2)
        self.assertEqual(s2.spec_attenuation, 1.0)
        self.assertEqual(s2.tag, "short")
        self.assertEqual(s2.model_seed, P.MODEL_SEED)
        s1 = parser.parse_args(["stage1"])
        self.assertEqual(s1.epochs, P.STAGE1_EPOCHS)
        self.assertEqual(s1.patience, P.STAGE1_PATIENCE)

    def test_tag_long_accepted_default_unchanged(self):
        parser = run_aliccp_benchmark.build_parser()
        self.assertEqual(parser.parse_args(["stage2", "--stage1-id", "x", "--tag", "long"]).tag, "long")
        self.assertEqual(parser.parse_args(["stage1", "--tag", "long"]).tag, "long")
        self.assertEqual(parser.parse_args(["stage1"]).tag, "short")       # 默认不变

    def test_override_parses(self):
        args = run_aliccp_benchmark.build_parser().parse_args(
            ["stage2", "--stage1-id", "x", "--epochs", "10", "--patience", "3", "--spec-attenuation", str(COEF)])
        self.assertEqual((args.epochs, args.patience, args.spec_attenuation), (10, 3, COEF))

    def test_main_threads_override_kwargs(self):
        """预注册配置经 main() 线程化到 run_stage2 调用参数（捕获 kwargs，不训练）。"""
        captured = {}
        fake = {"gates": {g: {"verdict": "PASS"} for g in ("A1", "A2", "A3", "A4", "A5", "A6",
                                                           "B1", "B2", "B3", "B4")},
                "run_id": "fake-run", "hard_pass": True, "metrics": {}}

        def fake_stage2(**kwargs):
            captured.update(kwargs)
            return fake

        with tempfile.TemporaryDirectory() as td, mock.patch.object(bench, "run_stage2", fake_stage2):
            rc = run_aliccp_benchmark.main(["stage2", "--stage1-id", "sid", "--epochs", "10", "--patience", "3",
                                            "--tag", "long", "--model-seed", str(SEED2), "--root", td])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["epochs"], 10)
        self.assertEqual(captured["patience"], 3)
        self.assertEqual(captured["spec_attenuation"], 1.0)
        self.assertEqual(captured["tag"], "long")
        self.assertEqual(captured["model_seed"], SEED2)
        self.assertEqual(captured["stage1_id"], "sid")
        self.assertTrue(captured["enforce_b"])                             # tag != smoke → B 类判定
        self.assertNotIn("sattn", captured["run_id"])

    def test_main_threads_arm_kwargs_and_run_id_suffix(self):
        captured = {}
        fake = {"gates": {g: {"verdict": "PASS"} for g in ("A1", "A2", "A3", "A4", "A5", "A6",
                                                           "B1", "B2", "B3", "B4")},
                "run_id": "fake-run-sattn", "hard_pass": True,
                "metrics": {"probe": {"pred_mean": 0.5, "pred_std": 0.1, "source_gate_mean": [0.5, 0.5]},
                            "trainable_params": 1}}

        def fake_stage2(**kwargs):
            captured.update(kwargs)
            return fake

        with tempfile.TemporaryDirectory() as td, mock.patch.object(bench, "run_stage2", fake_stage2):
            rc = run_aliccp_benchmark.main(["stage2", "--stage1-id", "sid", "--epochs", "10", "--patience", "3",
                                            "--tag", "long", "--model-seed", str(SEED2), "--root", td,
                                            "--spec-attenuation", str(COEF)])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["spec_attenuation"], COEF)
        self.assertTrue(captured["run_id"].endswith("-sattn"))


# ---- 极小 AliCCP 夹具（与 seed-2 复现测试同构，自包含）----
HEADER = "click,purchase,X,121,122,301"
TINY_VOCAB = {"121": 5, "122": 4}
TRAIN_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3], [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3], [1, 0, 7, 1, 1, 2], [0, 0, 9, 2, 2, 1], [1, 1, 9, 0, 1, 3],
    [0, 0, 8, 3, 0, 2], [1, 0, 8, 1, 1, 1], [0, 0, 7, 4, 2, 3], [1, 0, 7, 2, 1, 2],
]
VAL_ROWS = [
    [0, 0, 9, 1, 0, 1], [1, 0, 9, 2, 1, 2], [0, 0, 8, 0, 2, 3], [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3], [1, 0, 7, 1, 1, 2],
]
TEST_ROWS = [
    [0, 0, 9, 2, 2, 1], [1, 0, 9, 0, 1, 3], [0, 0, 8, 3, 0, 2], [1, 1, 8, 1, 1, 1],
    [0, 0, 7, 4, 2, 3], [1, 0, 7, 2, 1, 2], [0, 0, 9, 1, 0, 3], [1, 0, 8, 4, 1, 2],
]
TINY_DIMS = dict(expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4)


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
    data_files = {
        "train": str(data_dir / "train.csv"),
        "val": str(data_dir / "val.csv"),
        "test": str(data_dir / "test.csv"),
    }
    budgets = {"train": len(TRAIN_ROWS), "val": len(VAL_ROWS), "test": len(TEST_ROWS)}
    return root, data_files, budgets


class TestStage2FixtureOverrideAndArm(unittest.TestCase):
    """run_stage2 预算 override 与衰减臂落盘（CPU 极小夹具；只验证语义与接线）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root, cls.data_files, cls.budgets = _make_tiny_root(cls._tmp.name)
        cls.device = torch.device("cpu")
        cls.noop = staticmethod(lambda *a, **k: None)
        cls.meta = bench.run_stage1(
            root=cls.root, data_files=cls.data_files, budgets=cls.budgets, prefix_tag="tiny",
            model_seed=SEED2, env_seed=456, epochs=2, patience=2, device=cls.device, vocab=TINY_VOCAB,
            **TINY_DIMS, log=cls.noop,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run(self, **kwargs):
        params = dict(root=self.root, stage1_id=self.meta["stage1_id"], data_files=self.data_files,
                      budgets=self.budgets, prefix_tag="tiny", model_seed=SEED2, tag="smoke",
                      device=self.device, vocab=TINY_VOCAB, **TINY_DIMS, enforce_b=False, log=self.noop)
        params.update(kwargs)
        return bench.run_stage2(**params)

    def _json(self, out, name):
        return json.loads((Path(out["run_dir"]) / name).read_text(encoding="utf-8"))

    def test_override_epochs_patience_recorded_and_trajectory_respects_budget(self):
        out = self._run(epochs=3, patience=1)
        recorded = self._json(out, "metrics.json")
        self.assertEqual(recorded["epochs"], 3)
        self.assertEqual(recorded["patience"], 1)
        traj = [r["val_auc_bsi"] for r in recorded["per_epoch"]]
        self.assertLessEqual(len(traj), 3)
        self.assertGreaterEqual(len(traj), 1)
        # 预注册选点规则：strict > 首次取到最大值者；best checkpoint 即该 epoch
        self.assertEqual(recorded["best_val_auc_bsi"], max(traj))
        self.assertEqual(recorded["best_epoch"], traj.index(max(traj)) + 1)
        if len(traj) < 3:                    # patience=1：停止发生在首个非改进 epoch
            self.assertLessEqual(traj[-1], max(traj[:-1]))

    def test_default_and_arm_records(self):
        baseline = self._run(epochs=1, patience=1)
        arm = self._run(epochs=1, patience=1, spec_attenuation=COEF)
        self.assertNotIn("sattn", baseline["run_id"])
        self.assertTrue(arm["run_id"].endswith("-sattn"))
        base_rec, arm_rec = self._json(baseline, "metrics.json"), self._json(arm, "metrics.json")
        self.assertEqual(base_rec["spec_attenuation"], 1.0)
        self.assertNotIn("probe", base_rec)
        self.assertNotIn("trainable_params", base_rec)
        self.assertEqual(arm_rec["spec_attenuation"], COEF)
        probe = arm_rec["probe"]
        for key in ("pred_mean", "pred_std", "pred_var", "pred_min", "pred_max",
                    "pred_q10", "pred_q25", "pred_q50", "pred_q75", "pred_q90", "source_gate_mean"):
            self.assertIn(key, probe, key)
        self.assertEqual(len(probe["source_gate_mean"]), 2)
        self.assertEqual(arm_rec["trainable_params"], arm_rec["trainable_params_default"])
        self.assertEqual(arm_rec["backbone_sha256_before"], base_rec["backbone_sha256_before"])
        self.assertEqual(arm_rec["backbone_sha256_before"], arm_rec["backbone_sha256_after"])
        self.assertTrue(arm_rec["backbone_grads_none"])


class TestLongerBudgetAnalyzer(unittest.TestCase):
    """I6：分析器（只读、判定、identity、context、CLI）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root, cls.data_files, cls.budgets = _make_tiny_root(cls._tmp.name)
        cls.device = torch.device("cpu")
        cls.noop = staticmethod(lambda *a, **k: None)
        cls.meta = bench.run_stage1(
            root=cls.root, data_files=cls.data_files, budgets=cls.budgets, prefix_tag="tiny",
            model_seed=SEED2, env_seed=456, epochs=2, patience=2, device=cls.device, vocab=TINY_VOCAB,
            **TINY_DIMS, log=cls.noop,
        )
        common = dict(root=cls.root, stage1_id=cls.meta["stage1_id"], data_files=cls.data_files,
                      budgets=cls.budgets, prefix_tag="tiny", model_seed=SEED2, epochs=1, patience=1,
                      tag="smoke", device=cls.device, vocab=TINY_VOCAB, **TINY_DIMS,
                      enforce_b=False, log=cls.noop)
        cls.baseline = bench.run_stage2(**common)
        cls.arm = bench.run_stage2(**common, spec_attenuation=COEF)
        cls.expect_kwargs = dict(expected_stage1_id=cls.meta["stage1_id"], expected_model_seed=SEED2,
                                 expected_epochs=1, expected_patience=1)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_analyze_end_to_end_and_output(self):
        out_path = Path(self.arm["run_dir"]) / "longer_budget_compare.json"
        result = LB.analyze_runs(root=self.root, baseline_run_id=self.baseline["run_id"],
                                 arm_run_id=self.arm["run_id"], output_path=out_path, **self.expect_kwargs)
        self.assertTrue(all(result["identity_checks"].values()), result["identity_checks"])
        self.assertIn(result["classification"], {"PERSISTS", "NOT_PERSIST"})
        self.assertEqual(result["classification"], result["verdict"]["classification"])
        # 判定取自记录值（float64 重算）
        base_rec = json.loads((P.run_dir(self.root, self.baseline["run_id"]) / "metrics.json").read_text("utf-8"))
        arm_rec = json.loads((P.run_dir(self.root, self.arm["run_id"]) / "metrics.json").read_text("utf-8"))
        self.assertAlmostEqual(result["verdict"]["delta_test_auc"],
                               arm_rec["test_auc_bsi"] - base_rec["test_auc_bsi"], places=15)
        self.assertAlmostEqual(result["verdict"]["delta_val_auc"],
                               arm_rec["best_val_auc_bsi"] - base_rec["best_val_auc_bsi"], places=15)
        # 描述量：best epoch / 早停 / 轨迹
        for arm in ("baseline", "arm"):
            desc = result["description"][arm]
            for key in ("best_epoch", "epochs_run", "stopped_early", "best_is_last_epoch",
                        "val_traj", "wall_seconds", "gate_mean"):
                self.assertIn(key, desc, key)
            self.assertEqual(desc["epochs_run"], len(desc["val_traj"]))
            self.assertEqual(desc["best_epoch"], desc["val_traj"].index(max(desc["val_traj"])) + 1)
        self.assertIn("probe", result["description"]["arm"])               # 处理臂探针（从记录读取）
        self.assertEqual(result["description"]["arm"]["trainable_params"],
                         result["description"]["arm"]["trainable_params_default"])
        # context：5-epoch 钉死值，只作对照
        ctx = result["context"]
        self.assertEqual(ctx["five_epoch"]["delta_test_auc"], LB.SEED2_5EPOCH_DELTA_TEST)
        self.assertEqual(ctx["five_epoch"]["delta_val_auc"], LB.SEED2_5EPOCH_DELTA_VAL)
        self.assertEqual(ctx["five_epoch"]["baseline_run_id"], LB.SEED2_5EPOCH_BASELINE_RUN_ID)
        self.assertAlmostEqual(ctx["five_epoch"]["delta_test_auc"],
                               LB.SEED2_5EPOCH_ARM_TEST_AUC - LB.SEED2_5EPOCH_BASELINE_TEST_AUC, places=15)
        self.assertAlmostEqual(ctx["delta_test_change_vs_five_epoch"],
                               result["verdict"]["delta_test_auc"] - LB.SEED2_5EPOCH_DELTA_TEST, places=15)
        self.assertTrue(out_path.exists())
        on_disk = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["classification"], result["classification"])

    def test_analyze_identity_mismatch_flagged_invalid(self):
        result = LB.analyze_runs(root=self.root, baseline_run_id=self.baseline["run_id"],
                                 arm_run_id=self.arm["run_id"], expected_stage1_id="s1-wrong",
                                 expected_model_seed=SEED2, expected_epochs=1, expected_patience=1)
        self.assertEqual(result["classification"], "INVALID")
        self.assertFalse(result["identity_checks"]["stage1_id_is_pinned"])
        self.assertFalse(all(result["identity_checks"].values()))

    def test_analyze_epochs_mismatch_flagged_invalid(self):
        result = LB.analyze_runs(root=self.root, baseline_run_id=self.baseline["run_id"],
                                 arm_run_id=self.arm["run_id"], expected_stage1_id=self.meta["stage1_id"],
                                 expected_model_seed=SEED2, expected_epochs=10, expected_patience=3)
        self.assertEqual(result["classification"], "INVALID")
        self.assertFalse(result["identity_checks"]["epochs_recorded_pinned"])

    def test_analyze_is_read_only(self):
        run_path = P.run_dir(self.root, self.arm["run_id"])
        before = {p.name: p.read_bytes() for p in run_path.iterdir() if p.is_file()}
        LB.analyze_runs(root=self.root, baseline_run_id=self.baseline["run_id"],
                        arm_run_id=self.arm["run_id"], **self.expect_kwargs)
        after = {p.name: p.read_bytes() for p in run_path.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_cli_main_writes_invalid_for_non_prereg_runs(self):
        """main() 走预注册默认 expected_*；夹具 run 不满足 → INVALID（完备性分支端到端）。"""
        out_path = Path(self.arm["run_dir"]) / "cli_compare.json"
        rc = LB.main(["--root", str(self.root), "--baseline-run", self.baseline["run_id"],
                      "--arm-run", self.arm["run_id"], "--output", str(out_path)])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out_path.read_text(encoding="utf-8"))["classification"], "INVALID")


if __name__ == "__main__":
    unittest.main()
