"""忠实复现历史 Affinity Gate（OOD-aware 硬切换）的门控退化测试。

唯一事实来源：docs/superpowers/experiments/2026-09-30-stage2-affinity-gate-revalidation.md
历史实现（审计对象）：`archive/exploration` 分支的
  * `run_newtask_from_ckpt.py`（`ALL_MODES` 含 `"affinity_gate"`；Adam/BCE/val 选点/test 只评一次）
  * `multitaskrec/model.py` 的 `NewTask(fusion_mode="affinity_gate")`（双 sigmoid、全局 RNG 抽样、
    硬 STE、门控不进 L2）

本文件锁定九件事（I1–I9，与 experiments 文档第 4 节一致）：
  I1 **忠实公式**：cos_sim / 4 维特征 / g_logit / g_soft / g_hard / g / W_attn / W_fw / W 与历史逐项同式
     （人工复算逐位 `torch.equal`），且与**归档类**（`git show archive/exploration:multitaskrec/model.py`
     里真正的 `NewTask`）在同参数、同 seed 下**输出与梯度逐位一致**；
  I2 **忠实构造**：历史构造顺序（`affinity_mlp` 建在 projection 与 gate 之间，多消耗全局 RNG）、
     默认初始化（无常量初始化）、`state_dict` 键集 = 基线 + `affinity_mlp.*`；
  I3 **D1 双重 sigmoid**：eval 期 `g_hard ≡ 1`（对任意参数/输入）、`g_soft ∈ (0.5, sigmoid(2)]`、
     `W` 与 `W_attn` 的逐位差 ≤ STE 舍入量级（`routing_l1_max < 1e-6`，见 D1 精度留痕——**不得**写成
     "逐位恒等"）；训练期路由概率被夹在 `(0.6321, 0.9340)` 解析窗口内；
  I4 **D2 复核（撤回一条 + 保留一条）**：`env_embs` 元素是一维 `[rep_dim]` ⇒ `stack(dim=1)` = `[rep_dim, K]`，
     `normalize(E, dim=0)` 归一化的**就是特征维**（逐任务列）⇒ `cos_sim` 是真余弦——早期草稿"归一化维
     错误（应为 dim=-1）"的结论**已撤回**（布局守卫 + 归档 mm 形状佐证锁定）；保留的 D2 是设计层问题：
     4 维门控特征与 `W_attn` 同源（同一 `H(x)`），K=2 时第 4 维 `mean|C| = (|max|+|min|)/2` 是前三维的
     函数（恒等式，测试锁定）⇒ 门控实际只看到 2 个数；
  I5 **D3 全局 RNG**：训练期前向消耗全局 RNG、eval 期不消耗；探针一律走私有 Generator（不动全局 RNG）；
  I6 **D4/D5 梯度与正则**：affinity_mlp 梯度读数正确、修正路径（z 当 logit）的局部灵敏度上限 `1/τ = 2`
     远大于历史路径的 `max σ′ = 0.25`；`get_l2_reg()` 不含 affinity_mlp；
  I7 **D6 修正口径**：`corrected_routing_share_attn` = `P(z > 0)` 与 `corrected_gate_mean` = `mean σ(z/τ)`
     正确（无噪、同参数、输入相关 ⇒ 可检验）；
  I8 **预注册判据**：机制 STOP 三条（`eval_share >= 0.999` / `routing_l1 < 1e-6` / `grad_norm == 0`）
     + 保真 STOP（`noise_window > 0.32`，解析上界 0.3019）；**STOP 主导于 AUC**（效应阈值 0.8521 仅在
     无 STOP 时才解释）；
  I9 **接线**：`run_id` 后缀 `-affinity`、config / metrics 记录；基线臂不受影响（逐位同 master NewTask）。

定位：这不是新方法（硬门控 / 门控融合是已有大类，历史 affinity_gate 也从未超越 prompt），
而是一次**忠实复现 + 缺陷留痕**；失败结果按用户要求保留在本分支、永不合并 master。
"""
import ast
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import run_census_benchmark
from census_benchmark import affinity_gate as AFF
from census_benchmark import protocol as P
from multitaskrec.model import MLP, NewTask

# 与 test_protocol / test_smoke / test_cgr 同口径：dnn_input = 2 特征 × embedding 4 = 8
TINY_VOCAB, TINY_INPUT_SIZE, TINY_EMBEDDING, TINY_REP_DIM = {"a": 3, "b": 2}, 8, 4, 4
REPO = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parents[1] / "affinity_gate.py"
DOC = REPO / "docs" / "superpowers" / "experiments" / "2026-09-30-stage2-affinity-gate-revalidation.md"
ARCHIVE_REF = "archive/exploration"
# git 吐出的是 UTF-8 字节；Windows 中文环境下 `text=True` 默认按本地编码（GBK/cp936）解码，
# 读取线程抛 UnicodeDecodeError、`stdout` 变 None，下游 `ast.parse(None)` 才报成 TypeError
# （本文件最初的导入失败点）。所有 git 文本输出因此统一走 `_git_text()` 显式钉死 UTF-8；
# 严格模式（默认 `errors="strict"`）是有意为之：归档源码一旦损坏必须立刻报错，
# 不得静默降级成"取不到 ⇒ skip"。
GIT_TEXT_ENCODING = "utf-8"


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
    """处理臂头：忠实复现的历史 Affinity Gate（子类，不动 model.py）。"""
    return AFF.HistoricalAffinityGateNewTask(
        input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=list(P.TOWER_HIDDEN),
        reg_dnn=P.REG_DNN, device=device)


def tiny_reps(device=None, n=8, seed=3):
    """抽一批表征：(dnn_input, gen_rep, spec_reps, env_embs)——与真实路径同款 no_grad 抽取。"""
    gen = torch.Generator().manual_seed(seed)
    features = {"a": torch.randint(0, TINY_VOCAB["a"], (n,), generator=gen),
                "b": torch.randint(0, TINY_VOCAB["b"], (n,), generator=gen)}
    with torch.no_grad():
        return tiny_backbone(device).get_infos(features)


def _git_text(args):
    """跑 git 并把输出**显式按 UTF-8 严格解码**（见 `GIT_TEXT_ENCODING` 处的说明）。

    解码失败要清晰报错（RuntimeError 里带上命令），不能退化成 `stdout=None` 让下游炸在
    `ast.parse(None)` 上；`git` 本身失败（无该 ref / 浅克隆）不走异常路径，由调用方按
    `returncode` 判定并 skip。
    """
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              encoding=GIT_TEXT_ENCODING)
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"git {' '.join(args)} 的输出不是合法 {GIT_TEXT_ENCODING}: {exc}") from exc


def archive_newtask_class():
    """从 `archive/exploration` 取**真正的**历史 `NewTask` 类（AST 抽类定义后用当前解释器 exec）。

    取不到（浅克隆 / 无该 ref / 命令失败 / 输出为空）时返回 None，由调用方 skip。
    """
    source = _git_text(["show", f"{ARCHIVE_REF}:multitaskrec/model.py"])
    if source.returncode != 0 or not source.stdout:
        return None
    tree = ast.parse(source.stdout)
    node = next((item for item in tree.body
                 if isinstance(item, ast.ClassDef) and item.name == "NewTask"), None)
    if node is None:
        return None
    namespace = {"torch": torch, "nn": nn, "F": F, "MLP": MLP}
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, f"<{ARCHIVE_REF}:multitaskrec/model.py:NewTask>", "exec"), namespace)
    return namespace["NewTask"]


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


def tiny_loader(dataset, batch_size):
    """带私有 `torch.Generator` 的 `DataLoader`（迭代本身不得动全局 RNG）。

    `DataLoader.__iter__` 每次都会抽一次 `_base_seed`
    （`torch.empty((), dtype=torch.int64).random_(generator=loader.generator)`，与 `num_workers` 无关）：
    `generator=None` 时抽的是**全局 RNG**。这不属于门控行为，却会让"诊断/探针不消耗全局 RNG"的断言
    变成在测 DataLoader 的实现细节（`test_matches_recomputation_and_has_no_side_effects` 最初的失败点）
    ⇒ 夹具给每个 loader 配私有 generator，把全局随机流留给被测代码。
    """
    return DataLoader(dataset, batch_size=batch_size, generator=torch.Generator())


def tiny_inputs(n_train=64, n_val=16, n_test=16, batch_size=16):
    train_ds, test_ds = TinyCensus(n_train, 1), TinyCensus(n_val + n_test, 2)
    val_idx, test_idx = P.make_split(len(test_ds), P.SPLIT_SEED)
    loaders = {"train": tiny_loader(train_ds, batch_size),
               "val": tiny_loader(Subset(test_ds, val_idx.tolist()), batch_size),
               "test": tiny_loader(Subset(test_ds, test_idx.tolist()), batch_size)}
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


class TestGitSourceDecoding(unittest.TestCase):
    """读入口守卫（本模块最初的导入失败点）：git 文本输出一律按 UTF-8 解码，失败必须清晰。

    Windows 中文环境下 `text=True` 默认走 GBK；归档 `model.py` 有数百行非 ASCII 注解，读取线程
    抛 UnicodeDecodeError ⇒ `stdout` 变 None ⇒ 模块级 `skipUnless(...)` 在导入期炸成 TypeError，
    整个测试模块连带所有用例一起消失。这里把"钉死 UTF-8 + 失败要显式"锁成可回归的用例。
    """

    def test_git_helper_pins_utf8_and_keeps_strict_errors(self):
        seen = {}

        def fake_run(argv, **kwargs):
            seen.update(kwargs, argv=argv)
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        with mock.patch.object(subprocess, "run", fake_run):
            _git_text(["show", f"{ARCHIVE_REF}:multitaskrec/model.py"])
        self.assertEqual(seen["argv"][0], "git")
        self.assertTrue(seen["text"])
        self.assertEqual(seen["encoding"], GIT_TEXT_ENCODING)
        self.assertEqual(seen["encoding"], "utf-8")                 # 而不是 locale 的 gbk/cp936
        self.assertIn(seen.get("errors"), (None, "strict"))         # 不得放宽成 replace/ignore

    def test_decode_failure_is_a_clear_error(self):
        def boom(*args, **kwargs):
            raise UnicodeDecodeError("utf-8", b"\xd3\xd0", 0, 1, "invalid start byte")

        with mock.patch.object(subprocess, "run", boom):
            with self.assertRaises(RuntimeError) as caught:
                _git_text(["show", f"{ARCHIVE_REF}:multitaskrec/model.py"])
        self.assertIn("utf-8", str(caught.exception))

    def test_failed_or_empty_git_output_skips_instead_of_typeerror(self):
        """取不到源码 ⇒ 返回 None 交给调用方 skip；绝不能把 None 喂进 `ast.parse`。"""
        cases = ((128, None), (128, "class NewTask: pass"),      # 命令失败优先于输出内容
                 (0, ""), (0, None),                             # 空输出
                 (0, "class Other: pass"))                       # 没有 NewTask 类
        for returncode, stdout in cases:
            with self.subTest(returncode=returncode, stdout=stdout):
                with mock.patch.object(subprocess, "run",
                                       return_value=subprocess.CompletedProcess(["git"], returncode,
                                                                                stdout=stdout)):
                    self.assertIsNone(archive_newtask_class())

    def test_real_archive_source_decodes_as_utf8(self):
        """真实回归：归档源码必须能整段解码并解析出 `NewTask`（按本地编码解出来是另一串字符）。"""
        result = _git_text(["show", f"{ARCHIVE_REF}:multitaskrec/model.py"])
        if result.returncode != 0:
            self.skipTest(f"缺少 {ARCHIVE_REF} ref")
        raw = subprocess.run(["git", "show", f"{ARCHIVE_REF}:multitaskrec/model.py"], cwd=REPO,
                             capture_output=True).stdout             # 二进制：绕开 text= 的编码
        self.assertIsNotNone(raw)
        self.assertEqual(result.stdout.splitlines(), raw.decode(GIT_TEXT_ENCODING).splitlines())
        self.assertIn("class NewTask", result.stdout)
        self.assertIsNotNone(archive_newtask_class())

    def test_no_unpinned_text_subprocess_calls_in_this_module(self):
        """静态守卫：本文件里任何"要文本"的 subprocess 调用都必须显式钉 `encoding=`（否则跟随 locale）。"""
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), __file__)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and func.value.id == "subprocess"):
                continue
            keywords = {keyword.arg for keyword in node.keywords}
            if ("text" in keywords or "universal_newlines" in keywords) and "encoding" not in keywords:
                offenders.append(node.lineno)
        self.assertEqual(offenders, [], f"这些调用跟随 locale 解码，必须显式钉 encoding=: {offenders}")


class TestPreregisteredCriteria(unittest.TestCase):
    """预注册判据（只允许在看到结果之前修改）。"""

    def test_threshold_values(self):
        self.assertAlmostEqual(AFF.AUC_TEST_MIN, 0.8521, places=12)
        self.assertAlmostEqual(AFF.EVAL_ATTN_SHARE_MIN, 0.999, places=12)
        self.assertAlmostEqual(AFF.ROUTING_L1_MIN, 1e-6, places=18)
        self.assertAlmostEqual(AFF.GRAD_NORM_MIN, 0.0, places=12)
        self.assertAlmostEqual(AFF.NOISE_SHARE_WINDOW_MAX, 0.32, places=12)
        self.assertAlmostEqual(AFF.PRESIGMOID_SATURATION_ABS, 8.0, places=12)
        self.assertEqual(AFF.TEMPERATURE, 150)                    # 历史常量（= NewTask.temperature）
        self.assertEqual(AFF.AFFINITY_INPUT_DIM, 4)               # [max, min, max−min, mean|·|]
        self.assertEqual(AFF.AFFINITY_HIDDEN, 16)                 # 历史隐层宽度
        self.assertAlmostEqual(AFF.GUMBEL_TAU, 0.5, places=12)    # 历史硬编码 /0.5
        self.assertEqual(AFF.ENV_NORMALIZE_DIM, 0)                # D2 复核：dim=0 即特征维（非缺陷）
        self.assertEqual(AFF.ENV_LAYOUT, "feature_first_(rep_dim, K)")
        self.assertEqual(AFF.FUSION_MODE, "affinity_gate")
        self.assertEqual(AFF.VARIANT, "affinity")
        self.assertEqual(AFF.BASELINE_VARIANT, "baseline")
        self.assertEqual(AFF.RUN_ID_SUFFIX, "-affinity")
        self.assertEqual(tuple(AFF.VARIANTS), (AFF.BASELINE_VARIANT, AFF.VARIANT))

    def test_analytic_bounds_follow_from_double_sigmoid(self):
        """D1 的解析边界：g_logit = sigmoid(z) ∈ (0,1) ⇒ 训练期路由概率 ∈ (1−e^{−1}, 1−e^{−e})。"""
        self.assertAlmostEqual(AFF.GUMBEL_SHARE_LO, 1.0 - math.exp(-1.0), places=12)
        self.assertAlmostEqual(AFF.GUMBEL_SHARE_HI, 1.0 - math.exp(-math.e), places=12)
        self.assertAlmostEqual(AFF.GUMBEL_SHARE_LO, 0.6321205588285577, places=12)
        self.assertAlmostEqual(AFF.GUMBEL_SHARE_HI, 0.9340119641546875, places=12)
        self.assertAlmostEqual(AFF.GUMBEL_SHARE_WINDOW_MAX, math.exp(-1.0) - math.exp(-math.e), places=12)
        self.assertAlmostEqual(AFF.GUMBEL_SHARE_WINDOW_MAX, 0.30189140532612976, places=12)
        self.assertAlmostEqual(AFF.EVAL_G_SOFT_MIN, 0.5, places=12)                 # sigmoid(0)
        self.assertAlmostEqual(AFF.EVAL_G_SOFT_MAX, float(torch.sigmoid(torch.tensor(2.0))), places=12)
        self.assertAlmostEqual(AFF.TRAIN_DROP_MIN, 1.0 - AFF.GUMBEL_SHARE_HI, places=12)
        self.assertAlmostEqual(AFF.TRAIN_DROP_MAX, 1.0 - AFF.GUMBEL_SHARE_LO, places=12)
        self.assertGreater(AFF.FLIP_RATE_MAX, AFF.FLIP_RATE_MIN)

    def test_exact_share_sweep_hits_analytic_bounds(self):
        """闭式解在全 z 轴上的上下确界就是解析边界（D1 的可证后果，与参数无关）。"""
        z = torch.linspace(-100.0, 100.0, 20001, dtype=torch.float64)
        share = AFF.exact_attention_share(z)
        self.assertAlmostEqual(float(share.min()), AFF.GUMBEL_SHARE_LO, places=9)
        self.assertAlmostEqual(float(share.max()), AFF.GUMBEL_SHARE_HI, places=9)
        self.assertLessEqual(float(share.max() - share.min()), AFF.GUMBEL_SHARE_WINDOW_MAX + 1e-12)

    def test_mechanism_stop_rule_boundaries(self):
        fine = AFF.degeneracy_verdict(eval_share=0.5, routing_l1=0.1, grad_norm=1e-3, noise_window=0.01)
        self.assertEqual(fine["status"], "NOT_DEGENERATE")
        self.assertEqual(fine["triggered_rules"], [])

        at_bound = AFF.degeneracy_verdict(eval_share=0.999, routing_l1=0.1, grad_norm=1e-3, noise_window=0.01)
        self.assertEqual(at_bound["status"], "CONFIRMED_DEGENERATE")                 # >= 含等号
        just_below = AFF.degeneracy_verdict(eval_share=0.9989, routing_l1=0.1, grad_norm=1e-3,
                                            noise_window=0.01)
        self.assertEqual(just_below["status"], "NOT_DEGENERATE")

        self.assertEqual(AFF.degeneracy_verdict(eval_share=0.5, routing_l1=1e-7, grad_norm=1e-3,
                                                noise_window=0.01)["status"], "CONFIRMED_DEGENERATE")
        self.assertEqual(AFF.degeneracy_verdict(eval_share=0.5, routing_l1=1e-6, grad_norm=1e-3,
                                                noise_window=0.01)["status"], "NOT_DEGENERATE")  # < 严格
        self.assertEqual(AFF.degeneracy_verdict(eval_share=0.5, routing_l1=0.1, grad_norm=0.0,
                                                noise_window=0.01)["status"], "CONFIRMED_DEGENERATE")
        self.assertEqual(AFF.degeneracy_verdict(eval_share=0.5, routing_l1=0.1, grad_norm=1e-12,
                                                noise_window=0.01)["status"], "NOT_DEGENERATE")

    def test_fidelity_rule_boundary(self):
        """保真判据：解析上界 0.3019；阈值 0.32 只留余量（一旦超出 ⇒ 复现偏离历史）。"""
        self.assertFalse(AFF.fidelity_verdict(noise_window=AFF.GUMBEL_SHARE_WINDOW_MAX)["triggered"])
        self.assertFalse(AFF.fidelity_verdict(noise_window=0.32)["triggered"])        # <= 不算越界
        self.assertTrue(AFF.fidelity_verdict(noise_window=0.3200001)["triggered"])
        verdict = AFF.fidelity_verdict(noise_window=0.9)
        self.assertAlmostEqual(verdict["analytic_max"], AFF.GUMBEL_SHARE_WINDOW_MAX, places=12)
        self.assertEqual(verdict["rule"], "noise_share_window <= 0.32")

    def test_degeneracy_reports_every_triggered_rule(self):
        verdict = AFF.degeneracy_verdict(eval_share=1.0, routing_l1=0.0, grad_norm=0.0, noise_window=0.5)
        self.assertEqual(verdict["triggered_rules"],
                         ["eval_attn_share >= 0.999", "routing_l1 < 1e-6", "grad_norm == 0"])
        self.assertEqual(sorted(verdict["checks"]),
                         ["eval_share_ge_0.999", "grad_norm_eq_0", "routing_l1_lt_1e-6"])
        self.assertAlmostEqual(verdict["observed"]["eval_share"], 1.0, places=12)
        self.assertAlmostEqual(verdict["observed"]["routing_l1"], 0.0, places=12)
        self.assertAlmostEqual(verdict["observed"]["grad_norm"], 0.0, places=12)

    def test_arm_verdict_stops_when_degenerate(self):
        """退化（机制 STOP）主导于 AUC：即便 test AUC 再高也不解释效应阈值（预注册）。"""
        degeneracy = AFF.degeneracy_verdict(eval_share=1.0, routing_l1=0.0, grad_norm=1e-3,
                                            noise_window=0.01)
        arm = AFF.arm_verdict(0.99, degeneracy=degeneracy,
                              fidelity=AFF.fidelity_verdict(noise_window=0.01))
        self.assertEqual(arm["status"], "STOP")
        self.assertEqual(arm["stop_reason"], "degenerate")
        self.assertFalse(arm["pass"])
        self.assertFalse(arm["effect"]["evaluated"])
        self.assertIsNone(arm["effect"]["pass"])
        self.assertAlmostEqual(arm["effect"]["observed"], 0.99, places=12)

    def test_fidelity_break_dominates_degeneracy_and_effect(self):
        """保真 STOP 优先级最高：复现偏离历史时，退化与效应都不再解释。"""
        degeneracy = AFF.degeneracy_verdict(eval_share=0.5, routing_l1=0.5, grad_norm=1e-3,
                                            noise_window=0.9)
        arm = AFF.arm_verdict(0.99, degeneracy=degeneracy, fidelity=AFF.fidelity_verdict(noise_window=0.9))
        self.assertEqual(arm["status"], "STOP")
        self.assertEqual(arm["stop_reason"], "fidelity_break")
        self.assertIsNone(arm["effect"]["pass"])
        self.assertFalse(arm["effect"]["evaluated"])

    def test_arm_verdict_effect_only_when_no_stop(self):
        ok = AFF.degeneracy_verdict(eval_share=0.5, routing_l1=0.1, grad_norm=1e-3, noise_window=0.01)
        fidelity = AFF.fidelity_verdict(noise_window=0.01)
        self.assertEqual(AFF.arm_verdict(0.8521, degeneracy=ok, fidelity=fidelity)["status"],
                         "EFFECT_CONFIRMED")                                     # 含等号
        self.assertTrue(AFF.arm_verdict(0.8521, degeneracy=ok, fidelity=fidelity)["pass"])
        self.assertIsNone(AFF.arm_verdict(0.8521, degeneracy=ok, fidelity=fidelity)["stop_reason"])
        self.assertEqual(AFF.arm_verdict(0.850069, degeneracy=ok, fidelity=fidelity)["status"],
                         "EFFECT_NOT_CONFIRMED")
        self.assertFalse(AFF.arm_verdict(0.850069, degeneracy=ok, fidelity=fidelity)["pass"])
        self.assertTrue(AFF.arm_verdict(0.850069, degeneracy=ok, fidelity=fidelity)["effect"]["evaluated"])

    def test_payloads_are_json_serialisable(self):
        degeneracy = AFF.degeneracy_verdict(eval_share=1.0, routing_l1=0.0, grad_norm=0.0, noise_window=0.5)
        arm = AFF.arm_verdict(0.86, degeneracy=degeneracy, fidelity=AFF.fidelity_verdict(noise_window=0.5))
        self.assertEqual(arm["effect"]["rule"], "auc_test_education >= 0.8521")
        json.dumps(arm)
        json.dumps(AFF.preregistered_criteria())
        json.dumps(AFF.provenance())
        prereg = AFF.preregistered_criteria()
        self.assertEqual(prereg["dominance"], "stop_fidelity > stop_mechanism > effect")
        self.assertEqual(prereg["rules"]["stop_fidelity"], "noise_share_window <= 0.32")
        self.assertEqual(prereg["rules"]["effect"], "auc_test_education >= 0.8521")

    def test_provenance_records_the_historical_defects(self):
        prov = AFF.provenance()
        self.assertEqual(prov["env_normalize_dim"], 0)              # D2 复核：特征维（非缺陷）
        self.assertEqual(prov["env_layout"], AFF.ENV_LAYOUT)        # 布局自证：[rep_dim, K]
        self.assertEqual(prov["gumbel_noise_rng"], "global")        # D3
        self.assertEqual(prov["eval_hard_gate"], "constant_1")      # D1 / D6
        self.assertFalse(prov["affinity_mlp_in_l2"])                # D5
        self.assertIn(ARCHIVE_REF, prov["historical_source"])
        self.assertIn("2026-09-30-stage2-affinity-gate-revalidation.md", prov["spec"])

    def test_provenance_records_the_d2_retraction(self):
        """审计留痕：D2 的"归一化维错误"结论已撤回，撤回理由必须随 run 落盘（不得只改代码不改记录）。"""
        correction = AFF.provenance()["d2_audit_correction"]
        self.assertIn("撤回", correction)
        self.assertIn("rep_dim", correction)
        self.assertIn("dim=0", correction)

    def test_doc_records_the_d2_retraction_and_ste_precision(self):
        """文档必须与代码同步记录两处复核（否则"文档 = 唯一事实来源"失效）：
        D2 的"归一化维错误"结论**撤回**；D1 的 STE float32 舍入留痕（`ste_exact_frac` / `routing_l1_max`）。"""
        self.assertTrue(DOC.is_file(), f"未找到实验文档: {DOC}")
        text = DOC.read_text(encoding="utf-8")
        for token in ("撤回", "ste_exact_frac", "routing_l1_max", "env_layout",
                      "g_logit_in_unit_interval", "routing_identity_frac"):
            self.assertIn(token, text, f"文档缺少复核记号: {token}")

    def test_doc_preregisters_same_numbers(self):
        """文档与代码必须同数：阈值只允许在**看到结果之前**同步修改（改一处必须改两处）。"""
        self.assertTrue(DOC.is_file(), f"未找到实验文档: {DOC}")
        text = DOC.read_text(encoding="utf-8")
        for token in ("0.8521", "0.999", "1e-6", "0.32", "0.3019", "0.6321", "0.9340",
                      "eval_attn_share >= 0.999", "routing_l1 < 1e-6", "grad_norm == 0",
                      "auc_test_education >= 0.8521", "noise_share_window"):
            self.assertIn(token, text, f"文档缺少预注册记号: {token}")


class TestFaithfulConstruction(unittest.TestCase):
    """忠实构造（I2）：历史顺序 + RNG 混入 + 默认初始化（这是历史对比被审计出来的缺陷，必须原样复现）。"""

    def test_structure_and_default_init_match_history(self):
        head = tiny_variant()
        mlp = head.affinity_mlp
        self.assertIsInstance(mlp, nn.Sequential)
        self.assertEqual(tuple(mlp[0].weight.shape), (AFF.AFFINITY_HIDDEN, AFF.AFFINITY_INPUT_DIM))
        self.assertEqual(tuple(mlp[2].weight.shape), (1, AFF.AFFINITY_HIDDEN))
        self.assertIsInstance(mlp[1], nn.ReLU)
        self.assertIsInstance(mlp[3], nn.Sigmoid)          # ← D1 的根源：末端 Sigmoid
        self.assertAlmostEqual(head.temperature, 150.0, places=12)
        self.assertEqual(head.fusion_mode, AFF.FUSION_MODE)

        # 默认初始化（历史**没有**像 CGR 那样做常量初始化）：与"按历史顺序消耗 RNG"的参照逐位一致
        torch.manual_seed(P.MODEL_SEED)
        nn.Embedding(1, TINY_REP_DIM)                                          # 1) env_embedding_network
        nn.Sequential(nn.Linear(TINY_INPUT_SIZE, TINY_REP_DIM // 2, bias=False), nn.ReLU(),
                      nn.Linear(TINY_REP_DIM // 2, TINY_REP_DIM, bias=False),
                      nn.LayerNorm(TINY_REP_DIM))                              # 2) projection_network
        reference = nn.Sequential(nn.Linear(AFF.AFFINITY_INPUT_DIM, AFF.AFFINITY_HIDDEN), nn.ReLU(),
                                  nn.Linear(AFF.AFFINITY_HIDDEN, 1), nn.Sigmoid())   # 3) affinity_mlp
        torch.manual_seed(P.MODEL_SEED)
        head = tiny_variant()
        for index in (0, 2):
            self.assertTrue(torch.equal(head.affinity_mlp[index].weight, reference[index].weight),
                            f"affinity_mlp[{index}].weight 与历史初始化不一致")
            self.assertTrue(torch.equal(head.affinity_mlp[index].bias, reference[index].bias))
        self.assertGreater(int(torch.unique(head.affinity_mlp[0].weight).numel()), 1)   # 不是常量初始化

    def test_rng_construction_order_confound_is_reproduced(self):
        """历史把 affinity_mlp 建在 projection 与 gate 之间 → 同 seed 下 gate/tower 初始化与基线**不同**。"""
        device = torch.device("cpu")
        P.seed_model(P.MODEL_SEED)
        baseline = tiny_newtask(device=device)
        rng_after_baseline = torch.get_rng_state().clone()

        P.seed_model(P.MODEL_SEED)
        variant = tiny_variant(device=device)

        self.assertFalse(torch.equal(torch.get_rng_state(), rng_after_baseline),
                         "历史实现确实多消耗了全局 RNG——构造顺序必须原样保留（D3 的构造面）")
        got, want = variant.state_dict(), baseline.state_dict()
        for name in ("env_embedding_network.weight", "projection_network.0.weight",
                     "projection_network.2.weight"):
            self.assertTrue(torch.equal(got[name], want[name]), f"多抽之前的模块应与基线逐位相同: {name}")
        for name in ("gate_network.0.weight", "tower_network.mlp.linear0.weight",
                     "tower_network.mlp.linear2.weight"):
            self.assertFalse(torch.equal(got[name], want[name]),
                             f"多抽之后的模块必须体现 RNG 混入（历史缺陷留痕）: {name}")

    def test_shared_module_structure_matches_master_newtask(self):
        """防漂移：除 affinity_mlp 外，子模块名与形状必须与 master NewTask 完全一致。"""
        mine = {name: tuple(t.shape) for name, t in tiny_variant().state_dict().items()}
        master = {name: tuple(t.shape) for name, t in tiny_newtask().state_dict().items()}
        extra = sorted(set(mine) - set(master))
        self.assertEqual(extra, ["affinity_mlp.0.bias", "affinity_mlp.0.weight",
                                 "affinity_mlp.2.bias", "affinity_mlp.2.weight"])
        self.assertEqual(sorted(set(master) - set(mine)), [])
        for name in sorted(master):
            self.assertEqual(mine[name], master[name], f"共享张量形状漂移: {name}")

    def test_get_l2_reg_excludes_affinity_mlp(self):
        """D5：`get_l2_reg()` 只正则 tower（继承 master）——把门控权重放大 1000 倍，正则值不变。"""
        head = tiny_variant()
        before = float(head.get_l2_reg())
        with torch.no_grad():
            head.affinity_mlp[0].weight.mul_(1000.0)
            head.affinity_mlp[2].bias.add_(1000.0)
        self.assertEqual(before, float(head.get_l2_reg()))
        self.assertGreaterEqual(before, 0.0)

    def test_num_source_tasks_slicing_is_faithful(self):
        """历史可选参数 `num_source_tasks`：只取前 K 个源任务（默认 None = 全用）。"""
        device = torch.device("cpu")
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(device)
        head = AFF.HistoricalAffinityGateNewTask(
            input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM, tower_dnn_hidden_units=list(P.TOWER_HIDDEN),
            reg_dnn=P.REG_DNN, device=device, num_source_tasks=1)
        head.eval()
        terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        self.assertEqual(terms["num_tasks"], 1)
        self.assertEqual(tuple(terms["cos_sim"].shape), (dnn_input.shape[0], 1))
        self.assertTrue(torch.equal(terms["env_embs"], torch.stack(env_embs, dim=1)[:, :1]))


class TestFaithfulForward(unittest.TestCase):
    """忠实公式（I1/I3/I4）：与历史逐项同式；D1/D2 的签名读数逐位可验。"""

    def test_gate_terms_match_hand_computation(self):
        device = torch.device("cpu")
        head = tiny_variant(device=device)
        head.eval()                                                        # 无噪分支（确定性）
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(device)

        with torch.no_grad():
            terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
            exist = torch.stack(env_embs, dim=1)
            h_out = head.projection_network(dnn_input)
            h_p_norm = F.normalize(h_out, dim=-1)
            e_norm = F.normalize(exist, dim=0)                             # D2：历史归一化维 = 0
            want_cos = torch.mm(h_p_norm, e_norm)
            max_cos = want_cos.max(dim=-1, keepdim=True)[0]
            min_cos = want_cos.min(dim=-1, keepdim=True)[0]
            want_input = torch.cat([max_cos, min_cos, max_cos - min_cos,
                                    want_cos.abs().mean(dim=-1, keepdim=True)], dim=-1)
            want_z = head.affinity_mlp[2](head.affinity_mlp[1](head.affinity_mlp[0](want_input))).squeeze(-1)
            want_logit = torch.sigmoid(want_z)
            want_soft = torch.sigmoid(want_logit / 0.5)                    # eval：无 Gumbel 噪声
            want_hard = (want_soft > 0.5).float()
            want_g = want_hard.detach() + want_soft - want_soft.detach()
            want_w_attn = F.softmax(torch.mm(h_out, exist) / head.temperature, dim=-1).unsqueeze(2)
            want_w_fw = torch.ones_like(want_w_attn) / len(spec_reps)
            want_w = want_g.view(-1, 1, 1) * want_w_attn + (1 - want_g.view(-1, 1, 1)) * want_w_fw

        for got, want, name in ((terms["e_norm"], e_norm, "e_norm"),
                                (terms["cos_sim"], want_cos, "cos_sim"),
                                (terms["affinity_input"], want_input, "affinity_input"),
                                (terms["presigmoid"], want_z, "presigmoid"),
                                (terms["g_logit"], want_logit, "g_logit"),
                                (terms["g_soft"], want_soft, "g_soft"),
                                (terms["g_hard"], want_hard, "g_hard"),
                                (terms["g"], want_g, "g"),
                                (terms["w_attn"], want_w_attn, "w_attn"),
                                (terms["w_fw"], want_w_fw, "w_fw"),
                                (terms["w"], want_w, "w")):
            self.assertTrue(torch.equal(got, want), f"{name} 与历史公式不一致")
        # 手工重算前三层 = 历史 Sequential 去掉末端 Sigmoid（D6 修正口径要用 z）
        self.assertTrue(torch.equal(torch.sigmoid(terms["presigmoid"]),
                                    head.affinity_mlp(terms["affinity_input"]).squeeze(-1)))

    def test_forward_matches_the_preserved_pipeline(self):
        """前向 = 历史 affinity_gate 分支的整条管线（路由融合 → env scale → gate 融合 → tower）。"""
        device = torch.device("cpu")
        head = tiny_variant(device=device)
        head.eval()
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
        self.assertTrue(torch.equal(got, want), "前向不是历史 affinity_gate 管线的组合")
        self.assertTrue(torch.equal(AFF.routing_pipeline(head, dnn_input, gen_rep, spec_reps, env_embs,
                                                         terms["w"]), want))

    def test_eval_hard_gate_is_always_attention_for_any_parameters(self):
        """D1 的核心：eval 期 `g_hard ≡ 1`、路由等价于纯 attention（与参数、输入都无关）。

        两个口径必须分开（见模块 D1 "精度留痕"）：
          * `g_hard ≡ 1` 是**精确**的（`g_soft ∈ (0.5, 0.8808]` 严格大于 0.5）；
          * `W` 与 `W_attn` **不是**逐位恒等——STE 的 `fl(1 + g_soft) − g_soft` 在 float32 下可能给出
            `g = 1 − 2⁻²⁴`，于是逐样本 `‖W − W_attn‖₁` 是 ~1e-8…1e-7 量级。这里断言的是**可证上界**
            （判据 2 的 1e-6）而不是"逐位相等"；`g == 1` 的样本上仍逐位相等（按样本条件断言）。
        """
        device = torch.device("cpu")
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(device)
        for seed in (1, 2, 3, 1685480945):
            P.seed_model(seed)
            head = tiny_variant(device=device)
            head.eval()
            with torch.no_grad():
                terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
            self.assertTrue(torch.all(terms["g_hard"] == 1.0), f"seed={seed}: eval 硬门控不是全 1")
            exact = (terms["g"] == 1.0)
            self.assertTrue(torch.equal(terms["w"][exact], terms["w_attn"][exact]),
                            f"seed={seed}: g == 1 的样本上 W 必须逐位等于 W_attn")
            l1 = (terms["w"] - terms["w_attn"]).abs().reshape(dnn_input.shape[0], -1).sum(dim=1)
            self.assertLess(float(l1.max()), AFF.ROUTING_L1_MIN, f"seed={seed}: 路由 L1 越过判据 2")
            self.assertGreaterEqual(float(terms["g"].min()), 1.0 - AFF.STE_ULP)    # g ∈ {1, 1−2⁻²⁴}
            self.assertLessEqual(float(terms["g"].max()), 1.0)
            self.assertGreater(float(terms["g_soft"].min()), AFF.EVAL_G_SOFT_MIN)      # > 0.5
            self.assertLessEqual(float(terms["g_soft"].max()), AFF.EVAL_G_SOFT_MAX)    # ≤ sigmoid(2)
        # 极端参数也翻不过来（门控饱和方向不影响结论）
        head = tiny_variant(device=device)
        head.eval()
        for value in (50.0, -10.0, 0.0):
            with torch.no_grad():
                head.affinity_mlp[2].weight.zero_()
                head.affinity_mlp[2].bias.fill_(value)
                terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
            self.assertTrue(torch.all(terms["g_hard"] == 1.0), f"bias={value}: eval 硬门控不是全 1")

    def test_eval_float32_underflow_is_the_only_route_to_fw(self):
        """D1 的唯一例外（留痕）：`sigmoid(z)` 在 float32 下溢为 0 时 `g_soft = 0.5` ⇒ 判 0 ⇒ FW。"""
        head = tiny_variant()
        head.eval()
        with torch.no_grad():
            head.affinity_mlp[2].weight.zero_()
            head.affinity_mlp[2].bias.fill_(-200.0)
            terms = head.gate_terms(*tiny_reps())
        self.assertEqual(float(terms["g_logit"][0]), 0.0)                  # sigmoid(−200) 下溢
        self.assertEqual(float(terms["g_soft"][0]), 0.5)                   # sigmoid(0/0.5)
        self.assertTrue(torch.all(terms["g_hard"] == 0.0))
        self.assertTrue(torch.equal(terms["w"], terms["w_fw"]))

    def test_eval_forward_consumes_no_rng_and_training_forward_does(self):
        """D3：训练期前向每步消耗全局 RNG（`torch.rand_like`），eval 期不消耗。"""
        head = tiny_variant()
        reps = tiny_reps()
        head.eval()
        before = torch.get_rng_state().clone()
        with torch.no_grad():
            head(*reps)
        self.assertTrue(torch.equal(before, torch.get_rng_state()), "eval 前向不应消耗全局 RNG")
        head.train()
        before = torch.get_rng_state().clone()
        head(*reps)
        self.assertFalse(torch.equal(before, torch.get_rng_state()), "训练前向必须消耗全局 RNG（历史行为）")

    def test_training_hard_gate_flips_with_noise_but_g_logit_is_small(self):
        """训练期硬判据由噪声决定：同一样本换一份噪声就可能翻面，而 g_logit 始终 ∈ (0,1)。"""
        P.seed_model(P.MODEL_SEED)
        head = tiny_variant()
        head.train()
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps()
        decisions, zs = [], []
        with torch.no_grad():
            for _ in range(32):
                terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
                decisions.append(terms["g_hard"])
                zs.append(terms["presigmoid"])
                self.assertGreater(float(terms["g_logit"].min()), 0.0)
                self.assertLess(float(terms["g_logit"].max()), 1.0)
        hard = torch.stack(decisions)
        self.assertGreater(float((hard[1:] != hard[:-1]).double().mean()), 0.0, "噪声没有改变任何判据？")
        # 路由概率的上下界用**闭式解**核对（32 次抽样的经验值有 ~0.08 的抽样误差，不能直接卡窗口）
        exact = AFF.exact_attention_share(zs[-1])
        self.assertGreaterEqual(float(exact.min()), AFF.GUMBEL_SHARE_LO - 1e-9)
        self.assertLessEqual(float(exact.max()), AFF.GUMBEL_SHARE_HI + 1e-9)
        self.assertLessEqual(float(exact.max() - exact.min()), AFF.GUMBEL_SHARE_WINDOW_MAX + 1e-9)

    def test_gumbel_noise_matches_historical_formula(self):
        head = tiny_variant()
        like = torch.zeros(64)
        torch.manual_seed(11)
        want = -(-torch.rand_like(like).clamp(1e-10).log()).clamp(1e-10).log()
        torch.manual_seed(11)
        self.assertTrue(torch.equal(AFF.gumbel_noise_like(like), want))    # generator=None = 历史路径
        generator = torch.Generator().manual_seed(5)
        reference = torch.Generator().manual_seed(5)
        got = AFF.gumbel_noise_like(like, generator=generator)
        want = -(-torch.rand(like.shape, generator=reference).clamp(1e-10).log()).clamp(1e-10).log()
        self.assertTrue(torch.equal(got, want))                            # 私有 generator 走同一原式
        # 私有 generator 是**有状态的流**：上面的 `got` 已经把 `generator` 消耗到第二项，所以"同 seed ⇒ 同噪声"
        # 只能用另一只新 generator 复现**首抽**（即 `got`）来验；直接拿被消耗过的 `generator` 去比新 generator
        # 是在比"流的第二项 vs 首项"，恒不成立。流式语义本身也要锁：连续两次抽取必须给出不同噪声
        # （`noise_signal_probe` 的 `draws` 次重复抽样依赖它）。
        self.assertTrue(torch.equal(got, AFF.gumbel_noise_like(like, generator=torch.Generator().manual_seed(5))))
        self.assertFalse(torch.equal(got, AFF.gumbel_noise_like(like, generator=generator)),
                         "同一 generator 连续两次抽取必须给出不同噪声（否则探针的流式抽样退化）")

    def test_explicit_noise_is_used_and_consumes_no_rng(self):
        head = tiny_variant()
        head.train()
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps()
        before = torch.get_rng_state().clone()
        noise = torch.zeros(dnn_input.shape[0])
        terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs, noise=noise, use_noise=True)
        self.assertTrue(torch.equal(before, torch.get_rng_state()), "给定噪声时不得抽取 RNG")
        self.assertTrue(torch.equal(terms["gumbel_noise"], noise))
        want_soft = torch.sigmoid(terms["g_logit"] / AFF.GUMBEL_TAU)
        self.assertTrue(torch.equal(terms["g_soft"], want_soft))

    def test_env_layout_is_feature_first_and_cos_sim_is_a_true_cosine(self):
        """D2 复核（**撤回**早期结论）：`env_embs` 元素一维 `[rep_dim]` ⇒ `stack(dim=1)` = `[rep_dim, K]`，
        `normalize(E, dim=0)` 归一化的**就是特征维**（逐任务列）⇒ `cos_sim` 是真余弦。

        证据链（三处独立）：`get_infos` 用 0 维索引取 env 嵌入（元素 `[rep_dim]`）、历史 `num_source_tasks`
        切片用 `[:, :K]`、历史 `torch.mm(H_out, exist_env_embs)` 需要 `[B,d]@[d,K]`。

        边界口径（死 ReLU）：H(x) 为**零向量**的样本上余弦无定义，历史口径（`F.normalize`）给 0——
        这条约定单独断言，真余弦的逐位比对只在 H(x) ≠ 0 的样本上进行（见下方注释）。
        """
        device = torch.device("cpu")
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(device)          # 与真实路径同款抽取
        self.assertEqual(env_embs[0].dim(), 1, "env_embs 元素必须是 0 维索引取出的一维 [rep_dim]")
        self.assertEqual(tuple(env_embs[0].shape), (TINY_REP_DIM,))
        self.assertEqual(tuple(torch.stack(env_embs, dim=1).shape), (TINY_REP_DIM, len(env_embs)))

        head = tiny_variant(device=device)
        head.eval()
        with torch.no_grad():
            terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
            exist = torch.stack(env_embs, dim=1)                       # [rep_dim, K]
            e_norm = F.normalize(exist, dim=AFF.ENV_NORMALIZE_DIM)     # dim=0 = 特征维 ⇒ 逐任务列为单位向量
            # 真余弦：**独立**算法（einsum + 显式范数），不复用 F.normalize + mm 的实现路径
            # C[i,k] = <H_i, E_:,k> / (‖H_i‖·‖E_:,k‖)；历史 cos_sim 走的是 dim=0 归一化，二者必须一致
            true_cos = (torch.einsum("bd,dk->bk", terms["h_out"], exist)
                        / (terms["h_out"].norm(dim=-1, keepdim=True) * exist.norm(dim=0, keepdim=True)))
        self.assertTrue(torch.equal(terms["e_norm"], e_norm))
        self.assertLessEqual(float((e_norm.norm(dim=0) - 1.0).abs().max()), 1e-6)     # 逐任务列是单位向量
        # 真余弦只在 H(x) ≠ 0 的样本上有定义。`projection_network` 末端是 LayerNorm，若某样本把 ReLU 全灭
        # 则 H(x) 恰为**零向量** ⇒ `F.normalize` 按约定给 0（`0 / clamp_min(‖·‖, 1e-12)`），而独立算法的
        # `0 / (0·‖E‖)` 是 NaN ⇒ "逐行 allclose"在零投影样本上不再是良定义的比较。
        # 因此在有定义的样本上比对真余弦，零投影样本单独断言"约定值 0、不是 NaN"。
        defined = terms["h_out"].norm(dim=-1) > 0.0
        self.assertTrue(bool(defined.any()), "夹具退化：全部样本的投影都是零向量，无法检验真余弦")
        self.assertTrue(torch.equal(terms["cos_sim"][~defined], torch.zeros_like(terms["cos_sim"][~defined])),
                        "零投影样本上历史口径必须给出 0（F.normalize 的零向量约定），不得是 NaN")
        self.assertTrue(torch.allclose(terms["cos_sim"][defined], true_cos[defined], atol=1e-6),
                        "cos_sim 必须等于真余弦（dim=0 是特征维）")
        self.assertFalse(bool(torch.isnan(terms["cos_sim"]).any()), "cos_sim 不得出现 NaN")
        self.assertLessEqual(float(terms["cos_sim"].abs().max()), 1.0 + 1e-6)

    def test_layout_guard_rejects_wrong_env_layout(self):
        """守卫（fail-fast、不改数值）：[K, d] 或 [B, d] 的 env_embs 会让历史公式静默算错，这里必须显式报错。"""
        head = tiny_variant()
        head.eval()
        dnn_input, gen_rep, spec_reps, _ = tiny_reps()
        k_first = [torch.ones(1, TINY_REP_DIM) for _ in range(2)]                       # ⇒ stack(dim=1) = [1,K,d]
        with self.assertRaises(ValueError):
            head.gate_terms(dnn_input, gen_rep, spec_reps, k_first)
        batch_first = [torch.ones(dnn_input.shape[0], TINY_REP_DIM) for _ in range(2)]  # ⇒ [B,K,d]
        with self.assertRaises(ValueError):
            head.gate_terms(dnn_input, gen_rep, spec_reps, batch_first)

    def test_gate_features_are_redundant_for_k2(self):
        """D2（**保留**的设计层问题）：K=2 时 4 维门控特征是 `(C_1, C_2)` 的函数，且第 4 维
        `mean|C| = (|max|+|min|)/2` 是前三维的函数 ⇒ 门控实际只看到 2 个数（与 `W_attn` 同源）。"""
        head = tiny_variant()
        head.eval()
        with torch.no_grad():
            terms = head.gate_terms(*tiny_reps(seed=21))
        affinity_input = terms["affinity_input"]
        self.assertEqual(tuple(affinity_input.shape), (8, AFF.AFFINITY_INPUT_DIM))
        self.assertTrue(torch.equal(affinity_input[:, 2], affinity_input[:, 0] - affinity_input[:, 1]))
        redundant = (affinity_input[:, 3]
                     - (affinity_input[:, 0].abs() + affinity_input[:, 1].abs()) / 2.0).abs()
        self.assertLess(float(redundant.max()), 1e-6,
                        "第 4 维 = mean|C| 必须由 (max, min) 决定（恒等式，float32 舍入内）")

    def test_unnormalised_w_is_a_convex_combination(self):
        head = tiny_variant()
        terms = head.gate_terms(*tiny_reps())
        weights = terms["w"].reshape(terms["w"].shape[0], -1)
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(weights.shape[0]), atol=1e-6))
        self.assertTrue(torch.all(terms["g"] >= 0.0) and torch.all(terms["g"] <= 1.0))


@unittest.skipUnless(archive_newtask_class() is not None, f"缺少 {ARCHIVE_REF} ref，无法做归档逐位比对")
class TestArchiveFidelity(unittest.TestCase):
    """I1 的最强口径：与 `archive/exploration` 里**真正的** `NewTask` 逐位比对（参数/输出/梯度）。"""

    @staticmethod
    def _pair(seed=P.MODEL_SEED):
        archive_cls = archive_newtask_class()
        if archive_cls is None:                    # 类级 skipUnless 已经过了，这里再拿不到必须显式报错
            raise RuntimeError(f"导入时可读、现在读不到 {ARCHIVE_REF}:multitaskrec/model.py")
        kwargs = dict(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                      tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)
        torch.manual_seed(seed)
        archive = archive_cls(fusion_mode=AFF.FUSION_MODE, **kwargs)
        torch.manual_seed(seed)
        mine = tiny_variant()
        return archive, mine

    def test_parameters_are_bit_identical_after_same_seed_construction(self):
        archive, mine = self._pair()
        self.assertEqual(sorted(archive.state_dict()), sorted(mine.state_dict()))
        for name, value in archive.state_dict().items():
            self.assertTrue(torch.equal(mine.state_dict()[name], value), f"参数初始化漂移: {name}")

    def test_eval_and_train_forward_are_bit_identical(self):
        archive, mine = self._pair()
        reps = tiny_reps()
        archive.eval()
        mine.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(archive(*reps), mine(*reps)), "eval 前向与归档不一致")
        torch.manual_seed(7)                                   # 同 seed ⇒ 同 Gumbel 噪声 ⇒ 逐位可比
        archive.train()
        out_archive = archive(*reps)
        torch.manual_seed(7)
        mine.train()
        out_mine = mine(*reps)
        self.assertTrue(torch.equal(out_archive, out_mine), "train 前向与归档不一致")

    def test_gradients_are_bit_identical(self):
        archive, mine = self._pair()
        reps = tiny_reps()
        y = torch.randint(0, 2, (reps[0].shape[0],)).float()
        torch.manual_seed(13)
        archive.train()
        loss_archive = nn.BCELoss()(archive(*reps), y)
        loss_archive.backward()
        torch.manual_seed(13)
        mine.train()
        loss_mine = nn.BCELoss()(mine(*reps), y)
        loss_mine.backward()
        self.assertTrue(torch.equal(loss_archive, loss_mine), "损失与归档不一致")
        for (name_a, param_a), (name_m, param_m) in zip(archive.named_parameters(), mine.named_parameters()):
            self.assertEqual(name_a, name_m)
            self.assertTrue(torch.equal(param_a.grad, param_m.grad), f"梯度与归档不一致: {name_a}")

    def test_archive_eval_output_equals_pure_attention_pipeline(self):
        """D1 的独立复现：**归档类本身**在 eval 下的输出 == 把 W 换成纯 attention 权重的同一管线。

        这条不依赖我们的实现：它直接用归档 `NewTask` 的前向与 `softmax(H·E/T)` 的管线对比，
        说明"历史 affinity_gate 在评测时就是纯 attention"是历史实现自身的行为。
        """
        archive, _ = self._pair()
        archive.eval()
        reps = tiny_reps()
        with torch.no_grad():
            h_out = archive.projection_network(reps[0])
            w_attn = F.softmax(torch.mm(h_out, torch.stack(reps[3], dim=1)) / archive.temperature,
                               dim=-1).unsqueeze(2)
            want = AFF.routing_pipeline(archive, *reps, w_attn)
            got = archive(*reps)
        self.assertTrue(torch.equal(got, want), "归档 eval 输出不等于纯 attention 管线")


class TestAffinityStats(unittest.TestCase):
    """指标口径（I3/I4/I7）：硬切换占比、路由等价 L1、逐源任务列、解析读数、D6 修正读数。"""

    @staticmethod
    def terms(*, z, w, w_attn, w_fw, cos_sim, g_hard=None, g_soft=None, g=None, g_logit=None):
        z = torch.tensor(z, dtype=torch.float32).reshape(-1)
        shape = (z.shape[0],)
        soft = torch.sigmoid(z / AFF.GUMBEL_TAU) if g_soft is None else torch.tensor(g_soft, dtype=torch.float32)
        hard = (soft > 0.5).float() if g_hard is None else torch.tensor(g_hard, dtype=torch.float32)
        return {"presigmoid": z,
                "g_logit": torch.sigmoid(z) if g_logit is None else torch.tensor(g_logit, dtype=torch.float32),
                "g_soft": soft, "g_hard": hard,
                "g": hard.clone() if g is None else torch.tensor(g, dtype=torch.float32),
                "w": torch.tensor(w, dtype=torch.float32).reshape(*shape, 2, 1),
                "w_attn": torch.tensor(w_attn, dtype=torch.float32).reshape(*shape, 2, 1),
                "w_fw": torch.tensor(w_fw, dtype=torch.float32).reshape(*shape, 2, 1),
                "cos_sim": torch.tensor(cos_sim, dtype=torch.float32).reshape(*shape, 2),
                "num_tasks": 2}
        # 注：这里**故意**不放 `h_out` / `env_embs`——`AffinityStats.update` 只读门控输出与路由权重；
        # 摆一个形状随意的 `env_embs` 正是早期"D2 = 归一化维错误"误判的来源（见 experiments 文档 D2(a)）。

    def test_hand_computed_eval_batch_metrics(self):
        """eval 分支（硬门控 ≡ 1）的手算读数：路由等价 L1 == 0、逐位相同、逐源任务列均值。"""
        stats = AFF.AffinityStats()
        stats.update(self.terms(z=[1.0, 1.0],                                  # 两个样本都判 attention
                                w=[[[0.7], [0.3]], [[0.4], [0.6]]],
                                w_attn=[[[0.7], [0.3]], [[0.4], [0.6]]],
                                w_fw=[[[0.5], [0.5]], [[0.5], [0.5]]],
                                cos_sim=[[0.6, -0.2], [0.1, 0.3]]))
        got = stats.result()
        self.assertEqual((got["n_samples"], got["n_batches"], got["num_tasks"]), (2, 1, 2))
        self.assertAlmostEqual(got["g_hard_mean"], 1.0, places=6)
        self.assertAlmostEqual(got["routing_l1_mean"], 0.0, places=9)          # W ≡ W_attn
        self.assertAlmostEqual(got["routing_identity_frac"], 1.0, places=9)
        self.assertAlmostEqual(got["routing_fw_l1_mean"], 0.3, places=6)       # (0.4 + 0.2) / 2
        self.assertAlmostEqual(got["cos_max_mean"], 0.45, places=6)
        self.assertAlmostEqual(got["cos_min_mean"], -0.05, places=6)
        self.assertAlmostEqual(got["cos_range_mean"], 0.5, places=6)
        self.assertAlmostEqual(got["cos_absmean_mean"], 0.3, places=6)
        self.assertAlmostEqual(got["cos_col_mean"][0], 0.35, places=6)
        self.assertAlmostEqual(got["cos_col_mean"][1], 0.05, places=6)
        self.assertAlmostEqual(got["routing_col_mean"][0], 0.55, places=6)
        self.assertAlmostEqual(got["routing_col_mean"][1], 0.45, places=6)
        self.assertAlmostEqual(got["presigmoid_mean"], 1.0, places=6)
        self.assertAlmostEqual(got["presigmoid_abs_mean"], 1.0, places=6)
        self.assertAlmostEqual(got["presigmoid_saturation_frac"], 0.0, places=9)
        # D6 修正口径：corrected_gate_mean = mean σ(z/τ)、corrected_routing_share = P(z > 0)
        self.assertAlmostEqual(got["corrected_gate_mean"], float(torch.sigmoid(torch.tensor(2.0))), places=6)
        self.assertAlmostEqual(got["corrected_routing_share_attn"], 1.0, places=6)
        # 解析读数与闭式解同源
        exact = AFF.exact_attention_share(torch.tensor([1.0, 1.0], dtype=torch.float32))
        self.assertAlmostEqual(got["noise_share_exact_mean"], float(exact.mean()), places=6)
        self.assertAlmostEqual(got["noise_share_exact_min"], float(exact.min()), places=6)
        self.assertAlmostEqual(got["attention_drop_frac_exact"], 1.0 - float(exact.mean()), places=6)
        self.assertAlmostEqual(got["flip_rate_exact_mean"], float(AFF.exact_flip_rate(
            torch.tensor([1.0, 1.0], dtype=torch.float32)).mean()), places=6)
        json.dumps(got)

    def test_analytic_readings_respect_the_d1_window(self):
        """任何 z 下 `noise_share_exact_*` 都被夹在解析窗口内（D1 的结构性读数）。"""
        stats = AFF.AffinityStats()
        stats.update(self.terms(z=[-30.0, -5.0, 0.0, 5.0, 30.0],
                                w=[[[0.5], [0.5]]] * 5, w_attn=[[[0.5], [0.5]]] * 5,
                                w_fw=[[[0.5], [0.5]]] * 5, cos_sim=[[0.0, 0.0]] * 5))
        got = stats.result()
        self.assertGreaterEqual(got["noise_share_exact_min"], AFF.GUMBEL_SHARE_LO - 1e-9)
        self.assertLessEqual(got["noise_share_exact_max"], AFF.GUMBEL_SHARE_HI + 1e-9)
        self.assertLessEqual(got["noise_share_exact_window"], AFF.GUMBEL_SHARE_WINDOW_MAX + 1e-9)
        self.assertFalse(AFF.fidelity_verdict(noise_window=got["noise_share_exact_window"])["triggered"])

    def test_streaming_preserves_sample_level_aggregates(self):
        first = self.terms(z=[1.0, -1.0], w=[[[0.7], [0.3]], [[0.4], [0.6]]],
                           w_attn=[[[0.7], [0.3]], [[0.4], [0.6]]], w_fw=[[[0.5], [0.5]]] * 2,
                           cos_sim=[[0.6, -0.2], [0.1, 0.3]])
        second = self.terms(z=[2.0], w=[[[0.2], [0.8]]], w_attn=[[[0.5], [0.5]]], w_fw=[[[0.5], [0.5]]],
                            cos_sim=[[0.4, 0.4]])
        streamed = AFF.AffinityStats()
        streamed.update(first)
        streamed.update(second)
        single = AFF.AffinityStats()
        single.update({key: torch.cat([first[key], second[key]]) if torch.is_tensor(first[key]) else first[key]
                       for key in first})
        for key in ("g_hard_mean", "g_mean", "routing_l1_mean", "routing_l1_max", "routing_fw_l1_mean",
                    "routing_identity_frac", "ste_exact_frac", "cos_max_mean", "cos_min_mean",
                    "cos_range_mean", "cos_absmean_mean", "cos_entry_min", "cos_entry_max",
                    "g_logit_min", "g_logit_max", "presigmoid_mean", "presigmoid_abs_mean",
                    "noise_share_exact_mean", "noise_share_exact_window",
                    "attention_drop_frac_exact", "flip_rate_exact_mean", "corrected_routing_share_attn",
                    "corrected_gate_mean", "n_samples"):
            self.assertAlmostEqual(streamed.result()[key], single.result()[key], places=9,
                                   msg=f"流式聚合不一致: {key}")
        for key in ("cos_col_mean", "routing_col_mean"):            # 列表字段逐元素比较
            for streamed_value, single_value in zip(streamed.result()[key], single.result()[key]):
                self.assertAlmostEqual(streamed_value, single_value, places=9, msg=f"流式聚合不一致: {key}")
        self.assertEqual(streamed.result()["n_batches"], 2)

    def test_hard_decision_flip_is_visible_in_routing_l1(self):
        """训练期硬判据为 0 的样本 ⇒ `W = W_fw` ⇒ `routing_l1` 立刻反映"整支丢弃 attention"。"""
        stats = AFF.AffinityStats()
        stats.update(self.terms(z=[0.0], w=[[[0.5], [0.5]]], w_attn=[[[0.9], [0.1]]], w_fw=[[[0.5], [0.5]]],
                                cos_sim=[[0.0, 0.0]], g_hard=[0.0], g_soft=[0.4], g=[0.0]))
        got = stats.result()
        self.assertAlmostEqual(got["g_hard_mean"], 0.0, places=9)
        self.assertAlmostEqual(got["routing_l1_mean"], 0.8, places=6)      # |0.5−0.9| + |0.5−0.1|
        self.assertAlmostEqual(got["routing_identity_frac"], 0.0, places=9)
        self.assertEqual(got["routing_l1_max"] > 0.0, True)

    def test_d2_corrected_readings_and_cos_entry_bounds(self):
        """D2 复核后的落盘读数：布局/归一化维自证 + 真余弦的确定性边界 `|C| ≤ 1` + 逐源任务列。"""
        P.seed_model(P.MODEL_SEED)
        head = tiny_variant()
        head.eval()
        stats = AFF.AffinityStats()
        with torch.no_grad():
            stats.update(head.gate_terms(*tiny_reps(seed=9)))
        got = stats.result()
        self.assertEqual(got["env_normalize_dim"], AFF.ENV_NORMALIZE_DIM)
        self.assertEqual(got["env_layout"], AFF.ENV_LAYOUT)
        self.assertLessEqual(got["cos_entry_max"], 1.0 + 1e-6)                # 真余弦 ⇒ |C| ≤ 1
        self.assertGreaterEqual(got["cos_entry_min"], -1.0 - 1e-6)
        self.assertLessEqual(got["cos_entry_min"], got["cos_min_mean"])       # 均值必落在极值之间
        self.assertLessEqual(got["cos_max_mean"], got["cos_entry_max"])
        self.assertEqual(len(got["cos_col_mean"]), 2)                         # 逐源任务列读数
        self.assertEqual(len(got["routing_col_mean"]), 2)
        self.assertAlmostEqual(sum(got["routing_col_mean"]), 1.0, places=6)   # 路由权重逐样本和 = 1
        json.dumps(got)

    def test_ste_rounding_reading_is_recorded(self):
        """D1 精度留痕的**合成最坏情形**：`g = 1 − 2⁻²⁴` 的样本让 `ste_exact_frac < 1`、
        `routing_l1 ∈ (0, 1e-6)`（STE 残差的形状，不是实测值）；解析闭式解与 `g` 无关。"""
        residual = 1.0 - AFF.STE_ULP
        stats = AFF.AffinityStats()
        stats.update(self.terms(z=[1.0], w=[[[1.0 - AFF.STE_ULP], [AFF.STE_ULP]]],
                                w_attn=[[[1.0], [0.0]]], w_fw=[[[0.0], [1.0]]],
                                cos_sim=[[0.0, 0.0]], g_hard=[1.0], g_soft=[0.6], g=[residual]))
        got = stats.result()
        self.assertAlmostEqual(got["ste_exact_frac"], 0.0, places=9)          # g ≠ g_hard（舍入留痕）
        self.assertAlmostEqual(got["routing_identity_frac"], 0.0, places=9)
        self.assertGreater(got["routing_l1_mean"], 0.0)
        self.assertLess(got["routing_l1_mean"], AFF.ROUTING_L1_MIN)           # 判据 2 仍命中
        self.assertAlmostEqual(got["routing_l1_max"], got["routing_l1_mean"], places=9)
        self.assertAlmostEqual(got["g_hard_mean"], 1.0, places=9)             # 硬判据本身是精确的
        self.assertAlmostEqual(got["noise_share_exact_mean"],
                               float(AFF.exact_attention_share(torch.tensor([1.0])).mean()), places=6)
        json.dumps(got)

    def test_empty_accumulator_is_zero_safe(self):
        empty = AFF.AffinityStats().result()
        self.assertEqual((empty["n_samples"], empty["n_batches"]), (0, 0))
        for key, value in empty.items():
            if key not in ("n_samples", "n_batches", "presigmoid_saturation_abs", "gumbel_tau",
                           "num_tasks", "cos_col_mean", "routing_col_mean", "env_layout"):
                self.assertEqual(value, 0.0, f"空累加器字段应为 0.0: {key}")
        self.assertAlmostEqual(empty["presigmoid_saturation_abs"], 8.0, places=12)     # 阈值恒回显
        self.assertAlmostEqual(empty["gumbel_tau"], 0.5, places=12)
        self.assertEqual(empty["env_layout"], AFF.ENV_LAYOUT)                          # 字符串口径自证
        json.dumps(empty)

    def test_inconsistent_num_tasks_raises(self):
        stats = AFF.AffinityStats()
        stats.update(self.terms(z=[1.0], w=[[[0.5], [0.5]]], w_attn=[[[0.5], [0.5]]],
                                w_fw=[[[0.5], [0.5]]], cos_sim=[[0.0, 0.0]]))
        bad = self.terms(z=[1.0], w=[[[0.5], [0.5]]], w_attn=[[[0.5], [0.5]]],
                         w_fw=[[[0.5], [0.5]]], cos_sim=[[0.0, 0.0]])
        bad["num_tasks"] = 3
        with self.assertRaises(ValueError):
            stats.update(bad)


class TestNoiseSignalProbe(unittest.TestCase):
    """噪声/信号行为（I3/D4）：闭式解 vs 蒙特卡洛、翻面率、噪声轴 vs 样本轴离散度。"""

    def test_mc_agrees_with_closed_form_and_respects_window(self):
        device = torch.device("cpu")
        P.seed_model(P.MODEL_SEED)
        head = tiny_variant(device=device)
        reps = tiny_reps(device)
        got = AFF.noise_signal_probe(head, *reps, draws=256)
        self.assertEqual(got["n_samples"], reps[0].shape[0])
        self.assertAlmostEqual(got["noise_share_mc_mean"], got["noise_share_exact_mean"], delta=0.05)
        self.assertLess(got["mc_exact_max_abs_error"], 0.2)
        self.assertGreaterEqual(got["noise_share_exact_min"], AFF.GUMBEL_SHARE_LO - 1e-9)
        self.assertLessEqual(got["noise_share_exact_max"], AFF.GUMBEL_SHARE_HI + 1e-9)
        self.assertLessEqual(got["noise_share_exact_window"], AFF.GUMBEL_SHARE_WINDOW_MAX + 1e-9)
        # 翻面率：解析均值在理论区间内，蒙特卡洛与解析值在抽样误差内一致
        self.assertGreaterEqual(got["flip_rate_exact_mean"], AFF.FLIP_RATE_MIN - 1e-9)
        self.assertLessEqual(got["flip_rate_exact_mean"], AFF.FLIP_RATE_MAX + 1e-9)
        self.assertAlmostEqual(got["flip_rate_mc"], got["flip_rate_exact_mean"], delta=0.08)
        self.assertGreater(got["g_soft_within_sample_std"], 0.0)          # 噪声轴有离散度
        self.assertIsNotNone(got["noise_to_signal_ratio"])
        json.dumps(got)

    def test_probe_is_deterministic_and_consumes_no_global_rng(self):
        head = tiny_variant()
        reps = tiny_reps()
        before = torch.get_rng_state().clone()
        first = AFF.noise_signal_probe(head, *reps, draws=32)
        self.assertTrue(torch.equal(before, torch.get_rng_state()), "探针不得消耗全局 RNG（D3 留痕）")
        second = AFF.noise_signal_probe(head, *reps, draws=32)
        self.assertEqual(first, second)                                   # 同 seed ⇒ 逐位可复现

    def test_probe_restores_mode_and_parameters(self):
        head = tiny_variant()
        head.train()
        before = {name: value.clone() for name, value in head.state_dict().items()}
        AFF.noise_signal_probe(head, *tiny_reps(), draws=8)
        self.assertTrue(head.training, "探针必须还原训练模式")
        for name, value in head.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), f"探针改动了参数: {name}")
        head.eval()
        AFF.noise_signal_probe(head, *tiny_reps(), draws=8)
        self.assertFalse(head.training)

    def test_baseline_newtask_is_rejected(self):
        with self.assertRaises(ValueError):
            AFF.noise_signal_probe(tiny_newtask(), *tiny_reps(), draws=4)


class TestGradientProbe(unittest.TestCase):
    """梯度读数（I6）：affinity_mlp 梯度范数、三条路径的衰减比、修正路径的局部灵敏度上限。"""

    def test_affinity_grad_norm_matches_manual_computation(self):
        head = tiny_variant()
        head.train()
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps()
        pred = head(dnn_input, gen_rep, spec_reps, env_embs)
        nn.BCELoss()(pred, torch.randint(0, 2, (dnn_input.shape[0],)).float()).backward()
        want = sum(float(p.grad.detach().double().norm(2)) ** 2
                   for p in head.affinity_mlp.parameters() if p.grad is not None) ** 0.5
        self.assertGreater(want, 0.0)
        self.assertAlmostEqual(AFF.affinity_grad_norm(head), want, places=9)

    def test_probe_reports_three_paths_without_touching_grads_or_parameters(self):
        P.seed_model(P.MODEL_SEED)
        head = tiny_variant()
        head.train()
        reps = tiny_reps()
        before = {name: value.clone() for name, value in head.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        got = AFF.gradient_probe(head, *reps)
        for key in ("grad_norm_historical", "grad_norm_noiseless", "grad_norm_corrected"):
            self.assertGreater(got[key], 0.0, f"{key} 应为正（门控确实吃到梯度）")
        # 默认初始化夹具下：双 sigmoid + 噪声把学习信号显著压低（夹具读数，不是定理）
        self.assertGreater(got["grad_norm_corrected"], got["grad_norm_historical"])
        self.assertGreater(got["attenuation_total"], 1.0)
        self.assertIsNotNone(got["attenuation_noise"])
        self.assertIsNotNone(got["attenuation_double_sigmoid"])
        # 局部灵敏度上限（可证）：历史路径 max_z σ′(z) = 0.25，修正路径 max_z (1/τ)σ′(z/τ) = 1/τ = 2
        z = torch.linspace(-20.0, 20.0, 40001, dtype=torch.float64)
        s = torch.sigmoid(z)
        self.assertAlmostEqual(float((s * (1.0 - s)).max()), 0.25, places=6)
        self.assertAlmostEqual(got["local_sensitivity_logit_max"], 0.25, places=12)
        self.assertAlmostEqual(got["local_sensitivity_corrected_max"], 2.0, places=12)
        # 无副作用：不写 .grad、不改参数、不动全局 RNG
        self.assertTrue(all(param.grad is None for param in head.parameters()))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        for name, value in head.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), f"探针改动了参数: {name}")
        self.assertTrue(head.training)
        json.dumps(got)

    def test_probe_with_saturated_gate_reports_finite_readings(self):
        """门控饱和（bias=30）时读数仍有限、JSON 可序列化（衰减比可为 None，不得是 inf/NaN）。"""
        head = tiny_variant()
        head.eval()
        with torch.no_grad():
            head.affinity_mlp[2].weight.zero_()
            head.affinity_mlp[2].bias.fill_(30.0)
        got = AFF.gradient_probe(head, *tiny_reps())
        for key, value in got.items():
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value), f"{key} 不是有限数: {value}")
        json.dumps(got)

    def test_baseline_newtask_is_rejected(self):
        with self.assertRaises(ValueError):
            AFF.gradient_probe(tiny_newtask(), *tiny_reps())


class TestEvaluateGate(unittest.TestCase):
    """best_state 载入后的诊断：只用 val 前向一遍，不改参数、不训练、不抽 RNG、不参与选点。"""

    def test_matches_recomputation_and_has_no_side_effects(self):
        device = torch.device("cpu")
        P.seed_model(P.MODEL_SEED)
        head = tiny_variant(device=device)
        backbone = tiny_backbone(device)
        loaders, _, _ = tiny_inputs()
        before = {name: value.detach().clone() for name, value in head.state_dict().items()}
        sha_before = P.backbone_sha256(backbone)
        rng_before = torch.get_rng_state().clone()
        head.train()

        got = AFF.evaluate_gate(head, backbone, loaders["val"], device)
        # 模式还原只能在**诊断返回的那一刻**读：下面的重算循环自己要 `head.eval()`，把它留到末尾断言等于
        # 在测测试自己的模式切换（这条最初被前面的 RNG 断言挡住，修好 RNG 后才暴露出来）。
        self.assertTrue(head.training, "诊断必须还原调用前的模式")

        stats = AFF.AffinityStats()
        head.eval()
        for _, _, _, features in loaders["val"]:
            dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
            stats.update(head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs))
        for key, value in stats.result().items():
            self.assertEqual(got[key], value, f"诊断口径不一致: {key}")

        self.assertEqual(got["n_samples"], 16)
        self.assertEqual(got["mode"], "eval")
        self.assertAlmostEqual(got["eval_routing_share_attn"], 1.0, places=9)      # D1：eval 恒选 attention
        # STE 精度留痕（见模块 D1）：W 与 W_attn 的差是 ~1e-8…1e-7 量级，**不得**断言"恒为 0"；
        # 可证的是判据 2 的上界（1e-6）——退化判定不受这条舍入影响。
        self.assertLess(got["routing_l1"], AFF.ROUTING_L1_MIN)
        self.assertLess(got["routing_l1_max"], AFF.ROUTING_L1_MIN)
        self.assertAlmostEqual(got["routing_l1"], got["routing_l1_mean"], places=12)
        self.assertGreaterEqual(got["routing_identity_frac"], 0.0)
        self.assertLessEqual(got["routing_identity_frac"], 1.0)
        self.assertGreater(got["routing_fw_l1_mean"], 0.0)                         # 与 FW 明显不同
        self.assertLessEqual(got["noise_share_window"], AFF.GUMBEL_SHARE_WINDOW_MAX + 1e-9)
        # D1 的结构签名（g_logit = σ(z) ∈ (0,1)）与其规则文本随诊断落盘
        self.assertEqual(got["g_logit_interval"], AFF.RULE_G_LOGIT_INTERVAL)
        self.assertTrue(got["g_logit_in_unit_interval"])
        self.assertIsNone(got["grad_norm"])                                        # 判据 3 不在这里测
        self.assertAlmostEqual(got["temperature"], 150.0, places=12)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()), "诊断不得消耗全局 RNG")
        for name, value in head.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), f"诊断改动了参数: {name}")
        self.assertEqual(P.backbone_sha256(backbone), sha_before)
        P.assert_no_grads(backbone)
        json.dumps(got)

    def test_analytic_readings_pin_the_d1_signature(self):
        """诊断读数把 D1 的"代表性事实"钉死：eval 全 attention + 解析窗口内的训练期路由概率。"""
        device = torch.device("cpu")
        head = tiny_variant(device=device)
        loaders, _, _ = tiny_inputs()
        got = AFF.evaluate_gate(head, tiny_backbone(device), loaders["val"], device)
        verdict = AFF.degeneracy_verdict(eval_share=got["eval_routing_share_attn"],
                                         routing_l1=got["routing_l1"], grad_norm=1e-3,
                                         noise_window=got["noise_share_window"])
        self.assertEqual(verdict["status"], "CONFIRMED_DEGENERATE")
        self.assertIn("eval_attn_share >= 0.999", verdict["triggered_rules"])
        self.assertIn("routing_l1 < 1e-6", verdict["triggered_rules"])
        self.assertGreater(got["attention_drop_frac_exact"], 0.0)      # 训练期确有样本丢 attention
        self.assertLess(got["attention_drop_frac_exact"], 0.5)

    def test_baseline_newtask_is_rejected(self):
        device = torch.device("cpu")
        loaders, _, _ = tiny_inputs()
        with self.assertRaises(ValueError):
            AFF.evaluate_gate(tiny_newtask(device=device), tiny_backbone(device), loaders["val"], device)


class TestTrainGateTrace(unittest.TestCase):
    """训练轨迹：读取真实前向的 `last_terms`（不重算 ⇒ 不额外抽 RNG）+ 梯度范数。"""

    def test_trace_records_grad_stats_and_gate_readings(self):
        head = tiny_variant()
        head.train()
        reps = tiny_reps()
        trace = AFF.TrainGateTrace()
        trace.start_epoch(1)
        head.zero_grad()
        head(*reps).sum().backward()
        first = trace.record_step(head)
        first_hard = float(head.last_terms["g_hard"].mean())        # 每次前向抽新噪声 ⇒ 逐步记录
        head.zero_grad(set_to_none=True)                        # 梯度清空 → 本步范数为 0
        second = trace.record_step(head)
        second_hard = float(head.last_terms["g_hard"].mean())
        record = trace.end_epoch()

        self.assertGreater(first, 0.0)
        self.assertEqual(second, 0.0)
        self.assertEqual(record["epoch"], 1)
        self.assertEqual(record["steps"], 2)
        self.assertEqual(record["grad_nonzero_steps"], 1)
        self.assertAlmostEqual(record["grad_norm_mean"], first / 2, places=9)
        self.assertAlmostEqual(record["grad_norm_max"], first, places=9)
        self.assertAlmostEqual(record["g_hard_mean"], (first_hard + second_hard) / 2, places=6)
        self.assertLessEqual(record["noise_share_exact_window"], AFF.GUMBEL_SHARE_WINDOW_MAX + 1e-9)
        self.assertEqual(record["n_samples"], 2 * reps[0].shape[0])

        result = trace.result()
        self.assertEqual((result["epochs"], result["steps"]), (1, 2))
        self.assertAlmostEqual(result["grad_norm_max"], first, places=9)
        self.assertEqual(len(result["per_epoch"]), 1)
        json.dumps(result)

    def test_trace_consumes_no_extra_rng(self):
        """D3 的接线要求：记录读数**不得**重算门控（否则每步多抽一次 Gumbel 噪声）。"""
        head = tiny_variant()
        head.train()
        reps = tiny_reps()
        trace = AFF.TrainGateTrace()
        trace.start_epoch(1)
        head(*reps)
        before = torch.get_rng_state().clone()
        trace.record_step(head)
        self.assertTrue(torch.equal(before, torch.get_rng_state()), "轨迹记录不得消耗全局 RNG")
        trace.end_epoch()

    def test_zero_gradient_forever_fires_the_degeneracy_rule(self):
        head = tiny_variant()
        head.train()
        reps = tiny_reps()
        trace = AFF.TrainGateTrace()
        trace.start_epoch(1)
        head.zero_grad(set_to_none=True)
        head(*reps)
        trace.record_step(head)
        trace.end_epoch()
        result = trace.result()
        self.assertEqual(result["grad_norm_max"], 0.0)
        self.assertEqual(result["grad_nonzero_steps"], 0)
        verdict = AFF.degeneracy_verdict(eval_share=0.5, routing_l1=0.1,
                                         grad_norm=result["grad_norm_max"], noise_window=0.01)
        self.assertEqual(verdict["status"], "CONFIRMED_DEGENERATE")
        self.assertIn("grad_norm == 0", verdict["triggered_rules"])

    def test_empty_trace_reports_zero_grad_norm(self):
        result = AFF.TrainGateTrace().result()
        self.assertEqual((result["epochs"], result["steps"]), (0, 0))
        self.assertEqual(result["grad_norm_max"], 0.0)
        json.dumps(result)

    def test_epoch_bookkeeping_and_guard_errors_are_explicit(self):
        trace = AFF.TrainGateTrace()
        with self.assertRaises(RuntimeError):
            trace.record_step(tiny_variant())                            # 未 start_epoch
        trace.start_epoch(1)
        with self.assertRaises(RuntimeError):
            trace.start_epoch(2)                                         # 上一个 epoch 未收尾
        baseline = NewTask(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                           tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)
        with self.assertRaises(ValueError):
            trace.record_step(baseline)                                  # 基线头不属于处理臂
        head = tiny_variant()
        head.last_terms = None
        with self.assertRaises(RuntimeError):
            trace.record_step(head)                                      # forward 尚未调用


class TestVariantFactory(unittest.TestCase):
    def test_baseline_variant_is_the_untouched_newtask(self):
        head = AFF.build_newtask(AFF.BASELINE_VARIANT, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                 tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)
        self.assertIs(type(head), NewTask)                                # 不是子类，就是基线类本身
        self.assertNotIn("affinity_mlp.0.weight", head.state_dict())
        self.assertEqual(AFF.stage2_config(AFF.BASELINE_VARIANT), {})

    def test_baseline_variant_consumes_no_extra_rng(self):
        """基线臂逐位不变：同 seed 下与直接构造的 master NewTask 完全一致（含 RNG 终点）。"""
        P.seed_model(P.MODEL_SEED)
        reference = tiny_newtask()
        rng_after_reference = torch.get_rng_state().clone()

        P.seed_model(P.MODEL_SEED)
        head = AFF.build_newtask(AFF.BASELINE_VARIANT, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                 tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_after_reference))
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(head.state_dict()[name], value), f"基线初始化漂移: {name}")

    def test_treatment_variant_is_the_faithful_subclass(self):
        head = AFF.build_newtask(AFF.VARIANT, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                 tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)
        self.assertIsInstance(head, AFF.HistoricalAffinityGateNewTask)
        self.assertIsInstance(head, NewTask)
        self.assertIn("affinity_mlp.0.weight", head.state_dict())
        config = AFF.stage2_config(AFF.VARIANT)
        self.assertEqual(config["fusion_mode"], AFF.FUSION_MODE)
        self.assertAlmostEqual(config["affinity_tau"], 0.5, places=12)
        self.assertEqual(config["affinity_hidden_units"], 16)

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            AFF.build_newtask("no_such_variant", input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                              tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)


class TestRunnerIntegration(unittest.TestCase):
    """接线（I9）：标准 short 命令 + `--variant affinity` / `--affinity`；基线臂不受影响。"""

    def test_treatment_arm_records_arm_block_trace_and_probes(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            out = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=2,
                                                  device=device, model=tiny_backbone(device),
                                                  loaders=loaders, stats=stats, indices=indices,
                                                  input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                                  variant=AFF.VARIANT)
            run_path = Path(out["run_dir"])
            self.assertTrue(out["run_id"].endswith(AFF.RUN_ID_SUFFIX), out["run_id"])

            config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["variant"], AFF.VARIANT)
            self.assertEqual(config["fusion_mode"], AFF.FUSION_MODE)
            self.assertAlmostEqual(config["affinity_tau"], 0.5, places=12)

            payload = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["variant"], AFF.VARIANT)
            self.assertEqual(sorted(payload["mechanism"]),                     # M3/M4 键集不变
                             ["cos_gen_spec", "env_acc_stage1", "gate_mean", "gen_std"])
            arm = payload["affinity_arm"]
            self.assertEqual(arm["prereg"]["auc_test_min"], 0.8521)
            self.assertEqual(arm["prereg"]["rules"]["stop_fidelity"], "noise_share_window <= 0.32")
            self.assertIn(arm["degeneracy"]["status"], ("CONFIRMED_DEGENERATE", "NOT_DEGENERATE"))
            self.assertIn(arm["status"], ("STOP", "EFFECT_CONFIRMED", "EFFECT_NOT_CONFIRMED"))
            self.assertEqual(arm["effect"]["evaluated"],
                             not arm["fidelity"]["triggered"] and arm["degeneracy"]["status"] == "NOT_DEGENERATE")
            trace = arm["train_gate_trace"]
            self.assertEqual([record["epoch"] for record in trace], [1, 2])     # 逐 epoch 轨迹
            self.assertGreater(trace[0]["steps"], 0)
            self.assertGreater(arm["affinity_grad_norm"]["steps"], 0)

            diagnostics = arm["diagnostics"]
            self.assertEqual(diagnostics["n_samples"], 16)                      # val 只跑一遍
            self.assertAlmostEqual(diagnostics["eval_routing_share_attn"], 1.0, places=9)   # D1 签名
            self.assertLess(diagnostics["routing_l1"], AFF.ROUTING_L1_MIN)      # STE 留痕：< 1e-6，不恒为 0
            self.assertLessEqual(diagnostics["noise_share_window"], AFF.GUMBEL_SHARE_WINDOW_MAX + 1e-9)
            self.assertEqual(diagnostics["env_layout"], AFF.ENV_LAYOUT)         # D2 复核：布局自证
            self.assertIn("g_logit_in_unit_interval", diagnostics)              # D1 结构签名随诊断落盘
            self.assertIn("corrected_routing_share_attn", diagnostics)          # D6 修正口径
            self.assertIn("cos_col_mean", diagnostics)

            probe = arm["noise_signal_probe"]
            self.assertEqual(probe["n_samples"], 16)
            self.assertGreaterEqual(probe["noise_share_exact_min"], AFF.GUMBEL_SHARE_LO - 1e-9)
            self.assertLessEqual(probe["noise_share_exact_max"], AFF.GUMBEL_SHARE_HI + 1e-9)
            grads = arm["gradient_probe"]
            self.assertGreater(grads["grad_norm_historical"], 0.0)
            self.assertGreater(grads["grad_norm_corrected"], 0.0)
            self.assertIn(ARCHIVE_REF, arm["provenance"]["historical_source"])
            # 诊断/探针在 A1 取哈希之后运行 ⇒ 臂内补验冻结未被扰动
            self.assertEqual(arm["backbone_sha256_after_probes"], payload["backbone_sha256_after"])
            self.assertTrue(arm["grads_all_none_after_probes"])

            report = json.loads((run_path / "gate_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["A1"]["pass"])                               # 冻结三件套不受影响
            self.assertEqual(report["A3"]["status"], "on_demand")
            self.assertNotIn("affinity_arm", report)                            # 臂级判定不混入 A/B 门禁

            state = torch.load(run_path / "newtask.pt", map_location="cpu")
            tiny_variant(device=device).load_state_dict(state)                   # strict：门控权重在 checkpoint 里
            self.assertIn("affinity_mlp.0.weight", state)

            summary = (root / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn(out["run_id"], summary)                                # 失败也必须留痕（协议 7.3）

    def test_baseline_arm_is_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            with mock.patch.object(AFF, "HistoricalAffinityGateNewTask",
                                   side_effect=AssertionError("基线臂不得构造 affinity 头")):
                out = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=2,
                                                      device=device, model=tiny_backbone(device),
                                                      loaders=loaders, stats=stats, indices=indices,
                                                      input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM)
            run_path = Path(out["run_dir"])
            self.assertFalse(out["run_id"].endswith(AFF.RUN_ID_SUFFIX))
            self.assertIsNone(out["arm"])

            config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["variant"], AFF.BASELINE_VARIANT)
            self.assertNotIn("fusion_mode", config)                              # 处理臂专属字段不得出现
            self.assertEqual(sorted(config),                                     # 键集 = 基线键集 + variant 标签
                             ["batch_size", "commit", "env_seed", "epochs", "frozen", "input_size", "lr",
                              "model_seed", "patience", "rep_dim", "run_id", "split_seed", "stage1_id",
                              "tag", "variant"])

            payload = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["variant"], AFF.BASELINE_VARIANT)
            self.assertNotIn("affinity_arm", payload)
            self.assertNotIn("affinity_mlp", json.dumps(payload))
            self.assertEqual(sorted(payload),
                             ["backbone_sha256_after", "backbone_sha256_before", "commit",
                              "env_ids_sha256", "mechanism", "run_id", "split_sha256", "stage1",
                              "stage1_id", "stage2", "variant"])
            self.assertEqual(sorted(payload["mechanism"]),
                             ["cos_gen_spec", "env_acc_stage1", "gate_mean", "gen_std"])
            state = torch.load(run_path / "newtask.pt", map_location="cpu")
            tiny_newtask(device=device).load_state_dict(state)                   # strict 载入基线头

    def test_cli_exposes_variant_and_affinity_alias(self):
        parser = run_census_benchmark.build_parser()
        args = parser.parse_args(["stage2", "--stage1-dir", "x"])
        self.assertEqual(args.variant, AFF.BASELINE_VARIANT)
        args = parser.parse_args(["stage2", "--stage1-dir", "x", "--variant", AFF.VARIANT])
        self.assertEqual(args.variant, AFF.VARIANT)
        args = parser.parse_args(["stage2", "--stage1-dir", "x", "--affinity"])
        self.assertEqual(args.variant, AFF.VARIANT)
        with self.assertRaises(SystemExit):
            parser.parse_args(["stage2", "--stage1-dir", "x", "--variant", "nope"])

    def test_static_guard_protocol_and_model_untouched(self):
        """本分支只加复现件：模型/配置不动，协议与指标文件在本分支上不动。"""
        for args, note in (
                (["master", "--", "multitaskrec", "config.py"], "模型/配置"),
                (["HEAD", "--", "census_benchmark/protocol.py", "census_benchmark/metrics.py"], "协议/指标")):
            diff = _git_text(["diff", "--name-only", *args]).stdout.strip()
            self.assertEqual(diff, "", f"{note}文件不得改动: {diff}")


if __name__ == "__main__":
    unittest.main()
