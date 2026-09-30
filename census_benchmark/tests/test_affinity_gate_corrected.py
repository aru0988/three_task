"""修正版逐样本 Affinity 软门控（确定性、无 Gumbel、train/eval 同一公式）的测试。

唯一事实来源：docs/superpowers/experiments/2026-09-30-stage2-affinity-gate-corrected.md
被修正的历史缺陷（见 docs/superpowers/experiments/2026-09-30-stage2-affinity-gate-revalidation.md
第 3、8 节；实验结论 `STOP` / `CONFIRMED_DEGENERATE`）：

  D1 双 sigmoid + eval 期硬门控恒选 attention（`g_hard ≡ 1`）⇒ 评测路径上根本没有路由；
  D1(c)/D6 训练期 Gumbel 噪声主导（`noise_to_signal_ratio = 15.68`）且不存活到 eval；
  D3 训练期前向消耗全局 RNG + 构造顺序位移（共享参数初值与基线不同）；
  D4 硬 STE + 噪声污染梯度（门控被推向常数角）；
  D2(b) 门控特征与 `W_attn` 同源、K=2 时实际只有 2 个自由数。

本文件锁定 I1–I10（与 experiments 文档第 5 节一致）：

  I1 **确定性**：无 Gumbel、无 dropout、无任何随机分支；门控前向不消耗全局 RNG；train / eval 逐位一致；
  I2 **单 sigmoid**：`g = sigmoid(logit)`，logit 无约束、`routing_gate_mlp` 内不得有 `nn.Sigmoid`；
  I3 **中性初始化**：末层 bias 恒零、权重按 `GATE_INIT_WEIGHT_SCALE` 缩小 ⇒ 初始 gate 均值 ≈ 0.5、
     非饱和，且特征通路在**第 1 步**就有非零梯度（不做 zero-init 的"死启动"）；
  I4 **逐样本**：gate 输出只依赖该样本（batch 组成无关；源码守卫：门控路径不得有 `dim=0` 归约）；
  I5 **基线公平**：构造 = 基线 `NewTask` 构造 + 门控构造（共享参数与其 RNG 消耗与基线逐位一致）；
     `get_l2_reg()` 不含门控（D5 不改）；
  I6 **特征可解释且增加信息**：3 维 = [注意力熵, cos(gen, attention 融合表征), cos(gen, 均匀融合表征)]；
     与历史 4 维（只依赖 `H(x)`、K=2 时有精确恒等式）不同，本特征集依赖 `gen_rep` ⇒ 含历史特征集
     结构上无法看到的信息；满列秩（无精确线性恒等式）；
  I7 **blend**：`W = g·W_attn + (1−g)·W_fw`，端点上逐位等于纯 attention / 纯均匀；
  I8 **诊断（val、训练后、零泄漏）**：留一源（L1O）边际预测效应、逐源依赖份额、gate 与"attention 路由
     相对于均匀路由的逐样本效用"的秩相关（预注册保守阈值）；逐源"权重 vs 依赖"相关**只作诊断**并
     记录其力学耦合（标量 gate 没有逐源偏好 ⇒ 逐源相关性不可识别，不得伪造）；
  I9 **预注册判据**：机制 6 条 → 效应（`>= 0.8521`）→ 对齐（`spearman >= 0.05`）三层主导；
  I10 **接线**：`run_id` 后缀 `-affcorr`、config / metrics 记录；基线臂逐位不受影响。

测试用 CPU 极小夹具（`input_size=8`、`rep_dim=4`），**不构成**任何性能证据。
"""
import inspect
import json
import math
import re
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
from census_benchmark import affinity_gate_corrected as AGC
from census_benchmark import protocol as P
from multitaskrec.model import NewTask

# 与 test_protocol / test_smoke / test_affinity_gate 同口径：dnn_input = 2 特征 × embedding 4 = 8
TINY_VOCAB, TINY_INPUT_SIZE, TINY_EMBEDDING, TINY_REP_DIM = {"a": 3, "b": 2}, 8, 4, 4
REPO = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parents[1] / "affinity_gate_corrected.py"
DOC = REPO / "docs" / "superpowers" / "experiments" / "2026-09-30-stage2-affinity-gate-corrected.md"
# Windows 中文环境下 `text=True` 默认按本地编码（GBK/cp936）解码，git 输出是 UTF-8 ⇒ 显式钉死编码。
GIT_TEXT_ENCODING = "utf-8"


def _git_text(args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          encoding=GIT_TEXT_ENCODING)


# 文档阶段切分：`## 11.` 标题之前 = 头部声明 + 冻结的预注册/设计（第 1–10 节）；之后 = 结果回填。
DOC_PHASE_SPLIT = re.compile(r"(?m)^## 11\. ")


def split_doc_phases(text):
    """按 `## 11.` 结果小节标题切分文档，返回 `(冻结的预注册段, 结果回填段)`。

    标题缺失 ⇒ 直接失败（阶段结构被破坏），而不是静默地把整份文档都算成预注册段。
    """
    match = DOC_PHASE_SPLIT.search(text)
    if match is None:
        raise AssertionError("文档必须含 `## 11. ` 结果小节标题")
    return text[:match.start()], text[match.start():]


def tiny_backbone(device=None, num_tasks=2):
    """dnn_input 8 维、rep_dim 4、`num_tasks` 个 env（默认 2，与其它测试文件同一夹具口径）。"""
    return run_census_benchmark.MPTRec(
        num_tasks=num_tasks, feature_vocabulary=dict(TINY_VOCAB), embedding_size=TINY_EMBEDDING,
        input_size=TINY_INPUT_SIZE, expert_dnn_hidden_units=(8, TINY_REP_DIM),
        tower_dnn_hidden_units=(4, 2), device=device)


def tiny_newtask(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM, device=None):
    """基线头（不得改动）。"""
    return NewTask(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=list(P.TOWER_HIDDEN),
                   reg_dnn=P.REG_DNN, device=device)


def tiny_corrected(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM, device=None):
    """处理臂头：修正版逐样本软门控（子类，不动 model.py）。"""
    return AGC.CorrectedAffinityGateNewTask(
        input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=list(P.TOWER_HIDDEN),
        reg_dnn=P.REG_DNN, device=device)


def tiny_reps(device=None, n=8, seed=3, num_tasks=2):
    """抽一批表征：(dnn_input, gen_rep, spec_reps, env_embs)——与真实路径同款 no_grad 抽取。"""
    gen = torch.Generator().manual_seed(seed)
    features = {"a": torch.randint(0, TINY_VOCAB["a"], (n,), generator=gen),
                "b": torch.randint(0, TINY_VOCAB["b"], (n,), generator=gen)}
    with torch.no_grad():
        return tiny_backbone(device, num_tasks=num_tasks).get_infos(features)


def diverse_reps(device=None, n=64, seed=3, vocab_a=41, vocab_b=29):
    """取值多样的夹具（`TINY_VOCAB` 只有 3×2 = 6 种输入模式，表征在样本间几乎不变）。

    用于"特征必须有逐样本变化"的断言（满列秩、与历史特征集的对照）：模型结构/形状与 `tiny_backbone`
    完全相同（2 个特征 × embedding 4 = input_size 8），只是词表更大 ⇒ 64 个样本给出 64 种不同输入。
    """
    gen = torch.Generator().manual_seed(seed)
    features = {"a": torch.randint(0, vocab_a, (n,), generator=gen),
                "b": torch.randint(0, vocab_b, (n,), generator=gen)}
    backbone = run_census_benchmark.MPTRec(
        num_tasks=2, feature_vocabulary={"a": vocab_a, "b": vocab_b}, embedding_size=TINY_EMBEDDING,
        input_size=TINY_INPUT_SIZE, expert_dnn_hidden_units=(8, TINY_REP_DIM),
        tower_dnn_hidden_units=(4, 2), device=device)
    with torch.no_grad():
        return backbone.get_infos(features)


def historical_features(terms, dtype=torch.float64):
    """历史 4 维门控输入（**只**依赖 H(x) 与 E）：[max_k C, min_k C, max−min, mean_k |C|]。

    仅用于对照：证明本模块的特征集依赖 `gen_rep`（历史特征集结构上看不到的信息），
    以及"K=2 时 4 维里有精确线性恒等式"这一结构性事实。
    在 float64 下计算：恒等式（第 3 维 = 第 1 − 第 2 维）在精确算术下成立，float32 只会让它带上
    ~1e-7 的舍入量级，从而在数值秩检验里掩盖"精确线性相关"。
    """
    h_out = terms["h_out"].detach().to(dtype)
    env_embs = terms["env_embs"].detach().to(dtype)
    h_p_norm = F.normalize(h_out, dim=-1)
    e_norm = F.normalize(env_embs, dim=0)
    cos_sim = torch.mm(h_p_norm, e_norm)
    max_cos = cos_sim.max(dim=-1, keepdim=True)[0]
    min_cos = cos_sim.min(dim=-1, keepdim=True)[0]
    return torch.cat([max_cos, min_cos, max_cos - min_cos,
                      cos_sim.abs().mean(dim=-1, keepdim=True)], dim=-1)


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
    """带私有 `torch.Generator` 的 `DataLoader`：迭代本身不得动全局 RNG（口径同历史复现件）。"""
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


def probe_setup(n_train=64, n_val=16, n_test=16, batch_size=8, device=torch.device("cpu"), seed=5):
    """源贡献诊断夹具：(冻结 eval backbone, 处理臂头, loaders, stats, indices)。"""
    torch.manual_seed(seed)
    backbone = tiny_backbone(device)
    backbone.eval()
    head = tiny_corrected(device=device)
    loaders, stats, indices = tiny_inputs(n_train=n_train, n_val=n_val, n_test=n_test,
                                          batch_size=batch_size)
    return backbone, head, loaders, stats, indices


def pin_gate(head, value):
    """把门控钉死成常数：末层权重清零 + bias 取 ±40 ⇒ float32 下 sigmoid 恰好饱和到 1.0 / 0.0。"""
    with torch.no_grad():
        head.routing_gate_mlp[2].weight.zero_()
        head.routing_gate_mlp[2].bias.fill_(float(value))


class TestModuleSourceAndDoc(unittest.TestCase):
    """静态守卫（最便宜的用例）：源码可编译、文档阶段标记（预注册冻结 / 结果回填）、常量口径。"""

    def test_source_compiles(self):
        compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec")

    def test_doc_prereg_records_it_was_frozen_before_the_run(self):
        """阶段标记（预注册侧）："尚未运行"必须被**限定**为对第 1–10 节的描述。

        旧守卫只找"尚未运行"四个字——§11 结果回填后它匹配的是"这些小节写于运行之前"的阶段说明
        ⇒ 空转。这里要求同一行同时出现"第 1–10 节 / 冻结 / 一字未改"，并保留第 6、7 节
        "写在看到结果之前"的预注册标记。
        """
        prereg, _ = split_doc_phases(DOC.read_text(encoding="utf-8"))
        declarations = [line for line in prereg.splitlines()
                        if "第 1–10 节" in line and "尚未运行" in line]
        self.assertEqual(len(declarations), 1)
        self.assertIn("冻结", declarations[0])
        self.assertIn("一字未改", declarations[0])
        self.assertIn("只允许在看到结果之前修改", prereg)            # 第 6 节标题
        self.assertIn("写在看到任何结果之前", prereg)                # 第 6.4 / 7 节标记

    def test_doc_header_status_is_ran_with_stop_mechanism_result(self):
        """阶段标记（结果侧）：**状态**行记录"已运行"且结果串是 `STOP_MECHANISM`，第 11 节同串。"""
        text = DOC.read_text(encoding="utf-8")
        status_lines = [line for line in text.splitlines() if line.startswith("- **状态**：")]
        self.assertEqual(len(status_lines), 1)
        self.assertIn("已运行", status_lines[0])
        self.assertIn("STOP_MECHANISM", status_lines[0])
        self.assertNotIn("尚未运行", status_lines[0])               # 状态行不得再用"未运行"
        _, results = split_doc_phases(text)
        self.assertIn("STOP_MECHANISM", results)

    def test_doc_result_section_records_exact_run_id_and_result_markers(self):
        """阶段标记（结果侧）：§11 必须含本次 run 的精确身份与关键读数。

        防退化点：运行后才存在的标记（run_id 与结果读数）**不得**出现在冻结的预注册段——
        否则说明预注册被事后回写。
        """
        text = DOC.read_text(encoding="utf-8")
        prereg, results = split_doc_phases(text)
        run_only = (
            "20260930-1912-s20260929-m1685480945-short-87afe03-affcorr",   # 本次 run_id
            "0.8491303844406325",                                          # AUC-Test-Education
            "0.007132461220191373",                                        # 唯一失败读数 gate_std
            "0.0009381462769431",                                          # Δ vs 基线
            "0.24991698792103564",                                         # gate_utility_spearman
        )
        for marker in run_only:
            self.assertIn(marker, results, marker)
            self.assertNotIn(marker, prereg, marker)
        # 基线对照数在预注册段（第 6.3 节）也出现 ⇒ 这一组只查结果段。
        for marker in ("AUC-Test-Education = 0.8491303844406325",
                       "0.8500685307175756",
                       'failed_rules = ["gate_std >= 0.01"]',
                       "status = STOP_MECHANISM",
                       "stop_reason = mechanism_fail"):
            self.assertIn(marker, results, marker)

    def test_static_guard_model_config_protocol_metrics_untouched(self):
        """本分支只加隔离模块：模型/配置不动，协议与指标文件在本分支上不动。"""
        for args, note in (
                (["master", "--", "multitaskrec", "config.py",
                  "CensusIncome_MPTRec.py", "CensusIncome_NewTask.py"], "模型/配置"),
                (["HEAD", "--", "census_benchmark/protocol.py", "census_benchmark/metrics.py",
                  "census_benchmark/__init__.py"], "协议/指标")):
            diff = _git_text(["diff", "--name-only", *args]).stdout.strip()
            self.assertEqual(diff, "", f"{note}文件不得改动: {diff}")

    def test_feature_names_and_dims_are_declared(self):
        self.assertEqual(AGC.GATE_FEATURE_DIM, len(AGC.FEATURE_NAMES))
        self.assertEqual(tuple(AGC.FEATURE_NAMES),
                         ("attn_entropy_norm", "cos_gen_attn", "cos_gen_fw"))
        self.assertLessEqual(AGC.GATE_HIDDEN, 16)          # MLP 必须小
        self.assertEqual(AGC.ENV_LAYOUT, "feature_first_(rep_dim, K)")


class TestPreregisteredCriteria(unittest.TestCase):
    """I9：预注册阈值、规则文本、三层主导关系、JSON 可序列化、文档与代码同数。"""

    GOOD = dict(gate_std=0.2, gate_mean=0.5, routing_l1_mean=0.03, gate_grad_norm_min_step=1e-5,
                train_eval_identical=True, rng_unchanged=True)

    def test_threshold_values(self):
        self.assertEqual(AGC.AUC_TEST_MIN, 0.8521)
        self.assertEqual(AGC.BASELINE_TEST_AUC, 0.8500685307175756)
        self.assertEqual(AGC.GATE_STD_MIN, 0.01)
        self.assertEqual((AGC.GATE_MEAN_MIN, AGC.GATE_MEAN_MAX), (0.05, 0.95))
        self.assertEqual(AGC.ROUTING_L1_MIN, 1e-4)
        self.assertEqual(AGC.GATE_GRAD_NORM_MIN, 0.0)
        self.assertEqual(AGC.ALIGNMENT_SPEARMAN_MIN, 0.05)
        self.assertEqual(AGC.VARIANT, "affinity_corrected")
        self.assertEqual(AGC.RUN_ID_SUFFIX, "-affcorr")
        self.assertEqual(AGC.FUSION_MODE, "affinity_gate_corrected")

    def test_rule_strings_are_preregistration_verbatim(self):
        self.assertEqual(AGC.RULE_GATE_STD, "gate_std >= 0.01")
        self.assertEqual(AGC.RULE_GATE_MEAN, "0.05 <= gate_mean <= 0.95")
        self.assertEqual(AGC.RULE_ROUTING_L1, "routing_l1_mean >= 1e-4")
        self.assertEqual(AGC.RULE_GATE_GRAD, "gate_grad_norm_min_step > 0")
        self.assertEqual(AGC.RULE_TRAIN_EVAL, "train_eval_identical")
        self.assertEqual(AGC.RULE_RNG, "no_global_rng_consumed")
        self.assertEqual(AGC.RULE_EFFECT, "auc_test_education >= 0.8521")
        self.assertEqual(AGC.RULE_ALIGNMENT, "spearman(gate, utility) >= 0.05")

    def test_preregistered_criteria_snapshot(self):
        prereg = AGC.preregistered_criteria()
        self.assertEqual(prereg["auc_test_min"], 0.8521)
        self.assertEqual(prereg["baseline_test_auc"], 0.8500685307175756)
        self.assertEqual(prereg["gate_std_min"], 0.01)
        self.assertEqual((prereg["gate_mean_min"], prereg["gate_mean_max"]), (0.05, 0.95))
        self.assertEqual(prereg["routing_l1_min"], 1e-4)
        self.assertEqual(prereg["alignment_spearman_min"], 0.05)
        self.assertEqual(prereg["rules"]["stop_mechanism"],
                         [AGC.RULE_GATE_STD, AGC.RULE_GATE_MEAN, AGC.RULE_ROUTING_L1,
                          AGC.RULE_GATE_GRAD, AGC.RULE_TRAIN_EVAL, AGC.RULE_RNG])
        self.assertEqual(prereg["rules"]["effect"], AGC.RULE_EFFECT)
        self.assertEqual(prereg["rules"]["alignment"], AGC.RULE_ALIGNMENT)
        self.assertEqual(prereg["dominance"], "mechanism > effect > alignment")
        json.dumps(prereg)                                  # 必须 JSON 可序列化

    def test_provenance_records_the_fixed_defects_and_env_layout_fact(self):
        prov = AGC.provenance()
        json.dumps(prov)
        text = json.dumps(prov, ensure_ascii=False)
        for token in ("D1", "D2", "D3", "D4", "D6", "Gumbel", "STE"):     # 逐条记录被修的缺陷
            self.assertIn(token, text)
        self.assertEqual(prov["env_layout"], AGC.ENV_LAYOUT)
        self.assertIn("[rep_dim, K]", prov["env_normalize_note"])          # 唯一正确表述
        self.assertIn("dim=0", prov["env_normalize_note"])
        self.assertNotIn("应为 dim=-1", text)                              # 不得复述撤回结论
        self.assertIn("非新颖", prov["non_novelty_note"])
        self.assertEqual(prov["gate_in_l2"], False)                        # D5：门控不进 L2（不改）
        self.assertEqual(prov["gumbel_used"], False)                       # 无 Gumbel、无 STE
        self.assertEqual(prov["ste_used"], False)

    def test_mechanism_verdict_passes_on_good_readings(self):
        out = AGC.mechanism_verdict(**self.GOOD)
        self.assertEqual(out["status"], "MECHANISM_OK")
        self.assertEqual(out["failed_rules"], [])
        self.assertEqual(sorted(out["checks"]), sorted([
            "gate_std_ge_min", "gate_mean_in_window", "routing_l1_ge_min", "gate_grad_nonzero_every_step",
            "train_eval_identical", "no_global_rng_consumed"]))
        json.dumps(out)

    def test_mechanism_verdict_flags_each_rule(self):
        for key, bad in (("gate_std", 0.009), ("gate_mean", 0.99), ("gate_mean", 0.01),
                         ("routing_l1_mean", 1e-5), ("gate_grad_norm_min_step", 0.0),
                         ("train_eval_identical", False), ("rng_unchanged", False)):
            out = AGC.mechanism_verdict(**{**self.GOOD, key: bad})
            self.assertEqual(out["status"], "MECHANISM_FAIL", key)
            self.assertTrue(out["failed_rules"], key)

    def test_mechanism_verdict_boundaries_are_inclusive(self):
        self.assertEqual(AGC.mechanism_verdict(**{**self.GOOD, "gate_std": 0.01})["status"], "MECHANISM_OK")
        self.assertEqual(AGC.mechanism_verdict(**{**self.GOOD, "gate_mean": 0.05})["status"], "MECHANISM_OK")
        self.assertEqual(AGC.mechanism_verdict(**{**self.GOOD, "gate_mean": 0.95})["status"], "MECHANISM_OK")
        self.assertEqual(AGC.mechanism_verdict(**{**self.GOOD, "routing_l1_mean": 1e-4})["status"],
                         "MECHANISM_OK")
        self.assertEqual(AGC.mechanism_verdict(**{**self.GOOD, "gate_grad_norm_min_step": 0.0})["status"],
                         "MECHANISM_FAIL")                  # 严格 > 0

    def test_alignment_verdict_threshold_and_undefined(self):
        ok = AGC.alignment_verdict(spearman=0.2, auc=0.55, n=1000)
        self.assertEqual(ok["status"], "ALIGNMENT_OK")
        self.assertTrue(ok["pass"])
        fail = AGC.alignment_verdict(spearman=0.049, auc=0.52, n=1000)
        self.assertEqual(fail["status"], "ALIGNMENT_FAIL")
        self.assertFalse(fail["pass"])
        edge = AGC.alignment_verdict(spearman=0.05, auc=0.52, n=1000)
        self.assertEqual(edge["status"], "ALIGNMENT_OK")     # 阈值含端点
        undef = AGC.alignment_verdict(spearman=None, auc=None, n=0)
        self.assertEqual(undef["status"], "ALIGNMENT_UNDEFINED")
        self.assertFalse(undef["pass"])
        json.dumps(undef)

    def test_arm_verdict_priority_chain(self):
        mech_fail = AGC.mechanism_verdict(**{**self.GOOD, "gate_std": 0.0})
        mech_ok = AGC.mechanism_verdict(**self.GOOD)
        ali_ok = AGC.alignment_verdict(spearman=0.2, auc=0.55, n=1000)
        ali_fail = AGC.alignment_verdict(spearman=0.01, auc=0.51, n=1000)
        ali_undef = AGC.alignment_verdict(spearman=None, auc=None, n=0)

        arm = AGC.arm_verdict(0.99, mechanism=mech_fail, alignment=ali_ok)
        self.assertEqual(arm["status"], "STOP_MECHANISM")
        self.assertEqual(arm["stop_reason"], "mechanism_fail")
        self.assertFalse(arm["effect"]["evaluated"])
        self.assertIsNone(arm["effect"]["pass"])             # 机制不过 ⇒ 不得谈效应
        self.assertFalse(arm["pass"])

        arm = AGC.arm_verdict(0.80, mechanism=mech_ok, alignment=ali_ok)
        self.assertEqual(arm["status"], "STOP_EFFECT")
        self.assertTrue(arm["effect"]["evaluated"])
        self.assertFalse(arm["effect"]["pass"])

        arm = AGC.arm_verdict(0.8600, mechanism=mech_ok, alignment=ali_fail)
        self.assertEqual(arm["status"], "PASS_ALIGNMENT_FAIL")   # 对齐不过**不**翻转效应结论
        self.assertTrue(arm["effect"]["pass"])
        self.assertFalse(arm["pass"])

        arm = AGC.arm_verdict(0.8600, mechanism=mech_ok, alignment=ali_undef)
        self.assertEqual(arm["status"], "PASS_ALIGNMENT_FAIL")   # UNDEFINED 按保守处理

        arm = AGC.arm_verdict(0.8600, mechanism=mech_ok, alignment=ali_ok)
        self.assertEqual(arm["status"], "PASS")
        self.assertTrue(arm["pass"])
        self.assertIsNone(arm["stop_reason"])
        json.dumps(arm)

    def test_arm_verdict_effect_boundary(self):
        mech_ok = AGC.mechanism_verdict(**self.GOOD)
        ali_ok = AGC.alignment_verdict(spearman=0.2, auc=0.55, n=1000)
        self.assertTrue(AGC.arm_verdict(0.8521, mechanism=mech_ok, alignment=ali_ok)["effect"]["pass"])
        self.assertFalse(AGC.arm_verdict(0.85209, mechanism=mech_ok, alignment=ali_ok)["effect"]["pass"])

    def test_doc_preregisters_same_numbers_and_rules(self):
        text = DOC.read_text(encoding="utf-8")
        prereg = AGC.preregistered_criteria()
        for rule in (prereg["rules"]["stop_mechanism"] + [prereg["rules"]["effect"],
                                                          prereg["rules"]["alignment"]]):
            self.assertIn(rule, text, rule)
        for token in ("0.8521", "0.8500685307175756", "STOP_MECHANISM", "STOP_EFFECT",
                      "PASS_ALIGNMENT_FAIL", "MECHANISM_OK", "diagnostic", "诊断"):
            self.assertIn(token, text, token)

    def test_doc_records_env_layout_fact_and_retraction(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertIn("[rep_dim, K]", text)                  # 布局事实
        self.assertIn("撤回", text)                           # "归一化维错误"之说已撤回
        self.assertIn("不是缺陷", text)
        self.assertNotIn("应为 dim=-1", text)                 # 不得复述错误结论

    def test_doc_states_non_novelty(self):
        text = DOC.read_text(encoding="utf-8")
        for token in ("非新颖", "MoE", "soft gating"):
            self.assertIn(token, text, token)


class TestConstructionAndFairness(unittest.TestCase):
    """I5：构造顺序 / 共享参数逐位一致 / RNG 流 / 中性初始化 / 单 sigmoid / L2 口径。"""

    def test_shared_parameters_bit_identical_to_baseline_at_same_seed(self):
        torch.manual_seed(7)
        base = tiny_newtask()
        torch.manual_seed(7)
        head = tiny_corrected()
        base_state, head_state = base.state_dict(), head.state_dict()
        for key, value in base_state.items():
            self.assertIn(key, head_state)
            self.assertTrue(torch.equal(value, head_state[key]), key)
        self.assertEqual(sorted(set(head_state) - set(base_state)),
                         ["routing_gate_mlp.0.bias", "routing_gate_mlp.0.weight",
                          "routing_gate_mlp.2.bias", "routing_gate_mlp.2.weight"])

    def test_construction_is_baseline_modules_then_gate_only(self):
        """构造消耗的全局 RNG = 基线构造 + 门控构造；除此之外不多抽一次。"""
        torch.manual_seed(11)
        tiny_newtask()
        AGC.build_gate_mlp()
        expected = torch.get_rng_state()
        torch.manual_seed(11)
        tiny_corrected()
        self.assertTrue(torch.equal(torch.get_rng_state(), expected))
        # 初始化缩放（mul_ / zero_）不消耗 RNG：门控权重确实被缩小、bias 恒零
        head = tiny_corrected()
        self.assertTrue(torch.equal(head.routing_gate_mlp[2].bias, torch.zeros(1)))
        bound = AGC.GATE_INIT_WEIGHT_SCALE / math.sqrt(AGC.GATE_HIDDEN)
        self.assertGreater(float(head.routing_gate_mlp[2].weight.abs().max()), 0.0)
        self.assertLessEqual(float(head.routing_gate_mlp[2].weight.abs().max()), bound * (1.0 + 1e-6))

    def test_gate_mlp_has_no_sigmoid_module_and_ends_with_linear(self):
        head = tiny_corrected()
        self.assertFalse(any(isinstance(module, nn.Sigmoid) for module in head.routing_gate_mlp.modules()))
        self.assertIsInstance(head.routing_gate_mlp[-1], nn.Linear)
        self.assertEqual(len(list(head.routing_gate_mlp.parameters())), 4)

    def test_gate_init_is_neutral_and_nonsaturated(self):
        torch.manual_seed(3)
        head = tiny_corrected()
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(torch.device("cpu"), n=32)
        terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        g, logit = terms["g"], terms["gate_logit"]
        self.assertGreater(float(g.mean()), 0.49)             # 初始接近 0.5（中性）
        self.assertLess(float(g.mean()), 0.51)
        self.assertLess(float(g.std(unbiased=False)), 0.05)
        self.assertLess(float(logit.abs().max()), 1.0)        # 非饱和
        self.assertEqual(int((logit.abs() >= AGC.SATURATION_ABS).sum()), 0)

    def test_gate_receives_gradient_on_every_parameter_at_first_step(self):
        """中性初始化**不做** zero-init 的死启动：第 1 步特征通路就有非零梯度。"""
        torch.manual_seed(3)
        head = tiny_corrected()
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(torch.device("cpu"), n=16)
        pred = head(dnn_input, gen_rep, spec_reps, env_embs)
        loss = nn.BCELoss()(pred, torch.ones_like(pred)) + head.get_l2_reg()
        grads = torch.autograd.grad(loss, list(head.routing_gate_mlp.parameters()))
        for grad, (name, _) in zip(grads, head.routing_gate_mlp.named_parameters()):
            self.assertIsNotNone(grad, name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)

    def test_get_l2_reg_excludes_the_gate(self):
        torch.manual_seed(4)
        head = tiny_corrected()
        before = head.get_l2_reg()
        with torch.no_grad():                                 # 门控权重放大 1000 倍
            for param in head.routing_gate_mlp.parameters():
                param.mul_(1000.0)
        self.assertTrue(torch.equal(before, head.get_l2_reg()))   # D5 不改：门控不进 L2

    def test_variant_factory_baseline_is_exactly_newtask(self):
        torch.manual_seed(9)
        built = AGC.build_newtask(AGC.BASELINE_VARIANT, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                  tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN,
                                  device=None)
        self.assertIs(type(built), NewTask)
        torch.manual_seed(9)
        direct = tiny_newtask()
        for key, value in direct.state_dict().items():
            self.assertTrue(torch.equal(value, built.state_dict()[key]), key)

    def test_variant_factory_treatment_and_unknown(self):
        head = AGC.build_newtask(AGC.VARIANT, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                 tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)
        self.assertIs(type(head), AGC.CorrectedAffinityGateNewTask)
        with self.assertRaises(ValueError):
            AGC.build_newtask("nope", input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                              tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN)

    def test_stage2_config_keys(self):
        self.assertEqual(AGC.stage2_config(AGC.BASELINE_VARIANT), {})     # 基线臂不得被污染
        cfg = AGC.stage2_config(AGC.VARIANT)
        self.assertEqual(cfg["fusion_mode"], AGC.FUSION_MODE)
        self.assertEqual(cfg["gate_feature_dim"], AGC.GATE_FEATURE_DIM)
        self.assertEqual(cfg["gate_hidden_units"], AGC.GATE_HIDDEN)
        self.assertEqual(list(cfg["gate_feature_names"]), list(AGC.FEATURE_NAMES))
        json.dumps(cfg)


class TestGateForward(unittest.TestCase):
    """I1/I2/I4/I7：公式逐位可复算、单 sigmoid、确定性、逐样本、blend 端点。"""

    def setUp(self):
        torch.manual_seed(21)
        self.device = torch.device("cpu")
        self.head = tiny_corrected()
        self.reps = tiny_reps(self.device, n=6)

    def test_gate_terms_match_hand_computation(self):
        dnn_input, gen_rep, spec_reps, env_embs = self.reps
        terms = self.head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)

        exist_env_embs = torch.stack(env_embs, dim=1)
        h_out = self.head.projection_network(dnn_input)
        w_attn = F.softmax(torch.mm(h_out, exist_env_embs) / self.head.temperature, dim=-1)
        spec_stack = torch.stack(spec_reps, dim=2)
        fused_attn = torch.matmul(spec_stack, w_attn.unsqueeze(2)).squeeze(2)
        fused_fw = spec_stack.mean(dim=2)
        entropy = -(w_attn * w_attn.clamp_min(AGC.ENTROPY_EPS).log()).sum(dim=-1) / math.log(2)
        features = torch.stack([entropy,
                                F.cosine_similarity(gen_rep, fused_attn, dim=-1),
                                F.cosine_similarity(gen_rep, fused_fw, dim=-1)], dim=-1)
        logit = self.head.routing_gate_mlp(features).squeeze(-1)
        g = torch.sigmoid(logit)
        w_fw = torch.full_like(w_attn, 0.5)
        w = g.view(-1, 1, 1) * w_attn.unsqueeze(2) + (1 - g.view(-1, 1, 1)) * w_fw.unsqueeze(2)

        self.assertEqual(tuple(terms["gate_features"].shape), (6, AGC.GATE_FEATURE_DIM))
        self.assertEqual(terms["num_tasks"], 2)
        for key, expected in (("w_attn", w_attn), ("fused_attn", fused_attn), ("fused_fw", fused_fw),
                              ("attn_entropy", entropy), ("gate_features", features),
                              ("gate_logit", logit), ("g", g), ("w", w), ("w_fw", w_fw)):
            self.assertTrue(torch.equal(terms[key], expected), key)

    def test_gate_is_exactly_one_sigmoid_of_the_unconstrained_logit(self):
        dnn_input, gen_rep, spec_reps, env_embs = self.reps
        terms = self.head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        self.assertTrue(torch.equal(terms["g"], torch.sigmoid(terms["gate_logit"])))
        self.assertTrue(torch.equal(terms["gate_logit"],
                                    self.head.routing_gate_mlp(terms["gate_features"]).squeeze(-1)))
        # logit 无约束：人为放大末层权重后可以离开 [0, 1]
        with torch.no_grad():
            self.head.routing_gate_mlp[2].weight.mul_(1e4)
        wide = self.head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        self.assertGreater(float(wide["gate_logit"].abs().max()), 1.0)

    def test_forward_matches_the_preserved_pipeline_with_blended_w(self):
        dnn_input, gen_rep, spec_reps, env_embs = self.reps
        pred = self.head(dnn_input, gen_rep, spec_reps, env_embs)
        terms = self.head.last_terms                               # forward 存下的中间量（已 detach）
        w = AGC.blend_routing(terms["w_attn"], terms["g"])
        expected = AGC.routing_pipeline(self.head, dnn_input, gen_rep, spec_reps, w)
        self.assertTrue(torch.equal(pred, expected))
        self.assertTrue(torch.equal(w, terms["w"]))

    def test_train_and_eval_forward_bit_identical(self):
        dnn_input, gen_rep, spec_reps, env_embs = self.reps
        self.head.eval()
        out_eval = self.head(dnn_input, gen_rep, spec_reps, env_embs)
        self.head.train()
        out_train = self.head(dnn_input, gen_rep, spec_reps, env_embs)
        self.assertTrue(torch.equal(out_eval, out_train))

    def test_forward_consumes_no_global_rng_in_either_mode(self):
        dnn_input, gen_rep, spec_reps, env_embs = self.reps
        for training in (True, False):
            torch.manual_seed(13)
            self.head.train(training)
            before = torch.get_rng_state()
            self.head(dnn_input, gen_rep, spec_reps, env_embs)
            self.assertTrue(torch.equal(before, torch.get_rng_state()), f"training={training}")

    def test_forward_is_deterministic_across_repeated_calls(self):
        dnn_input, gen_rep, spec_reps, env_embs = self.reps
        first = self.head(dnn_input, gen_rep, spec_reps, env_embs)
        second = self.head(dnn_input, gen_rep, spec_reps, env_embs)
        self.assertTrue(torch.equal(first, second))

    def test_blend_endpoints_and_convexity(self):
        w_attn = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        self.assertTrue(torch.equal(AGC.blend_routing(w_attn, torch.ones(2)), w_attn.unsqueeze(2)))
        uniform = AGC.uniform_routing(w_attn)
        self.assertTrue(torch.equal(uniform, torch.full_like(w_attn, 0.5)))
        self.assertTrue(torch.equal(AGC.blend_routing(w_attn, torch.zeros(2)), uniform.unsqueeze(2)))
        mid = AGC.blend_routing(w_attn, torch.full((2,), 0.5))
        self.assertTrue(torch.allclose(mid.sum(dim=1), torch.ones(2, 1), atol=1e-6))
        lo = torch.minimum(w_attn.unsqueeze(2), uniform.unsqueeze(2))
        hi = torch.maximum(w_attn.unsqueeze(2), uniform.unsqueeze(2))
        self.assertTrue(torch.all(mid >= lo - 1e-6) and torch.all(mid <= hi + 1e-6))

    def test_uniform_routing_generalises_to_k_sources(self):
        """`W_fw = 1/K`：**均匀路由的生成**，与 K 无关。

        这里**不**断言留一源（L1O）重新归一化——那是 `leave_one_out_weights` 的语义（把某个源的权重置 0
        后在其余源上重新归一），与"均匀路由"是两件事，见 `TestSourceContributionProbe` 的两个用例。
        """
        for num_sources in (1, 2, 3, 5):
            w_attn = torch.full((4, num_sources), 1.0 / num_sources)
            uniform = AGC.uniform_routing(w_attn)
            self.assertEqual(tuple(uniform.shape), (4, num_sources))
            self.assertTrue(torch.equal(uniform, torch.full_like(w_attn, 1.0 / num_sources)))
        w_attn = torch.tensor([[0.5, 0.3, 0.2]])
        self.assertTrue(torch.equal(AGC.uniform_routing(w_attn), torch.full_like(w_attn, 1.0 / 3)))

    def test_routing_helpers_keep_the_bk1_layout_for_k1_k2_k3(self):
        """形状约定：`blend_routing` / `leave_one_out_weights` 输出 [B, K, 1]，`undefined` 是 [B]。"""
        for num_sources in (1, 2, 3):
            w_attn = torch.full((5, num_sources), 1.0 / num_sources)
            w = AGC.blend_routing(w_attn, torch.full((5,), 0.5))
            self.assertEqual(tuple(w.shape), (5, num_sources, 1), num_sources)
            for removed in range(num_sources):
                w_k, undefined = AGC.leave_one_out_weights(w, removed)
                self.assertEqual(tuple(w_k.shape), (5, num_sources, 1), (num_sources, removed))
                self.assertEqual(tuple(undefined.shape), (5,), (num_sources, removed))
                if num_sources > 1:                       # 软 gate ⇒ 幸存权重和恒 > 0 ⇒ 不会退化为回退
                    self.assertFalse(bool(undefined.any()), (num_sources, removed))
                    self.assertTrue(torch.allclose(w_k.sum(dim=1), torch.ones(5, 1), atol=1e-6))

    def test_gate_path_runs_for_k1_k2_k3_with_documented_shapes(self):
        """K=1/2/3 端到端：`gate_terms` / `forward` / L1O 的每个张量形状都符合布局约定。"""
        for num_tasks in (1, 2, 3):
            head = tiny_corrected()
            dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(self.device, n=5, num_tasks=num_tasks)
            terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
            self.assertEqual(terms["num_tasks"], num_tasks)
            self.assertEqual(tuple(terms["env_embs"].shape), (TINY_REP_DIM, num_tasks))
            self.assertEqual(tuple(terms["w_attn"].shape), (5, num_tasks))
            self.assertEqual(tuple(terms["w_fw"].shape), (5, num_tasks))
            self.assertEqual(tuple(terms["attn_entropy"].shape), (5,))
            self.assertEqual(tuple(terms["g"].shape), (5,))
            self.assertEqual(tuple(terms["w"].shape), (5, num_tasks, 1))
            self.assertTrue(torch.allclose(terms["w"].sum(dim=1), torch.ones(5, 1), atol=1e-6))
            if num_tasks == 1:                     # 唯一源被移除 ⇒ 无幸存源 ⇒ 无定义 + 权重恒 0
                w_0, undefined = AGC.leave_one_out_weights(terms["w"], 0)
                self.assertTrue(bool(undefined.all()))
                self.assertEqual(float(w_0.abs().sum()), 0.0)
            else:
                for removed in range(num_tasks):
                    w_k, undefined = AGC.leave_one_out_weights(terms["w"], removed)
                    self.assertEqual(tuple(w_k.shape), (5, num_tasks, 1), (num_tasks, removed))
                    self.assertFalse(bool(undefined.any()), (num_tasks, removed))
            self.assertEqual(tuple(head(dnn_input, gen_rep, spec_reps, env_embs).shape), (5,))
            self.assertEqual(tuple(AGC.routing_pipeline(head, dnn_input, gen_rep, spec_reps,
                                                        terms["w"]).shape), (5,))

    def test_gate_is_per_sample_and_batch_composition_independent(self):
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(self.device, n=8)
        # `env_embs` 的元素是**逐源**一维 [rep_dim]（布局见 ENV_LAYOUT），**没有 batch 轴** ⇒ 子批 / 重排
        # 时只能切真正带 batch 轴的量（dnn_input / gen_rep / spec_reps），env_embs 必须原样传入。
        for rep in env_embs:
            self.assertEqual(rep.dim(), 1)                           # 守卫夹具本身：一维 ⇒ 不可按样本切
        terms_all = self.head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        terms_head = self.head.gate_terms(dnn_input[:3], gen_rep[:3], [rep[:3] for rep in spec_reps],
                                          env_embs)
        self.assertTrue(torch.allclose(terms_all["g"][:3], terms_head["g"], atol=1e-6, rtol=0))
        perm = torch.tensor([4, 0, 7, 1, 6, 2, 5, 3])
        terms_perm = self.head.gate_terms(dnn_input[perm], gen_rep[perm],
                                          [rep[perm] for rep in spec_reps],
                                          env_embs)
        self.assertTrue(torch.allclose(terms_perm["g"], terms_all["g"][perm], atol=1e-6, rtol=0))

    def test_gate_path_has_no_batch_axis_reduction(self):
        """结构性守卫：门控路径的源码里不得出现任何 `dim=0` 归约（batch 轴）。"""
        for func in (AGC.CorrectedAffinityGateNewTask.gate_terms, AGC.blend_routing, AGC.uniform_routing,
                     AGC.normalized_attention_entropy, AGC.leave_one_out_weights, AGC.routing_pipeline,
                     AGC.CorrectedAffinityGateNewTask.forward):
            self.assertNotIn("dim=0", inspect.getsource(func), func.__qualname__)

    def test_layout_guard_rejects_wrong_env_layout(self):
        dnn_input, gen_rep, spec_reps, env_embs = self.reps
        bad = [env_embs[0].unsqueeze(0), env_embs[1].unsqueeze(0)]   # [1, rep_dim] ⇒ stack(dim=1) 形状错
        with self.assertRaises(ValueError):
            self.head.gate_terms(dnn_input, gen_rep, spec_reps, bad)

    def test_last_terms_are_detached_and_complete(self):
        dnn_input, gen_rep, spec_reps, env_embs = self.reps
        self.head(dnn_input, gen_rep, spec_reps, env_embs)
        terms = self.head.last_terms
        self.assertIsNotNone(terms)
        for key in ("g", "gate_logit", "gate_features", "w", "w_attn", "w_fw", "attn_entropy",
                    "cos_gen_attn", "cos_gen_fw", "fused_attn", "fused_fw", "num_tasks"):
            self.assertIn(key, terms)
        for key, value in terms.items():
            if torch.is_tensor(value):
                self.assertFalse(value.requires_grad, key)
                self.assertIsNone(value.grad_fn, key)


class TestGateFeatures(unittest.TestCase):
    """I6：特征可解释、逐样本、依赖 gen_rep（历史特征集看不到的新信息）、无精确线性恒等式。"""

    def setUp(self):
        torch.manual_seed(31)
        self.device = torch.device("cpu")
        self.head = tiny_corrected()

    def test_entropy_limits(self):
        uniform = torch.tensor([[0.5, 0.5]])
        one_hot = torch.tensor([[1.0, 0.0]])
        self.assertAlmostEqual(float(AGC.normalized_attention_entropy(uniform)), 1.0, places=12)
        self.assertAlmostEqual(float(AGC.normalized_attention_entropy(one_hot)), 0.0, places=9)
        three = torch.tensor([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
        self.assertAlmostEqual(float(AGC.normalized_attention_entropy(three)), 1.0, places=12)
        single = torch.tensor([[1.0]])                                    # K=1：唯一路由 ⇒ 熵 0（不除 log 1）
        self.assertAlmostEqual(float(AGC.normalized_attention_entropy(single)), 0.0, places=12)

    def test_features_use_gen_rep_while_historical_features_cannot(self):
        dnn_input, gen_rep, spec_reps, env_embs = diverse_reps(self.device, n=32)
        a = self.head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        b = self.head.gate_terms(dnn_input, gen_rep + 0.5, spec_reps, env_embs)
        # 历史 4 维特征只依赖 H(x) 与 E ⇒ 逐位不变
        self.assertTrue(torch.equal(historical_features(a), historical_features(b)))
        # 本特征集依赖 gen_rep：熵（只依赖路由）不变，两个一致性特征变化
        self.assertTrue(torch.equal(a["gate_features"][:, 0], b["gate_features"][:, 0]))
        self.assertFalse(torch.equal(a["gate_features"][:, 1], b["gate_features"][:, 1]))
        self.assertFalse(torch.equal(a["gate_features"][:, 2], b["gate_features"][:, 2]))

    def test_features_vary_across_samples_on_a_diverse_fixture(self):
        dnn_input, gen_rep, spec_reps, env_embs = diverse_reps(self.device, n=64)
        terms = self.head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        for column, name in enumerate(AGC.FEATURE_NAMES):
            self.assertGreater(float(terms["gate_features"][:, column].std(unbiased=False)), 0.0, name)

    def test_cos_features_are_true_cosines_of_the_right_pairs(self):
        dnn_input, gen_rep, spec_reps, env_embs = diverse_reps(self.device, n=16)
        terms = self.head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        self.assertTrue(torch.equal(terms["cos_gen_attn"],
                                    F.cosine_similarity(gen_rep, terms["fused_attn"], dim=-1)))
        self.assertTrue(torch.equal(terms["cos_gen_fw"],
                                    F.cosine_similarity(gen_rep, terms["fused_fw"], dim=-1)))
        self.assertTrue(bool(torch.all(terms["cos_gen_attn"] >= -1.0)))
        self.assertTrue(bool(torch.all(terms["cos_gen_attn"] <= 1.0)))
        self.assertTrue(bool(torch.all(terms["cos_gen_fw"] >= -1.0)))
        self.assertTrue(bool(torch.all(terms["cos_gen_fw"] <= 1.0)))
        # fused_fw = 逐源均匀平均（对 K 轴求平均，不是 batch 轴）
        self.assertTrue(torch.equal(terms["fused_fw"], torch.stack(spec_reps, dim=2).mean(dim=2)))
        # fused_attn = 按 W_attn 融合
        self.assertTrue(torch.equal(terms["fused_attn"],
                                    torch.matmul(torch.stack(spec_reps, dim=2),
                                                 terms["w_attn"].unsqueeze(2)).squeeze(2)))

    def test_feature_matrix_has_full_column_rank_whereas_historical_set_has_an_identity(self):
        dnn_input, gen_rep, spec_reps, env_embs = diverse_reps(self.device, n=64)
        terms = self.head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        features = terms["gate_features"].double()
        self.assertEqual(int(torch.linalg.matrix_rank(features)), AGC.GATE_FEATURE_DIM)
        hist = historical_features(terms)                              # 已在 float64 下计算
        self.assertLess(int(torch.linalg.matrix_rank(hist)), 4)        # 历史：c3 = c1 − c2（精确恒等式）
        self.assertTrue(torch.equal(hist[:, 2], hist[:, 0] - hist[:, 1]))
        expected_absmean = (hist[:, 0].abs() + hist[:, 1].abs()) / 2
        self.assertTrue(torch.allclose(hist[:, 3], expected_absmean, atol=1e-9))


class TestRoutingGateStatsAndTrace(unittest.TestCase):
    """I1/I8：流式累加器口径正确；训练轨迹不重算、不抽 RNG；空轨迹按 0 处理。"""

    def setUp(self):
        torch.manual_seed(41)
        self.device = torch.device("cpu")
        self.head = tiny_corrected()
        self.reps = tiny_reps(self.device, n=8)

    def test_stats_match_hand_computed_batch_readings(self):
        terms = self.head.gate_terms(*self.reps)
        stats = AGC.RoutingGateStats()
        stats.update(terms)
        out = stats.result()
        g = terms["g"].double()
        self.assertEqual(out["n_samples"], 8)
        self.assertEqual(out["num_tasks"], 2)
        self.assertAlmostEqual(out["gate_mean"], float(g.mean()), places=10)
        self.assertAlmostEqual(out["gate_std"], float(g.std(unbiased=False)), places=10)
        self.assertAlmostEqual(out["gate_min"], float(g.min()), places=10)
        self.assertAlmostEqual(out["gate_max"], float(g.max()), places=10)
        w = terms["w"].double().squeeze(2)
        w_attn = terms["w_attn"].double()
        self.assertAlmostEqual(out["routing_l1_mean"], float((w - w_attn).abs().sum(dim=1).mean()),
                               places=10)
        self.assertAlmostEqual(out["w_col_mean"][0], float(w[:, 0].mean()), places=10)
        self.assertAlmostEqual(out["w_attn_col_mean"][1], float(w_attn[:, 1].mean()), places=10)
        self.assertAlmostEqual(out["attn_entropy_mean"], float(terms["attn_entropy"].double().mean()),
                               places=10)
        self.assertEqual(out["gate_feature_names"], list(AGC.FEATURE_NAMES))
        json.dumps(out)

    def test_stats_streaming_equals_single_batch(self):
        dnn_input, gen_rep, spec_reps, env_embs = tiny_reps(self.device, n=8)
        whole = AGC.RoutingGateStats()
        whole.update(self.head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs))
        split = AGC.RoutingGateStats()
        for start, stop in ((0, 3), (3, 8)):
            # env_embs 是逐源一维 [rep_dim]（无 batch 轴）⇒ 分批时保持原样，只切带 batch 轴的量
            split.update(self.head.gate_terms(dnn_input[start:stop], gen_rep[start:stop],
                                              [rep[start:stop] for rep in spec_reps],
                                              env_embs))
        self.assertEqual(split.result()["n_samples"], 8)                 # 样本级累加与分批方式无关
        for key in ("gate_mean", "gate_std", "routing_l1_mean", "attn_entropy_mean"):
            self.assertAlmostEqual(split.result()[key], whole.result()[key], places=12, msg=key)

    def test_empty_stats_are_zero_safe(self):
        out = AGC.RoutingGateStats().result()
        self.assertEqual(out["n_samples"], 0)
        self.assertEqual(out["gate_mean"], 0.0)
        self.assertEqual(out["gate_std"], 0.0)
        self.assertEqual(out["routing_l1_mean"], 0.0)
        self.assertEqual(out["w_col_mean"], [])
        json.dumps(out)

    def test_every_accumulator_starts_at_numeric_zero_not_none(self):
        """回归：`_entropy_sum` 曾被与两个极值哨兵写在同一行初始化为 `None` ⇒ 首次 `update()` 抛
        `TypeError: unsupported operand type(s) for +=: 'NoneType' and 'float'`（12 个用例因此连锁报错）。

        累加器清单的**唯一事实来源是 `update()` 的源码**：抓出全部 `self.<name> +=`，逐个要求它在
        `__init__` 之后是**数值 0**（不是 `None`）；任务形状的累加器由 `_init_tasks` 惰性创建，
        单独要求它建成全零张量。`_g_min/_g_max/_entropy_min/_entropy_max` 是**极值哨兵**（不是累加器），
        由 `update()` 里的 `is None` 分支惰性初始化，刻意不在本断言范围内。
        """
        accumulators = set(re.findall(r"self\.(\w+)\s*\+=",
                                      inspect.getsource(AGC.RoutingGateStats.update)))
        lazily_allocated = set(re.findall(r"self\.(\w+)\s*=",
                                          inspect.getsource(AGC.RoutingGateStats._init_tasks)))
        self.assertIn("_entropy_sum", accumulators)                       # 断言本身不得空转
        self.assertTrue(accumulators & lazily_allocated)                  # 确有惰性分配的张量累加器
        stats = AGC.RoutingGateStats()
        for name in sorted(accumulators - lazily_allocated):
            value = getattr(stats, name)
            self.assertIsNotNone(value, f"{name} 必须初始化为数值 0，不是 None")
            self.assertIsInstance(value, (int, float), name)
            self.assertNotIsInstance(value, bool, name)
            self.assertEqual(value, 0, name)
        stats._init_tasks(3)                              # 惰性分配的张量累加器：必须建成全零张量
        for name in sorted(accumulators & lazily_allocated):
            value = getattr(stats, name)
            self.assertIsInstance(value, torch.Tensor, name)
            self.assertGreater(value.numel(), 0, name)
            self.assertEqual(float(value.abs().sum()), 0.0, name)

    def test_inconsistent_num_tasks_raises(self):
        dnn_input, gen_rep, spec_reps, env_embs = self.reps
        stats = AGC.RoutingGateStats()
        stats.update(self.head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs))
        with self.assertRaises(ValueError):
            stats.update(self.head.gate_terms(dnn_input, gen_rep, spec_reps[:1], env_embs[:1]))

    def test_trace_records_gradients_and_readings_without_recomputing(self):
        trace = AGC.TrainGateTrace()
        trace.start_epoch(1)
        norms = []
        for _ in range(3):
            pred = self.head(*self.reps)
            loss = nn.BCELoss()(pred, torch.ones_like(pred))
            self.head.zero_grad()
            loss.backward()
            norms.append(trace.record_step(self.head))
            self.head.zero_grad()
        record = trace.end_epoch()
        self.assertEqual(record["epoch"], 1)
        self.assertEqual(record["steps"], 3)
        self.assertEqual(record["grad_nonzero_steps"], 3)
        self.assertAlmostEqual(record["grad_norm_mean"], sum(norms) / 3, places=12)
        self.assertAlmostEqual(record["grad_norm_max"], max(norms), places=12)
        self.assertAlmostEqual(record["grad_norm_min"], min(norms), places=12)
        self.assertGreater(record["grad_norm_min"], 0.0)
        self.assertIn("gate_std", record)                                 # 逐 epoch 门控读数
        result = trace.result()
        self.assertEqual(result["epochs"], 1)
        self.assertEqual(result["steps"], 3)
        self.assertEqual(len(result["per_epoch"]), 1)
        json.dumps(result)

    def test_gate_grad_norm_matches_manual_computation(self):
        pred = self.head(*self.reps)
        loss = nn.BCELoss()(pred, torch.ones_like(pred))
        self.head.zero_grad()
        loss.backward()
        expected = math.sqrt(sum(float(param.grad.detach().double().norm(2)) ** 2
                                for param in self.head.routing_gate_mlp.parameters()))
        self.assertAlmostEqual(AGC.gate_grad_norm(self.head), expected, places=12)
        self.head.zero_grad()
        self.assertEqual(AGC.gate_grad_norm(self.head), 0.0)              # 无梯度 ⇒ 0.0

    def test_trace_consumes_no_extra_rng(self):
        trace = AGC.TrainGateTrace()
        torch.manual_seed(17)
        trace.start_epoch(1)
        pred = self.head(*self.reps)
        loss = nn.BCELoss()(pred, torch.ones_like(pred))
        self.head.zero_grad()
        loss.backward()
        before = torch.get_rng_state()
        trace.record_step(self.head)
        trace.end_epoch()
        trace.result()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_trace_zero_gradient_forever_fires_the_degeneracy_rule(self):
        trace = AGC.TrainGateTrace()
        trace.start_epoch(1)
        for _ in range(2):
            self.head(*self.reps)                                         # forward 但不 backward ⇒ grad 全 None
            trace.record_step(self.head)
        trace.end_epoch()
        result = trace.result()
        self.assertEqual(result["grad_zero_steps"], 2)
        self.assertEqual(result["grad_norm_min"], 0.0)
        verdict = AGC.mechanism_verdict(gate_std=0.2, gate_mean=0.5, routing_l1_mean=0.03,
                                        gate_grad_norm_min_step=result["grad_norm_min"],
                                        train_eval_identical=True, rng_unchanged=True)
        self.assertEqual(verdict["status"], "MECHANISM_FAIL")

    def test_empty_trace_reports_zero_grad_norm(self):
        result = AGC.TrainGateTrace().result()
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["grad_norm_min"], 0.0)                    # 空轨迹按 0 处理（判据会命中）
        self.assertEqual(result["per_epoch"], [])

    def test_epoch_bookkeeping_and_guard_errors_are_explicit(self):
        trace = AGC.TrainGateTrace()
        with self.assertRaises(RuntimeError):
            trace.record_step(self.head)                                  # 未 start_epoch
        with self.assertRaises(RuntimeError):
            trace.end_epoch()
        trace.start_epoch(1)
        with self.assertRaises(RuntimeError):
            trace.start_epoch(2)                                          # 未 end_epoch
        with self.assertRaises(ValueError):
            trace.record_step(tiny_newtask())                             # 只接受处理臂头
        with self.assertRaises(RuntimeError):
            trace.record_step(tiny_corrected())                           # 未 forward ⇒ last_terms 为 None
        trace.end_epoch()


class TestDiagnostics(unittest.TestCase):
    """I8：val 诊断的读数正确、无副作用、不消耗全局 RNG、拒绝基线头。"""

    def test_readings_match_recomputation(self):
        backbone, head, loaders, _, _ = probe_setup()
        out = AGC.evaluate_routing_gate(head, backbone, loaders["val"], torch.device("cpu"))
        stats = AGC.RoutingGateStats()
        with torch.no_grad():
            for _, _, _, features in loaders["val"]:
                dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
                stats.update(head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs))
        expected = stats.result()
        self.assertEqual(out["n_samples"], 16)
        for key in ("gate_mean", "gate_std", "routing_l1_mean", "routing_l1_max", "attn_entropy_mean",
                    "cos_gen_attn_mean", "cos_gen_fw_mean"):
            self.assertAlmostEqual(out[key], expected[key], places=9, msg=key)
        self.assertEqual(out["mode"], "eval")
        json.dumps(out)

    def test_runtime_invariants_are_checked_and_true(self):
        backbone, head, loaders, _, _ = probe_setup()
        out = AGC.evaluate_routing_gate(head, backbone, loaders["val"], torch.device("cpu"))
        invariants = out["invariants"]
        self.assertTrue(invariants["train_eval_identical"])
        self.assertTrue(invariants["no_global_rng_consumed"])
        self.assertGreater(invariants["checked_batch_size"], 1)
        self.assertIn("cuda_rng_checked", invariants)

    def test_diagnostics_have_no_side_effects_and_consume_no_rng(self):
        backbone, head, loaders, _, _ = probe_setup()
        head.train()
        before = {name: param.detach().clone() for name, param in head.named_parameters()}
        torch.manual_seed(19)
        rng_before = torch.get_rng_state()
        out = AGC.evaluate_routing_gate(head, backbone, loaders["val"], torch.device("cpu"))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))   # loader 自带私有 generator
        self.assertTrue(head.training)                                    # 模式恢复
        for name, param in head.named_parameters():
            self.assertTrue(torch.equal(param.detach(), before[name]), name)
            self.assertIsNone(param.grad, name)
        self.assertIn("gate_mean", out)

    def test_diagnostics_reject_baseline_head(self):
        backbone, _, loaders, _, _ = probe_setup()
        with self.assertRaises(ValueError):
            AGC.evaluate_routing_gate(tiny_newtask(), backbone, loaders["val"], torch.device("cpu"))
        with self.assertRaises(ValueError):
            AGC.source_contribution_probe(tiny_newtask(), backbone, loaders["val"], torch.device("cpu"))

    def test_diagnostics_report_a_constant_gate_as_degenerate(self):
        backbone, head, loaders, _, _ = probe_setup()
        pin_gate(head, 40.0)                                              # g ≡ 1（纯 attention）
        out = AGC.evaluate_routing_gate(head, backbone, loaders["val"], torch.device("cpu"))
        self.assertAlmostEqual(out["gate_mean"], 1.0, places=9)
        self.assertAlmostEqual(out["gate_std"], 0.0, places=9)
        self.assertLess(out["routing_l1_mean"], AGC.ROUTING_L1_MIN)       # 与纯 attention 等价 ⇒ 判据命中
        verdict = AGC.mechanism_verdict(gate_std=out["gate_std"], gate_mean=out["gate_mean"],
                                        routing_l1_mean=out["routing_l1_mean"],
                                        gate_grad_norm_min_step=1e-6,
                                        train_eval_identical=out["invariants"]["train_eval_identical"],
                                        rng_unchanged=out["invariants"]["no_global_rng_consumed"])
        self.assertEqual(verdict["status"], "MECHANISM_FAIL")


class TestSourceContributionProbe(unittest.TestCase):
    """I8：L1O 边际效应、逐源依赖份额、gate↔效用对齐与逐源相关性的诊断口径。"""

    def test_leave_one_out_weights_k2_is_one_hot_on_the_other_source(self):
        w = torch.tensor([[[0.7], [0.3]]])
        w0, undefined0 = AGC.leave_one_out_weights(w, 0)
        self.assertTrue(torch.equal(w0, torch.tensor([[[0.0], [1.0]]])))
        self.assertFalse(bool(undefined0.any()))
        w1, _ = AGC.leave_one_out_weights(w, 1)
        self.assertTrue(torch.equal(w1, torch.tensor([[[1.0], [0.0]]])))

    def test_leave_one_out_renormalises_the_surviving_weights_k3(self):
        """L1O 规则：移除源 k 后，**幸存的原始权重**在其余源上重新归一化（其余源之和 > eps 时）。

        与 `uniform_routing` 是两件事：这里重新归一化的是**原始权重**，不是换成 `1/K`。布局约定 [B, K, 1]。
        """
        w = torch.tensor([[[0.5], [0.3], [0.2]]])                          # [B, K, 1]
        w1, undefined1 = AGC.leave_one_out_weights(w, 1)
        self.assertFalse(bool(undefined1[0]))
        self.assertEqual(tuple(w1.shape), (1, 3, 1))
        self.assertTrue(torch.allclose(w1, torch.tensor([[[0.5 / 0.7], [0.0], [0.2 / 0.7]]]), atol=1e-6))
        for removed in (0, 1, 2):                                          # K=3 的三个移除逐条核对
            w_k, undefined = AGC.leave_one_out_weights(w, removed)
            survivors = [j for j in range(3) if j != removed]
            denom = float(w[0, survivors, 0].sum())
            self.assertGreater(denom, AGC.L1O_EPS)
            self.assertFalse(bool(undefined[0]), removed)
            self.assertEqual(float(w_k[0, removed, 0]), 0.0, removed)      # 被移除的源恒为 0
            for j in survivors:
                self.assertAlmostEqual(float(w_k[0, j, 0]), float(w[0, j, 0]) / denom,
                                       places=6, msg=(removed, j))
            self.assertAlmostEqual(float(w_k.sum()), 1.0, places=6, msg=removed)

    def test_leave_one_out_undefined_returns_uniform_over_the_rest(self):
        """其余源权重之和 ≤ eps ⇒ 该反事实**无定义** ⇒ 回退为其余源上的**均匀**权重（不是重新归一化）。"""
        w = torch.tensor([[[1.0], [0.0]]])                                 # K=2：权重全在被移除的源上
        w0, undefined0 = AGC.leave_one_out_weights(w, 0)
        self.assertTrue(bool(undefined0[0]))
        self.assertTrue(torch.equal(w0, torch.tensor([[[0.0], [1.0]]])))   # K=2 时回退值与极限一致
        w3 = torch.tensor([[[1.0], [0.0], [0.0]]])                         # K=3：只有移除源 0 才无定义
        w0_3, undefined0_3 = AGC.leave_one_out_weights(w3, 0)
        self.assertTrue(bool(undefined0_3[0]))
        self.assertEqual(tuple(w0_3.shape), (1, 3, 1))
        self.assertTrue(torch.allclose(w0_3, torch.tensor([[[0.0], [0.5], [0.5]]]), atol=1e-7))
        self.assertFalse(bool(AGC.leave_one_out_weights(w3, 1)[1][0]))      # 幸存源 0 的权重非零 ⇒ 有定义
        zeros = torch.zeros(1, 3, 1)                                       # 全零 W ⇒ 三个移除都无定义
        for removed in (0, 1, 2):
            w_k, undefined = AGC.leave_one_out_weights(zeros, removed)
            survivors = [j for j in range(3) if j != removed]
            self.assertTrue(bool(undefined[0]), removed)
            self.assertEqual(float(w_k[0, removed, 0]), 0.0, removed)
            for j in survivors:                                            # 其余源**等分**
                self.assertAlmostEqual(float(w_k[0, j, 0]), 0.5, places=7, msg=(removed, j))
            self.assertAlmostEqual(float(w_k.sum()), 1.0, places=6, msg=removed)

    def test_leave_one_out_k1_has_no_survivor(self):
        """K=1：移除唯一源后**没有幸存源** ⇒ 同样按"无定义"处理，权重恒 0（没有可均匀的对象）。"""
        w = torch.tensor([[[1.0]]])
        w0, undefined = AGC.leave_one_out_weights(w, 0)
        self.assertTrue(bool(undefined[0]))
        self.assertEqual(tuple(w0.shape), (1, 1, 1))
        self.assertEqual(float(w0.sum()), 0.0)

    def test_probe_matches_hand_computed_marginal_effects(self):
        backbone, head, loaders, _, _ = probe_setup()
        out = AGC.source_contribution_probe(head, backbone, loaders["val"], torch.device("cpu"))
        full, minus = [], {0: [], 1: []}
        with torch.no_grad():
            for _, _, _, features in loaders["val"]:
                dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
                terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
                full.append(AGC.routing_pipeline(head, dnn_input, gen_rep, spec_reps, terms["w"]))
                for k in (0, 1):
                    w_k, _ = AGC.leave_one_out_weights(terms["w"], k)
                    minus[k].append(AGC.routing_pipeline(head, dnn_input, gen_rep, spec_reps, w_k))
        full = torch.cat(full)
        self.assertEqual(out["n_samples"], 16)
        for k in (0, 1):
            self.assertAlmostEqual(out["source_reliance_mean_abs_delta"][k],
                                   float((full - torch.cat(minus[k])).abs().mean()), places=6)
            self.assertAlmostEqual(out["source_reliance_mean_delta"][k],
                                   float((full - torch.cat(minus[k])).mean()), places=6)
        self.assertEqual(out["l1o_undefined_count"], 0)                    # 软 gate ⇒ 分母恒 > 0
        self.assertAlmostEqual(sum(out["source_reliance_share_mean"]), 1.0, places=6)
        json.dumps(out)

    def test_probe_prediction_deltas_match_hand_computation(self):
        backbone, head, loaders, _, _ = probe_setup()
        out = AGC.source_contribution_probe(head, backbone, loaders["val"], torch.device("cpu"))
        full, attn, fw = [], [], []
        with torch.no_grad():
            for _, _, _, features in loaders["val"]:
                dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
                terms = head.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
                full.append(AGC.routing_pipeline(head, dnn_input, gen_rep, spec_reps, terms["w"]))
                attn.append(AGC.routing_pipeline(head, dnn_input, gen_rep, spec_reps,
                                                 terms["w_attn"].unsqueeze(2)))
                fw.append(AGC.routing_pipeline(head, dnn_input, gen_rep, spec_reps,
                                               terms["w_fw"].unsqueeze(2)))
        self.assertAlmostEqual(out["pred_delta_vs_attn_mean_abs"],
                               float((torch.cat(full) - torch.cat(attn)).abs().mean()), places=6)
        self.assertAlmostEqual(out["pred_delta_vs_fw_mean_abs"],
                               float((torch.cat(full) - torch.cat(fw)).abs().mean()), places=6)

    def test_probe_reads_zero_prediction_delta_when_gate_is_pinned_to_an_endpoint(self):
        backbone, head, loaders, _, _ = probe_setup()
        pin_gate(head, 40.0)                                               # g ≡ 1
        out = AGC.source_contribution_probe(head, backbone, loaders["val"], torch.device("cpu"))
        self.assertEqual(out["pred_delta_vs_attn_mean_abs"], 0.0)          # W 逐位 = W_attn
        self.assertGreater(out["pred_delta_vs_fw_mean_abs"], 0.0)
        pin_gate(head, -40.0)                                              # g ≡ 0
        out = AGC.source_contribution_probe(head, backbone, loaders["val"], torch.device("cpu"))
        self.assertEqual(out["pred_delta_vs_fw_mean_abs"], 0.0)            # W 逐位 = W_fw

    def test_probe_reports_utility_alignment_and_diagnostic_only_per_source_correlation(self):
        backbone, head, loaders, _, _ = probe_setup()
        out = AGC.source_contribution_probe(head, backbone, loaders["val"], torch.device("cpu"))
        for key in ("gate_utility_spearman", "gate_utility_pearson", "gate_utility_auc",
                    "n_utility_nonzero", "n_utility_zero", "routing_weight_vs_reliance_pearson",
                    "routing_weight_vs_reliance_spearman"):
            self.assertIn(key, out)
        self.assertEqual(out["n_utility_nonzero"] + out["n_utility_zero"], out["n_samples"])
        self.assertEqual(out["per_source_correlation_status"], "diagnostic_only")
        self.assertIn("标量 gate", out["per_source_correlation_note"])      # 逐源偏好不可识别的理由
        self.assertIn("力学", out["per_source_correlation_note"])           # 权重→效应的机械耦合
        self.assertIn("BCE", out["utility_definition"])

    def test_probe_is_deterministic_and_consumes_no_global_rng(self):
        backbone, head, loaders, _, _ = probe_setup()
        torch.manual_seed(23)
        before = torch.get_rng_state()
        first = AGC.source_contribution_probe(head, backbone, loaders["val"], torch.device("cpu"))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        second = AGC.source_contribution_probe(head, backbone, loaders["val"], torch.device("cpu"))
        self.assertEqual(first, second)

    def test_probe_restores_mode_and_touches_no_grads(self):
        backbone, head, loaders, _, _ = probe_setup()
        head.train()
        AGC.source_contribution_probe(head, backbone, loaders["val"], torch.device("cpu"))
        self.assertTrue(head.training)
        self.assertTrue(all(param.grad is None for param in head.parameters()))
        self.assertTrue(all(param.grad is None for param in backbone.parameters()))

    def test_probe_respects_max_samples(self):
        backbone, head, loaders, _, _ = probe_setup()
        out = AGC.source_contribution_probe(head, backbone, loaders["val"], torch.device("cpu"),
                                            max_samples=8)
        self.assertEqual(out["n_samples"], 8)


class TestCorrelationSummary(unittest.TestCase):
    """I8：相关性工具的口径（退化输入 ⇒ None，不得返回 nan/inf）。"""

    def test_detects_planted_monotone_signal(self):
        gen = torch.Generator().manual_seed(0)
        x = torch.rand(256, generator=gen)
        y = 3.0 * x + 0.01 * torch.rand(256, generator=gen)
        out = AGC.correlation_summary(x, y)
        self.assertEqual(out["n"], 256)
        self.assertGreater(out["spearman"], 0.95)
        self.assertGreater(out["pearson"], 0.95)

    def test_returns_none_for_constant_input(self):
        out = AGC.correlation_summary(torch.ones(32), torch.arange(32).float())
        self.assertIsNone(out["pearson"])
        self.assertIsNone(out["spearman"])
        self.assertIsNone(AGC.correlation_summary(torch.arange(32).float(), torch.ones(32))["spearman"])

    def test_handles_ties_via_average_ranks(self):
        x = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        self.assertAlmostEqual(AGC.correlation_summary(x, x.clone())["spearman"], 1.0, places=9)

    def test_small_and_empty_inputs_are_safe(self):
        self.assertIsNone(AGC.correlation_summary(torch.zeros(1), torch.zeros(1))["spearman"])
        out = AGC.correlation_summary(torch.zeros(0), torch.zeros(0))
        self.assertIsNone(out["pearson"])
        self.assertEqual(out["n"], 0)


class TestRunnerIntegration(unittest.TestCase):
    """I10：接线（`--variant affinity_corrected` / `--affinity-corrected`）；基线臂不受影响。"""

    def test_treatment_arm_records_arm_block_trace_and_probe(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            out = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=2,
                                                  device=device, model=tiny_backbone(device),
                                                  loaders=loaders, stats=stats, indices=indices,
                                                  input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                                  variant=AGC.VARIANT)
            run_path = Path(out["run_dir"])
            self.assertTrue(out["run_id"].endswith(AGC.RUN_ID_SUFFIX), out["run_id"])

            config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["variant"], AGC.VARIANT)
            self.assertEqual(config["fusion_mode"], AGC.FUSION_MODE)
            self.assertEqual(config["gate_feature_dim"], AGC.GATE_FEATURE_DIM)

            payload = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["variant"], AGC.VARIANT)
            self.assertEqual(sorted(payload["mechanism"]),                      # M3/M4 键集不变
                             ["cos_gen_spec", "env_acc_stage1", "gate_mean", "gen_std"])
            arm = payload["affinity_corrected_arm"]
            self.assertEqual(arm["prereg"]["auc_test_min"], 0.8521)
            self.assertEqual(arm["prereg"]["baseline_test_auc"], 0.8500685307175756)
            self.assertIn(arm["mechanism"]["status"], ("MECHANISM_OK", "MECHANISM_FAIL"))
            self.assertIn(arm["alignment"]["status"],
                          ("ALIGNMENT_OK", "ALIGNMENT_FAIL", "ALIGNMENT_UNDEFINED"))
            self.assertIn(arm["status"], ("STOP_MECHANISM", "STOP_EFFECT", "PASS_ALIGNMENT_FAIL", "PASS"))
            self.assertEqual(arm["effect"]["evaluated"], arm["mechanism"]["status"] == "MECHANISM_OK")
            self.assertEqual(arm["effect"]["observed"], payload["stage2"]["test_auc"])
            trace = arm["train_gate_trace"]
            self.assertEqual([record["epoch"] for record in trace], [1, 2])
            self.assertGreater(trace[0]["steps"], 0)
            self.assertGreater(arm["gate_grad_norm"]["steps"], 0)
            self.assertIn("gate_std", arm["gate_grad_norm"])
            self.assertIn("failed_rules", arm["mechanism"])
            self.assertEqual(sorted(arm["mechanism"]["checks"]), sorted([
                "gate_std_ge_min", "gate_mean_in_window", "routing_l1_ge_min",
                "gate_grad_nonzero_every_step", "train_eval_identical", "no_global_rng_consumed"]))

            diagnostics = arm["diagnostics"]
            self.assertEqual(diagnostics["n_samples"], 16)                      # val 只跑一遍
            self.assertTrue(diagnostics["invariants"]["train_eval_identical"])  # 机制判据 5 的读数
            self.assertTrue(diagnostics["invariants"]["no_global_rng_consumed"])
            self.assertEqual(diagnostics["env_layout"], AGC.ENV_LAYOUT)
            contribution = arm["source_contribution"]
            self.assertEqual(contribution["n_samples"], 16)
            self.assertEqual(len(contribution["source_reliance_share_mean"]), 2)
            self.assertEqual(contribution["per_source_correlation_status"], "diagnostic_only")
            self.assertIn("gate_utility_spearman", contribution)
            self.assertIn("D1", arm["provenance"]["fixed_defects"])
            # 诊断/探针在 A1 取哈希之后运行 ⇒ 臂内补验冻结未被扰动
            self.assertEqual(arm["backbone_sha256_after_probes"], payload["backbone_sha256_after"])
            self.assertTrue(arm["grads_all_none_after_probes"])

            report = json.loads((run_path / "gate_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["A1"]["pass"])                              # 冻结三件套不受影响
            self.assertEqual(report["A3"]["status"], "on_demand")
            self.assertNotIn("affinity_corrected_arm", report)                 # 臂级判定不混入 A/B 门禁

            state = torch.load(run_path / "newtask.pt", map_location="cpu")
            tiny_corrected(device=device).load_state_dict(state)                # strict：门控权重已落盘
            self.assertIn("routing_gate_mlp.2.weight", state)

            summary = (root / "SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn(out["run_id"], summary)                              # 失败也必须留痕（协议 7.3）

    def test_baseline_arm_is_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            with mock.patch.object(AGC, "CorrectedAffinityGateNewTask",
                                   side_effect=AssertionError("基线臂不得构造修正门控头")):
                out = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=2,
                                                      device=device, model=tiny_backbone(device),
                                                      loaders=loaders, stats=stats, indices=indices,
                                                      input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM)
            run_path = Path(out["run_dir"])
            self.assertFalse(out["run_id"].endswith(AGC.RUN_ID_SUFFIX))
            self.assertIsNone(out["arm"])

            config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["variant"], AGC.BASELINE_VARIANT)
            self.assertNotIn("fusion_mode", config)                            # 处理臂专属字段不得出现
            self.assertEqual(sorted(config),
                             ["batch_size", "commit", "env_seed", "epochs", "frozen", "input_size", "lr",
                              "model_seed", "patience", "rep_dim", "run_id", "split_seed", "stage1_id",
                              "tag", "variant"])

            payload = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["variant"], AGC.BASELINE_VARIANT)
            self.assertNotIn("affinity_corrected_arm", payload)
            self.assertNotIn("routing_gate_mlp", json.dumps(payload))
            self.assertEqual(sorted(payload),
                             ["backbone_sha256_after", "backbone_sha256_before", "commit",
                              "env_ids_sha256", "mechanism", "run_id", "split_sha256", "stage1",
                              "stage1_id", "stage2", "variant"])
            self.assertEqual(sorted(payload["mechanism"]),
                             ["cos_gen_spec", "env_acc_stage1", "gate_mean", "gen_std"])
            state = torch.load(run_path / "newtask.pt", map_location="cpu")
            tiny_newtask(device=device).load_state_dict(state)                  # strict 载入基线头

    def test_cli_exposes_variant_and_affinity_corrected_alias(self):
        parser = run_census_benchmark.build_parser()
        args = parser.parse_args(["stage2", "--stage1-dir", "x"])
        self.assertEqual(args.variant, AGC.BASELINE_VARIANT)
        args = parser.parse_args(["stage2", "--stage1-dir", "x", "--variant", AGC.VARIANT])
        self.assertEqual(args.variant, AGC.VARIANT)
        args = parser.parse_args(["stage2", "--stage1-dir", "x", "--affinity-corrected"])
        self.assertEqual(args.variant, AGC.VARIANT)
        with self.assertRaises(SystemExit):
            parser.parse_args(["stage2", "--stage1-dir", "x", "--variant", "nope"])


if __name__ == "__main__":
    unittest.main()
