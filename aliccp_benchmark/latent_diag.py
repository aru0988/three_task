"""AliCCP 潜在跨任务预测诊断（唯一事实来源：docs/superpowers/specs/2026-10-08-aliccp-latent-cross-task-diagnostic-design.md）。

问题：冻结 Stage-1 的 CTR/CVR 融合预测，是否携带超出单独训练 BSI 头自身预测的 BSI 排序增量信息？
本模块是信息诊断（非模型改进），协议全部由预注册文档冻结，不得按结果调整。

协议要点：
- A/B/C = 训练文件前 2M 行的连续区段：A 训练唯一 BSI 头（5 epoch 上限、patience 2、batch 2000、
  Adam 1e-4、官方 dev BSI AUC 选点）；B 拟合三个探针；C 机制外推检查；dev/test 为官方切分。
  三个探针全部冻结前本诊断不读 test CSV、不建 test 数据集/DataLoader、不扫描 test 标签；冻结后、
  test 评估前先执行完整指纹校验（含 test 源文件），随后 test 的唯一一次评估触碰（标签扫描 →
  数据集/DataLoader 构建 → 推理）全部发生在该点之后。
- 三探针共享同一 BSI 头与同一冻结 Stage-1；标准化均值/尺度只用 B 计算（ddof=0）：
  base=[bsi]；full=[bsi,ctr,cvr]；shuffle=[bsi,ctr[perm],cvr[perm]]。
- shuffle 置换：numpy.random.default_rng(20261008)，按固定顺序 (B,C,dev,test) 每切分独立
  rng.permutation(n)；同一排列同时作用于 CTR/CVR（联合置换），破坏行级对齐、保留边际分布与参数个数。
- ΔS = AUC(full) − max(AUC(base), AUC(shuffle))；正向信号要求 Δtest ≥ +0.001、ΔC ≥ +0.001、
  Δdev ≥ 0、B/C/test 上旧任务预测非常数、且全部身份/冻结/数据门禁 PASS。
- 拟合前中止（NO-GO，只判定 A/B/C/dev）：身份/指纹/检查点不匹配、A/B/C/dev 任一切分 BSI 负例
  < 100、B/C/dev 上任一切分两个旧任务预测方差同时 < 1e-8（"both old tasks" 的合取解读）、
  冻结不变量失败。test 的负例下限不参与拟合前判定，随最终门禁 D1_counts 在三探针冻结后判定。
- 指纹身份校验分两段，保证正式模式拟合前不触碰 test 源文件：拟合前仅执行 provisional 校验——
  指纹文件自哈希 + train/val 文件的前缀字节/表头/标签计数与预注册指纹逐项比对（复用协议函数，
  不调用会读取 test 的 verify_fingerprint/ensure_fingerprint）；三探针全部冻结后、test 评估前
  执行完整 protocol.verify_fingerprint(fp)（含 test 逐项比对），失败则中止且不产生任何 test
  指标。smoke（enforce=False，合成数据）豁免：ensure_fingerprint 在拟合前构建/校验完整指纹。
- 正式运行（enforce=True）额外要求：(model_seed, stage1_id) 命中冻结白名单
  FORMAL_SEED_STAGE1_PAIRS（不含 seed1）、Stage-1 meta 种子与运行 model_seed 一致、指纹命中
  正式常量、干净 git 树（dirty=False）。

不修改 multitaskrec/*；复用 aliccp_benchmark.protocol 与 bench 的纯工具（种子/指纹/冻结/哈希）。
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec, NewTask

from . import bench, metrics, protocol

# ---- 区段与预算（预注册冻结：A/B/C 连续区段，合计 == TRAIN_BUDGET）----
A_ROWS, B_ROWS, C_ROWS = 1_400_000, 300_000, 300_000
SMOKE_SIZES = {"A": 41, "B": 25, "C": 17, "dev": 13, "test": 11}  # batch=8 下余 1，覆盖单例

# ---- 探针与变换（冻结超参；无网格搜索）----
SHUFFLE_SEED = 20261008
CLIP_EPS = 1e-6
SCALE_FLOOR = 1e-12
PROBE_C = 1.0
PROBE_PENALTY = "l2"
PROBE_SOLVER = "lbfgs"
PROBE_MAX_ITER = 200
ARMS = ("base", "full", "shuffle")
ARM_FEATURES = {"base": ("bsi",), "full": ("bsi", "ctr", "cvr"), "shuffle": ("bsi", "ctr", "cvr")}
ALL_SPLITS = ("B", "C", "dev", "test")
PERM_ORDER = ("B", "C", "dev", "test")
PREFIT_SPLITS = ("A", "B", "C", "dev")  # 拟合前中止只判定这些切分；test 冻结后才触碰

# ---- 中止与判定阈值（看到结果前冻结）----
MIN_BSI_NEGATIVES = 100
VAR_FLOOR = 1e-8
DELTA_TEST_MIN = 0.001
DELTA_C_MIN = 0.001
DELTA_DEV_MIN = 0.0
CLASS_POS_MIN = 0.001
CLASS_NEG_MAX = -0.02

# ---- BSI 头超参（与公平 Stage-2 short 一致）----
HEAD_EPOCHS = protocol.STAGE2_EPOCHS
HEAD_PATIENCE = protocol.STAGE2_PATIENCE
HEAD_LR = protocol.LR

# ---- 正式运行的冻结身份（预注册正文；smoke 不强制）----
# 扩展分支（seeds2–3）：正式运行只认这些 (model_seed → Stage-1 ID) 冻结配对；不含 seed1——
# formal CLI --model-seed 无默认值，禁止在本分支静默重跑 seed1（见 resolve_formal_stage1）。
FORMAL_SEED_STAGE1_PAIRS = {
    1688723740: "s1-5c060b9c-m1688723740-e3-4e1b5c6f",
    1688738016: "s1-5c060b9c-m1688738016-e3-47619ce0",
}
FORMAL_FINGERPRINT_SHA256 = (
    "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
)

# ---- 产物 ----（*.npz/*.json/*.pt 均被 .gitignore 拦截；仅 SUMMARY 入库）
RAW_FILES = ("raw.npz", "perms.npz", "probe_probs.npz", "probes.json", "bsi_head.pt")
SUMMARY_FILE = "LATENT_DIAG_SUMMARY.md"
LATENT_DIAG_SUMMARY_COLUMNS = [
    "run_id", "commit", "tag",
    "auc_base_test", "auc_full_test", "auc_shuffle_test",
    "auc_base_C", "auc_full_C", "auc_shuffle_C",
    "delta_test", "delta_C", "delta_dev",
    "classification", "verdict", "stage1_id", "raw_sha256",
]

# ---- smoke 合成数据（真实表头；特征值按 vocab 取模，确定性）----
SMOKE_HEADER = (
    "click,purchase,101,121,122,124,125,126,127,128,129,205,206,207,216,508,509,702,853,301"
)
SMOKE_FEATURE_SIZES = (
    98, 14, 3, 8, 4, 4, 3, 5, 467298, 6929, 263942, 106399, 5888, 104830, 51878, 37148,
)


def split_ranges(a_rows: int = A_ROWS, b_rows: int = B_ROWS, c_rows: int = C_ROWS) -> dict:
    """A/B/C 连续区段（文件顺序；含 A 起点 0）。"""
    for name, n in (("A", a_rows), ("B", b_rows), ("C", c_rows)):
        if int(n) < 1:
            raise ValueError(f"区段行数必须为正整数: {name}={n}")
    return {
        "A": (0, int(a_rows)),
        "B": (int(a_rows), int(a_rows) + int(b_rows)),
        "C": (int(a_rows) + int(b_rows), int(a_rows) + int(b_rows) + int(c_rows)),
    }


def prefix_tag_for(train_budget: int, val_budget: int, test_budget: int) -> str:
    """与公平基准一致的前缀标识规则（正式预算 → 冻结 PREFIX_TAG）。"""
    if (train_budget, val_budget, test_budget) == (
        protocol.TRAIN_BUDGET, protocol.VAL_BUDGET, protocol.TEST_BUDGET,
    ):
        return protocol.PREFIX_TAG
    return f"p{train_budget}-v{val_budget}-t{test_budget}"


def resolve_formal_stage1(model_seed: int, stage1_id=None) -> str:
    """正式运行 (model_seed, stage1_id) 冻结配对的唯一解析入口（formal CLI 使用）。

    seed 必须命中 FORMAL_SEED_STAGE1_PAIRS；stage1_id 省略（None）时严格按白名单派生，
    显式提供时必须与白名单逐字一致。任何其他组合抛 ValueError——正式运行绝不静默回退到
    白名单之外的 Stage-1（含 seed1），杜绝"formal 无参默认重跑 seed1"。
    """
    expected = FORMAL_SEED_STAGE1_PAIRS.get(int(model_seed))
    if expected is None:
        raise ValueError(
            f"model_seed {int(model_seed)} 不在正式冻结白名单中: {sorted(FORMAL_SEED_STAGE1_PAIRS)}"
        )
    if stage1_id is None:
        return expected
    if stage1_id != expected:
        raise ValueError(
            f"stage1_id 与冻结白名单不匹配: model_seed={int(model_seed)} "
            f"expected={expected} got={stage1_id}"
        )
    return stage1_id


# ---- 拟合前指纹身份（provisional：自哈希 + 仅 train/val；完整校验在三探针冻结后）----
PROVISIONAL_FP_TAGS = ("train", "val")


def provisional_fingerprint_checks(fp: dict, tags=PROVISIONAL_FP_TAGS) -> dict:
    """拟合前（三探针冻结前）允许执行的指纹身份校验：自哈希 + 指定源文件的前缀字节/表头/标签计数
    与已冻结指纹逐项比对（复用既有 protocol 函数，不修改协议）。

    默认只覆盖 train/val：预注册要求三探针全部冻结前不读 test CSV、不扫描 test 标签，而既有
    protocol.verify_fingerprint 会对三个文件（含 test）重读前缀字节并扫描标签（protocol.py:249-264）
    → 正式模式拟合前不得调用它；test 部分的比对由完整校验在三探针冻结后、test 评估前执行。
    逐项布尔结果返回给 identity_checks 与日志；文件不可读/行数不足记为该文件校验失败，不向上抛。
    """
    out = {
        "scope": list(tags),
        "self_hash": bool(fp.get("fingerprint_sha256") == protocol.fingerprint_digest(fp)),
        "files": {},
    }
    for tag in tags:
        meta = fp["files"][tag]
        budget = fp["budgets"][tag]
        try:
            prefix_ok = protocol.prefix_sha256(meta["path"], budget) == meta["prefix_sha256"]
            counts_ok = protocol._normalize_counts(
                protocol.scan_label_counts(meta["path"], budget)
            ) == protocol._normalize_counts(fp["label_counts"][tag])
            header_ok = protocol._header_sha256(meta["path"]) == fp["header_sha256"]
            error = None
        except (ValueError, OSError) as exc:
            prefix_ok = counts_ok = header_ok = False
            error = f"{type(exc).__name__}: {exc}"
        out["files"][tag] = {
            "prefix_sha256": bool(prefix_ok),
            "label_counts": bool(counts_ok),
            "header_sha256": bool(header_ok),
            "error": error,
        }
    out["passed"] = out["self_hash"] and all(
        detail["prefix_sha256"] and detail["label_counts"] and detail["header_sha256"]
        for detail in out["files"].values()
    )
    return out


# ---- 纯变换（验证器以独立副本重实现同一格式；改动此处 = 协议改动）----
def clip_logit(p) -> np.ndarray:
    """裁剪到 [eps, 1-eps] 后做 logit 变换（float64）。"""
    p = np.asarray(p, dtype=np.float64)
    c = np.clip(p, CLIP_EPS, 1.0 - CLIP_EPS)
    return np.log(c / (1.0 - c))


def standardizer_stats(cols: dict) -> dict:
    """B 上的标准化统计（ddof=0）；近似常数（scale < SCALE_FLOOR）缩放置 1 并标记。"""
    means, scales, raw_scales, constant = {}, {}, {}, {}
    for name, col in cols.items():
        col = np.asarray(col, dtype=np.float64)
        mean = float(np.mean(col))
        scale = float(np.std(col))
        means[name] = mean
        raw_scales[name] = scale
        if scale < SCALE_FLOOR:
            scales[name] = 1.0
            constant[name] = True
        else:
            scales[name] = scale
            constant[name] = False
    return {"means": means, "scales": scales, "raw_scales": raw_scales, "constant": constant}


def apply_standardizer(cols: dict, stats: dict) -> dict:
    out = {}
    for name, col in cols.items():
        col = np.asarray(col, dtype=np.float64)
        out[name] = (col - stats["means"][name]) / stats["scales"][name]
    return out


def make_permutations(lengths: dict, seed: int = SHUFFLE_SEED) -> dict:
    """固定顺序 (B,C,dev,test) 下每切分独立 rng.permutation(n)；确定性、可独立复现。"""
    rng = np.random.default_rng(seed)
    return {split: rng.permutation(int(lengths[split])) for split in PERM_ORDER}


def arm_columns(logits: dict, perms: dict, arm: str, split: str) -> dict:
    """预注册冻结的臂-列映射（唯一臂语义实现）：
    base=[bsi]；full=[bsi,ctr,cvr]（行对齐原列）；shuffle=[bsi,ctr[perm],cvr[perm]]。

    只有 shuffle 臂重排行序（同一联合置换同时作用于 CTR/CVR；BSI 不置换）；full 臂必须
    保持逐行对齐。历史缺陷（2026-10-08 审计）：置换曾被同时应用到 full 与 shuffle →
    两臂输入逐位相同，正式输出 full=shuffle 全位数一致、探针系数相同（该运行已作废）。
    """
    cols = {"bsi": logits[split]["bsi"]}
    if arm == "base":
        return cols
    if arm == "shuffle":
        perm = perms[split]
        cols["ctr"] = logits[split]["ctr"][perm]
        cols["cvr"] = logits[split]["cvr"][perm]
        return cols
    if arm == "full":
        cols["ctr"] = logits[split]["ctr"]
        cols["cvr"] = logits[split]["cvr"]
        return cols
    raise ValueError(f"未知臂: {arm}")


def array_sha256(arr) -> str:
    """数组内容哈希：sha256("{dtype}:{shape}:" + bytes)。"""
    arr = np.ascontiguousarray(arr)
    header = f"{arr.dtype}:{list(arr.shape)}:".encode()
    return hashlib.sha256(header + arr.tobytes()).hexdigest()


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifacts_sha256(run_path, names=RAW_FILES) -> str:
    """原始产物身份：按固定名字顺序 sha256("name:file_sha\\n") 链式哈希（缺失文件跳过）。"""
    digest = hashlib.sha256()
    for name in names:
        path = Path(run_path) / name
        if not path.exists():
            continue
        digest.update(f"{name}:{_file_sha256(path)}\n".encode())
    return digest.hexdigest()


def split_range_hashes(path, ranges: dict) -> dict:
    """区段原始字节哈希：sha256("range:{start}:{end}\\n" + 表头 + 区段数据行)。"""
    if not ranges:
        return {}
    max_end = max(int(end) for _, end in ranges.values())
    digests = {
        name: hashlib.sha256(f"range:{int(start)}:{int(end)}\n".encode())
        for name, (start, end) in ranges.items()
    }
    with open(path, "rb") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"文件为空: {path}")
        for digest in digests.values():
            digest.update(header)
        for i in range(max_end):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} 数据行不足 {max_end}（在第 {i + 1} 行截断）")
            for name, (start, end) in ranges.items():
                if int(start) <= i < int(end):
                    digests[name].update(line)
    return {name: digest.hexdigest() for name, digest in digests.items()}


def range_label_counts(path, ranges: dict) -> dict:
    """区段原始扫描标签：n / click1 / purchase1 / bsi_pos / bsi_raw（与加载器无关）。"""
    if not ranges:
        return {}
    max_end = max(int(end) for _, end in ranges.values())
    counts = {
        name: {"n": 0, "click1": 0, "purchase1": 0, "bsi_pos": 0, "bsi_raw": {}}
        for name in ranges
    }
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i in range(max_end):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} 数据行不足 {max_end}（在第 {i + 1} 行截断）")
            for name, (start, end) in ranges.items():
                if int(start) <= i < int(end):
                    parts = line.strip().split(",")
                    click, purchase, raw = int(parts[0]), int(parts[1]), int(parts[-1])
                    entry = counts[name]
                    entry["n"] += 1
                    entry["click1"] += click
                    entry["purchase1"] += purchase
                    entry["bsi_pos"] += 0 if raw == 2 else 1
                    entry["bsi_raw"][raw] = entry["bsi_raw"].get(raw, 0) + 1
    return counts


def norm_counts(counts: dict) -> dict:
    """JSON 化（bsi_raw 键归一为 str；int 显式转换）。"""
    return {
        "n": int(counts["n"]),
        "click1": int(counts["click1"]),
        "purchase1": int(counts["purchase1"]),
        "bsi_pos": int(counts["bsi_pos"]),
        "bsi_raw": {str(k): int(v) for k, v in sorted(counts["bsi_raw"].items())},
    }


def corr_of(a, b):
    """Pearson 相关；任一侧标准差为 0 → None（非常数判断在下游单独给出）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa == 0.0 or sb == 0.0:
        return None
    cov = float(np.mean((a - np.mean(a)) * (b - np.mean(b))))
    return cov / (sa * sb)


# ---- 探针（sklearn LogisticRegression；概率由 coef/intercept 的 sigmoid 重算，保证可独立复现）----
def fit_probe(x_std: np.ndarray, y: np.ndarray) -> dict:
    clf = LogisticRegression(
        C=PROBE_C, penalty=PROBE_PENALTY, solver=PROBE_SOLVER, max_iter=PROBE_MAX_ITER
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(x_std, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return {
        "coef": [float(v) for v in np.ravel(clf.coef_)],
        "intercept": float(np.ravel(clf.intercept_)[0]),
        "converged": bool(converged),
        "n_iter": int(np.ravel(clf.n_iter_)[0]),
    }


def probe_prob(x_std: np.ndarray, probe: dict) -> np.ndarray:
    z = np.asarray(x_std, dtype=np.float64) @ np.asarray(probe["coef"], dtype=np.float64)
    z = z + float(probe["intercept"])
    return 1.0 / (1.0 + np.exp(-z))


def delta_of(aucs: dict) -> float:
    """Δ = AUC(full) − max(AUC(base), AUC(shuffle))。"""
    return float(aucs["full"]) - max(float(aucs["base"]), float(aucs["shuffle"]))


def classify_delta(delta: float) -> str:
    if delta >= CLASS_POS_MIN:
        return "POSITIVE"
    if delta <= CLASS_NEG_MAX:
        return "CLEAR_DECLINE"
    return "NO_CLEAR"


def evaluate_gates(facts: dict) -> dict:
    """门禁判定（纯函数）。signal_positive 要求全部门禁 PASS；否则 NO_GO_FOR_EXTENSION。"""

    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    variances = facts["variances"]
    nonconst_ok = all(
        float(variances[split][task]) > 0.0
        for split in ("B", "C", "test")
        for task in ("ctr", "cvr")
    )
    deltas = facts["deltas"]
    gates = {
        "I1_identity": {"verdict": verdict(facts["identity_ok"]), "detail": str(facts.get("identity_detail", ""))},
        "I2_freeze": {"verdict": verdict(facts["freeze_ok"]), "detail": str(facts.get("freeze_detail", ""))},
        "D1_counts": {"verdict": verdict(facts["counts_ok"]), "detail": str(facts.get("counts_detail", ""))},
        "D2_ranges": {"verdict": verdict(facts["ranges_ok"]), "detail": str(facts.get("ranges_detail", ""))},
        "P1_perm": {"verdict": verdict(facts["perms_ok"]), "detail": str(facts.get("perms_detail", ""))},
        "P2_nonconst": {
            "verdict": verdict(nonconst_ok),
            "detail": f"B/C/test 上 ctr、cvr 方差必须 > 0；variances={variances}",
        },
        "S1_delta_test": {
            "verdict": verdict(float(deltas["test"]) >= DELTA_TEST_MIN),
            "detail": f"delta_test={float(deltas['test']):.6f} >= {DELTA_TEST_MIN}",
        },
        "S2_delta_C": {
            "verdict": verdict(float(deltas["C"]) >= DELTA_C_MIN),
            "detail": f"delta_C={float(deltas['C']):.6f} >= {DELTA_C_MIN}",
        },
        "S3_delta_dev": {
            "verdict": verdict(float(deltas["dev"]) >= DELTA_DEV_MIN),
            "detail": f"delta_dev={float(deltas['dev']):.6f} >= {DELTA_DEV_MIN}",
        },
        "G1_git": {
            "verdict": verdict(facts["git_dirty"] is False),
            "detail": f"git.dirty={facts['git_dirty']}（正式运行必须为 False）",
        },
    }
    signal_positive = all(gate["verdict"] == "PASS" for gate in gates.values())
    return {
        "gates": gates,
        "signal_positive": bool(signal_positive),
        "verdict": "SIGNAL_POSITIVE" if signal_positive else "NO_GO_FOR_EXTENSION",
    }


def prefit_abort_reasons(facts: dict) -> list:
    """拟合前 NO-GO 中止原因（预注册冻结；空列表 = 继续拟合）。

    只判定 A/B/C/dev：三探针全部冻结前不触碰 test（预注册），因此 test 的负例下限、
    长度与加载器一致性不在此判定（test 负例下限随最终门禁 D1_counts 在冻结后判定）。
    facts 中即使携带 test 条目（negatives / loader_parity / length_mismatches），
    也不得在此产生中止原因。
    """
    reasons = []
    if not facts["identity_ok"]:
        reasons.append({"code": "IDENTITY_MISMATCH", "detail": facts.get("identity_detail", {})})
    if not facts["freeze_ok"]:
        reasons.append({"code": "FREEZE_FAIL", "detail": facts.get("freeze_detail", {})})
    for split in PREFIT_SPLITS:
        actual = int(facts["negatives"][split])
        if actual < MIN_BSI_NEGATIVES:
            reasons.append(
                {"code": "MIN_NEGATIVES", "split": split, "actual": actual, "required": MIN_BSI_NEGATIVES}
            )
    for split in ("B", "C", "dev"):  # test 未触碰：方差检查不含 test
        var = facts["old_task_var"][split]
        if float(var["ctr"]) < VAR_FLOOR and float(var["cvr"]) < VAR_FLOOR:
            reasons.append(
                {
                    "code": "CONSTANT_OLD_TASK_PREDS",
                    "split": split,
                    "var_ctr": float(var["ctr"]),
                    "var_cvr": float(var["cvr"]),
                }
            )
    for mismatch in facts.get("length_mismatches", []):
        if mismatch.get("split") in PREFIT_SPLITS:
            reasons.append({"code": "SPLIT_LENGTH_MISMATCH", **mismatch})
    loader_parity = facts.get("loader_parity", {})
    for split in PREFIT_SPLITS:
        if split in loader_parity and not loader_parity[split]:
            reasons.append({"code": "LOADER_SCAN_MISMATCH", "split": split})
    return reasons


def assert_clean_git_for_formal(git_state: dict) -> None:
    if git_state.get("dirty") is not False:
        raise SystemExit(
            f"正式运行要求干净 git 树（git.dirty={git_state.get('dirty')}）；先提交或清理后重跑"
        )


class RangeView(Dataset):
    """按文件顺序的区段视图（不复制底层数据）；越界抛 IndexError。"""

    def __init__(self, base, start: int, end: int):
        start, end = int(start), int(end)
        if start < 0 or end > len(base) or start >= end:
            raise ValueError(f"非法区段: [{start},{end}) 越界或为空（len={len(base)}）")
        self.base, self.start, self.end = base, start, end

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        return self.base[self.start + idx]


# ---- smoke 合成数据（真实表头 + 真实 vocab 范围；确定性）----
def make_smoke_rows(n: int, start: int = 0) -> list:
    rows = []
    for i in range(int(start), int(start) + int(n)):
        row = [i % 2, 1 if i % 10 == 4 else 0, i % 97]
        row += [(i * (k + 1)) % size for k, size in enumerate(SMOKE_FEATURE_SIZES)]
        row += [2 if i % 5 == 0 else (1 if i % 3 == 0 else 3)]
        rows.append(row)
    return rows


def write_aliccp_file(path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(SMOKE_HEADER + "\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")


# ---- 推理收集（no_grad；batch 单例安全：复制-丢弃 + 一律 reshape(-1)）----
def head_forward_batch_safe(head, dnn_input, gen_rep, spec_reps, env_embs):
    """runner 侧单例批次适配（核心模型不可改，仅在 runner 内处理）。

    NewTask.forward 内部多处 squeeze() 在 batch=1 时会折叠 batch 维，导致 model.py:926 的
    torch.stack(dim=2) 越界（"Dimension out of range"）。这里把单例复制为 2 行前向后
    **仅回传原始行输出**：样本预算不变；训练损失只按原始行/原始标签计算（复制行输出被丢弃，
    权重不放大——复制行输出即使进损失，均值归约下 (l+l)/2=l 亦等价，但切片后保证无歧义）；
    非单例批次直通，行为逐位不变。
    """
    if int(dnn_input.shape[0]) != 1:
        return head(dnn_input, gen_rep, spec_reps, env_embs)
    out = head(
        dnn_input.repeat(2, 1),
        gen_rep.repeat(2, 1),
        [rep.repeat(2, 1) for rep in spec_reps],
        env_embs,  # env_embs 每项形状 (rep_dim,)，与 batch 无关
    )
    return out[:1]


@torch.no_grad()
def collect_split_probs(model, head, loader, device) -> dict:
    """逐行收集：BSI 概率（头）+ CTR/CVR 融合概率（冻结 Stage-1）+ 三标签 + 加载器计数。"""
    model.eval()
    head.eval()
    ys, clicks, purchases, bsi_l, ctr_l, cvr_l = [], [], [], [], [], []
    for click, purchase, y, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        out = model(features)
        ctr = out["fused_preds"][0].reshape(-1).detach().to("cpu", torch.float32)
        cvr = out["fused_preds"][1].reshape(-1).detach().to("cpu", torch.float32)
        dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
        bsi = (
            head_forward_batch_safe(head, dnn_input, gen_rep, spec_reps, env_embs)
            .reshape(-1)
            .detach()
            .to("cpu", torch.float32)
        )
        ys.append(y.reshape(-1))
        clicks.append(click.reshape(-1))
        purchases.append(purchase.reshape(-1))
        bsi_l.append(bsi)
        ctr_l.append(ctr)
        cvr_l.append(cvr)
    y_arr = torch.cat(ys).numpy().astype(np.int8)
    click_arr = torch.cat(clicks).numpy().astype(np.int8)
    purchase_arr = torch.cat(purchases).numpy().astype(np.int8)
    return {
        "y": y_arr,
        "click": click_arr,
        "purchase": purchase_arr,
        "bsi_prob": torch.cat(bsi_l).numpy().astype(np.float32),
        "ctr_prob": torch.cat(ctr_l).numpy().astype(np.float32),
        "cvr_prob": torch.cat(cvr_l).numpy().astype(np.float32),
        "n": int(len(y_arr)),
        "click1": int(click_arr.sum()),
        "purchase1": int(purchase_arr.sum()),
        "bsi_pos": int(y_arr.sum()),
    }


@torch.no_grad()
def _head_auc(model, head, loader, device) -> float:
    model.eval()
    head.eval()
    ys, ps = [], []
    for _, _, y, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
        ps.append(head_forward_batch_safe(head, dnn_input, gen_rep, spec_reps, env_embs).reshape(-1).detach().cpu())
        ys.append(y.reshape(-1))
    return metrics.auc_score(torch.cat(ys), torch.cat(ps))


def train_bsi_head(
    model, head, a_loader, dev_loader, *, device, epochs=HEAD_EPOCHS, patience=HEAD_PATIENCE,
    lr=HEAD_LR, log=print,
) -> dict:
    """唯一 BSI 头：只在 A 段训练，按官方 dev BSI AUC 选点（与公平 Stage-2 short 同口径）。"""
    optimizer = torch.optim.Adam(params=head.parameters(), lr=lr)
    loss_func = torch.nn.BCELoss()
    best_auc, best_epoch, best_weight, earlystop_count, early_stopped = 0.0, None, None, 0, False
    fallback_used = False
    per_epoch = []
    for epoch in range(1, int(epochs) + 1):
        head.train()
        loss_sum, steps = 0.0, 0
        for _, _, y, features in a_loader:
            for key in features:
                features[key] = features[key].to(device)
            with torch.no_grad():  # 计算图级冻结
                dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
            # 单例批次：helper 只回传原始行输出 → loss 只按原始标签计，权重不放大（样本预算不变）
            pred = head_forward_batch_safe(head, dnn_input, gen_rep, spec_reps, env_embs)
            loss = loss_func(pred.reshape(-1).cpu(), y.float().reshape(-1)) + head.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss)
            steps += 1
        val_auc = _head_auc(model, head, dev_loader, device)
        per_epoch.append({"epoch": epoch, "train_loss": loss_sum / max(1, steps), "val_auc": float(val_auc)})
        log(f"[latdiag] head Epoch:{epoch} train_loss={loss_sum / max(1, steps):.4f} AUC-Val-BSI:{val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, best_epoch, earlystop_count = float(val_auc), epoch, 0
            best_weight = copy.deepcopy(head.state_dict())
        else:
            earlystop_count += 1
            log(f"[latdiag] head EarlyStopping count {earlystop_count}")
            if earlystop_count == patience:
                early_stopped = True
                log(f"[latdiag] head EarlyStopping at epoch {epoch}")
                break
    if best_weight is None:
        # 退化保护：全部 epoch 的 dev AUC 均为 0（与公平循环的 > 0 选点兼容的兜底）；
        # 该情形会在 reported.head.fallback_used 中如实记录，正式解读时必须披露。
        fallback_used = True
        best_epoch = len(per_epoch)
        best_weight = copy.deepcopy(head.state_dict())
        log("[latdiag] 警告：dev BSI AUC 全程为 0，回退到最后一个 epoch 的权重（记录 fallback_used=True）")
    head.load_state_dict(best_weight)
    return {
        "per_epoch": per_epoch,
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_auc),
        "early_stopped": bool(early_stopped),
        "fallback_used": bool(fallback_used),
    }


def _versions() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}


def _write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_diagnostic(
    *,
    root,
    stage1_id: str,
    data_files: dict,
    a_rows: int = A_ROWS,
    b_rows: int = B_ROWS,
    c_rows: int = C_ROWS,
    val_budget: int = protocol.VAL_BUDGET,
    test_budget: int = protocol.TEST_BUDGET,
    model_seed: int = protocol.MODEL_SEED,
    device,
    enforce: bool,
    require_clean_git=None,
    tag=None,
    batch_size: int = protocol.BATCH_SIZE,
    epochs: int = HEAD_EPOCHS,
    patience: int = HEAD_PATIENCE,
    run_id=None,
    log=print,
) -> dict:
    """唯一诊断运行：A 训头 → B/C/dev 收集 → 拟合前中止检查 → 拟合三探针 → 完整指纹校验 →
    test 单次触碰（扫描 → 数据集/DataLoader → 收集）→ 评分落盘。

    数据顺序（预注册）：三个探针全部冻结前，本诊断不读 test CSV、不构建 test 数据集/DataLoader、
    不扫描 test 标签，拟合前中止只判定 A/B/C/dev。拟合前指纹身份只做 provisional 校验（自哈希 +
    train/val 文件逐项比对，见 provisional_fingerprint_checks）；完整 protocol.verify_fingerprint
    （含 test 源文件）在三探针冻结后、test 评估前执行，失败即中止且不计算任何 test 指标。
    enforce=True（正式）：冻结白名单配对身份（(model_seed, stage1_id) 命中
    FORMAL_SEED_STAGE1_PAIRS 且 Stage-1 meta 种子一致）、全部中止原因、干净 git 树均为硬约束；
    enforce=False（smoke）：中止原因只记录不中止、不校验正式常量，且合成数据豁免——
    ensure_fingerprint 会在拟合前构建/校验含 test 的完整指纹，正式模式绝不走该路径。
    """
    root = Path(root)
    device = torch.device(device)
    t0 = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    train_budget = int(a_rows) + int(b_rows) + int(c_rows)
    ranges = split_ranges(a_rows, b_rows, c_rows)
    prefix_tag = prefix_tag_for(train_budget, val_budget, test_budget)
    tag = tag or ("latdiag" if enforce else "latdiag-smoke")
    git = protocol.git_state()
    if require_clean_git is None:
        require_clean_git = bool(enforce)
    if require_clean_git:
        assert_clean_git_for_formal(git)
    if run_id is None:
        run_id = protocol.make_run_id(
            datetime.now(), prefix_tag=prefix_tag, model_seed=model_seed, tag=tag, commit=git["commit"]
        )
    run_path = protocol.run_dir(root, run_id)
    if run_path.exists():
        raise FileExistsError(f"run 目录已存在，禁止覆盖: {run_path}")

    bench._reset_peak_vram(device)
    log(
        f"[latdiag] 开始：run_id={run_id} enforce={enforce} stage1_id={stage1_id} "
        f"prefix={prefix_tag} A/B/C={a_rows}/{b_rows}/{c_rows} val={val_budget} test={test_budget} "
        f"batch={batch_size} epochs={epochs} device={device}"
    )
    protocol.seed_model(model_seed)

    # ---- 前缀指纹（正式：只读 + provisional 身份校验；smoke：本地 root 可构建）----
    # 预注册：三探针全部冻结前不触碰 test。既有 protocol.verify_fingerprint 会对 train/val/test
    # 三个文件重读前缀字节并扫描标签（protocol.py:249-264，含 test）→ 正式模式拟合前不调用它，
    # 只做 provisional 校验（指纹文件自哈希 + train/val 文件逐项比对）；完整校验（含 test）推迟到
    # 三探针冻结后、test 评估前执行（见下方 full 段），失败则中止且不产生任何 test 指标。
    files_budgets = {
        "train": (Path(data_files["train"]), train_budget),
        "val": (Path(data_files["val"]), int(val_budget)),
        "test": (Path(data_files["test"]), int(test_budget)),
    }
    if enforce:
        fp = protocol.load_fingerprint(root, prefix_tag)  # 读取已冻结指纹文件本身属身份材料（允许）
        provisional = provisional_fingerprint_checks(fp)
    else:
        # smoke（合成数据）豁免：ensure_fingerprint 会在拟合前构建/校验含 test 的完整指纹；
        # 正式模式绝不走该路径（只用 load_fingerprint + provisional）。
        fp, _ = protocol.ensure_fingerprint(root, prefix_tag, files_budgets)
        provisional = provisional_fingerprint_checks(fp)
    fingerprint_verification = {
        "provisional": provisional,
        "full": None,  # 三探针冻结后、test 评估前填充；中止运行保持 null
    }
    budgets_match_fp = (
        fp["budgets"]["train"] == train_budget
        and fp["budgets"]["val"] == int(val_budget)
        and fp["budgets"]["test"] == int(test_budget)
    )
    log(f"[latdiag] 指纹：{fp['fingerprint_sha256'][:16]}（budgets 匹配={budgets_match_fp}）")
    log(
        f"[latdiag] 拟合前指纹身份（provisional，scope={'/'.join(provisional['scope'])}）："
        f"self_hash={provisional['self_hash']} files={provisional['files']} "
        f"→ passed={provisional['passed']}"
    )
    if enforce:
        log(
            "[latdiag] 正式模式：拟合前不读 test CSV、不扫描 test 标签（预注册）；完整指纹校验"
            "（含 test 源文件逐项比对）推迟到三探针冻结后、test 评估前执行。"
        )
    else:
        log(
            "[latdiag] smoke 模式（合成数据，豁免）：ensure_fingerprint 已在拟合前构建/校验含 "
            "test 的完整指纹；正式模式不使用该路径。"
        )

    # ---- 源数据扫描（拟合前：仅 train 的 A/B/C 区段与 dev；test 推迟到三探针冻结后）----
    range_hashes = split_range_hashes(data_files["train"], ranges)
    counts = range_label_counts(data_files["train"], ranges)
    dev_counts = protocol.scan_label_counts(data_files["val"], int(val_budget))
    expected_n = {"A": int(a_rows), "B": int(b_rows), "C": int(c_rows),
                  "dev": int(val_budget), "test": int(test_budget)}
    scan_counts = {**{s: counts[s] for s in ("A", "B", "C")}, "dev": dev_counts}
    length_mismatches = [
        {"split": s, "actual": int(scan_counts[s]["n"]), "expected": expected_n[s]}
        for s in ("A", "B", "C", "dev")
        if int(scan_counts[s]["n"]) != expected_n[s]
    ]
    negatives = {s: int(scan_counts[s]["n"]) - int(scan_counts[s]["bsi_pos"]) for s in scan_counts}
    log(
        f"[latdiag] 原始扫描（拟合前仅 A/B/C/dev；test 冻结后再扫描）："
        f"negatives={negatives} length_mismatches={length_mismatches}"
    )

    # ---- 数据加载（文件顺序；A/B/C 为视图，不复制）----
    # test 数据集/DataLoader 不在此构建：预注册要求三探针全部冻结后才触碰 test
    train_full = AliCCPDataset(data_files["train"], train_budget)
    dev_ds = AliCCPDataset(data_files["val"], int(val_budget))
    views = {
        "A": RangeView(train_full, *ranges["A"]),
        "B": RangeView(train_full, *ranges["B"]),
        "C": RangeView(train_full, *ranges["C"]),
        "dev": dev_ds,
    }
    loaders = {
        split: DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        for split, ds in views.items()
    }

    # ---- 加载固定 Stage-1（唯一 backbone 来源；正式：常量身份）----
    art = protocol.load_stage1(root, stage1_id)
    meta = art["meta"]
    model = MPTRec(
        num_tasks=protocol.NUM_TASKS,
        feature_vocabulary=protocol.build_vocab(),
        embedding_size=protocol.EMBEDDING_SIZE,
        input_size=protocol.INPUT_SIZE,
        expert_dnn_hidden_units=list(protocol.EXPERT_HIDDEN),
        tower_dnn_hidden_units=list(protocol.TOWER_HIDDEN),
        dropout=list(protocol.DROPOUT),
        reg_embedding=protocol.REG_EMBEDDING,
        reg_dnn=protocol.REG_DNN,
        device=device,
    ).to(device)
    model.load_state_dict(art["backbone_state"])
    backbone_sha_loaded = protocol.backbone_sha256(model)
    identity_checks = {
        "stage1_id_recorded": meta.get("stage1_id") == stage1_id,
        "stage1_meta_seed_match": int(meta.get("model_seed", -1)) == int(model_seed),
        "fingerprint_match": meta.get("fingerprint_sha256") == fp["fingerprint_sha256"],
        "budgets_match_fp": bool(budgets_match_fp),
        "backbone_sha_match": backbone_sha_loaded == meta.get("backbone_sha256"),
        # 拟合前指纹身份为 provisional（自哈希 + train/val）；test 部分由三探针冻结后的完整
        # 校验负责（失败即中止，不会进入任何 OK 结果）
        "provisional_fingerprint_train_val": bool(provisional["passed"]),
        # 正式：(model_seed → stage1_id) 必须命中冻结白名单（不含 seed1；见 FORMAL_SEED_STAGE1_PAIRS）
        "formal_seed_stage1_pair_match": (
            FORMAL_SEED_STAGE1_PAIRS.get(int(model_seed)) == stage1_id
        ) if enforce else True,
        "formal_fingerprint_match": (fp["fingerprint_sha256"] == FORMAL_FINGERPRINT_SHA256) if enforce else True,
    }
    identity_ok = all(identity_checks.values())
    log(f"[latdiag] Stage-1 身份：{identity_checks}")
    protocol.freeze_backbone(model)
    sha_before = protocol.backbone_sha256(model)

    # ---- 唯一 BSI 头：只在 A 训练，dev 选点 ----
    head = NewTask(
        input_size=protocol.INPUT_SIZE,
        rep_dim=protocol.NEWTASK_REP_DIM,
        tower_dnn_hidden_units=list(protocol.TOWER_HIDDEN),
        reg_dnn=protocol.REG_DNN,
        device=device,
    ).to(device)
    head_info = train_bsi_head(
        model, head, loaders["A"], loaders["dev"], device=device,
        epochs=epochs, patience=patience, lr=HEAD_LR, log=log,
    )
    sha_after_train = protocol.backbone_sha256(model)
    try:
        protocol.assert_no_grads(model)
        grads_none = True
    except AssertionError as exc:  # noqa: BLE001
        grads_none = False
        log(f"[latdiag] 冻结失败：{exc}")
    freeze_ok = (sha_before == sha_after_train) and grads_none

    # ---- 收集 B/C/dev（test 在探针冻结前不触碰）----
    raws = {split: collect_split_probs(model, head, loaders[split], device) for split in ("B", "C", "dev")}
    loader_parity = {}
    for split in ("B", "C", "dev"):
        raw = raws[split]
        scan = scan_counts[split]
        parity = (
            raw["n"] == expected_n[split]
            and raw["click1"] == int(scan["click1"])
            and raw["purchase1"] == int(scan["purchase1"])
            and raw["bsi_pos"] == int(scan["bsi_pos"])
        )
        loader_parity[split] = bool(parity)
        if raw["n"] != expected_n[split]:
            length_mismatches.append({"split": split, "actual": int(raw["n"]), "expected": expected_n[split]})
        log(f"[latdiag] 收集 {split}: n={raw['n']} bsi_pos={raw['bsi_pos']} loader-parity={parity}")

    old_task_var = {
        split: {
            "ctr": float(np.var(raws[split]["ctr_prob"].astype(np.float64))),
            "cvr": float(np.var(raws[split]["cvr_prob"].astype(np.float64))),
        }
        for split in ("B", "C", "dev")
    }
    prefit_facts = {
        "identity_ok": identity_ok,
        "identity_detail": {k: v for k, v in identity_checks.items() if not v},
        "freeze_ok": freeze_ok,
        "freeze_detail": {
            "sha_before": sha_before, "sha_after_train": sha_after_train, "grads_none": grads_none,
        },
        "negatives": negatives,
        "old_task_var": old_task_var,
        "loader_parity": loader_parity,
        "length_mismatches": length_mismatches,
        "fingerprint_verification": fingerprint_verification,
    }
    reasons = prefit_abort_reasons(prefit_facts)
    log(f"[latdiag] 拟合前中止检查：{len(reasons)} 条原因 {[r['code'] for r in reasons]}")
    if reasons and not enforce and length_mismatches:
        # smoke 且长度异常：硬失败（避免置换索引越界），正式运行走中止路径
        raise RuntimeError(f"smoke 数据长度与预算不一致: {length_mismatches}")

    # ---- config（身份与配置；含 config_sha256 自哈希）----
    config_body = {
        "run_id": run_id,
        "tag": tag,
        "enforce": bool(enforce),
        "require_clean_git": bool(require_clean_git),
        "created_at": started_at,
        "data_files": {k: str(v) for k, v in data_files.items()},
        "budgets": {
            "train": train_budget, "val": int(val_budget), "test": int(test_budget),
            "A": int(a_rows), "B": int(b_rows), "C": int(c_rows),
        },
        "model_seed": int(model_seed),
        "head": {
            "train_split": "A", "range": list(ranges["A"]), "epochs": int(epochs),
            "patience": int(patience), "batch_size": int(batch_size), "lr": float(HEAD_LR),
            "optimizer": "Adam", "loss": "BCELoss(pred,y)+l2", "selection": "official dev BSI AUC",
        },
        "probe": {
            "C": PROBE_C, "penalty": PROBE_PENALTY, "solver": PROBE_SOLVER, "max_iter": PROBE_MAX_ITER,
            "standardize": "B only, ddof=0", "clip_eps": CLIP_EPS, "scale_floor": SCALE_FLOOR,
            "prob": "sigmoid(coef @ z + intercept)",
        },
        "shuffle": {
            "seed": SHUFFLE_SEED, "order": list(PERM_ORDER),
            "scheme": "numpy.random.default_rng(seed); permutation(n) per split in fixed order; joint CTR/CVR",
        },
        "thresholds": {
            "min_bsi_negatives": MIN_BSI_NEGATIVES, "var_floor": VAR_FLOOR,
            "delta_test_min": DELTA_TEST_MIN, "delta_C_min": DELTA_C_MIN, "delta_dev_min": DELTA_DEV_MIN,
            "class_pos_min": CLASS_POS_MIN, "class_neg_max": CLASS_NEG_MAX,
        },
        "arms": list(ARMS),
        "arm_features": {arm: list(ARM_FEATURES[arm]) for arm in ARMS},
        "stage1": {
            "stage1_id": stage1_id,
            "backbone_sha256_loaded": backbone_sha_loaded,
            "expected_formal_stage1_id": FORMAL_SEED_STAGE1_PAIRS.get(int(model_seed)),
            "expected_formal_seed_stage1_pairs": {
                str(k): v for k, v in FORMAL_SEED_STAGE1_PAIRS.items()
            },
            "expected_formal_fingerprint_sha256": FORMAL_FINGERPRINT_SHA256,
        },
        "fingerprint": {
            "prefix_tag": prefix_tag,
            "fingerprint_sha256": fp["fingerprint_sha256"],
            "budgets": {k: int(v) for k, v in fp["budgets"].items()},
            "prefix_sha256": {s: fp["files"][s]["prefix_sha256"] for s in ("train", "val", "test")},
            # provisional = 拟合前（自哈希 + train/val）；full = 三探针冻结后、test 评估前
            # （含 test 逐项比对），中止运行保持 null
            "verification": fingerprint_verification,
        },
        "splits": {
            "A": {"range": list(ranges["A"]), "budget": int(a_rows), "range_sha256": range_hashes["A"], "counts_raw": norm_counts(counts["A"])},
            "B": {"range": list(ranges["B"]), "budget": int(b_rows), "range_sha256": range_hashes["B"], "counts_raw": norm_counts(counts["B"])},
            "C": {"range": list(ranges["C"]), "budget": int(c_rows), "range_sha256": range_hashes["C"], "counts_raw": norm_counts(counts["C"])},
            "dev": {"range": None, "budget": int(val_budget), "prefix_sha256": fp["files"]["val"]["prefix_sha256"], "counts_raw": norm_counts(dev_counts)},
            # test counts_raw 拟合前为 null（预注册：不触碰 test）：OK 运行在探针冻结、test 扫描后
            # 补齐；任何中止运行（拟合前 NO-GO 或冻结后完整指纹校验失败）保持 null。
            "test": {
                "range": None, "budget": int(test_budget),
                "prefix_sha256": fp["files"]["test"]["prefix_sha256"],
                "counts_raw": None,
                "counts_raw_note": "三探针冻结前不触碰 test；OK 运行在冻结后扫描补齐，任何中止运行保持 null",
            },
        },
        "commit": git["commit"],
        "git": git,
        "versions": _versions(),
        "device": str(device),
    }

    # ---- 中止路径（仅正式）：写 config/abort/raw/head，绝不拟合 ----
    if reasons and enforce:
        run_path.mkdir(parents=True, exist_ok=True)
        _save_raw_npz(run_path / "raw.npz", raws)
        torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, run_path / "bsi_head.pt")
        abort_doc = {
            "run_id": run_id,
            "status": "ABORT",
            "verdict": "ABORT_NO_GO",
            "abort_phase": "prefit",
            "reasons": reasons,
            "facts": prefit_facts,
            "artifacts_sha256": artifacts_sha256(run_path),
            "commit": git["commit"],
        }
        _write_json(
            run_path / "config.json",
            {"config": config_body, "config_sha256": protocol.config_hash(config_body)},
        )
        _write_json(run_path / "abort.json", abort_doc)
        log(f"[latdiag] 正式运行 NO-GO 中止：{[r['code'] for r in reasons]} → {run_path}")
        return {
            "run_id": run_id, "run_dir": str(run_path), "status": "ABORT", "verdict": "ABORT_NO_GO",
            "classification": None, "signal_positive": False, "deltas": None, "aucs": None,
            "gates": None, "abort_reasons": reasons, "recorded_reasons": reasons,
            "wall_seconds": round(time.time() - t0, 1),
        }

    # ---- 置换（每切分独立；联合 CTR/CVR）----
    lengths = {"B": int(b_rows), "C": int(c_rows), "dev": int(val_budget), "test": int(test_budget)}
    perms = make_permutations(lengths)
    perm_hashes = {split: array_sha256(perms[split]) for split in PERM_ORDER}
    log(f"[latdiag] 置换完成：{ {s: perm_hashes[s][:12] for s in PERM_ORDER} }")

    # ---- logit 列 + 只用 B 的标准化 + 拟合三探针 ----
    logits = {
        s: {name: clip_logit(raws[s][f"{name}_prob"]) for name in ("bsi", "ctr", "cvr")}
        for s in ("B", "C", "dev")
    }
    y_b = raws["B"]["y"].astype(np.int64)

    probes, stats = {}, {}
    for arm in ARMS:
        stats[arm] = standardizer_stats(arm_columns(logits, perms, arm, "B"))
        z = apply_standardizer(arm_columns(logits, perms, arm, "B"), stats[arm])
        x_b = np.column_stack([z[name] for name in ARM_FEATURES[arm]])
        probes[arm] = fit_probe(x_b, y_b)
        log(
            f"[latdiag] 探针 {arm}: coef={[round(c, 4) for c in probes[arm]['coef']]} "
            f"intercept={probes[arm]['intercept']:.4f} converged={probes[arm]['converged']} "
            f"n_iter={probes[arm]['n_iter']}"
        )
    log("[latdiag] 三探针已冻结")

    # ---- 完整指纹校验（既有协议函数；含 test 源文件逐项比对；预注册：test 评估前执行）----
    # 此调用会重读 train/val/test 前缀字节并扫描标签（protocol.py:249-264）——因此它只能出现在
    # 三探针冻结之后、test 评估触碰之前。失败 → 中止：不扫描 test 标签、不构建 test 数据集/
    # DataLoader、不推理、不计算任何 test 指标，也不写 probes 产物（该运行按无效处理）。
    try:
        protocol.verify_fingerprint(fp)
        full_verify_detail = None
    except (AssertionError, ValueError) as exc:
        full_verify_detail = f"{type(exc).__name__}: {exc}"
    fingerprint_verification["full"] = {
        "scope": ["train", "val", "test"],
        "timing": "post_freeze_pre_test_eval",
        "passed": full_verify_detail is None,
        "detail": full_verify_detail,
    }
    log(
        "[latdiag] 完整指纹校验（train/val/test vs 预注册指纹，test 评估前）："
        f"passed={full_verify_detail is None}"
        + ("" if full_verify_detail is None else f" detail={full_verify_detail}")
    )
    if full_verify_detail is not None:
        run_path.mkdir(parents=True, exist_ok=True)
        _save_raw_npz(run_path / "raw.npz", raws)
        torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, run_path / "bsi_head.pt")
        post_reasons = list(reasons) + [
            {"code": "FULL_FINGERPRINT_VERIFY_FAIL", "detail": full_verify_detail}
        ]
        abort_doc = {
            "run_id": run_id,
            "status": "ABORT",
            "verdict": "ABORT_NO_GO",
            "abort_phase": "post_freeze_pre_test_eval",
            "reasons": post_reasons,
            "facts": prefit_facts,
            "artifacts_sha256": artifacts_sha256(run_path),
            "commit": git["commit"],
        }
        _write_json(
            run_path / "config.json",
            {"config": config_body, "config_sha256": protocol.config_hash(config_body)},
        )
        _write_json(run_path / "abort.json", abort_doc)
        log(f"[latdiag] 完整指纹校验失败 → 中止（无 test 扫描/数据集/指标）→ {run_path}")
        return {
            "run_id": run_id, "run_dir": str(run_path), "status": "ABORT", "verdict": "ABORT_NO_GO",
            "classification": None, "signal_positive": False, "deltas": None, "aucs": None,
            "gates": None, "abort_reasons": post_reasons, "recorded_reasons": post_reasons,
            "wall_seconds": round(time.time() - t0, 1),
        }

    log("[latdiag] → 对官方 test 的唯一一次评估触碰（扫描 → 数据集/DataLoader → 收集）")

    # ---- test 单次触碰（探针冻结后才允许）：标签扫描 → 数据集 → DataLoader → 收集 ----
    test_scan = protocol.scan_label_counts(data_files["test"], int(test_budget))
    scan_counts["test"] = test_scan
    negatives["test"] = int(test_scan["n"]) - int(test_scan["bsi_pos"])
    if int(test_scan["n"]) != expected_n["test"]:
        length_mismatches.append(
            {"split": "test", "actual": int(test_scan["n"]), "expected": expected_n["test"]}
        )
    test_ds = AliCCPDataset(data_files["test"], int(test_budget))
    loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    raws["test"] = collect_split_probs(model, head, loaders["test"], device)
    loader_parity["test"] = bool(
        raws["test"]["n"] == expected_n["test"]
        and raws["test"]["click1"] == int(test_scan["click1"])
        and raws["test"]["purchase1"] == int(test_scan["purchase1"])
        and raws["test"]["bsi_pos"] == int(test_scan["bsi_pos"])
    )
    if raws["test"]["n"] != expected_n["test"]:
        length_mismatches.append({"split": "test", "actual": int(raws["test"]["n"]), "expected": expected_n["test"]})
    log(
        f"[latdiag] 收集 test（冻结后单次）：n={raws['test']['n']} bsi_pos={raws['test']['bsi_pos']} "
        f"negatives={negatives['test']} loader-parity={loader_parity['test']}"
    )
    logits["test"] = {name: clip_logit(raws["test"][f"{name}_prob"]) for name in ("bsi", "ctr", "cvr")}

    # ---- 评分（冻结探针 × 每切分列；概率由 coef/intercept 重算）----
    probe_probs = {}
    aucs = {}
    for split in ALL_SPLITS:
        aucs[split] = {}
        for arm in ARMS:
            z = apply_standardizer(arm_columns(logits, perms, arm, split), stats[arm])
            x_split = np.column_stack([z[name] for name in ARM_FEATURES[arm]])
            probs = probe_prob(x_split, probes[arm])
            probe_probs[f"{split}__{arm}_prob"] = probs
            aucs[split][arm] = metrics.auc_score(raws[split]["y"], probs)
    deltas = {split: delta_of(aucs[split]) for split in ALL_SPLITS}
    classification = classify_delta(deltas["test"])
    log(
        "[latdiag] AUC："
        + "；".join(
            f"{s} base={aucs[s]['base']:.4f} full={aucs[s]['full']:.4f} shuffle={aucs[s]['shuffle']:.4f}"
            for s in ALL_SPLITS
        )
    )
    log(f"[latdiag] Δ：test={deltas['test']:+.6f} C={deltas['C']:+.6f} dev={deltas['dev']:+.6f} 分类={classification}")

    # ---- 诊断量与门禁 ----
    variances = {
        split: {
            "ctr_raw": float(np.var(raws[split]["ctr_prob"].astype(np.float64))),
            "cvr_raw": float(np.var(raws[split]["cvr_prob"].astype(np.float64))),
            "ctr_logit": float(np.var(logits[split]["ctr"])),
            "cvr_logit": float(np.var(logits[split]["cvr"])),
        }
        for split in ALL_SPLITS
    }
    correlations = {
        split: {
            "bsi_ctr": corr_of(logits[split]["bsi"], logits[split]["ctr"]),
            "bsi_cvr": corr_of(logits[split]["bsi"], logits[split]["cvr"]),
            "ctr_cvr": corr_of(logits[split]["ctr"], logits[split]["cvr"]),
        }
        for split in ALL_SPLITS
    }
    split_counts = {}
    for split in ALL_SPLITS:
        raw = raws[split]
        split_counts[split] = {
            "n": int(raw["n"]),
            "click1": int(raw["click1"]),
            "purchase1": int(raw["purchase1"]),
            "bsi_pos": int(raw["bsi_pos"]),
            "bsi_neg": int(raw["n"]) - int(raw["bsi_pos"]),
            "loader_parity_ok": bool(loader_parity.get(split, False)),
        }
    sha_after_all = protocol.backbone_sha256(model)
    try:
        protocol.assert_no_grads(model)
        grads_none_end = True
    except AssertionError:  # noqa: BLE001
        grads_none_end = False
    freeze_ok_end = freeze_ok and (sha_before == sha_after_all) and grads_none_end
    scan_len_ok = all(int(scan_counts[s]["n"]) == expected_n[s] for s in scan_counts)
    counts_ok = (
        all(split_counts[s]["loader_parity_ok"] for s in ALL_SPLITS)
        and all(split_counts[s]["n"] == expected_n[s] for s in ALL_SPLITS)
        and scan_len_ok
        and all(negatives[s] >= MIN_BSI_NEGATIVES for s in scan_counts)
    )
    ranges_ok = all(range_hashes[s] for s in ("A", "B", "C"))
    perms_ok = all(
        sorted(perms[s].tolist()) == list(range(lengths[s])) for s in PERM_ORDER
    ) and all(perm_hashes[s] for s in PERM_ORDER)
    gate_out = evaluate_gates(
        {
            "identity_ok": identity_ok,
            "identity_detail": {k: v for k, v in identity_checks.items() if not v},
            "freeze_ok": freeze_ok_end,
            "freeze_detail": {
                "sha_before": sha_before, "sha_after_all": sha_after_all,
                "grads_none_train": grads_none, "grads_none_end": grads_none_end,
            },
            "counts_ok": bool(counts_ok),
            "counts_detail": f"negatives={negatives}",
            "ranges_ok": bool(ranges_ok),
            "ranges_detail": {s: range_hashes[s][:12] for s in ("A", "B", "C")},
            "perms_ok": bool(perms_ok),
            "perms_detail": {s: perm_hashes[s][:12] for s in PERM_ORDER},
            "variances": {s: {"ctr": variances[s]["ctr_raw"], "cvr": variances[s]["cvr_raw"]} for s in ALL_SPLITS},
            "deltas": deltas,
            "git_dirty": bool(git["dirty"]),
        }
    )
    log(f"[latdiag] 门禁：{ {k: v['verdict'] for k, v in gate_out['gates'].items()} } → {gate_out['verdict']}")

    # ---- config 补齐（仅 OK 路径）：test 计数在探针冻结后才写入；随后计算 config 自哈希 ----
    config_body["splits"]["test"]["counts_raw"] = norm_counts(scan_counts["test"])
    config_sha = protocol.config_hash(config_body)

    # ---- 落盘（原始产物 → 身份哈希 → reported/gate_report）----
    run_path.mkdir(parents=True, exist_ok=True)
    _save_raw_npz(run_path / "raw.npz", raws)
    np.savez(run_path / "perms.npz", **{f"{s}__perm": perms[s] for s in PERM_ORDER})
    np.savez(run_path / "probe_probs.npz", **probe_probs)
    _write_json(
        run_path / "probes.json",
        {
            "arms": {arm: probes[arm] for arm in ARMS},
            "standardization": stats,
            "arm_features": {arm: list(ARM_FEATURES[arm]) for arm in ARMS},
            "probe_params": {
                "C": PROBE_C, "penalty": PROBE_PENALTY, "solver": PROBE_SOLVER, "max_iter": PROBE_MAX_ITER,
            },
        },
    )
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, run_path / "bsi_head.pt")
    artifact_sha = artifacts_sha256(run_path)
    finished_at = datetime.now().isoformat(timespec="seconds")
    peak_vram = (
        round(torch.cuda.max_memory_allocated(device.index) / 1e6, 1) if device.type == "cuda" else None
    )
    reported = {
        "run_id": run_id,
        "tag": tag,
        "status": "OK",
        "enforce": bool(enforce),
        "config_sha256": config_sha,
        "stage1_id": stage1_id,
        "fingerprint_sha256": fp["fingerprint_sha256"],
        "fingerprint_verification": fingerprint_verification,
        "backbone_sha256_loaded": backbone_sha_loaded,
        "backbone_sha256_before": sha_before,
        "backbone_sha256_after": sha_after_all,
        "backbone_grads_none": bool(grads_none_end),
        "head": head_info,
        "splits": split_counts,
        "variances": variances,
        "correlations": correlations,
        "aucs": aucs,
        "deltas": {split: {"base": aucs[split]["base"], "full": aucs[split]["full"],
                           "shuffle": aucs[split]["shuffle"], "delta": deltas[split]} for split in ALL_SPLITS},
        "classification": classification,
        "signal_positive": bool(gate_out["signal_positive"]),
        "verdict": gate_out["verdict"],
        "range_sha256": range_hashes,
        "perm_sha256": perm_hashes,
        "artifacts_sha256": artifact_sha,
        "commit": git["commit"],
        "git": git,
        "versions": _versions(),
        "device": str(device),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": round(time.time() - t0, 1),
        "peak_vram_mb": peak_vram,
        "recorded_reasons": reasons,
    }
    _write_json(run_path / "config.json", {"config": config_body, "config_sha256": config_sha})
    _write_json(run_path / "reported.json", reported)
    _write_json(
        run_path / "gate_report.json",
        {
            "run_id": run_id,
            "tag": tag,
            "enforce": bool(enforce),
            "gates": gate_out["gates"],
            "signal_positive": bool(gate_out["signal_positive"]),
            "verdict": gate_out["verdict"],
            "classification": classification,
            "deltas": {s: float(deltas[s]) for s in ALL_SPLITS},
        },
    )
    log(f"[latdiag] 完成：run_id={run_id} verdict={gate_out['verdict']} wall={reported['wall_seconds']}s")
    return {
        "run_id": run_id,
        "run_dir": str(run_path),
        "status": "OK",
        "verdict": gate_out["verdict"],
        "classification": classification,
        "signal_positive": bool(gate_out["signal_positive"]),
        "deltas": deltas,
        "aucs": aucs,
        "gates": gate_out["gates"],
        "abort_reasons": [],
        "recorded_reasons": reasons,
        "wall_seconds": reported["wall_seconds"],
    }


def _save_raw_npz(path, raws: dict) -> None:
    payload = {}
    for split, raw in raws.items():
        for field in ("y", "click", "purchase", "bsi_prob", "ctr_prob", "cvr_prob"):
            payload[f"{split}__{field}"] = np.asarray(raw[field])
    np.savez(path, **payload)
