"""独立验证器（独立解析器）：只读 run 产物 + 数据文件 + Stage-1 产物，逐项重算并比对。

约束（预注册 2026-10-08 §Artifacts）：
- 不得 import aliccp_benchmark 的任何模块（常量在本文件独立复制一份，含
  (model_seed → Stage-1 ID) 冻结白名单；两份副本由测试锁定相等）。
- 不得调用训练/评分函数：AUC 直接调用 sklearn.metrics.roc_auc_score；
  探针概率由 probes.json 的 coef/intercept 独立重算；置换由 seed 独立重生成。
- SUMMARY 只在全部检查 PASS（且运行本身为正式运行或显式要求）之后追加；同一 run_id 幂等。

用法（见 run_aliccp_latent_diag.py verify）：verify_run(run_dir, root)。
"recomputed" 中的全部数值均为本验证器自算，不取自任何上报文件。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from multitaskrec.model import MPTRec

# ---- 冻结常量的独立副本（与 latent_diag 逐项相等；由测试锁定）----
# 扩展分支（seeds2–3）：正式运行只认这些 (model_seed → Stage-1 ID) 冻结配对；不含 seed1。
FORMAL_SEED_STAGE1_PAIRS = {
    1688723740: "s1-5c060b9c-m1688723740-e3-4e1b5c6f",
    1688738016: "s1-5c060b9c-m1688738016-e3-47619ce0",
}
FORMAL_FINGERPRINT_SHA256 = (
    "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
)
SHUFFLE_SEED = 20261008
CLIP_EPS = 1e-6
SCALE_FLOOR = 1e-12
MIN_BSI_NEGATIVES = 100
VAR_FLOOR = 1e-8
DELTA_TEST_MIN = 0.001
DELTA_C_MIN = 0.001
DELTA_DEV_MIN = 0.0
CLASS_POS_MIN = 0.001
CLASS_NEG_MAX = -0.02
PROBE_C = 1.0
PROBE_PENALTY = "l2"
PROBE_SOLVER = "lbfgs"
PROBE_MAX_ITER = 200
ARMS = ("base", "full", "shuffle")
ARM_FEATURES = {"base": ("bsi",), "full": ("bsi", "ctr", "cvr"), "shuffle": ("bsi", "ctr", "cvr")}
ALL_SPLITS = ("B", "C", "dev", "test")
PERM_ORDER = ("B", "C", "dev", "test")
# 诊断本地切分名 → 公平指纹键的唯一别名：指纹只有 train/val/test 三个键，诊断的 dev 即官方 val
# （config 的 budgets 键同指纹命名；验证器本地常量，不在 latent_diag 侧镜像）
FP_KEY_ALIAS = {"dev": "val"}
RAW_FILES = ("raw.npz", "perms.npz", "probe_probs.npz", "probes.json", "bsi_head.pt")
SUMMARY_FILE = "LATENT_DIAG_SUMMARY.md"
LATENT_DIAG_SUMMARY_COLUMNS = [
    "run_id", "commit", "tag",
    "auc_base_test", "auc_full_test", "auc_shuffle_test",
    "auc_base_C", "auc_full_C", "auc_shuffle_C",
    "delta_test", "delta_C", "delta_dev",
    "classification", "verdict", "stage1_id", "raw_sha256",
]

TOL_AUC = 1e-9
TOL_DELTA = 1e-9
TOL_PROB = 1e-9
TOL_STATS = 1e-12
TOL_REFIT = 1e-5


# ---- 基础工具（独立重实现，不 import 被测包）----
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_npz(path) -> dict:
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def _fp_key(split: str) -> str:
    """诊断本地切分名 → 指纹键（唯一别名 dev → val；其余原样）。"""
    return FP_KEY_ALIAS.get(split, split)


def _array_sha256(arr) -> str:
    arr = np.ascontiguousarray(arr)
    return _sha256_bytes(f"{arr.dtype}:{list(arr.shape)}:".encode() + arr.tobytes())


def _artifacts_sha256(run_dir, names=RAW_FILES) -> str:
    digest = hashlib.sha256()
    for name in names:
        path = Path(run_dir) / name
        if not path.exists():
            continue
        digest.update(f"{name}:{_sha256_file(path)}\n".encode())
    return digest.hexdigest()


def _range_sha256(path, start: int, end: int) -> str:
    digest = hashlib.sha256(f"range:{int(start)}:{int(end)}\n".encode())
    with open(path, "rb") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"文件为空: {path}")
        digest.update(header)
        for i in range(int(end)):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} 数据行不足 {end}（第 {i + 1} 行截断）")
            if int(start) <= i < int(end):
                digest.update(line)
    return digest.hexdigest()


def _header_sha256(path) -> str:
    with open(path, "rb") as handle:
        return _sha256_bytes(handle.readline())


def _prefix_sha256(path, n: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"文件为空: {path}")
        digest.update(header)
        for i in range(int(n)):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} 数据行不足 {n}（第 {i + 1} 行截断）")
            digest.update(line)
    return digest.hexdigest()


def _range_counts(path, ranges: dict) -> dict:
    max_end = max(int(end) for _, end in ranges.values())
    counts = {name: {"n": 0, "click1": 0, "purchase1": 0, "bsi_pos": 0, "bsi_raw": {}} for name in ranges}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i in range(max_end):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} 数据行不足 {max_end}（第 {i + 1} 行截断）")
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


def _prefix_counts(path, n: int) -> dict:
    return _range_counts(path, {"full": (0, int(n))})["full"]


def _norm_counts(counts: dict) -> dict:
    return {
        "n": int(counts["n"]),
        "click1": int(counts["click1"]),
        "purchase1": int(counts["purchase1"]),
        "bsi_pos": int(counts["bsi_pos"]),
        "bsi_raw": {str(k): int(v) for k, v in sorted(counts["bsi_raw"].items())},
    }


def _clip_logit(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    c = np.clip(p, CLIP_EPS, 1.0 - CLIP_EPS)
    return np.log(c / (1.0 - c))


def _std_stats(cols: dict) -> dict:
    stats = {"means": {}, "scales": {}, "raw_scales": {}, "constant": {}}
    for name, col in cols.items():
        col = np.asarray(col, dtype=np.float64)
        mean, scale = float(np.mean(col)), float(np.std(col))
        stats["means"][name] = mean
        stats["raw_scales"][name] = scale
        if scale < SCALE_FLOOR:
            stats["scales"][name] = 1.0
            stats["constant"][name] = True
        else:
            stats["scales"][name] = scale
            stats["constant"][name] = False
    return stats


def _apply_std(cols: dict, stats: dict) -> dict:
    return {
        name: (np.asarray(col, dtype=np.float64) - stats["means"][name]) / stats["scales"][name]
        for name, col in cols.items()
    }


def _sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa == 0.0 or sb == 0.0:
        return None
    return float(np.mean((a - np.mean(a)) * (b - np.mean(b)))) / (sa * sb)


def _regenerate_perms(lengths: dict) -> dict:
    rng = np.random.default_rng(SHUFFLE_SEED)
    return {split: rng.permutation(int(lengths[split])) for split in PERM_ORDER}


# 预注册臂-列映射（外部规范；本验证器直接实现该映射，不参考/复用 runner 代码）：
#   base    = [bsi]
#   full    = [bsi, ctr, cvr]（行对齐原列）
#   shuffle = [bsi, ctr[perm], cvr[perm]]（仅 CTR/CVR 行序被联合置换；BSI 不置换）
# 历史缺陷（2026-10-08 编排审计）：验证器曾复制 runner 的同一缺陷（置换同时应用于
# full 与 shuffle）→ 对作废运行给出虚假 PASS。
PERMUTED_FEATURES_BY_ARM = {
    "base": frozenset(),
    "full": frozenset(),
    "shuffle": frozenset({"ctr", "cvr"}),
}


def _arm_cols(logits: dict, perms: dict, arm: str, split: str) -> dict:
    if arm == "base":
        return {"bsi": logits[split]["bsi"]}
    permuted = PERMUTED_FEATURES_BY_ARM[arm]  # 未知臂 → KeyError（冻结 ARMS 之外无合法臂）
    cols = {"bsi": logits[split]["bsi"]}
    for name in ("ctr", "cvr"):
        source = logits[split][name]
        cols[name] = source[perms[split]] if name in permuted else source
    return cols


def _param_digest(param) -> str:
    t = param.detach().cpu().contiguous()
    return _sha256_bytes(str(t.dtype).encode() + str(tuple(t.shape)).encode() + t.numpy().tobytes())


def _backbone_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(_param_digest(param).encode())
    return digest.hexdigest()


def _build_model(meta_config: dict) -> MPTRec:
    return MPTRec(
        num_tasks=int(meta_config["num_tasks"]),
        feature_vocabulary={str(k): int(v) for k, v in meta_config["vocab"].items()},
        embedding_size=int(meta_config["embedding_size"]),
        input_size=int(meta_config["input_size"]),
        expert_dnn_hidden_units=[int(v) for v in meta_config["expert_hidden"]],
        tower_dnn_hidden_units=[int(v) for v in meta_config["tower_hidden"]],
        dropout=[float(v) for v in meta_config["dropout"]],
        reg_embedding=float(meta_config["reg_embedding"]),
        reg_dnn=float(meta_config["reg_dnn"]),
        device=torch.device("cpu"),
    )


def _git_output(*args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def _git_dirty_now():
    out = _git_output("status", "--porcelain", "--untracked-files=no")
    return None if out is None else bool(out)


def _auc(y, p) -> float:
    return float(roc_auc_score(np.asarray(y).astype(int), np.asarray(p, dtype=np.float64)))


def _delta(aucs: dict) -> float:
    return float(aucs["full"]) - max(float(aucs["base"]), float(aucs["shuffle"]))


def _classify(delta: float) -> str:
    if delta >= CLASS_POS_MIN:
        return "POSITIVE"
    if delta <= CLASS_NEG_MAX:
        return "CLEAR_DECLINE"
    return "NO_CLEAR"


def _gate_verdicts(facts: dict) -> dict:
    def v(ok):
        return "PASS" if ok else "FAIL"

    deltas = facts["deltas"]
    variances = facts["variances"]
    nonconst = all(
        float(variances[s][t]) > 0.0 for s in ("B", "C", "test") for t in ("ctr", "cvr")
    )
    return {
        "I1_identity": v(facts["identity_ok"]),
        "I2_freeze": v(facts["freeze_ok"]),
        "D1_counts": v(facts["counts_ok"]),
        "D2_ranges": v(facts["ranges_ok"]),
        "P1_perm": v(facts["perms_ok"]),
        "P2_nonconst": v(nonconst),
        "S1_delta_test": v(float(deltas["test"]) >= DELTA_TEST_MIN),
        "S2_delta_C": v(float(deltas["C"]) >= DELTA_C_MIN),
        "S3_delta_dev": v(float(deltas["dev"]) >= DELTA_DEV_MIN),
        "G1_git": v(facts["git_dirty"] is False),
    }


def _summary_rows(path) -> set:
    path = Path(path)
    if not path.exists():
        return set()
    rows = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and not line.startswith("| run_id"):
            rows.add(line.split("|")[1].strip())
    return rows


def _append_summary_row(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "| " + " | ".join(LATENT_DIAG_SUMMARY_COLUMNS) + " |\n"
            + "|" + "---|" * len(LATENT_DIAG_SUMMARY_COLUMNS) + "\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(str(row[c]) for c in LATENT_DIAG_SUMMARY_COLUMNS) + " |\n")


def verify_run(run_dir, root, *, append_summary=None, log=print) -> dict:
    """独立重算并比对一次诊断运行的全部上报值；返回检查清单与结论。"""
    run_dir, root = Path(run_dir), Path(root)
    checks = []

    def check(cid: str, ok: bool, detail="") -> bool:
        checks.append({"id": cid, "ok": bool(ok), "detail": str(detail)})
        return bool(ok)

    config_doc = _load_json(run_dir / "config.json")
    config = config_doc["config"]
    run_id = config["run_id"]
    stage1_id = config["stage1"]["stage1_id"]
    prefix_tag = config["fingerprint"]["prefix_tag"]
    enforce = bool(config["enforce"])
    if append_summary is None:
        append_summary = enforce
    abort_path = run_dir / "abort.json"
    status = "ABORT" if abort_path.exists() else "OK"
    log(f"[verify] run_id={run_id} status={status} enforce={enforce}")

    # ---- 配置身份与冻结常量 ----
    check(
        "config_sha256",
        _sha256_bytes(_canonical(config).encode()) == config_doc["config_sha256"],
        "配置自哈希重算",
    )
    frozen = (
        config["thresholds"]
        == {
            "min_bsi_negatives": MIN_BSI_NEGATIVES, "var_floor": VAR_FLOOR,
            "delta_test_min": DELTA_TEST_MIN, "delta_C_min": DELTA_C_MIN, "delta_dev_min": DELTA_DEV_MIN,
            "class_pos_min": CLASS_POS_MIN, "class_neg_max": CLASS_NEG_MAX,
        }
        and (config["probe"]["C"], config["probe"]["penalty"], config["probe"]["solver"], config["probe"]["max_iter"])
        == (PROBE_C, PROBE_PENALTY, PROBE_SOLVER, PROBE_MAX_ITER)
        and int(config["shuffle"]["seed"]) == SHUFFLE_SEED
        and tuple(config["shuffle"]["order"]) == PERM_ORDER
        and tuple(config["arms"]) == ARMS
        and {k: tuple(v) for k, v in config["arm_features"].items()} == ARM_FEATURES
    )
    check("frozen_config_match", frozen, "阈值/探针/置换种子/臂定义与冻结值逐项相等")

    recorded = _load_json(abort_path) if status == "ABORT" else _load_json(run_dir / "reported.json")
    recorded_sha = recorded.get("artifacts_sha256")
    recomputed_artifact_sha = _artifacts_sha256(run_dir)
    check("artifacts_sha256", recorded_sha == recomputed_artifact_sha, f"recorded={recorded_sha} recomputed={recomputed_artifact_sha}")

    # ---- 数据文件与前缀指纹 ----
    data_files = {k: Path(v) for k, v in config["data_files"].items()}
    files_ok = all(p.exists() for p in data_files.values())
    check("data_files_present", files_ok, {k: str(p) for k, p in data_files.items()})

    fp_path = root / "splits" / prefix_tag / "prefix_fingerprint.json"
    fp = _load_json(fp_path)
    fp_body = {k: v for k, v in fp.items() if k != "fingerprint_sha256"}
    check("fingerprint_self", _sha256_bytes(_canonical(fp_body).encode()) == fp["fingerprint_sha256"], "指纹自哈希重算")
    fp_files_ok, fp_detail = True, {}
    for tag in ("train", "val", "test"):
        meta = fp["files"][tag]
        budget = int(fp["budgets"][tag])
        path = data_files[tag]
        prefix_ok = _prefix_sha256(path, budget) == meta["prefix_sha256"]
        counts_ok = _norm_counts(_prefix_counts(path, budget)) == _norm_counts(fp["label_counts"][tag])
        header_ok = _header_sha256(path) == fp["header_sha256"]
        fp_detail[tag] = {"prefix_sha256": prefix_ok, "label_counts": counts_ok, "header": header_ok}
        fp_files_ok = fp_files_ok and prefix_ok and counts_ok and header_ok
    check("fingerprint_files", fp_files_ok, fp_detail)
    # 正式 OK 运行必须命中正式常量指纹；中止运行的常量不符本身由 abort 原因确认覆盖
    formal_fp_ok = (not enforce) or status == "ABORT" or fp["fingerprint_sha256"] == FORMAL_FINGERPRINT_SHA256
    check(
        "fingerprint_match",
        fp["fingerprint_sha256"] == config["fingerprint"]["fingerprint_sha256"] and formal_fp_ok,
        f"fp={fp['fingerprint_sha256'][:16]} config={config['fingerprint']['fingerprint_sha256'][:16]}",
    )
    budgets_match_fp = all(
        int(fp["budgets"][k]) == int(config["budgets"][k]) for k in ("train", "val", "test")
    )

    # ---- 区段身份（A/B/C 哈希与计数；dev/test 原始计数）----
    ranges = {s: tuple(int(v) for v in config["splits"][s]["range"]) for s in ("A", "B", "C")}
    own_range_hashes = {s: _range_sha256(data_files["train"], *ranges[s]) for s in ("A", "B", "C")}
    own_range_counts = _range_counts(data_files["train"], ranges)
    check(
        "ranges_hashes",
        all(own_range_hashes[s] == config["splits"][s]["range_sha256"] for s in ("A", "B", "C")),
        {s: own_range_hashes[s][:12] for s in ("A", "B", "C")},
    )
    scan_ok = all(
        _norm_counts(own_range_counts[s]) == config["splits"][s]["counts_raw"] for s in ("A", "B", "C")
    )
    dev_scan = _prefix_counts(data_files["val"], int(config["budgets"]["val"]))
    test_scan = _prefix_counts(data_files["test"], int(config["budgets"]["test"]))
    scan_ok = (
        scan_ok
        and _norm_counts(dev_scan) == config["splits"]["dev"]["counts_raw"]
        and int(dev_scan["n"]) == int(config["splits"]["dev"]["budget"])
    )
    test_cfg_counts = config["splits"]["test"].get("counts_raw")
    if status == "OK":
        scan_ok = (
            scan_ok
            and test_cfg_counts is not None
            and _norm_counts(test_scan) == test_cfg_counts
            and int(test_scan["n"]) == int(config["splits"]["test"]["budget"])
        )
    else:
        # 预注册：三探针冻结前不得触碰 test → 中止运行不得在 config 记录 test 标签计数
        # （验证器自身在运行结束后独立重扫 test，与冻结指纹的比对由 fingerprint_files 覆盖）
        check(
            "test_counts_deferred_prefit",
            test_cfg_counts is None,
            f"abort 运行 test counts_raw={test_cfg_counts}（必须为 null：拟合前不触碰 test）",
        )
    check("split_counts_scan", scan_ok, "原始扫描计数与 config 逐项相等")
    negatives = {
        "A": int(own_range_counts["A"]["n"]) - int(own_range_counts["A"]["bsi_pos"]),
        "B": int(own_range_counts["B"]["n"]) - int(own_range_counts["B"]["bsi_pos"]),
        "C": int(own_range_counts["C"]["n"]) - int(own_range_counts["C"]["bsi_pos"]),
        "dev": int(dev_scan["n"]) - int(dev_scan["bsi_pos"]),
        "test": int(test_scan["n"]) - int(test_scan["bsi_pos"]),
    }

    # ---- Stage-1 检查点（结构 + 全参数哈希链）----
    meta_path = root / "stage1" / stage1_id / "meta.json"
    backbone_path = root / "stage1" / stage1_id / "backbone.pt"
    meta = _load_json(meta_path) if meta_path.exists() else {}
    backbone_sha = None
    backbone_load_ok = False
    backbone_detail = ""
    if backbone_path.exists() and meta:
        try:
            state = torch.load(backbone_path, map_location="cpu")
            module = _build_model(meta["config"])
            module.load_state_dict(state, strict=True)
            backbone_sha = _backbone_sha256(module)
            backbone_load_ok = True
        except Exception as exc:  # noqa: BLE001
            backbone_detail = f"{type(exc).__name__}: {exc}"
    check("stage1_backbone_load", backbone_load_ok, backbone_detail or "strict 加载 + 结构一致")
    check(
        "stage1_backbone_sha",
        backbone_load_ok and backbone_sha == meta.get("backbone_sha256"),
        f"recomputed={backbone_sha} meta={meta.get('backbone_sha256')}",
    )

    # ---- 原始逐行数组（raw.npz）----
    raw = _load_npz(run_dir / "raw.npz") if (run_dir / "raw.npz").exists() else {}
    present_splits = [s for s in ("B", "C", "dev", "test") if f"{s}__y" in raw]
    raw_counts_ok = True
    raw_vs_fp_ok = True
    raw_detail = {}
    for split in present_splits:
        y = raw[f"{split}__y"]
        click1 = int(raw[f"{split}__click"].sum())
        purchase1 = int(raw[f"{split}__purchase"].sum())
        bsi_pos = int(y.sum())
        expected = config["splits"][split]["counts_raw"]
        ok = (
            len(y) == int(config["splits"][split]["budget"])
            and click1 == int(expected["click1"])
            and purchase1 == int(expected["purchase1"])
            and bsi_pos == int(expected["bsi_pos"])
        )
        raw_counts_ok = raw_counts_ok and ok
        raw_detail[split] = {"n": len(y), "click1": click1, "purchase1": purchase1, "bsi_pos": bsi_pos, "ok": ok}
        if split in ("dev", "test"):
            fp_counts = fp["label_counts"][_fp_key(split)]  # dev → 指纹 val 条目（唯一别名）
            raw_vs_fp_ok = raw_vs_fp_ok and (
                click1 == int(fp_counts["click1"])
                and purchase1 == int(fp_counts["purchase1"])
                and bsi_pos == int(fp_counts["bsi_pos"])
            )
    check("raw_counts_vs_scan", raw_counts_ok, raw_detail)
    check("raw_counts_vs_fingerprint", raw_vs_fp_ok, "dev（对指纹 val 条目）/test 逐行标签与指纹计数一致（BSI 定义闭环）")
    if status == "OK":
        check("raw_splits_complete", len(present_splits) == 4, present_splits)

    # ---- 身份/冻结门禁的独立重算 ----
    identity_checks = {
        "stage1_id_recorded": meta.get("stage1_id") == stage1_id,
        "stage1_meta_seed_match": int(meta.get("model_seed", -1)) == int(config["model_seed"]),
        "fingerprint_match": meta.get("fingerprint_sha256") == fp["fingerprint_sha256"],
        "budgets_match_fp": budgets_match_fp,
        "backbone_sha_match": backbone_sha == meta.get("backbone_sha256"),
        # 独立重判（不采信 runner 的接受决定）：(config.model_seed → stage1_id) 必须命中本验证器
        # 自持的冻结白名单副本（不含 seed1；见 FORMAL_SEED_STAGE1_PAIRS）
        "formal_seed_stage1_pair_match": (
            FORMAL_SEED_STAGE1_PAIRS.get(int(config["model_seed"])) == stage1_id
        ) if enforce else True,
        "formal_fingerprint_match": (fp["fingerprint_sha256"] == FORMAL_FINGERPRINT_SHA256) if enforce else True,
    }
    identity_ok = all(identity_checks.values())
    if status == "OK":
        recorded_chain = {
            "loaded": recorded.get("backbone_sha256_loaded"),
            "before": recorded.get("backbone_sha256_before"),
            "after": recorded.get("backbone_sha256_after"),
        }
    else:  # 中止路径：冻结事实记录在 abort.facts.freeze_detail
        freeze_detail = recorded.get("facts", {}).get("freeze_detail", {})
        recorded_chain = {
            "loaded": meta.get("backbone_sha256"),
            "before": freeze_detail.get("sha_before"),
            "after": freeze_detail.get("sha_after_train"),
        }
    chain_ok = backbone_load_ok and (
        recorded_chain["loaded"] == recorded_chain["before"] == recorded_chain["after"] == backbone_sha
        == meta.get("backbone_sha256")
    )
    check("chain_backbone", chain_ok, {"recorded": recorded_chain, "recomputed": backbone_sha, "meta": meta.get("backbone_sha256")})
    grads_none = bool(recorded.get("backbone_grads_none")) if status == "OK" else bool(
        recorded.get("facts", {}).get("freeze_detail", {}).get("grads_none")
    )
    freeze_ok = chain_ok and grads_none

    # ---- git 溯源 ----
    git_recorded = config.get("git", {})
    check(
        "git_recorded",
        isinstance(git_recorded.get("dirty"), bool)
        and git_recorded.get("commit") == config.get("commit")
        and (status != "OK" or git_recorded.get("dirty") == recorded.get("git", {}).get("dirty")),
        f"config.git={git_recorded}",
    )
    commit = config.get("commit")
    commit_valid = commit == "nogit" or _git_output("rev-parse", "--verify", f"{commit}^{{commit}}") is not None
    check("git_commit_valid", bool(commit_valid), f"commit={commit}")
    dirty_now = _git_dirty_now()
    if enforce:
        clean_ok = git_recorded.get("dirty") is False and dirty_now is False
        check("git_clean_required", clean_ok, f"recorded_dirty={git_recorded.get('dirty')} now_dirty={dirty_now}")
    else:
        check("git_clean_required", True, f"N/A（非正式运行）recorded_dirty={git_recorded.get('dirty')} now_dirty={dirty_now}")

    recomputed = {
        "status": status,
        "aucs": None, "deltas": None, "classification": None, "verdict": None,
        "signal_positive": None, "gates": None, "negatives": negatives,
    }

    if status == "ABORT":
        # ---- 中止路径：逐条独立确认中止原因 ----
        for index, reason in enumerate(recorded.get("reasons", [])):
            code = reason["code"]
            confirmed, detail = False, ""
            if code == "MIN_NEGATIVES":
                actual = negatives.get(reason["split"])
                confirmed = actual is not None and actual == int(reason["actual"]) and actual < int(reason["required"])
                detail = f"rescan negatives[{reason['split']}]={actual} required={reason['required']}"
            elif code == "CONSTANT_OLD_TASK_PREDS":
                split = reason["split"]
                if f"{split}__ctr_prob" in raw:
                    var_ctr = float(np.var(raw[f"{split}__ctr_prob"].astype(np.float64)))
                    var_cvr = float(np.var(raw[f"{split}__cvr_prob"].astype(np.float64)))
                    confirmed = var_ctr < VAR_FLOOR and var_cvr < VAR_FLOOR
                    detail = f"var_ctr={var_ctr:.3e} var_cvr={var_cvr:.3e}"
            elif code == "IDENTITY_MISMATCH":
                failed = {k: v for k, v in identity_checks.items() if not v}
                confirmed = bool(failed)
                detail = f"identity_recheck failed={failed}"
            elif code == "FREEZE_FAIL":
                detail = f"recorded={recorded.get('facts', {}).get('freeze_detail', {})}"
                confirmed = (
                    detail != "recorded={}" and not freeze_ok
                )
            elif code == "SPLIT_LENGTH_MISMATCH":
                split = reason.get("split")
                own_len = {
                    "A": int(own_range_counts["A"]["n"]), "B": int(own_range_counts["B"]["n"]),
                    "C": int(own_range_counts["C"]["n"]), "dev": int(dev_scan["n"]), "test": int(test_scan["n"]),
                }.get(split)
                confirmed = (
                    own_len is not None
                    and int(reason.get("actual", -1)) == own_len
                    and int(reason.get("actual", -1)) != int(reason.get("expected", -1))
                )
                detail = f"split={split} recorded_actual={reason.get('actual')} rescan_len={own_len} expected={reason.get('expected')}"
            elif code == "LOADER_SCAN_MISMATCH":
                split = reason.get("split")
                confirmed = split in raw_detail and not raw_detail[split]["ok"]
                detail = f"raw_counts[{split}]={raw_detail.get(split)}"
            check(f"abort_reason_{index}_{code}", confirmed, detail)
        check(
            "abort_verdict",
            recorded.get("verdict") == "ABORT_NO_GO" and recorded.get("status") == "ABORT",
            recorded.get("verdict"),
        )
        recomputed["verdict"] = "ABORT_NO_GO"
        # 诊断三元与 OK 路径一致地暴露（独立报告消费方不区分路径）：
        recomputed["identity_ok"] = identity_ok
        recomputed["freeze_ok"] = bool(freeze_ok)
        # 中止路径的计数一致性诊断：仅由中止路径已定义的事实构成（raw 逐切分计数 vs config、
        # 原始扫描计数 vs config）；负例下限（MIN_NEGATIVES）本身是合法中止原因，不并入此诊断。
        recomputed["counts_ok"] = bool(raw_counts_ok and scan_ok)
    else:
        # ---- 正常路径：置换 → 标准化 → 探针 → 概率 → AUC/Δ/门禁 全部独立重算 ----
        lengths = {
            "B": int(config["budgets"]["B"]), "C": int(config["budgets"]["C"]),
            # config budgets 键为 train/val/test/A/B/C（无 "dev"）：dev 长度 = val 预算（唯一别名）
            "dev": int(config["budgets"]["val"]), "test": int(config["budgets"]["test"]),
        }
        perms = _regenerate_perms(lengths)
        recorded_perms = _load_npz(run_dir / "perms.npz")
        perms_exact = all(
            np.array_equal(perms[s], recorded_perms.get(f"{s}__perm")) for s in PERM_ORDER
        )
        perms_valid = all(
            np.array_equal(np.sort(perms[s]), np.arange(lengths[s])) for s in PERM_ORDER
        )
        check("perms_regenerated", perms_exact and perms_valid, "由 seed 独立重生成并与 perms.npz 逐元素相等")
        # 恒等置换不破坏行级对齐（full 与 shuffle 输入会完全相同）→ shuffle 对照失效，
        # 冻结语义下必须判 FAIL（冻结种子的置换为非恒等；此处为结构性防回归）
        nonidentity = all(
            not np.array_equal(perms[s], np.arange(lengths[s])) for s in PERM_ORDER
        )
        check(
            "perms_nonidentity",
            nonidentity,
            "shuffle 置换必须为非恒等（恒等置换不破坏行级对齐，full 与 shuffle 输入相同）",
        )
        reported = recorded
        perm_hashes_ok = all(
            recorded_perms.get(f"{s}__perm") is not None
            and _array_sha256(recorded_perms[f"{s}__perm"]) == reported["perm_sha256"][s]
            for s in PERM_ORDER
        )
        check("perm_hashes", perm_hashes_ok, "置换内容哈希与 reported 一致")

        logits = {
            s: {name: _clip_logit(raw[f"{s}__{name}_prob"]) for name in ("bsi", "ctr", "cvr")}
            for s in ALL_SPLITS
        }

        probes_doc = _load_json(run_dir / "probes.json")
        std_ok = True
        std_detail = {}
        for arm in ARMS:
            own_stats = _std_stats(_arm_cols(logits, perms, arm, "B"))
            rec_stats = probes_doc["standardization"][arm]
            same = (
                all(abs(own_stats["means"][k] - rec_stats["means"][k]) <= TOL_STATS for k in own_stats["means"])
                and all(abs(own_stats["scales"][k] - rec_stats["scales"][k]) <= TOL_STATS for k in own_stats["scales"])
                and all(own_stats["constant"][k] == rec_stats["constant"][k] for k in own_stats["constant"])
            )
            std_ok = std_ok and same
            std_detail[arm] = {"ok": same}
        check("std_stats_match", std_ok, std_detail)

        y_b = raw["B__y"].astype(int)
        refit_ok = True
        refit_detail = {}
        for arm in ARMS:
            z = _apply_std(_arm_cols(logits, perms, arm, "B"), probes_doc["standardization"][arm])
            x_b = np.column_stack([z[name] for name in ARM_FEATURES[arm]])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                clf = LogisticRegression(C=PROBE_C, penalty=PROBE_PENALTY, solver=PROBE_SOLVER, max_iter=PROBE_MAX_ITER)
                clf.fit(x_b, y_b)
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            rec = probes_doc["arms"][arm]
            coef_close = np.allclose(np.ravel(clf.coef_), np.asarray(rec["coef"]), rtol=0, atol=TOL_REFIT)
            intercept_close = abs(float(np.ravel(clf.intercept_)[0]) - float(rec["intercept"])) <= TOL_REFIT
            same = coef_close and intercept_close and converged == bool(rec["converged"])
            refit_ok = refit_ok and same
            refit_detail[arm] = {
                "ok": same, "coef_close": bool(coef_close), "intercept_close": bool(intercept_close),
                "converged_match": converged == bool(rec["converged"]), "n_iter": int(np.ravel(clf.n_iter_)[0]),
            }
        check("probe_coef_refit", refit_ok, refit_detail)

        recorded_probs = _load_npz(run_dir / "probe_probs.npz")
        probs_ok = True
        aucs = {}
        for split in ALL_SPLITS:
            aucs[split] = {}
            for arm in ARMS:
                z = _apply_std(_arm_cols(logits, perms, arm, split), probes_doc["standardization"][arm])
                x_split = np.column_stack([z[name] for name in ARM_FEATURES[arm]])
                prob = _sigmoid(
                    x_split @ np.asarray(probes_doc["arms"][arm]["coef"], dtype=np.float64)
                    + float(probes_doc["arms"][arm]["intercept"])
                )
                saved = recorded_probs.get(f"{split}__{arm}_prob")
                probs_ok = probs_ok and saved is not None and np.max(np.abs(prob - saved)) <= TOL_PROB
                aucs[split][arm] = _auc(raw[f"{split}__y"], prob)
        check("probe_prob_match", probs_ok, f"逐行概率最大绝对偏差 <= {TOL_PROB}")

        auc_ok = True
        auc_detail = {}
        for split in ALL_SPLITS:
            auc_detail[split] = {}
            for arm in ARMS:
                recorded_auc = float(reported["aucs"][split][arm])
                auc_ok = auc_ok and abs(aucs[split][arm] - recorded_auc) <= TOL_AUC
                auc_detail[split][arm] = {"recomputed": aucs[split][arm], "recorded": recorded_auc}
        check("auc_match", auc_ok, auc_detail)

        deltas = {split: _delta(aucs[split]) for split in ALL_SPLITS}
        delta_ok = all(
            abs(deltas[split] - float(reported["deltas"][split]["delta"])) <= TOL_DELTA
            for split in ALL_SPLITS
        )
        check("delta_match", delta_ok, {s: {"recomputed": deltas[s], "recorded": reported["deltas"][s]["delta"]} for s in ALL_SPLITS})

        classification = _classify(deltas["test"])
        check(
            "classification_match",
            classification == reported["classification"],
            f"recomputed={classification} recorded={reported['classification']}",
        )

        variances = {
            s: {
                "ctr": float(np.var(raw[f"{s}__ctr_prob"].astype(np.float64))),
                "cvr": float(np.var(raw[f"{s}__cvr_prob"].astype(np.float64))),
            }
            for s in ALL_SPLITS
        }
        correlations = {
            s: {
                "bsi_ctr": _corr(logits[s]["bsi"], logits[s]["ctr"]),
                "bsi_cvr": _corr(logits[s]["bsi"], logits[s]["cvr"]),
                "ctr_cvr": _corr(logits[s]["ctr"], logits[s]["cvr"]),
            }
            for s in ALL_SPLITS
        }
        reported_corr = reported.get("correlations", {})
        corr_ok = True
        corr_detail = {}
        for s in ALL_SPLITS:
            for key in ("bsi_ctr", "bsi_cvr", "ctr_cvr"):
                own = correlations[s][key]
                rec = reported_corr.get(s, {}).get(key)
                if own is None or rec is None:
                    same = own is None and rec is None
                else:
                    same = abs(own - float(rec)) <= TOL_AUC
                corr_ok = corr_ok and same
                corr_detail[f"{s}.{key}"] = {"recomputed": own, "recorded": rec}
        check("correlation_match", corr_ok, corr_detail)
        counts_ok = (
            all(raw_detail[s]["ok"] for s in ALL_SPLITS)
            and all(negatives[s] >= MIN_BSI_NEGATIVES for s in ("A", "B", "C", "dev", "test"))
            and scan_ok
        )
        ranges_ok = all(own_range_hashes[s] == config["splits"][s]["range_sha256"] for s in ("A", "B", "C"))
        own_gates = _gate_verdicts(
            {
                "identity_ok": identity_ok,
                "freeze_ok": freeze_ok,
                "counts_ok": bool(counts_ok),
                "ranges_ok": bool(ranges_ok),
                "perms_ok": bool(perms_exact and perms_valid and perm_hashes_ok and nonidentity),
                "variances": variances,
                "deltas": deltas,
                "git_dirty": config.get("git", {}).get("dirty"),
            }
        )
        gate_report = _load_json(run_dir / "gate_report.json")
        gate_ok = all(
            own_gates[gid] == gate_report["gates"][gid]["verdict"] for gid in own_gates
        )
        check("gate_match", gate_ok, {gid: {"recomputed": own_gates[gid], "recorded": gate_report["gates"][gid]["verdict"]} for gid in own_gates})
        signal_positive = all(v == "PASS" for v in own_gates.values())
        verdict = "SIGNAL_POSITIVE" if signal_positive else "NO_GO_FOR_EXTENSION"
        check(
            "verdict_match",
            verdict == gate_report["verdict"] == reported["verdict"]
            and bool(signal_positive) == bool(reported["signal_positive"]) == bool(gate_report["signal_positive"]),
            f"recomputed={verdict} reported={reported['verdict']}",
        )
        head = reported.get("head", {})
        check(
            "head_fields",
            isinstance(head.get("best_epoch"), int) and 1 <= head["best_epoch"] <= int(config["head"]["epochs"])
            and 0.0 <= float(head.get("best_val_auc", -1)) <= 1.0,
            head,
        )
        recomputed.update(
            {
                "aucs": aucs, "deltas": deltas, "classification": classification,
                "verdict": verdict, "signal_positive": bool(signal_positive), "gates": own_gates,
                "identity_ok": identity_ok, "freeze_ok": bool(freeze_ok), "counts_ok": bool(counts_ok),
                "variances": variances,
            }
        )

    overall = "PASS" if all(c["ok"] for c in checks) else "FAIL"
    failed = [c["id"] for c in checks if not c["ok"]]
    log(f"[verify] overall={overall} checks={len(checks)} failed={failed}")

    # recomputed 白名单：identity/freeze/counts 诊断在 OK 与 ABORT 两条路径都必须落盘
    # （回归：中止路径曾把已写入的 identity_ok 过滤掉 → 独立报告消费方 KeyError）
    verification = {
        "run_id": run_id,
        "verified_at": datetime.now().isoformat(timespec="seconds"),
        "overall": overall,
        "failed": failed,
        "checks": checks,
        "recomputed": {
            k: v for k, v in recomputed.items()
            if k in (
                "status", "aucs", "deltas", "classification", "verdict", "signal_positive",
                "gates", "negatives", "identity_ok", "freeze_ok", "counts_ok",
            )
        },
    }
    (run_dir / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # ---- SUMMARY：只在 PASS 且（正式运行或显式要求）之后追加；同一 run_id 幂等 ----
    summary_info = {"appended": False, "already_present": False, "path": str(root / SUMMARY_FILE)}
    ledger = root / SUMMARY_FILE
    if overall == "PASS" and append_summary:
        if run_id in _summary_rows(ledger):
            summary_info["already_present"] = True
        else:
            if status == "OK":
                row = {
                    "run_id": run_id, "commit": config["commit"], "tag": config["tag"],
                    "auc_base_test": f"{recomputed['aucs']['test']['base']:.6f}",
                    "auc_full_test": f"{recomputed['aucs']['test']['full']:.6f}",
                    "auc_shuffle_test": f"{recomputed['aucs']['test']['shuffle']:.6f}",
                    "auc_base_C": f"{recomputed['aucs']['C']['base']:.6f}",
                    "auc_full_C": f"{recomputed['aucs']['C']['full']:.6f}",
                    "auc_shuffle_C": f"{recomputed['aucs']['C']['shuffle']:.6f}",
                    "delta_test": f"{recomputed['deltas']['test']:.6f}",
                    "delta_C": f"{recomputed['deltas']['C']:.6f}",
                    "delta_dev": f"{recomputed['deltas']['dev']:.6f}",
                    "classification": recomputed["classification"],
                    "verdict": recomputed["verdict"],
                    "stage1_id": stage1_id,
                    "raw_sha256": recomputed_artifact_sha,
                }
            else:
                row = {
                    "run_id": run_id, "commit": config["commit"], "tag": config["tag"],
                    "auc_base_test": "n/a", "auc_full_test": "n/a", "auc_shuffle_test": "n/a",
                    "auc_base_C": "n/a", "auc_full_C": "n/a", "auc_shuffle_C": "n/a",
                    "delta_test": "n/a", "delta_C": "n/a", "delta_dev": "n/a",
                    "classification": "n/a", "verdict": "ABORT_NO_GO",
                    "stage1_id": stage1_id, "raw_sha256": recomputed_artifact_sha,
                }
            _append_summary_row(ledger, row)
            summary_info["appended"] = True
            log(f"[verify] SUMMARY 已追加：{ledger}")
    return {
        "run_id": run_id,
        "overall": overall,
        "checks": checks,
        "recomputed": verification["recomputed"],
        "summary": summary_info,
    }
