"""三 seed 配对验证 + 二级分类（预注册 `docs/superpowers/specs/2026-10-05-census-stage2-null-expert-multiseed-design.md` §6）。

只读模块：扫描 `runs/` 下的落盘产物，产出 per-seed 配对记录与三 seed 检查点统计；
不写任何 artifacts（检查点 JSON 仅在 CLI 显式 `--out` 时落盘）。

硬约束（预注册 §4.1）：每 seed 的基线臂与处理臂必须引用**同一** stage1_id 与同一 split 指纹；
`find_runs` 对"多候选 / 缺臂 / 配对不一致"一律抛错——协议禁止挑 run。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

from census_benchmark import protocol as P

SPEC_PATH = "docs/superpowers/specs/2026-10-05-census-stage2-null-expert-multiseed-design.md"

# ---- Canonical seed 列表（预注册 §1；来自已提交证据，不发明）----
CANONICAL_MODEL_SEEDS = (1685480945, 1685463909, 1685477428, 1685459668, 1685496394)
CHECKPOINT_SEEDS = CANONICAL_MODEL_SEEDS[:3]

# ---- 二级分类（预注册 §6.1；边界含等号，逐字）----
POSITIVE_IMPROVEMENT = "POSITIVE_IMPROVEMENT"
NO_CLEAR_IMPROVEMENT = "NO_CLEAR_IMPROVEMENT"
CLEAR_DEGRADATION = "CLEAR_DEGRADATION"
RECLASS_POSITIVE_MIN = 0.001
RECLASS_DEGRADE_MAX = -0.02

# ---- 复现锚与统计参数（预注册 §3.4、§6.3）----
REPRO_TOL = 1e-9
TOP1_MIN, TOP1_MAX = 0.05, 0.95
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}      # key = df = n-1（Student t, 双侧 95%）

ANCHOR_BASELINE_RUN_ID = "20260929-1735-s20260929-m1685480945-short-904f8d0"
ANCHOR_NULLX_RUN_ID = "20260929-1802-s20260929-m1685480945-short-1eadd05-nullx"
ANCHOR_BASELINE_TEST_AUC = 0.8500685307175756
ANCHOR_NULLX_TEST_AUC = 0.8513648272440768


ANCHOR_RUN_IDS = (ANCHOR_BASELINE_RUN_ID, ANCHOR_NULLX_RUN_ID)


def reclassify_delta(delta_test: float) -> str:
    """预注册 §6.1 二级分类：≥ +0.001 → POSITIVE；≤ −0.02 → CLEAR_DEGRADATION；其余 NO_CLEAR。"""
    if delta_test >= RECLASS_POSITIVE_MIN:
        return POSITIVE_IMPROVEMENT
    if delta_test <= RECLASS_DEGRADE_MAX:
        return CLEAR_DEGRADATION
    return NO_CLEAR_IMPROVEMENT


def cross_seed_consistent(deltas) -> bool:
    """预注册 §6.2：多数（严格过半）Δtest > 0 **且** 无 seed Δtest ≤ −0.02。"""
    deltas = [float(d) for d in deltas]
    if not deltas:
        return False
    majority_positive = sum(1 for d in deltas if d > 0) * 2 > len(deltas)
    no_degradation = all(d > RECLASS_DEGRADE_MAX for d in deltas)
    return bool(majority_positive and no_degradation)


def seed_headroom(*, mechanism_active: bool, delta_val: float, budget_right_censored: bool,
                  budget_last_delta_positive: bool, cross_seed_consistent) -> dict:
    """预注册 §6.2：headroom_remains = 四维度全真；`cross_seed_consistent=None`（未知）时结果为 None。

    该记录只是 headroom 描述（适用 `NO_CLEAR_IMPROVEMENT`），不构成任何有效性主张。
    """
    val_direction_positive = bool(delta_val > 0)
    all_components = (bool(mechanism_active) and val_direction_positive
                      and bool(budget_right_censored) and bool(budget_last_delta_positive))
    headroom = None if cross_seed_consistent is None else bool(all_components and bool(cross_seed_consistent))
    return {"mechanism_active": bool(mechanism_active), "val_direction_positive": val_direction_positive,
            "budget_right_censored": bool(budget_right_censored),
            "budget_last_delta_positive": bool(budget_last_delta_positive),
            "cross_seed_consistent": cross_seed_consistent,
            "delta_val": delta_val, "headroom_remains": headroom}


def checkpoint_stats(deltas) -> dict:
    """预注册 §6.3：mean / 样本 std(ddof=1) / positive_count / worst / Student t 95% CI（方法原句落盘）。

    `worst_index` 取列表中 Δtest 最小者（并列取列表序靠前者，列表序 = `CHECKPOINT_SEEDS` 顺序）。
    """
    deltas = [float(d) for d in deltas]
    n = len(deltas)
    df = n - 1
    if df not in T_975:
        raise ValueError(f"t 表未覆盖 df={df}（n={n}）；预注册 §6.3 仅定义 n∈{{2,3,4,5}}")
    mean = sum(deltas) / n
    std = statistics.stdev(deltas)                       # ddof=1
    half = T_975[df] * std / (n ** 0.5)
    worst_index = min(range(n), key=lambda i: deltas[i])
    return {"n": n, "mean": mean, "std": std,
            "positive_count": sum(1 for d in deltas if d >= RECLASS_POSITIVE_MIN),
            "worst_index": worst_index, "worst_delta": deltas[worst_index],
            "ci95_low": mean - half, "ci95_high": mean + half,
            "ci_method": f"Student t 95% CI on paired deltas, df=n-1, t={T_975[df]} (n={n})"}


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_hashes(directory) -> dict | None:
    """目录下全部文件的 sha256（按文件名排序）；目录不存在 → None（纯 JSON 干跑夹具允许缺 stage1）。"""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}


def _is_nullx(config: dict) -> bool:
    """处理臂判定：run_id 的 `-nullx` 后缀是协议定义；config 标记存在时必须与之一致。"""
    by_suffix = str(config.get("run_id", "")).endswith("-nullx")
    if "null_expert" in config and bool(config["null_expert"]) != by_suffix:
        raise ValueError(f"run 内 null_expert 标记与 run_id 后缀不一致: {config.get('run_id')}")
    return by_suffix


def load_run(run_dir) -> dict:
    """读取一个 run 的 config / metrics / gate_report，并做内部一致性校验。"""
    run_dir = Path(run_dir)
    config = _load_json(run_dir / "config.json")
    metrics_doc = _load_json(run_dir / "metrics.json")
    for key in ("run_id", "stage1_id"):
        if config.get(key) != metrics_doc.get(key):
            raise ValueError(f"run 内部 {key} 不一致: {run_dir}")
    if config.get("model_seed") is not None:
        marker = f"-m{config['model_seed']}-"
        if marker not in str(config.get("run_id", "")):
            raise ValueError(f"run_id 与 config.model_seed 不一致: {run_dir}")
    _is_nullx(config)                                     # 触发一致性校验
    return {"run_dir": run_dir, "config": config, "metrics": metrics_doc,
            "gate": _load_json(run_dir / "gate_report.json")}


def find_runs(root, model_seed: int, *, tag: str = "short") -> dict:
    """定位某 model seed 的（唯一）基线臂与处理臂 run；歧义 / 缺臂 / 配对不一致一律抛错。

    预注册 §3.4 的两个历史锚 run（`ANCHOR_RUN_IDS`，committed evidence）不参与"本轮对子"的选择——
    否则 seed 1 会因历史对子与新一轮对子并存而判为"不唯一"。锚只用于 §6.5 的复现比对。

    本分支适配（longer-budget 预注册 §6/§11）：`tag` 参数（默认 `"short"`，缺省行为逐位不变）；
    `tag="long"` 只选择 long 族 run，short/long 并存时互不干扰，缺 long 对子时**不**回退到 short。
    """
    runs_dir = Path(root) / "runs"
    if not runs_dir.is_dir():
        raise ValueError(f"runs 目录不存在: {runs_dir}")
    candidates = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "config.json").exists()):
        config = _load_json(run_dir / "config.json")
        if config.get("run_id") in ANCHOR_RUN_IDS:
            continue
        if (config.get("model_seed") == model_seed and config.get("tag") == tag
                and config.get("split_seed") == P.SPLIT_SEED):
            candidates.append(load_run(run_dir))
    baselines = [c for c in candidates if not _is_nullx(c["config"])]
    arms = [c for c in candidates if _is_nullx(c["config"])]
    if len(baselines) != 1 or len(arms) != 1:
        raise ValueError(f"seed {model_seed}（tag={tag}）的 run 不唯一/缺失"
                         f"（baseline={len(baselines)}, nullx={len(arms)}）；协议禁止挑 run")
    baseline, arm = baselines[0], arms[0]
    if baseline["metrics"]["stage1_id"] != arm["metrics"]["stage1_id"]:
        raise ValueError(f"seed {model_seed} 两臂 stage1_id 不一致（禁止跨 artifact 配对）")
    if baseline["metrics"]["split_sha256"]["fingerprint"] != arm["metrics"]["split_sha256"]["fingerprint"]:
        raise ValueError(f"seed {model_seed} 两臂 split 指纹不一致（禁止跨划分配对）")
    if baseline["metrics"]["env_ids_sha256"] != arm["metrics"]["env_ids_sha256"]:
        raise ValueError(f"seed {model_seed} 两臂 env_ids 不一致")
    return {"baseline": baseline["run_dir"], "nullx": arm["run_dir"]}


def seed_record(root, model_seed: int, *, cross_consistent=None, tag: str = "short") -> dict:
    """某 seed 的配对记录：两臂 AUC、配对 Δ、二级分类、headroom、预算状态、门禁（预注册 §7）。

    `tag` 透传 `find_runs`（默认 `"short"`，缺省行为逐位不变）。
    """
    runs = find_runs(root, model_seed, tag=tag)
    baseline, arm = load_run(runs["baseline"]), load_run(runs["nullx"])
    b2, a2 = baseline["metrics"]["stage2"], arm["metrics"]["stage2"]
    delta_test = a2["test_auc"] - b2["test_auc"]
    delta_val = a2["best_val_auc"] - b2["best_val_auc"]
    b_records, a_records = b2["epoch_records"], a2["epoch_records"]
    common = min(len(b_records), len(a_records))
    delta_by_epoch = [a_records[i]["auc_val_education"] - b_records[i]["auc_val_education"] for i in range(common)]
    mechanism = arm["metrics"].get("mechanism", {})
    null_top1 = mechanism.get("null_top1_rate")
    mechanism_active = bool(null_top1 is not None and TOP1_MIN <= null_top1 <= TOP1_MAX)
    arm_right_censored = a2["best_epoch"] == a_records[-1]["epoch"]
    headroom = seed_headroom(mechanism_active=mechanism_active, delta_val=delta_val,
                             budget_right_censored=arm_right_censored,
                             budget_last_delta_positive=bool(delta_by_epoch and delta_by_epoch[-1] > 0),
                             cross_seed_consistent=cross_consistent)
    stage1_dir = Path(root) / "stage1" / arm["metrics"]["stage1_id"]
    stage1_meta = None
    meta_path = stage1_dir / "meta.json"
    if meta_path.exists():
        meta_full = _load_json(meta_path)
        stage1_meta = {key: meta_full.get(key) for key in
                       ("backbone_sha256", "env_ids_sha256", "split_fingerprint_sha256", "config_hash",
                        "epochs", "model_seed", "env_seed")}
    return {
        "model_seed": model_seed, "stage1_id": arm["metrics"]["stage1_id"],
        "stage1": {"stage1_id": arm["metrics"]["stage1_id"], "meta": stage1_meta,
                   "files": _file_hashes(stage1_dir)},
        "baseline": {"run_id": baseline["metrics"]["run_id"], "commit": baseline["metrics"].get("commit"),
                     "test_auc": b2["test_auc"], "best_val_auc": b2["best_val_auc"],
                     "best_epoch": b2["best_epoch"], "epoch_records": b_records,
                     "split_fingerprint": baseline["metrics"]["split_sha256"]["fingerprint"],
                     "env_ids_sha256": baseline["metrics"]["env_ids_sha256"],
                     "config": baseline["config"], "files": _file_hashes(baseline["run_dir"])},
        "arm": {"run_id": arm["metrics"]["run_id"], "commit": arm["metrics"].get("commit"),
                "test_auc": a2["test_auc"], "best_val_auc": a2["best_val_auc"],
                "best_epoch": a2["best_epoch"], "epoch_records": a_records,
                "mechanism": mechanism, "null_arm": arm["metrics"].get("null_arm"),
                "config": arm["config"], "files": _file_hashes(arm["run_dir"])},
        "delta_test": delta_test, "delta_val": delta_val,
        "classification": reclassify_delta(delta_test),
        "budget": {"baseline_best_epoch": b2["best_epoch"], "baseline_last_epoch": b_records[-1]["epoch"],
                   "arm_best_epoch": a2["best_epoch"], "arm_last_epoch": a_records[-1]["epoch"],
                   "baseline_right_censored": b2["best_epoch"] == b_records[-1]["epoch"],
                   "arm_right_censored": arm_right_censored,
                   "common_epochs": common, "delta_val_by_epoch": delta_by_epoch,
                   "delta_val_last_common": delta_by_epoch[-1] if delta_by_epoch else None},
        "headroom": headroom,
        "gates": {"baseline": baseline["gate"], "arm": arm["gate"]},
    }


def anchor_record(root) -> dict:
    """seed 1 新跑 vs 已提交历史对子（预注册 §3.4 / §6.5）：差 ≤ 1e-9 → REPRODUCED，否则 DIVERGED。"""
    root = Path(root)
    out = {"baseline_run_id": ANCHOR_BASELINE_RUN_ID, "nullx_run_id": ANCHOR_NULLX_RUN_ID,
           "historical_baseline_test_auc": ANCHOR_BASELINE_TEST_AUC,
           "historical_nullx_test_auc": ANCHOR_NULLX_TEST_AUC, "tolerance": REPRO_TOL}
    hist_b = root / "runs" / ANCHOR_BASELINE_RUN_ID / "metrics.json"
    hist_n = root / "runs" / ANCHOR_NULLX_RUN_ID / "metrics.json"
    if not (hist_b.exists() and hist_n.exists()):
        out["status"] = "REFERENCE_ABSENT"
        return out
    recorded_b = _load_json(hist_b)["stage2"]["test_auc"]
    recorded_n = _load_json(hist_n)["stage2"]["test_auc"]
    out["historical_dirs_match_constants"] = (recorded_b == ANCHOR_BASELINE_TEST_AUC
                                              and recorded_n == ANCHOR_NULLX_TEST_AUC)
    try:
        found = find_runs(root, CANONICAL_MODEL_SEEDS[0])
    except ValueError:
        out["status"] = "NO_FRESH_PAIR"
        return out
    fresh_b = load_run(found["baseline"])["metrics"]
    fresh_n = load_run(found["nullx"])["metrics"]
    diff_b = abs(fresh_b["stage2"]["test_auc"] - ANCHOR_BASELINE_TEST_AUC)
    diff_n = abs(fresh_n["stage2"]["test_auc"] - ANCHOR_NULLX_TEST_AUC)
    out.update({"fresh_baseline_run_id": fresh_b["run_id"], "fresh_nullx_run_id": fresh_n["run_id"],
                "fresh_baseline_test_auc": fresh_b["stage2"]["test_auc"],
                "fresh_nullx_test_auc": fresh_n["stage2"]["test_auc"],
                "abs_diff_baseline": diff_b, "abs_diff_nullx": diff_n,
                "status": "REPRODUCED" if max(diff_b, diff_n) <= REPRO_TOL else "DIVERGED"})
    return out


def checkpoint(root, seeds=CHECKPOINT_SEEDS) -> dict:
    """三 seed 检查点：per-seed 记录 + 统计（§6.3）+ 复现锚（§6.5）；cross_seed 一致性统一注入。"""
    seeds = tuple(seeds)
    records = [seed_record(root, seed) for seed in seeds]
    deltas = [record["delta_test"] for record in records]
    consistent = cross_seed_consistent(deltas)
    for record in records:
        head = record["headroom"]
        record["headroom"] = seed_headroom(mechanism_active=head["mechanism_active"], delta_val=head["delta_val"],
                                           budget_right_censored=head["budget_right_censored"],
                                           budget_last_delta_positive=head["budget_last_delta_positive"],
                                           cross_seed_consistent=consistent)
    if len(deltas) >= 2:
        stats = checkpoint_stats(deltas)
    else:                                   # 干跑/单 seed：CI 未定义，如实置 None（不伪造统计量）
        stats = {"n": len(deltas), "mean": deltas[0] if deltas else None, "std": None,
                 "positive_count": sum(1 for d in deltas if d >= RECLASS_POSITIVE_MIN),
                 "worst_index": 0 if deltas else None, "worst_delta": deltas[0] if deltas else None,
                 "ci95_low": None, "ci95_high": None, "ci_method": "undefined for n<2"}
    stats["worst_seed"] = seeds[stats["worst_index"]] if stats["worst_index"] is not None else None
    return {"spec": SPEC_PATH, "created": datetime.now().isoformat(timespec="seconds"),
            "seeds": records, "stats": stats, "anchor": anchor_record(root)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CensusIncome null-expert 多 seed 配对检查点（只读 run 目录）")
    parser.add_argument("--root", type=Path, default=P.ARTIFACT_ROOT)
    parser.add_argument("--out", type=Path, default=None, help="检查点 JSON 落盘路径（缺省只打印）")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="覆盖默认三 seed 检查点列表")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    seeds = tuple(args.seeds) if args.seeds else CHECKPOINT_SEEDS
    record = checkpoint(args.root, seeds=seeds)
    if args.out:
        Path(args.out).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = record["stats"]
    print(f"[multiseed] n={stats['n']} mean={stats['mean']!r} std={stats['std']!r} "
          f"positive={stats['positive_count']} worst_seed={stats['worst_seed']}")
    if stats["ci95_low"] is not None:
        print(f"[multiseed] ci95=[{stats['ci95_low']:+.9f}, {stats['ci95_high']:+.9f}] {stats['ci_method']}")
    for rec in record["seeds"]:
        print(f"[multiseed] seed={rec['model_seed']} delta_test={rec['delta_test']:+.9f} "
              f"delta_val={rec['delta_val']:+.9f} class={rec['classification']} "
              f"headroom={rec['headroom']['headroom_remains']}")
    print(f"[multiseed] anchor={record['anchor']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
