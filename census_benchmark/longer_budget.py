"""更长 Stage-2 预算持久性检验（预注册 `docs/superpowers/specs/2026-10-05-census-stage2-null-expert-longer-budget-design.md` §5/§6）。

纯分析：**只读** run 记录（config.json / metrics.json / gate_report.json），不重新评测、不新增 test 遍历、
不改 run 产物。

- 判定 = `persistence_verdict`（V1 mean Δtest ≥ +0.001 闭 × V2 严格过半为正且无 CLEAR_DEGRADATION ×
  V3 过半 Δval > 0 × V4 轨迹末端 mean(Δval@last-common) > 0 严格 × V5 5/5 机制激活）→
  `PERSISTS` / `NOT_PERSIST`。
- identity/记录/构造一致性（逐 seed 9 项具名检查）+ A 类门禁 → 任一不一致 ⇒ `INVALID`
  （完备性分支：出现即表示运行偏离预注册，按实记录并停止解读）。
- 短预算 5-seed 检查点 JSON（字节钉死 sha256）只作 **context**（动机与对照），**不入判定**。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from census_benchmark import multiseed as M
from census_benchmark import protocol as P

SPEC_PATH = "docs/superpowers/specs/2026-10-05-census-stage2-null-expert-longer-budget-design.md"

# ---- 预注册常量（先于任何 run 钉死；看到结果后不得修改）----
LONGER_BUDGET_EPOCHS = 10
LONGER_BUDGET_PATIENCE = 3
LONGER_BUDGET_TAG = "long"
CANONICAL_SEEDS = M.CANONICAL_MODEL_SEEDS                       # 5 个 canonical seed（同一列表序）

# 五份钉死的 Stage-1 产物（内容寻址 id；来源 = 已完成 5-seed 检查点，预注册 §1.3 75/75 核验通过）
PINNED_STAGE1_IDS = {
    1685480945: "s1-096f8f16-m1685480945-e2-cb2094b3",
    1685463909: "s1-096f8f16-m1685463909-e2-8b53a3bf",
    1685477428: "s1-096f8f16-m1685477428-e2-15eabcb4",
    1685459668: "s1-096f8f16-m1685459668-e2-8611b794",
    1685496394: "s1-096f8f16-m1685496394-e2-2231eae1",
}
PINNED_SPLIT_FINGERPRINT = "096f8f16db081d7ad024b7aa450a40510a50e0acdbb439d973cc8fc9557d460c"

# 短预算 context（唯一事实来源；字节钉死；`1196acd` 分支提交的 5-seed 检查点）
SHORT_CONTEXT_PATH = "docs/superpowers/specs/2026-10-05-census-stage2-null-expert-multiseed-checkpoint-5seed.json"
SHORT_CONTEXT_SHA256 = "605196e56b7e92a78a5953205c78b447e4691216593eb1436b1c8ab26b83e2ab"
MULTISEED_TIP = "1196acd04c4e5d2846e0644d6345532f3632d148"

# ---- 判定阈值（全部沿用既有冻结数字；预注册 §5.3）----
V1_MEAN_DELTA_TEST_MIN = 0.001               # = 本线 POSITIVE_IMPROVEMENT 阈值（同一数字）
V2_DEGRADE_MAX = M.RECLASS_DEGRADE_MAX       # = −0.02（CLEAR_DEGRADATION 边界，逐字沿用）
V5_TOP1_MIN, V5_TOP1_MAX = M.TOP1_MIN, M.TOP1_MAX   # = 0.05 / 0.95（历史机制判据，两端含等号）
A_CLASS_GATES = ("A1", "A2", "A4", "A5")     # 构造/冻结完整性门禁（B 类照常记录、不参与判定）
GATE_KEYS = ("A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4")   # 协议门禁全表（SUMMARY 同序）


def persistence_verdict(*, delta_test, delta_val, delta_val_last_common, null_top1) -> dict:
    """§5.3 判据（机械计算，阈值写死）。四个列表按同一 seed 顺序对齐；返回 checks + classification。"""
    delta_test = [float(x) for x in delta_test]
    delta_val = [float(x) for x in delta_val]
    last_common = [float(x) for x in delta_val_last_common]
    top1 = [float(x) for x in null_top1]
    n = len(delta_test)
    mean_delta_test = sum(delta_test) / n
    positive_delta_count = sum(1 for d in delta_test if d > 0)
    positive_val_count = sum(1 for v in delta_val if v > 0)
    checks = {
        "V1_mean_delta_test_ge_min": bool(mean_delta_test >= V1_MEAN_DELTA_TEST_MIN),
        "V2_majority_positive_no_degradation": bool(positive_delta_count * 2 > n
                                                    and min(delta_test) > V2_DEGRADE_MAX),
        "V3_majority_delta_val_positive": bool(positive_val_count * 2 > n),
        "V4_mean_last_common_delta_val_positive": bool(sum(last_common) / n > 0.0),
        "V5_mechanism_active_all_arms": bool(all(V5_TOP1_MIN <= t <= V5_TOP1_MAX for t in top1)),
    }
    return {
        "checks": checks,
        "classification": "PERSISTS" if all(checks.values()) else "NOT_PERSIST",
        "mean_delta_test": mean_delta_test,
        "positive_delta_count": positive_delta_count,
        "positive_val_count": positive_val_count,
        "mean_last_common_delta_val": sum(last_common) / n,
        "thresholds": {"V1_min": V1_MEAN_DELTA_TEST_MIN, "V2_degrade_max": V2_DEGRADE_MAX,
                       "V5_top1_range": [V5_TOP1_MIN, V5_TOP1_MAX]},
    }


def load_short_context(path) -> dict:
    """加载短预算 context；sha256 与钉死值不符 ⇒ 抛错（预注册 §7 I7）。"""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHORT_CONTEXT_SHA256:
        raise ValueError(f"短预算 context JSON sha256 不一致（禁止继续）: {digest} != {SHORT_CONTEXT_SHA256}")
    return json.loads(path.read_text(encoding="utf-8"))


def _a_class_pass(gate_report: dict) -> bool:
    return all(bool(gate_report.get(gate, {}).get("pass")) for gate in A_CLASS_GATES)


def _seed_identity(baseline: dict, arm: dict, seed: int, *, expected_epochs: int,
                   expected_patience: int) -> dict:
    """逐 seed 的 identity/记录/构造一致性检查（§5.3 INVALID 分支；9 项具名）。"""
    b_conv, a_conv = baseline["config"], arm["config"]
    b_metrics, a_metrics = baseline["metrics"], arm["metrics"]
    b_fp = b_metrics["split_sha256"]["fingerprint"]
    a_fp = a_metrics["split_sha256"]["fingerprint"]
    pinned = PINNED_STAGE1_IDS.get(seed)
    return {
        "same_stage1_id": b_metrics["stage1_id"] == a_metrics["stage1_id"],
        "stage1_id_is_pinned": b_metrics["stage1_id"] == a_metrics["stage1_id"] == pinned,
        "split_fingerprint_same": b_fp == a_fp,
        "split_fingerprint_is_pinned": b_fp == a_fp == PINNED_SPLIT_FINGERPRINT,
        "env_ids_same": b_metrics["env_ids_sha256"] == a_metrics["env_ids_sha256"],
        "tag_recorded_long": b_conv.get("tag") == a_conv.get("tag") == LONGER_BUDGET_TAG,
        "epochs_recorded_pinned": (b_conv.get("epochs") == a_conv.get("epochs") == expected_epochs),
        "patience_recorded_pinned": (b_conv.get("patience") == a_conv.get("patience") == expected_patience),
        "a_class_gates_pass": _a_class_pass(baseline["gate"]) and _a_class_pass(arm["gate"]),
    }


def analyze(root, *, seeds=CANONICAL_SEEDS, short_context_path=SHORT_CONTEXT_PATH,
            expected_epochs=LONGER_BUDGET_EPOCHS, expected_patience=LONGER_BUDGET_PATIENCE,
            output_path=None) -> dict:
    """只读 10 条 long run 记录 → 判定 + 逐 seed 记录 + 统计 + context + identity checks。

    `expected_*` 默认即预注册钉死值；仅测试夹具显式传入其它值（生产调用不得覆盖）。
    """
    root = Path(root)
    seeds = tuple(seeds)
    records, identities = [], {}
    for seed in seeds:
        runs = M.find_runs(root, seed, tag=LONGER_BUDGET_TAG)
        baseline, arm = M.load_run(runs["baseline"]), M.load_run(runs["nullx"])
        identities[str(seed)] = _seed_identity(baseline, arm, seed, expected_epochs=expected_epochs,
                                               expected_patience=expected_patience)
        rec = M.seed_record(root, seed, tag=LONGER_BUDGET_TAG, cross_consistent=None)
        rec["budget"]["baseline_stopped_early"] = (
            rec["baseline"]["epoch_records"][-1]["epoch"] < baseline["config"].get("epochs", expected_epochs))
        rec["budget"]["arm_stopped_early"] = (
            rec["arm"]["epoch_records"][-1]["epoch"] < arm["config"].get("epochs", expected_epochs))
        rec["gates"] = {
            "baseline": {g: bool(baseline["gate"].get(g, {}).get("pass")) for g in GATE_KEYS},
            "arm": {g: bool(arm["gate"].get(g, {}).get("pass")) for g in GATE_KEYS}}
        records.append(rec)

    deltas_test = [rec["delta_test"] for rec in records]
    deltas_val = [rec["delta_val"] for rec in records]
    last_common = [rec["budget"]["delta_val_last_common"] for rec in records]
    top1 = [rec["arm"]["mechanism"].get("null_top1_rate") for rec in records]
    consistent = M.cross_seed_consistent(deltas_test)
    for rec in records:
        head = rec["headroom"]
        rec["headroom"] = M.seed_headroom(mechanism_active=head["mechanism_active"], delta_val=head["delta_val"],
                                          budget_right_censored=head["budget_right_censored"],
                                          budget_last_delta_positive=head["budget_last_delta_positive"],
                                          cross_seed_consistent=consistent)

    verdict = persistence_verdict(delta_test=deltas_test, delta_val=deltas_val,
                                  delta_val_last_common=last_common, null_top1=top1)
    all_identity_pass = all(all(checks.values()) for checks in identities.values())
    classification = verdict["classification"] if all_identity_pass else "INVALID"

    if len(deltas_test) >= 2:
        stats = M.checkpoint_stats(deltas_test)
    else:                                        # 干跑/单 seed：CI 未定义，如实置 None（不伪造统计量）
        stats = {"n": len(deltas_test), "mean": deltas_test[0] if deltas_test else None, "std": None,
                 "positive_count": sum(1 for d in deltas_test if d >= M.RECLASS_POSITIVE_MIN),
                 "worst_index": 0 if deltas_test else None, "worst_delta": deltas_test[0] if deltas_test else None,
                 "ci95_low": None, "ci95_high": None, "ci_method": "undefined for n<2"}
    stats["worst_seed"] = seeds[stats["worst_index"]] if stats["worst_index"] is not None else None

    context = _short_context_view(short_context_path, seeds, deltas_test, deltas_val)

    payload = {
        "spec": SPEC_PATH,
        "created": datetime.now().isoformat(timespec="seconds"),
        "criteria": {
            "epochs": expected_epochs, "patience": expected_patience, "tag": LONGER_BUDGET_TAG,
            "seeds": list(seeds), "pinned_stage1_ids": {str(k): v for k, v in PINNED_STAGE1_IDS.items()},
            "pinned_split_fingerprint": PINNED_SPLIT_FINGERPRINT,
            "V1_mean_delta_test_min": V1_MEAN_DELTA_TEST_MIN, "V2_degrade_max": V2_DEGRADE_MAX,
            "V5_top1_range": [V5_TOP1_MIN, V5_TOP1_MAX], "majority": "strict (> n/2)",
            "a_class_gates": list(A_CLASS_GATES),
        },
        "run_ids": {str(rec["model_seed"]): {"baseline": rec["baseline"]["run_id"],
                                             "arm": rec["arm"]["run_id"]} for rec in records},
        "identity_checks": {"by_seed": identities, "all_pass": all_identity_pass},
        "verdict": verdict,
        "classification": classification,
        "stats": stats,
        "seeds": records,
        "context": context,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["output_path"] = str(output_path)
    return payload


def _short_context_view(path, seeds, deltas_test, deltas_val) -> dict:
    """短预算对照视图（context only，不入判定）：逐 seed 值与差值。"""
    ctx = load_short_context(path)
    by_seed = {rec["model_seed"]: rec for rec in ctx["seeds"]}
    short, changes_test, changes_val = {}, [], []
    for i, seed in enumerate(seeds):
        rec = by_seed.get(seed)
        if rec is None:
            short[str(seed)] = None
            changes_test.append(None)
            changes_val.append(None)
            continue
        short[str(seed)] = {"delta_test": rec["delta_test"], "delta_val": rec["delta_val"],
                            "baseline_test_auc": rec["baseline"]["test_auc"], "arm_test_auc": rec["arm"]["test_auc"],
                            "classification": rec["classification"]}
        changes_test.append(deltas_test[i] - rec["delta_test"])
        changes_val.append(deltas_val[i] - rec["delta_val"])
    return {
        "note": "短预算 5-seed 检查点仅作动机与对照（context_only），不构成本实验判定结果的任何部分",
        "source": {"path": str(path), "sha256": SHORT_CONTEXT_SHA256, "multiseed_tip": MULTISEED_TIP},
        "five_epoch": short,
        "delta_test_change_by_seed": changes_test,
        "delta_val_change_by_seed": changes_val,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CensusIncome null-expert 更长预算持久性检验（只读分析；预注册：%s）" % SPEC_PATH)
    parser.add_argument("--root", type=Path, default=P.ARTIFACT_ROOT)
    parser.add_argument("--out", type=Path, default=None, help="检查点 JSON 落盘路径（缺省只打印）")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="覆盖默认五 seed 列表（干跑用）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    seeds = tuple(args.seeds) if args.seeds else CANONICAL_SEEDS
    record = analyze(args.root, seeds=seeds, output_path=args.out)
    verdict, stats = record["verdict"], record["stats"]
    print(f"[longer-budget] classification={record['classification']} checks={verdict['checks']}")
    print(f"[longer-budget] mean_delta_test={verdict['mean_delta_test']!r} "
          f"mean_last_common_delta_val={verdict['mean_last_common_delta_val']!r}")
    if stats["ci95_low"] is not None:
        print(f"[longer-budget] ci95=[{stats['ci95_low']:+.9f}, {stats['ci95_high']:+.9f}] {stats['ci_method']}")
    for rec in record["seeds"]:
        print(f"[longer-budget] seed={rec['model_seed']} delta_test={rec['delta_test']:+.9f} "
              f"delta_val={rec['delta_val']:+.9f} class={rec['classification']} "
              f"headroom={rec['headroom']['headroom_remains']} best_epoch={rec['arm']['best_epoch']}/"
              f"{rec['arm']['epoch_records'][-1]['epoch']}")
    print(f"[longer-budget] identity_all_pass={record['identity_checks']['all_pass']}")
    if args.out:
        print(f"checkpoint={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
