"""AliCCP canonical 五 seed 配对汇总与二级分类（纯函数 + CLI）。

事实来源：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-five-seed-design.md §5/§6。
只读 run JSON（不训练、不触发 GPU/数据集）。本模块**不修改任何判定 machinery**：
历史 U1/U2（`+0.0055` / `>0`）与三标签分类原样取自 run 产物 `metrics.json:rp_arm`；
本模块只**新增**二级分类（§5.2）、NO_CLEAR_IMPROVEMENT 的机械 headroom 评估（§5.2）、
五 seed 汇总统计（§6.3）与 20-epoch 立项条件（§6.4）。判据常量先于结果写死，不得事后修改。

CLI：
    python -m aliccp_benchmark.five_seed --root artifacts/aliccp_bench \
        --pairs artifacts/aliccp_bench/audit/five-seed/pairs.json \
        --out artifacts/aliccp_bench/audit/five-seed/five_seed_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---- 预注册常量（§5.2/§6；看到结果前写死）----
CANONICAL_SEEDS = (1688723512, 1688723740, 1688738016, 1688749593, 1688762746)
SECONDARY_POSITIVE_MIN = 0.001          # Δtest ≥ +0.001 ⇒ POSITIVE_IMPROVEMENT
SECONDARY_DEGRADATION_MAX = -0.02       # Δtest ≤ −0.02 ⇒ CLEAR_DEGRADATION（之间为 NO_CLEAR）
HISTORIC_U1_MIN = 0.0055                # 历史 U1 阈值（不改写；仅用于计数报告）
T_975_DF4 = 2.7764451051977987          # scipy.stats.t.ppf(0.975, 4)（n=5、df=4、双侧 95%）
TWENTY_EPOCH_MEAN_MIN = 0.0055          # §6.4 C3
TWENTY_EPOCH_MIN_POSITIVE = 4           # §6.4 C5
TWENTY_EPOCH_MIN_VAL_POSITIVE = 4       # §6.4 C6

_MECHANISM_KEYS = ("M0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
_A_GATE_KEYS = ("A1", "A2", "A3", "A4", "A5", "A6")


def secondary_classification(delta_test: float) -> str:
    """§5.2 二级分类（闭开边界：≥+0.001 正；≤−0.02 退化；之间无清晰改善）。"""
    if delta_test >= SECONDARY_POSITIVE_MIN:
        return "POSITIVE_IMPROVEMENT"
    if delta_test <= SECONDARY_DEGRADATION_MAX:
        return "CLEAR_DEGRADATION"
    return "NO_CLEAR_IMPROVEMENT"


def mean_std(values):
    """(mean, 样本标准差 ddof=1)；n<2 时 std=None。"""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def t_ci(values, t: float = T_975_DF4):
    """Student-t 双侧 95% CI：mean ± t·s/√n（s 为样本标准差 ddof=1）。"""
    mean, std = mean_std(values)
    if std is None:
        return mean, mean
    half = t * std / (len(values) ** 0.5)
    return mean - half, mean + half


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _same_sign_count(signs, sign: int) -> int:
    return sum(1 for s in signs if s == sign)


def headroom_assessment(record: dict, *, same_sign_count: int, n_seeds: int) -> dict:
    """§5.2：NO_CLEAR_IMPROVEMENT 的机械 headroom 评估（四因子 + 距正分类余量）。"""
    active = (abs(float(record["alpha_final"])) > 0.0
              and float(record["generator_grad_last_epoch"]) > 0.0
              and float(record["geff"]["geff_std"]) > 0.0)
    return {
        "mechanism_activity": "ACTIVE" if active else "INACTIVE",
        "validation_direction": "POSITIVE" if float(record["delta_val"]) > 0.0 else "NON_POSITIVE",
        "cross_seed_consistency": "CONSISTENT" if same_sign_count >= n_seeds - 1 else "INCONSISTENT",
        "epoch_trajectory": ("RIGHT_CENSORED_STILL_IMPROVING"
                             if int(record["best_epoch"]) == int(record["epochs_run"]) else "EARLY_STOPPED"),
        "gap_to_positive_threshold": SECONDARY_POSITIVE_MIN - float(record["delta_test"]),
        "cross_seed_same_sign_count": int(same_sign_count),
    }


def aggregate(records: list[dict]) -> dict:
    """§6.3 汇总统计（全部报告值；机械计算；跨 seed 不混）。"""
    n = len(records)
    delta_test = [float(r["delta_test"]) for r in records]
    delta_val = [float(r["delta_val"]) for r in records]
    signs_test = [_sign(v) for v in delta_test]
    signs_val = [_sign(v) for v in delta_val]

    per_seed = []
    for record, sign in zip(records, signs_test):
        entry = dict(record)
        same_sign_count = _same_sign_count(signs_test, sign)
        entry["cross_seed_same_sign_count"] = same_sign_count
        if entry["secondary"] == "NO_CLEAR_IMPROVEMENT":
            entry["headroom"] = headroom_assessment(entry, same_sign_count=same_sign_count, n_seeds=n)
        else:
            entry["headroom"] = None
        per_seed.append(entry)

    mean_test, std_test = mean_std(delta_test)
    low_test, high_test = t_ci(delta_test)
    mean_val, std_val = mean_std(delta_val)
    low_val, high_val = t_ci(delta_val)
    worst_idx = min(range(n), key=lambda i: delta_test[i])

    def _seed_range(getter):
        cols = list(zip(*[getter(r) for r in records]))
        return [[min(col), max(col)] for col in cols]

    mech_pass = [bool(r["mechanism_pass"]) for r in records]
    a_pass = [bool(r["a_class_pass"]) for r in records]
    alpha_signs = {"positive": 0, "negative": 0, "zero": 0}
    for r in records:
        s = _sign(float(r["alpha_final"]))
        alpha_signs["positive" if s > 0 else "negative" if s < 0 else "zero"] += 1

    return {
        "n_seeds": n,
        "per_seed": per_seed,
        "delta_test": {"values": delta_test, "mean": mean_test, "std": std_test, "t": T_975_DF4,
                       "ci95_low": low_test, "ci95_high": high_test,
                       "min": delta_test[worst_idx], "min_model_seed": records[worst_idx]["model_seed"],
                       "min_arm_run": records[worst_idx]["arm_run"]},
        "delta_val": {"values": delta_val, "mean": mean_val, "std": std_val, "t": T_975_DF4,
                      "ci95_low": low_val, "ci95_high": high_val,
                      "min": min(delta_val), "max": max(delta_val)},
        "counts": {
            "delta_test_ge_0.001": sum(1 for v in delta_test if v >= SECONDARY_POSITIVE_MIN),
            "delta_test_ge_0.0055": sum(1 for v in delta_test if v >= HISTORIC_U1_MIN),
            "delta_val_gt_0": sum(1 for v in delta_val if v > 0.0),
        },
        "sign_consistency": {
            "delta_test_signs": signs_test, "delta_val_signs": signs_val,
            "delta_test_same_sign_count": max(_same_sign_count(signs_test, s) for s in (-1, 0, 1)),
            "delta_val_same_sign_count": max(_same_sign_count(signs_val, s) for s in (-1, 0, 1)),
            "delta_test_all_positive": all(s > 0 for s in signs_test),
            "delta_val_all_positive": all(s > 0 for s in signs_val),
        },
        "secondary_counts": {
            "POSITIVE_IMPROVEMENT": sum(1 for r in records if r["secondary"] == "POSITIVE_IMPROVEMENT"),
            "NO_CLEAR_IMPROVEMENT": sum(1 for r in records if r["secondary"] == "NO_CLEAR_IMPROVEMENT"),
            "CLEAR_DEGRADATION": sum(1 for r in records if r["secondary"] == "CLEAR_DEGRADATION"),
        },
        "mechanism_stability": {
            "n_mechanism_pass": sum(1 for m in mech_pass if m),
            "n_full_mechanism_pass": sum(1 for m, a in zip(mech_pass, a_pass) if m and a),
            "alpha_final_signs": alpha_signs,
            "ratio_mean_seed_range": _seed_range(lambda r: [float(v) for v in r["ratio_mean"]]),
            "cos_mean_seed_range": _seed_range(lambda r: [float(v) for v in r["cos_mean"]]),
            "geff_std_range": [min(float(r["geff"]["geff_std"]) for r in records),
                               max(float(r["geff"]["geff_std"]) for r in records)],
            "pred_std_over_ref_range": [min(float(r["pred_std"]) / float(r["ref_pred_std"]) for r in records),
                                        max(float(r["pred_std"]) / float(r["ref_pred_std"]) for r in records)],
        },
    }


def twenty_epoch_condition(agg: dict) -> dict:
    """§6.4 20-epoch 立项条件（C1–C6 机械真值表；全满足才 SATISFIED）。"""
    n = agg["n_seeds"]
    c1 = agg["mechanism_stability"]["n_full_mechanism_pass"] == n
    c2 = agg["secondary_counts"]["CLEAR_DEGRADATION"] == 0
    c3 = agg["delta_test"]["mean"] >= TWENTY_EPOCH_MEAN_MIN
    c4 = agg["delta_test"]["ci95_low"] > 0.0
    c5 = agg["counts"]["delta_test_ge_0.001"] >= TWENTY_EPOCH_MIN_POSITIVE
    c6 = agg["counts"]["delta_val_gt_0"] >= TWENTY_EPOCH_MIN_VAL_POSITIVE
    clauses = {"C1_mechanism": c1, "C2_no_degradation": c2, "C3_mean_at_least_0.0055": c3,
               "C4_ci_low_positive": c4, "C5_positive_count_ge_4": c5, "C6_val_positive_count_ge_4": c6}
    return {
        "status": ("TWENTY_EPOCH_CONDITION_SATISFIED" if all(clauses.values())
                   else "TWENTY_EPOCH_CONDITION_NOT_SATISFIED"),
        **clauses,
        "thresholds": {"mean_min": TWENTY_EPOCH_MEAN_MIN, "positive_min": TWENTY_EPOCH_MIN_POSITIVE,
                       "val_positive_min": TWENTY_EPOCH_MIN_VAL_POSITIVE, "t": T_975_DF4},
    }


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_record(root, *, model_seed: int, baseline_run: str, arm_run: str) -> dict:
    """从 run JSON 装配一条配对记录；配对身份不闭合即响亮失败（AssertionError）。"""
    root = Path(root)
    base = _read_json(root / "runs" / baseline_run / "metrics.json")
    arm = _read_json(root / "runs" / arm_run / "metrics.json")
    gate = _read_json(root / "runs" / arm_run / "gate_report.json")
    prompt = _read_json(root / "runs" / arm_run / "prompt_report.json")

    assert int(base["model_seed"]) == int(model_seed), "配对基线 model_seed 不符"
    assert int(arm["model_seed"]) == int(model_seed), "处理臂 model_seed 不符"
    assert base["stage1_id"] == arm["stage1_id"], "配对 stage1_id 不符（跨产物比较被禁止）"
    assert arm.get("variant") == "residual-prompt", "处理臂 variant 非 residual-prompt"

    rp_arm = arm["rp_arm"]
    base_test = float(base["test_auc_bsi"])
    base_val = float(base["best_val_auc_bsi"])
    observed_test = float(rp_arm["U"]["U1"]["observed"]["baseline_auc_test"])
    observed_val = float(rp_arm["U"]["U2"]["observed"]["baseline_auc_val"])
    assert observed_test == base_test, f"run-reference test 不闭合: {observed_test!r} != {base_test!r}"
    assert observed_val == base_val, f"run-reference val 不闭合: {observed_val!r} != {base_val!r}"

    grad_probe = prompt["grad_probe"]
    record = {
        "model_seed": int(model_seed),
        "stage1_id": arm["stage1_id"],
        "baseline_run": baseline_run,
        "arm_run": arm_run,
        "baseline_test_auc": base_test,
        "baseline_val_auc": base_val,
        "arm_test_auc": float(arm["test_auc_bsi"]),
        "arm_val_auc": float(arm["best_val_auc_bsi"]),
        "delta_test": float(arm["test_auc_bsi"]) - base_test,
        "delta_val": float(arm["best_val_auc_bsi"]) - base_val,
        "u1": bool(rp_arm["U"]["U1"]["pass"]),
        "u2": bool(rp_arm["U"]["U2"]["pass"]),
        "classification": rp_arm["classification"],
        "secondary": secondary_classification(float(arm["test_auc_bsi"]) - base_test),
        "mechanism_pass": all(bool(rp_arm[k]["pass"]) for k in _MECHANISM_KEYS),
        "a_class_pass": all(gate["gates"][k]["verdict"] in ("PASS", "SKIP") for k in _A_GATE_KEYS),
        "alpha_final": float(prompt["alpha_final"]),
        "generator_grad_last_epoch": float(grad_probe[-1]["generator_grad_norm"]),
        "ratio_mean": [float(v) for v in prompt["val_stats"]["ratio_mean"]],
        "ratio_std": [float(v) for v in prompt["val_stats"]["ratio_std"]],
        "cos_mean": [float(v) for v in prompt["val_stats"]["cos_mean"]],
        "geff": dict(prompt["gate"]),
        "pred_std": float(prompt["dispersion"]["pred_std"]),
        "ref_pred_std": float(prompt["reference_dispersion"]["pred_dispersion"]["pred_std"]),
        "best_epoch": int(arm["best_epoch"]),
        "epochs_run": int(arm["epochs"]),
        "patience": int(arm["patience"]),
        "per_epoch_val": [float(e["val_auc_bsi"]) for e in arm["per_epoch"]],
    }
    return record


def load_pairs(root, pairs: list[dict]) -> list[dict]:
    """按 pairs 逐条 load_record；seed 集合必须恰为 canonical 五 seed（顺序按 canonical）。"""
    root = Path(root)
    by_seed = {int(p["model_seed"]): p for p in pairs}
    assert set(by_seed) == set(CANONICAL_SEEDS), f"配对集合必须恰为 canonical 五 seed: {sorted(by_seed)}"
    return [load_record(root, model_seed=seed,
                        baseline_run=by_seed[seed]["baseline_run"], arm_run=by_seed[seed]["arm_run"])
            for seed in CANONICAL_SEEDS]


def build_report(root, pairs: list[dict]) -> dict:
    records = load_pairs(root, pairs)
    agg = aggregate(records)
    return {"pairs": pairs, "records": records, "aggregate": agg,
            "twenty_epoch_condition": twenty_epoch_condition(agg)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AliCCP 五 seed 配对汇总（预注册 §5/§6；只读 JSON）")
    parser.add_argument("--root", type=str, default="artifacts/aliccp_bench", help="bench root（含 runs/）")
    parser.add_argument("--pairs", type=str, required=True,
                        help='配对 JSON：[{"model_seed":..,"baseline_run":..,"arm_run":..}, ...]（恰 5 条）')
    parser.add_argument("--out", type=str, required=True, help="汇总报告输出路径（JSON）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
    report = build_report(args.root, pairs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    agg = report["aggregate"]
    print(f"delta_test mean={agg['delta_test']['mean']!r} std={agg['delta_test']['std']!r} "
          f"CI95=[{agg['delta_test']['ci95_low']!r}, {agg['delta_test']['ci95_high']!r}]")
    print(f"delta_val  mean={agg['delta_val']['mean']!r} std={agg['delta_val']['std']!r} "
          f"CI95=[{agg['delta_val']['ci95_low']!r}, {agg['delta_val']['ci95_high']!r}]")
    print(f"counts={agg['counts']} secondary={agg['secondary_counts']}")
    print(f"worst seed={agg['delta_test']['min_model_seed']} delta_test={agg['delta_test']['min']!r}")
    print(f"20-epoch condition: {report['twenty_epoch_condition']['status']}")
    print(f"报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
