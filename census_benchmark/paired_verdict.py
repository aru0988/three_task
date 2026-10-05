"""配对判定工具：从两条 Stage-2 run 目录计算配对差、新增分类与决策。

唯一事实来源：docs/superpowers/specs/2026-10-06-census-stage2-residual-prompt-seed-recheck-design.md
§4.2（新增分类）/ §4.3（决策与实质矛盾）。

只读两个 run 目录里的 `metrics.json` 与 `gate_report.json`；不加载权重、不触碰数据、不重训。
历史判据（E1 / G1–G5 / 结局分类）由迁移代码写好在 `metrics.json:rp_arm` 里，本工具**只读汇总**，
不重算、不覆盖；新增分类与其并列报告。

用法：
    python -m census_benchmark.paired_verdict --baseline-run <dir> --treatment-run <dir> [--out <json>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---- 新增分类（预注册 §4.2；用户指定；看到结果之前写定，测试钉死）----
DELTA_POSITIVE = 0.001            # delta_test >= +0.001 → POSITIVE_IMPROVEMENT（含端点）
DELTA_NEGATIVE = -0.02            # delta_test <= -0.02 → CLEAR_DEGRADATION（含端点）
POSITIVE_IMPROVEMENT = "POSITIVE_IMPROVEMENT"
NO_CLEAR_IMPROVEMENT = "NO_CLEAR_IMPROVEMENT"
CLEAR_DEGRADATION = "CLEAR_DEGRADATION"
# ---- 实质矛盾（预注册 §4.3）----
VAL_CONTRADICTION = -0.001        # (b) delta_val_best <= -0.001 → 与 test 正号矛盾
BAD_HISTORICAL = ("INVALID_IMPLEMENTATION", "MECHANISM_INACTIVE",
                  "MECHANISM_SILENT", "MECHANISM_OVER_PERTURB")   # (a)
GATE_KEYS = ("A1", "A2", "A4", "A5", "B1", "B2", "B4")            # (c)（B3 单独比对）


def classify_delta(delta_test: float) -> str:
    """预注册 §4.2 的三档分类（端点含入）。"""
    if delta_test >= DELTA_POSITIVE:
        return POSITIVE_IMPROVEMENT
    if delta_test <= DELTA_NEGATIVE:
        return CLEAR_DEGRADATION
    return NO_CLEAR_IMPROVEMENT


def _load_run(run_dir: Path, *, need_rp_arm: bool) -> dict:
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "gate_report.json").read_text(encoding="utf-8"))
    if need_rp_arm and "rp_arm" not in metrics:
        raise ValueError(f"处理臂 run 缺少 rp_arm（非 residual-prompt 臂？）: {run_dir}")
    s2 = metrics["stage2"]
    out = {
        "run_dir": str(run_dir), "run_id": metrics.get("run_id"), "commit": metrics.get("commit"),
        "variant": metrics.get("variant"), "stage1_id": metrics.get("stage1_id"),
        "auc_test": float(s2["test_auc"]), "best_val_auc": float(s2["best_val_auc"]),
        "best_epoch": int(s2["best_epoch"]), "epochs_run": len(s2["epoch_records"]),
        "gates": {key: bool(report[key]["pass"]) for key in GATE_KEYS},
        "b3_gate_mean": report["B3"]["detail"]["gate_mean"],
    }
    out["budget_capped"] = out["best_epoch"] == out["epochs_run"]
    if "rp_arm" in metrics:
        out["rp_arm"] = metrics["rp_arm"]
    return out


def paired_report(baseline_run, treatment_run) -> dict:
    """配对判定：delta、新增分类、历史判据（只读）、实质矛盾、决策。"""
    base = _load_run(Path(baseline_run), need_rp_arm=False)
    treat = _load_run(Path(treatment_run), need_rp_arm=True)
    if base["stage1_id"] != treat["stage1_id"]:
        raise ValueError(f"两臂 stage1_id 不一致，非有效配对: {base['stage1_id']} vs {treat['stage1_id']}")

    delta_test = treat["auc_test"] - base["auc_test"]
    delta_val = treat["best_val_auc"] - base["best_val_auc"]
    classification_new = classify_delta(delta_test)

    arm = treat["rp_arm"]
    g_all = all(bool(arm[f"G{i}"]["pass"]) for i in range(1, 6))
    historical = {"classification": arm.get("classification"), "e1_pass": bool(arm["E1"]["pass"]),
                  "g1_g5_all_pass": g_all}

    contradictions: list[str] = []
    if historical["classification"] in BAD_HISTORICAL or not g_all:                 # (a)
        contradictions.append(f"(a) 历史结局 {historical['classification']!r}（G1–G5 未全过）")
    if delta_val <= VAL_CONTRADICTION:                                              # (b)
        contradictions.append(f"(b) delta_val_best {delta_val:.6f} <= {VAL_CONTRADICTION}")
    failed = [key for key, ok in {**base["gates"], **treat["gates"]}.items() if not ok]  # (c)
    if failed:
        contradictions.append(f"(c) 门禁 FAIL: {failed}")
    if base["b3_gate_mean"] != treat["b3_gate_mean"]:
        contradictions.append(f"(c) B3 gate_mean 两臂不一致: {base['b3_gate_mean']} vs {treat['b3_gate_mean']}")

    decision = "EXPAND" if (classification_new == POSITIVE_IMPROVEMENT and not contradictions) else "STOP"
    return {
        "baseline": {key: value for key, value in base.items() if key != "rp_arm"},
        "treatment": {key: value for key, value in treat.items() if key != "rp_arm"},
        "paired": {"delta_test": delta_test, "delta_val_best": delta_val,
                   "classification_new": classification_new,
                   "historical_classification": historical["classification"],
                   "historical_e1_pass": historical["e1_pass"],
                   "historical_g1_g5_all_pass": g_all},
        "contradictions": contradictions,
        "decision": decision,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="配对判定：新增分类 + 历史判据（只读）+ 决策")
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--treatment-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="可选：写入 JSON 报告路径")
    args = parser.parse_args(argv)
    report = paired_report(args.baseline_run, args.treatment_run)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
