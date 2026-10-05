"""3-seed 聚合与终局稳定性判定（预注册 §4.4 / §4.5）。

唯一事实来源：docs/superpowers/specs/2026-10-06-census-stage2-residual-prompt-3seed-expansion-design.md

- 逐种子行：复用 `paired_verdict.paired_report`（只读两臂 run 目录；不重算层 A/层 B 判据）。
- 聚合量：mean / 样本标准差（ddof=1）/ 双侧 95% t 区间 / 单侧 95% 下界 / 符号与正提升计数 /
  最差种子 / 机制通过计数 / 验证方向一致性。
- 终局判定 S1–S8 与 verdict/decision（判据常量先于结果冻结，由测试钉死）。

本工具不加载权重、不读数据、不重训；不修改任何协议/迁移文件。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from census_benchmark import paired_verdict as PV

# ---- 冻结的三种子与预期 Stage-1 身份（预注册 §1.3 / §3.2）----
CANONICAL_SEEDS = (1685480945, 1685463909, 1685477428)
EXPECTED_STAGE1_IDS = {
    1685480945: "s1-096f8f16-m1685480945-e2-cb2094b3",
    1685463909: "s1-096f8f16-m1685463909-e2-8b53a3bf",
    1685477428: "s1-096f8f16-m1685477428-e2-15eabcb4",
}
# ---- 冻结的统计常量（预注册 §4.4；t 分布 df=2 分位点，scipy 实测值）----
T_TWO_SIDED_95_DF2 = 4.302652729696142
T_ONE_SIDED_95_DF2 = 2.919985580355516
# ---- 冻结的终局判定常量（预注册 §4.5）----
MEAN_MIN = 0.001
WORST_FLOOR = -0.001
SIGN_POSITIVE_REQUIRED = 3
PI_REQUIRED = 2
VAL_DIRECTION_REQUIRED = 3
MECHANISM_REQUIRED = 3
INTEGRITY_GATES = ("A1", "A2", "A4", "A5", "B1", "B2", "B4")
# ---- 臂身份（与迁移的机制模块常量一致；测试钉死）----
VARIANT_BASELINE = "baseline"
VARIANT_TREATMENT = "residual-prompt"
# ---- 终局词汇（预注册 §4.5）----
STABLE_UTILITY = "STABLE_UTILITY"
UTILITY_SEED_UNSTABLE = "UTILITY_SEED_UNSTABLE"
REGISTER_NEXT_VALIDATION = "REGISTER_NEXT_VALIDATION"
CLOSE_LINE = "CLOSE_LINE"
PREREG_PATH = "docs/superpowers/specs/2026-10-06-census-stage2-residual-prompt-3seed-expansion-design.md"

_RUN_ID_SEED_RE = re.compile(r"-m(\d+)-")


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def _check_run_id_seed(run: dict, seed: int) -> None:
    match = _RUN_ID_SEED_RE.search(run.get("run_id") or "")
    if match is None or int(match.group(1)) != seed:
        raise ValueError(f"run_id 与 seed 不匹配: seed={seed} run_id={run.get('run_id')!r}")


def seed_row(seed: int, baseline_run, treatment_run) -> dict:
    """单种子行：配对差值 + 层 A/层 B 只读汇总 + 完整性标志。"""
    baseline_run, treatment_run = Path(baseline_run), Path(treatment_run)
    report = PV.paired_report(baseline_run, treatment_run)
    base, treat = report["baseline"], report["treatment"]
    _check_run_id_seed(base, seed)
    _check_run_id_seed(treat, seed)
    # 基线臂 variant：历史 run（variant 字段引入之前，如 seed 1685480945 的 904f8d0）缺字段 ⇒ None，
    # 语义上即 baseline（paired_verdict._load_run 同样用 .get 读取）；处理臂必须显式 residual-prompt。
    if base["variant"] not in (None, VARIANT_BASELINE):
        raise ValueError(f"基线臂 variant 应为 {VARIANT_BASELINE!r} 或历史缺省 None: {base['variant']!r}")
    if treat["variant"] != VARIANT_TREATMENT:
        raise ValueError(f"处理臂 variant 应为 {VARIANT_TREATMENT!r}: {treat['variant']!r}")

    rp_arm = _load_json(treatment_run / "metrics.json")["rp_arm"]
    g1_g5 = {f"G{i}": bool(rp_arm[f"G{i}"]["pass"]) for i in range(1, 6)}
    integrity = all(base["gates"][key] and treat["gates"][key] for key in INTEGRITY_GATES) \
        and base["b3_gate_mean"] == treat["b3_gate_mean"]
    delta_test = report["paired"]["delta_test"]
    delta_val = report["paired"]["delta_val_best"]

    row = {
        "seed": seed,
        "stage1_id": base["stage1_id"],
        "baseline": {key: base[key] for key in ("run_id", "run_dir", "commit", "variant", "auc_test",
                                                "best_val_auc", "best_epoch", "epochs_run",
                                                "gates", "b3_gate_mean")},
        "treatment": {key: treat[key] for key in ("run_id", "run_dir", "commit", "variant", "auc_test",
                                                  "best_val_auc", "best_epoch", "epochs_run",
                                                  "gates", "b3_gate_mean")},
        "delta_test": delta_test,
        "delta_val_best": delta_val,
        "classification_new": report["paired"]["classification_new"],
        "historical": {"classification": report["paired"]["historical_classification"],
                       "e1_pass": report["paired"]["historical_e1_pass"],
                       "g1_g5_all_pass": report["paired"]["historical_g1_g5_all_pass"]},
        "g1_g5_pass": g1_g5,
        "mechanism_all_pass": all(g1_g5.values()),
        "integrity_all_pass": integrity,
        "sign_match_test_val": _sign(delta_test) == _sign(delta_val),
        "alpha_final": _optional_treatment_diagnostic(treatment_run, "alpha_final"),
        "ratio_mean": _optional_treatment_diagnostic(treatment_run, "ratio_mean"),
    }
    return row


def _optional_treatment_diagnostic(treatment_run: Path, key: str):
    """诊断量（只读、可缺失）：alpha_final 取 prompt_report.json；ratio_mean 取 metrics.json 机制段首流。"""
    try:
        if key == "alpha_final":
            return float(_load_json(treatment_run / "prompt_report.json")["alpha_final"])
        if key == "ratio_mean":
            prompt = _load_json(treatment_run / "metrics.json")["mechanism"]["prompt"]
            return float(prompt["val_stats"]["ratio_mean"][0])
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return None
    return None


def aggregate(rows: list[dict]) -> dict:
    """3-seed 聚合与 S1–S8 终局判定（预注册 §4.4 / §4.5）。"""
    deltas = [row["delta_test"] for row in rows]
    n = len(deltas)
    mean = sum(deltas) / n
    sd = math.sqrt(sum((delta - mean) ** 2 for delta in deltas) / (n - 1)) if n > 1 else 0.0
    root_n = math.sqrt(n)
    ci_half = T_TWO_SIDED_95_DF2 * sd / root_n
    worst = min(rows, key=lambda row: row["delta_test"])
    stats = {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_two_sided_95": [mean - ci_half, mean + ci_half],
        "lb_one_sided_95": mean - T_ONE_SIDED_95_DF2 * sd / root_n,
        "sign_positive_count": sum(1 for delta in deltas if delta > 0),
        "positive_improvement_count": sum(1 for delta in deltas if delta >= PV.DELTA_POSITIVE),
        "worst_seed": worst["seed"],
        "worst_delta": worst["delta_test"],
        "mechanism_pass_count": sum(1 for row in rows if row["mechanism_all_pass"]),
        "val_direction_consistency": sum(1 for row in rows if row["sign_match_test_val"]),
        "val_sign_table": [{"seed": row["seed"], "delta_test_sign": _sign(row["delta_test"]),
                            "delta_val_sign": _sign(row["delta_val_best"])} for row in rows],
    }
    conditions = {
        "S1": stats["sign_positive_count"] == SIGN_POSITIVE_REQUIRED,
        "S2": stats["positive_improvement_count"] >= PI_REQUIRED,
        "S3": stats["worst_delta"] > WORST_FLOOR,
        "S4": stats["mean"] >= MEAN_MIN,
        "S5": stats["lb_one_sided_95"] > 0,
        "S6": stats["val_direction_consistency"] == VAL_DIRECTION_REQUIRED,
        "S7": stats["mechanism_pass_count"] == MECHANISM_REQUIRED,
        "S8": all(row["integrity_all_pass"] for row in rows),
    }
    stable = all(conditions.values())
    return {
        "stats": stats,
        "S": conditions,
        "S_failed": [key for key, ok in conditions.items() if not ok],
        "verdict": STABLE_UTILITY if stable else UTILITY_SEED_UNSTABLE,
        "decision": REGISTER_NEXT_VALIDATION if stable else CLOSE_LINE,
    }


def three_seed_report(pairs: dict) -> dict:
    """主入口：`pairs` = {seed: (baseline_run_dir, treatment_run_dir)}，必须恰为三种子。"""
    if set(pairs) != set(CANONICAL_SEEDS):
        raise ValueError(f"必须恰为三个 canonical seed {CANONICAL_SEEDS}，收到 {sorted(pairs)}")
    rows = []
    for seed in CANONICAL_SEEDS:
        baseline_run, treatment_run = pairs[seed]
        row = seed_row(seed, baseline_run, treatment_run)
        if row["stage1_id"] != EXPECTED_STAGE1_IDS[seed]:
            raise ValueError(f"seed {seed} 的 stage1_id {row['stage1_id']!r} != 预期 "
                             f"{EXPECTED_STAGE1_IDS[seed]!r}（产物溯源失败）")
        rows.append(row)
    if len({row["stage1_id"] for row in rows}) != len(rows):
        raise ValueError("三个种子的 stage1_id 非互异（run 目录配错？）")
    return {
        "prereg": PREREG_PATH,
        "constants": {"T_TWO_SIDED_95_DF2": T_TWO_SIDED_95_DF2,
                      "T_ONE_SIDED_95_DF2": T_ONE_SIDED_95_DF2,
                      "MEAN_MIN": MEAN_MIN, "WORST_FLOOR": WORST_FLOOR,
                      "PI_THRESHOLD": PV.DELTA_POSITIVE},
        "seeds": rows,
        **aggregate(rows),
    }


def parse_pair_arg(text: str) -> tuple[int, Path, Path]:
    """解析 `--pair <seed>:<baseline_dir>,<treatment_dir>`（Windows 路径的盘符冒号安全）。"""
    seed_text, rest = text.split(":", 1)
    baseline_text, treatment_text = rest.rsplit(",", 1)
    return int(seed_text), Path(baseline_text), Path(treatment_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3-seed 聚合与终局稳定性判定（预注册 §4.4/§4.5）")
    parser.add_argument("--pair", action="append", required=True, metavar="SEED:BASELINE,TREATMENT",
                        help="格式：<seed>:<baseline_run_dir>,<treatment_run_dir>；三个 canonical seed 各一次")
    parser.add_argument("--out", type=Path, default=None, help="可选：写入 JSON 报告路径")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    pairs: dict[int, tuple[Path, Path]] = {}
    for text in args.pair:
        seed, baseline_run, treatment_run = parse_pair_arg(text)
        if seed in pairs:
            raise SystemExit(f"重复的 seed: {seed}")
        pairs[seed] = (baseline_run, treatment_run)
    report = three_seed_report(pairs)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
