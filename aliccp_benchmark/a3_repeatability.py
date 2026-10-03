"""A3 同配置复跑判定器（spec §10-A3；预注册见
docs/superpowers/specs/2026-10-04-aliccp-a3-repeatability-calibration-design.md §6/§8）。

只读既有 run 目录（config.json / metrics.json / gate_report.json / newtask.pt），
纯函数 + `python -m` CLI；不改任何管线文件，对 run 产物零写入。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

# spec §10-A3：两次 AUC-Test-BSI 差异 ≤ 1e-9（边界含等于）
A3_TOL = 1e-9
# 身份键子集（预注册 §6.1）；记录/溯源字段（run_id/commit/git/wall_seconds/spec_attenuation/variant 等）不参与
IDENTITY_KEYS = (
    "stage1_id", "tag", "model_seed", "budgets", "batch_size", "lr", "reg_dnn",
    "newtask_rep_dim", "expert_hidden", "tower_hidden", "input_size", "embedding_size", "enforce_b",
)
IDENTITY_METRIC_KEYS = ("epochs", "patience")
SHA_FIELDS = ("env_ids_sha256", "fingerprint_sha256")
A_CLASS_GATES = ("A1", "A2", "A4", "A5", "A6")

_EXIT_CODES = {"A3_PASS": 0, "A3_FAIL": 1, "INVALID": 2}


def _read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(run_dir) -> dict:
    """读取单个 run 的四个产物文件；任一缺失抛 FileNotFoundError（工具性无效信号）。"""
    run_dir = Path(run_dir)
    config = _read_json(run_dir / "config.json")
    return {
        "run_dir": str(run_dir),
        "config": config,
        "metrics": _read_json(run_dir / "metrics.json"),
        "gate_report": _read_json(run_dir / "gate_report.json"),
        "newtask_sha256": sha256_file(run_dir / "newtask.pt"),
        "run_id": config.get("run_id", run_dir.name),
    }


def check_config_identity(runs: list) -> dict:
    """预注册 §6.1 身份前提：identity 键子集 + 两个哈希字段必须全同（记录字段除外）。"""
    diffs = []
    for key in IDENTITY_KEYS:
        values = [run["config"].get(key) for run in runs]
        if any(value != values[0] for value in values[1:]):
            diffs.append({"field": f"config.{key}", "values": values})
    for key in IDENTITY_METRIC_KEYS:
        values = [run["metrics"].get(key) for run in runs]
        if any(value != values[0] for value in values[1:]):
            diffs.append({"field": f"metrics.{key}", "values": values})
    for key in SHA_FIELDS:
        values = [run["metrics"].get(key) for run in runs]
        if any(value != values[0] for value in values[1:]):
            diffs.append({"field": f"metrics.{key}", "values": values})
    return {"pass": not diffs, "diffs": diffs}


def check_a_class(runs: list) -> dict:
    """预注册 §6.1：每 run 的 A1/A2/A4/A5/A6 记录必须全 PASS（A3 行内为 SKIP，不参与）。"""
    per_run = {}
    for run in runs:
        gates = run["gate_report"].get("gates", {})
        per_run[run["run_id"]] = {gate: gates.get(gate, {}).get("verdict") for gate in A_CLASS_GATES}
    ok = all(verdict == "PASS" for verdicts in per_run.values() for verdict in verdicts.values())
    return {"pass": ok, "per_run": per_run}


def a3_verdict(runs: list) -> dict:
    """预注册 §6.2：Δ 项（成对 |ΔAUC-Test-BSI| ≤ A3_TOL）∧ 哈希项（backbone sha 三同 + 逐 run 稳定）。"""
    tests = [run["metrics"]["test_auc_bsi"] for run in runs]
    pairs = []
    for i, j in itertools.combinations(range(len(runs)), 2):
        delta = abs(tests[i] - tests[j])
        pairs.append({"pair": [runs[i]["run_id"], runs[j]["run_id"]],
                      "abs_delta_test_bsi": delta, "within_tol": delta <= A3_TOL})
    loaded = [run["metrics"]["backbone_sha256_loaded"] for run in runs]
    backbone_sha_equal = all(sha == loaded[0] for sha in loaded)
    backbone_sha_stable_within_run = all(
        run["metrics"]["backbone_sha256_loaded"] == run["metrics"]["backbone_sha256_before"]
        == run["metrics"]["backbone_sha256_after"] for run in runs
    )
    return {
        "pass": bool(all(p["within_tol"] for p in pairs) and backbone_sha_equal and backbone_sha_stable_within_run),
        "tol": A3_TOL,
        "max_abs_delta_test_bsi": max(p["abs_delta_test_bsi"] for p in pairs),
        "pairs": pairs,
        "backbone_sha_equal": backbone_sha_equal,
        "backbone_sha_stable_within_run": backbone_sha_stable_within_run,
        "backbone_sha256": loaded,
    }


def analyze(runs: list) -> dict:
    """完整报告：前提（身份/A 类）→ 判定（§6.2）→ 补充指标（§6.3，不入判定）。"""
    if len(runs) < 2:
        raise ValueError(f"A3 判定至少需要 2 个 run（给定 {len(runs)} 个）")
    identity = check_config_identity(runs)
    a_class = check_a_class(runs)
    verdict = a3_verdict(runs)

    val_aucs = [run["metrics"]["best_val_auc_bsi"] for run in runs]
    tests = [run["metrics"]["test_auc_bsi"] for run in runs]
    trajectories = [
        [(e["train_loss"], e["val_auc_bsi"]) for e in run["metrics"]["per_epoch"]] for run in runs
    ]
    gate_means = [list(run["metrics"]["gate_mean"]) for run in runs]
    newtask_shas = [run["newtask_sha256"] for run in runs]

    def _max_pairwise(values):
        return max(abs(a - b) for a, b in itertools.combinations(values, 2)) if len(values) > 1 else 0.0

    supplementary = {
        "max_abs_delta_val_bsi": _max_pairwise(val_aucs),
        "max_abs_delta_test_bsi": _max_pairwise(tests),
        "max_abs_delta_gate_mean": max(
            (abs(a - b) for xs, ys in itertools.combinations(gate_means, 2) for a, b in zip(xs, ys)),
            default=0.0,
        ),
        "trajectories_bit_equal": all(t == trajectories[0] for t in trajectories[1:]),
        "checkpoint_bit_equal": len(set(newtask_shas)) == 1,
        "gate_mean_bit_equal": all(g == gate_means[0] for g in gate_means[1:]),
        "env_ids_sha256": [run["metrics"]["env_ids_sha256"] for run in runs],
        "fingerprint_sha256": [run["metrics"]["fingerprint_sha256"] for run in runs],
        "newtask_sha256": newtask_shas,
        "wall_seconds": [run["metrics"].get("wall_seconds") for run in runs],
        "val_auc_bsi_best": val_aucs,
        "test_auc_bsi": tests,
        "gate_mean": gate_means,
    }

    if not (identity["pass"] and a_class["pass"]):
        status = "INVALID"
    else:
        status = "A3_PASS" if verdict["pass"] else "A3_FAIL"

    return {
        "status": status,
        "runs": [run["run_id"] for run in runs],
        "identity": identity,
        "a_class": a_class,
        "verdict": verdict,
        "supplementary": supplementary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A3 同配置复跑判定（spec 10-A3；预注册 2026-10-04-aliccp-a3-repeatability-calibration-design.md）"
    )
    parser.add_argument("--root", type=str, default="artifacts/aliccp_bench")
    parser.add_argument("--runs", nargs="+", required=True, help="run id 列表（约定第一个为参照 run）")
    parser.add_argument("--out", type=str, default=None, help="报告 JSON 落盘路径（可选）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    runs = [load_run(Path(args.root) / "runs" / run_id) for run_id in args.runs]
    report = analyze(runs)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return _EXIT_CODES[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
