"""更长 Stage-2 预算持久性检验（预注册：docs/superpowers/specs/2026-10-03-aliccp-stage2-attenuation-longer-budget-design.md §5/§6）。

纯分析：**只读** run 记录（metrics.json / gate_report.json），不重新评测、不新增 test 遍历、不改 run 产物。

- 判定 = `persistence_verdict`（C1 Δtest ≥ +0.0055 闭 × C2 Δval > 0 严格 × C3a A 类门禁）→ `PERSISTS` / `NOT_PERSIST`。
- identity/记录不一致（C3b）→ `INVALID`（完备性分支：出现即表示运行偏离预注册）。
- seed-2 5-epoch 先例值只作 context（动机与对照），**不入判定**。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import metrics, protocol

# ---- 预注册常量（先于任何 run 钉死；看到结果后不得修改）----
LONGER_BUDGET_EPOCHS = 10
LONGER_BUDGET_PATIENCE = 3
LONGER_BUDGET_TAG = "long"
LONGER_BUDGET_DELTA_TEST_MIN = 0.0055
SEED2_STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"
SEED2_MODEL_SEED = metrics.REPLICATION_MODEL_SEED          # = 1688723740（seed-2；不动）
SPEC_ATTENUATION_COEF = metrics.SPEC_ATTENUATION_COEF      # = 0.6972233730330467（不动）

# seed-2 5-epoch 先例（context only；逐位取自 on-disk run metrics.json，运行前已复核）
SEED2_5EPOCH_BASELINE_RUN_ID = "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07"
SEED2_5EPOCH_ARM_RUN_ID = "20261003-0627-p2M-v500k-t1M-m1688723740-short-79b5e07-sattn"
SEED2_5EPOCH_BASELINE_TEST_AUC = 0.5974422649550507
SEED2_5EPOCH_BASELINE_VAL_AUC = 0.5809347091990792
SEED2_5EPOCH_ARM_TEST_AUC = 0.6051559662367624
SEED2_5EPOCH_ARM_VAL_AUC = 0.5854178317514255
SEED2_5EPOCH_DELTA_TEST = 0.007713701281711671
SEED2_5EPOCH_DELTA_VAL = 0.004483122552346286

A_CLASS_GATES = ("A1", "A2", "A3", "A4", "A5", "A6")


def persistence_verdict(*, baseline_test_auc, baseline_val_auc, arm_test_auc, arm_val_auc,
                        a_class_pass) -> dict:
    """§5 判据（机械计算，阈值写死）：C1 Δtest ≥ +0.0055（闭）× C2 Δval > 0（严格）× C3a A 类门禁。

    任一不满足 → NOT_PERSIST。identity/记录校验（C3b）由调用方单独判定为 INVALID，不在此函数内。
    """
    delta_test = float(arm_test_auc) - float(baseline_test_auc)
    delta_val = float(arm_val_auc) - float(baseline_val_auc)
    checks = {
        "delta_test_ge_min": bool(delta_test >= LONGER_BUDGET_DELTA_TEST_MIN),
        "delta_val_positive": bool(delta_val > 0.0),
        "a_class_pass": bool(a_class_pass),
    }
    return {
        "baseline_test_auc": float(baseline_test_auc),
        "baseline_val_auc": float(baseline_val_auc),
        "arm_test_auc": float(arm_test_auc),
        "arm_val_auc": float(arm_val_auc),
        "delta_test_auc": delta_test,
        "delta_val_auc": delta_val,
        "delta_test_min": LONGER_BUDGET_DELTA_TEST_MIN,
        "checks": checks,
        "classification": "PERSISTS" if all(checks.values()) else "NOT_PERSIST",
    }


def _read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _a_class_ok(gates: dict) -> bool:
    """A 类门禁口径与协议 hard_pass 的 A 部分一致：PASS/SKIP 视为通过。"""
    return all(gates.get(gate, {}).get("verdict") in ("PASS", "SKIP") for gate in A_CLASS_GATES)


def _arm_description(rec: dict) -> dict:
    """描述量（不参与判定）：轨迹、best epoch、是否早停、gate 均值、运行时；处理臂附加探针。"""
    traj = [float(r["val_auc_bsi"]) for r in rec["per_epoch"]]
    best_epoch = int(rec["best_epoch"])
    desc = {
        "run_id": rec["run_id"],
        "best_epoch": best_epoch,
        "epochs_recorded": int(rec["epochs"]),
        "patience_recorded": int(rec["patience"]),
        "epochs_run": len(traj),
        "stopped_early": bool(len(traj) < int(rec["epochs"])),
        "best_is_last_epoch": bool(best_epoch == len(traj)),
        "val_traj": traj,
        "best_val_auc_bsi": float(rec["best_val_auc_bsi"]),
        "test_auc_bsi": float(rec["test_auc_bsi"]),
        "gate_mean": [float(g) for g in rec["gate_mean"]],
        "wall_seconds": rec["wall_seconds"],
        "peak_vram_mb": rec.get("peak_vram_mb"),
        "spec_attenuation": float(rec["spec_attenuation"]),
    }
    if "probe" in rec:
        desc["probe"] = rec["probe"]
        desc["trainable_params"] = rec.get("trainable_params")
        desc["trainable_params_default"] = rec.get("trainable_params_default")
    return desc


def analyze_runs(*, root, baseline_run_id, arm_run_id,
                 expected_stage1_id=SEED2_STAGE1_ID, expected_model_seed=SEED2_MODEL_SEED,
                 expected_epochs=LONGER_BUDGET_EPOCHS, expected_patience=LONGER_BUDGET_PATIENCE,
                 output_path=None) -> dict:
    """只读两 run 记录 → 判定 + 描述量 + context + identity checks；可选落盘 JSON。

    `expected_*` 默认即预注册钉死值；仅测试夹具显式传入小值（生产调用不得覆盖）。
    """
    root = Path(root)
    base = _read_json(protocol.run_dir(root, baseline_run_id) / "metrics.json")
    arm = _read_json(protocol.run_dir(root, arm_run_id) / "metrics.json")
    base_gates = _read_json(protocol.run_dir(root, baseline_run_id) / "gate_report.json")["gates"]
    arm_gates = _read_json(protocol.run_dir(root, arm_run_id) / "gate_report.json")["gates"]

    identity_checks = {
        "same_stage1_id": base["stage1_id"] == arm["stage1_id"],
        "stage1_id_is_pinned": base["stage1_id"] == expected_stage1_id == arm["stage1_id"],
        "same_model_seed": base["model_seed"] == arm["model_seed"],
        "model_seed_is_pinned": base["model_seed"] == expected_model_seed == arm["model_seed"],
        "baseline_spec_attenuation_default": float(base["spec_attenuation"]) == 1.0,
        "arm_spec_attenuation_frozen": float(arm["spec_attenuation"]) == float(SPEC_ATTENUATION_COEF),
        "epochs_recorded_pinned": base["epochs"] == expected_epochs and arm["epochs"] == expected_epochs,
        "patience_recorded_pinned": base["patience"] == expected_patience and arm["patience"] == expected_patience,
    }
    a_class_pass = _a_class_ok(base_gates) and _a_class_ok(arm_gates)
    verdict = persistence_verdict(
        baseline_test_auc=base["test_auc_bsi"], baseline_val_auc=base["best_val_auc_bsi"],
        arm_test_auc=arm["test_auc_bsi"], arm_val_auc=arm["best_val_auc_bsi"],
        a_class_pass=a_class_pass,
    )
    classification = verdict["classification"] if all(identity_checks.values()) else "INVALID"

    base_desc, arm_desc = _arm_description(base), _arm_description(arm)
    common = min(len(base_desc["val_traj"]), len(arm_desc["val_traj"]))
    delta_traj = [arm_desc["val_traj"][i] - base_desc["val_traj"][i] for i in range(common)]

    result = {
        "run_ids": {"baseline": baseline_run_id, "arm": arm_run_id},
        "stage1_id": base["stage1_id"],
        "identity_checks": identity_checks,
        "gates": {
            "baseline": {gate: base_gates.get(gate, {}).get("verdict") for gate in base_gates},
            "arm": {gate: arm_gates.get(gate, {}).get("verdict") for gate in arm_gates},
            "a_class_pass": bool(a_class_pass),
        },
        "verdict": verdict,
        "classification": classification,
        "criteria": {
            "delta_test_min": LONGER_BUDGET_DELTA_TEST_MIN,
            "epochs": expected_epochs,
            "patience": expected_patience,
            "tag": LONGER_BUDGET_TAG,
            "seed2_stage1_id": expected_stage1_id,
            "seed2_model_seed": expected_model_seed,
            "spec_attenuation_coef": float(SPEC_ATTENUATION_COEF),
        },
        "description": {
            "baseline": base_desc,
            "arm": arm_desc,
            "delta_traj": delta_traj,
            "delta_traj_common_epochs": common,
        },
        "context": {
            "note": "seed-2 5-epoch 先例：仅动机与对照（context_only），不构成本实验判定结果的任何部分",
            "five_epoch": {
                "baseline_run_id": SEED2_5EPOCH_BASELINE_RUN_ID,
                "arm_run_id": SEED2_5EPOCH_ARM_RUN_ID,
                "baseline_test_auc": SEED2_5EPOCH_BASELINE_TEST_AUC,
                "baseline_val_auc": SEED2_5EPOCH_BASELINE_VAL_AUC,
                "arm_test_auc": SEED2_5EPOCH_ARM_TEST_AUC,
                "arm_val_auc": SEED2_5EPOCH_ARM_VAL_AUC,
                "delta_test_auc": SEED2_5EPOCH_DELTA_TEST,
                "delta_val_auc": SEED2_5EPOCH_DELTA_VAL,
            },
            "delta_test_change_vs_five_epoch": verdict["delta_test_auc"] - SEED2_5EPOCH_DELTA_TEST,
            "delta_val_change_vs_five_epoch": verdict["delta_val_auc"] - SEED2_5EPOCH_DELTA_VAL,
        },
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["output_path"] = str(output_path)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="更长 Stage-2 预算持久性检验（只读分析；预注册：docs/superpowers/specs/"
                    "2026-10-03-aliccp-stage2-attenuation-longer-budget-design.md）"
    )
    parser.add_argument("--root", type=str, default=str(protocol.ARTIFACT_ROOT))
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--arm-run", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 路径（默认 runs/<arm-run>/longer_budget_compare.json）")
    args = parser.parse_args(argv)
    root = Path(args.root)
    output = Path(args.output) if args.output else protocol.run_dir(root, args.arm_run) / "longer_budget_compare.json"
    result = analyze_runs(root=root, baseline_run_id=args.baseline_run, arm_run_id=args.arm_run,
                          output_path=output)
    verdict = result["verdict"]
    print(f"classification={result['classification']} "
          f"delta_test_auc={verdict['delta_test_auc']:+.7f} delta_val_auc={verdict['delta_val_auc']:+.7f} "
          f"delta_test_min={verdict['delta_test_min']} checks={verdict['checks']}")
    ctx = result["context"]["five_epoch"]
    print(f"context(5-epoch, motivation only): delta_test={ctx['delta_test_auc']:+.7f} "
          f"delta_val={ctx['delta_val_auc']:+.7f}")
    for arm_name in ("baseline", "arm"):
        desc = result["description"][arm_name]
        print(f"{arm_name}: best_epoch={desc['best_epoch']} epochs_run={desc['epochs_run']} "
              f"stopped_early={desc['stopped_early']} test_auc={desc['test_auc_bsi']!r} "
              f"best_val_auc={desc['best_val_auc_bsi']!r} gate_mean={desc['gate_mean']}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
