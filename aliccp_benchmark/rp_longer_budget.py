"""残差 prompt 增量的更长 Stage-2 预算持久性检验（预注册：docs/superpowers/specs/2026-10-04-aliccp-stage2-residual-prompt-longer-budget-design.md §5/§6）。

纯分析：**只读** run 记录（metrics.json / gate_report.json / prompt_report.json）与参照头文件哈希，
不重新评测、不新增 test 遍历、不改 run 产物。

- 判定 = `persistence_verdict`（C1 Δtest ≥ +0.0055 闭 × C2 Δval > 0 严格 × C3a A 类门禁 × C3c 机制门禁
  M0+G1–G8）→ `PERSISTS` / `NOT_PERSIST`；identity/记录不一致（C3b）→ `INVALID`（完备性分支）。
- `persistence_verdict` / `_read_json` / `_a_class_ok` 自 exp/aliccp-stage2-attenuation-longer-budget @
  `bbd8a61:aliccp_benchmark/longer_budget.py`（blob `c8f66409a57d1e23d0629bedbae5ed89a6ef9523`）移植：
  前二者逐字；verdict 恰 3 处文档化适配（签名行 / 首行 docstring / checks 新增 `mechanism_gates_pass`），
  由守卫测试 `test_residual_prompt_longer_budget.py` 以重建等式钉死；其余按本实验判据新写。
- seed-2 5-epoch 残差 prompt 先例值只作 context（动机与对照），**不入判定**。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import protocol
from . import residual_prompt as RP

# ---- 预注册常量（先于任何 run 钉死；看到结果后不得修改）----
LONGER_BUDGET_EPOCHS = 10
LONGER_BUDGET_PATIENCE = 3
LONGER_BUDGET_TAG = "long"
LONGER_BUDGET_DELTA_TEST_MIN = 0.0055
SEED2_STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"
SEED2_MODEL_SEED = 1688723740
BASELINE_VARIANT = RP.BASELINE_VARIANT            # "baseline"（机制模块为唯一事实来源）
ARM_VARIANT = RP.VARIANT                          # "residual-prompt"
ARM_RUN_ID_SUFFIX = RP.RUN_ID_SUFFIX              # "-rpg"
REFERENCE_CHECKPOINT_REL = ("artifacts/aliccp_bench/runs/"
                            "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt")
REFERENCE_CHECKPOINT_SHA256 = "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f"
OUTPUT_NAME = "rp_longer_budget_compare.json"

# seed-2 5-epoch 残差 prompt 先例（context only；逐位取自 on-disk run 记录，运行前已复核）
SEED2_5EPOCH_BASELINE_RUN_ID = "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07"
SEED2_5EPOCH_ARM_RUN_ID = "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg"
SEED2_5EPOCH_BASELINE_TEST_AUC = 0.5974422649550507
SEED2_5EPOCH_BASELINE_VAL_AUC = 0.5809347091990792
SEED2_5EPOCH_ARM_TEST_AUC = 0.6055453782825184
SEED2_5EPOCH_ARM_VAL_AUC = 0.5895572066556973
SEED2_5EPOCH_DELTA_TEST = 0.008103113327467715
SEED2_5EPOCH_DELTA_VAL = 0.008622497456618139

A_CLASS_GATES = ("A1", "A2", "A3", "A4", "A5", "A6")
MECHANISM_GATES = ("M0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")


def persistence_verdict(*, baseline_test_auc, baseline_val_auc, arm_test_auc, arm_val_auc,
                        a_class_pass, mechanism_gates_pass) -> dict:
    """§5 判据（机械计算，阈值写死）：C1 Δtest ≥ +0.0055（闭）× C2 Δval > 0（严格）× C3a A 类门禁 × C3c 机制门禁（M0+G1–G8）。

    任一不满足 → NOT_PERSIST。identity/记录校验（C3b）由调用方单独判定为 INVALID，不在此函数内。
    """
    delta_test = float(arm_test_auc) - float(baseline_test_auc)
    delta_val = float(arm_val_auc) - float(baseline_val_auc)
    checks = {
        "delta_test_ge_min": bool(delta_test >= LONGER_BUDGET_DELTA_TEST_MIN),
        "delta_val_positive": bool(delta_val > 0.0),
        "a_class_pass": bool(a_class_pass),
        "mechanism_gates_pass": bool(mechanism_gates_pass),
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


def _read_json_optional(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _sha256_file(path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def _mechanism_gate_report(arm_rec: dict) -> dict:
    """从处理臂记录 `rp_arm` 提取 M0+G1–G8（缺 rp_arm ⇒ 全 FAIL，逐项可溯）。"""
    arm = arm_rec.get("rp_arm") or {}
    gate_pass = {gate: bool((arm.get(gate) or {}).get("pass")) for gate in MECHANISM_GATES}
    failures = [gate for gate in MECHANISM_GATES if not gate_pass[gate]]
    return {"gates": gate_pass, "failures": failures, "all_pass": all(gate_pass.values()),
            "classification_in_run": arm.get("classification"), "subreason_in_run": arm.get("subreason")}


def _arm_description(rec: dict) -> dict:
    """描述量（不参与判定）：轨迹、best epoch、是否早停、gate 均值、运行时；处理臂附加 rp_arm 摘要。"""
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
        "variant": rec.get("variant"),
    }
    if "rp_arm" in rec:
        arm = rec["rp_arm"]
        desc["rp_arm"] = {"classification": arm["classification"], "subreason": arm["subreason"],
                          "pass": arm["pass"], "U": arm["U"], "protocol_10_1": arm["protocol_10_1"],
                          "mechanism_gates": {gate: bool((arm.get(gate) or {}).get("pass"))
                                              for gate in MECHANISM_GATES}}
        desc["prompt_hidden"] = rec.get("prompt_hidden")
    return desc


_MECHANISM_DIAGNOSTIC_KEYS = ("construction_identity", "init_forward", "grad_probe", "alpha_final",
                              "gate", "val_stats", "dispersion", "reference_dispersion",
                              "source_gates", "params")


def analyze_runs(*, root, baseline_run_id, arm_run_id,
                 expected_stage1_id=SEED2_STAGE1_ID, expected_model_seed=SEED2_MODEL_SEED,
                 expected_epochs=LONGER_BUDGET_EPOCHS, expected_patience=LONGER_BUDGET_PATIENCE,
                 expected_reference_checkpoint=REFERENCE_CHECKPOINT_REL,
                 expected_reference_sha256=REFERENCE_CHECKPOINT_SHA256,
                 output_path=None) -> dict:
    """只读两 run 记录（+ 参照头文件哈希）→ 判定 + 描述量 + identity checks + context；可选落盘 JSON。

    `expected_*` 默认即预注册钉死值；仅测试夹具显式传入小值（生产调用不得覆盖）。
    参照路径按 cwd 解析（生产调用 cwd = 仓库根；预注册 §8）。
    """
    root = Path(root)
    base = _read_json(protocol.run_dir(root, baseline_run_id) / "metrics.json")
    arm = _read_json(protocol.run_dir(root, arm_run_id) / "metrics.json")
    base_gates = _read_json(protocol.run_dir(root, baseline_run_id) / "gate_report.json")["gates"]
    arm_gates = _read_json(protocol.run_dir(root, arm_run_id) / "gate_report.json")["gates"]
    arm_prompt = _read_json_optional(protocol.run_dir(root, arm_run_id) / "prompt_report.json")

    recorded_ref = None
    if arm_prompt:
        recorded_ref = (arm_prompt.get("reference_dispersion") or {}).get("newtask_checkpoint")
    reference_sha = _sha256_file(expected_reference_checkpoint)

    identity_checks = {
        "same_stage1_id": base["stage1_id"] == arm["stage1_id"],
        "stage1_id_is_pinned": base["stage1_id"] == expected_stage1_id == arm["stage1_id"],
        "same_model_seed": base["model_seed"] == arm["model_seed"],
        "model_seed_is_pinned": base["model_seed"] == expected_model_seed == arm["model_seed"],
        "baseline_variant_is_baseline": base.get("variant") == BASELINE_VARIANT,
        "arm_variant_is_residual_prompt": arm.get("variant") == ARM_VARIANT,
        "arm_run_id_has_suffix": bool(arm["run_id"].endswith(ARM_RUN_ID_SUFFIX)
                                      and not base["run_id"].endswith(ARM_RUN_ID_SUFFIX)),
        "reference_checkpoint_is_pinned": bool(recorded_ref is not None
                                               and Path(recorded_ref) == Path(expected_reference_checkpoint)),
        "reference_checkpoint_sha_pinned": bool(reference_sha == expected_reference_sha256),
        "epochs_recorded_pinned": base["epochs"] == expected_epochs and arm["epochs"] == expected_epochs,
        "patience_recorded_pinned": (base["patience"] == expected_patience
                                     and arm["patience"] == expected_patience),
    }
    a_class_pass = _a_class_ok(base_gates) and _a_class_ok(arm_gates)
    mechanism = _mechanism_gate_report(arm)
    verdict = persistence_verdict(
        baseline_test_auc=base["test_auc_bsi"], baseline_val_auc=base["best_val_auc_bsi"],
        arm_test_auc=arm["test_auc_bsi"], arm_val_auc=arm["best_val_auc_bsi"],
        a_class_pass=a_class_pass, mechanism_gates_pass=mechanism["all_pass"],
    )
    classification = verdict["classification"] if all(identity_checks.values()) else "INVALID"

    base_desc, arm_desc = _arm_description(base), _arm_description(arm)
    if arm_prompt is not None:
        arm_desc["mechanism_diagnostics"] = {key: arm_prompt.get(key)
                                             for key in _MECHANISM_DIAGNOSTIC_KEYS}
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
        "mechanism": mechanism,
        "reference": {"recorded": recorded_ref, "expected": str(expected_reference_checkpoint),
                      "sha256": reference_sha, "expected_sha256": expected_reference_sha256},
        "verdict": verdict,
        "classification": classification,
        "criteria": {
            "delta_test_min": LONGER_BUDGET_DELTA_TEST_MIN,
            "epochs": expected_epochs,
            "patience": expected_patience,
            "tag": LONGER_BUDGET_TAG,
            "seed2_stage1_id": expected_stage1_id,
            "seed2_model_seed": expected_model_seed,
            "baseline_variant": BASELINE_VARIANT,
            "arm_variant": ARM_VARIANT,
            "arm_run_id_suffix": ARM_RUN_ID_SUFFIX,
            "reference_checkpoint": str(expected_reference_checkpoint),
            "reference_checkpoint_sha256": expected_reference_sha256,
        },
        "description": {
            "baseline": base_desc,
            "arm": arm_desc,
            "delta_traj": delta_traj,
            "delta_traj_common_epochs": common,
        },
        "context": {
            "note": "seed-2 5-epoch 残差 prompt 先例：仅动机与对照（context_only），不构成本实验判定结果的任何部分",
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
        description="残差 prompt 增量更长预算持久性检验（只读分析；预注册：docs/superpowers/specs/"
                    "2026-10-04-aliccp-stage2-residual-prompt-longer-budget-design.md）"
    )
    parser.add_argument("--root", type=str, default=str(protocol.ARTIFACT_ROOT))
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--arm-run", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help=f"输出 JSON 路径（默认 runs/<arm-run>/{OUTPUT_NAME}）")
    args = parser.parse_args(argv)
    root = Path(args.root)
    output = Path(args.output) if args.output else protocol.run_dir(root, args.arm_run) / OUTPUT_NAME
    result = analyze_runs(root=root, baseline_run_id=args.baseline_run, arm_run_id=args.arm_run,
                          output_path=output)
    verdict = result["verdict"]
    print(f"classification={result['classification']} "
          f"delta_test_auc={verdict['delta_test_auc']:+.7f} delta_val_auc={verdict['delta_val_auc']:+.7f} "
          f"delta_test_min={verdict['delta_test_min']} checks={verdict['checks']}")
    print(f"mechanism_gates failures={result['mechanism']['failures']} "
          f"in_run_classification={result['mechanism']['classification_in_run']}")
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
