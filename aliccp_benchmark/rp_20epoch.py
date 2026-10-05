"""残差 prompt 20-epoch 持久性检验（终止判定点；预注册：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-20epoch-design.md §5/§6）。

纯分析：**只读** run 记录（metrics.json / gate_report.json / prompt_report.json）与参照头文件哈希，
不重新评测、不新增 test 遍历、不改 run 产物。

- 历史判定 = `persistence_verdict`（C1 Δtest ≥ +0.0055 闭 × C2 Δval > 0 严格 × C3a A 类门禁 × C3c 机制门禁
  M0+G1–G8）→ `PERSISTS` / `NOT_PERSIST`；identity/记录校验（C3b）由调用方单独判定为 INVALID（完备性分支）。
- `persistence_verdict` / `_read_json` / `_a_class_ok` 自 `f2ccec2:aliccp_benchmark/rp_longer_budget.py`
  （blob `e02da3071ded554175cfdad5d7a725dd43ed7c89`）移植：前二者逐字；verdict 恰 1 类适配（阈值常量名
  `LONGER_BUDGET_DELTA_TEST_MIN` → `DELTA_TEST_MIN`，函数段内恰 2 处），由守卫测试
  `test_residual_prompt_20epoch.py` 以重建等式钉死；其余（新增判定层 / analyze_runs / main）按本实验判据新写。
- 新增层（§5.2–§5.6，先于结果写死）：二级分类 `secondary_classification`、正持久性声明
  `positive_persistence_claim`、`NO_CLEAR_IMPROVEMENT` 的机械 headroom `headroom_assessment`、
  预算趋势 `budget_trend`（5/10/20）、消融资格 `ablation_eligibility`。
- seed-2 5-epoch 与 10-epoch 记录只作 context（动机与对照），**不入判定**。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import protocol
from . import residual_prompt as RP

# ---- 预注册常量（先于任何 run 钉死；看到结果后不得修改）----
TWENTY_EPOCH_EPOCHS = 20
TWENTY_EPOCH_PATIENCE = 3
TWENTY_EPOCH_TAG = "xlong"
DELTA_TEST_MIN = 0.0055                    # 历史判据 C1（= 前驱 LONGER_BUDGET_DELTA_TEST_MIN 同名值）
SECONDARY_POSITIVE_MIN = 0.001             # §5.2 二级分类边界（闭端）
SECONDARY_DEGRADATION_MAX = -0.02          # §5.2 二级分类边界（闭端）
CLAIM_DELTA_TEST_MIN = 0.001               # §5.3 P4 正持久性声明下界（数值同 §5.2 正类边界，语义位置不同）
SEED2_STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"
SEED2_MODEL_SEED = 1688723740
BASELINE_VARIANT = RP.BASELINE_VARIANT            # "baseline"（机制模块为唯一事实来源）
ARM_VARIANT = RP.VARIANT                          # "residual-prompt"
ARM_RUN_ID_SUFFIX = RP.RUN_ID_SUFFIX              # "-rpg"
REFERENCE_CHECKPOINT_REL = ("artifacts/aliccp_bench/runs/"
                            "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt")
REFERENCE_CHECKPOINT_SHA256 = "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f"
OUTPUT_NAME = "rp_20epoch_compare.json"

# seed-2 5-epoch 残差 prompt 先例（context only；审计逐位复核；§1）
SEED2_5EPOCH_BASELINE_RUN_ID = "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07"
SEED2_5EPOCH_ARM_RUN_ID = "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg"
SEED2_5EPOCH_BASELINE_TEST_AUC = 0.5974422649550507
SEED2_5EPOCH_BASELINE_VAL_AUC = 0.5809347091990792
SEED2_5EPOCH_ARM_TEST_AUC = 0.6055453782825184
SEED2_5EPOCH_ARM_VAL_AUC = 0.5895572066556973
SEED2_5EPOCH_DELTA_TEST = 0.008103113327467715
SEED2_5EPOCH_DELTA_VAL = 0.008622497456618139

# seed-2 10-epoch 直接前驱（context only；审计逐位复核；§1）
SEED2_10EPOCH_BASELINE_RUN_ID = "20261004-0427-p2M-v500k-t1M-m1688723740-long-2b1d585"
SEED2_10EPOCH_ARM_RUN_ID = "20261004-0431-p2M-v500k-t1M-m1688723740-long-f2ccec2-rpg"
SEED2_10EPOCH_BASELINE_TEST_AUC = 0.6614121223087556
SEED2_10EPOCH_BASELINE_VAL_AUC = 0.6401127811737995
SEED2_10EPOCH_ARM_TEST_AUC = 0.6688517876477933
SEED2_10EPOCH_ARM_VAL_AUC = 0.6449491259210769
SEED2_10EPOCH_DELTA_TEST = 0.0074396653390377265
SEED2_10EPOCH_DELTA_VAL = 0.0048363447472773435

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
        "delta_test_ge_min": bool(delta_test >= DELTA_TEST_MIN),
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
        "delta_test_min": DELTA_TEST_MIN,
        "checks": checks,
        "classification": "PERSISTS" if all(checks.values()) else "NOT_PERSIST",
    }


def _read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _a_class_ok(gates: dict) -> bool:
    """A 类门禁口径与协议 hard_pass 的 A 部分一致：PASS/SKIP 视为通过。"""
    return all(gates.get(gate, {}).get("verdict") in ("PASS", "SKIP") for gate in A_CLASS_GATES)


# ---- 新增判定层（§5.2–§5.6；先于结果写死）----


def secondary_classification(delta_test: float) -> str:
    """§5.2 二级分类（闭开边界，精确）：≥ +0.001 / (−0.02, +0.001) / ≤ −0.02。"""
    delta = float(delta_test)
    if delta >= SECONDARY_POSITIVE_MIN:
        return "POSITIVE_IMPROVEMENT"
    if delta > SECONDARY_DEGRADATION_MAX:
        return "NO_CLEAR_IMPROVEMENT"
    return "CLEAR_DEGRADATION"


def budget_trend(delta_test_5epoch: float, delta_test_10epoch: float, delta_test_20epoch: float) -> dict:
    """§5.4 预算趋势（描述性；三值均为"该预算下 arm − 同预算配对基线"）。"""
    d5, d10, d20 = float(delta_test_5epoch), float(delta_test_10epoch), float(delta_test_20epoch)
    e1, e2 = d10 - d5, d20 - d10
    if d5 == d10 == d20:
        trend = "FLAT"
    elif d5 > d10 > d20:
        trend = "MONOTONE_NARROWING"
    elif d5 < d10 < d20:
        trend = "MONOTONE_WIDENING"
    elif e1 < 0 and e2 > 0:
        trend = "TROUGH_AT_10"
    elif e1 > 0 and e2 < 0:
        trend = "PEAK_AT_10"
    else:
        trend = "NON_MONOTONE_MIXED"
    signs = {(d > 0) - (d < 0) for d in (d5, d10, d20)}
    return {"delta_test_5epoch": d5, "delta_test_10epoch": d10, "delta_test_20epoch": d20,
            "e1_10_minus_5": e1, "e2_20_minus_10": e2, "trend": trend,
            "sign_flip": len(signs) > 1}


def positive_persistence_claim(*, identity_all_pass, a_class_pass, mechanism_gates_pass,
                               delta_test, delta_val, delta_val_final) -> dict:
    """§5.3 正持久性声明（六项检查全真 ⇒ claim）：P1 identity ∧ P2 A 类 ∧ P3 机制门禁
    ∧ P4 Δtest ≥ +0.001（闭）∧ P5 Δval > 0（严格）∧ P6 末共同 epoch Δval > 0（严格；None ⇒ false）。"""
    checks = {
        "paired_identity_valid": bool(identity_all_pass),
        "a_class_pass": bool(a_class_pass),
        "mechanism_gates_pass": bool(mechanism_gates_pass),
        "delta_test_ge_0.001": bool(float(delta_test) >= CLAIM_DELTA_TEST_MIN),
        "delta_val_positive": bool(float(delta_val) > 0.0),
        "no_late_epoch_contradiction": bool(delta_val_final is not None and float(delta_val_final) > 0.0),
    }
    return {"checks": checks, "claim": all(checks.values()),
            "delta_test_min": CLAIM_DELTA_TEST_MIN, "delta_val_final": delta_val_final}


def headroom_assessment(*, secondary, delta_test, delta_val, delta_val_traj, arm_rec, arm_prompt) -> dict:
    """§5.4 机械 headroom（因子恒落盘；`applicable` 仅当二级分类为 NO_CLEAR_IMPROVEMENT）。

    - 机制活性：|α_final| > 0 ∧ 末 epoch generator_grad_norm > 0 ∧ geff_std > 0（prompt_report；缺 ⇒ INACTIVE）
    - 验证方向：Δval > 0 → POSITIVE / NON_POSITIVE
    - epoch 轨迹：臂 best_epoch == epochs_run（未早停）⇒ RIGHT_CENSORED_STILL_IMPROVING；否则 EARLY_STOPPED
    - 预算趋势：5/10/20 三值 + 标签（budget_trend）
    - 距正分类余量：+0.001 − Δtest
    """
    activity = "INACTIVE"
    if arm_prompt:
        alpha_final = float(arm_prompt.get("alpha_final") or 0.0)
        grad_probe = arm_prompt.get("grad_probe") or []
        last_gen_grad = float((grad_probe[-1] or {}).get("generator_grad_norm") or 0.0) if grad_probe else 0.0
        geff_std = float((arm_prompt.get("gate") or {}).get("geff_std") or 0.0)
        if abs(alpha_final) > 0.0 and last_gen_grad > 0.0 and geff_std > 0.0:
            activity = "ACTIVE"
    epochs_run = len(arm_rec["per_epoch"])
    right_censored = int(arm_rec["best_epoch"]) == epochs_run
    return {
        "applicable": secondary == "NO_CLEAR_IMPROVEMENT",
        "mechanism_activity": activity,
        "delta_val_direction": "POSITIVE" if float(delta_val) > 0.0 else "NON_POSITIVE",
        "epoch_trajectory": "RIGHT_CENSORED_STILL_IMPROVING" if right_censored else "EARLY_STOPPED",
        "budget_trend": budget_trend(SEED2_5EPOCH_DELTA_TEST, SEED2_10EPOCH_DELTA_TEST, float(delta_test)),
        "gap_to_positive_threshold": SECONDARY_POSITIVE_MIN - float(delta_test),
        "delta_val_traj_last3": [float(v) for v in (delta_val_traj or [])[-3:]],
    }


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
        "tag_recorded": rec.get("tag"),
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
                 expected_epochs=TWENTY_EPOCH_EPOCHS, expected_patience=TWENTY_EPOCH_PATIENCE,
                 expected_tag=TWENTY_EPOCH_TAG,
                 expected_reference_checkpoint=REFERENCE_CHECKPOINT_REL,
                 expected_reference_sha256=REFERENCE_CHECKPOINT_SHA256,
                 output_path=None) -> dict:
    """只读两 run 记录（+ 参照头文件哈希）→ 历史判定 + 二级分类 + claim + headroom + 消融资格 + 描述量。

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
        "tag_recorded_pinned": base.get("tag") == expected_tag == arm.get("tag"),
    }
    a_class_pass = _a_class_ok(base_gates) and _a_class_ok(arm_gates)
    mechanism = _mechanism_gate_report(arm)
    verdict = persistence_verdict(
        baseline_test_auc=base["test_auc_bsi"], baseline_val_auc=base["best_val_auc_bsi"],
        arm_test_auc=arm["test_auc_bsi"], arm_val_auc=arm["best_val_auc_bsi"],
        a_class_pass=a_class_pass, mechanism_gates_pass=mechanism["all_pass"],
    )
    identity_all_pass = all(identity_checks.values())
    classification = verdict["classification"] if identity_all_pass else "INVALID"

    base_desc, arm_desc = _arm_description(base), _arm_description(arm)
    if arm_prompt is not None:
        arm_desc["mechanism_diagnostics"] = {key: arm_prompt.get(key)
                                             for key in _MECHANISM_DIAGNOSTIC_KEYS}
    common = min(len(base_desc["val_traj"]), len(arm_desc["val_traj"]))
    delta_traj = [arm_desc["val_traj"][i] - base_desc["val_traj"][i] for i in range(common)]
    delta_val_final = delta_traj[-1] if common else None

    secondary = secondary_classification(verdict["delta_test_auc"])
    claim = positive_persistence_claim(
        identity_all_pass=identity_all_pass, a_class_pass=a_class_pass,
        mechanism_gates_pass=mechanism["all_pass"],
        delta_test=verdict["delta_test_auc"], delta_val=verdict["delta_val_auc"],
        delta_val_final=delta_val_final,
    )
    headroom = headroom_assessment(secondary=secondary, delta_test=verdict["delta_test_auc"],
                                   delta_val=verdict["delta_val_auc"], delta_val_traj=delta_traj,
                                   arm_rec=arm, arm_prompt=arm_prompt)

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
        "secondary": {"classification": secondary, "delta_test": verdict["delta_test_auc"],
                      "boundaries": {"positive_min": SECONDARY_POSITIVE_MIN,
                                     "degradation_max": SECONDARY_DEGRADATION_MAX}},
        "positive_persistence_claim": claim,
        "headroom": headroom,
        "ablation_eligibility": {"eligible": bool(claim["claim"]),
                                 "rule": "§5.6: claim（§5.3 六项全真）",
                                 "claim": bool(claim["claim"])},
        "criteria": {
            "delta_test_min": DELTA_TEST_MIN,
            "epochs": expected_epochs,
            "patience": expected_patience,
            "tag": expected_tag,
            "seed2_stage1_id": expected_stage1_id,
            "seed2_model_seed": expected_model_seed,
            "baseline_variant": BASELINE_VARIANT,
            "arm_variant": ARM_VARIANT,
            "arm_run_id_suffix": ARM_RUN_ID_SUFFIX,
            "reference_checkpoint": str(expected_reference_checkpoint),
            "reference_checkpoint_sha256": expected_reference_sha256,
            "secondary_positive_min": SECONDARY_POSITIVE_MIN,
            "secondary_degradation_max": SECONDARY_DEGRADATION_MAX,
            "claim_delta_test_min": CLAIM_DELTA_TEST_MIN,
        },
        "description": {
            "baseline": base_desc,
            "arm": arm_desc,
            "delta_traj": delta_traj,
            "delta_traj_common_epochs": common,
            "delta_val_final": delta_val_final,
            "late_epoch_contradiction": bool(delta_val_final is not None and delta_val_final <= 0.0),
        },
        "context": {
            "note": "seed-2 5-/10-epoch 记录：仅动机与对照（context_only），不构成本实验判定结果的任何部分",
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
            "ten_epoch": {
                "baseline_run_id": SEED2_10EPOCH_BASELINE_RUN_ID,
                "arm_run_id": SEED2_10EPOCH_ARM_RUN_ID,
                "baseline_test_auc": SEED2_10EPOCH_BASELINE_TEST_AUC,
                "baseline_val_auc": SEED2_10EPOCH_BASELINE_VAL_AUC,
                "arm_test_auc": SEED2_10EPOCH_ARM_TEST_AUC,
                "arm_val_auc": SEED2_10EPOCH_ARM_VAL_AUC,
                "delta_test_auc": SEED2_10EPOCH_DELTA_TEST,
                "delta_val_auc": SEED2_10EPOCH_DELTA_VAL,
            },
            "delta_test_change_vs_five_epoch": verdict["delta_test_auc"] - SEED2_5EPOCH_DELTA_TEST,
            "delta_val_change_vs_five_epoch": verdict["delta_val_auc"] - SEED2_5EPOCH_DELTA_VAL,
            "delta_test_change_vs_ten_epoch": verdict["delta_test_auc"] - SEED2_10EPOCH_DELTA_TEST,
            "delta_val_change_vs_ten_epoch": verdict["delta_val_auc"] - SEED2_10EPOCH_DELTA_VAL,
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
        description="残差 prompt 20-epoch（终止判定点）持久性检验（只读分析；预注册：docs/superpowers/specs/"
                    "2026-10-05-aliccp-stage2-residual-prompt-20epoch-design.md）"
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
    print(f"secondary={result['secondary']['classification']} "
          f"claim={result['positive_persistence_claim']['claim']} "
          f"ablation_eligible={result['ablation_eligibility']['eligible']}")
    print(f"mechanism_gates failures={result['mechanism']['failures']} "
          f"in_run_classification={result['mechanism']['classification_in_run']}")
    ctx = result["context"]
    print(f"context(5-epoch, motivation only): delta_test={ctx['five_epoch']['delta_test_auc']:+.7f} "
          f"delta_val={ctx['five_epoch']['delta_val_auc']:+.7f}")
    print(f"context(10-epoch): delta_test={ctx['ten_epoch']['delta_test_auc']:+.7f} "
          f"delta_val={ctx['ten_epoch']['delta_val_auc']:+.7f}")
    print(f"budget_trend={result['headroom']['budget_trend']['trend']} "
          f"late_epoch_contradiction={result['description']['late_epoch_contradiction']}")
    for arm_name in ("baseline", "arm"):
        desc = result["description"][arm_name]
        print(f"{arm_name}: best_epoch={desc['best_epoch']} epochs_run={desc['epochs_run']} "
              f"stopped_early={desc['stopped_early']} test_auc={desc['test_auc_bsi']!r} "
              f"best_val_auc={desc['best_val_auc_bsi']!r} gate_mean={desc['gate_mean']}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
