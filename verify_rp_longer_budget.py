"""运行后独立复核（只读）：残差 prompt 更长预算持久性检验（分支 exp/aliccp-stage2-residual-prompt-longer-budget）。

**不 import 任何机制/接线模块**（`rp_longer_budget` / `residual_prompt` / `bench` 均不 import）：
纯 JSON 重推 + 文件哈希 + 与预注册常量逐位比对。对象 = 本实验两条 run（10-epoch 基线臂 + 残差 prompt 臂）
+ 分析器输出 `runs/<arm>/rp_longer_budget_compare.json` + SUMMARY 行 + 参照头文件。

复核内容：
  A. 两条 run 记录齐全（metrics/gate/prompt_report）且 identity 与预注册钉死值一致；
  B. 真冻结链（backbone sha loaded==before==after==stage1 meta）+ env/fingerprint sha + grads_none；
  C. A 类门禁两 run 全 PASS（A3 SKIP）+ B4 继承 FAIL + hard_pass=false；
  D. 处理臂机制门禁 M0+G1–G8 全 PASS；M0 观测与钉死常量逐位一致；参照路径 + 文件 sha 钉死；
  E. rp_arm 三份 JSON 逐字互洽（metrics.rp_arm == gate_report.residual_prompt == prompt_report.arm）；
  F. 轨迹/best/early-stop 记录自洽；判定（C1/C2/C3）float64 独立重算 == 分析器 JSON（逐位）；
  G. 确定性附属核对（描述性，不入判定）：本实验基线轨迹 == 20261003-0724 旧 10-epoch 基线逐位；
     处理臂前 5 epoch == 5-epoch 残差臂 run 逐位；基线 test == 0.6614121223087556；
  H. SUMMARY.md 两行（append-only）字段与 run 记录一致。

退出码 0 = 全部一致；1 = 有不一致（逐项打印；不一致按实记录并停止解读）。
输出：artifacts/aliccp_bench/runs/<arm-run>/verify_report.json（+ stdout）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# ---- 预注册钉死常量（与 docs/superpowers/specs/2026-10-04-…-longer-budget-design.md 一致）----
SEED2_STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"
SEED2_MODEL_SEED = 1688723740
BASELINE_VARIANT = "baseline"
ARM_VARIANT = "residual-prompt"
ARM_RUN_ID_SUFFIX = "-rpg"
EPOCHS = 10
PATIENCE = 3
DELTA_TEST_MIN = 0.0055
REFERENCE_CHECKPOINT_REL = ("artifacts/aliccp_bench/runs/"
                            "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt")
REFERENCE_CHECKPOINT_SHA = "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f"
BACKBONE_SHA = "e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c"
ENV_IDS_SHA = "5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0"
FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
CONFIG_HASH = "4e1b5c6ffe9b6ff49cda34691de27390669396337e64e867f0f83b6ec4da7981"
STAGE1_FILE_SHAS = {
    "backbone.pt": "cd7b033423499e0d36eea988835cea4ac4e2a8e26427a2aea627333822101e0b",
    "env_ids.pt": "4660be5aaa3c59f53dd5b4394f77114a87db69064b32c45517db049a6b9157e7",
    "meta.json": "61a66d81ce3dcf6bae64f4c2b37cf12e722943def646336a588746aba932931d",
}
BASELINE_AUC_TEST_5EP = 0.5974422649550507      # rp_arm.U 口径（5-epoch 常量；class C 诊断）
BASELINE_AUC_VAL_5EP = 0.5809347091990792
REFERENCE_PRED_STD = 0.005217193225189258
REFERENCE_TOL = 1e-9

# 确定性附属核对（描述性；旧 run 只读）
OLD_10EP_BASELINE_RUN = "20261003-0724-p2M-v500k-t1M-m1688723740-long-bbd8a61"
OLD_10EP_BASELINE_TEST_AUC = 0.6614121223087556
OLD_5EP_ARM_RUN = "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg"
OLD_5EP_BASELINE_RUN = "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07"

MECHANISM_GATES = ("M0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")

checks: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="残差 prompt 更长预算持久性检验：运行后独立复核（只读）")
    parser.add_argument("--root", type=str, default="artifacts/aliccp_bench")
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--arm-run", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="报告路径（默认 runs/<arm-run>/verify_report.json）")
    args = parser.parse_args(argv)
    root = Path(args.root)
    base_dir = root / "runs" / args.baseline_run
    arm_dir = root / "runs" / args.arm_run

    # ---------------- A. run 记录与 identity ----------------
    base_m = load(base_dir / "metrics.json")
    base_g = load(base_dir / "gate_report.json")
    base_c = load(base_dir / "config.json")
    arm_m = load(arm_dir / "metrics.json")
    arm_g = load(arm_dir / "gate_report.json")
    arm_c = load(arm_dir / "config.json")
    arm_p = load(arm_dir / "prompt_report.json")

    check("identity.stage1_id_pinned",
          base_m["stage1_id"] == arm_m["stage1_id"] == SEED2_STAGE1_ID, base_m["stage1_id"])
    check("identity.model_seed_pinned",
          base_m["model_seed"] == arm_m["model_seed"] == SEED2_MODEL_SEED, str(base_m["model_seed"]))
    check("identity.variants",
          base_m.get("variant") == BASELINE_VARIANT and arm_m.get("variant") == ARM_VARIANT,
          f"baseline={base_m.get('variant')} arm={arm_m.get('variant')}")
    check("identity.run_id_suffix",
          arm_m["run_id"].endswith(ARM_RUN_ID_SUFFIX) and not base_m["run_id"].endswith(ARM_RUN_ID_SUFFIX),
          f"base={base_m['run_id']} arm={arm_m['run_id']}")
    check("identity.epochs_patience_recorded",
          base_m["epochs"] == arm_m["epochs"] == EPOCHS
          and base_m["patience"] == arm_m["patience"] == PATIENCE,
          f"base={base_m['epochs']}/{base_m['patience']} arm={arm_m['epochs']}/{arm_m['patience']}")
    check("identity.git_clean_both_runs",
          base_c["git"]["dirty"] is False and arm_c["git"]["dirty"] is False,
          f"base={base_c['git']} arm={arm_c['git']}")
    check("identity.run_id_embeds_config_commit",
          base_m["run_id"].endswith(f"-{base_c['commit']}")
          and arm_m["run_id"].endswith(f"-{arm_c['commit']}{ARM_RUN_ID_SUFFIX}"),
          f"base_commit={base_c['commit']} arm_commit={arm_c['commit']}")
    check("identity.budgets_and_tag",
          base_m["budgets"] == arm_m["budgets"] == {"train": 2000000, "val": 500000, "test": 1000000}
          and base_m["tag"] == arm_m["tag"] == "long", base_m["tag"])

    # ---------------- B. 真冻结链 + 张量 sha + 产物文件 ----------------
    stage_dir = root / "stage1" / SEED2_STAGE1_ID
    meta = load(stage_dir / "meta.json")
    for fname, pin in STAGE1_FILE_SHAS.items():
        check(f"artifact.{fname}.sha256", sha256_file(stage_dir / fname) == pin)
    check("artifact.stage1_id_recompute",
          f"s1-{meta['fingerprint_sha256'][:8]}-m{meta['model_seed']}-e{meta['epochs']}"
          f"-{meta['config_hash'][:8]}" == meta["stage1_id"] == SEED2_STAGE1_ID)
    check("artifact.fingerprint_field", meta["fingerprint_sha256"] == FINGERPRINT_SHA)
    check("artifact.config_hash_field", meta["config_hash"] == CONFIG_HASH)
    for tag, m in (("baseline", base_m), ("arm", arm_m)):
        check(f"freeze.{tag}.sha_chain",
              m["backbone_sha256_loaded"] == m["backbone_sha256_before"] == m["backbone_sha256_after"]
              == meta["backbone_sha256"] == BACKBONE_SHA)
        check(f"freeze.{tag}.env_fingerprint_grads",
              m["env_ids_sha256"] == meta["env_ids_sha256"] == ENV_IDS_SHA
              and m["fingerprint_sha256"] == meta["fingerprint_sha256"] == FINGERPRINT_SHA
              and m["backbone_grads_none"] is True)

    # ---------------- C. A/B 门禁 ----------------
    for tag, g in (("baseline", base_g), ("arm", arm_g)):
        gv = {k: v["verdict"] for k, v in g["gates"].items()}
        check(f"gates.{tag}.a_class",
              all(gv[k] == "PASS" for k in ("A1", "A2", "A4", "A5", "A6")) and gv["A3"] == "SKIP",
              json.dumps(gv))
        check(f"gates.{tag}.b_class_and_hard_pass",
              gv["B1"] == "PASS" and gv["B2"] == "PASS" and gv["B3"] == "PASS"
              and gv["B4"] == "FAIL" and g["hard_pass"] is False,
              f"B1={g['gates']['B1']['verdict']} B4={g['gates']['B4']['verdict']} hard_pass={g['hard_pass']}")
        b1 = g["gates"]["B1"]
        b1_detail = b1["detail"]
        check(f"gates.{tag}.b1_legs",
              f"CTR {meta['best_val_auc_ctr']:.4f}>=0.55:True" in b1_detail
              and f"CVR {meta['best_val_auc_cvr']:.4f}>=0.5:True" in b1_detail
              and f"BSI {(arm_m if tag == 'arm' else base_m)['test_auc_bsi']:.4f}>=0.53:True" in b1_detail,
              b1_detail)

    # ---------------- D/E. 机制门禁 + 三份 JSON 互洽 + M0/U 口径 ----------------
    rp = arm_m["rp_arm"]
    check("mechanism.three_json_agree",
          rp == arm_g["residual_prompt"] == arm_p["arm"],
          f"classification={rp['classification']}")
    check("mechanism.all_gates_pass",
          all(rp[gate]["pass"] is True for gate in MECHANISM_GATES),
          json.dumps({gate: rp[gate]["pass"] for gate in MECHANISM_GATES}))
    ref_recorded = arm_p["reference_dispersion"]["newtask_checkpoint"]
    check("mechanism.reference_path_pinned",
          Path(ref_recorded) == Path(REFERENCE_CHECKPOINT_REL), repr(ref_recorded))
    check("mechanism.reference_file_sha_pinned",
          sha256_file(Path(REFERENCE_CHECKPOINT_REL)) == REFERENCE_CHECKPOINT_SHA)
    m0 = rp["M0"]["observed"]
    check("mechanism.m0_observed_pinned",
          abs(m0["ref_val_auc"] - BASELINE_AUC_VAL_5EP) <= REFERENCE_TOL
          and abs(m0["ref_pred_std"] - REFERENCE_PRED_STD) <= REFERENCE_TOL,
          json.dumps(m0))
    refd = arm_p["reference_dispersion"]
    check("mechanism.reference_stats_agree",
          refd["val_auc"] == m0["ref_val_auc"]
          and refd["pred_dispersion"]["pred_std"] == m0["ref_pred_std"])
    u = rp["U"]
    check("mechanism.u_pairing_is_5epoch_constants",
          u["U1"]["observed"]["baseline_auc_test"] == BASELINE_AUC_TEST_5EP
          and u["U2"]["observed"]["baseline_auc_val"] == BASELINE_AUC_VAL_5EP,
          "class C 诊断口径（vs 5-epoch 常量；不参与持久性判定）")
    check("mechanism.u_observed_consistent",
          u["U1"]["observed"]["auc_test"] == arm_m["test_auc_bsi"]
          and abs(u["U1"]["observed"]["delta_test"]
                  - (arm_m["test_auc_bsi"] - BASELINE_AUC_TEST_5EP)) <= 1e-15
          and u["U2"]["observed"]["auc_val"] == arm_m["best_val_auc_bsi"]
          and abs(u["U2"]["observed"]["delta_val"]
                  - (arm_m["best_val_auc_bsi"] - BASELINE_AUC_VAL_5EP)) <= 1e-15)

    # ---------------- F. 轨迹自洽 + 判定重算 vs 分析器 ----------------
    for tag, m in (("baseline", base_m), ("arm", arm_m)):
        traj = [r["val_auc_bsi"] for r in m["per_epoch"]]
        check(f"trajectory.{tag}.self_consistent",
              m["best_val_auc_bsi"] == max(traj)
              and m["best_epoch"] == traj.index(max(traj)) + 1
              and len(traj) <= m["epochs"] and m["epochs"] == EPOCHS and m["patience"] == PATIENCE,
              f"epochs_run={len(traj)} best_epoch={m['best_epoch']}")

    delta_test = float(arm_m["test_auc_bsi"]) - float(base_m["test_auc_bsi"])
    delta_val = float(arm_m["best_val_auc_bsi"]) - float(base_m["best_val_auc_bsi"])
    a_class = all(load(root / "runs" / rid / "gate_report.json")["gates"][k]["verdict"] in ("PASS", "SKIP")
                  for rid in (args.baseline_run, args.arm_run)
                  for k in ("A1", "A2", "A3", "A4", "A5", "A6"))
    mech = all(rp[gate]["pass"] is True for gate in MECHANISM_GATES)
    checks_recomputed = {
        "delta_test_ge_min": bool(delta_test >= DELTA_TEST_MIN),
        "delta_val_positive": bool(delta_val > 0.0),
        "a_class_pass": bool(a_class),
        "mechanism_gates_pass": bool(mech),
    }
    classification = "PERSISTS" if all(checks_recomputed.values()) else "NOT_PERSIST"
    analysis_path = arm_dir / "rp_longer_budget_compare.json"
    if not analysis_path.is_file():
        check("judgment.analysis_json_present", False, str(analysis_path))
    else:
        analysis = load(analysis_path)
        v = analysis["verdict"]
        check("judgment.analysis_json_present", True)
        check("judgment.delta_test_bit_equal_vs_analysis",
              v["delta_test_auc"] == delta_test and v["baseline_test_auc"] == base_m["test_auc_bsi"]
              and v["arm_test_auc"] == arm_m["test_auc_bsi"])
        check("judgment.delta_val_bit_equal_vs_analysis",
              v["delta_val_auc"] == delta_val and v["baseline_val_auc"] == base_m["best_val_auc_bsi"]
              and v["arm_val_auc"] == arm_m["best_val_auc_bsi"])
        check("judgment.checks_and_classification_equal",
              v["checks"] == checks_recomputed and v["classification"] == classification
              and analysis["classification"] == classification,
              f"recomputed={classification} analysis={analysis['classification']}")
        check("judgment.identity_all_pass_in_analysis",
              all(analysis["identity_checks"].values()), json.dumps(analysis["identity_checks"]))
        check("judgment.mechanism_equal",
              analysis["mechanism"]["all_pass"] == mech
              and analysis["mechanism"]["failures"] == [g for g in MECHANISM_GATES if rp[g]["pass"] is not True])
        check("judgment.reference_sha_in_analysis",
              analysis["reference"]["sha256"] == REFERENCE_CHECKPOINT_SHA)
        ctx = analysis["context"]["five_epoch"]
        check("judgment.context_pinned",
              ctx["delta_test_auc"] == 0.008103113327467715 and ctx["delta_val_auc"] == 0.008622497456618139
              and abs(analysis["context"]["delta_test_change_vs_five_epoch"]
                      - (delta_test - 0.008103113327467715)) <= 1e-15)

    # ---------------- G. 确定性附属核对（描述性；不入判定） ----------------
    determinism = {}
    old10 = root / "runs" / OLD_10EP_BASELINE_RUN / "metrics.json"
    if old10.is_file():
        old10m = load(old10)
        fresh_traj = [r["val_auc_bsi"] for r in base_m["per_epoch"]]
        old_traj = [r["val_auc_bsi"] for r in old10m["per_epoch"]]
        determinism["fresh_baseline_traj_bit_equal_to_old_10ep"] = bool(fresh_traj == old_traj)
        determinism["fresh_baseline_test_bit_equal_to_old_10ep"] = bool(
            base_m["test_auc_bsi"] == old10m["test_auc_bsi"] == OLD_10EP_BASELINE_TEST_AUC)
        check("determinism.fresh_baseline_vs_old_10epoch",
              determinism["fresh_baseline_traj_bit_equal_to_old_10ep"]
              and determinism["fresh_baseline_test_bit_equal_to_old_10ep"],
              json.dumps(determinism))
    old5 = root / "runs" / OLD_5EP_ARM_RUN / "metrics.json"
    if old5.is_file():
        old5m = load(old5)
        arm_traj = [r["val_auc_bsi"] for r in arm_m["per_epoch"]]
        old5_traj = [r["val_auc_bsi"] for r in old5m["per_epoch"]]
        determinism["arm_first5_bit_equal_to_5epoch_arm"] = bool(arm_traj[:5] == old5_traj)
        check("determinism.arm_first5_vs_5epoch_arm",
              determinism["arm_first5_bit_equal_to_5epoch_arm"], json.dumps(determinism))
    old5b = root / "runs" / OLD_5EP_BASELINE_RUN / "metrics.json"
    if old5b.is_file():
        old5bm = load(old5b)
        determinism["baseline_first5_bit_equal_to_5epoch_baseline"] = bool(
            [r["val_auc_bsi"] for r in base_m["per_epoch"]][:5]
            == [r["val_auc_bsi"] for r in old5bm["per_epoch"]])
        check("determinism.baseline_first5_vs_5epoch_baseline",
              determinism["baseline_first5_bit_equal_to_5epoch_baseline"], json.dumps(determinism))

    # ---------------- H. SUMMARY 行 ----------------
    summary_lines = (root / "SUMMARY.md").read_text(encoding="utf-8").splitlines()
    for run_id, m, cfg in ((args.baseline_run, base_m, base_c), (args.arm_run, arm_m, arm_c)):
        rows = [line for line in summary_lines if f"| {run_id} |" in line]
        ok = len(rows) == 1
        detail = f"rows={len(rows)}"
        if ok:
            cells = [c.strip() for c in rows[0].strip("|").split("|")]
            ok = (cells[1] == cfg["commit"] and cells[2] == "long"
                  and cells[3] == f"{m['best_val_auc_bsi']:.6f}" and cells[4] == f"{m['test_auc_bsi']:.6f}"
                  and cells[-1] == SEED2_STAGE1_ID)
            detail = json.dumps(cells[:5] + cells[-1:], ensure_ascii=False)
        check(f"summary.{run_id}", ok, detail)

    ok_all = all(c["pass"] for c in checks)
    report = {
        "script": Path(__file__).name,
        "baseline_run": args.baseline_run, "arm_run": args.arm_run,
        "recomputed": {"delta_test_auc": delta_test, "delta_val_auc": delta_val,
                       "checks": checks_recomputed, "classification": classification},
        "determinism_checks": determinism,
        "checks": checks, "n_checks": len(checks), "all_pass": ok_all,
    }
    out_path = Path(args.output) if args.output else arm_dir / "verify_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nclassification(recomputed)={classification} delta_test={delta_test!r} delta_val={delta_val!r}")
    print(f"ALL_PASS = {ok_all}  ({len(checks)} checks; report -> {out_path})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
