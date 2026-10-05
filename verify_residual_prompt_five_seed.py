"""独立复核：canonical 五 seed 残差 prompt 配对验证（预注册 §11）。

不 import 机制模块（`aliccp_benchmark.residual_prompt`）与汇总模块（`five_seed`）：全部判定从 run JSON
与 stage1 meta 独立重推；Student-t 分位用 scipy.stats.t.ppf（不读 five_seed 的常量，只对照其数值）。

覆盖：
A. 三个新 arm 的逐臂复核（写入各 run 目录 verify_report.json）：A 类链、M0/G1–G8、U1/U2、三标签分类、
   run-reference pin 闭合（pin ↔ 配对基线 ↔ arm 记录，逐位）、cross-JSON 一致、B1/B4 继承、SUMMARY 行。
B. seed1/seed2 遗留臂的独立重推（复制自其分支 worktree，sha 已核）：数值/分类/机制与审计记录一致。
C. 重跑位等性（§10.6 披露项）：5 个 clean 重跑 vs 首跑（metrics/rp_arm/newtask.pt 逐位）。
D. 五 seed 汇总独立重算（均值/样本 std/Student-t CI/计数/最差 seed/符号一致性/二级分类/20-epoch 条件），
   与 five_seed_report.json 逐位对照；并核对文档 §10 记录值。

用法（cwd = worktree 根；主树 venv 解释器）：
    python verify_residual_prompt_five_seed.py
退出码 0 = 全部一致（ALL_PASS）；非 0 = 存在不一致（报告仍落盘）。
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

from scipy.stats import t as student_t

REPO = Path(__file__).resolve().parent
ROOT = REPO / "artifacts" / "aliccp_bench"
DOC = REPO / "docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-five-seed-design.md"

EXTRA_PARAM_NAMES = ("prompt_gate", "prompt_generator.0.bias", "prompt_generator.0.weight",
                     "prompt_generator.2.bias", "prompt_generator.2.weight")
RATIO_BAND = (0.005, 0.5)
BOUND_TOL = 1e-6
PRED_STD_MIN_RATIO = 0.5
REFERENCE_IDENTITY_TOL = 1e-9
U1_MIN = 0.0055
SECONDARY_POSITIVE_MIN = 0.001
SECONDARY_DEGRADATION_MAX = -0.02
T_975_DF4 = float(student_t.ppf(0.975, 4))

# 配对表（canonical 五 seed；§2/§10）
PAIRS = [
    {"model_seed": 1688723512, "baseline_run": "20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9",
     "arm_run": "20261004-0214-p2M-v500k-t1M-m1688723512-short-79ddefa-rpg", "legacy": True,
     "expected_ref_auc": 0.5781533414372665, "expected_ref_std": 0.0033728455401762676,
     "summary_branch": "exp/aliccp-stage2-residual-prompt-gate"},
    {"model_seed": 1688723740, "baseline_run": "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07",
     "arm_run": "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg", "legacy": True,
     "expected_ref_auc": 0.5809347091990792, "expected_ref_std": 0.005217193225189258,
     "summary_branch": "exp/aliccp-stage2-residual-prompt-seed-replication"},
    {"model_seed": 1688738016, "baseline_run": "20261005-0632-p2M-v500k-t1M-m1688738016-short-c17b100",
     "arm_run": "20261005-0701-p2M-v500k-t1M-m1688738016-short-ab7f337-rpg", "legacy": False, "pin_seed": 1688738016},
    {"model_seed": 1688749593, "baseline_run": "20261005-0654-p2M-v500k-t1M-m1688749593-short-23eb4d0",
     "arm_run": "20261005-0705-p2M-v500k-t1M-m1688749593-short-e86f938-rpg", "legacy": False, "pin_seed": 1688749593},
    {"model_seed": 1688762746, "baseline_run": "20261005-0657-p2M-v500k-t1M-m1688762746-short-2fba320",
     "arm_run": "20261005-0709-p2M-v500k-t1M-m1688762746-short-9f9da65-rpg", "legacy": False, "pin_seed": 1688762746},
]
# §10.6 重跑位等性（首跑 → clean 重跑）
RERUNS = [
    ("20261005-0635-p2M-v500k-t1M-m1688749593-short-c17b100",
     "20261005-0654-p2M-v500k-t1M-m1688749593-short-23eb4d0", "metrics"),
    ("20261005-0638-p2M-v500k-t1M-m1688762746-short-c17b100",
     "20261005-0657-p2M-v500k-t1M-m1688762746-short-2fba320", "metrics"),
    ("20261005-0642-p2M-v500k-t1M-m1688738016-short-c17b100-rpg",
     "20261005-0701-p2M-v500k-t1M-m1688738016-short-ab7f337-rpg", "arm"),
    ("20261005-0646-p2M-v500k-t1M-m1688749593-short-c17b100-rpg",
     "20261005-0705-p2M-v500k-t1M-m1688749593-short-e86f938-rpg", "arm"),
    ("20261005-0649-p2M-v500k-t1M-m1688762746-short-c17b100-rpg",
     "20261005-0709-p2M-v500k-t1M-m1688762746-short-9f9da65-rpg", "arm"),
]

_root_checks: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _root_checks.append({"name": name, "pass": bool(ok), "detail": detail})
    return bool(ok)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(run_id: str, name: str) -> dict:
    return json.loads((ROOT / "runs" / run_id / name).read_text(encoding="utf-8"))


def recompute_rp(metrics: dict, gate: dict, prompt: dict, fallback_ref_auc: float,
                 fallback_ref_std: float, *, tol: float = REFERENCE_IDENTITY_TOL) -> dict:
    """独立重推 M0/G1–G8/U/分类（不 import 机制模块；镜像预注册 §4/§5 规则）。"""
    arm = metrics["rp_arm"]
    ref = prompt["reference_dispersion"]
    ref_auc, ref_std = float(ref["val_auc"]), float(ref["pred_dispersion"]["pred_std"])
    m0 = (abs(ref_auc - fallback_ref_auc) <= tol) and (abs(ref_std - fallback_ref_std) <= tol)

    c = prompt["construction_identity"]
    g1 = (bool(c["shared_params_bit_identical"]) and bool(c["global_rng_endpoint_identical"])
          and c["extra_keys"] == sorted(EXTRA_PARAM_NAMES))
    g2 = bool(c["alpha_at_construction"] == 0.0 and prompt["init_forward"]["bit_identical"])
    records = prompt["grad_probe"]
    alpha_final = float(prompt["alpha_final"])
    if records:
        first, last = records[0], records[-1]
        g3 = bool(float(first.get("alpha_grad_norm") or 0.0) != 0.0
                  and float(last.get("alpha") or 0.0) != 0.0
                  and float(last.get("generator_grad_norm") or 0.0) > 0.0
                  and alpha_final != 0.0)
    else:
        g3 = False
    ratio_max = [float(v) for v in prompt["val_stats"]["ratio_max"]]
    g4 = all(v <= abs(alpha_final) + BOUND_TOL for v in ratio_max)
    ratio_mean = [float(v) for v in prompt["val_stats"]["ratio_mean"]]
    band_low = any(v < RATIO_BAND[0] for v in ratio_mean)
    band_high = any(v > RATIO_BAND[1] for v in ratio_mean)
    g5 = not band_low and not band_high
    gate_stats = prompt["gate"]
    g6 = bool(float(gate_stats["geff_std"]) > 0.0 and float(gate_stats["geff_max"]) > 0.0
              and float(gate_stats["geff_min"]) >= 0.0)
    pred_std = float(prompt["dispersion"]["pred_std"])
    g7 = bool(pred_std > 0.0 and ref_std > 0.0 and pred_std >= PRED_STD_MIN_RATIO * ref_std)
    names = sorted(str(i["name"]) for i in prompt["params"]["new_param_list"])
    g8 = bool(names == sorted(EXTRA_PARAM_NAMES)
              and int(prompt["params"]["new_params_total"]) == 2385
              and int(prompt["params"]["head_params"]) == 8129)

    delta_test = float(metrics["test_auc_bsi"]) - float(arm["U"]["U1"]["observed"]["baseline_auc_test"])
    delta_val = float(metrics["best_val_auc_bsi"]) - float(arm["U"]["U2"]["observed"]["baseline_auc_val"])
    u1, u2 = delta_test >= U1_MIN, delta_val > 0.0
    protocol_ok = all(gate["gates"][k]["verdict"] in ("PASS", "SKIP") for k in ("A1", "A2", "A3", "A4", "A5", "A6"))
    if not m0:
        cls, sub = "MECHANISM_FAIL", "REFERENCE_IDENTITY"
    elif not (g1 and g2 and g4 and g8):
        cls, sub = "MECHANISM_FAIL", "INVALID_IMPLEMENTATION"
    elif not g3:
        cls, sub = "MECHANISM_FAIL", "MECHANISM_INACTIVE"
    elif band_low:
        cls, sub = "MECHANISM_FAIL", "MECHANISM_SILENT"
    elif band_high:
        cls, sub = "MECHANISM_FAIL", "MECHANISM_OVER_PERTURB"
    elif not g6:
        cls, sub = "MECHANISM_FAIL", "GATE_DEGENERATE"
    elif not g7:
        cls, sub = "MECHANISM_FAIL", "PREDICTION_COLLAPSE"
    elif not protocol_ok:
        cls, sub = "MECHANISM_FAIL", "PROTOCOL_INVALID"
    elif u1 and u2:
        cls, sub = "VALID_POSITIVE", None
    else:
        cls, sub = "VALID_NEGATIVE", None
    return {"M0": m0, "G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6, "G7": g7, "G8": g8,
            "U1": u1, "U2": u2, "delta_test": delta_test, "delta_val": delta_val,
            "classification": cls, "subreason": sub, "protocol_ok": protocol_ok,
            "ref_auc": ref_auc, "ref_std": ref_std}


def secondary_classification(delta_test: float) -> str:
    if delta_test >= SECONDARY_POSITIVE_MIN:
        return "POSITIVE_IMPROVEMENT"
    if delta_test <= SECONDARY_DEGRADATION_MAX:
        return "CLEAR_DEGRADATION"
    return "NO_CLEAR_IMPROVEMENT"


def verify_pair(pair: dict, checks: list, tag: str) -> dict:
    def c(name, ok, detail=""):
        checks.append({"name": f"{tag}.{name}", "pass": bool(ok), "detail": detail})

    seed = pair["model_seed"]
    base_m = _read(pair["baseline_run"], "metrics.json")
    arm_m = _read(pair["arm_run"], "metrics.json")
    arm_g = _read(pair["arm_run"], "gate_report.json")
    arm_p = _read(pair["arm_run"], "prompt_report.json")
    art_meta = json.loads((ROOT / "stage1" / arm_m["stage1_id"] / "meta.json").read_text(encoding="utf-8"))

    c("pair_seed_and_stage1", base_m["model_seed"] == seed and arm_m["model_seed"] == seed
      and base_m["stage1_id"] == arm_m["stage1_id"], arm_m["stage1_id"])
    c("variant_is_residual_prompt", arm_m.get("variant") == "residual-prompt")

    pin = None
    if not pair["legacy"]:
        pin = json.loads((ROOT / "audit" / "five-seed" / f"reference_pin_seed{pair['pin_seed']}.json")
                         .read_text(encoding="utf-8"))
        c("pin_values_equal_baseline", pin["baseline_auc_test"] == base_m["test_auc_bsi"]
          and pin["baseline_auc_val"] == base_m["best_val_auc_bsi"],
          f"pin={pin['baseline_auc_test']!r}/{pin['baseline_auc_val']!r}")
        c("arm_recorded_baseline_equals_pin", float(arm_m["rp_arm"]["U"]["U1"]["observed"]["baseline_auc_test"]) == pin["baseline_auc_test"]
          and float(arm_m["rp_arm"]["U"]["U2"]["observed"]["baseline_auc_val"]) == pin["baseline_auc_val"])
        expected_ref_auc, expected_ref_std = pin["baseline_auc_val"], pin["reference_pred_std"]
        c("arm_git_clean", arm_m["git"]["dirty"] is False, json.dumps(arm_m["git"]))
    else:
        expected_ref_auc, expected_ref_std = pair["expected_ref_auc"], pair["expected_ref_std"]
        c("arm_recorded_baseline_equals_baseline", float(arm_m["rp_arm"]["U"]["U1"]["observed"]["baseline_auc_test"]) == base_m["test_auc_bsi"]
          and float(arm_m["rp_arm"]["U"]["U2"]["observed"]["baseline_auc_val"]) == base_m["best_val_auc_bsi"])

    # A 类链（从 run/meta 复核）
    a = arm_g["gates"]
    c("A_class_recorded", all(a[k]["verdict"] in ("PASS", "SKIP") for k in ("A1", "A2", "A3", "A4", "A5", "A6")),
      json.dumps({k: a[k]["verdict"] for k in a}))
    c("A1_freeze_chain", arm_m["backbone_sha256_before"] == arm_m["backbone_sha256_after"]
      == arm_m["backbone_sha256_loaded"] and arm_m["backbone_grads_none"] is True)
    c("A5_env_ids_identity", arm_m["env_ids_sha256"] == art_meta["env_ids_sha256"])
    c("A6_backbone_identity", arm_m["backbone_sha256_loaded"] == art_meta["backbone_sha256"]
      and art_meta["stage1_id"] == arm_m["stage1_id"])
    c("fingerprint_identity", arm_m["fingerprint_sha256"] == art_meta["fingerprint_sha256"]
      and art_meta["fingerprint_sha256"].startswith("5c060b9c"))

    rp = recompute_rp(arm_m, arm_g, arm_p, expected_ref_auc, expected_ref_std)
    recorded = arm_m["rp_arm"]
    for g in ("M0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"):
        c(f"{g}_recompute_matches_record", rp[g] == bool(recorded[g]["pass"]),
          f"recomputed={rp[g]} recorded={recorded[g]['pass']}")
    c("U1_recompute_matches_record", rp["U1"] == bool(recorded["U"]["U1"]["pass"]),
      f"delta_test={rp['delta_test']!r} (阈值 +0.0055)")
    c("U2_recompute_matches_record", rp["U2"] == bool(recorded["U"]["U2"]["pass"]),
      f"delta_val={rp['delta_val']!r} (阈值 > 0)")
    c("classification_recompute", (rp["classification"], rp["subreason"]) == (recorded["classification"], recorded["subreason"]),
      f"recomputed={rp['classification']}/{rp['subreason']} recorded={recorded['classification']}/{recorded['subreason']}")
    c("cross_json_consistency", recorded == arm_g["residual_prompt"] == arm_p["arm"])
    c("protocol_10_1_report", bool(recorded["protocol_10_1"]["delta_test_ge_0.005"]) == (rp["delta_test"] >= 0.005)
      and bool(recorded["protocol_10_1"]["val_positive"]) == rp["U2"])
    c("hard_pass_expected_false", arm_m["hard_pass"] is False)
    b4_meta = [(e["env_0"], e["env_1"]) for e in art_meta["cluster_events"]]
    c("B4_inherited_from_artifact", arm_g["gates"]["B4"]["verdict"] == ("FAIL" if b4_meta and any(
        e0 < 0.05 * 2_000_000 or e1 < 0.05 * 2_000_000 for e0, e1 in b4_meta) else "PASS"),
      f"events={b4_meta}")
    b1_ctr = art_meta["best_val_auc_ctr"] >= 0.55
    b1_cvr = art_meta["best_val_auc_cvr"] >= 0.50
    b1_expected = "PASS" if (b1_ctr and b1_cvr and float(arm_m["test_auc_bsi"]) >= 0.53) else "FAIL"
    c("B1_recompute_from_artifact_and_arm", arm_g["gates"]["B1"]["verdict"] == b1_expected,
      f"CTR {art_meta['best_val_auc_ctr']:.4f} CVR {art_meta['best_val_auc_cvr']:.4f} test {arm_m['test_auc_bsi']:.4f}")

    summary = (ROOT / "SUMMARY.md").read_text(encoding="utf-8")
    if pair.get("summary_branch"):
        # 遗留臂：台账行记录在其自身分支（本分支按先例不复制他分支行）→ 经 git 读取该分支台账
        import subprocess
        proc = subprocess.run(["git", "show", f"{pair['summary_branch']}:artifacts/aliccp_bench/SUMMARY.md"],
                              cwd=REPO, capture_output=True, text=True)
        summary = proc.stdout if proc.returncode == 0 else ""
    c("summary_row_present", any(pair["arm_run"] in line for line in summary.splitlines()),
      f"source={'branch' if pair.get('summary_branch') else 'local'}")

    return {"model_seed": seed, "stage1_id": arm_m["stage1_id"], "baseline_run": pair["baseline_run"],
            "arm_run": pair["arm_run"], "baseline_test_auc": base_m["test_auc_bsi"],
            "baseline_val_auc": base_m["best_val_auc_bsi"], "arm_test_auc": arm_m["test_auc_bsi"],
            "arm_val_auc": arm_m["best_val_auc_bsi"], "delta_test": rp["delta_test"], "delta_val": rp["delta_val"],
            "u1": rp["U1"], "u2": rp["U2"], "classification": rp["classification"],
            "secondary": secondary_classification(rp["delta_test"]),
            "mechanism_pass": all(rp[g] for g in ("M0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")),
            "a_class_pass": all(a[k]["verdict"] in ("PASS", "SKIP") for k in ("A1", "A2", "A3", "A4", "A5", "A6")),
            "alpha_final": float(arm_p["alpha_final"]),
            "ratio_mean": [float(v) for v in arm_p["val_stats"]["ratio_mean"]],
            "cos_mean": [float(v) for v in arm_p["val_stats"]["cos_mean"]],
            "geff_std": float(arm_p["gate"]["geff_std"]),
            "pred_std": float(arm_p["dispersion"]["pred_std"]), "ref_pred_std": rp["ref_std"],
            "best_epoch": int(arm_m["best_epoch"]), "epochs_run": int(arm_m["epochs"]),
            "per_epoch_val": [float(e["val_auc_bsi"]) for e in arm_m["per_epoch"]],
            "arm_gave_up_metrics": {"best_val_auc_bsi": arm_m["best_val_auc_bsi"], "test_auc_bsi": arm_m["test_auc_bsi"]}}


def verify_reruns() -> None:
    for first, rerun, kind in RERUNS:
        a, b = _read(first, "metrics.json"), _read(rerun, "metrics.json")
        same_metrics = (a["test_auc_bsi"] == b["test_auc_bsi"] and a["best_val_auc_bsi"] == b["best_val_auc_bsi"]
                        and a["per_epoch"] == b["per_epoch"] and a["best_epoch"] == b["best_epoch"])
        same_pt = sha256_file(ROOT / "runs" / first / "newtask.pt") == sha256_file(ROOT / "runs" / rerun / "newtask.pt")
        ok = same_metrics and same_pt and b["git"]["dirty"] is False
        if kind == "arm":
            ok = ok and a["rp_arm"] == b["rp_arm"]
        check(f"rerun_bitequal[{rerun[-7:]}]", ok,
              f"metrics={same_metrics} newtask_pt={same_pt} dirty={b['git']['dirty']} kind={kind}")


def verify_doc(records: list[dict], agg: dict, cond: dict) -> None:
    doc = DOC.read_text(encoding="utf-8")
    missing = []
    for r in records:
        for token in (repr(r["delta_test"]), repr(r["delta_val"]), r["arm_run"], r["baseline_run"], r["secondary"]):
            if token not in doc:
                missing.append(f"{r['model_seed']}:{token}")
    for token in (repr(agg["mean"]), repr(agg["std"]), repr(agg["ci_low"]), repr(agg["ci_high"]), cond["status"]):
        if token not in doc:
            missing.append(f"agg:{token}")
    check("doc_records_match_recomputation", not missing, f"missing={missing[:8]}")


def main() -> int:
    per_seed = []
    pair_checks_all: list = []
    for pair in PAIRS:
        tag = f"seed{pair['model_seed']}"

        def c(name, ok, detail="", _tag=tag):
            _root_checks.append({"name": f"{_tag}.{name}", "pass": bool(ok), "detail": detail})

        checks: list = []
        rec = verify_pair(pair, checks, tag)
        per_seed.append(rec)
        pair_checks_all.extend(checks)
        if not pair["legacy"]:
            run_dir = ROOT / "runs" / pair["arm_run"]
            (run_dir / "verify_report.json").write_text(
                json.dumps({"run_dir": str(run_dir), "checks": checks, "all_pass": all(x["pass"] for x in checks)},
                           ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{tag}] pair checks: {sum(1 for x in checks if x['pass'])}/{len(checks)}")
        for x in checks:
            if not x["pass"]:
                print(f"   FAIL {x['name']}: {x['detail']}")

    verify_reruns()

    dt = [r["delta_test"] for r in per_seed]
    dv = [r["delta_val"] for r in per_seed]
    n = len(dt)
    mean_dt = sum(dt) / n
    std_dt = math.sqrt(sum((v - mean_dt) ** 2 for v in dt) / (n - 1))
    mean_dv = sum(dv) / n
    std_dv = math.sqrt(sum((v - mean_dv) ** 2 for v in dv) / (n - 1))
    half_dt = T_975_DF4 * std_dt / math.sqrt(n)
    half_dv = T_975_DF4 * std_dv / math.sqrt(n)
    agg = {"values": dt, "mean": mean_dt, "std": std_dt, "ci_low": mean_dt - half_dt, "ci_high": mean_dt + half_dt,
           "val_mean": mean_dv, "val_std": std_dv, "val_ci_low": mean_dv - half_dv, "val_ci_high": mean_dv + half_dv,
           "t": T_975_DF4,
           "count_ge_0.001": sum(1 for v in dt if v >= SECONDARY_POSITIVE_MIN),
           "count_ge_0.0055": sum(1 for v in dt if v >= U1_MIN),
           "count_val_gt_0": sum(1 for v in dv if v > 0),
           "worst_seed": per_seed[dt.index(min(dt))]["model_seed"], "worst_delta_test": min(dt)}
    cond = {"C1_mechanism": all(r["mechanism_pass"] and r["a_class_pass"] for r in per_seed),
            "C2_no_degradation": all(r["secondary"] != "CLEAR_DEGRADATION" for r in per_seed),
            "C3_mean_at_least_0.0055": mean_dt >= U1_MIN,
            "C4_ci_low_positive": mean_dt - half_dt > 0.0,
            "C5_positive_count_ge_4": agg["count_ge_0.001"] >= 4,
            "C6_val_positive_count_ge_4": agg["count_val_gt_0"] >= 4}
    cond["status"] = ("TWENTY_EPOCH_CONDITION_SATISFIED" if all(v for k, v in cond.items() if k.startswith("C"))
                      else "TWENTY_EPOCH_CONDITION_NOT_SATISFIED")

    report = json.loads((ROOT / "audit" / "five-seed" / "five_seed_report.json").read_text(encoding="utf-8"))
    fa = report["aggregate"]
    check("agg_mean_matches", fa["delta_test"]["mean"] == mean_dt, f"{fa['delta_test']['mean']!r} vs {mean_dt!r}")
    check("agg_std_matches", fa["delta_test"]["std"] == std_dt)
    check("agg_ci_matches", (fa["delta_test"]["ci95_low"], fa["delta_test"]["ci95_high"]) == (agg["ci_low"], agg["ci_high"]))
    check("agg_val_matches", (fa["delta_val"]["mean"], fa["delta_val"]["std"]) == (mean_dv, std_dv))
    check("agg_counts_match", (fa["counts"]["delta_test_ge_0.001"], fa["counts"]["delta_test_ge_0.0055"],
                               fa["counts"]["delta_val_gt_0"]) == (agg["count_ge_0.001"], agg["count_ge_0.0055"], agg["count_val_gt_0"]))
    check("agg_worst_matches", (fa["delta_test"]["min_model_seed"], fa["delta_test"]["min"]) == (agg["worst_seed"], agg["worst_delta_test"]))
    check("agg_sign_matches", fa["sign_consistency"]["delta_test_all_positive"] is True
          and fa["sign_consistency"]["delta_val_all_positive"] is True)
    check("agg_secondary_matches", all(fa["secondary_counts"][k] == sum(1 for r in per_seed if r["secondary"] == k)
                                       for k in ("POSITIVE_IMPROVEMENT", "NO_CLEAR_IMPROVEMENT", "CLEAR_DEGRADATION")))
    check("agg_condition_matches", report["twenty_epoch_condition"]["status"] == cond["status"]
          and all(bool(report["twenty_epoch_condition"][k]) == v for k, v in cond.items() if k.startswith("C")))
    verify_doc(per_seed, agg, cond)

    all_checks = pair_checks_all + _root_checks
    out = {"pairs": PAIRS, "per_seed_recomputed": per_seed, "aggregate_recomputed": agg,
           "twenty_epoch_recomputed": cond, "checks": all_checks,
           "all_pass": all(x["pass"] for x in all_checks)}
    out_path = ROOT / "audit" / "five-seed" / "verify_five_seed.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    n_pass = sum(1 for x in all_checks if x["pass"])
    print(f"\nAGG recomputed: mean_dt={mean_dt!r} std={std_dt!r} CI95=[{agg['ci_low']!r}, {agg['ci_high']!r}]")
    print(f"counts ge0.001={agg['count_ge_0.001']} ge0.0055={agg['count_ge_0.0055']} val>0={agg['count_val_gt_0']} "
          f"worst={agg['worst_seed']} ({agg['worst_delta_test']!r})")
    print(f"20-epoch independent: {cond['status']}")
    print(f"ALL_PASS = {out['all_pass']}  ({n_pass}/{len(all_checks)} checks)  report={out_path}")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
