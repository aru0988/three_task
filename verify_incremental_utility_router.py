"""独立复核（不 import 任何实现模块）：从 run 产物重算 AUC/delta/阈值/混合/随机路由/判定。

预注册：docs/superpowers/specs/2026-10-06-aliccp-incremental-utility-verifier-design.md §11 前置
（本脚本只读 predictions.npz / routing_report.json / metrics.json / gate_report.json / SUMMARY.md，
独立实现 u、Ridge(lstsq)、阈值网格、混合网格、随机掩码与全部分类逻辑后逐位对照）。

用法：python verify_incremental_utility_router.py --run-id <iuv_run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROUTER_ALPHA = 1.0
THRESHOLD_QUANTILES = tuple(i / 20 for i in range(1, 20))
MIX_GRID = tuple(i / 20 for i in range(0, 21))
ROUTER_RANDOM_SALT = 20261006
EPS = 1e-12
ATOL = 1e-12
RIDGE_ATOL = 1e-8


def bce(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def auc(y, p):
    return float(roc_auc_score(np.asarray(y, dtype=int), np.asarray(p, dtype=np.float64)))


def classify(delta):
    if delta >= 0.001:
        return "POSITIVE_IMPROVEMENT"
    if delta <= -0.02:
        return "CLEAR_DEGRADATION"
    return "NO_CLEAR_IMPROVEMENT"


def close(a, b) -> bool:
    """None/NaN 语义一致的比较：同为 None（或同为 NaN）⇒ 一致；一侧 None 一侧数值 ⇒ 不一致。"""
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    if np.isnan(a) or np.isnan(b):
        return np.isnan(a) and np.isnan(b)
    return abs(a - b) <= ATOL


def auc_opt(y, p):
    """AUC；y 单类或结果非有限 ⇒ None（UNDEFINED）。独立实现，与实现侧 auc_or_none 同语义。"""
    yy = np.asarray(y).astype(np.int64).ravel()
    if yy.size == 0 or len(np.unique(yy)) < 2:
        return None
    v = float(roc_auc_score(yy, np.asarray(p, dtype=np.float64).ravel()))
    return None if np.isnan(v) else v


def fmt6(v) -> str:
    """SUMMARY 单元格口径（与 runner _fmt6 相同）。"""
    return "UNDEFINED" if v is None else f"{float(v):.6f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--root", default="artifacts/aliccp_bench")
    args = ap.parse_args(argv)
    run_dir = Path(args.root) / "runs" / args.run_id
    checks: dict = {}

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks[name] = {"pass": bool(ok), "detail": detail}

    z = np.load(run_dir / "predictions.npz")
    rep = json.loads((run_dir / "routing_report.json").read_text(encoding="utf-8"))
    mtx = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    gate = json.loads((run_dir / "gate_report.json").read_text(encoding="utf-8"))
    model_seed = int(mtx["model_seed"])

    # 1) 数组形状与分割
    nB = len(z["y_B"])
    nC = len(z["y_C"])
    nT = len(z["y_T"])
    for name, n in (("y_B", nB), ("pB_B", nB), ("pP_B", nB), ("X_B", nB), ("u_B", nB), ("s_B", nB)):
        check(f"shape_{name}", len(z[name]) == nB, f"{len(z[name])} vs {nB}")
    for name, n in (("y_C", nC), ("pB_C", nC), ("pP_C", nC), ("u_C", nC), ("s_C", nC)):
        check(f"shape_{name}", len(z[name]) == nC, f"{len(z[name])} vs {nC}")
    for name, n in (("y_T", nT), ("pB_T", nT), ("pP_T", nT), ("s_T", nT)):
        check(f"shape_{name}", len(z[name]) == nT, f"{len(z[name])} vs {nT}")
    bounds = {k: tuple(v) for k, v in rep["split"]["bounds"].items()}
    check("split_sizes", nB == bounds["B"][1] - bounds["B"][0] and nC == bounds["C"][1] - bounds["C"][0],
          f"bounds={bounds} nB={nB} nC={nC}")
    check("split_contiguous", bounds["A"][1] == bounds["B"][0] and bounds["B"][1] == bounds["C"][0]
          and bounds["A"][0] == 0 and bounds["C"][1] == rep["split"]["n_train"],
          f"bounds={bounds} n_train={rep['split']['n_train']}")

    # 2) 特征零标签一致性（列 0..2 必须逐位由两臂 logits 复现；8 列）
    X = np.asarray(z["X_B"], dtype=np.float64)
    check("feature_dim", X.shape[1] == 8, f"shape={X.shape}")
    check("feature_cols_0_2", np.allclose(X[:, 0], z["pB_B"], rtol=0, atol=ATOL)
          and np.allclose(X[:, 1], z["pP_B"], rtol=0, atol=ATOL)
          and np.allclose(X[:, 2], np.asarray(z["pP_B"]) - np.asarray(z["pB_B"]), rtol=0, atol=ATOL),
          "cols=[pB,pP,pP-pB]")

    # 3) 独立重推 u 与有益标签（1[u>0]）
    uB = bce(z["pB_B"], z["y_B"]) - bce(z["pP_B"], z["y_B"])
    uC = bce(z["pB_C"], z["y_C"]) - bce(z["pP_C"], z["y_C"])
    check("u_B_bitmatch", np.allclose(uB, z["u_B"], rtol=0, atol=ATOL))
    check("u_C_bitmatch", np.allclose(uC, z["u_C"], rtol=0, atol=ATOL))
    check("u_B_positive_rate", abs(float((uB > 0).mean()) - rep["diagnostics"]["u_B_positive_rate"]) <= ATOL)
    check("u_C_positive_rate", abs(float((uC > 0).mean()) - rep["diagnostics"]["u_C_positive_rate"]) <= ATOL)

    # 4) 独立重拟合 Ridge（lstsq 增广系统：权重受罚、截距不受罚）
    mean = X.mean(axis=0)
    std = np.where(X.std(axis=0) == 0.0, 1.0, X.std(axis=0))
    Xs = (X - mean) / std
    d = Xs.shape[1]
    design = np.vstack([np.hstack([Xs, np.ones((Xs.shape[0], 1))]),
                        np.hstack([np.sqrt(ROUTER_ALPHA) * np.eye(d), np.zeros((d, 1))])])
    target = np.concatenate([uB, np.zeros(d)])
    sol, *_ = np.linalg.lstsq(design, target, rcond=None)
    w_ref, b_ref = sol[:d], float(sol[d])
    w_rec = np.asarray(rep["router"]["ridge"]["w"], dtype=np.float64)
    b_rec = float(rep["router"]["ridge"]["b"])
    check("ridge_w", np.allclose(w_ref, w_rec, rtol=1e-6, atol=RIDGE_ATOL), f"max|Δw|={np.max(np.abs(w_ref - w_rec)):.3e}")
    check("ridge_b", abs(b_ref - b_rec) <= RIDGE_ATOL, f"|Δb|={abs(b_ref - b_rec):.3e}")
    check("ridge_std_mean", np.allclose(np.asarray(rep["router"]["ridge"]["mean"]), mean, rtol=0, atol=ATOL)
          and np.allclose(np.asarray(rep["router"]["ridge"]["std"]), std, rtol=0, atol=ATOL))

    def score(Z):
        return ((np.asarray(Z, dtype=np.float64) - mean) / std) @ w_rec + b_rec

    check("s_B_bitmatch", np.allclose(score(z["X_B"]), z["s_B"], rtol=0, atol=ATOL))
    check("s_C_bitmatch", np.allclose(score(z["X_C"]), z["s_C"], rtol=0, atol=ATOL))

    # 5) 阈值（仅 C）：独立重建网格、并列保守口径与 UNDEFINED 回退（照抄预注册/统筹第四轮口径）
    sC = np.asarray(z["s_C"], dtype=np.float64)
    pB_C = np.asarray(z["pB_C"], dtype=np.float64)
    pP_C = np.asarray(z["pP_C"], dtype=np.float64)
    yC = np.asarray(z["y_C"], dtype=int)
    cands = [-np.inf] + [float(np.quantile(sC, q)) for q in THRESHOLD_QUANTILES] + [np.inf]
    best, n_undef = None, 0
    for idx, thr in enumerate(cands):
        pred = np.where(sC > thr, pP_C, pB_C)
        a = auc_opt(yC, pred)
        pi = float((sC > thr).mean())
        if a is None:
            n_undef += 1
            continue
        key = (a, -pi, -idx)
        if best is None or key > best[0]:
            best = (key, thr, pi, a)
    if best is None:
        thr_ref, pi_ref, aucC_ref, status_ref = float("inf"), 0.0, None, "UNDEFINED"
    else:
        thr_ref, pi_ref, aucC_ref, status_ref = float(best[1]), float(best[2]), float(best[3]), "DEFINED"
    thr_rec = float(rep["router"]["threshold"]["thr"])
    pi_rec = rep["router"]["threshold"]["pi_hat"]
    check("threshold_status", status_ref == rep["router"]["threshold"]["calibration_status"],
          f"ref={status_ref} rec={rep['router']['threshold']['calibration_status']}")
    check("threshold_n_undefined", n_undef == int(rep["router"]["threshold"]["n_undefined_candidates"]),
          f"ref={n_undef} rec={rep['router']['threshold']['n_undefined_candidates']}")
    check("threshold_thr", thr_ref == thr_rec, f"ref={thr_ref} rec={thr_rec}")
    check("threshold_pi_hat", close(pi_ref, pi_rec), f"ref={pi_ref} rec={pi_rec}")
    check("threshold_auc_C", close(aucC_ref, rep["router"]["threshold"]["auc_C_routed"]),
          f"ref={aucC_ref} rec={rep['router']['threshold']['auc_C_routed']}")

    # 6) 固定混合（仅 C；UNDEFINED 口径同阈值）
    mix_best, n_undef_m = None, 0
    for a in MIX_GRID:
        v = auc_opt(yC, a * pP_C + (1.0 - a) * pB_C)
        if v is None:
            n_undef_m += 1
            continue
        key = (v, -float(a))
        if mix_best is None or key > mix_best[0]:
            mix_best = (key, float(a), v)
    if mix_best is None:
        alpha_ref, aucM_ref, statusM_ref = 0.0, None, "UNDEFINED"
    else:
        alpha_ref, aucM_ref, statusM_ref = mix_best[1], mix_best[2], "DEFINED"
    check("mix_status", statusM_ref == rep["router"]["mix"]["calibration_status"],
          f"ref={statusM_ref} rec={rep['router']['mix']['calibration_status']}")
    check("mix_n_undefined", n_undef_m == int(rep["router"]["mix"]["n_undefined_candidates"]),
          f"ref={n_undef_m} rec={rep['router']['mix']['n_undefined_candidates']}")
    check("mix_alpha", alpha_ref == float(rep["router"]["mix"]["alpha_hat"]),
          f"ref={alpha_ref} rec={rep['router']['mix']['alpha_hat']}")
    check("mix_auc_C", close(aucM_ref, rep["router"]["mix"]["auc_C_mix"]),
          f"ref={aucM_ref} rec={rep['router']['mix']['auc_C_mix']}")

    # 7) test 六配置（独立重算并进行逐位对照）
    yT = np.asarray(z["y_T"], dtype=int)
    pTb = np.asarray(z["pB_T"], dtype=np.float64)
    pTp = np.asarray(z["pP_T"], dtype=np.float64)
    sT = np.asarray(z["s_T"], dtype=np.float64)
    a_hat = float(rep["router"]["mix"]["alpha_hat"])
    thr = float(rep["router"]["threshold"]["thr"])
    seed = model_seed ^ ROUTER_RANDOM_SALT
    check("random_seed_recorded", int(rep["router"]["random_seed"]) == seed)
    mask = np.random.default_rng(seed).random(len(yT)) < pi_rec
    check("random_rate_matches_pi", abs(float(mask.mean()) - pi_rec) <= 1e-6, f"rate={mask.mean():.6f} pi={pi_rec:.6f}")
    uT = bce(pTb, yT) - bce(pTp, yT)
    preds = {
        "always_baseline": pTb,
        "always_prompt": pTp,
        "fixed_mix": a_hat * pTp + (1.0 - a_hat) * pTb,
        "routed": np.where(sT > thr, pTp, pTb),
        "random_router": np.where(mask, pTp, pTb),
        "label_assisted_bce_oracle_diag": np.where(uT > 0.0, pTp, pTb),
    }
    aucs_ref = {k: auc_opt(yT, v) for k, v in preds.items()}
    aucs_rec = mtx["aucs_test"]
    for k in preds:
        check(f"auc_test_{k}", close(aucs_ref[k], aucs_rec[k]),
              f"ref={aucs_ref[k]} rec={aucs_rec[k]}")

    def _delta(a, b):
        return None if (a is None or b is None) else float(a) - float(b)

    d_base = _delta(aucs_ref["routed"], aucs_ref["always_baseline"])
    d_prompt = _delta(aucs_ref["routed"], aucs_ref["always_prompt"])
    d_mix = _delta(aucs_ref["routed"], aucs_ref["fixed_mix"])
    for name, ref in (("base", d_base), ("prompt", d_prompt), ("mix", d_mix)):
        check(f"delta_{name}", close(ref, mtx["deltas_test"][name]),
              f"ref={ref} rec={mtx['deltas_test'][name]}")
    if d_base is None:
        cls_ref, rv_ref = "UNDEFINED", False
    else:
        cls_ref = classify(d_base)
        rv_ref = bool(d_base >= 0.001 and d_prompt > 0.0 and d_mix > 0.0)
    check("classification", cls_ref == mtx["classification"], f"ref={cls_ref} rec={mtx['classification']}")
    check("router_value", rv_ref == bool(mtx["router_value"]), f"ref={rv_ref} rec={mtx['router_value']}")

    # 7b) 官方 validation（独立复算；routed val AUC 即 SUMMARY 的 auc_val_bsi_best）
    yV = np.asarray(z["y_V"], dtype=int)
    pVb = np.asarray(z["pB_V"], dtype=np.float64)
    pVp = np.asarray(z["pP_V"], dtype=np.float64)
    sV = ((np.asarray(z["X_V"], dtype=np.float64) - mean) / std) @ w_rec + b_rec
    check("s_V_bitmatch", np.allclose(sV, z["s_V"], rtol=0, atol=ATOL))
    uV = bce(pVb, yV) - bce(pVp, yV)
    preds_V = {
        "always_baseline": pVb,
        "always_prompt": pVp,
        "fixed_mix": a_hat * pVp + (1.0 - a_hat) * pVb,
        "routed": np.where(sV > thr, pVp, pVb),
        "random_router": np.where(np.random.default_rng(seed).random(len(yV)) < pi_rec, pVp, pVb),
        "label_assisted_bce_oracle_diag": np.where(uV > 0.0, pVp, pVb),
    }
    aucs_V_ref = {k: auc_opt(yV, v) for k, v in preds_V.items()}
    for k in preds_V:
        check(f"auc_val_{k}", close(aucs_V_ref[k], mtx["aucs_val"][k]),
              f"ref={aucs_V_ref[k]} rec={mtx['aucs_val'][k]}")

    # 8) 诊断（AUROC/Spearman，仅机制性；单类/常量输入 ⇒ None=UNDEFINED）
    auroc_ref = auc_opt((uC > 0).astype(int), sC)
    check("auroc_C_vs_u_gt_0", close(auroc_ref, rep["diagnostics"]["auroc_C_vs_u_gt_0"]),
          f"ref={auroc_ref} rec={rep['diagnostics']['auroc_C_vs_u_gt_0']}")
    _sp = spearmanr(sC, uC).statistic
    sp_ref = None if (_sp is None or np.isnan(_sp)) else float(_sp)
    check("spearman_C", close(sp_ref, rep["diagnostics"]["spearman_C"]),
          f"ref={sp_ref} rec={rep['diagnostics']['spearman_C']}")

    # 9) 门禁与台账
    a_gates = gate["gates"]
    a_ok = all(a_gates[g]["verdict"] in ("PASS", "SKIP") for g in ("A1", "A2", "A3", "A4", "A5", "A6"))
    b_ok = all(a_gates[g]["verdict"] in ("PASS", "N/A") for g in ("B1", "B2", "B3", "B4"))
    hard_ref = a_ok and (b_ok if gate["enforce_b"] else True)
    check("hard_pass", hard_ref == bool(gate["hard_pass"]), f"ref={hard_ref} rec={gate['hard_pass']}")
    r_ok = all(a_gates[g]["verdict"] == "PASS" for g in ("R1", "R2", "R3", "R4", "R5"))
    check("R_gates", r_ok and bool(gate["all_gates_ok"]) == (hard_ref and r_ok))
    summary_lines = (Path(args.root) / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
    last = summary_lines[-1] if summary_lines else ""
    fields = [c.strip() for c in last.split("|")][1:-1]
    check("summary_row_id", bool(fields) and fields[0] == args.run_id, last[:80])
    check("summary_auc_val_routed", len(fields) > 4 and fields[3] == fmt6(aucs_V_ref["routed"]),
          f"cell={fields[3] if len(fields) > 3 else 'NA'}")
    check("summary_auc_test_routed", len(fields) > 4 and fields[4] == fmt6(aucs_ref["routed"]),
          f"cell={fields[4] if len(fields) > 4 else 'NA'}")
    check("git_dirty_false", mtx["git"]["dirty"] is False, str(mtx["git"]))

    all_pass = all(v["pass"] for v in checks.values())
    report = {"run_id": args.run_id, "all_pass": all_pass, "n_checks": len(checks),
              "n_failed": sum(1 for v in checks.values() if not v["pass"]),
              "checks": checks, "generated_at": datetime.now().isoformat(timespec="seconds")}
    (run_dir / "verify_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [k for k, v in checks.items() if not v["pass"]]
    print(f"[verify] {len(checks) - len(failed)}/{len(checks)} PASS" + (f"; FAIL: {failed}" if failed else ""))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
