"""seed3 判定：跑后只读验证与机械分类（预注册 §6；只读，不训练、不改产物）。

两个子命令：

1) `freeze-p4`（R1 认证捕获后、R2 运行前；登记 P4）
   在 R1 `--reproduce` 捕获的聚类时刻损失向量上，用**两个独立实现**
   （`normalized_clustering.per_task_rank01` 与 `audit_cluster_losses.ref_rank01`，
   由单测钉逐位等价）计算 argmin 分配，必须逐位一致；导出 R2 的预测事件三元组、
   `env_ids_sha256`、环境占比、`env_0∩purchase1`，写 JSON（提交前后皆可复核）。

2) `verify`（R1–R4 完成后；机械判定）
   读取四个产物，执行预注册 M/U 门禁、§6.2 三分法分类、§6.3 material contradiction、
   §6.6 headroom 旗标与 §6.4 扩展门，输出 JSON + 摘要。全部阈值来自预注册文档字面值，
   本脚本只做机械执行。
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch

from . import protocol
from . import audit_prior_seeds as aps

SEED3 = 1688738016
RAW_STAGE1_ID = "s1-5c060b9c-m1688738016-e3-47619ce0"
NORM_STAGE1_ID = "s1-5c060b9c-m1688738016-e3-5f899cad"
SEED3_RAW_CFG_HASH = "47619ce078ac75497b4da9a81f0db9de59d738f22759702788f0b4ae47640108"
SEED3_NORM_CFG_HASH = "5f899cad0d89972af1ecd2280d9933384ddcad9db84d06069a33a0fd4843f9d3"
FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
# §6.2 分类阈值（用户预注册）
POSITIVE_MIN = 0.001
CLEAR_DEGRADATION_MAX = -0.02
# §4.2 容差
U1_TOLERANCE = -0.005
# §6.3 material contradiction 阈值
DELTA_VAL_CONTRADICTION = -0.01
# M8b 恒等核验排除字段（非科学字段）
VOLATILE_KEYS = {"commit", "git", "versions", "device", "wall_seconds", "peak_vram_mb"}


def _jload(path: Path):
    return json.load(io.open(path, encoding="utf-8"))


# ---------------------------------------------------------------- freeze-p4
def freeze_p4(capture: Path, train_file: Path, env_seed: int, out_path: Path, log=print) -> dict:
    from .audit_cluster_losses import ref_rank01
    from .normalized_clustering import per_task_rank01

    losses = torch.load(capture, map_location="cpu")
    q_impl = per_task_rank01(losses)
    q_ref = ref_rank01(losses)
    agree = bool(torch.equal(q_impl, q_ref))
    if not agree:
        raise AssertionError("两独立实现的 rank01 输出不逐位相等——P4 不得登记（按预注册 §4.3 处置）")
    assign = torch.argmin(q_impl, dim=1)
    counts = torch.bincount(assign, minlength=2)
    initial = protocol.make_env_ids(losses.shape[0], env_seed)
    ys = aps._scan_labels(train_file, losses.shape[0])
    cross = aps._crosstab(assign, ys)
    p4 = {
        "derived_from": str(capture),
        "derived_by": ["normalized_clustering.per_task_rank01", "audit_cluster_losses.ref_rank01"],
        "bitwise_agree": agree,
        "n": int(losses.shape[0]),
        "event": {"epoch": 2, "diff_num": int((initial != assign).sum()),
                  "env_0": int(counts[0]), "env_1": int(counts[1])},
        "env_ids_sha256": protocol.sha256_tensor(assign),
        "env_0_share": float(counts[0]) / int(losses.shape[0]),
        "env_0_purchase1": cross["env_0_purchase1"],
        "crosstab": cross,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit": protocol.code_commit(),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(p4, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[p4] 预测登记：event={p4['event']} share={p4['env_0_share']:.5%} "
        f"env_ids_sha={p4['env_ids_sha256'][:16]}… env_0∩purchase1={p4['env_0_purchase1']} → {out_path}")
    return p4


# ---------------------------------------------------------------- verify
def _find_run(root: Path, stage1_id: str, tag: str, model_seed: int) -> Path:
    hits = []
    for run_dir in sorted((root / "runs").glob("*")):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        try:
            m = _jload(metrics_path)
        except Exception:  # noqa: BLE001
            continue
        if m.get("stage1_id") == stage1_id and m.get("tag") == tag and int(m.get("model_seed", -1)) == model_seed:
            hits.append(run_dir)
    if len(hits) != 1:
        raise FileNotFoundError(f"run 定位不唯一（{tag}/{stage1_id}）：{hits}")
    return hits[0]


def _epoch12(meta: dict):
    return meta["per_epoch"][:2]


def _run_tests(log) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "aliccp_benchmark/tests",
         "-t", "aliccp_benchmark/tests"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    tail = (proc.stderr or "").strip().splitlines()
    ok = proc.returncode == 0 and any(line.strip() == "OK" for line in tail)
    ran = next((line for line in tail if line.startswith("Ran ")), "")
    log(f"[verify] 测试套件：{'OK' if ok else 'FAILED'}（{ran}）")
    return {"ok": bool(ok), "ran": ran, "returncode": proc.returncode}


def _smoke_identity(root_hint: Path, reference_meta: Path, device, log) -> dict:
    """M8b：temp root 上跑 smoke 规模默认路径；与既有记录产物 meta 科学字段逐位比对。

    注意：必须在与记录产物相同的设备上运行（记录产物为 cuda:0；CPU 与 GPU 的浮点/随机
    路径不同，跨设备比较不是"逐位"口径——2026-10-05 首次 CPU 试跑已证差异 ~1e-4 量级）。
    """
    from . import bench
    ref = _jload(reference_meta)
    if str(device) != str(ref.get("device")):
        raise AssertionError(f"M8b 设备口径不符：smoke device={device} != 记录产物 device={ref.get('device')}")
    with tempfile.TemporaryDirectory() as td:
        meta = bench.run_stage1(
            root=Path(td) / "artifacts", data_files=protocol.DATA_FILES,
            budgets={"train": protocol.SMOKE_TRAIN_BUDGET, "val": protocol.SMOKE_VAL_BUDGET,
                     "test": protocol.SMOKE_TEST_BUDGET},
            prefix_tag="p20000-v5000-t10000", model_seed=protocol.MODEL_SEED, env_seed=protocol.ENV_SEED,
            epochs=1, patience=1, device=device, log=lambda *a, **k: None,
        )
    diffs = {}
    for key, ref_val in ref.items():
        if key in VOLATILE_KEYS:
            continue
        new_val = meta.get(key, "<missing>")
        if new_val != ref_val:
            diffs[key] = {"reference": ref_val if not isinstance(ref_val, (dict, list)) else "<omitted>",
                          "smoke": new_val if not isinstance(new_val, (dict, list)) else "<omitted>"}
    log(f"[verify] M8b smoke 恒等：stage1_id 一致={meta['stage1_id'] == ref['stage1_id']}；"
        f"科学字段差异={sorted(diffs)}")
    return {"stage1_id_match": meta["stage1_id"] == ref["stage1_id"], "field_diffs": diffs,
            "pass": bool(meta["stage1_id"] == ref["stage1_id"] and not diffs)}


def verify(root: Path, out_path: Path, *, raw_sid=RAW_STAGE1_ID, norm_sid=NORM_STAGE1_ID,
           p4_path: Path | None, repro_audit: Path | None, train_file: Path,
           smoke_reference_meta: Path | None, run_tests: bool, device, log=print) -> dict:
    t0 = time.time()
    raw_dir = root / "stage1" / raw_sid
    norm_dir = root / "stage1" / norm_sid
    raw_meta = _jload(raw_dir / "meta.json")
    norm_meta = _jload(norm_dir / "meta.json")
    raw_run_dir = _find_run(root, raw_sid, "short", SEED3)
    norm_run_dir = _find_run(root, norm_sid, "norm", SEED3)
    raw_run = _jload(raw_run_dir / "metrics.json")
    norm_run = _jload(norm_run_dir / "metrics.json")
    raw_gates = _jload(raw_run_dir / "gate_report.json")
    norm_gates = _jload(norm_run_dir / "gate_report.json")

    raw_env = torch.load(raw_dir / "env_ids.pt", map_location="cpu")
    norm_env = torch.load(norm_dir / "env_ids.pt", map_location="cpu")
    n = len(raw_env)
    ys = aps._scan_labels(train_file, n)
    raw_cross = aps._crosstab(raw_env, ys)
    norm_cross = aps._crosstab(norm_env, ys)

    checks: dict = {}

    # ---- M1: B4 修复 ----
    events = norm_meta["cluster_events"]
    m1 = (len(events) == 1 and events[0]["env_0"] >= protocol.ENV_SHARE_MIN * n
          and events[0]["env_1"] >= protocol.ENV_SHARE_MIN * n)
    m1 = bool(m1 and norm_gates["gates"]["B4"]["verdict"] == "PASS")
    checks["M1_B4_repair"] = m1

    # ---- M2a: 身份 ----
    checks["M2a_raw_identity"] = (raw_meta["stage1_id"] == raw_sid
                                  and raw_meta["config_hash"] == SEED3_RAW_CFG_HASH)
    checks["M2a_norm_identity"] = (norm_meta["stage1_id"] == norm_sid
                                   and norm_meta["config_hash"] == SEED3_NORM_CFG_HASH)

    # ---- M2b: epoch 1-2 逐位相等 ----
    checks["M2b_epoch12_bit_equal"] = _epoch12(raw_meta) == _epoch12(norm_meta)

    # ---- M2c: P4 ----
    p4 = _jload(p4_path) if p4_path and Path(p4_path).is_file() else None
    if p4 is None:
        checks["M2c_event_matches_P4"] = None  # 未判定（§4.3）
        checks["M2c_env_ids_sha_matches_P4"] = None
    else:
        checks["M2c_event_matches_P4"] = events == [p4["event"]]
        checks["M2c_env_ids_sha_matches_P4"] = norm_meta["env_ids_sha256"] == p4["env_ids_sha256"]

    # ---- M3: 有限性 ----
    diags = norm_meta.get("cluster_diagnostics", [])
    finite = bool(diags) and all(
        torch.isfinite(torch.tensor(list(v.values()))).all()
        for diag in diags for v in diag.get("normalized_loss_quantiles", {}).values()
    ) and all(
        torch.isfinite(torch.tensor(list(v.values()))).all()
        for diag in diags for v in diag.get("raw_loss_quantiles", {}).values()
    )
    checks["M3_finite_diagnostics"] = bool(finite)

    # ---- M4: 非标签恢复 ----
    m4 = (norm_cross["env_0"] > 0
          and norm_cross["env_0_purchase1"] <= 0.05 * norm_cross["env_0"]
          and not norm_cross["env0_equals_purchase1_set"]
          and not norm_cross["env0_subset_of_purchase1"])
    checks["M4_non_label_recovery"] = bool(m4)

    # ---- M4b: 退化签名消除（探针 recall 双正） ----
    model = aps._build_model(norm_meta, device)
    model.load_state_dict(torch.load(norm_dir / "backbone.pt", map_location="cpu"))
    from multitaskrec.dataset import AliCCPDataset
    ds = AliCCPDataset(str(train_file), n)
    probe = aps._probe(model, ds, norm_env, norm_meta["config"]["batch_size"], device)
    raw_model = aps._build_model(raw_meta, device)
    raw_model.load_state_dict(torch.load(raw_dir / "backbone.pt", map_location="cpu"))
    del ds
    ds_raw = AliCCPDataset(str(train_file), n)
    raw_probe = aps._probe(raw_model, ds_raw, raw_env, raw_meta["config"]["batch_size"], device)
    del ds_raw
    checks["M4b_recalls_positive"] = bool(probe["recall_env_0"] > 0 and probe["recall_env_1"] > 0)

    # ---- M5: 有效变化 ----
    checks["M5_effective_change"] = bool(
        events and events[0]["diff_num"] >= 0.05 * n
        and int((raw_env != norm_env).sum()) >= 0.05 * n
    )

    # ---- M6: 环境逐位可复现（reproduce） ----
    repro = _jload(repro_audit) if repro_audit and Path(repro_audit).is_file() else None
    if repro is None:
        checks["M6_reproduce_10of10"] = None
    else:
        rep = repro.get("reproduction", {})
        checks["M6_reproduce_10of10"] = bool(rep.get("all_pass") and rep.get("checks")
                                             and all(rep["checks"].values()))

    # ---- M7: A 类完整性 ----
    def a_ok(gate_doc):
        g = gate_doc["gates"]
        return all(g[k]["verdict"] == "PASS" for k in ("A1", "A2", "A4", "A5", "A6"))
    checks["M7_A_class_both_runs"] = bool(a_ok(raw_gates) and a_ok(norm_gates))

    # ---- M8a: 测试套件 ----
    checks["M8a_tests"] = _run_tests(log)["ok"] if run_tests else None
    # ---- M8b: smoke 恒等 ----
    if smoke_reference_meta and Path(smoke_reference_meta).is_file():
        checks["M8b_smoke_identity"] = _smoke_identity(root, Path(smoke_reference_meta), device, log)["pass"]
    else:
        checks["M8b_smoke_identity"] = None

    # ---- U 组 ----
    delta_test_ctr = norm_meta["test_auc_ctr"] - raw_meta["test_auc_ctr"]
    delta_test = norm_run["test_auc_bsi"] - raw_run["test_auc_bsi"]
    delta_val = norm_run["best_val_auc_bsi"] - raw_run["best_val_auc_bsi"]
    checks["U1_ctr_no_regression"] = bool(delta_test_ctr >= U1_TOLERANCE)

    classification = ("POSITIVE_IMPROVEMENT" if delta_test >= POSITIVE_MIN
                      else "CLEAR_DEGRADATION" if delta_test <= CLEAR_DEGRADATION_MAX
                      else "NO_CLEAR_IMPROVEMENT")

    m_gates = {k: v for k, v in checks.items() if k.startswith("M")}
    mechanism_repaired = all(v is True for v in m_gates.values())  # None（未判定）不算通过
    contradiction = {
        "(a)_delta_val_le_-0.01": bool(delta_val <= DELTA_VAL_CONTRADICTION),
        "(b)_mechanism_gate_fail": not mechanism_repaired,
        "(c)_U1_fail": not checks["U1_ctr_no_regression"],
        "(d)_A_class_fail": not checks["M7_A_class_both_runs"],
    }
    has_contradiction = any(contradiction.values())
    expand_condition = bool(delta_test >= POSITIVE_MIN and not has_contradiction)

    # ---- §6.6 headroom（仅 NO_CLEAR 需要判定；此处全部机械记录） ----
    prior = aps.ARMS
    prior_delta_test = {
        "seed1": prior[("seed1", "norm")]["run"]["test_auc_bsi"] - prior[("seed1", "raw")]["run"]["test_auc_bsi"],
        "seed2": prior[("seed2", "norm")]["run"]["test_auc_bsi"] - prior[("seed2", "raw")]["run"]["test_auc_bsi"],
    }
    deltas3 = [prior_delta_test["seed1"], prior_delta_test["seed2"], delta_test]
    h3 = bool(delta_test > 0 and sum(1 for d in deltas3 if d > 0) >= 2 and sum(deltas3) / 3.0 > 0)
    h2 = ("favors" if delta_val >= 0 else "contradicts" if delta_val <= DELTA_VAL_CONTRADICTION else "ambiguous")
    h4 = bool(raw_run["best_epoch"] == raw_run["epochs"] and norm_run["best_epoch"] == norm_run["epochs"])
    headroom = {
        "H1_mechanism": "favors" if mechanism_repaired else "contradicts",
        "H2_validation_direction": h2,
        "H3_cross_seed": "favors" if h3 else "not",
        "H4_budget_trajectory": "favors_longer_budget" if h4 else "not",
    }
    headroom_verdict = ("HEADROOM_PLAUSIBLE"
                        if (mechanism_repaired and h2 != "contradicts" and h3)
                        else "HEADROOM_NOT_EVIDENT")

    if expand_condition:
        verdict = "SEED3_CONDITION_PASSED__next: new branch from this tip, canonical seed4/5 (1688749593, 1688762746), same 1+1+1+1 paired protocol per prereg §6.4; do not run here"
    elif mechanism_repaired:
        verdict = "mechanism repair effective but utility unstable"
    else:
        verdict = "MECHANISM_NOT_REPAIRED_ON_SEED3"

    report = {
        "verify": {"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "commit": protocol.code_commit(), "root": str(root), "device": str(device)},
        "identities": {"raw_stage1_id": raw_sid, "norm_stage1_id": norm_sid,
                       "fingerprint_sha": FINGERPRINT_SHA,
                       "raw_run_id": raw_run["run_id"], "norm_run_id": norm_run["run_id"]},
        "provenance": {
            "raw_stage1": {"commit": raw_meta.get("commit"), "dirty": raw_meta.get("git", {}).get("dirty")},
            "norm_stage1": {"commit": norm_meta.get("commit"), "dirty": norm_meta.get("git", {}).get("dirty")},
            "raw_run": {"commit": raw_run.get("commit"), "dirty": raw_run.get("git", {}).get("dirty")},
            "norm_run": {"commit": norm_run.get("commit"), "dirty": norm_run.get("git", {}).get("dirty")},
        },
        "mechanism": {
            "raw_event": raw_meta["cluster_events"], "norm_event": norm_meta["cluster_events"],
            "raw_shares": {"env_0": raw_cross["env_0"] / n, "env_1": raw_cross["env_1"] / n},
            "norm_shares": {"env_0": norm_cross["env_0"] / n, "env_1": norm_cross["env_1"] / n},
            "raw_crosstab": raw_cross, "norm_crosstab": norm_cross,
            "norm_probe": probe, "raw_probe": raw_probe,
            "norm_diagnostics": diags,
            "env_ids_sha256": {"raw": raw_meta["env_ids_sha256"], "norm": norm_meta["env_ids_sha256"],
                               "norm_equals_p4": (p4 or {}).get("env_ids_sha256")},
        },
        "stage2": {
            "raw": {"run_id": raw_run["run_id"], "best_val_auc_bsi": raw_run["best_val_auc_bsi"],
                    "test_auc_bsi": raw_run["test_auc_bsi"], "gate_mean": raw_run["gate_mean"],
                    "per_epoch": raw_run["per_epoch"], "best_epoch": raw_run["best_epoch"],
                    "hard_pass": raw_run["hard_pass"],
                    "gates": {k: v["verdict"] for k, v in raw_gates["gates"].items()}},
            "norm": {"run_id": norm_run["run_id"], "best_val_auc_bsi": norm_run["best_val_auc_bsi"],
                     "test_auc_bsi": norm_run["test_auc_bsi"], "gate_mean": norm_run["gate_mean"],
                     "per_epoch": norm_run["per_epoch"], "best_epoch": norm_run["best_epoch"],
                     "hard_pass": norm_run["hard_pass"],
                     "gates": {k: v["verdict"] for k, v in norm_gates["gates"].items()}},
            "delta_test_bsi": delta_test, "delta_val_bsi": delta_val, "delta_test_ctr": delta_test_ctr,
        },
        "checks": checks,
        "mechanism_repaired": bool(mechanism_repaired),
        "classification": classification,
        "material_contradiction": {"flags": contradiction, "present": bool(has_contradiction)},
        "expand_condition": expand_condition,
        "headroom": {**headroom, "verdict": headroom_verdict},
        "cross_seed_delta_test": {**prior_delta_test, "seed3": delta_test},
        "verdict": verdict,
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("=" * 96)
    log(f"[verify] Δtest={delta_test:+.6f} Δval={delta_val:+.6f} Δtest_ctr={delta_test_ctr:+.6f}")
    log(f"[verify] 机制 M 组：" + ", ".join(f"{k}={'PASS' if v is True else v}" for k, v in m_gates.items()))
    log(f"[verify] 分类={classification}；material contradiction={has_contradiction}；"
        f"扩展条件={'PASSED' if expand_condition else 'not met'}")
    log(f"[verify] 结论：{verdict}")
    log(f"[verify] 输出：{out_path}（wall={report['wall_seconds']}s）")
    log("=" * 96)
    return report


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser(description="seed3 跑后验证与机械分类（只读）")
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("freeze-p4", help="在 R1 捕获上登记 P4 预测")
    f.add_argument("--capture", type=str, required=True)
    f.add_argument("--train-file", type=str, default=protocol.DATA_FILES["train"])
    f.add_argument("--env-seed", type=int, default=protocol.ENV_SEED)
    f.add_argument("--out", type=str, required=True)

    v = sub.add_parser("verify", help="执行 M/U 门禁与分类")
    v.add_argument("--root", type=str, default=str(protocol.ARTIFACT_ROOT))
    v.add_argument("--raw-stage1-id", type=str, default=RAW_STAGE1_ID)
    v.add_argument("--norm-stage1-id", type=str, default=NORM_STAGE1_ID)
    v.add_argument("--p4", type=str, default=None)
    v.add_argument("--repro-audit", type=str, default=None)
    v.add_argument("--train-file", type=str, default=protocol.DATA_FILES["train"])
    v.add_argument("--smoke-reference-meta", type=str, default=None)
    v.add_argument("--skip-tests", action="store_true")
    v.add_argument("--out", type=str, required=True)
    v.add_argument("--cpu", action="store_true")
    v.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args(argv)
    if args.command == "freeze-p4":
        freeze_p4(Path(args.capture), Path(args.train_file), args.env_seed, Path(args.out))
        return 0
    device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.gpu}")
    if device.type == "cuda":
        torch.cuda.init()
    report = verify(
        Path(args.root), Path(args.out),
        raw_sid=args.raw_stage1_id, norm_sid=args.norm_stage1_id,
        p4_path=Path(args.p4) if args.p4 else None,
        repro_audit=Path(args.repro_audit) if args.repro_audit else None,
        train_file=Path(args.train_file),
        smoke_reference_meta=Path(args.smoke_reference_meta) if args.smoke_reference_meta else None,
        run_tests=not args.skip_tests, device=device,
    )
    return 0 if report["mechanism_repaired"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
