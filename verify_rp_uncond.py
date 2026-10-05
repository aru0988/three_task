"""运行后独立复核（只读；本分支 exp/aliccp-stage2-residual-prompt-unconditional-control）。

用途：无条件固定条件对照消融预注册 §10.5 的运行后独立复核。**不 import 任何机制/接线/分析模块**
（不 import `residual_prompt` / `rp_pinned` / `rp_uncond` / `bench` / `protocol`）；全部判定由原始 run
JSON 记录 + 文件哈希 + 本文件内**独立重实现**的常量向量推导（预注册 §2.1：torch.Generator +
manual_seed(20261006) + 单次 randn(80)）与 UA/PA 门禁判据机械重推。

核验内容（恰一次；报告落 U run 目录 `verify_report.json`，exit 0/1）：
  1. 文件 LF sha256 钉死：mechanism / bench / CLI / rp_pinned / rp_uncond / 守卫测试 / 前置核验脚本；
  2. identity（三 run）/ REP_B（B vs 历史基线钉死 + newtask.pt sha）/ **REP_Cp（C_p vs f08ae6e F_c
     记录逐位 + checkpoint sha）** / A 类 / 对照链（五历史 run）独立重推；
  3. **UA1–UA11（U）与 PA1–PA8（C_p）：由记录的 `observed` 值独立重推 pass/fail，并核对记录布尔与重推
     一致**（记录一致性口径——门禁本身失败属判定事实，不使本脚本失败；失败状态在报告
     `recomputed.mechanism_gate_status` 单列）；
  4. **常量向量独立推导**：重抽 c → sha256 == 冻结值 ∧ **checkpoint 内 `uncond_condition` 逐位 == 重抽值**；
     两处理臂 checkpoint 内 `prompt_gate` == 钉死 α（逐位）；state 键集钉死（U=6 / C_p=5）；
  5. 效用分量与判定树独立重算（GAP_U/g_c/g_u/R_u/Δval_gap_u/verdict/subreason/两臂二级分类/headroom）与
     分析器报告逐位一致；
  6. 三处 JSON 互洽（metrics.rp_arm == gate_report.residual_prompt == prompt_report.arm）、SUMMARY 行、
     run 元数据（dirty=false、run_id 内嵌 commit）。

退出码 0 = 记录全部一致（含门禁记录与独立重推一致）；1 = 有不一致（逐项打印；不一致即停止，不改任何
run 产物）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent

# ---- §1.5/§5 钉死常量（与预注册 / 前置核验脚本逐位一致）----
STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"
BASELINE_RUN_ID = "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07"
CORRECT_RUN_ID = "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg"
SHUF_HIST_RUN_ID = "20261005-0933-p2M-v500k-t1M-m1688723740-short-9d26bc8-rpgs"
PINNED_CORRECT_RUN_ID = "20261005-1029-p2M-v500k-t1M-m1688723740-short-7d26918-rpp"
PINNED_SHUFFLED_RUN_ID = "20261005-1032-p2M-v500k-t1M-m1688723740-short-1c13841-rpps"
REF_HEAD_SHA = "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f"
PINNED_CORRECT_NEWTASK_SHA = "264aedbb9f3f605e5a8e5dd9b2aee8c1945d2dcf0388be722541d2a84f4062b8"
STAGE1_BACKBONE_SHA = "e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c"
STAGE1_ENV_IDS_SHA = "5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0"
STAGE1_FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
BASELINE_RECORD = {
    "best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
    "best_val_auc_bsi": 0.5809347091990792, "test_auc_bsi": 0.5974422649550507,
    "gate_mean": [0.789178, 0.2108221875],
    "per_epoch_val": [0.4645711559431739, 0.48564481224085004, 0.5142344439088465,
                      0.5508809596059387, 0.5809347091990792],
}
CORRECT_RECORD = {
    "best_val_auc_bsi": 0.5895572066556973, "test_auc_bsi": 0.6055453782825184,
    "variant": "residual-prompt", "classification": "VALID_POSITIVE",
    "alpha_final": 0.07101669907569885,
}
SHUF_HIST_RECORD = {
    "best_val_auc_bsi": 0.5819965357883198, "test_auc_bsi": 0.5989210260551207,
    "variant": "residual-prompt-shuffled", "classification": "MECHANISM_FAIL",
}
PINNED_CORRECT_RECORD = {
    "best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
    "best_val_auc_bsi": 0.5798886656441761, "test_auc_bsi": 0.5947650925865714,
    "gate_mean": [0.775494625, 0.224505265625],
    "per_epoch_val": [0.46759930720014714, 0.49016156687227064, 0.5156547237007145,
                      0.5480714746135289, 0.5798886656441761],
    "variant": "residual-prompt-pinned", "classification": "VALID_NEGATIVE",
}
PINNED_SHUFFLED_RECORD = {
    "best_val_auc_bsi": 0.5817369266195509, "test_auc_bsi": 0.5966263705980125,
    "variant": "residual-prompt-pinned-shuffled", "classification": "VALID_NEGATIVE",
}
DELTA_TEST_CORRECT = 0.008103113327467715
DELTA_VAL_CORRECT = 0.008622497456618139
DELTA_TEST_SHUFFLED_HIST = 0.0014787611000699474
PINNED_ALPHA = 0.07101669907569885
UNCOND_CONST_SEED = 20261006
EXPECTED_COND_NUMEL = 80
COND_VECTOR_SHA256 = "0d45cc4611cebde9675858f1afc633a91b3e9114d6ddc02899a5d97f34babf58"
MATERIAL_DELTA = 0.001
SECONDARY_POSITIVE_MIN = 0.001
SECONDARY_NEGATIVE_MAX = -0.02
VARIANT_PINNED = "residual-prompt-pinned"
VARIANT_UNCOND = "residual-prompt-uncond"
RUN_ID_SUFFIX_CORRECT = "-rpp"
RUN_ID_SUFFIX_UNCOND = "-rpu"
HISTORICAL_U1_THRESHOLD = 0.0055
RATIO_REL_DIFF_DISCLOSURE_TOL = 0.05

# ---- UA/PA 门禁判据常量（与钉死 `uncond_mechanism_gates` / `pinned_mechanism_gates` 语义逐字一致）----
REFERENCE_AUC = 0.5809347091990792
REFERENCE_PRED_STD = 0.005217193225189258
REFERENCE_IDENTITY_TOL = 1e-9
BOUND_TOL = 1e-6
RATIO_SPREAD_TOL = 1e-6
UNCOND_STD_TOL = 1e-6                       # §12 C1b-2 校正后
PRED_STD_MIN_RATIO = 0.5
EXPECTED_TRAINABLE_PROMPT_KEYS = ("prompt_generator.0.bias", "prompt_generator.0.weight",
                                  "prompt_generator.2.bias", "prompt_generator.2.weight")
EXPECTED_PINNED_STATE_KEYS = ("prompt_gate",) + EXPECTED_TRAINABLE_PROMPT_KEYS
EXPECTED_UNCOND_STATE_KEYS = tuple(sorted(("prompt_gate", "uncond_condition")
                                          + EXPECTED_TRAINABLE_PROMPT_KEYS))
EXPECTED_NEW_PARAMS_TOTAL = 2384
EXPECTED_HEAD_PARAMS = 8129

# ---- 文件 LF sha256 钉死（C2 时计算；与守卫测试一致；哈希链无环）----
MECHANISM_LF_SHA = "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc"
BENCH_LF_SHA = "686100c75c588f8833f16ffd8c2163efbef746b2a62a8d189cbe970122397977"
CLI_LF_SHA = "a4980e52d1126a9ddd88b01a0a34cbfc57772b9dbd4640320214cadba7bf1d26"
PINNED_MODULE_LF_SHA = "ba23bf473634ff647b482ebf6a92f27ca7f073432bb4ed7a1342b8b11101df9b"
UNCOND_MODULE_LF_SHA = "79ab481ad8420634c6bade5b22de78afb6255864ea75b9bc1039209a74e2cf42"
GUARD_TEST_LF_SHA = "3487246d7469c0d85536b3d6aa36a0e1df72c556589db6378c4ea770de4cd876"  # C7 审计重钉（白名单同步 +2 条目；原 bb218d3c…，见预注册 §12 C7；门禁语义零改动）
PRERUN_VERIFIER_LF_SHA = "479b12a24e360de94fc2fc53175ca7d35f2eae995e6201b239eb6731b893d9f2"

FILES = {
    "aliccp_benchmark/residual_prompt.py": MECHANISM_LF_SHA,
    "aliccp_benchmark/bench.py": BENCH_LF_SHA,
    "run_aliccp_benchmark.py": CLI_LF_SHA,
    "aliccp_benchmark/rp_pinned.py": PINNED_MODULE_LF_SHA,
    "aliccp_benchmark/rp_uncond.py": UNCOND_MODULE_LF_SHA,
    "aliccp_benchmark/tests/test_residual_prompt_uncond.py": GUARD_TEST_LF_SHA,
    "verify_uncond_prerun.py": PRERUN_VERIFIER_LF_SHA,
}

checks: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lf_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str(t.dtype).encode() + str(tuple(t.shape)).encode()
                          + t.numpy().tobytes()).hexdigest()


# ---- §2.1 规格独立重实现（不改任何共享代码）----
def draw_constant(input_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(int(input_size), generator=generator, dtype=torch.float32)


def a_class_ok(gate_doc: dict) -> bool:
    gates = gate_doc.get("gates") or {}
    return all(gates.get(g, {}).get("verdict") in ("PASS", "SKIP")
               for g in ("A1", "A2", "A3", "A4", "A5", "A6"))


def secondary_classification(delta_test: float) -> str:
    if float(delta_test) >= SECONDARY_POSITIVE_MIN:
        return "POSITIVE_IMPROVEMENT"
    if float(delta_test) <= SECONDARY_NEGATIVE_MAX:
        return "CLEAR_DEGRADATION"
    return "NO_CLEAR_IMPROVEMENT"


def rederive_pinned_gate_pass(gate: str, observed: dict) -> bool:
    """由记录的 observed 值独立重推 PA 门禁（判据 == 钉死 `pinned_mechanism_gates`）。"""
    o = observed or {}
    if gate == "M0":
        return (o.get("ref_val_auc") is not None and o.get("ref_pred_std") is not None
                and abs(float(o["ref_val_auc"]) - REFERENCE_AUC) <= REFERENCE_IDENTITY_TOL
                and abs(float(o["ref_pred_std"]) - REFERENCE_PRED_STD) <= REFERENCE_IDENTITY_TOL)
    if gate == "PA1":
        return bool(o.get("is_parameter") is False and o.get("requires_grad") is False
                    and o.get("in_named_parameters") is False
                    and o.get("alpha_at_construction") == PINNED_ALPHA
                    and o.get("alpha_after_training") == PINNED_ALPHA
                    and o.get("alpha_final") == PINNED_ALPHA
                    and o.get("pinned_alpha") == PINNED_ALPHA)
    if gate == "PA2":
        return bool(o.get("optimizer_excludes_prompt_gate") is True
                    and o.get("optimizer_covers_named_parameters") is True
                    and (o.get("probe_alpha_grad_norms") or []) != []
                    and all(g is None for g in (o.get("probe_alpha_grad_norms") or [])))
    if gate == "PA3":
        return bool(o.get("shared_params_bit_identical") and o.get("global_rng_endpoint_identical")
                    and o.get("extra_keys") == sorted(EXPECTED_PINNED_STATE_KEYS))
    if gate == "PA4":
        return bool(o.get("zero_alpha_bit_identical") and o.get("alpha_restored_exact")
                    and float(o.get("zero_alpha_max_abs_diff") or 0.0) == 0.0
                    and o.get("pinned_differs_from_reference"))
    if gate == "PA5":
        return all(float(r) <= abs(PINNED_ALPHA) + BOUND_TOL for r in (o.get("ratio_max") or [1e9]))
    if gate == "PA6":
        means = [float(r) for r in (o.get("ratio_mean") or [])]
        return bool(means) and (max(means) - min(means)) <= RATIO_SPREAD_TOL
    if gate == "PA7":
        return bool(float(o.get("pred_std") or 0.0) > 0.0 and o.get("ref_pred_std") is not None
                    and float(o["ref_pred_std"]) > 0.0
                    and float(o["pred_std"]) >= PRED_STD_MIN_RATIO * float(o["ref_pred_std"]))
    if gate == "PA8":
        return bool(sorted(str(n) for n in (o.get("names") or []))
                    == sorted(EXPECTED_TRAINABLE_PROMPT_KEYS)
                    and int(o.get("new_params_total", -1)) == EXPECTED_NEW_PARAMS_TOTAL
                    and int(o.get("head_params", -1)) == EXPECTED_HEAD_PARAMS)
    raise AssertionError(f"未知门禁 {gate}")


def rederive_uncond_gate_pass(gate: str, observed: dict) -> bool:
    """由记录的 observed 值独立重推 UA 门禁（判据 == 钉死 `uncond_mechanism_gates`）。"""
    o = observed or {}
    if gate == "M0":
        return rederive_pinned_gate_pass("M0", o)
    if gate == "UA1":
        return rederive_pinned_gate_pass("PA1", o)
    if gate == "UA2":
        return rederive_pinned_gate_pass("PA2", o)
    if gate == "UA3":
        return bool(o.get("shared_params_bit_identical") and o.get("global_rng_endpoint_identical")
                    and o.get("extra_keys") == sorted(EXPECTED_UNCOND_STATE_KEYS))
    if gate == "UA4":
        return rederive_pinned_gate_pass("PA4", o)
    if gate == "UA5":
        spread = o.get("ratio_rows_max_minus_min")
        return bool(o.get("direction_bit_identical_across_batch_rows") is True
                    and o.get("direction_bit_identical_under_zero_input") is True
                    and o.get("regen_bit_identical") is True
                    and spread is not None and float(spread) <= UNCOND_STD_TOL)
    if gate == "UA6":
        return rederive_pinned_gate_pass("PA5", o)
    if gate == "UA7":
        return rederive_pinned_gate_pass("PA6", o)
    if gate == "UA8":
        return bool(float(o.get("geff_std") if o.get("geff_std") is not None else 1e9)
                    <= UNCOND_STD_TOL
                    and float(o.get("geff_min") or 0.0) > 0.0
                    and all(float(r) <= UNCOND_STD_TOL for r in (o.get("ratio_std") or [1e9])))
    if gate == "UA9":
        return rederive_pinned_gate_pass("PA7", o)
    if gate == "UA10":
        return bool(sorted(str(n) for n in (o.get("names") or []))
                    == sorted(EXPECTED_TRAINABLE_PROMPT_KEYS)
                    and int(o.get("new_params_total", -1)) == EXPECTED_NEW_PARAMS_TOTAL
                    and int(o.get("head_params", -1)) == EXPECTED_HEAD_PARAMS
                    and o.get("alpha_numel") == 1
                    and o.get("cond_numel") == EXPECTED_COND_NUMEL
                    and o.get("state_extra_keys") == sorted(EXPECTED_UNCOND_STATE_KEYS))
    if gate == "UA11":
        norms = o.get("generator_grad_norms") or []
        return bool(norms) and all(g is not None and math.isfinite(float(g)) and float(g) > 0.0
                                   for g in norms)
    raise AssertionError(f"未知门禁 {gate}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="无条件固定条件对照消融运行后独立复核（只读）")
    parser.add_argument("--root", type=str, default="artifacts/aliccp_bench")
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--arm-cond-run", type=str, required=True)
    parser.add_argument("--arm-uncond-run", type=str, required=True)
    args = parser.parse_args(argv)
    root = Path(args.root)
    base_dir = root / "runs" / args.baseline_run
    cp_dir = root / "runs" / args.arm_cond_run
    up_dir = root / "runs" / args.arm_uncond_run

    base = _read_json(base_dir / "metrics.json")
    base_gate = _read_json(base_dir / "gate_report.json")
    cp = _read_json(cp_dir / "metrics.json")
    cp_gate = _read_json(cp_dir / "gate_report.json")
    cp_prompt = _read_json(cp_dir / "prompt_report.json")
    up = _read_json(up_dir / "metrics.json")
    up_gate = _read_json(up_dir / "gate_report.json")
    up_prompt = _read_json(up_dir / "prompt_report.json")
    compare = _read_json(up_dir / "rp_uncond_compare.json")
    ctx_base = _read_json(root / "runs" / BASELINE_RUN_ID / "metrics.json")
    ctx_corr = _read_json(root / "runs" / CORRECT_RUN_ID / "metrics.json")
    ctx_corr_prompt = _read_json(root / "runs" / CORRECT_RUN_ID / "prompt_report.json")
    ctx_shuf = _read_json(root / "runs" / SHUF_HIST_RUN_ID / "metrics.json")
    ctx_pc = _read_json(root / "runs" / PINNED_CORRECT_RUN_ID / "metrics.json")
    ctx_ps = _read_json(root / "runs" / PINNED_SHUFFLED_RUN_ID / "metrics.json")

    # ---------------- 1. 文件 LF sha256 钉死 ----------------
    for rel, pin in FILES.items():
        check(f"file.{rel}.exists", (ROOT / rel).is_file())
        if (ROOT / rel).is_file():
            actual = _lf_sha(ROOT / rel)
            check(f"file.{rel}.lf_sha256", actual == pin, actual)

    # ---------------- 2. identity / REP_B / REP_Cp / A / 对照链 ----------------
    expected_ref_path = root / "runs" / BASELINE_RUN_ID / "newtask.pt"
    cp_ref_recorded = (cp_prompt.get("reference_dispersion") or {}).get("newtask_checkpoint")
    up_ref_recorded = (up_prompt.get("reference_dispersion") or {}).get("newtask_checkpoint")
    identity = {
        "stage1_same_pinned": base["stage1_id"] == cp["stage1_id"] == up["stage1_id"] == STAGE1_ID,
        "seed_same_pinned": base["model_seed"] == cp["model_seed"] == up["model_seed"] == 1688723740,
        "epochs_patience_tag": all((r["epochs"], r["patience"], r["tag"]) == (5, 2, "short")
                                   for r in (base, cp, up)),
        "dirty_false": all(not r.get("git", {}).get("dirty") for r in (base, cp, up)),
        "baseline_variant_no_suffix": base.get("variant") == "baseline"
        and not base["run_id"].endswith(("-rpp", "-rpps", "-rpg", "-rpu")),
        "arm_cond_variant_suffix": cp.get("variant") == VARIANT_PINNED
        and cp["run_id"].endswith(RUN_ID_SUFFIX_CORRECT),
        "arm_uncond_variant_suffix": up.get("variant") == VARIANT_UNCOND
        and up["run_id"].endswith(RUN_ID_SUFFIX_UNCOND),
        "arms_ref_path_sha": (cp_ref_recorded is not None and up_ref_recorded is not None
                              and Path(str(cp_ref_recorded)) == expected_ref_path
                              and Path(str(up_ref_recorded)) == expected_ref_path
                              and expected_ref_path.is_file()
                              and _sha256_file(expected_ref_path) == REF_HEAD_SHA),
        "stage1_shas": all(r.get("backbone_sha256_loaded") == STAGE1_BACKBONE_SHA
                           and r.get("env_ids_sha256") == STAGE1_ENV_IDS_SHA
                           and r.get("fingerprint_sha256") == STAGE1_FINGERPRINT_SHA
                           for r in (base, cp, up)),
        "commit_embedded": all(str(r.get("commit")) in r["run_id"] for r in (base, cp, up)),
    }
    for name, ok in identity.items():
        check(f"identity.{name}", ok)

    base_pe = [e["val_auc_bsi"] for e in base["per_epoch"]]
    reproduction_b = {
        "per_epoch_val": base_pe == BASELINE_RECORD["per_epoch_val"],
        "best_epoch_val_test_gate": (base["best_epoch"] == BASELINE_RECORD["best_epoch"]
                                     and base["best_val_auc_bsi"] == BASELINE_RECORD["best_val_auc_bsi"]
                                     and base["test_auc_bsi"] == BASELINE_RECORD["test_auc_bsi"]
                                     and base["gate_mean"] == BASELINE_RECORD["gate_mean"]),
        "newtask_sha": (base_dir / "newtask.pt").is_file()
        and _sha256_file(base_dir / "newtask.pt") == REF_HEAD_SHA,
    }
    for name, ok in reproduction_b.items():
        check(f"reproduction_b.{name}", ok)

    cp_pe = [e["val_auc_bsi"] for e in cp["per_epoch"]]
    replay_cp = {
        "per_epoch_val": cp_pe == PINNED_CORRECT_RECORD["per_epoch_val"],
        "best_epoch_val_test_gate": (cp["best_epoch"] == PINNED_CORRECT_RECORD["best_epoch"]
                                     and cp["best_val_auc_bsi"] == PINNED_CORRECT_RECORD["best_val_auc_bsi"]
                                     and cp["test_auc_bsi"] == PINNED_CORRECT_RECORD["test_auc_bsi"]
                                     and cp["gate_mean"] == PINNED_CORRECT_RECORD["gate_mean"]),
        "newtask_sha": (cp_dir / "newtask.pt").is_file()
        and _sha256_file(cp_dir / "newtask.pt") == PINNED_CORRECT_NEWTASK_SHA,
        "alpha_final": cp_prompt.get("alpha_final") == PINNED_ALPHA,
    }
    for name, ok in replay_cp.items():
        check(f"replay_cp.{name}", ok)

    check("protocol.baseline_a_class", a_class_ok(base_gate))
    check("protocol.arm_cond_a_class", a_class_ok(cp_gate))
    check("protocol.arm_uncond_a_class", a_class_ok(up_gate))

    # ---------------- 3. UA1–UA11 / PA1–PA8 独立重推（observed → pass/fail）+ 记录一致性 ----------------
    gate_status = {}
    for label, doc, names, rederive in (
            ("cond", cp, ("M0",) + tuple(f"PA{i}" for i in range(1, 9)), rederive_pinned_gate_pass),
            ("uncond", up, ("M0",) + tuple(f"UA{i}" for i in range(1, 12)), rederive_uncond_gate_pass)):
        arm = doc.get("rp_arm") or {}
        for gate in names:
            rec = arm.get(gate) or {}
            rederived = rederive(gate, rec.get("observed") or {})
            gate_status[f"{label}.{gate}"] = "PASS" if rederived else "FAIL"
            check(f"mechanism.{label}.{gate}.record_consistent", bool(rec.get("pass")) == rederived,
                  f"recorded={rec.get('pass')} rederived={rederived}")

    # ---------------- 4. checkpoint 独立读取 + 常量向量独立推导 ----------------
    ref_state = torch.load(root / "runs" / BASELINE_RUN_ID / "newtask.pt", map_location="cpu")
    cp_state = torch.load(cp_dir / "newtask.pt", map_location="cpu")
    up_state = torch.load(up_dir / "newtask.pt", map_location="cpu")
    check("checkpoint.cond.prompt_gate_pinned",
          "prompt_gate" in cp_state and float(cp_state["prompt_gate"]) == PINNED_ALPHA,
          repr(cp_state.get("prompt_gate")))
    check("checkpoint.cond.key_set_pinned",
          sorted(set(cp_state) - set(ref_state)) == sorted(EXPECTED_PINNED_STATE_KEYS),
          sorted(set(cp_state) - set(ref_state)))
    check("checkpoint.uncond.prompt_gate_pinned",
          "prompt_gate" in up_state and float(up_state["prompt_gate"]) == PINNED_ALPHA,
          repr(up_state.get("prompt_gate")))
    check("checkpoint.uncond.key_set_pinned",
          sorted(set(up_state) - set(ref_state)) == sorted(EXPECTED_UNCOND_STATE_KEYS),
          sorted(set(up_state) - set(ref_state)))
    redraw = draw_constant(EXPECTED_COND_NUMEL, UNCOND_CONST_SEED)
    check("constant_vector.redraw_sha", sha256_tensor(redraw) == COND_VECTOR_SHA256,
          sha256_tensor(redraw))
    check("constant_vector.recorded_sha",
          (up_prompt.get("uncond") or {}).get("cond_sha256") == COND_VECTOR_SHA256,
          repr((up_prompt.get("uncond") or {}).get("cond_sha256")))
    check("constant_vector.checkpoint_bit_identical",
          "uncond_condition" in up_state
          and torch.equal(up_state["uncond_condition"].detach().cpu().float(), redraw),
          repr(up_state.get("uncond_condition") is not None))
    check("constant_vector.regen_flag", (up_prompt.get("uncond") or {}).get("regen_bit_identical") is True)

    comparator = {
        "baseline_values": (ctx_base["test_auc_bsi"] == BASELINE_RECORD["test_auc_bsi"]
                            and ctx_base["best_val_auc_bsi"] == BASELINE_RECORD["best_val_auc_bsi"]
                            and [e["val_auc_bsi"] for e in ctx_base["per_epoch"]]
                            == BASELINE_RECORD["per_epoch_val"]
                            and ctx_base["gate_mean"] == BASELINE_RECORD["gate_mean"]),
        "correct_values": (ctx_corr["test_auc_bsi"] == CORRECT_RECORD["test_auc_bsi"]
                           and ctx_corr["best_val_auc_bsi"] == CORRECT_RECORD["best_val_auc_bsi"]
                           and ctx_corr.get("variant") == CORRECT_RECORD["variant"]
                           and (ctx_corr.get("rp_arm") or {}).get("classification")
                           == CORRECT_RECORD["classification"]),
        "shuf_hist_values": (ctx_shuf["test_auc_bsi"] == SHUF_HIST_RECORD["test_auc_bsi"]
                             and ctx_shuf["best_val_auc_bsi"] == SHUF_HIST_RECORD["best_val_auc_bsi"]
                             and ctx_shuf.get("variant") == SHUF_HIST_RECORD["variant"]
                             and (ctx_shuf.get("rp_arm") or {}).get("classification")
                             == SHUF_HIST_RECORD["classification"]),
        "pinned_correct_values": (ctx_pc["test_auc_bsi"] == PINNED_CORRECT_RECORD["test_auc_bsi"]
                                  and ctx_pc["best_val_auc_bsi"] == PINNED_CORRECT_RECORD["best_val_auc_bsi"]
                                  and [e["val_auc_bsi"] for e in ctx_pc["per_epoch"]]
                                  == PINNED_CORRECT_RECORD["per_epoch_val"]
                                  and ctx_pc["gate_mean"] == PINNED_CORRECT_RECORD["gate_mean"]
                                  and ctx_pc.get("variant") == PINNED_CORRECT_RECORD["variant"]
                                  and (ctx_pc.get("rp_arm") or {}).get("classification")
                                  == PINNED_CORRECT_RECORD["classification"]),
        "pinned_shuffled_values": (ctx_ps["test_auc_bsi"] == PINNED_SHUFFLED_RECORD["test_auc_bsi"]
                                   and ctx_ps["best_val_auc_bsi"] == PINNED_SHUFFLED_RECORD["best_val_auc_bsi"]
                                   and ctx_ps.get("variant") == PINNED_SHUFFLED_RECORD["variant"]
                                   and (ctx_ps.get("rp_arm") or {}).get("classification")
                                   == PINNED_SHUFFLED_RECORD["classification"]),
        "pin_source_alpha": ctx_corr_prompt.get("alpha_final") == PINNED_ALPHA,
        "arms_alpha_equal_pin": (cp_prompt.get("alpha_final") == PINNED_ALPHA
                                 and up_prompt.get("alpha_final") == PINNED_ALPHA),
        "correct_deltas": (ctx_corr["test_auc_bsi"] - ctx_base["test_auc_bsi"] == DELTA_TEST_CORRECT
                           and ctx_corr["best_val_auc_bsi"] - ctx_base["best_val_auc_bsi"]
                           == DELTA_VAL_CORRECT),
        "shuf_hist_delta": (ctx_shuf["test_auc_bsi"] - ctx_base["test_auc_bsi"]
                            == DELTA_TEST_SHUFFLED_HIST),
        "replayed_g_c": (cp["test_auc_bsi"] - base["test_auc_bsi"]
                         == PINNED_CORRECT_RECORD["test_auc_bsi"] - BASELINE_RECORD["test_auc_bsi"]),
    }
    for name, ok in comparator.items():
        check(f"comparator.{name}", ok)

    # ---------------- 5. 效用与判定独立重算 ----------------
    base_test, base_val = base["test_auc_bsi"], base["best_val_auc_bsi"]
    cp_test, cp_val = cp["test_auc_bsi"], cp["best_val_auc_bsi"]
    up_test, up_val = up["test_auc_bsi"], up["best_val_auc_bsi"]
    gap_u = cp_test - up_test
    g_c, g_u = cp_test - base_test, up_test - base_test
    delta_val_gap_u = cp_val - up_val
    delta_val_c, delta_val_u = cp_val - base_val, up_val - base_val
    cp_ratio = [float(r) for r in ((cp_prompt.get("val_stats") or {}).get("ratio_mean") or [])]
    up_ratio = [float(r) for r in ((up_prompt.get("val_stats") or {}).get("ratio_mean") or [])]
    ratio_mean_c = sum(cp_ratio) / len(cp_ratio) if cp_ratio else None
    ratio_mean_u = sum(up_ratio) / len(up_ratio) if up_ratio else None
    ratio_rel_diff = (abs(ratio_mean_c - ratio_mean_u) / ratio_mean_c
                      if ratio_mean_c and ratio_mean_u is not None and ratio_mean_c > 0 else None)

    pin_fail = any(gate_status[k] == "FAIL" for k in
                   ("cond.PA1", "cond.PA2", "cond.PA8"))
    uncond_pin_fail = any(gate_status[k] == "FAIL" for k in
                          ("uncond.UA1", "uncond.UA2", "uncond.UA10"))
    any_gate_fail = any(v == "FAIL" for v in gate_status.values())
    const_ok = (sha256_tensor(redraw) == COND_VECTOR_SHA256
                and (up_prompt.get("uncond") or {}).get("cond_sha256") == COND_VECTOR_SHA256
                and (up_prompt.get("uncond") or {}).get("regen_bit_identical") is True
                and "uncond_condition" in up_state
                and torch.equal(up_state["uncond_condition"].detach().cpu().float(), redraw))

    if not all(identity.values()):
        expect_verdict, expect_sub = "INVALID", "IDENTITY_MISMATCH"
    elif not all(reproduction_b.values()):
        expect_verdict, expect_sub = "INVALID", "BASELINE_REPRODUCTION_FAILED"
    elif not all(replay_cp.values()):
        expect_verdict, expect_sub = "INVALID", "CROSS_EXPERIMENT_REPLAY_FAILED"
    elif not (a_class_ok(base_gate) and a_class_ok(cp_gate) and a_class_ok(up_gate)):
        expect_verdict, expect_sub = "INVALID", "PROTOCOL_INVALID"
    elif pin_fail:
        expect_verdict, expect_sub = "INVALID", "PINNED_ALPHA_INVALID"
    elif uncond_pin_fail:
        expect_verdict, expect_sub = "INVALID", "UNCONDITION_INVALID"
    elif any_gate_fail:
        expect_verdict, expect_sub = "INVALID", "MECHANISM_FAIL"
    elif not all(comparator.values()):
        expect_verdict, expect_sub = "INVALID", "COMPARATOR_MISMATCH"
    elif not const_ok:
        expect_verdict, expect_sub = "INVALID", "CONSTANT_VECTOR_MISMATCH"
    elif gap_u >= MATERIAL_DELTA and delta_val_gap_u > 0.0:
        expect_verdict = "SAMPLE_CONDITIONING_SUPPORTED"
        expect_sub = ("CONDITIONING_ESSENTIAL" if g_u < SECONDARY_POSITIVE_MIN
                      else "PARTIAL_CONDITIONING_CONTRIBUTION")
    else:
        expect_verdict = "SAMPLE_CONDITIONING_NOT_SUPPORTED"
        expect_sub = "GAP_BELOW_MATERIALITY" if gap_u < MATERIAL_DELTA else "VALIDATION_DISAGREEMENT"
    check("verdict.independent", compare.get("verdict") == expect_verdict
          and compare.get("subreason") == expect_sub,
          f"recomputed={expect_verdict}/{expect_sub} "
          f"recorded={compare.get('verdict')}/{compare.get('subreason')}")
    comp = compare.get("components") or {}
    check("components.bit_equal",
          comp.get("gap_u") == gap_u and comp.get("g_c") == g_c and comp.get("g_u") == g_u
          and comp.get("delta_val_gap_u") == delta_val_gap_u
          and comp.get("gap_material_positive") == (gap_u >= MATERIAL_DELTA)
          and comp.get("validation_agreement") == (delta_val_gap_u > 0.0)
          and comp.get("unconditional_matches_or_exceeds_conditioned") == (gap_u <= 0.0)
          and comp.get("retained_ratio_u_vs_cond") == ((g_u / g_c) if g_c > 0 else None)
          and comp.get("delta_val_c") == delta_val_c and comp.get("delta_val_u") == delta_val_u
          and comp.get("G_c_ge_historical_0.0055") == (g_c >= HISTORICAL_U1_THRESHOLD)
          and comp.get("G_u_ge_historical_0.0055") == (g_u >= HISTORICAL_U1_THRESHOLD)
          and comp.get("ratio_mean_cond") == ratio_mean_c
          and comp.get("ratio_mean_uncond") == ratio_mean_u
          and comp.get("ratio_relative_diff") == ratio_rel_diff
          and comp.get("ratio_disclosure_triggered")
          == (ratio_rel_diff is not None and ratio_rel_diff > RATIO_REL_DIFF_DISCLOSURE_TOL),
          json.dumps({"gap_u": gap_u, "g_c": g_c, "g_u": g_u}))
    secondary = compare.get("secondary") or {}
    check("secondary.independent",
          secondary.get("arm_cond") == secondary_classification(g_c)
          and secondary.get("arm_uncond") == secondary_classification(g_u),
          f"recomputed={secondary_classification(g_c)}/{secondary_classification(g_u)} "
          f"recorded={secondary.get('arm_cond')}/{secondary.get('arm_uncond')}")
    headroom = compare.get("headroom") or {}
    for label, delta_test, delta_val, arm_doc, prompt_doc in (
            ("arm_cond", g_c, delta_val_c, cp, cp_prompt),
            ("arm_uncond", g_u, delta_val_u, up, up_prompt)):
        rec = headroom.get(label) or {}
        if secondary_classification(delta_test) != "NO_CLEAR_IMPROVEMENT":
            check(f"headroom.{label}.not_applicable", rec.get("applicable") is False)
            continue
        ratio_mean = [float(r) for r in ((prompt_doc.get("val_stats") or {}).get("ratio_mean") or [])]
        active = (abs(float(prompt_doc.get("alpha_final") or 0.0)) > 0.0
                  and bool(ratio_mean) and all(r > 0.0 for r in ratio_mean)
                  and float((prompt_doc.get("gate") or {}).get("geff_max") or 0.0) > 0.0)
        right_censored = (arm_doc.get("best_epoch") == arm_doc.get("epochs")
                          == len(arm_doc.get("per_epoch") or []))
        check(f"headroom.{label}.independent",
              rec.get("applicable") is True
              and rec.get("mechanism_activity") == ("PIN_ACTIVE" if active else "INACTIVE")
              and rec.get("validation_direction") == ("POSITIVE" if delta_val > 0 else "NON_POSITIVE")
              and rec.get("epoch_trajectory")
              == ("RIGHT_CENSORED_STILL_IMPROVING" if right_censored else "EARLY_STOPPED")
              and rec.get("gap_to_positive_threshold") == SECONDARY_POSITIVE_MIN - delta_test)

    # ---------------- 6. JSON 互洽 / SUMMARY / 元数据 ----------------
    check("consistency.rp_arm_three_way",
          (cp.get("rp_arm") or {}) == (cp_gate.get("residual_prompt") or {})
          and (up.get("rp_arm") or {}) == (up_gate.get("residual_prompt") or {}))
    check("consistency.prompt_arm_equals_rp_arm",
          (cp_prompt.get("arm") or {}) == (cp.get("rp_arm") or {})
          and (up_prompt.get("arm") or {}) == (up.get("rp_arm") or {}))
    check("consistency.prompt_hidden",
          cp.get("prompt_hidden") == up.get("prompt_hidden") == 16)
    summary = (root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
    check("summary.baseline_row", any(args.baseline_run in line for line in summary))
    check("summary.arm_cond_row", any(args.arm_cond_run in line for line in summary))
    check("summary.arm_uncond_row", any(args.arm_uncond_run in line for line in summary))
    check("meta.commit_short7", all(len(str(r.get("commit"))) == 7 for r in (base, cp, up)))

    ok_all = all(c["pass"] for c in checks)
    report_doc = {
        "script": Path(__file__).name,
        "baseline_run": args.baseline_run,
        "arm_cond_run": args.arm_cond_run,
        "arm_uncond_run": args.arm_uncond_run,
        "recomputed": {"gap_u": gap_u, "g_c": g_c, "g_u": g_u, "delta_val_gap_u": delta_val_gap_u,
                       "verdict": expect_verdict, "subreason": expect_sub,
                       "secondary": {"arm_cond": secondary_classification(g_c),
                                     "arm_uncond": secondary_classification(g_u)},
                       "mechanism_gate_status": gate_status,
                       "cond_vector_redraw_sha256": sha256_tensor(redraw)},
        "checks": checks,
        "all_pass": ok_all,
        "n_checks": len(checks),
    }
    (up_dir / "verify_report.json").write_text(json.dumps(report_doc, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"\nALL_PASS = {ok_all}  ({len(checks)} checks; report -> {up_dir / 'verify_report.json'})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
