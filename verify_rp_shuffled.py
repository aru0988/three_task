"""运行后独立复核（只读；本分支 exp/aliccp-stage2-residual-prompt-shuffled-condition）。

用途：shuffled-condition 因果消融预注册 §10.5 的运行后独立复核。**不 import 任何机制/接线/分析
模块**（不 import `residual_prompt` / `rp_shuffled` / `bench` / `protocol`）；全部判定由原始 run JSON
记录 + 文件哈希 + 本文件内**独立重实现**的错排规格（预注册 §2.1：sha256 键推导 + Sattolo）机械重推。

核验内容（恰一次；报告落 P1 run 目录 `verify_report.json`，exit 0/1）：
  1. 文件 LF sha256 钉死：mechanism / bench / CLI / rp_shuffled / 守卫测试 / 前置核验脚本；
  2. 错排独立重推：按 (COND_PERM_SEED, split, batch_index, n) 键空间重生成全部 π → 逐 split digest 与
     总 digest 逐位 == 记录（⇒ 标签无关/确定性复现）；sampled 键的 perm_head 逐位核对；
  3. 置换门禁 PG1–PG6 独立重算（不信任记录布尔）：PG1–PG5 由 report 计数 + 覆盖核对；PG6 = 机制文件哈希；
  4. identity（14 项）/ REP（P0 vs 历史基线钉死 + newtask.pt sha）/ A 类 / 对照链 独立重推；
     **机制门禁 M0/G1–G8：由记录的 `observed` 值独立重推 pass/fail，并核对记录布尔与重推一致**
     （记录一致性口径——门禁本身失败属判定事实，不使本脚本失败；失败状态在报告
     `recomputed.mechanism_gate_status` 单列）；
  5. 效用分量与判定树独立重算（g_c/g_s/L/R/verdict/subreason/二级分类/headroom）与分析器报告逐位一致；
  6. 三处 JSON 互洽（metrics.rp_arm == gate_report.residual_prompt == prompt_report.arm）、SUMMARY 行、
     run 元数据（dirty=false、run_id 内嵌 commit）。

退出码 0 = 记录全部一致（含门禁记录与独立重推一致）；1 = 有不一致（逐项打印；不一致即停止，不改任何
run 产物）。首版检查器把"记录的门禁通过"当作完整性检查（G5 合法失败 ⇒ 误报 exit 1）；跑后修正为
记录一致性口径（同类先例：seed-2 复现预注册 §1.5 检查器缺陷披露）；首跑报告保留为
`verify_report_firstpass_exit1.json`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- §1.5/§3 钉死常量（与预注册 / 前置核验脚本逐位一致）----
STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"
BASELINE_RUN_ID = "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07"
CORRECT_RUN_ID = "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg"
REF_HEAD_SHA = "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f"
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
DELTA_TEST_CORRECT = 0.008103113327467715
DELTA_VAL_CORRECT = 0.008622497456618139
COND_PERM_SEED = 20261005
MATERIAL_DELTA = 0.001
SECONDARY_POSITIVE_MIN = 0.001
SECONDARY_NEGATIVE_MAX = -0.02
VARIANT_SHUFFLED = "residual-prompt-shuffled"
RUN_ID_SUFFIX = "-rpgs"

# ---- 机制门禁判据常量（与钉死 arm_verdict 语义逐字一致；供独立重推）----
REFERENCE_AUC = 0.5809347091990792          # M0 expected_reference_auc（== 历史基线 val）
REFERENCE_PRED_STD = 0.005217193225189258
REFERENCE_IDENTITY_TOL = 1e-9
RATIO_BAND = (0.005, 0.5)
BOUND_TOL = 1e-6
PRED_STD_MIN_RATIO = 0.5
EXPECTED_NEW_PARAMS_TOTAL = 2385
EXPECTED_HEAD_PARAMS = 8129
EXTRA_PARAM_NAMES = ("prompt_gate", "prompt_generator.0.bias", "prompt_generator.0.weight",
                     "prompt_generator.2.bias", "prompt_generator.2.weight")

# ---- 文件 LF sha256 钉死（C2 时计算；与守卫测试一致）----
MECHANISM_LF_SHA = "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc"
BENCH_LF_SHA = "84d631f7dfa12f50be0cfbfb94ffd23236c553c98fa949ce4a563556ee5f69dd"
CLI_LF_SHA = "abedd0b4471098d86b21fdfea040a0de04ceccae298ef5996198dc1e1250011d"
SHUFFLED_MODULE_LF_SHA = "81fb9ba9bec9864cfdcd71256fb44a2b10939466e3a01f39726e21b813a2e6bf"
GUARD_TEST_LF_SHA = "12fb2cf4ffa2c968065470795dc1730b9ff2b5b53fa2ff3cbd5ddf9d15a2fcc1"
PRERUN_VERIFIER_LF_SHA = "871e59a5915c1889773f14740925c49435687033f13efb783e4958f17f189b80"

FILES = {
    "aliccp_benchmark/residual_prompt.py": MECHANISM_LF_SHA,
    "aliccp_benchmark/bench.py": BENCH_LF_SHA,
    "run_aliccp_benchmark.py": CLI_LF_SHA,
    "aliccp_benchmark/rp_shuffled.py": SHUFFLED_MODULE_LF_SHA,
    "aliccp_benchmark/tests/test_residual_prompt_shuffled.py": GUARD_TEST_LF_SHA,
    "verify_shuffled_prerun.py": PRERUN_VERIFIER_LF_SHA,
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


# ---- §2.1 规格独立重实现（不改任何共享代码）----
def perm_seed_material(split: str, batch_index: int, n: int) -> int:
    key = f"{COND_PERM_SEED}|{split}|{int(batch_index)}|{int(n)}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16)


def sattolo(n: int, split: str, batch_index: int) -> list:
    if n <= 1:
        return list(range(n))
    rng = random.Random(perm_seed_material(split, batch_index, n))
    a = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randrange(0, i)
        a[i], a[j] = a[j], a[i]
    return a


def _perm_bytes(values: list) -> bytes:
    return struct.pack(f"<{len(values)}q", *values)


def rederive_split_digest(split: str, n_batches: int, n: int) -> str:
    digest = hashlib.sha256()
    for batch_index in range(n_batches):
        digest.update(f"{batch_index}|{n}|".encode("utf-8"))
        digest.update(_perm_bytes(sattolo(n, split, batch_index)))
    return digest.hexdigest()


def rederive_total_digest(per_split: dict) -> str:
    total = hashlib.sha256()
    for split in sorted(per_split):
        total.update(f"{split}:".encode("utf-8"))
        total.update(rederive_split_digest(split, per_split[split]["n_batches"],
                                           per_split[split]["n_rows"] // per_split[split]["n_batches"]
                                           ).encode("utf-8"))
    return total.hexdigest()


def a_class_ok(gate_doc: dict) -> bool:
    gates = gate_doc.get("gates") or {}
    return all(gates.get(g, {}).get("verdict") in ("PASS", "SKIP")
               for g in ("A1", "A2", "A3", "A4", "A5", "A6"))


def secondary_classification(g_s: float) -> str:
    if g_s >= SECONDARY_POSITIVE_MIN:
        return "POSITIVE_IMPROVEMENT"
    if g_s <= SECONDARY_NEGATIVE_MAX:
        return "CLEAR_DEGRADATION"
    return "NO_CLEAR_IMPROVEMENT"


def rederive_gate_pass(gate: str, observed: dict) -> bool:
    """由记录的 observed 值独立重推门禁通过与否（不信任记录的 pass 布尔；判据 == 钉死 arm_verdict）。"""
    o = observed or {}
    if gate == "M0":
        return (o.get("ref_val_auc") is not None and o.get("ref_pred_std") is not None
                and abs(float(o["ref_val_auc"]) - REFERENCE_AUC) <= REFERENCE_IDENTITY_TOL
                and abs(float(o["ref_pred_std"]) - REFERENCE_PRED_STD) <= REFERENCE_IDENTITY_TOL)
    if gate == "G1":
        return bool(o.get("shared_params_bit_identical") and o.get("global_rng_endpoint_identical")
                    and o.get("extra_keys") == sorted(EXTRA_PARAM_NAMES))
    if gate == "G2":
        return bool(o.get("alpha_at_construction") == 0.0 and o.get("init_bit_identical"))
    if gate == "G3":
        first, last = o.get("first") or {}, o.get("last") or {}
        return bool(float(first.get("alpha_grad_norm") or 0.0) != 0.0
                    and float(last.get("alpha") or 0.0) != 0.0
                    and float(last.get("generator_grad_norm") or 0.0) > 0.0
                    and float(o.get("alpha_final") or 0.0) != 0.0)
    if gate == "G4":
        return all(float(r) <= abs(float(o.get("alpha_abs") or 0.0)) + BOUND_TOL
                   for r in (o.get("ratio_max") or [1e9]))
    if gate == "G5":
        return all(RATIO_BAND[0] <= float(r) <= RATIO_BAND[1]
                   for r in (o.get("ratio_mean") or [1e9]))
    if gate == "G6":
        return bool(float(o.get("geff_std") or 0.0) > 0.0 and float(o.get("geff_max") or 0.0) > 0.0
                    and float(o.get("geff_min") or 0.0) >= 0.0)
    if gate == "G7":
        return bool(float(o.get("pred_std") or 0.0) > 0.0 and o.get("ref_pred_std") is not None
                    and float(o["ref_pred_std"]) > 0.0
                    and float(o["pred_std"]) >= PRED_STD_MIN_RATIO * float(o["ref_pred_std"]))
    if gate == "G8":
        return bool(sorted(str(n) for n in (o.get("names") or [])) == sorted(EXTRA_PARAM_NAMES)
                    and int(o.get("new_params_total", -1)) == EXPECTED_NEW_PARAMS_TOTAL
                    and int(o.get("head_params", -1)) == EXPECTED_HEAD_PARAMS)
    raise AssertionError(f"未知门禁 {gate}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="shuffled-condition 消融运行后独立复核（只读）")
    parser.add_argument("--root", type=str, default="artifacts/aliccp_bench")
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--arm-run", type=str, required=True)
    args = parser.parse_args(argv)
    root = Path(args.root)
    arm_dir = root / "runs" / args.arm_run
    base_dir = root / "runs" / args.baseline_run

    base = _read_json(base_dir / "metrics.json")
    base_gate = _read_json(base_dir / "gate_report.json")
    base_cfg = _read_json(base_dir / "config.json")
    arm = _read_json(arm_dir / "metrics.json")
    arm_gate = _read_json(arm_dir / "gate_report.json")
    arm_cfg = _read_json(arm_dir / "config.json")
    arm_prompt = _read_json(arm_dir / "prompt_report.json")
    compare = _read_json(arm_dir / "rp_shuffled_compare.json")
    ctx_base = _read_json(root / "runs" / BASELINE_RUN_ID / "metrics.json")
    ctx_corr = _read_json(root / "runs" / CORRECT_RUN_ID / "metrics.json")
    ctx_corr_prompt = _read_json(root / "runs" / CORRECT_RUN_ID / "prompt_report.json")

    # ---------------- 1. 文件 LF sha256 钉死 ----------------
    for rel, pin in FILES.items():
        check(f"file.{rel}.exists", (ROOT / rel).is_file())
        if (ROOT / rel).is_file():
            actual = _lf_sha(ROOT / rel)
            check(f"file.{rel}.lf_sha256", actual == pin, actual)

    # ---------------- 2. 错排独立重推 ----------------
    shuffle = arm_prompt.get("shuffle") or {}
    report = shuffle.get("report") or {}
    per_split = report.get("per_split") or {}
    check("shuffle.base_seed", report.get("base_seed") == COND_PERM_SEED, repr(report.get("base_seed")))
    check("shuffle.n_perms_total",
          report.get("n_perms_total") == sum(int(v.get("n_batches", -1)) for v in per_split.values()),
          repr(report.get("n_perms_total")))
    for split, rec in sorted(per_split.items()):
        n_batches, n_rows = int(rec["n_batches"]), int(rec["n_rows"])
        uniform = n_rows % max(n_batches, 1) == 0
        check(f"shuffle.{split}.uniform_batches", uniform, f"n_batches={n_batches} n_rows={n_rows}")
        if not uniform:
            continue
        n = n_rows // n_batches
        check(f"shuffle.{split}.digest_rederived",
              rederive_split_digest(split, n_batches, n) == rec["digest"], rec["digest"])
        check(f"shuffle.{split}.budget_covered", n_rows == int(arm_cfg["budgets"][split]),
              f"{n_rows} vs {arm_cfg['budgets'][split]}")
        check(f"shuffle.{split}.n_batches_expected",
              n_batches == -(-int(arm_cfg["budgets"][split]) // int(arm_cfg["batch_size"])),
              f"{n_batches}")
    check("shuffle.total_digest_rederived",
          all(int(v.get("n_rows", 0)) % max(int(v.get("n_batches", 1)), 1) == 0 for v in per_split.values())
          and rederive_total_digest(per_split) == report.get("total_digest"), repr(report.get("total_digest")))
    for split, samples in sorted((report.get("samples") or {}).items()):
        for sample in samples:
            expect = sattolo(int(sample["n"]), split, int(sample["batch_index"]))
            check(f"shuffle.{split}.sample[{sample['batch_index']}].perm_head",
                  sample["perm_head"] == expect[:len(sample["perm_head"])],
                  f"n={sample['n']} bk={sample['batch_index']}")

    # ---------------- 3. 置换门禁 PG1–PG6 独立重算 ----------------
    pg1 = int(report.get("fixed_points_total", -1)) == 0
    pg2a = int(report.get("bijection_failures", -1)) == 0
    pg2b = bool(per_split) and all(
        int(v["n_rows"]) == int(arm_cfg["budgets"][s])
        and int(v["n_batches"]) == -(-int(arm_cfg["budgets"][s]) // int(arm_cfg["batch_size"]))
        for s, v in per_split.items()) and set(per_split) == {"train", "val", "test"}
    pg3 = int(report.get("regeneration_mismatches", -1)) == 0
    pg4 = int(report.get("rng_isolation_violations", -1)) == 0
    pg5 = int(report.get("key_conflicts", -1)) == 0
    pg6 = _lf_sha(ROOT / "aliccp_benchmark/residual_prompt.py") == MECHANISM_LF_SHA
    recomputed_gates = {"PG1": pg1, "PG2": pg2a and pg2b, "PG3": pg3, "PG4": pg4, "PG5": pg5, "PG6": pg6}
    for gate, ok in recomputed_gates.items():
        check(f"perm.{gate}", ok)
    recorded = shuffle.get("gates") or {}
    check("perm.recorded_gates_consistent",
          all(bool((recorded.get(g) or {}).get("pass")) == ok for g, ok in recomputed_gates.items())
          and bool(shuffle.get("pass")) == all(recomputed_gates.values()))

    # ---------------- 4. identity / REP / A / M / 对照链 ----------------
    expected_ref_path = root / "runs" / BASELINE_RUN_ID / "newtask.pt"
    ref_recorded = (arm_prompt.get("reference_dispersion") or {}).get("newtask_checkpoint")
    identity = {
        "stage1_same_pinned": base["stage1_id"] == arm["stage1_id"] == STAGE1_ID,
        "seed_same_pinned": base["model_seed"] == arm["model_seed"] == 1688723740,
        "epochs_patience_tag": (base["epochs"], base["patience"], base["tag"])
        == (arm["epochs"], arm["patience"], arm["tag"]) == (5, 2, "short"),
        "dirty_false": (not base.get("git", {}).get("dirty")) and (not arm.get("git", {}).get("dirty")),
        "baseline_variant_no_suffix": base.get("variant") == "baseline"
        and not base["run_id"].endswith(RUN_ID_SUFFIX),
        "arm_variant_suffix": arm.get("variant") == VARIANT_SHUFFLED
        and arm["run_id"].endswith(RUN_ID_SUFFIX),
        "arm_ref_path_sha": ref_recorded is not None and Path(str(ref_recorded)) == expected_ref_path
        and expected_ref_path.is_file() and _sha256_file(expected_ref_path) == REF_HEAD_SHA,
        "stage1_shas": (base.get("backbone_sha256_loaded") == arm.get("backbone_sha256_loaded")
                        == STAGE1_BACKBONE_SHA
                        and base.get("env_ids_sha256") == arm.get("env_ids_sha256") == STAGE1_ENV_IDS_SHA
                        and base.get("fingerprint_sha256") == arm.get("fingerprint_sha256")
                        == STAGE1_FINGERPRINT_SHA),
        "commit_embedded": str(base.get("commit")) in base["run_id"]
        and str(arm.get("commit")) in arm["run_id"],
    }
    for name, ok in identity.items():
        check(f"identity.{name}", ok)

    base_pe = [e["val_auc_bsi"] for e in base["per_epoch"]]
    reproduction = {
        "per_epoch_val": base_pe == BASELINE_RECORD["per_epoch_val"],
        "best_epoch_val_test_gate": (base["best_epoch"] == BASELINE_RECORD["best_epoch"]
                                     and base["best_val_auc_bsi"] == BASELINE_RECORD["best_val_auc_bsi"]
                                     and base["test_auc_bsi"] == BASELINE_RECORD["test_auc_bsi"]
                                     and base["gate_mean"] == BASELINE_RECORD["gate_mean"]),
        "newtask_sha": (base_dir / "newtask.pt").is_file()
        and _sha256_file(base_dir / "newtask.pt") == REF_HEAD_SHA,
    }
    for name, ok in reproduction.items():
        check(f"reproduction.{name}", ok)

    check("protocol.baseline_a_class", a_class_ok(base_gate))
    check("protocol.arm_a_class", a_class_ok(arm_gate))
    arm_arm = arm.get("rp_arm") or {}
    gate_status = {}
    for gate in ("M0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"):
        rec = arm_arm.get(gate) or {}
        rederived = rederive_gate_pass(gate, rec.get("observed") or {})
        gate_status[gate] = "PASS" if rederived else "FAIL"
        check(f"mechanism.{gate}.record_consistent", bool(rec.get("pass")) == rederived,
              f"recorded={rec.get('pass')} rederived={rederived}")

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
                           == CORRECT_RECORD["classification"]
                           and ctx_corr_prompt.get("alpha_final") == CORRECT_RECORD["alpha_final"]),
        "correct_deltas": (ctx_corr["test_auc_bsi"] - ctx_base["test_auc_bsi"] == DELTA_TEST_CORRECT
                           and ctx_corr["best_val_auc_bsi"] - ctx_base["best_val_auc_bsi"]
                           == DELTA_VAL_CORRECT),
    }
    for name, ok in comparator.items():
        check(f"comparator.{name}", ok)

    # ---------------- 5. 效用与判定独立重算 ----------------
    g_c = ctx_corr["test_auc_bsi"] - base["test_auc_bsi"]
    g_s = arm["test_auc_bsi"] - base["test_auc_bsi"]
    L = g_c - g_s
    delta_val_s = arm["best_val_auc_bsi"] - base["best_val_auc_bsi"]
    if not all(identity.values()) :
        expect_verdict, expect_sub = "INVALID", "IDENTITY_MISMATCH"
    elif not all(reproduction.values()):
        expect_verdict, expect_sub = "INVALID", "BASELINE_REPRODUCTION_FAILED"
    elif not (a_class_ok(base_gate) and a_class_ok(arm_gate)):
        expect_verdict, expect_sub = "INVALID", "PROTOCOL_INVALID"
    elif not all(v == "PASS" for v in gate_status.values()):   # 用独立重推的门禁状态（不信任记录布尔）
        expect_verdict, expect_sub = "INVALID", "MECHANISM_FAIL"
    elif not all(comparator.values()):
        expect_verdict, expect_sub = "INVALID", "COMPARATOR_MISMATCH"
    elif not all(recomputed_gates.values()):
        expect_verdict, expect_sub = "INVALID", "PERMUTATION_INVALID"
    elif g_c < MATERIAL_DELTA:
        expect_verdict, expect_sub = "INVALID", "COMPARATOR_GAIN_NOT_MATERIAL"
    elif L >= MATERIAL_DELTA:
        expect_verdict = "SAMPLE_CONDITION_SUPPORTED"
        expect_sub = ("FULL_GAIN_REQUIRES_ALIGNMENT" if g_s < MATERIAL_DELTA
                      else "PARTIAL_ALIGNMENT_CONTRIBUTION")
    else:
        expect_verdict, expect_sub = "CONDITION_ALIGNMENT_NOT_SUPPORTED", "SHUFFLED_RETAINS_GAIN"
    check("verdict.independent", compare.get("verdict") == expect_verdict
          and compare.get("subreason") == expect_sub,
          f"recomputed={expect_verdict}/{expect_sub} recorded={compare.get('verdict')}/{compare.get('subreason')}")
    comp = compare.get("components") or {}
    check("components.bit_equal",
          comp.get("g_c") == g_c and comp.get("g_s") == g_s and comp.get("L") == L
          and comp.get("delta_val_s") == delta_val_s
          and comp.get("retained_ratio") == ((g_s / g_c) if g_c > 0 else None)
          and comp.get("shuffled_above_baseline") == (g_s > 0.0)
          and comp.get("shuffled_material_positive") == (g_s >= MATERIAL_DELTA)
          and comp.get("validation_agreement") == (delta_val_s > 0.0)
          and comp.get("G_s_ge_historical_0.0055") == (g_s >= 0.0055),
          json.dumps({"g_c": g_c, "g_s": g_s, "L": L}))
    check("secondary.independent", compare.get("secondary") == secondary_classification(g_s),
          f"recomputed={secondary_classification(g_s)} recorded={compare.get('secondary')}")
    if secondary_classification(g_s) == "NO_CLEAR_IMPROVEMENT":
        headroom = compare.get("headroom") or {}
        grad_probe = arm_prompt.get("grad_probe") or [{}]
        active = (abs(float(arm_prompt.get("alpha_final") or 0.0)) > 0.0
                  and float(grad_probe[-1].get("generator_grad_norm") or 0.0) > 0.0
                  and float((arm_prompt.get("gate") or {}).get("geff_std") or 0.0) > 0.0)
        right_censored = (arm.get("best_epoch") == arm.get("epochs") == len(arm.get("per_epoch") or []))
        check("headroom.independent",
              headroom.get("applicable") is True
              and headroom.get("mechanism_activity") == ("ACTIVE" if active else "INACTIVE")
              and headroom.get("validation_direction") == ("POSITIVE" if delta_val_s > 0 else "NON_POSITIVE")
              and headroom.get("epoch_trajectory")
              == ("RIGHT_CENSORED_STILL_IMPROVING" if right_censored else "EARLY_STOPPED")
              and headroom.get("gap_to_positive_threshold") == SECONDARY_POSITIVE_MIN - g_s)
    else:
        check("headroom.not_applicable", (compare.get("headroom") or {}).get("applicable") is False)

    # ---------------- 6. JSON 互洽 / SUMMARY / 元数据 ----------------
    check("consistency.rp_arm_three_way",
          (arm.get("rp_arm") or {}) == (arm_gate.get("residual_prompt") or {})
          and (arm.get("rp_arm") or {}).get("shuffle", {}).get("pass") == shuffle.get("pass"))
    check("consistency.prompt_arm_equals_rp_arm", (arm_prompt.get("arm") or {}) == (arm.get("rp_arm") or {}))
    check("consistency.prompt_hidden",
          arm.get("prompt_hidden") == arm_cfg.get("prompt_hidden") == 16)
    summary = (root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
    check("summary.arm_row", any(args.arm_run in line for line in summary))
    check("summary.baseline_row", any(args.baseline_run in line for line in summary))
    check("meta.commit_short7", len(str(arm.get("commit"))) == 7, str(arm.get("commit")))
    check("meta.arm_dir_exists", arm_dir.is_dir())

    ok_all = all(c["pass"] for c in checks)
    report_doc = {
        "script": Path(__file__).name,
        "baseline_run": args.baseline_run,
        "arm_run": args.arm_run,
        "recomputed": {"g_c": g_c, "g_s": g_s, "L": L, "delta_val_s": delta_val_s,
                       "verdict": expect_verdict, "subreason": expect_sub,
                       "secondary": secondary_classification(g_s),
                       "permutation_gates": recomputed_gates,
                       "mechanism_gate_status": gate_status},
        "checks": checks,
        "all_pass": ok_all,
        "n_checks": len(checks),
    }
    (arm_dir / "verify_report.json").write_text(json.dumps(report_doc, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"\nALL_PASS = {ok_all}  ({len(checks)} checks; report -> {arm_dir / 'verify_report.json'})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
