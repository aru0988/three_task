"""运行后独立复核（只读；本分支 exp/aliccp-stage2-residual-prompt-alpha-pinned-condition）。

用途：pinned-alpha 条件对应因果消融预注册 §10.5 的运行后独立复核。**不 import 任何机制/接线/分析
模块**（不 import `residual_prompt` / `rp_pinned` / `bench` / `protocol`）；全部判定由原始 run JSON
记录 + 文件哈希 + 本文件内**独立重实现**的错排规格（预注册 §2.2：sha256 键推导 + Sattolo）机械重推。

核验内容（恰一次；报告落 F_s run 目录 `verify_report.json`，exit 0/1）：
  1. 文件 LF sha256 钉死：mechanism / bench / CLI / rp_pinned / 守卫测试 / 前置核验脚本；
  2. 错排独立重推：按 (COND_PERM_SEED, split, batch_index, n) 键空间重生成全部 π → 逐 split digest 与
     总 digest 逐位 == 记录 == 历史钉死值（PG7/DD；⇒ 标签无关/确定性复现 + 同一张 deck）；sampled 键的
     perm_head 逐位核对；
  3. 置换门禁 PG1–PG7 独立重算（不信任记录布尔）：PG1–PG5 由 report 计数 + 覆盖核对；PG6 = 机制文件哈希；
     PG7 = deck 钉死值比对；
  4. identity / REP（B vs 历史基线钉死 + newtask.pt sha）/ A 类 / 对照链 独立重推；**PA1–PA8（含 M0）：
     由记录的 `observed` 值独立重推 pass/fail，并核对记录布尔与重推一致**（记录一致性口径——门禁本身
     失败属判定事实，不使本脚本失败；失败状态在报告 `recomputed.mechanism_gate_status` 单列）；
     PA1 另加 **checkpoint 独立读取**（`newtask.pt` 内 `prompt_gate` == 钉死 α，逐位）；
  5. 效用分量与判定树独立重算（gap/g_c/g_s/R/Δval_gap/verdict/subreason/两臂二级分类/headroom）与分析器
     报告逐位一致；
  6. 三处 JSON 互洽（metrics.rp_arm == gate_report.residual_prompt == prompt_report.arm）、SUMMARY 行、
     run 元数据（dirty=false、run_id 内嵌 commit）。

退出码 0 = 记录全部一致（含门禁记录与独立重推一致）；1 = 有不一致（逐项打印；不一致即停止，不改任何
run 产物）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent

# ---- §1.5/§5 钉死常量（与预注册 / 前置核验脚本逐位一致）----
STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"
BASELINE_RUN_ID = "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07"
CORRECT_RUN_ID = "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg"
SHUF_HIST_RUN_ID = "20261005-0933-p2M-v500k-t1M-m1688723740-short-9d26bc8-rpgs"
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
SHUF_HIST_RECORD = {
    "best_val_auc_bsi": 0.5819965357883198, "test_auc_bsi": 0.5989210260551207,
    "variant": "residual-prompt-shuffled", "classification": "MECHANISM_FAIL",
}
DELTA_TEST_CORRECT = 0.008103113327467715
DELTA_VAL_CORRECT = 0.008622497456618139
DELTA_TEST_SHUFFLED_HIST = 0.0014787611000699474
COND_PERM_SEED = 20261005
PINNED_ALPHA = 0.07101669907569885
MATERIAL_DELTA = 0.001
SECONDARY_POSITIVE_MIN = 0.001
SECONDARY_NEGATIVE_MAX = -0.02
VARIANT_PINNED = "residual-prompt-pinned"
VARIANT_PINNED_SHUFFLED = "residual-prompt-pinned-shuffled"
RUN_ID_SUFFIX_CORRECT = "-rpp"
RUN_ID_SUFFIX_SHUFFLED = "-rpps"
HISTORICAL_U1_THRESHOLD = 0.0055

# ---- PA 门禁判据常量（与钉死 `pinned_mechanism_gates` 语义逐字一致；供独立重推）----
REFERENCE_AUC = 0.5809347091990792
REFERENCE_PRED_STD = 0.005217193225189258
REFERENCE_IDENTITY_TOL = 1e-9
BOUND_TOL = 1e-6
RATIO_SPREAD_TOL = 1e-6
PRED_STD_MIN_RATIO = 0.5
EXPECTED_TRAINABLE_PROMPT_KEYS = ("prompt_generator.0.bias", "prompt_generator.0.weight",
                                  "prompt_generator.2.bias", "prompt_generator.2.weight")
EXPECTED_STATE_EXTRA_KEYS = ("prompt_gate",) + EXPECTED_TRAINABLE_PROMPT_KEYS
EXPECTED_NEW_PARAMS_TOTAL = 2384
EXPECTED_HEAD_PARAMS = 8129

# ---- deck 钉死（预注册 §1.2-A7）----
DECK_PIN = {
    "base_seed": COND_PERM_SEED,
    "n_perms_total": 1750,
    "per_split": {
        "train": {"n_batches": 1000, "n_rows": 2_000_000,
                  "digest": "50ecd19e5775069bfc1270e26c11f13feff34bfc238d61ae87a5e0f5c9ac4d60"},
        "val": {"n_batches": 250, "n_rows": 500_000,
                "digest": "2d85298c509acc26f2d9ee7654e2fb4accdf8d76d68e280e0946a68171b966c0"},
        "test": {"n_batches": 500, "n_rows": 1_000_000,
                 "digest": "239856ffef561c9a4c5dd19fee383a8c19f218fc92d8ea6aec66cb0acdd622cb"},
    },
    "total_digest": "dcf302c1a80ae7e8023345105e559d7a6daaa1ad492b016c39c22705886599d8",
}

# ---- 文件 LF sha256 钉死（C2 时计算；与守卫测试一致；哈希链无环）----
MECHANISM_LF_SHA = "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc"
BENCH_LF_SHA = "a02b8ea67d61ffdb5f6df0513c51d534e23653d647bc2663f13b2a1381f786d0"
CLI_LF_SHA = "98c0e5d373508c2c1b4b40cab7c0c400792bf70fd4151bc62f40592efad18fd8"
PINNED_MODULE_LF_SHA = "ba23bf473634ff647b482ebf6a92f27ca7f073432bb4ed7a1342b8b11101df9b"
GUARD_TEST_LF_SHA = "a0ffe303bf484e7d61a6dde3cd0a846039ac9051b9d308adf48b9a440593b7e1"
PRERUN_VERIFIER_LF_SHA = "13371ca418f284aedcb27439a9f36415bcf95d630fe431c020bc8e8d66844d2f"

FILES = {
    "aliccp_benchmark/residual_prompt.py": MECHANISM_LF_SHA,
    "aliccp_benchmark/bench.py": BENCH_LF_SHA,
    "run_aliccp_benchmark.py": CLI_LF_SHA,
    "aliccp_benchmark/rp_pinned.py": PINNED_MODULE_LF_SHA,
    "aliccp_benchmark/tests/test_residual_prompt_pinned.py": GUARD_TEST_LF_SHA,
    "verify_pinned_prerun.py": PRERUN_VERIFIER_LF_SHA,
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


# ---- §2.2 规格独立重实现（不改任何共享代码）----
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


def secondary_classification(delta_test: float) -> str:
    if float(delta_test) >= SECONDARY_POSITIVE_MIN:
        return "POSITIVE_IMPROVEMENT"
    if float(delta_test) <= SECONDARY_NEGATIVE_MAX:
        return "CLEAR_DEGRADATION"
    return "NO_CLEAR_IMPROVEMENT"


def rederive_gate_pass(gate: str, observed: dict) -> bool:
    """由记录的 observed 值独立重推门禁通过与否（不信任记录的 pass 布尔；判据 == 钉死 pinned_mechanism_gates）。"""
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
                    and (o.get("probe_alpha_grad_norms") or [1]) is not None
                    and all(g is None for g in (o.get("probe_alpha_grad_norms") or []))
                    and bool(o.get("probe_alpha_grad_norms") is not None))
    if gate == "PA3":
        return bool(o.get("shared_params_bit_identical") and o.get("global_rng_endpoint_identical")
                    and o.get("extra_keys") == sorted(EXPECTED_STATE_EXTRA_KEYS))
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="pinned-alpha 消融运行后独立复核（只读）")
    parser.add_argument("--root", type=str, default="artifacts/aliccp_bench")
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--arm-correct-run", type=str, required=True)
    parser.add_argument("--arm-shuffled-run", type=str, required=True)
    args = parser.parse_args(argv)
    root = Path(args.root)
    base_dir = root / "runs" / args.baseline_run
    fc_dir = root / "runs" / args.arm_correct_run
    fs_dir = root / "runs" / args.arm_shuffled_run

    base = _read_json(base_dir / "metrics.json")
    base_gate = _read_json(base_dir / "gate_report.json")
    fc = _read_json(fc_dir / "metrics.json")
    fc_gate = _read_json(fc_dir / "gate_report.json")
    fc_prompt = _read_json(fc_dir / "prompt_report.json")
    fs = _read_json(fs_dir / "metrics.json")
    fs_gate = _read_json(fs_dir / "gate_report.json")
    fs_cfg = _read_json(fs_dir / "config.json")
    fs_prompt = _read_json(fs_dir / "prompt_report.json")
    compare = _read_json(fs_dir / "rp_pinned_compare.json")
    ctx_base = _read_json(root / "runs" / BASELINE_RUN_ID / "metrics.json")
    ctx_corr = _read_json(root / "runs" / CORRECT_RUN_ID / "metrics.json")
    ctx_corr_prompt = _read_json(root / "runs" / CORRECT_RUN_ID / "prompt_report.json")
    ctx_shuf = _read_json(root / "runs" / SHUF_HIST_RUN_ID / "metrics.json")

    # ---------------- 1. 文件 LF sha256 钉死 ----------------
    for rel, pin in FILES.items():
        check(f"file.{rel}.exists", (ROOT / rel).is_file())
        if (ROOT / rel).is_file():
            actual = _lf_sha(ROOT / rel)
            check(f"file.{rel}.lf_sha256", actual == pin, actual)

    # ---------------- 2. 错排独立重推（PG3/PG5/PG7/DD）----------------
    shuffle = fs_prompt.get("shuffle") or {}
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
        check(f"shuffle.{split}.budget_covered", n_rows == int(fs_cfg["budgets"][split]),
              f"{n_rows} vs {fs_cfg['budgets'][split]}")
        check(f"shuffle.{split}.n_batches_expected",
              n_batches == -(-int(fs_cfg["budgets"][split]) // int(fs_cfg["batch_size"])),
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

    # ---------------- 3. 置换门禁 PG1–PG7 独立重算 ----------------
    pg1 = int(report.get("fixed_points_total", -1)) == 0
    pg2a = int(report.get("bijection_failures", -1)) == 0
    pg2b = bool(per_split) and all(
        int(v["n_rows"]) == int(fs_cfg["budgets"][s])
        and int(v["n_batches"]) == -(-int(fs_cfg["budgets"][s]) // int(fs_cfg["batch_size"]))
        for s, v in per_split.items()) and set(per_split) == {"train", "val", "test"}
    pg3 = int(report.get("regeneration_mismatches", -1)) == 0
    pg4 = int(report.get("rng_isolation_violations", -1)) == 0
    pg5 = int(report.get("key_conflicts", -1)) == 0
    pg6 = _lf_sha(ROOT / "aliccp_benchmark/residual_prompt.py") == MECHANISM_LF_SHA
    deck_ok = (report.get("base_seed") == DECK_PIN["base_seed"]
               and report.get("n_perms_total") == DECK_PIN["n_perms_total"]
               and report.get("total_digest") == DECK_PIN["total_digest"]
               and all((per_split.get(s) or {}).get("digest") == pin["digest"]
                       and ((per_split.get(s) or {}).get("n_batches"),
                            (per_split.get(s) or {}).get("n_rows")) == (pin["n_batches"], pin["n_rows"])
                       for s, pin in DECK_PIN["per_split"].items()))
    recomputed_gates = {"PG1": pg1, "PG2": pg2a and pg2b, "PG3": pg3, "PG4": pg4, "PG5": pg5,
                        "PG6": pg6, "PG7": deck_ok}
    for gate, ok in recomputed_gates.items():
        check(f"perm.{gate}", ok)
    recorded = shuffle.get("gates") or {}
    check("perm.recorded_gates_consistent",
          all(bool((recorded.get(g) or {}).get("pass")) == ok for g, ok in recomputed_gates.items())
          and bool(shuffle.get("pass")) == all(recomputed_gates.values()))

    # ---------------- 4. identity / REP / A / PA / 对照链 ----------------
    expected_ref_path = root / "runs" / BASELINE_RUN_ID / "newtask.pt"
    fc_ref_recorded = (fc_prompt.get("reference_dispersion") or {}).get("newtask_checkpoint")
    fs_ref_recorded = (fs_prompt.get("reference_dispersion") or {}).get("newtask_checkpoint")
    identity = {
        "stage1_same_pinned": base["stage1_id"] == fc["stage1_id"] == fs["stage1_id"] == STAGE1_ID,
        "seed_same_pinned": base["model_seed"] == fc["model_seed"] == fs["model_seed"] == 1688723740,
        "epochs_patience_tag": all((r["epochs"], r["patience"], r["tag"]) == (5, 2, "short")
                                   for r in (base, fc, fs)),
        "dirty_false": all(not r.get("git", {}).get("dirty") for r in (base, fc, fs)),
        "baseline_variant_no_suffix": base.get("variant") == "baseline"
        and not base["run_id"].endswith(("-rpp", "-rpps", "-rpg")),
        "arm_correct_variant_suffix": fc.get("variant") == VARIANT_PINNED
        and fc["run_id"].endswith(RUN_ID_SUFFIX_CORRECT),
        "arm_shuffled_variant_suffix": fs.get("variant") == VARIANT_PINNED_SHUFFLED
        and fs["run_id"].endswith(RUN_ID_SUFFIX_SHUFFLED),
        "arms_ref_path_sha": (fc_ref_recorded is not None and fs_ref_recorded is not None
                              and Path(str(fc_ref_recorded)) == expected_ref_path
                              and Path(str(fs_ref_recorded)) == expected_ref_path
                              and expected_ref_path.is_file()
                              and _sha256_file(expected_ref_path) == REF_HEAD_SHA),
        "stage1_shas": all(r.get("backbone_sha256_loaded") == STAGE1_BACKBONE_SHA
                           and r.get("env_ids_sha256") == STAGE1_ENV_IDS_SHA
                           and r.get("fingerprint_sha256") == STAGE1_FINGERPRINT_SHA
                           for r in (base, fc, fs)),
        "commit_embedded": all(str(r.get("commit")) in r["run_id"] for r in (base, fc, fs)),
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
    check("protocol.arm_correct_a_class", a_class_ok(fc_gate))
    check("protocol.arm_shuffled_a_class", a_class_ok(fs_gate))

    gate_status = {}
    for label, doc in (("correct", fc), ("shuffled", fs)):
        arm = doc.get("rp_arm") or {}
        for gate in ("M0", "PA1", "PA2", "PA3", "PA4", "PA5", "PA6", "PA7", "PA8"):
            rec = arm.get(gate) or {}
            rederived = rederive_gate_pass(gate, rec.get("observed") or {})
            gate_status[f"{label}.{gate}"] = "PASS" if rederived else "FAIL"
            check(f"mechanism.{label}.{gate}.record_consistent", bool(rec.get("pass")) == rederived,
                  f"recorded={rec.get('pass')} rederived={rederived}")

    # PA1 checkpoint 独立读取：newtask.pt 内 prompt_gate == 钉死 α（逐位）
    for label, run_dir in (("correct", fc_dir), ("shuffled", fs_dir)):
        state = torch.load(run_dir / "newtask.pt", map_location="cpu")
        check(f"checkpoint.{label}.prompt_gate_pinned",
              "prompt_gate" in state and float(state["prompt_gate"]) == PINNED_ALPHA,
              repr(state.get("prompt_gate")))
        check(f"checkpoint.{label}.pure_pinned_key_set",
              sorted(state) == sorted(set(state)) and "prompt_gate" in state
              and all(k in state for k in EXPECTED_TRAINABLE_PROMPT_KEYS))

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
        "pin_source_alpha": ctx_corr_prompt.get("alpha_final") == PINNED_ALPHA,
        "pinned_arms_alpha_equal_pin": (fc_prompt.get("alpha_final") == PINNED_ALPHA
                                        and fs_prompt.get("alpha_final") == PINNED_ALPHA),
        "correct_deltas": (ctx_corr["test_auc_bsi"] - ctx_base["test_auc_bsi"] == DELTA_TEST_CORRECT
                           and ctx_corr["best_val_auc_bsi"] - ctx_base["best_val_auc_bsi"]
                           == DELTA_VAL_CORRECT),
        "shuf_hist_delta": (ctx_shuf["test_auc_bsi"] - ctx_base["test_auc_bsi"]
                            == DELTA_TEST_SHUFFLED_HIST),
    }
    for name, ok in comparator.items():
        check(f"comparator.{name}", ok)

    # ---------------- 5. 效用与判定独立重算 ----------------
    base_test, base_val = base["test_auc_bsi"], base["best_val_auc_bsi"]
    fc_test, fc_val = fc["test_auc_bsi"], fc["best_val_auc_bsi"]
    fs_test, fs_val = fs["test_auc_bsi"], fs["best_val_auc_bsi"]
    gap = fc_test - fs_test
    g_c, g_s = fc_test - base_test, fs_test - base_test
    delta_val_gap = fc_val - fs_val
    delta_val_c, delta_val_s = fc_val - base_val, fs_val - base_val
    pa_fail_pin = any(gate_status[k] == "FAIL" for k in
                      ("correct.PA1", "correct.PA2", "correct.PA8",
                       "shuffled.PA1", "shuffled.PA2", "shuffled.PA8"))
    pa_fail_any = any(v == "FAIL" for v in gate_status.values())

    if not all(identity.values()):
        expect_verdict, expect_sub = "INVALID", "IDENTITY_MISMATCH"
    elif not all(reproduction.values()):
        expect_verdict, expect_sub = "INVALID", "BASELINE_REPRODUCTION_FAILED"
    elif not (a_class_ok(base_gate) and a_class_ok(fc_gate) and a_class_ok(fs_gate)):
        expect_verdict, expect_sub = "INVALID", "PROTOCOL_INVALID"
    elif pa_fail_pin:
        expect_verdict, expect_sub = "INVALID", "PINNED_ALPHA_INVALID"
    elif pa_fail_any:
        expect_verdict, expect_sub = "INVALID", "MECHANISM_FAIL"
    elif not all(comparator.values()):
        expect_verdict, expect_sub = "INVALID", "COMPARATOR_MISMATCH"
    elif not deck_ok:
        expect_verdict, expect_sub = "INVALID", "DECK_MISMATCH"
    elif not all(recomputed_gates.values()):
        expect_verdict, expect_sub = "INVALID", "PERMUTATION_INVALID"
    elif gap >= MATERIAL_DELTA and delta_val_gap > 0.0:
        expect_verdict, expect_sub = "SAMPLE_CONDITION_SUPPORTED", None
    else:
        expect_verdict = "CONDITION_ALIGNMENT_NOT_SUPPORTED"
        expect_sub = "GAP_BELOW_MATERIALITY" if gap < MATERIAL_DELTA else "VALIDATION_DISAGREEMENT"
    check("verdict.independent", compare.get("verdict") == expect_verdict
          and compare.get("subreason") == expect_sub,
          f"recomputed={expect_verdict}/{expect_sub} recorded={compare.get('verdict')}/{compare.get('subreason')}")
    comp = compare.get("components") or {}
    check("components.bit_equal",
          comp.get("gap") == gap and comp.get("g_c") == g_c and comp.get("g_s") == g_s
          and comp.get("delta_val_gap") == delta_val_gap
          and comp.get("gap_material_positive") == (gap >= MATERIAL_DELTA)
          and comp.get("validation_agreement") == (delta_val_gap > 0.0)
          and comp.get("shuffled_matches_or_exceeds_correct") == (gap <= 0.0)
          and comp.get("retained_ratio_vs_baseline") == ((g_s / g_c) if g_c > 0 else None)
          and comp.get("G_c_ge_historical_0.0055") == (g_c >= HISTORICAL_U1_THRESHOLD)
          and comp.get("G_s_ge_historical_0.0055") == (g_s >= HISTORICAL_U1_THRESHOLD),
          json.dumps({"gap": gap, "g_c": g_c, "g_s": g_s}))
    secondary = compare.get("secondary") or {}
    check("secondary.independent",
          secondary.get("arm_correct") == secondary_classification(g_c)
          and secondary.get("arm_shuffled") == secondary_classification(g_s),
          f"recomputed={secondary_classification(g_c)}/{secondary_classification(g_s)} "
          f"recorded={secondary.get('arm_correct')}/{secondary.get('arm_shuffled')}")
    headroom = compare.get("headroom") or {}
    for label, delta_test, delta_val, arm_doc, prompt_doc in (
            ("arm_correct", g_c, delta_val_c, fc, fc_prompt),
            ("arm_shuffled", g_s, delta_val_s, fs, fs_prompt)):
        rec = headroom.get(label) or {}
        if secondary_classification(delta_test) != "NO_CLEAR_IMPROVEMENT":
            check(f"headroom.{label}.not_applicable", rec.get("applicable") is False)
            continue
        grad_probe = prompt_doc.get("grad_probe") or [{}]
        ratio_mean = [float(r) for r in ((prompt_doc.get("val_stats") or {}).get("ratio_mean") or [])]
        active = (abs(float(prompt_doc.get("alpha_final") or 0.0)) > 0.0
                  and float((prompt_doc.get("gate") or {}).get("geff_std") or 0.0) > 0.0
                  and bool(ratio_mean) and all(r > 0.0 for r in ratio_mean))
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
          (fc.get("rp_arm") or {}) == (fc_gate.get("residual_prompt") or {})
          and (fs.get("rp_arm") or {}) == (fs_gate.get("residual_prompt") or {}))
    check("consistency.prompt_arm_equals_rp_arm",
          (fc_prompt.get("arm") or {}) == (fc.get("rp_arm") or {})
          and (fs_prompt.get("arm") or {}) == (fs.get("rp_arm") or {}))
    check("consistency.prompt_hidden",
          fc.get("prompt_hidden") == fs_cfg.get("prompt_hidden") == 16)
    summary = (root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
    check("summary.baseline_row", any(args.baseline_run in line for line in summary))
    check("summary.arm_correct_row", any(args.arm_correct_run in line for line in summary))
    check("summary.arm_shuffled_row", any(args.arm_shuffled_run in line for line in summary))
    check("meta.commit_short7", all(len(str(r.get("commit"))) == 7 for r in (base, fc, fs)))

    ok_all = all(c["pass"] for c in checks)
    report_doc = {
        "script": Path(__file__).name,
        "baseline_run": args.baseline_run,
        "arm_correct_run": args.arm_correct_run,
        "arm_shuffled_run": args.arm_shuffled_run,
        "recomputed": {"gap": gap, "g_c": g_c, "g_s": g_s, "delta_val_gap": delta_val_gap,
                       "verdict": expect_verdict, "subreason": expect_sub,
                       "secondary": {"arm_correct": secondary_classification(g_c),
                                     "arm_shuffled": secondary_classification(g_s)},
                       "permutation_gates": recomputed_gates,
                       "mechanism_gate_status": gate_status},
        "checks": checks,
        "all_pass": ok_all,
        "n_checks": len(checks),
    }
    (fs_dir / "verify_report.json").write_text(json.dumps(report_doc, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"\nALL_PASS = {ok_all}  ({len(checks)} checks; report -> {fs_dir / 'verify_report.json'})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
