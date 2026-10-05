"""AliCCP 阶段 2 残差 Prompt 延迟解冻消融——**运行后独立复核**（只读；纯 JSON 重推 + checkpoint 独立读取）。

唯一事实来源：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-delayed-unfreeze-dynamics-design.md
（§10.5）。本脚本**不导入** `aliccp_benchmark.rp_delay` 的判定代码：全部常量独立重打、
判定树独立重实现，只从两 run 的原始 JSON 与 checkpoint 字节重推，并与分析器
`rp_delay_compare.json` 逐位比对。只读：不写 run/stage1 产物；唯一写入 = 复核报告 JSON（落 D run 目录）。

用法（cwd 任意；路径锚定本文件所在仓库根）：
    python verify_rp_delay.py --baseline-run <B run_id> --delay-run <D run_id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
RUNS_DEFAULT = REPO / "artifacts" / "aliccp_bench" / "runs"

# ---- 独立重打的钉死常量（与分析器同值；此处为独立副本）----
STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"
STAGE1_BACKBONE_SHA = "e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c"
STAGE1_ENV_IDS_SHA = "5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0"
STAGE1_FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
MODEL_SEED = 1688723740
BASELINE_RUN_ID = "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07"
LEARNABLE_RUN_ID = "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg"
SHUF_HIST_RUN_ID = "20261005-0933-p2M-v500k-t1M-m1688723740-short-9d26bc8-rpgs"
PINNED_RUN_ID = "20261005-1029-p2M-v500k-t1M-m1688723740-short-7d26918-rpp"
PINNED_SHUFFLED_RUN_ID = "20261005-1032-p2M-v500k-t1M-m1688723740-short-1c13841-rpps"
REF_HEAD_SHA = "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f"
BASELINE_AUC_TEST = 0.5974422649550507
BASELINE_AUC_VAL = 0.5809347091990792
REFERENCE_PRED_STD = 0.005217193225189258
BASELINE_PER_EPOCH_VAL = [0.4645711559431739, 0.48564481224085004, 0.5142344439088465,
                          0.5508809596059387, 0.5809347091990792]
BASELINE_PER_EPOCH_LOSS = [0.08193688414408826, 0.05193358083860949, 0.04973645433795173,
                           0.04873575337347574, 0.04805051837593783]
BASELINE_EP1_TRAIN_LOSS = BASELINE_PER_EPOCH_LOSS[0]
BASELINE_EP1_VAL = BASELINE_PER_EPOCH_VAL[0]
BASELINE_GATE_MEAN = [0.789178, 0.2108221875]
DELTA_TEST_LEARNABLE = 0.008103113327467715
DELTA_TEST_PINNED = -0.002677172368479308
GAP = 0.010780285695947023
HALF_GAP_TARGET = 0.0027129704794942035
MATERIAL_DELTA = 0.001
SECONDARY_POSITIVE_MIN = 0.001
SECONDARY_NEGATIVE_MAX = -0.02
DELAY_EPOCHS = 1
UNFREEZE_EPOCH = 2
EXPECTED_PROMPT_KEYS = ("prompt_gate", "prompt_generator.0.bias", "prompt_generator.0.weight",
                        "prompt_generator.2.bias", "prompt_generator.2.weight")
REFERENCE_IDENTITY_TOL = 1e-9
BOUND_TOL = 1e-6
RATIO_BAND = (0.005, 0.5)
PRED_STD_MIN_RATIO = 0.5
LEARNABLE_PROMPT_REPORT_SHA = "185df4d0f9910508aeff84ed780e7df4fe8ad4fd6b8451ff429a74927d333c1e"
LEARNABLE_ALPHA_FINAL = 0.07101669907569885
LEARNABLE_TRAJECTORY_RUNS = [
    ("20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg",
     "185df4d0f9910508aeff84ed780e7df4fe8ad4fd6b8451ff429a74927d333c1e"),
    ("20261004-0431-p2M-v500k-t1M-m1688723740-long-f2ccec2-rpg",
     "ba1d41a777de4ec7713132a5946b994b871c4f4b1682f02e152ad23ff83d7bb8"),
    ("20261005-0829-p2M-v500k-t1M-m1688723740-xlong-eafc336-rpg",
     "d5a15cc479bdff605087fe8c5cda11556cd8ede5489bc36996026bcb70c0879c"),
    ("20261005-0642-p2M-v500k-t1M-m1688738016-short-c17b100-rpg",
     "daeb0d560b3c9c3bc8d9507579d43cdb6c75db317d31464b662e2dcaa511b5de"),
    ("20261005-0646-p2M-v500k-t1M-m1688749593-short-c17b100-rpg",
     "bc55349f07c26ec90078d43eaee5eb78388aa17396deaf970eb90dafdb28b51e"),
    ("20261005-0649-p2M-v500k-t1M-m1688762746-short-c17b100-rpg",
     "69cc23b070a22666594b09fcd30897da69a4b6246961cf15b02998148629d565"),
]
# 实现 LF sha 钉死（C2 冻结后填写；任何后续实现改动都会使复核失败——漂移检测）
FROZEN_LF_SHA = {
    "aliccp_benchmark/residual_prompt.py":
        "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc",
    "aliccp_benchmark/rp_pinned.py":
        "ba23bf473634ff647b482ebf6a92f27ca7f073432bb4ed7a1342b8b11101df9b",
    "aliccp_benchmark/rp_delay.py": "a3ef45d729ea7372da85b91484622e1e4fff24e5517a14a4c525fef1de49146b",
    "aliccp_benchmark/bench.py": "983ced60e9660bf3be337414b88d7925c66f6fed7f4c61962967dd4631da4bbc",
    "run_aliccp_benchmark.py": "dcdfed464de74ce8d485326bd3aecf17c18cd1f3a2c84f4831ed1165488f02db",
    "aliccp_benchmark/tests/test_residual_prompt.py": "b294f071bb8e508cdf46aa562579e793dc3891605edb6820c5c332b978fe2cc3",
    "aliccp_benchmark/tests/test_residual_prompt_delay.py": "d2bb4da504b79f5110844360a8e5e032b40c500919b476b633dff455d2604bd2",
    "verify_delay_prerun.py": "3c34ae48f7dccb3ec1df989b0469c35230d80e3a6a5b3fe8e2328faa44d47166",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_sha(rel: str) -> str:
    return hashlib.sha256((REPO / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Checker:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, group: str, name: str, ok: bool, detail=None) -> bool:
        self.checks.append({"group": group, "name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def eq(self, group: str, name: str, actual, expected) -> bool:
        return self.check(group, name, actual == expected, {"actual": actual, "expected": expected})

    @property
    def failed(self):
        return [c for c in self.checks if not c["ok"]]

    def summary(self) -> dict:
        return {"total": len(self.checks), "passed": len(self.checks) - len(self.failed),
                "failed": len(self.failed)}


def recompute_dd(report: dict) -> dict:
    """独立重实现 DD1–DD9（不导入分析器代码；键名兼容 construction/construction_identity）。"""
    construction = report.get("construction") or report.get("construction_identity") or {}
    init_forward = report.get("init_forward") or {}
    records = report.get("grad_probe") or []
    delay = report.get("delay") or {}
    alpha_final = float(report.get("alpha_final") or 0.0)
    gate = report.get("gate") or {}
    streams = report.get("val_stats") or {}
    ref_std = ((report.get("reference_dispersion") or {}).get("pred_dispersion") or {}).get("pred_std")
    rr_after = delay.get("requires_grad_after_unfreeze") or {}
    frozen_sig = bool(records and records[0].get("alpha") == 0.0
                      and records[0].get("alpha_grad_norm") is None
                      and records[0].get("generator_grad_norm") == 0.0)
    dd1 = bool(construction.get("shared_params_bit_identical")
               and construction.get("global_rng_endpoint_identical")
               and construction.get("extra_keys") == sorted(EXPECTED_PROMPT_KEYS))
    dd2 = bool(construction.get("alpha_at_construction") == 0.0 and init_forward.get("bit_identical"))
    dd3 = bool(delay.get("delay_epochs") == DELAY_EPOCHS and delay.get("unfreeze_epoch") == UNFREEZE_EPOCH
               and delay.get("alpha_at_construction") == 0.0
               and delay.get("alpha_requires_grad_at_construction") is False
               and delay.get("generator_requires_grad_at_construction") == [False] * 4
               and delay.get("alpha_at_unfreeze") == 0.0
               and delay.get("generator_sha_at_unfreeze") is not None
               and delay.get("generator_sha_at_unfreeze") == delay.get("generator_sha_at_construction")
               and delay.get("unfrozen_once") is True and rr_after.get("alpha") is True
               and rr_after.get("generator") == [True] * 4
               and delay.get("optimizer_covers_named_parameters") is True
               and delay.get("unfreeze_rng_endpoint_unchanged") is True and frozen_sig)
    if len(records) >= 2:
        dd4 = bool(records[1].get("alpha_grad_norm") is not None
                   and float(records[1].get("alpha_grad_norm") or 0.0) != 0.0
                   and records[1].get("generator_grad_norm") == 0.0
                   and float(records[-1].get("alpha") or 0.0) != 0.0
                   and float(records[-1].get("generator_grad_norm") or 0.0) > 0.0
                   and alpha_final != 0.0)
    else:
        dd4 = False
    ratio_max = [float(x) for x in (streams.get("ratio_max") or [])]
    ratio_mean = [float(x) for x in (streams.get("ratio_mean") or [])]
    dd5 = bool(ratio_max) and all(x <= abs(alpha_final) + BOUND_TOL for x in ratio_max)
    dd6 = bool(ratio_mean) and all(RATIO_BAND[0] <= x <= RATIO_BAND[1] for x in ratio_mean)
    dd7 = bool(float(gate.get("geff_std") or 0.0) > 0 and float(gate.get("geff_max") or 0.0) > 0
               and float(gate.get("geff_min") or 0.0) >= 0.0)
    pred_std = float((report.get("dispersion") or {}).get("pred_std") or 0.0)
    dd8 = bool(pred_std > 0 and ref_std is not None and float(ref_std) > 0
               and pred_std >= PRED_STD_MIN_RATIO * float(ref_std))
    params = report.get("params") or {}
    names = sorted(str(i.get("name")) for i in (params.get("new_param_list") or []))
    dd9 = bool(names == sorted(EXPECTED_PROMPT_KEYS) and params.get("new_params_total") == 2385
               and params.get("head_params") == 8129)
    ref_auc = (report.get("reference_dispersion") or {}).get("val_auc")
    m0 = bool(ref_auc is not None and abs(float(ref_auc) - BASELINE_AUC_VAL) <= REFERENCE_IDENTITY_TOL
              and ref_std is not None
              and abs(float(ref_std) - REFERENCE_PRED_STD) <= REFERENCE_IDENTITY_TOL)
    return {"M0": m0, "DD1": dd1, "DD2": dd2, "DD3": dd3, "DD4": dd4, "DD5": dd5, "DD6": dd6,
            "DD7": dd7, "DD8": dd8, "DD9": dd9}


def verdict_tree(*, invalid_subreason, delta_d: float, delta_val_d: float) -> tuple:
    if invalid_subreason is not None:
        return "INVALID", invalid_subreason
    if delta_d >= MATERIAL_DELTA and delta_val_d > 0.0:
        return "DYNAMICS_SUPPORTED", None
    return "DYNAMICS_NOT_SUPPORTED", ("BELOW_MATERIALITY" if delta_d < MATERIAL_DELTA
                                      else "VALIDATION_DISAGREEMENT")


def secondary_class(delta_d: float) -> str:
    if delta_d >= SECONDARY_POSITIVE_MIN:
        return "POSITIVE_IMPROVEMENT"
    if delta_d <= SECONDARY_NEGATIVE_MAX:
        return "CLEAR_DEGRADATION"
    return "NO_CLEAR_IMPROVEMENT"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="延迟解冻消融运行后独立复核（只读）")
    parser.add_argument("--runs-root", type=Path, default=RUNS_DEFAULT)
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--delay-run", type=str, required=True)
    args = parser.parse_args(argv)
    runs = Path(args.runs_root)
    chk = Checker()

    b_dir, d_dir = runs / args.baseline_run, runs / args.delay_run
    b = load_json(b_dir / "metrics.json")
    d = load_json(d_dir / "metrics.json")
    d_report = load_json(d_dir / "prompt_report.json")
    compare = load_json(d_dir / "rp_delay_compare.json")

    # ---- 身份重推 ----
    g = "identity"
    chk.eq(g, "b_dirty", b["git"]["dirty"], False)
    chk.eq(g, "d_dirty", d["git"]["dirty"], False)
    chk.eq(g, "b_variant", b.get("variant"), "baseline")
    chk.eq(g, "d_variant", d.get("variant"), "residual-prompt-delay")
    chk.check(g, "d_suffix", d["run_id"].endswith("-rpd"), d["run_id"])
    chk.eq(g, "stage1", (b["stage1_id"], d["stage1_id"]), (STAGE1_ID, STAGE1_ID))
    chk.eq(g, "seed", (b["model_seed"], d["model_seed"]), (MODEL_SEED, MODEL_SEED))
    chk.eq(g, "point", (d["epochs"], d["patience"], d["tag"]), (5, 2, "short"))
    chk.eq(g, "lineage", (d["backbone_sha256_loaded"], d["env_ids_sha256"],
                          d["fingerprint_sha256"]),
           (STAGE1_BACKBONE_SHA, STAGE1_ENV_IDS_SHA, STAGE1_FINGERPRINT_SHA))
    ref_head = runs / BASELINE_RUN_ID / "newtask.pt"
    chk.eq(g, "ref_head_sha", sha256_file(ref_head), REF_HEAD_SHA)

    # ---- REP_B 重推 ----
    g = "rep_b"
    chk.eq(g, "per_epoch_val", [e["val_auc_bsi"] for e in b["per_epoch"]], BASELINE_PER_EPOCH_VAL)
    chk.eq(g, "per_epoch_loss", [e["train_loss"] for e in b["per_epoch"]], BASELINE_PER_EPOCH_LOSS)
    chk.eq(g, "test", b["test_auc_bsi"], BASELINE_AUC_TEST)
    chk.eq(g, "val", b["best_val_auc_bsi"], BASELINE_AUC_VAL)
    chk.eq(g, "best_epoch", b["best_epoch"], 5)
    chk.eq(g, "gate_mean", b["gate_mean"], BASELINE_GATE_MEAN)
    chk.eq(g, "b_newtask_sha", sha256_file(b_dir / "newtask.pt"), REF_HEAD_SHA)

    # ---- FROZEN_ID 重推 ----
    g = "frozen_id"
    delay_block = d_report.get("delay") or {}
    chk.eq(g, "ep1_loss_triple", (d["per_epoch"][0]["train_loss"], b["per_epoch"][0]["train_loss"]),
           (BASELINE_EP1_TRAIN_LOSS, BASELINE_EP1_TRAIN_LOSS))
    chk.eq(g, "ep1_val_triple", (d["per_epoch"][0]["val_auc_bsi"], b["per_epoch"][0]["val_auc_bsi"]),
           (BASELINE_EP1_VAL, BASELINE_EP1_VAL))
    chk.eq(g, "delay_epochs", delay_block.get("delay_epochs"), DELAY_EPOCHS)
    chk.eq(g, "unfreeze_epoch", delay_block.get("unfreeze_epoch"), UNFREEZE_EPOCH)
    chk.eq(g, "alpha_at_construction", delay_block.get("alpha_at_construction"), 0.0)
    chk.eq(g, "alpha_at_unfreeze", delay_block.get("alpha_at_unfreeze"), 0.0)
    chk.check(g, "gen_sha_equal", delay_block.get("generator_sha_at_unfreeze") is not None
              and delay_block.get("generator_sha_at_unfreeze") == delay_block.get("generator_sha_at_construction"))
    chk.eq(g, "unfrozen_once", delay_block.get("unfrozen_once"), True)

    # ---- DD 重推 + 记录一致性 ----
    g = "dd_recompute"
    dd = recompute_dd(d_report)
    recorded = d.get("rp_arm") or {}
    for gate_id, ok in dd.items():
        chk.check(g, f"{gate_id}_observed", ok)
        chk.eq(g, f"{gate_id}_recorded_consistent", bool((recorded.get(gate_id) or {}).get("pass")), ok)

    # ---- 判定重推与比对 ----
    g = "verdict_recompute"
    delta_d = d["test_auc_bsi"] - b["test_auc_bsi"]
    delta_val_d = d["best_val_auc_bsi"] - b["best_val_auc_bsi"]
    all_ok = all(dd.values())
    invalid = None if all_ok else "MECHANISM_FAIL"
    v, sr = verdict_tree(invalid_subreason=invalid, delta_d=delta_d, delta_val_d=delta_val_d)
    chk.eq(g, "delta_d", compare["components"]["delta_d"], delta_d)
    chk.eq(g, "delta_val_d", compare["components"]["delta_val_d"], delta_val_d)
    chk.eq(g, "verdict", compare["verdict"]["verdict"], v)
    chk.eq(g, "subreason", compare["verdict"]["subreason"], sr)
    chk.eq(g, "half_gap_met", compare["descriptive"]["half_gap_met"], bool(delta_d >= HALF_GAP_TARGET))
    chk.eq(g, "secondary", compare["secondary"]["class"], secondary_class(delta_d))

    # ---- 对照链重推 ----
    g = "comparator"
    hist = {rid: load_json(runs / rid / "metrics.json")
            for rid in (BASELINE_RUN_ID, LEARNABLE_RUN_ID, SHUF_HIST_RUN_ID, PINNED_RUN_ID,
                        PINNED_SHUFFLED_RUN_ID)}
    chk.eq(g, "delta_l", hist[LEARNABLE_RUN_ID]["test_auc_bsi"] - hist[BASELINE_RUN_ID]["test_auc_bsi"],
           DELTA_TEST_LEARNABLE)
    chk.eq(g, "delta_p", hist[PINNED_RUN_ID]["test_auc_bsi"] - hist[BASELINE_RUN_ID]["test_auc_bsi"],
           DELTA_TEST_PINNED)
    chk.eq(g, "gap", DELTA_TEST_LEARNABLE - DELTA_TEST_PINNED, GAP)
    chk.eq(g, "half_gap_target", DELTA_TEST_PINNED + 0.5 * GAP, HALF_GAP_TARGET)
    chk.eq(g, "learnable_prompt_sha", sha256_file(runs / LEARNABLE_RUN_ID / "prompt_report.json"),
           LEARNABLE_PROMPT_REPORT_SHA)
    chk.eq(g, "learnable_alpha_final",
           load_json(runs / LEARNABLE_RUN_ID / "prompt_report.json")["alpha_final"],
           LEARNABLE_ALPHA_FINAL)
    for run_id, want_sha in LEARNABLE_TRAJECTORY_RUNS:
        path = runs / run_id / "prompt_report.json"
        probe = load_json(path)["grad_probe"]
        chk.check(g, f"trajectory:{run_id[-10:]}",
                  sha256_file(path) == want_sha and probe[0]["alpha"] == 0.0
                  and probe[0]["generator_grad_norm"] == 0.0 and probe[1]["alpha"] != 0.0
                  and probe[1]["generator_grad_norm"] > 0.0)

    # ---- checkpoint 独立读取 ----
    g = "checkpoint"
    state = torch.load(d_dir / "newtask.pt", map_location="cpu")
    b_state = torch.load(b_dir / "newtask.pt", map_location="cpu")
    chk.eq(g, "d_prompt_gate", float(state["prompt_gate"]), d_report["alpha_final"])
    chk.eq(g, "d_extra_keys_vs_b", sorted(set(state) - set(b_state)), sorted(EXPECTED_PROMPT_KEYS))
    chk.check(g, "d_state_prompt_keys", all(k in state for k in EXPECTED_PROMPT_KEYS))

    # ---- 三件 JSON 互洽 + SUMMARY ----
    g = "docs_consistency"
    gate_doc = load_json(d_dir / "gate_report.json")
    config_doc = load_json(d_dir / "config.json")
    chk.eq(g, "run_ids", (d["run_id"], gate_doc["run_id"], config_doc["run_id"], d_report["run_id"]),
           (d["run_id"],) * 4)
    chk.eq(g, "stage1_ids", (d["stage1_id"], d_report["stage1_id"]), (STAGE1_ID, STAGE1_ID))
    chk.eq(g, "arm_class", (gate_doc.get("residual_prompt") or {}).get("classification"),
           recorded.get("classification"))
    # 臂块形状（DD 门禁集；variant 身份由 identity/metrics/config 独立校验，臂块本身不含 variant）
    chk.eq(g, "arm_block_keys", sorted((gate_doc.get("residual_prompt") or {}).keys()),
           sorted(["M0"] + [f"DD{i}" for i in range(1, 10)]
                  + ["U", "protocol_10_1", "classification", "subreason", "pass"]))
    chk.eq(g, "arm_block_subreason", (gate_doc.get("residual_prompt") or {}).get("subreason"),
           recorded.get("subreason"))
    chk.eq(g, "variant_config", config_doc.get("variant"), "residual-prompt-delay")
    summary = (REPO / "artifacts" / "aliccp_bench" / "SUMMARY.md").read_text(encoding="utf-8")
    rows = [line for line in summary.splitlines() if args.baseline_run in line or args.delay_run in line]
    chk.eq(g, "summary_rows", len(rows), 2)

    # ---- 实现文件 LF sha 钉死（漂移检测） ----
    g = "file_pins"
    for rel, want in FROZEN_LF_SHA.items():
        if str(want).startswith("__FILL"):
            chk.check(g, f"lf_sha:{rel}", False, "未填写钉死值（C2 冻结后必须填写）")
            continue
        chk.eq(g, f"lf_sha:{rel}", lf_sha(rel), want)

    result = {"script": "verify_rp_delay.py", "read_only": True,
              "baseline_run": args.baseline_run, "delay_run": args.delay_run,
              "summary": chk.summary(), "checks": chk.checks}
    out_path = d_dir / "verify_report.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    s = result["summary"]
    print(f"verify_rp_delay: {s['passed']}/{s['total']} passed")
    for c in chk.failed:
        print(f"  FAIL [{c['group']}] {c['name']}: {c['detail']}")
    print(f"report: {out_path} sha256={sha256_file(out_path)}")
    return 0 if not chk.failed else 1


if __name__ == "__main__":
    sys.exit(main())
