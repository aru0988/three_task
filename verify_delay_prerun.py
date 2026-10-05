"""AliCCP 阶段 2 残差 Prompt 延迟解冻学习动力学消融——**预注册前完整性核验**（只读；先于预注册提交）。

唯一事实来源：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-delayed-unfreeze-dynamics-design.md
（§1.5）。本脚本在**任何实现、任何 run 之前**核验：固定 Stage-1 产物（逐文件 sha256 + 内容寻址重算）、
钉死参照头、五个历史 run 链（逐文件 sha256 + 记录值逐位）、六条学习臂 α 轨迹（延迟定界规则的全部
证据）、延迟定界规则复算（k=1 / UNFREEZE_EPOCH=2）、选项 (A)"α 轨迹精确回放"不可行性的机械证据
（采样粒度与 checkpoint 粒度，取自钉死 runner 的 git 对象）、钉死 blob 清单（端口靶）。

只读原则：不写任何 run/stage1/splits 产物、不训练、不重新评测、不导入消融模块（rp_delay 尚不存在于
预注册前；存在性探针记录了该状态）。唯一写入 = 本脚本的报告 JSON + α 轨迹导出（落
artifacts/aliccp_bench/audit/rp-delay/，gitignore）。

用法（cwd 任意；路径锚定本文件所在仓库根）：
    python verify_delay_prerun.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from aliccp_benchmark import protocol as P

REPO = Path(__file__).resolve().parent
ROOT = REPO / "artifacts" / "aliccp_bench"
RUNS = ROOT / "runs"
AUDIT_DIR = ROOT / "audit" / "rp-delay"

F08AE6E = "f08ae6e459ae9fe048daf11eda43799d19ccc7d1"

# ---- §1.5-1/2 Stage-1 产物钉死 ----
STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"
STAGE1_FILES = {
    "backbone.pt": "cd7b033423499e0d36eea988835cea4ac4e2a8e26427a2aea627333822101e0b",
    "env_ids.pt": "4660be5aaa3c59f53dd5b4394f77114a87db69064b32c45517db049a6b9157e7",
    "meta.json": "61a66d81ce3dcf6bae64f4c2b37cf12e722943def646336a588746aba932931d",
}
STAGE1_BACKBONE_SHA = "e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c"
STAGE1_ENV_IDS_SHA = "5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0"
STAGE1_FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
STAGE1_SEED, STAGE1_EPOCHS, STAGE1_ENV_SEED = 1688723740, 3, 20261003
ENV_ID_COUNTS = (1532, 1998468)

# ---- §1.5-3 五个历史 run 链钉死（文件 sha256）----
HIST_RUNS = {
    "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07": {
        "config.json": "1b639caddbc072ad4083eab4f9f06bc5833528f626cb486b20c2be8ed5d60007",
        "gate_report.json": "7e715ffd46b6b8e58149d510cf9e9c790da34c15b7c8159d75f6bb841c7075be",
        "metrics.json": "be437542ddd53c1bb7573f19e777cdc9d0d5ba0798a6d6d7998ee4ce6d034fd7",
        "newtask.pt": "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f",
    },
    "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg": {
        "config.json": "5508364b5383b75f962734bd76ea5bb7e24ce5435d4724d920795ec221d8e702",
        "gate_report.json": "c5d0514f6e75695f585a29730810df682889479083d5c3c384f29b3701757637",
        "metrics.json": "f5619bb0dead85e2ae601e9525ba5a5e4ce9c7302297e02152a2aeda733cdc7f",
        "newtask.pt": "4fdf724c1824ae426246b264339b6e05932682363ceaa7d73dfd6e958182df96",
        "prompt_report.json": "185df4d0f9910508aeff84ed780e7df4fe8ad4fd6b8451ff429a74927d333c1e",
        "verify_report.json": "0e1879ec936e3963e1b8176b0bd380b1ec5384fd3f34b0b118602104ea509ce9",
    },
    "20261005-0933-p2M-v500k-t1M-m1688723740-short-9d26bc8-rpgs": {
        "metrics.json": "6f82c61ca1e2dd5bbeba9c198f9db924442487b71e4531655bf54de876900da8",
        "prompt_report.json": "28636895e9f5ce7be7ce3908aaa80c0e4c7213f633f4087086159f4e1e60c95e",
    },
    "20261005-1029-p2M-v500k-t1M-m1688723740-short-7d26918-rpp": {
        "config.json": "baec4bbc5c22d0033240c0d0bd3eaf7fbc7206014361ca19bacad229c7596a29",
        "gate_report.json": "573ebaf36338140f1c0251cf0497b77fa9e4c632a400365b44aaec16e8d55c89",
        "metrics.json": "f0a884ce6c59e0b37657cec4de3fc012d53539e175571d1ee70aa9d98ced14e9",
        "newtask.pt": "264aedbb9f3f605e5a8e5dd9b2aee8c1945d2dcf0388be722541d2a84f4062b8",
        "prompt_report.json": "77f028993e8b1ea85210ac1b8d08f44698433264c36997bb850e10789d62e64b",
    },
    "20261005-1032-p2M-v500k-t1M-m1688723740-short-1c13841-rpps": {
        "metrics.json": "b1d6925edc259bf58891ecfc4e1dfe494c88074e49f4bf3d721458b2c83e62d3",
    },
}

# 记录值钉死（§1.5-3；逐位）
HIST_RECORDS = {
    "79b5e07": {"test": 0.5974422649550507, "val": 0.5809347091990792, "best_epoch": 5,
                "gate_mean": [0.789178, 0.2108221875],
                "per_epoch_val": [0.4645711559431739, 0.48564481224085004, 0.5142344439088465,
                                  0.5508809596059387, 0.5809347091990792],
                "per_epoch_loss": [0.08193688414408826, 0.05193358083860949, 0.04973645433795173,
                                   0.04873575337347574, 0.04805051837593783]},
    "013e105": {"test": 0.6055453782825184, "val": 0.5895572066556973, "best_epoch": 5,
                "gate_mean": [0.7686165625, 0.23138340625],
                "per_epoch_val": [0.4650886037939688, 0.4904911795680732, 0.5231212372261194,
                                  0.5597976191334388, 0.5895572066556973],
                "variant": "residual-prompt", "classification": "VALID_POSITIVE",
                "alpha_final": 0.07101669907569885},
    "9d26bc8": {"test": 0.5989210260551207, "val": 0.5819965357883198, "best_epoch": 5,
                "variant": "residual-prompt-shuffled", "classification": "MECHANISM_FAIL"},
    "7d26918": {"test": 0.5947650925865714, "val": 0.5798886656441761, "best_epoch": 5,
                "gate_mean": [0.775494625, 0.224505265625],
                "per_epoch_val": [0.46759930720014714, 0.49016156687227064, 0.5156547237007145,
                                  0.5480714746135289, 0.5798886656441761],
                "variant": "residual-prompt-pinned", "classification": "VALID_NEGATIVE",
                "alpha_final": 0.07101669907569885},
    "1c13841": {"test": 0.5966263705980125, "val": 0.5817369266195509, "best_epoch": 5,
                "variant": "residual-prompt-pinned-shuffled", "classification": "VALID_NEGATIVE",
                "alpha_final": 0.07101669907569885},
}

# ---- §1.5-4 参照头钉死 ----
REF_HEAD_SHA = "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f"
REF_HEAD = RUNS / "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07" / "newtask.pt"

# ---- §1.5-5 三常量与半程门（由记录逐位导出）----
DELTA_TEST_LEARNABLE = 0.008103113327467715
DELTA_TEST_PINNED = -0.002677172368479308
GAP = 0.010780285695947023
HALF_GAP_TARGET = 0.0027129704794942035
BASELINE_EP1_TRAIN_LOSS = 0.08193688414408826
BASELINE_EP1_VAL = 0.4645711559431739

# ---- §2.2 六条学习臂 α 轨迹钉死（prompt_report 文件 sha256 + 逐 epoch 探针 α）----
LEARNABLE_TRAJECTORIES = {
    "013e105-rpg": {
        "run": "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg", "budget": "short",
        "prompt_report_sha": "185df4d0f9910508aeff84ed780e7df4fe8ad4fd6b8451ff429a74927d333c1e",
        "alphas": [0.0, 0.016439981758594513, 0.037462275475263596, 0.050921376794576645,
                   0.06157483905553818], "alpha_final": 0.07101669907569885},
    "f2ccec2-rpg": {
        "run": "20261004-0431-p2M-v500k-t1M-m1688723740-long-f2ccec2-rpg", "budget": "long",
        "prompt_report_sha": "ba1d41a777de4ec7713132a5946b994b871c4f4b1682f02e152ad23ff83d7bb8",
        "metrics_sha": "9b225f6e943f82e9af14641c5251ad3892381febb00bb598d104258a692eb513",
        "alphas": [0.0, 0.016439981758594513, 0.037462275475263596, 0.050921376794576645,
                   0.06157483905553818, 0.07101669907569885, 0.07947666943073273,
                   0.08645110577344894, 0.09248566627502441, 0.09788396954536438],
        "alpha_final": 0.1025848239660263},
    "eafc336-rpg": {
        "run": "20261005-0829-p2M-v500k-t1M-m1688723740-xlong-eafc336-rpg", "budget": "xlong",
        "prompt_report_sha": "d5a15cc479bdff605087fe8c5cda11556cd8ede5489bc36996026bcb70c0879c",
        "metrics_sha": "d32a0fbff15d6bb560d64721f738604beb93b9f17b671364bcd4485308c35615",
        "alpha_final": 0.12710827589035034},
    "c17b100-rpg.s3": {
        "run": "20261005-0642-p2M-v500k-t1M-m1688738016-short-c17b100-rpg", "budget": "short",
        "prompt_report_sha": "daeb0d560b3c9c3bc8d9507579d43cdb6c75db317d31464b662e2dcaa511b5de",
        "metrics_sha": "d69f5092eeee96efc959e4520cffbbf208dd2c1b912518bf542cf8ab58fc9259",
        "alphas": [0.0, -0.046634916216135025, -0.04634027183055878, -0.04700079560279846,
                   -0.04794004559516907],
        "alpha_final": -0.04903412237763405},
    "c17b100-rpg.s4": {
        "run": "20261005-0646-p2M-v500k-t1M-m1688749593-short-c17b100-rpg", "budget": "short",
        "prompt_report_sha": "bc55349f07c26ec90078d43eaee5eb78388aa17396deaf970eb90dafdb28b51e",
        "metrics_sha": "2dcb008fa2a73528861d0f46bad772c49af66bbfeb75ffdd8093512a7d59aba0",
        "alphas": [0.0, 0.019704077392816544, 0.023464083671569824, 0.032033320516347885,
                   0.04252833500504494],
        "alpha_final": 0.053119830787181854},
    "c17b100-rpg.s5": {
        "run": "20261005-0649-p2M-v500k-t1M-m1688762746-short-c17b100-rpg", "budget": "short",
        "prompt_report_sha": "69cc23b070a22666594b09fcd30897da69a4b6246961cf15b02998148629d565",
        "metrics_sha": "96ce5a9a7966f7ad3a8530fa77254a1933bf986fd0cee59eb841ac77054f9619",
        "alphas": [0.0, -0.026758631691336632, -0.02457469142973423, -0.02771887741982937,
                   -0.03476540744304657],
        "alpha_final": -0.04552163928747177},
}

# ---- §1.5-8 钉死 blob 清单（端口靶；git 对象）----
PINNED_BLOBS = {
    "aliccp_benchmark/residual_prompt.py": (
        "674213f619c5d5039811c71242a7727118348daf",
        "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc"),
    "aliccp_benchmark/rp_pinned.py": (
        "432fa9bb6b3bb9d537753599499b693197e8c898",
        "ba23bf473634ff647b482ebf6a92f27ca7f073432bb4ed7a1342b8b11101df9b"),
    "aliccp_benchmark/bench.py": ("b676a7976f9c33cb249fec8789fcbe9b155ca7d1", None),
    "run_aliccp_benchmark.py": ("74aaaf05defc650ed80cff00e3f52ff31125c568", None),
    "aliccp_benchmark/tests/test_residual_prompt.py": (
        "2b86d39d8afa3f65fc974bc5bfa482c81e070ff5", None),
}

# ---- §2.2 延迟定界规则（机械、唯一）----
DELAY_EPOCHS = 1
UNFREEZE_EPOCH = 2


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Checker:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, group: str, name: str, ok: bool, detail=None) -> bool:
        self.checks.append({"group": group, "name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def eq(self, group: str, name: str, actual, expected) -> bool:
        return self.check(group, name, actual == expected,
                          {"actual": actual, "expected": expected})

    @property
    def failed(self) -> list[dict]:
        return [c for c in self.checks if not c["ok"]]

    def summary(self) -> dict:
        return {"total": len(self.checks), "passed": len(self.checks) - len(self.failed),
                "failed": len(self.failed)}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, encoding="utf-8").strip()


def check_stage1(chk: Checker) -> dict:
    g = "1/2_stage1"
    s1 = ROOT / "stage1" / STAGE1_ID
    for fname, want in STAGE1_FILES.items():
        path = s1 / fname
        chk.check(g, f"file_sha:{fname}", path.is_file() and sha256_file(path) == want,
                  {"sha256": sha256_file(path) if path.is_file() else None})
    meta = json.loads((s1 / "meta.json").read_text(encoding="utf-8"))
    chk.eq(g, "stage1_id", meta["stage1_id"], STAGE1_ID)
    cfg_sha = P.config_hash(meta["config"])
    chk.eq(g, "config_hash_recompute", cfg_sha, meta["config_hash"])
    chk.eq(g, "stage1_id_recompute",
           P.make_stage1_id(meta["fingerprint_sha256"], meta["model_seed"], meta["epochs"], cfg_sha),
           STAGE1_ID)
    chk.eq(g, "backbone_sha_recorded", meta["backbone_sha256"], STAGE1_BACKBONE_SHA)
    chk.eq(g, "env_ids_sha_recorded", meta["env_ids_sha256"], STAGE1_ENV_IDS_SHA)
    chk.eq(g, "fingerprint_recorded", meta["fingerprint_sha256"], STAGE1_FINGERPRINT_SHA)
    chk.eq(g, "seed/epochs/env_seed",
           (meta["model_seed"], meta["epochs"], meta["env_seed"]),
           (STAGE1_SEED, STAGE1_EPOCHS, STAGE1_ENV_SEED))
    # 张量重算
    backbone_state = torch.load(s1 / "backbone.pt", map_location="cpu")
    digest = hashlib.sha256()
    for key in sorted(backbone_state):
        if key == "env_indices":                       # MPTRec 唯一 persistent buffer（模型零改动）
            continue
        digest.update(key.encode())
        digest.update(P.sha256_tensor(backbone_state[key]).encode())
    chk.eq(g, "backbone_sha_recompute", digest.hexdigest(), STAGE1_BACKBONE_SHA)
    env_ids = torch.load(s1 / "env_ids.pt", map_location="cpu")
    chk.eq(g, "env_ids_sha_recompute", P.sha256_tensor(env_ids), STAGE1_ENV_IDS_SHA)
    counts = tuple(int(v) for v in torch.bincount(env_ids, minlength=2))
    chk.eq(g, "env_ids_shape_counts", (tuple(env_ids.shape), counts),
           ((2000000,), ENV_ID_COUNTS))
    fp = json.loads((ROOT / "splits" / "p2M-v500k-t1M" / "prefix_fingerprint.json")
                    .read_text(encoding="utf-8"))
    chk.eq(g, "fingerprint_file_recompute", fp["fingerprint_sha256"], STAGE1_FINGERPRINT_SHA)
    return {"stage1_id": STAGE1_ID}


def check_historical_runs(chk: Checker) -> dict:
    g = "3_hist_runs"
    records = {}
    for run, files in HIST_RUNS.items():
        d = RUNS / run
        for fname, want in files.items():
            path = d / fname
            chk.check(g, f"{run[-20:]}:{fname}", path.is_file() and sha256_file(path) == want,
                      {"sha256": sha256_file(path) if path.is_file() else None})
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        records[run] = metrics
        short = metrics["commit"]
        want = HIST_RECORDS[short]
        chk.eq(g, f"{short}:test", metrics["test_auc_bsi"], want["test"])
        chk.eq(g, f"{short}:val", metrics["best_val_auc_bsi"], want["val"])
        chk.eq(g, f"{short}:best_epoch", metrics["best_epoch"], want["best_epoch"])
        chk.eq(g, f"{short}:git_dirty", metrics["git"]["dirty"], False)
        chk.eq(g, f"{short}:variant", metrics.get("variant"), want.get("variant"))
        if "classification" in want:
            chk.eq(g, f"{short}:classification", metrics["rp_arm"]["classification"],
                   want.get("classification"))
        if "gate_mean" in want:
            chk.eq(g, f"{short}:gate_mean", metrics["gate_mean"], want["gate_mean"])
        if "per_epoch_val" in want:
            chk.eq(g, f"{short}:per_epoch_val",
                   [e["val_auc_bsi"] for e in metrics["per_epoch"]], want["per_epoch_val"])
        if "per_epoch_loss" in want:
            chk.eq(g, f"{short}:per_epoch_loss",
                   [e["train_loss"] for e in metrics["per_epoch"]], want["per_epoch_loss"])
        if "alpha_final" in want:
            report = json.loads((d / "prompt_report.json").read_text(encoding="utf-8"))
            chk.eq(g, f"{short}:alpha_final", report["alpha_final"], want["alpha_final"])
    # L 的 prompt_report alpha_final 精确来源
    l_report = json.loads((RUNS / "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg"
                           / "prompt_report.json").read_text(encoding="utf-8"))
    chk.eq(g, "L:alpha_final_exact", l_report["alpha_final"],
           HIST_RECORDS["013e105"]["alpha_final"])
    # 三常量逐位重推
    b_test = records["20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07"]["test_auc_bsi"]
    l_test = records["20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg"]["test_auc_bsi"]
    p_test = records["20261005-1029-p2M-v500k-t1M-m1688723740-short-7d26918-rpp"]["test_auc_bsi"]
    chk.eq(g, "delta_L_recompute", l_test - b_test, DELTA_TEST_LEARNABLE)
    chk.eq(g, "delta_P_recompute", p_test - b_test, DELTA_TEST_PINNED)
    chk.eq(g, "gap_recompute", (l_test - b_test) - (p_test - b_test), GAP)
    chk.eq(g, "half_gap_target_recompute", (p_test - b_test) + 0.5 * GAP, HALF_GAP_TARGET)
    b_ep1 = records["20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07"]["per_epoch"][0]
    chk.eq(g, "baseline_ep1_train_loss", b_ep1["train_loss"], BASELINE_EP1_TRAIN_LOSS)
    chk.eq(g, "baseline_ep1_val", b_ep1["val_auc_bsi"], BASELINE_EP1_VAL)
    chk.eq(g, "gap_equals_half_gap_target_x2",
           GAP, 2 * (HALF_GAP_TARGET - DELTA_TEST_PINNED))
    return records


def check_reference_head(chk: Checker) -> None:
    g = "4_ref_head"
    chk.check(g, "file_sha", REF_HEAD.is_file() and sha256_file(REF_HEAD) == REF_HEAD_SHA,
              {"sha256": sha256_file(REF_HEAD) if REF_HEAD.is_file() else None})
    from multitaskrec.model import NewTask
    state = torch.load(REF_HEAD, map_location="cpu")
    try:
        with torch.random.fork_rng():
            head = NewTask(input_size=P.INPUT_SIZE, rep_dim=P.NEWTASK_REP_DIM,
                           tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN,
                           device=torch.device("cpu"))
            head.load_state_dict(state)
        chk.check(g, "strict_load", True)
    except Exception as exc:  # noqa: BLE001
        chk.check(g, "strict_load", False, {"error": str(exc)})


def check_trajectories(chk: Checker) -> dict:
    g = "6/7_alpha_trajectory"
    exported = {}
    for tag, spec in LEARNABLE_TRAJECTORIES.items():
        d = RUNS / spec["run"]
        report_path = d / "prompt_report.json"
        chk.check(g, f"{tag}:prompt_report_sha",
                  report_path.is_file() and sha256_file(report_path) == spec["prompt_report_sha"],
                  {"sha256": sha256_file(report_path) if report_path.is_file() else None})
        if "metrics_sha" in spec:
            metrics_path = d / "metrics.json"
            chk.eq(g, f"{tag}:metrics_sha", sha256_file(metrics_path), spec["metrics_sha"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        probe = report["grad_probe"]
        alphas = [p["alpha"] for p in probe]
        gen_grads = [p["generator_grad_norm"] for p in probe]
        exported[tag] = {"run": spec["run"], "budget": spec["budget"], "alpha_probes": alphas,
                         "alpha_grad_probes": [p["alpha_grad_norm"] for p in probe],
                         "generator_grad_probes": gen_grads,
                         "alpha_final": report["alpha_final"]}
        # §2.2 定界证据：epoch-1 探针 α == 0.0 精确 ∧ 生成器梯度 == 0.0 精确
        chk.eq(g, f"{tag}:probe1_alpha_exact_zero", alphas[0], 0.0)
        chk.eq(g, f"{tag}:probe1_gen_grad_exact_zero", gen_grads[0], 0.0)
        # epoch-2 探针 α != 0 ∧ 生成器梯度 > 0
        chk.check(g, f"{tag}:probe2_alpha_nonzero", alphas[1] != 0.0, {"alpha": alphas[1]})
        chk.check(g, f"{tag}:probe2_gen_grad_positive", gen_grads[1] > 0.0,
                  {"gen_grad": gen_grads[1]})
        chk.eq(g, f"{tag}:alpha_final", report["alpha_final"], spec["alpha_final"])
        if "alphas" in spec:
            chk.eq(g, f"{tag}:probes_match_pin", alphas, spec["alphas"])
        # 采样粒度（选项 A 不可行性证据）：探针数 == epochs（1 点/epoch + 终值）
        chk.eq(g, f"{tag}:probe_count_equals_epochs", len(alphas),
               int(json.loads((d / "metrics.json").read_text(encoding="utf-8"))["epochs"]))
    return exported


def check_boundary_rule(chk: Checker, exported: dict) -> dict:
    g = "8_boundary_rule"
    # 规则复算：probe(ep1) 全部精确为 0（六条学习臂）⇒ 离开 0 发生在 epoch 1 ⇒ k=1
    all_zero = all(v["alpha_probes"][0] == 0.0 for v in exported.values())
    all_depart = all(v["alpha_probes"][1] != 0.0 for v in exported.values())
    chk.check(g, "universal_probe1_exact_zero", all_zero)
    chk.check(g, "universal_departure_by_epoch2", all_depart)
    chk.eq(g, "delay_epochs", DELAY_EPOCHS, 1)
    chk.eq(g, "unfreeze_epoch", UNFREEZE_EPOCH, DELAY_EPOCHS + 1)
    # 边界比例（seed2 short；描述）
    a = exported["013e105-rpg"]["alpha_probes"]
    final = exported["013e105-rpg"]["alpha_final"]
    fractions = [x / final for x in a[1:]]
    chk.check(g, "seed2_boundary_fractions",
              all(0.0 < f < 1.0 for f in fractions), {"fractions": fractions})
    # 选项 A 不可行性：钉死 runner 只做首 batch 探针、只存 best checkpoint（git 对象机械核验）
    bench_blob = _git("show", f"{F08AE6E}:aliccp_benchmark/bench.py")
    rp_blob = _git("show", f"{F08AE6E}:aliccp_benchmark/residual_prompt.py")
    chk.eq(g, "runner_has_single_torch_save", bench_blob.count("torch.save("), 1)
    chk.check(g, "probe_is_first_batch_only", "if step != 0:" in rp_blob
              and "return" in rp_blob.split("if step != 0:")[1][:40])
    chk.check(g, "no_per_epoch_checkpoint", "for epoch in range(1, epochs + 1):" in bench_blob
              and bench_blob.count("best_weight = copy.deepcopy") == 1)
    return {"delay_epochs": DELAY_EPOCHS, "unfreeze_epoch": UNFREEZE_EPOCH,
            "rule": "k = 学习臂 α 仍精确等于构造值 0.0 的 epoch 起始探针数（六条学习臂一致：k=1）",
            "seed2_boundary_fractions": fractions}


def check_pinned_blobs(chk: Checker) -> None:
    g = "9_pinned_blobs"
    for path, (blob, lf_sha) in PINNED_BLOBS.items():
        actual = _git("rev-parse", f"{F08AE6E}:{path}")
        chk.eq(g, f"blob:{path}", actual, blob)
        if lf_sha:
            data = subprocess.check_output(["git", "cat-file", "blob", blob], cwd=REPO)
            chk.eq(g, f"lf_sha:{path}", hashlib.sha256(data).hexdigest(), lf_sha)


def check_import_purity(chk: Checker) -> dict:
    g = "10_import_purity"
    state_before = torch.get_rng_state()
    try:
        import aliccp_benchmark.rp_delay  # noqa: F401
        present = True
    except ModuleNotFoundError:
        present = False
    state_after = torch.get_rng_state()
    chk.check(g, "rng_endpoint_unchanged", torch.equal(state_before, state_after))
    chk.check(g, "import_status_recorded", True, {"rp_delay_present": present})
    return {"rp_delay_present_at_prerun": present,
            "note": "预注册前 rp_delay 尚不存在属预期（C2 提交实现；存在性由 C2 后单元测试与复核覆盖）"}


def check_dataset(chk: Checker) -> None:
    g = "11_dataset"
    for split, rel in P.DATA_FILES.items():
        path = REPO / rel
        chk.check(g, f"{split}_exists_readable", path.is_file() and path.stat().st_size > 0,
                  {"path": str(path), "size": path.stat().st_size if path.is_file() else None})


def main() -> int:
    chk = Checker()
    check_stage1(chk)
    records = check_historical_runs(chk)
    check_reference_head(chk)
    exported = check_trajectories(chk)
    boundary = check_boundary_rule(chk, exported)
    check_pinned_blobs(chk)
    purity = check_import_purity(chk)
    check_dataset(chk)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "alpha_trajectories.json").write_text(
        json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "script": "verify_delay_prerun.py",
        "design_doc": "docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-delayed-unfreeze-dynamics-design.md",
        "read_only": True,
        "summary": chk.summary(),
        "boundary_rule": boundary,
        "import_purity": purity,
        "checks": chk.checks,
    }
    out = AUDIT_DIR / "verify_delay_prerun_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    s = result["summary"]
    print(f"verify_delay_prerun: {s['passed']}/{s['total']} passed")
    for c in chk.failed:
        print(f"  FAIL [{c['group']}] {c['name']}: {c['detail']}")
    print(f"report: {out} sha256={sha256_file(out)}")
    return 0 if not chk.failed else 1


if __name__ == "__main__":
    sys.exit(main())
