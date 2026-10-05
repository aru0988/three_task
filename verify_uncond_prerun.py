"""预注册前 seed2 产物 + 对照链 + 无条件常量向量完整性核验（只读；本分支 exp/aliccp-stage2-residual-prompt-unconditional-control）。

用途：无条件固定条件对照消融预注册 §1.5 的前置核验。对象 = 固定 Stage-1 产物
s1-5c060b9c-m1688723740-e3-4e1b5c6f、残差 prompt 参照头 checkpoint（run
20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt）、五个历史 run
（79b5e07 基线 / 013e105-rpg 学习 correct 臂 / 9d26bc8-rpgs 学习 shuffled 臂 /
7d26918-rpp 钉死 α correct 臂【本实验 C_p 臂的跨实验重放靶】/ 1c13841-rpps 钉死 α shuffled 臂）
的逐文件字节与记录值、以及 §2.1 无条件常量向量 c 的确定性推导（独立重实现）。

移植自 `f08ae6e:verify_pinned_prerun.py`（alpha-pinned 分支；其 §1.5 已在该分支 89/89
通过）；相对该版适配：模块 docstring / 输出目录（audit/rp-uncond）/ 第 7 组扩展为五个对照 run
（新增 7d26918-rpp、1c13841-rpps 两组文件 sha256 + 记录值钉死）/ 第 9 组改为**无条件常量向量
推导钉死**（原 deck 组移除——本实验无错排臂）/ 新增第 10 组（跨实验重放靶与三常量重推：
7d26918-rpp 的 checkpoint/prompt_report 文件 sha + G_c/G_s/GAP 逐位重推）。钉死值（第 1–6 组）不变。

**不写任何 run/stage1 产物、不训练、不重新评测**；唯一输出为
artifacts/aliccp_bench/audit/rp-uncond/verify_uncond_prerun_result.json（audit 目录）。

核验内容（独立重算，不 import 机制模块与 bench）：
  1. 三个产物文件字节 sha256 == 钉死值（backbone.pt / env_ids.pt / meta.json）；
  2. meta 内容寻址：stage1_id 重算 == 记录 == 期望；config_hash 重算；id 分量；
  3. 前缀指纹 A2 级独立复算：自哈希 + 前缀字节 sha256 ×3 + 表头 sha256 + 文件大小 + 原始扫描标签计数 ×3；
  4. env_ids 张量独立重算 sha256 == meta == 钉死；形状 (2000000,) 且取值 ∈{0,1}；计数 1532/1998468；
  5. backbone 张量独立重算 sha256（排除 buffer env_indices）== meta == 钉死；
  6. 参照头 checkpoint 文件 sha256 == 钉死 + NewTask(80/64/[32,32]) strict 载入成功；
  7. 对照 run 完整性（五 run）：79b5e07（4 文件）、013e105-rpg（6 文件）、9d26bc8-rpgs（8 文件）、
     7d26918-rpp（5 文件）、1c13841-rpps（7 文件）逐文件 sha256 == 钉死；记录值钉死：五 run 的
     test/val/逐 epoch val/gate_mean/best_epoch/epochs/patience/seed；013e105-rpg
     variant/rp_arm.classification/VALID_POSITIVE；9d26bc8-rpgs MECHANISM_FAIL；
     7d26918-rpp variant=="residual-prompt-pinned"、VALID_NEGATIVE、α==钉死；1c13841-rpps
     variant=="residual-prompt-pinned-shuffled"、VALID_NEGATIVE；
  8. α 钉死来源：013e105-rpg/prompt_report.json sha256 == 钉死 ∧ alpha_final == PINNED_ALPHA（精确）
     ∧ float32 往返精确；
  9. 无条件常量向量：以 UNCOND_CONST_SEED=20261006 独立重实现 torch.Generator+randn(80) →
     sha256 == 冻结值 ∧ numel==80 ∧ float32 ∧ 全有限 ∧ norm>0 ∧ 全局 RNG 隔离（推导前后端点不变）；
  10. 跨实验重放靶与三常量（预注册 §5.1-REP_Cp/CC 的钉死来源）：7d26918-rpp newtask.pt sha256 ==
      264aedbb… ∧ prompt_report.json sha256 == 77f02899…；由五 run 记录值重推
      G_c==−0.002677172368479308 ∧ G_s==−0.000815894357038216 ∧ GAP==−0.001861278011441092（逐位）。

退出码 0 = 全部一致；1 = 有不一致（逐项打印；不一致即停止，不重训、不替换）。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts" / "aliccp_bench"

SID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"
REF_RUN = "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07"
CORRECT_RUN = "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg"
SHUF_HIST_RUN = "20261005-0933-p2M-v500k-t1M-m1688723740-short-9d26bc8-rpgs"
PINNED_CORRECT_RUN = "20261005-1029-p2M-v500k-t1M-m1688723740-short-7d26918-rpp"
PINNED_SHUFFLED_RUN = "20261005-1032-p2M-v500k-t1M-m1688723740-short-1c13841-rpps"

# 钉死值（与预注册文档 / 前序核验一致；看到任何结果前写死）
FILE_SHAS = {
    "backbone.pt": "cd7b033423499e0d36eea988835cea4ac4e2a8e26427a2aea627333822101e0b",
    "env_ids.pt": "4660be5aaa3c59f53dd5b4394f77114a87db69064b32c45517db049a6b9157e7",
    "meta.json": "61a66d81ce3dcf6bae64f4c2b37cf12e722943def646336a588746aba932931d",
}
FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
CONFIG_HASH = "4e1b5c6ffe9b6ff49cda34691de27390669396337e64e867f0f83b6ec4da7981"
BACKBONE_SHA = "e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c"
ENV_IDS_SHA = "5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0"
ENV_COUNTS = (1532, 1998468)
REF_HEAD_SHA = "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f"
BUDGETS = {"train": 2_000_000, "val": 500_000, "test": 1_000_000}
NON_PARAM_STATE_KEYS = {"env_indices"}          # MPTRec 的持久 buffer（backbone_sha 只覆盖参数）

# ---- 第 7 组：对照 run 钉死（预注册 §1.5；文件 sha256 + 记录值）----
BASELINE_RUN_FILE_SHAS = {
    "config.json": "1b639caddbc072ad4083eab4f9f06bc5833528f626cb486b20c2be8ed5d60007",
    "gate_report.json": "7e715ffd46b6b8e58149d510cf9e9c790da34c15b7c8159d75f6bb841c7075be",
    "metrics.json": "be437542ddd53c1bb7573f19e777cdc9d0d5ba0798a6d6d7998ee4ce6d034fd7",
    "newtask.pt": REF_HEAD_SHA,
}
CORRECT_RUN_FILE_SHAS = {
    "config.json": "5508364b5383b75f962734bd76ea5bb7e24ce5435d4724d920795ec221d8e702",
    "gate_report.json": "c5d0514f6e75695f585a29730810df682889479083d5c3c384f29b3701757637",
    "metrics.json": "f5619bb0dead85e2ae601e9525ba5a5e4ce9c7302297e02152a2aeda733cdc7f",
    "newtask.pt": "4fdf724c1824ae426246b264339b6e05932682363ceaa7d73dfd6e958182df96",
    "prompt_report.json": "185df4d0f9910508aeff84ed780e7df4fe8ad4fd6b8451ff429a74927d333c1e",
    "verify_report.json": "0e1879ec936e3963e1b8176b0bd380b1ec5384fd3f34b0b118602104ea509ce9",
}
SHUF_HIST_RUN_FILE_SHAS = {
    "config.json": "b44a50445611adcda3e42da28f8f2758c620d99aa0ae098becd53acb0f59cfe2",
    "gate_report.json": "0206c97d09f32ac7b377a1c855d3336ca2c6e104d358a1d18fee8bf3dc32216a",
    "metrics.json": "6f82c61ca1e2dd5bbeba9c198f9db924442487b71e4531655bf54de876900da8",
    "newtask.pt": "9e22e9dd5a0a8bfa1e623b731a6f37e0e68319eb6b8300bcf3ec2fcd2b0e68b5",
    "prompt_report.json": "28636895e9f5ce7be7ce3908aaa80c0e4c7213f633f4087086159f4e1e60c95e",
    "rp_shuffled_compare.json": "74f3f4d79ad53cbfa3cc7a59180667777062625b2a3d0e9e095842c1726cee39",
    "verify_report.json": "94ea3506deb01d7e0c636db47c8ab3896e038c399c83c9f333761bc6bfed61e2",
    "verify_report_firstpass_exit1.json": "b55b3fcfed3b31021c47d295e74c3652f1aa1dadc70de32ff3e83d77ae5e9b28",
}
PINNED_CORRECT_RUN_FILE_SHAS = {
    "config.json": "baec4bbc5c22d0033240c0d0bd3eaf7fbc7206014361ca19bacad229c7596a29",
    "gate_report.json": "573ebaf36338140f1c0251cf0497b77fa9e4c632a400365b44aaec16e8d55c89",
    "metrics.json": "f0a884ce6c59e0b37657cec4de3fc012d53539e175571d1ee70aa9d98ced14e9",
    "newtask.pt": "264aedbb9f3f605e5a8e5dd9b2aee8c1945d2dcf0388be722541d2a84f4062b8",
    "prompt_report.json": "77f028993e8b1ea85210ac1b8d08f44698433264c36997bb850e10789d62e64b",
}
PINNED_SHUFFLED_RUN_FILE_SHAS = {
    "config.json": "b6139d74a7004277077ae4b51139f455390935f0a0287536c0f134f714a102e9",
    "gate_report.json": "5f47cbeaec34b6c3da27e6f3a95fbca85865cd00c95ee4f7f323702160d6a6f8",
    "metrics.json": "b1d6925edc259bf58891ecfc4e1dfe494c88074e49f4bf3d721458b2c83e62d3",
    "newtask.pt": "03d14ec4fcb98e72f9c1cd4a13005f33660703082db6f8b54e63459d02da6dbc",
    "prompt_report.json": "a24b66a7befa592887f28567796fd2a8d3e0896c6b7131b0af1b6d7874d01ffd",
    "rp_pinned_compare.json": "01322a72a20f507dddc6969e3a64ea6b8fbf4f67d8e934df66f27bffca841f29",
    "verify_report.json": "d1893916d264b51ba19633ec9e9ccc9a9bf1a5febecfeeae0dbcbb23a381db9e",
}
BASELINE_RECORD = {
    "best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
    "best_val_auc_bsi": 0.5809347091990792, "test_auc_bsi": 0.5974422649550507,
    "gate_mean": [0.789178, 0.2108221875],
    "per_epoch_val": [0.4645711559431739, 0.48564481224085004, 0.5142344439088465,
                      0.5508809596059387, 0.5809347091990792],
    # 79b5e07 早于 variant 字段引入（2026-10-03 原始协议 run）；不设 variant 钉死
}
CORRECT_RECORD = {
    "best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
    "best_val_auc_bsi": 0.5895572066556973, "test_auc_bsi": 0.6055453782825184,
    "gate_mean": [0.7686165625, 0.23138340625],
    "per_epoch_val": [0.4650886037939688, 0.4904911795680732, 0.5231212372261194,
                      0.5597976191334388, 0.5895572066556973],
    "variant": "residual-prompt", "classification": "VALID_POSITIVE",
    "alpha_final": 0.07101669907569885,
}
SHUF_HIST_RECORD = {
    "best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
    "best_val_auc_bsi": 0.5819965357883198, "test_auc_bsi": 0.5989210260551207,
    "gate_mean": [0.7904243125, 0.2095755],
    "per_epoch_val": [0.4646865641709375, 0.48723682075012587, 0.5170954166630655,
                      0.5533860188554449, 0.5819965357883198],
    "variant": "residual-prompt-shuffled", "classification": "MECHANISM_FAIL",
    "alpha_final": 0.010569889098405838,
}
PINNED_CORRECT_RECORD = {
    "best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
    "best_val_auc_bsi": 0.5798886656441761, "test_auc_bsi": 0.5947650925865714,
    "gate_mean": [0.775494625, 0.224505265625],
    "per_epoch_val": [0.46759930720014714, 0.49016156687227064, 0.5156547237007145,
                      0.5480714746135289, 0.5798886656441761],
    "variant": "residual-prompt-pinned", "classification": "VALID_NEGATIVE",
    "alpha_final": 0.07101669907569885,
}
PINNED_SHUFFLED_RECORD = {
    "best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
    "best_val_auc_bsi": 0.5817369266195509, "test_auc_bsi": 0.5966263705980125,
    "gate_mean": [0.7925029375, 0.207497140625],
    "per_epoch_val": [0.4657800526815393, 0.48481854615132286, 0.5134964621264595,
                      0.5511490407183938, 0.5817369266195509],
    "variant": "residual-prompt-pinned-shuffled", "classification": "VALID_NEGATIVE",
    "alpha_final": 0.07101669907569885,
}
DELTA_TEST_CORRECT = 0.008103113327467715      # test_c − test_baseline（记录值重推，逐位）
DELTA_VAL_CORRECT = 0.008622497456618139

# ---- 第 8 组：α 钉死来源（预注册 §1.5；含 float32 往返精确）----
PINNED_ALPHA = 0.07101669907569885
CORRECT_PROMPT_REPORT_SHA = "185df4d0f9910508aeff84ed780e7df4fe8ad4fd6b8451ff429a74927d333c1e"

# ---- 第 9 组：无条件常量向量推导钉死（预注册 §2.1）----
UNCOND_CONST_SEED = 20261006
COND_VECTOR_SHAPE = (80,)
COND_VECTOR_SHA256 = "0d45cc4611cebde9675858f1afc633a91b3e9114d6ddc02899a5d97f34babf58"

# ---- 第 10 组：跨实验重放靶与三常量（预注册 §5.1-REP_Cp/CC）----
PINNED_CORRECT_METRICS_SHA = "f0a884ce6c59e0b37657cec4de3fc012d53539e175571d1ee70aa9d98ced14e9"
PINNED_CORRECT_NEWTASK_SHA = "264aedbb9f3f605e5a8e5dd9b2aee8c1945d2dcf0388be722541d2a84f4062b8"
PINNED_CORRECT_PROMPT_SHA = "77f028993e8b1ea85210ac1b8d08f44698433264c36997bb850e10789d62e64b"
HIST_GAP = -0.001861278011441092
HIST_G_C = -0.002677172368479308
HIST_G_S = -0.000815894357038216

checks: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha256_tensor(tensor: torch.Tensor) -> str:
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str(t.dtype).encode() + str(tuple(t.shape)).encode()
                          + t.numpy().tobytes()).hexdigest()


def draw_constant(input_size: int, seed: int) -> torch.Tensor:
    """§2.1 的独立重实现（不得 import 机制模块）：torch.Generator + 单次 randn。"""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(int(input_size), generator=generator, dtype=torch.float32)


def prefix_sha256(path: Path, n: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.readline())
        for _ in range(n):
            digest.update(handle.readline())
    return digest.hexdigest()


def scan_label_counts(path: Path, n: int) -> dict:
    counts = {"n": 0, "click1": 0, "purchase1": 0, "bsi_pos": 0, "bsi_raw": {}}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for _ in range(n):
            parts = handle.readline().strip().split(",")
            click, purchase, raw = int(parts[0]), int(parts[1]), int(parts[-1])
            counts["n"] += 1
            counts["click1"] += click
            counts["purchase1"] += purchase
            counts["bsi_pos"] += 0 if raw == 2 else 1
            counts["bsi_raw"][str(raw)] = counts["bsi_raw"].get(str(raw), 0) + 1
    return counts


def main() -> int:
    print(f"cwd={Path.cwd()}")
    stage_dir = ART / "stage1" / SID

    # ---------------- 1. 产物文件字节 sha256 ----------------
    for fname, pin in FILE_SHAS.items():
        path = stage_dir / fname
        if not path.is_file():
            check(f"file.{fname}.exists", False, "缺失")
            continue
        actual = sha256_file(path)
        check(f"file.{fname}.sha256", actual == pin, actual)

    # ---------------- 2. meta 内容寻址 ----------------
    meta = json.loads((stage_dir / "meta.json").read_text(encoding="utf-8"))
    sid_recomputed = (f"s1-{meta['fingerprint_sha256'][:8]}-m{meta['model_seed']}"
                      f"-e{meta['epochs']}-{meta['config_hash'][:8]}")
    check("meta.stage1_id_recompute",
          sid_recomputed == meta["stage1_id"] == SID, sid_recomputed)
    cfg_sha = hashlib.sha256(canonical_json(meta["config"]).encode()).hexdigest()
    check("meta.config_hash_recompute", cfg_sha == meta["config_hash"] == CONFIG_HASH, cfg_sha)
    check("meta.id_components",
          meta["fingerprint_sha256"][:8] == "5c060b9c" and meta["config_hash"][:8] == "4e1b5c6f"
          and meta["model_seed"] == 1688723740 and meta["epochs"] == 3
          and meta["env_seed"] == 20261003 and meta["budgets"] == BUDGETS,
          f"fp={meta['fingerprint_sha256'][:8]} cfg={meta['config_hash'][:8]} "
          f"seed={meta['model_seed']} epochs={meta['epochs']}")
    check("meta.fingerprint_field", meta["fingerprint_sha256"] == FINGERPRINT_SHA,
          meta["fingerprint_sha256"])
    check("meta.backbone_sha_field", meta["backbone_sha256"] == BACKBONE_SHA, meta["backbone_sha256"])
    check("meta.env_ids_sha_field", meta["env_ids_sha256"] == ENV_IDS_SHA, meta["env_ids_sha256"])

    # ---------------- 3. 前缀指纹 A2 级独立复算 ----------------
    fp = json.loads((ART / "splits" / "p2M-v500k-t1M" / "prefix_fingerprint.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in fp.items() if k != "fingerprint_sha256"}
    self_sha = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    check("fingerprint.self_hash", self_sha == fp["fingerprint_sha256"] == FINGERPRINT_SHA, self_sha)
    check("fingerprint.budgets", fp["budgets"] == BUDGETS, json.dumps(fp["budgets"]))
    for tag in ("train", "val", "test"):
        rec = fp["files"][tag]
        path = Path(rec["path"])
        if not path.is_file():
            path = ROOT / rec["path"]            # 指纹记录的是原始 worktree 相对路径；本 worktree 经 junction 同源
        size_ok = path.stat().st_size == rec["size_bytes"]
        check(f"fingerprint.{tag}.size_bytes", size_ok, f"{rec['size_bytes']}")
        pfx = prefix_sha256(path, BUDGETS[tag])
        check(f"fingerprint.{tag}.prefix_sha256", pfx == rec["prefix_sha256"], pfx)
        with open(path, "rb") as handle:
            header_sha = hashlib.sha256(handle.readline()).hexdigest()
        check(f"fingerprint.{tag}.header_sha256", header_sha == fp["header_sha256"], header_sha)
        counts = scan_label_counts(path, BUDGETS[tag])
        rec_counts = {**fp["label_counts"][tag], "bsi_raw": {str(k): v for k, v in
                                                             fp["label_counts"][tag]["bsi_raw"].items()}}
        check(f"fingerprint.{tag}.label_counts", counts == rec_counts,
              json.dumps({"n": counts["n"], "click1": counts["click1"],
                          "purchase1": counts["purchase1"], "bsi_pos": counts["bsi_pos"]}))

    # ---------------- 4. env_ids 张量独立复算 ----------------
    env_ids = torch.load(stage_dir / "env_ids.pt", map_location="cpu")
    env_sha = sha256_tensor(env_ids)
    check("env_ids.sha256_recompute",
          env_sha == meta["env_ids_sha256"] == ENV_IDS_SHA, env_sha)
    counts = torch.bincount(env_ids.long(), minlength=2)
    check("env_ids.shape_range_counts",
          tuple(env_ids.shape) == (2_000_000,) and int(env_ids.min()) == 0 and int(env_ids.max()) == 1
          and (int(counts[0]), int(counts[1])) == ENV_COUNTS,
          f"shape={tuple(env_ids.shape)} counts={(int(counts[0]), int(counts[1]))}")

    # ---------------- 5. backbone 张量独立复算 ----------------
    state = torch.load(stage_dir / "backbone.pt", map_location="cpu")
    digest = hashlib.sha256()
    for name in sorted(k for k in state if k not in NON_PARAM_STATE_KEYS):
        digest.update(name.encode())
        digest.update(sha256_tensor(state[name]).encode())
    bb_sha = digest.hexdigest()
    check("backbone.sha256_recompute",
          bb_sha == meta["backbone_sha256"] == BACKBONE_SHA, bb_sha)

    # ---------------- 6. 参照头 checkpoint ----------------
    ref_path = ART / "runs" / REF_RUN / "newtask.pt"
    if not ref_path.is_file():
        check("reference_head.exists", False, str(ref_path))
    else:
        ref_sha = sha256_file(ref_path)
        check("reference_head.sha256", ref_sha == REF_HEAD_SHA, ref_sha)
        from multitaskrec.model import NewTask
        head = NewTask(input_size=80, rep_dim=64, tower_dnn_hidden_units=[32, 32],
                       reg_dnn=7e-6, device=torch.device("cpu"))
        head.load_state_dict(torch.load(ref_path, map_location="cpu"))      # strict=True
        check("reference_head.strict_load", True, "NewTask(80/64/[32,32]) strict=True OK")

    # ---------------- 7. 对照 run 完整性（五 run；文件 sha + 记录值钉死）----------------
    for run_id, pins in ((REF_RUN, BASELINE_RUN_FILE_SHAS), (CORRECT_RUN, CORRECT_RUN_FILE_SHAS),
                         (SHUF_HIST_RUN, SHUF_HIST_RUN_FILE_SHAS),
                         (PINNED_CORRECT_RUN, PINNED_CORRECT_RUN_FILE_SHAS),
                         (PINNED_SHUFFLED_RUN, PINNED_SHUFFLED_RUN_FILE_SHAS)):
        run_dir = ART / "runs" / run_id
        for fname, pin in sorted(pins.items()):
            path = run_dir / fname
            if not path.is_file():
                check(f"comparator.{run_id}.{fname}.exists", False, "缺失")
                continue
            actual = sha256_file(path)
            check(f"comparator.{run_id}.{fname}.sha256", actual == pin, actual)

    docs = {}
    for run_id in (REF_RUN, CORRECT_RUN, SHUF_HIST_RUN, PINNED_CORRECT_RUN, PINNED_SHUFFLED_RUN):
        docs[run_id] = json.loads((ART / "runs" / run_id / "metrics.json").read_text(encoding="utf-8"))
    for label, rec, run_id in (("baseline", BASELINE_RECORD, REF_RUN),
                               ("correct", CORRECT_RECORD, CORRECT_RUN),
                               ("shuf_hist", SHUF_HIST_RECORD, SHUF_HIST_RUN),
                               ("pinned_correct", PINNED_CORRECT_RECORD, PINNED_CORRECT_RUN),
                               ("pinned_shuffled", PINNED_SHUFFLED_RECORD, PINNED_SHUFFLED_RUN)):
        doc = docs[run_id]
        for key in ("best_epoch", "epochs", "patience", "model_seed",
                    "best_val_auc_bsi", "test_auc_bsi", "gate_mean"):
            check(f"comparator.{label}.{key}", doc[key] == rec[key], repr(doc[key]))
        check(f"comparator.{label}.per_epoch_val",
              [e["val_auc_bsi"] for e in doc["per_epoch"]] == rec["per_epoch_val"],
              repr([e["val_auc_bsi"] for e in doc["per_epoch"]][:2]) + "…")
        if rec.get("variant") is not None:
            check(f"comparator.{label}.variant", doc.get("variant") == rec["variant"],
                  repr(doc.get("variant")))
        if rec.get("classification") is not None:
            check(f"comparator.{label}.classification",
                  doc["rp_arm"]["classification"] == rec["classification"],
                  doc["rp_arm"]["classification"])
        check(f"comparator.{label}.run_id", doc["run_id"] == run_id, doc["run_id"])
    base_m = docs[REF_RUN]
    correct_m = docs[CORRECT_RUN]
    for label, run_id in (("correct", CORRECT_RUN), ("pinned_correct", PINNED_CORRECT_RUN),
                          ("pinned_shuffled", PINNED_SHUFFLED_RUN)):
        prompt = json.loads((ART / "runs" / run_id / "prompt_report.json").read_text(encoding="utf-8"))
        check(f"comparator.{label}.alpha_final",
              prompt["alpha_final"] == PINNED_ALPHA, repr(prompt["alpha_final"]))
    check("comparator.correct.delta_test_recompute",
          correct_m["test_auc_bsi"] - base_m["test_auc_bsi"] == DELTA_TEST_CORRECT,
          repr(correct_m["test_auc_bsi"] - base_m["test_auc_bsi"]))
    check("comparator.correct.delta_val_recompute",
          correct_m["best_val_auc_bsi"] - base_m["best_val_auc_bsi"] == DELTA_VAL_CORRECT,
          repr(correct_m["best_val_auc_bsi"] - base_m["best_val_auc_bsi"]))

    # ---------------- 8. α 钉死来源（预注册 §1.5；精确相等 + float32 往返）----------------
    correct_prompt_path = ART / "runs" / CORRECT_RUN / "prompt_report.json"
    correct_p = json.loads(correct_prompt_path.read_text(encoding="utf-8"))
    check("pin.source_file_sha256",
          sha256_file(correct_prompt_path) == CORRECT_PROMPT_REPORT_SHA,
          sha256_file(correct_prompt_path))
    check("pin.alpha_exact_equal",
          correct_p["alpha_final"] == PINNED_ALPHA, repr(correct_p["alpha_final"]))
    rt = float(torch.tensor(PINNED_ALPHA, dtype=torch.float32))
    check("pin.float32_roundtrip_exact", rt == PINNED_ALPHA, repr(rt))

    # ---------------- 9. 无条件常量向量推导钉死（预注册 §2.1）----------------
    rng_before = torch.get_rng_state()
    cond = draw_constant(80, UNCOND_CONST_SEED)
    rng_after = torch.get_rng_state()
    check("uncond_const.rng_isolation", bool(torch.equal(rng_before, rng_after)),
          "推导前后全局 CPU RNG 端点不变")
    cond2 = draw_constant(80, UNCOND_CONST_SEED)
    check("uncond_const.redraw_bit_identical", bool(torch.equal(cond, cond2)), "")
    cond_sha = sha256_tensor(cond)
    check("uncond_const.sha256", cond_sha == COND_VECTOR_SHA256, cond_sha)
    check("uncond_const.shape_dtype",
          tuple(cond.shape) == COND_VECTOR_SHAPE and cond.dtype == torch.float32
          and int(cond.numel()) == 80, f"{tuple(cond.shape)} {cond.dtype}")
    check("uncond_const.finite_norm_positive",
          bool(torch.isfinite(cond).all()) and float(cond.double().norm()) > 0.0,
          f"norm={float(cond.double().norm())!r}")

    # ---------------- 10. 跨实验重放靶与三常量（预注册 §5.1-REP_Cp/CC）----------------
    fc_dir = ART / "runs" / PINNED_CORRECT_RUN
    check("replay_target.metrics_sha",
          sha256_file(fc_dir / "metrics.json") == PINNED_CORRECT_METRICS_SHA,
          sha256_file(fc_dir / "metrics.json"))
    check("replay_target.newtask_sha",
          sha256_file(fc_dir / "newtask.pt") == PINNED_CORRECT_NEWTASK_SHA,
          sha256_file(fc_dir / "newtask.pt"))
    check("replay_target.prompt_report_sha",
          sha256_file(fc_dir / "prompt_report.json") == PINNED_CORRECT_PROMPT_SHA,
          sha256_file(fc_dir / "prompt_report.json"))
    fc_m = docs[PINNED_CORRECT_RUN]
    fs_m = docs[PINNED_SHUFFLED_RUN]
    g_c = fc_m["test_auc_bsi"] - base_m["test_auc_bsi"]
    g_s = fs_m["test_auc_bsi"] - base_m["test_auc_bsi"]
    gap = fc_m["test_auc_bsi"] - fs_m["test_auc_bsi"]
    check("replay_target.g_c_recompute", g_c == HIST_G_C, repr(g_c))
    check("replay_target.g_s_recompute", g_s == HIST_G_S, repr(g_s))
    check("replay_target.gap_recompute", gap == HIST_GAP, repr(gap))

    ok_all = all(c["pass"] for c in checks)
    out_dir = ART / "audit" / "rp-uncond"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_doc = {
        "script": Path(__file__).name,
        "stage1_id": SID,
        "reference_head": str(ref_path),
        "comparator_runs": [REF_RUN, CORRECT_RUN, SHUF_HIST_RUN, PINNED_CORRECT_RUN,
                            PINNED_SHUFFLED_RUN],
        "pinned_alpha": PINNED_ALPHA,
        "uncond_const_seed": UNCOND_CONST_SEED,
        "uncond_const_sha256": COND_VECTOR_SHA256,
        "checks": checks,
        "all_pass": ok_all,
        "n_checks": len(checks),
    }
    (out_dir / "verify_uncond_prerun_result.json").write_text(
        json.dumps(report_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nALL_PASS = {ok_all}  ({len(checks)} checks; "
          f"report -> {out_dir / 'verify_uncond_prerun_result.json'})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
