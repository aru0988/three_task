"""预注册前 seed2 产物完整性核验（只读；本分支 exp/aliccp-stage2-residual-prompt-longer-budget）。

用途：residual-prompt 更长预算持久性检验的预注册 §1.5 前置核验。对象 = 固定 Stage-1 产物
s1-5c060b9c-m1688723740-e3-4e1b5c6f 与残差 prompt 参照头 checkpoint
（run 20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt）。

**不写任何 run/stage1 产物、不训练、不重新评测**；唯一输出为
artifacts/aliccp_bench/audit/longer-budget-rp/verify_seed2_artifact_integrity_result.json（audit 目录）。

核验内容（全部独立重算，不 import 机制模块与 bench）：
  1. 三个产物文件字节 sha256 == 钉死值（backbone.pt / env_ids.pt / meta.json）；
  2. meta 内容寻址：stage1_id 重算 == 记录 == 期望；config_hash 重算；id 分量（fp/config 前缀、seed、epochs）；
  3. 前缀指纹 A2 级独立复算：自哈希 + 前缀字节 sha256 ×3 + 表头 sha256 + 文件大小 + 原始扫描标签计数 ×3；
  4. env_ids 张量独立重算 sha256 == meta == 钉死；形状 (2000000,) 且取值 ∈{0,1}；计数 1532/1998468；
  5. backbone 张量独立重算 sha256（排除 buffer env_indices）== meta == 钉死；
  6. 参照头 checkpoint 文件 sha256 == 钉死 + NewTask(80/64/[32,32]) strict 载入成功。

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

# 钉死值（与预注册文档 / 前序 21/21 核验一致；看到任何结果前写死）
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

    ok_all = all(c["pass"] for c in checks)
    out_dir = ART / "audit" / "longer-budget-rp"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "script": Path(__file__).name,
        "stage1_id": SID,
        "reference_head": str(ref_path),
        "checks": checks,
        "all_pass": ok_all,
        "n_checks": len(checks),
    }
    (out_dir / "verify_seed2_artifact_integrity_result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nALL_PASS = {ok_all}  ({len(checks)} checks; "
          f"report -> {out_dir / 'verify_seed2_artifact_integrity_result.json'})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
