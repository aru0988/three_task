"""更长预算实验（``exp/census-stage2-null-expert-longer-budget``）预注册前的**只读**完整性核验。

对象：
1. 五份复用的 Stage-1 产物（内容寻址；选取自 ``exp/census-stage2-null-expert-multiseed`` 已完成
   5-seed 检查点），逐文件 sha256 / config 重算 / stage1_id 重算 / split 指纹重算；
2. 已提交的 5-seed 短预算检查点 JSON（作为短预算 context 的唯一事实来源）字节一致；
3. Census 数据文件（train.gz / test.gz）字节一致。

硬约束（预注册文档 §4/§7）：核验**不通过即停止**——不重训、不替换、不静默换产物；本脚本只读，
不写任何 stage1 / run 产物（报告 JSON 由 ``--out`` 显式指定）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from census_benchmark import protocol as P

# ---- 钉死常量（预注册前写死；来源 = origin/exp/census-stage2-null-expert-multiseed @ 1196acd）----
MULTISEED_TIP = "1196acd04c4e5d2846e0644d6345532f3632d148"
CHECKPOINT_REL = "docs/superpowers/specs/2026-10-05-census-stage2-null-expert-multiseed-checkpoint-5seed.json"
CHECKPOINT_SHA256 = "605196e56b7e92a78a5953205c78b447e4691216593eb1436b1c8ab26b83e2ab"
CHECKPOINT_BLOB_SHA = "074d893a801e118598c416accfc6e94a05b550eb"
DATASET_SHA256 = {"train.gz": "e2f2d57925a8020d23fcd05802284b3bc1d91fcdd451f1eaaefd683e76d1b2dc",
                  "test.gz": "8de690375028ac531de37fcbb117bd6527c5f3542d3bfead8975226e45100cec"}
CANONICAL_SEEDS = (1685480945, 1685463909, 1685477428, 1685459668, 1685496394)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Report:
    def __init__(self):
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        self.checks.append((name, bool(cond), detail))

    @property
    def ok(self) -> bool:
        return all(cond for _, cond, _ in self.checks)

    def summary(self) -> dict:
        return {"total": len(self.checks), "passed": sum(1 for _, c, _ in self.checks if c),
                "all_pass": self.ok,
                "failed": [name for name, c, _ in self.checks if not c],
                "checks": [{"name": n, "pass": c, "detail": d} for n, c, d in self.checks]}


def verify(*, root: Path, checkpoint_path: Path, dataset_dir: Path, report: Report) -> None:
    root = Path(root)

    # 1) 检查点 JSON 字节一致（短预算 context 的唯一事实来源）
    report.check("checkpoint file exists", checkpoint_path.is_file(), str(checkpoint_path))
    report.check("checkpoint sha256 pinned", sha256_file(checkpoint_path) == CHECKPOINT_SHA256,
                 sha256_file(checkpoint_path))
    ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    report.check("checkpoint has 5 seed records", len(ckpt["seeds"]) == 5)
    sids = {rec["model_seed"]: rec["stage1_id"] for rec in ckpt["seeds"]}
    report.check("checkpoint seeds are canonical 5", set(sids) == set(CANONICAL_SEEDS),
                 ",".join(map(str, sorted(sids))))

    # 2) 数据文件
    for fname, want in DATASET_SHA256.items():
        path = Path(dataset_dir) / fname
        report.check(f"dataset {fname} exists", path.is_file())
        report.check(f"dataset {fname} sha256", path.is_file() and sha256_file(path) == want)

    # 3) split 基准落盘与重算一致（A2 级）
    fp_path = P.split_dir(root, P.SPLIT_SEED) / "split_fingerprint.json"
    npz_path = P.split_dir(root, P.SPLIT_SEED) / "split_indices.npz"
    report.check("split_fingerprint.json exists", fp_path.is_file())
    report.check("split_indices.npz exists", npz_path.is_file())

    # 4) 逐 seed 的 stage1 产物
    from run_census_benchmark import build_mptrec          # 延迟导入：核验脚本只读，不触发 CUDA

    for seed in CANONICAL_SEEDS:
        sid = sids[seed]
        sdir = root / "stage1" / sid
        report.check(f"{sid}: dir exists", sdir.is_dir())
        if not sdir.is_dir():
            continue
        files = {p.name: sha256_file(p) for p in sorted(sdir.iterdir()) if p.is_file()}
        recorded = next(r for r in ckpt["seeds"] if r["model_seed"] == seed)["stage1"]["files"]
        for fname, want in recorded.items():
            report.check(f"{sid}: {fname} sha256", files.get(fname) == want, str(files.get(fname)))
        report.check(f"{sid}: file set exact", set(files) == set(recorded),
                     str(set(files) ^ set(recorded)))

        meta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
        cfg = {k: meta[k] for k in ("model_seed", "env_seed", "epochs", "patience", "batch_size", "lr",
                                    "uni_coe", "env_coe", "reg_embedding", "reg_dnn", "input_size",
                                    "embedding_size", "expert_hidden", "tower_hidden", "num_tasks")}
        report.check(f"{sid}: config_hash recompute", P.config_hash(cfg) == meta["config_hash"])
        report.check(f"{sid}: stage1_id content-addressed",
                     P.stage1_id(split_sha=meta["split_fingerprint_sha256"], model_seed=seed,
                                 epochs=meta["epochs"], cfg_sha=P.config_hash(cfg)) == sid)
        report.check(f"{sid}: protocol constants",
                     meta["model_seed"] == seed and meta["env_seed"] == P.ENV_SEED
                     and meta["split_seed"] == P.SPLIT_SEED and meta["epochs"] == P.STAGE1_EPOCHS
                     and meta["patience"] == P.PATIENCE and meta["batch_size"] == P.BATCH_SIZE
                     and meta["lr"] == P.LR and meta["input_size"] == P.INPUT_SIZE
                     and meta["embedding_size"] == P.EMBEDDING_SIZE
                     and meta["expert_hidden"] == list(P.EXPERT_HIDDEN)
                     and meta["tower_hidden"] == list(P.TOWER_HIDDEN))

        state = torch.load(sdir / "backbone.pt", map_location="cpu")
        model = build_mptrec(torch.device("cpu"))
        model.load_state_dict(state, strict=True)
        report.check(f"{sid}: backbone_sha256 recompute", P.backbone_sha256(model) == meta["backbone_sha256"],
                     P.backbone_sha256(model))
        env_ids = torch.load(sdir / "env_ids.pt", map_location="cpu")
        report.check(f"{sid}: env_ids_sha256 recompute", P.sha256_tensor(env_ids) == meta["env_ids_sha256"])
        report.check(f"{sid}: env shape/values",
                     env_ids.dim() == 1 and int(env_ids.min()) >= 0 and int(env_ids.max()) < P.NUM_ENVS
                     and int(env_ids.shape[0]) == meta["n_train"], str(tuple(env_ids.shape)))

        # split 指纹重算（用落盘 indices 与 meta 的 n_train）
        if npz_path.is_file():
            data = np.load(npz_path)
            stats = P.split_stats(data["val"], data["test"], n_train=meta["n_train"])
            fp = P.split_fingerprint(split_seed=P.SPLIT_SEED, stats=stats,
                                     val_idx=data["val"], test_idx=data["test"])
            report.check(f"{sid}: split fingerprint recompute",
                         fp["fingerprint_sha256"] == meta["split_fingerprint_sha256"]
                         and stats["disjoint"] and stats["union_complete"]
                         and stats["n_train"] == 199523 and stats["n_val"] == 49881 and stats["n_test"] == 49881,
                         json.dumps(stats))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="更长预算实验预注册前完整性核验（只读）")
    parser.add_argument("--root", type=Path, default=P.ARTIFACT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=Path(CHECKPOINT_REL))
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/Census-income"))
    parser.add_argument("--out", type=Path, default=None, help="报告 JSON 落盘路径（缺省只打印）")
    args = parser.parse_args(argv)

    report = Report()
    verify(root=args.root, checkpoint_path=args.checkpoint, dataset_dir=args.dataset_dir, report=report)
    payload = {"multiseed_tip": MULTISEED_TIP, "checkpoint_blob_sha": CHECKPOINT_BLOB_SHA,
               "root": str(args.root), **report.summary()}
    for name, cond, detail in report.checks:
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    print(f"TOTAL: {'ALL PASS' if report.ok else 'FAILURES PRESENT'} ({payload['passed']}/{payload['total']})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report={args.out}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
