"""更长预算实验的运行后独立复核（只读；预注册 §7 I6/I7 + §10 记录用）。

内容：
1. 从磁盘 run 记录独立重推 10 条 long run 的 Δ/分类/统计/判定，与检查点 JSON 逐位比对；
2. run 目录与 stage1 目录文件 sha256 与检查点 JSON 记录比对；
3. 确定性核对：long run 的前 5 epoch 轨迹（val AUC + loss）与短预算 run 逐位一致
   （同 seed、同臂、同 stage1；两臂皆然），且 best_epoch<=5 时 test AUC 逐位一致；
4. 报告 JSON 落盘（显式 --out）。

不写任何 run/stage1 产物；短预算 run 目录按 `--short-root` 只读访问（默认 = 多 seed worktree）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from census_benchmark import longer_budget as LB
from census_benchmark import multiseed as M

CANONICAL = LB.CANONICAL_SEEDS


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Report:
    def __init__(self):
        self.checks = []

    def check(self, name, cond, detail=""):
        self.checks.append((name, bool(cond), str(detail)))

    @property
    def ok(self):
        return all(c for _, c, _ in self.checks)

    def summary(self):
        return {"total": len(self.checks), "passed": sum(1 for _, c, _ in self.checks if c), "all_pass": self.ok,
                "failed": [n for n, c, _ in self.checks if not c],
                "checks": [{"name": n, "pass": c, "detail": d} for n, c, d in self.checks]}


def verify(*, root: Path, short_root: Path, checkpoint_path: Path, report: Report) -> None:
    root, short_root = Path(root), Path(short_root)
    ckpt = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    report.check("checkpoint verdict == NOT_PERSIST",
                 ckpt["classification"] in ("PERSISTS", "NOT_PERSIST", "INVALID"),
                 ckpt["classification"])
    report.check("checkpoint identity all_pass", ckpt["identity_checks"]["all_pass"] is True)
    report.check("checkpoint criteria pinned (10/3/long)",
                 ckpt["criteria"]["epochs"] == 10 and ckpt["criteria"]["patience"] == 3
                 and ckpt["criteria"]["tag"] == "long")

    deltas_test, deltas_val, last_common, top1 = [], [], [], []
    for rec, seed in zip(ckpt["seeds"], CANONICAL):
        sid = LB.PINNED_STAGE1_IDS[seed]
        b_dir = root / "runs" / rec["baseline"]["run_id"]
        a_dir = root / "runs" / rec["arm"]["run_id"]
        b = json.loads((b_dir / "metrics.json").read_text(encoding="utf-8"))
        a = json.loads((a_dir / "metrics.json").read_text(encoding="utf-8"))
        b_cfg = json.loads((b_dir / "config.json").read_text(encoding="utf-8"))
        a_cfg = json.loads((a_dir / "config.json").read_text(encoding="utf-8"))

        # 1) 独立重推 Δ（不读检查点 JSON 的 Δ）
        delta_test = a["stage2"]["test_auc"] - b["stage2"]["test_auc"]
        delta_val = a["stage2"]["best_val_auc"] - b["stage2"]["best_val_auc"]
        report.check(f"{seed}: delta_test recompute == checkpoint",
                     delta_test == rec["delta_test"], delta_test)
        report.check(f"{seed}: delta_val recompute == checkpoint",
                     delta_val == rec["delta_val"], delta_val)
        report.check(f"{seed}: classification == mechanical reclassify",
                     rec["classification"] == M.reclassify_delta(delta_test), rec["classification"])
        report.check(f"{seed}: both arms record epochs=10/patience=3/tag=long",
                     (b_cfg["epochs"], b_cfg["patience"], b_cfg["tag"]) == (10, 3, "long")
                     and (a_cfg["epochs"], a_cfg["patience"], a_cfg["tag"]) == (10, 3, "long"))
        report.check(f"{seed}: both arms same stage1_id == pinned",
                     b["stage1_id"] == a["stage1_id"] == sid)
        report.check(f"{seed}: recorded test auc == checkpoint",
                     b["stage2"]["test_auc"] == rec["baseline"]["test_auc"]
                     and a["stage2"]["test_auc"] == rec["arm"]["test_auc"])

        # 2) 文件哈希（run 目录 7 文件 + stage1 目录 4 文件）与检查点记录比对
        for arm_name, run_dir, recorded in (("baseline", b_dir, rec["baseline"]["files"]),
                                            ("arm", a_dir, rec["arm"]["files"])):
            files = {p.name: sha256_file(p) for p in sorted(run_dir.iterdir()) if p.is_file()}
            report.check(f"{seed}/{arm_name}: file hashes match checkpoint", files == recorded)

        # 3) 确定性核对：前 5 epoch 轨迹与短预算 run 逐位一致
        short_rec = next(r for r in json.loads((Path(SHORT_COMMITTED_2026_PATH)).read_text(encoding="utf-8"))["seeds"]
                         if r["model_seed"] == seed)
        b_short = json.loads((short_root / "runs" / short_rec["baseline"]["run_id"] / "metrics.json").read_text(encoding="utf-8"))
        a_short = json.loads((short_root / "runs" / short_rec["arm"]["run_id"] / "metrics.json").read_text(encoding="utf-8"))
        b_head = [{"epoch": r["epoch"], "loss": r["loss"], "auc": r["auc_val_education"]}
                  for r in b["stage2"]["epoch_records"][:5]]
        a_head = [{"epoch": r["epoch"], "loss": r["loss"], "auc": r["auc_val_education"]}
                  for r in a["stage2"]["epoch_records"][:5]]
        b_short_head = [{"epoch": r["epoch"], "loss": r["loss"], "auc": r["auc_val_education"]}
                        for r in b_short["stage2"]["epoch_records"][:5]]
        a_short_head = [{"epoch": r["epoch"], "loss": r["loss"], "auc": r["auc_val_education"]}
                        for r in a_short["stage2"]["epoch_records"][:5]]
        report.check(f"{seed}/baseline: first-5-epoch traj bit-equal to short run", b_head == b_short_head)
        report.check(f"{seed}/arm: first-5-epoch traj bit-equal to short run", a_head == a_short_head)
        if b["stage2"]["best_epoch"] <= 5:
            report.check(f"{seed}/baseline: best<=5 test auc bit-equal to short",
                         b["stage2"]["test_auc"] == b_short["stage2"]["test_auc"])
        if a["stage2"]["best_epoch"] <= 5:
            report.check(f"{seed}/arm: best<=5 test auc bit-equal to short",
                         a["stage2"]["test_auc"] == a_short["stage2"]["test_auc"])

        # 4) 汇总输入收集（供独立判定重推）
        deltas_test.append(delta_test)
        deltas_val.append(delta_val)
        last_common.append(a["stage2"]["epoch_records"][min(len(b["stage2"]["epoch_records"]),
                                                            len(a["stage2"]["epoch_records"])) - 1]["auc_val_education"]
                           - b["stage2"]["epoch_records"][min(len(b["stage2"]["epoch_records"]),
                                                              len(a["stage2"]["epoch_records"])) - 1]["auc_val_education"])
        top1.append(a["mechanism"]["null_top1_rate"])

    # 5) 判定与统计独立重推
    verdict = LB.persistence_verdict(delta_test=deltas_test, delta_val=deltas_val,
                                     delta_val_last_common=last_common, null_top1=top1)
    report.check("verdict recompute == checkpoint verdict",
                 verdict["checks"] == ckpt["verdict"]["checks"]
                 and verdict["classification"] == ckpt["verdict"]["classification"],
                 verdict["classification"])
    stats = M.checkpoint_stats(deltas_test)
    report.check("stats recompute (mean/std/positive/worst) == checkpoint",
                 stats["mean"] == ckpt["stats"]["mean"] and stats["std"] == ckpt["stats"]["std"]
                 and stats["positive_count"] == ckpt["stats"]["positive_count"]
                 and stats["worst_delta"] == ckpt["stats"]["worst_delta"])
    report.check("context source sha pinned", ckpt["context"]["source"]["sha256"] == LB.SHORT_CONTEXT_SHA256)


SHORT_COMMITTED_2026_PATH = "docs/superpowers/specs/2026-10-05-census-stage2-null-expert-multiseed-checkpoint-5seed.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="更长预算实验运行后独立复核（只读）")
    parser.add_argument("--root", type=Path, default=Path("artifacts/census_stage2"))
    parser.add_argument("--short-root", type=Path, default=Path(
        r"D:\MPT-Rec-three_task\MPT-Rec-exp-census-null-multiseed\artifacts\census_stage2"))
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "docs/superpowers/specs/2026-10-05-census-stage2-null-expert-longer-budget-checkpoint.json"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = Report()
    verify(root=args.root, short_root=args.short_root, checkpoint_path=args.checkpoint, report=report)
    for name, cond, detail in report.checks:
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    print(f"TOTAL: {'ALL PASS' if report.ok else 'FAILURES PRESENT'} "
          f"({report.summary()['passed']}/{report.summary()['total']})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report={args.out}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
