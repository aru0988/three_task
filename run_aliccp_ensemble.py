"""AliCCP 参数高效集成（方向 3）入口。

spec: docs/superpowers/specs/2026-10-08-aliccp-parameter-efficient-ensemble-design.md
预注册用法（正式短实验 = B/C/E 三臂各一次；不写 SUMMARY、不推送，待独立复核）：

  # 臂 B（原始 NewTask）/ C（+rank-16）/ E（+2×rank-8），须先提交代码使 git.dirty=false
  python run_aliccp_ensemble.py stage2 --arm B --stage1-id <stage1_id> --tag short
  python run_aliccp_ensemble.py stage2 --arm C --stage1-id <stage1_id> --tag short
  python run_aliccp_ensemble.py stage2 --arm E --stage1-id <stage1_id> --tag short

  # 独立复核（从 raw 预测/标签/配置/checkpoint 重算，只打印不写文件）
  python run_aliccp_ensemble.py report --run-dir artifacts/aliccp_bench/runs/<runB> \
      --run-dir artifacts/aliccp_bench/runs/<runC> --run-dir artifacts/aliccp_bench/runs/<runE>

smoke（管线正确性；预算取协议 smoke 值，须与 smoke Stage-1 配套）：
  python run_aliccp_ensemble.py stage2 --arm E --tag smoke --stage1-id <smoke_stage1_id> \
      --train-budget 20000 --val-budget 5000 --test-budget 10000 --epochs 1 --patience 1
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

from aliccp_benchmark import ensemble, protocol
from run_aliccp_benchmark import _Tee, _prefix_tag_for


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AliCCP 参数高效集成三臂（spec: docs/superpowers/specs/2026-10-08-aliccp-parameter-efficient-ensemble-design.md）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p2 = sub.add_parser("stage2", help="阶段 2：加载固定 Stage-1 + 真冻结 + 按臂训练头 + 门禁 + raw 落盘（不写 SUMMARY）")
    p2.add_argument("--arm", type=str, required=True, choices=sorted(ensemble.ARM_RANKS))
    p2.add_argument("--stage1-id", type=str, required=True)
    p2.add_argument("--gpu", type=int, default=0)
    p2.add_argument("--root", type=str, default=str(protocol.ARTIFACT_ROOT))
    p2.add_argument("--tag", type=str, default="short", choices=["short", "smoke"])
    p2.add_argument("--train-budget", type=int, default=protocol.TRAIN_BUDGET)
    p2.add_argument("--val-budget", type=int, default=protocol.VAL_BUDGET)
    p2.add_argument("--test-budget", type=int, default=protocol.TEST_BUDGET)
    p2.add_argument("--model-seed", type=int, default=protocol.MODEL_SEED)
    p2.add_argument("--epochs", type=int, default=protocol.STAGE2_EPOCHS)
    p2.add_argument("--patience", type=int, default=protocol.STAGE2_PATIENCE)

    pr = sub.add_parser("report", help="独立复核：从 raw 预测重算 AUC/val 选点并做三臂配对核验（只打印）")
    pr.add_argument("--run-dir", type=str, action="append", required=True, help="可重复；B/C/E 各一")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "report":
        report = ensemble.compare_arms([Path(p) for p in args.run_dir])
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0

    device = torch.device(f"cuda:{args.gpu}")
    root = Path(args.root)
    budgets = {"train": args.train_budget, "val": args.val_budget, "test": args.test_budget}
    prefix_tag = _prefix_tag_for(args.train_budget, args.val_budget, args.test_budget)
    now = datetime.now()
    commit = protocol.code_commit()
    run_id = ensemble.make_arm_run_id(
        now, prefix_tag=prefix_tag, model_seed=args.model_seed, tag=args.tag, commit=commit, arm=args.arm
    )
    run_path = protocol.run_dir(root, run_id)
    if run_path.exists():
        raise SystemExit(f"run 目录已存在，禁止覆盖：{run_path}")

    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{now:%Y%m%d-%H%M%S}-stage2-{args.tag}-ens-{args.arm}.log"
    with open(log_path, "w", encoding="utf-8") as log_file, contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
        print(f"command=stage2 arm={args.arm} tag={args.tag} prefix_tag={prefix_tag} budgets={budgets}")
        print(f"model_seed={args.model_seed} device={device} commit={commit} root={root}")
        result = ensemble.run_arm(
            root=root, arm=args.arm, stage1_id=args.stage1_id,
            data_files=protocol.DATA_FILES, budgets=budgets, prefix_tag=prefix_tag,
            model_seed=args.model_seed, epochs=args.epochs, patience=args.patience,
            tag=args.tag, device=device, enforce_b=False, require_clean=True, run_id=run_id,
        )
        verdicts = {gate: value["verdict"] for gate, value in result["gates"].items()}
        print(f"gates: {json.dumps(verdicts, ensure_ascii=False)}")
        print(f"run_id={result['run_id']} hard_pass={result['hard_pass']}")
    print(f"日志已写入: {log_path}")
    print(f"run 目录: {run_path}（SUMMARY 未写，待独立复核）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
