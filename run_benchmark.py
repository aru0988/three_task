"""CensusIncome / AliCCP 统一阶段 1 / 阶段 2 基准入口（替代各自独立脚本）。

用法（各数据集保持既有产物路径与字段名：census → artifacts/census_stage2；aliccp → artifacts/aliccp_bench）：
  census：python run_benchmark.py census stage1 [--root ...] [--gpu 0] [--tag short|full]
                         [--epochs ...] [--split-seed ...] [--model-seed ...] [--env-seed ...]
          python run_benchmark.py census stage2 --stage1-dir <root>/<stage1_id 目录> [--tag ...] [--epochs ...]
  aliccp smoke（管线正确性，A 类门禁执行，B 类只记录）：
    python run_benchmark.py aliccp stage1 --tag smoke --train-budget 20000 --val-budget 5000 \
           --test-budget 10000 --epochs 1 --patience 1
    python run_benchmark.py aliccp stage2 --tag smoke --stage1-id <sid> --train-budget 20000 \
           --val-budget 5000 --test-budget 10000 --epochs 1 --patience 1
  aliccp short（唯一一次基线运行，A+B 全判定）：
    python run_benchmark.py aliccp stage1 --tag short
    python run_benchmark.py aliccp stage2 --tag short --stage1-id <sid>

全部 stdout 落盘 root/logs/{ts}-{dataset}-{command}-{tag}.log（tqdm 进度条走 stderr，不入日志）。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

from benchmark.datasets import aliccp, census
from benchmark.pipeline import run_stage1, run_stage2
from benchmark.protocol import code_commit, make_run_id


class _Tee:
    """stdout 同时写控制台与日志文件（tqdm 进度条走 stderr，不入日志）。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def resolve_enforce_b(tag: str, no_enforce_b: bool) -> bool:
    """aliccp：smoke 只记录 B 类门禁；其余 tag 默认 enforce（--no-enforce-b 可显式关闭）。"""
    return (tag != "smoke") and not no_enforce_b


def _resolve_device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def _add_common(parser: argparse.ArgumentParser, default_root: Path) -> None:
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--gpu", type=int, default=0)


def _add_aliccp_data_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tag", type=str, default="short", choices=["short", "smoke"])
    parser.add_argument("--train-budget", type=int, default=aliccp.TRAIN_BUDGET)
    parser.add_argument("--val-budget", type=int, default=aliccp.VAL_BUDGET)
    parser.add_argument("--test-budget", type=int, default=aliccp.TEST_BUDGET)
    parser.add_argument("--model-seed", type=int, default=aliccp.MODEL_SEED)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CensusIncome / AliCCP 阶段 1/2 统一基准入口")
    datasets = parser.add_subparsers(dest="dataset", required=True)

    p_census = datasets.add_parser("census", help="CensusIncome 协议（产物 artifacts/census_stage2）")
    census_sub = p_census.add_subparsers(dest="command", required=True)
    c1 = census_sub.add_parser("stage1", help="阶段 1：训练 + 选点 + 保存内容寻址产物")
    _add_common(c1, census.ARTIFACT_ROOT)
    c1.add_argument("--tag", type=str, default="short", choices=["short", "full"])
    c1.add_argument("--epochs", type=int, default=census.STAGE1_EPOCHS)
    c1.add_argument("--split-seed", type=int, default=census.SPLIT_SEED)
    c1.add_argument("--model-seed", type=int, default=census.MODEL_SEED)
    c1.add_argument("--env-seed", type=int, default=census.ENV_SEED)
    c2 = census_sub.add_parser("stage2", help="阶段 2：加载固定 Stage-1 产物，只训练新任务头")
    c2.add_argument("--stage1-dir", type=Path, required=True)      # 唯一引用方式
    _add_common(c2, census.ARTIFACT_ROOT)
    c2.add_argument("--tag", type=str, default="short", choices=["short", "full"])
    c2.add_argument("--epochs", type=int, default=census.STAGE2_EPOCHS)

    p_aliccp = datasets.add_parser("aliccp", help="AliCCP 协议（产物 artifacts/aliccp_bench）")
    aliccp_sub = p_aliccp.add_subparsers(dest="command", required=True)
    a1 = aliccp_sub.add_parser("stage1", help="阶段 1：训练 + 选点 + 保存内容寻址产物")
    _add_common(a1, aliccp.ARTIFACT_ROOT)
    _add_aliccp_data_opts(a1)
    a1.add_argument("--env-seed", type=int, default=aliccp.ENV_SEED)
    a1.add_argument("--epochs", type=int, default=aliccp.STAGE1_EPOCHS)
    a1.add_argument("--patience", type=int, default=aliccp.STAGE1_PATIENCE)
    a2 = aliccp_sub.add_parser("stage2", help="阶段 2：加载固定产物 + 真冻结 + NewTask 头 + 门禁")
    a2.add_argument("--stage1-id", type=str, required=True)
    _add_common(a2, aliccp.ARTIFACT_ROOT)
    _add_aliccp_data_opts(a2)
    a2.add_argument("--epochs", type=int, default=aliccp.STAGE2_EPOCHS)
    a2.add_argument("--patience", type=int, default=aliccp.STAGE2_PATIENCE)
    a2.add_argument("--no-enforce-b", action="store_true", help="只记录 B 类门禁（smoke 默认如此）")
    return parser


def _run_census(args, root: Path, device: torch.device) -> None:
    if args.command == "stage1":
        profile = census.build_profile(split_seed=args.split_seed, model_seed=args.model_seed,
                                       env_seed=args.env_seed)
        out = run_stage1(profile, root, epochs=args.epochs, device=device)
        print(f"[cli] stage1_dir={out['dir']}")
    else:
        profile = census.build_profile()          # split/model seed 以 stage1 meta 为准
        out = run_stage2(profile, root, stage1_dir=args.stage1_dir, epochs=args.epochs,
                         tag=args.tag, device=device)
        print(f"[cli] run_dir={out['run_dir']}")


def _run_aliccp(args, root: Path, device: torch.device, now: datetime, commit: str) -> None:
    budgets = {"train": args.train_budget, "val": args.val_budget, "test": args.test_budget}
    prefix_tag = aliccp.prefix_tag_for(**budgets)
    enforce_b = resolve_enforce_b(args.tag, getattr(args, "no_enforce_b", False))
    profile = aliccp.build_profile(prefix_tag=prefix_tag, budgets=budgets, model_seed=args.model_seed,
                                   stage1_patience=args.patience, stage2_patience=args.patience,
                                   enforce_b=enforce_b)
    print(f"command={args.command} tag={args.tag} prefix_tag={prefix_tag} budgets={budgets}")
    print(f"model_seed={args.model_seed} device={device} commit={commit} root={root}")
    if args.command == "stage1":
        out = run_stage1(profile, root, epochs=args.epochs, device=device)
        meta = out["meta"]
        print(f"stage1_id={out['stage1_id']}")
        print(f"summary: best_epoch={meta['best_epoch']} test_auc_ctr={meta['test_auc_ctr']:.4f} "
              f"test_auc_cvr={meta['test_auc_cvr']:.4f} env_acc={meta['env_acc']:.4f} "
              f"cluster_events={len(meta['cluster_events'])} wall={meta['wall_seconds']}s")
    else:
        run_id = make_run_id(now, prefix=prefix_tag, model_seed=args.model_seed,
                             tag=args.tag, commit=commit)
        run_path = root / "runs" / run_id
        if run_path.exists():
            raise SystemExit(f"run 目录已存在，禁止覆盖（换一分钟重跑或清理旧 run）：{run_path}")
        result = run_stage2(profile, root, stage1_dir=root / "stage1" / args.stage1_id,
                            epochs=args.epochs, tag=args.tag, device=device, now=now)
        verdicts = {gate: value["verdict"] for gate, value in result["gates"].items()}
        print(f"gates: {json.dumps(verdicts, ensure_ascii=False)}")
        print(f"run_id={result['run_id']} hard_pass={result['hard_pass']}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    device = _resolve_device(args.gpu)
    root = args.root
    now, commit = datetime.now(), code_commit()

    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{now:%Y%m%d-%H%M%S}-{args.dataset}-{args.command}-{args.tag}.log"

    with open(log_path, "w", encoding="utf-8") as log_file, \
            contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
        print(f"[cli] dataset={args.dataset} command={args.command} device={device} "
              f"root={root} commit={commit}")
        if args.dataset == "census":
            _run_census(args, root, device)
        else:
            _run_aliccp(args, root, device, now, commit)
    print(f"日志已写入: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
