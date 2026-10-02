"""AliCCP 阶段 1/2 公平基准入口（协议唯一事实来源：docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md）。

预注册用法（spec 8.2/8.3）：
  smoke（管线正确性，A 类门禁执行，B 类只记录）：
    python run_aliccp_benchmark.py stage1 --tag smoke --train-budget 20000 --val-budget 5000 --test-budget 10000 --epochs 1 --patience 1
    python run_aliccp_benchmark.py stage2 --tag smoke --stage1-id <sid> --train-budget 20000 --val-budget 5000 --test-budget 10000 --epochs 1 --patience 1
  short（唯一一次基线运行，A+B 全判定）：
    python run_aliccp_benchmark.py stage1 --tag short
    python run_aliccp_benchmark.py stage2 --tag short --stage1-id <sid>
  long（更长 Stage-2 预算持久性检验，预注册见
  docs/superpowers/specs/2026-10-03-aliccp-stage2-attenuation-longer-budget-design.md §8）：
    python run_aliccp_benchmark.py stage2 --tag long --epochs 10 --patience 3 --model-seed 1688723740 --stage1-id <sid>
    python run_aliccp_benchmark.py stage2 --tag long --epochs 10 --patience 3 --model-seed 1688723740 --stage1-id <sid> --spec-attenuation 0.6972233730330467
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

from aliccp_benchmark import bench, protocol


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


def _prefix_tag_for(train_budget: int, val_budget: int, test_budget: int) -> str:
    if (train_budget, val_budget, test_budget) == (protocol.TRAIN_BUDGET, protocol.VAL_BUDGET, protocol.TEST_BUDGET):
        return protocol.PREFIX_TAG
    return f"p{train_budget}-v{val_budget}-t{test_budget}"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--root", type=str, default=str(protocol.ARTIFACT_ROOT))
    parser.add_argument("--tag", type=str, default="short", choices=["short", "smoke", "long"])
    parser.add_argument("--train-budget", type=int, default=protocol.TRAIN_BUDGET)
    parser.add_argument("--val-budget", type=int, default=protocol.VAL_BUDGET)
    parser.add_argument("--test-budget", type=int, default=protocol.TEST_BUDGET)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AliCCP 阶段 1/2 公平基准（spec: docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("stage1", help="阶段 1：训练 + 选点 + 保存内容寻址产物")
    _add_common(p1)
    p1.add_argument("--model-seed", type=int, default=protocol.MODEL_SEED)
    p1.add_argument("--env-seed", type=int, default=protocol.ENV_SEED)
    p1.add_argument("--epochs", type=int, default=protocol.STAGE1_EPOCHS)
    p1.add_argument("--patience", type=int, default=protocol.STAGE1_PATIENCE)

    p2 = sub.add_parser("stage2", help="阶段 2：加载固定产物 + 真冻结 + NewTask 头 + 门禁")
    _add_common(p2)
    p2.add_argument("--model-seed", type=int, default=protocol.MODEL_SEED)
    p2.add_argument("--stage1-id", type=str, required=True)
    p2.add_argument("--epochs", type=int, default=protocol.STAGE2_EPOCHS)
    p2.add_argument("--patience", type=int, default=protocol.STAGE2_PATIENCE)
    p2.add_argument("--no-enforce-b", action="store_true", help="只记录 B 类门禁（smoke 默认如此）")
    p2.add_argument("--spec-attenuation", type=float, default=1.0,
                    help="阶段 2 specific 混合固定衰减系数 c∈(0,1]（默认 1.0 = 基线行为；更长预算持久性处理臂）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(f"cuda:{args.gpu}")
    root = Path(args.root)
    budgets = {"train": args.train_budget, "val": args.val_budget, "test": args.test_budget}
    prefix_tag = _prefix_tag_for(args.train_budget, args.val_budget, args.test_budget)
    now = datetime.now()
    commit = protocol.code_commit()

    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{now:%Y%m%d-%H%M%S}-{args.command}-{args.tag}.log"

    run_id = None
    if args.command == "stage2":
        run_id = bench.stage2_run_id(
            protocol.make_run_id(now, prefix_tag=prefix_tag, model_seed=args.model_seed, tag=args.tag, commit=commit),
            args.spec_attenuation,
        )
        run_path = protocol.run_dir(root, run_id)
        if run_path.exists():
            raise SystemExit(f"run 目录已存在，禁止覆盖（换一分钟重跑或清理旧 run）：{run_path}")

    with open(log_path, "w", encoding="utf-8") as log_file, contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
        print(f"command={args.command} tag={args.tag} prefix_tag={prefix_tag} budgets={budgets}")
        print(f"model_seed={args.model_seed} device={device} commit={commit} root={root}")
        if args.command == "stage1":
            meta = bench.run_stage1(
                root=root, data_files=protocol.DATA_FILES, budgets=budgets, prefix_tag=prefix_tag,
                model_seed=args.model_seed, env_seed=args.env_seed,
                epochs=args.epochs, patience=args.patience, device=device,
            )
            print(f"stage1_id={meta['stage1_id']}")
            print(f"summary: best_epoch={meta['best_epoch']} test_auc_ctr={meta['test_auc_ctr']:.4f} "
                  f"test_auc_cvr={meta['test_auc_cvr']:.4f} env_acc={meta['env_acc']:.4f} "
                  f"cluster_events={len(meta['cluster_events'])} wall={meta['wall_seconds']}s")
        else:
            result = bench.run_stage2(
                root=root, stage1_id=args.stage1_id, data_files=protocol.DATA_FILES, budgets=budgets,
                prefix_tag=prefix_tag, model_seed=args.model_seed,
                epochs=args.epochs, patience=args.patience, tag=args.tag, device=device,
                enforce_b=(args.tag != "smoke") and not args.no_enforce_b, run_id=run_id,
                spec_attenuation=args.spec_attenuation,
            )
            verdicts = {gate: value["verdict"] for gate, value in result["gates"].items()}
            print(f"gates: {json.dumps(verdicts, ensure_ascii=False)}")
            print(f"run_id={result['run_id']} hard_pass={result['hard_pass']}")
            if args.spec_attenuation != 1.0:
                probe = result["metrics"]["probe"]
                print(f"probe: pred_mean={probe['pred_mean']:.10f} pred_std={probe['pred_std']:.10f} "
                      f"source_gate_mean={probe['source_gate_mean']} "
                      f"trainable_params={result['metrics']['trainable_params']}")
    print(f"日志已写入: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
