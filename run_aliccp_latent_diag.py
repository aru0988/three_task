"""AliCCP 潜在跨任务诊断入口（预注册：docs/superpowers/specs/2026-10-08-aliccp-latent-cross-task-diagnostic-design.md）。

用法：
  smoke（管线正确性；合成小数据 + 微型 Stage-1；CPU；< 1 分钟；不写正式产物目录）：
    python run_aliccp_latent_diag.py smoke
  formal（每个白名单 seed 唯一一次筛查；要求干净 git 树；先经审阅再跑；预计 4–6 分钟）：
    --model-seed 必填且必须命中冻结白名单（latent_diag.FORMAL_SEED_STAGE1_PAIRS，seeds2–3）；
    省略 --stage1-id 时按白名单派生，显式提供时必须与白名单逐字一致：
    python run_aliccp_latent_diag.py formal --model-seed 1688723740
    python run_aliccp_latent_diag.py verify --run-id <run_id>     # 独立验证，PASS 后才追加 SUMMARY
  verify 预计 2–3 分钟（重扫数据文件 + 全量重算）。

纪律：formal 命令自身绝不写 SUMMARY；SUMMARY 只在 verify 全部 PASS 后追加（同一 run_id 幂等）。
"""
from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import datetime
from pathlib import Path

import torch

from aliccp_benchmark import bench, latent_diag, protocol, verify_latent_diag


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


def _log_path(root: Path, command: str) -> Path:
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{datetime.now():%Y%m%d-%H%M%S}-latdiag-{command}.log"


def _resolve_device(device_arg, *, default_cpu: bool) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if default_cpu:
        return torch.device("cpu")
    return torch.device("cuda:0")


def _cmd_smoke(args) -> int:
    # 每次 smoke 用带时间戳的新子目录：微型 Stage-1 身份确定，协议禁止覆盖旧产物
    root = Path(args.root) / f"{datetime.now():%Y%m%d-%H%M%S}"
    device = _resolve_device(args.device, default_cpu=True)
    sizes = latent_diag.SMOKE_SIZES
    total = sizes["A"] + sizes["B"] + sizes["C"]
    data_dir = root / "smoke_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    latent_diag.write_aliccp_file(data_dir / "train.csv", latent_diag.make_smoke_rows(total, start=0))
    latent_diag.write_aliccp_file(data_dir / "dev.csv", latent_diag.make_smoke_rows(sizes["dev"], start=1000))
    latent_diag.write_aliccp_file(data_dir / "test.csv", latent_diag.make_smoke_rows(sizes["test"], start=2000))
    data_files = {
        "train": str(data_dir / "train.csv"),
        "val": str(data_dir / "dev.csv"),
        "test": str(data_dir / "test.csv"),
    }
    prefix_tag = latent_diag.prefix_tag_for(total, sizes["dev"], sizes["test"])
    log_path = _log_path(root, "smoke")
    with open(log_path, "w", encoding="utf-8") as log_file, contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
        print(f"command=smoke root={root} device={device} sizes={sizes} prefix_tag={prefix_tag}")
        meta = bench.run_stage1(
            root=root, data_files=data_files,
            budgets={"train": total, "val": sizes["dev"], "test": sizes["test"]},
            prefix_tag=prefix_tag, model_seed=protocol.MODEL_SEED, env_seed=protocol.ENV_SEED,
            epochs=1, patience=1, device=device,
        )
        print(f"[smoke] 微型 Stage-1：{meta['stage1_id']}")
        result = latent_diag.run_diagnostic(
            root=root, stage1_id=meta["stage1_id"], data_files=data_files,
            a_rows=sizes["A"], b_rows=sizes["B"], c_rows=sizes["C"],
            val_budget=sizes["dev"], test_budget=sizes["test"], model_seed=protocol.MODEL_SEED,
            device=device, enforce=False, tag="latdiag-smoke", batch_size=8, epochs=1, patience=1,
            log=print,
        )
        print(f"[smoke] 诊断 status={result['status']} verdict={result['verdict']} "
              f"classification={result['classification']} wall={result['wall_seconds']}s")
        if result["recorded_reasons"]:
            print(f"[smoke] 记录（不中止）的正式中止原因：{[r['code'] for r in result['recorded_reasons']]}")
        verdict = verify_latent_diag.verify_run(
            Path(result["run_dir"]), root, append_summary=False, log=print
        )
        print(f"[smoke] 独立验证 overall={verdict['overall']} "
              f"failed={[c['id'] for c in verdict['checks'] if not c['ok']]}")
        print(f"[smoke] run_dir={result['run_dir']}")
    print(f"日志已写入: {log_path}")
    print("smoke 结论：验证 %s；正式运行请用 formal 子命令（先审阅后执行）" % verdict["overall"])
    return 0 if verdict["overall"] == "PASS" else 1


def _cmd_formal(args) -> int:
    root = Path(args.root)
    try:
        # 冻结白名单配对解析：seed 必填命中；显式 stage1_id 必须逐字一致；绝不静默回退 seed1
        stage1_id = latent_diag.resolve_formal_stage1(args.model_seed, args.stage1_id)
    except ValueError as exc:
        raise SystemExit(f"[formal] {exc}")
    device = torch.device(args.device) if args.device else torch.device(f"cuda:{args.gpu}")
    log_path = _log_path(root, "formal")
    with open(log_path, "w", encoding="utf-8") as log_file, contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
        print(
            f"command=formal root={root} device={device} stage1_id={stage1_id} "
            f"model_seed={args.model_seed} A/B/C={args.a_rows}/{args.b_rows}/{args.c_rows} "
            f"val={args.val_budget} test={args.test_budget} batch={args.batch_size} "
            f"epochs={args.epochs} patience={args.patience}"
        )
        result = latent_diag.run_diagnostic(
            root=root, stage1_id=stage1_id, data_files=protocol.DATA_FILES,
            a_rows=args.a_rows, b_rows=args.b_rows, c_rows=args.c_rows,
            val_budget=args.val_budget, test_budget=args.test_budget, model_seed=args.model_seed,
            device=device, enforce=True, tag=args.tag, batch_size=args.batch_size,
            epochs=args.epochs, patience=args.patience, log=print,
        )
        print(f"[formal] status={result['status']} verdict={result['verdict']} "
              f"classification={result['classification']} wall={result['wall_seconds']}s")
        if result["status"] == "ABORT":
            print(f"[formal] NO-GO 中止原因：{[r['code'] for r in result['abort_reasons']]}")
        if result["aucs"]:
            for split in latent_diag.ALL_SPLITS:
                aucs = result["aucs"][split]
                print(f"[formal] AUC[{split}] base={aucs['base']:.6f} full={aucs['full']:.6f} "
                      f"shuffle={aucs['shuffle']:.6f} delta={result['deltas'][split]:+.6f}")
        print(f"[formal] 下一步（独立验证 + PASS 后追加 SUMMARY）："
              f"python run_aliccp_latent_diag.py verify --run-id {result['run_id']}")
    print(f"日志已写入: {log_path}")
    return 0


def _cmd_verify(args) -> int:
    root = Path(args.root)
    run_dir = protocol.run_dir(root, args.run_id)
    if not run_dir.is_dir():
        raise SystemExit(f"run 目录不存在: {run_dir}")
    if args.append_summary:
        append_summary = True
    elif args.no_append_summary:
        append_summary = False
    else:
        append_summary = None  # 自动：正式运行（enforce）PASS 后追加
    log_path = _log_path(root, "verify")
    with open(log_path, "w", encoding="utf-8") as log_file, contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
        verdict = verify_latent_diag.verify_run(run_dir, root, append_summary=append_summary, log=print)
        failed = [c["id"] for c in verdict["checks"] if not c["ok"]]
        print(f"[verify] overall={verdict['overall']} failed={failed}")
        print(f"[verify] SUMMARY: {verdict['summary']}")
    print(f"日志已写入: {log_path}")
    return 0 if verdict["overall"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AliCCP 潜在跨任务诊断（预注册：docs/superpowers/specs/2026-10-08-aliccp-latent-cross-task-diagnostic-design.md）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="管线烟雾测试（合成小数据 + 微型 Stage-1 + 独立验证；CPU）")
    smoke.add_argument("--root", type=str, default="artifacts/aliccp_latent_diag_smoke")
    smoke.add_argument("--device", type=str, default="cpu")

    formal = sub.add_parser("formal", help="正式筛查运行（需干净 git 树；每个白名单 seed 唯一一次）")
    formal.add_argument("--root", type=str, default=str(protocol.ARTIFACT_ROOT))
    formal.add_argument(
        "--model-seed", type=int, required=True,
        help="必填：必须命中冻结白名单 latent_diag.FORMAL_SEED_STAGE1_PAIRS（无默认值，禁止静默重跑 seed1）",
    )
    formal.add_argument(
        "--stage1-id", type=str, default=None,
        help="省略时按 --model-seed 从冻结白名单派生；显式提供时必须与白名单逐字一致",
    )
    formal.add_argument("--a-rows", type=int, default=latent_diag.A_ROWS)
    formal.add_argument("--b-rows", type=int, default=latent_diag.B_ROWS)
    formal.add_argument("--c-rows", type=int, default=latent_diag.C_ROWS)
    formal.add_argument("--val-budget", type=int, default=protocol.VAL_BUDGET)
    formal.add_argument("--test-budget", type=int, default=protocol.TEST_BUDGET)
    formal.add_argument("--batch-size", type=int, default=protocol.BATCH_SIZE)
    formal.add_argument("--epochs", type=int, default=latent_diag.HEAD_EPOCHS)
    formal.add_argument("--patience", type=int, default=latent_diag.HEAD_PATIENCE)
    formal.add_argument("--gpu", type=int, default=0)
    formal.add_argument("--device", type=str, default=None, help="显式设备（默认 cuda:<gpu>）")
    formal.add_argument("--tag", type=str, default="latdiag")

    verify = sub.add_parser("verify", help="独立验证一个 run（重算全部上报值；PASS 后追加 SUMMARY）")
    verify.add_argument("--root", type=str, default=str(protocol.ARTIFACT_ROOT))
    verify.add_argument("--run-id", type=str, required=True)
    verify.add_argument("--append-summary", action="store_true", help="PASS 后强制追加 SUMMARY")
    verify.add_argument("--no-append-summary", action="store_true", help="即使 PASS 也不追加 SUMMARY")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        return _cmd_smoke(args)
    if args.command == "formal":
        return _cmd_formal(args)
    return _cmd_verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
