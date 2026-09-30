"""CensusIncome 阶段 1 梯度冲突审计入口（**诊断分支；零训练语义改动**）。

用法（人工审阅后自行运行；本分支不代跑）：

    python run_census_gradient_audit.py                    # 正式审计：2 epoch + 固定采样节奏
    python run_census_gradient_audit.py --max-batches 5    # smoke：只验链路，不产生门禁判定

定性：梯度手术（PCGrad/GradNorm/CAGrad 等）是**既有文献**。本脚本只做审计，
不改变训练更新、不主张任何性能提升。阈值预注册于
`docs/superpowers/specs/2026-09-30-census-stage1-gradient-conflict-audit-prereg.md`。
"""
from __future__ import annotations

import argparse
import io
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from census_benchmark import gradient_audit as GA
from census_benchmark import protocol as P
from run_census_benchmark import build_census_loaders, build_mptrec, tee_stdout


def truncate_loader(loader, max_batches):
    """smoke 用：只保留**前** max_batches 个 batch。

    保序截断 → 父类 `env_ids[batch_size*step : batch_size*(step+1)]` 的切片仍然对齐；
    batch_size 不随截断改变，故 alpha 日程与正式审计同形。
    """
    if max_batches is None:
        return loader
    size = max_batches * loader.batch_size
    indices = list(range(min(size, len(loader.dataset))))
    return DataLoader(Subset(loader.dataset, indices), batch_size=loader.batch_size)


def run_audit(root=GA.ARTIFACT_ROOT, *, split_seed=P.SPLIT_SEED, model_seed=P.MODEL_SEED,
              env_seed=P.ENV_SEED, epochs=P.STAGE1_EPOCHS, device=None, max_batches=None,
              now=None, model=None, loaders=None, stats=None, indices=None) -> dict:
    """跑一次带插桩的阶段 1，落盘审计产物并给出（或抑制）门禁判定。

    与 `run_census_benchmark.run_stage1` 同 seed / 同划分 / 同超参 / 同 batch 顺序；
    差别只有：多了一个只读采样点，且**不保存** backbone（本分支不产出模型）。
    """
    root = Path(root)
    device = device or torch.device("cuda:0")
    commit, log_buffer = P.code_commit(), io.StringIO()

    # 1) 划分：只由 split seed 决定；首次落盘基准，之后必须一致
    if loaders is None:
        loaders, stats, indices = build_census_loaders(split_seed)
    val_idx, test_idx = indices
    fp = P.split_fingerprint(split_seed=split_seed, stats=stats, val_idx=val_idx, test_idx=test_idx)
    baseline = P.load_split_fingerprint(root, split_seed)
    if baseline is None:
        P.write_split_fingerprint(root, split_seed, fp)
        P.save_split_indices(root, split_seed, val_idx, test_idx)
    elif not P.verify_split_fingerprint(fp, baseline):
        raise RuntimeError("划分指纹与基准不一致（A2 失败），协议禁止继续")

    # 2) 初始环境分配：独立 env seed，长度 = 完整训练集长度
    env_ids = P.make_env_ids(len(loaders["train"].dataset), env_seed)

    # 3) 带插桩的训练：模型构建/初始化顺序与 run_stage1 完全一致
    P.seed_model(model_seed)
    model = model or build_mptrec(device)
    model.to(device)
    train_loader = truncate_loader(loaders["train"], max_batches)
    # smoke 截断只取训练集**前缀**（保序、batch_size 不变），env_ids 必须同步取同一前缀：
    # 否则 epoch2 的 `cluster_2` 会按截断 loader 重聚类出长度 max_batches*batch_size 的新
    # env_ids，与仍为全长的 `self.env_ids` 相减时形状不匹配（smoke 崩在聚类而不是审计逻辑）。
    # 完整运行时 `len(train_loader.dataset)` 不变，这里是空操作。
    env_ids = env_ids[: len(train_loader.dataset)]
    audit = GA.GradientAudit(model=model, uni_coe=P.UNI_COE, env_coe=P.ENV_COE)
    manager = GA.GradientAuditTrainManager(
        model=model, train_loader=train_loader, val_loader=loaders["val"], env_ids=env_ids,
        task_name=["Income", "Marital"], lr=P.LR, batch_size=loaders["train"].batch_size,
        uni_coe=P.UNI_COE, env_coe=P.ENV_COE, epochs=epochs, patience=P.PATIENCE, audit=audit)

    with tee_stdout(log_buffer):
        manager.train_two_task()
        groups = GA.parameter_groups(model)
        print(f"[audit] 采样步数 = {len(audit.records)}（计划 {len(manager.audit_schedule)} 步/epoch）")
        print(f"[audit] 损失分解恒等偏差（相对）= {audit.identity_gap}")

    # 4) 产物：audit.json 不含时间戳/路径 → 同输入逐字节可复现
    smoke = max_batches is not None
    meta = {
        "commit": commit, "smoke": smoke, "max_batches": max_batches,
        "split_seed": split_seed, "model_seed": model_seed, "env_seed": env_seed,
        "epochs": epochs, "patience": P.PATIENCE,
        "batch_size": loaders["train"].batch_size, "lr": P.LR,
        "uni_coe": P.UNI_COE, "env_coe": P.ENV_COE,
        "reg_embedding": P.REG_EMBEDDING, "reg_dnn": P.REG_DNN,
        "n_train": stats["n_train"], "n_batches": len(train_loader),
        "n_sampled_steps": len(audit.records),
        "sampled_steps_per_epoch": len(manager.audit_schedule),
        "meets_min_sampled_steps": len(audit.records) >= GA.MIN_SAMPLED_STEPS,
        "warmup_steps": GA.WARMUP_STEPS, "spaced_steps": GA.SPACED_STEPS,
        "split_fingerprint_sha256": fp["fingerprint_sha256"],
        "loss_identity_gap": audit.identity_gap,
        "loss_identity_ok": audit.identity_gap is not None and audit.identity_gap < 1e-5,
        "group_sizes": GA.group_sizes(groups),
        "unclassified_parameters": groups[GA.GROUP_UNCLASSIFIED],
        "n_parameters": sum(param.numel() for param in model.parameters()),
    }
    payload = GA.build_payload(step_records=audit.records, meta=meta, smoke=smoke)
    run_id = P.make_run_id(now or datetime.now(), split_seed=split_seed, model_seed=model_seed,
                           tag="smoke" if smoke else "short", commit=commit)
    run_dir = root / "runs" / run_id
    GA.write_audit(run_dir, payload, prereg_text=Path(GA.PREREG_SPEC).read_text(encoding="utf-8"),
                   stdout_text=log_buffer.getvalue())
    GA.append_summary_row(root / "SUMMARY.md",
                          GA.summary_row(run_id=run_id, commit=commit, payload=payload))

    gate = payload["gate"]
    print(f"[audit] run_id={run_id} status={gate['status']} dir={run_dir}")
    for reason in gate["reasons"]:
        print(f"[audit] {reason}")
    return {"run_id": run_id, "run_dir": str(run_dir), "payload": payload, "gate": gate}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CensusIncome 阶段 1 梯度冲突审计（诊断分支）")
    parser.add_argument("--root", type=Path, default=GA.ARTIFACT_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=P.STAGE1_EPOCHS)
    parser.add_argument("--split-seed", type=int, default=P.SPLIT_SEED)
    parser.add_argument("--model-seed", type=int, default=P.MODEL_SEED)
    parser.add_argument("--env-seed", type=int, default=P.ENV_SEED)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="smoke：只跑前 N 个 batch；不产生门禁判定，不得当作审计结果")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[cli] device={device} root={args.root} epochs={args.epochs} "
          f"max_batches={args.max_batches}")
    run_audit(args.root, split_seed=args.split_seed, model_seed=args.model_seed,
              env_seed=args.env_seed, epochs=args.epochs, device=device,
              max_batches=args.max_batches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
