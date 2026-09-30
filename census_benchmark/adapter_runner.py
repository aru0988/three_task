"""CensusIncome 阶段 2 低秩残差适配 runner 变体（协议路径；**不修改任何协议文件**）。

预注册文档：docs/superpowers/specs/2026-09-30-stage2-lora-adapter-design.md

与 `run_census_benchmark.run_stage2` 的差异（仅三处，其余逐行同构）：
1. 新任务头换成 `adapter.LoRANewTask`（共享一份 rank=4 残差适配器，作用于冻结表征）；
2. 训练循环追加适配器梯度探针，每个 epoch 记录适配器范数；
3. 评测后追加一遍 val 统计适配器输出占比/余弦，并把 A6/G/R1/S1 四项**实验门禁**写进
   `gate_report.json` 的 `"adapter"` 子对象（**不并入**协议 A/B 的 `overall_pass`）。

复用（import，而非复制）：`build_census_loaders` / `build_mptrec`（= 同一划分、同一 backbone 构建）、
`tee_stdout`；协议语义全部来自 `census_benchmark.protocol` 与 `census_benchmark.metrics`
（含 AUC 评测 `metrics.evaluate_newtask` → 与基线同一段评测代码、同一口径）。

`run_id` 沿用协议格式（`<short|full>` 表示跑量档，机制标识落在 `config.json` / `adapter_report.json`），
`SUMMARY.md` 列结构与基线完全一致，且只追加、不重写历史。

用法（**必须用 `-m` 从仓库根执行**，与 `baseline.*` 的既有约定一致；直接 `python census_benchmark/adapter_runner.py`
会因 `sys.path[0]` 变成包目录而无法 `import run_census_benchmark`）：

    .venv\\Scripts\\python.exe -m census_benchmark.adapter_runner --stage1-dir <stage1 产物目录> --epochs 5
"""
from __future__ import annotations

import argparse
import copy
import io
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from census_benchmark import adapter
from census_benchmark import metrics
from census_benchmark import protocol as P
from run_census_benchmark import build_census_loaders, build_mptrec, tee_stdout


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_stage2_adapter(root=P.ARTIFACT_ROOT, *, stage1_dir, epochs=P.STAGE2_EPOCHS, tag="short",
                       device=None, model=None, loaders=None, stats=None, indices=None,
                       input_size=P.INPUT_SIZE, rep_dim=P.EXPERT_HIDDEN[-1],
                       rank=adapter.RANK, alpha=adapter.ALPHA, now=None) -> dict:
    """阶段 2（适配器变体）：与 run_stage2 同协议，只把新任务头换成 LoRANewTask。"""
    root = Path(root)
    device = device or torch.device("cuda:0")
    sid = Path(stage1_dir).name
    checkpoint = P.load_stage1(root, sid)                          # spec 4.3：唯一 backbone 来源
    meta, commit, log_buffer = checkpoint["meta"], P.code_commit(), io.StringIO()

    # 1) 同一划分：用 stage1 的 split seed 重建并双向校验（A2）
    if loaders is None:
        loaders, stats, indices = build_census_loaders(meta["split_seed"])
    val_idx, test_idx = indices
    fp = P.split_fingerprint(split_seed=meta["split_seed"], stats=stats, val_idx=val_idx, test_idx=test_idx)
    split_ok = (P.verify_split_fingerprint(fp, P.load_split_fingerprint(root, meta["split_seed"]))
                and fp["fingerprint_sha256"] == meta["split_fingerprint_sha256"])

    # 2) backbone 加载 + 真冻结三件套之 1、2；env_ids 只读取（A5）
    P.seed_model(meta["model_seed"])                               # 让 NewTask 初始化可复现
    backbone = model or build_mptrec(device)
    backbone.to(device)
    backbone.load_state_dict(checkpoint["backbone_state"])
    P.freeze_backbone(backbone)
    sha_before = P.backbone_sha256(backbone)
    env_ids = checkpoint["env_ids"]
    env_ids_ok = P.sha256_tensor(env_ids) == meta["env_ids_sha256"]

    # 3) 新任务头（适配器版）：在 seed_model 之后、按与基线相同的顺序构造
    #    → 共享参数与基线逐元素相等；适配器初始化被 isolated_cpu_rng 隔离 → 全局 RNG 端点不变
    rng_before = torch.get_rng_state().clone()
    newtask = adapter.LoRANewTask(input_size=input_size, rep_dim=rep_dim,
                                  tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN,
                                  device=device, rank=rank, alpha=alpha).to(device)
    rng_after = torch.get_rng_state().clone()

    # 4) 构造期审计（A6）：对**真实构造出来的实例**核对共享参数与全局 RNG 端点；
    #    审计在 isolated_cpu_rng 内完成，自身不消耗全局 RNG，也不参与训练
    identity = adapter.actual_instance_identity_report(
        newtask, rng_state_before=rng_before, rng_state_after=rng_after, input_size=input_size,
        rep_dim=rep_dim, tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN, device=device)
    params = adapter.param_report(newtask)
    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=P.LR)   # 同一优化器、同一 lr，含适配器参数
    loss_func = nn.BCELoss()
    probe = adapter.AdapterGradProbe(newtask.rep_adapter)

    best_auc, best_epoch, best_state, stale, epoch_records = -1.0, 0, None, 0, []
    norm_records, global_step = [], 0

    # 5) 训练：与基线同一循环，仅追加探针（val 只用于 early stop 选点；test 全程不参与选择）
    with tee_stdout(log_buffer):
        for epoch in range(1, epochs + 1):
            newtask.train()
            loss_sum, steps = 0.0, 0
            for _, _, y, features in loaders["train"]:
                features = {key: value.to(device) for key, value in features.items()}
                with torch.no_grad():                                # 三件套之 3：计算图级冻结
                    dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
                pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
                loss = loss_func(pred, y.float().to(device)) + newtask.get_l2_reg()
                optimizer.zero_grad()
                loss.backward()
                global_step += 1
                probe.observe(epoch=epoch, step=global_step, is_first=(global_step == 1))
                optimizer.step()
                loss_sum += float(loss)
                steps += 1
            probe.end_epoch()
            norm_records.append({"epoch": epoch, **adapter.adapter_norms(newtask.rep_adapter)})
            val = metrics.evaluate_newtask(newtask, backbone, loaders["val"], device)
            epoch_records.append({"epoch": epoch, "loss": loss_sum / max(steps, 1),
                                  "auc_val_education": val["auc"],
                                  "adapter_effective_fro": norm_records[-1]["effective_fro"]})
            print(f"[stage2-adapter] epoch={epoch} loss={loss_sum / max(steps, 1):.4f} "
                  f"auc_val_education={val['auc']:.4f} "
                  f"adapter_effective_fro={norm_records[-1]['effective_fro']:.6f}")
            if val["auc"] > best_auc:
                best_auc, best_epoch, stale = val["auc"], epoch, 0
                best_state = copy.deepcopy(newtask.state_dict())
            else:
                stale += 1
                if stale == P.PATIENCE:
                    print(f"[stage2-adapter] early stop at epoch {epoch}")
                    break

    # 6) 冻结校验 + 最终评测（val：机制指标 + 适配器统计；test 只评一次）
    newtask.load_state_dict(best_state)
    sha_after = P.backbone_sha256(backbone)
    grads_all_none = all(param.grad is None for param in backbone.parameters())
    P.assert_no_grads(backbone)
    val_final = metrics.evaluate_newtask(newtask, backbone, loaders["val"], device, mechanism=True)
    adapter_stats = adapter.evaluate_adapter_stats(newtask, backbone, loaders["val"], device)
    test_final = metrics.evaluate_newtask(newtask, backbone, loaders["test"], device)

    # 7) 门禁：协议 A/B 与基线同口径；实验门禁单独放 "adapter" 子对象，互不污染
    env_shares = [count / meta["n_train"] for record in meta["cluster_records"] for count in record["env_counts"]]
    report = metrics.judge(
        backbone_sha_equal=(sha_before == sha_after), grads_all_none=grads_all_none, split_ok=split_ok,
        split_stats_ok=bool(stats["disjoint"] and stats["union_complete"]), env_ids_ok=env_ids_ok,
        auc_val_income=meta["val_auc_income_max"], auc_val_marital=meta["val_auc_marital_max"],
        auc_test_education=test_final["auc"], auc_val_education_best=best_auc,
        gate_mean=val_final["gate_mean"], env_shares=env_shares)
    report["A3"] = {"status": "on_demand",
                    "detail": "按需复跑同一配置，比较 AUC-Test-Education 差 ≤ 1e-9 与 backbone_sha256 一致"}
    adapter_gates = {
        "A6": adapter.identity_gate(identity),
        "G": adapter.grad_gate(probe.first, probe.per_epoch),
        "R1": adapter.ratio_gate(adapter_stats),
        "S1": adapter.verdict(test_final["auc"] >= adapter.AUC_TEST_MIN,
                              auc_test_education=test_final["auc"], baseline_auc=adapter.BASELINE_AUC,
                              baseline_run_id=adapter.BASELINE_RUN_ID, threshold=adapter.AUC_TEST_MIN),
    }
    adapter_gates["overall_pass"] = all(item["pass"] for item in adapter_gates.values())
    adapter_gates["failures"] = [key for key, item in adapter_gates.items()
                                 if isinstance(item, dict) and "pass" in item and not item["pass"]]
    report["adapter"] = adapter_gates

    # 8) 产物落盘（与基线同一目录规范；SUMMARY 列结构不变，只追加）
    run_id = P.make_run_id(now or datetime.now(), split_seed=meta["split_seed"],
                           model_seed=meta["model_seed"], tag=tag, commit=commit)
    run_path = root / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    torch.save(newtask.state_dict(), run_path / "newtask.pt")
    torch.save(env_ids, run_path / "env_ids.pt")
    _write_json(run_path / "split_fingerprint.json", fp)
    _write_json(run_path / "config.json", {
        "run_id": run_id, "stage1_id": sid, "commit": commit, "tag": tag, "frozen": True,
        "mechanism": adapter.MECHANISM, "rank": rank, "alpha": alpha, "adapter_scale": alpha / rank,
        "adapter_injection": "gen_rep+spec_reps_pre_fusion", "adapter_shared_across_streams": True,
        "split_seed": meta["split_seed"], "model_seed": meta["model_seed"], "env_seed": meta["env_seed"],
        "epochs": epochs, "patience": P.PATIENCE, "lr": P.LR, "batch_size": loaders["train"].batch_size,
        "input_size": input_size, "rep_dim": rep_dim, **params})
    _write_json(run_path / "adapter_report.json", {
        "run_id": run_id, "stage1_id": sid, "commit": commit, "mechanism": adapter.MECHANISM,
        "preregistration": "docs/superpowers/specs/2026-09-30-stage2-lora-adapter-design.md",
        "config": {"rank": rank, "alpha": alpha, "scale": alpha / rank, "lr": P.LR, "epochs": epochs,
                   "patience": P.PATIENCE, "shared_across_streams": True,
                   "injection": "gen_rep+spec_reps_pre_fusion", "optimizer": "Adam(newtask.parameters())",
                   "l2_reg_on_adapter": False},
        "params": params, "construction_identity": identity,
        "grad_probe": {"first": probe.first, "per_epoch": probe.per_epoch},
        "adapter_norms_by_epoch": norm_records, "val_stats_final": adapter_stats,
        "gates": adapter_gates,
        "auc": {"test_education": test_final["auc"], "val_education_best": best_auc,
                "baseline_test_education": adapter.BASELINE_AUC,
                "delta_vs_baseline": test_final["auc"] - adapter.BASELINE_AUC}})
    _write_json(run_path / "metrics.json", {
        "run_id": run_id, "stage1_id": sid, "commit": commit, "mechanism": adapter.MECHANISM,
        "split_sha256": {"val": fp["val_sha256"], "test": fp["test_sha256"],
                         "fingerprint": fp["fingerprint_sha256"]},
        "env_ids_sha256": meta["env_ids_sha256"],
        "backbone_sha256_before": sha_before, "backbone_sha256_after": sha_after,
        "stage1": {"epoch_records": meta["epoch_records"], "best_epoch": meta["best_epoch"],
                   "uni_loss_0": meta["uni_loss_0_list"], "uni_loss_1": meta["uni_loss_1_list"],
                   "fuse_loss_0": meta["fuse_loss_0_list"], "fuse_loss_1": meta["fuse_loss_1_list"],
                   "env_loss": meta["env_loss_list"], "cluster_records": meta["cluster_records"]},
        "stage2": {"epoch_records": epoch_records, "best_epoch": best_epoch,
                   "best_val_auc": best_auc, "test_auc": test_final["auc"],
                   "adapter_norms_by_epoch": norm_records},
        "mechanism": {"gate_mean": val_final["gate_mean"], "cos_gen_spec": val_final["cos_gen_spec"],
                      "gen_std": val_final["gen_std"], "adapter": adapter_stats,
                      "env_acc_stage1": [record["env_acc"] for record in meta["epoch_records"]]}})
    _write_json(run_path / "gate_report.json", report)
    P.append_summary_row(root / "SUMMARY.md", {
        "run_id": run_id, "commit": commit, "auc_test_education": f"{test_final['auc']:.6f}", "stage1_id": sid,
        **{key: ("PASS" if report[key]["pass"] else "FAIL")
           for key in ("A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4")}})
    (run_path / "stdout.log").write_text(log_buffer.getvalue(), encoding="utf-8")
    print(f"[stage2-adapter] run_id={run_id} test_auc={test_final['auc']:.4f} "
          f"protocol_overall_pass={report['overall_pass']} adapter_overall_pass={adapter_gates['overall_pass']}")
    print(f"[stage2-adapter] protocol_failures={report['failures']} "
          f"adapter_failures={adapter_gates['failures']} run_dir={run_path}")
    return {"run_id": run_id, "run_dir": str(run_path), "report": report,
            "adapter_report": adapter_gates, "test_auc": test_final["auc"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CensusIncome 阶段 2 低秩残差适配变体（协议路径）")
    parser.add_argument("--stage1-dir", type=Path, required=True)          # spec 4.3：唯一引用方式
    parser.add_argument("--root", type=Path, default=P.ARTIFACT_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tag", choices=["short", "full"], default="short")
    parser.add_argument("--epochs", type=int, default=P.STAGE2_EPOCHS)
    parser.add_argument("--rank", type=int, default=adapter.RANK)          # 改这里即偏离预注册
    parser.add_argument("--alpha", type=float, default=adapter.ALPHA)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[cli] command=stage2-adapter device={device} root={args.root} stage1_dir={args.stage1_dir} "
          f"rank={args.rank} alpha={args.alpha}")
    if (args.rank, args.alpha) != (adapter.RANK, adapter.ALPHA):
        print(f"[cli] 警告：rank/alpha 偏离预注册值 ({adapter.RANK}, {adapter.ALPHA})，"
              f"该 run 不可用于预注册结论")
    run_stage2_adapter(args.root, stage1_dir=args.stage1_dir, epochs=args.epochs, tag=args.tag,
                       device=device, rank=args.rank, alpha=args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
