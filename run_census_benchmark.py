"""CensusIncome 阶段 1 / 阶段 2 公平评测入口（协议路径；零模型语义改动）。"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from census_benchmark import affinity_gate_corrected as AGC
from census_benchmark import metrics
from census_benchmark import protocol as P
from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import MPTRec, NewTask
from multitaskrec.train import MPTRecTrainManager


@contextlib.contextmanager
def tee_stdout(stream):
    """stdout 同时写真实终端与给定文本流（文件或 StringIO）→ 落盘 stdout.log。"""
    class _Tee:
        def write(self, text):
            sys.__stdout__.write(text)
            stream.write(text)
            return len(text)

        def flush(self):
            sys.__stdout__.flush()
            stream.flush()

    original, sys.stdout = sys.stdout, _Tee()
    try:
        yield
    finally:
        sys.stdout = original


class Stage1HookTrainManager(MPTRecTrainManager):
    """只挂记录钩子，loss / 优化器 / early-stop 语义全部沿用父类（spec 4）。

    父类 train_two_task 每个 epoch 调一次 self.evaluation_two_task(self.val_loader)；
    在该调用点记录 val AUC 与该 epoch 的训练集 env_acc（M1），并重写 cluster_2 记录 M2。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_records: list[dict] = []
        self.cluster_records: list[dict] = []

    def evaluation_two_task(self, data_loader):
        auc_pair = super().evaluation_two_task(data_loader)
        if data_loader is self.val_loader:
            self.epoch_records.append({"epoch": len(self.epoch_records) + 1,
                                       "auc_val_income": float(auc_pair[0]),
                                       "auc_val_marital": float(auc_pair[1]),
                                       "env_acc": self.train_env_acc()})
        return auc_pair

    def cluster_2(self):
        previous = self.env_ids.clone()
        updated = super().cluster_2()
        counts = torch.bincount(updated, minlength=P.NUM_ENVS).tolist()
        self.cluster_records.append({"epoch": len(self.epoch_records),
                                     "diff_num": int((previous != updated).sum()),
                                     "env_counts": [int(c) for c in counts]})
        return updated

    @torch.no_grad()
    def train_env_acc(self) -> float:
        """M1：训练集上 env_pred 与当前 env_ids 的一致率（父类每轮开头重设 train()，无副作用）。"""
        self.model.eval()
        log_probs, ids = [], []
        for step, batch in enumerate(self.train_loader):
            features = {key: value.to(self.device) for key, value in batch[-1].items()}
            log_probs.append(self.model(features)["env_pred"].cpu())
            ids.append(self.env_ids[self.batch_size * step: self.batch_size * (step + 1)])
        return metrics.env_accuracy(torch.cat(log_probs), torch.cat(ids))


def build_census_loaders(split_seed: int):
    """返回 (loaders, stats, (val_idx, test_idx))：train.gz 全量训练；test.gz 按 split seed 切成
    val / test，两侧都是 `Subset(test_dataset, idx.tolist())` + `batch_size=P.BATCH_SIZE`（spec 2.1、4.6）。"""
    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
    val_idx, test_idx = P.make_split(len(test_dataset), split_seed)
    loaders = {"train": DataLoader(train_dataset, batch_size=P.BATCH_SIZE),
               "val": DataLoader(Subset(test_dataset, val_idx.tolist()), batch_size=P.BATCH_SIZE),
               "test": DataLoader(Subset(test_dataset, test_idx.tolist()), batch_size=P.BATCH_SIZE)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=len(train_dataset)), (val_idx, test_idx)


def build_mptrec(device) -> MPTRec:
    """vocabulary 必须 `CensusIncome_Vocabulary_Size.copy()` 后再 `pop("education")`（spec 2.2.1）。"""
    vocabulary = CensusIncome_Vocabulary_Size.copy()
    vocabulary.pop("education")
    return MPTRec(num_tasks=P.NUM_TASKS, feature_vocabulary=vocabulary, embedding_size=P.EMBEDDING_SIZE,
                  input_size=P.INPUT_SIZE, expert_dnn_hidden_units=list(P.EXPERT_HIDDEN),
                  tower_dnn_hidden_units=list(P.TOWER_HIDDEN),
                  reg_embedding=P.REG_EMBEDDING, reg_dnn=P.REG_DNN, device=device)


def run_stage1(root=P.ARTIFACT_ROOT, *, split_seed=P.SPLIT_SEED, model_seed=P.MODEL_SEED,
               env_seed=P.ENV_SEED, epochs=P.STAGE1_EPOCHS, tag="short", device=None,
               model=None, loaders=None, stats=None, indices=None) -> dict:
    root = Path(root)
    device = device or torch.device("cuda:0")
    commit, log_buffer = P.code_commit(), io.StringIO()

    # 1) 划分：只由 split seed 决定；首次落盘基准，之后必须一致（A2、A4）
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

    # 2) 初始环境分配：独立 env seed（spec 5.3.2），长度 = 训练集长度
    env_ids = P.make_env_ids(len(loaders["train"].dataset), env_seed)

    # 3) 训练：复用 MPTRecTrainManager 语义，只额外记录
    P.seed_model(model_seed)                              # 模型初始化 / batch 顺序 / dropout
    model = model or build_mptrec(device)
    model.to(device)
    manager = Stage1HookTrainManager(
        model=model, train_loader=loaders["train"], val_loader=loaders["val"], env_ids=env_ids,
        task_name=["Income", "Marital"], lr=P.LR, batch_size=loaders["train"].batch_size,
        uni_coe=P.UNI_COE, env_coe=P.ENV_COE, epochs=epochs, patience=P.PATIENCE)
    with tee_stdout(log_buffer):
        manager.train_two_task()
    model.load_state_dict(manager.best_weight)
    final_env_ids = manager.env_ids.clone()               # cluster 覆盖后的最终环境分配 → 阶段 2 读取

    # 4) 内容寻址落盘（spec 4.2；只读、不覆盖）
    cfg = {"model_seed": model_seed, "env_seed": env_seed, "epochs": epochs, "patience": P.PATIENCE,
           "batch_size": loaders["train"].batch_size, "lr": P.LR, "uni_coe": P.UNI_COE, "env_coe": P.ENV_COE,
           "reg_embedding": P.REG_EMBEDDING, "reg_dnn": P.REG_DNN, "input_size": P.INPUT_SIZE,
           "embedding_size": P.EMBEDDING_SIZE, "expert_hidden": list(P.EXPERT_HIDDEN),
           "tower_hidden": list(P.TOWER_HIDDEN), "num_tasks": P.NUM_TASKS}
    cfg_sha = P.config_hash(cfg)
    sid = P.stage1_id(split_sha=fp["fingerprint_sha256"], model_seed=model_seed, epochs=epochs, cfg_sha=cfg_sha)
    records = manager.epoch_records
    meta = {"stage1_id": sid, "commit": commit, "config_hash": cfg_sha, **cfg, **stats,
            "created": datetime.now().isoformat(timespec="seconds"),
            "split_seed": split_seed, "model_seed": model_seed, "env_seed": env_seed,
            "split_fingerprint_sha256": fp["fingerprint_sha256"],
            "val_sha256": fp["val_sha256"], "test_sha256": fp["test_sha256"],
            "env_ids_sha256": P.sha256_tensor(final_env_ids), "backbone_sha256": P.backbone_sha256(model),
            "best_epoch": max(records, key=lambda r: r["auc_val_income"] + r["auc_val_marital"])["epoch"],
            "best_val_auc_sum": max(r["auc_val_income"] + r["auc_val_marital"] for r in records),
            "val_auc_income_max": max(r["auc_val_income"] for r in records),
            "val_auc_marital_max": max(r["auc_val_marital"] for r in records),
            "epoch_records": records, "cluster_records": manager.cluster_records,
            "uni_loss_0_list": manager.uni_loss_0_list, "uni_loss_1_list": manager.uni_loss_1_list,
            "fuse_loss_0_list": manager.fused_loss_0_list, "fuse_loss_1_list": manager.fused_loss_1_list,
            "env_loss_list": manager.env_loss_list}
    stage1_path = P.save_stage1(root, sid, backbone_state=model.state_dict(), env_ids=final_env_ids, meta=meta)
    (stage1_path / "stdout.log").write_text(log_buffer.getvalue(), encoding="utf-8")
    print(f"[stage1] id={sid} dir={stage1_path} backbone_sha256={meta['backbone_sha256']}")
    return {"stage1_id": sid, "dir": str(stage1_path), "meta": meta}


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_stage2(root=P.ARTIFACT_ROOT, *, stage1_dir, epochs=P.STAGE2_EPOCHS, tag="short", device=None,
               model=None, loaders=None, stats=None, indices=None, input_size=P.INPUT_SIZE,
               rep_dim=P.EXPERT_HIDDEN[-1], variant=AGC.BASELINE_VARIANT, now=None) -> dict:
    """阶段 2：只从 Stage-1 产物加载 backbone，训练新任务头，做门禁与 SUMMARY。

    `variant="affinity_corrected"` 时换成**修正版逐样本软门控**头（见
    docs/superpowers/experiments/2026-09-30-stage2-affinity-gate-corrected.md）：逐 epoch 记录门控读数
    与门控梯度范数（读真实前向的 `last_terms`，不重算、不抽 RNG），best_state 载入后做 val 诊断与
    源贡献探针，并按预注册判据（机制 > 效应 > 对齐）落盘 `metrics.json:affinity_corrected_arm`。
    默认 `baseline` 与未启用时行为完全一致（新任务头、config / metrics 的基线键集都不变）。
    """
    root = Path(root)
    device = device or torch.device("cuda:0")
    sid = Path(stage1_dir).name
    checkpoint = P.load_stage1(root, sid)                  # spec 4.3：唯一 backbone 来源（缺失即抛错）
    meta, commit, log_buffer = checkpoint["meta"], P.code_commit(), io.StringIO()
    treatment = variant != AGC.BASELINE_VARIANT

    # 1) 同一划分：用 stage1 的 split seed 重建并双向校验（A2）
    if loaders is None:
        loaders, stats, indices = build_census_loaders(meta["split_seed"])
    val_idx, test_idx = indices
    fp = P.split_fingerprint(split_seed=meta["split_seed"], stats=stats, val_idx=val_idx, test_idx=test_idx)
    split_ok = (P.verify_split_fingerprint(fp, P.load_split_fingerprint(root, meta["split_seed"]))
                and fp["fingerprint_sha256"] == meta["split_fingerprint_sha256"])

    # 2) backbone 加载 + 真冻结三件套之 1、2；env_ids 只读取（A5）
    P.seed_model(meta["model_seed"])                       # 让 NewTask 初始化可复现
    backbone = model or build_mptrec(device)
    backbone.to(device)
    backbone.load_state_dict(checkpoint["backbone_state"])
    P.freeze_backbone(backbone)
    sha_before = P.backbone_sha256(backbone)
    env_ids = checkpoint["env_ids"]
    env_ids_ok = P.sha256_tensor(env_ids) == meta["env_ids_sha256"]

    newtask = AGC.build_newtask(variant, input_size=input_size, rep_dim=rep_dim,
                                tower_dnn_hidden_units=list(P.TOWER_HIDDEN),
                                reg_dnn=P.REG_DNN, device=device).to(device)
    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=P.LR)      # 只含 NewTask 参数
    loss_func = nn.BCELoss()
    best_auc, best_epoch, best_state, stale, epoch_records = -1.0, 0, None, 0, []
    trace = AGC.TrainGateTrace() if treatment else None                     # 处理臂：门控轨迹

    # 3) 训练：val 只用于 early stop 选点；test 全程不参与选择（spec 4.6、7.3）
    with tee_stdout(log_buffer):
        for epoch in range(1, epochs + 1):
            if treatment:
                trace.start_epoch(epoch)
            newtask.train()
            loss_sum, steps = 0.0, 0
            for _, _, y, features in loaders["train"]:
                features = {key: value.to(device) for key, value in features.items()}
                with torch.no_grad():                                        # 三件套之 3：计算图级冻结
                    dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
                pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
                loss = loss_func(pred, y.float().to(device)) + newtask.get_l2_reg()
                optimizer.zero_grad()
                loss.backward()
                if treatment:                     # 读梯度 + 真实前向的门控读数（step 之前；不重算、不抽 RNG）
                    trace.record_step(newtask)
                optimizer.step()
                loss_sum += float(loss)
                steps += 1
            if treatment:
                trace.end_epoch()
            val = metrics.evaluate_newtask(newtask, backbone, loaders["val"], device)
            epoch_records.append({"epoch": epoch, "loss": loss_sum / max(steps, 1),
                                  "auc_val_education": val["auc"]})
            print(f"[stage2] epoch={epoch} loss={loss_sum / max(steps, 1):.4f} auc_val_education={val['auc']:.4f}")
            if val["auc"] > best_auc:
                best_auc, best_epoch, stale = val["auc"], epoch, 0
                best_state = copy.deepcopy(newtask.state_dict())
            else:
                stale += 1
                if stale == P.PATIENCE:
                    print(f"[stage2] early stop at epoch {epoch}")
                    break

    # 4) 冻结校验 + 最终评测（val 带机制指标一次；test 只评一次）
    newtask.load_state_dict(best_state)
    sha_after = P.backbone_sha256(backbone)
    grads_all_none = all(param.grad is None for param in backbone.parameters())
    P.assert_no_grads(backbone)
    val_final = metrics.evaluate_newtask(newtask, backbone, loaders["val"], device, mechanism=True)
    test_final = metrics.evaluate_newtask(newtask, backbone, loaders["test"], device)

    # 5) 门禁判定 + 产物落盘（spec 8、9.1、9.2）
    env_shares = [count / meta["n_train"] for record in meta["cluster_records"] for count in record["env_counts"]]
    report = metrics.judge(
        backbone_sha_equal=(sha_before == sha_after), grads_all_none=grads_all_none, split_ok=split_ok,
        split_stats_ok=bool(stats["disjoint"] and stats["union_complete"]), env_ids_ok=env_ids_ok,
        auc_val_income=meta["val_auc_income_max"], auc_val_marital=meta["val_auc_marital_max"],
        auc_test_education=test_final["auc"], auc_val_education_best=best_auc,
        gate_mean=val_final["gate_mean"], env_shares=env_shares)
    report["A3"] = {"status": "on_demand",
                    "detail": "按需复跑同一配置，比较 AUC-Test-Education 差 ≤ 1e-9 与 backbone_sha256 一致"}

    run_id = P.make_run_id(now or datetime.now(), split_seed=meta["split_seed"],
                           model_seed=meta["model_seed"], tag=tag, commit=commit)
    mechanism = {"gate_mean": val_final["gate_mean"], "cos_gen_spec": val_final["cos_gen_spec"],
                 "gen_std": val_final["gen_std"],
                 "env_acc_stage1": [record["env_acc"] for record in meta["epoch_records"]]}
    arm = None
    if treatment:
        run_id += AGC.RUN_ID_SUFFIX                        # 臂后缀：与基线 run 在 SUMMARY.md 中天然可区分
        # 诊断（机制判据 1–3、5、6 的读数）：best_state 已载入，只用 val 前向一遍
        diagnostics = AGC.evaluate_routing_gate(newtask, backbone, loaders["val"], device)
        trace_result = trace.result()
        # 源贡献诊断（L1O 边际效应 + gate↔效用对齐）：同一次训练与选点全部结束之后，零泄漏
        contribution = AGC.source_contribution_probe(newtask, backbone, loaders["val"], device)
        mech = AGC.mechanism_verdict(
            gate_std=diagnostics["gate_std"], gate_mean=diagnostics["gate_mean"],
            routing_l1_mean=diagnostics["routing_l1_mean"],
            gate_grad_norm_min_step=trace_result["grad_norm_min"],     # 判据 4：每个训练步
            train_eval_identical=diagnostics["invariants"]["train_eval_identical"],
            rng_unchanged=diagnostics["invariants"]["no_global_rng_consumed"])
        alignment = AGC.alignment_verdict(spearman=contribution["gate_utility_spearman"],
                                          auc=contribution["gate_utility_auc"],
                                          n=contribution["n_samples"])
        arm = AGC.arm_verdict(test_final["auc"], mechanism=mech, alignment=alignment)
        arm["diagnostics"] = diagnostics
        arm["source_contribution"] = contribution
        arm["train_gate_trace"] = trace_result["per_epoch"]
        arm["gate_grad_norm"] = {key: value for key, value in trace_result.items()
                                 if key != "per_epoch"}
        arm["prereg"] = AGC.preregistered_criteria()
        arm["provenance"] = AGC.provenance()
        # 诊断/探针在 A1 取哈希之后运行 ⇒ 这里补一次冻结复验（A1 语义不变，只是把缺口留痕）
        arm["backbone_sha256_after_probes"] = P.backbone_sha256(backbone)
        arm["grads_all_none_after_probes"] = all(param.grad is None for param in backbone.parameters())
    run_path = root / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    torch.save(newtask.state_dict(), run_path / "newtask.pt")
    torch.save(env_ids, run_path / "env_ids.pt")
    _write_json(run_path / "split_fingerprint.json", fp)
    _write_json(run_path / "config.json", {
        "run_id": run_id, "stage1_id": sid, "commit": commit, "tag": tag, "frozen": True,
        "variant": variant,
        "split_seed": meta["split_seed"], "model_seed": meta["model_seed"], "env_seed": meta["env_seed"],
        "epochs": epochs, "patience": P.PATIENCE, "lr": P.LR, "batch_size": loaders["train"].batch_size,
        "input_size": input_size, "rep_dim": rep_dim, **AGC.stage2_config(variant)})
    payload = {
        "run_id": run_id, "stage1_id": sid, "commit": commit, "variant": variant,
        "split_sha256": {"val": fp["val_sha256"], "test": fp["test_sha256"],
                         "fingerprint": fp["fingerprint_sha256"]},
        "env_ids_sha256": meta["env_ids_sha256"],
        "backbone_sha256_before": sha_before, "backbone_sha256_after": sha_after,
        "stage1": {"epoch_records": meta["epoch_records"], "best_epoch": meta["best_epoch"],
                   "uni_loss_0": meta["uni_loss_0_list"], "uni_loss_1": meta["uni_loss_1_list"],
                   "fuse_loss_0": meta["fuse_loss_0_list"], "fuse_loss_1": meta["fuse_loss_1_list"],
                   "env_loss": meta["env_loss_list"], "cluster_records": meta["cluster_records"]},
        "stage2": {"epoch_records": epoch_records, "best_epoch": best_epoch,
                   "best_val_auc": best_auc, "test_auc": test_final["auc"]},
        "mechanism": mechanism}
    if arm is not None:
        payload["affinity_corrected_arm"] = arm   # 臂级判定：不并入 judge() 的 overall_pass（A/B 门禁语义不变）
    _write_json(run_path / "metrics.json", payload)
    _write_json(run_path / "gate_report.json", report)
    P.append_summary_row(root / "SUMMARY.md", {
        "run_id": run_id, "commit": commit, "auc_test_education": f"{test_final['auc']:.6f}", "stage1_id": sid,
        **{key: ("PASS" if report[key]["pass"] else "FAIL")
           for key in ("A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4")}})
    (run_path / "stdout.log").write_text(log_buffer.getvalue(), encoding="utf-8")
    print(f"[stage2] run_id={run_id} test_auc={test_final['auc']:.4f} overall_pass={report['overall_pass']}")
    print(f"[stage2] failures={report['failures']} run_dir={run_path}")
    if arm is not None:
        print(f"[stage2] variant={variant} mechanism={arm['mechanism']['status']} "
              f"failed={arm['mechanism']['failed_rules']} alignment={arm['alignment']['status']} "
              f"status={arm['status']} (auc_test_education {test_final['auc']:.6f} vs {AGC.AUC_TEST_MIN})")
    return {"run_id": run_id, "run_dir": str(run_path), "report": report, "test_auc": test_final["auc"],
            "variant": variant, "arm": arm}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CensusIncome 阶段 2 公平评测（协议路径）")
    parser.add_argument("--root", type=Path, default=P.ARTIFACT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("stage1", help="阶段 1：预训练并落盘内容寻址产物")
    for flag, kwargs in (("--gpu", dict(type=int, default=0)),
                         ("--tag", dict(choices=["short", "full"], default="short")),
                         ("--epochs", dict(type=int, default=P.STAGE1_EPOCHS)),
                         ("--split-seed", dict(type=int, default=P.SPLIT_SEED)),
                         ("--model-seed", dict(type=int, default=P.MODEL_SEED)),
                         ("--env-seed", dict(type=int, default=P.ENV_SEED))):
        first.add_argument(flag, **kwargs)
    second = sub.add_parser("stage2", help="阶段 2：加载固定 Stage-1 产物，只训练新任务头")
    second.add_argument("--stage1-dir", type=Path, required=True)       # spec 4.3：唯一引用方式
    second.add_argument("--gpu", type=int, default=0)
    second.add_argument("--tag", choices=["short", "full"], default="short")
    second.add_argument("--epochs", type=int, default=P.STAGE2_EPOCHS)
    second.add_argument("--variant", choices=list(AGC.VARIANTS), default=AGC.BASELINE_VARIANT,
                        help="新任务头变体：baseline=基线 NewTask（默认，逐位同 master）；"
                             "affinity_corrected=修正版逐样本软门控（spec 2026-09-30，"
                             "run_id 带 -affcorr 后缀）")
    second.add_argument("--affinity-corrected", action="store_const", const=AGC.VARIANT, dest="variant",
                        help="等价于 --variant affinity_corrected（后出现者生效）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[cli] command={args.command} device={device} root={args.root}")
    if args.command == "stage1":
        run_stage1(args.root, split_seed=args.split_seed, model_seed=args.model_seed,
                   env_seed=args.env_seed, epochs=args.epochs, tag=args.tag, device=device)
    else:
        run_stage2(args.root, stage1_dir=args.stage1_dir, epochs=args.epochs, tag=args.tag, device=device,
                   variant=args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
