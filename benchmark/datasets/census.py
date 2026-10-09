"""CensusIncome 数据集适配：常量、划分/指纹、加载器与模型构造。

唯一事实来源：docs/superpowers/specs/2026-09-29-*.md。函数体搬自 census_benchmark/protocol.py
与 run_census_benchmark.py，行为必须保持位级一致；共享纯工具见 benchmark/protocol.py。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from benchmark import gates
from benchmark.cliutil import write_json
from benchmark.metrics import auc
from benchmark.profile import DatasetProfile
from benchmark.protocol import (
    append_summary_row,
    backbone_sha256,
    canonical_json,
    sha256_bytes,
    sha256_tensor,
)
from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import MPTRec, NewTask

# ---- 三个独立种子（spec 5.2）----
SPLIT_SEED = 20260929          # 唯一决定 val/test 划分，跨 model seed 恒定
MODEL_SEED = 1685480945        # 首轮唯一 model seed（模型初始化 / batch 顺序 / dropout）
ENV_SEED = 20260929            # 只决定初始 env_ids
# ---- 首轮 smoke 配置（spec 6.1：只降 epochs，不动数据与超参）----
STAGE1_EPOCHS, STAGE2_EPOCHS, PATIENCE = 2, 5, 2
# ---- 论文超参（spec 2.2.3，本分支冻结）----
BATCH_SIZE, LR = 256, 1e-3
REG_EMBEDDING, REG_DNN = 0.006, 3e-5
UNI_COE, ENV_COE = 0.9, 0.1
INPUT_SIZE, EMBEDDING_SIZE = 123, 4
EXPERT_HIDDEN, TOWER_HIDDEN = (256, 128), (64, 32)
NUM_TASKS, NUM_ENVS = 2, 2
# ---- 门禁阈值（spec 8；只允许在看到结果之前修改）----
AUC_FLOOR, VAL_TEST_GAP, GATE_MIN, GATE_MAX, ENV_SHARE_MIN = 0.60, 0.03, 0.05, 0.95, 0.05

ARTIFACT_ROOT = Path("artifacts/census_stage2")
SUMMARY_COLUMNS = ["run_id", "commit", "auc_test_education",
                   "A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4", "stage1_id"]


def make_split(n_test: int, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """按索引切分 test.gz，返回排序后的 (val_idx, test_idx)：与 train_test_split(dataset, ...,
    random_state=split_seed) 同一排列，但可落盘、可校验。"""
    val_idx, test_idx = train_test_split(np.arange(n_test), test_size=0.5, random_state=split_seed)
    return np.sort(val_idx), np.sort(test_idx)


def split_stats(val_idx: np.ndarray, test_idx: np.ndarray, n_train: int) -> dict:
    """A4 的三集合实测行数与交并检查。"""
    n = len(val_idx) + len(test_idx)
    return {"n_train": int(n_train), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
            "disjoint": bool(set(val_idx.tolist()).isdisjoint(test_idx.tolist())),
            "union_complete": bool(sorted(val_idx.tolist() + test_idx.tolist()) == list(range(n)))}


def split_fingerprint(*, split_seed: int, stats: dict, val_idx, test_idx) -> dict:
    """确定性指纹：不含时间戳，同 split seed 必须逐字节一致（A2）。"""
    fp = {"split_seed": int(split_seed), **stats,
          "val_sha256": sha256_tensor(torch.from_numpy(val_idx)),
          "test_sha256": sha256_tensor(torch.from_numpy(test_idx)),
          "val_indices_head": val_idx[:5].tolist(), "test_indices_head": test_idx[:5].tolist()}
    fp["fingerprint_sha256"] = sha256_bytes(canonical_json(fp).encode())
    return fp


def split_dir(root: Path, split_seed: int) -> Path:
    return Path(root) / "splits" / str(split_seed)


def write_split_fingerprint(root: Path, split_seed: int, fp: dict) -> Path:
    path = split_dir(root, split_seed) / "split_fingerprint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_split_fingerprint(root: Path, split_seed: int) -> dict | None:
    path = split_dir(root, split_seed) / "split_fingerprint.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def verify_split_fingerprint(fp: dict, baseline: dict) -> bool:
    """A2：指纹只由 split seed 决定，必须与基准逐字节一致。"""
    return bool(baseline) and fp["fingerprint_sha256"] == baseline["fingerprint_sha256"]


def save_split_indices(root: Path, split_seed: int, val_idx, test_idx) -> Path:
    path = split_dir(root, split_seed) / "split_indices.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, val=val_idx, test=test_idx)
    return path


def build_census_loaders(split_seed: int):
    """返回 (loaders, stats, (val_idx, test_idx))：train.gz 全量训练；test.gz 按 split seed 切成
    val / test，两侧都是 `Subset(test_dataset, idx.tolist())` + `batch_size=BATCH_SIZE`（spec 2.1、4.6）。"""
    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
    val_idx, test_idx = make_split(len(test_dataset), split_seed)
    loaders = {"train": DataLoader(train_dataset, batch_size=BATCH_SIZE),
               "val": DataLoader(Subset(test_dataset, val_idx.tolist()), batch_size=BATCH_SIZE),
               "test": DataLoader(Subset(test_dataset, test_idx.tolist()), batch_size=BATCH_SIZE)}
    return loaders, split_stats(val_idx, test_idx, n_train=len(train_dataset)), (val_idx, test_idx)


def build_mptrec(device) -> MPTRec:
    """vocabulary 必须 `CensusIncome_Vocabulary_Size.copy()` 后再 `pop("education")`（spec 2.2.1）。"""
    vocabulary = CensusIncome_Vocabulary_Size.copy()
    vocabulary.pop("education")
    return MPTRec(num_tasks=NUM_TASKS, feature_vocabulary=vocabulary, embedding_size=EMBEDDING_SIZE,
                  input_size=INPUT_SIZE, expert_dnn_hidden_units=list(EXPERT_HIDDEN),
                  tower_dnn_hidden_units=list(TOWER_HIDDEN),
                  reg_embedding=REG_EMBEDDING, reg_dnn=REG_DNN, device=device)


def judge(*, backbone_sha_equal: bool, grads_all_none: bool, split_ok: bool, split_stats_ok: bool,
          env_ids_ok: bool, auc_val_income: float, auc_val_marital: float, auc_test_education: float,
          auc_val_education_best: float, gate_mean: list[float], env_shares: list[float]) -> dict:
    """A1/A2/A4/A5 + B1–B4（spec 8）。A3 按需触发，由调用方另行写入，不进入 overall_pass。"""
    outcomes = gates.evaluate([
        ("A1", backbone_sha_equal and grads_all_none,
         {"backbone_sha_equal": backbone_sha_equal, "grads_all_none": grads_all_none}),
        ("A2", split_ok, {"split_fingerprint_consistent": split_ok}),
        ("A4", split_stats_ok, {"disjoint_and_complete": split_stats_ok}),
        ("A5", env_ids_ok, {"env_ids_sha256_matches_stage1": env_ids_ok}),
        ("B1", min(auc_val_income, auc_val_marital, auc_test_education) >= AUC_FLOOR,
         {"auc_val_income": auc_val_income, "auc_val_marital": auc_val_marital,
          "auc_test_education": auc_test_education, "floor": AUC_FLOOR}),
        ("B2", abs(auc_val_education_best - auc_test_education) <= VAL_TEST_GAP,
         {"gap": abs(auc_val_education_best - auc_test_education), "limit": VAL_TEST_GAP}),
        ("B3", all(GATE_MIN <= w <= GATE_MAX for w in gate_mean), {"gate_mean": gate_mean}),
        ("B4", all(share >= ENV_SHARE_MIN for share in env_shares), {"env_shares": env_shares}),
    ])
    report = gates.render_pass_detail(outcomes)
    report["overall_pass"] = gates.hard_pass((outcome.state for outcome in outcomes), allowed=(gates.PASS,))
    report["failures"] = [outcome.gate_id for outcome in outcomes if outcome.state == gates.FAIL]
    return report


# ---- 机制指标 M3/M4 与新任务头评估（搬自 census_benchmark/metrics.py，行为不变）----
class GateStats:
    """M3：各 gate 在样本维度的平均权重（流式累加，显存 O(1)）。

    gate_networks[i] 输出 2 维（specific 分支 / general 分支）且过 softmax → 两维互补，
    故 result() 报告**每个任务 specific 分支**的样本均值（= 1 − general 分支均值），长度 = num_tasks；
    这正是 B3「未坍缩到单一分支」判据所需的量（0.05 / 0.95 即两端的坍缩边界）。
    """

    def __init__(self, num_tasks: int = NUM_TASKS):
        self.total = torch.zeros(num_tasks, 2, dtype=torch.float64)
        self.count = 0

    def update(self, gate_outs: list[torch.Tensor]) -> None:
        for i, gate_out in enumerate(gate_outs):
            # 累加器恒在 CPU：先在原设备归约，再显式 .cpu()，否则 CPU 累加器接 CUDA 张量会 device mismatch
            self.total[i] += gate_out.detach().double().sum(dim=0).cpu()
        self.count += gate_outs[0].shape[0]

    def result(self) -> list[float]:
        return (self.total[:, 0] / max(self.count, 1)).tolist()


class RepStats:
    """M4：cos(gen_rep, spec_rep_i) 均值 + gen_rep 各维标准差均值（防常量表征）。

    用累加和 / 平方和代替"按固定 seed 抽样后保存中间张量"：全量、显存 O(1)、不受抽样影响。
    """

    def __init__(self, num_tasks: int = NUM_TASKS):
        self.cos_sum = torch.zeros(num_tasks, dtype=torch.float64)
        self.gen_sum = self.gen_sq_sum = None
        self.count = 0

    def update(self, gen_rep: torch.Tensor, spec_reps: list[torch.Tensor]) -> None:
        gen = gen_rep.detach().double()
        # 同 GateStats：归约留在原设备，结果显式 .cpu()，三个累加器恒为 CPU（result() 的返回类型不变）
        gen_sum = gen.sum(dim=0).cpu()
        gen_sq_sum = (gen * gen).sum(dim=0).cpu()
        self.gen_sum = gen_sum if self.gen_sum is None else self.gen_sum + gen_sum
        self.gen_sq_sum = gen_sq_sum if self.gen_sq_sum is None else self.gen_sq_sum + gen_sq_sum
        for i, spec in enumerate(spec_reps):
            self.cos_sum[i] += F.cosine_similarity(gen, spec.detach().double(), dim=1).sum().cpu()
        self.count += gen.shape[0]

    def result(self) -> dict:
        n = max(self.count, 1)
        mean = self.gen_sum / n
        std = (self.gen_sq_sum / n - mean * mean).clamp_min(0).sqrt()
        return {"cos_gen_spec": (self.cos_sum / n).tolist(), "gen_std": float(std.mean())}


@torch.no_grad()
def evaluate_newtask(newtask, backbone, loader, device, *, mechanism: bool = False) -> dict:
    """评估新任务头。mechanism=True 时额外算 M3/M4（只在 val 上开一次）。"""
    newtask.eval(); backbone.eval()                 # 三件套之 2：backbone 恒为 eval
    ys, preds = [], []
    gate = GateStats() if mechanism else None
    rep = RepStats() if mechanism else None
    for _, _, y, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)   # 三件套之 3：no_grad 抽取
        pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
        ys.append(y); preds.append(pred.detach())
        if mechanism:
            gate.update([backbone.gate_networks[i](dnn_input) for i in range(NUM_TASKS)])
            rep.update(gen_rep, spec_reps)
    out = {"auc": auc(torch.cat(ys), torch.cat(preds))}
    if mechanism:
        out["gate_mean"] = gate.result(); out.update(rep.result())
    return out


# ---- 数据集 profile（供 benchmark/pipeline 两阶段编排）----
def stage1_epoch_records(manager) -> list:
    """census 侧 epoch_records（旧 Stage1HookTrainManager.epoch_records 的等价重建）。"""
    return [{"epoch": i + 1, "auc_val_income": aucs[0], "auc_val_marital": aucs[1],
             "env_acc": manager.env_accs[i]} for i, aucs in enumerate(manager.val_epoch_aucs)]


def build_profile(*, split_seed: int = SPLIT_SEED, model_seed: int = MODEL_SEED,
                  env_seed: int = ENV_SEED, loaders=None, stats=None, indices=None,
                  input_size: int = INPUT_SIZE, rep_dim: int = EXPERT_HIDDEN[-1]) -> DatasetProfile:
    """census 两阶段适配；model_seed/env_seed 供 CLI 透传；loaders/stats/indices 与
    input_size/rep_dim 仅供 tiny e2e 测试注入。"""

    def prepare(root):
        if loaders is not None:
            ld, st, ix = loaders, stats, indices
        else:
            ld, st, ix = build_census_loaders(split_seed)
        val_idx, test_idx = ix
        fp = split_fingerprint(split_seed=split_seed, stats=st, val_idx=val_idx, test_idx=test_idx)
        baseline = load_split_fingerprint(root, split_seed)
        if baseline is None:
            write_split_fingerprint(root, split_seed, fp)
            save_split_indices(root, split_seed, val_idx, test_idx)
        elif not verify_split_fingerprint(fp, baseline):
            raise RuntimeError("划分指纹与基准不一致（A2 失败），协议禁止继续")
        return {"loaders": ld, "stats": st, "indices": ix, "fp": fp, "split_seed": split_seed}

    def manager_kwargs(ctx, epochs):
        ld = ctx["loaders"]
        return {"train_loader": ld["train"], "val_loader": ld["val"], "task_name": ["Income", "Marital"],
                "lr": LR, "batch_size": ld["train"].batch_size, "uni_coe": UNI_COE, "env_coe": ENV_COE,
                "epochs": epochs, "patience": PATIENCE, "record_env_acc": True, "cluster_epoch_offset": 0}

    def build_cfg(ctx, epochs):
        return {"model_seed": model_seed, "env_seed": env_seed, "epochs": epochs, "patience": PATIENCE,
                "batch_size": ctx["loaders"]["train"].batch_size, "lr": LR, "uni_coe": UNI_COE,
                "env_coe": ENV_COE, "reg_embedding": REG_EMBEDDING, "reg_dnn": REG_DNN,
                "input_size": INPUT_SIZE, "embedding_size": EMBEDDING_SIZE,
                "expert_hidden": list(EXPERT_HIDDEN), "tower_hidden": list(TOWER_HIDDEN),
                "num_tasks": NUM_TASKS}

    def build_meta(*, sid, cfg, cfg_sha, ctx, model, manager, env_ids, post, commit, epochs):
        fp, stats = ctx["fp"], ctx["stats"]
        records = stage1_epoch_records(manager)
        return {"stage1_id": sid, "commit": commit, "config_hash": cfg_sha, **cfg, **stats,
                "created": datetime.now().isoformat(timespec="seconds"),
                "split_seed": ctx["split_seed"], "model_seed": model_seed, "env_seed": env_seed,
                "split_fingerprint_sha256": fp["fingerprint_sha256"],
                "val_sha256": fp["val_sha256"], "test_sha256": fp["test_sha256"],
                "env_ids_sha256": sha256_tensor(env_ids), "backbone_sha256": backbone_sha256(model),
                "best_epoch": manager.best_epoch(),
                "best_val_auc_sum": max(r["auc_val_income"] + r["auc_val_marital"] for r in records),
                "val_auc_income_max": max(r["auc_val_income"] for r in records),
                "val_auc_marital_max": max(r["auc_val_marital"] for r in records),
                "epoch_records": records, "cluster_records": manager.cluster_records,
                "uni_loss_0_list": manager.uni_loss_0_list, "uni_loss_1_list": manager.uni_loss_1_list,
                "fuse_loss_0_list": manager.fused_loss_0_list, "fuse_loss_1_list": manager.fused_loss_1_list,
                "env_loss_list": manager.env_loss_list}

    def after_save(stage1_path, log_buffer):
        (stage1_path / "stdout.log").write_text(log_buffer.getvalue(), encoding="utf-8")

    # ---- 阶段 2 ----
    def prepare_stage2(root, meta, device):
        if loaders is not None:
            ld, st, ix = loaders, stats, indices
        else:
            ld, st, ix = build_census_loaders(meta["split_seed"])
        val_idx, test_idx = ix
        fp = split_fingerprint(split_seed=meta["split_seed"], stats=st, val_idx=val_idx, test_idx=test_idx)
        split_ok = (verify_split_fingerprint(fp, load_split_fingerprint(root, meta["split_seed"]))
                    and fp["fingerprint_sha256"] == meta["split_fingerprint_sha256"])
        return {"loaders": ld, "model_seed": meta["model_seed"], "split_seed": meta["split_seed"],
                "split_ok": split_ok, "split_stats_ok": bool(st["disjoint"] and st["union_complete"]),
                "fp": fp, "input_size": input_size, "rep_dim": rep_dim}

    def build_newtask(device, ctx2):
        return NewTask(input_size=ctx2["input_size"], rep_dim=ctx2["rep_dim"],
                       tower_dnn_hidden_units=list(TOWER_HIDDEN), reg_dnn=REG_DNN,
                       device=device).to(device)

    def val_auc(newtask, backbone, loader, device):
        return evaluate_newtask(newtask, backbone, loader, device)["auc"]

    def epoch_record(epoch, loss, auc_val):
        return {"epoch": epoch, "loss": loss, "auc_val_education": auc_val}

    def log_stage2_epoch(epoch, loss, auc_val, stale, stopped, log):
        log(f"[stage2] epoch={epoch} loss={loss:.4f} auc_val_education={auc_val:.4f}")
        if stopped:
            log(f"[stage2] early stop at epoch {epoch}")

    def final_eval(newtask, backbone, ctx2, device):
        return {"val": evaluate_newtask(newtask, backbone, ctx2["loaders"]["val"], device, mechanism=True),
                "test": evaluate_newtask(newtask, backbone, ctx2["loaders"]["test"], device)}

    def build_gates(final, ctx2, meta, facts, best_auc, log):
        env_shares = [count / meta["n_train"] for record in meta["cluster_records"]
                      for count in record["env_counts"]]
        report = judge(backbone_sha_equal=(facts["sha_before"] == facts["sha_after"]),
                       grads_all_none=facts["grads_all_none"], split_ok=ctx2["split_ok"],
                       split_stats_ok=ctx2["split_stats_ok"], env_ids_ok=facts["env_ids_ok"],
                       auc_val_income=meta["val_auc_income_max"], auc_val_marital=meta["val_auc_marital_max"],
                       auc_test_education=final["test"]["auc"], auc_val_education_best=best_auc,
                       gate_mean=final["val"]["gate_mean"], env_shares=env_shares)
        report["A3"] = {"status": "on_demand",
                        "detail": "按需复跑同一配置，比较 AUC-Test-Education 差 ≤ 1e-9 与 backbone_sha256 一致"}
        return report, report["overall_pass"]

    def write_artifacts(run_path, bundle):
        ctx2, meta, fp = bundle["ctx2"], bundle["meta"], bundle["ctx2"]["fp"]
        final, facts = bundle["final"], bundle["facts"]
        test_auc = final["test"]["auc"]
        torch.save(bundle["newtask"].state_dict(), run_path / "newtask.pt")
        torch.save(bundle["env_ids"], run_path / "env_ids.pt")
        write_json(run_path / "split_fingerprint.json", fp)
        write_json(run_path / "config.json", {
            "run_id": bundle["run_id"], "stage1_id": bundle["stage1_id"], "commit": bundle["commit"],
            "tag": bundle["tag"], "frozen": True, "split_seed": meta["split_seed"],
            "model_seed": meta["model_seed"], "env_seed": meta["env_seed"], "epochs": bundle["epochs"],
            "patience": PATIENCE, "lr": LR, "batch_size": bundle["loaders"]["train"].batch_size,
            "input_size": ctx2["input_size"], "rep_dim": ctx2["rep_dim"]})
        write_json(run_path / "metrics.json", {
            "run_id": bundle["run_id"], "stage1_id": bundle["stage1_id"], "commit": bundle["commit"],
            "split_sha256": {"val": fp["val_sha256"], "test": fp["test_sha256"],
                             "fingerprint": fp["fingerprint_sha256"]},
            "env_ids_sha256": meta["env_ids_sha256"],
            "backbone_sha256_before": facts["sha_before"], "backbone_sha256_after": facts["sha_after"],
            "stage1": {"epoch_records": meta["epoch_records"], "best_epoch": meta["best_epoch"],
                       "uni_loss_0": meta["uni_loss_0_list"], "uni_loss_1": meta["uni_loss_1_list"],
                       "fuse_loss_0": meta["fuse_loss_0_list"], "fuse_loss_1": meta["fuse_loss_1_list"],
                       "env_loss": meta["env_loss_list"], "cluster_records": meta["cluster_records"]},
            "stage2": {"epoch_records": bundle["epoch_records"], "best_epoch": bundle["best_epoch"],
                       "best_val_auc": bundle["best_auc"], "test_auc": test_auc},
            "mechanism": {"gate_mean": final["val"]["gate_mean"],
                          "cos_gen_spec": final["val"]["cos_gen_spec"], "gen_std": final["val"]["gen_std"],
                          "env_acc_stage1": [record["env_acc"] for record in meta["epoch_records"]]}})
        write_json(run_path / "gate_report.json", bundle["gates_doc"])
        append_summary_row(bundle["root"] / "SUMMARY.md", {
            "run_id": bundle["run_id"], "commit": bundle["commit"],
            "auc_test_education": f"{test_auc:.6f}", "stage1_id": bundle["stage1_id"],
            **{key: ("PASS" if bundle["gates_doc"][key]["pass"] else "FAIL")
               for key in ("A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4")}}, SUMMARY_COLUMNS)
        (run_path / "stdout.log").write_text(bundle["log_buffer"].getvalue(), encoding="utf-8")
        return {"report": bundle["gates_doc"], "test_auc": test_auc}

    def log_stage2_end(bundle, log):
        report = bundle["gates_doc"]
        log(f"[stage2] run_id={bundle['run_id']} test_auc={bundle['final']['test']['auc']:.4f} "
            f"overall_pass={report['overall_pass']}")
        log(f"[stage2] failures={report['failures']} run_dir={bundle['run_path']}")

    return DatasetProfile(
        name="census", model_seed=model_seed, env_seed=env_seed,
        stage1_epochs=STAGE1_EPOCHS, stage2_epochs=STAGE2_EPOCHS, stage2_patience=PATIENCE,
        stage2_lr=LR, stage2_cpu_loss=False, stage2_catch_grads=False,
        prepare=prepare, build_model=build_mptrec,
        train_size=lambda ctx: len(ctx["loaders"]["train"].dataset),
        manager_kwargs=manager_kwargs, post_train=lambda model, manager, ctx, device: {},
        build_cfg=build_cfg, fingerprint_sha=lambda ctx: ctx["fp"]["fingerprint_sha256"],
        state_dict=lambda model: model.state_dict(), build_meta=build_meta, after_save=after_save,
        log_summary=lambda sid, path, meta: print(
            f"[stage1] id={sid} dir={path} backbone_sha256={meta['backbone_sha256']}"),
        prepare_stage2=prepare_stage2, log_stage2_start=lambda sid, tag, ctx2, log: None,
        backbone_ready=lambda backbone, checkpoint, ctx2, log: {},
        build_newtask=build_newtask, val_auc=val_auc, epoch_record=epoch_record,
        log_stage2_epoch=log_stage2_epoch, final_eval=final_eval,
        log_grads_failure=lambda exc, log: None, build_gates=build_gates,
        run_id_prefix=lambda ctx2: f"s{ctx2['split_seed']}",
        write_artifacts=write_artifacts, log_stage2_end=log_stage2_end)
