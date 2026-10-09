"""AliCCP 数据集适配：常量、前缀指纹/标签计数、加载器与模型构造、阶段 1 记录探针。

唯一事实来源：docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md。
函数体搬自 aliccp_benchmark/protocol.py 与 aliccp_benchmark/bench.py，行为必须保持位级一致；
共享纯工具见 benchmark/protocol.py（不修改任何模型/训练代码）。
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from benchmark import gates
from benchmark.cliutil import write_json
from benchmark.metrics import auc
from benchmark.profile import DatasetProfile
from benchmark.protocol import (
    append_summary_row,
    backbone_sha256,
    canonical_json,
    git_state,
    sha256_bytes,
    sha256_tensor,
    stage1_id,
)
from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec, NewTask

# ---- 前缀预算（spec 5.1）----
PREFIX_TAG = "p2M-v500k-t1M"
TRAIN_BUDGET, VAL_BUDGET, TEST_BUDGET = 2_000_000, 500_000, 1_000_000
SMOKE_TRAIN_BUDGET, SMOKE_VAL_BUDGET, SMOKE_TEST_BUDGET = 20_000, 5_000, 10_000
DATA_FILES = {
    "train": "dataset/AliCCP/ctr_cvr.train",
    "val": "dataset/AliCCP/ctr_cvr.dev",
    "test": "dataset/AliCCP/ctr_cvr.test",
}
# AliCCP 无 split 种子：前缀由预算确定性定义（spec 6.1）
MODEL_SEED = 1688723512
ENV_SEED = 20261003
# ---- 训练配置（spec 8.1/8.3；论文值冻结）----
STAGE1_EPOCHS, STAGE1_PATIENCE = 3, 2
STAGE2_EPOCHS, STAGE2_PATIENCE = 5, 2
BATCH_SIZE, LR = 2000, 1e-4
UNI_COE, ENV_COE = 0.9, 0.1
REG_EMBEDDING, REG_DNN = 1e-4, 7e-6
INPUT_SIZE, EMBEDDING_SIZE = 80, 5
EXPERT_HIDDEN, TOWER_HIDDEN = (128, 64), (32, 32)
DROPOUT = (0.1, 0.3)
NUM_TASKS, NUM_ENVS = 2, 2
NEWTASK_REP_DIM = 64
# ---- 门禁阈值（spec 10；只允许在看到结果之前修改）----
AUC_FLOOR_CTR, AUC_FLOOR_CVR, AUC_FLOOR_BSI = 0.55, 0.50, 0.53
VAL_TEST_GAP_BSI = 0.05
GATE_MIN, GATE_MAX = 0.05, 0.95
ENV_SHARE_MIN = 0.05
IMPROVE_DELTA_AUC_TEST_BSI = 0.005  # provisional（spec 10.1）

ARTIFACT_ROOT = Path("artifacts/aliccp_bench")
SUMMARY_COLUMNS = [
    "run_id", "commit", "tag", "auc_val_bsi_best", "auc_test_bsi",
    "A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "stage1_id",
]


def prefix_tag_for(*, train: int = TRAIN_BUDGET, val: int = VAL_BUDGET,
                   test: int = TEST_BUDGET) -> str:
    """全量预算 → PREFIX_TAG（唯一正式基线）；否则 p{train}-v{val}-t{test}（spec 5.1）。"""
    if (train, val, test) == (TRAIN_BUDGET, VAL_BUDGET, TEST_BUDGET):
        return PREFIX_TAG
    return f"p{train}-v{val}-t{test}"


def build_vocab() -> dict:
    """AliCCP vocab 副本（去掉 101 干扰特征与 301 第三标签列）；绝不修改全局字典（spec 2.2）。"""
    vocab = AliCCP_Vocabulary_Size.copy()
    for key in ("101", "301"):
        vocab.pop(key)
    return vocab


# ---- 前缀身份（spec 5.2）----
def _prefix_budget_check(n: int) -> None:
    if n < 1:
        raise ValueError(f"预算必须为正整数: {n}")


def prefix_sha256(path, n: int) -> str:
    """sha256(表头 + 前 n 条数据行的原始字节)；数据行不足 n 时抛 ValueError。"""
    _prefix_budget_check(n)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"文件为空: {path}")
        digest.update(header)
        for i in range(n):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} 数据行不足 {n}（在第 {i + 1} 行截断）")
            digest.update(line)
    return digest.hexdigest()


def scan_label_counts(path, n: int) -> dict:
    """独立于加载器的原始扫描：click1 / purchase1 / bsi_pos（raw != 2）/ bsi_raw 分布。"""
    _prefix_budget_check(n)
    counts = {"n": 0, "click1": 0, "purchase1": 0, "bsi_pos": 0, "bsi_raw": {}}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i in range(n):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} 数据行不足 {n}（在第 {i + 1} 行截断）")
            parts = line.strip().split(",")
            click, purchase, raw = int(parts[0]), int(parts[1]), int(parts[-1])
            counts["n"] += 1
            counts["click1"] += click
            counts["purchase1"] += purchase
            counts["bsi_pos"] += 0 if raw == 2 else 1
            counts["bsi_raw"][raw] = counts["bsi_raw"].get(raw, 0) + 1
    return counts


def dataset_label_counts(dataset) -> dict:
    """从已加载的 AliCCPDataset 统计标签（bsi 为加载器变换后的值；raw 分布不可得）。"""
    click1 = purchase1 = bsi_pos = 0
    for row in dataset.data:
        click1 += int(row[0])
        purchase1 += int(row[1])
        bsi_pos += int(row[-1])
    return {"n": len(dataset.data), "click1": click1, "purchase1": purchase1, "bsi_pos": bsi_pos}


def verify_label_counts(dataset, budget: int, expected: dict) -> None:
    """A4：加载器实际样本数与标签计数必须与指纹逐项相等（比 AUC 更硬的标签对齐检查）。"""
    if len(dataset) != budget:
        raise AssertionError(f"样本数不符: len(dataset)={len(dataset)} != budget={budget}")
    actual = dataset_label_counts(dataset)
    for key in ("n", "click1", "purchase1", "bsi_pos"):
        if actual[key] != expected[key]:
            raise AssertionError(f"标签计数不符 {key}: loader={actual[key]} != fingerprint={expected[key]}")


def _row_digests(path, n: int) -> set:
    _prefix_budget_check(n)
    digests = set()
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i in range(n):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} 数据行不足 {n}（在第 {i + 1} 行截断）")
            digests.add(int.from_bytes(hashlib.blake2b(line.strip().encode(), digest_size=8).digest(), "big"))
    return digests


def duplicate_stats(files_budgets) -> dict:
    """整行精确重复统计（切分内 + 两两跨切分）。预算相关量，随指纹固化（spec 2.5）。"""
    sets = {}
    stats = {}
    for tag, (path, budget) in files_budgets.items():
        sets[tag] = _row_digests(path, budget)
        stats[f"{tag}_within"] = int(budget) - len(sets[tag])
    for a, b in itertools.combinations(files_budgets.keys(), 2):
        stats[f"{a}_{b}"] = len(sets[a] & sets[b])
    return stats


def _header_sha256(path) -> str:
    with open(path, "rb") as handle:
        return sha256_bytes(handle.readline())


def _normalize_counts(counts: dict) -> dict:
    """JSON 往返后 bsi_raw 键会变成字符串；归一化后再比较。"""
    return {
        "n": counts["n"],
        "click1": counts["click1"],
        "purchase1": counts["purchase1"],
        "bsi_pos": counts["bsi_pos"],
        "bsi_raw": {str(k): v for k, v in counts["bsi_raw"].items()},
    }


def fingerprint_digest(fp: dict) -> str:
    body = {k: v for k, v in fp.items() if k != "fingerprint_sha256"}
    return sha256_bytes(canonical_json(body).encode())


def build_fingerprint(prefix_tag: str, files_budgets) -> dict:
    files, label_counts, budgets = {}, {}, {}
    for tag, (path, budget) in files_budgets.items():
        files[tag] = {
            "path": str(path),
            "size_bytes": os.path.getsize(path),
            "prefix_sha256": prefix_sha256(path, budget),
        }
        label_counts[tag] = scan_label_counts(path, budget)
        budgets[tag] = int(budget)
    first_tag = next(iter(files_budgets))
    fp = {
        "prefix_tag": prefix_tag,
        "budgets": budgets,
        "files": files,
        "header_sha256": _header_sha256(files_budgets[first_tag][0]),
        "label_counts": label_counts,
        "duplicate_stats": duplicate_stats(files_budgets),
    }
    fp["fingerprint_sha256"] = fingerprint_digest(fp)
    return fp


def verify_fingerprint(fp: dict) -> None:
    """A2：前缀字节、表头、标签计数与自哈希必须与指纹一致；不一致抛 AssertionError。"""
    if fp.get("fingerprint_sha256") != fingerprint_digest(fp):
        raise AssertionError("指纹自哈希不一致（字段被改动）")
    for tag, meta in fp["files"].items():
        budget = fp["budgets"][tag]
        if prefix_sha256(meta["path"], budget) != meta["prefix_sha256"]:
            raise AssertionError(f"前缀字节与指纹不符: {tag} ({meta['path']})")
        counts = scan_label_counts(meta["path"], budget)
        if _normalize_counts(counts) != _normalize_counts(fp["label_counts"][tag]):
            raise AssertionError(f"原始扫描标签计数与指纹不符: {tag}")
        if _header_sha256(meta["path"]) != fp["header_sha256"]:
            raise AssertionError(f"表头与指纹不符: {tag}")


def splits_dir(root, prefix_tag: str) -> Path:
    return Path(root) / "splits" / prefix_tag


def save_fingerprint(root, prefix_tag: str, fp: dict) -> Path:
    path = splits_dir(root, prefix_tag) / "prefix_fingerprint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_fingerprint(root, prefix_tag: str) -> dict:
    path = splits_dir(root, prefix_tag) / "prefix_fingerprint.json"
    if not path.exists():
        raise FileNotFoundError(f"前缀指纹不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_fingerprint(root, prefix_tag: str, files_budgets):
    """存在则读取并校验，不存在则构建、校验、落盘。返回 (fp, created)。"""
    path = splits_dir(root, prefix_tag) / "prefix_fingerprint.json"
    if path.exists():
        fp = load_fingerprint(root, prefix_tag)
        verify_fingerprint(fp)
        return fp, False
    fp = build_fingerprint(prefix_tag, files_budgets)
    verify_fingerprint(fp)
    save_fingerprint(root, prefix_tag, fp)
    return fp, True


def run_dir(root, run_id: str) -> Path:
    return Path(root) / "runs" / run_id


def stage1_id_for(meta: dict) -> str:
    return stage1_id(fingerprint_sha=meta["fingerprint_sha256"], model_seed=meta["model_seed"],
                     epochs=meta["epochs"], cfg_sha=meta["config_hash"])


# ---- 加载器与模型构造（原 bench.py）----
def build_loaders(data_files, budgets, batch_size):
    datasets = {split: AliCCPDataset(data_files[split], budgets[split]) for split in ("train", "val", "test")}
    loaders = {
        split: DataLoader(datasets[split], batch_size=batch_size, shuffle=False, num_workers=0)
        for split in ("train", "val", "test")
    }
    return datasets, loaders


def build_mptrec(device, *, vocab=None, expert_hidden=EXPERT_HIDDEN, tower_hidden=TOWER_HIDDEN,
                 embedding_size=EMBEDDING_SIZE, input_size=INPUT_SIZE,
                 reg_embedding=REG_EMBEDDING, reg_dnn=REG_DNN, dropout=DROPOUT) -> MPTRec:
    vocab = dict(vocab) if vocab is not None else build_vocab()
    return MPTRec(
        num_tasks=NUM_TASKS,
        feature_vocabulary=vocab,
        embedding_size=embedding_size,
        input_size=input_size,
        expert_dnn_hidden_units=list(expert_hidden),
        tower_dnn_hidden_units=list(tower_hidden),
        dropout=list(dropout),
        reg_embedding=reg_embedding,
        reg_dnn=reg_dnn,
        device=device,
    )


def _budget_of(datasets, budgets) -> bool:
    return all(len(datasets[split]) == budgets[split] for split in ("train", "val", "test"))


def _versions() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}


def _reset_peak_vram(device) -> None:
    """torch 2.6 在 CUDA 未初始化时，_cuda_resetPeakMemoryStats 对任何实参都报 Invalid device argument；
    必须先 torch.cuda.init()（实测确认，2026-10-03 smoke）。"""
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)


@torch.no_grad()
def env_accuracy_probe(model, loader, env_ids, batch_size, device, max_batches=200) -> float:
    """M1：训练集前 400k 行（前 200 个 batch，确定性）上 env_pred 与最终 env_ids 的一致率。"""
    model.eval()
    correct = total = 0
    for step, (_, _, _, features) in enumerate(loader):
        if step >= max_batches:
            break
        for key in features:
            features[key] = features[key].to(device)
        env_pred = model(features)["env_pred"]
        ids = env_ids[batch_size * step : batch_size * (step + 1)].to(device)
        n = len(ids)
        correct += int((env_pred.argmax(dim=1)[:n] == ids).sum())
        total += n
    return correct / max(1, total)


@torch.no_grad()
def evaluate_newtask(newtask, model, loader, device) -> float:
    """阶段 2 评测口径与仓库 AliCCP_NewTask.py 的 evaluation() 一致（spec 7.5）。"""
    newtask.eval()
    y_true, y_hat = [], []
    for _, _, y, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
        pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
        y_true.append(y)
        y_hat.append(pred)
    return auc(torch.cat(y_true), torch.cat(y_hat))


@torch.no_grad()
def newtask_gate_mean(newtask, model, loader, device) -> list:
    """M3：val 上 NewTask.gate_network 的平均输出（2 维），no_grad 累积（spec 9.2）。"""
    newtask.eval()
    total = None
    count = 0
    for _, _, _, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        dnn_input, _, _, _ = model.get_infos(features)
        gate = newtask.gate_network(dnn_input)
        total = gate.sum(dim=0) if total is None else total + gate.sum(dim=0)
        count += gate.shape[0]
    return [float(v) / max(1, count) for v in total]


# ---- 门禁判定（原 aliccp_benchmark.metrics；spec 10）----
def _a_outcomes(facts: dict) -> list:
    """A 类协议正确性门禁（spec 10）。A3 为按需复跑，首轮 SKIP。"""
    a1_ok = facts["backbone_sha_before"] == facts["backbone_sha_after"] and facts["backbone_grads_none"]
    a2_ok = facts["prefix_sha_ok"] and facts["fingerprint_sha_match"]
    a4_ok = facts["len_ok"] and facts["counts_ok"]
    a5_ok = facts["env_ids_sha_match"]
    a6_ok = facts["backbone_sha_matches_stage1"] and facts["stage1_id_recorded"]
    return gates.evaluate([
        ("A1", a1_ok, f"sha_before==sha_after:{facts['backbone_sha_before'] == facts['backbone_sha_after']}, "
                      f"grads_none:{facts['backbone_grads_none']}"),
        ("A2", a2_ok, f"prefix_sha_ok:{facts['prefix_sha_ok']}, fingerprint_sha_match:{facts['fingerprint_sha_match']}"),
        gates.GateOutcome("A3", gates.SKIP, "按需复跑（spec 10，不阻塞首轮）"),
        ("A4", a4_ok, f"len_ok:{facts['len_ok']}, counts_ok:{facts['counts_ok']}"),
        ("A5", a5_ok, f"env_ids_sha_match:{facts['env_ids_sha_match']}"),
        ("A6", a6_ok, f"backbone_sha_matches_stage1:{facts['backbone_sha_matches_stage1']}, "
                      f"stage1_id_recorded:{facts['stage1_id_recorded']}"),
    ])


def _b_outcomes(facts: dict) -> list:
    """B 类活性与机制门禁（spec 10；smoke 只记录不判定）。"""
    ctr_ok = facts["auc_val_ctr"] >= AUC_FLOOR_CTR
    cvr_ok = facts["auc_val_cvr"] >= AUC_FLOOR_CVR
    bsi_ok = facts["auc_test_bsi"] >= AUC_FLOOR_BSI
    b1_ok = ctr_ok and cvr_ok and bsi_ok

    gap = abs(facts["auc_val_bsi_best"] - facts["auc_test_bsi"])
    b2_ok = gap <= VAL_TEST_GAP_BSI

    gate_mean = facts["gate_mean"]
    b3_ok = bool(gate_mean) and all(GATE_MIN <= float(g) <= GATE_MAX for g in gate_mean)

    events = facts["cluster_events"]
    train_size = facts.get("train_size", 0)
    if not events:
        b4 = gates.GateOutcome("B4", gates.NA, "无 cluster 调用（0 次，未判定）")
    else:
        share = ENV_SHARE_MIN * train_size
        shares = [(e["env_0"], e["env_1"]) for e in events]
        b4_ok = all(e0 >= share and e1 >= share for e0, e1 in shares)
        b4 = gates.GateOutcome("B4", gates.PASS if b4_ok else gates.FAIL,
                               f"env 占比下限 {ENV_SHARE_MIN:.0%}；events={shares}")

    return gates.evaluate([
        ("B1", b1_ok, f"CTR {facts['auc_val_ctr']:.4f}>={AUC_FLOOR_CTR}:{ctr_ok}, "
                      f"CVR {facts['auc_val_cvr']:.4f}>={AUC_FLOOR_CVR}:{cvr_ok}, "
                      f"BSI {facts['auc_test_bsi']:.4f}>={AUC_FLOOR_BSI}:{bsi_ok}"),
        ("B2", b2_ok, f"|val-test|={gap:.4f} <= {VAL_TEST_GAP_BSI}"),
        ("B3", b3_ok, f"gate_mean={list(gate_mean)} ∈ [{GATE_MIN}, {GATE_MAX}]"),
        b4,
    ])


def evaluate_a_gates(facts: dict) -> dict:
    return gates.render_verdict_detail(_a_outcomes(facts))


def evaluate_b_gates(facts: dict) -> dict:
    return gates.render_verdict_detail(_b_outcomes(facts))


def hard_pass(a_gates: dict, enforce_b: bool, b_gates: dict | None = None) -> bool:
    """A 类必须全 PASS（A3 SKIP 视为通过）；enforce_b 时 B 类必须 PASS 或 N/A。"""
    a_ok = gates.hard_pass((g["verdict"] for g in a_gates.values()), allowed=(gates.PASS, gates.SKIP))
    if not enforce_b:
        return a_ok
    return a_ok and b_gates is not None and gates.hard_pass(
        (g["verdict"] for g in b_gates.values()), allowed=(gates.PASS, gates.NA))


def build_profile(*, prefix_tag: str = PREFIX_TAG, data_files=None, budgets=None,
                  model_seed: int = MODEL_SEED, env_seed: int = ENV_SEED,
                  stage1_epochs: int = STAGE1_EPOCHS, stage1_patience: int = STAGE1_PATIENCE,
                  stage2_epochs: int = STAGE2_EPOCHS, stage2_patience: int = STAGE2_PATIENCE,
                  vocab=None, expert_hidden=EXPERT_HIDDEN, tower_hidden=TOWER_HIDDEN,
                  embedding_size: int = EMBEDDING_SIZE, input_size: int = INPUT_SIZE,
                  batch_size: int = BATCH_SIZE, lr: float = LR, uni_coe: float = UNI_COE,
                  env_coe: float = ENV_COE, reg_embedding: float = REG_EMBEDDING,
                  reg_dnn: float = REG_DNN, dropout=DROPOUT, newtask_rep_dim=None,
                  enforce_b: bool = True, log_batches: int = 200, log=print) -> DatasetProfile:
    """aliccp 两阶段适配；data_files/budgets/vocab 与模型结构参数供 tiny e2e 测试注入。

    enforce_b 只影响阶段 2 门禁与落盘字段；log 只接管阶段 1 控制台输出（阶段 2 用 pipeline 的 log）。
    """
    resolved_vocab = dict(vocab) if vocab is not None else build_vocab()
    resolved_files = dict(DATA_FILES) if data_files is None else dict(data_files)
    resolved_budgets = ({"train": TRAIN_BUDGET, "val": VAL_BUDGET, "test": TEST_BUDGET}
                        if budgets is None else dict(budgets))
    rep_dim = int(newtask_rep_dim) if newtask_rep_dim is not None else int(list(expert_hidden)[-1])

    def prepare(root):
        t0 = time.time()
        log(f"[stage1] 开始：prefix={prefix_tag} budgets={resolved_budgets} "
            f"model_seed={model_seed} env_seed={env_seed}")
        datasets, loaders = build_loaders(resolved_files, resolved_budgets, batch_size)
        files_budgets = {split: (Path(resolved_files[split]), resolved_budgets[split])
                         for split in ("train", "val", "test")}
        fp, created = ensure_fingerprint(root, prefix_tag, files_budgets)
        log(f"[stage1] 前缀指纹 {'构建并落盘' if created else '读取校验'}：{fp['fingerprint_sha256'][:16]}")
        if not _budget_of(datasets, resolved_budgets):
            raise AssertionError("A4: 样本数与预算不符")
        for split in ("train", "val", "test"):
            verify_label_counts(datasets[split], resolved_budgets[split], fp["label_counts"][split])
        log("[stage1] A4 通过：三切分样本数与标签计数与指纹逐项相等")
        return {"loaders": loaders, "datasets": datasets, "fp": fp, "budgets": resolved_budgets,
                "data_files": resolved_files, "t0": t0}

    def build_model(device):
        _reset_peak_vram(device)
        return build_mptrec(device, vocab=resolved_vocab, expert_hidden=expert_hidden,
                            tower_hidden=tower_hidden, embedding_size=embedding_size,
                            input_size=input_size, reg_embedding=reg_embedding,
                            reg_dnn=reg_dnn, dropout=dropout)

    def manager_kwargs(ctx, epochs):
        ld = ctx["loaders"]
        return {"train_loader": ld["train"], "val_loader": ld["val"], "task_name": ["CTR", "CVR"],
                "lr": lr, "batch_size": batch_size, "uni_coe": uni_coe, "env_coe": env_coe,
                "epochs": epochs, "patience": stage1_patience,
                "record_env_acc": False, "cluster_epoch_offset": 1}

    def post_train(model, manager, ctx, device):
        test_aucs = manager.evaluation_two_task(ctx["loaders"]["test"])
        env_acc = env_accuracy_probe(model, ctx["loaders"]["train"], manager.env_ids,
                                     batch_size, device, log_batches)
        return {"test_aucs": test_aucs, "env_acc": env_acc}

    def build_cfg(ctx, epochs):
        return {
            "prefix_tag": prefix_tag,
            "budgets": dict(ctx["budgets"]),
            "model_seed": int(model_seed),
            "env_seed": int(env_seed),
            "epochs": int(epochs),
            "patience": int(stage1_patience),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "uni_coe": float(uni_coe),
            "env_coe": float(env_coe),
            "reg_embedding": float(reg_embedding),
            "reg_dnn": float(reg_dnn),
            "embedding_size": int(embedding_size),
            "input_size": int(input_size),
            "expert_hidden": list(expert_hidden),
            "tower_hidden": list(tower_hidden),
            "dropout": [float(d) for d in dropout],
            "num_tasks": NUM_TASKS,
            "vocab": {str(k): int(v) for k, v in sorted(resolved_vocab.items())},
        }

    def build_meta(*, sid, cfg, cfg_sha, ctx, model, manager, env_ids, post, commit, epochs):
        per_epoch = []
        for i in range(len(manager.val_epoch_aucs)):
            per_epoch.append({
                "epoch": i + 1,
                "auc_val_ctr": manager.val_epoch_aucs[i][0],
                "auc_val_cvr": manager.val_epoch_aucs[i][1],
                "uni_loss_0": float(manager.uni_loss_0_list[i]),
                "uni_loss_1": float(manager.uni_loss_1_list[i]),
                "fuse_loss_0": float(manager.fused_loss_0_list[i]),
                "fuse_loss_1": float(manager.fused_loss_1_list[i]),
                "env_loss": float(manager.env_loss_list[i]),
            })
        best_epoch = manager.best_epoch()
        device = next(model.parameters()).device
        return {
            "stage1_id": sid,
            "prefix_tag": prefix_tag,
            "budgets": dict(ctx["budgets"]),
            "model_seed": int(model_seed),
            "env_seed": int(env_seed),
            "epochs": int(epochs),
            "patience": int(stage1_patience),
            "config_hash": cfg_sha,
            "config": cfg,
            "fingerprint_sha256": ctx["fp"]["fingerprint_sha256"],
            "per_epoch": per_epoch,
            "best_epoch": best_epoch,
            "best_val_auc_ctr": manager.val_epoch_aucs[best_epoch - 1][0],
            "best_val_auc_cvr": manager.val_epoch_aucs[best_epoch - 1][1],
            "test_auc_ctr": float(post["test_aucs"][0]),
            "test_auc_cvr": float(post["test_aucs"][1]),
            "env_acc": float(post["env_acc"]),
            "cluster_events": [{"epoch": r["epoch"], "diff_num": r["diff_num"],
                                "env_0": r["env_counts"][0], "env_1": r["env_counts"][1]}
                               for r in manager.cluster_records],
            "backbone_sha256": backbone_sha256(model),
            "env_ids_sha256": sha256_tensor(env_ids),
            "commit": commit,
            "git": git_state(),
            "versions": _versions(),
            "device": str(device),
            "wall_seconds": round(time.time() - ctx["t0"], 1),
            "peak_vram_mb": (round(torch.cuda.max_memory_allocated(device.index) / 1e6, 1)
                             if device.type == "cuda" else None),
        }

    def log_summary(sid, path, meta):
        log(f"[stage1] 完成：stage1_id={sid} best_epoch={meta['best_epoch']} "
            f"test_auc_ctr={meta['test_auc_ctr']:.4f} test_auc_cvr={meta['test_auc_cvr']:.4f} "
            f"env_acc={meta['env_acc']:.4f} wall={meta['wall_seconds']}s")

    # ---- 阶段 2 ----
    def prepare_stage2(root, meta, device):
        t0 = time.time()
        datasets, loaders = build_loaders(resolved_files, resolved_budgets, batch_size)
        log_lines = []
        prefix_sha_ok = True
        fp = None
        try:
            fp = load_fingerprint(root, prefix_tag)
            verify_fingerprint(fp)
        except (AssertionError, FileNotFoundError) as exc:  # noqa: BLE001
            prefix_sha_ok = False
            log_lines.append(f"[stage2] A2 前缀指纹校验失败：{exc}")
            if fp is None:
                fp = {"fingerprint_sha256": None, "label_counts": {}}
        len_ok = _budget_of(datasets, resolved_budgets)
        counts_ok = True
        for split in ("train", "val", "test"):
            try:
                verify_label_counts(datasets[split], resolved_budgets[split], fp["label_counts"][split])
            except (AssertionError, KeyError) as exc:  # noqa: BLE001
                counts_ok = False
                log_lines.append(f"[stage2] A4 标签计数失败（{split}）：{exc}")
        return {"loaders": loaders, "datasets": datasets, "model_seed": model_seed,
                "prefix_tag": prefix_tag, "budgets": resolved_budgets, "data_files": resolved_files,
                "fp": fp, "prefix_sha_ok": prefix_sha_ok, "len_ok": len_ok, "counts_ok": counts_ok,
                "rep_dim": rep_dim, "t0": t0, "log_lines": log_lines}

    def log_stage2_start(sid, tag, ctx2, log):
        ctx2["sid"] = sid                       # backbone_ready 的 A6 需要比对的请求 id
        log(f"[stage2] 开始：stage1_id={sid} tag={tag} model_seed={ctx2['model_seed']} "
            f"enforce_b={enforce_b}")
        for line in ctx2["log_lines"]:
            log(line)

    def backbone_ready(backbone, checkpoint, ctx2, log):
        art_meta = checkpoint["meta"]
        sha_loaded = backbone_sha256(backbone)
        matches = sha_loaded == art_meta.get("backbone_sha256")
        log(f"[stage2] backbone 已加载并冻结：sha={sha_loaded[:16]}（A6 匹配={matches}）")
        return {"backbone_sha_loaded": sha_loaded, "backbone_sha_matches_stage1": matches,
                "stage1_id_recorded": art_meta.get("stage1_id") == ctx2["sid"],
                "fingerprint_sha_match":
                    art_meta.get("fingerprint_sha256") == ctx2["fp"].get("fingerprint_sha256")}

    def build_newtask(device, ctx2):
        if ctx2["rep_dim"] != int(list(expert_hidden)[-1]):
            raise AssertionError("NewTask rep_dim 必须等于 expert_dnn_hidden_units[-1]（env_embs 维度约束）")
        return NewTask(input_size=input_size, rep_dim=ctx2["rep_dim"],
                       tower_dnn_hidden_units=list(tower_hidden), reg_dnn=reg_dnn,
                       device=device).to(device)

    def log_stage2_epoch(epoch, loss, auc_val, stale, stopped, log):
        log(f"[stage2] Epoch:{epoch} train_loss={loss:.4f} AUC-Val-BSI:{auc_val:.4f}")
        if stale:
            log(f"[stage2] EarlyStopping count {stale}")
        if stopped:
            log(f"[stage2] EarlyStopping at epoch {epoch}")

    def final_eval(newtask, backbone, ctx2, device):
        return {"test_auc": evaluate_newtask(newtask, backbone, ctx2["loaders"]["test"], device),
                "gate_mean": newtask_gate_mean(newtask, backbone, ctx2["loaders"]["val"], device)}

    def build_gates(final, ctx2, meta, facts, best_auc, log):
        a_gates = evaluate_a_gates({
            "backbone_sha_before": facts["sha_before"],
            "backbone_sha_after": facts["sha_after"],
            "backbone_grads_none": facts["grads_all_none"],
            "prefix_sha_ok": ctx2["prefix_sha_ok"],
            "fingerprint_sha_match": facts["fingerprint_sha_match"],
            "len_ok": ctx2["len_ok"],
            "counts_ok": ctx2["counts_ok"],
            "env_ids_sha_match": facts["env_ids_ok"],
            "backbone_sha_matches_stage1": facts["backbone_sha_matches_stage1"],
            "stage1_id_recorded": facts["stage1_id_recorded"],
        })
        b_gates = evaluate_b_gates({
            "auc_val_ctr": meta.get("best_val_auc_ctr", 0.0),
            "auc_val_cvr": meta.get("best_val_auc_cvr", 0.0),
            "auc_val_bsi_best": best_auc,
            "auc_test_bsi": final["test_auc"],
            "gate_mean": final["gate_mean"],
            "cluster_events": meta.get("cluster_events", []),
            "train_size": ctx2["budgets"]["train"],
        })
        return {**a_gates, **b_gates}, hard_pass(a_gates, enforce_b, b_gates)

    def write_artifacts(run_path, bundle):
        ctx2, meta, facts, final = bundle["ctx2"], bundle["meta"], bundle["facts"], bundle["final"]
        run_id = bundle["run_id"]
        torch.save({k: v.detach().cpu() for k, v in bundle["newtask"].state_dict().items()},
                   run_path / "newtask.pt")
        wall = round(time.time() - ctx2["t0"], 1)
        metrics_doc = {
            "run_id": run_id,
            "tag": bundle["tag"],
            "stage1_id": bundle["stage1_id"],
            "prefix_tag": ctx2["prefix_tag"],
            "budgets": dict(ctx2["budgets"]),
            "model_seed": int(ctx2["model_seed"]),
            "epochs": int(bundle["epochs"]),
            "patience": int(stage2_patience),
            "enforce_b": bool(enforce_b),
            "best_epoch": bundle["best_epoch"],
            "best_val_auc_bsi": float(bundle["best_auc"]),
            "test_auc_bsi": float(final["test_auc"]),
            "gate_mean": final["gate_mean"],
            "per_epoch": bundle["epoch_records"],
            "backbone_sha256_loaded": facts["backbone_sha_loaded"],
            "backbone_sha256_before": facts["sha_before"],
            "backbone_sha256_after": facts["sha_after"],
            "backbone_grads_none": facts["grads_all_none"],
            "env_ids_sha256": sha256_tensor(bundle["env_ids"]),
            "fingerprint_sha256": ctx2["fp"].get("fingerprint_sha256"),
            "hard_pass": bool(bundle["passed"]),
            "commit": bundle["commit"],
            "git": git_state(),
            "versions": _versions(),
            "device": str(bundle["device"]),
            "wall_seconds": wall,
            "peak_vram_mb": (round(torch.cuda.max_memory_allocated(bundle["device"].index) / 1e6, 1)
                             if bundle["device"].type == "cuda" else None),
        }
        config_doc = {
            "run_id": run_id,
            "tag": bundle["tag"],
            "stage1_id": bundle["stage1_id"],
            "data_files": {k: str(v) for k, v in ctx2["data_files"].items()},
            "budgets": dict(ctx2["budgets"]),
            "model_seed": int(ctx2["model_seed"]),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "reg_dnn": float(reg_dnn),
            "newtask_rep_dim": ctx2["rep_dim"],
            "expert_hidden": list(expert_hidden),
            "tower_hidden": list(tower_hidden),
            "input_size": int(input_size),
            "embedding_size": int(embedding_size),
            "enforce_b": bool(enforce_b),
            "commit": bundle["commit"],
            "git": git_state(),
        }
        gate_doc = {"run_id": run_id, "tag": bundle["tag"], "enforce_b": bool(enforce_b),
                    "gates": bundle["gates_doc"], "hard_pass": bool(bundle["passed"])}
        write_json(run_path / "metrics.json", metrics_doc)
        write_json(run_path / "config.json", config_doc)
        write_json(run_path / "gate_report.json", gate_doc)
        append_summary_row(bundle["root"] / "SUMMARY.md", {
            "run_id": run_id, "commit": bundle["commit"], "tag": bundle["tag"],
            "auc_val_bsi_best": f"{bundle['best_auc']:.6f}",
            "auc_test_bsi": f"{final['test_auc']:.6f}",
            "stage1_id": bundle["stage1_id"],
            **{gate_id: bundle["gates_doc"][gate_id]["verdict"]
               for gate_id in ("A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4")},
        }, SUMMARY_COLUMNS)
        ctx2["wall_seconds"] = wall
        return {"gates": bundle["gates_doc"], "metrics": metrics_doc,
                "hard_pass": bool(bundle["passed"])}

    def log_stage2_end(bundle, log):
        log(f"[stage2] 完成：run_id={bundle['run_id']} "
            f"AUC-Val-BSI(best)={bundle['best_auc']:.4f} "
            f"AUC-Test-BSI={bundle['final']['test_auc']:.4f} hard_pass={bundle['passed']} "
            f"wall={bundle['ctx2']['wall_seconds']}s")

    return DatasetProfile(
        name="aliccp", model_seed=model_seed, env_seed=env_seed,
        stage1_epochs=stage1_epochs, stage2_epochs=stage2_epochs,
        stage2_patience=stage2_patience, stage2_lr=lr,
        stage2_cpu_loss=True, stage2_catch_grads=True,
        prepare=prepare, build_model=build_model,
        train_size=lambda ctx: len(ctx["datasets"]["train"]),
        manager_kwargs=manager_kwargs, post_train=post_train, build_cfg=build_cfg,
        fingerprint_sha=lambda ctx: ctx["fp"]["fingerprint_sha256"],
        state_dict=lambda model: {k: v.detach().cpu() for k, v in model.state_dict().items()},
        build_meta=build_meta, after_save=lambda stage1_path, log_buffer: None,
        log_summary=log_summary, prepare_stage2=prepare_stage2,
        log_stage2_start=log_stage2_start, backbone_ready=backbone_ready,
        build_newtask=build_newtask,
        val_auc=lambda newtask, backbone, loader, device: evaluate_newtask(newtask, backbone, loader, device),
        epoch_record=lambda epoch, loss, auc_val: {"epoch": epoch, "train_loss": loss, "val_auc_bsi": auc_val},
        log_stage2_epoch=log_stage2_epoch, final_eval=final_eval,
        log_grads_failure=lambda exc, log: log(f"[stage2] A1 失败：{exc}"),
        build_gates=build_gates, run_id_prefix=lambda ctx2: ctx2["prefix_tag"],
        write_artifacts=write_artifacts, log_stage2_end=log_stage2_end)
