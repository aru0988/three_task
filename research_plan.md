# T4 借鉴12 vs 借鉴123 实验 — 完整执行计划

## 实验目的

回答：T4 的 prompt-tuning 同时关注 [E_T1, E_T2, E_T3] 是否比只关注 [E_T1, E_T2] 更好？

两个 T4（Sex 和 Race）覆盖不同的相关性格局，与 source task 的 Pearson 相关性（已实测）：

```
            Income   Marital   Education   Sex      Race
----------------------------------------------------------
 Income        -      0.005     0.222     0.110   -0.024
Marital     0.005       -       0.006    -0.001    0.002
Education   0.222     0.006        -       0.015   -0.022
    Sex     0.110    -0.001     0.015        -     -0.002
   Race    -0.024     0.002    -0.022    -0.002       -
```

Sex：T1 有信号 (0.11)，T3 无用 (~0.01) — 测"噪声 source 是否干扰已有有用的 attention"
Race：全无用 (max 0.02) — 测"没有信号时 Prompt 是否学到虚假 attention，KL 能否自动退化为 FW"

## 实验设计

使用**同一个 3-task backbone**（T1 Income + T2 Marital + T3 Education），冻结不变：

```
条件 A（借鉴12）：NewTask 只关注前 2 个 source task
    attention 候选池 = [E_T1, E_T2], spec 池 = [spec_0, spec_1]

条件 B（借鉴123）：NewTask 关注全部 3 个 source task
    attention 候选池 = [E_T1, E_T2, E_T3], spec 池 = [spec_0, spec_1, spec_2]
```

唯一变量：T4 的 softmax 多了一个 E_T3 候选项。

三种 fusion mode：`prompt`、`kl_prompt`（β=0.1）、`fw`。
FW 作为下界，Prompt 作为原始 attention，KL-Prompt 作为安全上限。

## 第一步：修改代码

### 1.1 multitaskrec/dataset.py — 添加 CensusIncome4TaskDataset

在文件末尾追加以下类（放在 `if __name__ == "__main__"` 之前，最后一个已有 class 之后）：

```python
class CensusIncome4TaskDataset(Dataset):
    """4-label dataset for T4 experiment.

    Returns (income, marital, education, new_task_label, features).
    Removes both 'education' and new_task from features.
    Supports new_task='sex' or new_task='race'.
    """
    def __init__(self, datafile, new_task):
        self.feature_names = [
            "age", "class_worker", "det_ind_code", "det_occ_code", "education",
            "wage_per_hour", "hs_college", "major_ind_code", "major_occ_code",
            "race", "hisp_origin", "sex", "union_member", "unemp_reason",
            "full_or_part_emp", "capital_gains", "capital_losses", "stock_dividends",
            "tax_filer_stat", "region_prev_res", "state_prev_res", "det_hh_fam_stat",
            "det_hh_summ", "instance_weight", "mig_chg_msa", "mig_chg_reg",
            "mig_move_reg", "mig_same", "mig_prev_sunbelt", "num_emp", "fam_under_18",
            "country_father", "country_mother", "country_self", "citizenship",
            "own_or_self", "vet_question", "vet_benefits", "weeks_worked", "year",
        ]
        import pandas as pd
        df = pd.read_csv(datafile, delimiter=",")

        self.feature_names.remove("education")
        df["label_education"] = df["education"].apply(lambda x: 1 if x == 9 else 0)
        df.pop("education")

        self.feature_names.remove(new_task)
        if new_task == "sex":
            df[f"label_{new_task}"] = df[new_task]
        elif new_task == "race":
            df[f"label_{new_task}"] = df[new_task].apply(lambda x: 1 if x == 0 else 0)
        else:
            raise ValueError(f"Unsupported new task: {new_task}")
        df.pop(new_task)

        self.data = df.values
        self.new_task = new_task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        income = line[-4]
        marital = line[-3]
        education = line[-2]
        new_task_label = line[-1]
        features = dict(zip(self.feature_names, line[:-4]))
        return income, marital, education, new_task_label, features
```

验证：Python 能 import 且不报错。

### 1.2 multitaskrec/model.py — NewTask 增加 num_source_tasks 参数

修改 `NewTask.__init__`，增加参数 `num_source_tasks=None`（None 表示不限制，即使用全部 source task）：

找到 `def __init__(self, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None, fusion_mode="prompt", ...)` 这一行，在最后追加 `, num_source_tasks=None`。

在 `__init__` 方法体内（`self.fusion_mode = fusion_mode` 之后）添加：
```python
self.num_source_tasks = num_source_tasks  # None = use all, K = use first K only
```

修改 `forward` 方法。在 `exist_env_embs = torch.stack(env_embs, dim=1)` 这一行之后（约984行），`num_tasks = len(spec_reps)` 之前，添加：

```python
if self.num_source_tasks is not None:
    exist_env_embs = exist_env_embs[:, :self.num_source_tasks, :]
    spec_reps = spec_reps[:self.num_source_tasks]
    num_tasks = self.num_source_tasks
else:
    num_tasks = len(spec_reps)
```

注意：这会改变后续 `num_tasks` 变量。检查原有代码中 `num_tasks = len(spec_reps)` 的位置（约986行），确保替换后不冲突——需要把对应的原有赋值删掉或包裹在 else 分支里。

原有代码约 986 行：`num_tasks = len(spec_reps)` — 这一行需要被包裹进 else 分支（见上面代码块的 else 部分）。

验证逻辑：

- `num_source_tasks=None`（默认）：行为完全不变，向后兼容所有已有实验
- `num_source_tasks=2`：只取前 2 个 source task 的 embedding 和 spec_rep 做 attention
- `num_source_tasks=3`：取全部 3 个

### 1.3 新建 run_t4_experiment.py

创建文件 `run_t4_experiment.py` 在项目根目录。完整内容见下方（独立创建）。

**核心逻辑**：

```
Stage 1（仅 run_stage1）：
  - 创建 CensusIncome4TaskDataset(datafile, t4)
  - 创建 MPTRec(num_tasks=3)
  - 训练 3-task backbone
  - 保存 ckpt 到 checkpoints/t4_experiment/
  - 记录 T1/T2/T3 AUC

Stage 2（仅 run_stage2）：
  - 加载 3-task backbone ckpt
  - 条件 A：创建 NewTask(num_source_tasks=2)
  - 条件 B：创建 NewTask(num_source_tasks=3)
  - 各跑 prompt / kl_prompt / fw 三种 mode × 3 seeds
  - 输出 T4 Test AUC
  - 输出格式：RESULT|t4=<sex/race>|cond=<A/B>|mode=<prompt/kl_prompt/fw>|seed=<N>|test=<auc>
```

**脚本结构**：

```python
"""
T4 experiment: compare 2-source vs 3-source attention for learning T4.

Usage:
    # Full sweep
    python run_t4_experiment.py --all --all-seeds --gpu 0

    # Stage 1 only
    python run_t4_experiment.py --stage1 --t4 sex --all-seeds --gpu 0
    python run_t4_experiment.py --stage1 --t4 race --all-seeds --gpu 0

    # Stage 2 only (requires Stage 1 ckpts)
    python run_t4_experiment.py --stage2 --t4 sex --all-seeds --gpu 0
    python run_t4_experiment.py --stage2 --t4 race --all-seeds --gpu 0
"""
import argparse
import copy
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncome4TaskDataset
from multitaskrec.model import MPTRec, NewTask
from multitaskrec.train import MPTRecTrainManager

warnings.filterwarnings("ignore")

CENSUS_SEEDS = [1685480945, 1688723512, 1689453621]
ALL_MODES = ["prompt", "kl_prompt", "fw"]
CKPT_DIR = "checkpoints/t4_experiment"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Config ──────────────────────────────────────────────
def get_cfg(t4):
    vocab = CensusIncome_Vocabulary_Size.copy()
    vocab.pop("education")
    if t4 in vocab:
        vocab.pop(t4)
    return dict(
        vocabulary=vocab,
        embedding_size=4, input_size=123, rep_dim=128,
        expert_hidden=(256, 128), tower_hidden=(64, 32),
        reg_embedding=0.006, reg_dnn=3e-5,
        lr_stage1=1e-3, lr_stage2=1e-3,
        batch_size=256, epochs_stage1=10, epochs_stage2=30,
        patience=5,
        train_path="dataset/Census-income/train.gz",
        test_path="dataset/Census-income/test.gz",
    )


# ── Stage 1 ─────────────────────────────────────────────
def run_stage1(seed, t4, gpu):
    print(f"\n[Stage1] t4={t4} seed={seed}")
    sys.stdout.flush()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device(f"cuda:{gpu}")

    cfg = get_cfg(t4)
    train_ds = CensusIncome4TaskDataset(cfg["train_path"], t4)
    test_ds = CensusIncome4TaskDataset(cfg["test_path"], t4)
    val_ds, test_ds = train_test_split(test_ds, test_size=0.5, random_state=seed)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"])

    mptrec = MPTRec(
        num_tasks=3,
        feature_vocabulary=cfg["vocabulary"],
        embedding_size=cfg["embedding_size"],
        input_size=cfg["input_size"],
        expert_dnn_hidden_units=cfg["expert_hidden"],
        tower_dnn_hidden_units=cfg["tower_hidden"],
        reg_embedding=cfg["reg_embedding"],
        reg_dnn=cfg["reg_dnn"],
        device=device,
    ).to(device)

    env_ids = torch.randint(0, 3, size=(len(train_ds),))
    train_manager = MPTRecTrainManager(
        model=mptrec, train_loader=train_loader, val_loader=val_loader,
        env_ids=env_ids, task_name=['Income', 'Marital', 'Education'],
        lr=cfg["lr_stage1"], batch_size=cfg["batch_size"],
        uni_coe=0.9, env_coe=0.1, epochs=cfg["epochs_stage1"],
    )
    train_manager.train_three_task()
    mptrec.load_state_dict(train_manager.best_weight)

    ckpt_path = os.path.join(CKPT_DIR, f"backbone_3task_t4_{t4}_seed{seed}.pt")
    torch.save(mptrec.state_dict(), ckpt_path)
    print(f"  Saved: {ckpt_path}")

    aucs = train_manager.evaluation_three_task(test_loader)
    print(f"  T1_Income={aucs[0]:.4f}  T2_Marital={aucs[1]:.4f}  T3_Education={aucs[2]:.4f}")
    sys.stdout.flush()
    return ckpt_path


# ── Stage 2 ─────────────────────────────────────────────
@torch.no_grad()
def evaluate_t4(newtask, mptrec, data_loader):
    newtask.eval()
    device = next(newtask.parameters()).device
    t_true, t_hat = [], []
    for _, _, _, y_t4, features in data_loader:
        for k in features.keys():
            features[k] = features[k].to(device)
        dnn_in, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)
        pred = newtask(dnn_in, gen_rep, spec_reps, env_embs)
        t_true.append(y_t4)
        t_hat.append(pred)
    return roc_auc_score(torch.cat(t_true).int(), torch.cat(t_hat).cpu())


def run_stage2(seed, t4, cond, mode, gpu):
    """cond: "A" (2 source) or "B" (3 source)"""
    num_src = 2 if cond == "A" else 3
    print(f"\n[Stage2] t4={t4} cond={cond}({num_src}src) mode={mode} seed={seed}")
    sys.stdout.flush()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device(f"cuda:{gpu}")

    cfg = get_cfg(t4)
    train_ds = CensusIncome4TaskDataset(cfg["train_path"], t4)
    test_ds = CensusIncome4TaskDataset(cfg["test_path"], t4)
    val_ds, test_ds = train_test_split(test_ds, test_size=0.5, random_state=seed)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"])

    mptrec = MPTRec(
        num_tasks=3,
        feature_vocabulary=cfg["vocabulary"],
        embedding_size=cfg["embedding_size"],
        input_size=cfg["input_size"],
        expert_dnn_hidden_units=cfg["expert_hidden"],
        tower_dnn_hidden_units=cfg["tower_hidden"],
        reg_embedding=cfg["reg_embedding"],
        reg_dnn=cfg["reg_dnn"],
        device=device,
    ).to(device)

    ckpt_path = os.path.join(CKPT_DIR, f"backbone_3task_t4_{t4}_seed{seed}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run Stage 1 first.")
    mptrec.load_state_dict(torch.load(ckpt_path))
    mptrec.eval()

    newtask = NewTask(
        input_size=cfg["input_size"], rep_dim=cfg["rep_dim"],
        tower_dnn_hidden_units=list(cfg["tower_hidden"]),
        reg_dnn=cfg["reg_dnn"], device=device,
        fusion_mode=mode, num_source_tasks=num_src,
    ).to(device)

    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=cfg["lr_stage2"])
    loss_func = nn.BCELoss()
    best_val, best_w, earlystop = 0, None, 0

    for epoch in range(1, cfg["epochs_stage2"] + 1):
        newtask.train()
        for _, _, _, y_t4, features in train_loader:
            for k in features.keys():
                features[k] = features[k].to(device)
            dnn_in, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)
            pred = newtask(dnn_in, gen_rep, spec_reps, env_embs)
            loss = loss_func(pred.cpu(), y_t4.float()) + newtask.get_l2_reg()
            if mode == "kl_prompt":
                loss += 0.1 * newtask.kl_loss.to(loss.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_auc = evaluate_t4(newtask, mptrec, val_loader)
        arrow = " *" if val_auc > best_val else ""
        print(f"  Epoch {epoch:2d}/{cfg['epochs_stage2']}  val_auc={val_auc:.4f}{arrow}")
        sys.stdout.flush()

        if val_auc > best_val:
            earlystop, best_val, best_w = 0, val_auc, copy.deepcopy(newtask.state_dict())
        else:
            earlystop += 1
            if earlystop >= cfg["patience"]:
                print(f"  Early stop at epoch {epoch}")
                sys.stdout.flush()
                break

    newtask.load_state_dict(best_w)
    test_auc = evaluate_t4(newtask, mptrec, test_loader)
    print(f"RESULT|t4={t4}|cond={cond}|mode={mode}|seed={seed}|test={test_auc:.4f}")
    sys.stdout.flush()
    return test_auc


# ── Main ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", action="store_true")
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--t4", type=str, choices=["sex", "race"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.all:
        for t4 in ["sex", "race"]:
            for seed in CENSUS_SEEDS:
                run_stage1(seed, t4, args.gpu)
            for cond in ["A", "B"]:
                for mode in ALL_MODES:
                    for seed in CENSUS_SEEDS:
                        run_stage2(seed, t4, cond, mode, args.gpu)
        print("\n" + "=" * 60)
        print("T4 EXPERIMENT COMPLETE")
        print("=" * 60)
        return

    seeds = CENSUS_SEEDS if args.all_seeds else [args.seed]

    if args.stage1:
        for seed in seeds:
            run_stage1(seed, args.t4, args.gpu)

    if args.stage2:
        for cond in ["A", "B"]:
            for mode in ALL_MODES:
                for seed in seeds:
                    run_stage2(seed, args.t4, cond, mode, args.gpu)


if __name__ == "__main__":
    main()
```

**关键设计点**：

1. `num_source_tasks` 作为 NewTask 参数，默认 `None`（向后兼容）。Stage 2 条件 A 传 2，条件 B 传 3。
2. env_ids 初始化为 `torch.randint(0, 3, ...)`（3 个 task 的 env clustering）。
3. `make_vocab` 改为函数 `get_cfg(t4)`，包含完整配置。
4. Stage 2 的 evaluate 函数命名为 `evaluate_t4`，避免与现有 evaluate 冲突。
5. 条件 A/B 在 stage2 中自动遍历，不需要手动指定。

## 第二步：执行

```bash
# 进入项目
cd D:\MPT-Rec-three_task\MPT-Rec

# 激活环境
.venv\Scripts\activate

# 完整运行（推荐，~58 min）
python run_t4_experiment.py --all --all-seeds --gpu 0

# 或分步执行：
# Stage 1（先跑完所有 backbone，~30 min）
python run_t4_experiment.py --stage1 --t4 sex --all-seeds --gpu 0
python run_t4_experiment.py --stage1 --t4 race --all-seeds --gpu 0

# Stage 2（依赖 Stage 1 ckpt，~28 min）
python run_t4_experiment.py --stage2 --t4 sex --all-seeds --gpu 0
python run_t4_experiment.py --stage2 --t4 race --all-seeds --gpu 0
```

## 第三步：验证

运行完成后检查：

- [ ] `checkpoints/t4_experiment/` 下有 6 个 `.pt` 文件
  - `backbone_3task_t4_sex_seed1685480945.pt` 等 × 3
  - `backbone_3task_t4_race_seed1685480945.pt` 等 × 3
- [ ] Stage 1 输出 T1 ~0.94, T2 ~0.99, T3 ~0.86（与已有 CensusIncome 数据一致）
- [ ] Stage 2 输出 36 行 `RESULT|t4=...|cond=...|mode=...|seed=...|test=...`
- [ ] 条件 A vs B 的对比有明确模式（不是随机波动）
- [ ] `kl_prompt` 在 Race 上 ≤ FW 的波动范围，在 Sex 上 ≥ FW
- [ ] 汇总结果追加到 `logs/all_experiment_results.md`
