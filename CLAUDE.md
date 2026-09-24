# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

MPT-Rec 多任务推荐系统研究项目，复现论文实验。包含 MPTRec（主模型）和多个 baseline（SharedBottom、MMOE、PLE、STEM、SparseSharing、SingleTask），支持两个数据集（CensusIncome、AliCCP）和两个实验协议（联合训练、新任务泛化）。

## 分支约定（重要）

- `master`：**干净基线**，目录结构与官方 BAI-LAB/MPT-Rec `three_task` 分支一致，只含论文所需代码 + 运行修复。新想法必须从 `master` 拉分支开发
- `archive/exploration`：历史探索归档（TC-Prompt / CGR / affinity gate / KL-Prompt / T4 实验 / 论文草稿 / logs），**不要在 master 上重建这些模块**
- 工作流：想法分支 → 验证有效则合并回 `master`，无效则删除
- 任何超参改动都要**全项目检查一致性**（入口脚本 + baseline 是否同步），避免"只改一半"导致对比失真

## 运行环境

```bash
# 激活虚拟环境
.venv\Scripts\activate

# Python 3.10, PyTorch (CUDA), 单 GPU: NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM)
```

## 运行命令

```bash
# ===== 预训练（阶段 1）=====
python -m baseline.sharedbottom.CensusIncome_SharedBottom     # CensusIncome + SharedBottom
python -m baseline.mmoe.CensusIncome_MMOE                     # CensusIncome + MMOE
python -m baseline.ple.CensusIncome_PLE                       # CensusIncome + PLE
python baseline/stem/CensusIncome_STEM.py                     # CensusIncome + STEM

python -m baseline.sharedbottom.AliCCP_SharedBottom           # AliCCP + SharedBottom
python -m baseline.mmoe.AliCCP_MMOE                           # AliCCP + MMOE
python -m baseline.ple.AliCCP_PLE                             # AliCCP + PLE
python baseline/stem/AliCCP_STEM.py                           # AliCCP + STEM

# ===== 新任务泛化（阶段 2，需要先跑阶段 1 产出 .pt 权重）=====
python -m baseline.sharedbottom.CensusIncome_NewTask
python -m baseline.mmoe.CensusIncome_NewTask
# 依此类推...

# ===== MPTRec（阶段 1+2 合并在一个脚本）=====
python CensusIncome_MPTRec.py --seed 1685480945
python CensusIncome_NewTask.py --seed 1685480945
python AliCCP_MPTRec.py --seed 1688723512
python AliCCP_NewTask.py --seed 1688723512
```

## 代码架构

```
multitaskrec/              # 核心包
  model.py                 # EmbeddingNetwork + 所有模型类
  train.py                 # TrainManager, MPTRecTrainManager 等
  dataset.py               # CensusIncomeDataset, AliCCPDataset, ByteRecDataset
  utils.py                 # FLOPs 计算、参数统计
  evaluation.py            # 评估辅助

baseline/                  # 各 baseline 模型的训练脚本
  sharedbottom/            # SharedBottom: CensusIncome_SharedBottom.py / _NewTask.py
  mmoe/                    # MMOE: 同上模式
  ple/                     # PLE: 同上模式
  stem/                    # STEM: 同上模式
  singletask/              # SingleTask: 单任务 baseline
  sparsesharing/           # SparseSharing + CsRec: 稀疏共享 baseline
  csrec/                   # CsRec 剪枝训练 + mask 文件

config.py                  # 三个数据集的 Vocabulary_Size 定义
analysis/                  # 表征可视化（t-SNE）
```

## 关键模型（model.py）

- **EmbeddingNetwork**: 所有模型共享的 embedding 层。将 sparse 特征过 embedding、dense 特征直接拼接，输出固定维度。维度 = vocab_size × embedding_size + dense_feature_count
- **SharedBottom / MMOE / PLE / STEM / SparseSharing**: 经典多任务模型，forward 返回 `[task0_pred, task1_pred, ...]`
- **MPTRec**: 本项目的核心模型。forward 返回 `{"gen_preds", "fused_preds", "env_pred"}`，通过 `predict()` 获取最终预测，`get_infos()` 提取表征
- **NewTask**: 新任务泛化头，接收 `(dnn_input, gen_rep, spec_reps, env_embs)` 而非原始特征
- **MLP**: 通用多层感知机，支持 dropout 和多种输出激活函数

## 数据集关键行为

### CensusIncomeDataset
- 构造函数 `CensusIncomeDataset(datafile, new_task)`：`new_task` 参数指定的列会被从特征中**移除**并转为标签
- 返回 `(income, marital, new_task_label, features)` 四元组
- **必须用 `CensusIncome_Vocabulary_Size.copy().pop("education")`** 否则 `EmbeddingNetwork` 会因 vocab 有 education 但数据没有而 KeyError

### AliCCPDataset
- `AliCCPDataset(datafile, data_size)`：固定三标签（CTR、CVR、BSI），不删除任何特征列
- **不需要** vocabulary copy+pop
- 但 STEM 基线会 remove vocab 中的 `"101"` 和 `"301"`（标注为干扰特征），也需要 `.copy()` 保护全局

## 两阶段训练协议

阶段 1 预训练脚本必须保存权重（`torch.save`），阶段 2 的 NewTask 脚本加载这些权重并冻结 backbone。保存路径约定：
- `baseline/{model}/census_income_{seed}.pt` 或 `baseline/{model}/AliCCP_{seed}.pt`
- `.pt` 文件在 `.gitignore` 中被排除，需本地运行生成

## 常见踩坑

1. **vocab 不匹配**：CensusIncome 用 `new_task='education'` 但 vocab 里有 education → 必须 copy+pop
2. **input_size 计算**：vocab_size × embedding_size + 数据集中不在 vocab 的 dense 特征数。CensusIncome: 123，AliCCP: 80
3. **FLOPs 计算**：`fvcore.FlopCountAnalysis` 在 CPU 上极慢 / 假死，测试时建议先注释掉
4. **多 seed 循环**：预训练脚本通常跑 5 个 seed，很耗时，debug 时减少到 1 个
5. **AliCCP_SharedBottom.py** 之前 `torch.save` 被 `if task_num == 2` 锁死（task_num 实际是 3）
