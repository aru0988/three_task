# MPT-Rec 项目架构笔记

> 生成日期: 2026-05-28  
> 论文: *Efficient Multi-task Prompt Tuning for Recommendation* (ACM TOIS 2025)  
> 作者: Bai et al. (Beijing Univ. Posts & Telecom / Tencent AI Lab)

---

## 1. 项目概述

MPT-Rec 是一个**两阶段多任务推荐系统框架**，核心创新点：

1. **阶段 1（预训练）**：用 GAN 将表征解耦为 task-sharing（共享）和 task-specific（专属）两部分，同时通过 env classifier 做对抗学习
2. **阶段 2（新任务泛化）**：冻结阶段 1 的 backbone，仅训练一个轻量 NewTask 模块，通过 task-aware prompt 从已有任务提取有用信息

训练参数只占全量训练的 **~10%**，FLOPs 降至 **~9%**。

### 论文发表

- ACM Transactions on Information Systems, Vol. 43, No. 4, Article 104 (July 2025)
- DOI: https://doi.org/10.1145/3736403
- 代码: https://github.com/BAI-LAB/MPT-Rec

---

## 2. 目录结构

```
MPT-Rec/
├── multitaskrec/                # 核心包
│   ├── __init__.py
│   ├── model.py                 # 所有模型类 (约1000行)
│   ├── train.py                 # 训练管理器
│   ├── dataset.py               # 数据集加载
│   ├── evaluation.py            # 评估工具
│   ├── utils.py                 # 参数/FLOP统计
│   └── tc_prompt.py             # TC-Prompt 融合模块
├── baseline/                    # Baseline 模型
│   ├── sharedbottom/            # SharedBottom + NewTask
│   ├── mmoe/                    # MMOE + NewTask
│   ├── ple/                     # PLE + NewTask
│   ├── stem/                    # STEM + NewTask
│   ├── singletask/              # SingleTask
│   ├── sparsesharing/           # SparseSharing + CsRec
│   └── csrec/                   # CsRec 剪枝训练
├── analysis/                    # 分析脚本
│   ├── cgr_diagnosis.py         # CGR 置信度分析
│   ├── cgr_gradient_diagnosis.py# 梯度方差分析
│   ├── compute_silhouette.py    # Silhouette 分数
│   └── projection_analysis.py   # 表征可视化
├── mask/                        # CsRec 剪枝 mask 生成
├── logs/                        # 实验结果
│   ├── all_experiment_results.md
│   └── diagnostic_analysis.md
├── config.py                    # Vocabulary_Size 配置
├── compute_results.py           # 聚合实验结果
├── eval_checkpoints.py          # 评估 checkpoint
├── run_ablation.py              # 消融实验 (Fig 4-6)
├── run_lambda_sensitivity.py    # λ 敏感性分析
├── run_tcprompt_experiment.py   # TC-Prompt 实验
├── run_aliccp_tcprompt.py       # AliCCP TC-Prompt
├── resume_lambda_aliccp.py      # λ 断点恢复
└── CensusIncome_MPTRec.py       # 入口: CensusIncome
├── CensusIncome_NewTask.py      # 入口: CensusIncome 新任务
├── AliCCP_MPTRec.py             # 入口: AliCCP
└── AliCCP_NewTask.py            # 入口: AliCCP 新任务
```

---

## 3. 数据集

### CensusIncome
- 来源: UCI Census-income (1994/1995)
- 样本: train~200K, test~100K
- 特征: 40 列 (去除 education 后 39 个特征列)
- 任务: T1=Income, T2=Marital, T3=Education(新任务)
- Vocab size: 去除 education 后 30 个离散特征
- input_size=123 (30个vocab×4emb + 3个dense特征)
- 新任务标签: education==9 → 1, else 0

### AliCCP
- 来源: 淘宝推荐系统日志
- 样本: train~5M, dev~500K, test~5M (实验用子集)
- 特征: 18 列 (含"101"干扰特征)
- 任务: T1=CTR, T2=CVR, T3=BSI(新任务)
- Vocab size: 去除 101/301 后保留
- input_size=80
- 标注: BSI label 会将 2→0

### ByteRec (论文中用到，代码中仅 PLE 有脚本)
- 来源: 短视频推荐比赛
- 任务: finish, like, duration_time

---

## 4. 模型架构

### 4.1 EmbeddingNetwork
- 统一 embedding 层，所有模型共享
- 离散特征: nn.Embedding(vocab_size, embedding_size)
- 连续特征: 直接拼接 unsqueeze 后 concat
- 输出: [batch, num_features, embedding_size]

### 4.2 MLP
- 通用多层感知机
- 支持 dropout list、多种输出激活(relu/sigmoid/softmax)
- 可选 L2 regularization

### 4.3 Baseline 模型

| 模型 | 机制 | 代码位置 |
|------|------|----------|
| **SingleTask** | 单任务独立训练 | model.py:80 |
| **SharedBottom** | 共享底层 + 独立 tower | model.py:142 |
| **MMOE** | 多专家 + gate 网络 | model.py:193 |
| **PLE** | CGC 模块多层堆叠 | model.py:547 |
| **STEM** | 每个任务独立 embedding + 共享 expert | model.py:281 |
| **SparseSharing** | 剪枝子网 + 掩码训练 | model.py:658 |
| **CsRec** | 对比学习 + 剪枝 | train.py:270 |

### 4.4 MPTRec (核心模型)

```
输入特征 x
  └→ EmbeddingNetwork → dnn_input
      ├→ Shared Expert → gen_rep (task-sharing)
      │   └→ Tower → gen_preds (仅共享信息的预测)
      │   └→ ReverseLayerF → EnvClassifier → env_pred (GAN对抗)
      └→ [Specific Expert i → spec_rep_i × env_embedding_i]
          └→ env_aware_rep_i
          └→ [env_aware_rep_i, gen_rep] → Gate → fused_rep_i → Tower → fused_preds
```

关键组件:
- **SharedExpert**: 生成 task-sharing 表征 (generator)
- **EnvClassifier**: 鉴别表征来自哪个 task (discriminator) — 通过 ReverseLayerF 做对抗
- **SpecificExpert**: 每个 task 独立专属表征
- **EnvEmbedding**: 可学习的 task embedding，与 spec_rep 做 element-wise product
- **Gate**: 学习融合 gen_rep 和 env_aware_rep 的权重

三种 variant:
- `full` (默认): gen + spec + GAN
- `share_only`: 仅用共享信息
- `specific_only`: 仅用专属信息 (但 GAN 仍训练)
- `no_gan`: 有 gen+spec，但移除 GAN 对抗部分 (env_pred 全零)

### 4.5 NewTask (新任务泛化头)

```
dnn_input, gen_rep, spec_reps, env_embs
  └→ 计算 W 权重 (不同 fusion_mode)
      └→ spec_reps × W → new_spec_rep
          └→ × new_env_emb → env_aware_rep
              └→ [env_aware_rep, gen_rep] → Gate → fused_rep → Tower → pred
```

**Fusion Modes** (新任务如何从已有任务提取信息):

| Mode | 权重计算方式 | 说明 |
|------|-------------|------|
| `prompt` | h_p · E_i / τ → Softmax | 原始 MPT-Rec 的 instance-level attention |
| `fw` | 1/K (等权重) | 固定权重 |
| `tes` | new_env_emb · exist_env_embs / τ → Softmax | Task Embedding Similarity |
| `tcprompt` | h_p · E_i / τ + λ·ρ_i → Softmax | TC-Prompt: 加入 Pearson 相关性先验 |
| `cgr` | g·W_attn + (1-g)·W_fw | Confidence-Gated Routing |
| `affinity_gate` | Gumbel-Sigmoid 硬切换 Attention/FW | OOD-aware 硬切换 |

---

## 5. 训练流程

### 5.1 两阶段协议

**阶段 1 — 多任务预训练**:
- 训练所有已有任务 (CensusIncome: Income+Marital, AliCCP: CTR+CVR)
- Loss = fused_loss + uni_coe × gen_loss + env_coe × env_loss + l2_reg
- 每 2 epoch 重新聚类 env_ids (基于最小预测损失)
- Early stopping (patience=5)
- epoch=10 (论文正式 30)，seed 通常 3~5 个

**阶段 2 — 新任务泛化**:
- 冻结 MPTRec backbone
- 仅训练 NewTask 模块 (~27K 参数，全量 9.3%)
- 支持多种 fusion_mode 切换
- 同样 early stopping

### 5.2 训练器层次

```
TrainManager (通用)
  ├── SparseSharingTrainManager (掩码训练)
  │   └── CsRecTrainManager (对比学习)
  └── MPTRecTrainManager (GAN + 聚类)
      ├── train_two_task()
      ├── train_three_task()
      ├── cluster_2() / cluster_3() — 环境聚类
      └── evaluation_two_task() / evaluation_three_task()
```

### 5.3 损失函数

MPTRec 总损失:
```
L = Σ(fused_loss_i) + uni_coe × Σ(gen_loss_i) + env_coe × NLL(env_pred, env_ids) + l2_reg
```

- `uni_coe` (uninformative coefficient): 控制共享预测的权重 (default 0.9)
- `env_coe` (environment coefficient): 控制 GAN 对抗权重 (default 0.1)
- `reg_embedding` / `reg_dnn`: L2 正则化系数

---

## 6. 实验与结果

### 6.1 硬件环境

- GPU: NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM)
- CensusIncome: ~5 min/seed (阶段1), ~7 min/mode (阶段2)
- AliCCP: ~55 min/seed (阶段1), ~30 min/mode (阶段2)
- 总运行约 10 小时

### 6.2 MTL 结果 (Table 3, 论文原文)

| 数据集 | 最佳方法 | AUC/T1 | AUC/T2 |
|--------|---------|--------|--------|
| CensusIncome | MPT-Rec | 0.9517 | 0.9926 |
| AliCCP | MPT-Rec | 0.5990 | 0.6274 |
| ByteRec | MPT-Rec | 0.7485 | 0.9339 |

### 6.3 新任务泛化结果 (Table 5, 论文原文)

| 方法 | CensusIncome AUC/T3 | 参数比 | AliCCP AUC/T3 | 参数比 |
|------|-------------------|--------|--------------|--------|
| MAML | 0.8563 | 26K | 0.7155 | 1748K |
| Shared* | 0.8133 | 21K (21.2%) | 0.6279 | 6K (0.12%) |
| MMOE* | 0.8497 | 21K (9.4%) | 0.6862 | 7K (0.13%) |
| PLE* | 0.8617 | 118K (26.0%) | 0.6997 | 33K (0.62%) |
| **MPT-Rec** | **0.8626** | **27K (9.3%)** | **0.6968** | **8K (0.15%)** |

### 6.4 New-task Fusion 实验 (2026-05-30 完整结果)

使用 `run_newtask_from_ckpt.py` 统一框架，GAN-on (uni_coe=0.9, env_coe=0.1)，独立 RNG。3 seeds × 8 modes。

| 策略 | CensusIncome | AliCCP |
|------|-------------|--------|
| FW | 0.8548 | 0.6744 |
| TES | 0.8546 | 0.6725 |
| Prompt | **0.8629** | 0.6719 |
| TC-Prompt(fixed) | 0.8618 | 0.6731 |
| TC-Prompt(learn) | 0.8625 | 0.6740 |
| CGR | 0.8618 | 0.6632 ⚠️ |
| Affinity Gate | 0.8601 | 0.6714 |
| KL-Prompt (β=0.1) | 0.8609 | **0.6765** |

关键发现:
- CensusIncome: 所有 attention-based 方法挤在 0.860-0.863 区间，Prompt 微弱最优，无显著差异
- AliCCP: KL-Prompt 和 FW 并列最优（差距 +0.0021 在噪声范围内）。FW 本身是一个被低估的强基线
- Prompt 在 AliCCP 上均值不如 FW 且方差更大（±0.0073），attention 在此数据集上不帮助反而增加不稳定性
- TC-Prompt 的相关性先验 λ 始终收敛到 ~0.5，λ·ρ 仅 ~0.03-0.08，对 attention 无实质影响
- CGR 在 AliCCP seed3 崩溃（0.6354），根因是全局标量门控设计缺陷（见 `logs/all_experiment_results.md` 第 4 节）

### 6.5 已修复的 Bug (2026-05-22)

`AliCCP_NewTask.py` / `CensusIncome_NewTask.py` 中 `uni_coe` 和 `env_coe` 默认值从 (0,0) 改为 (0.9, 0.1)。修复后:
- CensusIncome T3: 0.8597 → 0.8629 (+0.0032)
- AliCCP T3: 0.6662 → 0.6720 (+0.0058)

---

## 7. 已知问题与常见踩坑

1. **vocab 不匹配**: CensusIncome 用 `new_task='education'`，但 vocab 里有 education
   - 必须 `copy()+pop()`，否则 EmbeddingNetwork KeyError
2. **input_size 计算**: 不是固定值，取决于 vocab_size × embedding_size + dense 特征数
   - CensusIncome: 123, AliCCP: 80
3. **FLOPs 计算卡死**: `fvcore.FlopCountAnalysis` 在 CPU 上极慢，测试时注释
4. **多 seed 循环耗时**: 预训练 5 seed × baseline 很慢，debug 时减少到 1~3 个
5. **AliCCP_SharedBottom.py torch.save bug**: 被 `if task_num == 2` 锁死 (实际是 3)
6. **env_ids 随机初始化**: `torch.randint(0, 2, ...)` 随机聚类，不是从数据中学习的

---

## 8. 后续扩展方向 (论文+实验已暗示)

1. **Lifelong Learning**: 论文提及的 future work，连续新增任务场景
2. **KL-Prompt (β=0.1)**: 已实现，在 AliCCP 上与 FW 并列最优（+0.0021 无统计显著性），零额外参数，方差控制好。β sensitivity 待分析
3. **CGR 修复**: 当前设计存在全局标量门控缺陷，在 AliCCP seed3 崩溃（σ=0.0241）。改进方向：替换门控输入为逐样本特征
4. **Affinity Gate**: OOD-aware 硬切换，最稳定（σ=0.0014）但均值偏低，过于保守
5. **更多数据集**: ByteRec 脚本不完整，可补全
