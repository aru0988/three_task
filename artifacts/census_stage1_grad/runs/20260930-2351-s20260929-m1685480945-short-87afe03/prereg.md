# CensusIncome 阶段 1 梯度冲突审计 —— 预注册（Pre-registration）

- 日期：2026-09-30
- 分支：`exp/stage1-gradient-conflict-audit`（基于 `infra/fair-stage2-benchmark`）
- 状态：**阈值在本文件写定之后才允许运行审计**。运行后不得回改阈值；若要改，只能新开一份预注册并从零重跑。

## 0. 定性：这是审计，不是创新

梯度手术/梯度冲突消解（PCGrad、GradNorm、CAGrad、Gradient Vaccine、PCGrad 的诸多变体）是**已成型的既有文献**，不是本项目的贡献点。本审计唯一要回答的问题是：

> 在 MPT-Rec 阶段 1 的 CensusIncome 设定下，Income / Marital / 环境目标这三个损失分量，在**真正共享**的参数上是否已经存在可测的梯度冲突，从而**值得**再开一个分支去试 PCGrad/GradNorm？

如果证据不过门禁，结论就是 `NO_ACTION`：不做梯度手术，把精力放回主线。**本审计本身不改变任何训练更新，也不主张任何性能提升。**

## 1. 审计对象与口径

### 1.1 损失分量（与 `multitaskrec/train.py::MPTRecTrainManager.train_two_task` 逐字对应）

阶段 1（2 任务）每 batch 的优化目标：

```
total = fused_loss_0 + fused_loss_1 + uni_coe * (uni_loss_0 + uni_loss_1)
        + env_coe * env_loss + model.get_l2_reg()
```

按"每个分量对 total 的实际贡献"切分：

| 分量 | 定义 | 是否任务 |
|---|---|---|
| `income` | `fused_loss_0 + uni_coe * uni_loss_0` | 是 |
| `marital` | `fused_loss_1 + uni_coe * uni_loss_1` | 是 |
| `env` | `env_coe * env_loss` | 否（环境目标） |
| `reg` | `model.get_l2_reg()` | 否（正则项） |

约束：`sum(分量) ≈ total`（浮点再结合，非逐位相等），该恒等式由单元测试守护。

**不定义 general-vs-specific 分量。** `fused_rep_i = gate_i[0] · (spec_rep_i ⊙ env_emb_i) + gate_i[1] · gen_rep`，
gate 权重依赖输入样本，因此"fused 里属于 specific 的那部分损失"在损失空间里**没有良定义**。
按"only if mathematically defined"原则，本审计不产出该复合量，并在文档中显式记录为"不适用"。

### 1.2 参数分组（只有真正共享的才算 shared）

分组按参数名前缀，**按"是否被 ≥2 个分量可达"判定共享**（可达性由 autograd 图实测，见 §3 自检）：

| 组名 | 参数前缀 | 被哪些分量可达 |
|---|---|---|
| `shared_embeddings` | `embedding_network.*` | income / marital / env（三者） |
| `shared_expert` | `shared_expert_network.*` | income / marital / env（三者） |
| `shared_trunk` | 上面两组的并集 | **门禁判定所用组** |
| `env_classifier` | `env_classifier.*` | 仅 env |
| `task_exclusive` | `specific_expert_networks.*`、`gate_networks.*`、`tower_networks.*`、`env_embedding_network.*` | 仅对应单任务 |

- `tower_networks[i]` 同时服务 `gen_preds[i]`（uni 分支）与 `fused_preds[i]`（fused 分支），
  但**只服务任务 i**，故按任务独有处理，绝不进入 `shared_trunk`。
- `env_embedding_network.weight` 形状 `(num_tasks, H)`，第 i 行只被任务 i 使用；按张量粒度整体归入 `task_exclusive`。
- `env_classifier` 只被 env 分量可达 → 归入环境独有，**不计入 shared**。

### 1.3 梯度的取法（只读，无副作用）

对每个分量做一次 `torch.autograd.grad(component, params, retain_graph=True, allow_unused=True)`：

- **不写 `.grad`**、不调 `optimizer.step()`、不消耗 RNG、不改参数。
- 未被该分量可达的参数返回 `None`，一律按**精确零张量**处理并计数。
- `env` 分量的梯度**包含** `ReverseLayerF` 的反转与 `alpha` 缩放（即优化器实际看到的那一支），
  因为"冲突"问的是实际进入更新的方向。每步记录 `alpha` 以便解释。注意这是**构造性对抗**，
  故 task-vs-env 的负余弦有设计上的先验，不能单独作为任务间 PCGrad 的依据（见 §2）。
- 向量化统计前统一 `detach().to("cpu", float64)`；拼接为每个参数组的单一扁平向量
  （**不是**逐参数余弦再平均——组内余弦按拼接后的整体向量定义）。

## 2. 门禁（在看到任何结果之前写定）

**唯一可触发后续 PCGrad 实验的条件**：`income-vs-marital` 这一对，在 `shared_trunk` 这一组上，
两个 epoch 合并统计，同时满足下表四条。

| 编号 | 条件 | 阈值 |
|---|---|---|
| G0 | 采样步数（两 epoch 合计） | `>= 50` |
| G1 | `income-vs-marital` 的 `cos(income, marital)` 为负的步数占比 | `>= 0.25` |
| G2 | `income-vs-marital` 的 `cos(income, marital)` 均值 | `<= -0.05` |
| G3 | income 与 marital 两侧梯度范数**同时**非零的步数占比 | `>= 0.95` |

- G0–G3 全部通过 → 判定 `ACTIONABLE`，允许**另开分支**试 PCGrad；本分支不实现 PCGrad。
- 任一不通过 → 判定 `NO_ACTION`，记录原因，主线继续，不再讨论梯度手术。
- **`income-vs-env` / `marital-vs-env` 的冲突只作诊断量报告，单独不构成行动依据。**
  理由：梯度反转使 env 支路被设计成与任务方向相反，负余弦是机制本身而非新发现。

## 3. 实验设计（Option A：插桩一次公平的 2-epoch 阶段 1）

- 数据、划分、超参、seed 全部沿用既有协议 `census_benchmark/protocol.py`：
  `SPLIT_SEED=20260929`、`MODEL_SEED=1685480945`、`ENV_SEED=20260929`、
  `STAGE1_EPOCHS=2`、`BATCH_SIZE=256`、`LR=1e-3`、`UNI_COE=0.9`、`ENV_COE=0.1`。
  **只用 CensusIncome。**
- 采样节奏固定并预先声明：每个 epoch 采 **前 10 个 batch（warmup）+ 均匀间隔 25 个 batch**，
  去重后约 34 步/epoch，两个 epoch 合计 `>= 50` 步。间隔位置用
  `(k * n_batches) // spaced`（k = 0..spaced-1）这一确定性整数公式，不用随机抽样。
- 插桩点在 forward 之后、`zero_grad/backward` 之前，**复用同一次 forward 的输出**，
  因此不额外消耗 RNG、不改变数据顺序。
- **逐位保真**：`audit=None` 时插桩版 `train_two_task` 必须与被插桩的父类实现
  产生逐位相同（`torch.equal`）的参数；开启采集时同样必须逐位相同。
  由单元测试在同进程、同 seed 下守护（跨进程/GPU 归约非确定性不在保证范围内）。
- **诚实的边界**：本审计复现的是"同 seed / 同划分 / 同超参 / 同 batch 顺序"的公平阶段 1，
  而不是 `run_stage1` 那一次具体轨迹的逐位重放。`nn.Embedding` 的反向在 CUDA 上用
  `atomicAdd`，跨进程本就不可逐位复现；门禁判据（负余弦占比、平均余弦）也不依赖这一点。
- 开销上限：采样步约占全部 batch 的 4.4%，每步多 4 次反向 → 预计 < 20% 墙钟开销。
- smoke 模式（`--max-batches`）只用于跑通链路，其结论强制标为 `NOT_EVALUATED_SMOKE`，
  **不产生门禁判定**，不得当作审计结果引用。

## 4. 产出（compact JSON/Markdown，不做逐参数 dump）

`artifacts/census_stage1_grad/`：

- `runs/<run_id>/audit.json`：预注册阈值自述块 + 每步标量记录（约 39 个标量/步）+ 汇总统计 + 门禁判定；
  不含时间戳/绝对路径 → 同输入逐字节可复现。
- `runs/<run_id>/prereg.md`：本文件的快照副本（阈值与代码常量一致性由测试校验）。
- `runs/<run_id>/stdout.log`：含墙钟开销等非确定性信息（**只在 log 里，不进 JSON**）。
- `SUMMARY.md`：只追加的门禁结论表。

全局与分组统计量定义：

- `mean_cos`：各步余弦的算术平均（仅统计有定义的步）。
- `neg_cos_rate` = `#(cos < 0) / #定义步`；`dot sign` 与余弦同号，等价。
- `mean_of_ratio` = `mean(‖g_a‖ / ‖g_b‖)`（逐步比值再平均，对 ‖g_b‖=0 的步跳过）。
- `ratio_of_means` = `mean(‖g_a‖) / mean(‖g_b‖)`（先平均再比值，更稳）。
- `dominance_ratio` = `mean( ‖g_a‖ / (‖g_a‖ + ‖g_b‖) )`，∈[0,1]，0.5 为均衡；按 (a,b) 顺序非对称。
- `nonzero_rate_a/b`：该分量在该组上范数严格非零的步占比。
- 零梯度计数：① `zero_norm_steps`（分量 × 组 × epoch：范数精确为 0 的步数）；
  ② `zero_tensor`（分量 × 组：梯度张量精确全零的 (步, 参数) 对数与占比）——
  后者能暴露 embedding 行未被 batch 命中导致的固有稀疏零梯度。

## 5. 已知的构造性零梯度（预期中出现，不算异常）

`alpha = 2/(1+e^{-10p}) - 1`，`p = (step + (epoch-1)*batch_size) / (epochs*batch_size)`。
epoch 1 的 `step=0` 时 `alpha = 0`，故 env 支路在 shared trunk 上的梯度在该步**精确为 0**。
该现象必须出现在记录中并保持可见。

## 6. 决策规则

```
if G0 and G1 and G2 and G3:  status = ACTIONABLE -> 另开分支试 PCGrad（本分支不做）
else:                        status = NO_ACTION  -> 记录原因，主线继续
```

env 相关的负余弦一律只写入 `env_diagnostic` 段落，**不参与** status 计算。

## 7. 不做的事

- 不改 `multitaskrec/`、`config.py`、`CensusIncome_MPTRec.py`、`CensusIncome_NewTask.py`（既有测试 `test_static_guards_*` 已强制）。
- 不改损失、不改优化器、不改更新规则、不声称任何性能提升。
- 不跑 AliCCP、不新增 seed、不重划数据。
- 不 commit / push / merge；由人工审阅后决定。
