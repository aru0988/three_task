# CensusIncome 阶段 2 公平评测协议

- **状态**：已批准，待实现（本轮只交付本文档）
- **日期**：2026-09-29
- **适用分支**：`infra/fair-stage2-benchmark`（基线：`master` @ `fd75198`）
- **唯一事实来源**：本文件。任何与本文件冲突的实验做法以本文件为准；如需变更先改本文件再改代码。
- **本文档不含任何模型改进**，也不承诺任何性能提升。它只定义"怎么跑才算公平、可复现、可审计"。

---

## 1. 目标与非目标

### 1.1 目标

| 编号 | 目标 |
|---|---|
| G1 | **协议固化**：把阶段 1 / 阶段 2 的数据划分、随机性来源、权重传递、评测时点写成唯一事实来源。 |
| G2 | **公平比较地基**：让"不同模型 seed"与"不同模型想法"都在同一 val/test 划分、同一冻结语义、同一评测口径下比较。 |
| G3 | **真实冻结**：阶段 2 的 backbone 必须是"加载自固定 checkpoint 且参数不可变"，而不是"恰好没被优化器更新"。 |
| G4 | **可复现**：同配置同种子重复运行时指标一致；配置与种子全部落盘。复跑验证按需执行（门禁 A3），不阻塞首轮。 |
| G5 | **可审计**：每次运行产出配置、种子、划分指纹、权重指纹、日志，并由门禁脚本判定通过与否。 |

### 1.2 非目标

| 编号 | 非目标 |
|---|---|
| N1 | **不做任何模型改进**：Null Expert、TC-Prompt、CGR、affinity gate、KL-Prompt、T4 等一律不在本分支（它们属 `archive/exploration`，需另开 `exp/*` 分支）。详见第 10 节。 |
| N2 | **不产出可写入论文的性能结论**：本分支只验证协议正确性。性能对齐属后续多 seed 全量阶段。 |
| N3 | **不运行 AliCCP / ByteRec**：不跑、不调参、不新增配置、不扩写文档。 |
| N4 | **不修改模型语义**：结构、损失项、超参语义（`uni_coe` / `env_coe` / `reg_*`）全部保持论文值。 |
| N5 | **本轮不写任何实现代码**：当前交付物仅为本文档。第 11 节列出后续实现面，供下一轮执行。 |
| N6 | **不改变 master 既有脚本的入口命令与模型相关超参默认值**。协议所需的固定值通过新增的可选参数/配置显式传入。 |

---

## 2. 只使用 CensusIncome

**本分支本轮只支持 CensusIncome**：所有实现、配置、脚本与运行都只针对该数据集；AliCCP / ByteRec 一律不涉及（见 2.3）。

### 2.1 数据与任务

| 项 | 取值 |
|---|---|
| 训练集 | `dataset/Census-income/train.gz`（全量） |
| 评测源 | `dataset/Census-income/test.gz`（全量），50/50 切分为 val / test |
| 新任务 | `new_task='education'`，标签 = `1 if education == 9 else 0` |
| 样本结构 | `(income, marital, label_education, features)` |
| 阶段 1 任务 | Income、Marital（两任务） |
| 阶段 2 任务 | Education（新任务泛化） |

### 2.2 必须遵守的数据约定

1. vocab 一律 `CensusIncome_Vocabulary_Size.copy()` 后再 `pop("education")`，**禁止**直接修改 `config.py` 里的全局字典。
2. `input_size=123`（vocab_size × embedding_size + dense 特征数）；`embedding_size=4`。
3. 模型超参（论文值，本分支冻结，不得改动）：`expert_dnn_hidden_units=(256,128)`、`tower_dnn_hidden_units=(64,32)`、`lr=1e-3`、`batch_size=256`、`reg_embedding=0.006`、`reg_dnn=3e-5`、`uni_coe=0.9`、`env_coe=0.1`。
4. FLOPs / 参数量单独一次性计算并落盘，**不得**放进训练循环（`fvcore.FlopCountAnalysis` 在 CPU 上极慢/假死）。

### 2.3 明确的排除项

- **AliCCP**（CTR / CVR / BSI 三标签）：本分支不运行、不配置、不调参、不写入任何新文档段落。
- **ByteRec**：官方未提供数据，本就无法运行，本分支亦不处理。
- AliCCP / ByteRec 的代码路径在本分支**保持不变**（不删、不改、不重构），以维持与 `master` 的对等性，避免"只改一半"导致后续对比失真。

---

## 3. 分支隔离规则

### 3.1 允许的改动（白名单）

- 数据划分与随机性来源的管理（划分种子、环境种子、指纹落盘）
- Stage-1 checkpoint 的保存规范与加载路径
- 冻结与 eval 语义（外部参数遍历实现，不改模型类）
- 日志、指标、门禁脚本
- `artifacts/` 目录规则与 `.gitignore` 规则
- 文档

### 3.2 禁止的改动（黑名单）

- 模型结构、损失项、门控/专家/投影设计
- 新增任何模型模块（Null Expert 等，见第 10 节）
- baseline 模型的实现
- 模型超参语义（第 2.3 节所列论文值）

### 3.3 工作流

1. 本分支验证通过 → 合并回 `master`；`master` 是唯一基线。
2. 任何模型想法必须**先等本分支合并**，再从 `master` 拉 `exp/<name>`；想法分支**只能引用**本协议，不得修改本协议文件，并按第 6.3 节阶梯验证（先单 seed 单次短跑）。
3. 一票否决：任何使 val/test 划分指纹不一致、或使冻结检查失败的改动，不得进入 `master`。
4. 本分支本轮**只支持 CensusIncome**：不运行、不配置其他数据集（AliCCP / ByteRec 见 2.3）。
5. 不推送远端；推送与合并需用户显式批准。

---

## 4. 固定 Stage-1 checkpoint 与真实冻结 / eval

### 4.1 master 现状（`fd75198`）——三条必须修掉的事实

| 编号 | 现状 | 后果 |
|---|---|---|
| F1 | `CensusIncome_NewTask.py` 在阶段 2 脚本**内部重新训练阶段 1**（`epochs=10`），全程不加载任何 checkpoint | "先预训练、再泛化"这一语义没有被强制执行；阶段 1 与阶段 2 无法分离复现 |
| F2 | `MPTRec` **没有** `freeze_params()`（`SharedBottom` / `MMOE` / `PLE` / `STEM` 都有）；阶段 2 只是"优化器里没放 backbone 参数"，且 `get_infos()` 在训练循环中未加 `torch.no_grad()` | 梯度仍会回传并累积到 backbone 的 `.grad`；"冻结"是巧合而非保证 |
| F3 | 阶段 1 的 `best_weight` 只 `load_state_dict` 回内存对象，**不落盘** | 阶段 2 无法复用同一次预训练；无法审计阶段 1 到底产出了什么 |

### 4.2 Stage-1 产物规范

```
artifacts/census_stage2/stage1/<stage1_id>/
  backbone.pt      # mptrec.state_dict()（含 env_indices 等 buffer）
  env_ids.pt       # 阶段 1 结束后、被 cluster 覆盖过的最终 env_ids
  meta.json        # 见下
```

- `stage1_id = s1-<split指纹前8>-m<modelSeed>-e<epochs>-<config哈希前8>`（内容可寻址）
- `meta.json` 记录：model seed / env seed / split seed、全部模型超参、train+val+test 索引指纹、best epoch、best val AUC、代码 commit
- **只读**：生成后不得原地覆盖；配置或种子变化 → 新的 `stage1_id`

### 4.3 阶段 2 的引用方式

阶段 2 **只允许**通过 `--stage1-dir` 引用 Stage-1 产物。禁止在阶段 2 内训练、微调或重新初始化 backbone。

### 4.4 真冻结三件套（全部满足才算冻结）

1. **参数级**：对 backbone 全部参数 `requires_grad_(False)`
2. **模式级**：`backbone.eval()`，且阶段 2 训练循环内不得调用 `backbone.train()`
3. **计算图级**：表征抽取 `get_infos()` 必须在 `torch.no_grad()` 下调用；优化器只接收 `NewTask` 参数

> 冻结通过**外部参数遍历**实现，不要求给 `MPTRec` 新增 `freeze_params()`。这样 `multitaskrec/model.py` 的模型语义与 `master` 完全一致（第 10 节）。

### 4.5 冻结的机器验证

- 阶段 2 **前后**各算一次 backbone 全参数（按参数名排序后）的 sha256，两者必须完全一致 → 门禁 **A1**
- 阶段 2 结束后，backbone 所有参数的 `.grad` 必须全为 `None`（验证 4.4 第 3 条确实生效）

### 4.6 真实 eval 语义

- 三个 loader 由**同一份划分**派生；阶段 1 与阶段 2 必须使用同一划分（同一 `split_sha256`）→ 门禁 **A2**（同一 `split-seed` 下指纹恒定）
- backbone 恒为 eval 模式；AUC 输入为 `predict()` 返回的 `fused_preds`（与 `MPTRecTrainManager.evaluation_two_task` 口径一致）
- 阶段 2 的 val **仅**用于 early stop 选点；test **只在训练全部结束后评一次**，全程不得据 test 调参
- NewTask 头的最优权重同样落盘为 `newtask.pt`，并在最终评估前重新加载
- 阶段 2 复用阶段 1 的环境定义：`env_ids` 从 Stage-1 产物**读取**，不重抽 → 门禁 **A5**

---

## 5. 固定 val/test 划分种子与模型 seed 解耦

### 5.1 master 现状

两个脚本都是 `train_test_split(test_dataset, test_size=0.5, random_state=seed)`，其中 `seed` 同时是模型种子。**换模型 seed 就换了 val/test 划分**，导致不同 seed 的 AUC 不可比——这是当前"公平比较"的最大障碍。

### 5.2 三个独立种子

| 参数 | 默认值 | 职责 | 不得影响 |
|---|---|---|---|
| `--split-seed` | 协议常量 `20260929` | **唯一**决定 val/test 划分，跨模型 seed 恒定 | 模型初始化、训练顺序 |
| `--model-seed` | `1685480945`（论文 seed 之一；首轮唯一 seed） | 模型初始化、batch 顺序、dropout 等训练随机性 | val/test 划分 |
| `--env-seed` | 协议常量 `20260929` | 只决定初始 `env_ids` | 模型初始化、划分 |

### 5.3 指纹与实现约束

1. **划分指纹**：对 val / test 的原始索引（排序后）分别 sha256，写入 `split_fingerprint.json`；门禁 A2 要求同一 `--split-seed` 下指纹**恒定**——首轮只跑 1 个 model seed，该指纹即基准；后续若扩展多 model seed，每个 seed 的指纹都必须与基准一致。
2. **env_ids**：初始环境分配由 `--env-seed` 经独立 `torch.Generator` 产生，**不消耗全局 RNG**，避免与模型初始化互相干扰（现状 `torch.randint` 直接吃全局 RNG，是一次隐式耦合）。
3. **落盘**：阶段 1 结束时把（被 cluster 覆盖后的）最终 `env_ids` 落盘，阶段 2 读取。
4. **反向兼容**：`master` 既有脚本"划分跟随 model seed"的行为，只有显式传 `--split-seed=<model seed>` 才能复现，以保证旧结果可追溯。协议默认**不**这么做。
5. **禁止**：`random_state=model_seed` 的写法不得出现在协议路径上；任何新的划分逻辑必须落盘指纹。

---

## 6. 首轮 smoke 配置（单 seed 单次）

**目的**：用最小代价验证整条协议（划分 → stage1 → 冻结 → stage2 → 指标 → 门禁）端到端跑通。**不用于任何性能结论**（对应 N2）。

### 6.1 首轮只跑一次

**"单次"的含义**：一套固定配置，不搜参、不换结构、不调超参；首轮只执行 **1 次**，只用 model seed `1685480945`。可复现性复跑（门禁 A3）**不作为首轮阻塞条件**——需要时再对同一配置做第二次复跑。

| 项 | 取值 |
|---|---|
| 数据 | CensusIncome 全量（不做子采样） |
| split seed | `20260929`（固定） |
| model seed | `1685480945`（首轮唯一 seed） |
| env seed | `20260929`（固定） |
| 阶段 1 | `epochs=2`、`patience=2`、`batch_size=256`、`lr=1e-3`、`uni_coe=0.9`、`env_coe=0.1` |
| 阶段 2 | `epochs=5`、`patience=2`、`batch_size=256`、`lr=1e-3` |
| 设备 | 单卡 `cuda:0`（RTX 3060 Laptop，6 GB 显存） |
| FLOPs | **跳过**训练循环内的 FLOPs 计算；参数量与 FLOPs 单独一次性计算后落盘 |
| 预期时长 | 单次 run ≤ 15 分钟（预期值，非门禁） |

### 6.2 与全量配置的关系

短跑只降 `epochs`，不改数据、不改划分、不改模型超参。因此短跑通过即协议通过。全量跑（`epochs` 回 30）属后续阶段，不在本分支。

### 6.3 模型改进的验证阶梯（后续 `exp/<name>` 分支适用）

1. **先单 seed 单次短跑**：每个模型改进先在 model seed `1685480945` 上按 6.1 配置跑 **1 次**，不搜参、不换结构。
2. **接受阈值先行**：跑之前必须在该分支文档写明本次改进的接受阈值（相对基线的判据）；阈值不得在看到结果后修改（与第 8 节阈值纪律一致）。
3. **达标才扩展**：只有第 1 步达到接受阈值，才允许扩展到多 seed；扩展的 seed 列表由用户在该分支显式指定，本文档不预设。
4. **未达标即止损**：未达阈值 → 如实记录该次结果，不得跳到多 seed 或全量跑；分支按 3.3 节第 1 条处置。
5. **对照一致**：改进与基线必须在同一 `--split-seed`、同一冻结语义（第 4 节）下比较。

---

## 7. AUC 与机制指标

### 7.1 主指标（性能）：AUC

| 位置 | 指标 | 用途 |
|---|---|---|
| 阶段 1 | `AUC-Val-Income`、`AUC-Val-Marital`（每 epoch） | 阶段 1 选点 + 活性检查 |
| 阶段 2 | `AUC-Val-Education`（每 epoch） | 阶段 2 early stop 选点 |
| 阶段 2 | `AUC-Test-Education`（结束后一次） | **主报告值** |
| 汇总 | `metrics.json` | 每 epoch 值、best epoch、best val、final test，以及 epoch 级 `uni_loss_0/1`、`fuse_loss_0/1`、`env_loss` |

### 7.2 机制指标（解释"是否真按论文机制在工作"，不参与性能判定）

| 编号 | 指标 | 定义 |
|---|---|---|
| M1 | `env_acc` | 阶段 1 每 epoch 在**训练集**上 `env_pred` 与当前 `env_ids` 的一致率。梯度反转生效时该值应被压向随机水平 `1/num_tasks = 0.5` |
| M2 | `cluster_diff_num` / `env_0, env_1` | 每次 `cluster_2()` 的 diff_num 与两环境样本数（`master` 已有打印，只需落盘） |
| M3 | `gate_w` | `fused_preds` 路径上 `gate_networks[i]` 在验证集上的平均权重（2 维：specific 分支 / general 分支） |
| M4 | 表征几何 | 验证集上 `cos(gen_rep, spec_rep_i)` 均值（i=0,1），以及 `gen_rep` 各维标准差均值（防常量表征） |
| M5 | `backbone_sha256` | 阶段 2 前后各一次 |
| M6 | `split_sha256` / `env_ids_sha256` | 划分与环境标签指纹 |

**实现约束**：M1 / M3 / M4 在 `torch.no_grad()` 下按 batch 累积后一次算，不保留中间张量；M4 在验证集上按固定抽样 seed 抽样，避免显存峰值。

### 7.3 报告纪律

- 不得用 test 集做任何模型选择
- 不得只报告最好的 seed
- 不得省略失败的 run

---

## 8. 验收阈值

### A 类：协议正确性（A1 / A2 / A4 / A5 为硬门禁必须全过；A3 按需触发、不阻塞首轮）

| 编号 | 判据 |
|---|---|
| A1 | 阶段 2 前后 `backbone_sha256` 完全一致，且阶段 2 结束后 backbone 参数 `.grad` 全为 `None` |
| A2 | 划分指纹只由 `--split-seed` 决定：`split_sha256` 与 `split_fingerprint.json` 基准一致，且阶段 1 / 阶段 2 读到同一指纹；后续若扩展多 model seed，每个 seed 的指纹都必须与基准一致 |
| A3 | **可复现性（按需触发，不阻塞首轮）**：需要时对同一 `(split seed, model seed, env seed, 配置)` 做第二次复跑，两次 `AUC-Test-Education` 差异 ≤ `1e-9` 且 `backbone_sha256` 一致；首轮只跑一次，不因未复跑而失败，复跑一旦执行即按本判据判定 |
| A4 | train / val / test 索引两两交集为空；val ∪ test = `test.gz` 全量；train = `train.gz` 全量。三个集合的实际行数实测后写入 `split_fingerprint.json`，作为后续 run 的对照基准 |
| A5 | 阶段 2 读到的 `env_ids_sha256` 与阶段 1 落盘的完全一致 |

### B 类：模型活性（硬门禁）

| 编号 | 判据 |
|---|---|
| B1 | `AUC-Val-Income ≥ 0.60`、`AUC-Val-Marital ≥ 0.60`、`AUC-Test-Education ≥ 0.60`。此下界远高于随机 0.5、远低于论文级预期，**只**用于捕获流程性破坏（标签错位、冻结失效、划分错乱），不是性能目标 |
| B2 | `\|AUC-Val-Education(best) − AUC-Test-Education\| ≤ 0.03`（val 与 test 是同一 `test.gz` 的随机两半，同模型上不应有大差） |
| B3 | 验证集上平均 gate 权重每一维 ∈ `[0.05, 0.95]`（未坍缩到单一分支） |
| B4 | 每次 `cluster_2()` 后两个 env 各占训练集 ≥ 5%（聚类未退化） |

### C 类：诊断（只落盘，不判定）

`env_acc`、cluster 计数、gate 权重、表征余弦/标准差、loss 轨迹。

### 明确不设的阈值

- 不设"论文对齐 / 超越 baseline"的性能门槛（属后续阶段，见 N2）
- 不设运行时长门槛
- 不设显存门槛

### 阈值变更纪律

阈值只允许在**看到 run 结果之前**修改；每次修改单独 commit，并在本文件记录理由。已被旧阈值判定的 run 不得用新阈值重判。

---

## 9. 实验产物目录与 Git 保留策略

### 9.1 目录结构

```
artifacts/census_stage2/
  stage1/<stage1_id>/
    backbone.pt
    env_ids.pt
    meta.json
  runs/<run_id>/
    config.json
    metrics.json
    split_fingerprint.json
    env_ids.pt
    newtask.pt
    gate_report.json
    stdout.log
  SUMMARY.md          # 唯一入库文件
```

`run_id` 格式：`<YYYYMMDD-HHmm>-s<splitSeed>-m<modelSeed>-<short|full>-<commit7>`
例：`20260929-1530-s20260929-m1685480945-short-fd75198`

### 9.2 Git 保留策略

| 类别 | 内容 | 处置 |
|---|---|---|
| 入库 | 本协议文档、`artifacts/census_stage2/SUMMARY.md` | commit |
| 不入库 | 全部权重（`*.pt`）、指标（`*.json`）、日志（`*.log`） | 由现有 `.gitignore` 规则天然拦截 |
| 需新增的忽略规则 | `artifacts/census_stage2/stage1/`、`artifacts/census_stage2/runs/`，并保留 `!artifacts/census_stage2/SUMMARY.md` | 实现时加入 `.gitignore` |

`SUMMARY.md` 每个 run 一行，含：`run_id`、代码 commit、`AUC-Test-Education`、A/B 门禁逐项结果、`stage1_id`。

### 9.3 保留与容量

- `stage1/` 中每个 split 指纹只保留最新一个 `stage1_id`，旧的删除。
- `runs/` 中只保留"支撑过 `SUMMARY.md` 结论或门禁判定"的 run，其余可删。
- `artifacts/` 总计 ≤ 2 GB；超限时按 run 时间从旧到新删除，但**永不删除** `SUMMARY.md` 引用且尚未被取代的 run。
- 永不入库：数据集、`.venv/`、任何 test 集预测明细。
- 禁止 `git add -f` 绕过忽略规则提交权重或日志。

---

## 10. 本分支明确不含的模型改进

本分支**不含**以下任何内容，它们属 `archive/exploration` 归档，需从 `master` 另拉 `exp/*` 分支才能重启：

- **Null Expert**（空专家 / 额外专家位）
- TC-Prompt、CGR、affinity gate、KL-Prompt
- T4 实验（2src vs 3src attention 的 source task 数量研究）
- 任何新的专家、门控、投影、温度、损失设计

**强约束**：本分支对 `multitaskrec/model.py` 的模型语义改动为**零**。冻结通过外部参数遍历实现（4.4），不新增 `freeze_params()`，不修改任何模型类。阶段 1 / 阶段 2 的模型超参一律为第 2.3 节所列论文值。

---

## 11. 本分支的最小实现面（供下一轮执行，本轮不写代码）

1. 划分与环境种子外置 + `split_fingerprint.json` / `env_ids` 落盘（第 5 节）
2. Stage-1 保存规范：`backbone.pt` + `env_ids.pt` + `meta.json` + 内容寻址 `stage1_id`（4.2）
3. Stage-2 加载 + 真冻结三件套 + 冻结指纹断言（4.3–4.5）
4. 指标与门禁脚本：`metrics.json` / `gate_report.json` + A/B 类判定 + `SUMMARY.md` 追加（第 7、8、9 节）
5. `artifacts/` 的 `.gitignore` 规则（9.2）

---

## 附录：协议相对 master 现状的差异清单

| master 现状 | 本协议处置 | 对应节 |
|---|---|---|
| F1 阶段 2 脚本内重训阶段 1 | 阶段 2 只从 `--stage1-dir` 加载，禁止训练 backbone | 4.3 |
| F2 无 `freeze_params()`，`get_infos()` 无 `no_grad` | 外部参数遍历冻结 + 三件套 + `.grad` 断言；不改 `model.py` | 4.4、4.5、10 |
| F3 Stage-1 权重不落盘 | 内容寻址的 Stage-1 产物目录 + `meta.json` | 4.2 |
| `random_state=model_seed`（划分与模型种子耦合） | 拆成 `--split-seed` / `--model-seed` / `--env-seed` 三个独立种子 + 指纹 | 5.2、5.3 |
| `torch.randint` 初始 `env_ids` 吃全局 RNG，且不落盘 | 独立 `torch.Generator`，阶段 1 结束时落盘，阶段 2 读取 | 5.3、4.6 |
| 阶段 2 无 test 访问纪律 | val 只用于 early stop，test 只评一次 | 4.6、7.3 |
| 脚本内训练前对单 batch 跑 `FlopCountAnalysis`（CPU 上极慢/假死） | 移出训练路径，单独一次性计算 | 2.2、6 |
