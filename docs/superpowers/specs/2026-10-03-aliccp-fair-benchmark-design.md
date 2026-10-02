# AliCCP 阶段 1 / 阶段 2 公平评测协议（基础设施，非模型改进）

- **状态**：已批准，按本文档实现（先提交本文档，再写代码）
- **日期**：2026-10-03
- **适用分支**：`infra/aliccp-fair-benchmark`（基线：`master` @ `fd75198`，独立于任何 CensusIncome 实验分支）
- **唯一事实来源**：本文件。任何与本文件冲突的实验做法以本文件为准；需变更先改本文件再改代码。
- **本文档不含任何模型改进**，也不承诺任何性能提升。它只定义"AliCCP 上怎么跑才算公平、可复现、可审计"。
- **与 CensusIncome 协议的关系**：沿用同一套方法论（种子解耦、内容寻址 Stage-1 产物、真冻结三件套、A/B/C 门禁、SUMMARY 纪律），但**预算、阈值、指纹定义全部为 AliCCP 专属，不复用 CensusIncome 的任何数值**。本分支从 `master` 拉出，不含 `census_benchmark/` 代码，两者互不依赖。

---

## 1. 目标与非目标

### 1.1 目标

| 编号 | 目标 |
|---|---|
| G1 | **协议固化**：把 AliCCP 阶段 1 / 阶段 2 的样本预算（前缀）、随机性来源、权重传递、评测时点写成唯一事实来源。 |
| G2 | **公平比较地基**：让不同模型想法在同一前缀样本、同一冻结语义、同一评测口径下比较，并为后续跨数据集复现提供模板。 |
| G3 | **真实冻结**：阶段 2 的 backbone 必须是"加载自固定 checkpoint 且参数不可变"，而不是"恰好没被优化器更新"。 |
| G4 | **可复现**：同配置同种子重复运行时指标一致；配置、种子、前缀指纹全部落盘。复跑验证按需执行（门禁 A3），不阻塞首轮。 |
| G5 | **可审计**：每次运行产出配置、种子、前缀指纹、权重指纹、日志，并由门禁脚本判定通过与否。 |

### 1.2 非目标

| 编号 | 非目标 |
|---|---|
| N1 | **不做任何模型改进**：新专家、门控、投影、prompt、adapter、损失设计等一律不在本分支。 |
| N2 | **不产出可写入论文的性能结论**：首轮短跑只验证协议正确性与活性；性能对齐属后续多 seed 全量阶段。 |
| N3 | **不运行其他数据集**：CensusIncome / ByteRec 不涉及、不配置、不改其代码路径。 |
| N4 | **不修改模型语义**：`multitaskrec/*`（model/train/dataset/utils）、`config.py`、master 既有脚本、`baseline/*` 一律零修改；本分支只新增文件、只做 import 复用。 |
| N5 | **不跑 baseline 模型族**（SharedBottom/MMOE/PLE/STEM/SparseSharing）：首轮只跑 MPTRec 基线一条链；baseline 比较政策见第 12 节。 |
| N6 | **不做数据清洗/去重**：前缀中存在的重复行（见 2.5）量化记录但不修改（修改会偏离仓库/论文管线，属独立的数据卫生议题）。 |

---

## 2. AliCCP 数据与任务审计（2026-10-03，只读扫描）

### 2.1 磁盘事实

| 文件 | 字节数 | 数据行数（不含表头） |
|---|---|---|
| `dataset/AliCCP/ctr_cvr.train` | 2,473,647,855 | 38,070,670 |
| `dataset/AliCCP/ctr_cvr.dev` | 274,769,757 | 4,229,235 |
| `dataset/AliCCP/ctr_cvr.test` | 2,711,167,840 | 43,016,614 |
| `dataset/AliCCP/ctrcvr_enum.pkl` | 8,588,099 | 非标准 pickle（首字节 `x`），评测不需要 |

三个文件表头一致：`click,purchase,101,121,122,124,125,126,127,128,129,205,206,207,216,508,509,702,853,301`。

### 2.2 加载器语义（`multitaskrec/dataset.py: AliCCPDataset`，冻结不改）

- 特征 = 表头第 3 列到倒数第 2 列（16 个：121…853）；`101`（干扰特征）与 `301`（第三标签列）被加载器排除。
- 标签：`click=col0`、`purchase=col1`、`business_scenario_information=col[-1]`；第三标签被就地变换：**raw==2 → 0，其余 → 1**（源码注释 `TODO 处理最后一个label`）。
- 模型侧 vocab = `AliCCP_Vocabulary_Size.copy()` 后 `pop("101")`、`pop("301")`（16 个特征 × embedding 5 = `input_size=80`，无 dense 特征）。**禁止**修改 `config.py` 全局字典。
- `data_size` 语义 = **按文件顺序取前 N 行**（前缀），不是随机采样。
- 数据行访问顺序 = 文件顺序；`env_ids` 按行号对齐（见 5.3）。

### 2.3 标签统计（实测）

审计预算下（train 前 1,000,000 / dev 前 200,000 / test 前 500,000 行）：

| 切分 | click=1 | purchase=1 | 第三标签变换后=1 | raw 分布 {1,2,3} |
|---|---|---|---|---|
| train | 4.66% | 264（0.0264%） | 99.22% | 385,740 / 7,839 / 606,421 |
| dev | 4.69% | 61（0.0305%） | 99.12% | 80,375 / 1,752 / 117,873 |
| test | 4.73% | 150（0.0300%） | 98.42% | 247,409 / 7,897 / 244,694 |

- **CVR 嵌套成立**：两文件中 `purchase=1 ⇒ click=1` 无例外（标准 CVR 语义）。
- **第三标签极不平衡**：变换后正例约 98–99%；raw=2 是稀有一侧。本协议冻结该定义（不做反向映射、不改数据集代码）；下游阈值按此校准。
- **CVR 正例极稀少**：val 61 / test 150（审计预算），任何 CVR 粒度结论都不可靠（Hanley–McNeil SE ≈ 0.038 / 0.025 @AUC 0.6）——协议只把 CVR 当活性/诊断指标（见第 10 节）。

### 2.4 顺序漂移（实测，决定"前缀"选择的依据）

沿文件向尾部扫描（各 20 万行）：train click 4.52% → 3.66% → 3.15%（头/中/尾），test 4.80% → 3.64% → 3.20%；第三标签正例率 97.7% → 99.3% → 97.8%（train）。**文件内分布有漂移**，因此：

- **选择确定性前缀**（三个文件各取头部固定行数）：三份前缀的头部统计彼此接近（click 4.66/4.69/4.73%），训练/验证/测试分布**互相可比**；零索引机制、零种子依赖；加载成本 O(预算)（加载器到预算即 break）。
- 拒绝"随机子采样"：需要全文件扫描物化索引，且会把漂移混入各切分、使 train 与 test 的整文件分布差异重新引入；还引入额外 split 种子。

### 2.5 重复行（实测，数据卫生披露）

对所用前缀做整行 sha1 精确比对（审计预算）：

| 项 | train 1M | dev 200k | test 500k |
|---|---|---|---|
| 切分内重复行 | 50,976 | 1,550 | 29,297 |
| train∩dev / train∩test / dev∩test | 9,338 | 10,493 | 1,513 |

- 特征全为高基数枚举，随机碰撞概率可忽略 → 这些是**真实重复记录**，且**跨文件重复**（供应商切分并非严格互斥）。约 2.1% 的 test 前缀行也出现在 train 前缀中。
- **本协议不修改数据**（见 N6），但把重复统计写入前缀指纹（第 5 节），并在 SUMMARY/报告中如实披露。该效应对所有 arm 同等存在。
- 正式运行预算下的重复统计将在指纹生成时计算并固化（审计值为预算相关量）。
- 全文件级别的重复规模未测量（超出本轮范围，属后续数据卫生议题）。

### 2.6 阶段语义（冻结）

- **阶段 1**：MPTRec（`num_tasks=2`），任务 = CTR、CVR；`MPTRecTrainManager.train_two_task`（含 `alpha` 调度、`cluster_2` 每 2 个 epoch、按 `sum(AUC_val)` 选点、patience）。
- **阶段 2（新任务泛化）**：第三任务 = `business_scenario_information`（列 `301`；MPTRec 脚本称 BSI、baseline 脚本称 ESI，**同一列，命名不一致仅为历史遗留**）。用 `NewTask` 头接收 `(dnn_input, gen_rep, spec_reps, env_embs)`，在**冻结 backbone**上训练，val BSI AUC 选点，test BSI AUC 单次评估。
- baseline 脚本的阶段 2 语义 = 载入 `.pt` + 新 tower（`train_one_task(2)`），与本协议不冲突，但首轮不跑。

### 2.7 计时实测（RTX 3060 Laptop 6GB，torch 2.6.0+cu124）

| 项 | 实测 |
|---|---|
| 加载 1M / 200k / 500k 行 | 3.8s / 0.9s / 2.0s |
| 阶段 1 训练步（batch 2000） | 51.7 ms → **每 1M 行 epoch ≈ 26s** |
| `cluster_2` 全训练集推理 | ≈ 7.3s / 1M 行 |
| val 评估（200k，双任务） | 1.5s |
| 阶段 2 训练步 | 19.8 ms → **每 1M 行 epoch ≈ 10s** |
| 峰值显存 | 138 MB（批量 2000） |

---

## 3. master 现状缺陷清单（本协议必须消除的公平性缺陷）

| 编号 | 现状（`fd75198`） | 后果 |
|---|---|---|
| F1 | `AliCCP_NewTask.py` 在阶段 2 脚本**内部重训阶段 1**（epochs=10，预算 5M/500k/5M）；`AliCCP_MPTRec.py` 完全不落盘任何 checkpoint | "先预训练、再泛化"没有产物身份；阶段 1 与阶段 2 无法分离复现；MPTRec 自家两脚本互不一致 |
| F2 | 样本上限不一致：`AliCCP_MPTRec.py` 10M/100k/10M；`AliCCP_NewTask.py` 与全部 baseline 5M/500k/5M | 同一"阶段 1"在不同脚本里训练集大小、val 集、test 集都不同，任何跨脚本比较失真 |
| F3 | 默认种子不一致：`AliCCP_MPTRec.py` 默认 `1688723512`，`AliCCP_NewTask.py` 默认 `1688738016` | 阶段 2 脚本内重训的阶段 1 与独立阶段 1 脚本默认就不是同一次预训练 |
| F4 | `env_ids = torch.randint(...)` 直接吃全局 RNG，且在模型初始化**之前**执行 | 改预算 → 改变 RNG 消耗 → 同种子下模型初始化也不同（隐式耦合）；env_ids 不落盘 |
| F5 | MPTRec **没有** `freeze_params()`；阶段 2 训练循环里 `get_infos()` 无 `torch.no_grad()`；backbone 未被显式 `eval()` | 梯度回传并累积到 backbone 的 `.grad`；"冻结"靠"优化器没放它"和"上一次调用恰好是 eval"两个巧合，不是保证 |
| F6 | 阶段 1 权重仅 baseline 脚本落盘（seed 命名，STEM 存到仓库根目录 `stem_alicpp_{seed}.pt`）；无 meta、无指纹 | 无法审计阶段 1 产出；阶段 2 靠 seed 字符串与阶段 1 隐式耦合 |
| F7 | 阶段 2 头初始化依赖阶段 1 训练轨迹消耗后的全局 RNG 状态 | 阶段 1 任何改动（预算/epoch）都会静默改变阶段 2 初始化 |
| F8 | 第三标签语义无文档：`TODO 处理最后一个label`，变换后正例 98–99% | 易被误读为"标签错位"；需在协议中固化并披露 |
| F9 | 供应商切分存在跨文件精确重复行（2.5 实测） | 训练/测试非严格互斥；需量化披露而非无视 |
| F10 | 文件内分布漂移（2.4 实测）；data_size 语义为前缀这一点未文档化 | 不同脚本取不同前缀长度时，比较基准悄悄变化 |

### 处置原则

F1/F2/F3/F4/F5/F6/F7 由本协议的分层设计消除（第 5–7 节）；F8/F9/F10 属数据事实，**冻结披露**（量化写入指纹与 SUMMARY），不在本分支修改数据或标签定义。

---

## 4. 分支隔离规则

### 4.1 允许的改动（白名单）

- 新增 `aliccp_benchmark/`（协议、指标、测试）与 `run_aliccp_benchmark.py`
- 前缀指纹 / 环境种子 / Stage-1 产物规范的实现
- 冻结与 eval 语义（外部参数遍历实现，不改模型类）
- 日志、指标、门禁与 SUMMARY 追加
- `artifacts/aliccp_bench/` 目录规则与 `.gitignore` 规则
- 文档

### 4.2 禁止的改动（黑名单）

- `multitaskrec/*`、`config.py` 的任何修改（import 复用允许）
- master 既有脚本（`AliCCP_*.py`、`CensusIncome_*.py`）与 `baseline/*` 的任何修改
- 模型结构、损失项、超参语义（第 8 节论文值）
- 数据文件、标签定义、去重/清洗
- 复用或修改 CensusIncome 协议的常量/阈值

### 4.3 工作流

1. 本分支验证通过 → 经用户批准后才能合并回 `master`；**本分支不合并、不推送、不触碰 `master`**（推送需用户显式批准）。
2. 任何模型想法必须**先等本协议合并**，再从 `master` 拉 `exp/<name>`；想法分支**只能引用**本协议，不得修改本协议文件。
3. 一票否决：任何使前缀指纹不一致、或使冻结检查失败的改动，不得进入原型对比。
4. 本分支**只支持 AliCCP**；CensusIncome / ByteRec 不运行、不配置。

---

## 5. 固定前缀样本与指纹（A2 / A4 的定义）

### 5.1 预算（协议常量；首轮短跑与复跑必须一致）

| 项 | 常量 | 来源与理由 |
|---|---|---|
| train | `dataset/AliCCP/ctr_cvr.train` 前 **2,000,000** 行 | 与仓库自带的快速调参脚本 `test/AliCCP_NewTask.py`（2M）同量级；阶段 1 每 epoch ≈ 52s，全程可控 |
| val | `dataset/AliCCP/ctr_cvr.dev` 前 **500,000** 行 | 与全部 baseline 阶段 1 脚本的 val 上限一致；BSI 负例 ≈ 4.4k，val 侧 SE ≈ 0.005 |
| test | `dataset/AliCCP/ctr_cvr.test` 前 **1,000,000** 行 | BSI 负例 ≈ 1.6 万，test 侧 SE ≈ 0.0025 @AUC0.7；CVR/CTR 侧亦优于 500k |
| 前缀标识 | `PREFIX_TAG = "p2M-v500k-t1M"` | 进入 run_id / stage1_id / 指纹 |

**为什么是前缀**：见 2.4（漂移实测 + 与仓库 `data_size` 语义完全一致 + 零种子依赖 + 加载 O(预算)）。任何预算变更 = 新的前缀身份（新指纹、新 stage1_id），且**必须在看到结果前**改本文件。

### 5.2 前缀指纹（内容寻址，写入 `artifacts/aliccp_bench/splits/<PREFIX_TAG>/prefix_fingerprint.json`）

```json
{
  "prefix_tag": "p2M-v500k-t1M",
  "budgets": {"train": 2000000, "val": 500000, "test": 1000000},
  "files": {"train": {"path": "...", "size_bytes": ..., "prefix_sha256": "..."}, "...": {}},
  "header_sha256": "...",
  "label_counts": {"train": {"click1": ..., "purchase1": ..., "bsi_pos": ..., "bsi_raw": {"1": ..., "2": ..., "3": ...}}, "...": {}},
  "duplicate_stats": {"train_within": ..., "val_within": ..., "test_within": ...,
                       "train_val": ..., "train_test": ..., "val_test": ...},
  "fingerprint_sha256": "..."
}
```

- `prefix_sha256` = 对"表头 + 前 N 条数据行"的**原始字节**做 sha256（确定性、与加载器读取范围一致）。
- `label_counts` / `duplicate_stats` 在指纹构建时计算一次并固化（阶段 1 进程生成；此后只读）。
- **A2 门禁**：每次运行重算各前缀的 `prefix_sha256` 与加载器实际标签计数，必须与指纹一致；阶段 1 / 阶段 2 进程报出的 `fingerprint_sha256` 必须相同。
- **A4 门禁**：`len(dataset) == budget`（三切分）；加载器读出的 click1 / purchase1 / bsi_pos / bsi_raw 与指纹**逐项相等**（这是比 AUC 更硬的标签对齐检查，覆盖 F8 的语义风险）。
- 指纹改动 = 协议改动：先改本文件，再重建指纹；已判定过的 run 不得用新指纹重判。

---

## 6. 随机性来源与顺序纪律

### 6.1 两个种子（AliCCP 无 split 种子——前缀是确定性的）

| 参数 | 常量 | 职责 | 不得影响 |
|---|---|---|---|
| `--model-seed` | `1688723512`（仓库 AliCCP 全脚本默认、注释中的 5 seed 之一；首轮唯一 seed） | 模型初始化、训练随机性（dropout）、阶段 2 头初始化 | 前缀构成 |
| `--env-seed` | `20261003`（协议常量） | 只决定初始 `env_ids` | 模型初始化、前缀 |

- `env_ids` 用**独立 `torch.Generator`** 产生（`make_env_ids`，不消耗全局 RNG），长度 = train 预算，入口位置对齐（6.2）。
- 阶段 2 头初始化：阶段 2 进程**重新播种** `model_seed` 后先建数据、再从头初始化 `NewTask`（backbone 来自产物、`env_ids` 来自产物、两者均不消耗 RNG）→ 头初始化与阶段 1 训练轨迹**解耦**（消除 F7）。
- 模型初始化前不得有任何全局 RNG 消耗（数据加载、`Generator` 均不消耗）。

### 6.2 顺序纪律（硬约束）

1. 三个 `DataLoader` 一律 `shuffle=False`、`batch_size=2000`、`num_workers=0`、`drop_last=False`（与仓库一致；`batch_size` 同时进入 `alpha` 调度公式，必须与仓库相同）。
2. `env_ids` 按行号对齐训练集位置；`MPTRecTrainManager` 用 `self.batch_size*step` 切片取 batch 环境——因此**任何 shuffle 化都会静默错位**，协议禁止。
3. `cluster_2` 每 2 epoch 在训练集上重聚类（含 `evaluation` 模式），聚类结果覆盖 `env_ids`；阶段 1 结束时把**最终** `env_ids` 落盘（而非初始值）。
4. 前缀 = 文件顺序前 N 行；禁止在加载器之外重排样本。

---

## 7. 固定 Stage-1 产物与真冻结 / eval

### 7.1 Stage-1 产物规范（内容寻址、只读、禁止覆盖）

```
artifacts/aliccp_bench/stage1/<stage1_id>/
  backbone.pt      # mptrec.state_dict()（CPU）
  env_ids.pt       # 阶段 1 结束时（cluster 覆盖后）的最终 env_ids
  meta.json        # 见下
```

- `stage1_id = s1-<指纹前8>-m<modelSeed>-e<epochs>-<配置哈希前8>`（配置哈希含 PREFIX_TAG、预算、全部模型超参与 batch）。
- `meta.json`：model/env seed、PREFIX_TAG、预算、`fingerprint_sha256`、全部模型超参、逐 epoch 指标、best epoch、best val AUC（CTR/CVR）、best 权重 `backbone_sha256`、test CTR/CVR AUC（单次）、cluster 事件、env_acc 诊断、代码 commit、torch/cuda/python 版本、墙钟耗时。
- 生成后**不得原地覆盖**：配置/种子/预算任一变化 → 新 `stage1_id`。
- 生成失败或被打断 → 该目录不得作为后续阶段 2 的输入（meta.json 缺失即视为无效）。

### 7.2 阶段 2 的引用方式

阶段 2 **只允许**通过 `--stage1-dir` 引用产物；禁止在阶段 2 内训练、微调或重新初始化 backbone；禁止重抽 `env_ids`。

### 7.3 真冻结三件套（全部满足才算冻结；外部参数遍历实现，不改模型类）

1. **参数级**：backbone 全部参数 `requires_grad_(False)`
2. **模式级**：`backbone.eval()`，且阶段 2 全程不得调用 `backbone.train()`
3. **计算图级**：`get_infos()` 始终在 `torch.no_grad()` 下调用（训练循环与评估一致）；优化器只接收 `NewTask` 参数

### 7.4 冻结与身份的机器验证

| 门禁 | 判据 |
|---|---|
| **A1** | 阶段 2 前后 backbone 全参数（按名排序）sha256 完全一致；阶段 2 结束后 backbone 所有参数 `.grad is None` |
| **A5** | 阶段 2 读到的 `env_ids` sha256 == 阶段 1 落盘值 |
| **A6** | 阶段 2 加载后立即计算的 backbone sha256 == 阶段 1 `meta.json` 记录值；阶段 2 的运行记录写明所用 `stage1_id` |

### 7.5 评测语义

- 阶段 1：val 只用于选点；test CTR/CVR AUC **在选点结束后评一次**（用于活性记录，不参与任何选择，此后 backbone 冻结不再触碰）。
- 阶段 2：val BSI AUC 只用于 early stop；test BSI AUC **在全部训练结束后评一次**，为**主报告值**；best 头权重落盘 `newtask.pt` 并在最终评估前重新加载。
- 全程禁止用 test 做任何模型选择。

---

## 8. 首轮配置（smoke + 单次 short 基线，均已预先登记）

### 8.1 模型与优化超参（论文值，冻结；与 master 脚本逐项一致）

| 项 | 阶段 1（MPTRec, num_tasks=2） | 阶段 2（NewTask 头） |
|---|---|---|
| vocab | `AliCCP_Vocabulary_Size.copy()` 后 pop `101`,`301`（16 特征） | 同左 |
| embedding_size / input_size | 5 / 80 | `input_size=80`, `rep_dim=64` |
| expert / tower | experts `[128,64]`、towers `[32,32]` | tower `[32,32]`（接在 rep_dim 上） |
| dropout | `[0.1, 0.3]` | 无（与仓库一致） |
| lr / batch | `1e-4` / `2000` | `1e-4` / `2000` |
| uni_coe / env_coe | `0.9` / `0.1` | — |
| reg_embedding / reg_dnn | `1e-4` / `7e-6` | `reg_dnn=7e-6` |
| 损失 | 仓库 `MPTRecTrainManager.train_two_task` 原样 | `BCELoss(pred, y) + newtask.get_l2_reg()`（仓库口径） |
| 设备 | `cuda:0` | `cuda:0` |

**训练循环复用纪律**：阶段 1 直接调用仓库 `MPTRecTrainManager.train_two_task`（只允许以子类方式**记录** `cluster_2` 的事件与返回值，不得改变其计算行为）。阶段 2 头训练循环按仓库 `AliCCP_NewTask.py` 的口径重写为运行器代码（backbone 部分改为 no_grad + 真冻结）。

### 8.2 Smoke 运行（管线正确性，仅此一次）

| 项 | 取值 |
|---|---|
| 预算 | train 20,000 / val 5,000 / test 10,000 |
| 阶段 1 | epochs=1、patience=1（不触发 cluster，B4 记录为"0 次"） |
| 阶段 2 | epochs=1、patience=1 |
| 种子 | model `1688723512`、env `20261003` |
| 门禁 | **执行 A 类**；B/C 类只记录不判定（短跑不用于性能判定） |
| 预期时长 | < 1 分钟 |

### 8.3 单次 short 基线运行（唯一一次；结果进 SUMMARY，不用于性能结论）

| 项 | 取值 |
|---|---|
| 预算 | train 2,000,000 / val 500,000 / test 1,000,000（5.1） |
| 阶段 1 | epochs=3、patience=2、batch 2000（其余 8.1） |
| 阶段 2 | epochs=5、patience=2、batch 2000 |
| 种子 | model `1688723512`（唯一）、env `20261003` |
| 门禁 | A 类 + B 类全部执行判定；C 类落盘 |
| 预期时长 | 6–9 分钟（基于 2.7 实测外推：阶段1 ≈ 3×(52s)+cluster(15s)+val(3×4s) ≈ 3min；阶段2 ≈ 5×20s+eval ≈ 2min；加载 ≈ 15s×2 进程） |
| 播报纪律 | 只跑 1 次；不搜参、不换结构、不调超参；A3 复跑按需（不阻塞） |

---

## 9. 指标定义

### 9.1 主指标（性能）

| 位置 | 指标 | 用途 |
|---|---|---|
| 阶段 1 | `AUC-Val-CTR`、`AUC-Val-CVR`（每 epoch） | 选点（仓库 `sum(auc_val)` 规则原样）+ 活性检查 |
| 阶段 1 | `AUC-Test-CTR`、`AUC-Test-CVR`（结束后一次） | 活性记录（不参与选择） |
| 阶段 2 | `AUC-Val-BSI`（每 epoch） | early stop 选点 |
| 阶段 2 | `AUC-Test-BSI`（结束后一次） | **主报告值** |
| 汇总 | `metrics.json` | 逐 epoch 值、best epoch、best val、final test |

### 9.2 机制与完整性指标

| 编号 | 指标 | 定义 | 类别 |
|---|---|---|---|
| M1 | `env_acc` | 阶段 1 结束后，在**训练集前 40 万行（前 200 个 batch，确定性）**上 `env_pred` 与最终 `env_ids` 的一致率（no_grad） | C（诊断） |
| M2 | cluster 事件 | 每次 `cluster_2()` 的 epoch、`diff_num`、`env_0/env_1` 计数（子类记录，不改行为） | B4 判定 + C |
| M3 | `gate_mean` | 阶段 2 结束时 `NewTask.gate_network` 在 val 上的平均输出（2 维：新 spec 分支 / gen 分支），no_grad 累积 | B3 判定 + C |
| M4 | `backbone_sha256` | 阶段 2 前后各一次 | A1 |
| M5 | `prefix_sha256` / `fingerprint_sha256` / `env_ids_sha256` | 指纹三件 | A2 / A5 |
| M6 | 运行元数据 | 墙钟、峰值显存、commit、torch/cuda 版本、逐 epoch loss（uni/fuse/env） | C |

**实现约束**：M1/M3 在 `torch.no_grad()` 下按 batch 累积后一次算，不保留中间张量；不做 FLOPs（`fvcore` 在 CPU 上极慢，与仓库脚本一致地跳过）。

### 9.3 报告纪律

- 不得用 test 集做任何模型选择
- 不得只报告最好的 seed / 最好的 run
- 不得省略失败的 run 或失败的门禁
- smoke 运行的结果不得用于任何性能陈述

---

## 10. 验收阈值（只允许在看到结果前修改；每次修改单独 commit 并在此记录理由）

### A 类：协议正确性（硬门禁，smoke 与 short 均执行）

| 编号 | 判据 |
|---|---|
| A1 | 阶段 2 前后 `backbone_sha256` 一致，且结束后 backbone 参数 `.grad` 全为 `None` |
| A2 | 每次运行重算的 `prefix_sha256` ×3 与指纹一致；阶段 1 / 阶段 2 报出的 `fingerprint_sha256` 相同 |
| A3 | **可复现性（按需触发，不阻塞首轮）**：同 `(model seed, env seed, 预算, 配置)` 复跑，两次 `AUC-Test-BSI` 差异 ≤ `1e-9` 且 `backbone_sha256` 一致；首轮只跑一次，不因未复跑而失败 |
| A4 | `len(dataset)` == 预算 ×3；加载器标签计数（click1/purchase1/bsi_pos/bsi_raw）与指纹逐项相等 |
| A5 | 阶段 2 读到的 `env_ids_sha256` == 阶段 1 落盘值 |
| A6 | 阶段 2 加载后 backbone sha == 阶段 1 meta 记录值；run 记录含 `stage1_id` |

### B 类：活性与机制（硬门禁，仅 short 运行判定；smoke 只记录）

| 编号 | 判据 | 依据 |
|---|---|---|
| B1 | `AUC-Val-CTR ≥ 0.55`；`AUC-Val-CVR ≥ 0.50`；`AUC-Test-BSI ≥ 0.53` | CTR：远高于随机、远低于预期（只抓流程性破坏）；CVR：**刻意近乎无约束**——val 正例仅 61，SE≈0.038，任何更紧阈值在统计上无意义，标签错位由 A4 的精确计数门禁负责；BSI：抓标签对不齐/冻结失效类破坏。三者均非性能目标 |
| B2 | `\|AUC-Val-BSI(best) − AUC-Test-BSI\| ≤ 0.05` | dev 与 test 为不同文件（BSI 正例率实测差 0.7pp），≈7× 合并 SE，只抓粗破坏 |
| B3 | val 上 `gate_mean` 每维 ∈ `[0.05, 0.95]`（未坍缩） | 机制活性 |
| B4 | 每次 `cluster_2` 后两环境各占训练集 ≥ 5% | 聚类未退化（smoke 无 cluster 调用 → 记为"0 次，未判定"） |

### C 类：诊断（只落盘，不判定）

`env_acc`、cluster 计数、`gate_mean` 数值、loss 轨迹（uni/fuse/env 逐 epoch）、CVR 的 val/test AUC（作为诊断参考，见下）、运行时/显存。

### 10.1 有意义的改进阈值（**provisional**，供后续 arm 使用）

- **主判据**：同一 `stage1_id` 下，`ΔAUC-Test-BSI ≥ +0.005`（绝对）**且** `ΔAUC-Val-BSI > 0`（方向一致）。
- 依据：test 1M 下 BSI 负例 ≈ 1.6 万，Hanley–McNeil SE ≈ 0.0025 @AUC0.7 → 0.005 ≈ 2×SE；要求 val 同向以防单侧噪声。
- **provisional 标记**：该阈值在拿到 A3 复跑证据（本数据集实测 run-to-run 噪声）之前不得视为定稿；证据到位后按 10.2 流程收紧或放宽。
- **禁止**用 CVR 或 CTR 的差值宣称改进（CVR 正例过少、CTR 非新任务指标）；禁止用 smoke 数据宣称任何改进。

### 10.2 阈值变更纪律

阈值只允许在看到受影响 run 的结果**之前**修改；每次修改单独 commit 并在本节记录理由；已被旧阈值判定的 run 不得用新阈值重判。

### 明确不设的阈值

- 不设"论文对齐 / 超越基线"的性能门槛（属后续阶段，N2）
- 不设运行时长 / 显存门槛
- 不设 CVR/CTR 的"改进"门槛

---

## 11. 实验产物目录与 Git 保留策略

### 11.1 目录结构

```
artifacts/aliccp_bench/
  splits/<PREFIX_TAG>/prefix_fingerprint.json
  stage1/<stage1_id>/{backbone.pt, env_ids.pt, meta.json}
  runs/<run_id>/{config.json, metrics.json, gate_report.json, newtask.pt, run.log}
  SUMMARY.md          # 唯一入库文件
```

- `run_id`：`<YYYYMMDD-HHmm>-<PREFIX_TAG>-m<modelSeed>-<smoke|short>-<commit7>`
  例：`20261003-1530-p2M-v500k-t1M-m1688723512-short-fd75198`

### 11.2 Git 保留策略

| 类别 | 内容 | 处置 |
|---|---|---|
| 入库 | 本协议、`aliccp_benchmark/`（含 tests）、`run_aliccp_benchmark.py`、`artifacts/aliccp_bench/SUMMARY.md`、`.gitignore` 规则 | commit |
| 不入库 | 全部权重（`*.pt`）、指标/指纹（`*.json`）、日志（`*.log`） | 现有 `.gitignore` 已拦截（`*.pt`/`*.json`/`*.log`），另加目录级规则 |
| 新增忽略规则 | `artifacts/aliccp_bench/splits/`、`artifacts/aliccp_bench/stage1/`、`artifacts/aliccp_bench/runs/`，保留 `!artifacts/aliccp_bench/SUMMARY.md` | 实现时加入 `.gitignore` |

`SUMMARY.md` 每 run 一行：`run_id`、commit、`auc_test_bsi`、A/B 门禁逐项、`stage1_id`、备注（smoke/short 标记）。**只追加，永不重写已有行。**

### 11.3 保留与容量

- `stage1/`：同一 `stage1_id` 只读；空间紧张时优先删 smoke 产物。
- `runs/`：只保留"支撑过 SUMMARY 结论或门禁判定"的 run。
- `artifacts/aliccp_bench/` 总计 ≤ 2 GB；超限从旧到新删 run，但**永不删除** SUMMARY 引用且尚未被取代的 run。
- 永不入库：数据集、`.venv/`、任何 test 集预测明细。
- 禁止 `git add -f` 绕过忽略规则；**禁止触碰或添加 `artifacts/census_stage1_grad/` 等无关目录**（提交只允许显式路径，禁止 `git add -A`/`git add .`）。

---

## 12. 基线比较政策（后续 arm 必须遵守）

1. **Stage-2-only arm（预期的主要形态）**：必须引用与基线相同的 `stage1_id`；禁止重训/重初始化 backbone；比较对象 = SUMMARY 中同指纹、同 `stage1_id` 的基线 run；判据见 10.1；所有 A 类门禁必须通过，否则比较作废。
2. **动阶段 1 的 arm**：必须在分支文档中显式声明；产生**新** `stage1_id`；并在新 backbone 上**重跑未改动的阶段 2 头**作为同 backbone 对照——只允许"同 backbone 配对"比较。
3. arm 不得修改协议常量、预算、指纹；指纹变化 = 可比性作废（需先改本文件）。
4. 单 seed 筛查先行（本协议 `model_seed`）；只有达到 10.1 阈值且 A3 复跑证据到位后，才允许扩展多 seed；扩展 seed 列表由用户显式指定。
5. 报告纪律同 9.3；任何 B 类失败必须原样披露（不得只报通过的 arm）。

---

## 13. 本分支明确不含的内容

- 任何模型改进机制（新专家、门控、投影、prompt、adapter、温度、损失设计）
- baseline 模型族（SharedBottom/MMOE/PLE/STEM/SparseSharing/CsRec）的运行与改造
- 数据去重、标签重定义、分布校正
- CensusIncome / ByteRec 的任何运行、配置、常量复用
- 论文级多 seed / 全量预算运行
- 对 `multitaskrec/*`、`config.py`、master 脚本、`baseline/*` 的任何修改

**强约束**：本分支对 `multitaskrec/model.py` 与 `multitaskrec/train.py` 的改动为**零**。冻结通过外部参数遍历实现；阶段 1 训练循环复用仓库 `MPTRecTrainManager`（子类仅记录不改行为）。

---

## 14. 最小实现面（TDD，本轮交付）

1. `aliccp_benchmark/protocol.py`：常量、前缀指纹构建/校验、`make_env_ids`、stage1 产物读写（禁止覆盖）、冻结三件套、`backbone_sha256`、`.grad` 断言、run_id、SUMMARY 追加
2. `aliccp_benchmark/metrics.py`：AUC 封装、env_acc、cluster 事件记录（子类）、`gate_mean`、A/B/C 门禁判定（纯函数）、metrics/gate JSON schema
3. `run_aliccp_benchmark.py`：`stage1` / `stage2` 两个子命令；stdout 同步落盘 `run.log`；写 `config.json`/`metrics.json`/`gate_report.json`/`newtask.pt`；追加 SUMMARY
4. `aliccp_benchmark/tests/{test_protocol,test_metrics,test_smoke}.py`：`unittest`（venv 无 pytest）、CPU-only、小合成数据；test_smoke 在临时 AliCCP 格式小文件上走通"stage1 产物 → stage2 加载 → 冻结 → 1 步训练 → 门禁"全链
5. `.gitignore` 规则（11.2）
6. `artifacts/aliccp_bench/SUMMARY.md`（运行后生成）

顺序：**先提交本协议 → 再写测试 → 再实现 → smoke → short 运行 → 记录与本地提交**。实现遵循仓库现有代码风格，不做无关重构。

---

## 附录 A：协议相对 master 现状的差异清单

| master 现状 | 本协议处置 | 对应节 |
|---|---|---|
| F1 阶段 2 脚本内重训阶段 1；阶段 1 不落盘 | 阶段 2 只从 `--stage1-dir` 加载；Stage-1 产物内容寻址落盘 | 7.1–7.2 |
| F2 样本上限不一致（10M/100k/10M vs 5M/500k/5M） | 固定前缀预算 2M/500k/1M，指纹固化 | 5.1–5.2 |
| F3 两脚本默认种子不同 | 唯一 model seed `1688723512`；预算/种子进指纹与 stage1_id | 6.1 |
| F4 env_ids 吃全局 RNG | 独立 `torch.Generator(env_seed)`；最终 env_ids 落盘、阶段 2 读取 | 6.1、7.1 |
| F5 无冻结保证、get_infos 无 no_grad | 真冻结三件套 + A1 断言；不改模型类 | 7.3–7.4 |
| F6 无 meta/指纹/（MPTRec）无落盘 | meta.json + 指纹三件 + SUMMARY | 7.1、5.2、11 |
| F7 头初始化依赖阶段 1 轨迹 | 阶段 2 独立进程重播种，头初始化与阶段 1 解耦 | 6.1 |
| F8 第三标签语义无文档、极不平衡 | 冻结定义（2→0 其余→1），A4 精确计数门禁，阈值按此校准 | 2.3、5.2、10 |
| F9 跨文件重复行 | 量化写入指纹并在 SUMMARY 披露；不修改数据 | 2.5、5.2 |
| F10 文件内分布漂移、前缀语义未文档化 | 前缀选择显式文档化 + 指纹固化 | 2.4、5.1 |

## 附录 B：审计方法与证据

- 审计脚本（只读，临时目录 `.audit_tmp/`，不入库）：行数/字节（`wc -l`/`ls`）、标签与漂移扫描（前缀与偏移段各 20 万行）、click×purchase 交叉表、前缀整行 sha1 重复比对、`AliCCPDataset` 实例化计时、MPTRec 训练/评估步计时探针。
- 关键证据数字见 2.1–2.7；正式运行的**canonical** 计数以前缀指纹为准（预算 2M/500k/1M）。
- 本协议对模型的全部行为假设（训练循环、选点规则、`alpha` 调度、`cluster_2` 语义、NewTask 头结构）均来自 `fd75198` 的 `multitaskrec/train.py`、`multitaskrec/model.py`、`AliCCP_MPTRec.py`、`AliCCP_NewTask.py` 源码走读，未做任何修改。
