# AliCCP 阶段 2 范数受控残差 Prompt 的条件对应因果消融（shuffled conditioning；canonical seed-2、5-epoch short 单因子）

- **状态**：预注册已写死（本文件在本分支任何实现、任何 run **之前单独提交**；commit C1）。结果只在第 10 节以新增小节回填；第 0–9 节的判据、数字、构造与运行规程不得在看到结果后改动。
- **日期**：2026-10-05
- **适用分支**：`exp/aliccp-stage2-residual-prompt-shuffled-condition`（自 `infra/aliccp-fair-benchmark` @ `8133d32` **独立拉出**；不从任何 exp 分支拉出、不继承其运行时状态；不合并 `master`、不触碰其它 worktree；`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`aliccp_benchmark/metrics.py`、既有测试三件、master 脚本与 `baseline/*` 零修改）。
- **被消融对象（只读引用；系谱钉死）**：残差 prompt 机制 = `exp/aliccp-stage2-residual-prompt-seed-replication`（预注册 `e931cd7`，结果 `a40836e`）@ `013e105:aliccp_benchmark/residual_prompt.py`（blob `674213f619c5d5039811c71242a7727118348daf`，LF sha256 `b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc`）；其系谱：seed1 线（预注册 `221580a` / 实现 `79ddefa`（blob `5dc7158ca999c9e7e6217a999c37869231411549`）/ 结果 `3e2f083`）→ Census 祖本 `99b9510`（blob `107221b26382da7fd44990e167d9a61da2680d92`）。
- **上游协议（只引用、不修改）**：
  1. `docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（AliCCP 公平评测协议）——前缀预算、种子、Stage-1 产物、真冻结三件套、A/B/C 门禁、SUMMARY 台账、止损纪律全部沿用；
  2. `docs/superpowers/specs/2026-10-04-aliccp-stage2-residual-prompt-seed-replication-design.md`（seed-2 复现预注册）——机制门禁 M0+G1–G8、臂级三标签、U1/U2 口径、参照头钉死逐字沿用（本文件 §5.3 引用即为该预注册 §4/§5）；
  3. `exp/aliccp-stage2-residual-prompt-20epoch` @ `10e86dc`（20-epoch 终止判定点）——审计方法（以磁盘原始 run 产物 + git 对象为准）、二级分类三标签边界、`NO_CLEAR_IMPROVEMENT` 的机械 headroom 结构、`commit-between-runs` 清洁树纪律、独立复核脚本形态（纯 JSON 重推）**逐字沿用**；该分支结论（`NOT_PERSIST` ∧ claim=true ∧ `ABLATION_ELIGIBLE`）为本实验的立项依据；
  4. `exp/aliccp-stage2-residual-prompt-five-seed` @ `0935257`（结果）/ `28aa1fe`（provenance tip）——五 seed canonical 配对与跨进程逐位确定性证据（§1.2-A6）。
- **定位声明**：本实验是残差 prompt 线的**首个机制消融**（条件对应因果检验）。被检验的问题不是"该机制是否有效"（既有记录：5-epoch seed2 `VALID_POSITIVE` +0.0081；20-epoch 判定点 `NOT_PERSIST`但二级 `POSITIVE_IMPROVEMENT`、`ABLATION_ELIGIBLE`），而是：**观测到的效用是否依赖"每个样本用自身的特征条件化"这一对应关系**，而非额外参数、全局扰动或范数/门控效应。单一变化 = 目标样本与 prompt 生成器条件特征之间的对应关系（§2）。无新颖性主张、无改进主张、不调参、不放宽任何门限。

**提交时序（预注册纪律）**：C1 本文件（审计 + 预注册，单独提交）→ C2 实现 + 测试（TDD；两 run 的 `commit` 字段即该提交）→ C2 之后跑 **P0 配对基线臂**（前台）→ C3 提交 P0 的 SUMMARY 行 → 跑 **P1 shuffled 臂**（前台）→ 分析恰一次 → 独立复核恰一次 → C4 结果回填（§10）+ P1 的 SUMMARY 行 → push。

---

## 0. 判定问题

对照三分：**P0 = 本实验新跑的配对基线臂**（seed2、short、5-epoch；协议默认）；**P1 = 本实验新跑的 shuffled-condition 臂**（唯一变化 = §2 的条件对应置换）；**C = 已提交 correct-condition 臂**（`20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg`，hash 核验后**仅作上下文对照**，不重跑）。

记 `G_c := test(C) − test(P0)`（correct 臂相对配对基线的 test 增量；已有记录 +0.008103113327467715）、`G_s := test(P1) − test(P0)`（shuffled 臂增量）、`L := G_c − G_s`（条件错配造成的增量损失）、`R := G_s / G_c`（保留比）。

- **Q1（因果主判定）**：shuffled 臂是否相对 correct 臂**实质损失**（`L ≥ +0.001`），且 correct 臂本身实质为正（`G_c ≥ +0.001`）？→ §5.5 判定树：`SAMPLE_CONDITION_SUPPORTED` / `CONDITION_ALIGNMENT_NOT_SUPPORTED` / `INVALID`。
- **Q2（保留比）**：`R` 的实测值与"shuffled 是否仍高于配对基线"（`G_s > 0` 与 `G_s ≥ +0.001` 两口径，均报告）；不参与 Q1 分类，仅注解子类（§5.5 子类）。
- **Q3（二级效用分类）**：P1 相对 P0 的 `Δtest` 落入 `POSITIVE_IMPROVEMENT` / `NO_CLEAR_IMPROVEMENT` / `CLEAR_DEGRADATION` 哪一类（边界先于结果写死，§5.6）；若 `NO_CLEAR`，按 §5.6 机械评估 headroom。
- **Q4（完备性）**：前置条件（identity / 基线复现 / A 类 / 机制门禁 / 置换门禁）是否全真？任一假 ⇒ `INVALID`（不解读效用）。

**纪律声明（预注册）**：历史判定与全部数值（seed1 `VALID_NEGATIVE`、seed2 `VALID_POSITIVE`、10-epoch `PERSISTS`、20-epoch `NOT_PERSIST`/claim/`ABLATION_ELIGIBLE`、5-seed 均值等）**原样保留、不改写、不重判**；`+0.0055` 历史效用门**不适用于** shuffled 臂的主判定（§5.7 单列保留其真值）。本实验只用 §4 新跑的两条 run 做因果与分类判定；correct 臂 run 仅作 hash 核验后的上下文对照。

---

## 1. 审计（2026-10-05，只读；先于本文件提交）

### 1.1 审计方法与范围

- **原则**：不信任任何文档转述，全部以 **git 对象**（blob / `ls-remote` tip）与**磁盘上的原始 run 产物**（本 worktree 内按 §1.5 逐文件 sha256 核验后的只读副本）为准重新核对；不重跑、不修改任何历史 run。
- 审计对象与 remote tip（`git ls-remote origin` 实测）：seed-replication `a40836e`（`a40836e3d6a7f73c4943bada98139d414957301a`）、five-seed `28aa1fe`（`28aa1feae1fea0762136beebdfe6eb8387edb170`，结果提交 `0935257`）、gate `3e2f083`、longer-budget `e46e5d2`、20epoch `10e86dc`（`10e86dc6cbd047b103211ab75dd99c1c68db3526`）、基点 `8133d32`。
- blob 钉死实测（`git rev-parse`）：`013e105:aliccp_benchmark/residual_prompt.py` = `674213f619c5d5039811c71242a7727118348daf`（LF sha256 `b3b93b3a…`）；`013e105:aliccp_benchmark/bench.py` = `bf3647686838157bf5fd7645615faad9f2d7185d`（LF `b063a366…`）；`013e105:run_aliccp_benchmark.py` = `229c25a1c86719fa6fb6b057d6cff3f4120fa053`（LF `2966ec39…`）；`013e105:aliccp_benchmark/tests/test_residual_prompt.py` = `768a477b5c71af32c5c59ec6feb20c29f7873557`（LF `dab737a1…`）；`79ddefa:…residual_prompt.py` = `5dc7158c…`。
- 审计为只读核对（git + 原始 JSON 重读），无可跟踪审计脚本；全部命令可复现。

### 1.2 审计结果（逐项）

| # | 审计项 | 证据（独立重推） | 结果 |
|---|---|---|---|
| A1 | seed2 5-epoch 配对完整性（本实验的参照链底座） | 基线 `20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07`（commit `79b5e07`，dirty=false，epochs 5/patience 2/seed 1688723740）：test `0.5974422649550507` / val `0.5809347091990792` / best=5/5 / gate_mean `[0.789178, 0.2108221875]`；臂 `20261004-0325-…-013e105-rpg`（commit `013e105`，dirty=false，variant `residual-prompt`）：test `0.6055453782825184` / val `0.5895572066556973` / best=5/5 / gate_mean `[0.7686165625, 0.23138340625]`；`Δtest = +0.008103113327467715`、`Δval = +0.008622497456618139`（逐位重推）；臂 `rp_arm` = `VALID_POSITIVE`，M0+G1–G8 全 PASS | PASS |
| A2 | seed2 10-epoch / 20-epoch 链（上下文 + 立项依据） | 10-epoch：基线 `2b1d585`（test `0.6614121223087556` / val `0.6401127811737995`）vs 臂 `f2ccec2-rpg`（test `0.6688517876477933` / val `0.6449491259210769`），`Δtest +0.0074396653390377265`、`Δval +0.0048363447472773435`，`PERSISTS`；20-epoch：基线 `45ab7e7`（test `0.6862927818762538` / val `0.6652705173239858`）vs 臂 `eafc336-rpg`（test `0.6904487492541437` / val `0.667583990949807`），`Δtest +0.004155967377889924`、`Δval +0.0023134736258212385`，`NOT_PERSIST`（C1 单项失败）∧ 二级 `POSITIVE_IMPROVEMENT` ∧ claim=true（六项）∧ `ABLATION_ELIGIBLE`；预算趋势 `MONOTONE_NARROWING`（+0.00810 → +0.00744 → +0.00416），三判定点两臂**均右删失**（best=末 epoch） | PASS |
| A3 | 跨进程/跨 commit 确定性（支撑"新跑基线可与历史记录逐位对账"） | five-seed 记录：seed3–5 的两次 pass 重跑 **逐位相等**（含 `newtask.pt` 字节；仅 run_id/路径字符串不同）；20-epoch 记录：20-epoch 基线前 10 epoch 逐位等于 10-epoch 基线、前 5 epoch 逐位等于 5-epoch 基线（两臂同理）⇒ 同 seed 的 stage2 训练在本环境确定性可复现 | PASS |
| A4 | 参照头身份 | `79b5e07/newtask.pt` sha256 = `90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f`（与 20-epoch 钉死值一致） | PASS |
| A5 | 机制实现的输入使用面（消融靶点定位） | `NewTask.forward`（`8133d32:multitaskrec/model.py`，零改动）在 3 处使用 `dnn_input`：`projection_network`、`gate_network`；`ResidualPromptNewTask.forward` 追加第 4 处：`prompt_generator(dnn_input)`（经 `prompt_deltas`）。本消融**只动第 4 处的对应关系**（§2）；前三处保持正确样本输入 | PASS |

### 1.3 seed 与预算选择冻结（**先于任何本实验指标**）

**canonical seed 冻结为 seed2 = `1688723740`；预算冻结为 short（5-epoch / patience 2）。** 理由（全部先于新指标、由 §1.2 证据支撑）：

1. **唯一具备完整既有证据链的 seed**：五个 canonical seed 中，seed2 是唯一同时拥有 (a) 精确 5-epoch 配对（基线 `79b5e07` + 臂 `013e105-rpg`，两 run dirty=false、全产物在盘）、(b) 精确 10-epoch 配对、(c) 20-epoch 终止判定点配对、(d) 钉死参照头（sha `90ee06da…`）、(e) 已复用且内容寻址的 Stage-1 产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f` 的 seed（20-epoch 分支 §1.3 同一论证）。其余 seed 只有 5-epoch 配对。
2. **short（5-epoch）是成本最低的完整协议点**：本消融需要 P0+P1 两条新 run；5-epoch 实测墙钟 ≈143–157 s/run（A1），总计 ≈5–6 min，单次前景窗口可控；10/20-epoch 无必要（预算效应不是本实验变量，且本实验不重跑历史预算点）。
3. **short 上 correct 臂效用最强且已过线**：`G_c = +0.0081 ≥ +0.0055`（`VALID_POSITIVE`）——"被消融的效用"在 short 上量级最大（预算趋势单调收窄），若条件对应因果有效，short 是**最容易**观测到损失的判定点（对 `CONDITION_ALIGNMENT_NOT_SUPPORTED` 是最保守的检验：若连 short 都不损失，更长预算更难支持）。
4. **单因子**：P1 相对 C 的唯一差异 = §2 的条件对应置换；P0 相对 C 的唯一差异 = 无 prompt 头。P0 与 `79b5e07` 的历史记录预期逐位一致（A3 确定性），构成对运行环境与配对链的**独立复核**（§5.1-REP）。

### 1.4 机制审计（消融靶点的精确语义）

机制（`013e105:aliccp_benchmark/residual_prompt.py`，公式与结构化事实逐字沿用 seed-2 复现预注册 §1.2）：

    P(x)        = tanh(W2·ReLU(W1·x + b1) + b2)            # 有界方向，来自条件特征 x
    g_eff(x)    = |alpha| · ||P(x)|| / sqrt(d)             # 逐样本有效门控
    delta_s(x)  = alpha · (||h_s|| / sqrt(d)) · P(x)       # 残差 = 门控 × 单位方向 × 基向量范数
    h_s'        = h_s + delta_s                             # 注入融合前三路（gen + spec_0 + spec_1）

- **条件特征**：`x = dnn_input` `[B,80]`，只经 `self.prompt_generator(x)` 进入机制（`prompt_deltas` 内 `m = tanh(generator(x))`）；`forward` 随后以**正确的** `dnn_input` 原样委托 `super().forward`（`projection_network`/`gate_network` 不受影响）。
- **结构性事实（测试锁定，本消融全部保留）**：S1 范数受控（`ratio_s ≤ |α|`）；S2 逐样本跨流一致（同一 x 下三路相对扰动相同）；S3 α=0 ⇒ 与基线逐位一致。
- **本消融恰好改变**：目标样本 i 的残差所用方向 `P(x_{π(i)})`（π = §2 的确定性错排），即"用哪个样本的特征去生成作用在样本 i 上的方向"。**不改变**：α 语义、范数标定、注入流集合、`super().forward` 委托、损失、优化器、训练协议。

### 1.5 预注册前参照核验（只读；先于本文件提交；`verify_shuffled_prerun.py`）

对固定产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f` + 参照头 + 两个对照 run 做只读复算（**59/59 通过**，报告 `artifacts/aliccp_bench/audit/rp-shuffled/verify_shuffled_prerun_result.json`；不写任何 run/stage1 产物、不训练、不重新评测）：

| 组 | 核对 | 结果 |
|---|---|---|
| 1 产物文件字节 | `backbone.pt`/`env_ids.pt`/`meta.json` sha256 == 钉死值（`cd7b0334…` / `4660be5a…` / `61a66d81…`） | PASS |
| 2 内容寻址 | `stage1_id` 重算 == 记录 == 期望；`config_hash` 重算 == `4e1b5c6f…`；id 分量（fp/config 前缀、seed=1688723740、epochs=3、env_seed=20261003、budgets） | PASS |
| 3 前缀指纹 A2 级 | 自哈希 == `5c060b9c…`；前缀字节 sha256 ×3 + 表头 + 文件大小（2473647855/274769757/2711167840）+ 原始扫描标签计数 ×3 | PASS |
| 4 env_ids | 张量 sha 重算 == `5cd198f1…` == meta；形状 (2000000,)∈{0,1}；计数 1532/1998468 | PASS |
| 5 backbone | 张量 sha 重算（排除 buffer `env_indices`）== `e5e7e610…` == meta | PASS |
| 6 参照头 | 文件 sha256 == `90ee06da…`；`NewTask(80/64/[32,32])` strict 载入成功 | PASS |
| 7 对照 run 链 | 79b5e07（4 文件）与 013e105-rpg（6 文件）逐文件 sha256 == 钉死；两 run 记录值（test/val/逐 epoch val/gate_mean/best_epoch/epochs/patience/seed）逐位 == 钉死；correct 臂 `variant=="residual-prompt"`、`rp_arm.classification=="VALID_POSITIVE"`、`α_final==0.07101669907569885`；`Δtest/Δval` 重推逐位 == 钉死 | PASS |

- **产物复用决议**：按"先验证、后复用，否则确定性重生成"规程——59/59 通过 ⇒ **复用**该产物（不重训 Stage-1）。产物、参照头、两个对照 run、前缀指纹均自 five-seed worktree **只读复制**入本 worktree（逐文件 sha256 与源相等，复制后复验）。
- 钉死参照常量（后续判定全部引用）：`BASELINE_AUC_TEST = 0.5974422649550507`、`BASELINE_AUC_VAL = 0.5809347091990792`、`REFERENCE_PRED_STD = 0.005217193225189258`（与机制文件 `013e105` 内常量逐位一致）；`CORRECT_AUC_TEST = 0.6055453782825184`、`CORRECT_AUC_VAL = 0.5895572066556973`。

---

## 2. 消融设计（唯一变化 = 条件对应置换）

### 2.1 构造（deterministic label-blind derangement；先于实现写死）

- **对象**：每个 batch（batch 内行序 = 数据文件序，`shuffle=False`、batch_size=2000 协议默认）构造一个**错排** π：目标样本 i 的 prompt 条件特征取 `dnn_input[π(i)]`。
- **键**：`(split ∈ {"train","val","test"}, batch_index ≥ 0, n = batch size)`。同一键在任何时刻、任何 epoch、任何评估中产生**同一个** π（缓存不重抽）——即全训练使用**一张固定的错配映射**（每 split 各自一张），使"错配"成为单一、良定义的反事实，而非随 epoch 变化的平均化扰动。
- **种子推导（逐字写死，供独立复核重实现）**：`seed_material = sha256(f"{COND_PERM_SEED}|{split}|{batch_index}|{n}".encode("utf-8")).hexdigest()[:16]`；`rng = random.Random(int(seed_material, 16))`（Python `random` 模块的**局部实例**，不触碰全局 `random` 状态，不触碰 torch 全局 RNG）。`COND_PERM_SEED = 20261005`（冻结常量）。
- **算法（Sattolo，无不动点）**：`a = list(range(n))`；`for i in range(n-1, 0, -1): j = rng.randrange(0, i); a[i], a[j] = a[j], a[i]`。数学性质：n ≥ 2 时输出为**均匀随机的 n-轮换**（单循环）⇒ **不动点恒为 0**、每个下标作为条件源恰被使用一次（双射）。`n ≤ 1`：π = 恒等（退化情形，**定义为恒等并如实记录**；协议 batch=2000 ⇒ 预期 0 次出现；全数据 2M/500k/1M 均为 2000 整除 ⇒ 无尾批）。
- **标签无关（label-blind）**：π 仅由键与 `COND_PERM_SEED` 决定；生成函数不接收任何标签、特征、预测或梯度。**边际保持**：每 batch 内"条件特征的多重集 == 目标特征的多重集"（双射）；跨 split 全量：条件行总覆盖数 == 目标行总数 == split 大小。
- **RNG 隔离**：π 生成只用上述局部 `random.Random`；**不消耗/不改变**全局 torch RNG 与全局 Python RNG（运行期探针断言逐位不变）。因此训练 dropout、头初始化、数据顺序等随机流与 P0/C **逐位一致**。

### 2.2 应用面（forward 语义；只动条件对应）

- 新类 `ShuffledConditionResidualPromptNewTask(ResidualPromptNewTask)`：**只重写 `forward`**，签名为 `forward(dnn_input, gen_rep, spec_reps, env_embs, cond_perm)`；`cond_perm` 为**必需参数**（哨兵缺省 + 显式抛错：缺省调用 ⇒ `ValueError`，**拒绝静默回退为正确条件**）。
- 语义：`cond = dnn_input[cond_perm]`；`deltas, _ = self.prompt_deltas(cond, [gen_rep, *spec_reps])`（`prompt_deltas` 继承钉死实现，零改动）；注入后以**正确的** `dnn_input` 委托 `super().forward`。基头（`projection_network`/`gate_network`）与标签、目标、`h_s`、优化器更新**全部保持正确对应**。
- 构造：`super().__init__()` 链与 correct 头**逐位同路径**（共享参数/RNG 端点/新增键集不变；错排器为普通属性对象，**不是** `nn.Module`、不进入 `state_dict`、不新增参数——`new_params_total=2385`/`head_params=8129` 预期不变）。
- **同一预声明原则应用于 train/val/test**：三 split 各用自己的键空间（split 字符串进键），训练循环、val 评估、test 评估、prompt 诊断遍历一律经同一 `ConditioningShuffler` 取 π；`reference_head_stats`（纯基线参照头、零 prompt）不涉及条件、不需置换。
- **correct-condition 默认逐位不变（PG6 的对象）**：(a) `aliccp_benchmark/residual_prompt.py` 相对 `013e105` 钉死 blob **逐字节一致**（本分支移植零文本适配）；(b) bench 接线在 `shuffler=None`（baseline/correct 臂）时调用签名与行为**逐字节等价**（重建等式守卫）；(c) 等价性测试：恒等置换下 shuffled 头输出与 correct 头输出在 α≠0 时**逐位相等**（同权重、同条件 ⇒ 置换机制引入零漂移）。

### 2.3 明确不改变（禁止清单）

标签/目标/损失/优化器与更新规则、数据顺序与 batch 组成、`h_s` 基表征、`super().forward` 委托、α 初值与语义、范数标定 `scale`、注入流集合 `("gen","spec_0","spec_1")`、hidden=16、生成器初始化（隔离 RNG）、活性带、`BOUND_TOL`、G7 比值、U1/U2 口径、协议文件与 `multitaskrec/*`。

---

## 3. 实现面（最小；TDD；等价证明）与适配清单

| 编号 | 文件 | 处置 | 守卫 |
|---|---|---|---|
| A1 | `aliccp_benchmark/residual_prompt.py` | **逐字节移植** `013e105:aliccp_benchmark/residual_prompt.py`（blob `674213f619c5d5039811c71242a7727118348daf`；LF sha256 `b3b93b3a…`）。**零文本适配** | 守卫测试：工作树 == 钉死 blob 逐字节 + LF sha + `git hash-object` == blob + 继承的 Census 祖本 AST 钉死（`TestMechanismPin` 原样移植） |
| A2a | `aliccp_benchmark/bench.py` | `013e105:` 版（blob `bf36476…`；LF `b063a366…`）**+ 文档化最小适配**：import `rp_shuffled`；`variant=="residual-prompt-shuffled"` 时 ① 经 `RPS.build_shuffled_newtask` 构造、② 建 `ConditioningShuffler`、③ 用 `ShuffledPromptAudit`、④ 训练/评估前向经 `RPS.forward_with_conditioning` 分派（`shuffler=None` 时调用**逐字**为原调用）、⑤ 评估/诊断/臂级判定分派到 `RPS.*` 包装、⑥ run_id 追加 `-rpgs` 后缀、⑦ `prompt_report.json` 增 `shuffle` 段。baseline/correct 臂路径的所有改动均为"经 `shuffler=None` 的恒等分派" | 守卫测试：重建等式（钉死文本 + 恰好替换 == 工作树）＋ `shuffler=None` 分派恒等测试（tiny 端到端输出逐位不变）＋ AST 差异集合恰为文档化集合 |
| A2b | `run_aliccp_benchmark.py` | `013e105:` 版（blob `229c25a1…`；LF `2966ec39…`）**+ 恰好 1 行适配**：`--variant` choices 增加 `"residual-prompt-shuffled"`（后缀分派读 `RPS.RUN_ID_SUFFIX`） | 守卫测试：单行替换重建等式（替换前断言旧行恰出现 1 次）+ 新 LF sha 钉死 |
| A3 | `aliccp_benchmark/tests/test_residual_prompt.py` | `013e105:` 同名测试（blob `768a477b…`；LF `dab737a1…`）**恰 4 处**适配：(a) 模块 docstring；(b) `DOC_PATH` → 本文件；(c) `WHITELIST` → 本分支 10 项（逐项集合钉死）；(d) `TestPreregConstants.test_prereg_doc_tokens_and_summary_ledger` 的 token 表 → 本文件 token 表（`test_thresholds_and_baseline_reference` 免改：本实验机制常量与 013e105 相同）。**其余逐类/逐方法/逐模块级语句不变** | 守卫测试：重建等式（docstring + `DOC_PATH` + `WHITELIST` + 该 1 个方法段替换）+ AST 差异集合 == 恰 1 个方法 + 模块级函数逐字 + 新 LF sha 钉死 |
| A4 | `aliccp_benchmark/rp_shuffled.py`（新） | 消融本体：`ConditioningShuffler`（§2.1 构造 + 运行期完整性记录：双射/不动点/重生成一致/RNG 隔离探针 + 逐 split 与总 digest）、`ShuffledConditionResidualPromptNewTask`、`ShuffledPromptAudit`（继承 `PromptAudit`，仅重写 `init_forward_check` 以传入真实 π；G1 构造审计逐字继承）、`forward_with_conditioning` 分派、shuffled 版评估/诊断包装（复用钉死的 `PromptStats`/`effective_gate`）、`shuffled_arm_verdict`（钉死 `RP.arm_verdict` + `shuffle` 段：PG1–PG5 运行期门禁 + PG6 构建期证据钉死）、冻结常量（`COND_PERM_SEED`、`MATERIAL_DELTA`、二级分类边界、判定树标签）、`analyze_runs`（只读 P0/P1/对照 run 记录 → §5 全量判定）+ `python -m` CLI（输出 JSON） | 守卫测试（本分支 `test_residual_prompt_shuffled.py`）：§4 全部不变量 + 判定树真值表/边界 + 构造/双射/不动点/确定性/隔离/标签无关 |
| A5 | `aliccp_benchmark/tests/test_residual_prompt_shuffled.py`（新） | 本分支守卫 + 消融/接线/分析器测试（§4） | 本文件即为守卫 |
| A6 | `verify_shuffled_prerun.py`（预注册前完整性核验；§1.5 已运行 59/59）与 `verify_rp_shuffled.py`（运行后独立复核） | 只读脚本；**跟踪入库**（沿用 20-epoch 先例；两处 WHITELIST 副本同步包含二者）；报告 JSON 落 `artifacts/aliccp_bench/audit/rp-shuffled/` 与 P1 run 目录（gitignore）。前置脚本自 `10e86dc:verify_seed2_artifact_integrity.py`（blob 见 §1.5 报告）移植（docstring/输出目录/新增对照链核验；钉死值 §1.5 不变） | 前置核验 59/59（§1.5）；复核脚本 §10.5 |

**被禁止的适配**：机制公式/初始化/门控语义/注入点/活性带/`BOUND_TOL`/G7 比值/U1 阈值/`arm_verdict` 判据语义/判定树/阈值；`protocol.py`/`metrics.py`/`multitaskrec/*`/`config.py`——零改动（静态守卫：`git diff --name-only 8133d32` ⊆ 白名单）。

**等价性证明口径（三层，全部由测试守卫执行）**：① 重建等式（文本级）：钉死 blob + 恰好替换 == 工作树（bench/CLI/移植测试）；② 机制文件逐字节 == 钉死 blob + Census 祖本 AST 钉死（原样继承）；③ 行为级：`shuffler=None` 分派恒等 + 恒等置换等价（α≠0 逐位）+ α=0 初值恒等（G2 运行期）。

---

## 4. 不变量（由 `test_residual_prompt_shuffled.py` 与移植测试强制，不接受人工目测）

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 基线/正确臂逐位不变：`shuffler=None` 分派 ≡ 钉死调用；correct 臂端到端记录键集与钉死一致（除既有 `variant`/`rp_arm`/`prompt_hidden`） | 移植测试（构造恒等/初值恒等/tiny 端到端 purity）+ bench 重建等式 |
| I2 | 适配逐字守卫：A1/A2a/A2b/A3 的字节/重建等式钉死 | `git show` + sha256(LF) + 段级重建等式 |
| I3 | 机制语义不变：α 初值 0、恒等/界/活性带/参数预算测试全绿（移植测试原样通过） | 移植测试（TestConstructionIdentity/TestNormControl/TestArmVerdict/TestParamReport…） |
| I4 | 错排构造：双射（`sorted(π)==range(n)`）∧ n≥2 不动点==0 ∧ n≤1 ⇒ 恒等 ∧ 同键同 π（跨实例逐位）∧ 不同键（抽样）不同 π ∧ `COND_PERM_SEED` 冻结 | 构造单元测试（n ∈ {0,1,2,3,7,100,2000}，多键） |
| I5 | RNG 隔离：π 生成前后 `torch.get_rng_state()` 与 `random.getstate()` 逐位不变；连续生成不改变后续正确路径输出 | 探针测试 + tiny 端到端 |
| I6 | 标签无关：π 生成函数只接收键（构造签名）∧ 同键在任何数据/标签下产生同 π（以不同 tensor 场景复算） | 签名断言 + 键复算测试 |
| I7 | forward 语义：缺 `cond_perm` ⇒ 抛错（哨兵）；恒等置换 ⇒ 与 correct 头逐位相等（α≠0）；非恒等置换 + α≠0 ⇒ 输出改变；条件对应正确（手算：`delta_i` 用 `x[π(i)]` 生成——小规模显式计算）；范数界在置换下保持（ratio ≤ |α|） | 行为测试 |
| I8 | 判定树/边界：§5.5 全部标签真值表；`L` 恰 `+0.001` ⇒ `SUPPORTED`（闭），`−1e-12` ⇒ `NOT_SUPPORTED`；`G_s` 恰 `+0.001` ⇒ `PARTIAL`，`−1e-12` ⇒ `FULL`；二级分类边界 `+0.001`/`−0.02`（闭端）；每条 INVALID 子原因可达；REP 失败 ⇒ INVALID | 边界/真值表测试（CPU 夹具） |
| I9 | 分析器只读：不改任何 run 产物；identity/对照 mismatch ⇒ `INVALID`；输出键齐全（§5 全部判定 + 全部分量 + 描述量） | 夹具端到端 + 篡改 fixture |
| I10 | 静态守卫：相对 `8133d32` 的全部跟踪改动 ⊆ 白名单；`protocol.py`/`metrics.py`/`multitaskrec/*`/`config.py` 零 diff | `git diff --name-only` |
| I11 | 预注册文档 token 钉死：seed `1688723740`、stage1_id、COND_PERM_SEED `20261005`、阈值 `0.0055`/`0.001`/`−0.02`、判定标签四项、适配清单 blob、对照 run_id/常量 | 文档文本断言（doc-token 方法） |

CPU 极小夹具不构成任何性能证据，只验证语义与接线。

---

## 5. 预注册判据（看到结果前写死；看到结果后不得修改）

对照定义：**P0 = 本实验新跑的配对基线臂**；P1 = shuffled 臂；C = 已提交 correct 臂（只读对照）。`G_c := test(C) − test(P0)`；`G_s := test(P1) − test(P0)`；`L := G_c − G_s`；`R := G_s / G_c`；`Δval_s := val(P1) − val(P0)`；`Δval_c := val(C) − val(P0)`。

### 5.1 前置条件（全部必须为真；任一假 ⇒ `INVALID`，不解读效用）

| 编号 | 判据 | 落盘 |
|---|---|---|
| **ID**（identity，14 项） | P0/P1 两 run：`stage1_id` 相同且 == 钉死值；`model_seed` 相同且 == 1688723740；`epochs==5` ∧ `patience==2` ∧ `tag=="short"`；`dirty==false`（两 run 记录时）；P0 `variant=="baseline"` 且 run_id 无 `-rpgs` 后缀；P1 `variant=="residual-prompt-shuffled"` 且 run_id 以 `-rpgs` 结尾；P1 记录参照路径 == 钉死路径且文件 sha256 == `90ee06da…`；两 run 记录 `stage1_id` 的 backbone/env/fingerprint sha 与钉死值一致 | `rp_shuffled_compare.json:identity` |
| **REP**（基线复现） | P0 的逐 epoch val 轨迹、best_epoch、best_val、test、gate_mean 与 `79b5e07` 记录**逐位相等** 且 P0 的 `newtask.pt` sha256 == `90ee06da…`（A3 确定性预期成立；失败 ⇒ 环境偏离历史链，对照组不可比） | 同上 `reproduction` |
| **A**（协议） | P0 与 P1 的 A1/A2/A4/A5/A6 全 PASS（A3 SKIP 视为通过）；任一 FAIL ⇒ 该 run 比较作废 | 两 run `gate_report.json` |
| **M**（机制） | P1 的 `rp_arm`：M0 + G1–G8 全 PASS（判据语义 == seed-2 复现预注册 §4，**不放宽**） | P1 `metrics.json:rp_arm` |
| **CC**（对照链） | C 与 `79b5e07` 的记录值 == §1.5 钉死常量（test/val/Δ/α/分类）；对照 run 文件 sha == 钉死（复核脚本执行） | `rp_shuffled_compare.json:comparator` |

### 5.2 置换门禁（PG1–PG6；运行期 PG1–PG5 由 P1 记录 → 复核脚本独立重推）

| 编号 | 判据 | 口径 |
|---|---|---|
| **PG1 无不动点** | 运行期生成的每个 π（n≥2）不动点数 == 0；n≤1 的恒等（若有）逐项记录 | 运行期逐 π 检查 + 记录计数（预期 0 次退化） |
| **PG2 双射/边际保持** | 每个 π 为 `range(n)` 的置换（多重集相等）；逐 split：条件行总覆盖 == 目标行总数 == split 大小 | 运行期 + 独立重推 |
| **PG3 确定性/可复现** | 每个 π 以**全新** `random.Random` 重生成逐位相等（运行期探针）；**独立复核脚本以 §2.1 规格重实现并对逐 split digest 与总 digest 逐位复算** | 运行期 + verifier 重推 |
| **PG4 RNG 隔离** | 每次生成前后 `torch.get_rng_state()` 与 `random.getstate()` 不变 | 运行期探针 |
| **PG5 标签无关/键唯一** | π 仅由键决定：独立复核以（`COND_PERM_SEED`, split, batch_index, n）重推全键空间 digest == 记录 digest；同键不同 n ⇒ 拒绝（`ValueError`） | verifier 重推 + 单元测试 |
| **PG6 correct 默认逐位不变** | §2.2(a)(b)(c)：机制文件 LF sha == 钉死；bench 重建等式；恒等置换等价测试全绿 | 构建期（测试 + verifier 独立重哈希机制文件） |

### 5.3 机制门禁（M0+G1–G8，pinned）

逐字沿用 seed-2 复现预注册 §4（本文件不重写其数值口径）；P1 必须全 PASS。G7 的参照为 `REFERENCE_PRED_STD = 0.005217193225189258`（M0 同款身份检查）。**已知预期**：G3 首步生成器梯度为 0（α=0 的数学后果，探针口径）；G4/G5 在置换下结构性保持（S1/S2 与条件无关）。

### 5.4 效用分量（判定的原子量；全部报告）

| 量 | 定义 | 预声明真值门限 |
|---|---|---|
| `G_c` | `test(C) − test(P0)` | 实质为正：`G_c ≥ +0.001`（**MATERIAL_DELTA**） |
| `G_s` | `test(P1) − test(P0)` | — |
| `L` | `G_c − G_s` | 实质损失：`L ≥ +0.001` |
| `R` | `G_s / G_c`（`G_c ≤ 0` ⇒ NA） | 报告 |
| `shuffled_above_baseline` | `G_s > 0` | 报告（符号口径） |
| `shuffled_material_positive` | `G_s ≥ +0.001` | 报告（实质口径） |
| `validation_agreement` | `Δval_s > 0`（U2 同款方向口径） | 报告 + §5.6 headroom 因子（**不进入 §5.5 判定树合取**：因果对比由 test 侧 Δ 差值定义，val 为方向佐证，预声明其不改变分类） |
| `val_sign_matches_correct` | `sgn(Δval_s) == sgn(Δval_c)` | 报告 |

**阈值来源与读法（预声明）**：`+0.001` = 用户指定的最小实质差异，与既有分支二级分类边界一致；test 1M 前缀 BSI AUC 的 SE ≈ 0.0025 @AUC0.7（沿用 20-epoch §5.1 口径）⇒ `+0.001` ≈ 0.4×SE，为**分类标签**而非显著性强断言；单 seed、每臂 1 run、无 run-to-run 噪声估计（A3 SKIP）。

### 5.5 因果判定树（**先于结果写死**；优先级自上而下，标签集固定为以下四种）

| 优先级 | 判定 | 条件 | 含义（写死） |
|---|---|---|---|
| 1 | **`INVALID`** | 任一前置条件假（ID / REP / A / M / CC / PG1–PG6 任一 FAIL） | 不解读效用；`subreason` 按首个失败项机械给出（`IDENTITY_MISMATCH` / `BASELINE_REPRODUCTION_FAILED` / `PROTOCOL_INVALID` / `MECHANISM_FAIL` / `COMPARATOR_MISMATCH` / `PERMUTATION_INVALID` / `COMPARATOR_GAIN_NOT_MATERIAL`） |
| 2 | **`SAMPLE_CONDITION_SUPPORTED`** | `G_c ≥ +0.001` ∧ `L ≥ +0.001`（闭） | 条件对应承载实质效用：错配使 correct 臂的实质增量**损失 ≥ +0.001**。子类（**报告、不改变本判定**）：`subreason = FULL_GAIN_REQUIRES_ALIGNMENT` 若 `G_s < +0.001`（错配下增量不再实质为正，含 `G_s ≤ 0` 情形）；`subreason = PARTIAL_ALIGNMENT_CONTRIBUTION` 若 `G_s ≥ +0.001`（错配下仍有实质增量，对齐只解释一部分） |
| 3 | **`CONDITION_ALIGNMENT_NOT_SUPPORTED`** | `L < +0.001`（即 shuffled 保留 correct 臂实质增量的全部/几乎全部） | 效用**不依赖**样本级条件对齐：与"额外参数 / 全局扰动 / 范数-门控效应"等 nuisance 解释一致。`subreason = SHUFFLED_RETAINS_GAIN`；`G_s > G_c` 时附注 `shuffled_exceeds_correct=true`（报告） |
| 4 | **`INVALID`（对照分支）** | `G_c < +0.001`（REP 通过时由 §1.5 钉死值排除，预期不可达） | `subreason = COMPARATOR_GAIN_NOT_MATERIAL`；出现即如实记录并停止解读 |

### 5.6 二级效用分类（用户指定；对 P1 的 `G_s` 机械分类；与 §5.5 并存、不覆盖）

| 类别 | 判据（精确边界） |
|---|---|
| `POSITIVE_IMPROVEMENT` | `G_s ≥ +0.001` |
| `NO_CLEAR_IMPROVEMENT` | `−0.02 < G_s < +0.001`（严格双开） |
| `CLEAR_DEGRADATION` | `G_s ≤ −0.02` |

**`NO_CLEAR_IMPROVEMENT` 的机械 headroom 评估（仅当落入该区间；规则先于结果写死；结构沿用 20-epoch §5.4，因子按本实验适配）**：

| 因子 | 机械口径 | 取值 |
|---|---|---|
| 机制活性 | `|α_final| > 0` ∧ P1 末 epoch 首 batch `generator_grad_norm > 0` ∧ `geff_std > 0`（取自 P1 JSON） | `ACTIVE` / `INACTIVE` |
| 验证方向 | `Δval_s > 0` | `POSITIVE` / `NON_POSITIVE` |
| epoch 轨迹 | P1 `best_epoch == epochs_run == 记录 epochs`（未早停）⇒ 右删失；否则早停 | `RIGHT_CENSORED_STILL_IMPROVING` / `EARLY_STOPPED` |
| 距正分类余量 | `gap_to_positive_threshold = +0.001 − G_s`（>0） | 数值 |
| 保留比 | `R`（描述性；与 §5.5 子类互参） | 数值 |

### 5.7 历史 `+0.0055` 判定单列保留（不适用、不改写）

- 历史成效门槛 `+0.0055` 为 correct 臂效用口径（seed-2 复现预注册 §3-U1）；**P1（shuffled 臂）不适用该门槛**，其真值 `G_s ≥ +0.0055` 作为**报告项单列**（不参与 §5.5/§5.6 判定）。
- 历史判定原样引用：seed1 `VALID_NEGATIVE`（+0.005394543882337954，差 1.05e-4）、seed2 `VALID_POSITIVE`（+0.008103113327467715）、10-epoch `PERSISTS`、20-epoch `NOT_PERSIST`/claim=true/`ABLATION_ELIGIBLE`。本实验不重判任何历史结论。

---

## 6. 运行规程（恰好两次；清洁树纪律；前台）

**运行次数**：P0 stage2 **恰好 1 次**；P1 stage2 **恰好 1 次**；分析（`analyze_runs`，只读）**恰好 1 次**；运行后独立复核 **恰好 1 次**。不重训 Stage-1；任何历史 run 不重跑；correct 臂不重跑（§5 对照仅取已提交记录）。

**清洁树纪律（run 间提交，沿用 20-epoch §4）**：① 提交实现与测试（C2）→ `git.dirty=false`；② 跑 **P0**（run 记录 commit=C2）；③ 提交 P0 的 SUMMARY 追加行（C3，仅 `artifacts/aliccp_bench/SUMMARY.md` 变化）；④ 跑 **P1**（run 记录 commit=C3）；⑤ 核验 `git diff C3 C2 -- . ':(exclude)artifacts/aliccp_bench/SUMMARY.md'` 为空；⑥ 分析 ×1 → 复核 ×1 → C4（§10 回填 + P1 SUMMARY 行）→ push。

```powershell
# cwd = 本 worktree 根；本 worktree 无 .venv，用主树解释器；dataset 经 junction 复用（只读）

# 0) 测试（先红后绿；全部不变量）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests

# 1) P0 配对基线臂（唯一一次；协议默认：epochs 5 / patience 2 / tag short）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag short --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f

# 2) P1 shuffled 臂（唯一一次；同产物、同 seed、同预算；残差 prompt + 条件错排 + 钉死参照头）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag short --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f `
    --variant residual-prompt-shuffled `
    --prompt-reference-newtask artifacts/aliccp_bench/runs/20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt

# 3) 分析（纯分析，恰一次）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m aliccp_benchmark.rp_shuffled `
    --baseline-run <P0 run_id> --arm-run <P1 run_id>

# 4) 运行后独立复核（只读；JSON 重推 + 文件哈希 + 错排独立重实现）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe verify_rp_shuffled.py `
    --baseline-run <P0 run_id> --arm-run <P1 run_id>
```

- **前台执行并等待至完成**（先例：后台执行被会话终止杀死）；开跑前提交实现（§6 步骤①）；每臂 run 启动时 `git.dirty` 必须为 false。
- **无效执行处置**：仅当 run 因**工具性原因**（崩溃/中断/环境故障、产物未完整落盘）无效时，保留现场、记录原因后可重跑一次；因**结果原因**不构成无效，不重跑。无任何事后调参（COND_PERM_SEED/构造/阈值/参照/判定树均不得动）。
- **墙钟预算**：每臂 ≈2–3 min（short 实测 143–157 s + 错排开销），总计可控。

---

## 7. 必录诊断（落盘 run 目录；class C，不判定）

| 量 | 字段 | 用途 |
|---|---|---|
| 错排完整性 | `shuffle`：逐 split 键数/行数/digest、总 digest、不动点计数（必须 0）、双射检查、重生成一致计数、RNG 隔离探针计数、退化（n≤1）计数 | PG1–PG5 |
| 错排样例 | `shuffle.samples`：每 split 前 3 个键的 π 前 16 元素 + 首/末 5 行错配对（i→π(i)） | 人审 + 复现 |
| α / 梯度轨迹 | `grad_probe`（逐 epoch 首 batch α、`alpha_grad_norm`、`generator_grad_norm`）、`alpha_final` | G3 + 与 correct 臂（+0.07101669907569885）对照 |
| 残差/基范数比 | 逐流 `ratio_mean/std/max`、`delta_norm_mean`、`h_norm_mean` | G4/G5 + 对照（correct 臂 ratio_mean 0.02813939；h_norm 168.97/23.98/22.45） |
| 余弦方向 | 逐流 `cos_mean` | 对照（correct 臂 +0.01287/+0.08432/+0.05899） |
| 逐样本门控 | `gate.geff_mean/std/min/max` | G6 + 对照 |
| 预测离散度 | `dispersion` + `reference_dispersion`（M0 口径） | G7 + 对照（correct 臂 pred_std 0.0049475979 / 参照 0.0052171933） |
| 源/头门 | `gate_mean` + Stage-1 环境上下文（`cluster_events`、`env_acc`） | 落盘；B3 原样披露 |
| 参数清单 | `params`（`new_params_total=2385`、`head_params=8129`） | G8（预期与 correct 臂相同） |
| 臂级判定 | `arm`（M0+G1–G8+U1/U2+classification+subreason）+ `shuffle` 段（PG1–PG5 + PG6 构建期证据） | §5.2/§5.3 机械分类 |
| 配对判定 | `rp_shuffled_compare.json`：§5 全部前置条件、分量、判定树、二级分类、headroom（若适用）、context 表 | §5.5/§5.6 |
| 停止/删失 | 两 run 的 `best_epoch`/epochs、是否早停、逐 epoch 轨迹 | 描述 |

---

## 8. 局限（预声明）

1. **单 seed、每臂 1 run**：无 run-to-run 噪声估计（A3 SKIP）；`+0.001` 为分类标签而非显著性强断言（≈0.4×SE）。
2. **单一错排抽样**：本实验检验的是**一张**冻结错配映射（由 `COND_PERM_SEED=20261005` 决定）的效应；不回答"错排分布上的平均效应"。若结果落在 `SAMPLE_CONDITION_SUPPORTED`，稳健性需后续独立复现（换 perm seed 的第二张错排，§9.3 预声明建议）。
3. **batch 内错排（窗口局部）**：条件源取自同 batch（同 2000 行窗口）；对齐度高于"全库随机错配"。这是冻结构造；全局错配为另一实验（§9.3）。
4. **判定依赖单一汇总统计**（test 1M 前缀 BSI AUC 差值）；无独立重评测 pass（判定取 run 记录值；正确性依赖代码路径守卫 + 同类路径既有重评测先例）。
5. **in-run `rp_arm.U` 的配对对象是历史 5-epoch 常量**（class C 诊断）；因果判定只用 §5 的 P0 配对。
6. **B4 若 FAIL 为继承缺陷**（Stage-1 共享语义）；`hard_pass` 不参与有效性定义（同先例）。
7. **单数据集（AliCCP）、单 seed**：结论不推及 CensusIncome / ByteRec、不推及其它 seed / 预算点。
8. **本实验不是"更优方法"主张**；是机制因果消融的一次性判定。

---

## 9. 非目标与止损

1. 不做：其它消融（无条件的固定向量 prompt、跨 split 条件、label 置换——**明确排除**：本实验不动标签）、不写论文/图表、不做聚类、不做 Census、不跑 `full` tag、不做 FLOPs、不扩 seed、不重跑任何历史 run。
2. 不因结果修改：COND_PERM_SEED、错排构造/算法、MATERIAL_DELTA、二级边界、判定树、§5.2–§5.5 任何规则、参照文件、clean-tree 纪律。
3. **后续独立消融的预声明建议（本分支不启动，仅登记）**：
   - 若 `CONDITION_ALIGNMENT_NOT_SUPPORTED` ⇒ 下一步最优先：**无条件的固定条件对照**（把逐样本 `P(x_i)` 换成与样本无关的常量方向，范数标定与参数预算匹配），定位增量由哪个非对齐通道（参数/全局扰动/范数-门控）承载；
   - 若 `SAMPLE_CONDITION_SUPPORTED` ⇒ 下一步最优先：**第二张独立错排复现**（仅更换 `COND_PERM_SEED`，其余逐字节不动），排除单张错排抽样伪影；
   - 两分支共同的前提：`INVALID` 情形下先修复执行/实现，不启动任何后续消融。

---

## 10. 结果（运行后回填，不回写第 0–9 节）

_（C4 回填）_

---

## 11. 非新颖性声明

构件全部为既有方法族（prompt tuning、FiLM 族条件化调制、adapter/LoRA 残差、门控残差、范数归一、条件置换/错排对照为消融常用手段），先行工作清单沿用 seed1 设计 §11（写作引用前必须逐条核实，本会话未做系统检索）。本文件**不**声称独立于先行工作；可报告内容仅限本仓库冻结协议下的一次性判定。

---

## 12. 偏离披露（预声明 + 运行后补）

- **分支工作流**：自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；机制经 §3 三层守卫钉死等价；协议不合并 `master`；push 经用户显式指示（本任务含 "commit and push"）。
- **分支本地 SUMMARY 系谱**：本分支 `SUMMARY.md` 基线含 smoke 行与 seed1 基线行（`8133d32` 携带）；其余历史行记录在各自分支，本分支**不复制**其它分支行，只由本实验 run 追加自己的行（P0 + P1）。
- **产物/对照 run 只读复制**：§1.5 所列产物、参照头、两个对照 run、前缀指纹均自 five-seed worktree 只读复制入本 worktree（逐文件 sha 相等）；dataset 经 junction 复用 main tree（只读、gitignore）。
- **非结果文件（未跟踪）**：`artifacts/aliccp_bench/audit/rp-shuffled/` 报告 JSON、`artifacts/aliccp_bench/logs/` run 日志。
- **执行记录（运行后补）**：_（C4 回填）_。

---

**改动清单（白名单；静态守卫以 `8133d32` 为基准）**

| 文件 | 处置 |
|---|---|
| `docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-shuffled-condition-design.md` | 本文件（C1 单独提交） |
| `aliccp_benchmark/residual_prompt.py` | 新增（A1：逐字节 == `013e105` 钉死 blob） |
| `aliccp_benchmark/rp_shuffled.py` | 新增（A4：消融本体 + 判定 + 分析 CLI） |
| `aliccp_benchmark/bench.py` | A2a：== `013e105` + 文档化最小适配（shuffled 分派；shuffler=None 恒等） |
| `run_aliccp_benchmark.py` | A2b：== `013e105` 恰 1 行适配（variant choices） |
| `aliccp_benchmark/tests/test_residual_prompt.py` | 新增（A3：== `013e105` 恰 4 处适配） |
| `aliccp_benchmark/tests/test_residual_prompt_shuffled.py` | 新增（A5：本分支守卫 + 消融/接线/分析器测试） |
| `verify_shuffled_prerun.py` | 新增（A6：预注册前完整性核验；§1.5 已运行 59/59） |
| `verify_rp_shuffled.py` | 新增（A6：运行后独立复核） |
| `artifacts/aliccp_bench/SUMMARY.md` | 只追加 2 行（runner 自动） |
| 其余全部（`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`metrics.py`、既有测试三件、`AliCCP_*.py`、`CensusIncome_*.py`、`baseline/*`、`mask/*`、`analysis/*`） | **零改动**（静态守卫断言 `git diff --name-only 8133d32` ⊆ 白名单） |
