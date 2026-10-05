# AliCCP 阶段 2 残差 Prompt 的**无条件固定条件对照**消融（unconditional fixed-condition control；canonical seed-2、5-epoch short、三臂单因子）

- **状态**：预注册已写死（本文件在本分支任何实现、任何 run **之前单独提交**；commit C1）。结果只在第 10 节以新增小节回填；第 0–9 节的判据、数字、构造与运行规程不得在看到结果后改动。
- **日期**：2026-10-05
- **适用分支**：`exp/aliccp-stage2-residual-prompt-unconditional-control`（自 `infra/aliccp-fair-benchmark` @ `8133d32` **独立拉出**；不从任何 exp 分支拉出、不继承其运行时状态；不合并 `master`、不触碰其它 worktree；`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`aliccp_benchmark/metrics.py`、既有测试三件、master 脚本与 `baseline/*` 零修改）。
- **立项依据**：`exp/aliccp-stage2-residual-prompt-alpha-pinned-condition` @ `f08ae6e`（`f08ae6e459ae9fe048daf11eda43799d19ccc7d1`）的结果——因果判定 `CONDITION_ALIGNMENT_NOT_SUPPORTED/GAP_BELOW_MATERIALITY`（`GAP = test(F_c) − test(F_s) = −0.001861278011441092`，未达 `+0.001` 实质门且方向为负；两处理臂二级分类均 `NO_CLEAR_IMPROVEMENT`；PA1–PA8 + PG1–PG7 全 PASS；独立复核 101/101）——及其 §9.3 / §10.7.1 **预声明的下一步消融建议**：**无条件固定条件对照**（`CONDITION_ALIGNMENT_NOT_SUPPORTED` 分支 ⇒ 下一步最优先：把逐样本 `P(x_i)` 换成与样本无关的常量方向，**α 与参数预算匹配**，定位增量由哪个非对齐通道——额外参数 / 全局扰动 / 范数-门控效应——承载）。
- **被消融对象（只读引用；系谱钉死）**：残差 prompt 机制 = `exp/aliccp-stage2-residual-prompt-seed-replication` @ `013e105`（`013e105f6dfb35e630d261696290635d0263ce99`）`:aliccp_benchmark/residual_prompt.py`（blob `674213f619c5d5039811c71242a7727118348daf`，LF sha256 `b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc`）；钉死 α 实现（本实验 C_p 臂 = 重放对象）= `f08ae6e:aliccp_benchmark/rp_pinned.py`（blob `432fa9bb6b3bb9d537753599499b693197e8c898`，LF sha256 `ba23bf473634ff647b482ebf6a92f27ca7f073432bb4ed7a1342b8b11101df9b`）；其系谱：seed1 线（预注册 `221580a` / 实现 `79ddefa`（blob `5dc7158ca999c9e7e6217a999c37869231411549`）/ 结果 `3e2f083`）→ Census 祖本 `99b9510`（blob `107221b26382da7fd44990e167d9a61da2680d92`）。
- **α 钉死值来源（只读引用；系谱钉死）**：correct-condition seed2 学习臂 run `20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg`（commit `013e105`，dirty=false）的 `prompt_report.json:alpha_final = +0.07101669907569885`（该文件 sha256 `185df4d0f9910508aeff84ed780e7df4fe8ad4fd6b8451ff429a74927d333c1e`）。该值 float32 往返**逐位精确**，作为 buffer 常量以精确相等比较（§1.5-8 复核）。
- **上游协议（只引用、不修改）**：
  1. `docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（AliCCP 公平评测协议）——前缀预算、种子、Stage-1 产物、真冻结三件套、A/B/C 门禁、SUMMARY 台账、止损纪律全部沿用；
  2. `docs/superpowers/specs/2026-10-04-aliccp-stage2-residual-prompt-seed-replication-design.md`（seed-2 复现预注册）——机制语义、参照头钉死、U1/U2 口径（历史 `+0.0055`）逐字沿用；
  3. `exp/aliccp-stage2-residual-prompt-20epoch` @ `10e86dc`——审计方法（以磁盘原始 run 产物 + git 对象为准）、二级分类三标签边界、`NO_CLEAR_IMPROVEMENT` 的机械 headroom 结构、`commit-between-runs` 清洁树纪律、独立复核脚本形态（纯 JSON 重推）**逐字沿用**；
  4. `exp/aliccp-stage2-residual-prompt-alpha-pinned-condition` @ `f08ae6e`（`docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-alpha-pinned-condition-design.md`，blob `6e033b74275e8e78438720d1aabb888ae4352317`）——α 钉死构造（非参数 buffer、优化器排除）、PA1–PA8 结构门禁形态、判定树/二级分类/headroom 形态、`REP` 前置条件口径、独立复核形态**逐字沿用**；其 `CONDITION_ALIGNMENT_NOT_SUPPORTED` 与 §10.7.1 建议为本实验立项依据；
  5. `exp/aliccp-stage2-residual-prompt-five-seed` @ `0935257`（结果）/ `28aa1fe`（provenance tip）——五 seed canonical 配对与跨进程逐位确定性证据（§1.2-A5）。
- **定位声明**：本实验是 f08ae6e §10.7.1 的**预注册化执行**：新增第三处理臂 **U**——把逐样本条件 `P(x_i)` 换成**与样本无关的常量方向**（学习参数化：同一生成器架构、同一参数预算、喂入**冻结常量条件向量** `c`；§2），α 与 C_p 完全相同的钉死值 `+0.07101669907569885`，范数标定/注入流/协议逐字沿用。C_p 臂 = f08ae6e F_c 的**跨实验逐位重放**（同构造、同 commit 谱系、`REP_Cp` 前置条件），使 `GAP_U = test(C_p) − test(U)` 成为**同分支、同代码同一性**之下的直接配对。学习活性带类门禁（G5 等）**不适用**（a priori，同 f08ae6e §1.4）；U 臂门禁为先于结果写死的结构式 UA1–UA11（§5.2）。无新颖性主张、无改进主张、不调参、不放宽任何门限。
- **提交时序（预注册纪律）**：C1 本文件（审计 + 预注册，单独提交）→ C2 实现 + 测试（TDD；三 run 的 `commit` 字段即 C2 与后续两个 SUMMARY 提交）→ C2 之后跑 **B 臂（配对基线）**（前台）→ C3 提交 B 的 SUMMARY 行 → 跑 **C_p 臂（钉死 α、逐样本条件；f08ae6e F_c 的逐位重放）**（前台）→ C4 提交 C_p 的 SUMMARY 行 → 跑 **U 臂（钉死 α、无条件/全局条件）**（前台）→ C5 提交 U 的 SUMMARY 行 → 分析恰一次 → 独立复核恰一次 → C6 结果回填（§10）+ push。

---

## 0. 判定问题

对照三分：**B = 本实验新跑的配对基线臂**（seed2、short、5-epoch；协议默认）；**C_p = 本实验新跑的钉死 α 逐样本条件臂**（构造与 f08ae6e 的 F_c **逐字相同**，须逐位重放其记录 `7d26918-rpp`）；**U = 本实验新跑的钉死 α 无条件/全局条件臂**（唯一变化 vs C_p = 条件输入 `x_i` → 冻结常量 `c`；§2）。历史学习臂（`013e105-rpg`）与 f08ae6e 两处理臂（`7d26918-rpp` / `1c13841-rpps`）与学习 shuffled 臂（`9d26bc8-rpgs`）仅作 hash 核验后的**上下文对照**，不重跑、不进任何门禁与主判定。

记 `GAP_U := test(C_p) − test(U)`（**因果主量**；样本级条件化 − 无条件/全局；两臂标量幅度由同一钉死 α 匹配、可训练参数预算匹配）；`G_c := test(C_p) − test(B)`；`G_u := test(U) − test(B)`；`Δval_gap_u := val(C_p) − val(U)`；`R_u := G_u / G_c`（增益保留比）。

- **Q1（因果主判定）**：在**标量幅度匹配**（同钉死 α、同范数标定、同注入流）与**参数预算匹配**（同 2384 可训练生成器参数）之下，逐样本条件化臂是否相对无条件/全局臂**实质更优**（`GAP_U ≥ +0.001`）且验证方向一致（`Δval_gap_u > 0`）？→ §5.4 判定树：`SAMPLE_CONDITIONING_SUPPORTED` / `SAMPLE_CONDITIONING_NOT_SUPPORTED` / `INVALID`。
- **Q2（无条件通道与保留比）**：`G_u` 的实测值——"额外参数 / 全局扰动 / 范数-门控效应"通道（即无条件臂相对基线）是否承载可观测效用；`R_u`、`G_c`/`G_u` 相对历史学习臂增量（`+0.008103113327467715`）的保留比——均报告、不进 Q1 分类（`G_u` 参与 §5.5 二级分类与 §9.3 后续消融的机械选择）。
- **Q3（二级效用分类）**：C_p 与 U **各自**相对 B 的 `Δtest` 落入用户三标签（边界先于结果写死，§5.5）；任一落入 `NO_CLEAR_IMPROVEMENT` 时按 §5.5 机械评估 headroom。
- **Q4（完备性）**：前置条件（ID / 基线复现 REP_B / **跨实验重放 REP_Cp** / A 类 / U 臂 UA1–UA11 与 C_p 臂 PA1–PA8 / 对照链 CC / 常量向量 CV）是否全真？任一假 ⇒ `INVALID`（不解读效用）。

**纪律声明（预注册）**：历史判定与全部数值（seed1 `VALID_NEGATIVE`、seed2 `VALID_POSITIVE` +0.0081、10-epoch `PERSISTS`、20-epoch `NOT_PERSIST`/claim/`ABLATION_ELIGIBLE`、5-seed 均值、学习 shuffled `INVALID/MECHANISM_FAIL`、f08ae6e `CONDITION_ALIGNMENT_NOT_SUPPORTED`（`GAP=−0.001861`）、`+0.0055` 历史效用门等）**原样保留、不改写、不重判**。本实验只用 §4/§6 新跑的三条 run 做因果与分类判定；历史 run 仅作 hash 核验后的上下文对照（C_p 除外——C_p 是新跑臂，其与 `7d26918-rpp` 的逐位相等是**前置条件**，不是效用证据）。

---

## 1. 审计（2026-10-05，只读；先于本文件提交）

### 1.1 审计方法与范围

- **原则**：不信任任何文档转述，全部以 **git 对象**（blob / `ls-remote` tip）与**磁盘上的原始 run 产物**（本 worktree 内按 §1.5 逐文件 sha256 核验后只读复制入的副本）为准重新核对；不重跑、不修改任何历史 run。
- 审计对象与 remote tip（`git ls-remote origin` 实测与本地一致）：alpha-pinned `f08ae6e`（`f08ae6e459ae9fe048daf11eda43799d19ccc7d1`）、shuffled `fbfff09`、seed-replication `a40836e`（其预注册 `e931cd7`）、five-seed `28aa1fe`（结果提交 `0935257`）、20epoch `10e86dc`、gate `3e2f083`、longer-budget `e46e5d2`、基点 `8133d32`。
- 审计为只读核对（git + 原始 JSON 重读 + sha256 复算），无可跟踪审计脚本；全部命令可复现。

### 1.2 审计结果（逐项）

| # | 审计项 | 证据（独立重推） | 结果 |
|---|---|---|---|
| A1 | **f08ae6e 结果链复核（立项核心）** | 原始 JSON 重读三条 run：B `20261005-1026-…-bf322df`（test `0.5974422649550507` / val `0.5809347091990792` / 逐 epoch val 轨迹 / gate_mean `[0.789178, 0.2108221875]`）；F_c `20261005-1029-…-7d26918-rpp`（test `0.5947650925865714` / val `0.5798886656441761` / 轨迹 `[0.46759930720014714, 0.49016156687227064, 0.5156547237007145, 0.5480714746135289, 0.5798886656441761]` / gate_mean `[0.775494625, 0.224505265625]` / `rp_arm.M0+PA1–PA8` 全 true / classification `VALID_NEGATIVE` / `alpha_final == 0.07101669907569885`）；F_s `20261005-1032-…-1c13841-rpps`（test `0.5966263705980125` / val `0.5817369266195509` / classification `VALID_NEGATIVE`）；三 run `git.dirty=false`、A 类 PASS（A3 SKIP）、B4 FAIL（继承）、`hard_pass=false`；`rp_pinned_compare.json:verdict=CONDITION_ALIGNMENT_NOT_SUPPORTED`、`GAP=−0.001861278011441092`、`G_c=−0.002677172368479308`、`G_s=−0.000815894357038216` 逐位复算 | PASS |
| A2 | B 臂重放链（本实验 B 的预期） | `bf322df` 记录与历史基线 `79b5e07`（`20261003-0624-…`，commit `79b5e07`，dirty=false）：test/val/逐 epoch val/gate_mean **逐位相等**；两 run `newtask.pt` sha256 均 == `90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f`（参照头钉死值） | PASS |
| A3 | **correct-condition 学习臂记录（α 钉死值来源 + 上下文）** | `20261004-0325-…-013e105-rpg`（commit `013e105`，dirty=false）：test `0.6055453782825184` / val `0.5895572066556973`；`prompt_report.json:alpha_final = 0.07101669907569885`；`rp_arm = VALID_POSITIVE`，M0+G1–G8 全 PASS；`Δtest = +0.008103113327467715`、`Δval = +0.008622497456618139`（逐位重推）；文件 sha256 `prompt_report.json = 185df4d0…` | PASS |
| A4 | **f08ae6e F_c 记录（C_p 的跨实验重放靶）** | `7d26918-rpp`：variant `residual-prompt-pinned`、run_id 以 `-rpp` 结尾；`newtask.pt sha256 = 264aedbb9f3f605e5a8e5dd9b2aee8c1945d2dcf0388be722541d2a84f4062b8`；`prompt_report.json sha256 = 77f028993e8b…`、`metrics.json = f0a884ce6c59…`、`config.json = baec4bbc5c22…`、`gate_report.json = 573ebaf36338…`；参照路径记录 == `artifacts\aliccp_bench\runs\20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07\newtask.pt` | PASS |
| A5 | 跨进程/跨 commit 确定性（支撑"三臂可对账 + REP_Cp 前置条件"） | five-seed 记录：seed3–5 两次 pass 重跑**逐位相等**（含 `newtask.pt` 字节）；20-epoch 记录：20-epoch 基线前 10 epoch 逐位等于 10-epoch、前 5 epoch 逐位等于 5-epoch（两臂同理）；f08ae6e §10.1：B 逐位重放 `79b5e07`、REP 6 项全 true；A2 本次再次实测 ⇒ 同 seed 的 stage2 训练在本环境确定性可复现 | PASS |
| A6 | 机制实现的输入使用面（消融靶点定位） | `NewTask.forward`（`8133d32:multitaskrec/model.py`，零改动）2 处使用 `dnn_input`：`projection_network`、`gate_network`；`ResidualPromptNewTask.forward` 追加第 3 处：`prompt_deltas(dnn_input)`。本实验的消融面 = 第 3 处的**条件输入**（`x_i` → 常量 `c`；§2）；前三处（基头）保持正确输入 | PASS |
| A7 | 五 seed 立项条件（只读引用，不重推） | five-seed 分支：5 个 canonical seed 短预算配对逐 seed M0+G1–G8 全 PASS ∧ A 类 PASS，二级分类 5×`POSITIVE_IMPROVEMENT`，`mean_Δtest = +0.005910402347593546`（样本 std 0.0031538117454772883，Student-t 95% CI 下界 > 0） | PASS |
| A8 | **f08ae6e 审计/复核完整性（只读引用）** | f08ae6e §10.0/§10.5：预注册前核验 89/89；独立复核 101/101 ALL_PASS；三 run 均前台、无工具性中断、无重跑；`git diff 1c13841 bf322df -- . ':(exclude)artifacts/aliccp_bench/SUMMARY.md'` 为空 | PASS |
| A9 | **无条件常量向量预冻结（新）** | 以 §2.1 规程（`torch.Generator(manual_seed(20261006))` + 单次 `randn(80)`，float32）独立重实现：`sha256 = 0d45cc4611cebde9675858f1afc633a91b3e9114d6ddc02899a5d97f34babf58`（形状 (80,)、dtype float32、全有限、norm 8.512103733551866）；重抽逐位相同；推导前后全局 CPU RNG 端点不变（§1.5-9） | PASS |

### 1.3 seed / 预算 / α / 常量向量冻结（**先于任何本实验指标**）

**canonical seed 冻结为 seed2 = `1688723740`；预算冻结为 short（5-epoch / patience 2）；α 冻结为 `+0.07101669907569885`（signed）；无条件常量向量 c 冻结为 §2.1 的种子 `20261006` 单次 `torch.randn(80)`（sha256 `0d45cc46…`）。** 理由：

1. **seed2 是唯一具备完整既有证据链的 seed**（A2–A5）：同 seed 的 5/10/20-epoch 配对、钉死参照头（sha `90ee06da…`）、内容寻址 Stage-1 产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f`、f08ae6e 三臂记录齐备；其余 seed 只有 5-epoch 配对。
2. **short（5-epoch）是成本最低的完整协议点**：三臂实测墙钟 ≈160–200 s/run，总计 ≈10 min，单次前景窗口可控；预算效应不是本实验变量（历史 10/20-epoch 对照仅作上下文）。
3. **short 上学习 correct 臂效用最强且已过线**（`G_c^hist = +0.0081 ≥ +0.0055`）：若条件化在"最容易"的判定点上仍不显出实质差，`SAMPLE_CONDITIONING_NOT_SUPPORTED` 的结论最保守（同 f08ae6e §1.3-3 逻辑）。
4. **α 取 correct 臂 5-epoch 终值**且与 f08ae6e 两处理臂**逐位相同**：C_p 因此是 F_c 的逐位重放候选、U 与 C_p 的标量幅度匹配直接成立（唯一选出规则；不扫描 α、不试多个值，§9）。
5. **三因子分解（本实验的核心结构）**：C_p vs f08ae6e 学习臂 C 的差异 = α 学习性→钉死（被 §8-6 式训练动态混淆，仅描述）；C_p vs U 的唯一差异 = **条件输入逐样本 → 常量**（同架构、同参数、同初始化路径、同 α、同协议，单因子）；B vs C_p/U = 无 prompt 头。
6. **单常量单抽（c 一张）**：c 由冻结种子确定性定义，仅推及该向量；对"c 的分布"不做泛化主张（§8-3）。

### 1.4 机制审计（α 与条件的角色；UA 门禁的 a priori 依据）

机制（`013e105:aliccp_benchmark/residual_prompt.py`，公式与结构化事实逐字沿用 f08ae6e §1.4）：

    P(x)        = tanh(W2·ReLU(W1·x + b1) + b2)            # 有界方向，来自条件输入 x
    g_eff(x)    = |alpha| · ||P(x)|| / sqrt(d)             # 逐样本有效门控
    delta_s(x)  = alpha · (||h_s|| / sqrt(d)) · P(x)       # 残差 = 门控 × 单位方向 × 基向量范数
    h_s'        = h_s + delta_s                             # 注入融合前三路（gen + spec_0 + spec_1）

- **学习臂中 α 的角色**：`prompt_gate` 为零初始化 `nn.Parameter`；α=0 使首步生成器梯度为 0（∂loss/∂m ∝ α）。G2/G3/G5 都是**针对学习 α/学习幅度**的门禁——本实验不适用（a priori 声明，同 f08ae6e §1.4）。
- **本实验的 α**：C_p 与 U 的 α 同为非参数 buffer（`PINNED_ALPHA`，优化器结构性排除，训练全程逐位不变）⇒ 无"学习活性"命题；"激活"由 UA4(ii) 结构性保证，"幅度上界"由 UA6 结构性保证，"跨流一致"由 UA7 保证。
- **本实验的条件**：C_p = 逐样本 `x_i`（正确条件化）；U = **常量 c**（与样本无关；§2.1）。方向仍**可学习**（生成器 2384 参数全部在优化器内）⇒ C_p 与 U 的差异**恰为**"生成器的输入是样本还是常量"这一个因子；生成器的训练轨迹差异（对 C_p 的输入分布 vs 固定输入）是**设计的一部分**（预声明披露，§8-4）。
- **本消融恰好改变**：（i）U：生成器条件输入 `x_i` → `c`（常量 buffer）。（ii）C_p 相对 f08ae6e F_c：无（逐位重放，REP_Cp 前置条件）。**不改变**：α 语义（signed 标量、乘在范数标定后的方向前）、范数标定 `scale = ||h_s||/√d`、注入流集合、`super().forward` 委托、损失、优化器类型/超参、训练协议、数据顺序、参照头。

### 1.5 预注册前参照核验（只读；先于本文件提交；`verify_uncond_prerun.py`）

对固定产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f` + 参照头 + 五个历史 run 链（79b5e07 基线 / 013e105-rpg 学习 correct / 9d26bc8-rpgs 学习 shuffled / 7d26918-rpp 钉死 correct【重放靶】/ 1c13841-rpps 钉死 shuffled）+ 无条件常量向量推导做只读复算（**130/130 通过**，报告 `artifacts/aliccp_bench/audit/rp-uncond/verify_uncond_prerun_result.json`，sha256 `daa9e78a000f49e6e1785d4b3cc47e4e3fbe2baa4331332c1ae47afb530c0d07`；不写任何 run/stage1 产物、不训练、不重新评测）：

| 组 | 核对 | 结果 |
|---|---|---|
| 1 产物文件字节 | `backbone.pt`/`env_ids.pt`/`meta.json` sha256 == 钉死值（`cd7b0334…` / `4660be5a…` / `61a66d81…`） | PASS |
| 2 内容寻址 | `stage1_id` 重算 == 记录 == 期望；`config_hash` 重算 == `4e1b5c6f…`；id 分量（fp/config 前缀、seed=1688723740、epochs=3、env_seed=20261003、budgets） | PASS |
| 3 前缀指纹 A2 级 | 自哈希 == `5c060b9c…`；前缀字节 sha256 ×3 + 表头 + 文件大小 + 原始扫描标签计数 ×3 | PASS |
| 4 env_ids | 张量 sha 重算 == `5cd198f1…` == meta；形状 (2000000,)∈{0,1}；计数 1532/1998468 | PASS |
| 5 backbone | 张量 sha 重算（排除 buffer `env_indices`）== `e5e7e610…` == meta | PASS |
| 6 参照头 | 文件 sha256 == `90ee06da…`；`NewTask(80/64/[32,32])` strict 载入成功 | PASS |
| 7 历史 run 链（五 run） | 79b5e07（4 文件）、013e105-rpg（6 文件）、9d26bc8-rpgs（8 文件）、7d26918-rpp（5 文件）、1c13841-rpps（7 文件）逐文件 sha256 == 钉死；记录值钉死：五 run 的 test/val/逐 epoch val/gate_mean/best_epoch/epochs/patience/seed/run_id/variant/classification；013e105-rpg `VALID_POSITIVE`、`alpha_final == 钉死`；9d26bc8-rpgs `MECHANISM_FAIL`；7d26918-rpp `VALID_NEGATIVE`；1c13841-rpps `VALID_NEGATIVE` | PASS |
| 8 α 钉死来源 | `013e105-rpg/prompt_report.json` 文件 sha256 == `185df4d0…` ∧ `alpha_final == PINNED_ALPHA`（精确相等）∧ `float32` 往返精确 | PASS |
| **9 无条件常量向量（新）** | 以 §2.1 规程独立重实现：RNG 隔离（推导前后端点不变）∧ 重抽逐位相同 ∧ `sha256 == 0d45cc46…` ∧ 形状 (80,) float32 numel 80 ∧ 全有限 ∧ norm > 0 | PASS |
| **10 跨实验重放靶与三常量（新）** | `7d26918-rpp` `metrics.json`/`newtask.pt`/`prompt_report.json` sha256 == 钉死（`f0a884ce…`/`264aedbb…`/`77f02899…`）；由五 run 记录值逐位重推 `G_c == −0.002677172368479308` ∧ `G_s == −0.000815894357038216` ∧ `GAP == −0.001861278011441092` | PASS |

- **产物复用决议**：按"先验证、后复用，否则确定性重生成"规程——130/130 通过 ⇒ **复用**该产物（不重训 Stage-1）；产物、参照头、五个历史 run、前缀指纹均自 alpha-pinned worktree **只读复制**入本 worktree（逐文件 sha256 与源相等，复制后由本核验复验）；dataset 经 junction 复用 main tree（只读、gitignore）。
- 钉死参照常量（后续判定全部引用）：`BASELINE_AUC_TEST = 0.5974422649550507`、`BASELINE_AUC_VAL = 0.5809347091990792`、`REFERENCE_PRED_STD = 0.005217193225189258`；`CORRECT_AUC_TEST = 0.6055453782825184`、`CORRECT_AUC_VAL = 0.5895572066556973`、`DELTA_TEST_CORRECT = 0.008103113327467715`、`DELTA_VAL_CORRECT = 0.008622497456618139`；`PINNED_ALPHA = 0.07101669907569885`；**重放靶**：`PINNED_CORRECT_AUC_TEST = 0.5947650925865714`、`PINNED_CORRECT_AUC_VAL = 0.5798886656441761`、`PINNED_CORRECT_NEWTASK_SHA = 264aedbb9f3f…`、`PINNED_CORRECT_PROMPT_SHA = 77f02899…`；f08ae6e 钉死 shuffled 臂：`PINNED_SHUFFLED_AUC_TEST = 0.5966263705980125`、`PINNED_SHUFFLED_AUC_VAL = 0.5817369266195509`；历史学习 shuffled 臂：`SHUF_HIST_AUC_TEST = 0.5989210260551207`、`DELTA_TEST_SHUFFLED_HIST = 0.0014787611000699474`；**常量向量**：`UNCOND_CONST_SEED = 20261006`、`COND_VECTOR_SHA256 = 0d45cc4611cebde9675858f1afc633a91b3e9114d6ddc02899a5d97f34babf58`。

### 1.6 审计结论（立项裁定的确认 + 单因子陈述）

1. **f08ae6e 结果链验证成立**（A1/A4/A8 逐位复核）：`CONDITION_ALIGNMENT_NOT_SUPPORTED/GAP_BELOW_MATERIALITY`（`GAP=−0.001861`；两臂二级 `NO_CLEAR_IMPROVEMENT`），执行/实现/置换/复现全无故障 ⇒ 该结论有效，本实验的立项前提成立。
2. **单因子陈述**：在标量幅度匹配之后，条件对应仍未有实质贡献；但 f08ae6e 的两臂都**保留**"逐样本条件输入 + 可学习生成器"结构（含其有效幅度残留差异 32.7%）。本实验将条件输入因子**整体替换**为常量（同参数预算）⇒ 回答"逐样本条件化相对无条件/全局是否可分离"，并给出"无条件通道自身（vs B）是否承载效用"的读数。
3. 历史判定原样保留（§0 纪律声明）；本实验不重判任何历史结论。

---

## 2. 消融设计（唯一变化 = 条件输入；常量向量与 α 均先于实现写死）

### 2.1 无条件常量向量构造（**先于实现写死**）

- **种子**：`UNCOND_CONST_SEED = 20261006`（冻结常量）。
- **推导规程（逐字，供独立复核重实现；已在 §1.5-9 预核验）**：

      generator = torch.Generator(device="cpu")
      generator.manual_seed(20261006)
      c = torch.randn(80, generator=generator, dtype=torch.float32)

  单次 `randn(80)`；**专用 generator**（不触碰全局 torch RNG；`random` 模块不参与）；float32。
- **冻结值**：`sha256(c) = 0d45cc4611cebde9675858f1afc633a91b3e9114d6ddc02899a5d97f34babf58`（张量 sha = dtype+shape+bytes，与 §1.5 核验脚本一致）；形状 (80,)、dtype float32、全有限、`||c||₂ = 8.512103733551866`；描述统计（mean `−0.17465121473069303`、std `0.9355190152142492`、min `−2.2204647064208984`、max `1.713071584701538`）随 run 落盘。
- **性质（结构性 + 运行期探针双重证明）**：
  1. **确定性**：同种子同规程 ⇒ 逐位同一（§1.5-9 重抽逐位相同；实现内 `UncondPromptAudit` 再以**全新** generator 重抽并逐位比对）；
  2. **标签无关（label-blind）**：推导只依赖冻结种子与常数形状，不接收任何标签/特征/预测/梯度；
  3. **train/val/test 一致**：c 是构造期注册的 **buffer**（`uncond_condition`），三个 split 的每次前向都读同一 buffer（结构性；运行期 UA5 在真实 train/val 数据上复验）；
  4. **无 batch-composition 依赖（结构性）**：c 不来自任何 batch / 不依赖 batch 组成 / 不依赖数据顺序；换任何 batch、任何行序、任何 split，注入方向逐位不变（UA5 探针实测）；
  5. **RNG 隔离**：推导在专用 generator 上完成，全局 torch/Python RNG 端点逐位不变（§1.5-9；单元测试再验）；
  6. **落盘可核**：c 作为 buffer 进入 `state_dict`/`newtask.pt` ⇒ 独立复核可**直接读取**并与重抽值逐位比对（§10.5）。

### 2.2 无条件头构造（`UnconditionalResidualPromptNewTask`；先于实现写死）

- **继承**：`UnconditionalResidualPromptNewTask(PinnedAlphaResidualPromptNewTask)`——共享参数、α buffer（= `PINNED_ALPHA`，非参数、优化器排除）、生成器初始化路径（隔离 RNG 内）、`forward`/注入/委托全部**逐字继承**。
- **新增**：`self.register_buffer("uncond_condition", c)`（§2.1 的常量，float32，形状 == `input_size`；**非参数**、`requires_grad=False`、不进 `named_parameters()`、不进优化器）。
- **唯一重写（`prompt_deltas`）**：条件输入替换为常量广播：

      m = tanh(self.prompt_generator(self.uncond_condition.unsqueeze(0)))   # (1, d)
      m = m.expand(B, d)                                                    # 批内逐行逐位相同
      delta_s = alpha · (||h_s||_detach / sqrt(d)) · m                      # 公式逐字不变

  即：`deltas, m = prompt_deltas(dnn_input, reps)` 的签名与返回形状契约不变（调用方零适配），但**忽略 `dnn_input` 内容**（批大小仅用于 broadcast 形状）；`forward`、`super().forward` 委托、alpha 语义、范数标定**零改动**。
- **行为等价链（TDD 钉死，§4-I6/I8）**：① 任意两个不同 `dnn_input`（含 `zeros_like`、极端值、不同 batch 大小）⇒ `m` 逐位相同；② 同 batch 内所有行 `m`/逐行 ratio 逐位相同；③ 若把某 batch 的所有行都置为 c，则 U 的 `deltas` 与该 batch 下 C_p（同权重同 α）的 `deltas` **逐位相等**——即 U **恰为**"C_p 的条件输入替换为常量"。
- **梯度有效性**：α ≠ 0（钉死）且 c ≠ 0 ⇒ 生成器 4 个参数张量自首步即有非零有限梯度（UA11 运行期 + 单元测试；对照：零 α 探针下生成器梯度为 0，属预期）。

### 2.3 参数预算匹配量化（"α 与参数预算匹配"的落实）

| 量 | C_p（逐样本条件） | U（无条件/全局） | 匹配性 |
|---|---|---|---|
| 可训练参数：生成器 | 4 键（`prompt_generator.0/2.weight/bias`）= **2384** | 同 4 键 = **2384**（同架构、同形状、同隔离 RNG 初始化路径） | **逐项相同** |
| 可训练参数：α | 无（非参数 buffer，numel 1） | 无（同左） | **逐项相同** |
| 可训练参数总计 | **2384** | **2384** | **相同** |
| 优化器覆盖 | `named_parameters()` 全量（16 张量） | 同（16 张量，其中 prompt 4 键） | 相同 |
| buffer 新增 | α ×1（numel 1） | α ×1 + c ×1（numel 1 + 80） | **披露差异：U 多 80 个非可训练元素**（§8-10） |
| 有效函数类 | `m(x) = tanh(W2·ReLU(W1·x + b1) + b2)`，逐样本 | `m* = tanh(W2·ReLU(W1·c + b1) + b2)`，常量 | **同一参数化、同一容量，仅输入不同（单因子）** |

**读法（防误读，先于结果声明）**：C_p 与 U 的对照**不是**"多参数 vs 少参数"或"大模型 vs 小模型"，而是**同一参数预算、同一架构下的条件输入单因子**；两臂的可训练容量、优化器、初始化 RNG 消耗逐项相同；"额外参数通道"在**两臂中都存在**（这正是本实验要检验的"非对齐通道"之一）。

### 2.4 明确不改变（禁止清单）

标签/目标/损失/优化器与更新规则（Adam、lr=1e-4、`get_l2_reg`）、数据顺序与 batch 组成、`h_s` 基表征、`super().forward` 委托、α 值（= `PINNED_ALPHA`）与语义、范数标定 `scale`、注入流集合 `("gen","spec_0","spec_1")`、hidden=16、生成器初始化（隔离 RNG）、`BOUND_TOL`、参照头、U1/U2 口径（历史 `+0.0055`）、协议文件与 `multitaskrec/*`、以及 f08ae6e 已钉死的全部 RPP 代码（逐字节移植）。

---

## 3. 实现面（最小；TDD；等价证明）与适配清单

| 编号 | 文件 | 处置 | 守卫 |
|---|---|---|---|
| A1 | `aliccp_benchmark/residual_prompt.py` | **逐字节移植** `013e105:` 版（blob `674213f6…`；LF sha256 `b3b93b3a…`）。**零文本适配** | 守卫测试：工作树 == 钉死 blob 逐字节 + LF sha + `git hash-object` == blob + 继承的 Census 祖本 AST 钉死（`TestMechanismPin` 原样移植） |
| A2 | `aliccp_benchmark/rp_pinned.py` | **逐字节移植** `f08ae6e:` 版（blob `432fa9bb…`；LF sha256 `ba23bf47…`）。**零文本适配**（C_p 臂 = 该模块原样运行；其 shuffled 类/错排器为**休眠代码**：CLI 不暴露、本实验不接线、不运行） | 守卫测试：工作树 == 钉死 blob 逐字节 + LF sha + `git hash-object` == blob |
| A3 | `aliccp_benchmark/bench.py` | `f08ae6e:` 版（blob `b676a797…`）**+ 文档化最小适配**：import 段（增 `rp_uncond as RPU`）；`run_stage2` 段（新增 uncond 分支：构造/审计/臂级判定/prompt_doc 的 `uncond`+`pin` 块/run_id 后缀分派经 `RPU.arm_suffix_for`）；**其余模块级函数（含 `evaluate_newtask`）与全部控制流逐字不变** | 守卫测试：重建等式（钉死文本 + 恰好 2 段替换 == 工作树）＋ AST 差异集合恰为 `{run_stage2}`＋ tiny 端到端 C_p 路径与 f08ae6e 语义一致（另由真实 REP_Cp 逐位背书） |
| A4 | `run_aliccp_benchmark.py` | `f08ae6e:` 版（blob `74aaaf05…`）**+ 恰好 3 行适配**：import 行（增 `rp_uncond as RPU`）；`--variant` choices 行（增 `residual-prompt-uncond`；移除 CLI 暴露的 pinned-shuffled——本实验无错排臂）；臂后缀行（改经 `RPU.arm_suffix_for` 分派） | 守卫测试：行级替换重建等式 + 新 LF sha 钉死 |
| A5 | `aliccp_benchmark/rp_uncond.py`（新） | 消融本体：`UnconditionalResidualPromptNewTask`（§2.2）、`UncondPromptAudit`（继承 `RPP.PinnedPromptAudit`：PA1/PA2 记录 + **UA5 不变性探针** + c 重抽/隔离/摘要记录）、`draw_uncond_condition`（§2.1）、`arm_suffix_for`、`uncond_mechanism_gates`（UA1–UA11）、`uncond_arm_verdict`、`analyze_runs`（只读三 run 记录 → §5 全量判定）+ `python -m` CLI（输出 JSON） | 守卫测试（本分支 `test_residual_prompt_uncond.py`）：§4 全部不变量 + 判定树真值表/边界 + 常量推导/不变性/梯度 + 分析器夹具 |
| A6 | `aliccp_benchmark/tests/test_residual_prompt.py` | `f08ae6e:` 同名测试（blob `2b86d39d…`）**恰 4 处**适配：(a) 模块 docstring；(b) `DOC_PATH` → 本文件；(c) `WHITELIST` → 本分支 11 项（逐项集合钉死）；(d) `TestPreregConstants.test_prereg_doc_tokens_and_summary_ledger` 的 token 表 → 本文件 token 表。**其余逐类/逐方法/逐模块级语句不变** | 守卫测试：重建等式（docstring + `DOC_PATH` + `WHITELIST` + 该 1 个方法段替换）+ AST 差异集合 == 恰 1 个方法 + 新 LF sha 钉死 |
| A7 | `aliccp_benchmark/tests/test_residual_prompt_uncond.py`（新） | 本分支守卫 + 消融/接线/分析器测试（§4） | 本文件即为守卫 |
| A8 | `verify_uncond_prerun.py`（预注册前完整性核验；§1.5）与 `verify_rp_uncond.py`（运行后独立复核） | 只读脚本；**跟踪入库**（沿用 20-epoch/shuffled/pinned 先例；两处 WHITELIST 副本同步包含二者）；报告 JSON 落 `artifacts/aliccp_bench/audit/rp-uncond/` 与 U run 目录（gitignore）。前置脚本自 `f08ae6e:verify_pinned_prerun.py` 移植（五 run 链 + 第 9 组常量向量 + 第 10 组重放靶/三常量；钉死值 §1.5 不变）；复核脚本自 `f08ae6e:verify_rp_pinned.py` 移植（三 run 形态 B/C_p/U；记录一致性口径沿用） | 前置核验 130/130（§1.5）；复核脚本 §10.5 |

**被禁止的适配**：机制公式/初始化/门控语义/注入点/`BOUND_TOL`/U1 阈值/判定树/阈值/常量推导规程/α 值；`protocol.py`/`metrics.py`/`multitaskrec/*`/`config.py`——零改动（静态守卫：`git diff --name-only 8133d32` ⊆ 白名单）。

**等价性证明口径（三层，全部由测试守卫执行）**：
1. **重建等式（文本级）**：钉死 blob（f08ae6e）+ 恰好替换 == 工作树（bench/CLI/移植测试）；机制与 rp_pinned **逐字节**。
2. **机制 pin**：`residual_prompt.py` 逐字节 == 674213f6 + Census 祖本 AST 钉死；`rp_pinned.py` 逐字节 == 432fa9bb。
3. **行为级**：① `shuffler=None` 分派恒等（baseline/学习/钉死路径逐位不变）；② U 常量条件极限 ≡ C_p（§2.2-③）；③ α=0 下 U 与同权重参照 `NewTask` 逐位相等（UA4 运行期 + 单元测试）；④ 导入 `rp_uncond` 不消耗/改变任何 RNG 端点。

---

## 4. 不变量（由 `test_residual_prompt_uncond.py` 与移植测试强制，不接受人工目测）

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 基线/学习 correct/钉死 correct 臂逐位不变：`shuffler=None` 分派 ≡ 钉死调用；学习臂端到端记录键集与钉死一致 | 移植测试 + bench 重建等式 + 真实 REP_B/REP_Cp |
| I2 | 适配逐字守卫：A1–A4/A6 的字节/重建等式钉死 | `git show` + sha256(LF) + 段级重建等式 |
| I3 | 学习机制语义不变：α 初值 0、恒等/界/活性带测试全绿 | 移植测试原样通过 |
| I4 | **α 钉死恒常性（U 与 C_p）**：构造/训练前/训练后/checkpoint 四处 α == `PINNED_ALPHA` 精确相等；非参数 buffer、`requires_grad=False`、不在 `named_parameters()` | 单元测试 + 运行期 UA1/UA2（两臂） |
| I5 | **常量向量推导**：种子/规程冻结；同种子重抽逐位相同；全局 torch/Python RNG 端点不变；形状 (80,)/float32；sha256 == `0d45cc46…`；buffer 随 checkpoint 落盘且独立读取 == 重抽值 | 单元测试 + §1.5-9 + 运行期 UA5(c)/CV + 复核脚本 |
| I6 | **U 样本不变性**：任意输入（含 `zeros_like`/极端值/不同 batch 大小/单样本）⇒ `prompt_deltas` 的 `m` 逐位不变；同 batch 行间 `m` 与逐行 ratio 逐位相同 | 单元测试（n ∈ {1, 2, 7, 64}）+ 运行期 UA5 |
| I7 | **梯度有效**：钉死 α 下反传，生成器 4 张量梯度全非零且有限；零 α 下生成器梯度为 0（预期，供注入纯度探针） | 单元测试 + 运行期 UA11 |
| I8 | **U ≡ C_p（常量条件极限）**：同权重同 α 下，将 batch 全行置为 c 时两臂 `deltas` 逐位相等 | 行为测试 |
| I9 | state_dict/checkpoint：U 新增键恰 6（4 生成器参数 + α buffer + c buffer）；α 与 c 经 checkpoint 往返逐位保持 | 单元测试 + 复核脚本独立读取 |
| I10 | 判定树/边界：§5.4 全部标签真值表；`GAP_U` 恰 `+0.001` ⇒ `SUPPORTED`（闭），`−1e-12` ⇒ `NOT_SUPPORTED`；`Δval_gap_u` 恰 0 ⇒ 验证不一致；二级分类边界 `+0.001`/`−0.02`（闭端）；每条 INVALID 子原因可达；UA1–UA11 每条 FAIL 可达（夹具） | 边界/真值表测试（CPU 夹具） |
| I11 | 分析器只读：不改任何 run 产物；identity/REP_B/REP_Cp/UA/CC/CV mismatch ⇒ `INVALID`；输出键齐全（§5 全部判定 + 全部分量 + 描述量）；静态守卫：相对 `8133d32` 的全部跟踪改动 ⊆ 白名单；`protocol.py`/`metrics.py`/`multitaskrec/*`/`config.py` 零 diff；预注册文档 token 钉死（seed、stage1_id、`PINNED_ALPHA`、`UNCOND_CONST_SEED`、`COND_VECTOR_SHA256`、阈值、判定标签、重放靶 sha、适配清单 blob）；两处 WHITELIST 副本逐项相等 | 夹具端到端 + 篡改 fixture + `git diff --name-only` + 文档文本断言 + 双副本交叉测试 |
| I12 | 端口完整性：`rp_pinned.py` 逐字节 == f08ae6e blob；`residual_prompt.py` 逐字节 == 013e105 blob；导入 `rp_uncond` 无 RNG 副作用 | 守卫测试（§3） |

CPU 极小夹具不构成任何性能证据，只验证语义与接线。

---

## 5. 预注册判据（看到结果前写死；看到结果后不得修改）

对照定义：**B = 本实验新跑的配对基线臂**；**C_p = 本实验新跑的钉死 α 逐样本条件臂**（f08ae6e F_c 的逐位重放）；**U = 本实验新跑的钉死 α 无条件/全局臂**。`GAP_U := test(C_p) − test(U)`；`G_c := test(C_p) − test(B)`；`G_u := test(U) − test(B)`；`R_u := G_u / G_c`；`Δval_gap_u := val(C_p) − val(U)`；`Δval_c := val(C_p) − val(B)`；`Δval_u := val(U) − val(B)`。

### 5.1 前置条件（全部必须为真；任一假 ⇒ `INVALID`，不解读效用）

| 编号 | 判据 | 落盘 |
|---|---|---|
| **ID**（identity） | 三 run：`stage1_id` 相同且 == 钉死值；`model_seed` 相同且 == 1688723740；`epochs==5` ∧ `patience==2` ∧ `tag=="short"`；`dirty==false`（三 run 记录时）；B `variant=="baseline"` 且 run_id 无后缀；C_p `variant=="residual-prompt-pinned"` 且 run_id 以 `-rpp` 结尾；U `variant=="residual-prompt-uncond"` 且 run_id 以 `-rpu` 结尾；C_p 与 U 记录参照路径 == 钉死路径且文件 sha256 == `90ee06da…`；三 run `stage1_id` 的 backbone/env/fingerprint sha 与钉死值一致 | `rp_uncond_compare.json:identity` |
| **REP_B**（基线复现） | B 的逐 epoch val 轨迹（×5）、best_epoch、best_val、test、gate_mean 与 `79b5e07` 记录**逐位相等** 且 B 的 `newtask.pt` sha256 == `90ee06da…`（A2/A5 确定性预期成立；失败 ⇒ 环境偏离历史链，对照组不可比） | 同上 `reproduction_b` |
| **REP_Cp**（跨实验重放） | C_p 的逐 epoch val 轨迹（×5）、best_epoch、best_val、test、gate_mean 与 `7d26918-rpp` 记录**逐位相等** 且 C_p 的 `newtask.pt` sha256 == `264aedbb…` ∧ `prompt_report.json` 的 `alpha_final == PINNED_ALPHA`（失败 ⇒ 钉死路径相对 f08ae6e 发生实现分叉，等价性证明被证伪，全实验作废） | 同上 `replay_cp` |
| **A**（协议） | 三 run 的 A1/A2/A4/A5/A6 全 PASS（A3 SKIP 视为通过）；任一 FAIL ⇒ 该 run 比较作废 | 三 run `gate_report.json` |
| **UA**（结构机制门禁） | U 的 `rp_arm`：**UA1–UA11 全 PASS**（§5.2）；C_p 的 `rp_arm`：**PA1–PA8 全 PASS**（f08ae6e §5.3 口径，由本分析器对记录 `observed` 值机械重算，不信任记录布尔） | 两 run `metrics.json:rp_arm` |
| **CC**（对照链与钉死来源） | 五个历史 run（79b5e07 / 013e105-rpg / 9d26bc8-rpgs / 7d26918-rpp / 1c13841-rpps）的记录值 == §1.5 钉死常量（含 variant/classification/run_id；复核脚本执行文件 sha 核对）；**`PINNED_ALPHA == 013e105-rpg:prompt_report.json:alpha_final`（精确相等）且两处理臂 `alpha_final == PINNED_ALPHA`**；历史 Δ 重推逐位；**`G_c == −0.002677172368479308`（本实验 C_p−B 的记录值 == f08ae6e 的 G_c，逐位——REP_B∧REP_Cp 的端到端一致性核对）** | 同上 `comparator` |
| **CV**（常量向量完整性） | U 的 `prompt_report.json:uncond.cond_sha256 == COND_VECTOR_SHA256` ∧ `cond_regen_bit_identical == true` ∧ `cond.shape == (80,)` ∧ dtype float32；复核脚本独立重抽 c 并与 **checkpoint 内 `uncond_condition` 张量逐位比对** | 同上 `constant_vector` |

### 5.2 U 臂结构机制门禁（UA1–UA11；**替换**学习门禁——a priori 理由见 §1.4）

| 编号 | 判据 | 口径 |
|---|---|---|
| **UA1 钉死身份** | `prompt_gate` 为 buffer（非 `nn.Parameter`、`requires_grad False`、不在 `named_parameters()`）；构造时、训练后（best_state 载入后）、checkpoint（`newtask.pt` 独立读取）三处 `float(prompt_gate) == PINNED_ALPHA`（精确相等） | 运行期记录 + verifier 读 checkpoint 重推 |
| **UA2 优化器排除** | α 不在 `named_parameters()`；训练 optimizer 的 `param_groups` 覆盖全部 `named_parameters()` 且**不含** α（对象同一性）；逐 epoch 首 batch 探针 `alpha_grad_norm is None` 且 `alpha == PINNED_ALPHA`；训练前后 α 逐位不变 | 运行期 + 单元测试（I4） |
| **UA3 头身份（M0/G1 等价）** | 从构造前 RNG 现场重建 baseline `NewTask`：共享参数与 buffer 逐位一致 ∧ RNG 端点一致 ∧ 新增 `state_dict` 键恰 = `{prompt_gate, uncond_condition, prompt_generator.0.bias, prompt_generator.0.weight, prompt_generator.2.bias, prompt_generator.2.weight}`（6 键） | 运行期（`UncondPromptAudit` 继承构造审计）+ 单元测试 |
| **UA4 注入纯度与激活** | (i) α 临时置 0（探针内；恢复后逐位校验 `alpha_restored_exact`）⇒ 与同共享权重参照头在真实首 batch 上**逐位相等**（`zero_alpha_bit_identical`，`max_abs_diff == 0.0`）；(ii) α 钉死值下与参照头 `max_abs_diff > 0`（干预自首步即激活） | 运行期首 batch 探针 |
| **UA5 样本不变性（结构核心）** | (a) 真实首 train batch：`m` 行间逐位相同（`m` vs `m[0:1].expand` 逐位，`direction_max_abs_diff_across_rows == 0.0`）；(b) `zeros_like(dnn_input)` 输入下的方向与真实 batch 方向**逐位相同**（`direction_max_abs_diff_zero_input == 0.0`）；(c) c 以全新 generator 重抽与 buffer **逐位相同**（`cond_regen_bit_identical`）；(d) 记录 `cond_sha256`。另记录真实 batch 上逐行 ratio 的 max−min（须 ≤ 1e-9） | 运行期探针 + 单元测试（I6） |
| **UA6 范数界（S1）** | 逐流 `ratio_max ≤ |PINNED_ALPHA| + 1e-6` | val 诊断（fp64 流式） |
| **UA7 跨流一致（S2）** | 三流 `ratio_mean` 两两 spread ≤ 1e-6 | val 诊断 |
| **UA8 逐样本有效门控恒定（真无条件性实测）** | `geff_std ≤ 1e-9` ∧ `geff_min > 0` ∧ 逐流 `ratio_std ≤ 1e-9`（500k val 行；对照 f08ae6e C_p：geff_std `0.006211574794529912`） | val 诊断 |
| **UA9 无坍缩** | `pred_std > 0` ∧ `pred_std ≥ 0.5 × ref_pred_std` | val 诊断 + 参照头 |
| **UA10 参数预算/核算** | 可训练 prompt 键恰 4（generator；`new_params_total == 2384`）；`head_params == 8129`；α buffer numel == 1；c buffer numel == 80；state_dict 新增键 == 6（UA3 同集） | `param_report` + checkpoint 读取 |
| **UA11 梯度有效** | 逐 epoch 首 batch 记录齐全；每个 epoch 的 `generator_grad_norm` 均为有限且 > 0（常数方向可训练、非结构性死路） | 运行期梯度探针 |

**分类映射（先于结果写死）**：`not UA1 or not UA2 or not UA10` ⇒ `MECHANISM_FAIL / UNCONDITION_PIN_INVALID`；`not UA3 or not UA4` ⇒ `MECHANISM_FAIL / INVALID_IMPLEMENTATION`；`not UA5` ⇒ `MECHANISM_FAIL / NOT_TRULY_UNCONDITIONAL`；`not UA8` ⇒ `MECHANISM_FAIL / UNCONDITIONAL_GATE_NOT_CONSTANT`；`not UA11` ⇒ `MECHANISM_FAIL / GRADIENT_INVALID`；`not UA6` ⇒ `MECHANISM_FAIL / NORM_BOUND_VIOLATION`；`not UA7` ⇒ `MECHANISM_FAIL / CROSS_STREAM_INCONSISTENT`；`not UA9` ⇒ `MECHANISM_FAIL / PREDICTION_COLLAPSE`；`not M0` ⇒ `MECHANISM_FAIL / REFERENCE_IDENTITY`；protocol 失败 ⇒ `MECHANISM_FAIL / PROTOCOL_INVALID`；否则按 U1（`Δtest ≥ +0.0055` vs 基线常量）/U2（`Δval > 0`）给 `VALID_POSITIVE`/`VALID_NEGATIVE`（in-run，class C 诊断）。**学习活性带（原 G5）不设、`ratio_mean` 不判定**（描述量）；**"geff_std > 0"（原 G6 形态）对 U 结构性不适用**（真无条件 ⇒ 恒定是预期，其证明对象恰为 UA8 的 `geff_std ≤ 1e-9`）。

### 5.3 效用分量（判定的原子量；全部报告）

| 量 | 定义 | 预声明口径 |
|---|---|---|
| `GAP_U` | `test(C_p) − test(U)` | **因果主量**；实质为正：`GAP_U ≥ +0.001`（`MATERIAL_DELTA`） |
| `Δval_gap_u` | `val(C_p) − val(U)` | 验证方向：`> 0` ⇒ `validation_agreement = true`（**进 §5.4 判定树合取**；两臂同源对照，val 方向一致为因果对比的佐证；不进 UA/CC/CV 门禁） |
| `G_c` | `test(C_p) − test(B)` | 报告 + 二级分类 + 保留比 + CC 逐位核对 |
| `G_u` | `test(U) − test(B)` | 报告 + 二级分类 + **§9.3 后续消融的机械选择依据** |
| `R_u` | `G_u / G_c`（`G_c ≤ 0` ⇒ NA） | 报告（增益保留比） |
| `Δval_c` / `Δval_u` | `val(·) − val(B)` | 报告 + headroom |
| `retention_c_vs_hist_correct` | `G_c / 0.008103113327467715` | 报告（增益保留比） |
| `retention_u_vs_hist_correct` | `G_u / 0.008103113327467715` | 报告 |
| `unconditional_matches_or_exceeds_conditioned` | `GAP_U ≤ 0` | 报告（flag） |
| `G_c_ge_historical_0.0055` / `G_u_ge_historical_0.0055` | `≥ +0.0055` | §5.6 报告项（不判定） |
| `ratio_mean_c` / `ratio_mean_u` / `ratio_relative_diff` | 两臂 val 逐流 `ratio_mean` 均值与**相对差** | **有效幅度披露（预声明触发规则，同 f08ae6e §8-5 口径）**：相对差 > 5% ⇒ 解读时必须并读该披露（描述性，不作门禁） |

**阈值来源与读法（预声明）**：`+0.001` = 用户指定的最小实质差异，与既有分支分类边界一致；test 1M 前缀 BSI AUC 的 SE ≈ 0.0025 @AUC0.7 ⇒ `+0.001` ≈ 0.4×SE，为**分类标签**而非显著性强断言；单 seed、每臂 1 run、无 run-to-run 噪声估计（A5 沿用）。

### 5.4 因果判定树（**先于结果写死**；优先级自上而下，标签集固定为以下三种）

| 优先级 | 判定 | 条件 | 含义（写死） |
|---|---|---|---|
| 1 | **`INVALID`** | 任一前置条件假（ID / REP_B / REP_Cp / A / UA / CC / CV 任一失败） | 不解读效用；`subreason` 按优先级取首个失败项（`IDENTITY_MISMATCH` / `BASELINE_REPRODUCTION_FAILED` / `CROSS_EXPERIMENT_REPLAY_FAILED` / `PROTOCOL_INVALID` / `PINNED_ALPHA_INVALID` / `UNCONDITION_INVALID` / `MECHANISM_FAIL` / `COMPARATOR_MISMATCH` / `CONSTANT_VECTOR_MISMATCH`） |
| 2 | **`SAMPLE_CONDITIONING_SUPPORTED`** | `GAP_U ≥ +0.001`（闭）∧ `Δval_gap_u > 0` ∧ 前置条件全真 | 在标量幅度与参数预算匹配之下，**逐样本条件化承载实质效用**：条件化臂相对无条件臂 test 增量 **≥ +0.001** 且验证方向一致。子类（**报告、不改变本判定**）：`subreason = CONDITIONING_ESSENTIAL` 若 `G_u < +0.001`（无条件臂相对基线不再实质为正）；`subreason = PARTIAL_CONDITIONING_CONTRIBUTION` 若 `G_u ≥ +0.001` |
| 3 | **`SAMPLE_CONDITIONING_NOT_SUPPORTED`** | 前置条件全真 ∧ **实质规则失败**：`GAP_U < +0.001` **或** `Δval_gap_u ≤ 0` | 逐样本条件化相对无条件/全局提示**无可分离的实质效用**（或证据方向不一致）：与"效用由全局扰动 / 额外参数（两臂共有）/ 范数-门控 / 学习动态等非条件通道承载"一致。`subreason = GAP_BELOW_MATERIALITY`（`GAP_U < +0.001`）或 `VALIDATION_DISAGREEMENT`（`GAP_U ≥ +0.001` ∧ `Δval_gap_u ≤ 0`）；`GAP_U ≤ 0` 时附注 `unconditional_matches_or_exceeds_conditioned = true`（报告） |

**读法声明**：判定树**不设**"条件化臂相对基线实质为正"的前置（C_p 的 `G_c` 已知为 f08ae6e 的 `−0.00268`，由 REP_Cp ∧ CC 钉死；`G_c`/`G_u` 作为分量报告，不进门禁）——本实验只回答"同幅度、同预算下，**逐样本条件化相对无条件/全局是否承载可分离的实质效用**"。`G_u` 的全部解读在 §5.5 与 §9.3（机械规则），不改变本判定。

### 5.5 二级效用分类（用户指定；对 C_p 与 U **各自**相对 B 的 `Δtest` 机械分类；与 §5.4 并存、不覆盖）

| 类别 | 判据（精确边界） |
|---|---|
| `POSITIVE_IMPROVEMENT` | `Δtest ≥ +0.001` |
| `NO_CLEAR_IMPROVEMENT` | `−0.02 < Δtest < +0.001`（严格双开） |
| `CLEAR_DEGRADATION` | `Δtest ≤ −0.02` |

**`NO_CLEAR_IMPROVEMENT` 的机械 headroom 评估（仅当某臂落入该区间；规则先于结果写死；结构沿用 20-epoch/f08ae6e，因子对本实验适配）**：

| 因子 | 机械口径 | 取值 |
|---|---|---|
| 钉死机制活性 | `\|α_pin\| > 0` ∧ 三流 `ratio_mean > 0` ∧ `geff_max > 0`（取自该臂 JSON） | `PIN_ACTIVE` / `INACTIVE` |
| 验证方向 | `Δval_arm > 0` | `POSITIVE` / `NON_POSITIVE` |
| epoch 轨迹 | 该臂 `best_epoch == epochs_run == 记录 epochs`（未早停）⇒ 右删失；否则早停 | `RIGHT_CENSORED_STILL_IMPROVING` / `EARLY_STOPPED` |
| 距正分类余量 | `gap_to_positive_threshold = +0.001 − Δtest_arm`（>0） | 数值 |
| 保留比 | `Δtest_arm / 0.008103113327467715`（描述性） | 数值 |

（U 臂的"机制活性"因子说明：`geff_std ≈ 0` 是其**真无条件性证明**（UA8），不构成"失活"；活性以 α、`ratio_mean`、`geff_max` 判定，a priori 声明。）

### 5.6 历史 `+0.0055` 判定单列保留（不改写）

- 历史成效门槛 `+0.0055` 为学习 correct 臂效用口径（seed-2 复现预注册 §3-U1）；本实验两处理臂的 `G_c ≥ +0.0055`、`G_u ≥ +0.0055` 作为**报告项单列**（不参与 §5.4/§5.5 判定）。
- 历史判定原样引用：seed1 `VALID_NEGATIVE`（+0.005394543882337954）、seed2 `VALID_POSITIVE`（+0.008103113327467715）、10-epoch `PERSISTS`、20-epoch `NOT_PERSIST`/claim=true/`ABLATION_ELIGIBLE`、学习 shuffled `INVALID/MECHANISM_FAIL`、f08ae6e `CONDITION_ALIGNMENT_NOT_SUPPORTED`（`GAP=−0.001861278011441092`）。本实验不重判任何历史结论。

---

## 6. 运行规程（恰好三次；清洁树纪律；前台）

**运行次数**：B stage2 **恰好 1 次**；C_p stage2 **恰好 1 次**；U stage2 **恰好 1 次**；分析（`analyze_runs`，只读）**恰好 1 次**；运行后独立复核 **恰好 1 次**。不重训 Stage-1；任何历史 run 不重跑（§5 对照仅取已提交记录）。

**清洁树纪律（run 间提交，沿用 20-epoch §4 / f08ae6e §6）**：① 提交实现与测试（C2）→ `git.dirty=false`；② 跑 **B**（run 记录 commit=C2）；③ 提交 B 的 SUMMARY 追加行（C3，仅 `artifacts/aliccp_bench/SUMMARY.md` 变化）；④ 跑 **C_p**（run 记录 commit=C3）；⑤ 提交 C_p 的 SUMMARY 行（C4）；⑥ 跑 **U**（run 记录 commit=C4）；⑦ 提交 U 的 SUMMARY 行（C5）；⑧ 核验 `git diff C5 C2 -- . ':(exclude)artifacts/aliccp_bench/SUMMARY.md'` 为空；⑨ 分析 ×1 → 复核 ×1 → C6（§10 回填）→ push。

```powershell
# cwd = 本 worktree 根；本 worktree 无 .venv，用主树解释器；dataset 经 junction 复用（只读）

# 0) 测试（先红后绿；全部不变量）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests

# 1) B 配对基线臂（唯一一次；协议默认：epochs 5 / patience 2 / tag short）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag short --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f

# 2) C_p 钉死 α 逐样本条件臂（唯一一次；= f08ae6e F_c 的逐位重放；α=+0.07101669907569885、钉死参照头）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag short --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f `
    --variant residual-prompt-pinned `
    --prompt-reference-newtask artifacts/aliccp_bench/runs/20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt

# 3) U 钉死 α 无条件/全局臂（唯一一次；同产物、同 seed、同预算、同 α；条件 = 冻结常量 c、sha 0d45cc46…）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag short --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f `
    --variant residual-prompt-uncond `
    --prompt-reference-newtask artifacts/aliccp_bench/runs/20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt

# 4) 分析（纯分析，恰一次；只读三 run 记录）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m aliccp_benchmark.rp_uncond `
    --baseline-run <B run_id> --arm-cond-run <C_p run_id> --arm-uncond-run <U run_id>

# 5) 运行后独立复核（只读；JSON 重推 + 文件哈希 + c 独立重抽 + checkpoint 独立读 c/α）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe verify_rp_uncond.py `
    --baseline-run <B run_id> --arm-cond-run <C_p run_id> --arm-uncond-run <U run_id>
```

- **前台执行并等待至完成**（先例：后台执行被会话终止杀死）；开跑前提交实现（§6 步骤①）；每臂 run 启动时 `git.dirty` 必须为 false。
- **无效执行处置**：仅当 run 因**工具性原因**（崩溃/中断/环境故障、产物未完整落盘）无效时，保留现场、记录原因后可重跑一次；因**结果原因**不构成无效，不重跑。无任何事后调参（`PINNED_ALPHA`/`UNCOND_CONST_SEED`/构造/阈值/参照/判定树均不得动）。
- **墙钟预算**：每臂 ≈2–4 min（short 实测 156–199 s + 无条件臂开销），总计可控。

---

## 7. 必录诊断（落盘 run 目录；class C，不判定）

| 量 | 字段 | 用途 |
|---|---|---|
| **钉死 α 证明** | `pin`（U 与 C_p 各有）：`pinned_alpha`、`source_run`/`source_field`/`source_file_sha256`、`is_parameter=false`、`requires_grad=false`、`in_named_parameters=false`、`alpha_at_construction`、`alpha_after_training`、`alpha_restored_exact`、`optimizer_covers_named_parameters`、`optimizer_contains_alpha=false`、`optimizer_params_total` | UA1/UA2（+C_p PA1/PA2） |
| **常量向量证明（U）** | `uncond`：`const_seed`、`cond_sha256`、`cond_numel/dim`、`cond_stats`（mean/std/min/max/norm）、`regen_bit_identical`、`rng_isolation_ok`、`invariance`：`direction_bit_identical_across_batch_rows`、`direction_max_abs_diff_across_rows`、`direction_bit_identical_under_zero_input`、`direction_max_abs_diff_zero_input`、`ratio_rows_max_minus_min` | UA5/CV |
| 构造身份 | `construction_identity`：共享参数逐位/RNG 端点/新增键集（U 6 键；C_p 5 键） | UA3（+PA3） |
| 注入纯度 | `init_forward`：`zero_alpha_bit_identical`、`zero_alpha_max_abs_diff`、`pinned_max_abs_diff`、`alpha_restored_exact` | UA4（+PA4） |
| 梯度探针 | `grad_probe`（逐 epoch 首 batch：`alpha`、`alpha_grad_norm`(=None)、`generator_grad_norm`） | UA2/UA11 + 描述 |
| 残差/基范数比 | 逐流 `ratio_mean/std/max`、`delta_norm_mean`、`h_norm_mean` | UA6/UA7 + 有效幅度披露 |
| 余弦方向 | 逐流 `cos_mean` | 描述（对照历史：学习 correct +0.01287/+0.08432/+0.05899；钉死 correct −0.03315/+0.06856/−0.00197） |
| 逐样本门控 | `gate.geff_mean/std/min/max` | UA8 + headroom（对照：钉死 correct `geff_std = 0.006211574794529912`；U 预期 ≤1e-9） |
| 预测离散度 | `dispersion` + `reference_dispersion`（M0 口径） | UA9 + 描述 |
| 源/头门 | `gate_mean` + Stage-1 环境上下文（`cluster_events`、`env_acc`） | 落盘；B4 原样披露 |
| 参数清单 | `params`：4 键 generator、`new_params_total=2384`、`head_params=8129`；α buffer numel=1；c buffer numel=80 | UA10 |
| 臂级判定 | 两处理臂 `arm`：U = UA1–UA11 + U1/U2 + 分类 + `pin`；C_p = PA1–PA8 + U1/U2 + 分类 + `pin` | §5.2 机械分类 |
| 配对判定 | `rp_uncond_compare.json`：§5 全部前置条件、分量（`GAP_U`/`G_c`/`G_u`/`R_u`/`Δval_gap_u`/保留比/披露量）、判定树、两臂二级分类、headroom（若适用）、context 表 | §5.4/§5.5 |
| 停止/删失 | 三 run 的 `best_epoch`/epochs、是否早停、逐 epoch 轨迹 | 描述 |
| provenance | 三 run 的 `run_id`/`commit`/`git.dirty`/`wall_seconds`/`peak_vram_mb`/`hard_pass`/A 类/B 类（继承披露） | §10 记录 |

---

## 8. 局限（预声明）

1. **单 seed、每臂 1 run**：无 run-to-run 噪声估计（A5 沿用）；`+0.001` 为分类标签而非显著性强断言（≈0.4×SE）。
2. **单一 α 值**：钉死 α = correct 臂 5-epoch 终值；结论条件于 α = `+0.07101669907569885`，不外推到其它幅度（无 α 扫描——§9）。
3. **单一常量向量（单抽）**：c 由冻结种子定义的**一张**向量；结论只推及该 c（"c 分布上的平均效应"为另一实验）；c 的推导已冻结且处处可核（§1.5-9/§10.5），但不构成"任意全局提示"的泛化证据。
4. **α 钉死但有效幅度仍含学习组分**：逐样本有效幅度 `|α|·||m||/√d` 中 `m` 在两臂仍被学习（C_p 学自 `x_i` 分布、U 学自常量 c）⇒ 若两臂 `ratio_mean` 出现实质差异（预声明：相对差 > 5% 视为实质），则"条件化 vs 无条件"仍残留一个**可量化**的幅度分量；报告两臂 `ratio_mean` 与相对差（§5.3），判定树不含该量（描述性披露，不作门禁；对照 f08ae6e §8-5/§10.3 先例）。
5. **学习性 vs 钉死的对照仍被混淆（预声明）**：C_p/U 与学习臂（`013e105-rpg` 等）的差异不止 α 的学习性（生成器首步梯度 0 vs >0、α 轨迹、生成器共适应）——本实验**不**提供"学习 vs 钉死"的因果证据（f08ae6e §10.7.2 口径沿用）；如需该结论，须另立预注册（§9.3）。
6. **判定依赖单一汇总统计**（test 1M 前缀 BSI AUC 差值）；判定取 run 记录值；正确性依赖代码路径守卫 + REP_B/REP_Cp 逐位 + 独立复核。
7. **REP_Cp 前置条件依赖跨进程确定性**：由 A5 的多次逐位重放背书；若失败 ⇒ `INVALID`（钉死路径实现分叉的证伪信号，保留现场核查）。
8. **in-run `rp_arm.U`（U1/U2）的配对对象是历史 5-epoch 常量**（class C 诊断）；因果判定只用 §5 的 B/C_p/U 配对。
9. **B4 若 FAIL 为继承缺陷**（Stage-1 共享语义）；`hard_pass` 不参与有效性定义（同先例）。
10. **U 多 80 个非可训练 buffer 元素**（c）；可训练预算与优化器覆盖与 C_p 逐项相同（§2.3）；该差异不复用为"容量"解释。
11. **单数据集（AliCCP）、单 seed**：结论不推及 CensusIncome / ByteRec、不推及其它 seed / 预算点。
12. **本实验不是"更优方法"主张**；是机制因果消融的一次性判定。

---

## 9. 非目标与止损

1. 不做：其它消融（α 调度/延迟解冻、norm-control 移除、跨 split 条件、第二张 c、学习 shuffled 变体接线——**明确排除**）、不写论文/图表、不做聚类、不做 Census、不跑 `full` tag、不做 FLOPs、不扩 seed、不重跑任何历史 run、**不做 α 扫描**。
2. 不因结果修改：`PINNED_ALPHA`、`UNCOND_CONST_SEED`、常量推导规程、`MATERIAL_DELTA`、二级边界、判定树、§5.1–§5.4 任何规则、参照文件、clean-tree 纪律。
3. **后续独立消融的预声明建议（本分支不启动，仅登记；按结果机械选择，先于结果写死）**：
   - 若 `SAMPLE_CONDITIONING_SUPPORTED` ⇒ 条件化承载可分离实质效用 ⇒ 下一步最优先：**matched-parameter adapter** 对照（与 prompt 同预算的另一种逐样本条件化构件，检验条件是"prompt 方向机制专属"还是"任何同容量条件化"均可承载——机制泛化性检验）；
   - 若 `SAMPLE_CONDITIONING_NOT_SUPPORTED` 且 `G_u ≥ +0.001`（无条件通道相对基线实质为正）⇒ 全局通道承载效用 ⇒ 下一步最优先：**norm-control 移除**（把范数标定 `scale = ||h_s||/√d` 换成常数标定，其余逐字不动，检验"范数-门控"是否为该通道的承载者）；
   - 若 `SAMPLE_CONDITIONING_NOT_SUPPORTED` 且 `G_u < +0.001`（无条件通道相对基线无实质增益）⇒ 固定注入族（条件化与无条件皆然）不承载可观测效用 ⇒ 机制证据唯一指向**学习动态**（学习 α 臂 `+0.0081` 是全线唯一正读数）⇒ 下一步最优先：**学习-vs-钉死解释消融**（α 轨迹回放/延迟解冻设计，直测"学习动态/共适应承载增量"假说）；
   - `INVALID` 情形下先修复执行/实现，不启动任何后续消融。
   - 上述三者均需**全新预注册**；本分支不启动。（α 调度消融 = f08ae6e §10.7.1 的"可选二元"，在本实验的 `NOT_SUPPORTED ∧ G_u < +0.001` 分支下与"学习-vs-钉死"合并为同一优先项。）

---

## 10. 结果（运行后回填，不回写第 0–9 节）

（待运行后回填：§10.0 非结果核验与执行记录；§10.1 三条 run 与臂级身份；§10.2 判定；§10.3 结构机制门禁与诊断；§10.4 常量向量完整性；§10.5 独立复核；§10.6 纪律核对；§10.7 解读与下一消融建议。）

---

## 11. 非新颖性声明

构件全部为既有方法族（prompt tuning、FiLM 族条件化调制、adapter/LoRA 残差、门控残差、范数归一、全局/无条件提示为消融常用手段），先行工作清单沿用 seed1 设计 §11（写作引用前必须逐条核实，本会话未做系统检索）。本文件**不**声称独立于先行工作；可报告内容仅限本仓库冻结协议下的一次性判定。

---

## 12. 偏离披露（预声明 + 运行后补）

- **预注册后澄清提交（C1b，如需要）**：若本文件在实现前需澄清（补全 blob/run id 等），以独立提交修正；判据、阈值、构造、判定树、运行规程**零改动**（同类先例：fbfff09 C1b `768ba8b`、f08ae6e C1b `0dfb11a`）。
- **分支工作流**：自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；机制与 rp_pinned 经 §3 三层守卫钉死等价；协议不合并 `master`；push 经用户显式指示（本任务含 "commit and push"）。
- **分支本地 SUMMARY 系谱**：本分支 `SUMMARY.md` 基线含 smoke 行与 seed1 基线行（`8133d32` 携带）；其余历史行记录在各自分支，本分支**不复制**其它分支行，只由本实验 run 追加自己的行（B + C_p + U）。
- **产物/历史 run 只读复制**：§1.5 所列产物、参照头、五个历史 run、前缀指纹均自 alpha-pinned worktree **只读复制**入本 worktree（逐文件 sha 相等）；dataset 经 junction 复用 main tree（只读、gitignore）。
- **非结果文件（未跟踪）**：`artifacts/aliccp_bench/audit/rp-uncond/` 报告 JSON、`artifacts/aliccp_bench/logs/` run 日志、U run 目录内 `rp_uncond_compare.json` / `verify_report*.json`。
- **与前置分支的关系**：f08ae6e 的 shuffled 臂（F_s）不移植入本分支运行时（CLI 不暴露 `residual-prompt-pinned-shuffled`；`rp_pinned.py` 内的错排类为逐字节移植带来的休眠代码，不接线、不运行、不进任何门禁）。
- **执行记录（运行后补）**。
- **跑后代码改动披露（运行后补）**。
- **实际结果形态（运行后补）**。
- **下一消融登记（运行后补，按 §9.3 机械选择）**。

---

**改动清单（白名单；静态守卫以 `8133d32` 为基准）**

| 文件 | 处置 |
|---|---|
| `docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-unconditional-control-design.md` | 本文件（C1 单独提交） |
| `aliccp_benchmark/residual_prompt.py` | 新增（A1：逐字节 == `013e105` 钉死 blob `674213f6…`） |
| `aliccp_benchmark/rp_pinned.py` | 新增（A2：逐字节 == `f08ae6e` 钉死 blob `432fa9bb…`） |
| `aliccp_benchmark/rp_uncond.py` | 新增（A5：消融本体 + 判定 + 分析 CLI） |
| `aliccp_benchmark/bench.py` | A3：== `f08ae6e` + 文档化最小适配（uncond 分派；其余逐字） |
| `run_aliccp_benchmark.py` | A4：== `f08ae6e` + 文档化 3 行适配（import/choices/后缀分派） |
| `aliccp_benchmark/tests/test_residual_prompt.py` | 新增（A6：== `f08ae6e` 恰 4 处适配） |
| `aliccp_benchmark/tests/test_residual_prompt_uncond.py` | 新增（A7：本分支守卫 + 消融/接线/分析器测试） |
| `verify_uncond_prerun.py` | 新增（A8：预注册前完整性核验；§1.5 已运行） |
| `verify_rp_uncond.py` | 新增（A8：运行后独立复核） |
| `artifacts/aliccp_bench/SUMMARY.md` | 只追加 3 行（runner 自动） |
| 其余全部（`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`metrics.py`、既有测试三件、`AliCCP_*.py`、`CensusIncome_*.py`、`baseline/*`、`mask/*`、`analysis/*`） | **零改动**（静态守卫断言 `git diff --name-only 8133d32` ⊆ 白名单） |
