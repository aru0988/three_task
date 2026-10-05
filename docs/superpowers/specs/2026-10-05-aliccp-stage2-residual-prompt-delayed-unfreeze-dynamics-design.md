# AliCCP 阶段 2 残差 Prompt 的**延迟解冻学习动力学**消融（delayed-unfreeze dynamics；canonical seed-2、5-epoch short、双臂单因子）

- **状态**：预注册已写死（本文件在本分支任何实现、任何 run **之前单独提交**；commit C1）。结果只在第 10 节以新增小节回填；第 0–9 节的判据、数字、构造与运行规程不得在看到结果后改动。
- **日期**：2026-10-05
- **适用分支**：`exp/aliccp-stage2-residual-prompt-alpha-dynamics`（自 `infra/aliccp-fair-benchmark` @ `8133d32` **独立拉出**；不从任何 exp 分支拉出、不继承其运行时状态；不合并 `master`、不触碰其它 worktree；`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`aliccp_benchmark/metrics.py`、既有测试三件、master 脚本与 `baseline/*` 零修改）。
- **立项依据（机械选择，先于结果写死）**：`exp/aliccp-stage2-residual-prompt-unconditional-control` @ `180d8d5`（`180d8d535831db7d40ee268023506f23e359c480`）的结果——因果判定 `SAMPLE_CONDITIONING_NOT_SUPPORTED/GAP_BELOW_MATERIALITY`（`GAP_U = −5.8180870650015315e-05`；C_p 与 U 两臂二级分类均 `NO_CLEAR_IMPROVEMENT`）——按该分支预注册 §9.3 的机械规则：`SAMPLE_CONDITIONING_NOT_SUPPORTED` **且** `G_u = −0.00262 < +0.001` ⇒ **下一步最优先 = 学习-vs-钉死解释消融**（α 轨迹回放 / 延迟解冻设计，直测"学习动态/共适应承载增量"假说；§10.7.1 已登记）。本实验即该登记条目的**预注册化执行**；其余两个候选（norm-control 移除、matched-parameter adapter）按机械规则**未满足选择条件**，本分支不得启动（§9）。
- **被消融对象（只读引用；系谱钉死）**：残差 prompt 机制 = `exp/aliccp-stage2-residual-prompt-seed-replication` @ `013e105`（`013e105f6dfb35e630d261696290635d0263ce99`）`:aliccp_benchmark/residual_prompt.py`（blob `674213f619c5d5039811c71242a7727118348daf`，LF sha256 `b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc`）；其系谱：seed1 线（预注册 `221580a` / 实现 `79ddefa`（blob `5dc7158ca999c9e7e6217a999c37869231411549`）/ 结果 `3e2f083`）→ Census 祖本 `99b9510`（blob `107221b26382da7fd44990e167d9a61da2680d92`）。钉死 α 消融实现（本实验只读引用其核查常量与运行期审计形态）= `f08ae6e:aliccp_benchmark/rp_pinned.py`（blob `432fa9bb6b3bb9d537753599499b693197e8c898`，LF sha256 `ba23bf473634ff647b482ebf6a92f27ca7f073432bb4ed7a1342b8b11101df9b`）。
- **审计对象（只读；§1）**：unconditional `180d8d5`、alpha-pinned `f08ae6e`（`f08ae6e459ae9fe048daf11eda43799d19ccc7d1`）、five-seed `28aa1fe`（`28aa1feae1fea0762136beebdfe6eb8387edb170`；结果提交 `0935257`）、seed2 学习参照 `a40836e`（`a40836e3d6a7f73c4943bada98139d414957301a`）、20-epoch `10e86dc`（`10e86dc6cbd047b103211ab75dd99c1c68db3526`）；全部判定以 **git 对象 + 磁盘原始 run 产物逐文件 sha256 复核**为准，不信任文档转述。
- **上游协议（只引用、不修改）**：
  1. `docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（AliCCP 公平评测协议）——前缀预算、种子、Stage-1 产物、真冻结三件套、A/B/C 门禁、SUMMARY 台账、止损纪律全部沿用；
  2. `docs/superpowers/specs/2026-10-04-aliccp-stage2-residual-prompt-seed-replication-design.md`（seed-2 复现预注册）——机制语义、参照头钉死、U1/U2 口径（历史 `+0.0055`）逐字沿用；
  3. `exp/aliccp-stage2-residual-prompt-20epoch` @ `10e86dc`——审计方法（以磁盘原始 run 产物 + git 对象为准）、二级分类三标签边界、`NO_CLEAR_IMPROVEMENT` 的机械 headroom 结构、`commit-between-runs` 清洁树纪律、独立复核脚本形态（纯 JSON 重推）**逐字沿用**；
  4. `exp/aliccp-stage2-residual-prompt-alpha-pinned-condition` @ `f08ae6e` 与 `exp/aliccp-stage2-residual-prompt-unconditional-control` @ `180d8d5`——判定树/二级分类/headroom 形态、`REP` 前置条件口径、预注册前完整性核验 + 运行后独立复核形态、常量向量式"先于实现写死"纪律**逐字沿用**；
  5. `exp/aliccp-stage2-residual-prompt-five-seed` @ `28aa1fe`（结果 `0935257`）——五 seed canonical 配对与跨进程逐位确定性证据（§1.2-A5）。
- **定位声明**：本实验新增**唯一处理臂 D（delayed-unfreeze）**——把可学习标量门控 α 与 prompt 生成器整体**冻结于构造值**（α = 0.0；生成器 = 隔离 RNG 初始权重）**恰好 1 个 epoch**（= 历史学习臂 α 离开 0 的那个 epoch；§2.2 的机械定界规则），在 epoch 2 起点解冻（恢复与学习臂相同的可学习语义；优化器状态自首个有效梯度起与学习臂同构）。其余一切（架构、初始化路径、范数标定、注入流、损失、优化器类型/超参、数据顺序、参照头、协议）逐字沿用学习臂。**配对基线 B 为本分支新跑**（REP_B 前置条件）。**无新颖性主张、无改进主张、不调参、不放宽任何门限；不重跑任何历史 run。**
- **提交时序（预注册纪律）**：C1 本文件（审计 + 预注册，单独提交）→ C2 实现 + 测试（TDD；两 run 的 `commit` 字段即 C2 与 C3）→ 跑 **B 臂（配对基线）**（前台）→ C3 提交 B 的 SUMMARY 行 → 跑 **D 臂（延迟解冻）**（前台）→ C4 提交 D 的 SUMMARY 行 → 分析恰一次 → 独立复核恰一次 → C5 结果回填（§10）+ push。

---

## 0. 判定问题

对照三分：**B = 本实验新跑的配对基线臂**（seed2、short、5-epoch；协议默认）；**D = 本实验新跑的延迟解冻臂**（唯一变化 vs 学习臂 = α 与生成器的学习期延迟 1 个 epoch；§2）；**L = 历史学习 correct 臂**（`20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg`，commit `013e105`，dirty=false）与 **C_p = 历史钉死 α correct 臂**（`20261005-1029-p2M-v500k-t1M-m1688723740-short-7d26918-rpp`，commit `7d26918`，dirty=false）为 **hash 核验后的只读上下文对照**（不重跑、不进任何门禁，其数值作为钉死常量进入描述统计与前置条件 CC；§1.5/§3）。

记 `ΔD := test(D) − test(B)`（**因果主量**；同分支、同流、同 seed 配对）；`Δval_D := val(D) − val(B)`；`ΔL := test(L) − test(B) = +0.008103113327467715`（钉死常量，逐位重推）；`ΔP := test(C_p) − test(B) = −0.002677172368479308`（钉死常量，逐位重推）；`GAP := ΔL − ΔP = +0.010780285695947023`（钉死 α 与学习 α 之间的全部差值）；`gap_closure := (ΔD − ΔP) / GAP`；**半程门量**：`half_gap_met := ΔD ≥ ΔP + 0.5·GAP = +0.0027129704794942035`。

- **Q1（因果主判定）**：在**同一构造、同一初始化路径、同一优化器语义**之下，把 prompt 模块（α + 生成器）的学习**恰好延迟 1 个 epoch**（冻结期与配对基线逐位同流；§2.3 冻结恒等），是否仍相对配对基线取得**实质正增量**（`ΔD ≥ +0.001`）且验证方向一致（`Δval_D > 0`）？→ §5.4 判定树：`DYNAMICS_SUPPORTED` / `DYNAMICS_NOT_SUPPORTED` / `INVALID`。
- **Q2（半程回收，预声明描述统计）**：D 是否**回收固定 α 与学习 α 之间差距的至少一半**（`half_gap_met`；等价 `gap_closure ≥ 0.5`）？`gap_closure`、`ΔD` 相对 `ΔP`/`ΔL` 的位置、`ΔD/ΔL`（相对回收比）全部报告；**`half_gap_met` 为预声明描述统计，不进判定树**（§5.3/§5.4 读法）。
- **Q3（二级效用分类）**：D 相对 B 的 `ΔD` 落入用户三标签（边界先于结果写死，§5.5）；若落入 `NO_CLEAR_IMPROVEMENT` 则按 §5.5 机械评估 headroom。
- **Q4（完备性）**：前置条件（ID / REP_B / FROZEN_ID / A 类 / D 臂 DD1–DD9 / CC 对照链）是否全真？任一假 ⇒ `INVALID`（不解读效用）。

**纪律声明（预注册）**：历史判定与全部数值（seed1 `VALID_NEGATIVE`、seed2 `VALID_POSITIVE` +0.0081、10-epoch `PERSISTS`、20-epoch `NOT_PERSIST`/claim/`ABLATION_ELIGIBLE`、5-seed 均值 `+0.005910402347593546`、学习 shuffled `INVALID/MECHANISM_FAIL`、f08ae6e `CONDITION_ALIGNMENT_NOT_SUPPORTED`（`GAP=−0.001861`）、uncond `SAMPLE_CONDITIONING_NOT_SUPPORTED`（`GAP_U=−5.8e-5`）、`+0.0055` 历史效用门等）**原样保留、不改写、不重判**。本实验只用 §4/§6 新跑的两条 run 做因果与分类判定；历史 run 仅作 hash 核验后的上下文对照（L 与 C_p 的数值为钉死常量，由 CC 逐位重推背书）。

---

## 1. 审计（2026-10-05，只读；先于本文件提交）

### 1.1 审计方法与范围

- **原则**：不信任任何文档转述，全部以 **git 对象**（blob / `rev-parse`）与**磁盘上的原始 run 产物**（本 worktree 内按 §1.5 逐文件 sha256 核验后只读复制入的副本）为准重新核对；不重跑、不修改任何历史 run；全部命令可复现。
- 审计对象与 commit（`git rev-parse` 实测）：`180d8d5` = `180d8d535831db7d40ee268023506f23e359c480`；`f08ae6e` = `f08ae6e459ae9fe048daf11eda43799d19ccc7d1`；`28aa1fe` = `28aa1feae1fea0762136beebdfe6eb8387edb170`；`a40836e` = `a40836e3d6a7f73c4943bada98139d414957301a`；`10e86dc` = `10e86dc6cbd047b103211ab75dd99c1c68db3526`；另引 `013e105` = `013e105f6dfb35e630d261696290635d0263ce99`、`79ddefa` = `79ddefa98b8a4c4d32403ac5027c24bd7420844a`、`e931cd7`、`0935257` = `0935257079c769f45e6c65cd0419c7abe5bba725`、`fbfff09`、`e46e5d2`、`3e2f083`（five-seed/20-epoch/seed1 链，只读引用）。
- 审计为只读核对（git + 原始 JSON 重读 + sha256 复算）；无可跟踪审计脚本；审计证据 dump（比较链全量与六条学习臂 α 轨迹）由 §1.5 的预核验脚本一并复算落盘。

### 1.2 审计结果（逐项）

| # | 审计项 | 证据（独立重推） | 结果 |
|---|---|---|---|
| A1 | **uncond 结果链（立项核心）** | `180d8d5` 原文与 §10 回填：判定 `SAMPLE_CONDITIONING_NOT_SUPPORTED/GAP_BELOW_MATERIALITY`；`GAP_U = −5.8180870650015315e-05`；`G_u = −0.0026189914978292927`；`G_c = −0.002677172368479308`（== f08ae6e 重放靶，逐位）；两臂二级 `NO_CLEAR_IMPROVEMENT`；UA1–UA11/PA1–PA8 全 PASS；独立复核 85/85；→ §9.3/§10.7.1 机械登记"学习-vs-钉死解释消融"为下一步最优先（`NOT_SUPPORTED ∧ G_u < +0.001` 分支） | PASS |
| A2 | **f08ae6e 结果链（钉死 α 对照常量来源）** | 原始 JSON 重读三 run：B `bf322df`（test `0.5974422649550507`）；F_c `7d26918-rpp`（test `0.5947650925865714` / val `0.5798886656441761` / 逐 epoch val 轨迹 `[0.46759930720014714, 0.49016156687227064, 0.5156547237007145, 0.5480714746135289, 0.5798886656441761]` / gate_mean `[0.775494625, 0.224505265625]` / classification `VALID_NEGATIVE` / `alpha_final == 0.07101669907569885`）；`GAP = −0.001861278011441092`、`G_c = −0.002677172368479308` 逐位复算 | PASS |
| A3 | **seed2 学习参照（a40836e 结果链；L 与 α 轨迹来源）** | `20261004-0325-…-013e105-rpg`（commit `013e105`，dirty=false）：test `0.6055453782825184` / val `0.5895572066556973`；逐 epoch val `[0.4650886037939688, 0.4904911795680732, 0.5231212372261194, 0.5597976191334388, 0.5895572066556973]`；`prompt_report.json:alpha_final = 0.07101669907569885`；`grad_probe`（逐 epoch 首 batch）：α `[0.0, 0.016439981758594513, 0.037462275475263596, 0.050921376794576645, 0.06157483905553818]`、α_grad `[0.05110896751284599, 0.10020145773887634, 0.22529752552509308, 0.1570289582014084, 0.06819286942481995]`、gen_grad `[0.0, 0.025772185068553728, 0.08280410243250655, 0.11413855714947566, 0.12780488809679513]`；`rp_arm = VALID_POSITIVE`，M0+G1–G8 全 PASS；`Δtest = +0.008103113327467715`、`Δval = +0.008622497456618139`（逐位重推）；`prompt_report.json` sha256 `185df4d0…`；`a40836e` 结果提交原文 `REPLICATION_SUPPORTED`、复核 27/27 | PASS |
| A4 | **五 seed 与预算链（只读引用，不重推）** | five-seed：5 个 canonical seed 短预算配对逐 seed M0+G1–G8 全 PASS ∧ A 类 PASS；per-seed `Δtest`：s1 `+0.005394543882337954` / s2 `+0.008103113327467715` / s3 `+0.001456339669678841` / s4 `+0.004951998671071434` / s5 `+0.009646016187411788`，均值 `+0.005910402347593546`（std `0.0031538117454772883`，CI95 下界 > 0）；预算链 `+0.00810 → +0.00744 (10ep) → +0.00416 (20ep)`，`MONOTONE_NARROWING`；20-epoch `NOT_PERSIST`（`Δtest +0.004155967377889924`）/claim=true/`ABLATION_ELIGIBLE`；学习 shuffled `MECHANISM_FAIL` | PASS |
| A5 | **跨进程/跨 commit 确定性（支撑"历史 L/C_p 常量可在本分支流上复用"）** | five-seed：seed3–5 两次 pass 重跑**逐位相等**（含 `newtask.pt` 字节）；20-epoch：20-epoch 基线前 10 epoch 逐位等于 10-epoch、前 5 epoch 逐位等于 5-epoch（两臂同理；L 学习路径的跨分支逐位重放实测）；uncond：B 逐位重放 `79b5e07` + `G_c` 逐位等于 f08ae6e；f08ae6e：B 逐位重放 `79b5e07`、REP 6 项全 true | PASS |
| A6 | **机制实现的输入使用面（消融靶点定位）** | `NewTask.forward`（`8133d32:multitaskrec/model.py`，零改动）2 处使用 `dnn_input`：`projection_network`、`gate_network`；`ResidualPromptNewTask.forward` 追加第 3 处：`prompt_deltas(dnn_input)`（经 `prompt_generator`）。本实验的消融面 = **第 3 处的学习时序**（α + 生成器冻结 1 epoch），前三处（基头）保持正确输入与正确训练 | PASS |
| A7 | **六条学习臂的 α 轨迹与"离开 0 的边界"（本实验延迟定界规则的全部证据）** | 逐 run `grad_probe` 重读（§1.5-11 复算）：**全部六条**学习臂在 epoch-1 首 batch 探针处 `α == 0.0`（精确）且 `generator_grad_norm == 0.0`（精确），在 epoch-2 起点探针处 `α != 0` 且 `generator_grad_norm > 0`——即"α 离开构造值 0 发生在 epoch 1 期间"是**跨 seed、跨预算的普遍事实**（seed2 short/long/xlong + seed3/4/5 short）；seed2 边界比例：α(ep2 起点) = 23.15%·α_final、α(ep3 起点) = 52.75%、α(ep4) = 71.70%、α(ep5) = 86.70% | PASS |
| A8 | **uncond/f08ae6e 审计与复核完整性（只读引用）** | f08ae6e：预注册前核验 89/89、独立复核 101/101、run 间 diff 为空、无重跑；uncond：预核验 130/130、独立复核 85/85（首跑 83/85 为 checker-semantics 缺陷、已披露修正）、三臂前台单次 | PASS |
| A9 | **f08ae6e/20-epoch 对"学习 vs 钉死"可用性的既有裁定（本实验立项的必要性）** | f08ae6e §10.7.1/§10.7.2：钉死-α 与学习臂的对照**不可作因果证据**（训练动态混淆：生成器首步梯度 0 vs >0、α 轨迹、共适应）；uncond §10.7.3/§10.7.4：固定注入族（条件化/无条件/错配）三臂一致 `NO_CLEAR`，全线唯一正读数仍是学习 α 臂 ⇒ 机制证据唯一指向**学习动态**；因果检验需另立消融（即本实验） | PASS |
| A10 | **20-epoch 停止/删失形态（D 臂预期的删失口径）** | 20-epoch 与 5-epoch 短预算各臂 `best_epoch == epochs_run`（右删失、未早停；patience=2 在 5-epoch 下结构性可能但历史未触发）；D 臂按同口径记录 `best_epoch`/是否删失 | PASS |

### 1.3 seed / 预算 / 延迟界冻结（**先于任何本实验指标**）

**canonical seed 冻结为 seed2 = `1688723740`；预算冻结为 short（5-epoch / patience 2）；延迟冻结为 `DELAY_EPOCHS = 1`、`UNFREEZE_EPOCH = 2`（§2.2 机械定界规则）。** 理由：

1. **seed2 是唯一具备完整既有证据链的 seed**（A2–A5）：同 seed 的 5/10/20-epoch 配对、钉死参照头（sha `90ee06da…`）、内容寻址 Stage-1 产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f`、六条 α 轨迹证据（含 seed2 三条）齐备；其余 seed 只有 5-epoch 配对。
2. **short（5-epoch）是成本最低的完整协议点**：两臂实测墙钟 ≈150–250 s/run，总计 ≈8 min，单次前景窗口可控；预算效应不是本实验变量。
3. **short 上学习臂效用最强且已过线**（`ΔL = +0.0081 ≥ +0.0055`，A4）：延迟 1 epoch 后若仍无实质差，`DYNAMICS_NOT_SUPPORTED` 的结论最保守（在"最容易"的判定点上延迟即失效）。
4. **延迟界 = epoch 1（唯一机械可辩护的边界，§2.2）**：六条学习臂一致显示 α 的"离开构造值 0"发生在 epoch 1 期间（A7）；延迟恰好覆盖该 epoch、在 epoch 2 起点解冻 = 在"学习臂的 α 已离开 0 的第一个边界"恢复学习。**不扫描 k、不试多个 schedule、不事后选择**（§9）。
5. **双臂设计（B + D）**：因果主量 `ΔD` 只需 B 与 D 同分支新跑；L 与 C_p 为 hash 核验后的钉死常量（CC 逐位重推 + A5 跨分支确定性背书 + §3 端口逐字节守卫）——见 §3.5 的"必要性裁定"。无第三臂、无重跑、无条件性补跑。

### 1.4 机制审计（α 与生成器的学习耦合；延迟干预的靶点）

机制（`013e105:aliccp_benchmark/residual_prompt.py`，公式与结构化事实逐字沿用 f08ae6e §1.4）：

    P(x)        = tanh(W2·ReLU(W1·x + b1) + b2)            # 有界方向，来自条件输入 x
    g_eff(x)    = |alpha| · ||P(x)|| / sqrt(d)             # 逐样本有效门控
    delta_s(x)  = alpha · (||h_s|| / sqrt(d)) · P(x)       # 残差 = 门控 × 单位方向 × 基向量范数
    h_s'        = h_s + delta_s                             # 注入融合前三路（gen + spec_0 + spec_1）

**学习臂中 α 与生成器的耦合（本实验的靶点）**：
1. `prompt_gate` 为零初始化 `nn.Parameter`；α = 0 ⇒ `∂delta/∂θ_gen ∝ α = 0` ⇒ **首步生成器梯度精确为 0**（A3 实测 gen_grad(ep1 探针) = 0.0）；α 在 epoch 1 内被 Adam 更新离开 0 后，生成器梯度才逐步涌现（ep2 探针 0.0258 → ep5 探针 0.1278）。
2. **α≡0 的结构惰性引理（本实验的构造基石）**：若 α 恒为 0，则 `delta ≡ ±0.0` ⇒ 头部前向与参照头**逐位一致**（S3；G2/PA4 先例实测 `bit_identical=true, max_abs_diff=0.0`），且**生成器参数在任何优化器算术下都无法移动**（梯度为精确 ±0 的有限组合；Adam 在 m=v=0、grad=±0 下更新量 `lr·0/ε = 0` 精确）——因此"延迟 α"与"延迟整个 prompt 模块"在**模型动力学上同一**（不是两个可区分干预）；差异仅在优化器状态簿记（§2.3-3）。
3. **学习臂的学习期结构（epoch 1 的两步序列）**：step 1（α=0）：α.grad ≠ 0（实测 0.0511）且生成器梯度 = 0（状态创建 t=1、值为 0）；step 2 起：α ≠ 0、生成器梯度非零且 t=2 的偏差校正因子 ≈0.744（首步归一化偏小）。**D 臂在解冻时的优化器状态序列与之同构**（α 首梯度 t=1；生成器先零梯度步 t=1、再首非零梯度 t=2；§2.3-3）。
4. **本消融恰好改变**：D 相对学习臂——α 与生成器的学习期整体延迟 1 个 epoch（epoch 1 冻结于构造值；epoch 2 起点解冻）。**不改变**：α 语义（signed 标量、乘在范数标定后的方向前）、α 初值（0.0）、生成器初始化（隔离 RNG，同 seed 同路径）、范数标定 `scale = ||h_s||/√d`、注入流集合、`super().forward` 委托、损失、优化器类型/超参、训练协议、数据顺序、参照头。

### 1.5 预注册前参照核验（只读；先于本文件提交；`verify_delay_prerun.py`）

对固定产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f` + 参照头 + 五个历史 run 链（`20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07` 基线 / `20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg` 学习 correct【L】/ `20261005-0933-p2M-v500k-t1M-m1688723740-short-9d26bc8-rpgs` 学习 shuffled / `20261005-1029-p2M-v500k-t1M-m1688723740-short-7d26918-rpp` 钉死 correct【C_p】/ `20261005-1032-p2M-v500k-t1M-m1688723740-short-1c13841-rpps` 钉死 shuffled）+ 六条学习臂 α 轨迹导出 + 延迟定界规则复算做只读核验（**153/153 通过**，2026-10-05 预注册前实测；报告 `artifacts/aliccp_bench/audit/rp-delay/verify_delay_prerun_result.json`，sha256 `7e35d9e77e1b23c4cbeb16dcf91c63964f397a433dc07c9556b5cfc2c663b28b`；α 轨迹导出 `alpha_trajectories.json`；不写任何 run/stage1 产物、不训练、不重新评测）：

| 组 | 核对 | 结果 |
|---|---|---|
| 1 产物文件字节 | `backbone.pt`/`env_ids.pt`/`meta.json` sha256 == 钉死值（`cd7b0334…` / `4660be5a…` / `61a66d81…`） | **PASS** |
| 2 内容寻址 | `stage1_id` 重算 == 记录 == `s1-5c060b9c-m1688723740-e3-4e1b5c6f`；`backbone_sha256 == e5e7e610…`；`env_ids_sha256 == 5cd198f1…`；`fingerprint_sha256 == 5c060b9c…` | **PASS** |
| 3 历史 run 链（五 run） | 逐文件 sha256 == §1.5 钉死（§1.2-A2/A3 所列）；记录值钉死：五 run 的 test/val/逐 epoch val/gate_mean/best_epoch/epochs/patience/seed/run_id/variant/classification；L `VALID_POSITIVE`、`alpha_final == 0.07101669907569885` | **PASS** |
| 4 参照头 | `79b5e07/newtask.pt` sha256 == `90ee06da…`；`NewTask(80/64/[32,32])` strict 载入成功（M0 的 val 重算 `0.5809347091990792`/pred_std `0.005217193225189258` 由运行期门禁与复核脚本在真实数据上执行） | **PASS** |
| 5 三常量与半程门 | 由原始记录逐位重推 `ΔL == +0.008103113327467715` ∧ `ΔP == −0.002677172368479308` ∧ `GAP == +0.010780285695947023` ∧ 半程门 `+0.0027129704794942035` | **PASS** |
| 6 基线 epoch-1 钉死（FROZEN_ID 常量） | `79b5e07` per_epoch[0]：train_loss == `0.08193688414408826` ∧ val == `0.4645711559431739` | **PASS** |
| 7 **α 轨迹导出（六条学习臂；定界规则证据）** | 逐 run 重读 `grad_probe`；断言六条全部满足：probe(ep1).α == 0.0 ∧ probe(ep1).gen_grad == 0.0 ∧ probe(ep2).α != 0 ∧ probe(ep2).gen_grad > 0；导出全量轨迹 JSON 至 audit 目录（含 §1.2-A7 的边界比例） | **PASS** |
| 8 **延迟定界规则复算** | 按 §2.2 冻结规则机械复算 `k = 1`、`UNFREEZE_EPOCH = 2`；选项 (A) 不可行性证据（逐 epoch 采样粒度 = 1 点/epoch + 终值；无 per-step 日志/逐 epoch checkpoint）随报告记录 | **PASS** |
| 9 导入纯性与数据集 | 导入探针在预核验首次运行时按 `rp_delay` 存在性记录（C1 时按设计尚不存在 ⇒ 记录 pending；C2 后由单元测试 I8 全覆盖）；全局 torch/Python RNG 端点不变；dataset junction 三切分可读 | **PASS** |

- **产物复用决议**：按"先验证、后复用，否则确定性重生成"规程——全部通过 ⇒ **复用**该产物（不重训 Stage-1）；产物、参照头、五个历史 run 均自 uncond worktree **只读复制**入本 worktree（逐文件 sha256 与源相等，复制后由本核验复验；§1.2 已完成复制复核）；dataset 经 junction 复用 main tree（只读、gitignore）。
- 钉死参照常量（后续判定全部引用）：`BASELINE_AUC_TEST = 0.5974422649550507`、`BASELINE_AUC_VAL = 0.5809347091990792`、`REFERENCE_PRED_STD = 0.005217193225189258`、`BASELINE_EP1_TRAIN_LOSS = 0.08193688414408826`、`BASELINE_EP1_VAL = 0.4645711559431739`；`LEARNABLE_AUC_TEST = 0.6055453782825184`、`LEARNABLE_AUC_VAL = 0.5895572066556973`、`DELTA_TEST_LEARNABLE = +0.008103113327467715`、`DELTA_VAL_LEARNABLE = +0.008622497456618139`、`LEARNABLE_ALPHA_FINAL = 0.07101669907569885`；`PINNED_CORRECT_AUC_TEST = 0.5947650925865714`、`PINNED_CORRECT_AUC_VAL = 0.5798886656441761`、`DELTA_TEST_PINNED = −0.002677172368479308`、`DELTA_VAL_PINNED = −0.00104604355490312`；`GAP = +0.010780285695947023`、`HALF_GAP_TARGET = +0.0027129704794942035`。

### 1.6 审计结论（立项裁定的确认 + 单因子陈述）

1. **uncond 的机械选择成立**（A1）：`NOT_SUPPORTED ∧ G_u < +0.001` ⇒ 学习-vs-钉死解释消融为登记下一步；本实验为其预注册化执行。
2. **单因子陈述（本实验实际可主张的范围）**：D 相对 **B** 的对照是**单干预**（prompt 模块学习期延迟 1 epoch；冻结期与 B 逐位同流，§2.3）；D 相对 **L** 的差异**不止一个因子**（epoch-2 边界的头部权重状态、α 边界值 0.0 vs 0.01644、4 vs 5 个学习 epoch 及其全部下游），**不构成"延迟本身"的因果对照**——只作描述统计（gap_closure 等，§5.3/§5.4 读法）；D 相对 **C_p** 的差异同时含 α 可学习性与头部 warmup 两因子，同不主张因果。
3. 历史判定原样保留（§0 纪律声明）；本实验不重判任何历史结论。

---

## 2. 消融设计（唯一处理臂 D；延迟界与构造先于实现写死）

### 2.1 选项裁定：(A) α 轨迹精确回放 **不可行** ⇒ 选定 (B) 单一延迟解冻

**裁定规则（先于结果写死）**：仅当存在"足够细粒度、可独立核验、无需插值/调参即可定义"的历史 α 轨迹时选 (A)；否则选 (B)。**裁定：选 (B)。** 依据（全部来自 §1.2-A3/A7 的已提交证据）：

1. **采样粒度不足**：全部六条学习臂的 α 轨迹记录为 **1 点/epoch**（每 epoch 首 batch 的 `grad_probe`）+ 训练终值（`alpha_final`）——seed2 5-epoch 共 6 个时点（0, 0, 0.01644, 0.03746, 0.05092, 0.06157）+ 终值 0.07102；**没有任何 per-step/per-batch α 记录**（`grad_probe` 只记首 batch；层日志、逐 epoch checkpoint、优化器状态快照均不存在——runner 仅保存 best `state_dict`）。
2. **epoch 内 α 运动是一阶量**：seed2 的 epoch 2 内 α 从 0.01644 升至 0.03746（运动量 +0.0210，为 epoch-2→3 边界读数的全部来源）；任何"回放"必须先发明 epoch 内的 α(step) 行为（零阶保持/线性插值/其它）——**均属插值/约定发明**，直接违反 (A) 的前提（"without interpolation/tuning"）。
3. **零阶保持本身自相矛盾**：记录值显示 epoch 1 的边界读数为 0.0，但学习臂的 α 在 epoch 1 **期间**离开 0（这正是 A7 的普遍事实）——按记录值做 ZOH 回放会把 epoch 1 的 α 定为恒 0，与记录所证明的真实行为矛盾；"回放"从第一段起就不是轨迹。
4. **回放无法表达耦合**：假说对象是 α+生成器的**共适应**；把 α 换成从另一条轨迹抄来的 schedule，生成器将对"别人的 schedule"共适应——即便可定义，也引入不可分离的混杂（登记条目中 (B) 的存在理由）。
5. 因此按裁定规则选 **(B)：单一延迟解冻 schedule**——延迟长度由 §2.2 的机械规则唯一确定（`k = 1`），**不试多个 k、不做 α 扫描、不事后选择**。

### 2.2 延迟定界规则（**先于实现写死**；机械、唯一、跨 run 普遍）

**规则**：`DELAY_EPOCHS k :=` 学习臂记录中"α 仍精确等于其构造值 0.0 的 epoch 起始探针"所覆盖的 epoch 数；`UNFREEZE_EPOCH := k + 1`。

**证据（A7；六条学习臂全量重读）**：

| run | seed | 预算 | probe(ep1).α | probe(ep1).gen_grad | probe(ep2).α | probe(ep2).gen_grad |
|---|---|---|---|---|---|---|
| `013e105-rpg` | 1688723740 | short | **0.0** | **0.0** | +0.016439981758594513 | 0.025772185068553728 |
| `f2ccec2-rpg` | 1688723740 | long(10) | **0.0** | **0.0** | +0.016439981758594513 | 0.025772185068553728 |
| `eafc336-rpg` | 1688723740 | xlong(20) | **0.0** | **0.0** | +0.016439981758594513 | 0.025772185068553728 |
| `c17b100-rpg` | 1688738016 | short | **0.0** | **0.0** | −0.046634916216135025 | 0.025634397865251665 |
| `c17b100-rpg` | 1688749593 | short | **0.0** | **0.0** | +0.019704077392816544 | 0.023493472415480506 |
| `c17b100-rpg` | 1688762746 | short | **0.0** | **0.0** | −0.026758631691336632 | 0.042282424045870896 |

⇒ 在**全部六条**已提交学习臂中，α 恰在构造值（0.0）上停留通过 epoch-1 探针、并在 epoch 1 期间离开 0（epoch-2 边界探针非零、生成器首现非零梯度）⇒ **`k = 1`、`UNFREEZE_EPOCH = 2`**（冻结期 = 恰好 1000 个训练步；解冻在 epoch 2 第一个 batch 之前）。

**为何不是其它机械候选（先于结果说明，不事后改）**：候选"α 首次越过终值一半的边界"（seed2 于 epoch-3 起点达 52.75%）需引入任意的 50% 阈值且为 seed2 特异（seed3 在同一边界已达 95%）、并把学习窗口砍到 3 个 epoch（与"预算收窄"效应混淆、负结果不可解释）；候选"按 val 拐点"在噪声轨迹上无机械定义。**唯一跨 run 普遍且无自由参数的边界 = 离开构造值的边界 ⇒ k = 1**（最小延迟，恰好移出"离开 0"这一个 epoch）。

### 2.3 延迟解冻头构造（`DelayedUnfreezeResidualPromptNewTask`；**先于实现写死**）

- **继承**：`DelayedUnfreezeResidualPromptNewTask(ResidualPromptNewTask)`——共享参数、生成器结构/初始化（隔离 CPU RNG 内，逐位同学习臂）、`prompt_deltas`/`forward`/`super().forward` 委托/α 语义/范数标定全部**逐字继承**（零重写）。
- **冻结实现（唯一重写 = 构造期的可学习性门控）**：
  1. `prompt_gate` 仍为 **`nn.Parameter(torch.zeros(()))`**（关键：与学习臂**同一参数身份**、同一 state_dict 键、同一优化器覆盖、同一新参数计数 2385），但构造时 `requires_grad_(False)`；
  2. 生成器 4 个参数张量构造后 `requires_grad_(False)`；
  3. `unfreeze_prompt_module(newtask, *, audit)`（模块级函数，runner 于 epoch 2 起点调用恰一次）：校验（α 精确 == 0.0 ∧ 生成器参数逐位等于构造时 ∧ 未重复解冻 ∧ epoch == `UNFREEZE_EPOCH`）→ 置 α 与全部生成器参数 `requires_grad_(True)` → 记录解冻事件。
- **梯度门控 vs"零梯度步进"的裁定（先于结果写死）**：若不门控生成器（保持 `requires_grad=True`），α ≡ 0 使生成器梯度为**精确 0**（引理 1.4-2）⇒ 参数同样不动，但 Adam 的 `t` 计数在冻结期空转 1000 步，使解冻后**最初 ~一个 epoch** 的偏差校正因子失真（首非零梯度步的归一化被放大至 ≈3.16× 且需 ~1000 步回正）——这是对"重启学习动力学"的**非受控扰动**。门控实现使解冻时生成器的 Adam 状态**全新**，首非零梯度序列（先 1 步零梯度 t=1、再首非零 t=2）与学习臂自身的 epoch-1 前两步**同构**（引理 1.4-3）；且由引理 1.4-2，门控与不门控在**冻结期模型行为上无任何差异**（都不动）。故**裁定：门控实现**；该差异仅属优化器状态簿记，随 run 落盘披露（§7）。
- **α 解冻初值 = 0.0（其构造值/学习臂同款初值）**；**不**warm-start 到历史边界值 0.01644（裁定：解冻语义 = 从冻结值释放学习；warm-start 会把历史 run 在**不同头部状态**下产生的特定取值注入当前流，属不可机械辩护的混合干预；登记为已考虑并否决的变体）。
- **行为等价链（TDD 钉死，§4）**：① 冻结期（α=0）任意前向与同共享权重参照头**逐位相等**；② 冻结期反传：α/生成器 `.grad is None`、头部各参数梯度与参照头**逐位相等**；③ 解冻后：α 梯度非零、生成器在 α=0 的那一步梯度精确 0（随后恢复非零）；④ 冻结期任意"无梯度 step"下 α/生成器逐位不变；⑤ 构造的共享参数/RNG 端点/生成器初始化与学习臂类在同 RNG 现场下**逐位相同**。
- **参数/键口径（与学习臂逐项相同）**：可训练参数 5 键（α + 生成器 4 张量）计 2385（解冻后名义全量）；冻结期有效可学 = 生成器 4 张量（α 亦被门控，但两者在 α=0 下皆不动——引理 1.4-2）；state_dict 新增键恰 5；头部 8129；优化器 = `newtask.parameters()` 全量（16 张量，与学习臂**同一集合**，对象同一性记录）。无新增 buffer、无参数预算差异（**与学习臂 2385 == 2385**）。
- **单因子陈述**：D 相对 B 的唯一差异 = "prompt 模块在 epoch 2–5 学习（其中 epoch 2 起 α 从 0 增长）"——冻结期（epoch 1）两臂**逐位同流**（§2.4）。

### 2.4 冻结期恒等（为什么 epoch 1 可逐位核验）

由引理 1.4-2（α=0 ⇒ delta 精确 ±0 ⇒ 前向逐位同参照头；生成器梯度精确 0）与结构事实（新模块构造不改 RNG 端点、stage2 头部 MLP **无 dropout**、DataLoader `shuffle=False`、无其它随机算子）⇒ D 臂 epoch 1 的**整条训练流**（每个 batch 的前向/损失/梯度/更新）与配对基线 B 逐位相同 ⇒ 可运行期核验的强不变量：

- `D.per_epoch[0].train_loss == B.per_epoch[0].train_loss`（逐位；== 钉死常量 `0.08193688414408826`）；
- `D.per_epoch[0].val_auc_bsi == B.per_epoch[0].val_auc_bsi`（逐位；== 钉死常量 `0.4645711559431739`）；
- 冻结期 α 恒 0.0、生成器参数逐位不变（解冻记录内的构造/解冻两点 sha 相等）。

第三条为**运行期**记录；前两条为 `FROZEN_ID` 前置条件（§5.1；分析器与独立复核双重核验）。此即"延迟被严格执行"的端到端证明；任一失配 ⇒ `INVALID/FROZEN_PHASE_IDENTITY_FAILED`。

### 2.5 可识别性与残留混杂（预声明）

**可识别（本实验能站住的因果陈述）**：

1. **D vs B 是单干预、同分支、同流、同 seed 的配对对照**：冻结期逐位同流（§2.4 三重核验）⇒ 两臂的全部差异 = prompt 模块在 epoch 2–5 的学习；`ΔD` 因此是"延迟 1 epoch 的学习动力学"相对基线的直接因果读数。
2. **D 的构造复刻学习臂的学习期起点语义**：同一 α 初值 0.0、同一生成器初始化、同一优化器覆盖；解冻时优化器状态序列与学习臂 epoch-1 前两步同构（引理 1.4-3；§2.3 裁定）⇒ "学习期重启"的语义边界干净。
3. **α≡0 惰性引理**使干预不可再分解为"延迟 α"与"延迟生成器"两个变体（§1.4-2）——不存在被遗漏的中间干预。

**残留混杂（必须并读，不得越界）**：

1. **D vs L 不是单因子对照**：epoch-2 边界处头部权重状态不同（B 流 warmup vs L 的含注入流）、α 边界值不同（0.0 vs 0.01644）、学习 epoch 数不同（4 vs 5）及其全部下游——`gap_closure`/`half_gap_met` 按 §5.3 仅作**描述统计**；不得表述为"延迟多少导致差距回收多少"的因果量。
2. **D vs C_p 含两因子**（α 可学习性 + 头部 warmup），不作因果陈述。
3. **单 seed、单协议点、每臂 1 run**：无 run-to-run 噪声估计（A5 沿用）；`+0.001` 为分类标签而非显著性强断言（test 1M 前缀 BSI AUC 的 SE ≈ 0.0025 @AUC0.7 ⇒ ≈0.4×SE）。
4. **删失/停止**：5-epoch 短预算下历史两臂均右删失（best_epoch = 5/5，A10）；D 若同删失，headroom 按 §5.5 机械记录；不早停即为预期形态，不解读为"仍在上升"的因果证据。
5. **学习期长度差异**（4 vs 5 epochs）与**预算收窄效应**（A4：`+0.0081 → +0.0042`）在 NO_CLEAR/负结果时**无法在本实验内分离**——读法必须以"最小延迟即失效/仍成立"为主体，不推及"任意延迟"。
6. **判定依赖单一汇总统计**（test 1M 前缀 BSI AUC 差值）；判定取 run 记录值；正确性依赖代码路径守卫 + REP_B/FROZEN_ID 双重逐位 + 独立复核。
7. **B4 继承 FAIL**（Stage-1 共享语义；臂无关）；`hard_pass` 不参与有效性定义（同先例）。
8. **单数据集（AliCCP）、单 seed**：结论不推及 CensusIncome / ByteRec、不推及其它 seed / 预算点。
9. **本实验不是"更优方法"主张**；是机制解释消融的一次性判定。

---

## 3. 实现面（最小；TDD；等价证明）与适配清单

| 编号 | 文件 | 处置 | 守卫 |
|---|---|---|---|
| A1 | `aliccp_benchmark/residual_prompt.py` | **逐字节移植** `f08ae6e:` 版（blob `674213f6…`；LF sha256 `b3b93b3a…`）。**零文本适配** | 守卫测试：工作树 == 钉死 blob 逐字节 + LF sha + `git hash-object` == blob + Census 祖本 AST 钉死（`TestMechanismPin` 移植） |
| A2 | `aliccp_benchmark/rp_pinned.py` | **逐字节移植** `f08ae6e:` 版（blob `432fa9bb…`；LF sha256 `ba23bf47…`）。**零文本适配**（其 shuffled 类/错排器为**休眠代码**：CLI 不暴露、本实验不接线、不运行） | 守卫测试：工作树 == 钉死 blob 逐字节 + LF sha + `git hash-object` == blob |
| A3 | `aliccp_benchmark/bench.py` | `f08ae6e:` 版（blob `b676a797…`）**+ 文档化最小适配**：import 段（增 `rp_delay as RPD`）；`run_stage2` 段（新增 delay 分支：构造/审计/解冻钩子/臂级判定/prompt_doc 的 `delay` 块；run_id 回退路径经 `RPD.arm_suffix_for`）；**其余模块级函数（含 `evaluate_newtask`）与全部控制流逐字不变** | 守卫测试：重建等式（钉死文本 + 恰好替换 == 工作树）＋ AST 差异集合恰为 `{run_stage2}`＋ tiny 端到端 delay 路径语义测试 |
| A4 | `run_aliccp_benchmark.py` | `f08ae6e:` 版（blob `74aaaf05…`）**+ 恰好 3 行适配**：import 行（增 `rp_delay as RPD`）；`--variant` choices 行（增 `residual-prompt-delay`；CLI 仍暴露 pinned-shuffled 原样，本实验不运行）；臂后缀行（改经 `RPD.arm_suffix_for` 分派） | 守卫测试：行级替换重建等式 + 新 LF sha 钉死 |
| A5 | `aliccp_benchmark/rp_delay.py`（新） | 消融本体：`DelayedUnfreezeResidualPromptNewTask`（§2.3）、`unfreeze_prompt_module`（模块级、恰一次、校验齐全）、`DelayPromptAudit`（继承 `RP.PromptAudit`：G1/G2 记录 + 冻结期结构记录 + 解冻事件记录 + 逐 epoch 梯度探针沿用）、`arm_suffix_for`、`delay_mechanism_gates`（DD1–DD9）、`delay_arm_verdict`、`analyze_runs`（只读两 run + 历史常量 → §5 全量判定）+ `python -m` CLI（输出 JSON） | 守卫测试（本分支 `test_residual_prompt_delay.py`）：§4 全部不变量 + 判定树真值表/边界 + 冻结/解冻行为 + 分析器夹具 |
| A6 | `aliccp_benchmark/tests/test_residual_prompt.py` | `f08ae6e:` 同名测试（blob `2b86d39d…`；LF sha256 `55cef9e5…`）**恰 4 处**适配：(a) 模块 docstring；(b) `DOC_PATH` → 本文件；(c) `WHITELIST` → 本分支 11 项（逐项集合钉死）；(d) `TestPreregConstants.test_prereg_doc_tokens_and_summary_ledger` 的 token 表 → 本文件 token 表。**其余逐类/逐方法/逐模块级语句不变** | 守卫测试：重建等式（docstring + `DOC_PATH` + `WHITELIST` + 该 1 个方法段替换）+ AST 差异集合 == 恰 1 个方法 + 新 LF sha 钉死 |
| A7 | `aliccp_benchmark/tests/test_residual_prompt_delay.py`（新） | 本分支守卫 + 延迟/接线/分析器测试（§4）；含 A1–A4/A6 的重建等式与 AST 差异守卫 | 本文件即为守卫 |
| A8 | `verify_delay_prerun.py`（预注册前完整性核验；§1.5）与 `verify_rp_delay.py`（运行后独立复核） | 只读脚本；**跟踪入库**（沿用 five-seed/alpha-pinned/uncond 先例；两处 WHITELIST 副本同步包含二者）；报告 JSON 落 `artifacts/aliccp_bench/audit/rp-delay/` 与 D run 目录（gitignore）。前置脚本自 `180d8d5:verify_uncond_prerun.py` 形态移植（产物 + 参考头 + 五 run 链 + α 轨迹导出 + 定界复算）；复核脚本自 `180d8d5:verify_rp_uncond.py` 形态移植（两 run 形态 B/D；记录一致性口径沿用） | 前置核验 §1.5 全通过（§10.0 回填）；复核脚本 §10.5 |

**被禁止的适配**：机制公式/初始化/门控语义/注入点/`BOUND_TOL`/U1 阈值/判定树/阈值/延迟界（`k=1`）/解冻实现裁定；`protocol.py`/`metrics.py`/`multitaskrec/*`/`config.py`——零改动（静态守卫：`git diff --name-only 8133d32` ⊆ 白名单）。

**等价性证明口径（三层，全部由测试守卫执行）**：
1. **重建等式（文本级）**：钉死 blob（f08ae6e）+ 恰好替换 == 工作树（bench/CLI/移植测试）；机制与 rp_pinned **逐字节**。
2. **机制 pin**：`residual_prompt.py` 逐字节 == 674213f6 + Census 祖本 AST 钉死；`rp_pinned.py` 逐字节 == 432fa9bb。
3. **行为级**：① `shuffler=None` 分派恒等（baseline/学习路径逐位不变）；② 冻结期（α=0）前向/梯度恒等参照头（§2.3 等价链）；③ 解冻后 α 梯度非零、生成器梯度序列与学习臂同构；④ 导入 `rp_delay` 不消耗/改变任何 RNG 端点；⑤ D 类与学习臂类在同 RNG 现场下共享参数与生成器初始化**逐位相同**（"same Prompt Generator"的构造级证明）。

### 3.5 第三臂（学习臂重放）必要性裁定（**先于结果，冻结**）

**裁定：不设第三臂；L 以 hash 核验后的已提交 run 作上下文常量。** 依据（本实验的身份链闭合论证）：

1. **本分支流身份**由 **REP_B** 运行期证明（B 与 `79b5e07` 逐位重放，含 `newtask.pt` sha）——stage2 训练流（数据顺序、RNG、优化器轨迹）在本分支与历史链**同一**；
2. **学习路径的跨分支逐位确定性**已由已提交证据背书（A5：20-epoch 学习臂前 10 epoch 逐位等于 10-epoch、前 5 逐位等于 5-epoch；five-seed 双 pass 逐位；本实验的 D 臂不触碰该路径的代码——`residual_prompt.py` 逐字节钉死、学习臂分派分支逐字不变）；
3. **D vs L 不存在任何逐位配对关系**（D 在 epoch 2 从 α=0 重启，按设计即与 L 的任何子轨迹不等）——重放 L 不会改变 `ΔL` 这个量（逐位确定性下重放结果 == 历史记录），只增加一次验证；该验证已由 1+2 的结构论证 + `FROZEN_ID`/REP_B 的运行期逐位证据覆盖；
4. 故第三臂**非必要**（任务纪律：仅在预注册认定必要才重放）⇒ **两臂、恰两次 run**（§6）；L 与 C_p 的数值以 CC（§5.1）逐位重推 + 文件 sha 钉死进入判定。

---

## 4. 不变量（由 `test_residual_prompt_delay.py` 与移植测试强制，不接受人工目测）

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 基线/学习臂逐位不变：`shuffler=None` 分派 ≡ 钉死调用；学习臂端到端记录键集与钉死一致 | 移植测试 + bench 重建等式 + 真实 REP_B |
| I2 | 适配逐字守卫：A1–A4/A6 的字节/重建等式钉死 | `git show` + sha256(LF) + 段级重建等式 |
| I3 | 学习机制语义不变：α 初值 0、恒等/界/活性带测试全绿（移植测试原样通过） | 移植测试 |
| I4 | **D 构造身份**：共享参数/buffer 逐位同参照头 ∧ RNG 端点一致 ∧ 新增键恰 5（与学习臂同集）∧ α == 0.0 精确 ∧ α 为 `nn.Parameter` ∧ α 与生成器 4 张量 `requires_grad == False` ∧ 生成器初始化与学习臂类同现场逐位同 | 单元测试 + 运行期 DD1/DD2 |
| I5 | **冻结期行为恒等**：任意输入（含 `zeros_like`/极端值/多 batch 大小）下 D 前向 == 参照头前向**逐位**；冻结期反传后 α/生成器 `.grad is None` ∧ 头部各参数梯度 == 参照头逐位；优化器 step 后 α/生成器逐位不变 | 单元测试（CPU tiny）+ 运行期 G2/init_forward |
| I6 | **解冻语义**：`unfreeze_prompt_module` 恰一次（第二次调用 raise）；epoch != 2 raise；解冻时 α == 0.0 精确 ∧ 生成器逐位不变（构造 sha == 解冻 sha）；解冻后 α/生成器 `requires_grad == True` ∧ 优化器覆盖含 α 与生成器（对象同一性） | 单元测试 + 运行期 DD3 |
| I7 | **解冻后动力学同构**：解冻后首 batch 反传：α.grad ≠ 0 ∧ 生成器梯度精确 0（α=0 步）；α 更新一步后生成器梯度非零有限；α 更新一步（手动）后生成器在优化器 step 下可移动；未更新时生成器逐位不变 | 单元测试 + 运行期 DD4/DD11 探针 |
| I8 | **RNG 隔离与导入纯性**：构造/导入 `rp_delay` 不改变全局 torch/Python RNG 端点；`isolated_cpu_rng` 语义不变 | 单元测试 + §1.5-9 |
| I9 | state_dict/checkpoint：D 新增键恰 5（α + 生成器 4）；α 值经 checkpoint 往返逐位保持；参数计数 2385/8129 | 单元测试 + 复核脚本独立读取 |
| I10 | 判定树/边界：§5.4 全部标签真值表；`ΔD` 恰 `+0.001` ⇒ `SUPPORTED`（闭），`−1e-12` ⇒ `NOT_SUPPORTED`；`Δval_D` 恰 0 ⇒ 验证不一致；半程门恰 `+0.0027129704794942035` ⇒ `half_gap_met`（闭）；二级分类边界 `+0.001`/`−0.02`（闭端）；每条 INVALID 子原因可达；DD1–DD9 每条 FAIL 可达（夹具） | 边界/真值表测试（CPU 夹具） |
| I11 | 分析器只读：不改任何 run 产物；ID/REP_B/FROZEN_ID/DD/CC 任一 mismatch ⇒ `INVALID`；输出键齐全（§5 全部判定 + 全部分量 + 描述量）；静态守卫：相对 `8133d32` 的全部跟踪改动 ⊆ 白名单；`protocol.py`/`metrics.py`/`multitaskrec/*`/`config.py` 零 diff；预注册文档 token 钉死（seed、stage1_id、`DELAY_EPOCHS`、`UNFREEZE_EPOCH`、`ΔL`/`ΔP`/`GAP`/半程门、阈值、判定标签、重放靶 sha、适配清单 blob）；两处 WHITELIST 副本逐项相等 | 夹具端到端 + 篡改 fixture + `git diff --name-only` + 文档文本断言 + 双副本交叉测试 |
| I12 | 端口完整性：`rp_pinned.py` 逐字节 == f08ae6e blob；`residual_prompt.py` 逐字节 == 013e105 blob；L 的 `grad_probe` 六 run 导出与 §2.2 证据表逐位一致 | 守卫测试（§3）+ 预核验脚本 |

CPU 极小夹具不构成任何性能证据，只验证语义与接线。

---

## 5. 预注册判据（看到结果前写死；看到结果后不得修改）

对照定义：**B = 本实验新跑的配对基线臂**；**D = 本实验新跑的延迟解冻臂**。`ΔD := test(D) − test(B)`；`Δval_D := val(D) − val(B)`；钉死常量 `ΔL = +0.008103113327467715`、`ΔP = −0.002677172368479308`、`GAP = ΔL − ΔP = +0.010780285695947023`；`gap_closure := (ΔD − ΔP)/GAP`；`half_gap_met := ΔD ≥ ΔP + 0.5·GAP`。

### 5.1 前置条件（全部必须为真；任一假 ⇒ `INVALID`，不解读效用）

| 编号 | 判据 | 落盘 |
|---|---|---|
| **ID**（identity） | 两 run：`stage1_id` 相同且 == `s1-5c060b9c-m1688723740-e3-4e1b5c6f`；`model_seed == 1688723740`；`epochs==5` ∧ `patience==2` ∧ `tag=="short"`；`dirty==false`（两 run 记录时）；B `variant=="baseline"` 且 run_id 无后缀；D `variant=="residual-prompt-delay"` 且 run_id 以 `-rpd` 结尾；D 记录参照路径 == 钉死路径且文件 sha256 == `90ee06da…`；两 run `stage1_id` 的 backbone/env/fingerprint sha 与钉死值一致 | `rp_delay_compare.json:identity` |
| **REP_B**（基线复现） | B 的逐 epoch val 轨迹（×5）、逐 epoch train_loss（×5）、best_epoch、best_val、test、gate_mean 与 `79b5e07` 记录**逐位相等** 且 B 的 `newtask.pt` sha256 == `90ee06da…`（A5 确定性预期成立；失败 ⇒ 环境偏离历史链，对照组不可比） | 同上 `reproduction_b` |
| **FROZEN_ID**（冻结期恒等；本实验特有） | D 的 epoch-1 `train_loss == 0.08193688414408826` 且 `val_auc_bsi == 0.4645711559431739`（**逐位**；同时与 B 的 epoch-1 逐位相等——三方恒等）；D 的 `delay` 块：`alpha_at_construction == 0.0` ∧ `alpha_at_unfreeze == 0.0`（精确）∧ `generator_sha_at_construction == generator_sha_at_unfreeze` ∧ `unfreeze_epoch == 2` ∧ `delay_epochs == 1` ∧ 解冻后 α/生成器 `requires_grad` 全 True（失败 ⇒ 延迟未被严格执行，全实验作废） | 同上 `frozen_phase_identity` |
| **A**（协议） | 两 run 的 A1/A2/A4/A5/A6 全 PASS（A3 SKIP 视为通过）；任一 FAIL ⇒ 该 run 比较作废 | 两 run `gate_report.json` |
| **DD**（结构机制门禁） | D 的 `rp_arm`：**DD1–DD9 全 PASS**（§5.2；由本分析器对记录 `observed` 值机械重算，不信任记录布尔） | D run `metrics.json:rp_arm` |
| **CC**（对照链与钉死来源） | 五个历史 run（`79b5e07`/`013e105-rpg`/`9d26bc8-rpgs`/`7d26918-rpp`/`1c13841-rpps`）的记录值 == §1.5 钉死常量（含 variant/classification/run_id；复核脚本执行文件 sha 核对）；L 的 `alpha_final == 0.07101669907569885`（精确）∧ `prompt_report.json` sha256 == `185df4d0…`；**`ΔL`/`ΔP` 由原始记录逐位重推 == 钉死值**；六条学习臂 α 轨迹导出 == §2.2 证据表逐位 | 同上 `comparator` |

### 5.2 D 臂结构机制门禁（DD1–DD9；**先于结果写死**）

| 编号 | 判据 | 口径 |
|---|---|---|
| **DD1 构造身份（G1 等价）** | 共享参数与 buffer 逐位同参照头 ∧ RNG 端点一致 ∧ 新增键恰 5（`prompt_gate` + 生成器 4 键） | 运行期 `DelayPromptAudit`（继承构造审计）+ 单元测试（I4） |
| **DD2 保守初值（G2 等价）** | `alpha_at_construction == 0.0` 精确 ∧ 首 batch 前向与参照头**逐位相等**（`bit_identical == true` ∧ `max_abs_diff == 0.0`） | 运行期首 batch 探针 + 单元测试（I5） |
| **DD3 延迟完整性（结构核心）** | 构造期：α 为 `nn.Parameter` 且 `requires_grad == False` ∧ 生成器 4 张量 `requires_grad == False` ∧ 优化器覆盖全部 `named_parameters()`（16 张量，含 α）∧ α 不在任何 `param_groups` 之外；解冻期：`unfreeze_epoch == 2` ∧ `delay_epochs == 1` ∧ α 精确 == 0.0 ∧ 生成器构造/解冻两点 sha 相等 ∧ 解冻后 α + 生成器 `requires_grad` 全 True ∧ 解冻恰一次 ∧ RNG 端点不变 | 运行期记录 + 单元测试（I6） |
| **DD4 学习期重启活性（学习臂 G3 的位移形态）** | epoch-2 首 batch 探针：`alpha_grad_norm` 非 None 且 != 0（解冻即接上学梯度）∧ 同探针 `generator_grad_norm == 0.0`（α=0 步，结构预期）；末 epoch 探针：`alpha != 0` ∧ `generator_grad_norm > 0`；`alpha_final != 0` | 运行期梯度探针 + 单元测试（I7） |
| **DD5 范数界（G4）** | 逐流 `ratio_max ≤ \|alpha_final\| + 1e-6` | val 诊断（fp64 流式） |
| **DD6 活性带（G5）** | 逐流 `ratio_mean ∈ [0.005, 0.5]`（含端点）；`band_low ⇒ MECHANISM_FAIL/MECHANISM_SILENT`、`band_high ⇒ MECHANISM_FAIL/MECHANISM_OVER_PERTURB`（**a priori 说明**：4-epoch 延迟重启下若有效扰动跌出该带，因果问题不再被有效提出 ⇒ `MECHANISM_FAIL` ⇒ 实验 `INVALID`；带数值与学习臂 G5 完全相同，不放宽） | val 诊断 |
| **DD7 门控非退化（G6）** | `geff_std > 0` ∧ `geff_max > 0` ∧ `geff_min >= 0` | val 诊断 |
| **DD8 无坍缩（G7）** | `pred_std > 0` ∧ `pred_std ≥ 0.5 × ref_pred_std` | val 诊断 + 参照头 |
| **DD9 参数预算（G8 等价）** | 5 exact `prompt_` 键 ∧ `new_params_total == 2385` ∧ `head_params == 8129`（**与学习臂逐项相同**） | `param_report` + checkpoint 读取 |
| **M0 参照身份** | `ref_val_auc == 0.5809347091990792` ∧ `ref_pred_std == 0.005217193225189258`（`REFERENCE_IDENTITY_TOL = 1e-9`） | 参照头重算 |

**分类映射（先于结果写死）**：`not DD1 or not DD2 or not DD3 or not DD9` ⇒ `MECHANISM_FAIL / INVALID_IMPLEMENTATION`；`not DD4` ⇒ `MECHANISM_FAIL / MECHANISM_INACTIVE`；`band_low` ⇒ `MECHANISM_FAIL / MECHANISM_SILENT`；`band_high` ⇒ `MECHANISM_FAIL / MECHANISM_OVER_PERTURB`；`not DD7` ⇒ `MECHANISM_FAIL / GATE_DEGENERATE`；`not DD8` ⇒ `MECHANISM_FAIL / PREDICTION_COLLAPSE`；`not DD5` ⇒ `MECHANISM_FAIL / NORM_BOUND_VIOLATION`；`not M0` ⇒ `MECHANISM_FAIL / REFERENCE_IDENTITY`；protocol 失败 ⇒ `MECHANISM_FAIL / PROTOCOL_INVALID`；否则按 U1（`ΔD ≥ +0.0055` vs 基线常量）/U2（`Δval_D > 0`）给 `VALID_POSITIVE`/`VALID_NEGATIVE`（in-run，class C 诊断；阈值口径与学习臂相同）。

### 5.3 效用分量（判定的原子量；全部报告）

| 量 | 定义 | 预声明口径 |
|---|---|---|
| `ΔD` | `test(D) − test(B)` | **因果主量**；实质为正：`ΔD ≥ +0.001`（`MATERIAL_DELTA`，闭） |
| `Δval_D` | `val(D) − val(B)` | 验证方向：`> 0` ⇒ `validation_agreement = true`（**进 §5.4 判定树合取**） |
| `gap_closure` | `(ΔD − ΔP)/GAP` | **预声明描述统计**（不进判定树）；GAP 与 ΔP 为钉死常量 |
| `half_gap_met` | `ΔD ≥ ΔP + 0.5·GAP = +0.0027129704794942035` | **预声明描述统计**（闭端）；解读权重：SUPPORTED ∧ 半程达成 ⇒ 强形态；SUPPORTED ∧ 未达半程 ⇒ 部分回收；NOT_SUPPORTED ⇒ 最小延迟即失效 |
| `recovery_ratio` | `ΔD/ΔL`（`ΔL > 0`） | 报告（相对学习臂增量的回收比） |
| `ΔD_vs_U` | `ΔD − G_u`（`G_u = −0.0026189914978292927`，uncond 钉死） | 报告（vs 无条件钉死臂的上下文位次） |
| `G_d_ge_historical_0.0055` | `ΔD ≥ +0.0055` | §5.6 报告项（不判定） |
| `retention_vs_five_seed_mean` | `ΔD / 0.005910402347593546` | 报告（vs 五 seed 均值） |
| `abs_delta_D`、`delta_val_D` 全量 | — | 报告（§7 明细） |

**阈值来源与读法（预声明）**：`+0.001` = 用户指定的最小实质差异，与既有分支分类边界一致；test 1M 前缀 BSI AUC 的 SE ≈ 0.0025 @AUC0.7 ⇒ `+0.001` ≈ 0.4×SE，为**分类标签**而非显著性强断言；单 seed、每臂 1 run、无 run-to-run 噪声估计（A5 沿用）。半程门 `+0.002713` 由两个钉死常量确定性导出（无自由参数）。

### 5.4 因果判定树（**先于结果写死**；优先级自上而下，标签集固定为以下三种）

| 优先级 | 判定 | 条件 | 含义（写死） |
|---|---|---|---|
| 1 | **`INVALID`** | 任一前置条件假（ID / REP_B / FROZEN_ID / A / DD / CC 任一失败） | 不解读效用；`subreason` 按优先级取首个失败项（`IDENTITY_MISMATCH` / `BASELINE_REPRODUCTION_FAILED` / `FROZEN_PHASE_IDENTITY_FAILED` / `PROTOCOL_INVALID` / `MECHANISM_FAIL` / `COMPARATOR_MISMATCH`） |
| 2 | **`DYNAMICS_SUPPORTED`** | `ΔD ≥ +0.001`（闭）∧ `Δval_D > 0` ∧ 前置条件全真 | 把 prompt 模块的学习期延迟 1 个 epoch（冻结期与配对基线逐位同流）之后，学习动力学仍相对基线取得**实质正增量** ⇒ "学习动态/共适应承载增量"的机械解释获得**因果支持**（D vs B 单干预配对）。子态（报告、不改变本判定）：`half_gap_met` ⇒ 强形态（≥半程回收固定↔学习差距）；未达半程 ⇒ 部分形态 |
| 3 | **`DYNAMICS_NOT_SUPPORTED`** | 前置条件全真 ∧ **实质规则失败**：`ΔD < +0.001` **或** `Δval_D ≤ 0` | 最小延迟（仅移出"α 离开 0"的 epoch）即消除实质增量（或方向不一致）⇒ "学习动态承载增量"在本干预形态下**未获支持**；机制解释收窄向 schedule 特异耦合 / 种子脆弱性（§8 读法）。`subreason = BELOW_MATERIALITY`（`ΔD < +0.001`）或 `VALIDATION_DISAGREEMENT`（`ΔD ≥ +0.001` ∧ `Δval_D ≤ 0`）；负值时报 `dynamics_arm_matches_or_exceeds_...` 无此量——仅报告 `ΔD` 与 `gap_closure` |

**读法声明**：判定树**不设**"半程回收"与"vs 历史 +0.0055"为合取项（前者为描述统计、后者为历史口径单列）；本实验只回答"延迟 1 epoch 的学习动力学是否仍承载实质正增量"。`ΔD` 的任何解读必须与 `gap_closure`/`half_gap_met`（描述）以及 §2.5 的混杂清单并读。

### 5.5 二级效用分类（用户指定；对 D 相对 B 的 `ΔD` 机械分类；与 §5.4 并存、不覆盖）

| 类别 | 判据（精确边界） |
|---|---|
| `POSITIVE_IMPROVEMENT` | `ΔD ≥ +0.001` |
| `NO_CLEAR_IMPROVEMENT` | `−0.02 < ΔD < +0.001`（严格双开） |
| `CLEAR_DEGRADATION` | `ΔD ≤ −0.02` |

**`NO_CLEAR_IMPROVEMENT` 的机械 headroom 评估（仅当落入该区间；规则先于结果写死；结构沿用 20-epoch/uncond，因子对本实验适配）**：

| 因子 | 机械口径 | 取值 |
|---|---|---|
| 机制活性 | `alpha_final != 0` ∧ 三流 `ratio_mean > 0` ∧ `geff_max > 0`（取自 D run JSON） | `UNFROZEN_ACTIVE` / `INACTIVE` |
| 解冻接续 | epoch-2 探针 `alpha_grad_norm != 0`（DD4） | `ENGAGED` / `NOT_ENGAGED` |
| 验证方向 | `Δval_D > 0` | `POSITIVE` / `NON_POSITIVE` |
| epoch 轨迹 | D `best_epoch == epochs_run == 5`（未早停）⇒ 右删失；否则早停 | `RIGHT_CENSORED_STILL_IMPROVING` / `EARLY_STOPPED` |
| 距正分类余量 | `gap_to_positive_threshold = +0.001 − ΔD`（>0） | 数值 |
| 回收比 | `gap_closure`（描述性） | 数值 |

### 5.6 历史 `+0.0055` 判定单列保留（不改写）

- 历史成效门槛 `+0.0055` 为学习 correct 臂效用口径（seed-2 复现预注册 §3-U1）；本实验 `ΔD ≥ +0.0055` 作为**报告项单列**（不参与 §5.4/§5.5 判定）。
- 历史判定原样引用：seed1 `VALID_NEGATIVE`（+0.005394543882337954）、seed2 `VALID_POSITIVE`（+0.008103113327467715）、10-epoch `PERSISTS`、20-epoch `NOT_PERSIST`/claim=true/`ABLATION_ELIGIBLE`、五 seed 均值 `+0.005910402347593546`、学习 shuffled `INVALID/MECHANISM_FAIL`、f08ae6e `CONDITION_ALIGNMENT_NOT_SUPPORTED`（`GAP=−0.001861278011441092`）、uncond `SAMPLE_CONDITIONING_NOT_SUPPORTED`（`GAP_U=−5.8180870650015315e-05`）。本实验不重判任何历史结论。

---

## 6. 运行规程（恰好两次；清洁树纪律；前台）

**运行次数**：B stage2 **恰好 1 次**；D stage2 **恰好 1 次**；分析（`analyze_runs`，只读）**恰好 1 次**；运行后独立复核 **恰好 1 次**。不重训 Stage-1；任何历史 run 不重跑（§3.5 裁定；§5 对照仅取已提交记录）。**不设第三臂。**

**清洁树纪律（run 间提交，沿用 20-epoch §4 / f08ae6e §6 / uncond §6）**：① 提交实现与测试（C2）→ `git.dirty=false`；② 跑 **B**（run 记录 commit=C2）；③ 提交 B 的 SUMMARY 追加行（C3，仅 `artifacts/aliccp_bench/SUMMARY.md` 变化）；④ 跑 **D**（run 记录 commit=C3）；⑤ 提交 D 的 SUMMARY 行（C4）；⑥ 核验 `git diff C4 C2 -- . ':(exclude)artifacts/aliccp_bench/SUMMARY.md'` 为空；⑦ 分析 ×1 → 复核 ×1 → C5（§10 回填）→ push。

```powershell
# cwd = 本 worktree 根；本 worktree 无 .venv，用主树解释器；dataset 经 junction 复用（只读）

# 0) 测试（先红后绿；全部不变量）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests

# 1) B 配对基线臂（唯一一次；协议默认：epochs 5 / patience 2 / tag short）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag short --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f

# 2) D 延迟解冻臂（唯一一次；α/生成器冻结 1 epoch、epoch 2 起点解冻；钉死参照头）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag short --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f `
    --variant residual-prompt-delay `
    --prompt-reference-newtask artifacts/aliccp_bench/runs/20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt

# 3) 分析（纯分析，恰一次；只读两 run 记录 + 历史常量）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m aliccp_benchmark.rp_delay `
    --baseline-run <B run_id> --delay-run <D run_id>

# 4) 运行后独立复核（只读；JSON 重推 + 文件哈希 + checkpoint 独立读取 α）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe verify_rp_delay.py `
    --baseline-run <B run_id> --delay-run <D run_id>
```

- **前台执行并等待至完成**（先例：后台执行被会话终止杀死）；开跑前提交实现（§6 步骤①）；每臂 run 启动时 `git.dirty` 必须为 false。
- **无效执行处置**：仅当 run 因**工具性原因**（崩溃/中断/环境故障、产物未完整落盘）无效时，保留现场、记录原因后可重跑一次；因**结果原因**不构成无效，不重跑。无任何事后调参（延迟界/解冻实现/构造/阈值/参照/判定树均不得动）。
- **墙钟预算**：每臂 ≈2–5 min（short 实测 143–276 s + 延迟审计开销），总计可控。

---

## 7. 必录诊断（落盘 run 目录；class C，不判定）

| 量 | 字段 | 用途 |
|---|---|---|
| **延迟/解冻证明（D）** | `delay`：`delay_epochs`、`unfreeze_epoch`、`alpha_at_construction`、`alpha_requires_grad_at_construction`、`generator_requires_grad_at_construction`（×4）、`generator_sha_at_construction`、`alpha_at_unfreeze`、`generator_sha_at_unfreeze`、`unfrozen_once`、`requires_grad_after_unfreeze`（α + 生成器）、`optimizer_covers_named_parameters`、`optimizer_params_total`、`unfreeze_rng_endpoint_unchanged` | DD3（+DD1/DD2） |
| 构造身份 | `construction_identity`：共享参数逐位/RNG 端点/新增键集（5 键） | DD1 |
| 注入纯度/初值 | `init_forward`：`bit_identical`、`max_abs_diff`、`n_samples` | DD2 |
| 梯度探针 | `grad_probe`（逐 epoch 首 batch：`alpha`、`alpha_grad_norm`（epoch 1 为 None）、`generator_grad_norm`（epoch 1/2 首探针结构为 0.0）） | DD4 + 描述 |
| 残差/基范数比 | 逐流 `ratio_mean/std/max`、`delta_norm_mean`、`h_norm_mean` | DD5/DD6 + 有效幅度披露 |
| 余弦方向 | 逐流 `cos_mean` | 描述（对照：学习 correct `+0.01287/+0.08432/+0.05899`；钉死 correct `−0.03315/+0.06856/−0.00197`） |
| 逐样本门控 | `gate.geff_mean/std/min/max` | DD7 + headroom（对照：学习 correct `geff_std = 0.006022934967912283`；钉死 correct `0.006211574794529912`） |
| 预测离散度 | `dispersion` + `reference_dispersion`（M0 口径） | DD8 + 描述 |
| 源/头门 | `gate_mean` + Stage-1 环境上下文（`cluster_events`、`env_acc`） | 落盘；B4 原样披露 |
| 参数清单 | `params`：5 键、`new_params_total = 2385`、`head_params = 8129` | DD9 |
| 臂级判定 | `rp_arm`：DD1–DD9 + M0 + U1/U2 + 分类 | §5.2 机械分类 |
| 配对判定 | `rp_delay_compare.json`：§5 全部前置条件、分量（`ΔD`/`Δval_D`/`gap_closure`/`half_gap_met`/`recovery_ratio`/保留比/披露量）、判定树、二级分类、headroom（若适用）、context 表（L/C_p/五 seed/预算链/U 臂） | §5.4/§5.5 |
| 停止/删失 | 两 run 的 `best_epoch`/epochs、是否早停、逐 epoch 轨迹 | 描述 |
| provenance | 两 run 的 `run_id`/`commit`/`git.dirty`/`wall_seconds`/`peak_vram_mb`/`hard_pass`/A 类/B 类（继承披露） | §10 记录 |

---

## 8. 局限（预声明）

1. **单 seed、每臂 1 run**：无 run-to-run 噪声估计（A5 沿用）；`+0.001` 为分类标签而非显著性强断言（≈0.4×SE）。
2. **单一延迟值（k=1）**：结论条件于"最小延迟"这一个干预点；**不外推**到任意延迟/调度（无 k 扫描——§9）。
3. **D vs L / D vs C_p 非单因子**（§2.5）：`gap_closure`/`half_gap_met` 为描述统计；不得作"延迟量→增量量"的因果拟合。
4. **学习期长度差异**（4 vs 5 epoch）与预算收窄效应（`+0.0081→+0.0042`）在负结果时无法在本实验内分离（§2.5-5）。
5. **D 臂解冻初值 = 0.0（非 warm-start）**：不主张"从历史边界值继续"的等价性（已否决变体，§2.3）。
6. **判定依赖单一汇总统计**（test 1M 前缀 BSI AUC 差值）；判定取 run 记录值；正确性依赖代码路径守卫 + REP_B/FROZEN_ID 双重逐位 + 独立复核。
7. **冻结期恒等依赖结构论证**（引理 1.4-2 + 无 dropout/无随机算子 + 相同数据顺序）；由 FROZEN_ID 端到端逐位核验背书；若失败 ⇒ `INVALID`（保留现场核查）。
8. **B4 若 FAIL 为继承缺陷**（Stage-1 共享语义；臂无关）；`hard_pass` 不参与有效性定义（同先例）。
9. **单数据集（AliCCP）、单 seed**：结论不推及 CensusIncome / ByteRec、不推及其它 seed / 预算点。
10. **本实验不是"更优方法"主张**；是机制解释消融的一次性判定。

---

## 9. 非目标与止损

1. 不做：其它消融（norm-control 移除、matched-parameter adapter、聚类、Census、第二 seed、α 扫描、多延迟/调度扫描、warm-start 变体、学习臂重放、shuffled 变体接线、论文/图表——**明确排除**）、不跑 `full` tag、不做 FLOPs、不重跑任何历史 run。
2. 不因结果修改：`DELAY_EPOCHS`、`UNFREEZE_EPOCH`、解冻实现裁定、`MATERIAL_DELTA`、半程门定义、二级边界、判定树、§5.1–§5.4 任何规则、参照文件、clean-tree 纪律。
3. **后续独立消融的预声明建议（本分支不启动，仅登记；按结果机械选择，先于结果写死）**：
   - 若 `DYNAMICS_SUPPORTED`（含半程达成与否两形态）⇒ 学习动力学承载可分离实质效用 ⇒ 机制线继续价值有限；下一步最优先：**延迟扫描的最小扩展**（k=2 单点，验证"回收随延迟的单调性"）或 **warm-start 变体**（从历史边界值继续）——均需全新预注册；**norm-control 移除 / matched-parameter adapter 仍不满足选择条件**（其机械触发条件为 uncond 分支的 `G_u ≥ +0.001`，从未满足），维持更低优先。
   - 若 `DYNAMICS_NOT_SUPPORTED` ⇒ 最小延迟即抹平增量 ⇒ "学习动态承载增量"的稳健形态未获支持 ⇒ **机制线建议停止**（剩余解释 = schedule 特异耦合/种子脆弱性，均非可操作机制）；**norm-control 移除与 matched-parameter adapter 的信息价值同步判定为消失**（它们检验的是"固定注入族内部哪个通道承载效用"，而固定注入族已被 f08ae6e/uncond 与本实验三重钉死为无可观测效用）——建议不再启动。
   - `INVALID` 情形下先修复执行/实现，不启动任何后续消融。
   - 上述均需**全新预注册**；本分支不启动。

---

## 10. 结果（运行后回填，不回写第 0–9 节）

### 10.0 非结果核验与执行记录

§1.5 预核验 **153/153 通过**（报告 sha256 `7e35d9e7…`，§1.5）。执行链：C1（本文件）→ C2（实现+测试）→ B 臂前台单次 → C3（B SUMMARY 行）→ D 臂前台单次 → C4（D SUMMARY 行）→ 分析 ×1 → 独立复核 ×1 → C5（§10 回填）。

### 10.1 两条 run 与臂级身份

（待回填：run_id/commit/dirty/stage1_id/epochs/patience/best_epoch/best val/test/early-stop/墙钟/峰值显存/A 类/B 类/hard_pass/臂级分类；REP_B 逐位复现；FROZEN_ID 三方逐位；产物 sha256。）

### 10.2 判定（预注册 §5 机械计算；分析器 + 独立复核逐位一致）

（待回填：前置条件各组；因果判定；二级分类；历史 `+0.0055` 单列；全量分量 `ΔD`/`Δval_D`/`gap_closure`/`half_gap_met`/`recovery_ratio`；headroom（若适用）；有效幅度披露。）

### 10.3 结构机制门禁与诊断（DD1–DD9 关键读数）

（待回填。）

### 10.4（保留节位）延迟/解冻完整性核对

（待回填。）

### 10.5 独立复核

（待回填：`verify_rp_delay.py` 计数、重推一致性、checkpoint 独立读取。）

### 10.6 纪律核对

（待回填：运行次数 ×2、分析 ×1、复核 ×1；测试计数；静态守卫；run 间 diff；SUMMARY 行数；跑后代码改动披露。）

### 10.7 解读（预注册口径；不越界）

（待回填：机械结论；与 L/C_p 的关系与 gap 回收读数；机制指向；历史判定原样保留声明；继承披露。）

---

## 11. 非新颖性声明

构件全部为既有方法族（prompt tuning、FiLM 族条件化调制、adapter/LoRA 残差、门控残差、范数归一、warmup/延迟解冻调度为训练常用手段），先行工作清单沿用 seed1 设计 §11（写作引用前必须逐条核实，本会话未做系统检索）。本文件**不**声称独立于先行工作；可报告内容仅限本仓库冻结协议下的一次性判定。

---

## 12. 偏离披露（预声明 + 运行后补）

- **预注册纪律**：本文件先于任何实现与运行单独提交（C1）；判据、阈值、构造、延迟界（k=1）、解冻实现、判定树、运行规程在看到结果前写死。
- **分支工作流**：自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；机制与 rp_pinned 经 §3 三层守卫钉死等价；协议不合并 `master`；push 经用户显式指示（本任务含 "commit and push"）。
- **分支本地 SUMMARY 系谱**：本分支 `SUMMARY.md` 基线含 smoke 行与 seed1 基线行（`8133d32` 携带）；其余历史行记录在各自分支，本分支**不复制**其它分支行，只由本实验 run 追加自己的行（B + D）。
- **产物/历史 run 只读复制**：§1.5 所列产物、参照头、五个历史 run 均自 uncond worktree **只读复制**入本 worktree（逐文件 sha 相等；§1.2 已复核）；dataset 经 junction 复用 main tree（只读、gitignore）。
- **非结果文件（未跟踪）**：`artifacts/aliccp_bench/audit/rp-delay/` 报告 JSON 与 α 轨迹导出、`artifacts/aliccp_bench/logs/` run 日志、D run 目录内 `rp_delay_compare.json` / `verify_report*.json`。
- **与前置分支的关系**：`rp_pinned.py` 的 shuffled 类/错排器为逐字节移植带来的休眠代码（CLI 不暴露 delay 相关接线以外的变化；本实验不运行 pinned-shuffled）。
- **执行记录（运行后补）**：（待回填。）
- **实际结果形态（运行后补）**：（待回填。）

---

**改动清单（白名单；静态守卫以 `8133d32` 为基准）**

| 文件 | 处置 |
|---|---|
| `docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-delayed-unfreeze-dynamics-design.md` | 本文件（C1 单独提交） |
| `aliccp_benchmark/residual_prompt.py` | 新增（A1：逐字节 == `f08ae6e` 钉死 blob `674213f6…`） |
| `aliccp_benchmark/rp_pinned.py` | 新增（A2：逐字节 == `f08ae6e` 钉死 blob `432fa9bb…`） |
| `aliccp_benchmark/rp_delay.py` | 新增（A5：消融本体 + 判定 + 分析 CLI） |
| `aliccp_benchmark/bench.py` | A3：== `f08ae6e` + 文档化最小适配（delay 分派/解冻钩子/臂判定/prompt_doc；其余逐字） |
| `run_aliccp_benchmark.py` | A4：== `f08ae6e` + 文档化 3 行适配（import/choices/后缀分派） |
| `aliccp_benchmark/tests/test_residual_prompt.py` | 新增（A6：== `f08ae6e` 恰 4 处适配） |
| `aliccp_benchmark/tests/test_residual_prompt_delay.py` | 新增（A7：本分支守卫 + 延迟/接线/分析器测试） |
| `verify_delay_prerun.py` | 新增（A8：预注册前完整性核验；§1.5） |
| `verify_rp_delay.py` | 新增（A8：运行后独立复核） |
| `artifacts/aliccp_bench/SUMMARY.md` | 只追加 2 行（runner 自动） |
| 其余全部（`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`metrics.py`、既有测试三件、`AliCCP_*.py`、`CensusIncome_*.py`、`baseline/*`、`mask/*`、`analysis/*`） | **零改动**（静态守卫断言 `git diff --name-only 8133d32` ⊆ 白名单） |
