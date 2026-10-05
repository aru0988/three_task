# AliCCP 阶段 2 残差 Prompt 的 α 钉死条件对应因果消融（pinned-alpha conditioning；canonical seed-2、5-epoch short、三臂单因子）

- **状态**：预注册已写死（本文件在本分支任何实现、任何 run **之前单独提交**；commit C1）。结果只在第 10 节以新增小节回填；第 0–9 节的判据、数字、构造与运行规程不得在看到结果后改动。
- **日期**：2026-10-05
- **适用分支**：`exp/aliccp-stage2-residual-prompt-alpha-pinned-condition`（自 `infra/aliccp-fair-benchmark` @ `8133d32` **独立拉出**；不从任何 exp 分支拉出、不继承其运行时状态；不合并 `master`、不触碰其它 worktree；`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`aliccp_benchmark/metrics.py`、既有测试三件、master 脚本与 `baseline/*` 零修改）。
- **立项依据**：`exp/aliccp-stage2-residual-prompt-shuffled-condition` @ `fbfff09`（`fbfff0945c77c74866955238f67e9aeaa0774670`）的结果——因果判定 `INVALID/MECHANISM_FAIL`（错配条件下学习到的 `|α|` 收缩至 `0.010569889098405838` ⇒ `ratio_mean 0.003168184354…` < 活性带下界 `0.005`，唯一失败门 G5；PG1–PG6 全 PASS）——及其 §10.7.1 **预声明的下一步消融建议**：**α 匹配的条件消融**（两臂钉死同一 `α = +0.07101669907569885`，唯一差异 = 样本↔条件对应）。
- **被消融对象（只读引用；系谱钉死）**：残差 prompt 机制 = `exp/aliccp-stage2-residual-prompt-seed-replication` @ `013e105`（`013e105f6dfb35e630d261696290635d0263ce99`）`:aliccp_benchmark/residual_prompt.py`（blob `674213f619c5d5039811c71242a7727118348daf`，LF sha256 `b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc`）；其系谱：seed1 线（预注册 `221580a` / 实现 `79ddefa`（blob `5dc7158ca999c9e7e6217a999c37869231411549`）/ 结果 `3e2f083`）→ Census 祖本 `99b9510`（blob `107221b26382da7fd44990e167d9a61da2680d92`）。
- **α 钉死值来源（只读引用；系谱钉死）**：correct-condition seed2 学习臂 run `20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg`（commit `013e105`，dirty=false）的 `prompt_report.json:alpha_final = +0.07101669907569885`（该文件 sha256 `185df4d0f9910508aeff84ed780e7df4fe8ad4fd6b8451ff429a74927d333c1e`，见 §1.5 钉死）。该值 float32 往返**逐位精确**（`float(torch.tensor(v, dtype=torch.float32)) == v`，本会话实测），可作为 buffer 常量以精确相等比较。
- **上游协议（只引用、不修改）**：
  1. `docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（AliCCP 公平评测协议）——前缀预算、种子、Stage-1 产物、真冻结三件套、A/B/C 门禁、SUMMARY 台账、止损纪律全部沿用；
  2. `docs/superpowers/specs/2026-10-04-aliccp-stage2-residual-prompt-seed-replication-design.md`（seed-2 复现预注册）——机制语义、参照头钉死、U1/U2 口径（历史 `+0.0055`）逐字沿用；
  3. `exp/aliccp-stage2-residual-prompt-20epoch` @ `10e86dc`（20-epoch 终止判定点）——审计方法（以磁盘原始 run 产物 + git 对象为准）、二级分类三标签边界、`NO_CLEAR_IMPROVEMENT` 的机械 headroom 结构、`commit-between-runs` 清洁树纪律、独立复核脚本形态（纯 JSON 重推）**逐字沿用**；
  4. `exp/aliccp-stage2-residual-prompt-five-seed` @ `0935257`（结果）/ `28aa1fe`（provenance tip）——五 seed canonical 配对与跨进程逐位确定性证据（§1.2-A4）；
  5. `exp/aliccp-stage2-residual-prompt-shuffled-condition` @ `fbfff09`——**错排构造规格（`COND_PERM_SEED=20261005`、sha256 键推导 + Sattolo、RNG 隔离）逐字沿用**；置换门禁 PG1–PG6 及判定树形态沿用；其 INVALID 结论与 §10.7.1 建议为本实验立项依据。
- **定位声明**：本实验是 fbfff09 `INVALID` 的**预注册化解决**：把学习门控 `prompt_gate` 在**两条处理臂**中替换为**同一个 fixed signed α = +0.07101669907569885**（非参数、排除于优化器、训练全程逐位不变），使两臂注入的**标量幅度相同**，唯一变化 = 目标样本与 prompt 生成器条件特征之间的**对应关系**（§2）。学习活性带 G5（"以结局为门"）**不再用作门禁**，替换为**先于结果写死**的钉死 α 结构式门禁 PA1–PA8（§5.3，附 a priori 不适用声明）。无新颖性主张、无改进主张、不调参、不放宽任何门限。

**提交时序（预注册纪律）**：C1 本文件（审计 + 预注册，单独提交）→ C2 实现 + 测试（TDD；三 run 的 `commit` 字段即 C2 与后续两个 SUMMARY 提交）→ C2 之后跑 **B 臂（配对基线）**（前台）→ C3 提交 B 的 SUMMARY 行 → 跑 **F_c 臂（钉死 α、correct-condition）**（前台）→ C4 提交 F_c 的 SUMMARY 行 → 跑 **F_s 臂（钉死 α、shuffled-condition）**（前台）→ 分析恰一次 → 独立复核恰一次 → C5 结果回填（§10）+ F_s 的 SUMMARY 行 → push。

---

## 0. 判定问题

对照三分：**B = 本实验新跑的配对基线臂**（seed2、short、5-epoch；协议默认）；**F_c = 本实验新跑的钉死 α correct-condition 臂**（唯一变化 vs 历史 correct 臂 = α 由学习改为钉死）；**F_s = 本实验新跑的钉死 α shuffled-condition 臂**（唯一变化 vs F_c = §2.2 的条件对应错排）；**C = 已提交 correct-condition 学习臂**（`013e105-rpg`，hash 核验后**仅作上下文对照**，不重跑）。

记 `GAP := test(F_c) − test(F_s)`（**因果主量**；两臂标量幅度由钉死 α 匹配）；`G_c := test(F_c) − test(B)`；`G_s := test(F_s) − test(B)`；`R := G_s / G_c`（保留比）；`Δval_gap := val(F_c) − val(F_s)`。

- **Q1（因果主判定）**：在**标量幅度匹配**（同 `α`、同范数标定、同注入流）之下，correct 臂是否相对 shuffled 臂**实质更优**（`GAP ≥ +0.001`）且验证方向一致（`Δval_gap > 0`）？→ §5.5 判定树：`SAMPLE_CONDITION_SUPPORTED` / `CONDITION_ALIGNMENT_NOT_SUPPORTED` / `INVALID`。
- **Q2（保留比与历史对照）**：`R` 的实测值；`G_c`/`G_s` 相对历史学习臂增量的**保留比**（vs `C` 的 `+0.008103113327467715`、vs 历史学习 shuffled 臂 `9d26bc8-rpgs` 的 `+0.0014787611000699474`）——均报告、不进 Q1 分类。
- **Q3（二级效用分类）**：F_c 与 F_s **各自**相对 B 的 `Δtest` 落入用户三标签（边界先于结果写死，§5.6）；任一落入 `NO_CLEAR_IMPROVEMENT` 时按 §5.6 机械评估 headroom。
- **Q4（完备性）**：前置条件（ID / 基线复现 REP / A 类 / 钉死 α 与结构机制门禁 PA / 置换门禁 PG / 对照链 CC / deck 重放 DD）是否全真？任一假 ⇒ `INVALID`（不解读效用）。

**纪律声明（预注册）**：历史判定与全部数值（seed1 `VALID_NEGATIVE`、seed2 `VALID_POSITIVE` +0.0081、10-epoch `PERSISTS`、20-epoch `NOT_PERSIST`/claim/`ABLATION_ELIGIBLE`、5-seed 均值、shuffled 学习臂 `INVALID`、`+0.0055` 历史效用门等）**原样保留、不改写、不重判**。本实验只用 §4/§6 新跑的三条 run 做因果与分类判定；学习臂 run（C 与历史 shuffled）仅作 hash 核验后的上下文对照（**不进任何门禁与主判定**）。

---

## 1. 审计（2026-10-05，只读；先于本文件提交）

### 1.1 审计方法与范围

- **原则**：不信任任何文档转述，全部以 **git 对象**（blob / `ls-remote` tip）与**磁盘上的原始 run 产物**（本 worktree 内按 §1.5 逐文件 sha256 核验后只读复制入的副本）为准重新核对；不重跑、不修改任何历史 run。
- 审计对象与 remote tip（`git ls-remote origin` 实测与本地一致）：shuffled `fbfff09`（`fbfff0945c77c74866955238f67e9aeaa0774670`）、seed-replication `a40836e`（`a40836e3d6a7f73c4943bada98139d414957301a`；其预注册 `e931cd7`）、five-seed `28aa1fe`（结果提交 `0935257`）、20epoch `10e86dc`（`10e86dc6cbd047b103211ab75dd99c1c68db3526`）、gate `3e2f083`、longer-budget `e46e5d2`、基点 `8133d32`（`8133d32cfa722b084c558b334f4792d85ce95c67`）。
- blob 钉死实测（`git rev-parse`）：`013e105:aliccp_benchmark/residual_prompt.py` = `674213f6…`；`fbfff09:…residual_prompt.py` = `674213f6…`（**逐字节一致**，消融分支零文本适配，实测 `git diff --stat 013e105 fbfff09 -- …residual_prompt.py` 为空）；`013e105:aliccp_benchmark/bench.py` = `bf3647686838157bf5fd7645615faad9f2d7185d`；`013e105:run_aliccp_benchmark.py` = `229c25a1c86719fa6fb6b057d6cff3f4120fa053`；`013e105:tests/test_residual_prompt.py` = `768a477b5c71af32c5c59ecfeb20c29f7873557`。
- 审计为只读核对（git + 原始 JSON 重读 + sha256 复算），无可跟踪审计脚本；全部命令可复现。

### 1.2 审计结果（逐项）

| # | 审计项 | 证据（独立重推） | 结果 |
|---|---|---|---|
| A1 | **fbfff09 失败链复核（立项核心）** | 原始 JSON 重读 `20261005-0933-p2M-v500k-t1M-m1688723740-short-9d26bc8-rpgs`：`metrics.json:rp_arm.G5 = {pass:false, ratio_mean:[0.003168184354059116, 0.0031681843541297287, 0.0031681843543092778], band_low:true}`；`rp_arm.G3.observed.alpha_final = 0.010569889098405838`；`classification = MECHANISM_FAIL / MECHANISM_SILENT`；`prompt_report.shuffle.report`：`n_perms_total=1750`、`fixed_points_total=0`、`bijection_failures=0`、`regeneration_mismatches=0`、`rng_isolation_violations=0`、`key_conflicts=0`、`degenerate_identity_batches=0`；`rp_shuffled_compare.json`：`verdict=INVALID / subreason=MECHANISM_FAIL`，ID 14 项全 true、REP 全 true、PG1–PG6 全 PASS、CC 全 true | PASS |
| A2 | 配对基线复现（B 臂的预期） | `20261005-0930-…-e5619a7`（commit `e5619a7`，dirty=false）与历史基线 `20261003-0624-…-79b5e07`（commit `79b5e07`，dirty=false）：test `0.5974422649550507`、val `0.5809347091990792`、best=5/5、gate_mean `[0.789178, 0.2108221875]`、逐 epoch val 轨迹**逐位相等**；`newtask.pt` sha256 两 run 均 == `90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f`（参照头钉死值） | PASS |
| A3 | **correct-condition 学习臂记录（α 钉死值来源）** | `20261004-0325-…-013e105-rpg`（commit `013e105`，dirty=false）：test `0.6055453782825184` / val `0.5895572066556973`；`prompt_report.json:alpha_final = 0.07101669907569885`；`rp_arm = VALID_POSITIVE`，M0+G1–G8 全 PASS（G5 `ratio_mean=0.028139385824218544`；G4 `ratio_max=0.049055453891063694` ≤ |α|+1e-6）；`Δtest = +0.008103113327467715`、`Δval = +0.008622497456618139`（逐位重推）；文件 sha256 `prompt_report.json = 185df4d0…` | PASS |
| A4 | 跨进程/跨 commit 确定性（支撑"新跑三臂可对账"） | five-seed 记录：seed3–5 两次 pass 重跑**逐位相等**（含 `newtask.pt` 字节）；20-epoch 记录：20-epoch 基线前 10 epoch 逐位等于 10-epoch、前 5 epoch 逐位等于 5-epoch（两臂同理）；A2 再次实测（e5619a7 == 79b5e07 逐位）⇒ 同 seed 的 stage2 训练在本环境确定性可复现 | PASS |
| A5 | 机制实现的输入使用面（消融靶点定位） | `NewTask.forward`（`8133d32:multitaskrec/model.py`，零改动）2 处使用 `dnn_input`：`projection_network`、`gate_network`；`ResidualPromptNewTask.forward` 追加第 3 处：`prompt_generator(dnn_input)`（经 `prompt_deltas`）。本实验的消融面 = （i）第 3 处的**条件对应**（F_s 的错排；§2.2）与（ii）**α 的学习性→钉死**（两处理臂同值；§2.1）。前三处（基头）保持正确样本输入 | PASS |
| A6 | 五 seed 立项条件（只读引用，不重推） | five-seed 分支：5 个 canonical seed 短预算配对逐 seed M0+G1–G8 全 PASS ∧ A 类 PASS，二级分类 5×`POSITIVE_IMPROVEMENT`，`mean_Δtest = +0.005910402347593546`（样本 std 0.0031538117454772883，Student-t 95% CI 下界 > 0） | PASS |
| A7 | **错排 deck 钉死（F_s 复用同一张错排）** | fbfff09 §10.4 记录（本会话原始 JSON 重读复核）：`total_digest = dcf302c1a80ae7e8023345105e559d7a6daaa1ad492b016c39c22705886599d8`；train `50ecd19e5775069bfc1270e26c11f13feff34bfc238d61ae87a5e0f5c9ac4d60`（1000 批/2 000 000 行）、val `2d85298c509acc26f2d9ee7654e2fb4accdf8d76d68e280e0946a68171b966c0`（250 批/500 000 行）、test `239856ffef561c9a4c5dd19fee383a8c19f218fc92d8ea6aec66cb0acdd622cb`（500 批/1 000 000 行）。F_s 以同一 `COND_PERM_SEED=20261005` 键空间重建同一 deck ⇒ digest 必须逐位复现（§5.2-PG7） | PASS |

### 1.3 seed / 预算 / α 值冻结（**先于任何本实验指标**）

**canonical seed 冻结为 seed2 = `1688723740`；预算冻结为 short（5-epoch / patience 2）；α 冻结为 `+0.07101669907569885`（signed，correct 臂 5-epoch 终值）。** 理由（全部先于新指标、由 §1.2 证据支撑）：

1. **seed2 是唯一具备完整既有证据链的 seed**（A2/A3/A4；同 20-epoch 分支 §1.3 论证）：同 seed 的 5/10/20-epoch 配对、钉死参照头（sha `90ee06da…`）、内容寻址 Stage-1 产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f` 齐备；其余 seed 只有 5-epoch 配对。
2. **short（5-epoch）是成本最低的完整协议点**：三臂实测墙钟 ≈160–200 s/run（A2 + fbfff09 §10.1），总计 ≈10 min，单次前景窗口可控；预算效应不是本实验变量。
3. **short 上学习 correct 臂效用最强且已过线**（`G_c^hist = +0.0081 ≥ +0.0055`）：若条件对应因果有效，short 是**最容易**观测到 `GAP` 的判定点（对 `CONDITION_ALIGNMENT_NOT_SUPPORTED` 是最保守的检验）。
4. **α 取 correct 臂终值**：这是"钉死 α 匹配"唯一有系谱依据的选出规则（fbfff09 §10.7.1 预声明：`α = +0.07101669907569885`，correct 臂 5-epoch 终值）；不扫描 α、不试多个值（§9）。
5. **单因子**：F_c 相对 C 的唯一差异 = α 学习性→钉死同值；F_s 相对 F_c 的唯一差异 = §2.2 条件对应置换；B 相对 C 的差异 = 无 prompt 头。B 与 `79b5e07` 的历史记录预期逐位一致（A2/A4 确定性），构成对运行环境与配对链的**独立复核**（§5.1-REP）。

### 1.4 机制审计（α 的角色；及为何 G2/G3/G5 在钉死设计下不适用——a priori 声明）

机制（`013e105:aliccp_benchmark/residual_prompt.py`，公式与结构化事实逐字沿用 seed-2 复现预注册 §1.2）：

    P(x)        = tanh(W2·ReLU(W1·x + b1) + b2)            # 有界方向，来自条件特征 x
    g_eff(x)    = |alpha| · ||P(x)|| / sqrt(d)             # 逐样本有效门控
    delta_s(x)  = alpha · (||h_s|| / sqrt(d)) · P(x)       # 残差 = 门控 × 单位方向 × 基向量范数
    h_s'        = h_s + delta_s                             # 注入融合前三路（gen + spec_0 + spec_1）

- **学习臂中 α 的角色**：`prompt_gate` 是零初始化 `nn.Parameter`（标量）；α=0 使首步生成器梯度为 0（∂loss/∂m ∝ α），α 自身首步即有梯度（G3 探针：0.0511/0.0592）。G2（α=0 初值恒等）与 G3（α 学习活性）、G5（`ratio_mean ∈ [0.005, 0.5]` 学习幅度带）都是**针对学习 α 的**门禁。
- **钉死设计下不适用（先于结果声明）**：
  1. **G2**：α 构造值 ≠ 0 ⇒ "初值恒等"命题不存在。替换为 **PA4**：α 临时置 0 时与同权重参照头**逐位相等**（注入纯度；恢复后逐位校验）+ α 钉死值下首 batch 与参照头 `max_abs_diff > 0`（干预自首步即激活）。
  2. **G3**：α 非参数 ⇒ 无梯度、无"学习活性"命题。替换为 **PA1/PA2**：钉死身份（buffer、非参数、`requires_grad=False`、构造/训练后/checkpoint 三处逐位相等）与优化器排除（不在 `named_parameters()`、不在任何 param group、探针 `alpha_grad_norm is None`、训练前后逐位不变）。
  3. **G5**：原口径以**学到的门控幅度**为门——正是使 fbfff09 判定 `INVALID` 的"以结局为门"缺陷（错配使 α 收缩 6.72×，条件对应与干预幅度不可分离，`|α|` correct/shuffled = `0.07101669907569885/0.010569889098405838 = 6.718774285570287`）。钉死设计下 `ratio = |α_pin| · ||m||/√d` 的上界由 |α_pin| **结构性地**保证（**PA5**），跨流一致性由构造保证（**PA6**），"激活"由 PA4(ii) 结构性保证。**不设学习幅度带**；`ratio_mean/std/max`、`cos_mean`、`geff_*` 全量记录为**描述量**（§7），不判定。
- **本消融恰好改变**：（i）F_s：目标样本 i 的残差所用方向 `P(x_{π(i)})`（π = §2.2 的确定性错排）；（ii）F_c/F_s 两臂：α 的学习性→钉死同值。**不改变**：α 语义（signed 标量、乘在范数标定后的方向前）、范数标定 `scale`、注入流集合、`super().forward` 委托、损失、优化器类型/超参、训练协议、数据顺序。

### 1.5 预注册前参照核验（只读；先于本文件提交；`verify_pinned_prerun.py`）

对固定产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f` + 参照头 + 历史 run 链（79b5e07 基线 / 013e105-rpg 学习 correct 臂 / 9d26bc8-rpgs 学习 shuffled 臂）做只读复算（**89/89 通过**，报告 `artifacts/aliccp_bench/audit/rp-pinned/verify_pinned_prerun_result.json`；不写任何 run/stage1 产物、不训练、不重新评测）：

| 组 | 核对 | 结果 |
|---|---|---|
| 1 产物文件字节 | `backbone.pt`/`env_ids.pt`/`meta.json` sha256 == 钉死值（`cd7b0334…` / `4660be5a…` / `61a66d81…`） | PASS |
| 2 内容寻址 | `stage1_id` 重算 == 记录 == 期望；`config_hash` 重算 == `4e1b5c6f…`；id 分量（fp/config 前缀、seed=1688723740、epochs=3、env_seed=20261003、budgets） | PASS |
| 3 前缀指纹 A2 级 | 自哈希 == `5c060b9c…`；前缀字节 sha256 ×3 + 表头 + 文件大小 + 原始扫描标签计数 ×3 | PASS |
| 4 env_ids | 张量 sha 重算 == `5cd198f1…` == meta；形状 (2000000,)∈{0,1}；计数 1532/1998468 | PASS |
| 5 backbone | 张量 sha 重算（排除 buffer `env_indices`）== `e5e7e610…` == meta | PASS |
| 6 参照头 | 文件 sha256 == `90ee06da…`；`NewTask(80/64/[32,32])` strict 载入成功 | PASS |
| 7 历史 run 链 | 79b5e07（4 文件）、013e105-rpg（6 文件）、9d26bc8-rpgs（8 文件）逐文件 sha256 == 钉死；记录值钉死：三 run 的 test/val/逐 epoch val/gate_mean/best_epoch/epochs/patience/seed；013e105-rpg `variant=="residual-prompt"`、`rp_arm.classification=="VALID_POSITIVE"`、α_final == 钉死；9d26bc8-rpgs `variant=="residual-prompt-shuffled"`、`rp_arm.classification=="MECHANISM_FAIL"` | PASS |
| **8 α 钉死来源（新）** | `013e105-rpg/prompt_report.json` 文件 sha256 == `185df4d0…` ∧ `alpha_final == PINNED_ALPHA`（精确相等）∧ `float32` 往返精确 | PASS |
| **9 deck 钉死（新）** | 9d26bc8-rpgs `shuffle.report`：`base_seed==20261005`、`n_perms_total==1750`、逐 split digest ×3 + total digest + 批数/行数 == §1.2-A7 钉死值 | PASS |

- **产物复用决议**：按"先验证、后复用，否则确定性重生成"规程——N/N 通过 ⇒ **复用**该产物（不重训 Stage-1）；产物、参照头、三个历史 run、前缀指纹均自 shuffled worktree **只读复制**入本 worktree（逐文件 sha256 与源相等，复制后复验）；dataset 经 junction 复用 main tree（只读、gitignore）。
- 钉死参照常量（后续判定全部引用）：`BASELINE_AUC_TEST = 0.5974422649550507`、`BASELINE_AUC_VAL = 0.5809347091990792`、`REFERENCE_PRED_STD = 0.005217193225189258`；`CORRECT_AUC_TEST = 0.6055453782825184`、`CORRECT_AUC_VAL = 0.5895572066556973`、`DELTA_TEST_CORRECT = 0.008103113327467715`、`DELTA_VAL_CORRECT = 0.008622497456618139`；`PINNED_ALPHA = 0.07101669907569885`；历史学习 shuffled 臂：`SHUF_HIST_AUC_TEST = 0.5989210260551207`、`SHUF_HIST_AUC_VAL = 0.5819965357883198`、`DELTA_TEST_SHUFFLED_HIST = 0.0014787611000699474`。

### 1.6 审计结论（fbfff09 失败链验证 + confound 陈述）

1. **失败链验证成立**（A1 逐位复核）：`INVALID/MECHANISM_FAIL` 的唯一失败面 = 处理臂 G5 学习幅度带（错配条件下 `|α|` 坍缩至 0.010569889098405838 ⇒ `ratio_mean 0.00317 < 0.005`）；置换、身份、复现、对照、实现面全部通过 ⇒ **不是执行/实现故障**。
2. **Confound 陈述（立项依据）**：在"唯一变化 = 条件对应"的设计中，α 仍是被优化的参数 ⇒ 错配条件化**同时**改变了"样本↔条件对应"与"学习到的干预幅度"（两臂 `|α|` 相差 6.72×；有效幅度 `ratio_mean` 0.0281 vs 0.0032）⇒ 即使 G5 越过，`G_c − G_s` 也无法唯一归因于条件对应。**α 钉死把该 confound 从设计中移除**：两臂 α 逐位相同，`GAP` 的差异只能来自（a）条件对应（b）随之而来的生成器学习轨迹差异（见 §8-6，双臂对称、预声明披露）。
3. 历史判定原样保留（§0 纪律声明）；本实验不重判任何历史结论。

---

## 2. 消融设计（唯一变化 = 条件对应；α 双臂同值钉死）

### 2.1 α 钉死构造（先于实现写死）

- **常量**：`PINNED_ALPHA = 0.07101669907569885`（signed；= §1.5 核验后的 correct 学习臂 `prompt_report.json:alpha_final`，精确相等；float32 往返逐位精确）。
- **实现语义**：新类 `PinnedAlphaResidualPromptNewTask(RP.ResidualPromptNewTask)`：`super().__init__()`（机制构造路径逐位不变，含隔离 RNG 生成器初始化）后，**删除** `prompt_gate` 参数并**注册同名 buffer** `prompt_gate = torch.tensor(PINNED_ALPHA, dtype=torch.float32)`。因此：
  1. `prompt_deltas`（继承，零改动）以 buffer 作 `self.prompt_gate` 参与 `α · scale · m`，公式逐字不变；
  2. α **不在** `named_parameters()` ⇒ `torch.optim.Adam(params=newtask.parameters())` **结构性地**不含 α（无梯度、无更新）；
  3. α **在** `state_dict`（buffer）⇒ 随 checkpoint 落盘，可逐位核验；
  4. `float(newtask.prompt_gate)` 可直接与 `PINNED_ALPHA` 做**精确相等**比较（float32 往返精确）。
- **训练全程不变**：初始（构造后）、每个 epoch 首 batch 探针、训练结束（best_state 载入后）、checkpoint 读取，四处 α == `PINNED_ALPHA` 逐位相等（PA1/PA2）。
- **shuffled 臂**：`PinnedAlphaShuffledResidualPromptNewTask(PinnedAlphaResidualPromptNewTask)`，只重写 `forward` 加必需参数 `cond_perm`（哨兵缺省 ⇒ 显式 `ValueError`，拒绝静默回退为正确条件；语义与 fbfff09 的 `ShuffledConditionResidualPromptNewTask` 逐字相同）。

### 2.2 错排构造（deterministic label-blind derangement；**逐字沿用 fbfff09 §2.1**）

- **对象**：每个 batch（batch 内行序 = 数据文件序，`shuffle=False`、batch_size=2000 协议默认）构造**错排** π：目标样本 i 的 prompt 条件特征取 `dnn_input[π(i)]`。
- **键**：`(split ∈ {"train","val","test"}, batch_index ≥ 0, n = batch size)`；同键任何时刻产生**同一个** π（缓存不重抽）。
- **种子推导（逐字，供独立复核重实现）**：`seed_material = sha256(f"{COND_PERM_SEED}|{split}|{batch_index}|{n}".encode("utf-8")).hexdigest()[:16]`；`rng = random.Random(int(seed_material, 16))`（Python `random` **局部实例**，不触碰全局 RNG）。`COND_PERM_SEED = 20261005`（冻结常量，**与 fbfff09 相同** ⇒ 同一张 deck，§5.2-PG7）。
- **算法（Sattolo，无不动点）**：`a = list(range(n))`；`for i in range(n-1, 0, -1): j = rng.randrange(0, i); a[i], a[j] = a[j], a[i]`。n ≥ 2 ⇒ 均匀 n-轮换（单循环）⇒ 不动点恒 0、双射；`n ≤ 1`：π = 恒等（退化，定义为恒等并如实记录；协议 batch=2000 ⇒ 预期 0 次；**batch-size-1 语义保留**，§4-I5）。
- **标签无关（label-blind）**：π 仅由键与 `COND_PERM_SEED` 决定；生成函数不接收任何标签、特征、预测或梯度。**边际保持**：每 batch 内条件特征多重集 == 目标特征多重集（双射）。
- **RNG 隔离**：π 生成只用局部 `random.Random`；**不消耗/不改变**全局 torch 与全局 Python RNG（运行期探针断言逐位不变）⇒ 训练 dropout、头初始化、数据顺序等随机流与 B/F_c **逐位一致**。

### 2.3 应用面（forward 语义）

- **F_c**：继承 `RP.ResidualPromptNewTask.forward`（正确 `dnn_input` 条件化；注入后 `super().forward` 委托逐字不变）。
- **F_s**：`forward(dnn_input, gen_rep, spec_reps, env_embs, cond_perm)`：`cond = dnn_input[cond_perm]`；`deltas, _ = self.prompt_deltas(cond, [gen_rep, *spec_reps])`；注入后以**正确的** `dnn_input` 委托 `NewTask.forward`（显式，不经 `super()`，防二次注入）。基头（`projection_network`/`gate_network`）与标签、目标、`h_s`、优化器更新**全部保持正确对应**。
- **同一预声明原则应用于 train/val/test**：F_s 的三 split 各用自己的键空间；训练循环、val 评估、test 评估、prompt 诊断一律经同一 `ConditioningShuffler` 取 π。
- **B/F_c 默认路径逐位不变（PG6 的对象）**：(a) `aliccp_benchmark/residual_prompt.py` 相对 `013e105` 钉死 blob **逐字节一致**（零文本适配）；(b) bench 接线在 `shuffler=None`（baseline/学习 correct 臂）时调用签名与行为**逐字节等价**（重建等式守卫）；(c) 等价性测试：**钉死头与学习头在同 α 下逐位等价**（§3 三层证明）。

### 2.4 明确不改变（禁止清单）

标签/目标/损失/优化器与更新规则（Adam、lr=1e-4、`get_l2_reg`）、数据顺序与 batch 组成、`h_s` 基表征、`super().forward` 委托、α 初值与语义（除学习性→钉死）、范数标定 `scale`、注入流集合 `("gen","spec_0","spec_1")`、hidden=16、生成器初始化（隔离 RNG）、`BOUND_TOL`、参照头、U1/U2 口径（历史 `+0.0055`）、协议文件与 `multitaskrec/*`。

---

## 3. 实现面（最小；TDD；等价证明）与适配清单

| 编号 | 文件 | 处置 | 守卫 |
|---|---|---|---|
| A1 | `aliccp_benchmark/residual_prompt.py` | **逐字节移植** `013e105:` 版（blob `674213f619c5d5039811c71242a7727118348daf`；LF sha256 `b3b93b3a…`）。**零文本适配** | 守卫测试：工作树 == 钉死 blob 逐字节 + LF sha + `git hash-object` == blob + 继承的 Census 祖本 AST 钉死（`TestMechanismPin` 原样移植） |
| A2a | `aliccp_benchmark/bench.py` | `013e105:` 版（blob `bf36476…`）**+ 文档化最小适配**：import `rp_pinned`；`variant ∈ {pinned, pinned-shuffled}` 时 ① 经 `RPP.build_*_newtask` 构造、②（shuffled 时）建 `ConditioningShuffler`、③ 用 `PinnedPromptAudit`、④ 训练/评估前向经 `RPP.forward_with_conditioning` 分派（`shuffler=None` 时调用**逐字**为原调用）、⑤ 评估/诊断/臂级判定分派到 `RPP.*` 包装、⑥ run_id 追加 variant 后缀、⑦ `prompt_report.json` 增 `pin` 段（shuffled 另增 `shuffle` 段）、⑧ `metrics.json`/`gate_report.json`/`config.json` 的臂字段沿用 | 守卫测试：重建等式（钉死文本 + 恰好替换 == 工作树）＋ `shuffler=None` 恒等分派测试（tiny 端到端输出逐位不变）＋ AST 差异集合恰为文档化集合 |
| A2b | `run_aliccp_benchmark.py` | `013e105:` 版（blob `229c25a1…`）**+ 恰好 3 行适配**：import 行（增 `rp_pinned as RPP`）；`--variant` choices 行（4 变体）；臂后缀行（改经 `RPP.run_id_suffix_for` 分派） | 守卫测试：行级替换重建等式（替换前断言旧行恰出现 1 次）+ 新 LF sha 钉死 |
| A3 | `aliccp_benchmark/tests/test_residual_prompt.py` | `013e105:` 同名测试（blob `768a477b…`）**恰 4 处**适配：(a) 模块 docstring；(b) `DOC_PATH` → 本文件；(c) `WHITELIST` → 本分支 10 项（逐项集合钉死）；(d) `TestPreregConstants.test_prereg_doc_tokens_and_summary_ledger` 的 token 表 → 本文件 token 表（`test_thresholds_and_baseline_reference` 免改：学习机制常量与 013e105 相同）。**其余逐类/逐方法/逐模块级语句不变** | 守卫测试：重建等式（docstring + `DOC_PATH` + `WHITELIST` + 该 1 个方法段替换）+ AST 差异集合 == 恰 1 个方法 + 模块级函数逐字 + 新 LF sha 钉死 |
| A4 | `aliccp_benchmark/rp_pinned.py`（新） | 消融本体：`PinnedAlphaResidualPromptNewTask` / `PinnedAlphaShuffledResidualPromptNewTask`（§2.1/§2.3）、`ConditioningShuffler`（§2.2 构造 + 运行期完整性记录 + 逐 split/总 digest；自 fbfff09 逐字移植）、`PinnedPromptAudit`（继承 `RP.PromptAudit`：G1 构造审计逐字继承 + PA1/PA2 钉死记录 + PA4 零 α 探针/激活探针；shuffled 时以真实 train/0 π 调用）、`forward_with_conditioning` 分派、诊断包装（复用钉死的 `PromptStats`/`effective_gate`）、`pinned_arm_verdict`（新 PA1–PA8 + U1/U2 + 分类；**不含** G2/G3/G5）、冻结常量（`PINNED_ALPHA`、`COND_PERM_SEED`、`MATERIAL_DELTA`、二级边界、判定树标签、deck 钉死 digest）、`analyze_runs`（只读三 run 记录 → §5 全量判定）+ `python -m` CLI（输出 JSON） | 守卫测试（本分支 `test_residual_prompt_pinned.py`）：§4 全部不变量 + 判定树真值表/边界 + 钉死恒常性 + 零 α 探针 + 构造/双射/不动点/确定性/隔离/标签无关 |
| A5 | `aliccp_benchmark/tests/test_residual_prompt_pinned.py`（新） | 本分支守卫 + 消融/接线/分析器测试（§4） | 本文件即为守卫 |
| A6 | `verify_pinned_prerun.py`（预注册前完整性核验；§1.5）与 `verify_rp_pinned.py`（运行后独立复核） | 只读脚本；**跟踪入库**（沿用 20-epoch/shuffled 先例；两处 WHITELIST 副本同步包含二者）；报告 JSON 落 `artifacts/aliccp_bench/audit/rp-pinned/` 与 F_s run 目录（gitignore）。前置脚本自 `fbfff09:verify_shuffled_prerun.py` 移植（输出目录/第 7 组三 run 链/新增第 8/9 组；钉死值 §1.5 不变）；复核脚本自 `fbfff09:verify_rp_shuffled.py` 移植（三 run 形态；记录一致性口径沿用） | 前置核验 N/N（§10.0）；复核脚本 §10.5 |

**被禁止的适配**：机制公式/初始化/门控语义/注入点/`BOUND_TOL`/G7 比值/U1 阈值/判定树/阈值；`protocol.py`/`metrics.py`/`multitaskrec/*`/`config.py`——零改动（静态守卫：`git diff --name-only 8133d32` ⊆ 白名单）。

**等价性证明口径（三层，全部由测试守卫执行）**：
1. **重建等式（文本级）**：钉死 blob（013e105）+ 恰好替换 == 工作树（机制/bench/CLI/移植测试）。
2. **机制 pin**：机制文件逐字节 == 钉死 blob + Census 祖本 AST 钉死（原样继承）。
3. **行为级**：① `shuffler=None` 分派恒等（learning correct/baseline 路径逐位不变）；② **钉死头 ≡ 学习头（同 α 逐位）**：`PinnedAlphaResidualPromptNewTask` 与 `RP.ResidualPromptNewTask`（其 `prompt_gate` 参数置为同 α、同生成器权重）在同一输入上输出**逐位相等**（correct 与 shuffled 两路径各测）；③ α=0 下钉死头与同权重参照 `NewTask` 逐位相等（PA4 运行期 + 单元测试）。

---

## 4. 不变量（由 `test_residual_prompt_pinned.py` 与移植测试强制，不接受人工目测）

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 基线/学习 correct 臂逐位不变：`shuffler=None` 分派 ≡ 钉死调用；学习臂端到端记录键集与钉死一致（除既有 `variant`/`rp_arm`/`prompt_hidden`） | 移植测试（构造恒等/初值恒等/tiny 端到端 purity）+ bench 重建等式 |
| I2 | 适配逐字守卫：A1/A2a/A2b/A3 的字节/重建等式钉死 | `git show` + sha256(LF) + 段级重建等式 |
| I3 | 学习机制语义不变：α 初值 0、恒等/界/活性带/参数预算测试全绿（移植测试原样通过） | 移植测试原样通过 |
| I4 | **α 钉死恒常性**：构造/训练前/训练后/checkpoint 四处 α == `PINNED_ALPHA` 精确相等；`isinstance(buffer)` 且非参数；`requires_grad=False` | 单元测试（构造、fill/zero 探针后恢复）+ 运行期 PA1/PA2 |
| I5 | **错排构造（含 batch-size-1）**：双射（`sorted(π)==range(n)`）∧ n≥2 不动点==0 ∧ **n≤1 ⇒ 恒等** ∧ 同键同 π（跨实例逐位）∧ 不同键（抽样）不同 π ∧ `COND_PERM_SEED` 冻结 | 构造单元测试（n ∈ {0,1,2,3,7,100,2000}，多键）+ 钉死头 batch=1 前向测试 |
| I6 | RNG 隔离：π 生成前后 `torch.get_rng_state()` 与 `random.getstate()` 逐位不变；连续生成不改变后续正确路径输出 | 探针测试 + tiny 端到端 |
| I7 | 标签无关：π 生成函数只接收键（构造签名）∧ 同键在任何数据/标签下产生同 π | 签名断言 + 键复算测试 |
| I8 | **钉死头 ≡ 学习头（同 α 逐位）**：两实现（correct/shuffled 路径）同权重同输入输出逐位相等；α=0 下钉死头 ≡ 参照 `NewTask` 逐位 | 行为测试（含零 α 探针恢复校验） |
| I9 | **优化器排除**：α 不在 `named_parameters()`；Adam 的 param_groups 覆盖 `named_parameters()` 全量且不含 α（对象同一性 `is` 比较） | 构造单元测试 |
| I10 | forward 语义（shuffled 钉死头）：缺 `cond_perm` ⇒ 抛错（哨兵）；恒等置换 ⇒ 与钉死 correct 头逐位相等；非恒等置换 + α≠0 ⇒ 输出改变；范数界在置换下保持（ratio ≤ \|α\|） | 行为测试 |
| I11 | 判定树/边界：§5.5 全部标签真值表；`GAP` 恰 `+0.001` ⇒ `SUPPORTED`（闭），`−1e-12` ⇒ `NOT_SUPPORTED`；`Δval_gap` 恰 0 ⇒ 验证不一致；二级分类边界 `+0.001`/`−0.02`（闭端）；每条 INVALID 子原因可达；PA1–PA8 每条 FAIL 可达（夹具） | 边界/真值表测试（CPU 夹具） |
| I12 | 分析器只读：不改任何 run 产物；identity/对照 mismatch ⇒ `INVALID`；输出键齐全（§5 全部判定 + 全部分量 + 描述量）；静态守卫：相对 `8133d32` 的全部跟踪改动 ⊆ 白名单；`protocol.py`/`metrics.py`/`multitaskrec/*`/`config.py` 零 diff；预注册文档 token 钉死（seed、stage1_id、`PINNED_ALPHA`、`COND_PERM_SEED`、阈值、判定标签、deck digest、适配清单 blob） | 夹具端到端 + 篡改 fixture + `git diff --name-only` + 文档文本断言 |

CPU 极小夹具不构成任何性能证据，只验证语义与接线。

---

## 5. 预注册判据（看到结果前写死；看到结果后不得修改）

对照定义：**B = 本实验新跑的配对基线臂**；**F_c = 钉死 α correct 臂**；**F_s = 钉死 α shuffled 臂**；C = 已提交学习 correct 臂（只读对照）。`GAP := test(F_c) − test(F_s)`；`G_c := test(F_c) − test(B)`；`G_s := test(F_s) − test(B)`；`R := G_s / G_c`；`Δval_gap := val(F_c) − val(F_s)`；`Δval_c := val(F_c) − val(B)`；`Δval_s := val(F_s) − val(B)`。

### 5.1 前置条件（全部必须为真；任一假 ⇒ `INVALID`，不解读效用）

| 编号 | 判据 | 落盘 |
|---|---|---|
| **ID**（identity） | 三 run：`stage1_id` 相同且 == 钉死值；`model_seed` 相同且 == 1688723740；`epochs==5` ∧ `patience==2` ∧ `tag=="short"`；`dirty==false`（三 run 记录时）；B `variant=="baseline"` 且 run_id 无后缀；F_c `variant=="residual-prompt-pinned"` 且 run_id 以 `-rpp` 结尾；F_s `variant=="residual-prompt-pinned-shuffled"` 且 run_id 以 `-rpps` 结尾；两处理臂记录参照路径 == 钉死路径且文件 sha256 == `90ee06da…`；三 run `stage1_id` 的 backbone/env/fingerprint sha 与钉死值一致 | `rp_pinned_compare.json:identity` |
| **REP**（基线复现） | B 的逐 epoch val 轨迹、best_epoch、best_val、test、gate_mean 与 `79b5e07` 记录**逐位相等** 且 B 的 `newtask.pt` sha256 == `90ee06da…`（A2/A4 确定性预期成立；失败 ⇒ 环境偏离历史链，对照组不可比） | 同上 `reproduction` |
| **A**（协议） | 三 run 的 A1/A2/A4/A5/A6 全 PASS（A3 SKIP 视为通过）；任一 FAIL ⇒ 该 run 比较作废 | 三 run `gate_report.json` |
| **PA**（钉死 α 与结构机制门禁） | F_c 与 F_s 的 `rp_arm`：**PA1–PA8 全 PASS**（§5.3；判据先于结果写死、不放宽） | 两 run `metrics.json:rp_arm` |
| **PG**（置换门禁） | F_s：**PG1–PG7 全 PASS**（§5.2；PG7 = deck 重放 == 历史 digest）；PG6 的构建期部分为分支级证据（机制字节一致 + 恒等分派 + 等价测试） | F_s `prompt_report.json:shuffle` |
| **CC**（对照链与钉死来源） | 79b5e07 / 013e105-rpg / 9d26bc8-rpgs 的记录值 == §1.5 钉死常量；对照 run 文件 sha == 钉死（复核脚本执行）；**`PINNED_ALPHA == 013e105-rpg:prompt_report.json:alpha_final`（精确相等）且该文件 sha256 == `185df4d0…`** | `rp_pinned_compare.json:comparator` |
| **DD**（deck 钉死） | F_s `shuffle.report` 的逐 split digest ×3 + total digest == §1.2-A7 历史钉死值（逐位） | 同上 `deck` |

### 5.2 置换门禁（PG1–PG7；运行期 PG1–PG5 + PG7 由 F_s 记录 → 复核脚本独立重推）

| 编号 | 判据 | 口径 |
|---|---|---|
| **PG1 无不动点** | 运行期生成的每个 π（n≥2）不动点数 == 0；n≤1 的恒等（若有）逐项记录 | 运行期逐 π 检查 + 记录计数（预期 0 次退化） |
| **PG2 双射/边际保持** | 每个 π 为 `range(n)` 的置换；逐 split：条件行总覆盖 == 目标行总数 == split 大小 | 运行期 + 独立重推 |
| **PG3 确定性/可复现** | 每个 π 以**全新** `random.Random` 重生成逐位相等（运行期探针）；**独立复核脚本以 §2.2 规格重实现并对逐 split digest 与总 digest 逐位复算** | 运行期 + verifier 重推 |
| **PG4 RNG 隔离** | 每次生成前后 `torch.get_rng_state()` 与 `random.getstate()` 不变 | 运行期探针 |
| **PG5 标签无关/键唯一** | π 仅由键决定：独立复核以（`COND_PERM_SEED`, split, batch_index, n）重推全键空间 digest == 记录 digest；同键不同 n ⇒ 拒绝（`ValueError`） | verifier 重推 + 单元测试 |
| **PG6 correct 默认逐位不变（构建期）** | 机制文件 LF sha == 钉死；bench 重建等式；`shuffler=None` 恒等分派 + 恒等置换等价测试全绿 | 构建期（测试 + verifier 独立重哈希机制文件） |
| **PG7 deck 重放一致（新）** | F_s `shuffle.report`（逐 split digest ×3 + total）== §1.2-A7 历史钉死 digest（同一 `COND_PERM_SEED` 键空间 ⇒ 结构性必然；逐位比对） | 运行期记录 + 分析器 + verifier 重推 |

### 5.3 钉死 α 结构机制门禁（PA1–PA8，pinned；**替换**学习门禁 G2/G3/G5——a priori 理由见 §1.4）

| 编号 | 判据 | 口径 |
|---|---|---|
| **PA1 钉死身份** | `prompt_gate` 为 buffer（`isinstance(…, nn.Parameter) == False` 且 `requires_grad == False`）；构造时、训练后（best_state 载入后）、checkpoint（`newtask.pt` 独立读取）三处 `float(prompt_gate) == PINNED_ALPHA`（精确相等） | 运行期记录 + verifier 读 checkpoint 重推 |
| **PA2 优化器排除** | α 不在 `named_parameters()`；训练 optimizer 的 `param_groups` 覆盖全部 `named_parameters()` 且**不含** α（对象同一性）；逐 epoch 首 batch 探针 `alpha_grad_norm is None` 且 `alpha == PINNED_ALPHA`；训练前后 α 逐位不变 | 运行期 + 单元测试（I9） |
| **PA3 头身份（G1 等价）** | 从构造前 RNG 现场重建 baseline `NewTask`：共享参数逐位一致 ∧ RNG 端点一致 ∧ 新增 `state_dict` 键恰 = `{prompt_gate, prompt_generator.0.bias, prompt_generator.0.weight, prompt_generator.2.bias, prompt_generator.2.weight}` | 运行期（继承 `PromptAudit` 构造审计）+ 单元测试 |
| **PA4 注入纯度与激活** | (i) α 临时置 0（探针内；恢复后逐位校验 `alpha_restored_exact`）⇒ 与同共享权重参照头在真实首 batch 上**逐位相等**（`zero_alpha_bit_identical`；F_s 用真实 train/0 π）；(ii) α 钉死值下 `max_abs_diff > 0`（干预自首步即激活） | 运行期首 batch 探针 |
| **PA5 范数界（S1）** | 逐流 `ratio_max ≤ \|PINNED_ALPHA\| + 1e-6` | val 诊断（fp64 流式） |
| **PA6 跨流一致（S2）** | 三流 `ratio_mean` 两两 spread ≤ 1e-6（构造性；与条件来源无关） | val 诊断 |
| **PA7 无坍缩** | `pred_std > 0 ∧ pred_std ≥ 0.5 × ref_pred_std`（G7 原语义） | val 诊断 + 参照头 |
| **PA8 参数预算/钉死核算** | 可训练 prompt 键恰 4（generator；`new_params_total == 2384` == 2385−1）；`head_params == 8129`；α buffer numel == 1 且落盘于 checkpoint | `param_report` + checkpoint 读取 |

**分类映射（先于结果写死）**：`not PA1 or not PA2 or not PA8` ⇒ `MECHANISM_FAIL / PINNED_ALPHA_INVALID`；`not PA3 or not PA4` ⇒ `MECHANISM_FAIL / INVALID_IMPLEMENTATION`；`not PA5` ⇒ `MECHANISM_FAIL / NORM_BOUND_VIOLATION`；`not PA6` ⇒ `MECHANISM_FAIL / CROSS_STREAM_INCONSISTENT`；`not PA7` ⇒ `MECHANISM_FAIL / PREDICTION_COLLAPSE`；α 恒常性单独列 `pin` 段（§7）。**学习活性带（原 G5）不设、`ratio_mean` 不判定**（描述量）。

### 5.4 效用分量（判定的原子量；全部报告）

| 量 | 定义 | 预声明口径 |
|---|---|---|
| `GAP` | `test(F_c) − test(F_s)` | **因果主量**；实质为正：`GAP ≥ +0.001`（`MATERIAL_DELTA`） |
| `G_c` | `test(F_c) − test(B)` | 报告 + 二级分类 + 保留比 |
| `G_s` | `test(F_s) − test(B)` | 报告 + 二级分类 + 保留比 |
| `R` | `G_s / G_c`（`G_c ≤ 0` ⇒ NA） | 报告（保留比） |
| `Δval_gap` | `val(F_c) − val(F_s)` | 验证方向：`> 0` ⇒ `validation_agreement = true`（**进 §5.5 判定树合取**——两臂同源对照，val 方向一致为因果对比的佐证；预声明其不进 PA/PG 门禁） |
| `G_c_ge_historical_0.0055` | `G_c ≥ +0.0055` | §5.7 报告项（不判定） |
| `G_s_ge_historical_0.0055` | `G_s ≥ +0.0055` | 同上 |
| `retention_correct_vs_hist` | `G_c / DELTA_TEST_CORRECT`（vs 学习 correct `+0.008103113327467715`） | 报告（增益保留比） |
| `retention_shuffled_vs_hist_correct` | `G_s / DELTA_TEST_CORRECT` | 报告 |
| `shuffled_pinned_vs_learnable` | `G_s / DELTA_TEST_SHUFFLED_HIST`（vs 学习 shuffled `+0.0014787611000699474`） | 报告 |
| `gap_sign` | `sign(GAP)`；`GAP ≤ 0` ⇒ `shuffled_matches_or_exceeds_correct = true`（报告） | 报告 |

**阈值来源与读法（预声明）**：`+0.001` = 用户指定的最小实质差异，与既有分支二分类边界一致；test 1M 前缀 BSI AUC 的 SE ≈ 0.0025 @AUC0.7（沿用 20-epoch §5.1 口径）⇒ `+0.001` ≈ 0.4×SE，为**分类标签**而非显著性强断言；单 seed、每臂 1 run、无 run-to-run 噪声估计（A4 沿用）。

### 5.5 因果判定树（**先于结果写死**；优先级自上而下，标签集固定为以下三种）

| 优先级 | 判定 | 条件 | 含义（写死） |
|---|---|---|---|
| 1 | **`INVALID`** | 任一前置条件假（ID / REP / A / PA / PG / CC / DD 任一失败） | 不解读效用；`subreason` 按首个失败项机械给出（`IDENTITY_MISMATCH` / `BASELINE_REPRODUCTION_FAILED` / `PROTOCOL_INVALID` / `PINNED_ALPHA_INVALID` / `MECHANISM_FAIL` / `COMPARATOR_MISMATCH` / `PERMUTATION_INVALID` / `DECK_MISMATCH`） |
| 2 | **`SAMPLE_CONDITION_SUPPORTED`** | `GAP ≥ +0.001`（闭）∧ `Δval_gap > 0` ∧ 前置条件全真 | 在标量幅度匹配之下，条件对应承载实质效用：correct 臂相对 shuffled 臂 test 增量 **≥ +0.001** 且验证方向一致。子类（**报告、不改变本判定**）：`subreason = FULL_GAIN_REQUIRES_ALIGNMENT` 若 `G_s < +0.001`（错配下不再实质为正）；`subreason = PARTIAL_ALIGNMENT_CONTRIBUTION` 若 `G_s ≥ +0.001` |
| 3 | **`CONDITION_ALIGNMENT_NOT_SUPPORTED`** | 前置条件全真 ∧ **实质规则失败**：`GAP < +0.001` **或** `Δval_gap ≤ 0` | 效用**不依赖**样本级条件对齐（或证据方向不一致）：与"额外参数 / 全局扰动 / 范数-门控效应"等 nuisance 解释一致。`subreason = GAP_BELOW_MATERIALITY`（`GAP < +0.001`）或 `VALIDATION_DISAGREEMENT`（`GAP ≥ +0.001` ∧ `Δval_gap ≤ 0`）；`GAP ≤ 0` 时附注 `shuffled_matches_or_exceeds_correct=true`（报告） |

**读法声明**：判定树**不设**"correct 臂相对基线实质为正"的前置（学习臂的 `G_c^hist ≥ +0.001` 已由 §1.5 钉死；钉死臂的 `G_c`/`G_s` 作为分量报告，不进门禁）——本实验只回答"同幅度下对齐是否贡献实质差异"。

### 5.6 二级效用分类（用户指定；对 F_c 与 F_s **各自**相对 B 的 `Δtest` 机械分类；与 §5.5 并存、不覆盖）

| 类别 | 判据（精确边界） |
|---|---|
| `POSITIVE_IMPROVEMENT` | `Δtest ≥ +0.001` |
| `NO_CLEAR_IMPROVEMENT` | `−0.02 < Δtest < +0.001`（严格双开） |
| `CLEAR_DEGRADATION` | `Δtest ≤ −0.02` |

**`NO_CLEAR_IMPROVEMENT` 的机械 headroom 评估（仅当某臂落入该区间；规则先于结果写死；结构沿用 20-epoch §5.4，因子按本实验适配）**：

| 因子 | 机械口径 | 取值 |
|---|---|---|
| 钉死机制活性 | `\|α_pin\| > 0` ∧ `geff_std > 0` ∧ 三流 `ratio_mean > 0`（取自该臂 JSON） | `PIN_ACTIVE` / `INACTIVE` |
| 验证方向 | `Δval_arm > 0` | `POSITIVE` / `NON_POSITIVE` |
| epoch 轨迹 | 该臂 `best_epoch == epochs_run == 记录 epochs`（未早停）⇒ 右删失；否则早停 | `RIGHT_CENSORED_STILL_IMPROVING` / `EARLY_STOPPED` |
| 距正分类余量 | `gap_to_positive_threshold = +0.001 − Δtest_arm`（>0） | 数值 |
| 保留比 | `Δtest_arm / DELTA_TEST_CORRECT`（描述性；与 §5.4 互参） | 数值 |

### 5.7 历史 `+0.0055` 判定单列保留（不改写）

- 历史成效门槛 `+0.0055` 为学习 correct 臂效用口径（seed-2 复现预注册 §3-U1）；本实验两处理臂的 `G_c ≥ +0.0055`、`G_s ≥ +0.0055` 作为**报告项单列**（不参与 §5.5/§5.6 判定）。
- 历史判定原样引用：seed1 `VALID_NEGATIVE`（+0.005394543882337954）、seed2 `VALID_POSITIVE`（+0.008103113327467715）、10-epoch `PERSISTS`、20-epoch `NOT_PERSIST`/claim=true/`ABLATION_ELIGIBLE`、shuffled 学习臂 `INVALID/MECHANISM_FAIL`。本实验不重判任何历史结论。

---

## 6. 运行规程（恰好三次；清洁树纪律；前台）

**运行次数**：B stage2 **恰好 1 次**；F_c stage2 **恰好 1 次**；F_s stage2 **恰好 1 次**；分析（`analyze_runs`，只读）**恰好 1 次**；运行后独立复核 **恰好 1 次**。不重训 Stage-1；任何历史 run 不重跑；学习臂（C、历史 shuffled）不重跑（§5 对照仅取已提交记录）。

**清洁树纪律（run 间提交，沿用 20-epoch §4 / fbfff09 §6）**：① 提交实现与测试（C2）→ `git.dirty=false`；② 跑 **B**（run 记录 commit=C2）；③ 提交 B 的 SUMMARY 追加行（C3，仅 `artifacts/aliccp_bench/SUMMARY.md` 变化）；④ 跑 **F_c**（run 记录 commit=C3）；⑤ 提交 F_c 的 SUMMARY 行（C4）；⑥ 跑 **F_s**（run 记录 commit=C4）；⑦ 核验 `git diff C4 C2 -- . ':(exclude)artifacts/aliccp_bench/SUMMARY.md'` 为空；⑧ 分析 ×1 → 复核 ×1 → C5（§10 回填 + F_s SUMMARY 行）→ push。

```powershell
# cwd = 本 worktree 根；本 worktree 无 .venv，用主树解释器；dataset 经 junction 复用（只读）

# 0) 测试（先红后绿；全部不变量）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests

# 1) B 配对基线臂（唯一一次；协议默认：epochs 5 / patience 2 / tag short）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag short --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f

# 2) F_c 钉死 α correct 臂（唯一一次；同产物、同 seed、同预算；α=+0.07101669907569885、钉死参照头）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag short --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f `
    --variant residual-prompt-pinned `
    --prompt-reference-newtask artifacts/aliccp_bench/runs/20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt

# 3) F_s 钉死 α shuffled 臂（唯一一次；同产物、同 seed、同预算、同 α；+ 同 deck 错排）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag short --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f `
    --variant residual-prompt-pinned-shuffled `
    --prompt-reference-newtask artifacts/aliccp_bench/runs/20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt

# 4) 分析（纯分析，恰一次；只读三 run 记录）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m aliccp_benchmark.rp_pinned `
    --baseline-run <B run_id> --arm-correct-run <F_c run_id> --arm-shuffled-run <F_s run_id>

# 5) 运行后独立复核（只读；JSON 重推 + 文件哈希 + 错排独立重实现 + deck 重放）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe verify_rp_pinned.py `
    --baseline-run <B run_id> --arm-correct-run <F_c run_id> --arm-shuffled-run <F_s run_id>
```

- **前台执行并等待至完成**（先例：后台执行被会话终止杀死）；开跑前提交实现（§6 步骤①）；每臂 run 启动时 `git.dirty` 必须为 false。
- **无效执行处置**：仅当 run 因**工具性原因**（崩溃/中断/环境故障、产物未完整落盘）无效时，保留现场、记录原因后可重跑一次；因**结果原因**不构成无效，不重跑。无任何事后调参（`PINNED_ALPHA`/`COND_PERM_SEED`/构造/阈值/参照/判定树均不得动）。
- **墙钟预算**：每臂 ≈2–4 min（short 实测 156–198 s + 钉死/错排开销），总计可控。

---

## 7. 必录诊断（落盘 run 目录；class C，不判定）

| 量 | 字段 | 用途 |
|---|---|---|
| **钉死 α 证明** | `pin`：`pinned_alpha`、`source_run`/`source_field`/`source_file_sha256`、`is_parameter=false`、`requires_grad=false`、`in_named_parameters=false`、`alpha_at_construction`、`alpha_after_training`、`alpha_restored_exact`、`optimizer_covers_named_parameters`、`optimizer_contains_alpha=false`、`optimizer_params_total` | PA1/PA2 + 固定 α 证据 |
| 构造身份 | `construction_identity`：共享参数逐位/RNG 端点/新增键集（含 buffer） | PA3 |
| 注入纯度 | `init_forward`：`zero_alpha_bit_identical`、`zero_alpha_max_abs_diff`、`pinned_max_abs_diff`、`alpha_restored_exact`（shuffled 另记 `cond_perm`=真实 train/0 键） | PA4 |
| 梯度探针 | `grad_probe`（逐 epoch 首 batch：`alpha`、`alpha_grad_norm`(=None)、`generator_grad_norm`——**首 epoch 即 > 0**，与学习臂的 0 对照为描述差异） | PA2 + 描述 |
| 残差/基范数比 | 逐流 `ratio_mean/std/max`、`delta_norm_mean`、`h_norm_mean` | PA5/PA6 + 描述（对照：学习 correct 0.02814；学习 shuffled 0.00317） |
| 余弦方向 | 逐流 `cos_mean` | 描述（对照历史：correct +0.01287/+0.08432/+0.05899；shuffled +0.06290/+0.13314/+0.07916） |
| 逐样本门控 | `gate.geff_mean/std/min/max` | 描述 + headroom |
| 预测离散度 | `dispersion` + `reference_dispersion`（M0 口径） | PA7 + 描述 |
| 源/头门 | `gate_mean` + Stage-1 环境上下文（`cluster_events`、`env_acc`） | 落盘；B3 原样披露 |
| 参数清单 | `params`：4 键 generator、`new_params_total=2384`、`head_params=8129`；α buffer numel=1 | PA8 |
| 错排完整性 | `shuffle`：逐 split 键数/行数/digest、总 digest、不动点计数（必须 0）、双射、重生成一致、RNG 隔离、退化计数；**digest 与历史钉死值逐位比对** | PG1–PG5、PG7 |
| 错排样例 | `shuffle.samples`：每 split 前 3 键 π 前 16 元素 + 首/末 5 行错配对 | 人审 + 复现 |
| 臂级判定 | 两处理臂 `arm`：PA1–PA8 + U1/U2 + 分类 + `pin` 段（shuffled 另含 PG 段） | §5.3 机械分类 |
| 配对判定 | `rp_pinned_compare.json`：§5 全部前置条件、分量（`GAP`/`G_c`/`G_s`/`R`/`Δval_gap`/全部保留比）、判定树、两臂二级分类、headroom（若适用）、deck 比对、context 表 | §5.5/§5.6 |
| 停止/删失 | 三 run 的 `best_epoch`/epochs、是否早停、逐 epoch 轨迹 | 描述 |
| provenance | 三 run 的 `run_id`/`commit`/`git.dirty`/`wall_seconds`/`peak_vram_mb`/`hard_pass`/A 类/B 类（继承披露） | §10 记录 |

---

## 8. 局限（预声明）

1. **单 seed、每臂 1 run**：无 run-to-run 噪声估计（A4 沿用）；`+0.001` 为分类标签而非显著性强断言（≈0.4×SE）。
2. **单一 α 值**：钉死 α = correct 臂 5-epoch 终值；结论条件于 α = +0.07101669907569885，不外推到其它幅度（无 α 扫描——§9）。
3. **单一错排抽样**：F_s 检验的是与 fbfff09 **同一张**冻结 deck 的效应（§5.2-PG7 结构性复用）；不回答"错排分布上的平均效应"。若结果落在 `SAMPLE_CONDITION_SUPPORTED`，稳健性需后续独立复现（换 deck seed 的第二张错排，§9.3 预声明建议）。
4. **batch 内错排（窗口局部）**：条件源取自同 batch（同 2000 行窗口）；对齐度高于"全库随机错配"。冻结构造；全局错配为另一实验（§9.3）。
5. **α 钉死但有效幅度仍含学习组分**：逐样本有效幅度 `|α|·||m||/√d` 中 `m`（生成器方向）在两臂仍被学习 ⇒ 若两臂 `ratio_mean` 出现实质差异（预声明：相对差 > 5% 视为实质），则"对齐 vs 幅度"仍残留一个**可量化**的幅度分量；报告两臂 `ratio_mean` 与相对差，判定树不含该量（描述性披露，不作门禁）。
6. **学习性 vs 钉死的对照被混淆（预声明）**：F_c vs 学习臂 C 差异**不止** α 的学习性——α 学习臂中生成器首步梯度为 0（∂loss/∂m ∝ α=0），钉死臂中生成器**首步即得梯度**（`generator_grad_norm > 0`）⇒ 生成器训练轨迹不同。**本实验对"学习 vs 钉死门控"问题只提供描述性证据（不可作因果归因）**；若未来需要该结论，须另立预注册（如 α 冻结在前 k 步后解冻等设计）。
7. **判定依赖单一汇总统计**（test 1M 前缀 BSI AUC 差值）；无独立重评测 pass（判定取 run 记录值；正确性依赖代码路径守卫 + 同类路径既有重评测先例）。
8. **in-run `rp_arm.U` 的配对对象是历史 5-epoch 常量**（class C 诊断）；因果判定只用 §5 的 B/F_c/F_s 配对。
9. **B4 若 FAIL 为继承缺陷**（Stage-1 共享语义）；`hard_pass` 不参与有效性定义（同先例）。
10. **单数据集（AliCCP）、单 seed**：结论不推及 CensusIncome / ByteRec、不推及其它 seed / 预算点。
11. **本实验不是"更优方法"主张**；是机制因果消融的一次性判定。

---

## 9. 非目标与止损

1. 不做：其它消融（无条件的固定向量 prompt、跨 split 条件、label 置换——**明确排除**：本实验不动标签；学习 shuffled 变体**不接线、不运行**，历史 run 只读对照）、不写论文/图表、不做聚类、不做 Census、不跑 `full` tag、不做 FLOPs、不扩 seed、不重跑任何历史 run、**不做 α 扫描**。
2. 不因结果修改：`PINNED_ALPHA`、`COND_PERM_SEED`、错排构造/算法、`MATERIAL_DELTA`、二级边界、判定树、§5.1–§5.5 任何规则、参照文件、clean-tree 纪律。
3. **后续独立消融的预声明建议（本分支不启动，仅登记）**：
   - 若 `SAMPLE_CONDITION_SUPPORTED` ⇒ 下一步最优先：**第二张独立错排复现**（仅更换 `COND_PERM_SEED`，其余逐字节不动），排除单张 deck 抽样伪影；
   - 若 `CONDITION_ALIGNMENT_NOT_SUPPORTED` ⇒ 下一步最优先：**无条件固定条件对照**（把逐样本 `P(x_i)` 换成与样本无关的常量方向，α 与参数预算匹配），定位增量由哪个非对齐通道（参数/全局扰动/范数-门控）承载；
   - **学习 vs 钉死门控**：两分支共同——本实验的该对照被 §8-6 混淆；如需该结论，另立预注册（如 α 延迟解冻/部分冻结设计），不在本分支做；
   - `INVALID` 情形下先修复执行/实现，不启动任何后续消融。

---

## 10. 结果（运行后回填，不回写第 0–9 节）

（待运行后回填：§10.0 非结果核验与执行记录；§10.1 三条 run 与臂级身份；§10.2 判定；§10.3 门禁与诊断；§10.4 错排完整性；§10.5 独立复核；§10.6 纪律核对；§10.7 解读与下一消融建议。）

---

## 11. 非新颖性声明

构件全部为既有方法族（prompt tuning、FiLM 族条件化调制、adapter/LoRA 残差、门控残差、范数归一、条件置换/错排对照为消融常用手段），先行工作清单沿用 seed1 设计 §11（写作引用前必须逐条核实，本会话未做系统检索）。本文件**不**声称独立于先行工作；可报告内容仅限本仓库冻结协议下的一次性判定。

---

## 12. 偏离披露（预声明 + 运行后补）

- **预注册后澄清提交（C1b）**：在本分支任何实现与任何 run 之前，修正 §1.1（补 seed-replication 预注册号 `e931cd7`）、§1.2-A1（补全 9d26bc8-rpgs 完整 run id）、§3-A2b（CLI 适配 2 行 → 3 行：import/choices/后缀）；判据、阈值、构造、判定树、运行规程**零改动**（同类先例：fbfff09 分支 C1b `768ba8b`）。
- **分支工作流**：自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；机制经 §3 三层守卫钉死等价；协议不合并 `master`；push 经用户显式指示（本任务含 "commit and push"）。
- **分支本地 SUMMARY 系谱**：本分支 `SUMMARY.md` 基线含 smoke 行与 seed1 基线行（`8133d32` 携带）；其余历史行记录在各自分支，本分支**不复制**其它分支行，只由本实验 run 追加自己的行（B + F_c + F_s）。
- **产物/历史 run 只读复制**：§1.5 所列产物、参照头、三个历史 run（79b5e07 / 013e105-rpg / 9d26bc8-rpgs）、前缀指纹均自 shuffled worktree **只读复制**入本 worktree（逐文件 sha 相等）；dataset 经 junction 复用 main tree（只读、gitignore）。
- **非结果文件（未跟踪）**：`artifacts/aliccp_bench/audit/rp-pinned/` 报告 JSON、`artifacts/aliccp_bench/logs/` run 日志、run 目录内 `rp_pinned_compare.json` / `verify_report*.json`。
- **与本分支前置分支的关系**：学习 shuffled 消融（fbfff09）不移植入本分支运行时（无 `VARIANT_SHUFFLED` 接线）；错排机制作为 F_s 的组成部分移植入 `rp_pinned.py`（同一规格、同一 deck）。
- **执行记录（运行后补）**。
- **跑后代码改动披露（运行后补）**。
- **下一消融登记（运行后补）**。

---

**改动清单（白名单；静态守卫以 `8133d32` 为基准）**

| 文件 | 处置 |
|---|---|
| `docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-alpha-pinned-condition-design.md` | 本文件（C1 单独提交） |
| `aliccp_benchmark/residual_prompt.py` | 新增（A1：逐字节 == `013e105` 钉死 blob） |
| `aliccp_benchmark/rp_pinned.py` | 新增（A4：消融本体 + 判定 + 分析 CLI） |
| `aliccp_benchmark/bench.py` | A2a：== `013e105` + 文档化最小适配（钉死分派；`shuffler=None` 恒等） |
| `run_aliccp_benchmark.py` | A2b：== `013e105` + 文档化 2 行适配（variant choices + 后缀分派） |
| `aliccp_benchmark/tests/test_residual_prompt.py` | 新增（A3：== `013e105` 恰 4 处适配） |
| `aliccp_benchmark/tests/test_residual_prompt_pinned.py` | 新增（A5：本分支守卫 + 消融/接线/分析器测试） |
| `verify_pinned_prerun.py` | 新增（A6：预注册前完整性核验） |
| `verify_rp_pinned.py` | 新增（A6：运行后独立复核） |
| `artifacts/aliccp_bench/SUMMARY.md` | 只追加 3 行（runner 自动） |
| 其余全部（`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`metrics.py`、既有测试三件、`AliCCP_*.py`、`CensusIncome_*.py`、`baseline/*`、`mask/*`、`analysis/*`） | **零改动**（静态守卫断言 `git diff --name-only 8133d32` ⊆ 白名单） |
