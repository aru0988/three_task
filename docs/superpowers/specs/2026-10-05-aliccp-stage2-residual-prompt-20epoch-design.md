# AliCCP 阶段 2 范数受控残差 Prompt（可学习门控）在 20-epoch 预算下的持久性检验（seed-2 单配对；终止判定点）

- **状态**：预注册已写死（本文件在本分支任何实现、任何 run **之前单独提交**，commit C1）。结果只在第 10/12 节以新增小节回填；第 0–9 节的判据、数字、预算与运行规程不得在看到结果后改动。
- **日期**：2026-10-05
- **适用分支**：`exp/aliccp-stage2-residual-prompt-20epoch`（自 `infra/aliccp-fair-benchmark` @ `8133d32` **独立拉出**；不从任何 exp 分支拉出、不继承其运行时状态；不合并 `master`、不触碰其它 worktree；`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`aliccp_benchmark/metrics.py`、既有测试三件、master 脚本与 `baseline/*` 零修改）。
- **被检验对象（只读引用；系谱钉死）**：残差 prompt 机制 = `exp/aliccp-stage2-residual-prompt-seed-replication`（预注册 `e931cd7`，机制系谱文档 `docs/superpowers/specs/2026-10-04-aliccp-stage2-residual-prompt-seed-replication-design.md`）@ `013e105:aliccp_benchmark/residual_prompt.py`（blob `674213f619c5d5039811c71242a7727118348daf`，LF sha256 `b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc`；其自身为 seed1 线（预注册 `221580a` / 实现 `79ddefa`（blob `5dc7158ca999c9e7e6217a999c37869231411549`）/ 结果 `3e2f083`）的 A1′ 适配移植，机制系谱终点为 Census 祖本 `99b9510`，blob `107221b26382da7fd44990e167d9a61da2680d92`）。
- **上游协议（只引用、不修改）**：
  1. `docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（AliCCP 公平评测协议）——前缀预算、种子、Stage-1 产物、真冻结三件套、A/B/C 门禁、SUMMARY 台账全部沿用；
  2. `exp/aliccp-stage2-attenuation-longer-budget` @ `47ecb0c`（下称"更长预算协议"）——已建立的更长预算协议：唯一变化 = Stage2 max epochs 5→10 与 patience 2→3（tag `"long"`）、判定结构 C1/C2/C3 → `PERSISTS`/`NOT_PERSIST`/`INVALID`、`commit-between-runs` 清洁树纪律——**逐字沿用**；
  3. `exp/aliccp-stage2-residual-prompt-longer-budget` 预注册 `d990cb0`（C1b 澄清 `296e03a`）/ 实现 `2b1d585` / 结果 `e46e5d2`（10-epoch 持久性检验；`PERSISTS`）——本实验的**直接前驱**与唯一运行规程模板；
  4. `exp/aliccp-stage2-residual-prompt-five-seed` @ `0935257`（五 seed canonical 配对；20-epoch 立项条件 `TWENTY_EPOCH_CONDITION_SATISFIED` 的来源）。
- **定位声明**：本实验是**稳健性 / 替代解释检验的终止判定点**（残差 prompt 的正增量在 10-epoch 判定点仍成立但收窄；20 = 10 的 2×，为本线**最后一次**预算延长），不是新方法，不主张任何新颖性；不改机制、不改系数、不调参、不换结构、不改损失、不放宽任何门限。与 10-epoch 持久性检验相比**唯一的变化 = Stage2 max epochs 10→20**（patience 3 不变；§3）。结论口径：seed-2 残差 prompt 的正增量在 20-epoch 判定点是否按 §5 三层规则（历史阈值持久性 / 二级分类 / 正持久性声明）成立，以及本线是否具备进入后续独立消融的条件（§5.6）。

**提交时序（预注册纪律）**：C1 本文件（审计 + 预注册，单独提交）→ C2 实现 + 测试（TDD；两 run 的 `commit` 字段即该提交）→ C2b baseline 臂 run（前台）→ C3 baseline SUMMARY 行提交 → C3b 处理臂 run（前台；记录 commit = C3）→ 分析恰一次 → 复核恰一次 → C4 结果回填（§10/§12）+ 处理臂 SUMMARY 行 → push。

---

## 0. 目的与判定问题

seed-2（`m1688723740`，Stage-1 产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f`）证据链（§1 审计）：

| 判定点 | 配对基线 run | 处理臂 run | 配对 Δtest | 配对 Δval | 结果 |
|---|---|---|---|---|---|
| 5-epoch（canonical，`a40836e`） | `20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07`（test `0.5974422649550507` / val `0.5809347091990792`） | `20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg`（test `0.6055453782825184` / val `0.5895572066556973`） | +0.008103113327467715 | +0.008622497456618139 | `VALID_POSITIVE`（U1/U2 PASS）；两臂 best=ep5 **右删失** |
| 10-epoch（更长预算，`e46e5d2`） | `20261004-0427-p2M-v500k-t1M-m1688723740-long-2b1d585`（test `0.6614121223087556` / val `0.6401127811737995`） | `20261004-0431-p2M-v500k-t1M-m1688723740-long-f2ccec2-rpg`（test `0.6688517876477933` / val `0.6449491259210769`） | +0.0074396653390377265 | +0.0048363447472773435 | `PERSISTS`；两臂 best=ep10 **仍右删失**；Δval 自 ep4 峰值（+0.008916659527500093）**单调收窄**至 ep10（+0.0048363447472773435）但始终为正 |

10-epoch 判定点仍未回答"收窄趋势是否最终抹平增量"（其设计 §9.7 明确不外推）。五 seed canonical 配对（短预算）已按预注册 §6.4 满足 20-epoch 立项条件（`TWENTY_EPOCH_CONDITION_SATISFIED`，§1 审计独立重推）。

**判定问题 Q**：在同一 seed-2 Stage-1 产物上，把 Stage2 预算延长到 **20 epochs / patience 3**（其余全部钉死），残差 prompt 处理臂相对**本实验新跑的配对 20-epoch 基线臂**：

- **Q1（历史阈值持久性）**：`Δtest ≥ +0.0055`（闭）∧ `Δval > 0`（严格）∧ A 类门禁 ∧ identity ∧ 机制门禁是否成立 → `PERSISTS` / `NOT_PERSIST`（`INVALID` 为完备性分支）；
- **Q2（二级分类，新增层）**：`Δtest` 落入 `POSITIVE_IMPROVEMENT` / `NO_CLEAR_IMPROVEMENT` / `CLEAR_DEGRADATION` 中哪一类（边界先于结果写死，§5.2）；若为 `NO_CLEAR_IMPROVEMENT`，按 §5.4 机械评估 headroom；
- **Q3（正持久性声明）**：§5.3 的六项检查（identity ∧ A 类 ∧ 机制门禁 ∧ `Δtest ≥ +0.001` ∧ `Δval > 0` ∧ 无末段矛盾趋势）是否全真 → `positive_persistence_claim`；据此按 §5.6 给出本线是否具备后续独立消融资格的机械结论。

**纪律声明（预注册）**：历史 5-epoch/10-epoch 判定（`VALID_POSITIVE`、`PERSISTS`、各自 Δ 值与轨迹）**原样保留、不改写、不重判**；本实验的判定只用 §4 新跑的两条成对 run（§5），历史数值只作 context 与预算趋势对照。

---

## 1. 审计（2026-10-05，只读；先于本文件提交）

### 1.1 审计方法与范围

- **原则**：不信任任何文档转述，全部以**磁盘上的原始 run 产物**（各 worktree `artifacts/aliccp_bench/` 下的 `metrics.json` / `config.json` / `gate_report.json` / `prompt_report.json`）与 **git 对象**（blob、ls-remote tip）为准重新核对；不重跑、不修改任何历史 run。
- 审计对象与 remote tip：five-seed `0935257`（`0935257079c769f45e6c65cd0419c7abe5bba725`；其后 tip 前进至 `28aa1fe` = 纯 provenance 提交——入库核验脚本 + whitelist 同步，结果记录零改动，本审计仍针对结果提交 `0935257`）、longer-budget `e46e5d2`（`e46e5d28ea572c1d6d5ace68b8100b848fe374e7`）、gate `3e2f083`（`3e2f0833fdb8a4966d5d815b41a8cefcfb0887d6`）、seed-replication `a40836e`（`a40836e3d6a7f73c4943bada98139d414957301a`）、基点 `8133d32`（`8133d32cfa722b084c558b334f4792d85ce95c67`）。
- 审计脚本为一次性只读脚本（`$CLAUDE_JOB_DIR/tmp/audit_20epoch_branch.py`，未跟踪、不入库）；结果 JSON 落 job tmp。

### 1.2 审计结果（逐项）

| # | 审计项 | 证据（全部独立重推） | 结果 |
|---|---|---|---|
| A1 | 五 seed 配对完整性（评立条件 C1 底座） | 五对 `(baseline, arm)` run 从 `pairs.json` + 原始 `metrics.json` 重读：逐 seed `stage1_id`/`model_seed` 相等、baseline/arm variant 正确、`rp_arm == gate_report.residual_prompt` 三处 JSON 互洽 ×5 | PASS |
| A2 | 逐 seed Δ 与分类重推 | seed1 `Δtest=+0.005394543882337954`/`Δval=+0.00448174078674779`（U1 FAIL/U2 PASS → `VALID_NEGATIVE`）；seed2 `+0.008103113327467715`/`+0.008622497456618139`（U1/U2 PASS → `VALID_POSITIVE`）；seed3 `+0.001456339669678841`/`+0.0013698631373653125`；seed4 `+0.004951998671071434`/`+0.004212769651563031`；seed5 `+0.009646016187411788`/`+0.011667457451331131` | PASS |
| A3 | **20-epoch 立项条件独立重推（§6.4 真值表）** | 五 seed 逐 seed：M0+G1–G8 全 PASS ∧ A 类 PASS ∧ A3 SKIP ⇒ C1；二级分类 5×`POSITIVE_IMPROVEMENT` ⇒ C2；`mean_Δtest=+0.005910402347593546`（样本 std `0.0031538117454772883`，ddof=1）⇒ C3；Student-t 95% CI（df=4，`t=2.7764451051977987`，scipy 独立复核逐位一致）`[+0.0019944278461222166, +0.009826376849064875]` 下界 > 0 ⇒ C4；`n(Δtest ≥ +0.001)=5 ≥ 4` ⇒ C5；`n(Δval > 0)=5 ≥ 4` ⇒ C6 | **`TWENTY_EPOCH_CONDITION_SATISFIED`**（C1–C6 全真） |
| A4 | 与 canonical 报告逐位对照 | 独立重推的 `mean/std/CI95_low/CI95_high/counts(5/2/5)/secondary_counts(5,0,0)/worst_seed(1688738016)` 与 `five_seed_report.json:aggregate` **逐位一致**；六项条件布尔与 `twenty_epoch_condition` 逐项一致；status 一致 | PASS |
| A5 | seed2 10-epoch 链（直接前驱）完整核对 | 基线 `20261004-0427-…-2b1d585` / 臂 `20261004-0431-…-f2ccec2-rpg`：两 run `dirty=false`、run_id 内嵌 commit、同 `stage1_id`、`epochs=10/patience=3`；A 类 PASS ×2、`hard_pass=false`（仅 B4 继承 FAIL）；臂 M0+G1–G8 全 PASS、三份 JSON 互洽；`Δ` 重推逐位一致（`+0.0074396653390377265`/`+0.0048363447472773435`）；两臂 best=10/10 右删失；Δval 自 ep4 峰值 `+0.008916659527500093` **单调收窄**至 ep10 | PASS |
| A6 | 跨预算确定性（描述性事实，支撑预算趋势可读性） | 10-epoch 基线前 5 epoch **逐位等于** 5-epoch 基线；10-epoch 臂前 5 epoch **逐位等于** 5-epoch 臂；10-epoch 基线轨迹/test **逐位等于** 旧衰减线 10-epoch 基线 `20261003-0724-p2M-v500k-t1M-m1688723740-long-bbd8a61`（test `0.6614121223087556`） | PASS |
| A7 | Stage-1 产物与参照头（§1.5 预注册前完整性核验） | 28/28 全过（见 §1.5） | PASS |
| A8 | 跨 worktree 复制字节同等 | 复制入本 worktree 的产物/run 目录逐文件 sha256 与源目录相等；seed2 Stage-1 产物的两份 on-disk 副本（five-seed / atten-seed2 worktree）逐文件相等 | PASS |

### 1.3 audit 结论与 seed 选择冻结（**先于任何 20-epoch 指标**）

1. 五 seed 短预算结果**满足其冻结的 20-epoch 立项条件**（`TWENTY_EPOCH_CONDITION_SATISFIED`，C1–C6 独立重推全真，与 canonical 报告逐位一致）⇒ 20-epoch 分支依预注册程序立项。
2. **canonical seed 选择冻结为 seed2 = `1688723740`**。理由（先于新指标）：五个 canonical seed 中**唯一**具备完整既有证据链的 seed——精确 5-epoch 配对（基线 `79b5e07` + 臂 `013e105-rpg`）+ 精确 10-epoch 配对（基线 `2b1d585` + 臂 `f2ccec2-rpg`）+ 钉死参照头（`79b5e07/newtask.pt`）+ 已复用产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f`。其余 seed 只有 5-epoch 配对；为其立 20-epoch 判定需先补 10-epoch 配对（超出本分支范围）。单 seed、单配对；**本分支不做五 seed × 20-epoch**。
3. 历史 5-epoch/10-epoch 判定与全部数值**原样保留**（本文件只作引用与对照）。

---

## 1.5 预注册前完整性核验（只读；先于本文件提交；`verify_seed2_artifact_integrity.py`）

对固定产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f` + 参照头 checkpoint 做只读复算（**28/28 通过**，报告 `artifacts/aliccp_bench/audit/20epoch-rp/verify_seed2_artifact_integrity_result.json`；不写任何 run/stage1 产物、不训练、不重新评测）：

| 组 | 核对 | 结果 |
|---|---|---|
| 产物文件字节 | `backbone.pt`/`env_ids.pt`/`meta.json` sha256 == 钉死值（`cd7b033423499e0d36eea988835cea4ac4e2a8e26427a2aea627333822101e0b` / `4660be5aaa3c59f53dd5b4394f77114a87db69064b32c45517db049a6b9157e7` / `61a66d81ce3dcf6bae64f4c2b37cf12e722943def646336a588746aba932931d`） | PASS |
| 产物内容寻址 | `stage1_id` 重算 == 记录 == 期望；`config_hash` 重算 == `4e1b5c6ffe9b6ff49cda34691de27390669396337e64e867f0f83b6ec4da7981`；id 分量（fp/config 前缀、seed=1688723740、epochs=3、env_seed=20261003、budgets） | PASS |
| 前缀指纹 A2 级 | 自哈希 == `5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8`；前缀字节 sha256 ×3 + 表头 + 文件大小（2473647855/274769757/2711167840）+ 原始扫描标签计数 ×3 | PASS |
| env_ids | 张量 sha 重算 == `5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0`；形状 (2000000,)∈{0,1}；env_0=1532 / env_1=1998468 | PASS |
| backbone | 张量 sha 重算（排除 buffer `env_indices`）== `e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c` | PASS |
| 参照头 | 文件 sha256 == `90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f`；`NewTask(80/64/[32,32])` strict 载入成功 | PASS |

**产物复用决议**：按"先验证、后复用，否则确定性重生成"的规程——28/28 验证通过 ⇒ **复用**该产物（不重训 Stage-1）。产物与参照头均自 five-seed worktree（产物另与 atten-seed2 worktree 副本交叉核对）**只读复制**入本 worktree，逐文件 sha256 与源相等（A8）。

**context run 只读复制（不参与判定，§5.5 对照用）**：5-epoch 基线 `20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07`（含钉死参照头 `newtask.pt`）、5-epoch 臂 `20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg`、10-epoch 基线 `20261004-0427-p2M-v500k-t1M-m1688723740-long-2b1d585`、10-epoch 臂 `20261004-0431-p2M-v500k-t1M-m1688723740-long-f2ccec2-rpg`，连同 `splits/p2M-v500k-t1M/prefix_fingerprint.json`，逐文件 sha256 校验后复制入本 worktree `artifacts/aliccp_bench/`（gitignore）。

---

## 2. 预算选择（基于 §1 轨迹事实，不基于任何新结果；先于 run 写死）

| 项 | 取值 | 依据 |
|---|---|---|
| Stage2 **max epochs** | **20** | 5-epoch（`best=5/5`）与 10-epoch（`best=10/10`）判定点**均右删失**；10-epoch Δval 自 ep4 峰值单调收窄（`+0.00892 → +0.00484`）但判据全过——收窄终点未知。更长预算协议 §9.7 明确"10-epoch 为固定预算、不外推"；本实验经用户显式指示做**一次**延长：20 = 10 的 2×，为**终止判定点**（§9.6：不外推至 20 以上）。单臂墙钟 ≈ 8–10 min（10-epoch 实测 249.5 s/283.4 s 的 2× 外推），两 run 合计 ≈ 20 min，可控 |
| Stage2 **patience** | **3**（不变） | 与 10-epoch 检验完全相同：patience=3 允许至多 2 个连续非改进 epoch 后停止；两臂**同一规则**。相对 5-epoch canonical 预算（patience=2）的 +1 已在 10-epoch 协议中引入并沿用，本实验不再变更 |
| 其余全部 | 不变 | §3 |

- 若某臂在 20 epoch 内触发 early stop：属预注册行为，**照实记录**（`stopped_early`、停止 epoch、best epoch 仍取 val 最优并重载），不作为失败、不重跑。
- **禁止**：看到轨迹后改预算/patience；事后挑选 epoch 报数（判定只用各臂 val 选出的 best checkpoint 与单次 test 评估，§5）。

## 3. 控制变量（哪些变、哪些钉死）

| 项 | 5-epoch（canonical 先例） | 10-epoch（直接前驱） | 本实验（20-epoch） | 说明 |
|---|---|---|---|---|
| **Stage2 epochs** | 5 | 10 | **20** | 相对前驱的唯一变化 |
| **Stage2 patience** | 2 | 3 | **3**（不变） | — |
| tag / run_id | `short` | `long` | **`xlong`**（新 tag 词汇，沿用"预算延长即新 tag"先例） | 处理臂 run_id 由 runner 追加 `-rpg` 后缀 |
| Stage-1 产物 | `s1-5c060b9c-m1688723740-e3-4e1b5c6f` | 同左 | **同一产物（钉死；不重训 Stage-1）** | §1.5 28/28 通过 |
| model seed | `1688723740` | 同左 | 同左（不变） | 头初始化 + 训练 dropout；两臂同进程重播种 → 头初始化与样本顺序构造性相同 |
| env seed / 预算 / 前缀 | `20261003` / p2M-v500k-t1M | 同左 | 同左（不变） | `env_ids` 来自产物（不重抽） |
| 机制超参 | hidden=16；α 初值 0；注入三路 `("gen","spec_0","spec_1")`；RNG 隔离；新增参数 2385/8129 | 同左 | **逐字不变（不搜索、不调参）** | §6-A1‴ 逐字节移植保证 |
| 参照头 checkpoint（M0/G7 锚，**不随预算变化**） | run `20261003-0624-…-79b5e07/newtask.pt` | 同左 | **同文件钉死**（路径 + sha `90ee06da…`） | M0 常量不变：`ref_val_auc = 0.5809347091990792`、`ref_pred_std = 0.005217193225189258`（≤1e-9） |
| batch / lr / 结构 / dropout / loss / 评测口径 | 协议 §8.1 默认；val 选点 + best 重载 + test 单次评估 | 同左 | 同左（不变） | **no post-hoc epoch selection** |

**描述性确定性核对（预注册，不参与判定）**：同 seed 的 stage2 训练在本环境已被证明跨进程/跨 commit 逐位可复现（§1-A6）。因此：(a) 本实验新跑的 20-epoch 基线臂前 10 epoch 轨迹**预期与 10-epoch 基线 run 逐位相同**；(b) 处理臂前 10 epoch 轨迹**预期与 10-epoch 臂 run 逐位相同**（进而前 5 epoch 与 5-epoch 臂相同）。两核对**照实报告**；若不一致，如实披露并排查，但不改变 §5 判定所用 run（判定只用本实验新跑的两条 run）。

---

## 4. 运行计划（恰好一次）与清洁树纪律

**运行次数**：基线臂 stage2 **恰好 1 次**；处理臂 stage2 **恰好 1 次**；持久性分析（纯分析：只读 run 记录 + 参照文件哈希，**不重新评测、不新增 test 遍历**）**恰好 1 次**；运行后独立复核脚本 **恰好 1 次**。**不重训 Stage-1**；已完成的有效 run 一律不重跑。

**无效执行处置（预注册）**：仅当 run 因**工具性原因**（进程崩溃/中断/环境故障，run 未完整落盘）无效时，保留全部现场（含失败日志与半成品目录，不删除），记录原因后可重跑一次；因**结果原因**（判据未过/门禁失败）不构成无效，不重跑，如实记录。

**清洁树纪律（run 间提交）**：
1. 提交实现与测试（commit `C2`）→ 树干净（未跟踪文件不计入 `git.dirty`）；
2. 跑**基线臂**（`git.dirty` 必须为 false，run 记录 commit = `C2`）；
3. **提交基线 run 的 SUMMARY 追加行**（commit `C3`，只允许 `artifacts/aliccp_bench/SUMMARY.md` 一个文件变化）；
4. 跑**处理臂**（`git.dirty` 必须为 false，run 记录 commit = `C3`）；
5. 运行后核验代码同一性：`git diff C3 C2 -- . ':(exclude)artifacts/aliccp_bench/SUMMARY.md'` 必须为空（两 run 的代码逐字相同，run_id 记不同 commit 仅为 SUMMARY 追加行的归属）；
6. 提交结果（§10/§12 文档更新 + 处理臂 SUMMARY 追加行，commit `C4`；`rp_20epoch_compare.json` 与 `verify_report.json` 落 run 目录，gitignore）→ push。

## 5. 预注册判据（看到结果前写死；看到结果后不得修改）

对照 = **本实验新跑的 20-epoch 基线臂**（同 `stage1_id`、同 model seed、同顺序；**不得**用 5-epoch/10-epoch run 做对照）。`Δtest := test_auc_bsi(arm) − test_auc_bsi(baseline)`；`Δval := best_val_auc_bsi(arm) − best_val_auc_bsi(baseline)`。

### 5.1 历史阈值持久性（C1/C2/C3a/C3b/C3c；与 10-epoch 检验逐字相同，阈值不放宽）

| 条件 | 判据 |
|---|---|
| C1 | `Δtest ≥ +0.0055`（闭区间） |
| C2 | `Δval > 0`（严格；best 由各臂 val 选点规则确定，非事后挑选） |
| C3a | **A 类门禁全过**：两 run 的 A1/A2/A4/A5/A6 全 PASS（A3 SKIP 视为通过，与协议 `hard_pass` 的 A 类口径一致） |
| C3b | **identity/记录校验全过**（实现为 12 项具名检查，逐项落盘）：两臂 `stage1_id` 相同**且** == §3 钉死值；两臂 `model_seed` 相同**且** == 1688723740；baseline `variant == "baseline"`；arm `variant == "residual-prompt"`；arm run_id 以 `-rpg` 结尾且 baseline 不以之结尾；arm 记录的参照头路径 == §3 钉死路径**且**文件 sha256 == 钉死值；两 run 记录 `epochs == 20` 且 `patience == 3`；两 run 记录 `tag == "xlong"` |
| C3c | **处理臂机制/构造门禁全过**：arm 记录 `rp_arm` 的 **M0 + G1–G8 全 PASS**（判据语义与 seed-2 复现预注册 §4 逐字相同，不放宽） |

- **C1 ∧ C2 ∧ C3a ∧ C3b ∧ C3c → `PERSISTS`**；**C3b 任一 false → `INVALID`**（完备性分支：预期不可达；出现即表示运行偏离预注册，按实记录并停止解读）；其余情形 → **`NOT_PERSIST`**（含 C3a/C3c 失败；失败门禁集合逐项落盘，照实披露）。
- 阈值来源：+0.0055 沿用 seed-2 复现/更长预算协议的 provisional 效用口径（协议 §10.1 "+0.005" 的保守取整版本；test 1M 前缀 BSI 负例 ≈1.6 万 ⇒ SE ≈ 0.0025 @AUC0.7，+0.0055 ≈ 2.2×SE）；**不放宽**。

### 5.2 二级分类（新增层；与历史 U1/U2/C 类判定**并存**，不覆盖任何旧字段）

对配对 `Δtest` 机械分类（边界与五 seed 分支 §5.2 逐字相同）：

| 类别 | 判据（精确边界） |
|---|---|
| `POSITIVE_IMPROVEMENT` | `Δtest ≥ +0.001` |
| `NO_CLEAR_IMPROVEMENT` | `−0.02 < Δtest < +0.001`（严格双开） |
| `CLEAR_DEGRADATION` | `Δtest ≤ −0.02` |

### 5.3 正持久性声明（`positive_persistence_claim`；新增层）

六项检查全真 ⇒ `claim = true`（全部机械落盘）：

| # | 检查 | 判据 |
|---|---|---|
| P1 | 配对 identity 有效 | C3b 全 12 项 true |
| P2 | A 类门禁 | C3a true（协议：A 类 FAIL ⇒ 比较作废；并入使 claim 更保守） |
| P3 | 全部机制门禁 | C3c true（M0+G1–G8 全 PASS） |
| P4 | 效用下界 | `Δtest ≥ +0.001`（闭） |
| P5 | 验证方向为正（**预声明的对照 = 本实验新跑的 20-epoch 配对基线的 best-val 值**，非 5-epoch 常量） | `Δval > 0` |
| P6 | 无末段矛盾趋势 | `Δval@末共同 epoch > 0`：设两臂均按 val 选点、逐 epoch val 轨迹的**末共同 epoch**（`min(epochs_run_base, epochs_run_arm)`）上 `Δval(e) = arm_val(e) − baseline_val(e)`；要求 `Δval(末共同) > 0`（"增量在终点前未被对手反超"的机械口径；test 为单次评估、无逐 epoch test 轨迹，故末段趋势只可由 val 侧定义）。**任一臂无共同 epoch ⇒ P6 false** |

- claim 与历史 `PERSISTS` **并列报告**：`PERSISTS=false` 而 `claim=true`（即 `+0.001 ≤ Δtest < +0.0055` 且方向/门禁全过）是合法组合；反之 `claim=false ∧ PERSISTS=true` 不可能（PERSISTS ⇒ Δtest ≥ 0.0055 ⇒ P4；P6 仍需单独检查）。
- **报告项（不判定）**：末段趋势形态（Δval 最后 3 个共同 epoch 的走势：收窄/走平/走宽），仅供解读。

### 5.4 `NO_CLEAR_IMPROVEMENT` 的机械 headroom 评估（仅当 §5.2 落入该区间；规则先于结果写死）

| 因子 | 机械口径 | 取值 |
|---|---|---|
| 机制活性 | `|α_final| > 0` ∧ 末 epoch `generator_grad_norm > 0` ∧ `geff_std > 0`（均取自 arm run JSON） | `ACTIVE` / `INACTIVE` |
| 验证方向 | `Δval > 0` | `POSITIVE` / `NON_POSITIVE` |
| epoch 轨迹 | 臂 `best_epoch == epochs_run == 记录 epochs`（未早停）⇒ 右删失仍上升；否则早停 | `RIGHT_CENSORED_STILL_IMPROVING` / `EARLY_STOPPED` |
| 预算趋势（short/10/20） | `d5 = +0.008103113327467715`、`d10 = +0.0074396653390377265`（§1 审计过的记录值）、`d20 = 本实验 Δtest`；`e1 = d10 − d5`、`e2 = d20 − d10`；标签：`d5>d10>d20` ⇒ `MONOTONE_NARROWING`；`d5<d10<d20` ⇒ `MONOTONE_WIDENING`；`e1<0 ∧ e2>0` ⇒ `TROUGH_AT_10`；`e1>0 ∧ e2<0` ⇒ `PEAK_AT_10`；`d5==d10==d20` ⇒ `FLAT`；其余 ⇒ `NON_MONOTONE_MIXED`；另报 `sign_flip = not (三个 Δtest 同号)` | 标签 + 三值 + 两差分 |
| 距正分类余量 | `gap_to_positive_threshold = +0.001 − Δtest`（>0，越小越接近） | 数值 |

- 预算趋势三值各自均为"该预算下 arm − 同预算配对基线"（跨预算不混基线）；`d5/d10` 为已审计记录值，`d20` 为本实验实测。本因子为**描述性**，不改变任何分类。

### 5.5 报告项（不判定）

- 评测协议 §10.1 provisional 双条件（`Δtest ≥ +0.005` ∧ `Δval > 0`）原样报告。
- **历史 +0.0055 是否保留**：显式单列（= C1 的真值；另报 Δtest 的 5→10→20 预算序列与相对 10-epoch 的增减）。
- **与精确 5-/10-epoch seed2 轨迹的对照**：逐 epoch val 轨迹并列、逐 epoch Δval 并列（含 5→10→20 判定点 Δtest/Δval 序列）。
- 是否 early stop、best epoch、逐 epoch 轨迹形态、in-run `rp_arm.U`（其配对对象是 5-epoch 常量——class C 诊断，**不得**与持久性判定混读）均为描述量，不参与判定。

### 5.6 后续独立消融资格（预先声明；本分支不启动任何消融）

`residual_prompt_ablation_eligible := claim（§5.3 六项全真）`。即：只在"配对 identity 有效 ∧ A 类 ∧ M0+G1–G8 全过 ∧ Δtest ≥ +0.001 ∧ Δval > 0 ∧ 末段无矛盾"时才判本线**具备**进入后续独立消融的资格（`ABLATION_ELIGIBLE`）；否则 `ABLATION_NOT_ELIGIBLE`。该字段为**机械布尔**，与 `PERSISTS` 分开报告；本分支**不启动**任何消融（§9）。

---

## 6. 实现面（最小；TDD；等价证明）与适配清单

| 编号 | 文件 | 处置 | 守卫 |
|---|---|---|---|
| A1‴ | `aliccp_benchmark/residual_prompt.py` | **逐字节移植** `013e105:aliccp_benchmark/residual_prompt.py`（blob `674213f619c5d5039811c71242a7727118348daf`；LF sha256 `b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc`）。**零文本适配**（seed2 run-reference 常量与 §3 钉死参照一致；docstring 指向 seed-2 复现设计文档 = 机制系谱文档，不改） | 守卫测试：工作树 == 钉死 blob 逐字节 + sha/blob 钉死 + docstring token |
| A2‴a | `aliccp_benchmark/bench.py` | **逐字节移植** `013e105:aliccp_benchmark/bench.py`（blob `bf3647686838157bf5fd7645615faad9f2d7185d`；LF sha256 `b063a36667173b66fc8a7ac932502cc34753c91e2bfcfdafaedfce443f1db3b4`；== `79ddefa`） | 守卫测试：逐字节 + sha/blob 钉死 + 相对 `8133d32` 的 diff 与 `013e105` 相同 |
| A2‴b | `run_aliccp_benchmark.py` | `013e105:run_aliccp_benchmark.py`（blob `229c25a1c86719fa6fb6b057d6cff3f4120fa053`；LF sha256 `2966ec3985e5b91e277f429c2f236fb551de7479d114def90f767f2d813ad285`）**恰 1 行**适配：`--tag` choices 增加 `"xlong"`（`["short","smoke"]` → `["short","smoke","xlong"]`；默认仍 `short`） | 守卫测试：单行替换重建等式（替换前断言旧行恰出现 1 次）+ 新 LF sha 钉死 |
| A3‴ | `aliccp_benchmark/tests/test_residual_prompt.py` | `013e105:` 同名测试（blob `768a477b5c71af32c5c59ec6feb20c29f7873557`；LF sha256 `dab737a1422b88befc4c19f2bee408618061de7fd3bf0b2f83ebcdd85fd1cbc2`）**恰 4 处**适配：(a) 模块 docstring；(b) `DOC_PATH` → 本文件；(c) `WHITELIST` → 本分支 10 项白名单（逐项集合钉死）；(d) `TestPreregConstants.test_prereg_doc_tokens_and_summary_ledger` 的 token 表 → 本文件 token 表。**其余逐类/逐方法/逐模块级语句不变**（含 `TestMechanismPin` 对 Census 祖本的全套 AST 钉死与 `test_thresholds_and_baseline_reference` 的常量断言——本实验常量与 seed-2 复现相同，**无需改动**） | 守卫测试：重建等式（docstring + `DOC_PATH` 段 + `WHITELIST` 段 + 该 1 个方法段替换）+ AST 差异集合 == 恰 1 个方法 + 模块级函数逐字 + 新 LF sha 钉死 |
| A4‴ | `aliccp_benchmark/rp_20epoch.py`（新） | 20-epoch 持久性分析器：钉死常量 + 历史判定 `persistence_verdict` + **新增层**（二级分类 / 正持久性声明 / headroom / 预算趋势 / 消融资格）+ `analyze_runs`（只读两 run 记录 → 全量判定/描述/identity/context）+ `python -m` CLI（输出 JSON）。`persistence_verdict`/`_read_json`/`_a_class_ok` 自 **`f2ccec2:aliccp_benchmark/rp_longer_budget.py`**（blob `e02da3071ded554175cfdad5d7a725dd43ed7c89`；LF sha256 `b8d62c5735715440a7e23dcbc4f206ee38ec7f6f9a6e5aac1184ceb8dcb15bb0`）移植：`_read_json`/`_a_class_ok` **逐字**；`persistence_verdict` **恰 1 类适配**（阈值常量名 `LONGER_BUDGET_DELTA_TEST_MIN` → `DELTA_TEST_MIN`，函数段内恰 2 处；签名/docstring/checks 结构与前驱逐字一致）；其余（常量块 / `_arm_description` / 新增判定函数 / `analyze_runs` / `main`）为按本实验判据新写 | 守卫测试：段级重建等式（上述 2 处替换）+ `_read_json`/`_a_class_ok` 段逐字节 + 行为测试（边界/identity/只读/fixture 端到端/claim 真值表） |
| A5‴ | `aliccp_benchmark/tests/test_residual_prompt_20epoch.py`（新） | 本分支守卫 + 接线 + 分析器测试（§7 I2/I4/I5/I6/I7/I8） | 本文件即为守卫 |
| A6‴ | `verify_seed2_artifact_integrity.py`（预注册前完整性核验）与 `verify_rp_20epoch.py`（运行后独立复核） | 只读脚本；**跟踪入库**（沿用更长预算分支先例；两处 WHITELIST 副本同步包含二者）；报告 JSON 落 `artifacts/aliccp_bench/audit/20epoch-rp/` 与 run 目录（gitignore）。完整性脚本自 `e46e5d2:verify_seed2_artifact_integrity.py`（blob `f2aa202c426d619cd752035e19cf8803dc5afccec0ce32f923c48a20ff1d72f3`）移植（仅 docstring/输出目录适配，钉死值不变；§1.5 28/28）；复核脚本为纯 JSON 重推新写 | 完整性核验结果 28/28（§1.5）；复核脚本 §10.5 |

**被禁止的适配**：机制公式/初始化/门控语义/注入点/活性带/`BOUND_TOL`/G7 比值/U1 阈值/分类规则/claim 规则/headroom 规则/消融资格规则；`protocol.py`/`metrics.py`/`multitaskrec/*`/`config.py` —— 零改动（静态守卫：`git diff --name-only 8133d32` ⊆ 白名单）。

**tag 词汇说明**：`"xlong"` 为本实验引入的最小 CLI 支持（1 行 choices 扩展；`smoke`/`short`/`long` 语义不变），与更长预算分支引入 `"long"` 的先例同型；`epochs/patience` 为已有 override（基点 `8133d32` 起存在），本实验不新增数值语义。

**等价性证明口径（三层，全部由测试守卫执行）**：
1. **重建等式（文本级）**：钉死 blob 文本 + 恰好替换（CLI 1 行；移植测试 4 处）== 工作树文件文本（逐字节）；
2. **AST 钉死（祖先级）**：`TestMechanismPin`（原样移植）断言工作树机制文件与 Census 祖本 blob：类名/方法集合相等、全部方法 `ast.dump` 逐字相等、模块级函数逐字相等、`reference_head_stats` 归一化 `auc_score→auc` 后逐字相等（豁免集合恰为 {模块 docstring、import、模块级常量赋值、`arm_verdict`}）；
3. **接线字节级**：`bench.py`/`residual_prompt.py` 工作树 LF sha256 == pin；`run_aliccp_benchmark.py` == pin + 恰 1 行；分析器移植段重建等式。
   （1+2 合取 ⇒ 工作树机制 ≡ pin ≡ seed1/seed2/10-epoch 实现在一切类/函数上；3 ⇒ 接线 ≡ 前驱实现在一切行为路径上。）

---

## 7. 不变量（由 `test_residual_prompt_20epoch.py` 与移植测试强制，不接受人工目测）

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 基线臂逐位不变：`RP.build_newtask(baseline)` 即 `NewTask`；`run_stage2` 默认参数下记录与机制臂键集差异恰为 `variant`/`rp_arm`/`prompt_hidden`（及 `-rpg` 后缀、prompt_report）；共享参数/前向在 α=0 时逐位一致 | 移植测试（构造恒等/初值恒等/端到端 purity） |
| I2 | **适配逐字守卫**：A1‴/A2‴/A3‴/A4‴ 的字节/AST/重建等式钉死 | `git show` + sha256(LF) + 段级重建等式 |
| I3 | 机制语义不变：α 初值 0、恒等/界/活性带/参数预算测试全绿（移植测试原样通过） | 移植测试（TestConstructionIdentity/TestNormControl/TestArmVerdict/TestParamReport…） |
| I4 | **CLI/override**：`--epochs/--patience/--variant/--prompt-reference-newtask` 默认值 == 协议/机制常量（逐位不变）；`--tag xlong` 可解析且默认仍 `short`；`--epochs 20 --patience 3 --tag xlong` 经 `main()` 正确线程化到 `run_stage2` 调用参数（monkeypatch 捕获 kwargs）；处理臂 `--variant/--prompt-reference-newtask` 线程化 + run_id `-rpg` 后缀 + 基线 run_id 无后缀 | parser 断言 + `main()` 捕获式测试 |
| I5 | 判定边界：历史 C1 Δtest 恰 `+0.0055` 过（闭）、`−1e-12` 不过；C2 Δval 恰 `0` 不过（严格）、`+1e-12` 过；C3a=false 或机制门禁=false ⇒ `NOT_PERSIST`（即使 Δ 达标）；四条件全过 ⇒ `PERSISTS`。二级分类边界：`Δtest` 恰 `+0.001` ⇒ `POSITIVE_IMPROVEMENT`；恰 `−0.02` ⇒ `CLEAR_DEGRADATION`（闭端）；`−0.02+ε` ⇒ `NO_CLEAR`。claim 真值表：六项任一 false ⇒ claim=false；`Δval(末共同)` 恰 `0` ⇒ P6 false | 边界/真值表测试 |
| I6 | 分析器：只读（不改 run 产物）；identity mismatch ⇒ `INVALID`；机制门禁失败集合逐项落盘；输出键齐全（历史判定/二级分类/claim/headroom/预算趋势/消融资格/描述量/context）；context 常量 == 5-/10-epoch 记录钉死值；夹具端到端 + 篡改 fixture（门禁全过 + Δ 达标 ⇒ `PERSISTS` ∧ claim=true ∧ `ABLATION_ELIGIBLE`） | CPU 极小夹具端到端 + 篡改 fixture |
| I7 | 静态守卫：相对 `8133d32` 的全部跟踪改动 ⊆ §6 白名单；`protocol.py`/`metrics.py`/`multitaskrec/*`/`config.py` 零 diff | `git diff --name-only` |
| I8 | 预注册文档 token 钉死：预算 20/3、tag `"xlong"`、seed `1688723740`、stage1_id、参照路径/sha、5-/10-epoch context run_id/Δ、阈值 `0.0055`/`0.001`/`−0.02`、`PERSISTS`/`NOT_PERSIST`/`INVALID`、二级分类三标签、`positive_persistence_claim`、`ABLATION_ELIGIBLE`、适配清单 commit/blob | 文档文本断言（doc-token 方法） |

CPU 极小夹具不构成任何性能证据，只验证语义与接线。

---

## 8. 运行方式（恰好一次；命令留档；cwd = 本 worktree 根；本 worktree 无 .venv，用主树解释器）

```powershell
# 0) 测试（全部不变量；先写测试后写实现）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests -v

# 1) 基线臂（唯一一次；tag xlong；epochs 20 / patience 3；seed-2 产物）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag xlong --epochs 20 --patience 3 --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f

# 2) 处理臂（唯一一次；同产物、同 seed、同预算；残差 prompt + 钉死参照头）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag xlong --epochs 20 --patience 3 --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f `
    --variant residual-prompt `
    --prompt-reference-newtask artifacts/aliccp_bench/runs/20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt

# 3) 持久性分析（纯分析，恰一次；只读 run 记录 + 参照文件哈希）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m aliccp_benchmark.rp_20epoch `
    --baseline-run <基线 run_id> --arm-run <处理臂 run_id>

# 4) 运行后独立复核（只读；JSON 重推 + 交叉一致性）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe verify_rp_20epoch.py `
    --baseline-run <基线 run_id> --arm-run <处理臂 run_id>
```

- **前置条件**：开跑前先提交实现（§4 步骤 1）；每臂 run 启动时 `git.dirty` 必须为 false（§4 清洁树纪律）。**前台执行并等待至完成**（更长预算协议先例记载后台执行被会话终止杀死；本预注册明确要求前台）。
- 数据文件经目录联接（junction）复用 main tree 的 `dataset/AliCCP/`（只读；`dataset/` 被 gitignore，不入库）。
- **墙钟预算**：每臂 ≈ 8–10 min（10-epoch 实测 249.5 s/283.4 s 的 2× 外推 + 加载/评测），分析/复核 < 1 min。

## 9. 局限（预声明）

1. **单 seed、每臂 1 run**：无同 seed run-to-run 噪声估计（A3 SKIP）；20-epoch 与 10-epoch 的 Δ 差异混合了预算效应与 run 噪声，二者不可分离（§3 确定性核对为描述性佐证，不作噪声估计）。本实验回答"20-epoch 判定点增量是否仍在/在何量级"，不回答问题"增量随预算连续变化的曲线"。
2. **判定依赖单一汇总统计**（test 1M 前缀的 BSI AUC 差值；SE ≈ 0.0025 @AUC0.7）；单点估计，不引入事后置信区间叙事。`+0.001` 二级阈值约 0.4×SE，为**分类标签**而非显著性强断言。
3. **无独立重评测 pass**：判定直接取 run 记录值（单次 test 评估纪律，§4）；记录值的正确性依赖 bench 代码路径与复制源**逐字一致**（I2 静态守卫）+ 同类代码路径的独立重评测先例（seed-2 复现参照核验 21/21，diff=0.0）。此为范围决策，如实披露。
4. **in-run `rp_arm.U` 的配对对象是 5-epoch 常量**（class C 诊断，§5.5）——不得与 §5 持久性判定混读。
5. **B4 若 FAIL 为继承缺陷**（共享 Stage-1 语义，两臂逐位相同），本实验无法修复也不据此做任何主张；`hard_pass` 不参与臂有效性定义。
6. **单数据集（AliCCP）、单 seed**：结论不推及 CensusIncome / ByteRec、不推及其它 seed；**20 为终止判定点**——不外推至更长预算（如 40）。
7. **本实验不是"更优方法"主张**：机制无消融、无多 seed 显著性；§5.6 的消融资格仅为流程结论。
8. **P6（末段趋势）只由 val 侧定义**（test 单次评估无逐 epoch 轨迹）；若两臂早停于不同 epoch，以**末共同 epoch** 为准（定义先于结果写死）。

## 10. 结果（运行后回填，不回写第 0–9 节）

### 10.0 非结果核验与执行记录

- **§1.5 预注册前完整性核验**（只读）：28/28 通过；产物未动。运行内 A2/A5/A6 两次逐位复验通过；运行后独立复核 53/53 ALL_PASS（§10.5）。
- **执行链**：C1 `32e1303`（预注册）→ C1b `b6e5ed5`（预注册前澄清，判据零改动）→ C2 `45ab7e7`（移植 + 分析器 + 守卫，117/117 测试绿）→ 基线臂前台单次 run（commit `45ab7e7`，wall 523.8 s）→ C3 `eafc336`（基线 SUMMARY 行）→ 处理臂前台单次 run（commit `eafc336`，wall 591.9 s）→ 分析 ×1（`rp_20epoch_compare.json`）→ 复核 ×1（53/53）→ C4（本 §10/§12 回填 + 处理臂 SUMMARY 行）。
- **无工具性中断、无重跑、无事后调参**；两臂均**前台**执行并等待至完成（wall 523.8 s / 591.9 s，均在 600 s 前台窗口内）。运行间代码同一性：`git diff eafc336 45ab7e7 -- . ':(exclude)artifacts/aliccp_bench/SUMMARY.md'` 为**空**。

### 10.1 唯一有效 run 与臂级身份

| 项 | 基线臂 | 处理臂（residual-prompt） |
|---|---|---|
| run_id | `20261005-0820-p2M-v500k-t1M-m1688723740-xlong-45ab7e7` | `20261005-0829-p2M-v500k-t1M-m1688723740-xlong-eafc336-rpg` |
| commit / dirty | `45ab7e7`（C2）/ **false** | `eafc336`（C3）/ **false** |
| stage1_id | `s1-5c060b9c-m1688723740-e3-4e1b5c6f` | 同左（A5/A6 逐位校验通过） |
| epochs / patience / tag（记录值） | 20 / 3 / `xlong` | 20 / 3 / `xlong` |
| best_epoch / best val | **20** / `0.6652705173239858` | **20** / `0.667583990949807` |
| test AUC（单次） | `0.6862927818762538` | `0.6904487492541437` |
| early stop | **未触发**（20/20 严格选点；ep18 非改进计数后被 ep19 打破） | 同左（20/20） |
| 墙钟 / 峰值显存 | 523.8 s / 52.0 MB | 591.9 s / 55.7 MB |
| 臂级分类（in-run，class C） | — | `VALID_POSITIVE`（U1 +0.09300648429909297 / U2 +0.08664928175072784，**vs 5-epoch 常量**，非持久性判定） |

- identity（C3b，12/12 全 true）：两臂同 `stage1_id`/`model_seed`/`epochs`/`patience`/`tag`；baseline `variant=="baseline"`、arm `variant=="residual-prompt"` 且 run_id `-rpg` 后缀；arm 参照路径 == 钉死路径且文件 sha256 == `90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f`（逐位：`artifacts/aliccp_bench/runs/20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt`）。
- Stage-1 哈希（运行内复验逐位一致）：`backbone_sha256 = e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c`、`env_ids_sha256 = 5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0`、`fingerprint_sha256 = 5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8`、`config_hash = 4e1b5c6ffe9b6ff49cda34691de27390669396337e64e867f0f83b6ec4da7981`。

### 10.2 判定（§5 三层规则，机械计算；分析器 `rp_20epoch_compare.json` + 独立复核逐位一致）

| 条件 | 判据 | 实测 | 结果 |
|---|---|---|---|
| C1 | Δtest ≥ +0.0055（闭） | **+0.004155967377889924** | **FAIL**（差 0.001344 ≈ 0.54×SE） |
| C2 | Δval > 0（严格） | **+0.0023134736258212385** | PASS |
| C3a | A 类两 run 全 PASS（A3 SKIP） | A1/A2/A4/A5/A6 PASS ×2 | PASS |
| C3b | identity/记录 12/12 | 全 true（§10.1） | PASS |
| C3c | 机制门禁 M0+G1–G8 | 全 PASS（§10.4） | PASS |

- **历史判定（§5.1）：`NOT_PERSIST`**——恰由 C1（`+0.0055`）单项失败决定；C2/C3 全过。**历史 +0.0055 在 20-epoch 判定点未被保留**（5/10-epoch 判定点保留：`+0.008103113327467715` / `+0.0074396653390377265`；三判定点序列见 §10.3）。
- **二级分类（§5.2）：`POSITIVE_IMPROVEMENT`**（Δtest ≥ +0.001）；`NO_CLEAR_IMPROVEMENT` headroom 不适用（`applicable=false`）。
- **正持久性声明（§5.3）：`claim = true`**（六项检查全 true：identity ✓ / A 类 ✓ / M0+G1–G8 ✓ / `Δtest ≥ +0.001` ✓ / `Δval > 0` ✓ / 末共同 epoch `Δval = +0.0023134736258212385 > 0`（无末段矛盾趋势）✓）。
- **消融资格（§5.6）：`ABLATION_ELIGIBLE`**（:= claim 全真）。
- 报告项（不判定）：评测协议 §10.1 provisional 双条件在 20-epoch **未满足**（`Δtest ≥ +0.005` 为 **false**（+0.00416 < +0.005）；`Δval > 0` 为 true）——原样报告。

### 10.3 轨迹与描述量（20-epoch 全轨迹 + 跨预算对照）

逐 epoch val AUC（BSI，fp64 记录值；两臂 best=20/20 右删失，无早停）：

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 基线 | 0.4645711559 | 0.4856448122 | 0.5142344439 | 0.5508809596 | 0.5809347092 | 0.6024307437 | 0.6173782928 | 0.6269454538 | 0.6342338048 | 0.6401127812 |
| 残差臂 | 0.4650886038 | 0.4904911796 | 0.5231212372 | 0.5597976191 | 0.5895572067 | 0.6103371408 | 0.6240056179 | 0.6330036266 | 0.6396889474 | 0.6449491259 |

| epoch | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| 基线 | 0.6448133213 | 0.6486086949 | 0.6523950744 | 0.6548255863 | 0.6566516198 | 0.6588996629 | 0.6607770599 | 0.6607741599 | 0.6639277561 | 0.6652705173 |
| 残差臂 | 0.6492697341 | 0.6525659933 | 0.6553505683 | 0.6576950437 | 0.6598143347 | 0.6615652409 | 0.6632340829 | 0.6647673741 | 0.6662288952 | 0.6675839909 |

逐 epoch Δval（arm − baseline，20 个共同 epoch）：+0.0005174479 / +0.0048463673 / +0.0088867933 / **+0.0089166595**（ep4 峰值）/ +0.0086224975 / +0.0079063972 / +0.0066273251 / +0.0060581728 / +0.0054551426 / +0.0048363447 / +0.0044564128 / +0.0039572983 / +0.0029554939 / +0.0028694574 / +0.0031627149 / +0.0026655780 / +0.0024570230 / +0.0039932142 / +0.0023011392 / +0.0023134736（ep20）。

- **形态（描述）**：Δval 自 ep4 峰值起总体收窄（ep5–ep14 单调降至 +0.00287），ep15–ep20 在 **+0.0023 ~ +0.0040** 区间波动、**始终为正**；ep18 的局部回升由基线该 epoch 的微小回落（0.6607741599 < ep17 0.6607770599）机械导致。**末段无矛盾趋势**（§5.3-P6 过）。
- **跨预算对照（context + 本实验；三判定点各自的同预算配对 Δ）**：

| 判定点 | Δtest | Δval | 臂 best/epochs | 早停 | arm test | baseline test | arm val | baseline val |
|---|---|---|---|---|---|---|---|---|
| 5-epoch（`a40836e`，context） | +0.008103113327467715 | +0.008622497456618139 | 5/5 | 无 | 0.6055453782825184 | 0.5974422649550507 | 0.5895572066556973 | 0.5809347091990792 |
| 10-epoch（`e46e5d2`，context） | +0.0074396653390377265 | +0.0048363447472773435 | 10/10 | 无 | 0.6688517876477933 | 0.6614121223087556 | 0.6449491259210769 | 0.6401127811737995 |
| **20-epoch（本实验）** | **+0.004155967377889924** | **+0.0023134736258212385** | 20/20 | 无 | 0.6904487492541437 | 0.6862927818762538 | 0.667583990949807 | 0.6652705173239858 |

- 预算趋势（§5.4 机械标签）：`MONOTONE_NARROWING`（+0.00810 → +0.00744 → +0.00416，严格递减）；`sign_flip = false`；相对 10-epoch 变化：Δtest `−0.003283697961147802`、Δval `−0.002522871121456105`。**三个判定点的两臂均右删失**（best=末 epoch），无任何早停。
- 确定性附属核对（描述性，独立复核逐位断言）：本实验基线前 10 epoch **逐位等于** 10-epoch 基线；处理臂前 10 epoch **逐位等于** 10-epoch 臂；两臂前 5 epoch **逐位等于** 5-epoch 两臂 ⇒ 5→10→20 的 Δ 变化来源是**预算（判定点）**而非 run 噪声。
- 头/源门（B3 量）：`gate_mean` 基线 `[0.859728375, 0.140271765625]` → 残差臂 `[0.851118125, 0.1488816875]`（与 5/10-epoch 线同方向的小幅重平衡）。

### 10.4 机制门禁与诊断（M0 + G1–G8 全 PASS）

| 门禁 | 实测 |
|---|---|
| M0 参照身份 | `ref_val_auc = 0.5809347091990792`、`ref_pred_std = 0.005217193225189258`，与预注册常量逐位一致（≤1e-9） |
| G1 构造恒等 | 共享参数逐位相同、全局 RNG 端点一致；新增键恰 5 键（`prompt_gate` + 生成器 4 参数） |
| G2 初值恒等 | `α@构造 = 0.0`；真实首 batch 前向逐位一致（`max_abs_diff = 0.0`，n=2000） |
| G3 门控活 | 逐 epoch 首 batch（α, α_grad, gen_grad）：(0, 0.05111, **0.0**) / (0.01644, 0.10020, 0.02577) / … / (0.12383, 0.02666, 0.26064) / (0.12552, 0.03079, 0.26521)；`α_final = +0.12710827589035034`（全程单调上升）。首步 gen_grad=0 为预注册已知代价 |
| G4 范数受控 | 三路 `ratio_max = 0.09686915…` ≤ 0.12710828 + 1e-6 |
| G5 活性带 | 三路 `ratio_mean = 0.051863893563…` ∈ [0.005, 0.5]（三路 spread <1e-11） |
| G6 门控方差 | `geff` mean/std/min/max = 0.05186389356082932 / 0.01200169567020517 / 0.017817062738039714 / 0.09686915759902573 |
| G7 无坍缩 | `pred_std = 0.005657412024608277` = 1.084 × 参照（≥ 0.5 × 0.005217193225189258） |
| G8 参数预算 | 恰 5 键；`new_params_total = 2385`、`head_params = 8129`（29.34%，total 10514） |

逐流读数（val，n=500000，fp64 流式），流序 `("gen","spec_0","spec_1")`：`ratio_mean = 0.051863893563`（三路）；`delta_norm_mean = 8.9409 / 1.2455 / 1.1650`；`h_norm_mean = 168.9746 / 23.9812 / 22.4535`；`cos_mean = −0.045076 / +0.032769 / +0.047836`。

预测离散度：`pred_mean/std/min/max = 0.9934390616734028 / 0.005657412024608277 / 0.8520079851150513 / 0.9998692274093628`；`q05/q50/q95 = 0.9817896246910095 / 0.9950555562973022 / 0.9984723925590515`。

A 类：A1/A2/A4/A5/A6 PASS、A3 SKIP（×2 run）。B 类（继承事实、不归因于本臂）：B1 PASS（CTR 0.5559 / CVR 0.5028 继承；BSI 腿 = 本臂 0.6904）、B2 PASS（|val−test| = 0.0229 ≤ 0.05）、B3 PASS、B4 FAIL（`events=[(1532, 1998468)]`，与 5/10-epoch 线逐字相同）；两臂 `hard_pass = false`（仅 B4 继承失败，与先例同型）。

### 10.5 独立复核（`verify_rp_20epoch.py`，运行后，恰一次）

- 独立脚本（**不 import 任何机制/接线/分析模块**，纯 JSON 重推 + 文件哈希）：**53/53 ALL_PASS（exit 0）**，报告落臂 run 目录 `verify_report.json`。
- 判定独立重算与分析器 **逐位一致**：`delta_test_auc = 0.004155967377889924`、`delta_val_auc = 0.0023134736258212385`、`checks` 全同、`classification = NOT_PERSIST`、`secondary = POSITIVE_IMPROVEMENT`、`claim_checks` 六项全 true、`ablation_eligible = true`、`budget_trend = MONOTONE_NARROWING`。
- identity 12/12、机制门禁、真冻结 sha 链、A/B 门禁（含 B1 三腿按钉死精度解析）、M0/U 观测口径（U 为 5-epoch 常量 class C 口径）、参照路径+文件 sha、SUMMARY 两行、四项确定性逐位核对全部通过。

### 10.6 纪律核对（§4/§7/§9 落实）

- **运行次数**：基线 stage2 ×1、处理臂 stage2 ×1（均前台；无工具性中断；无重跑）、分析 ×1、复核 ×1；未改预算/patience/tag/阈值/常量/对照；未做任何事后调参；未重训/微调 Stage-1；未重跑任何历史 run；未扩 seed；未启动任何消融。
- 测试：117/117 通过（基点 37 = 22+12+3 + 移植 38 + 本分支守卫/分析器 42）；静态守卫：`git diff --name-only 8133d32` ⊆ 白名单、受保护文件零 diff（测试内断言）。
- `SUMMARY.md` 恰追加 2 行（基线 + 处理臂，tag `xlong`，append-only；无重写）。

### 10.7 解读（预注册口径；不越界）

1. **机械结论**：历史阈值 `+0.0055` 在 20-epoch 判定点**未保留**（`NOT_PERSIST`，恰由 C1 单项失败）；但 20-epoch 配对仍是**正向**（Δtest `+0.00416` ≥ `+0.001`、Δval `+0.00231` > 0、末段无矛盾）⇒ 二级分类 `POSITIVE_IMPROVEMENT`、`claim = true`、按 §5.6 `ABLATION_ELIGIBLE`。
2. **形态（描述）**：Δ 随预算 5→10→20 **单调收窄**（`+0.00810 → +0.00744 → +0.00416`），与"仅加速收敛"替代解释的预期方向一致，但截至 20-epoch 判定点该解释**尚未完全兑现**（增量仍为正且 val 同向）。三判定点两臂**均为右删失**（best=末 epoch）——收窄是否继续、是否归零，**不外推**（§9.6：20 为终止判定点）。
3. **历史判定原样保留**：5-epoch `VALID_POSITIVE`（`a40836e`）与 10-epoch `PERSISTS`（`e46e5d2`）两处记录**未改写、未重判**；本实验只新增 20-epoch 判定点。
4. **不是"更优方法"主张**：单 seed、每臂 1 run、无 run-to-run 噪声估计（A3 SKIP；确定性核对为描述性佐证）；本实验回答"20-epoch 判定点三层规则的结果是什么"，不作机理归因。
5. **class C 读法**：in-run `rp_arm.U`（+0.0930/+0.0866）是 vs **5-epoch 常量**的机制诊断（臂相对历史基线已远超），**不是**持久性判定。
6. **继承披露**：B4 两臂 FAIL（共享 stage-1 cluster 退化；臂无关）；`hard_pass=false` 如常披露。
7. **后续消融口径（§5.6，机械）**：`ABLATION_ELIGIBLE`——残差 prompt 线**具备**进入后续独立消融的条件（配对身份 ∧ A 类 ∧ M0+G1–G8 ∧ Δtest ≥ +0.001 ∧ Δval > 0 ∧ 末段无矛盾全真）。本分支**不启动**任何消融；是否执行由用户决定。

## 11. 非新颖性声明

构件全部为既有方法族（prompt tuning、FiLM 族条件化调制、adapter/LoRA 残差、门控残差、范数归一），先行工作清单沿用 seed1 设计 §11（写作引用前必须逐条核实，本会话未做系统检索）。本文件**不**声称独立于先行工作；可报告内容仅限本仓库冻结协议下的一次性判定。

## 12. 偏离披露（预声明 + 运行后补）

- **预注册前澄清提交**（先于任何实现与 run；判据、预算、阈值、分类/claim/headroom/消融资格规则**零改动**；仅把 §0/§1/§1.5/§6 中已按"…"省略的内容寻址哈希与 run id 写全，并注明 five-seed tip 前进至 `28aa1fe` 为纯 provenance 提交）：与更长预算协议 `296e03a` 先例同类。
- **分支工作流**：自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；机制经 §6 三层守卫钉死等价；协议不合并 `master`；**push 经用户显式指示**（本任务含"commit and push"）。
- **分支本地 SUMMARY 系谱**：本分支 `SUMMARY.md` 基线含 smoke 行与 seed1 基线行（`8133d32` 携带）；其余历史行记录在各自分支，本分支**不复制**其它分支行，只由本次运行追加自己的行（2 行：基线 + 臂）。
- **产物/context run 只读复制**：§1.5 所列产物/run/指纹均自 five-seed / atten-seed2 worktree 只读复制入本 worktree（逐文件 sha 相等；仅分析/参照用，不修改）。
- **非结果文件（未跟踪）**：`artifacts/aliccp_bench/audit/20epoch-rp/` 报告 JSON、`artifacts/aliccp_bench/logs/` run 日志；一次性只读审计脚本与复制脚本置于 job tmp。
- **说明（预期形态，非判据）**：若 10-epoch 的单调收窄趋势延续，20-epoch 判定点可能落入 `NO_CLEAR_IMPROVEMENT`（届时按 §5.4 headroom 机械评估）或 `NOT_PERSIST`——两者均属**预注册可预期结果**，照实记录、不重跑、不调参；任何结果（含 `PERSISTS`/`NOT_PERSIST`/`CLEAR_DEGRADATION`）都不触发布局外动作。
- **执行记录（运行后补）**：C1 `32e1303` → C1b `b6e5ed5`（判据零改动）→ C2 `45ab7e7`（117/117 测试绿、`git.dirty=false`）→ 基线臂前台单次 run `20261005-0820-…-xlong-45ab7e7`（wall 523.8 s）→ C3 `eafc336`（基线 SUMMARY 行）→ 处理臂前台单次 run `20261005-0829-…-xlong-eafc336-rpg`（wall 591.9 s，**无工具性中断、无重跑**）→ 分析 ×1（`NOT_PERSIST` ∧ `POSITIVE_IMPROVEMENT` ∧ claim=true ∧ `ABLATION_ELIGIBLE`）→ 复核 ×1（53/53 ALL_PASS）→ C4（本 §10/§12 回填 + 处理臂 SUMMARY 行）。运行命令与 §8 逐字一致（cwd=本 worktree，主树 venv 解释器）；两 run `git.dirty=false`；run 间代码同一性 diff 为空。
- **实际结果形态（运行后补）**：落在预注册预期分支（§12 上一条的 `NOT_PERSIST`）——历史 `+0.0055` 未保留；无任何判据/阈值/规则被事后修改。
- **20-epoch 终止判定点落实（运行后补）**：本分支结论仅限 20-epoch 判定点；不外推至更长预算；不因本结果启动任何消融/新 seed/新数据集工作（§9）。
- **运行时长贴近前台窗口的披露（运行后补）**：处理臂 wall 591.9 s，接近前台执行窗口上限（600 s）；无中断发生，如实披露。

---

**改动清单（白名单；静态守卫以 `8133d32` 为基准）**

| 文件 | 处置 |
|---|---|
| `docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-20epoch-design.md` | 本文件（C1 单独提交） |
| `aliccp_benchmark/residual_prompt.py` | 新增（A1‴：逐字节 == `013e105`） |
| `aliccp_benchmark/bench.py` | 新增（A2‴a：逐字节 == `013e105`） |
| `run_aliccp_benchmark.py` | A2‴b：== `013e105` 恰 1 行适配（`--tag` choices 增加 `"xlong"`） |
| `aliccp_benchmark/tests/test_residual_prompt.py` | 新增（A3‴：== `013e105` 恰 4 处适配） |
| `aliccp_benchmark/rp_20epoch.py` | 新增（A4‴：`persistence_verdict`/`_read_json`/`_a_class_ok` 自 `f2ccec2:rp_longer_budget.py` 移植 + 本实验判定/描述/identity/claim/headroom） |
| `aliccp_benchmark/tests/test_residual_prompt_20epoch.py` | 新增（A5‴：守卫 + 接线 + 分析器测试） |
| `verify_seed2_artifact_integrity.py` | 新增（A6‴：预注册前完整性核验脚本；自 `e46e5d2` 移植，§1.5 已运行 28/28） |
| `verify_rp_20epoch.py` | 新增（A6‴：运行后独立复核脚本） |
| `artifacts/aliccp_bench/SUMMARY.md` | 只追加 2 行（runner 自动） |
| 其余全部（`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`metrics.py`、既有测试三件、`AliCCP_*.py`、`CensusIncome_*.py`、`baseline/*`、`mask/*`、`analysis/*`） | **零改动**（静态守卫断言 `git diff --name-only 8133d32` ⊆ 白名单） |
