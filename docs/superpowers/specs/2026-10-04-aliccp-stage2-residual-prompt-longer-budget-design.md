# AliCCP 阶段 2 残差 Prompt 增量在更长优化预算下的持久性检验（robustness / alternative-explanation，非新方法）

- **状态**：预注册已写死（本文件在本分支任何实现、任何 run 之前**单独提交**；commit C1）。结果只在第 10 节以新增小节回填；第 1–9 节的判据与数字不得在看到结果后改动。
- **日期**：2026-10-04
- **适用分支**：`exp/aliccp-stage2-residual-prompt-longer-budget`（自 `infra/aliccp-fair-benchmark` @ `8133d32` **独立拉出**；不从任何 exp 分支拉出、不继承其运行时状态；不合并 `master`、不触碰其它 worktree；`aliccp_benchmark/protocol.py`、`aliccp_benchmark/metrics.py` 语义、`multitaskrec/*`、`config.py`、master 脚本与 `baseline/*` 零修改）。
- **被检验对象（只读引用）**：残差 prompt 机制 = `exp/aliccp-stage2-residual-prompt-seed-replication` @ `013e105:aliccp_benchmark/residual_prompt.py`（其自身为 `79ddefa`（seed1 实现）的 A1′ 适配移植——机制 blob `5dc7158ca999c9e7e6217a999c37869231411549`，机制系谱 `99b9510` Census 祖本）。该机制的 canonical seed-2 5-epoch 配对结果（run `20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg` vs 基线 run `20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07`）：`VALID_POSITIVE`（M0+G1–G8+A 类全 PASS；U1 PASS `Δtest = +0.008103113327467715 ≥ +0.0055`；U2 PASS `Δval = +0.008622497456618139 > 0`），记录 @`a40836e`（§1 为运行前对本机磁盘 run 记录的只读审计）。
- **上游协议（只引用、不修改）**：
  1. `docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（AliCCP 公平评测协议）——前缀预算、种子、Stage-1 产物、真冻结三件套、A/B/C 门禁、SUMMARY 台账全部沿用；
  2. `exp/aliccp-stage2-attenuation-longer-budget` @ `47ecb0c:docs/superpowers/specs/2026-10-03-aliccp-stage2-attenuation-longer-budget-design.md`（下称"更长预算协议"）——**已建立的更长预算协议**：唯一变化 = Stage2 max epochs 5→10 与 patience 2→3（tag `"long"`）、判定结构 C1/C2/C3 → `PERSISTS`/`NOT_PERSIST`/`INVALID`、`commit-between-runs` 清洁树纪律、`persistence_verdict` 阈值语义——**逐字沿用（§2/§4/§5）**。其实现提交 `bbd8a61`（`aliccp_benchmark/longer_budget.py` 为本次分析器 `persistence_verdict`/`_read_json`/`_a_class_ok` 的逐字来源）。
- **定位声明**：本实验是**稳健性 / 替代解释检验**（5-epoch 观察到的正增量是持久收益还是仅加速收敛），不是新方法，不主张任何新颖性；不改机制、不改系数、不调参、不换结构、不改损失、不放宽任何门限。与 seed-2 5-epoch 配对相比**唯一的变化 = Stage2 max epochs 5→10 与 patience 2→3**（§3）。结论口径：seed-2 残差 prompt 观察到的正增量在更长 Stage2 预算下是否按预注册判据**仍然成立**（PERSISTS vs NOT_PERSIST）。

**提交时序（预注册纪律）**：本文件先于实现与运行**单独提交**（C1）→ 实现 + 测试（TDD；两 run 的 `commit` 字段即后续提交）→ **恰好一次**基线臂 run、**恰好一次**处理臂 run（均前台、串行、run 间提交 SUMMARY 行，§4）→ 分析恰一次 → 结果只在第 10 节与 `SUMMARY.md` 以**新增小节 / 新行**回填（C3）。

---

## 0. 目的与判定问题

seed-2（`m1688723740`，Stage-1 产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f`）在 5-epoch Stage2 预算下观察到残差 prompt 处理臂相对配对基线 `Δtest = +0.008103113327467715`、`Δval = +0.008622497456618139`（同 `stage1_id`、同顺序、单次 test 评测；`VALID_POSITIVE`）。但两条 5-epoch 轨迹**都在最后一个 epoch 仍在改进**（best_epoch = 5/5，§1），判定点落在轨迹的右删失处，无法区分"持久收益"与"仅加速收敛"。同一个 seed-2 产物上，固定衰减机制线的 10-epoch 检验（基线 `20261003-0724-p2M-v500k-t1M-m1688723740-long-bbd8a61` / 处理臂 `20261003-0729-p2M-v500k-t1M-m1688723740-long-2d2bc5e-sattn`）已观察到增量被基线追平（`NOT_PERSIST`：`Δtest = +0.0005273306552253665`、`Δval = −0.002653370173158476`）——替代解释在另一机制线上成立；本实验检验其在**残差 prompt 线**上是否同样成立。

**判定问题 Q**：在同一 seed-2 Stage-1 产物上，把 Stage2 预算延长到 **10 epochs / patience 3**（其余全部钉死），残差 prompt 处理臂相对**本实验新跑的配对 10-epoch 基线臂**的 **Δtest ≥ +0.0055 且 Δval > 0** 是否仍成立，且全部机制/构造/冻结门禁通过？
- 成立 → **`PERSISTS`**（增量在更长预算下持续；非纯加速解释）
- 不成立 → **`NOT_PERSIST`**（增量被更长预算抹平或反向；与"仅加速收敛"解释一致）
- identity/记录不一致 → **`INVALID`**（完备性分支：出现即表示运行偏离预注册，按实记录并停止解读）

**纪律声明（预注册）**：旧 5-epoch 两条 run 及其 `Δ`、旧 10-epoch 基线 run `20261003-0724` 与衰减臂 `20261003-0729` **只作本实验的动机与事后对照（context），不构成本实验判定结果的任何部分**；本实验的判定只用 §4 新跑的两条成对 run（§5 判据），不使用旧 run 的任何数值入判。

---

## 1. 5-epoch 轨迹审计（先于任何新 run；只读，来自 on-disk run 记录）

来源：`artifacts/aliccp_bench/runs/<run_id>/metrics.json` 与 `gate_report.json`（本机磁盘留存，逐位复核）。两臂同 `stage1_id`、同 model seed、同顺序、单次 test 评测。

| 项 | 基线臂（`20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07`） | 处理臂（`20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg`） |
|---|---|---|
| `variant`（记录值） | （无字段；raw seed2 基线） | `residual-prompt` |
| `epochs / patience`（记录值） | 5 / 2 | 5 / 2 |
| 逐 epoch val AUC（BSI） | 0.4645711559431739 / 0.48564481224085004 / 0.5142344439088465 / 0.5508809596059387 / 0.5809347091990792 | 0.4650886037939688 / 0.4904911795680732 / 0.5231212372261194 / 0.5597976191334388 / 0.5895572066556973 |
| 逐 epoch train loss | 0.0819 / 0.0519 / 0.0497 / 0.0487 / 0.0481 | 0.0819 / 0.0517 / 0.0494 / 0.0485 / 0.0478 |
| best_epoch / best val | 5 / 0.5809347091990792 | 5 / 0.5895572066556973 |
| test AUC（单次） | 0.5974422649550507 | 0.6055453782825184 |
| 逐 epoch Δval(arm−baseline) | — | +0.0005174478507949387 / +0.0048463673272231556 / +0.008886793317272823 / **+0.008916659527500093** / +0.008622497456618139 |
| early stop | **未触发**（5/5 epoch 严格改进，patience=2 从未被行使） | 同左 |
| A 类门禁 | A1/A2/A4/A5/A6 PASS，A3 SKIP | 同左 |
| B 类门禁 | B1/B2/B3 PASS；B4 FAIL（继承 seed-2 stage-1 cluster 退化） | 同左（逐位相同 stage-1） |
| Δtest / Δval（配对） | — | **+0.008103113327467715 / +0.008622497456618139** |
| 臂级分类（in-run） | — | `VALID_POSITIVE`（M0+G1–G8 全 PASS） |
| 墙钟 | 142.9 s | 156.4 s |

**审计结论（决定预算选择的轨迹事实，与结果无关）**：
1. 两臂 5/5 epoch **单调改进、best=ep5（右删失）**——5-epoch 判定点截断了仍在下降的优化轨迹，无法回答"持久 vs 加速"。
2. 逐 epoch Δval 前 4 epoch 上升、**末段收窄**（+0.008887 → +0.008917 → +0.008622）：增量是否在更长预算下继续收窄至阈值以下，必须实测。
3. early stop 在 5 epoch 内从未被行使——patience 的取值在延长预算时必须**先于 run** 重新钉死（§2）。
4. **既存 10-epoch 基线 run（衰减线）context**：`20261003-0724-p2M-v500k-t1M-m1688723740-long-bbd8a61`（同产物/同 seed/同预算/epochs=10/patience=3）已在本机磁盘；其前 5 epoch 与 5-epoch 基线 run **逐位相同**（跨 commit 确定性证据，见更长预算协议 §10.2 附注）；其 10-epoch 记录：best=ep10、`best_val = 0.6401127811737995`、`test = 0.6614121223087556`。**该 run 只作 context**（§0 纪律）；本实验对照 = 本实验新跑的 10-epoch 基线臂（§5）。

### 1.5 预注册前完整性核验（只读；先于本文件提交；`verify_seed2_artifact_integrity.py`）

对固定产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f` + 参照头 checkpoint 做只读复算（**28/28 通过**，报告 `artifacts/aliccp_bench/audit/longer-budget-rp/verify_seed2_artifact_integrity_result.json`；不写任何 run/stage1 产物、不训练、不重新评测）：

| 组 | 核对 | 结果 |
|---|---|---|
| 产物文件字节 | `backbone.pt`/`env_ids.pt`/`meta.json` sha256 == 钉死值 | PASS |
| 产物内容寻址 | `stage1_id` 重算 == 记录 == 期望；`config_hash` 重算 == `4e1b5c6ffe9b6ff49cda34691de27390669396337e64e867f0f83b6ec4da7981`；id 分量（seed/epochs/env_seed/budgets）；`fingerprint_sha256 == 5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8`；`backbone_sha256` 重算 == `e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c`；`env_ids` sha 重算 == `5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0` | PASS |
| 前缀指纹 A2 级 | 自哈希 + 前缀字节 sha256 ×3 + 表头 + 文件大小 + 原始扫描标签计数 ×3（2473647855 / 274769757 / 2711167840） | PASS |
| env_ids | 形状 (2000000,)∈{0,1}；env_0=1532 / env_1=1998468 | PASS |
| 参照头 | 文件 sha256 == `90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f`；`NewTask(80/64/[32,32])` strict 载入成功 | PASS |

**失败处置**：任一不一致 ⇒ **停止**（不重训、不替换、不静默换产物），保留现场并如实报告。

---

## 2. 预算选择（基于 §1 轨迹，不基于任何结果；先于 run 写死；沿用已建立的更长预算协议）

| 项 | 取值 | 依据 |
|---|---|---|
| Stage2 **max epochs** | **10** | 5-epoch 为右删失点（§1 结论 1）；10 = canonical 5 的 2×，与更长预算协议（`47ecb0c` §2）**完全相同**的固定预算；单臂墙钟 ≈ 4–5 min，总运行预算可控 |
| Stage2 **patience** | **3** | 与更长预算协议（`47ecb0c` §2）完全相同：canonical 值 2 在 5 epoch 内从未被行使；延长预算后允许**至多 2 个连续非改进 epoch** 后停止；两臂**同一规则**，先于 run 写死 |
| 其余全部 | 不变 | §3 |

- 若某臂在 10 epoch 内触发 early stop：属预注册行为，**照实记录**（`stopped_early`、停止 epoch、best epoch 仍取 val 最优并重载），不作为失败、不重跑。
- **禁止**：看到轨迹后改预算/patience；事后挑选 epoch 报数（判定只用各臂 val 选出的 best checkpoint 与单次 test 评估，§5）。

## 3. 控制变量（哪些变、哪些钉死）

| 项 | seed-2 5-epoch（先例，仅对照） | 本实验（10-epoch） | 说明 |
|---|---|---|---|
| **Stage2 epochs** | 5 | **10** | 唯一变化之一（§2） |
| **Stage2 patience** | 2 | **3** | 唯一变化之二（§2） |
| tag / run_id | `short` | **`long`**（沿用更长预算协议引入的 tag 词汇；`smoke`/`short` 语义不变） | 处理臂 run_id 由 runner 追加 `-rpg` 后缀 |
| Stage-1 产物 | `s1-5c060b9c-m1688723740-e3-4e1b5c6f` | **同一产物（钉死；不重训 Stage-1）** | 预注册前完整性核验 28/28 通过（§1.5）；`backbone_sha256 = e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c`；`env_ids_sha256 = 5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0`；`fingerprint_sha256 = 5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8`；`config_hash = 4e1b5c6ffe9b6ff49cda34691de27390669396337e64e867f0f83b6ec4da7981`；文件 sha256：`backbone.pt = cd7b033423499e0d36eea988835cea4ac4e2a8e26427a2aea627333822101e0b`、`env_ids.pt = 4660be5aaa3c59f53dd5b4394f77114a87db69064b32c45517db049a6b9157e7`、`meta.json = 61a66d81ce3dcf6bae64f4c2b37cf12e722943def646336a588746aba932931d`（A2/A5/A6 在 run 内再次逐位校验） |
| model seed | `1688723740` | `1688723740`（不变） | 头初始化 + 训练 dropout；两臂同进程重播种 → 头初始化与样本顺序构造性相同 |
| env seed / 预算 / 前缀 | `20261003` / p2M-v500k-t1M | 同（不变） | `env_ids` 来自产物（不重抽） |
| 机制超参 | hidden=16；α 初值 0；注入三路 `("gen","spec_0","spec_1")`；RNG 隔离；新增参数 2385/8129 | **逐字不变（不搜索、不调参）** | §6-A1″ 逐字节移植保证 |
| 参照头 checkpoint | run `20261003-0624-…-79b5e07/newtask.pt`（5-epoch 基线头；M0/G7 锚） | **同文件钉死**：路径 `artifacts/aliccp_bench/runs/20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt`、文件 sha256 `90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f` | M0 常量不变：`ref_val_auc = 0.5809347091990792`、`ref_pred_std = 0.005217193225189258`（≤1e-9）；参照是机制/构造锚，**不随预算变化**（用本实验新基线头将无法在 run 前钉死常量——预注册纪律不允许） |
| batch / lr / 结构 / dropout / loss / 评测口径 | 协议 §8.1 默认；val 选点 + best 重载 + test 单次评估 | 同（不变） | 协议 §7.5/§9.1；**no post-hoc epoch selection** |

**描述性确定性核对（预注册，不参与判定）**：同 seed 的 stage2 训练在本环境已被证明跨进程/跨 commit 逐位可复现（更长预算协议 §10.2 附注）。因此：(a) 本实验新跑的 10-epoch 基线臂的逐 epoch 轨迹**预期与 `20261003-0724-…-bbd8a61` 逐位相同**、test 预期 `= 0.6614121223087556`；(b) 处理臂的前 5 epoch 轨迹**预期与 5-epoch 处理臂 run 逐位相同**。两核对**照实报告**；若不一致，如实披露并排查，但不改变 §5 判定所用 run（判定只用本实验新跑的两条 run）。

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
6. 提交结果（§10/§11 文档更新 + 处理臂 SUMMARY 追加行，commit `C4`；`rp_longer_budget_compare.json` 与 `verify_report.json` 落 run 目录，gitignore）。

**命令（留档，见 §8）**。分析输出：`artifacts/aliccp_bench/runs/<arm-run>/rp_longer_budget_compare.json`。

## 5. 预注册判据（看到结果前写死；看到结果后不得修改）

对照 = **本实验新跑的 10-epoch 基线臂**（同 `stage1_id`、同 model seed、同顺序；**不得**用 5-epoch run 或旧 10-epoch run 做对照）。

| 条件 | 判据 |
|---|---|
| C1 | Δtest := `test_auc_bsi(arm)` − `test_auc_bsi(baseline)` **≥ +0.0055**（闭区间） |
| C2 | Δval := `best_val_auc_bsi(arm)` − `best_val_auc_bsi(baseline)` **> 0**（严格；best 由各臂 val 选点规则确定，非事后挑选） |
| C3a | **A 类门禁全过**：两 run 的 A1/A2/A4/A5/A6 全 PASS（A3 SKIP 视为通过，与协议 `hard_pass` 的 A 类口径一致） |
| C3b | **分析步 identity/记录校验全过**（实现为 11 项具名检查，逐项落盘）：两臂 `stage1_id` 相同**且** == §3 钉死值；两臂 `model_seed` 相同**且** == 1688723740；baseline `variant == "baseline"`；arm `variant == "residual-prompt"`；arm run_id 以 `-rpg` 结尾且 baseline 不以之结尾；arm 记录的参照头路径 == §3 钉死路径**且**文件 sha256 == 钉死值；两 run 记录 `epochs == 10` 且 `patience == 3` |
| C3c | **处理臂机制/构造门禁全过**：arm 记录 `rp_arm` 的 **M0 + G1–G8 全 PASS**（恒等构造、零初值恒等、门控/生成器梯度活、范数受控界、活性带、逐样本门控方差、无预测坍缩（相对参照）、参数预算、参照身份——判据语义与 seed-2 复现预注册 §4 逐字相同，不放宽） |

- **C1 ∧ C2 ∧ C3a ∧ C3c ∧ C3b → `PERSISTS`**；**C3b 任一 false → `INVALID`**（完备性分支：预期不可达；出现即表示运行偏离预注册，按实记录并停止解读）；其余情形 → **`NOT_PERSIST`**（含 C3a/C3c 失败；失败门禁集合逐项落盘，照实披露）。
- 阈值来源：+0.0055 沿用 seed-2 复现/更长预算协议的 provisional 效用口径（协议 §10.1 "+0.005" 的保守取整版本；test 1M 前缀 BSI 负例 ≈1.6 万 ⇒ SE ≈ 0.0025 @AUC0.7，+0.0055 ≈ 2.2×SE）；**不放宽**。
- **报告项（不判定）**：评测协议 10.1 provisional 双条件（`Δtest ≥ +0.005` ∧ `Δval > 0`）原样报告；**best epoch 差异、是否 early stop、逐 epoch Δ 形态、in-run `rp_arm.U`（其配对对象是 5-epoch 常量——class C 诊断，见 §6 注）** 均为描述量，不参与判定——防止任何事后 epoch 挑选。
- **B 类**（B1–B4）照常判定并披露；**继承披露口径**：B4 预期 FAIL 由共享 Stage-1 产物（cluster 退化）决定（两臂逐位相同），属继承缺陷、不归因于处理臂、不据此否定本臂；本臂有效性定义 = C3，**不依赖 `hard_pass`**。

## 6. 实现面（最小；TDD；默认臂逐位一致）与适配清单

| 编号 | 文件 | 处置 | 守卫 |
|---|---|---|---|
| A1″ | `aliccp_benchmark/residual_prompt.py` | **逐字节移植** `013e105:aliccp_benchmark/residual_prompt.py`（blob `674213f619c5d5039811c71242a7727118348daf`；LF sha256 `b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc`）。**零文本适配**（seed2 run-reference 常量已就位，与 §3 钉死参照一致；docstring 的"唯一事实来源"指向 seed-2 复现设计文档 `docs/superpowers/specs/2026-10-04-aliccp-stage2-residual-prompt-seed-replication-design.md` = 机制系谱文档，本分支不改）| 守卫测试：工作树 == 钉死 blob 逐字节 + sha/blob 钉死 + docstring token |
| A2″a | `aliccp_benchmark/bench.py` | **逐字节移植** `013e105:aliccp_benchmark/bench.py`（blob `bf3647686838157bf5fd7645615faad9f2d7185d`；LF sha256 `b063a36667173b66fc8a7ac932502cc34753c91e2bfcfdafaedfce443f1db3b4`；== `79ddefa`） | 守卫测试：逐字节 + sha/blob 钉死 |
| A2″b | `run_aliccp_benchmark.py` | `013e105:run_aliccp_benchmark.py`（blob `229c25a1c86719fa6fb6b057d6cff3f4120fa053`；LF sha256 `2966ec3985e5b91e277f429c2f236fb551de7479d114def90f767f2d813ad285`）**恰 1 行**适配：`--tag` choices 增加 `"long"`（`["short", "smoke"]` → `["short", "smoke", "long"]`；默认仍 `short`）| 守卫测试：单行替换重建等式（替换前断言旧行恰出现 1 次）+ 新 LF sha 钉死 |
| A3″ | `aliccp_benchmark/tests/test_residual_prompt.py` | `013e105:aliccp_benchmark/tests/test_residual_prompt.py`（blob `768a477b5c71af32c5c59ec6feb20c29f7873557`；LF sha256 `dab737a1422b88befc4c19f2bee408618061de7fd3bf0b2f83ebcdd85fd1cbc2`）**恰 4 处**适配：(a) 模块 docstring；(b) `DOC_PATH` → 本文件；(c) `WHITELIST` → 本分支 10 项白名单（逐项集合钉死）；(d) `TestPreregConstants.test_prereg_doc_tokens_and_summary_ledger` 的 token 表 → 本文件 token 表。**其余逐类/逐方法/逐模块级语句不变**（含 `TestMechanismPin` 对 Census 祖本的全套 AST 钉死与 `test_thresholds_and_baseline_reference` 的常量断言——本实验常量与 seed-2 复现相同，**无需改动**）| 守卫测试：重建等式（docstring + `DOC_PATH` 段 + `WHITELIST` 段 + 该 1 个方法段替换）+ AST 差异集合 == 恰 1 个方法 + 模块级函数逐字 + 新 LF sha 钉死 |
| A4″ | `aliccp_benchmark/rp_longer_budget.py`（新） | 持久性分析器：钉死常量 + `persistence_verdict` + `analyze_runs`（只读两 run 记录 → 判定 + 描述量 + identity checks + context）+ `python -m` CLI（输出 JSON）。`persistence_verdict`/`_read_json`/`_a_class_ok` 自 `bbd8a61:aliccp_benchmark/longer_budget.py`（blob `c8f66409a57d1e23d0629bedbae5ed89a6ef9523`；LF sha256 `831897fde2b55a115e3a5b362a73de64bd3fdd8cd703227caab3ce5469a7149f`）移植：`_read_json`/`_a_class_ok` **逐字**；`persistence_verdict` **恰 3 处**适配（签名行 + 首行 docstring + checks 字典新增 `"mechanism_gates_pass"` 一项——落实 §5-C3c）；常量块 / `_arm_description` / `analyze_runs` / `main` 为按本实验判据新写（RP variant 记录、参照路径+sha 钉死、机制门禁、5-epoch 残差 prompt context）| 守卫测试：段级重建等式（上述 3 处替换）+ `_read_json`/`_a_class_ok` 段逐字节 + 行为测试（边界/identity/只读/fixture 端到端） |
| A5″ | `aliccp_benchmark/tests/test_residual_prompt_longer_budget.py`（新） | 本分支守卫 + 接线 + 分析器测试（§7 I2/I4/I5/I6/I7/I8）| 本文件即为守卫 |
| A6″ | `verify_seed2_artifact_integrity.py`（新；预注册前完整性核验）与 `verify_rp_longer_budget.py`（新；运行后独立复核）| 只读脚本；报告 JSON 落 `artifacts/aliccp_bench/audit/longer-budget-rp/` 与 run 目录（gitignore）。**本分支将脚本本身跟踪入库**（与 seed1/seed2 未跟踪先例的偏离，§11 披露）| 完整性核验结果 28/28（§1.5）；复核脚本纯 JSON 重推（§10.5）|

**被禁止的适配**：机制公式/初始化/门控语义/注入点/活性带/`BOUND_TOL`/G7 比值/U1 阈值/分类规则；`protocol.py`/`metrics.py`/`multitaskrec/*`/`config.py` —— 零改动（静态守卫：`git diff --name-only 8133d32` ⊆ 白名单）。

**epoch/patience override 说明**：`--epochs`/`--patience` 自基点 `8133d32` 已存在（默认 = `protocol.STAGE2_EPOCHS/PATIENCE`），本实验**不新增**数值语义，只测试其 override 接线（§7 I4）并在 run 记录中钉死 `epochs=10/patience=3`；默认行为逐位不变。

**class C 注（in-run `rp_arm.U`）**：处理臂 run 内 `rp_arm.U` 的 U1/U2 按机制模块钉死常量（5-epoch 基线值）计算，属 5-epoch 复现口径的**class C 诊断**；本实验的持久性判定（§5）由 A4″ 分析器对**新 10-epoch 基线臂**机械计算，**不得**用 in-run `U` 读数替代。

---

## 7. 不变量（由 `test_residual_prompt_longer_budget.py` 与移植测试强制，不接受人工目测）

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 基线臂逐位不变：`RP.build_newtask(baseline)` 即 `NewTask`；`run_stage2` 默认参数下记录与机制臂键集差异恰为 `variant`/`rp_arm`/`prompt_hidden`（及 `-rpg` 后缀、prompt_report）；共享参数/前向在 α=0 时逐位一致 | 移植测试（构造恒等/初值恒等/端到端 purity） |
| I2 | **适配逐字守卫**：A1″/A2″/A3″/A4″ 的字节/AST/重建等式钉死（§6 表） | `git show` + sha256(LF) + 段级重建等式 |
| I3 | 机制语义不变：α 初值 0、恒等/界/活性带/参数预算测试全绿（移植测试原样通过） | 移植测试（TestConstructionIdentity/TestNormControl/TestArmVerdict/TestParamReport…） |
| I4 | **CLI/override**：`--epochs/--patience/--variant/--prompt-reference-newtask` 默认值 == 协议/机制常量（逐位不变）；`--tag long` 可解析且默认仍 `short`；`--epochs 10 --patience 3 --tag long` 经 `main()` 正确线程化到 `run_stage2` 调用参数（monkeypatch 捕获 kwargs）；处理臂 `--variant/--prompt-reference-newtask` 线程化 + run_id `-rpg` 后缀 + 基线 run_id 无后缀 | parser 断言 + `main()` 捕获式测试 |
| I5 | 判定边界：Δtest 恰 `+0.0055` 过（闭）、`−1e-12` 不过；Δval 恰 `0` 不过（严格）、`+1e-12` 过；`a_class_pass=False` ⇒ `NOT_PERSIST`；`mechanism_gates_pass=False` ⇒ `NOT_PERSIST`（即使 Δ 达标）；四条件全过 ⇒ `PERSISTS` | 边界/真值表测试 |
| I6 | 分析器：只读（不改 run 产物）；identity mismatch ⇒ `INVALID`；机制门禁失败集合逐项落盘；输出键齐全（判定/描述量/context/identity/mechanism）；context 常量 == 5-epoch 残差 prompt 记录钉死值；夹具端到端 + 篡改 fixture（门禁全过 + Δ 达标 ⇒ `PERSISTS`）| CPU 极小夹具端到端 + 篡改 fixture |
| I7 | 静态守卫：相对 `8133d32` 的全部跟踪改动 ⊆ §6 白名单；`protocol.py`/`metrics.py`/`multitaskrec/*`/`config.py` 零 diff | `git diff --name-only` |
| I8 | 预注册文档 token 钉死：预算 10/3、tag `"long"`、seed `1688723740`、stage1_id、参照路径/sha、5-epoch context run_id/Δ、阈值 `0.0055`、`PERSISTS`/`NOT_PERSIST`/`INVALID`、适配清单 commit/blob、确定性预期表述 | 文档文本断言（doc-token 方法） |

CPU 极小夹具不构成任何性能证据，只验证语义与接线。

---

## 8. 运行方式（恰好一次；命令留档；cwd = 本 worktree 根；本 worktree 无 .venv，用主树解释器）

```powershell
# 0) 测试（全部不变量；先写测试后写实现）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests -v

# 1) 基线臂（唯一一次；tag long；epochs 10 / patience 3；seed-2 产物）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag long --epochs 10 --patience 3 --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f

# 2) 处理臂（唯一一次；同产物、同 seed、同预算；残差 prompt + 钉死参照头）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag long --epochs 10 --patience 3 --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f `
    --variant residual-prompt `
    --prompt-reference-newtask artifacts/aliccp_bench/runs/20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt

# 3) 持久性分析（纯分析，恰一次；只读 run 记录 + 参照文件哈希）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m aliccp_benchmark.rp_longer_budget `
    --baseline-run <基线 run_id> --arm-run <处理臂 run_id>

# 4) 运行后独立复核（只读；JSON 重推 + 交叉一致性）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe verify_rp_longer_budget.py `
    --baseline-run <基线 run_id> --arm-run <处理臂 run_id>
```

- **前置条件**：开跑前先提交实现（§4 步骤 1）；每臂 run 启动时 `git.dirty` 必须为 false（§4 清洁树纪律）。**前台执行并等待至完成**（更长预算协议先例记载后台执行被会话终止杀死；本预注册明确要求前台）。
- 数据文件经目录联接（junction）复用 main tree 的 `dataset/AliCCP/`（只读；`dataset/` 被 gitignore，不入库）。
- **墙钟预算**：每臂 ≈ 4–5 min（5-epoch 实测 143–156 s 的 2× 外推 + 加载/评测），分析/复核 < 1 min。

## 9. 局限（预声明）

1. **单 seed、每臂 1 run**：仍无同 seed run-to-run 噪声估计（A3 SKIP）；10-epoch 与 5-epoch 的 Δ 差异混合了预算效应与 run 噪声，二者不可分离（确定性核对 §3 为描述性佐证，不作噪声估计）。本实验回答"更长预算下增量是否仍在"，不回答问题"增量随预算如何变化的曲线"。
2. **判定依赖单一汇总统计**（test 1M 前缀的 BSI AUC 差值；SE ≈ 0.0025 @AUC0.7 ⇒ +0.0055 ≈ 2.2×SE）；单点估计，不引入事后置信区间叙事。
3. **无独立重评测 pass**：判定直接取 run 记录值（单次 test 评估纪律，§4）；记录值的正确性依赖 bench 代码路径与复制源**逐字一致**（I2 静态守卫）+ 同类代码路径的独立重评测先例（seed-2 复现参照核验 21/21，diff=0.0）。此为范围决策，如实披露。
4. **in-run `rp_arm.U` 的配对对象是 5-epoch 常量**（class C 诊断，§6 注）——不得与 §5 持久性判定混读。
5. **B4 若 FAIL 为继承缺陷**（共享 Stage-1 语义，两臂逐位相同），本实验无法修复也不据此做任何主张。
6. **单数据集（AliCCP）**：结论不推及 CensusIncome / ByteRec。
7. **10-epoch 为固定预算**：更长（如 20+）预算下的行为不外推；本轮不外延预算。若两臂在 ep10 仍改进（右删失再现），结论仅限"10-epoch 判定点"。

## 10. 结果（运行后回填，不回写第 1–9 节）

（待运行后回填）

## 11. 偏离披露（预声明 + 运行后补）

- **预注册前澄清提交**（先于任何实现与 run；判据、预算、阈值、分类规则**零改动**；仅把 §0/§1.5/§3/§6 中已按"…"省略的内容寻址哈希与 run id 写全，并补 79ddefa 系谱 blob）：`docs: complete full content-addressed hashes and run ids in longer-budget prereg (pre-run clarification, criteria unchanged)`。与更长预算协议 `53346eb` 先例同类。

- **分支工作流**：自 `infra/aliccp-fair-benchmark` @ `8133d32` 拉出（不从 seed-2 复现分支拉出）；机制经 A1″ 逐字节守卫钉死等价于 `013e105`（后者经种子复现分支的重建等式钉死于 `79ddefa`）。协议未合并入 master，与同批 exp 分支先例一致。
- **tag `"long"` 词汇**：沿用更长预算协议（`47ecb0c`/`bbd8a61`）先例；`protocol.py` 零修改。
- **校验脚本入库**：`verify_seed2_artifact_integrity.py` / `verify_rp_longer_budget.py` 本分支**跟踪入库**（seed1/seed2 先例为未跟踪）——改进证据链可复现性；报告 JSON 仍不入库（`*.json` gitignore）。
- **run 间提交**（§4）：基线 run 的 SUMMARY 追加行在跑处理臂前提交，两 run 记录 `git.dirty=false`。
- **不做第二次 test 遍历**（§9.3）：本实验判定只用 run 内单次 test 评估值与只读分析。
- **执行记录（运行后补）**：C1（预注册）→ C2（移植 + 分析器 + 守卫，测试全绿）→ 基线臂前台单次 run → C3（基线 SUMMARY 行）→ 处理臂前台单次 run → 分析 + 复核 → C4（§10/§11 回填 + 处理臂 SUMMARY 行）。**未 push**（本地提交；推送需用户显式批准）。

---

**改动清单（白名单；静态守卫以 `8133d32` 为基准）**

| 文件 | 处置 |
|---|---|
| `docs/superpowers/specs/2026-10-04-aliccp-stage2-residual-prompt-longer-budget-design.md` | 本文件（C1 单独提交） |
| `aliccp_benchmark/residual_prompt.py` | 新增（A1″：逐字节 == `013e105`） |
| `aliccp_benchmark/bench.py` | 新增（A2″a：逐字节 == `013e105`） |
| `run_aliccp_benchmark.py` | A2″b：== `013e105` 恰 1 行适配（`--tag` choices 增加 `"long"`） |
| `aliccp_benchmark/tests/test_residual_prompt.py` | 新增（A3″：== `013e105` 恰 4 处适配） |
| `aliccp_benchmark/rp_longer_budget.py` | 新增（A4″：`persistence_verdict`/`_read_json`/`_a_class_ok` 自 `bbd8a61` 移植 + 本实验判定/描述/identity） |
| `aliccp_benchmark/tests/test_residual_prompt_longer_budget.py` | 新增（A5″：守卫 + 接线 + 分析器测试） |
| `verify_seed2_artifact_integrity.py` | 新增（A6″：预注册前完整性核验脚本） |
| `verify_rp_longer_budget.py` | 新增（A6″：运行后独立复核脚本） |
| `artifacts/aliccp_bench/SUMMARY.md` | 只追加 2 行（runner 自动） |
| 其余全部（`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`aliccp_benchmark/metrics.py`、既有测试三件、`AliCCP_*.py`、`CensusIncome_*.py`、`baseline/*`、`mask/*`、`analysis/*`） | **零改动**（静态守卫断言 `git diff --name-only 8133d32` ⊆ 白名单） |
