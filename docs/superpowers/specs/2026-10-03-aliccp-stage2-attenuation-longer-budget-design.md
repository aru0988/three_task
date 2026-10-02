# AliCCP Stage-2 固定衰减增量在更长优化预算下的持久性检验（robustness / alternative-explanation，非新方法）

- **状态**：预注册已写死（本文件在本分支任何 run 之前提交）；实现与结果见 §10（运行后追加，不回填）
- **日期**：2026-10-03
- **适用分支**：`exp/aliccp-stage2-attenuation-longer-budget`（自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；**不从**任何 exp 分支拉出、不继承其运行时状态；`aliccp_benchmark/protocol.py` 零修改）
- **协议依赖**：`docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（AliCCP 公平评测协议）与 `aliccp_benchmark/protocol.py`。本分支**只引用**，不修改其任何文件与常量（SUMMARY 列、A/B 门禁阈值、预算、指纹定义均不变）
- **机制与先例来源（只读引用）**：处理臂机制 = `exp/aliccp-stage2-specific-attenuation-control` @ `3314420`，经 `exp/aliccp-stage2-attenuation-seed-replication` @ `f589e61` **逐字移植**（本分支自 `f589e61` 逐字复用，静态守卫见 §7）；seed-2 5-epoch 配对记录 = 同分支 run `20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07`（基线）/ `20261003-0627-p2M-v500k-t1M-m1688723740-short-79b5e07-sattn`（处理臂）
- **定位声明**：本实验是**稳健性 / 替代解释检验**（benefit 是持久收益还是仅仅加速收敛），不是新方法，不主张任何新颖性；不改系数、不调参、不换结构、不改损失。与 seed-2 5-epoch 配对相比**唯一的变化 = Stage2 max epochs 5→10 与 patience 2→3**（§3）。结论口径：seed-2 观察到的固定衰减正增量在更长 Stage2 预算下是否按预注册判据**仍然成立**（PERSISTS vs NOT_PERSIST）。

---

## 0. 目的与判定问题

seed-2（`m1688723740`，Stage-1 产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f`）在 5-epoch Stage2 预算下观察到处理臂（specific 混合乘以固定常数 c = 0.6972233730330467）相对配对基线 Δtest = `+0.007713701281711671`、Δval = `+0.004483122552346286`（同 stage1_id、同顺序、单次 test 评测；`STABILITY_SUPPORTED`）。但两条 5-epoch 轨迹**都在最后一个 epoch 仍在改进**（见 §1），判定点落在轨迹的右删失处，无法区分"持久收益"与"仅加速收敛"。

**判定问题 Q**：在同一 seed-2 Stage-1 产物上，把 Stage2 预算延长到 **10 epochs / patience 3**（其余全部钉死），处理臂相对配对基线的 **Δtest ≥ +0.0055 且 Δval > 0** 是否仍成立，且全部构造/冻结门禁通过？
- 成立 → **`PERSISTS`**（增量在更长预算下持续；非纯加速解释）
- 不成立 → **`NOT_PERSIST`**（增量被更长预算抹平或反向；与"仅加速收敛"解释一致）

**纪律声明（预注册）**：旧 5-epoch 两条 run 与 Δ = +0.0077 **只作本实验的动机与事后对照（context），不构成本实验判定结果的任何部分**；本实验的判定只用 §4 新跑的成对 run（§5 判据），不使用旧 run 的任何数值入判。

---

## 1. 5-epoch 轨迹审计（先于任何新 run；只读，来自 on-disk run 记录）

来源：`artifacts/aliccp_bench/runs/<run_id>/metrics.json` 与 `gate_report.json`（本机磁盘留存，逐位复核；与 `f589e61` 文档 §10.1 记录一致）。两臂同 `stage1_id`、同 model seed、同顺序、单次 test 评测。

| 项 | 基线臂（`20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07`） | 处理臂（`20261003-0627-p2M-v500k-t1M-m1688723740-short-79b5e07-sattn`） |
|---|---|---|
| `epochs / patience`（记录值） | 5 / 2 | 5 / 2 |
| 逐 epoch val AUC（BSI） | 0.4645711559 / 0.4856448122 / 0.5142344439 / 0.5508809596 / 0.5809347092 | 0.4650750989 / 0.4898335356 / 0.5283062258 / 0.5623536460 / 0.5854178318 |
| 逐 epoch train loss | 0.0819 / 0.0519 / 0.0497 / 0.0487 / 0.0481 | 0.0831 / 0.0528 / 0.0500 / 0.0486 / 0.0479 |
| best_epoch / best val | 5 / 0.5809347091990792 | 5 / 0.5854178317514255 |
| test AUC（单次） | 0.5974422649550507 | 0.6051559662367624 |
| 逐 epoch Δval(arm−baseline) | — | +0.000504 / +0.004189 / +0.014072 / +0.011473 / +0.004483 |
| early stop | **未触发**（5/5 epoch 严格改进，patience=2 从未被行使；`len(per_epoch)=5=epochs`） | 同左 |
| A 类门禁 | A1/A2/A4/A5/A6 PASS，A3 SKIP | 同左 |
| B 类门禁 | B1/B2/B3 PASS；B4 FAIL（继承 seed-2 stage-1 cluster 退化） | 同左（逐位相同 stage-1） |
| Δtest / Δval（配对） | — | **+0.007713701281711671 / +0.004483122552346286** |
| 墙钟 | 142.9 s | 147.2 s |

**审计结论（决定预算选择的轨迹事实，与结果无关）**：
1. 两臂 5/5 epoch **单调改进、best=ep5（右删失）**——5-epoch 判定点截断了仍在下降的优化轨迹，无法回答"持久 vs 加速"。
2. 逐 epoch Δval 全程为正但**末段收窄**（+0.0141 → +0.0115 → +0.0045）：增量是否在更长预算下继续收窄至阈值以下，必须实测。
3. early stop 在 5 epoch 内从未被行使——patience 的取值在延长预算时必须**先于 run** 重新钉死（§2）。

**附注（顺序纪律，与本实验设计相关的事实）**：5-epoch 处理臂 run 的磁盘记录里 `git.dirty = true`（基线 run 先追加了受跟踪的 `SUMMARY.md`，处理臂 run 启动时树已脏）。本实验预注册**run 间提交**规避（§4），使两条新 run 都在 `git.dirty = false` 下运行。

---

## 2. 预算选择（基于 §1 轨迹，不基于任何结果；先于 run 写死）

| 项 | 取值 | 依据 |
|---|---|---|
| Stage2 **max epochs** | **10** | 5-epoch 为右删失点（§1 结论 1）；10 = canonical 5 的 2×，为"modest but informative"的固定预算；单臂墙钟 ≈ 4–5 min（按 5-epoch 实测 143–147 s 外推），总运行预算可控 |
| Stage2 **patience** | **3** | canonical 值 2 在 5 epoch 内从未被行使（无任何非改进 epoch）；延长预算后若出现 1–2 个 epoch 的暂时平台/回落，patience=2 会在恢复前截断，而 patience=3 允许**至多 2 个连续非改进 epoch** 后停止，既容忍暂时波动又保证终止；两臂**同一规则**，先于 run 写死 |
| 其余全部 | 不变 | §3 |

- 若某臂在 10 epoch 内触发 early stop：属预注册行为，**照实记录**（`stopped_early`、停止 epoch、best epoch 仍取 val 最优并重载），不作为失败、不重跑。
- **禁止**：看到轨迹后改预算/patience；事后挑选 epoch 报数（判定只用各臂 val 选出的 best checkpoint 与单次 test 评估，§5）。

---

## 3. 控制变量（哪些变、哪些钉死）

| 项 | seed-2 5-epoch（先例，仅对照） | 本实验（10-epoch） | 说明 |
|---|---|---|---|
| **Stage2 epochs** | 5 | **10** | 唯一变化之一（§2） |
| **Stage2 patience** | 2 | **3** | 唯一变化之二（§2） |
| Stage-1 产物 | `s1-5c060b9c-m1688723740-e3-4e1b5c6f` | **同一产物（钉死；不重训 Stage-1）** | 预跑校验（本分支 8133d32 上复验，分支切换前后两份文件字节一致）：`backbone_sha256 = e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c`；`env_ids_sha256 = 5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0`；文件 sha256：`backbone.pt = cd7b033423499e0d36eea988835cea4ac4e2a8e26427a2aea627333822101e0b`、`env_ids.pt = 4660be5aaa3c59f53dd5b4394f77114a87db69064b32c45517db049a6b9157e7`、`meta.json = 61a66d81ce3dcf6bae64f4c2b37cf12e722943def646336a588746aba932931d`；`fingerprint_sha256 = 5c060b9c…`（A2/A5/A6 在 run 内再次逐位校验） |
| model seed | `1688723740` | `1688723740`（不变） | 头初始化 + 训练 dropout；两臂同进程重播种 → 头初始化与样本顺序构造性相同 |
| env seed / 预算 / 前缀 | `20261003` / p2M-v500k-t1M | 同（不变） | `env_ids` 来自产物（不重抽） |
| 系数 c | `0.6972233730330467` | **同值钉死**（不调参） | float64 字面量，`c := 1 − 0.3027766269669533`；float32 铸造值 `0.6972233653068542`（记录备查，判定不依赖） |
| batch / lr / 结构 / dropout / loss / 温度 | 协议 §8.1 默认 | 同（不变） | — |
| 评测口径 | val 选点 + best 重载 + test 单次评估 | 同（不变） | 协议 §7.5/§9.1；**no post-hoc epoch selection** |
| tag / run_id | `short` | **`long`**（新 tag 词汇，见 §11） | run_id 身份标识；`smoke`/`short` 语义不变 |

---

## 4. 运行计划（恰好一次）与清洁树纪律

**运行次数**：基线臂 stage2 **恰好 1 次**；处理臂 stage2 **恰好 1 次**；`longer_budget` 分析（纯分析：只读 run 记录，**不重新评测、不新增 test 遍历**）**恰好 1 次**。**不重训 Stage-1**；已完成的有效 run 一律不重跑。

**无效执行处置（预注册）**：仅当 run 因**工具性原因**（进程崩溃/中断/环境故障，run 未完整落盘）无效时，保留全部现场（含失败日志与半成品目录，不删除），记录原因后可重跑一次；因**结果原因**（判据未过/门禁失败）不构成无效，不重跑，如实记录。

**清洁树纪律（run 间提交；见 §1 附注的动机）**：
1. 提交实现与测试（commit `C1`）→ 树干净；
2. 跑**基线臂**（`git.dirty` 必须为 false，run 记录 commit = `C1`）；
3. **提交基线 run 的 SUMMARY 追加行**（commit `C2`，只允许 `artifacts/aliccp_bench/SUMMARY.md` 一个文件变化）；
4. 跑**处理臂**（`git.dirty` 必须为 false，run 记录 commit = `C2`）；
5. 运行后核验代码同一性：`git diff C2 C1 -- . ':(exclude)artifacts/aliccp_bench/SUMMARY.md'` 必须为空（两 run 的代码逐字相同，run_id 记不同 commit 仅为 SUMMARY 追加行的归属）；
6. 提交结果（§10 文档更新 + 处理臂 SUMMARY 追加行，commit `C3`）。

**命令（留档，见 §8）**。分析输出：`artifacts/aliccp_bench/runs/<arm-run>/longer_budget_compare.json`（gitignore）。

---

## 5. 预注册判据（看到结果前写死；看到结果后不得修改）

对照 = **本实验新跑的 10-epoch 基线臂**（同 `stage1_id`、同 model seed、同顺序；**不得**用 5-epoch run 做对照）。

| 条件 | 判据 |
|---|---|
| C1 | Δtest := `test_auc_bsi(arm)` − `test_auc_bsi(baseline)` **≥ +0.0055**（闭区间） |
| C2 | Δval := `best_val_auc_bsi(arm)` − `best_val_auc_bsi(baseline)` **> 0**（严格；best 由各臂 val 选点规则确定，非事后挑选） |
| C3 | **构造/冻结校验全过**：两 run 的 A1/A2/A4/A5/A6 全 PASS（A3 SKIP 视为通过，与协议 `hard_pass` 的 A 类口径一致），**且**分析步 identity checks 全过：两臂同 `stage1_id`（== §3 钉死值）、两臂 `model_seed` 相等（== 1688723740）、baseline `spec_attenuation == 1.0`、arm `spec_attenuation == 0.6972233730330467`、两 run 记录 `epochs == 10` 且 `patience == 3` |

- C1 ∧ C2 ∧ C3 → **`PERSISTS`**；否则 → **`NOT_PERSIST`**。
- identity/记录不一致（C3 的 identity 部分失败）→ **`INVALID`**（完备性分支：预期不可达；出现即表示运行偏离预注册，按实记录并停止解读）。
- 阈值来源：+0.0055 沿用 seed-2 复现预注册的 provisional PI 效用口径（协议 §10.1 "+0.005" 的保守取整版本；test 1M 前缀 BSI 负例 ≈1.6 万 ⇒ SE ≈ 0.0025 @AUC0.7，+0.0055 ≈ 2.2×SE）。
- **best epoch 差异、是否 early stop、逐 epoch Δ 形态 = 描述量（§10 照实报告），不参与判定**——防止任何事后 epoch 挑选。

**B 类**（B1–B4）照常判定并披露；**继承披露口径**：B4 预期 FAIL 由共享 Stage-1 产物（cluster 退化）决定（两臂逐位相同），属继承缺陷、不归因于处理臂、不据此否定本臂。

---

## 6. 实现面（最小；TDD；默认关逐位一致）

| 文件 | 改动 | 来源 |
|---|---|---|
| `multitaskrec/model.py` → `NewTask` | `spec_attenuation=1.0` 构造参数（校验 `0 < c ≤ 1`）+ `forward` 条件乘（默认 1.0 整句跳过） | **逐字复用** `f589e61`（其自身逐字移植自 seed-1 处理臂 `3314420`）；全文件字节相等，静态守卫 §7 |
| `aliccp_benchmark/metrics.py` | 追加：种子/系数常量块（`CANONICAL_ALICCP_SEED_LIST`、`PROTOCOL_MODEL_SEED`、`REPLICATION_MODEL_SEED`、`NULL_RUN_NULL_MEAN`、`SPEC_ATTENUATION_COEF`、`SPEC_ATTENUATION_COEF_F32`）+ `SourceGateStats` + `_f64` + `pred_dispersion` | 逐字复用 `f589e61`（AST 段字节守卫 §7）；**不移植** replicate 专属函数（pearson/spearman/logit_clip/paired/cross_arm/判定函数）——本实验无独立重评测步（§9.3） |
| `aliccp_benchmark/bench.py` | `stage2_run_id` + `newtask_attenuation_probe`（逐字复用）；`run_stage2(..., spec_attenuation=1.0)` 接线、probe 落盘（`probe`/`trainable_params`）、`config/metrics` 记录字段、日志行 | 逐字复用 `f589e61`：两函数 AST 段字节守卫；`run_stage2` 可执行结构 AST 等价（去 docstring，docstring 的文档引用适配为本文件） |
| `run_aliccp_benchmark.py` | stage2 新增 `--spec-attenuation`（默认 1.0）；`stage2_run_id` 接线与 probe 打印；`--tag` choices 增加 `"long"`（默认仍 `"short"`）；**移除** replicate 子命令（本分支不含 `replicate.py`） | 前两项逐字复用 `f589e61`；tag 扩展为本实验所需的最小 CLI 支持（§11） |
| `aliccp_benchmark/longer_budget.py`（新） | 钉死常量（§3/§5 的判据、seed2 产物 id、5-epoch context 记录）+ `persistence_verdict`（C1/C2/C3 纯函数）+ `analyze_runs`（只读两 run 的 metrics/gate 记录 → 判定 + 描述量 + identity checks）+ `python -m` CLI（输出 JSON） | 本分支新写（判定规则 §5） |
| `aliccp_benchmark/tests/test_longer_budget.py`（新） | §7 不变量 | 改编自 `f589e61` 测试（I1/I2/判定边界/CLI 模式） |
| `artifacts/aliccp_bench/SUMMARY.md` | 运行后追加 2 行（append-only） | 协议 §11.2 |

**明确不改**：`aliccp_benchmark/protocol.py`（零改动）、`config.py`、`AliCCP_MPTRec.py`、`AliCCP_NewTask.py`、`baseline/*`、`multitaskrec/train.py`、`multitaskrec/dataset.py`、`multitaskrec/model.py` 中 `NewTask` 以外的任何类；CensusIncome / ByteRec 任何代码路径不涉及。SUMMARY 只追加行、**永不重写**。

**epoch/patience override 说明**：`--epochs`/`--patience` 自基点 `8133d32` 已存在（默认 = `protocol.STAGE2_EPOCHS/PATIENCE`），本实验**不新增**数值语义，只测试其 override 接线（§7 I4）并在 run 记录中钉死 `epochs=10/patience=3`；默认行为逐位不变。

---

## 7. 不变量（由 `test_longer_budget.py` 强制，不接受人工目测）

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 默认臂（`spec_attenuation=1.0`）前向/反向与基点 `8133d32:multitaskrec/model.py` 逐位一致；参数集合/state_dict 相同；衰减不消耗 RNG | `torch.equal`（前向 + 全参数梯度）+ `load_state_dict(strict=True)` + 种子对照 |
| I2 | **机制逐字守卫**：`model.py` 全文件字节 == `f589e61` == `exp/aliccp-stage2-specific-attenuation-control`；`metrics.py` 移植项（常量 Assign 段 + 3 个函数）与 `bench.py` 移植函数（`stage2_run_id`、`newtask_attenuation_probe`）AST 段字节 == `f589e61`；`run_stage2` 可执行结构（AST，去 docstring）== `f589e61` | `git show` + AST 段比较（`ast.get_source_segment` / `ast.dump`）|
| I3 | 开启臂前向等于写死公式（`new_spec_rep' = c × Σ W_k spec_rep_k`）；零新增可训练参数；`get_l2_reg()` 不变；系数钉死（`== 1.0 − 0.3027766269669533`、f32 铸造值）；非法值构造抛错 | 独立参考实现 `torch.equal` + 计数 + 异常测试 |
| I4 | **CLI/override**：`--epochs/--patience/--spec-attenuation` 默认值 == 协议常量（逐位不变）；`--tag long` 可解析且默认仍 `short`；`--epochs 10 --patience 3` 经 `main()` 正确线程化到 `run_stage2` 调用参数（monkeypatch 捕获 kwargs），arm 臂 `--spec-attenuation` 线程化 + run_id `-sattn` 后缀 | parser 断言 + `main()` 捕获式测试 |
| I5 | 判定边界：Δtest 恰 `+0.0055` 过（闭）、`−1e-12` 不过；Δval 恰 `0` 不过（严格）、`+1e-12` 过；`a_class_pass=False` ⇒ `NOT_PERSIST`（即使 Δ 达标）；三条件全过 ⇒ `PERSISTS` | 边界/真值表测试 |
| I6 | 分析器：只读（不改 run 产物）；identity mismatch ⇒ `INVALID`；输出键齐全（判定/描述量/context/identity checks）；context 常量 == 5-epoch 记录钉死值；夹具端到端 | CPU 极小夹具端到端 + 篡改 fixture |
| I7 | 静态守卫：相对 `8133d32` 的全部跟踪改动 ⊆ §6 白名单；`protocol.py` 零 diff | `git diff --name-only` |
| I8 | 预注册文档 token 钉死：预算 10/3、tag `long`、seed `1688723740`、stage1_id、系数、阈值 `0.0055`、5-epoch context run_id/Δ、"仅动机"表述、`PERSISTS`/`NOT_PERSIST` | 文档文本断言 |

CPU 极小夹具不构成任何性能证据，只验证语义与接线。

---

## 8. 运行方式（恰好一次；命令留档）

```powershell
# 0) 测试（全部不变量；先写测试后写实现）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests -v

# 1) 基线臂（唯一一次；tag long；epochs 10 / patience 3；seed-2 产物）
python run_aliccp_benchmark.py stage2 --tag long --epochs 10 --patience 3 --model-seed 1688723740 --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f

# 2) 处理臂（唯一一次；同产物、同 seed、同预算，仅加固定系数）
python run_aliccp_benchmark.py stage2 --tag long --epochs 10 --patience 3 --model-seed 1688723740 --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f --spec-attenuation 0.6972233730330467

# 3) 持久性分析（纯分析，恰一次；只读 run 记录）
python -m aliccp_benchmark.longer_budget --baseline-run <基线 run_id> --arm-run <处理臂 run_id>
```

- **前置条件**：开跑前先提交本批改动（§4 步骤 1）；每臂 run 启动时 `git.dirty` 必须为 false（§4 清洁树纪律）。
- 数据文件经目录联接（junction）复用 main tree 的 `dataset/AliCCP/`（只读；`dataset/` 被 gitignore，不入库）。
- **墙钟预算**：每臂 ≈ 4–5 min（5-epoch 实测 143–147 s 的 2× 外推 + 加载/评测），分析 < 1 min。

---

## 9. 局限（预声明）

1. **单 seed、每臂 1 run**：仍无同 seed run-to-run 噪声估计（A3 SKIP）；10-epoch 与 5-epoch 的 Δ 差异混合了预算效应与 run 噪声，二者不可分离。本实验回答"更长预算下增量是否仍在"，不回答问题"增量随预算如何变化的曲线"。
2. **判定依赖单一汇总统计**（test 1M 前缀的 BSI AUC 差值；SE ≈ 0.0025 @AUC0.7 ⇒ +0.0055 ≈ 2.2×SE）；单点估计，不引入事后置信区间叙事。
3. **无独立重评测 pass**：本分支不含 `replicate.py`，判定直接取 run 记录值（单次 test 评估纪律，§4）；记录值的正确性依赖 bench 代码路径与复制源**逐字一致**（I2 静态守卫）+ 5-epoch 先例中同代码路径的独立重评测完整性通过（19/19，diff=0.0）。此为范围决策，如实披露。
4. **B4 若 FAIL 为继承缺陷**（共享 Stage-1 语义，两臂逐位相同），本实验无法修复也不据此做任何主张。
5. **单数据集（AliCCP）**：结论不推及 CensusIncome / ByteRec。
6. **10-epoch 为固定预算**：更长（如 20+）预算下的行为不外推；本轮不外延预算。

---

## 10. 结果（实测；运行后追加，不得回填预期值）

**判定：`NOT_PERSIST`（§5 机械计算：C1 与 C2 均未通过；C3 通过——非 INVALID，亦非构造缺陷导致的假阴性）。**
Δtest = `+0.0005273306552253665`（< +0.0055，**C1 失败**）且 Δval = `-0.002653370173158476`（≤ 0，**C2 失败**）；两 run A 类门禁全 PASS（A3 SKIP）、identity/记录校验 8/8 通过；B4 继承 FAIL 照实披露。**检测结论：seed-2 上 5-epoch 观察到的固定衰减正增量在 10-epoch 预算下未能持续——更长预算下基线在 val 侧追平并小幅反超，test 侧基本打平（+0.0005，≈0.2×SE）。**

### 10.1 运行记录（各恰一次；无重跑、无无效执行）

| 项 | 基线臂 | 处理臂 |
|---|---|---|
| run_id | `20261003-0724-p2M-v500k-t1M-m1688723740-long-bbd8a61` | `20261003-0729-p2M-v500k-t1M-m1688723740-long-2d2bc5e-sattn` |
| commit / dirty | `bbd8a61`（C1）/ **false** | `2d2bc5e`（C2）/ **false** |
| stage1_id | `s1-5c060b9c-m1688723740-e3-4e1b5c6f` | 同左（A5/A6 逐位校验通过） |
| model seed / env seed | `1688723740` / `20261003` | 同左（头 seed/顺序构造性相同） |
| epochs / patience（记录值） | 10 / 3 | 10 / 3 |
| `spec_attenuation`（记录值） | `1.0` | `0.6972233730330467` |
| 逐 epoch val AUC | 0.4645711559 / 0.4856448122 / 0.5142344439 / 0.5508809596 / 0.5809347092 / 0.6024307437 / 0.6173782928 / 0.6269454538 / 0.6342338048 / 0.6401127812 | 0.4650750989 / 0.4898335356 / 0.5283062258 / 0.5623536460 / 0.5854178318 / 0.6044688301 / 0.6160047496 / 0.6247308861 / 0.6316211842 / 0.6374594110 |
| 逐 epoch train loss | 0.0819 / 0.0519 / 0.0497 / 0.0487 / 0.0481 / 0.0475 / 0.0473 / 0.0469 / 0.0467 / 0.0465 | 0.0831 / 0.0528 / 0.0500 / 0.0486 / 0.0479 / 0.0476 / 0.0472 / 0.0470 / 0.0468 / 0.0466 |
| **best_epoch / best val** | 10 / `0.6401127811737995` | 10 / `0.6374594110006411` |
| **test AUC（单次）** | `0.6614121223087556` | `0.6619394529639809` |
| **Δtest / Δval（配对，本实验判定值）** | — | **`+0.0005273306552253665` / `-0.002653370173158476`** |
| 逐 epoch Δval(arm−baseline) | — | +0.000504 / +0.004189 / +0.014072 / +0.011473 / +0.004483 / +0.002038 / **−0.001374** / −0.002215 / −0.002613 / −0.002653 |
| early stop | **未触发**（10/10 epoch 改进，best=ep10，`stopped_early=false`） | 同左（10/10 改进，best=ep10） |
| 墙钟 / 峰值显存 | 250.0 s / 52.0 MB | 260.6 s / 52.0 MB |
| 日志 | `logs/20261003-072450-stage2-long.log` | `logs/20261003-072926-stage2-long.log` |

- 分析输出（恰一次）：`python -m aliccp_benchmark.longer_budget --baseline-run 20261003-0724-…-bbd8a61 --arm-run 20261003-0729-…-2d2bc5e-sattn` → `runs/<arm-run>/longer_budget_compare.json`。
- **两臂 best epoch 相同（=10）且均为末 epoch**：10-epoch 预算对两臂同样仍是右删失（都在继续改进）；增量差异不是"某臂提前选点/早停"的产物。

### 10.2 判定与描述量（分析步机械输出）

| 项 | 值 |
|---|---|
| `verdict.checks` | `delta_test_ge_min = false`（+0.00053 < +0.0055）× `delta_val_positive = false`（−0.00265 ≤ 0）× `a_class_pass = true` |
| `classification` | **`NOT_PERSIST`** |
| `identity_checks`（8 项） | 全部 `true`（同 stage1_id / 钉死 / 同 seed / 钉死 / baseline c=1.0 / arm c=钉死 / epochs=10 / patience=3） |
| `context.five_epoch` | Δtest `+0.007713701281711671`、Δval `+0.004483122552346286`（先例记录；**context only**） |
| `context.delta_test_change_vs_five_epoch` | `-0.007186370626486305` |
| `context.delta_val_change_vs_five_epoch` | `-0.007136492725504762` |

**确定性附注（描述性）**：两条新 run 前 5 epoch 的 val AUC 与 5-epoch 先例 run（`…-79b5e07` / `…-79b5e07-sattn`）**逐位相等**（两臂皆然；跨进程、跨 commit）——同 seed 的 stage2 训练在本环境逐位可复现。因此 5→10 epoch 的结果差异来源是**预算（判定点）**而非 run 噪声。该事实不改变协议 A3 状态（仍 SKIP；预算配置不同，不构成 A3 定义的复跑）。

### 10.3 门禁

| 检查 | 基线臂 | 处理臂 |
|---|---|---|
| A1 / A2 / A4 / A5 / A6 | **PASS** | **PASS**（backbone sha before==after==loaded；`.grad` 全 None） |
| A3 | SKIP（按需复跑，不阻塞） | SKIP |
| B1 | **PASS**（CTR 0.5559、CVR 0.5028、BSI 0.6614） | **PASS**（BSI 0.6619） |
| B2 | PASS（\|val−test\| = 0.0213 ≤ 0.05） | PASS（\|val−test\| = 0.0245 ≤ 0.05） |
| B3 | PASS（gate_mean ∈ [0.05, 0.95]） | PASS |
| B4 | **FAIL（继承 seed-2 stage-1 cluster 退化，非臂引入；逐位同 stage-1）** | 同左 |

`hard_pass = false`（仅 B4 继承失败，与 5-epoch 先例同型）。

### 10.4 诊断（C 类；记录不判定）

| 项 | 基线臂 | 处理臂 |
|---|---|---|
| 头 gate 均值（val） | `[0.84101275, 0.1589871875]` | `[0.807770125, 0.19222978125]` |
| 处理臂 in-run 探针（val 一次遍历，no_grad） | — | `pred_mean=0.9930221257048846`、`pred_std=0.005221704008876466`、`pred_var=2.7266192756316557e-05`、`pred_min=0.7808335423469543`、`pred_max=0.9996875524520874`、`q10/q25/q50/q75/q90=0.9875617861747742/0.9913785457611084/0.9942547082901001/0.996184766292572/0.9973938465118408` |
| 源任务 gate 均值（backbone，冻结） | — | `[0.9240377414264083, 0.539432083903104]`（与 seed-2 5-epoch 臂**逐位相同**——同一 backbone 固有值） |
| trainable params | 8129 | 8129（零新增；`trainable_params == trainable_params_default`） |

- 头 gate 重平衡方向与 5-epoch 复现一致（baseline `[0.8410, 0.1590]` → arm `[0.8078, 0.1922]`，朝均衡小幅移动），但 10-epoch 下该重平衡**未转化为** test/val 的持续增益。

### 10.5 纪律核对

- **运行次数**：基线 stage2 ×1、处理臂 stage2 ×1、`longer_budget` 分析 ×1（纯分析、只读）；**无重跑、无无效执行**、无挑 seed/epoch；系数/阈值/预算自预注册提交换起未修改；无系数调参。
- **commit 链**：`47ecb0c`（预注册）→ `53346eb`（run ID 钉死澄清，判定标准未动）→ `bbd8a61`（实现 + 测试）→ `2d2bc5e`（基线 SUMMARY 追加行）→ 本次（结果追加 + 处理臂 SUMMARY 追加行，C3）。两 run 记录 `git.dirty=false`（§4 清洁树纪律成立）。
- **run 间代码同一性核验**（§4 步骤 5）：`git diff bbd8a61 2d2bc5e -- . ':(exclude)artifacts/aliccp_bench/SUMMARY.md'` 为**空**。
- **SUMMARY**：追加 2 行（append-only，未重写任何既有行）。
- **测试**：72/72 通过（37 基点 + 35 本实验；含机制逐字守卫 I1–I2、判定边界 I5、CLI override I4、静态白名单 I7）。
- **未 push**（本地提交；推送需用户显式批准）。

### 10.6 解读（预注册口径；不越界）

1. **判据机械未过 → `NOT_PERSIST`**：Δtest `+0.00053` 远低于 +0.0055；Δval `−0.00265` 反向。C3 全过排除了"构造/冻结问题导致假阴性"。
2. **形态（描述）**：Δ 全程为正至 ep5，ep6 收窄（+0.0020），**ep7 起转负**并缓慢走低（−0.0027 @ ep10，斜率趋平）——更长预算下基线追平并小幅反超。
3. **不是"衰减有害"主张**：10-epoch 的 test 差 `+0.00053 ≈ 0.2×SE`、val 差 `−0.00265` 均在单 run 噪声量级内；本实验的结论是"**未观察到持续增益（判据未过）**"。
4. **与 5-epoch 对照（context only）**：两臂前 5 epoch 与 5-epoch 先例逐位一致（§10.2 附注）⇒ 5-epoch 的正增量在 ep5 判定点真实存在，但它在更长预算下被基线追平。**该结果与"固定衰减主要加速早期收敛、而非持久抬升"的替代解释一致**——这正是本实验设计要检验的解释；本实验为其提供了支持性证据，但**不做机理归因**（无 null 臂、无重复噪声估计、单 seed）。
5. **跨 seed 原始 AUC 未比较**（§5.3 口径延续）：唯一报告量为同 seed 配对 Δ。
6. **继承披露**：B4 两臂 FAIL（共享 stage-1 cluster 退化；臂无关）；`hard_pass=false` 如常披露。

---

## 11. 偏离披露

- **§4.3.2 工作流偏离**（与 seed-1/seed-2 分支相同先例）：协议"先等协议合并再从 master 拉 exp 分支"；本分支自 `infra/aliccp-fair-benchmark` @ `8133d32` 拉出（未合并入 master），协议文件零修改、SUMMARY 只追加。如实披露。
- **tag `"long"` 词汇扩展**：协议 §11.1 的 run_id 语法为 `<smoke|short>`；本实验引入 `tag="long"`（只改 `run_aliccp_benchmark.py` 的 choices 与 run 记录，`protocol.py` 零修改），使 10-epoch run 的身份不冒用 canonical `short`（5 epoch）语义。属最小 CLI 支持（§6），先于 run 提交。
- **run 间提交**（§4）：基线 run 的 SUMMARY 追加行在跑处理臂前提交，避免 5-epoch 先例中处理臂 run 记录 `git.dirty=true` 的情形（§1 附注）；两 run 代码同一性由 §4 步骤 5 的 diff 核验保证。
- **不做第二次 test 遍历**（§9.3）：与 seed-2 复现分支的 `replicate` 步不同，本实验判定只用 run 内单次 test 评估值。
