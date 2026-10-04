# AliCCP 阶段 2 范数受控残差 Prompt（可学习门控）：canonical 五 seed 配对验证——审计与预注册

- **状态**：预注册已写死（本文件在本分支任何实现、任何 seed3–5 运行**之前单独提交**，commit C1）。结果只在第 10/11 节以新增小节回填；第 0–9 节的判据、数字与运行规程不得在看到结果后改动。
- **日期**：2026-10-05
- **适用分支**：`exp/aliccp-stage2-residual-prompt-five-seed`（自 `infra/aliccp-fair-benchmark` @ `8133d32` **独立拉出**；`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`aliccp_benchmark/metrics.py`、既有测试三件、master 脚本与 `baseline/*` 零修改；不合并 `master`、不触碰其它 worktree）。
- **被审计 / 复现对象（只读引用）**：
  1. 上游评测协议：`docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（下称"AliCCP 评测协议"）——前缀预算、种子、Stage-1 产物、真冻结三件套、A/B/C 门禁、SUMMARY 台账、`provisional` 改进阈值**全部沿用**。
  2. seed1 线：分支 `exp/aliccp-stage2-residual-prompt-gate`，预注册 `221580a`、实现 `79ddefa`、结果 `3e2f083`；唯一有效 arm run `20261004-0214-p2M-v500k-t1M-m1688723512-short-79ddefa-rpg`（`VALID_NEGATIVE`）。
  3. seed2 线：分支 `exp/aliccp-stage2-residual-prompt-seed-replication`，预注册 `e931cd7`、实现 `013e105`、结果 `a40836e`；唯一有效 arm run `20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg`（`VALID_POSITIVE`）。
  4. A3 复跑校准：分支 `infra/aliccp-a3-repeatability-calibration`，预注册 `10ea6fa`、实现 `acba208`、复跑 `428c76f`、报告 `f65412f`（`A3_PASS`：三 run 逐位相同）。
  5. Census 侧祖本（机制钉死的最终祖先）：`99b9510:census_benchmark/residual_prompt.py`（blob `107221b26382da7fd44990e167d9a61da2680d92`）。
- **定位声明**：本分支是**用户显式指定的 canonical 五 seed 配对验证**（评测协议 §12.4 允许的显式扩 seed；A3 证据已到位，见 §1）。**无新颖性主张、无改进主张、不放宽任何门限**；可报告内容仅限：本仓库冻结协议下五 seed 逐 seed 配对判定、二级分类（用户新增）、五 seed 汇总统计、以及是否为 20-epoch 分支立项（预注册条件 §6.4）。历史 U1/U2（`+0.0055` / `>0`）与三标签判定**原样保留、不改写**；二级分类为**新增字段/新增层**，不覆盖任何旧字段。

**提交时序（预注册纪律）**：C1 本文件（审计 + 预注册，单独提交）→ C2 实现 + 测试（TDD；本次运行的 `commit` 字段即该提交）→ 逐 seed 运行（stage1 → 配对 baseline → 只读参照核验/pin → 处理臂；全为前台）→ C3 结果回填（§10/§11）+ SUMMARY 追加 → push。

---

## 0. 判定问题

- **Q1（机制复现，seed3–5）**：在 canonical seed 3/4/5（`1688738016` / `1688749593` / `1688762746`）上，逐字移植（§3 适配清单）的范数受控残差 prompt 是否复现 seed1/seed2 形态：M0 参照身份、G1 构造恒等、G2 零初值恒等、G3 门控/生成器梯度活、G4 范数受控界、G5 活性带、G6 逐样本门控方差、G7 无预测坍缩、G8 参数预算、A 类完整性全过？
- **Q2（within-seed 效用与二级分类）**：每个新 seed 与其**同 seed、同 stage1_id** 配对基线的精确差值下：(i) 历史判据 U1（`Δtest ≥ +0.0055`）与 U2（`Δval > 0`）分别是什么；(ii) 用户新增二级分类（§5.2）落在哪个类别；方向与量级相对 seed1/seed2 如何。
- **Q3（五 seed 汇总与 20-epoch 立项）**：跨五个 canonical seed（逐 seed 与其自身基线配对，**跨 seed 不混**）,(i) §6.3 汇总统计（均值/标准差、两阈值下正例计数、最差 seed、Student-t 95% CI、val 方向、符号一致性、机制稳定性）是什么；(ii) §6.4 预注册条件是否满足（决定是否立项独立 20-epoch 分支）。

---

## 1. 审计（2026-10-05，只读；先于本文件提交）

### 1.1 审计方法与范围

- **原则**：不信任种子分支文档中的转述数字，全部以**磁盘上的原始 run 产物**（各 worktree `artifacts/aliccp_bench/` 下的 `metrics.json` / `config.json` / `gate_report.json` / `prompt_report.json` / `verify_report.json`）与 **git 对象**（blob 哈希、diff）为准重新核对；不重跑、不修改任何历史 run；不创建 canonical 产物。
- 审计脚本为一次性只读脚本（`$CLAUDE_JOB_DIR/tmp/audit*.py`，未跟踪、不入库）；审计结论以下述 A1–A8 八项等价性为准。

### 1.2 审计结果（逐项，全部 PASS）

| # | 审计项 | 方法与证据 | 结果 |
|---|---|---|---|
| A1 | seed1 配对完整性 | 基线 `20261003-0130-…-b2e17f9` 与 arm `20261004-0214-…-79ddefa-rpg` 的 `config/metrics.json`：`model_seed` 均 `1688723512`、`stage1_id` 均 `s1-5c060b9c-m1688723512-e3-3a30e2c0`；budgets `{2M,500k,1M}`、batch 2000、lr 1e-4、epochs 5、patience 2、`enforce_b=true` 全同 | PASS |
| A2 | seed2 配对完整性 | 基线 `20261003-0624-…-79b5e07` 与 arm `20261004-0325-…-013e105-rpg`：`model_seed` 均 `1688723740`、`stage1_id` 均 `s1-5c060b9c-m1688723740-e3-4e1b5c6f`；其余同上 | PASS |
| A3 | 数值重推（不信任转述） | 从原始 JSON 重推：seed1 `Δtest=+0.005394543882337954`、`Δval=+0.00448174078674779`（U1 FAIL / U2 PASS）；seed2 `Δtest=+0.008103113327467715`、`Δval=+0.008622497456618139`（U1/U2 PASS）；`α_final`、三路 ratio/cos、`geff`、`pred_std` 与各自文档 §9.3/§10.3 表逐位一致；`rp_arm == gate_report.residual_prompt == prompt_report.arm` 三处 JSON 互洽 | PASS |
| A4 | 参照身份闭合 | 两 arm 的 `prompt_report.reference_dispersion.newtask_checkpoint` 分别指向**其配对基线 run 的** `newtask.pt`；M0 复算值（`ref_val_auc`/`ref_pred_std`）与 arm 使用的预注册常量逐位一致（≤1e-9），且 `ref_val_auc` == 配对基线 `best_val_auc_bsi` 逐位相等 ⇒ 参照文件确为配对基线头 | PASS |
| A5 | 实现等价链（Census → seed1 → seed2） | git blob：`79ddefa:aliccp_benchmark/residual_prompt.py` = `5dc7158c…`（LF sha256 `1c6131c7…`，与 seed2 文档记录一致）；`013e105:…` = `674213f6…`（LF sha256 `b3b93b3a…`）；两文件 unified diff **恰为**模块 docstring + 3 个 run-reference 常量行（即 seed2 预注册 §1.4-A1′ 所声明的 2 处适配）；Census 祖本 blob `107221b2…` 在库中可得（`git cat-file` 成功）；`bench.py` 在两实现提交处 blob 同为 `bf364768…`（LF `b063a366…`）、`run_aliccp_benchmark.py` 同为 `229c25a1…`（LF `2966ec39…`）——与 seed2 文档 §1.4-A2′ 记录值一致 | PASS |
| A6 | 基线代码态等价（seed2 线特有问题） | seed2 基线 `79b5e07` 运行于含 `spec_attenuation`（默认 1.0、`!=1.0` 才生效）的代码态（`git diff 8133d32 79b5e07` 非空：`model.py` 守卫改动 + `bench.py`/runner 处理臂接线 + `metrics.py` **纯追加**）。**独立实验证据**：A3 分支复跑 R1 `acba208`/R2 `428c76f` 的代码态 = `8133d32` + 纯新增分析模块（`git diff --name-only 8133d32 <commit>` 不含 `model.py`/`bench.py`/`metrics.py`/runner），其 `test/val/gate_mean` 与 `newtable.pt` 字节（sha `90ee06da…`）与 `79b5e07` **逐位相同** ⇒ 该代码态差异对基线 stage2 路径经实验证明惰性。seed1 基线 `b2e17f9` 的代码态与 `8133d32` 对全部协议 machinery 的 diff 为空（直接一致） | PASS |
| A7 | A3 判定可用性 | A3 报告（`f65412f`）与原始产物核对：三 run `test_auc` 逐位相同（0.0 差）、`backbone_sha256` 同；同配置 run-to-run 噪声 = 0（逐位）。⇒ 本协议下 seed 间差异不含 run-to-run 噪声成分；同时满足评测协议 §12.4"达到 10.1 阈值（seed2 arm U1/U2 通过）且 A3 证据到位"的扩 seed 前提 | PASS |
| A8 | 门禁记录一致性 | 两 arm 的 M0+G1–G8 全 PASS、A 类（A1/A2/A4/A5/A6）PASS、A3 SKIP、`hard_pass=false`（seed1 由 B1/B4 继承 FAIL 决定；seed2 仅 B4）与文档一致；两个 run 的 `git.dirty=false` | PASS |

### 1.3 审计发现的差异与披露（不改变任何结论）

1. **verify 报告条数文档转述偏差**：两臂文档与提交信息称"27/27"；原始 `verify_report.json` 实际 `checks` 条数为 **26（seed1）/ 28（seed2）**，两份 `all_pass=true`、逐项 `pass=true`、无 FAIL 项。（条数差异为文档转述不精确；被复核的实质——全项通过、分类独立重推一致——不受影响。）
2. seed2 文档记录 `REFERENCE_PRED_STD=0.005217193225189258` 显著大于 seed1（0.003373），为不同 backbone/head 的自然差异（seed2 文档已如实标注）；不影响等价性。
3. 上述差异均属**记录层**；A1–A8 的实质等价性不依赖文档转述。

### 1.4 审计结论

**seed1 与 seed2 两条已提交证据在"协议等价 + 实现等价 + 配对完整 + 数值可重推"四方面成立（A1–A8 全 PASS）**，满足"仅当 exact protocol and implementation equivalence are proven 才可使用既有结果"的条件 ⇒ 二者作为 canonical seed 位置 1/2 **进入五 seed 汇总**（历史 U1/U2/分类原样保留）。

---

## 2. canonical seed 列表与五 seed 配对表

评测协议 §6.1 注释与 `metrics.py`（attenuation-seed-replication 分支）`CANONICAL_ALICCP_SEED_LIST` 一致的五 seed：

| 位置 | `model_seed` | 状态 | 配对基线 run | 处理臂 run | `stage1_id`（内容寻址） |
|---|---|---|---|---|---|
| 1 | `1688723512` | 已记录（`3e2f083`，`VALID_NEGATIVE`） | `20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9` | `20261004-0214-p2M-v500k-t1M-m1688723512-short-79ddefa-rpg` | `s1-5c060b9c-m1688723512-e3-3a30e2c0` |
| 2 | `1688723740` | 已记录（`a40836e`，`VALID_POSITIVE`） | `20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07` | `20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg` | `s1-5c060b9c-m1688723740-e3-4e1b5c6f` |
| 3 | `1688738016` | **本次运行** | 本次生成（同名 `-rpg` 无后缀基线） | 本次生成（`-rpg` 后缀） | 本次生成（前缀指纹不变：`5c060b9c…`） |
| 4 | `1688749593` | **本次运行** | 同上 | 同上 | 同上 |
| 5 | `1688762746` | **本次运行** | 同上 | 同上 | 同上 |

- **跨 seed 不混**：一切效用判定均为"该 seed 的 arm − 该 seed 的配对基线（同 `stage1_id`、同 `model_seed`、同预算）"；五 seed 汇总只对**逐 seed 配对差值**做统计（§6），任何形式都不允许跨 seed 拼基线。
- 三个新 seed 的 `stage1_id` 依赖运行期 config 哈希，**运行前不可预知**；运行后逐字记录于 §10 与 SUMMARY。
- 前缀身份三 seed 共用（同一 `PREFIX_TAG`/指纹 `5c060b9c…`，A2/A4 每次运行重算校验）。

---

## 3. 机制重建与移植（pin @ `013e105`）

### 3.1 钉死（内容寻址；§1 审计已复核）

| 项 | 值 |
|---|---|
| 机制实现（pin） | `013e105:aliccp_benchmark/residual_prompt.py`，blob `674213f619c5d5039811c71242a7727118348daf`，LF sha256 `b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc` |
| 该 pin 与 seed1 实现的关系 | 与 `79ddefa:…`（blob `5dc7158ca999c9e7e6217a999c37869231411549`）的差异**恰为** docstring + 3 个 run-reference 常量行（§1-A5） |
| 测试（pin） | `013e105:aliccp_benchmark/tests/test_residual_prompt.py`，blob `768a477b5c71af32c5c59ec6feb20c29f7873557`，LF sha256 `dab737a1422b88befc4c19f2bee408618061de7fd3bf0b2f83ebcdd85fd1cbc2` |
| 守卫测试（pin） | `013e105:aliccp_benchmark/tests/test_residual_prompt_seed_replication.py`，blob `e53a0284ea2d51dbc30eac5133dfbc272f73c20d` |
| 接线（pin，逐字节） | `bench.py` blob `bf3647686838157bf5fd7645615faad9f2d7185d`（LF `b063a366…`）；`run_aliccp_benchmark.py` blob `229c25a1c86719fa6fb6b057d6cff3f4120fa053`（LF `2966ec39…`） |
| Census 祖本（AST 钉死的最终依据） | `99b9510:census_benchmark/residual_prompt.py`，blob `107221b26382da7fd44990e167d9a61da2680d92`，LF sha256 `8139e067a57fd3e144846ca5d39015a1d51657d55da67b9da9e4e06c007658df`；测试 blob `847097b7cb48a56c8593dd76d63a71a21a4c9e4f` |

机制本体（公式 `P(x)=tanh(W2·ReLU(W1x+b1)+b2)`、`g_eff=|α|·‖P‖/√d`、`δ_s=α·(‖h_s‖/√d)·P`、融合前三路注入、`h.detach()` 范数、零初始化标量门控、`isolated_cpu_rng`、`PromptAudit`、`PromptStats`、`evaluate_prompt_diagnostics`、`reference_head_stats`、`param_report`、`arm_verdict` 的 M0+G1–G8+U 判定逻辑与三标签分类）**逐字沿用** pin。

### 3.2 五 seed 移植适配清单（**恰好以下 4 组**；其余钉死）

| 编号 | 适配 | 内容 | 守卫 |
|---|---|---|---|
| A1″ | 机制文件 docstring | `aliccp_benchmark/residual_prompt.py` 模块 docstring → 指向本文件 + 五 seed 上下文 + pin 系谱 | 重建等式：把工作树文件按下列替换序列逆向归一化回 pin 文本 |
| A2″ | `import os` 一行 | `import math` 之后新增 `import os`（import 语句属既有豁免集合） | 同上（重建等式含此行替换） |
| A3″ | **3 个 run-reference 常量行 → 运行期解析** | pin 的 3 行字面量常量改为：`BASELINE_AUC_TEST = float(os.environ["RP_BASELINE_AUC_TEST"]) if "RP_BASELINE_AUC_TEST" in os.environ else None`（`BASELINE_AUC_VAL`、`REFERENCE_PRED_STD` 同式，env 名对应）。语义：**每 seed 的参照/基线值由 runner 运行期显式提供**（配对基线的 `test_auc_bsi` / `best_val_auc_bsi` 与参照头 `pred_std`；来源与 pin 程序见 §4.3）；**未提供 = `None` ⇒ 处理臂在 M0/效用计算处响亮失败（TypeError），绝不静默沿用任何他 seed 参照**；基线臂不读取这些常量，行为不受影响 | 重建等式（3 行替换 + 断言旧文本在 pin 中恰出现 1 次）；行为测试：未设 env ⇒ 三常量为 `None` 且 `arm_verdict` 默认路径抛错；设 env ⇒ 逐位解析 |
| A4″ | 测试文件适配 | `aliccp_benchmark/tests/test_residual_prompt.py` 移植自 pin 测试 blob，**恰好 4 处**：模块 docstring、`DOC_PATH`、`WHITELIST`（9 项，逐项集合钉死）、`TestPreregConstants` 两方法（常量断言改为运行期解析语义；文档 token 表 → 本文件表） | 新增守卫测试以重建等式钉死（同 seed2 先例口径） |
| A5″ | 新增守卫/分析测试 | `aliccp_benchmark/tests/test_residual_prompt_five_seed.py`（A1″–A3″ 与接线逐字节守卫）；`aliccp_benchmark/five_seed.py` + `aliccp_benchmark/tests/test_five_seed.py`（§5/§6 的纯函数与 CLI，新分析层，不改判定 machinery） | 本文件即为守卫 |

**接线零改动**：`bench.py` 与 `run_aliccp_benchmark.py` 与 pin **逐字节相同**（守卫测试断言工作树 LF sha256 == 上表 pin 值，且 `git diff 8133d32` 对二文件的 diff 文本与 pin 提交的 diff 文本相同）。这样处理臂的调用链（`run_stage2` → `RP.build_newtask` → `PromptAudit` → `RP.arm_verdict`（默认参数取模块常量））与 seed1/seed2 逐字一致，per-seed 参照经由 A3″ 的运行期常量进入 `arm_verdict` 默认参数。

**被禁止的适配**：机制函数类/公式/初始化/门控语义/注入点/活性带/`BOUND_TOL`/G7 比值/U1 阈值/分类规则/`protocol.py`/`metrics.py`/`multitaskrec/*` —— 零改动（静态守卫：`git diff --name-only 8133d32` ⊆ §白名单）。

**等价性证明口径（三层，全部由测试守卫执行）**：
1. **重建等式（文本级）**：pin blob 文本 + 恰好替换（docstring、`import os`、3 常量行）== 工作树文件文本（逐字节）；
2. **AST 钉死（祖先级）**：`TestMechanismPin`（原样移植）断言工作树机制文件与 Census 祖本 blob：类名/方法集合相等、全部方法 `ast.dump` 逐字相等、模块级函数（`isolated_cpu_rng`/`effective_gate`/`build_newtask`/`evaluate_prompt_diagnostics`/`param_report`）逐字相等、`reference_head_stats` 归一化 `auc_score→auc` 后逐字相等（豁免集合恰为 {模块 docstring、import、模块级常量赋值、`arm_verdict`}）；
3. **接线字节级**：`bench.py`/`run_aliccp_benchmark.py` 工作树 LF sha256 == pin（上表）且 diff-文本相等。
   （1+2 合取 ⇒ 工作树机制 ≡ pin ≡ seed1 实现在一切类/函数上；3 ⇒ 接线 ≡ seed1/seed2 实现在一切行为路径上。）

---

## 4. 运行规程（每 seed 四步；全前台、等待至完成）

### 4.1 逐 seed 流程（seed S ∈ {`1688738016`, `1688749593`, `1688762746`}）

```powershell
# cwd = 本 worktree 根；解释器 = 主树 venv（本 worktree 无 .venv）
$PY = D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe

# (1) Stage-1（内容寻址产物；model_seed=S、env_seed=20261003、epochs=3、patience=2 全为协议默认）
& $PY run_aliccp_benchmark.py stage1 --tag short --model-seed <S>
#     → stage1_id = s1-5c060b9c-m<S>-e3-<cfg8>（前缀指纹沿用并校验：5c060b9c…）

# (2) 配对基线（同 seed、同 stage1_id；5 epochs/patience 2/batch 2000/lr 1e-4 全为协议默认）
& $PY run_aliccp_benchmark.py stage2 --tag short --model-seed <S> --stage1-id <sid_S>

# (3) 只读参照核验与 pin（不训练、不写 run/SUMMARY；见 §4.3）
& $PY verify_reference_residual_prompt_five_seed.py --seed <S>
#     → artifacts/aliccp_bench/audit/five-seed/reference_pin_seed<S>.json

# (4) 处理臂（env 三常量逐位取自 (3) 的 pin；其余全为协议默认；run_id 由 runner 追加 -rpg）
$env:RP_BASELINE_AUC_TEST = "<pin.baseline_auc_test>"
$env:RP_BASELINE_AUC_VAL  = "<pin.baseline_auc_val>"
$env:RP_REFERENCE_PRED_STD = "<pin.reference_pred_std>"
& $PY run_aliccp_benchmark.py stage2 --tag short --model-seed <S> --stage1-id <sid_S> `
    --variant residual-prompt `
    --prompt-reference-newtask artifacts/aliccp_bench/runs/<baseline_run_id>/newtask.pt
```

（步骤 (4) 的 `--prompt-reference-newtask` 路径为 `artifacts/aliccp_bench/runs/<baseline_run_id>/newtask.pt`，逐字以 pin 记录的 run_id 构造。）

- 步骤顺序**固定为 1→2→3→4**；任何 seed 的 arm 运行**不得先于**该 seed 的基线与其 pin。
- 三 seed 可（且计划）先集中跑 (1)(2)（6 个运行），再集中跑 (3)（无 GPU），再逐 seed 跑 (4)；该批次安排不影响任何配对关系，且保证 pin 在 arm 之前。
- 运行前：全量单元测试绿 + 静态/重建/AST 守卫 + `git_state.dirty=False`（未跟踪文件不计入）；运行后：`SUMMARY.md` 由 runner 只追加。

### 4.2 无效执行处置

仅当运行因**工具性原因**（崩溃/中断/环境故障、产物未完整落盘、**run-reference 与配对 pin 不一致**——如 env 变量误置）无效时，保留现场、记录原因后可**重跑一次**；因结果原因不构成无效。无任何事后调参救结果（hidden/初始化/α 初值/注入流集合/活性带/阈值/参照文件/常量均不得动）。

### 4.3 run-reference pin（新增机械规程；判定不依赖"预注册时已知值"的假设）

对每个 seed 的固定产物 + 配对基线 `newtask.pt` 做**只读**复算（同 `reference_head_stats` 口径：val 全量 500k、fp64），产出 `reference_pin_seed<S>.json`：

| 字段 | 来源（逐位） |
|---|---|
| `baseline_auc_test` | 配对基线 run `metrics.json:test_auc_bsi` |
| `baseline_auc_val` | 配对基线 run `metrics.json:best_val_auc_bsi` |
| `reference_pred_std` | 参照头 val `pred_std`（fp64、n=500000、`unbiased=False`；与运行期 `reference_head_stats` 同口径） |
| provenance | `model_seed`、`stage1_id`、`baseline_run_id`、`newtask.pt` sha256、pin 脚本 sha256、生成时间 |

- pin 一旦写出不得修改；`<pin.*>` 三值逐位进入 arm 运行的 env（A3″）。
- **闭合校验（独立复核执行）**：arm `metrics.json:rp_arm.U.*.observed.baseline_*` 与 `M0.observed/rule` 中的参照值 == pin 值 == 配对基线 `metrics.json` 值（三方逐位相等）；任一处不等 ⇒ 该 arm 无效（工具性），按 §4.2 处置。
- `REFERENCE_IDENTITY_TOL = 1e-9` 与 `PRED_STD_MIN_RATIO = 0.5` 不变；M0 判据形式与 seed1/seed2 逐字相同，仅期望值改为 per-seed。

---

## 5. 效用、历史判定与二级分类

### 5.1 历史判据（与 seed1/seed2 逐字相同；**不改写、不放宽**）

| 编号 | 判据 | 落盘 |
|---|---|---|
| U1 | `Δtest = test_bsi_arm − 配对基线 test_bsi ≥ +0.0055` | `metrics.json:rp_arm.U.U1` |
| U2 | `Δval = val_bsi_arm − 配对基线 val_bsi > 0` | `metrics.json:rp_arm.U.U2` |

臂级三标签（`VALID_POSITIVE` / `VALID_NEGATIVE` / `MECHANISM_FAIL`+subreason）与 §7 门禁逻辑**不变**；`arm.pass ⇔ classification == "VALID_POSITIVE"` 不变。

### 5.2 新增二级分类（用户指定；**新字段/新层，不覆盖旧字段**）

对每个 seed 的配对 `Δtest` 机械分类（边界口径写死如下）：

| 类别 | 判据（闭开区间，精确边界） |
|---|---|
| `POSITIVE_IMPROVEMENT` | `Δtest ≥ +0.001` |
| `NO_CLEAR_IMPROVEMENT` | `−0.02 < Δtest < +0.001`（严格双开） |
| `CLEAR_DEGRADATION` | `Δtest ≤ −0.02` |

- 二级分类与历史 U1/U2/三标签**并存**（前者为分析层字段，后者原样保留在 run 产物中）。
- 对 `NO_CLEAR_IMPROVEMENT` 的 seed，按下表做**机械 headroom 评估**（全部由 `five_seed.py` 从 run JSON 计算，规则先于结果写死）：

| 因子 | 机械口径 | 取值 |
|---|---|---|
| 机制活性 | `|α_final| > 0` ∧ 末 epoch `generator_grad_norm > 0` ∧ `geff_std > 0`（均取自 run JSON） | `ACTIVE` / `INACTIVE` |
| 验证方向 | `Δval > 0` | `POSITIVE` / `NON_POSITIVE` |
| 跨 seed 一致性 | 该 seed `Δtest` 符号为负而五 seed 全正（或反之）即 `INCONSISTENT`；与该 seed 同号计数 ≥ 4 记 `CONSISTENT` | `CONSISTENT` / `INCONSISTENT` |
| epoch 轨迹 | `best_epoch == epochs_run` ⇒ 右删失仍上升；否则早停 | `RIGHT_CENSORED_STILL_IMPROVING` / `EARLY_STOPPED` |
| 距正分类余量 | `gap_to_positive_threshold = +0.001 − Δtest`（>0，越小越接近） | 数值 |

---

## 6. 五 seed 汇总（预注册规则）

### 6.1 输入与配对纪律

- 输入 = 五条 `(配对基线 run, arm run)` 对（§2 表）；一切字段从 run JSON / `prompt_report.json` 读取并**独立重推**（§11）。
- 只对**逐 seed 配对差值** `{Δtest_i}`、`{Δval_i}` 做统计；机制诊断逐 seed 列出后做稳定性描述；**禁止**跨 seed 拼基线、禁止用均值替代任何逐 seed 判定。

### 6.2 逐 seed 记录字段

`model_seed`、`stage1_id`、基线/arm `run_id`、`test/val AUC`（两臂）、`Δtest/Δval`、U1/U2、历史分类 `classification`、二级分类 `secondary`、M0+G1–G8/A 类 pass 布尔、`α_final`、三路 `ratio_mean`/`ratio_std`/`cos_mean`、`geff mean/std/min/max`、`pred_std`/`ref_pred_std`、`best_epoch`/`epochs_run`/是否右删失、逐 epoch val 轨迹、参数计数（2385/8129/10514）。

### 6.3 汇总统计（全部报告值；机械计算）

| 量 | 口径 |
|---|---|
| 均值/标准差 | `mean_Δtest`、`std_Δtest`（**样本标准差**，ddof=1）、`mean_Δval`、`std_Δval` 同式；n=5 |
| 正例计数（两阈值） | `n(Δtest ≥ +0.001)`（二级阈值）与 `n(Δtest ≥ +0.0055)`（历史 U1 阈值）；另报 `n(Δval > 0)` |
| 最差 seed | `min Δtest` 及其 `model_seed`/`run_id` |
| Student-t 95% CI | `mean ± t_{0.975,4}·s/√5`，`t_{0.975,4} = 2.7764451051977987`（df=4；对 `Δtest` 与 `Δval` 各报一组，含下界/上界） |
| 验证方向 | 逐 seed `Δval` 列表 + `n(Δval > 0)` |
| 符号一致性 | `all(sgn(Δtest) 全同)` 与 `all(sgn(Δval) 全同)` 布尔 + 各自同号计数 |
| 机制稳定性 | `n(M0+G1–G8+A 全 PASS)`；`α_final` 符号分布；三路 `ratio_mean` 的 seed 间 min/max；三路 `cos_mean` 的 seed 间 min/max；`geff_std` min/max；`pred_std/ref_pred_std` 比值 min/max（全部描述性） |
| 二级分类计数 | `n(POSITIVE_IMPROVEMENT)` / `n(NO_CLEAR_IMPROVEMENT)` / `n(CLEAR_DEGRADATION)`；对每个 `NO_CLEAR_IMPROVEMENT` seed 附 §5.2 headroom 表 |

- 统计仅覆盖**同配置 run-to-run 噪声为零**（A3）前提下的 **seed 间变异**；不做假设检验推断超总体（n=5，CI 仅描述这五个 canonical seed 的均值）；结论措辞不得超出该范围。

### 6.4 20-epoch 分支立项条件（**先于结果写死**；六条全满足才记 `TWENTY_EPOCH_CONDITION_SATISFIED`，否则 `TWENTY_EPOCH_CONDITION_NOT_SATISFIED`）

| # | 条件 |
|---|---|
| C1 | 机制：五 seed 全部 M0+G1–G8 PASS 且 A 类 PASS（比较全部有效） |
| C2 | 无退化：无 seed 二级分类为 `CLEAR_DEGRADATION` |
| C3 | 中心：`mean_Δtest ≥ +0.0055`（历史 U1 阈值；不允许因汇总而放宽） |
| C4 | 不确定性：`Δtest` 的 Student-t 95% CI **下界 > 0** |
| C5 | 广度：`n(Δtest ≥ +0.001) ≥ 4` |
| C6 | 验证方向：`n(Δval > 0) ≥ 4` |

- 判定为逐条机械真值表；**本分支本身不启动 20-epoch 工作**（非目标，§9）；判定结果只回答"是否满足预注册立项条件"，供用户决定是否另开分支。

---

## 7. 机制门禁（M0 + G1–G8；与 seed1/seed2 逐字相同；全部硬判据）

| 编号 | 判据（不变） | 实测口径 |
|---|---|---|
| M0 参照身份 | 运行期 `reference_head_stats` 复算：`|ref_val_auc − <pin.baseline_auc_val>| ≤ 1e-9` 且 `|ref_pred_std − <pin.reference_pred_std>| ≤ 1e-9` | 处理臂专用；不满足 = 参照/产物/pin 不一致，本次不作效用解释 |
| G1 构造恒等 | 共享参数/buffer 与基线 `NewTask` 逐位相同；全局 RNG 端点逐位一致；新增键恰 5 键 | 运行期 `construction_identity` |
| G2 初值恒等 | `α@构造 == 0.0`；真实首 batch 前向与基线头 `torch.equal` | 运行期探针 |
| G3 门控/生成器梯度活 | 首 batch α 梯度 ≠ 0；末 epoch 首 batch α ≠ 0 ∧ 生成器梯度范数 > 0；`best_state` 载入后 `α_final ≠ 0` | 逐 epoch `grad_probe` |
| G4 范数受控界 | val 全部样本、三路：`ratio_max ≤ |α_final| + 1e-6` | `prompt_report.json:val_stats` |
| G5 活性带 | val 每路平均 `ratio ∈ [0.005, 0.5]`（含端点） | 同上 |
| G6 逐样本门控方差 | `geff_std > 0` ∧ `geff_max > 0` ∧ `geff_min ≥ 0` | 同上 |
| G7 无预测坍缩 | `pred_std > 0` ∧ `pred_std ≥ 0.5 × ref_pred_std`（相对参照） | `dispersion` + `reference` |
| G8 参数预算 | 键集恰 5 键；`new_params_total == 2385`；`head_params == 8129` | `prompt_report.json:params` |
| A 类 | A1/A2/A4/A5/A6 必须 PASS（A3 SKIP）；任一 FAIL ⇒ 该臂比较作废 | `gate_report.json`（不改） |

**B 类说明**：B1–B4 逐 seed 原样披露；B1/B4 的 FAIL 若出现，需核对该 seed 产物 meta 是否为**继承事实**（CTR/CVR 腿由 Stage-1 决定；B4 由该 seed 的 `cluster_events` 决定），不得归因于本臂。`hard_pass` 不参与臂有效性定义（沿用 seed1/seed2 口径）。

---

## 8. 继承事实与预期噪声

- 每个 seed 的 Stage-1 产物决定其 B1（CTR/CVR 腿）与 B4（cluster 占比）；三新 seed 的具体取值运行后记录（seed1 线 B1 FAIL/B4 FAIL；seed2 线 B1 PASS/B4 FAIL，均为产物决定的历史形态）。
- seed 间机制诊断量（α 符号/量级、ratio、cos）在 seed1（α −0.0605、cos 三路 +0.36/+0.48/+0.47）与 seed2（α +0.0710、cos 三路 ≈ +0.01~+0.08）间已有显著差异（seed2 文档已披露）；本实验**不**对 seed 间诊断差异做机制层解读，只做稳定性描述（§6.3）。
- 参照 `pred_std` 逐 seed 不同（seed1 0.003373 / seed2 0.005217），为不同 backbone/head 的自然差异；G7 用相对判据（不变）。

---

## 9. 非目标与止损

- **本分支不做**：20-epoch / 长预算运行、任何消融、A3 复跑（已有证据）、`full` tag、FLOPs、归一化聚类修复、TC-Prompt / CGR / affinity gate / KL-Prompt / 温度与损失改动、CensusIncome 相关工作、论文/图表/新颖性主张、聚类分析。
- 不因结果修改：hidden=16、初始化、α 初值、注入流集合、活性带、`BOUND_TOL`、G7 比值、U1 阈值、二级分类边界、五 seed 汇总规则、20-epoch 条件、pin 规程。
- **恰好一次**每 seed 一次 stage1、一次基线、一次 arm（工具性无效按 §4.2 留痕重跑一次）；`SUMMARY.md` 只追加。
- 不触碰 Stage-1 既有产物（seed1/2 产物与 run 只读）；不重训/不微调/不重新初始化任何 backbone。

---

## 10. 结果（运行后回填，不回写第 0–9 节）

**（占位：待三新 seed 运行与 pin 完成后回填逐 seed 表、机制诊断、二级分类、5-seed 汇总与 20-epoch 条件判定。）**

---

## 11. 独立复核（运行后回填）

**（占位：独立 verifier 逐 arm 复核 + 五 seed 汇总独立重推 + 与本文档记录值逐位对照。）**

---

## 12. 非新颖性声明

构件全部为既有方法族（prompt tuning、FiLM 族条件化调制、adapter/LoRA 残差、门控残差、范数归一），先行工作清单沿用 seed1 设计 §11（写作引用前必须逐条核实，本会话未做系统检索）。本文件**不**声称独立于先行工作；可报告内容仅限本仓库冻结协议下的五 seed 一次性判定。

---

## 13. 偏离披露（预声明 + 运行后补）

- **分支工作流**：自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；机制经 §3 三层守卫钉死等价；协议不合并 `master`。
- **分支本地 SUMMARY 系谱**：本分支 `SUMMARY.md` 基线含 smoke 行与 seed1 基线行（`8133d32` 携带）；seed1/seed2 的 arm 行与其 seed2 基线行记录在各自分支，本分支**不复制**其它分支行，只由本次运行追加自己的行（6 行：3 基线 + 3 臂）。
- **文档转述偏差披露**：§1.3-1（verify 计数 26/28 vs "27/27"）为对历史文档转述的更正性披露，不改历史结论。
- **非结果文件（未跟踪，同 seed1/seed2 先例）**：`artifacts/aliccp_bench/audit/five-seed/`（pin、汇总报告、审计报告）、worktree 根的核验/分析脚本（`verify_reference_residual_prompt_five_seed.py`、`verify_residual_prompt_five_seed.py`、一次性审计脚本）。
- **预注册前参照值不可预置**：seed3–5 的配对基线与参照值只能在各自基线 run 完成后取得（§4.3 机械规程），本预注册因此把"pin 手续与闭合校验"写死为流程判据，而非假设运行前已知数值。

---

**改动清单（白名单；静态守卫以 `8133d32` 为基准）**

| 文件 | 处置 |
|---|---|
| `docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-five-seed-design.md` | 本文件（C1 单独提交） |
| `aliccp_benchmark/residual_prompt.py` | 新增（§3.2 A1″–A3″ 适配后的移植机制） |
| `aliccp_benchmark/tests/test_residual_prompt.py` | 新增（§3.2 A4″ 适配后的移植测试） |
| `aliccp_benchmark/tests/test_residual_prompt_five_seed.py` | 新增（A1″–A3″ 与接线字节守卫） |
| `aliccp_benchmark/five_seed.py` | 新增（二级分类 / headroom / 五 seed 汇总 / 20-epoch 条件；纯函数 + CLI） |
| `aliccp_benchmark/tests/test_five_seed.py` | 新增（上者单测） |
| `run_aliccp_benchmark.py` | 逐字节移植（pin `2966ec39…`） |
| `aliccp_benchmark/bench.py` | 逐字节移植（pin `b063a366…`） |
| `artifacts/aliccp_bench/SUMMARY.md` | 只追加 6 行（runner 自动） |
| 其余全部（`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`metrics.py`、既有测试三件、`AliCCP_*.py`、`CensusIncome_*.py`、`baseline/*`、`mask/*`、`analysis/*`） | **零改动**（静态守卫断言 `git diff --name-only 8133d32` ⊆ 白名单） |
