# AliCCP A3 同配置复跑校准（基础设施；预注册）

- **状态**：预注册——本文件先于本实验的任何新 run 提交。提交后判据冻结；任何澄清以单独 commit 追加且判据数值零改动（沿用 `296e03a` 先例）。
- **日期**：2026-10-04
- **分支**：`infra/aliccp-a3-repeatability-calibration`（基点 `infra/aliccp-fair-benchmark` @ `8133d32cfa722b084c558b334f4792d85ce95c67`；不合并 `master`、不触碰其他 worktree）
- **上位协议**：`docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（唯一事实来源）。本文件只**实例化执行**其 §10-A3（按需复跑）与 §10.1（provisional 阈值），**不修改**其任何常量、阈值、预算、指纹或产物。
- **出处**：`docs/mptrec-screening-synthesis` @ `cb5c735` §9-D2——对 AliCCP 管线正式执行协议定义的 A3 复跑（该合成明确记载：所有分支 A3 均 SKIP；位等观察为描述性，从未被计为噪声估计）。
- **本实验不含任何模型改进、无超参搜索、不新增方法主张、不改变 `+0.0055`。**

---

## 1. 目标与非目标

### 1.1 目标

| 编号 | 目标 |
|---|---|
| G1 | **正式执行 A3**：按 spec §10-A3 的定义，对 seed-2 短（5-epoch）Stage-2 原始基线配置做同配置复跑，机械判定 PASS/FAIL。 |
| G2 | **量化同配置 run-to-run 噪声**：val/test 最大成对差（绝对）、逐 epoch 轨迹位等性、checkpoint 字节相等性、gate 均值、哈希与耗时。 |
| G3 | **校准 `+0.0055`**：以 G2 的噪声为证据，评估 provisional 阈值在**同配置 run-to-run 维度**上的保守性（仅此维度；见 §6.4 的强制口径）。 |

### 1.2 非目标

| 编号 | 非目标 |
|---|---|
| N1 | **不重训 Stage-1**（论证见 §2.2）。 |
| N2 | **不改任何协议常量/阈值/预算/指纹/模型代码**：`bench.py`/`protocol.py`/`metrics.py`/`run_aliccp_benchmark.py`/`multitaskrec/*`/`config.py` 零改动（以 `git diff` 空为证，§8）。 |
| N3 | 不跑新 arm、不调超参、不搜参。 |
| N4 | 不重跑任何既有有效 run；不重判任何已判定 run 的结论。 |
| N5 | 不产出性能结论；不把本校准当作 +0.0055 的全面验证（§11 局限）。 |
| N6 | **不修改 `+0.0055`**（spec §10.2 阈值纪律；本实验只补充其校准证据）。 |

---

## 2. A3 定义与实例化

### 2.1 定义原文（spec §10 A 类，逐字）

> **A3**：**可复现性（按需触发，不阻塞首轮）**：同 `(model seed, env seed, 预算, 配置)` 复跑，两次 `AUC-Test-BSI` 差异 ≤ `1e-9` 且 `backbone_sha256` 一致；首轮只跑一次，不因未复跑而失败。

### 2.2 实例化（本实验的操作化）

- **比较单元** = "固定 `stage1_id` 上的 Stage-2 run"——与 spec §12.1 定义的臂比较单元一致（全部筛查比较都发生在同一冻结产物上的 Stage-2 run 之间）。
- 三要素映射：
  - `model seed` = `1688723740`（筛选用例中的 seed-2，全脚本钉死）；
  - `env seed` = `20261003`（经产物 `env_ids` 固化；三 run 的 `env_ids_sha256` 必须逐位一致，§6.1）；
  - `预算` = `p2M-v500k-t1M`（2,000,000 / 500,000 / 1,000,000，协议默认）；
  - `配置` = §4 的 identity 键子集（配置身份），排除记录/溯源字段（§6.1、§9-D2）。
- **判据的 `backbone_sha256` 项**：对固定产物上的 Stage-2 复跑，此项 = "两 run 加载/使用/冻结后的 backbone 哈希逐位一致"（A6 语义的运行间复验），即防产物漂移；三 run 的 `backbone_sha256_loaded/before/after` 必须全部相等（§6.2）。
- **不重训 Stage-1 的论证（预注册）**：
  1. A3 文本不含"重训 Stage-1"要求；其"复跑"作用于"同配置"的**被检验单元**——本协议的被检验单元是"在内容寻址固定产物上的 Stage-2 run"（§7.2 阶段 2 只允许 `--stage1-dir` 引用；§12.1 臂比较单元同此）。
  2. Stage-1 层的身份已由内容寻址（`stage1_id` = 指纹+seed+epochs+配置哈希）与归档 repro 审计覆盖（§3.1b：3-epoch 含 cluster 的 Stage-1 完整重训逐位复现）；重训 Stage-1 会产生**新** `stage1_id`，属 spec §12.2 的"动阶段 1 的 arm"程序（另需配对对照），不是 A3。
  3. 筛查家族的全部结论（Δ 比较）都发生在 Stage-2 run 之间；本校准要回答的 run-to-run 噪声，恰是 Stage-2 层的噪声。

---

## 3. 预注册前审计（只读；全部来自既有磁盘产物与已提交记录）

### 3.1 既有位等/确定性证据（**均未被既有文档记为 A3**；对应分支与文档均记 A3 SKIP）

| 证据 | 内容 | 来源 |
|---|---|---|
| (a) 环境探针 | 两个独立进程：同 `model_seed` → 同 init（`23485bb8…`）→ 同真实 batch（`features_hash 100b97d7…`）→ 3 个训练步 → loss / embed-grad-norm / 最终参数哈希（`573a1749…`）**全部逐位相同**；两日志逐字节相同 | `artifacts/aliccp_bench/audit/env_determinism_probe.py` + `probe_runA/B.log`（本机磁盘） |
| (b) Stage-1 重训 repro 审计 | `s1-…-m1688723512-e3-3a30e2c0` 完整重训（3 epoch、含 `cluster_2`）：per-epoch 值逐位同、best backbone sha 同（`5553640b…`）、test CTR/CVR 逐位同、`env_ids_sha` 同、cluster 事件同——10/10 检查 PASS，wall 232.1s | `artifacts/aliccp_bench/audit/s1-…-3a30e2c0-repro/audit.json` |
| (c) 轨迹前缀位等 | 5-epoch 基线 `79b5e07` 的 5 个 epoch 的 val AUC 与其后独立进程跑的 10-epoch 基线 `bbd8a61` 的**前 5 epoch 逐位相同**（`[0.4645711559, 0.4856448122, 0.5142344439, 0.5508809596, 0.5809347092]`） | 两 run 的 `metrics.json`（本机磁盘） |
| (d) 长配置独立重跑对 | `bbd8a61`（`20261003-0724-…`，衰减线代码）vs `2b1d585`（`20261004-0427-…`，残差 prompt 线代码，`multitaskrec/` 与基点逐字节相同）：`newtask.pt` **字节相同**（sha `ea988a3633c6e7130e5b9dadfd6c543a9a8a5398ca0460f7f17b083b98a82a98`）、全部数值指标逐位相同（差异仅 `run_id`/`commit`/`git`/`wall_seconds`/`spec_attenuation` 记录字段） | 两 run 目录；其文档记"逐位等于"且明确 **A3 SKIP、不作噪声估计**：`e46e5d2:…2026-10-04-aliccp-stage2-residual-prompt-longer-budget-design.md` §10.2/§11 |
| (e) 推理层复评 | `79b5e07` 保存头重载后重评 val/test/gate_mean **逐位同**；35/35 独立核验通过 | `artifacts/aliccp_bench/audit/verify_reference_seed2_result.json` |

### 3.2 既有重跑是否已满足 A3（决定"是否需要新训练"的审计）

| 配置 | 既有同配置重跑？ | 处置 |
|---|---|---|
| **短配置（5-epoch，本实验对象）** | **否**——(c) 只是长 run 的轨迹前缀（非独立整 run）；(e) 是推理层复评（非训练复跑）。无任何先前完整同配置重跑。 | **必须新跑 2 次**（非重复训练） |
| 长配置（10-epoch） | 是——(d) 的独立重跑对**数值上满足 A3 判据**（Δtest = 0.0 ≤ 1e-9；backbone sha 一致）。但其执行从未以 A3 名义预注册，且跨 commit / 记录字段有差异。 | **不重跑**（避免重复训练）；作为补充证据完整记录（§7） |
| Stage-1 | repro 审计 (b) 已覆盖（另一协议程序） | 不重训（§2.2） |

### 3.3 确定性设置审计（CUDA/框架）

- runner **未启用** `cudnn.deterministic` / `torch.use_deterministic_algorithms`——保持仓库默认设置（`git grep` 于基点零命中；探针 (a) 证明该默认在本环境已逐位可复现）。
- 随机性来源暴露面：`protocol.seed_model(model_seed)`（torch/cuda/np）唯一播种点；数据顺序 `shuffle=False, batch_size=2000, num_workers=0, drop_last=False`（spec §6.2 硬约束）；`env_ids` 来自产物、不消耗 RNG；NewTask 头无 dropout（spec §8.1）；Stage-2 训练循环无其他随机算子。即：**设计上完全确定**，本实验检验其实现层面是否逐位成立。

### 3.4 参照 run（REF）有效性核验（预注册前）

- `20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07`：`gate_report.json` 中 A1/A2/A4/A5/A6 全 **PASS**（A3 记 SKIP）；B1/B2/B3 PASS、B4 FAIL（继承 stage-1 cluster 退化，非本实验对象）；`git.dirty=false`；35/35 独立核验（§3.1e）通过；SUMMARY 行与磁盘值逐位一致。→ **REF 有效，可入比较。**

---

## 4. 被复跑配置（逐字钉死）

### 4.1 运行命令（R1/R2 必须逐字执行）

```
python run_aliccp_benchmark.py stage2 --tag short \
  --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f \
  --model-seed 1688723740 --epochs 5 --patience 2
```

（其余为协议默认：budgets 2M/500k/1M、batch 2000、lr 1e-4、reg_dnn 7e-6、rep_dim 64、experts [128,64]、towers [32,32]、input 80、emb 5、`enforce_b=True`、`cuda:0`、root `artifacts/aliccp_bench`。）

### 4.2 固定产物（唯一 backbone 来源；钉死）

| 项 | 值 |
|---|---|
| `stage1_id` | `s1-5c060b9c-m1688723740-e3-4e1b5c6f` |
| `backbone_sha256` | `e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c` |
| `env_ids_sha256` | `5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0` |
| `fingerprint_sha256` | `5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8` |
| `config_hash` | `4e1b5c6ffe9b6ff49cda34691de27390669396337e64e867f0f83b6ec4da7981` |
| 文件 sha256 | `backbone.pt = cd7b033423499e0d36eea988835cea4ac4e2a8e26427a2aea627333822101e0b`；`env_ids.pt = 4660be5aaa3c59f53dd5b4394f77114a87db69064b32c45517db049a6b9157e7`；`meta.json = 61a66d81ce3dcf6bae64f4c2b37cf12e722943def646336a588746aba932931d` |

（上述哈希与既有完整性核验 28/28 一致；run 内 A2/A5/A6 会再次逐位复验。）

### 4.3 参照 run（REF，既有；不重跑）

| 项 | 值 |
|---|---|
| `run_id` | `20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07` |
| commit / dirty | `79b5e07` / `false` |
| `AUC-Val-BSI(best)` | `0.5809347091990792` |
| `AUC-Test-BSI` | `0.5974422649550507` |
| `gate_mean` | `[0.789178, 0.2108221875]` |
| `newtask.pt` sha256 | `90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f` |

---

## 5. 运行计划与纪律

1. **恰好 2 次新 Stage-2 run**（R1、R2），前台逐次执行、逐次等待；除 run tag/id 与"仅含 SUMMARY 行的提交"外，R1 与 R2 之间**零代码/配置变更**（§9-D3；以 `git diff <R1 时点的 commit> <R2 时点的 commit> -- . ':(exclude)artifacts/aliccp_bench/SUMMARY.md'` 为空为证）。
2. **清洁树纪律**（既有先例：`47ecb0c`/`bbd8a61`）：R1 完成后提交 SUMMARY 新增行（提交只含该行），使 R2 在清洁树上运行；两 run 记录 `git.dirty=false`。
3. **SUMMARY**：由 runner 既有 `append_summary_row` 自动追加，**只追加不重写**；行内 A3 列照旧记 `SKIP`（A3 是跨 run 判据，行内无法自判）；正式判定在分析器报告与结果文档（§6）。
4. **无效执行处置（预注册）**：仅当 run 因**工具性原因**（进程崩溃/中断/环境故障、未完整落盘）无效时，保留全部现场（失败日志与半成品目录不删除），记录原因后可重跑一次；因**结果原因**（判据/门禁不达预期）不构成无效、不重跑、照实记录。
5. 不重跑任何既有有效 run；不重训 Stage-1；不触碰其他 worktree。

---

## 6. 预注册判据（只看结果之前钉死）

### 6.1 身份前提（三 run：REF、R1、R2；不满足 → 本次比较 INVALID，照实记录）

- `config.json` 的 identity 键子集全同：`stage1_id`、`tag`、`model_seed`、`budgets`、`batch_size`、`lr`、`reg_dnn`、`newtask_rep_dim`、`expert_hidden`、`tower_hidden`、`input_size`、`embedding_size`、`enforce_b`；
- `metrics.json`：`epochs=5`、`patience=2` 全同；
- 三 run 的 `gate_report.json`：A1/A2/A4/A5/A6 全 PASS；
- 三 run 的 `env_ids_sha256`、`fingerprint_sha256` 全同。
- 记录/溯源字段（`run_id`、`commit`、`git`、`wall_seconds`、`spec_attenuation`、`variant`、`peak_vram_mb`）**不参与**身份判定（§9-D2）。

### 6.2 A3 核心判据（spec §10-A3 逐字机械计算）

- **Δ 项**：所有成对（REF-R1、REF-R2、R1-R2）`|ΔAUC-Test-BSI| ≤ 1e-9`（边界含等于）；
- **哈希项**：三 run 的 `backbone_sha256_loaded` 全同**且**各自 `loaded == before == after`；
- **A3_PASS ⟺ Δ 项 ∧ 哈希项**；否则 **A3_FAIL**（如实披露，不重跑）。

### 6.3 补充报告指标（预注册，报告用，**不参与** A3 判定）

- val/test 的最大成对绝对差；
- 轨迹位等：三 run 的逐 epoch `(train_loss, val_auc_bsi)` 序列精确相等（逐对布尔）；
- `newtask.pt` 的 sha256 三同；
- `gate_mean` 逐元素精确相等 + 最大逐元素差；
- 三 run `wall_seconds`；哈希表（backbone/env_ids/fingerprint/newtask）。

### 6.4 阈值校准口径（先于结果钉死；`+0.0055` 零改动）

- 记 `F := max_pair |ΔAUC-Test-BSI|`。
- 若 A3_PASS（`F ≤ 1e-9`）：报告"**同配置 run-to-run 维度**上，噪声 ≤ `1e-9`（观测 `F`）；`+0.0055` 相对该噪声的保守倍数 ≥ `5.5e6`（`F=0` 时表述为 ≥ 5.5e6）"。
- **任何相关表述必须同时写入以下边界**（缺一即违规）：
  1. 本校准只覆盖**同配置 run-to-run** 噪声；**不覆盖** seed 间方差、预算效应、机器/版本差异；
  2. 筛查家族内已观察的 ±1e-4…1e-3 级摆幅属**跨 seed/跨配置**量，**不由**本实验解释，其噪声学地位不因本实验改变；
  3. `+0.0055` 的 provisional 状态照旧（spec §10.1），本实验**不修改**其数值，也不单独将其"定稿"。
- 若 A3_FAIL：如实记录差异位置与幅度；`F` 即为该配置的实测噪声下界（≥），据此重述上述口径，不重跑。

---

## 7. 既有长配置重跑对（补充证据；不重跑）

- 对（d）：`20261003-0724-p2M-v500k-t1M-m1688723740-long-bbd8a61` vs `20261004-0427-p2M-v500k-t1M-m1688723740-long-2b1d585`，同产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f`、同 seed、`epochs=10`/`patience=3`。
- 审计结论（只读复算）：`ΔAUC-Test-BSI = 0.0`（两 run 均 `0.6614121223087556`）；`AUC-Val-BSI(best)` 均 `0.6401127811737995`；逐 epoch 轨迹逐位相同；`gate_mean` 均 `[0.84101275, 0.1589871875]`；`newtask.pt` 字节相同（sha `ea988a36…`）；`backbone_sha256` 均 `e5e7e610…`。记录字段差异恰为 `run_id`/`commit`/`git`/`wall_seconds`/`spec_attenuation`（`1.0` vs 无键）。
- **定位**：数值上满足 A3 判据；但执行时未以 A3 预注册，属**补充证据**，不替代本实验对短配置的正式执行；**不重跑**（既有效重跑不重复训练）。

---

## 8. 实现面（最小；TDD）

| 项 | 内容 |
|---|---|
| 新增 | `aliccp_benchmark/a3_repeatability.py`：只读判定器——`load_run` / `check_config_identity` / `a3_verdict` / `analyze` 纯函数 + `python -m` CLI（输出 JSON 报告）；容差常量 `A3_TOL = 1e-9`（spec §10-A3） |
| 新增 | `aliccp_benchmark/tests/test_a3_repeatability.py`：`unittest`、CPU-only、合成 run 目录；覆盖：位等 PASS、超容差 FAIL、**容差边界（恰 1e-9）PASS**、backbone 不一致 FAIL、身份不符 INVALID、A 类门禁失败 INVALID、轨迹不等但 Δ 达标仍 PASS（补充指标不入判定）、sha 计算、CLI 退出码（0/1/2） |
| **零改动** | `aliccp_benchmark/bench.py`、`aliccp_benchmark/protocol.py`、`aliccp_benchmark/metrics.py`、`run_aliccp_benchmark.py`、`multitaskrec/*`、`config.py`——以 `git diff 8133d32 <run 时点 commit> -- <上述路径>` 为空为证 |
| SUMMARY | 由既有 `append_summary_row` 追加，无新代码 |

---

## 9. 偏离披露（预注册）

| 编号 | 偏离 | 处置与论证 |
|---|---|---|
| D1 | **代码溯源差异**：REF 跑在 `79b5e07`（衰减分支代码：`multitaskrec/model.py` +14/−1 的 `spec_attenuation` 钩子、`bench.py`/`metrics.py` 纯增量钩子）；R1/R2 跑在本分支 tip（基点管线代码 + 本实验新增的 `a3_repeatability.py`/测试/文档，管线文件零改动） | 惰性论证：① 源码——`spec_attenuation==1.0` 时乘式整句跳过（`if self.spec_attenuation != 1.0:`），无参数/RNG/state_dict 变化；bench 钩子全部由 `spec_attenuation != 1.0` 门控或纯记录。② 实证——(d) 的长配置对恰跨这两类代码（含 `multitaskrec/` 逐字节相同 vs 带钩子版本），产出字节相同 checkpoint。③ 判定同时依赖 R1-vs-R2（同代码）对。若 REF-vs-R*/的差异与 R1-vs-R2 差异不同型，如实归因该溯源差异 |
| D2 | **记录模式差异**：REF `config.json` 含 `spec_attenuation: 1.0`（R1/R2 无该键）；`variant` 键同理（长配置对内部差异，与本实验无关） | identity 键子集（§6.1）排除记录字段；两模式在运行路径上的等价性由 D1 论证覆盖 |
| D3 | **R1/R2 间 commit/run_id 差异**：来自 §5 的"仅 SUMMARY 行"提交 | 预注册允许；代码同一性以 §5.1 的空 diff 为证 |
| D4 | **不重训 Stage-1** | §2.2 论证 |
| D5 | **分析器为本实验新增代码** | 容差/口径在 §6 钉死；TDD 测试覆盖边界；分析器只读，不触碰 run 产物 |

---

## 10. 预期（描述性；不入判定；落空不重跑）

依 §3.1 的既有证据，预期 R1、R2 与 REF 逐位相同（轨迹、`newtask.pt`、`gate_mean`、val/test）→ 预期 A3_PASS、`F = 0`。若出现任何不一致：如实记录差异位置与幅度，A3 按 §6.2 机械判定；不因预期落空而重跑（结果性）。

---

## 11. 局限（预注册）

- 单配置（短、seed-2、单一产物）、n = 3（1 既有 + 2 新）、同机同 GPU、单一环境版本（python 3.10.11 / torch 2.6.0+cu124 / cuda 12.4）。
- 不外推到其他 seed、预算、配置、机器或框架版本；不构成性能结论、方法主张或对筛查家族结论的重判。
- 不足以单独决定 `+0.0055` 的最终取值：只补充"同配置 run-to-run"一个噪声维度的证据。
