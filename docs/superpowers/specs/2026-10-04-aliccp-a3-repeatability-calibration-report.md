# AliCCP A3 同配置复跑校准——结果（已完成）

- **状态**：已执行完毕，本文件为结果记录（判定按预注册 §6 机械计算；预注册见同目录 `2026-10-04-aliccp-a3-repeatability-calibration-design.md`）
- **日期**：2026-10-04
- **分支**：`infra/aliccp-a3-repeatability-calibration`（基点 `infra/aliccp-fair-benchmark` @ `8133d32`；未合并 `master`、未触碰其他 worktree、未推送）
- **提交链**：`10ea6fa`（C1 预注册）→ `acba208`（C2 实现 + 测试）→ `428c76f`（C3 R1 SUMMARY 行）→ `f7c1292`（C4 R2 SUMMARY 行）→ 本次 C5（本文件）
- **判定**：**A3_PASS**（短配置 / seed-2 / 固定产物；`max_pair |ΔAUC-Test-BSI| = 0.0 ≤ 1e-9` 且三 run `backbone_sha256` 一致）

---

## 1. 执行摘要

- 按预注册 §4 逐字命令，前台执行 **2 次**新 Stage-2 run（R1 `acba208`、R2 `428c76f`；run 间仅"SUMMARY 行"提交，空 diff 已证），与既有参照 run（REF `79b5e07`）构成 n = 3 比较；无工具性无效执行、无重跑。
- **A3 核心判据（预注册 §6.2）**：三个成对 `|ΔAUC-Test-BSI| = 0.0`（逐位相同，非仅 ≤ 1e-9）；三 run `backbone_sha256_loaded = e5e7e610…` 全同且各自 `loaded == before == after` → **A3_PASS**。
- **补充指标（预注册 §6.3）全部达到最强形式**：val/test/gate_mean 的最大成对差均为 `0.0`；三 run 的逐 epoch 轨迹（`train_loss`/`val_auc_bsi`）逐位相同；**三个 `newtask.pt` 字节相同**（sha `90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f`，含 REF）；`env_ids`/`fingerprint` sha 三同。
- 校准结论（口径按预注册 §6.4）：同配置 run-to-run 噪声 `F = 0.0`（逐位）；`+0.0055` 相对该噪声的保守倍数 ≥ `5.5e6`；**仅覆盖同配置 run-to-run 这一维度**（强制边界见 §7）。
- 分析器机械判定与**独立重算**（不经 `a3_repeatability` 模块、直接读 JSON/文件）逐位一致（§3.3）。

---

## 2. Run 记录（全部前台、`git.dirty=false`、无中断）

| 项 | REF（既有，不重跑） | R1（新） | R2（新） |
|---|---|---|---|
| `run_id` | `20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07` | `20261004-0622-p2M-v500k-t1M-m1688723740-short-acba208` | `20261004-0625-p2M-v500k-t1M-m1688723740-short-428c76f` |
| commit / dirty | `79b5e07` / `false` | `acba208` / `false` | `428c76f` / `false` |
| 运行日志 | `logs/20261003-062449-stage2-short.log` | `logs/20261004-062230-stage2-short.log` | `logs/20261004-062506-stage2-short.log` |
| `AUC-Val-BSI(best)` | `0.5809347091990792` | `0.5809347091990792` | `0.5809347091990792` |
| `AUC-Test-BSI` | `0.5974422649550507` | `0.5974422649550507` | `0.5974422649550507` |
| `gate_mean` | `[0.789178, 0.2108221875]` | `[0.789178, 0.2108221875]` | `[0.789178, 0.2108221875]` |
| `newtask.pt` sha256 | `90ee06da…b94f` | 同左（字节相同） | 同左（字节相同） |
| A 类（A1/A2/A4/A5/A6） | 全 PASS | 全 PASS | 全 PASS |
| B 类 | B1/B2/B3 PASS；**B4 FAIL（继承）** | 同左 | 同左 |
| `hard_pass` | `false`（仅 B4 继承失败） | `false`（同型） | `false`（同型） |
| `wall_seconds` | 142.9 | 139.9 | 141.1 |

- 逐 epoch 轨迹（三 run 逐位相同，引用共享值）：`train_loss = [0.08193688414408826, 0.05193358083860949, 0.04973645433795173, 0.04873575337347574, 0.04805051837593783]`；`val_auc_bsi = [0.4645711559431739, 0.48564481224085004, 0.5142344439088465, 0.5508809596059387, 0.5809347091990792]`。
- B4 FAIL 由共享 Stage-1 产物（cluster 退化，`env_0 = 1532`）决定，三 run 逐位相同 stage-1；属继承缺陷、非本实验对象、不归因于任何新代码（预注册 §3.4 已核验 REF 同型）。
- SUMMARY：R1/R2 行已按既有 `append_summary_row` 只追加（行内 A3 列仍为 `SKIP`——行不重写；正式判定 = 本文件 + 分析器报告）。

---

## 3. 判定过程（预注册 §6，机械计算）

### 3.1 身份前提（§6.1）——通过

- identity 键子集（13 项）与 `metrics.epochs=5 / patience=2` 三 run 全同；记录/溯源字段差异见 §5-D2。
- 三 run A1/A2/A4/A5/A6 全 PASS；`env_ids_sha256`、`fingerprint_sha256` 三同。

### 3.2 A3 核心判据（§6.2）——PASS

| 成对 | `|ΔAUC-Test-BSI|` | ≤ 1e-9 |
|---|---|---|
| REF–R1 | `0.0` | ✓ |
| REF–R2 | `0.0` | ✓ |
| R1–R2 | `0.0` | ✓ |

`backbone_sha256`：三 run 均 `e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c`（loaded==before==after）→ **A3_PASS**。

### 3.3 分析器与独立重算的一致性

- 分析器（`python -m aliccp_benchmark.a3_repeatability --root artifacts/aliccp_bench --runs <REF> <R1> <R2> --out artifacts/aliccp_bench/audit/a3-short-seed2/a3_report.json`，退出码 0）→ `status = A3_PASS`。
- 独立重算（单独脚本直接读取三个 run 目录的 JSON 与文件字节，不经分析器模块）：`max_pair|Δtest| = 0.0`；backbone/env_ids/fingerprint/newtask sha 集合大小均为 1；轨迹原始字典比较全同；identity 键全同；A 类全 PASS。**与分析器报告逐位一致。**
- 测试：`python -m unittest discover -s aliccp_benchmark/tests` → **56/56 OK**（含本实验新增 18 个 A3 判定器测试）。

---

## 4. 补充指标（预注册 §6.3；全部为报告值，不入判定）

| 指标 | 值 |
|---|---|
| `max_pair |ΔAUC-Val-BSI|` | `0.0` |
| `max_pair |ΔAUC-Test-BSI|` | `0.0` |
| `max` 逐元素 `|Δgate_mean|` | `0.0` |
| 轨迹位等（train_loss + val_auc_bsi） | `True` |
| checkpoint 位等（`newtask.pt` sha 三同） | `True` |
| `gate_mean` 精确相等 | `True` |
| `env_ids_sha256` / `fingerprint_sha256` | 三同 |
| `wall_seconds` | 142.9 / 139.9 / 141.1（跨 run 差 ≤ 3.0s，≈2%） |

---

## 5. 偏离与纪律遵守（逐条对照预注册 §5/§9）

| 项 | 实际 | 说明 |
|---|---|---|
| D1 代码溯源 | REF @ `79b5e07`（衰减分支代码）vs R1/R2 @ `acba208`/`428c76f`（基点管线代码，管线文件与 `8133d32` **零 diff**，已证） | 预注册 §10 的预期（逐位相同）实测成立；且**跨代码**字节相同（含 `newtask.pt`）实测成立。惰性论证（源码 skip 分支 + 长配置对实证）与本复跑实测一致 |
| D2 记录模式 | REF `config.json` 含 `spec_attenuation: 1.0`（R1/R2 无该键；R1/R2 亦无 `variant`） | 记录字段差异，identity 键子集不含它们（§6.1）；不影响判定 |
| D3 run 间 commit/run_id | R1 `acba208` ≠ R2 `428c76f`（差一个"仅 SUMMARY 行"提交；`git diff acba208 428c76f -- . ':(exclude)artifacts/aliccp_bench/SUMMARY.md'` 为空） | 预注册允许 |
| 运行次数 | stage2 ×2（R1/R2）；无工具性中断；无重跑；未重训 Stage-1；未重跑 REF 与任何既有 run | 预注册 §5 纪律 |
| 无效执行 | 无 | — |
| SUMMARY 行 | 只追加 2 行；未重写任何既有行 | 预注册 §5.3 |
| 阈值/常量 | `+0.0055` 与一切协议常量零改动 | 预注册 N6 |

---

## 6. 既有长配置重跑对（补充证据；按预注册 §7 记录，未重跑）

`20261003-0724-…-long-bbd8a61` vs `20261004-0427-…-long-2b1d585`（同产物、同 seed、`epochs=10/patience=3`）：

- `ΔAUC-Test-BSI = 0.0`（均 `0.6614121223087556`）；`AUC-Val-BSI(best)` 均 `0.6401127811737995`；轨迹、`gate_mean`（`[0.84101275, 0.1589871875]`）逐位同；`newtask.pt` 字节相同（sha `ea988a3633c6e7130e5b9dadfd6c543a9a8a5398ca0460f7f17b083b98a82a98`）。
- 记录字段差异恰为 `run_id`/`commit`/`git`/`wall_seconds`/`spec_attenuation`（`1.0` vs 无键）。
- **定位**：数值上满足 A3 判据，但执行时未以 A3 预注册（其文档自记 A3 SKIP、确定性观察为"描述性、不作噪声估计"）→ 记录为补充证据；本实验**未重跑**该配置（避免重复训练）。

---

## 7. 校准结论（口径严格按预注册 §6.4）

- 实测：同配置 run-to-run 噪声 `F := max_pair |ΔAUC-Test-BSI| = 0.0`（逐位；val 同）。
- **在"同配置 run-to-run"这一维度上**：`+0.0055` 相对该噪声的保守倍数 ≥ `5.5e6`；即在该维度上，`+0.0055` 是高度保守的。
- **强制边界（预注册 §6.4，缺一即违规，随本结论一体引用）**：
  1. 本校准只覆盖**同配置 run-to-run** 噪声；**不覆盖** seed 间方差、预算效应、机器/版本差异。
  2. 筛查家族内已观察的 ±1e-4…1e-3 级摆幅属**跨 seed/跨配置**量，**不由**本实验解释，其噪声学地位不因本实验改变（该家族文档已自记"无同 seed run-to-run 噪声估计、A3 SKIP"）。
  3. `+0.0055` 的 provisional 状态照旧（spec §10.1）；本实验**不修改**其数值，也不单独将其"定稿"。

---

## 8. 对既有 AliCCP 结论的含义（不重判任何已判定 run）

1. **固定 seed 的单 run Δ 结论在本环境/版本/机器上不受同配置 run 噪声污染**：任何按协议复现该配置的第三次运行都会得到逐位相同的数值；因此筛查家族中"差一点"的判定（如残差 prompt seed-1 `U1 FAIL by 1.05e-4`）**不是** run 噪声可解释的——在该协议下它是真实的（该配置单次运行的）缺口，重跑同 seed 不会翻转判定。同理，衰减线 seed-2 `+0.0077`、残差 prompt seed-2 `+0.0081` 等正差值也不是同配置噪声。
2. **阈值校准的适用边界**：`+0.0055` 面对的主要不确定性来源（seed 间方差、预算敏感性）**不在**本次证据覆盖内；合成文档已记录的 seed 级敏感现象（衰减线量级 0.61×、残差 prompt α 符号翻转等）**不因本实验改变**，其相关 caveat 继续有效。
3. **长配置（10-epoch）**：既有重跑对（§6）给出同型结论（逐位相同）；因此 `NOT_PERSIST`（衰减线 10-epoch `Δ = +0.0005273 < +0.0055`）等 10-epoch 判定同样不可用"run 噪声"解释。
4. **A3 状态更新（本分支范围内）**：短配置（seed-2）A3 = **PASS**（n = 3，含跨代码对照）；长配置 = 数值判据满足（补充证据，未加 A3 预注册标签）；其余 seed/配置 A3 状态**不变**（仍无正式复跑证据，逐位观察仍为描述性）。
5. **无性能/方法主张**：本实验不改变任何 arm 分类、不新增机制结论、不构成论文级证据（预注册 §11 局限照旧）。

---

## 9. 复算入口

```powershell
# 判定（只读；退出码 0=PASS / 1=FAIL / 2=INVALID）
<repo-python> -m aliccp_benchmark.a3_repeatability --root artifacts\aliccp_bench `
  --runs 20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07 `
         20261004-0622-p2M-v500k-t1M-m1688723740-short-acba208 `
         20261004-0625-p2M-v500k-t1M-m1688723740-short-428c76f `
  --out artifacts\aliccp_bench\audit\a3-short-seed2\a3_report.json
# 测试
<repo-python> -m unittest discover -s aliccp_benchmark\tests
```

报告 JSON：`artifacts/aliccp_bench/audit/a3-short-seed2/a3_report.json`（本机磁盘；`*.json` 按 §11.2 不入库，数值已全文引用）。

## 10. 局限（同预注册 §11）

单配置（短、seed-2、单一产物）、n = 3、同机同 GPU、单一环境版本（python 3.10.11 / torch 2.6.0+cu124 / cuda 12.4）；不外推；不构成性能结论；不足单独决定 `+0.0055` 最终取值。
