# exp/stage2-attn-env-prior：Stage-2 新环境向量改用 attention 加权（实验说明）

- **状态**：已实现（代码 + 测试），**未运行训练**；接受阈值已在开跑前写死（见第 3 节）
- **日期**：2026-09-29
- **分支**：`exp/stage2-attn-env-prior`（基线：`infra/fair-stage2-benchmark` @ `87afe03`）
- **协议**：只**引用** `docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`；
  `census_benchmark/protocol.py` **零改动**（想法分支不得修改协议文件，spec 3.3.2）
- **数据集**：只跑 CensusIncome（spec 2.3）

---

## 1. 唯一改动

Stage-2 `NewTask` 的新环境向量来源，由"随机初始化、可学习的 `nn.Embedding(1, rep_dim)`"改为
**每样本用既有 attention 权重 W 对 Stage-1 env_embs 加权**：

```
W(x)          = softmax_k( H(x) · E / temperature )        # 既有 attention 权重，未改动
new_env_emb(x) = Σ_k W_k(x) · E_k                          # 新增：attn 模式
env_aware_rep = new_spec_rep * new_env_emb(x)              # 与 master 同一处使用点
```

- `H(x)` = `projection_network(dnn_input)`，`E_k` = Stage-1 `get_infos()` 给出的 `env_embs`，
  `temperature = 150` 沿用 master 既有值（**不新增、不调温度**）。
- W 与 `new_spec_rep` 的 spec 加权用的是**同一份** softmax 权重（不额外算一套注意力）。
- 由外部参数 `env_prior` 控制，**默认 `"learned"`**（= master 行为，逐位一致；见第 5 节测试）。

### 1.1 明确不含（强约束）

Null Expert、温度改动/可学习温度、LoRA、任何 loss 项改动、门控/专家/投影结构改动、
Stage-1（`EmbeddingNetwork` / `MPTRec` / `train.py`）改动、AliCCP / ByteRec 入口改动。
`env_prior="attn"` 之外的一切模型语义与 master 完全一致。

### 1.2 一个实现取舍（写清楚，便于复核）

`env_prior="attn"` 时仍**保留** `env_embedding_network` 模块，但 forward 不再引用它
（其 `.grad` 恒为 `None`，不参与优化）。理由：两种模式在**同一 model seed 下全部参数与 buffer
逐位一致**（含共享的 projection / gate / tower 初始化）→ A/B 对照只差"env 向量来源"这一个变量；
若在 attn 模式跳过该 Embedding 的构造，随机流会错位，共享子模块初始化将与对照臂不同，
小效应（阈值差 ~0.002）下无法归因。

---

## 2. 对照（单 seed 单次，spec 6.3 阶梯第 1 步）

| 项 | 两条臂共同设置 |
|---|---|
| Stage-1 产物 | `artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3`（唯一 backbone 来源，只读） |
| split seed / model seed / env seed | `20260929` / `1685480945` / `20260929`（协议默认值） |
| 阶段 2 | `epochs=5`、`patience=2`、`batch_size=256`、`lr=1e-3`、冻结语义与门禁同协议 |
| 差异 | 仅 `--env-prior`：`learned`（对照） vs `attn`（处理） |

**先决条件**：跑之前先把本分支改动 commit —— `run_id` 与 `config.json` 里的 `commit`
取 `git rev-parse HEAD`，未提交时该字段不指向实验代码。

### 2.1 运行命令（单 seed 对照，两条各跑一次）

```powershell
# 对照臂：env_prior=learned（master 行为）
.venv\Scripts\python.exe run_census_benchmark.py stage2 --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3 --epochs 5 --tag short --env-prior learned

# 处理臂：env_prior=attn（本次改进）
.venv\Scripts\python.exe run_census_benchmark.py stage2 --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3 --epochs 5 --tag short --env-prior attn
```

两条命令都只训练 NewTask 头、复用同一 Stage-1 产物（不重训 Stage-1），结果各自追加一行到
`artifacts/census_stage2/SUMMARY.md`，run 目录 = `artifacts/census_stage2/runs/<run_id>/`。

### 2.2 记录位置

- `config.json` / `metrics.json`：新增 `"env_prior"` 字段；
- `run_id`：协议 run_id + `-p<env_prior>` 后缀（如 `...-short-<commit7>-pattn`），
  协议 `make_run_id` 与 `SUMMARY.md` 列定义均未改动（**protocol.py 零改动**）。

---

## 3. 接受阈值（开跑前写死，不得看到结果后修改）

| 判据 | 阈值 |
|---|---|
| **主判据** | `AUC-Test-Education`（`env_prior="attn"`）**≥ 0.8521** |
| 参照 | 同 stage1 的既存对照行 `AUC-Test-Education = 0.850069`（`learned`，commit `904f8d0`） |
| 附注 | 对照臂 `learned` 复跑**预期复现 0.850069**（同 stage1、同 seed；A3 判据允许 ≤ 1e-9 差异）——复跑值若明显偏离，说明"默认路径不变"假设被破坏，先查因再谈 attn 结果 |
| 不作数 | 协议 B 类门禁（B1–B4）照常落盘，但**性能判定只认主判据 ≥ 0.8521**（B 类是流程性门禁，见协议第 8 节） |

阈值纪律（spec 6.3 / 第 8 节）：阈值由用户在开跑前指定，**不得事后修改**；未达标即止损。

---

## 4. 判定与处置

1. **达标（≥ 0.8521）**：允许扩展到多 seed（seed 列表由用户显式指定），之后才谈是否合并回 `master`。
2. **未达标**：如实把该 run 留在 `SUMMARY.md`（协议 7.3：不得只报告成功的 run），**不**跳到多 seed / 全量跑。
3. **附带的免费校验**：`learned` 复跑值 + 单测里的"逐位一致"契约共同验证默认路径零回归。

---

## 5. 测试（TDD 契约，先写测试后改代码）

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v
```

新增/更新的用例：

| 文件 | 用例 | 断言 |
|---|---|---|
| `census_benchmark/tests/test_newtask_env_prior.py` | `test_default_arg_keeps_learned_env_embedding` | 不传 `env_prior` 可构造，默认 `"learned"`，参数/`buffer` 与 master 同形 |
| 同上 | `test_learned_forward_bit_identical_to_master_reference` | 默认路径输出与 master `forward` 冻结副本**逐位一致**（rtol=atol=0） |
| 同上 | `test_learned_backward_bit_identical_to_master_reference` | 默认路径梯度逐位一致；`env_embedding_network.weight` 仍吃梯度 |
| 同上 | `test_attn_forward_equals_attention_weighted_env_sum` | attn 输出 == `Σ_k W_k(x)·E_k` 参考实现 |
| 同上 | `test_attn_output_independent_of_env_embedding_params` | 大幅扰动该参数输出逐位不变（不再依赖） |
| 同上 | `test_attn_backward_grads_shapes_and_env_dependence` | 除该参数（`.grad is None`）外全部参数梯度形状正确；`E_k` 有非零梯度 |
| 同上 | `test_attn_shares_identical_init_with_learned` | 同 seed 下两模式参数/buffer 逐位一致（第 1.2 节的取舍被钉死） |
| 同上 | `test_attn_output_differs_from_learned_on_identical_weights` | 同权重下两模式输出不同（开关确实生效） |
| 同上 | `test_unknown_env_prior_rejected` | 非法 `env_prior` 抛 `ValueError` |
| `census_benchmark/tests/test_smoke.py` | `test_default_env_prior_learned_recorded_everywhere` / `test_attn_env_prior_recorded_everywhere` | `config.json` / `metrics.json` / `run_id` 三处记录；`SUMMARY.md` 行含 run_id |
| 同上 | `test_cli_env_prior_flag_and_default` / `test_cli_stage2_forwards_env_prior_to_runner` | CLI 默认 `learned`、`--env-prior attn` 生效、非法值被 argparse 拒绝、且确实传到 runner |
| 同上 | `test_static_guards_only_model_py_changed_and_no_flops` | `git diff master` 只允许 `multitaskrec/model.py` |
| 同上 | `test_stage1_classes_byte_identical_to_master` | `EmbeddingNetwork` / `MPTRec` 等类源码与 master 逐字节一致（**不改 Stage1**） |
