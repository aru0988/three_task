# CensusIncome 阶段 2 公平评测基础设施 实现计划（精简版）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 `docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md` 落地 CensusIncome 阶段 1 / 阶段 2 公平评测基础设施：三种子解耦、内容寻址 Stage-1 产物、阶段 2 真冻结与 eval、AUC + 必要机制指标、A/B 门禁、`.gitignore` 规则与 SUMMARY 追加。

**Scope（硬边界）**：首轮**只跑 CensusIncome**、**只跑 model seed `1685480945`**、stage1 `epochs=2`、stage2 `epochs=5`（spec 6.1）；**不做 FLOPs**；**不做通用框架**；**不实现 AliCCP / ByteRec**；**不改** `multitaskrec/model.py`、`multitaskrec/train.py`、`multitaskrec/dataset.py`、`config.py`、`CensusIncome_MPTRec.py`、`CensusIncome_NewTask.py`、`baseline/`；**不含任何模型改进**（Null Expert / TC-Prompt / CGR / affinity gate / KL-Prompt / T4 一律不出现——须等本分支合并回 `master` 后另开 `exp/<name>` 分支，按 spec 6.3 阶梯验证）。

**Architecture:** 新增纯基础设施包 `census_benchmark/`（`protocol.py` + `metrics.py`，零模型语义改动）+ 单入口 `run_census_benchmark.py`（`stage1` / `stage2` 子命令）。阶段 1 复用 `MPTRecTrainManager`（子类只挂记录钩子，loss / 优化语义零改动），训练完把 backbone / env_ids / meta 落盘为内容寻址目录；阶段 2 **只**从 `--stage1-dir` 读取，用外部参数遍历实现真冻结三件套，跑完写 run 产物、做 A/B 门禁、追加 `SUMMARY.md`。

**Tech Stack:** Python 3.10.11、PyTorch 2.6.0+cu124、scikit-learn 1.7.2、pandas 2.3.3、numpy 2.2.6；测试用 stdlib `unittest`（venv 内没有 pytest，**不新增依赖**）。

## 环境与约定（执行者必读）

- 工作目录 `D:\MPT-Rec-three_task\MPT-Rec`（仓库根；测试依赖 cwd = 仓库根）；解释器 `.venv\Scripts\python.exe`（主 shell 为 PowerShell）。
- 全量测试：`.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v`
  （`-t` 与 `-s` 相同 → 不需要 `census_benchmark/tests/__init__.py`；`-m` 把 cwd 放进 `sys.path`，故 `import census_benchmark.protocol` 可解析。）单文件加 `-p "test_protocol.py"`。
- 单元测试全部 CPU、秒级、**不读真实数据集**；真实数据只出现在 T7 的手工 GPU 单跑。**不要** `git add -f`；push 与合并需用户显式批准；每个任务结束提交一次。

**六条已核实的实现判断（不要在实现时"顺手改回去"）**

1. **划分用索引不用 Dataset 对象**：同一 `random_state` 下 `train_test_split(dataset, ...)` 与 `train_test_split(np.arange(n), ...)` 是**同一排列**（permutation 只由 `random_state` 与 `n_samples` 决定），但索引可落盘、可校验。T2 的 P1 固定这一事实。
2. **`env_ids` 长度必须 = 训练集长度**：`MPTRecTrainManager` 用 `env_ids[batch_size*step : batch_size*(step+1)]` 取环境标签，末尾不满 batch 时切片自然截断对齐；故传给 manager 的 `batch_size` 必须 = loader 的 `batch_size`（T5 用 `loaders["train"].batch_size`）。
3. **阶段 2 的 loss 用 `.to(device)`**（master 写的是 `pred.cpu()`）：BCE 数值等价、AUC 等价，但设备一致。只在新文件里这么写。
4. **不实现 FLOPs**：协议路径一律不 `import fvcore`（T1 静态守卫锁定）。
5. **冻结靠外部参数遍历**：不给 `MPTRec` 新增 `freeze_params()`，`model.py` 与 `master` 逐字节一致（spec 4.4、§10）。
6. **阶段 2 的种子全部从 Stage-1 `meta.json` 读取**（split / model / env），CLI 不重复传入。

## 文件结构（先锁定，不再增加）

```
新建（6 个）
census_benchmark/__init__.py                 # 一行 docstring
census_benchmark/protocol.py                 # 常量 + seed/split/指纹 + Stage-1 产物 + 冻结 + hash/run_id/SUMMARY
census_benchmark/metrics.py                  # AUC + 机制指标 M1/M3/M4 + A/B 门禁判定
census_benchmark/tests/{test_protocol.py,test_metrics.py,test_smoke.py}   # P1–P8 / M1–M7 / S1–S5
run_census_benchmark.py                      # 入口：stage1 / stage2 子命令 + Stage1HookTrainManager
修改（1 个）：.gitignore                      # 新增 artifacts/census_stage2 忽略规则（spec 9.2）
运行期产出（不入库，除 SUMMARY.md）
artifacts/census_stage2/
  splits/<split_seed>/split_fingerprint.json + split_indices.npz   # 划分基准（A2）
  stage1/<stage1_id>/{backbone.pt,env_ids.pt,meta.json,stdout.log}
  runs/<run_id>/{config.json,metrics.json,split_fingerprint.json,env_ids.pt,newtask.pt,gate_report.json,stdout.log}
  SUMMARY.md                                                       # 唯一入库产物
```

`run_id` 格式：`<YYYYMMDD-HHmm>-s<splitSeed>-m<modelSeed>-<short|full>-<commit7>`（spec 9.1）。

---

## T1 — 包骨架、`.gitignore` 规则、静态守卫

**Files:** Create `census_benchmark/__init__.py`、`census_benchmark/tests/test_smoke.py`；Modify `.gitignore`

- [ ] **Step 1: 写失败测试**（本任务只放 S1、S2；S3–S5 在 T5 / T6 追加）

```python
import subprocess, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # 测试依赖 cwd = 仓库根


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def _ignored(rel_path):
    """git check-ignore 返回 0 表示被忽略。"""
    return subprocess.run(["git", "check-ignore", "-q", rel_path], cwd=REPO).returncode == 0


class TestStaticGuards(unittest.TestCase):
    def test_gitignore_blocks_artifacts_and_keeps_summary(self):
        for rel in ("artifacts/census_stage2/stage1", "artifacts/census_stage2/runs",
                    "artifacts/census_stage2/splits",
                    "artifacts/census_stage2/stage1/s1-deadbeef-m1-e2-cafe0000/backbone.pt",
                    "artifacts/census_stage2/runs/20260929-1530-s20260929-m1685480945-short-fd75198/metrics.json"):
            self.assertTrue(_ignored(rel), f"应被忽略: {rel}")
        self.assertFalse(_ignored("artifacts/census_stage2/SUMMARY.md"))     # SUMMARY 必须能入库

    def test_static_guards_master_untouched_and_no_flops(self):
        diff = _git("diff", "--name-only", "master", "--", "multitaskrec", "config.py",
                    "CensusIncome_MPTRec.py", "CensusIncome_NewTask.py").stdout.strip()
        self.assertEqual(diff, "", f"协议分支不得改动模型/master 文件: {diff}")
        protocol_src = (REPO / "census_benchmark" / "protocol.py").read_text(encoding="utf-8")
        runner_src = (REPO / "run_census_benchmark.py").read_text(encoding="utf-8")
        for src in (protocol_src, runner_src):
            self.assertNotIn("fvcore", src)                    # 不做 FLOPs（spec 2.2.4）
        self.assertIn("random_state=split_seed", protocol_src)  # 划分只由 split seed 决定（spec 5.3.5）
        self.assertNotIn("random_state=model_seed", protocol_src)
```

- [ ] **Step 2: 跑测试确认失败**

```powershell
New-Item -ItemType Directory -Force census_benchmark\tests | Out-Null
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v
```

预期：`Ran 2 tests ... FAILED (failures=1, errors=1)`——S1 失败于 `stage1` / `runs` / `splits` 未被忽略（现有 `.gitignore` 只有 `*.json` / `*.pt` / `*.log` 等文件级规则），S2 因 `census_benchmark/protocol.py` 不存在而 `FileNotFoundError`。

- [ ] **Step 3: 建包骨架 + 追加 `.gitignore` 规则**

创建 `census_benchmark/__init__.py`（一行：`"""CensusIncome 阶段 2 公平评测协议实现。"""`），`.gitignore` 末尾追加（**不带**结尾斜杠，`git check-ignore` 对不存在的目录也能判定）：

```gitignore
artifacts/census_stage2/stage1
artifacts/census_stage2/runs
artifacts/census_stage2/splits
!artifacts/census_stage2/SUMMARY.md
```

- [ ] **Step 4: 跑测试**

预期：`Ran 2 tests ... FAILED (errors=1)`（S1 转绿；S2 仍缺 `protocol.py`——**不要**为让它变绿而删断言，T2 / T5 会自然转绿）。

- [ ] **Step 5: 提交**

```powershell
git add .gitignore census_benchmark/__init__.py census_benchmark/tests/test_smoke.py
git commit -m "infra: 协议包骨架与 artifacts 忽略规则（spec 9.2）"
```

---

## T2 — `protocol.py` 第一段：常量、三种子、划分与指纹

**Files:** Create `census_benchmark/protocol.py`、`census_benchmark/tests/test_protocol.py`

- [ ] **Step 1: 写失败测试（P1–P4）**

```python
import tempfile, unittest
from datetime import datetime
from pathlib import Path

import numpy as np, torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from census_benchmark import protocol


class ListDataset(Dataset):
    """最小 list 型 Dataset，复现 CensusIncomeDataset 的索引行为（无 .shape）。"""
    def __init__(self, n): self.data = [(i, {"x": i}) for i in range(n)]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


class TestSplitAndSeeds(unittest.TestCase):
    def test_index_split_matches_dataset_split(self):
        ds = ListDataset(37)
        direct_val, direct_test = train_test_split(ds, test_size=0.5, random_state=protocol.SPLIT_SEED)
        perm_val, perm_test = train_test_split(np.arange(len(ds)), test_size=0.5, random_state=protocol.SPLIT_SEED)
        self.assertEqual([ds[int(i)] for i in perm_val], direct_val)      # 同一排列
        self.assertEqual([ds[int(i)] for i in perm_test], direct_test)
        val_idx, test_idx = protocol.make_split(len(ds), protocol.SPLIT_SEED)
        self.assertEqual(sorted(perm_val.tolist()), val_idx.tolist())     # 规范形式 = 排序后的索引
        self.assertEqual(sorted(perm_test.tolist()), test_idx.tolist())

    def test_split_disjoint_complete_and_fingerprint_deterministic(self):
        val_idx, test_idx = protocol.make_split(1000, protocol.SPLIT_SEED)
        stats = protocol.split_stats(val_idx, test_idx, n_train=2000)
        self.assertTrue(stats["disjoint"]); self.assertTrue(stats["union_complete"])
        self.assertEqual((stats["n_train"], stats["n_val"], stats["n_test"]), (2000, 500, 500))
        fp1 = protocol.split_fingerprint(split_seed=protocol.SPLIT_SEED, stats=stats, val_idx=val_idx, test_idx=test_idx)
        fp2 = protocol.split_fingerprint(split_seed=protocol.SPLIT_SEED, stats=stats, val_idx=val_idx, test_idx=test_idx)
        self.assertEqual(fp1, fp2)                                        # 同 split seed → 指纹恒定（A2）
        v2, t2 = protocol.make_split(1000, protocol.SPLIT_SEED + 1)
        fp3 = protocol.split_fingerprint(split_seed=protocol.SPLIT_SEED + 1,
                                         stats=protocol.split_stats(v2, t2, 2000), val_idx=v2, test_idx=t2)
        self.assertNotEqual(fp1["fingerprint_sha256"], fp3["fingerprint_sha256"])

    def test_protocol_constants_match_spec(self):
        self.assertEqual((protocol.SPLIT_SEED, protocol.MODEL_SEED, protocol.ENV_SEED), (20260929, 1685480945, 20260929))
        self.assertEqual((protocol.STAGE1_EPOCHS, protocol.STAGE2_EPOCHS, protocol.PATIENCE), (2, 5, 2))
        self.assertEqual((protocol.BATCH_SIZE, protocol.LR), (256, 1e-3))
        self.assertEqual((protocol.UNI_COE, protocol.ENV_COE), (0.9, 0.1))
        self.assertEqual((protocol.REG_EMBEDDING, protocol.REG_DNN), (0.006, 3e-5))
        self.assertEqual((protocol.INPUT_SIZE, protocol.EMBEDDING_SIZE), (123, 4))
        self.assertEqual((protocol.EXPERT_HIDDEN, protocol.TOWER_HIDDEN), ((256, 128), (64, 32)))

    def test_env_ids_independent_of_global_rng(self):
        torch.manual_seed(0); x1 = torch.rand(3)
        torch.manual_seed(0); e1 = protocol.make_env_ids(1000, protocol.ENV_SEED); x2 = torch.rand(3)
        e2 = protocol.make_env_ids(1000, protocol.ENV_SEED)
        self.assertTrue(torch.equal(x1, x2))                              # 全局 RNG 未被消耗（spec 5.3.2）
        self.assertTrue(torch.equal(e1, e2))                              # 同 env seed 可复现
        self.assertEqual(set(e1.tolist()), {0, 1})
        self.assertNotEqual(protocol.sha256_tensor(e1),
                            protocol.sha256_tensor(protocol.make_env_ids(1000, protocol.ENV_SEED + 1)))
```

- [ ] **Step 2: 跑测试确认失败**

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -p "test_protocol.py" -v
```

预期：`ERROR`（`ModuleNotFoundError: No module named 'census_benchmark.protocol'`）。

- [ ] **Step 3: 实现第一段（`census_benchmark/protocol.py`）**

```python
"""CensusIncome 阶段 2 公平评测协议（唯一事实来源：docs/superpowers/specs/2026-09-29-*.md）。"""
from __future__ import annotations

import hashlib, json, subprocess
from pathlib import Path

import numpy as np, torch
from sklearn.model_selection import train_test_split

# ---- 三个独立种子（spec 5.2）----
SPLIT_SEED = 20260929          # 唯一决定 val/test 划分，跨 model seed 恒定
MODEL_SEED = 1685480945        # 首轮唯一 model seed（模型初始化 / batch 顺序 / dropout）
ENV_SEED = 20260929            # 只决定初始 env_ids
# ---- 首轮 smoke 配置（spec 6.1：只降 epochs，不动数据与超参）----
STAGE1_EPOCHS, STAGE2_EPOCHS, PATIENCE = 2, 5, 2
# ---- 论文超参（spec 2.2.3，本分支冻结）----
BATCH_SIZE, LR = 256, 1e-3
REG_EMBEDDING, REG_DNN = 0.006, 3e-5
UNI_COE, ENV_COE = 0.9, 0.1
INPUT_SIZE, EMBEDDING_SIZE = 123, 4
EXPERT_HIDDEN, TOWER_HIDDEN = (256, 128), (64, 32)
NUM_TASKS, NUM_ENVS = 2, 2
# ---- 门禁阈值（spec 8；只允许在看到结果之前修改）----
AUC_FLOOR, VAL_TEST_GAP, GATE_MIN, GATE_MAX, ENV_SHARE_MIN = 0.60, 0.03, 0.05, 0.95, 0.05

ARTIFACT_ROOT = Path("artifacts/census_stage2")
SUMMARY_COLUMNS = ["run_id", "commit", "auc_test_education",
                   "A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4", "stage1_id"]


def seed_model(model_seed: int) -> None:
    """只播种训练随机性来源；绝不参与划分（spec 5.2）。"""
    torch.manual_seed(model_seed); torch.cuda.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed); np.random.seed(model_seed)


def make_env_ids(n: int, env_seed: int, num_envs: int = NUM_ENVS) -> torch.Tensor:
    """初始环境分配：独立 Generator，不消耗全局 RNG（spec 5.3.2）。长度必须 = 训练集长度。"""
    return torch.randint(0, num_envs, size=(n,), generator=torch.Generator().manual_seed(env_seed))


def make_split(n_test: int, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """按索引切分 test.gz，返回排序后的 (val_idx, test_idx)：与 train_test_split(dataset, ...,
    random_state=split_seed) 同一排列，但可落盘、可校验。"""
    val_idx, test_idx = train_test_split(np.arange(n_test), test_size=0.5, random_state=split_seed)
    return np.sort(val_idx), np.sort(test_idx)


def split_stats(val_idx: np.ndarray, test_idx: np.ndarray, n_train: int) -> dict:
    """A4 的三集合实测行数与交并检查。"""
    n = len(val_idx) + len(test_idx)
    return {"n_train": int(n_train), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
            "disjoint": bool(set(val_idx.tolist()).isdisjoint(test_idx.tolist())),
            "union_complete": bool(sorted(val_idx.tolist() + test_idx.tolist()) == list(range(n)))}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    t = tensor.detach().cpu().contiguous()
    return sha256_bytes(str(t.dtype).encode() + str(tuple(t.shape)).encode() + t.numpy().tobytes())


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def config_hash(cfg: dict) -> str:
    return sha256_bytes(canonical_json(cfg).encode())


def code_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def split_fingerprint(*, split_seed: int, stats: dict, val_idx, test_idx) -> dict:
    """确定性指纹：不含时间戳，同 split seed 必须逐字节一致（A2）。"""
    fp = {"split_seed": int(split_seed), **stats,
          "val_sha256": sha256_tensor(torch.from_numpy(val_idx)),
          "test_sha256": sha256_tensor(torch.from_numpy(test_idx)),
          "val_indices_head": val_idx[:5].tolist(), "test_indices_head": test_idx[:5].tolist()}
    fp["fingerprint_sha256"] = sha256_bytes(canonical_json(fp).encode())
    return fp


def split_dir(root: Path, split_seed: int) -> Path:
    return Path(root) / "splits" / str(split_seed)


def write_split_fingerprint(root: Path, split_seed: int, fp: dict) -> Path:
    path = split_dir(root, split_seed) / "split_fingerprint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_split_fingerprint(root: Path, split_seed: int) -> dict | None:
    path = split_dir(root, split_seed) / "split_fingerprint.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def verify_split_fingerprint(fp: dict, baseline: dict) -> bool:
    """A2：指纹只由 split seed 决定，必须与基准逐字节一致。"""
    return bool(baseline) and fp["fingerprint_sha256"] == baseline["fingerprint_sha256"]


def save_split_indices(root: Path, split_seed: int, val_idx, test_idx) -> Path:
    path = split_dir(root, split_seed) / "split_indices.npz"
    path.parent.mkdir(parents=True, exist_ok=True); np.savez(path, val=val_idx, test=test_idx)
    return path
```

- [ ] **Step 4: 跑全量测试**

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v
```

预期：`Ran 6 tests ... FAILED (errors=1)`（P1–P4 与 S1 通过；S2 仍因 `run_census_benchmark.py` 不存在而 `FileNotFoundError`，T5 转绿）。

- [ ] **Step 5: 提交**

```powershell
git add census_benchmark/protocol.py census_benchmark/tests/test_protocol.py
git commit -m "infra: 协议常量、三种子解耦、划分与指纹（spec 5）"
```

---

## T3 — `protocol.py` 第二段：Stage-1 产物、真冻结、run_id 与 SUMMARY

**Files:** Modify `census_benchmark/protocol.py`、`census_benchmark/tests/test_protocol.py`

- [ ] **Step 1: 写失败测试（追加 P5–P8）**

追加到 `test_protocol.py`（`input_size` 必须 = `(3+2)*4 = 20`，因为该夹具没有 dense 特征）：

```python
from multitaskrec.model import MPTRec

TINY_VOCAB, TINY_INPUT_SIZE = {"a": 3, "b": 2}, 20


def tiny_mptrec():
    return MPTRec(num_tasks=2, feature_vocabulary=dict(TINY_VOCAB), embedding_size=4,
                  input_size=TINY_INPUT_SIZE, expert_dnn_hidden_units=(8, 4),
                  tower_dnn_hidden_units=(4, 2), device=torch.device("cpu"))


class TestStage1ArtifactsAndFreeze(unittest.TestCase):
    def test_freeze_backbone_and_backbone_sha256(self):
        model = tiny_mptrec()
        before = protocol.backbone_sha256(model)
        protocol.freeze_backbone(model)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertEqual(before, protocol.backbone_sha256(model))          # 冻结不改变参数值
        with torch.no_grad():
            model({"a": torch.zeros(2, dtype=torch.long), "b": torch.zeros(2, dtype=torch.long)})
        protocol.assert_no_grads(model)                                   # 无梯度 → 通过
        param = model.shared_expert_network.mlp[0].weight                  # MLP.mlp 是 nn.Sequential
        param.grad = torch.zeros_like(param)
        with self.assertRaises(AssertionError):
            protocol.assert_no_grads(model)
        with torch.no_grad():
            param.add_(0.1)
        self.assertNotEqual(before, protocol.backbone_sha256(model))       # 参数一变哈希即变

    def test_stage1_id_content_addressed(self):
        base = dict(split_sha="ab" * 32, model_seed=1685480945, epochs=2, cfg_sha="cd" * 32)
        sid = protocol.stage1_id(**base)
        self.assertTrue(sid.startswith("s1-"))
        self.assertEqual(sid, protocol.stage1_id(**base))
        for changed in ({"model_seed": 1}, {"epochs": 3}, {"cfg_sha": "ef" * 32}):
            self.assertNotEqual(sid, protocol.stage1_id(**{**base, **changed}))

    def test_stage1_save_load_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root, sid = Path(td), "s1-deadbeef-m1685480945-e2-cafe0000"
            model, env_ids = tiny_mptrec(), protocol.make_env_ids(16, protocol.ENV_SEED)
            out_dir = protocol.save_stage1(root, sid, backbone_state=model.state_dict(),
                                           env_ids=env_ids, meta={"stage1_id": sid})
            for name in ("backbone.pt", "env_ids.pt", "meta.json"):
                self.assertTrue((out_dir / name).exists())
            loaded = protocol.load_stage1(root, sid)
            self.assertEqual(loaded["meta"]["stage1_id"], sid)
            self.assertTrue(torch.equal(loaded["env_ids"], env_ids))
            self.assertEqual(protocol.sha256_tensor(loaded["env_ids"]), protocol.sha256_tensor(env_ids))
            with self.assertRaises(FileExistsError):                      # 只读产物，永不覆盖（spec 4.2）
                protocol.save_stage1(root, sid, backbone_state=model.state_dict(), env_ids=env_ids, meta={})
            with self.assertRaises(FileNotFoundError):
                protocol.load_stage1(root, "s1-missing")

    def test_run_id_format_and_summary_append(self):
        rid = protocol.make_run_id(datetime(2026, 9, 29, 15, 30), split_seed=20260929,
                                   model_seed=1685480945, tag="short", commit="fd75198")
        self.assertEqual(rid, "20260929-1530-s20260929-m1685480945-short-fd75198")
        self.assertRegex(rid, r"^\d{8}-\d{4}-s\d+-m\d+-(short|full)-[0-9a-f]{7}$")
        with tempfile.TemporaryDirectory() as td:
            summary = Path(td) / "SUMMARY.md"
            row = {"run_id": rid, "commit": "fd75198", "auc_test_education": "0.900000", "stage1_id": "s1-x",
                   **{k: "PASS" for k in ("A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4")}}
            protocol.append_summary_row(summary, row)
            protocol.append_summary_row(summary, {**row, "run_id": rid.replace("1530", "1545")})
            lines = summary.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 4)                                # 表头 + 分隔 + 2 行
            self.assertIn(rid, lines[2]); self.assertIn("1545", lines[3])  # 只追加，不重写历史（spec 7.3）
```

- [ ] **Step 2: 跑测试确认失败**

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -p "test_protocol.py" -v
```

预期：`Ran 8 tests ... FAILED (errors=4)`（`freeze_backbone` / `backbone_sha256` / `assert_no_grads` / `stage1_id` / `save_stage1` / `load_stage1` / `make_run_id` / `append_summary_row` 尚未定义 → `AttributeError`）。

- [ ] **Step 3: 实现第二段（追加到 `protocol.py`）**

```python
# ---- Stage-1 产物（spec 4.2：内容寻址、只读）----
def stage1_id(*, split_sha: str, model_seed: int, epochs: int, cfg_sha: str) -> str:
    """s1-<split指纹前8>-m<modelSeed>-e<epochs>-<config哈希前8>"""
    return f"s1-{split_sha[:8]}-m{model_seed}-e{epochs}-{cfg_sha[:8]}"


def stage1_dir(root: Path, sid: str) -> Path:
    return Path(root) / "stage1" / sid


def save_stage1(root: Path, sid: str, *, backbone_state: dict, env_ids: torch.Tensor, meta: dict) -> Path:
    out_dir = stage1_dir(root, sid)
    if out_dir.exists():
        raise FileExistsError(f"stage1 产物已存在，协议禁止覆盖: {out_dir}")
    out_dir.mkdir(parents=True)
    torch.save(backbone_state, out_dir / "backbone.pt")
    torch.save(env_ids, out_dir / "env_ids.pt")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def load_stage1(root: Path, sid: str) -> dict:
    """阶段 2 的唯一 backbone 来源（spec 4.3）。"""
    out_dir = stage1_dir(root, sid)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"stage1 产物不存在: {out_dir}")
    return {"backbone_state": torch.load(out_dir / "backbone.pt", map_location="cpu"),
            "env_ids": torch.load(out_dir / "env_ids.pt", map_location="cpu"),
            "meta": json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))}


# ---- 真冻结三件套（spec 4.4、4.5；外部参数遍历，不改模型类）----
def freeze_backbone(backbone: torch.nn.Module) -> None:
    """三件套之 1（参数级）+ 2（模式级）；第 3 件（no_grad 抽表征）由调用方保证。"""
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)


def backbone_sha256(backbone: torch.nn.Module) -> str:
    """按参数名排序拼 (name, 张量哈希) 的 sha256 → A1 的 before/after 比对。"""
    digest = hashlib.sha256()
    for name, param in sorted(backbone.named_parameters()):
        digest.update(name.encode()); digest.update(sha256_tensor(param).encode())
    return digest.hexdigest()


def assert_no_grads(backbone: torch.nn.Module) -> None:
    dirty = [name for name, param in backbone.named_parameters() if param.grad is not None]
    if dirty:
        raise AssertionError(f"backbone 参数残留梯度（冻结失效）: {dirty}")


# ---- run_id 与 SUMMARY（spec 9.1、9.2）----
def make_run_id(now, *, split_seed: int, model_seed: int, tag: str, commit: str) -> str:
    return f"{now:%Y%m%d-%H%M}-s{split_seed}-m{model_seed}-{tag}-{commit}"


def append_summary_row(path: Path, row: dict) -> None:
    """只追加，永不重写已有行（spec 7.3；不得只报告成功的 run）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("| " + " | ".join(SUMMARY_COLUMNS) + " |\n" + "|" + "---|" * len(SUMMARY_COLUMNS) + "\n",
                        encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(str(row[col]) for col in SUMMARY_COLUMNS) + " |\n")
```

- [ ] **Step 4: 跑测试**

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -p "test_protocol.py" -v
```

预期：`Ran 8 tests ... OK`。

- [ ] **Step 5: 提交**

```powershell
git add census_benchmark/protocol.py census_benchmark/tests/test_protocol.py
git commit -m "infra: Stage-1 内容寻址产物、真冻结与指纹（spec 4）"
```

---

## T4 — `metrics.py`：AUC、机制指标 M1/M3/M4、A/B 门禁

**Files:** Create `census_benchmark/metrics.py`、`census_benchmark/tests/test_metrics.py`

- [ ] **Step 1: 写失败测试（M1–M7）**

```python
import unittest

import torch
from sklearn.metrics import roc_auc_score

from census_benchmark import metrics


def _ok_kwargs(**over):
    """全部门禁通过的基线参数；单个测试按需覆盖。"""
    base = dict(backbone_sha_equal=True, grads_all_none=True, split_ok=True, split_stats_ok=True,
                env_ids_ok=True, auc_val_income=0.90, auc_val_marital=0.90, auc_test_education=0.90,
                auc_val_education_best=0.90, gate_mean=[0.5, 0.5], env_shares=[0.40, 0.60])
    base.update(over)
    return base


class TestMetrics(unittest.TestCase):
    def test_auc_matches_sklearn(self):
        gen = torch.Generator().manual_seed(0)
        y = torch.randint(0, 2, (500,), generator=gen)
        score = torch.rand(500, generator=gen) + y.float() * 0.5
        self.assertAlmostEqual(metrics.auc(y, score), roc_auc_score(y.tolist(), score.tolist()), places=12)

    def test_env_accuracy(self):
        log_prob = torch.log(torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.1, 0.9]]))
        self.assertAlmostEqual(metrics.env_accuracy(log_prob, torch.tensor([0, 1, 1, 1])), 0.75)

    def test_gate_mean_weights(self):
        stats = metrics.GateStats(num_tasks=2)
        stats.update([torch.tensor([[0.8, 0.2], [0.6, 0.4]]), torch.tensor([[0.1, 0.9], [0.3, 0.7]])])
        out = stats.result()
        self.assertAlmostEqual(out[0], 0.70, places=6); self.assertAlmostEqual(out[1], 0.20, places=6)

    def test_rep_geometry_and_constant_rep(self):
        stats = metrics.RepStats(num_tasks=2)
        stats.update(torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
                     [torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]])])
        out = stats.result()
        self.assertAlmostEqual(out["cos_gen_spec"][0], 0.5, places=6)     # cos: 1.0 与 0.0
        self.assertAlmostEqual(out["cos_gen_spec"][1], 1.0, places=6)
        self.assertAlmostEqual(out["gen_std"], 0.75, places=6)
        flat = metrics.RepStats(num_tasks=1)
        flat.update(torch.ones(4, 3), [torch.ones(4, 3)])
        self.assertAlmostEqual(flat.result()["gen_std"], 0.0, places=9)   # 常量表征 → std 0（防塌缩诊断）

    def test_judge_a_class_hard_gates(self):
        report = metrics.judge(**_ok_kwargs())
        self.assertTrue(report["overall_pass"]); self.assertEqual(report["failures"], [])
        for key in ("A1", "A2", "A4", "A5"):
            self.assertTrue(report[key]["pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(backbone_sha_equal=False))["A1"]["pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(grads_all_none=False))["A1"]["pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(split_ok=False))["overall_pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(env_ids_ok=False))["A5"]["pass"])

    def test_judge_b1_b2_thresholds(self):
        self.assertTrue(metrics.judge(**_ok_kwargs(auc_val_income=0.60))["B1"]["pass"])     # 边界含等号
        self.assertFalse(metrics.judge(**_ok_kwargs(auc_val_income=0.599))["B1"]["pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(auc_test_education=0.59,
                                                    auc_val_education_best=0.59))["B1"]["pass"])
        self.assertTrue(metrics.judge(**_ok_kwargs(auc_val_education_best=0.90,
                                                   auc_test_education=0.88))["B2"]["pass"])   # 差 = 0.02
        self.assertFalse(metrics.judge(**_ok_kwargs(auc_val_education_best=0.90,
                                                    auc_test_education=0.86))["B2"]["pass"])  # 差 = 0.04
        # 阈值 0.03 在二进制浮点下不可精确表示（0.9 - 0.87 = 0.030000000000000027），故用 0.02 / 0.04 覆盖两侧

    def test_judge_b3_b4_degenerate(self):
        self.assertTrue(metrics.judge(**_ok_kwargs(gate_mean=[0.05, 0.95]))["B3"]["pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(gate_mean=[0.02, 0.98]))["B3"]["pass"])    # 坍缩到单一分支
        self.assertTrue(metrics.judge(**_ok_kwargs(env_shares=[0.05, 0.95]))["B4"]["pass"])
        self.assertFalse(metrics.judge(**_ok_kwargs(env_shares=[0.049, 0.951]))["B4"]["pass"])  # 聚类退化
```

- [ ] **Step 2: 跑测试确认失败**

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -p "test_metrics.py" -v
```

预期：`ERROR`（`ModuleNotFoundError: No module named 'census_benchmark.metrics'`）。

- [ ] **Step 3: 实现 `metrics.py`**

```python
"""AUC、机制指标 M1/M3/M4、A/B 门禁判定（spec 7、8）。不 import fvcore，不做 FLOPs。"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from census_benchmark import protocol as P


def auc(y_true: torch.Tensor, y_hat: torch.Tensor) -> float:
    return float(roc_auc_score(y_true.int().cpu(), y_hat.detach().cpu()))


def env_accuracy(env_pred_logprob: torch.Tensor, env_ids: torch.Tensor) -> float:
    """M1：env_pred 与当前 env_ids 的一致率（梯度反转生效时应趋近 1/NUM_TASKS = 0.5）。"""
    return float((env_pred_logprob.argmax(dim=1).cpu() == env_ids.cpu()).float().mean())


class GateStats:
    """M3：各 gate 在样本维度的平均权重（流式累加，显存 O(1)）。"""

    def __init__(self, num_tasks: int = P.NUM_TASKS):
        self.total = torch.zeros(num_tasks, 2, dtype=torch.float64); self.count = 0

    def update(self, gate_outs: list[torch.Tensor]) -> None:
        for i, gate_out in enumerate(gate_outs):
            self.total[i] += gate_out.detach().double().sum(dim=0)
        self.count += gate_outs[0].shape[0]

    def result(self) -> list[float]:
        return (self.total / max(self.count, 1)).tolist()


class RepStats:
    """M4：cos(gen_rep, spec_rep_i) 均值 + gen_rep 各维标准差均值（防常量表征）。

    用累加和 / 平方和代替"按固定 seed 抽样后保存中间张量"：全量、显存 O(1)、不受抽样影响
    （spec 7.2 实现约束以显存峰值为目的，流式累加达成同一目的且更强）。
    """

    def __init__(self, num_tasks: int = P.NUM_TASKS):
        self.cos_sum = torch.zeros(num_tasks, dtype=torch.float64)
        self.gen_sum = self.gen_sq_sum = None
        self.count = 0

    def update(self, gen_rep: torch.Tensor, spec_reps: list[torch.Tensor]) -> None:
        gen = gen_rep.detach().double()
        self.gen_sum = gen.sum(dim=0) if self.gen_sum is None else self.gen_sum + gen.sum(dim=0)
        self.gen_sq_sum = (gen * gen).sum(dim=0) if self.gen_sq_sum is None else self.gen_sq_sum + (gen * gen).sum(dim=0)
        for i, spec in enumerate(spec_reps):
            self.cos_sum[i] += F.cosine_similarity(gen, spec.detach().double(), dim=1).sum()
        self.count += gen.shape[0]

    def result(self) -> dict:
        n = max(self.count, 1)
        mean = self.gen_sum / n
        std = (self.gen_sq_sum / n - mean * mean).clamp_min(0).sqrt()
        return {"cos_gen_spec": (self.cos_sum / n).tolist(), "gen_std": float(std.mean())}


@torch.no_grad()
def evaluate_newtask(newtask, backbone, loader, device, *, mechanism: bool = False) -> dict:
    """评估新任务头。mechanism=True 时额外算 M3/M4（只在 val 上开一次）。"""
    newtask.eval(); backbone.eval()                 # 三件套之 2：backbone 恒为 eval
    ys, preds = [], []
    gate = GateStats() if mechanism else None
    rep = RepStats() if mechanism else None
    for _, _, y, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)   # 三件套之 3：no_grad 抽取
        pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
        ys.append(y); preds.append(pred.detach())
        if mechanism:
            gate.update([backbone.gate_networks[i](dnn_input) for i in range(P.NUM_TASKS)])
            rep.update(gen_rep, spec_reps)
    out = {"auc": auc(torch.cat(ys), torch.cat(preds))}
    if mechanism:
        out["gate_mean"] = gate.result(); out.update(rep.result())
    return out


def judge(*, backbone_sha_equal: bool, grads_all_none: bool, split_ok: bool, split_stats_ok: bool,
          env_ids_ok: bool, auc_val_income: float, auc_val_marital: float, auc_test_education: float,
          auc_val_education_best: float, gate_mean: list[float], env_shares: list[float]) -> dict:
    """A1/A2/A4/A5 + B1–B4（spec 8）。A3 按需触发，由调用方另行写入，不进入 overall_pass。"""

    def verdict(passed: bool, **detail) -> dict:
        return {"pass": bool(passed), "detail": detail}

    report = {
        "A1": verdict(backbone_sha_equal and grads_all_none,
                      backbone_sha_equal=backbone_sha_equal, grads_all_none=grads_all_none),
        "A2": verdict(split_ok, split_fingerprint_consistent=split_ok),
        "A4": verdict(split_stats_ok, disjoint_and_complete=split_stats_ok),
        "A5": verdict(env_ids_ok, env_ids_sha256_matches_stage1=env_ids_ok),
        "B1": verdict(min(auc_val_income, auc_val_marital, auc_test_education) >= P.AUC_FLOOR,
                      auc_val_income=auc_val_income, auc_val_marital=auc_val_marital,
                      auc_test_education=auc_test_education, floor=P.AUC_FLOOR),
        "B2": verdict(abs(auc_val_education_best - auc_test_education) <= P.VAL_TEST_GAP,
                      gap=abs(auc_val_education_best - auc_test_education), limit=P.VAL_TEST_GAP),
        "B3": verdict(all(P.GATE_MIN <= w <= P.GATE_MAX for w in gate_mean), gate_mean=gate_mean),
        "B4": verdict(all(share >= P.ENV_SHARE_MIN for share in env_shares), env_shares=env_shares),
    }
    report["overall_pass"] = all(item["pass"] for item in report.values())
    report["failures"] = [key for key, item in report.items() if isinstance(item, dict) and not item["pass"]]
    return report
```

- [ ] **Step 4: 跑测试**

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -p "test_metrics.py" -v
```

预期：`Ran 7 tests ... OK`。

- [ ] **Step 5: 提交**

```powershell
git add census_benchmark/metrics.py census_benchmark/tests/test_metrics.py
git commit -m "infra: AUC、机制指标与 A/B 门禁判定（spec 7、8）"
```

---

## T5 — 入口 `run_census_benchmark.py`：`stage1` 子命令与记录钩子

**Files:** Create `run_census_benchmark.py`；Modify `census_benchmark/tests/test_smoke.py`

- [ ] **Step 1: 写失败测试（追加 S3）**

追加到 `test_smoke.py`（夹具与 `test_protocol.py` 的同名夹具各自独立，刻意不新增共享模块）：

```python
import json, tempfile

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from census_benchmark import protocol as P
import run_census_benchmark


class TinyCensus(Dataset):
    """极小 CensusIncome 形状：(income, marital, new_task, features)。"""
    def __init__(self, n, seed):
        gen = torch.Generator().manual_seed(seed)
        self.features = {"a": torch.randint(0, 3, (n,), generator=gen),
                         "b": torch.randint(0, 2, (n,), generator=gen)}
        self.labels = [torch.randint(0, 2, (n,), generator=gen).float() for _ in range(3)]
    def __len__(self): return int(self.features["a"].shape[0])
    def __getitem__(self, i):
        return (self.labels[0][i], self.labels[1][i], self.labels[2][i],
                {"a": self.features["a"][i], "b": self.features["b"][i]})


def tiny_model():
    """与 test_protocol.tiny_mptrec 同口径：dnn_input 维度 = (3+2)*4 = 20。"""
    return run_census_benchmark.MPTRec(num_tasks=2, feature_vocabulary={"a": 3, "b": 2}, embedding_size=4,
                                       input_size=20, expert_dnn_hidden_units=(8, 4),
                                       tower_dnn_hidden_units=(4, 2), device=torch.device("cpu"))


def tiny_inputs(n_train=64, n_val=16, n_test=16, batch_size=16):
    """train_ds 是「全量训练集」，test_ds 按 P.SPLIT_SEED 切成 val / test，与真实路径同构。"""
    train_ds, test_ds = TinyCensus(n_train, 1), TinyCensus(n_val + n_test, 2)
    val_idx, test_idx = P.make_split(len(test_ds), P.SPLIT_SEED)
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size),
               "val": DataLoader(Subset(test_ds, val_idx.tolist()), batch_size=batch_size),
               "test": DataLoader(Subset(test_ds, test_idx.tolist()), batch_size=batch_size)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=len(train_ds)), (val_idx, test_idx)


class TestStage1Smoke(unittest.TestCase):
    def test_stage1_smoke_cpu_tiny(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            loaders, stats, indices = tiny_inputs()
            out = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_model(),
                                                  loaders=loaders, stats=stats, indices=indices, tag="short")
            stage1_path = Path(out["dir"])
            for name in ("backbone.pt", "env_ids.pt", "meta.json"):
                self.assertTrue((stage1_path / name).exists())
            self.assertTrue(out["stage1_id"].startswith("s1-"))
            meta = json.loads((stage1_path / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual((meta["epochs"], meta["model_seed"], meta["env_seed"]), (2, P.MODEL_SEED, P.ENV_SEED))
            self.assertEqual(len(meta["epoch_records"]), 2)                 # M1 每 epoch 一条
            self.assertIn("env_acc", meta["epoch_records"][0])
            self.assertEqual(len(meta["cluster_records"]), 1)               # epochs=2 → 第 2 轮后聚一次（M2）
            self.assertEqual(len(meta["fuse_loss_0_list"]), 2)
            self.assertEqual(meta["env_ids_sha256"], P.sha256_tensor(P.load_stage1(root, out["stage1_id"])["env_ids"]))
            self.assertTrue((root / "splits" / str(P.SPLIT_SEED) / "split_fingerprint.json").exists())
            with self.assertRaises(FileExistsError):                        # 内容寻址 + 只读
                run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_model(),
                                                loaders=loaders, stats=stats, indices=indices, tag="short")
```

- [ ] **Step 2: 跑测试确认失败**

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v
```

预期：`Ran 18 tests ... FAILED (errors=2)`（S3 报 `ModuleNotFoundError: No module named 'run_census_benchmark'`；S2 仍是 `FileNotFoundError`；其余 16 个通过）。

- [ ] **Step 3: 实现入口的 stage1 部分**

```python
"""CensusIncome 阶段 1 / 阶段 2 公平评测入口（协议路径；零模型语义改动）。"""
from __future__ import annotations

import argparse, contextlib, copy, io, json, sys
from datetime import datetime
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset

from census_benchmark import metrics
from census_benchmark import protocol as P
from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import MPTRec, NewTask
from multitaskrec.train import MPTRecTrainManager


@contextlib.contextmanager
def tee_stdout(stream):
    """stdout 同时写真实终端与给定文本流（文件或 StringIO）→ 落盘 stdout.log。"""
    class _Tee:
        def write(self, text):
            sys.__stdout__.write(text); stream.write(text); return len(text)
        def flush(self):
            sys.__stdout__.flush(); stream.flush()
    original, sys.stdout = sys.stdout, _Tee()
    try:
        yield
    finally:
        sys.stdout = original


class Stage1HookTrainManager(MPTRecTrainManager):
    """只挂记录钩子，loss / 优化器 / early-stop 语义全部沿用父类（spec 4）。

    父类 train_two_task 每个 epoch 调一次 self.evaluation_two_task(self.val_loader)；
    在该调用点记录 val AUC 与该 epoch 的训练集 env_acc（M1），并重写 cluster_2 记录 M2。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_records: list[dict] = []
        self.cluster_records: list[dict] = []

    def evaluation_two_task(self, data_loader):
        auc_pair = super().evaluation_two_task(data_loader)
        if data_loader is self.val_loader:
            self.epoch_records.append({"epoch": len(self.epoch_records) + 1,
                                       "auc_val_income": float(auc_pair[0]),
                                       "auc_val_marital": float(auc_pair[1]),
                                       "env_acc": self.train_env_acc()})
        return auc_pair

    def cluster_2(self):
        previous = self.env_ids.clone()
        updated = super().cluster_2()
        counts = torch.bincount(updated, minlength=P.NUM_ENVS).tolist()
        self.cluster_records.append({"epoch": len(self.epoch_records),
                                     "diff_num": int((previous != updated).sum()),
                                     "env_counts": [int(c) for c in counts]})
        return updated

    @torch.no_grad()
    def train_env_acc(self) -> float:
        """M1：训练集上 env_pred 与当前 env_ids 的一致率（父类每轮开头重设 train()，无副作用）。"""
        self.model.eval()
        log_probs, ids = [], []
        for step, batch in enumerate(self.train_loader):
            features = {key: value.to(self.device) for key, value in batch[-1].items()}
            log_probs.append(self.model(features)["env_pred"].cpu())
            ids.append(self.env_ids[self.batch_size * step: self.batch_size * (step + 1)])
        return metrics.env_accuracy(torch.cat(log_probs), torch.cat(ids))


def build_census_loaders(split_seed: int):
    """返回 (loaders, stats, (val_idx, test_idx))：train.gz 全量训练；test.gz 按 split seed 切成
    val / test，两侧都是 `Subset(test_dataset, idx.tolist())` + `batch_size=P.BATCH_SIZE`（spec 2.1、4.6）。"""
    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
    val_idx, test_idx = P.make_split(len(test_dataset), split_seed)
    loaders = {"train": DataLoader(train_dataset, batch_size=P.BATCH_SIZE),
               "val": DataLoader(Subset(test_dataset, val_idx.tolist()), batch_size=P.BATCH_SIZE),
               "test": DataLoader(Subset(test_dataset, test_idx.tolist()), batch_size=P.BATCH_SIZE)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=len(train_dataset)), (val_idx, test_idx)


def build_mptrec(device) -> MPTRec:
    """vocabulary 必须 `CensusIncome_Vocabulary_Size.copy()` 后再 `pop("education")`（spec 2.2.1）。"""
    vocabulary = CensusIncome_Vocabulary_Size.copy(); vocabulary.pop("education")
    return MPTRec(num_tasks=P.NUM_TASKS, feature_vocabulary=vocabulary, embedding_size=P.EMBEDDING_SIZE,
                  input_size=P.INPUT_SIZE, expert_dnn_hidden_units=list(P.EXPERT_HIDDEN),
                  tower_dnn_hidden_units=list(P.TOWER_HIDDEN),
                  reg_embedding=P.REG_EMBEDDING, reg_dnn=P.REG_DNN, device=device)


def run_stage1(root=P.ARTIFACT_ROOT, *, split_seed=P.SPLIT_SEED, model_seed=P.MODEL_SEED,
               env_seed=P.ENV_SEED, epochs=P.STAGE1_EPOCHS, tag="short", device=None,
               model=None, loaders=None, stats=None, indices=None) -> dict:
    root = Path(root); device = device or torch.device("cuda:0")
    commit, log_buffer = P.code_commit(), io.StringIO()

    # 1) 划分：只由 split seed 决定；首次落盘基准，之后必须一致（A2、A4）
    if loaders is None:
        loaders, stats, indices = build_census_loaders(split_seed)
    val_idx, test_idx = indices
    fp = P.split_fingerprint(split_seed=split_seed, stats=stats, val_idx=val_idx, test_idx=test_idx)
    baseline = P.load_split_fingerprint(root, split_seed)
    if baseline is None:
        P.write_split_fingerprint(root, split_seed, fp)
        P.save_split_indices(root, split_seed, val_idx, test_idx)
    elif not P.verify_split_fingerprint(fp, baseline):
        raise RuntimeError("划分指纹与基准不一致（A2 失败），协议禁止继续")

    # 2) 初始环境分配：独立 env seed（spec 5.3.2），长度 = 训练集长度
    env_ids = P.make_env_ids(len(loaders["train"].dataset), env_seed)

    # 3) 训练：复用 MPTRecTrainManager 语义，只额外记录
    P.seed_model(model_seed)                              # 模型初始化 / batch 顺序 / dropout
    model = model or build_mptrec(device)
    model.to(device)
    manager = Stage1HookTrainManager(
        model=model, train_loader=loaders["train"], val_loader=loaders["val"], env_ids=env_ids,
        task_name=["Income", "Marital"], lr=P.LR, batch_size=loaders["train"].batch_size,
        uni_coe=P.UNI_COE, env_coe=P.ENV_COE, epochs=epochs, patience=P.PATIENCE)
    with tee_stdout(log_buffer):
        manager.train_two_task()
    model.load_state_dict(manager.best_weight)
    final_env_ids = manager.env_ids.clone()               # cluster 覆盖后的最终环境分配 → 阶段 2 读取

    # 4) 内容寻址落盘（spec 4.2；只读、不覆盖）
    cfg = {"model_seed": model_seed, "env_seed": env_seed, "epochs": epochs, "patience": P.PATIENCE,
           "batch_size": loaders["train"].batch_size, "lr": P.LR, "uni_coe": P.UNI_COE, "env_coe": P.ENV_COE,
           "reg_embedding": P.REG_EMBEDDING, "reg_dnn": P.REG_DNN, "input_size": P.INPUT_SIZE,
           "embedding_size": P.EMBEDDING_SIZE, "expert_hidden": list(P.EXPERT_HIDDEN),
           "tower_hidden": list(P.TOWER_HIDDEN), "num_tasks": P.NUM_TASKS}
    cfg_sha = P.config_hash(cfg)
    sid = P.stage1_id(split_sha=fp["fingerprint_sha256"], model_seed=model_seed, epochs=epochs, cfg_sha=cfg_sha)
    records = manager.epoch_records
    meta = {"stage1_id": sid, "commit": commit, "config_hash": cfg_sha, **cfg, **stats,
            "created": datetime.now().isoformat(timespec="seconds"),
            "split_seed": split_seed, "model_seed": model_seed, "env_seed": env_seed,
            "split_fingerprint_sha256": fp["fingerprint_sha256"],
            "val_sha256": fp["val_sha256"], "test_sha256": fp["test_sha256"],
            "env_ids_sha256": P.sha256_tensor(final_env_ids), "backbone_sha256": P.backbone_sha256(model),
            "best_epoch": max(records, key=lambda r: r["auc_val_income"] + r["auc_val_marital"])["epoch"],
            "best_val_auc_sum": max(r["auc_val_income"] + r["auc_val_marital"] for r in records),
            "val_auc_income_max": max(r["auc_val_income"] for r in records),
            "val_auc_marital_max": max(r["auc_val_marital"] for r in records),
            "epoch_records": records, "cluster_records": manager.cluster_records,
            "uni_loss_0_list": manager.uni_loss_0_list, "uni_loss_1_list": manager.uni_loss_1_list,
            "fuse_loss_0_list": manager.fused_loss_0_list, "fuse_loss_1_list": manager.fused_loss_1_list,
            "env_loss_list": manager.env_loss_list}
    stage1_path = P.save_stage1(root, sid, backbone_state=model.state_dict(), env_ids=final_env_ids, meta=meta)
    (stage1_path / "stdout.log").write_text(log_buffer.getvalue(), encoding="utf-8")
    print(f"[stage1] id={sid} dir={stage1_path} backbone_sha256={meta['backbone_sha256']}")
    return {"stage1_id": sid, "dir": str(stage1_path), "meta": meta}
```

- [ ] **Step 4: 跑全量测试**

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v
```

预期：`Ran 18 tests ... OK`（S2 转绿：`run_census_benchmark.py` 已存在且不含 `fvcore`）。

- [ ] **Step 5: 提交**

```powershell
git add run_census_benchmark.py census_benchmark/tests/test_smoke.py
git commit -m "infra: stage1 子命令与阶段 1 记录钩子（spec 4.2、7.2 M1/M2）"
```

---

## T6 — `stage2` 子命令、门禁产物与 SUMMARY 追加

**Files:** Modify `run_census_benchmark.py`、`census_benchmark/tests/test_smoke.py`

- [ ] **Step 1: 写失败测试（追加 S4、S5）**

```python
class TestStage2Smoke(unittest.TestCase):
    def test_stage2_smoke_cpu_tiny(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            loaders, stats, indices = tiny_inputs()
            stage1 = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_model(),
                                                     loaders=loaders, stats=stats, indices=indices)
            out = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=3, device=device,
                                                  model=tiny_model(), loaders=loaders, stats=stats,
                                                  indices=indices, input_size=20, rep_dim=4)
            run_path = Path(out["run_dir"])
            for name in ("config.json", "metrics.json", "split_fingerprint.json", "env_ids.pt",
                         "newtask.pt", "gate_report.json", "stdout.log"):
                self.assertTrue((run_path / name).exists())
            report = json.loads((run_path / "gate_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["A1"]["pass"])                      # 同一 checkpoint + 真冻结
            self.assertTrue(report["A1"]["detail"]["backbone_sha_equal"])
            self.assertTrue(report["A2"]["pass"])
            self.assertTrue(report["A5"]["pass"])                      # env_ids 只读取、不重抽
            self.assertIn("overall_pass", report)                      # 小夹具上 B 类可能为 False，属预期
            self.assertEqual(report["A3"]["status"], "on_demand")      # spec 8：A3 按需触发
            metrics_json = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(metrics_json["stage2"]["epoch_records"]), 3)
            self.assertGreaterEqual(metrics_json["stage2"]["test_auc"], 0.0)
            self.assertEqual(len(metrics_json["mechanism"]["gate_mean"]), 2)
            self.assertEqual(metrics_json["backbone_sha256_before"], metrics_json["backbone_sha256_after"])
            self.assertEqual(metrics_json["stage1_id"], stage1["stage1_id"])
            summary = (root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(summary), 3)                          # 表头 + 分隔 + 1 行
            self.assertIn(out["run_id"], summary[2])

    def test_stage2_missing_stage1_dir_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                run_census_benchmark.run_stage2(Path(td) / "artifacts",
                                                stage1_dir=Path(td) / "no_such_stage1",
                                                device=torch.device("cpu"))
```

- [ ] **Step 2: 跑测试确认失败**

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -p "test_smoke.py" -v
```

预期：`Ran 5 tests ... FAILED (errors=2)`（`AttributeError: module 'run_census_benchmark' has no attribute 'run_stage2'`）。

- [ ] **Step 3: 实现 stage2 与 CLI（追加到 `run_census_benchmark.py`）**

产物用 `_write_json(path, payload)` 落盘（`json.dumps(..., ensure_ascii=False, indent=2)` + `write_text`）；payload 键名由 S4 与 spec 7.1 / 9.1 固定，不得改名：

- `config.json`：`run_id` / `stage1_id` / `commit` / `tag` / `frozen` / `split_seed` / `model_seed` / `env_seed` / `epochs` / `patience` / `lr` / `batch_size` / `input_size` / `rep_dim`
- `metrics.json`：`run_id`、`stage1_id`、`commit`、`split_sha256{val,test,fingerprint}`、`env_ids_sha256`、`backbone_sha256_before`、`backbone_sha256_after`、`stage1{epoch_records,best_epoch,uni_loss_0,uni_loss_1,fuse_loss_0,fuse_loss_1,env_loss,cluster_records}`、`stage2{epoch_records,best_epoch,best_val_auc,test_auc}`、`mechanism{gate_mean,cos_gen_spec,gen_std,env_acc_stage1}`
- `gate_report.json`：`metrics.judge(...)` 返回值 + `A3 = {"status": "on_demand", "detail": "按需复跑同一配置，比较 AUC-Test-Education 差 ≤ 1e-9 与 backbone_sha256 一致"}`

```python
def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_stage2(root=P.ARTIFACT_ROOT, *, stage1_dir, epochs=P.STAGE2_EPOCHS, tag="short", device=None,
               model=None, loaders=None, stats=None, indices=None, input_size=P.INPUT_SIZE,
               rep_dim=P.EXPERT_HIDDEN[-1], now=None) -> dict:
    """阶段 2：只从 Stage-1 产物加载 backbone，训练新任务头，做门禁与 SUMMARY。"""
    root = Path(root); device = device or torch.device("cuda:0")
    sid = Path(stage1_dir).name
    checkpoint = P.load_stage1(root, sid)                  # spec 4.3：唯一 backbone 来源（缺失即抛错）
    meta, commit, log_buffer = checkpoint["meta"], P.code_commit(), io.StringIO()

    # 1) 同一划分：用 stage1 的 split seed 重建并双向校验（A2）
    if loaders is None:
        loaders, stats, indices = build_census_loaders(meta["split_seed"])
    val_idx, test_idx = indices
    fp = P.split_fingerprint(split_seed=meta["split_seed"], stats=stats, val_idx=val_idx, test_idx=test_idx)
    split_ok = (P.verify_split_fingerprint(fp, P.load_split_fingerprint(root, meta["split_seed"]))
                and fp["fingerprint_sha256"] == meta["split_fingerprint_sha256"])

    # 2) backbone 加载 + 真冻结三件套之 1、2；env_ids 只读取（A5）
    P.seed_model(meta["model_seed"])                       # 让 NewTask 初始化可复现
    backbone = model or build_mptrec(device)
    backbone.to(device)
    backbone.load_state_dict(checkpoint["backbone_state"])
    P.freeze_backbone(backbone)
    sha_before = P.backbone_sha256(backbone)
    env_ids = checkpoint["env_ids"]
    env_ids_ok = P.sha256_tensor(env_ids) == meta["env_ids_sha256"]

    newtask = NewTask(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=list(P.TOWER_HIDDEN),
                      reg_dnn=P.REG_DNN, device=device).to(device)
    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=P.LR)      # 只含 NewTask 参数
    loss_func = nn.BCELoss()
    best_auc, best_epoch, best_state, stale, epoch_records = -1.0, 0, None, 0, []

    # 3) 训练：val 只用于 early stop 选点；test 全程不参与选择（spec 4.6、7.3）
    with tee_stdout(log_buffer):
        for epoch in range(1, epochs + 1):
            newtask.train()
            loss_sum, steps = 0.0, 0
            for _, _, y, features in loaders["train"]:
                features = {key: value.to(device) for key, value in features.items()}
                with torch.no_grad():                                        # 三件套之 3：计算图级冻结
                    dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
                pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
                loss = loss_func(pred, y.float().to(device)) + newtask.get_l2_reg()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                loss_sum += float(loss); steps += 1
            val = metrics.evaluate_newtask(newtask, backbone, loaders["val"], device)
            epoch_records.append({"epoch": epoch, "loss": loss_sum / max(steps, 1),
                                  "auc_val_education": val["auc"]})
            print(f"[stage2] epoch={epoch} loss={loss_sum / max(steps, 1):.4f} auc_val_education={val['auc']:.4f}")
            if val["auc"] > best_auc:
                best_auc, best_epoch, stale = val["auc"], epoch, 0
                best_state = copy.deepcopy(newtask.state_dict())
            else:
                stale += 1
                if stale == P.PATIENCE:
                    print(f"[stage2] early stop at epoch {epoch}"); break

    # 4) 冻结校验 + 最终评测（val 带机制指标一次；test 只评一次）
    newtask.load_state_dict(best_state)
    sha_after = P.backbone_sha256(backbone)
    grads_all_none = all(param.grad is None for param in backbone.parameters())
    P.assert_no_grads(backbone)
    val_final = metrics.evaluate_newtask(newtask, backbone, loaders["val"], device, mechanism=True)
    test_final = metrics.evaluate_newtask(newtask, backbone, loaders["test"], device)

    # 5) 门禁判定 + 产物落盘（spec 8、9.1、9.2）
    env_shares = [count / meta["n_train"] for record in meta["cluster_records"] for count in record["env_counts"]]
    report = metrics.judge(
        backbone_sha_equal=(sha_before == sha_after), grads_all_none=grads_all_none, split_ok=split_ok,
        split_stats_ok=bool(stats["disjoint"] and stats["union_complete"]), env_ids_ok=env_ids_ok,
        auc_val_income=meta["val_auc_income_max"], auc_val_marital=meta["val_auc_marital_max"],
        auc_test_education=test_final["auc"], auc_val_education_best=best_auc,
        gate_mean=val_final["gate_mean"], env_shares=env_shares)
    report["A3"] = {"status": "on_demand",
                    "detail": "按需复跑同一配置，比较 AUC-Test-Education 差 ≤ 1e-9 与 backbone_sha256 一致"}

    run_id = P.make_run_id(now or datetime.now(), split_seed=meta["split_seed"],
                           model_seed=meta["model_seed"], tag=tag, commit=commit)
    run_path = root / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    torch.save(newtask.state_dict(), run_path / "newtask.pt")
    torch.save(env_ids, run_path / "env_ids.pt")
    _write_json(run_path / "split_fingerprint.json", fp)
    _write_json(run_path / "config.json", {
        "run_id": run_id, "stage1_id": sid, "commit": commit, "tag": tag, "frozen": True,
        "split_seed": meta["split_seed"], "model_seed": meta["model_seed"], "env_seed": meta["env_seed"],
        "epochs": epochs, "patience": P.PATIENCE, "lr": P.LR, "batch_size": loaders["train"].batch_size,
        "input_size": input_size, "rep_dim": rep_dim})
    _write_json(run_path / "metrics.json", {
        "run_id": run_id, "stage1_id": sid, "commit": commit,
        "split_sha256": {"val": fp["val_sha256"], "test": fp["test_sha256"],
                         "fingerprint": fp["fingerprint_sha256"]},
        "env_ids_sha256": meta["env_ids_sha256"],
        "backbone_sha256_before": sha_before, "backbone_sha256_after": sha_after,
        "stage1": {"epoch_records": meta["epoch_records"], "best_epoch": meta["best_epoch"],
                   "uni_loss_0": meta["uni_loss_0_list"], "uni_loss_1": meta["uni_loss_1_list"],
                   "fuse_loss_0": meta["fuse_loss_0_list"], "fuse_loss_1": meta["fuse_loss_1_list"],
                   "env_loss": meta["env_loss_list"], "cluster_records": meta["cluster_records"]},
        "stage2": {"epoch_records": epoch_records, "best_epoch": best_epoch,
                   "best_val_auc": best_auc, "test_auc": test_final["auc"]},
        "mechanism": {"gate_mean": val_final["gate_mean"], "cos_gen_spec": val_final["cos_gen_spec"],
                      "gen_std": val_final["gen_std"],
                      "env_acc_stage1": [record["env_acc"] for record in meta["epoch_records"]]}})
    _write_json(run_path / "gate_report.json", report)
    P.append_summary_row(root / "SUMMARY.md", {
        "run_id": run_id, "commit": commit, "auc_test_education": f"{test_final['auc']:.6f}", "stage1_id": sid,
        **{key: ("PASS" if report[key]["pass"] else "FAIL")
           for key in ("A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4")}})
    (run_path / "stdout.log").write_text(log_buffer.getvalue(), encoding="utf-8")
    print(f"[stage2] run_id={run_id} test_auc={test_final['auc']:.4f} overall_pass={report['overall_pass']}")
    print(f"[stage2] failures={report['failures']} run_dir={run_path}")
    return {"run_id": run_id, "run_dir": str(run_path), "report": report, "test_auc": test_final["auc"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CensusIncome 阶段 2 公平评测（协议路径）")
    parser.add_argument("--root", type=Path, default=P.ARTIFACT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("stage1", help="阶段 1：预训练并落盘内容寻址产物")
    for flag, kwargs in (("--gpu", dict(type=int, default=0)),
                         ("--tag", dict(choices=["short", "full"], default="short")),
                         ("--epochs", dict(type=int, default=P.STAGE1_EPOCHS)),
                         ("--split-seed", dict(type=int, default=P.SPLIT_SEED)),
                         ("--model-seed", dict(type=int, default=P.MODEL_SEED)),
                         ("--env-seed", dict(type=int, default=P.ENV_SEED))):
        first.add_argument(flag, **kwargs)
    second = sub.add_parser("stage2", help="阶段 2：加载固定 Stage-1 产物，只训练新任务头")
    second.add_argument("--stage1-dir", type=Path, required=True)       # spec 4.3：唯一引用方式
    second.add_argument("--gpu", type=int, default=0)
    second.add_argument("--tag", choices=["short", "full"], default="short")
    second.add_argument("--epochs", type=int, default=P.STAGE2_EPOCHS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[cli] command={args.command} device={device} root={args.root}")
    if args.command == "stage1":
        run_stage1(args.root, split_seed=args.split_seed, model_seed=args.model_seed,
                   env_seed=args.env_seed, epochs=args.epochs, tag=args.tag, device=device)
    else:
        run_stage2(args.root, stage1_dir=args.stage1_dir, epochs=args.epochs, tag=args.tag, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑全量测试**

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v
```

预期：`Ran 20 tests ... OK`。

- [ ] **Step 5: 提交**

```powershell
git add run_census_benchmark.py census_benchmark/tests/test_smoke.py
git commit -m "infra: stage2 子命令、真冻结校验与 A/B 门禁产物（spec 4.3-4.6、8、9）"
```

---

## T7 — 首轮 smoke 单跑（真实数据，GPU，单 seed 单次）

**目的**：按 spec 6.1 验证整条链路端到端跑通。**不产出任何性能结论**（spec N2）。非单元测试，产出物不入库。

- [ ] **Step 1: 前置检查**

```powershell
git status --short                     # 预期：无未提交改动（T1–T6 均已提交）
git diff --stat master                 # 预期：只出现 .gitignore / census_benchmark/ / run_census_benchmark.py
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期：`True NVIDIA GeForce RTX 3060 Laptop GPU`。若为 `False`，先修环境再继续（不得改用 CPU 跑首轮，否则 A3 复跑口径不一致）。

- [ ] **Step 2: 阶段 1（预期 ≤ 5 分钟）**

```powershell
.venv\Scripts\python.exe run_census_benchmark.py stage1 --tag short
.venv\Scripts\python.exe -c "import json,pathlib; m=json.loads(pathlib.Path('artifacts/census_stage2/stage1/<id>/meta.json').read_text(encoding='utf-8')); print(m['best_epoch'], m['val_auc_income_max'], m['val_auc_marital_max'], [r['env_acc'] for r in m['epoch_records']], m['cluster_records'][-1])"
```

预期：stdout 打印 `[stage1] id=s1-<8位>-m1685480945-e2-<8位> dir=... backbone_sha256=<hex>`；该目录下 `backbone.pt` / `env_ids.pt` / `meta.json` / `stdout.log` 齐全，`splits\20260929\` 下 `split_fingerprint.json` / `split_indices.npz` 齐全。**记下 `id` 与 `backbone_sha256`**。第二条命令预期：`val_auc_income_max` 与 `val_auc_marital_max` 均 ≥ 0.60（B1 的阶段 1 部分）；`env_acc` 落在 0.5–1.0（M1 只诊断不判定）；`cluster_records[-1]["env_counts"]` 两个环境各 ≥ 5% 训练集样本（B4）。

- [ ] **Step 3: 阶段 2（预期 ≤ 10 分钟）**

```powershell
.venv\Scripts\python.exe run_census_benchmark.py stage2 --stage1-dir artifacts\census_stage2\stage1\<id> --tag short
```

预期 stdout：`[stage2] run_id=<YYYYMMDD-HHmm>-s20260929-m1685480945-short-<commit7> test_auc=<0.6~1.0> overall_pass=True`、`failures=[]`。若 `overall_pass=False`，读 `runs\<run_id>\gate_report.json` 各项 `detail`：

- `A1` 失败 → 冻结被破坏（`.grad` 残留或参数被改），回到 T5/T6 检查 `freeze_backbone` 与 `no_grad` 包裹。
- `A2` 失败 → 划分指纹漂移，检查是否有代码用 `random_state=model_seed`。
- `A5` 失败 → 阶段 2 重抽了 env_ids，检查是否绕过 `load_stage1`。
- `B1/B2` 失败 → 活性问题（标签错位 / epochs 太少）；`B3/B4` 失败 → gate 或聚类退化。两者都是诊断信息，先如实记录再决定是否调整。

- [ ] **Step 4: 验收核对**

```powershell
.venv\Scripts\python.exe -c "import json,pathlib; r=json.loads(pathlib.Path('artifacts/census_stage2/runs/<run_id>/gate_report.json').read_text(encoding='utf-8')); print({k:v['pass'] for k,v in r.items() if isinstance(v,dict) and 'pass' in v})"
Get-Content artifacts\census_stage2\SUMMARY.md
git status --short
```

预期：A1/A2/A4/A5 全 `True`；`SUMMARY.md` = 表头 + 分隔 + 1 行 run 记录（含 `AUC-Test-Education`、A/B 逐项、`stage1_id`）；`git status --short` **只**出现 `?? artifacts/`（SUMMARY.md 位于未被忽略的路径，需显式 `git add` 才入库，见 Step 7）。

- [ ] **Step 5: A3 复跑（按需触发，不阻塞首轮；一旦执行就必须如实记录）**

阶段 2 复跑（`run_id` 含时间戳 → 新目录，无需删除任何东西）：

```powershell
.venv\Scripts\python.exe run_census_benchmark.py stage2 --stage1-dir artifacts\census_stage2\stage1\<id> --tag short
.venv\Scripts\python.exe -c "import json,pathlib; a=json.loads(pathlib.Path('artifacts/census_stage2/runs/<run_id_1>/metrics.json').read_text(encoding='utf-8')); b=json.loads(pathlib.Path('artifacts/census_stage2/runs/<run_id_2>/metrics.json').read_text(encoding='utf-8')); print(abs(a['stage2']['test_auc']-b['stage2']['test_auc']), a['backbone_sha256_before']==b['backbone_sha256_before'])"
```

预期（A3）：差值 ≤ `1e-9` 且打印 `True`。未执行时 `gate_report.json` 的 `A3.status` 保持 `on_demand`，首轮判定不受影响（spec 8）。

阶段 1 复跑（可选，确认内容寻址与确定性）：先删除旧产物目录 `artifacts\census_stage2\stage1\<id>\`（只读产物不可原地覆盖，删除是唯一路径），再执行 Step 2，比较新的 `backbone_sha256` 与旧值——一致即证明阶段 1 可复现。

- [ ] **Step 6: 容量与保留（spec 9.3）**

```powershell
Get-ChildItem -Recurse artifacts | Measure-Object -Property Length -Sum | Select-Object Sum
```

预期：`Sum` ≤ 2 GB。同一 split 指纹下 `stage1/` 只保留最新一个 `stage1_id`；`runs/` 只保留支撑过 `SUMMARY.md` 结论的 run；**永不删除** `SUMMARY.md` 引用且尚未被取代的 run。

- [ ] **Step 7: 提交唯一入库产物**

```powershell
git add artifacts/census_stage2/SUMMARY.md
git commit -m "infra: 首轮 smoke 单跑记录（stage1 + stage2，单 seed 单次）"
```

---

## 完成后自查清单（执行者收尾时逐条确认）

- [ ] 全量测试：`.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v` → `Ran 20 tests ... OK`
- [ ] 单元测试不读真实数据集（只有 T7 的手工单跑读 `dataset/Census-income/*.gz`）
- [ ] `git diff --name-only master` 里没有 `multitaskrec/`、`config.py`、`CensusIncome_MPTRec.py`、`CensusIncome_NewTask.py`、`baseline/`
- [ ] 协议路径没有 `import fvcore` / `FlopCountAnalysis`
- [ ] `artifacts/census_stage2/runs/<run_id>/gate_report.json` 的 `overall_pass` 为 `true`，A1/A2/A4/A5 逐项 `pass`
- [ ] `artifacts/census_stage2/SUMMARY.md` 有 1 行 run 记录（含 `AUC-Test-Education`、A/B 逐项、`stage1_id`）
- [ ] 没有 `git add -f` 的痕迹；push 与合并留给用户批准

## 附：与 spec 的覆盖对照（自查用）

- **§2.1** 数据与任务（train.gz 全量 / test.gz 50/50 / `new_task='education'`）→ T5 `build_census_loaders`、T7；**§2.2** 数据约定（vocab copy+pop / `input_size=123` / 论文超参 / FLOPs 不进训练循环）→ T2 常量 + P3、T5 `build_mptrec`、T1 S2；**§2.3** 排除 AliCCP / ByteRec → T1 S2（`git diff master` 断言）
- **§4.2** Stage-1 产物 + `stage1_id` + `meta.json` + 只读 → T3（`stage1_id`/`save_stage1`/`load_stage1` + P6/P7）、T5；**§4.3** 只经 `--stage1-dir` → T6（required 参数）+ S5；**§4.4** 真冻结三件套（外部遍历，不改 `model.py`）→ T3 `freeze_backbone`、T6（`no_grad` 抽取 + 优化器只含 NewTask 参数）；**§4.5** 冻结机器验证（A1 + `.grad` 全 None）→ T3 `backbone_sha256`/`assert_no_grads`、T6 `report["A1"]`；**§4.6** 同一划分 / eval 语义 / val 只选点 / test 一次 / `newtask.pt` / env_ids 读取 → T6 Step 3 + S4
- **§5.2** 三个独立种子 → T2 常量 + P3、T5/T6 CLI；**§5.3** 指纹落盘 / 独立 Generator / 不消耗全局 RNG / 禁止 `random_state=model_seed` → T2 `make_env_ids` + P2/P4、T1 S2
- **§6.1** 首轮 smoke 配置（单 seed 单次）→ T2 P3、T5/T6 CLI 默认值、T7
- **§7.1** AUC 与 `metrics.json` 字段 → T6（`stage1`/`stage2`/`mechanism` 三段）、T4 `auc`；**§7.2** M1–M6 → T4（M1 `env_accuracy`/M3 `GateStats`/M4 `RepStats`）、T5 钩子（M1 每 epoch、M2 cluster）、T3+T6（M5 `backbone_sha256`、M6 指纹）；**§7.3** 报告纪律 → T6（test 只评一次）、T3 `append_summary_row` + P8
- **§8** A1/A2/A4/A5 硬门禁 + B1–B4 + A3 按需 → T4 `judge` + M5/M6/M7、T6（`A3.status = on_demand`）、T7 Step 5
- **§9.1** 目录结构与 `run_id` → T3 `make_run_id` + P8、T6；**§9.2** `.gitignore` 与 SUMMARY 入库 → T1 Step 3 + S1、T6、T7 Step 7；**§9.3** 保留与容量 → T7 Step 6
- **§10** 不含任何模型改进（`model.py` 零改动）→ T1 S2、T3 外部遍历冻结（不新增 `freeze_params()`）；**§11** 五项最小实现面 → 划分（T2）、Stage-1 产物（T3/T5）、冻结（T3/T6）、指标与门禁 + SUMMARY（T4/T6）、`.gitignore`（T1）
