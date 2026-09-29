"""Null Expert：阶段 2 源任务路由追加"零候选"。

对应实验说明 `docs/superpowers/specs/2026-09-29-stage2-null-expert-design.md`。

- 默认 `use_null_expert=False`：NewTask 必须与 `master` **逐位一致**（前向 + 反向 + 参数集合）。
- `use_null_expert=True`：在既有 K 个 spec_rep 候选之后追加一个**值恒为零**的候选，
  router key 追加一个可学习 `null_key`（rep_dim），与既有 H_out 点积后同温度 softmax 得 K+1 权重；
  其余 new_env_emb / gate / tower / loss 完全不变。
"""
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from census_benchmark import metrics
from census_benchmark import protocol as P
from multitaskrec.model import MPTRec, NewTask
import run_census_benchmark

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "superpowers" / "specs" / "2026-09-29-stage2-null-expert-design.md"
BATCH, INPUT_SIZE, REP_DIM, NUM_SRC = 6, 8, 4, 2


def _master_newtask_class():
    """从 git `master` 动态加载 NewTask 作为逐位对照的参考实现（不落地任何文件）。

    显式 `encoding="utf-8"`：Windows 下 `text=True` 默认走 locale 编码（GBK），
    model.py 含中文注释 → UnicodeDecodeError 会把源码变成 None；errors="strict" 保证解码问题不被静默吞掉。
    """
    src = subprocess.run(["git", "show", "master:multitaskrec/model.py"], cwd=REPO, check=True,
                         capture_output=True, text=True, encoding="utf-8", errors="strict").stdout
    module = types.ModuleType("model_master")
    exec(compile(src, "master:multitaskrec/model.py", "exec"), module.__dict__)
    return module.NewTask


def _ours(use_null_expert=False):
    return NewTask(input_size=INPUT_SIZE, rep_dim=REP_DIM, tower_dnn_hidden_units=(4, 2),
                   reg_dnn=P.REG_DNN, device=torch.device("cpu"), use_null_expert=use_null_expert)


def _master():
    return _master_newtask_class()(input_size=INPUT_SIZE, rep_dim=REP_DIM, tower_dnn_hidden_units=(4, 2),
                                   reg_dnn=P.REG_DNN, device=torch.device("cpu"))


def _inputs(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(BATCH, INPUT_SIZE, generator=gen), torch.randn(BATCH, REP_DIM, generator=gen),
            [torch.randn(BATCH, REP_DIM, generator=gen) for _ in range(NUM_SRC)],
            [torch.randn(REP_DIM, generator=gen) for _ in range(NUM_SRC)],
            torch.randint(0, 2, (BATCH,), generator=gen).float())


def _forward_backward(module, inputs):
    dnn_input, gen_rep, spec_reps, env_embs, target = inputs
    module.zero_grad(set_to_none=True)
    pred = module(dnn_input, gen_rep, spec_reps, env_embs)
    F.binary_cross_entropy(pred, target).backward()
    return pred


class TestDefaultArmBitIdenticalToMaster(unittest.TestCase):
    """默认臂（use_null_expert=False）必须是 master 的逐位复制。"""

    def test_default_arm_has_no_null_key(self):
        ours = _ours()
        self.assertNotIn("null_key", dict(ours.named_parameters()))
        self.assertEqual(set(ours.state_dict()), set(_master().state_dict()))
        ours.load_state_dict(_master().state_dict())                      # strict=True：参数集合必须一致

    def test_default_forward_bit_identical(self):
        master, ours, inputs = _master(), _ours(), _inputs(1)
        ours.load_state_dict(master.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(ours(*inputs[:4]), master(*inputs[:4])))

    def test_default_backward_bit_identical(self):
        master, ours, inputs = _master(), _ours(), _inputs(2)
        ours.load_state_dict(master.state_dict())
        pred_master, pred_ours = _forward_backward(master, inputs), _forward_backward(ours, inputs)
        self.assertTrue(torch.equal(pred_master, pred_ours))
        grads_ours = dict(ours.named_parameters())
        for name, param in master.named_parameters():
            self.assertIsNotNone(param.grad, f"master 该参数无梯度，反向对照无效: {name}")
            self.assertTrue(torch.equal(param.grad, grads_ours[name].grad), f"梯度非逐位一致: {name}")


class TestNullCandidateMechanics(unittest.TestCase):
    """开启 Null Expert 后的候选/权重语义。"""

    def test_routing_weights_shape_is_k_then_k_plus_one(self):
        off, on, inputs = _ours(), _ours(use_null_expert=True), _inputs(3)
        self.assertEqual(tuple(off.routing_weights(inputs[0], inputs[3]).shape), (BATCH, NUM_SRC))
        self.assertEqual(tuple(on.routing_weights(inputs[0], inputs[3]).shape), (BATCH, NUM_SRC + 1))

    def test_routing_weights_are_probabilities(self):
        on, inputs = _ours(use_null_expert=True), _inputs(4)
        weights = on.routing_weights(inputs[0], inputs[3])
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(BATCH), atol=1e-6))
        self.assertTrue(bool((weights >= 0).all()))

    def test_null_value_is_exactly_zero_and_other_columns_untouched(self):
        on, inputs = _ours(use_null_expert=True), _inputs(5)
        values = on.routing_values(inputs[2])
        self.assertEqual(tuple(values.shape), (BATCH, REP_DIM, NUM_SRC + 1))
        self.assertTrue(torch.equal(values[:, :, -1], torch.zeros(BATCH, REP_DIM)))   # 精确零，非近似零
        self.assertTrue(torch.equal(values[:, :, :NUM_SRC], torch.stack(inputs[2], dim=2)))
        off = _ours()
        self.assertTrue(torch.equal(off.routing_values(inputs[2]), torch.stack(inputs[2], dim=2)))

    def test_null_key_is_the_only_new_parameter(self):
        ours = _ours(use_null_expert=True)
        extra = set(ours.state_dict()) - set(_master().state_dict())
        self.assertEqual(extra, {"null_key"})
        self.assertEqual(set(_master().state_dict()) - set(ours.state_dict()), set())
        self.assertEqual(tuple(ours.null_key.shape), (REP_DIM,))
        self.assertTrue(ours.null_key.requires_grad)

    def test_null_key_receives_gradient(self):
        ours, inputs = _ours(use_null_expert=True), _inputs(6)
        _forward_backward(ours, inputs)
        self.assertIsNotNone(ours.null_key.grad)
        self.assertGreater(float(ours.null_key.grad.abs().sum()), 0.0)
        for name, param in ours.named_parameters():
            if name != "null_key":
                self.assertIsNotNone(param.grad, f"开启 Null Expert 后该参数丢失梯度: {name}")

    def test_enabled_forward_matches_reference_formula(self):
        """K+1 前向必须等于实验说明写死的公式（同温度 softmax，最后一列为零值）。"""
        ours, inputs = _ours(use_null_expert=True), _inputs(7)
        dnn_input, gen_rep, spec_reps, env_embs, _ = inputs
        keys = torch.cat([torch.stack(env_embs, dim=1), ours.null_key.unsqueeze(1)], dim=1)
        logits = torch.mm(ours.projection_network(dnn_input), keys) / ours.temperature
        weights = F.softmax(logits, dim=-1).unsqueeze(2)
        values = torch.cat([torch.stack(spec_reps, dim=2), torch.zeros(BATCH, REP_DIM, 1)], dim=2)
        new_spec_rep = torch.matmul(values, weights).squeeze()
        env_aware_rep = new_spec_rep * ours.env_embedding_network(ours.new_env_idx).squeeze(0)
        fused_rep = torch.matmul(torch.stack([env_aware_rep, gen_rep], dim=2),
                                 ours.gate_network(dnn_input).unsqueeze(dim=2)).squeeze()
        expected = ours.tower_network(fused_rep).squeeze()
        with torch.no_grad():
            self.assertTrue(torch.equal(ours(*inputs[:4]), expected))


class TestNullRouteStats(unittest.TestCase):
    """M7：null 候选的样本均值权重与 top1 占比。"""

    # dtype=float64：下面的 places=9 断言要求字面量在累加精度下可精确表示。
    # float32 下 0.2/0.7 本身就有 ~3e-9 / ~1.2e-8 的表示误差，null_mean 会偏离
    # (0.2+0.7+0.2)/3 约 2e-9，超出 places=9 的 5e-10 容差（与实现无关）。
    WEIGHTS = torch.tensor([[0.5, 0.3, 0.2],       # argmax = 0
                            [0.1, 0.2, 0.7],       # argmax = 2 = null → top1
                            [0.4, 0.4, 0.2]],      # 平局取首个 → null 不算 top1
                           dtype=torch.float64)

    def test_null_mean_and_top1_rate(self):
        stats = metrics.NullRouteStats()
        stats.update(self.WEIGHTS)
        out = stats.result()
        self.assertAlmostEqual(out["null_mean"], (0.2 + 0.7 + 0.2) / 3, places=9)
        self.assertAlmostEqual(out["null_top1_rate"], 1 / 3, places=9)

    def test_streaming_updates_equal_single_update(self):
        single, streamed = metrics.NullRouteStats(), metrics.NullRouteStats()
        single.update(self.WEIGHTS)
        streamed.update(self.WEIGHTS[:2])
        streamed.update(self.WEIGHTS[2:])
        self.assertEqual(single.result(), streamed.result())

    def test_single_row_batch(self):
        stats = metrics.NullRouteStats()
        stats.update(torch.tensor([[0.1, 0.2, 0.7]]))
        self.assertAlmostEqual(stats.result()["null_top1_rate"], 1.0, places=9)


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 才能复现累加器 device mismatch")
class TestNullRouteStatsCuda(unittest.TestCase):
    def test_cuda_weights_accumulate_on_cpu(self):
        stats = metrics.NullRouteStats()
        stats.update(TestNullRouteStats.WEIGHTS.cuda())
        out = stats.result()
        self.assertAlmostEqual(out["null_mean"], (0.2 + 0.7 + 0.2) / 3, places=9)
        self.assertAlmostEqual(out["null_top1_rate"], 1 / 3, places=9)


class TestNullArmVerdict(unittest.TestCase):
    """预注册接受标准：同 checkpoint 基线 0.8500685307；AUC ≥ 0.8521 且 top1 ∈ [0.05, 0.95]。"""

    def test_constants_pin_baseline_and_thresholds(self):
        self.assertEqual(metrics.NULL_ARM_BASELINE_TEST_AUC, 0.8500685307)
        self.assertEqual(metrics.NULL_ARM_AUC_MIN, 0.8521)
        self.assertEqual((metrics.NULL_ARM_TOP1_MIN, metrics.NULL_ARM_TOP1_MAX), (0.05, 0.95))

    def test_boundaries_inclusive(self):
        self.assertTrue(metrics.null_arm_verdict(0.8521, 0.05)["pass"])
        self.assertTrue(metrics.null_arm_verdict(0.8521, 0.95)["pass"])
        self.assertTrue(metrics.null_arm_verdict(0.99, 0.5)["pass"])

    def test_below_threshold_stops(self):
        self.assertFalse(metrics.null_arm_verdict(0.8520999, 0.5)["pass"])      # AUC 未达标
        self.assertFalse(metrics.null_arm_verdict(0.99, 0.049)["pass"])         # null 从不被选
        self.assertFalse(metrics.null_arm_verdict(0.99, 0.951)["pass"])         # null 总被选

    def test_verdict_records_evidence(self):
        verdict = metrics.null_arm_verdict(0.86, 0.3)
        self.assertEqual(verdict["test_auc"], 0.86)
        self.assertEqual(verdict["null_top1_rate"], 0.3)
        self.assertEqual(verdict["baseline_test_auc"], 0.8500685307)
        self.assertEqual(verdict["top1_range"], [0.05, 0.95])
        self.assertEqual(verdict["checks"], {"auc": True, "top1": True})

    def test_doc_preregisters_same_numbers(self):
        text = DOC.read_text(encoding="utf-8")
        for token in ("0.8500685307", "0.8521", "[0.05, 0.95]"):
            self.assertIn(token, text, f"实验说明未写死接受标准: {token}")


class TinyCensus(Dataset):
    """极小 CensusIncome 形状：(income, marital, new_task, features)。"""

    def __init__(self, n, seed):
        gen = torch.Generator().manual_seed(seed)
        self.features = {"a": torch.randint(0, 3, (n,), generator=gen),
                         "b": torch.randint(0, 2, (n,), generator=gen)}
        self.labels = [torch.randint(0, 2, (n,), generator=gen).float() for _ in range(3)]

    def __len__(self):
        return int(self.features["a"].shape[0])

    def __getitem__(self, i):
        return (self.labels[0][i], self.labels[1][i], self.labels[2][i],
                {"a": self.features["a"][i], "b": self.features["b"][i]})


def _tiny_model():
    return MPTRec(num_tasks=2, feature_vocabulary={"a": 3, "b": 2}, embedding_size=4, input_size=8,
                  expert_dnn_hidden_units=(8, 4), tower_dnn_hidden_units=(4, 2), device=torch.device("cpu"))


def _tiny_inputs(n_train=64, n_val=16, n_test=16, batch_size=16):
    train_ds, test_ds = TinyCensus(n_train, 1), TinyCensus(n_val + n_test, 2)
    val_idx, test_idx = P.make_split(len(test_ds), P.SPLIT_SEED)
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size),
               "val": DataLoader(Subset(test_ds, val_idx.tolist()), batch_size=batch_size),
               "test": DataLoader(Subset(test_ds, test_idx.tolist()), batch_size=batch_size)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=len(train_ds)), (val_idx, test_idx)


class TestCliFlag(unittest.TestCase):
    def test_null_expert_flag_defaults_to_false(self):
        parser = run_census_benchmark.build_parser()
        self.assertFalse(parser.parse_args(["stage2", "--stage1-dir", "x"]).null_expert)
        self.assertTrue(parser.parse_args(["stage2", "--stage1-dir", "x", "--null-expert"]).null_expert)


class TestRunnerNullArm(unittest.TestCase):
    """runner 接线：两臂共享同一 Stage-1 产物，各自落盘 config / metrics / run_id。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root, cls.device = Path(cls._tmp.name) / "artifacts", torch.device("cpu")
        cls.loaders, cls.stats, cls.indices = _tiny_inputs()
        cls.stage1 = run_census_benchmark.run_stage1(cls.root, epochs=2, device=cls.device, model=_tiny_model(),
                                                     loaders=cls.loaders, stats=cls.stats,
                                                     indices=cls.indices, tag="short")
        cls.baseline = cls._stage2(null_expert=False)
        cls.null = cls._stage2(null_expert=True)

    @classmethod
    def _stage2(cls, null_expert):
        return run_census_benchmark.run_stage2(cls.root, stage1_dir=Path(cls.stage1["dir"]), epochs=2,
                                               device=cls.device, model=_tiny_model(), loaders=cls.loaders,
                                               stats=cls.stats, indices=cls.indices, input_size=8, rep_dim=4,
                                               null_expert=null_expert)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _json(self, out, name):
        return json.loads((Path(out["run_dir"]) / name).read_text(encoding="utf-8"))

    def test_baseline_arm_artifact_shape_unchanged(self):
        self.assertNotIn("nullx", self.baseline["run_id"])
        self.assertFalse(self._json(self.baseline, "config.json")["null_expert"])
        recorded = self._json(self.baseline, "metrics.json")
        self.assertFalse(recorded["null_expert"])
        self.assertNotIn("null_arm", recorded)
        self.assertEqual(set(recorded["mechanism"]),
                         {"gate_mean", "cos_gen_spec", "gen_std", "env_acc_stage1"})

    def test_null_arm_records_config_metrics_and_run_id(self):
        self.assertTrue(self.null["run_id"].endswith("-nullx"))
        self.assertEqual(self._json(self.null, "config.json")["null_expert"], True)
        recorded = self._json(self.null, "metrics.json")
        self.assertEqual(recorded["null_expert"], True)
        self.assertTrue(0.0 <= recorded["mechanism"]["null_mean"] <= 1.0)
        self.assertTrue(0.0 <= recorded["mechanism"]["null_top1_rate"] <= 1.0)
        self.assertEqual(recorded["null_arm"]["test_auc"], recorded["stage2"]["test_auc"])
        self.assertEqual(recorded["null_arm"]["top1_range"], [0.05, 0.95])
        self.assertIn("pass", recorded["null_arm"])

    def test_null_arm_keeps_freeze_and_gates(self):
        report = self._json(self.null, "gate_report.json")
        self.assertTrue(report["A1"]["pass"])                       # 新头未破坏阶段 2 冻结三件套
        self.assertEqual(report["A3"]["status"], "on_demand")

    def test_null_arm_uses_same_stage1_checkpoint(self):
        recorded, baseline = self._json(self.null, "metrics.json"), self._json(self.baseline, "metrics.json")
        self.assertEqual(recorded["stage1_id"], self.stage1["stage1_id"])
        self.assertEqual(recorded["backbone_sha256_before"], baseline["backbone_sha256_before"])
