"""预注册不变量测试：CensusIncome 跨数据集固定衰减迁移臂（design doc §3；先写测试后写实现）。

参考实现从 git 基点 87afe03:multitaskrec/model.py 动态加载（= 全部对照 run 使用的模型代码）。
CPU 极小夹具不构成任何性能证据，只验证语义与接线。
"""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from census_benchmark import metrics
from census_benchmark import protocol as P
import run_census_benchmark
from multitaskrec.model import MPTRec, NewTask

REPO = Path(__file__).resolve().parents[2]
BASE_REV = "87afe03"
DESIGN_DOC = REPO / "docs" / "superpowers" / "specs" / "2026-10-03-census-stage2-attenuation-transfer-design.md"
NULL_HEAD_BLOB_SHA256 = "836db5327b550cdcd07dee004c1c30e75010df43b38d5f950a41f1977539786c"

TINY = dict(input_size=8, rep_dim=4, tower_dnn_hidden_units=(4, 2), reg_dnn=3e-5)


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def _load_module_from_git(rev, path, name):
    blob = subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=REPO)
    with tempfile.TemporaryDirectory() as td:
        file = Path(td) / f"{name}.py"
        file.write_bytes(blob)
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, blob


def reference_newtask_class():
    module, _ = _load_module_from_git(BASE_REV, "multitaskrec/model.py", "base_model_ref")
    return module.NewTask


def _make_inputs(batch=16, rep_dim=4, input_size=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    dnn_input = torch.randn(batch, input_size, generator=gen)
    gen_rep = torch.randn(batch, rep_dim, generator=gen)
    spec_reps = [torch.randn(batch, rep_dim, generator=gen) for _ in range(2)]
    env_embs = [torch.randn(rep_dim, generator=gen) for _ in range(2)]
    return dnn_input, gen_rep, spec_reps, env_embs


# ---------------- I1 / I2：默认臂逐位一致 + 参数集合与 RNG ----------------

class TestDefaultArmBitIdenticalToBase(unittest.TestCase):
    def test_default_forward_backward_bit_identical(self):
        RefNewTask = reference_newtask_class()
        torch.manual_seed(0)
        ref = RefNewTask(**TINY)
        torch.manual_seed(0)
        new = NewTask(**TINY)
        self.assertEqual(sorted(ref.state_dict()), sorted(new.state_dict()))
        for key, value in ref.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        x = _make_inputs(seed=1)
        out_ref, out_new = ref(*x), new(*x)
        self.assertTrue(torch.equal(out_ref, out_new))                     # 前向逐位一致
        out_ref.sum().backward()
        out_new.sum().backward()
        for (n1, p1), (n2, p2) in zip(ref.named_parameters(), new.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1.grad, p2.grad), f"梯度不一致: {n1}")


class TestParamSetsAndRng(unittest.TestCase):
    def test_attenuation_is_not_a_param_or_buffer(self):
        head = NewTask(**TINY, spec_attenuation=metrics.SPEC_ATTENUATION_COEF)
        self.assertNotIn("spec_attenuation", set(head.state_dict()))
        self.assertNotIn("spec_attenuation", dict(head.named_parameters()))
        self.assertNotIn("spec_attenuation", dict(head.named_buffers()))

    def test_default_state_dict_loads_strict_from_base(self):
        RefNewTask = reference_newtask_class()
        torch.manual_seed(3)
        ref = RefNewTask(**TINY)
        NewTask(**TINY).load_state_dict(ref.state_dict(), strict=True)

    def test_construction_consumes_no_extra_rng(self):
        RefNewTask = reference_newtask_class()
        torch.manual_seed(7)
        ref = RefNewTask(**TINY)
        tail_ref = torch.rand(3)
        torch.manual_seed(7)
        default = NewTask(**TINY)
        tail_default = torch.rand(3)
        torch.manual_seed(7)
        attn = NewTask(**TINY, spec_attenuation=metrics.SPEC_ATTENUATION_COEF)
        tail_attn = torch.rand(3)
        self.assertTrue(torch.equal(tail_ref, tail_default))
        self.assertTrue(torch.equal(tail_ref, tail_attn))
        for key, value in ref.state_dict().items():
            self.assertTrue(torch.equal(value, attn.state_dict()[key]), key)


# ---------------- I3：开启臂前向 = 预注册公式 ----------------

class TestEnabledForwardFormula(unittest.TestCase):
    def test_enabled_forward_matches_preregistered_formula(self):
        c = metrics.SPEC_ATTENUATION_COEF
        torch.manual_seed(11)
        head = NewTask(**TINY, spec_attenuation=c)
        dnn_input, gen_rep, spec_reps, env_embs = _make_inputs(seed=12)
        got = head(dnn_input, gen_rep, spec_reps, env_embs)
        exist_env_embs = torch.stack(env_embs, dim=1)
        new_env_emb = head.env_embedding_network(head.new_env_idx).squeeze(0)
        W = F.softmax(torch.mm(head.projection_network(dnn_input), exist_env_embs) / head.temperature, dim=-1)
        gate_out = head.gate_network(dnn_input).unsqueeze(dim=2)
        spec = torch.matmul(torch.stack(spec_reps, dim=2), W.unsqueeze(2)).squeeze() * c   # 固定常数乘
        env_aware = spec * new_env_emb
        fused = torch.matmul(torch.stack([env_aware, gen_rep], dim=2), gate_out).squeeze()
        want = head.tower_network(fused).squeeze()
        self.assertTrue(torch.equal(got, want))


# ---------------- I4：零新增可训练参数 + 梯度/L2 回归 ----------------

class TestZeroExtraParamsAndGrads(unittest.TestCase):
    def test_enabled_arm_has_zero_extra_trainable_params(self):
        torch.manual_seed(5)
        default = NewTask(**TINY)
        torch.manual_seed(5)
        attn = NewTask(**TINY, spec_attenuation=metrics.SPEC_ATTENUATION_COEF)
        self.assertEqual(sum(p.numel() for p in default.parameters()), sum(p.numel() for p in attn.parameters()))
        self.assertEqual(sorted(n for n, _ in default.named_parameters()), sorted(n for n, _ in attn.named_parameters()))
        self.assertTrue(all(p.requires_grad for p in attn.parameters()))

    def test_all_params_receive_gradient_and_l2_reg_matches_default(self):
        torch.manual_seed(6)
        default = NewTask(**TINY)
        torch.manual_seed(6)
        attn = NewTask(**TINY, spec_attenuation=metrics.SPEC_ATTENUATION_COEF)
        attn.load_state_dict(default.state_dict(), strict=True)
        attn(*_make_inputs(seed=13)).sum().backward()
        for name, param in attn.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.any(param.grad != 0), f"梯度全零: {name}")
        self.assertTrue(torch.equal(default.get_l2_reg(), attn.get_l2_reg()))   # loss 形式不变


# ---------------- I5：常量与文档钉死 ----------------

class TestConstantsPinned(unittest.TestCase):
    def test_coefficient_literals(self):
        self.assertEqual(metrics.ALICCP_NULL_MEAN, 0.3027766269669533)
        self.assertEqual(metrics.SPEC_ATTENUATION_COEF, 1.0 - 0.3027766269669533)
        self.assertEqual(metrics.SPEC_ATTENUATION_COEF, 0.6972233730330467)
        self.assertEqual(metrics.SPEC_ATTENUATION_COEF_F32,
                         float(torch.tensor(metrics.SPEC_ATTENUATION_COEF, dtype=torch.float32)))
        self.assertEqual(metrics.SPEC_ATTENUATION_COEF_F32, 0.6972233653068542)

    def test_baseline_null_threshold_literals(self):
        self.assertEqual(metrics.TRANSFER_BASELINE_TEST_AUC, 0.8500685307175756)
        self.assertEqual(metrics.TRANSFER_BASELINE_VAL_BEST, 0.8527881905614896)
        self.assertEqual(metrics.TRANSFER_TEST_THRESHOLD, 0.8521)
        self.assertEqual(metrics.TRANSFER_NULL_TEST_AUC, 0.8513648272440768)
        self.assertEqual(metrics.TRANSFER_NULL_VAL_BEST, 0.853020431042571)
        self.assertEqual(metrics.TRANSFER_NULL_DELTA_TEST, 0.001296296526501206)
        self.assertEqual(metrics.TRANSFER_NULL_DELTA_TEST,
                         metrics.TRANSFER_NULL_TEST_AUC - metrics.TRANSFER_BASELINE_TEST_AUC)
        self.assertEqual(metrics.ALICCP_REPRO_RATIO, 0.917640706647481)

    def test_doc_preregisters_same_numbers(self):
        doc = DESIGN_DOC.read_text(encoding="utf-8")
        for literal in ("0.6972233730330467", "0.6972233653068542", "0.8500685307175756",
                        "0.8527881905614896", "0.8521", "0.8513648272440768", "0.853020431042571",
                        "0.001296296526501206", "TRANSFER_SUPPORTED", "TRANSFER_NOT_SUPPORTED",
                        "s1-096f8f16-m1685480945-e2-cb2094b3"):
            self.assertIn(literal, doc)

    def test_invalid_attenuation_rejected(self):
        for bad in (0.0, -0.1, 1.0000001):
            with self.assertRaises(ValueError, msg=f"应拒绝: {bad}"):
                NewTask(**TINY, spec_attenuation=bad)
        NewTask(**TINY, spec_attenuation=1.0)                              # 上界含等号


# ---------------- I6：诊断纯函数 ----------------

class TestDiagnostics(unittest.TestCase):
    def test_pred_dispersion_matches_numpy(self):
        values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9], dtype=torch.float32)
        out = metrics.pred_dispersion(values)
        arr = values.double().numpy()
        self.assertAlmostEqual(out["mean"], float(arr.mean()), places=12)
        self.assertAlmostEqual(out["std"], float(arr.std(ddof=0)), places=12)
        self.assertAlmostEqual(out["var"], float(arr.var(ddof=0)), places=12)
        self.assertAlmostEqual(out["q50"], float(np.quantile(arr, 0.5)), places=9)
        self.assertAlmostEqual(out["min"], 0.1, places=6)
        self.assertAlmostEqual(out["max"], 0.9, places=6)
        self.assertEqual(out["n"], 5)

    def test_correlations_match_scipy(self):
        from scipy import stats
        rng = np.random.default_rng(0)
        a = rng.normal(size=200)
        b = a * 0.5 + rng.normal(size=200) * 0.5
        self.assertAlmostEqual(metrics.pearson_corr(a, b), float(stats.pearsonr(a, b)[0]), places=10)
        c = np.round(a, 1)                                                 # 制造并列，测平均秩
        self.assertAlmostEqual(metrics.spearman_corr(c, b), float(stats.spearmanr(c, b)[0]), places=10)
        self.assertTrue(np.isnan(metrics.pearson_corr(torch.ones(5), torch.ones(5))))  # 常量输入 → 无定义，返回 nan

    def test_logit_clip(self):
        p = torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64)
        out = metrics.logit_clip(p, eps=1e-6)
        self.assertTrue(torch.isfinite(out).all())
        self.assertAlmostEqual(float(out[2]), 0.0, places=12)

    def test_paired_delta_stats_exact(self):
        a = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
        b = torch.tensor([0.0, 2.0, 3.0], dtype=torch.float64)
        out = metrics.paired_delta_stats(a, b)
        self.assertEqual(out["mean"], 2.0 / 3.0)
        self.assertEqual(out["min"], 0.0)
        self.assertEqual(out["max"], 1.0)
        self.assertAlmostEqual(out["frac_pos"], 2.0 / 3.0, places=12)


# ---------------- I7：判定函数边界 ----------------

class TestTransferVerdict(unittest.TestCase):
    def test_boundaries(self):
        ok = dict(test_auc=0.8521, val_best=0.8527881905614897, construction_ok=True, a_gates_ok=True)
        self.assertEqual(metrics.transfer_arm_verdict(**ok)["verdict"], "TRANSFER_SUPPORTED")   # test 闭边界
        for broken, criterion in (({"test_auc": 0.8520999999999}, "test_auc_ge_threshold"),
                                  ({"val_best": 0.8527881905614896}, "val_direction_improves"),
                                  ({"construction_ok": False}, "construction_audit_ok"),
                                  ({"a_gates_ok": False}, "a_gates_ok")):
            out = metrics.transfer_arm_verdict(**{**ok, **broken})
            self.assertEqual(out["verdict"], "TRANSFER_NOT_SUPPORTED", broken)
            self.assertFalse(out["criteria"][criterion], broken)

    def test_evidence_and_reference(self):
        out = metrics.transfer_arm_verdict(test_auc=0.851, val_best=0.853, construction_ok=True, a_gates_ok=True)
        ev = out["evidence"]
        self.assertEqual(ev["delta_test_vs_baseline"], 0.851 - metrics.TRANSFER_BASELINE_TEST_AUC)
        self.assertEqual(ev["reproduction_ratio_vs_null"],
                         (0.851 - metrics.TRANSFER_BASELINE_TEST_AUC) / metrics.TRANSFER_NULL_DELTA_TEST)
        self.assertEqual(ev["aliccp_reference"]["reproduction_ratio"], 0.917640706647481)
        self.assertEqual(ev["aliccp_reference"]["delta_test"], 0.012706144575588052)
        self.assertFalse(out["criteria"]["test_auc_ge_threshold"])


# ---------------- I9：runner 两臂接线（CPU 极小夹具端到端） ----------------

class TinyCensus(Dataset):
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


def tiny_model():
    return MPTRec(num_tasks=2, feature_vocabulary={"a": 3, "b": 2}, embedding_size=4,
                  input_size=8, expert_dnn_hidden_units=(8, 4),
                  tower_dnn_hidden_units=(4, 2), device=torch.device("cpu"))


def tiny_inputs(n_train=64, n_val=16, n_test=16, batch_size=16):
    train_ds, test_ds = TinyCensus(n_train, 1), TinyCensus(n_val + n_test, 2)
    val_idx, test_idx = P.make_split(len(test_ds), P.SPLIT_SEED)
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size),
               "val": DataLoader(Subset(test_ds, val_idx.tolist()), batch_size=batch_size),
               "test": DataLoader(Subset(test_ds, test_idx.tolist()), batch_size=batch_size)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=len(train_ds)), (val_idx, test_idx)


class TestRunnerAttenuationArm(unittest.TestCase):
    def test_runner_attenuation_arm_records_probe_and_verdict(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            loaders, stats, indices = tiny_inputs()
            stage1 = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_model(),
                                                     loaders=loaders, stats=stats, indices=indices)
            out = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=3, device=device,
                                                  model=tiny_model(), loaders=loaders, stats=stats,
                                                  indices=indices, input_size=8, rep_dim=4,
                                                  spec_attenuation=metrics.SPEC_ATTENUATION_COEF)
            self.assertTrue(out["run_id"].endswith("-sattn"))
            run_path = Path(out["run_dir"])
            metrics_json = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics_json["spec_attenuation"], metrics.SPEC_ATTENUATION_COEF)
            probe = metrics_json["probe"]
            self.assertEqual(len(probe["head_gate_mean"]), 2)
            self.assertEqual(len(probe["source_gate_mean"]), 2)
            self.assertEqual(set(probe["norm_mean"]),
                             {"spec_mix_pre", "spec_mix_post", "env_aware", "fused", "gen"})
            self.assertEqual(metrics_json["trainable_params"], metrics_json["reference_trainable_params"])
            self.assertEqual(metrics_json["extra_trainable_params"], 0)
            self.assertTrue(metrics_json["construction"]["construction_ok"])
            self.assertIn("transfer_arm", metrics_json)
            self.assertEqual(metrics_json["transfer_arm"]["verdict"],
                             "TRANSFER_NOT_SUPPORTED")                     # 小夹具远不达阈值，判定必须如实
            report = json.loads((run_path / "gate_report.json").read_text(encoding="utf-8"))
            self.assertIn("transfer_arm", report)
            self.assertTrue(report["A1"]["pass"])
            config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["spec_attenuation"], metrics.SPEC_ATTENUATION_COEF)

    def test_runner_default_arm_has_no_arm_keys(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            loaders, stats, indices = tiny_inputs()
            stage1 = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_model(),
                                                     loaders=loaders, stats=stats, indices=indices)
            out = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=2, device=device,
                                                  model=tiny_model(), loaders=loaders, stats=stats,
                                                  indices=indices, input_size=8, rep_dim=4)
            self.assertFalse(out["run_id"].endswith("-sattn"))
            run_path = Path(out["run_dir"])
            metrics_json = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics_json["spec_attenuation"], 1.0)        # 仅新增标记，语义不变
            for key in ("probe", "transfer_arm", "trainable_params", "extra_trainable_params", "construction"):
                self.assertNotIn(key, metrics_json)
            report = json.loads((run_path / "gate_report.json").read_text(encoding="utf-8"))
            self.assertNotIn("transfer_arm", report)


# ---------------- I8：对照管线 ----------------

class TestComparePipeline(unittest.TestCase):
    def _tiny_backbone_and_loaders(self, root, device):
        loaders, stats, indices = tiny_inputs()
        stage1 = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_model(),
                                                 loaders=loaders, stats=stats, indices=indices)
        checkpoint = P.load_stage1(root, stage1["stage1_id"])
        backbone = tiny_model()
        backbone.load_state_dict(checkpoint["backbone_state"])
        P.freeze_backbone(backbone)
        return stage1, loaders, stats, indices, backbone

    def test_null_head_blob_pinned(self):
        from census_benchmark import compare
        head_cls = compare.load_null_newtask_class(REPO)
        head = head_cls(**TINY, use_null_expert=True)
        self.assertTrue(hasattr(head, "null_key"))
        with self.assertRaises(RuntimeError):                              # sha 钉死：错误的 rev 必须拒绝
            compare.load_null_newtask_class(REPO, rev=BASE_REV)

    def test_predict_arm_auc_matches_evaluate_newtask(self):
        from census_benchmark import compare
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices, backbone = self._tiny_backbone_and_loaders(root, device)
            torch.manual_seed(P.MODEL_SEED)
            head = NewTask(**TINY)
            preds, ys = compare.predict_arm(head, backbone, loaders["test"], device)
            self.assertEqual(metrics.auc(ys, preds),
                             metrics.evaluate_newtask(head, backbone, loaders["test"], device)["auc"])

    def test_compare_e2e_cpu_tiny(self):
        from census_benchmark import compare
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices, backbone = self._tiny_backbone_and_loaders(root, device)
            baseline = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=2,
                                                       device=device, model=tiny_model(), loaders=loaders,
                                                       stats=stats, indices=indices, input_size=8, rep_dim=4)
            control = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=2,
                                                      device=device, model=tiny_model(), loaders=loaders,
                                                      stats=stats, indices=indices, input_size=8, rep_dim=4,
                                                      spec_attenuation=metrics.SPEC_ATTENUATION_COEF)
            null_run_id = self._fabricate_null_run(root, stage1, loaders, device, compare)
            result = compare.run_compare(root, stage1_dir=Path(stage1["dir"]),
                                         baseline_run=baseline["run_id"], null_run=null_run_id,
                                         control_run=control["run_id"], device=device, model=tiny_model(),
                                         loaders=loaders, stats=stats, indices=indices)
            self.assertTrue(result["integrity"]["all_pass"])
            for arm in ("baseline", "null", "control"):
                self.assertIn(arm, result["test"]["dispersion"])
                self.assertIn(arm, result["gates"]["head_gate_mean"])
            self.assertIn("control_minus_baseline", result["test"]["paired_delta"])
            self.assertIn("control_baseline", result["test"]["corr"])
            self.assertIn("control_null", result["test"]["corr"])
            self.assertEqual(result["heads"]["null"]["trainable_params"],
                             result["heads"]["baseline"]["trainable_params"] + 4)   # null_key = rep_dim = 4
            written = Path(control["run_dir"]) / "three_way_compare.json"
            self.assertTrue(written.exists())
            self.assertEqual(json.loads(written.read_text(encoding="utf-8"))["integrity"]["all_pass"], True)

    @staticmethod
    def _fabricate_null_run(root, stage1, loaders, device, compare):
        """参照真实 null 臂 run 的产物结构，用动态载入的 null 头类生成一个极小 null run。"""
        head_cls = compare.load_null_newtask_class(REPO)
        torch.manual_seed(P.MODEL_SEED)
        head = head_cls(input_size=8, rep_dim=4, tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=3e-5,
                        device=device, use_null_expert=True)
        checkpoint = P.load_stage1(root, stage1["stage1_id"])
        backbone = tiny_model()
        backbone.load_state_dict(checkpoint["backbone_state"])
        P.freeze_backbone(backbone)
        test_auc = metrics.evaluate_newtask(head, backbone, loaders["test"], device)["auc"]
        fp = json.loads((Path(stage1["dir"]) / "meta.json").read_text(encoding="utf-8"))
        run_id = "29990101-0000-s20260929-m1685480945-short-0000000-nullx"
        run_dir = root / "runs" / run_id
        run_dir.mkdir(parents=True)
        torch.save(head.state_dict(), run_dir / "newtask.pt")
        (run_dir / "config.json").write_text(json.dumps(
            {"run_id": run_id, "stage1_id": stage1["stage1_id"], "commit": "0000000", "frozen": True}),
            encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(
            {"run_id": run_id, "stage1_id": stage1["stage1_id"], "commit": "0000000",
             "split_sha256": {"val": fp["val_sha256"], "test": fp["test_sha256"],
                              "fingerprint": fp["split_fingerprint_sha256"]},
             "stage2": {"test_auc": test_auc}}), encoding="utf-8")
        return run_id


# ---------------- I10：静态守卫 ----------------

class TestStaticGuards(unittest.TestCase):
    def test_only_whitelisted_files_changed_since_base(self):
        diff = _git("diff", "--name-only", BASE_REV).stdout.split()
        allowed = {"multitaskrec/model.py", "census_benchmark/metrics.py", "census_benchmark/compare.py",
                   "census_benchmark/tests/test_spec_attenuation.py", "census_benchmark/tests/test_smoke.py",
                   "run_census_benchmark.py",
                   "docs/superpowers/specs/2026-10-03-census-stage2-attenuation-transfer-design.md",
                   "artifacts/census_stage2/SUMMARY.md"}
        self.assertTrue(set(diff) <= allowed, f"改动超出白名单: {set(diff) - allowed}")

    def test_protocol_and_entry_files_zero_diff(self):
        self.assertEqual(_git("diff", "--name-only", BASE_REV, "--", "census_benchmark/protocol.py").stdout.strip(), "")
        self.assertEqual(_git("diff", "--name-only", BASE_REV, "--", "config.py", "CensusIncome_MPTRec.py",
                              "CensusIncome_NewTask.py", "multitaskrec/train.py", "multitaskrec/dataset.py"
                              ).stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
