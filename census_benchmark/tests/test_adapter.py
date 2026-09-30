"""T1–T19：低秩残差适配（adapter.py）与 runner 变体（adapter_runner.py）的单元测试。

用例编号用 T* 前缀，避免与门禁名（协议 A1–A5/B1–B4、实验 A6/G/R1/S1）混淆。
全部 CPU、秒级、不读真实数据集。运行方式（cwd = 仓库根）：

    .venv\\Scripts\\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests
"""
import json, subprocess, tempfile, unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from census_benchmark import adapter
from census_benchmark import protocol as P
from multitaskrec.model import MPTRec, NewTask
import run_census_benchmark
from census_benchmark.adapter_runner import run_stage2_adapter

REPO = Path(__file__).resolve().parents[2]          # 测试依赖 cwd = 仓库根
# 本实验分支的起点 = infra/fair-stage2-benchmark tip（2026-09-30）。用它做“协议文件未被改动”的
# 硬对照：固定 SHA 不依赖 ref 是否存在；若分支被 rebase，--is-ancestor 会失败并给出原因。
EXP_BASE_COMMIT = "87afe037505a29046e1a9a8ba66b950fb30fe820"
PROTECTED_PATHS = ("multitaskrec", "census_benchmark/protocol.py", "census_benchmark/metrics.py",
                   "run_census_benchmark.py", "config.py")

TINY_VOCAB, TINY_INPUT_SIZE, TINY_REP_DIM = {"a": 3, "b": 2}, 8, 8


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def tiny_mptrec():
    """dnn_input = 2 特征 × embedding_size 4 = 8；expert 末层 8 → rep_dim=8 > rank=4（真低秩）。"""
    return MPTRec(num_tasks=2, feature_vocabulary=dict(TINY_VOCAB), embedding_size=4,
                  input_size=TINY_INPUT_SIZE, expert_dnn_hidden_units=(8, 8),
                  tower_dnn_hidden_units=(4, 2), device=torch.device("cpu"))


def tiny_newtask(**kwargs):
    return adapter.LoRANewTask(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                               tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN,
                               device=torch.device("cpu"), **kwargs)


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


def tiny_inputs(n_train=64, n_val=16, n_test=16, batch_size=16):
    train_ds, test_ds = TinyCensus(n_train, 1), TinyCensus(n_val + n_test, 2)
    val_idx, test_idx = P.make_split(len(test_ds), P.SPLIT_SEED)
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size),
               "val": DataLoader(Subset(test_ds, val_idx.tolist()), batch_size=batch_size),
               "test": DataLoader(Subset(test_ds, test_idx.tolist()), batch_size=batch_size)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=len(train_ds)), (val_idx, test_idx)


def _summary_rows():
    path = REPO / "artifacts" / "census_stage2" / "SUMMARY.md"
    lines = [line for line in path.read_text(encoding="utf-8").strip().splitlines() if line.startswith("|")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    return [dict(zip(header, [cell.strip() for cell in line.strip("|").split("|")])) for line in lines[2:]]


class TestPreregistration(unittest.TestCase):
    """T1–T2：预注册常量与文档一致，且引用的基线数值与仓库内 SUMMARY.md 对得上。"""

    def test_preregistered_constants(self):
        self.assertEqual((adapter.RANK, adapter.ALPHA), (4, 4.0))
        self.assertEqual(adapter.SCALE, 1.0)                                  # alpha/r 惯例
        self.assertEqual((adapter.RATIO_MIN, adapter.RATIO_MAX), (0.005, 0.5))
        self.assertEqual(adapter.AUC_TEST_MIN, 0.8521)
        self.assertGreater(adapter.AUC_TEST_MIN, adapter.BASELINE_AUC)        # 阈值必须高于基线

    def test_baseline_reference_matches_recorded_summary(self):
        rows = _summary_rows()
        row = next((r for r in rows if r["run_id"] == adapter.BASELINE_RUN_ID), None)
        self.assertIsNotNone(row, f"SUMMARY.md 缺少预注册引用的基线 run: {adapter.BASELINE_RUN_ID}")
        self.assertEqual(row["auc_test_education"], f"{adapter.BASELINE_AUC:.6f}")
        self.assertEqual(row["stage1_id"], adapter.BASELINE_STAGE1_ID)


class TestConstructionIdentity(unittest.TestCase):
    """T3–T9：初始恒等、共享参数逐元素相等、全局 RNG 端点不变，且审计本身有判别力。"""

    def test_adapter_is_identity_at_init(self):
        torch.manual_seed(0)
        net = tiny_newtask()
        h = torch.randn(16, TINY_REP_DIM)
        self.assertTrue(torch.equal(net.rep_adapter.delta(h), torch.zeros_like(h)))   # delta ≡ 0
        self.assertTrue(torch.equal(net.rep_adapter(h), h))                           # h' = h

    def test_forward_matches_baseline_newtask_at_init(self):
        torch.manual_seed(P.MODEL_SEED)
        baseline = NewTask(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM, tower_dnn_hidden_units=(4, 2),
                           reg_dnn=P.REG_DNN, device=torch.device("cpu"))
        torch.manual_seed(P.MODEL_SEED)
        net = tiny_newtask()
        dnn_input, gen_rep = torch.randn(8, TINY_INPUT_SIZE), torch.randn(8, TINY_REP_DIM)
        spec_reps = [torch.randn(8, TINY_REP_DIM) for _ in range(2)]
        env_embs = [torch.randn(TINY_REP_DIM) for _ in range(2)]
        baseline.eval(); net.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(baseline(dnn_input, gen_rep, spec_reps, env_embs),
                                        net(dnn_input, gen_rep, spec_reps, env_embs)))

    def test_construction_identity_report(self):
        torch.manual_seed(P.MODEL_SEED)
        report = adapter.construction_identity_report(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                                      tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN)
        self.assertTrue(report["shared_params_bit_identical"])
        self.assertTrue(report["global_rng_endpoint_identical"])
        self.assertEqual(report["extra_keys"], ["rep_adapter.down.weight", "rep_adapter.up.weight"])
        self.assertEqual(report["adapted_state_numel"] - report["baseline_state_numel"],
                         2 * adapter.RANK * TINY_REP_DIM)
        self.assertTrue(adapter.identity_gate(report)["pass"])

    def test_identity_gate_rejects_non_adapter_extra_keys(self):
        report = adapter.construction_identity_report(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                                      tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN)
        self.assertFalse(adapter.identity_gate({**report, "extra_keys": ["something_else"]})["pass"])
        self.assertFalse(adapter.identity_gate({**report, "global_rng_endpoint_identical": False})["pass"])

    def test_real_construction_preserves_global_rng_endpoint(self):
        """构造 LoRANewTask 之后的下一次全局抽样，必须与构造基线 NewTask 之后完全相同。"""
        torch.manual_seed(P.MODEL_SEED)
        with adapter.isolated_cpu_rng():
            NewTask(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM, tower_dnn_hidden_units=(4, 2),
                    reg_dnn=P.REG_DNN, device=torch.device("cpu"))
            expected = torch.rand(4).clone()
        torch.manual_seed(P.MODEL_SEED)
        tiny_newtask()
        self.assertTrue(torch.equal(torch.rand(4), expected))

    def test_actual_instance_identity_report(self):
        """A6 的强证据：对真实构造的实例核对共享参数、RNG 端点、零初始化，且审计自身不动 RNG。"""
        torch.manual_seed(P.MODEL_SEED)
        before = torch.get_rng_state().clone()
        net = tiny_newtask()
        after = torch.get_rng_state().clone()
        state_before_audit = torch.get_rng_state().clone()
        report = adapter.actual_instance_identity_report(
            net, rng_state_before=before, rng_state_after=after, input_size=TINY_INPUT_SIZE,
            rep_dim=TINY_REP_DIM, tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN)
        self.assertTrue(report["shared_params_bit_identical"])
        self.assertTrue(report["global_rng_endpoint_identical"])
        self.assertEqual(report["extra_keys"], ["rep_adapter.down.weight", "rep_adapter.up.weight"])
        self.assertEqual(report["up_fro_at_construction"], 0.0)              # B 零初始化
        self.assertEqual(report["adapted_state_numel"] - report["baseline_state_numel"],
                         2 * adapter.RANK * TINY_REP_DIM)
        self.assertTrue(adapter.identity_gate(report)["pass"])
        self.assertTrue(torch.equal(torch.get_rng_state(), state_before_audit))   # 审计不改全局 RNG

    def test_actual_instance_report_detects_tampering(self):
        """审计必须真有判别力：RNG 端点被扰动 / 非零初始化 / 共享参数被改，都要能被查出来。"""
        torch.manual_seed(P.MODEL_SEED)
        before = torch.get_rng_state().clone()
        net = tiny_newtask()
        after = torch.get_rng_state().clone()
        kwargs = dict(rng_state_before=before, rng_state_after=after, input_size=TINY_INPUT_SIZE,
                      rep_dim=TINY_REP_DIM, tower_dnn_hidden_units=(4, 2), reg_dnn=P.REG_DNN)
        self.assertTrue(adapter.actual_instance_identity_report(net, **kwargs)["shared_params_bit_identical"])
        self.assertFalse(adapter.actual_instance_identity_report(
            net, **{**kwargs, "rng_state_after": after + 1})["global_rng_endpoint_identical"])
        with torch.no_grad():
            net.rep_adapter.up.weight.normal_(std=0.01)                      # 破坏零初始化
        self.assertGreater(adapter.actual_instance_identity_report(net, **kwargs)["up_fro_at_construction"], 0.0)
        with torch.no_grad():
            net.tower_network.mlp[0].weight.add_(0.1)                        # 篡改共享参数（放最后：不可逆）
        self.assertFalse(adapter.actual_instance_identity_report(net, **kwargs)["shared_params_bit_identical"])


class TestFreezeAndGradients(unittest.TestCase):
    """T10–T11：backbone 未被改动且无梯度；适配器梯度活性（含 B=0 的固有性质）。"""

    def test_training_steps_keep_backbone_frozen_and_activate_adapter(self):
        torch.manual_seed(0)
        backbone = tiny_mptrec()
        P.freeze_backbone(backbone)
        sha_before = P.backbone_sha256(backbone)
        net = tiny_newtask()
        optimizer = torch.optim.Adam(params=net.parameters(), lr=P.LR)
        loss_func = torch.nn.BCELoss()
        probe = adapter.AdapterGradProbe(net.rep_adapter)
        features = {"a": torch.randint(0, 3, (16,)), "b": torch.randint(0, 2, (16,))}
        y = torch.randint(0, 2, (16,)).float()
        for step in range(1, 3):
            with torch.no_grad():
                dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
            loss = loss_func(net(dnn_input, gen_rep, spec_reps, env_embs), y)
            optimizer.zero_grad()
            loss.backward()
            probe.observe(epoch=1, step=step, is_first=(step == 1))
            optimizer.step()
        probe.end_epoch()
        self.assertGreater(probe.first["up_grad_norm"], 0.0)                   # B 第 1 步即有梯度
        self.assertEqual(probe.first["down_grad_norm"], 0.0)                   # dL/dA ∝ B = 0
        self.assertGreater(probe.per_epoch[-1]["down_grad_norm"], 0.0)         # 更新一步后 A 有梯度
        self.assertTrue(adapter.grad_gate(probe.first, probe.per_epoch)["pass"])
        self.assertEqual(P.backbone_sha256(backbone), sha_before)              # A1：backbone 未变
        P.assert_no_grads(backbone)                                           # backbone 梯度全 None
        self.assertFalse(any(p.requires_grad for p in backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in net.parameters()))        # 头（含适配器）可训练

    def test_grad_gate_semantics(self):
        first = {"epoch": 1, "step": 1, "down_grad_norm": 0.0, "up_grad_norm": 0.5}
        late = {**first, "step": 9, "down_grad_norm": 0.3}
        self.assertTrue(adapter.grad_gate(first, [late])["pass"])
        self.assertFalse(adapter.grad_gate({**first, "up_grad_norm": 0.0}, [late])["pass"])
        self.assertFalse(adapter.grad_gate(first, [{**late, "down_grad_norm": 0.0}])["pass"])
        self.assertFalse(adapter.grad_gate({**first, "down_grad_norm": 0.1}, [late])["pass"])   # B 非零初始化
        self.assertFalse(adapter.grad_gate(None, [])["pass"])


class TestParamsAndStats(unittest.TestCase):
    """T12–T15：参数量核算（小夹具 + 真实维度）、逐流占比/余弦统计与 R1 判定。"""

    def test_param_report_counts_and_ratio(self):
        report = adapter.param_report(tiny_newtask())
        self.assertEqual(report["adapter_params"], 2 * adapter.RANK * TINY_REP_DIM)
        self.assertEqual(report["newtask_total_params"] - report["newtask_head_params"],
                         report["adapter_params"])
        self.assertEqual((report["rank"], report["scale"]), (adapter.RANK, 1.0))
        self.assertGreater(report["adapter_ratio_of_newtask"], 0.0)
        # 小夹具的 rep_dim=8 会把占比放大（真实维度见下一个用例），此处只要求"仍是少数派"
        self.assertLess(report["adapter_ratio_of_newtask"], 0.5)

    def test_param_report_real_dimensions_match_preregistration(self):
        """真实维度（input 123 / rep 128 / tower (64,32)）下核算预注册文档 §5 写的参数量。"""
        net = adapter.LoRANewTask(input_size=P.INPUT_SIZE, rep_dim=P.EXPERT_HIDDEN[-1],
                                  tower_dnn_hidden_units=list(P.TOWER_HIDDEN), reg_dnn=P.REG_DNN,
                                  device=torch.device("cpu"))
        report = adapter.param_report(net)
        self.assertEqual(report["adapter_params"], 2 * adapter.RANK * P.EXPERT_HIDDEN[-1])     # 1024
        self.assertEqual(report["newtask_head_params"], 27063)
        self.assertEqual(report["newtask_total_params"], 28087)
        self.assertAlmostEqual(report["adapter_ratio_of_newtask"], 1024 / 28087, places=12)

    def test_adapter_stats_match_reference(self):
        torch.manual_seed(0)
        net = tiny_newtask()
        with torch.no_grad():
            torch.nn.init.normal_(net.rep_adapter.up.weight, std=0.05)         # 让 delta 非零
        gen_rep = torch.randn(7, TINY_REP_DIM)
        spec_reps = [torch.randn(7, TINY_REP_DIM) for _ in range(2)]
        _, _, gen_delta, spec_deltas = net.adapt_with_delta(gen_rep, spec_reps)
        stats = adapter.AdapterStats(num_tasks=2)
        stats.update(gen_rep, gen_delta, spec_reps, spec_deltas)
        out = stats.result()
        reference = [(delta.double().norm(dim=1) / rep.double().norm(dim=1)).mean().item()
                     for rep, delta in zip((gen_rep, *spec_reps), (gen_delta, *spec_deltas))]
        self.assertEqual(out["streams"], ["gen", "spec_0", "spec_1"])
        self.assertEqual((out["count"], out["n_zero_rep"]), (7, 0))
        for got, want in zip(out["ratio_mean"], reference):
            self.assertAlmostEqual(got, want, places=10)
        for got, want in zip(out["delta_norm_mean"],
                             [delta.double().norm(dim=1).mean().item() for delta in (gen_delta, *spec_deltas)]):
            self.assertAlmostEqual(got, want, places=10)

    def test_ratio_gate_band(self):
        in_band = {"streams": ["gen"], "ratio_mean": [0.05], "ratio_max": [0.2], "n_zero_rep": 0}
        self.assertTrue(adapter.ratio_gate(in_band)["pass"])
        for out_of_band in (0.0, adapter.RATIO_MIN - 1e-9, adapter.RATIO_MAX + 1e-9, 1.0):
            self.assertFalse(adapter.ratio_gate({**in_band, "ratio_mean": [out_of_band]})["pass"])
        self.assertFalse(adapter.ratio_gate({**in_band, "ratio_mean": [0.05, 0.9]})["pass"])   # 任一路越界即失败


class TestBranchIsolation(unittest.TestCase):
    """T16–T17：本实验分支不得触碰协议/模型文件；基线 runner 不得耦合适配器。"""

    def test_protected_files_untouched_since_branch_point(self):
        if _git("cat-file", "-e", EXP_BASE_COMMIT).returncode != 0:
            self.skipTest(f"基准 commit {EXP_BASE_COMMIT[:7]} 不在本地（浅克隆？）")
        self.assertEqual(_git("merge-base", "--is-ancestor", EXP_BASE_COMMIT, "HEAD").returncode, 0,
                         "分支已被 rebase：请更新 EXP_BASE_COMMIT 并复核协议文件差异")
        diff = _git("diff", "--name-only", EXP_BASE_COMMIT, "--", *PROTECTED_PATHS).stdout.strip()
        self.assertEqual(diff, "", f"实验分支不得改动协议/模型文件: {diff}")

    def test_baseline_runner_and_protocol_have_no_adapter_coupling(self):
        sources = [REPO / "run_census_benchmark.py", REPO / "census_benchmark" / "protocol.py",
                   REPO / "census_benchmark" / "metrics.py"]
        for path in sources:
            source = path.read_text(encoding="utf-8")
            for token in ("adapter", "Adapter", "LoRA", "lora", "rep_adapter"):
                self.assertNotIn(token, source, f"{path.name} 不应出现 {token}（协议路径必须与实验解耦）")


class TestAdapterRunnerSmoke(unittest.TestCase):
    """T18–T19：runner 变体端到端（CPU 小夹具）与 Stage-1 产物只读。"""

    def test_adapter_runner_smoke_cpu_tiny(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            loaders, stats, indices = tiny_inputs()
            stage1 = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_mptrec(),
                                                     loaders=loaders, stats=stats, indices=indices)
            out = run_stage2_adapter(root, stage1_dir=Path(stage1["dir"]), epochs=3, device=device,
                                     model=tiny_mptrec(), loaders=loaders, stats=stats, indices=indices,
                                     input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM)
            run_path = Path(out["run_dir"])
            for name in ("config.json", "metrics.json", "split_fingerprint.json", "env_ids.pt",
                         "newtask.pt", "gate_report.json", "adapter_report.json", "stdout.log"):
                self.assertTrue((run_path / name).exists(), f"缺少产物: {name}")

            report = json.loads((run_path / "gate_report.json").read_text(encoding="utf-8"))
            for key in ("A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4"):        # 协议门禁仍按原口径产出
                self.assertIn(key, report)
            self.assertTrue(report["A1"]["pass"])
            self.assertTrue(report["A2"]["pass"])
            self.assertTrue(report["A5"]["pass"])
            self.assertEqual(report["A3"]["status"], "on_demand")
            self.assertTrue(report["adapter"]["A6"]["pass"])                  # 构造期恒等
            self.assertTrue(report["adapter"]["G"]["pass"])                   # 梯度活性
            self.assertIn("pass", report["adapter"]["R1"])
            self.assertFalse(report["adapter"]["S1"]["pass"])                 # 小夹具达不到 0.8521，属预期
            self.assertEqual(report["adapter"]["S1"]["detail"]["baseline_auc"], adapter.BASELINE_AUC)

            metrics_json = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics_json["backbone_sha256_before"], metrics_json["backbone_sha256_after"])
            self.assertEqual(len(metrics_json["stage2"]["epoch_records"]), 3)
            self.assertEqual(metrics_json["mechanism"]["adapter"]["streams"], ["gen", "spec_0", "spec_1"])
            self.assertEqual(len(metrics_json["stage2"]["adapter_norms_by_epoch"]), 3)

            adapter_json = json.loads((run_path / "adapter_report.json").read_text(encoding="utf-8"))
            self.assertEqual(adapter_json["params"]["adapter_params"], 2 * adapter.RANK * TINY_REP_DIM)
            self.assertTrue(adapter_json["construction_identity"]["shared_params_bit_identical"])
            self.assertTrue(adapter_json["construction_identity"]["global_rng_endpoint_identical"])
            self.assertEqual(adapter_json["construction_identity"]["up_fro_at_construction"], 0.0)  # B 零初始化
            self.assertEqual(adapter_json["grad_probe"]["first"]["down_grad_norm"], 0.0)
            self.assertEqual(adapter_json["config"]["shared_across_streams"], True)
            self.assertEqual(len(adapter_json["val_stats_final"]["cos_mean"]), 3)
            self.assertEqual(adapter_json["preregistration"],
                             "docs/superpowers/specs/2026-09-30-stage2-lora-adapter-design.md")

            config_json = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config_json["mechanism"], adapter.MECHANISM)
            self.assertEqual(config_json["stage1_id"], stage1["stage1_id"])
            self.assertEqual(config_json["adapter_injection"], "gen_rep+spec_reps_pre_fusion")

            summary = (root / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(summary), 3)                                  # 表头 + 分隔 + 1 行
            self.assertIn(out["run_id"], summary[2])
            self.assertEqual([c.strip() for c in summary[0].strip("|").split("|")], P.SUMMARY_COLUMNS)

    def test_adapter_runner_keeps_stage1_artifact_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            loaders, stats, indices = tiny_inputs()
            stage1 = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_mptrec(),
                                                     loaders=loaders, stats=stats, indices=indices)
            stage1_path = Path(stage1["dir"])
            before = {path.name: path.read_bytes() for path in stage1_path.iterdir() if path.is_file()}
            self.assertIn("stdout.log", before)                  # Stage-1 产物含 stdout.log，必须一并纳入只读检查
            run_stage2_adapter(root, stage1_dir=stage1_path, epochs=2, device=device, model=tiny_mptrec(),
                               loaders=loaders, stats=stats, indices=indices,
                               input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM)
            after = {path.name: path.read_bytes() for path in stage1_path.iterdir() if path.is_file()}
            self.assertEqual(sorted(after), sorted(before), "Stage-1 产物目录不得新增/删除文件")
            for name, payload in before.items():
                self.assertEqual(after[name], payload, f"Stage-1 产物被改写: {name}")


if __name__ == "__main__":
    unittest.main()
