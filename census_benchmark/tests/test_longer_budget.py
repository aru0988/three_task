"""更长预算持久性检验（预注册 `2026-10-05-census-stage2-null-expert-longer-budget-design.md` §5–§7）。

- 判定 V1–V5 边界/真值表、INVALID 分支、identity/记录校验（§5.3）。
- 短预算 context JSON 字节钉死（§7 I7）。
- runner CLI：`--epochs/--patience/--tag` 默认逐位不变、override 正确线程化、run 记录实际预算（§7 I4/I5）。
- 分析器只读（§7 I6）与文档 token 钉死（§7 I8）。
"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from census_benchmark import longer_budget as LB
from census_benchmark import multiseed as M
from census_benchmark import protocol as P
from multitaskrec.model import MPTRec
import run_census_benchmark

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "superpowers" / "specs" / "2026-10-05-census-stage2-null-expert-longer-budget-design.md"
SEEDS = LB.CANONICAL_SEEDS

ALL_PASS = dict(delta_test=[0.002, 0.0015, 0.0012, -0.0004, 0.0016],
                delta_val=[0.001, 0.0005, 0.0002, -0.0001, 0.0003],
                delta_val_last_common=[0.0005, 0.0002, 0.0001, -0.0002, 0.0004],
                null_top1=[0.37, 0.41, 0.42, 0.45, 0.30])


class TestPersistenceVerdict(unittest.TestCase):
    """§5.3 判据的机械函数：逐条边界 + 真值表。"""

    def _verdict(self, **overrides):
        return LB.persistence_verdict(**dict(ALL_PASS, **overrides))

    def test_all_pass_is_persists(self):
        out = self._verdict()
        self.assertEqual(out["classification"], "PERSISTS")
        self.assertTrue(all(out["checks"].values()))
        self.assertAlmostEqual(out["mean_delta_test"], sum(ALL_PASS["delta_test"]) / 5, places=15)
        self.assertEqual(out["positive_delta_count"], 4)

    def test_v1_boundary_inclusive(self):
        exact = self._verdict(delta_test=[0.001] * 5)            # mean 恰 +0.001 → 过（闭区间）
        self.assertTrue(exact["checks"]["V1_mean_delta_test_ge_min"])
        self.assertEqual(exact["classification"], "PERSISTS")
        below = self._verdict(delta_test=[0.001, 0.001, 0.001, 0.001, 0.000999])
        self.assertFalse(below["checks"]["V1_mean_delta_test_ge_min"])
        self.assertEqual(below["classification"], "NOT_PERSIST")

    def test_v2_strict_majority_and_no_degradation(self):
        three_positive = self._verdict(delta_test=[0.0009, 0.0008, 0.0007, -0.0001, -0.0002])   # 3/5 > 0
        self.assertTrue(three_positive["checks"]["V2_majority_positive_no_degradation"])
        two_positive = self._verdict(delta_test=[0.0009, 0.0008, -0.0001, -0.0002, -0.0003])    # 2/5
        self.assertFalse(two_positive["checks"]["V2_majority_positive_no_degradation"])
        zero_not_positive = self._verdict(delta_test=[0.0009, 0.0008, 0.0, -0.0001, -0.0002])
        self.assertFalse(zero_not_positive["checks"]["V2_majority_positive_no_degradation"])
        degrade = self._verdict(delta_test=[0.002, 0.002, 0.002, 0.002, -0.02])                  # 恰 −0.02
        self.assertFalse(degrade["checks"]["V2_majority_positive_no_degradation"])
        near = self._verdict(delta_test=[0.002, 0.002, 0.002, 0.002, -0.0199999])
        self.assertTrue(near["checks"]["V2_majority_positive_no_degradation"])

    def test_v3_majority_val_direction(self):
        three = self._verdict(delta_val=[0.001, 0.001, 0.001, -0.001, -0.001])
        self.assertTrue(three["checks"]["V3_majority_delta_val_positive"])
        two = self._verdict(delta_val=[0.001, 0.001, -0.001, -0.001, -0.001])
        self.assertFalse(two["checks"]["V3_majority_delta_val_positive"])

    def test_v4_strictly_positive_trajectory_endpoint(self):
        zero = self._verdict(delta_val_last_common=[0.0] * 5)
        self.assertFalse(zero["checks"]["V4_mean_last_common_delta_val_positive"])
        tiny = self._verdict(delta_val_last_common=[1e-12] * 5)
        self.assertTrue(tiny["checks"]["V4_mean_last_common_delta_val_positive"])

    def test_v5_mechanism_band_inclusive_ends(self):
        ends = self._verdict(null_top1=[0.05, 0.95, 0.5, 0.5, 0.5])
        self.assertTrue(ends["checks"]["V5_mechanism_active_all_arms"])
        out_low = self._verdict(null_top1=[0.0499, 0.5, 0.5, 0.5, 0.5])
        self.assertFalse(out_low["checks"]["V5_mechanism_active_all_arms"])
        out_high = self._verdict(null_top1=[0.5, 0.5, 0.5, 0.5, 0.9501])
        self.assertFalse(out_high["checks"]["V5_mechanism_active_all_arms"])

    def test_threshold_constants_are_the_frozen_numbers(self):
        self.assertEqual(LB.V1_MEAN_DELTA_TEST_MIN, 0.001)
        self.assertEqual(LB.V2_DEGRADE_MAX, -0.02)
        self.assertEqual((LB.V5_TOP1_MIN, LB.V5_TOP1_MAX), (0.05, 0.95))
        self.assertEqual(LB.LONGER_BUDGET_EPOCHS, 10)
        self.assertEqual(LB.LONGER_BUDGET_PATIENCE, 3)
        self.assertEqual(LB.LONGER_BUDGET_TAG, "long")


class TestShortContextPin(unittest.TestCase):
    """§7 I7：短预算 context JSON 字节钉死（sha256），不一致必须抛错。"""

    def test_real_context_loads_with_pinned_values(self):
        ctx = LB.load_short_context(REPO / LB.SHORT_CONTEXT_PATH)
        by_seed = {rec["model_seed"]: rec for rec in ctx["seeds"]}
        self.assertEqual(set(by_seed), set(SEEDS))
        self.assertAlmostEqual(by_seed[1685480945]["delta_test"], 0.001296296526501206, places=18)
        self.assertAlmostEqual(by_seed[1685463909]["delta_test"], -0.0005284892173337274, places=18)
        self.assertAlmostEqual(by_seed[1685477428]["delta_test"], 0.00287759533010834, places=18)
        self.assertAlmostEqual(by_seed[1685459668]["delta_test"], 0.0008191587308337134, places=18)
        self.assertAlmostEqual(by_seed[1685496394]["delta_test"], 0.0002448654571886033, places=18)

    def test_tampered_context_raises(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.json"
            bad.write_text('{"seeds": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                LB.load_short_context(bad)


def _write_long_run(root, run_id, *, model_seed, null_expert, test_auc, best_val_auc, stage1_id,
                    delta_traj=None, epochs=10, patience=3, tag="long", split_fp=LB.PINNED_SPLIT_FINGERPRINT,
                    env_sha="cd" * 32, a_pass=True):
    """伪造 long run 目录（仅含有趣字段；结构对齐真实 runner 落盘）。"""
    run_dir = Path(root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    traj = delta_traj or [best_val_auc - 0.005 + 0.001 * i for i in range(epochs)]
    records = [{"epoch": i + 1, "loss": 0.2, "auc_val_education": auc} for i, auc in enumerate(traj)]
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "stage1_id": stage1_id, "commit": "c0ffee0", "tag": tag, "frozen": True,
        "null_expert": null_expert, "split_seed": P.SPLIT_SEED, "model_seed": model_seed,
        "env_seed": P.ENV_SEED, "epochs": epochs, "patience": patience, "lr": 1e-3, "batch_size": 256,
        "input_size": 123, "rep_dim": 128}), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "stage1_id": stage1_id, "commit": "c0ffee0", "null_expert": null_expert,
        "split_sha256": {"val": "v", "test": "t", "fingerprint": split_fp},
        "env_ids_sha256": env_sha,
        "stage2": {"epoch_records": records, "best_epoch": len(records), "best_val_auc": best_val_auc,
                   "test_auc": test_auc},
        "mechanism": ({"null_mean": 0.34, "null_top1_rate": 0.37, "null_std": 0.17,
                       "route_entropy_mean": 0.91, "route_var_mean": 0.04} if null_expert
                      else {"gate_mean": [0.5, 0.6], "gen_std": 0.1})}), encoding="utf-8")
    (run_dir / "gate_report.json").write_text(json.dumps({
        "A1": {"pass": a_pass}, "A2": {"pass": a_pass}, "A4": {"pass": a_pass}, "A5": {"pass": a_pass},
        "B1": {"pass": True}, "B2": {"pass": True}, "B3": {"pass": False}, "B4": {"pass": True},
        "overall_pass": False, "failures": ["B3"]}), encoding="utf-8")
    return run_dir


def _build_long_root(root, *, deltas, val_deltas=None, null_top1=None, stage1_ids=None):
    """为 5 个 canonical seed 写 long 对子；返回 {seed: (baseline_dir, arm_dir)}。"""
    val_deltas = val_deltas or [0.0005] * len(deltas)
    null_top1 = null_top1 or [0.37] * len(deltas)
    stage1_ids = stage1_ids or {seed: LB.PINNED_STAGE1_IDS[seed] for seed in SEEDS}
    out = {}
    for i, seed in enumerate(SEEDS):
        base_val = 0.8500
        b = _write_long_run(root, f"20261005-1000-s20260929-m{seed}-long-c0ffee0", model_seed=seed,
                            null_expert=False, test_auc=base_val, best_val_auc=base_val,
                            stage1_id=stage1_ids[seed])
        a = _write_long_run(root, f"20261005-1010-s20260929-m{seed}-long-c0ffee0-nullx", model_seed=seed,
                            null_expert=True, test_auc=base_val + deltas[i],
                            best_val_auc=base_val + val_deltas[i], stage1_id=stage1_ids[seed])
        out[seed] = (b, a)
    out["_null_top1"] = null_top1
    return out


def _patch_null_top1(fixture):
    for i, seed in enumerate(SEEDS):
        path = fixture[seed][1] / "metrics.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["mechanism"]["null_top1_rate"] = fixture["_null_top1"][i]
        path.write_text(json.dumps(doc), encoding="utf-8")


class TestAnalyzeFixture(unittest.TestCase):
    """夹具端到端：判定、identity、context、只读性。"""

    DELTAS = [0.002, 0.0015, 0.0012, -0.0001, 0.0016]

    def test_end_to_end_persists_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = _build_long_root(root, deltas=self.DELTAS)
            _patch_null_top1(fixture)
            out = LB.analyze(root)
            self.assertEqual(out["verdict"]["classification"], "PERSISTS")
            self.assertTrue(out["identity_checks"]["all_pass"])
            self.assertEqual(len(out["seeds"]), 5)
            self.assertEqual(out["stats"]["n"], 5)
            self.assertAlmostEqual(out["stats"]["mean"], sum(self.DELTAS) / 5, places=15)
            self.assertEqual(out["stats"]["worst_seed"], SEEDS[3])
            rec0 = out["seeds"][0]
            self.assertEqual(rec0["stage1_id"], LB.PINNED_STAGE1_IDS[SEEDS[0]])
            self.assertIn("headroom", rec0)
            self.assertIn("budget", rec0)
            ctx = out["context"]["five_epoch"]
            self.assertAlmostEqual(ctx[str(SEEDS[0])]["delta_test"], 0.001296296526501206, places=18)
            self.assertAlmostEqual(out["context"]["delta_test_change_by_seed"][0],
                                   0.002 - 0.001296296526501206, places=15)

    def test_stage1_not_pinned_is_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = _build_long_root(root, deltas=self.DELTAS)
            _patch_null_top1(fixture)
            # 篡改 seed0 两臂的 stage1_id（两臂相互一致但与钉死值不符 → 对子成立、identity 不过）
            for arm_dir in fixture[SEEDS[0]]:
                for name in ("config.json", "metrics.json"):
                    path = arm_dir / name
                    doc = json.loads(path.read_text(encoding="utf-8"))
                    doc["stage1_id"] = "s1-deadbeef-m1685480945-e2-0000aaaa"
                    path.write_text(json.dumps(doc), encoding="utf-8")
            out = LB.analyze(root)
            self.assertEqual(out["classification"], "INVALID")
            self.assertFalse(out["identity_checks"]["all_pass"])
            self.assertFalse(out["identity_checks"]["by_seed"][str(SEEDS[0])]["stage1_id_is_pinned"])
            self.assertTrue(out["identity_checks"]["by_seed"][str(SEEDS[0])]["same_stage1_id"])

    def test_mismatched_arms_stage1_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = _build_long_root(root, deltas=self.DELTAS)
            _patch_null_top1(fixture)
            path = fixture[SEEDS[0]][1] / "config.json"                 # 只改处理臂 → 配对本身违规
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["stage1_id"] = "s1-deadbeef-m1685480945-e2-0000aaaa"
            path.write_text(json.dumps(doc), encoding="utf-8")
            path = fixture[SEEDS[0]][1] / "metrics.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["stage1_id"] = "s1-deadbeef-m1685480945-e2-0000aaaa"
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(ValueError):                          # 禁止跨 artifact 配对（find_runs 硬约束）
                LB.analyze(root)

    def test_wrong_epochs_or_patience_is_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = _build_long_root(root, deltas=self.DELTAS)
            _patch_null_top1(fixture)
            path = fixture[SEEDS[1]][1] / "config.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["patience"] = 2
            path.write_text(json.dumps(doc), encoding="utf-8")
            out = LB.analyze(root)
            self.assertEqual(out["classification"], "INVALID")
            self.assertFalse(out["identity_checks"]["by_seed"][str(SEEDS[1])]["patience_recorded_pinned"])

    def test_a_class_fail_is_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = _build_long_root(root, deltas=self.DELTAS)
            _patch_null_top1(fixture)
            path = fixture[SEEDS[2]][0] / "gate_report.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["A1"]["pass"] = False
            path.write_text(json.dumps(doc), encoding="utf-8")
            out = LB.analyze(root)
            self.assertEqual(out["classification"], "INVALID")
            self.assertFalse(out["identity_checks"]["by_seed"][str(SEEDS[2])]["a_class_gates_pass"])

    def test_missing_arm_raises_and_no_implicit_short_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _build_long_root(root, deltas=self.DELTAS)
            # 缺 seed4 处理臂
            arm = next((root / "runs").glob(f"*m{SEEDS[4]}-long-c0ffee0-nullx"))
            for f in arm.iterdir():
                f.unlink()
            arm.rmdir()
            with self.assertRaises(ValueError):
                LB.analyze(root)

    def test_analyze_is_read_only(self):
        import hashlib
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = _build_long_root(root, deltas=self.DELTAS)
            _patch_null_top1(fixture)
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (root / "runs").rglob("*") if p.is_file()}
            LB.analyze(root)
            after = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (root / "runs").rglob("*") if p.is_file()}
            self.assertEqual(before, after)


class TestCli(unittest.TestCase):
    def test_main_writes_json_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root, out_path = Path(td) / "artifacts", Path(td) / "longer_budget.json"
            fixture = _build_long_root(root, deltas=[0.002, 0.0015, 0.0012, -0.0001, 0.0016])
            _patch_null_top1(fixture)
            code = LB.main(["--root", str(root), "--out", str(out_path)])
            self.assertEqual(code, 0)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["verdict"]["classification"], "PERSISTS")
            self.assertEqual(payload["spec"], LB.SPEC_PATH)


class TestStage2BudgetCli(unittest.TestCase):
    """§7 I4/I5：`--epochs/--patience/--tag` 默认逐位不变；override 正确线程化并落盘。"""

    def test_parser_defaults_match_protocol(self):
        args = run_census_benchmark.build_parser().parse_args(
            ["stage2", "--stage1-dir", "x"])
        self.assertEqual(args.epochs, P.STAGE2_EPOCHS)
        self.assertEqual(args.patience, P.PATIENCE)
        self.assertEqual(args.tag, "short")
        self.assertFalse(args.null_expert)

    def test_tag_choices_include_long(self):
        args = run_census_benchmark.build_parser().parse_args(
            ["stage2", "--stage1-dir", "x", "--tag", "long"])
        self.assertEqual(args.tag, "long")

    def test_main_threads_epochs_patience_tag(self):
        captured = {}

        def fake_run_stage2(root, **kwargs):
            captured.update(kwargs)
            return {"run_id": "fake"}

        original = run_census_benchmark.run_stage2
        run_census_benchmark.run_stage2 = fake_run_stage2
        try:
            code = run_census_benchmark.main(["stage2", "--stage1-dir", "s1-x", "--tag", "long",
                                              "--epochs", "10", "--patience", "3", "--null-expert", "--gpu", "0"])
        finally:
            run_census_benchmark.run_stage2 = original
        self.assertEqual(code, 0)
        self.assertEqual(captured["epochs"], 10)
        self.assertEqual(captured["patience"], 3)
        self.assertEqual(captured["tag"], "long")
        self.assertTrue(captured["null_expert"])

    def test_run_config_records_actual_patience(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            loaders, stats, indices = _tiny_inputs()
            stage1 = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=_tiny_model(),
                                                     loaders=loaders, stats=stats, indices=indices)
            out = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=3, patience=3,
                                                  tag="long", device=device, model=_tiny_model(),
                                                  loaders=loaders, stats=stats, indices=indices,
                                                  input_size=8, rep_dim=4)
            config = json.loads((Path(out["run_dir"]) / "config.json").read_text(encoding="utf-8"))
            self.assertEqual((config["epochs"], config["patience"], config["tag"]), (3, 3, "long"))
            out_default = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=3,
                                                          device=device, model=_tiny_model(),
                                                          loaders=loaders, stats=stats, indices=indices,
                                                          input_size=8, rep_dim=4)
            config = json.loads((Path(out_default["run_dir"]) / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["patience"], P.PATIENCE)                 # 默认值 == 协议常量（逐位不变）


class TestPreregDocTokens(unittest.TestCase):
    """§7 I8：预注册文档 token 钉死。"""

    def test_doc_pins_budget_seeds_and_rules(self):
        text = DOC.read_text(encoding="utf-8")
        for token in ("10", "3", "`long`", "1685480945", "1685463909", "1685477428", "1685459668", "1685496394",
                      "s1-096f8f16-m1685480945-e2-cb2094b3", "s1-096f8f16-m1685463909-e2-8b53a3bf",
                      "s1-096f8f16-m1685477428-e2-15eabcb4", "s1-096f8f16-m1685459668-e2-8611b794",
                      "s1-096f8f16-m1685496394-e2-2231eae1",
                      "+0.001", "-0.02", "0.05", "0.95", "2.776", "ddof=1",
                      "PERSISTS", "NOT_PERSIST", "INVALID", "只作本实验的动机", "605196e5"):
            self.assertIn(token, text, f"预注册文档缺少 token: {token}")


class TestStaticGuardWhitelist(unittest.TestCase):
    """§7 I9：相对基点 87e2b8d 的跟踪改动 ⊆ 白名单；protocol.py 零 diff。"""

    WHITELIST = {
        "docs/superpowers/specs/2026-10-05-census-stage2-null-expert-longer-budget-design.md",
        "docs/superpowers/specs/2026-10-05-census-stage2-null-expert-multiseed-checkpoint-5seed.json",
        "verify_census_stage1_artifacts.py",
        "census_benchmark/metrics.py",
        "census_benchmark/multiseed.py",
        "census_benchmark/longer_budget.py",
        "census_benchmark/tests/test_multiseed.py",
        "census_benchmark/tests/test_null_diagnostics.py",
        "census_benchmark/tests/test_longer_budget.py",
        "run_census_benchmark.py",
        "artifacts/census_stage2/SUMMARY.md",
    }

    def test_only_whitelisted_paths_differ_from_base(self):
        base = subprocess.run(["git", "rev-parse", "exp/stage2-null-expert"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
        changed = subprocess.run(["git", "diff", "--name-only", base], cwd=REPO,
                                 capture_output=True, text=True).stdout.split()
        unknown = set(changed) - self.WHITELIST
        self.assertEqual(unknown, set(), f"白名单之外的改动: {sorted(unknown)}")
        protocol_diff = subprocess.run(["git", "diff", "--name-only", base, "--", "census_benchmark/protocol.py",
                                        "multitaskrec", "config.py", "CensusIncome_MPTRec.py",
                                        "CensusIncome_NewTask.py", "baseline"], cwd=REPO,
                                       capture_output=True, text=True).stdout.split()
        self.assertEqual(protocol_diff, [], f"禁改文件出现 diff: {protocol_diff}")


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


if __name__ == "__main__":
    unittest.main()
