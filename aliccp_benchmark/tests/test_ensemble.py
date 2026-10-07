"""参数高效集成（方向 3）机制与运行器测试（TDD：先于实现编写）。

spec: docs/superpowers/specs/2026-10-08-aliccp-parameter-efficient-ensemble-design.md
覆盖预注册要求的机制门禁：两头参数不共享、各自非零梯度、零残差时输出与主头逐位相同、
对照与集成参数预算匹配、集成均值计算正确、冻结 backbone 参数/梯度/SHA 不变；
另含塌缩判定、结局分类边界、独立重算解析器与三臂端到端小流程，以及真实性守卫
（身份字段 config/metrics/predictions 三方互证、raw 载荷长度/有限性/均值自洽、
产物树锚定）及其篡改单测。CPU-only、临时合成数据。
"""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from aliccp_benchmark import bench, ensemble, metrics, protocol
from multitaskrec.model import NewTask

HEADER = "click,purchase,X,121,122,301"
TINY_VOCAB = {"121": 5, "122": 4}
DEFAULT_STAGE1_ID = "s1-tiny-m123-e1-cafe0000"
DEFAULT_FP_SHA = "f" * 64
DEFAULT_BB_SHA = "b" * 64

TRAIN_ROWS = [
    [0, 0, 9, 1, 0, 1],
    [1, 0, 9, 2, 1, 2],
    [0, 0, 8, 0, 2, 3],
    [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3],
    [1, 0, 7, 1, 1, 2],
    [0, 0, 9, 2, 2, 1],
    [1, 1, 9, 0, 1, 3],
    [0, 0, 8, 3, 0, 2],
    [1, 0, 8, 1, 1, 1],
    [0, 0, 7, 4, 2, 3],
    [1, 0, 7, 2, 1, 2],
]
VAL_ROWS = [
    [0, 0, 9, 1, 0, 1],
    [1, 0, 9, 2, 1, 2],
    [0, 0, 8, 0, 2, 3],
    [1, 1, 8, 3, 1, 1],
    [0, 0, 7, 4, 0, 3],
    [1, 0, 7, 1, 1, 2],
]
TEST_ROWS = [
    [0, 0, 9, 2, 2, 1],
    [1, 0, 9, 0, 1, 3],
    [0, 0, 8, 3, 0, 2],
    [1, 1, 8, 1, 1, 1],
    [0, 0, 7, 4, 2, 3],
    [1, 0, 7, 2, 1, 2],
    [0, 0, 9, 1, 0, 3],
    [1, 0, 8, 4, 1, 2],
]


def write_aliccp_file(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")


def make_tiny_root(td: str):
    root = Path(td) / "artifacts"
    data_dir = Path(td) / "data"
    data_dir.mkdir()
    write_aliccp_file(data_dir / "train.csv", TRAIN_ROWS)
    write_aliccp_file(data_dir / "val.csv", VAL_ROWS)
    write_aliccp_file(data_dir / "test.csv", TEST_ROWS)
    data_files = {
        "train": str(data_dir / "train.csv"),
        "val": str(data_dir / "val.csv"),
        "test": str(data_dir / "test.csv"),
    }
    budgets = {"train": len(TRAIN_ROWS), "val": len(VAL_ROWS), "test": len(TEST_ROWS)}
    return root, data_files, budgets


def tiny_head(arm, seed=123, rep_dim=8):
    torch.manual_seed(seed)
    return ensemble.build_arm_head(
        arm,
        input_size=4,
        rep_dim=rep_dim,
        tower_dnn_hidden_units=(4,),
        reg_dnn=7e-6,
        device=torch.device("cpu"),
    )


def tiny_inputs(batch=16, rep_dim=8, input_size=4, seed=7):
    gen = torch.Generator().manual_seed(seed)
    dnn_input = torch.randn(batch, input_size, generator=gen)
    gen_rep = torch.randn(batch, rep_dim, generator=gen)
    spec_reps = [torch.randn(batch, rep_dim, generator=gen) for _ in range(2)]
    env_embs = [torch.randn(rep_dim, generator=gen) for _ in range(2)]
    return dnn_input, gen_rep, spec_reps, env_embs


def tiny_labels(batch=16, seed=11):
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(batch, generator=gen) > 0.5).long()


def noop_log(*args, **kwargs):
    pass


def run_tiny_stage1_at(root, data_files, budgets, *, model_seed=123, env_seed=456, epochs=2, patience=2):
    """tiny Stage-1（CPU、合成数据）；端到端与产物树锚定测试共用。"""
    return bench.run_stage1(
        root=root, data_files=data_files, budgets=budgets, prefix_tag="tiny",
        model_seed=model_seed, env_seed=env_seed, epochs=epochs, patience=patience,
        device=torch.device("cpu"), vocab=TINY_VOCAB,
        expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
        log=noop_log,
    )


class TestResidualHead(unittest.TestCase):
    def test_up_layer_zero_initialized_outputs_exact_zero(self):
        head = ensemble.ResidualHead(rep_dim=6, rank=2)
        x = torch.randn(5, 6)
        self.assertTrue(torch.equal(head(x), torch.zeros(5, 1)))

    def test_rank_is_bounded_by_bottleneck(self):
        head = ensemble.ResidualHead(rep_dim=6, rank=2)
        self.assertEqual(tuple(head.down.weight.shape), (2, 6))
        self.assertEqual(tuple(head.up.weight.shape), (1, 2))
        composed = head.up.weight @ head.down.weight  # (1, 6)，等效线性映射
        self.assertLessEqual(int(torch.linalg.matrix_rank(composed)), 2)

    def test_param_counts_match_rank_formula(self):
        # 无偏置两层的参数量 = rep_dim * rank + rank
        self.assertEqual(sum(p.numel() for p in ensemble.ResidualHead(64, 8).parameters()), 64 * 8 + 8)
        self.assertEqual(sum(p.numel() for p in ensemble.ResidualHead(64, 16).parameters()), 64 * 16 + 16)


class TestBuildArmHead(unittest.TestCase):
    def test_arm_ranks_and_types(self):
        b = tiny_head("B")
        c = tiny_head("C")
        e = tiny_head("E")
        self.assertIs(type(b), NewTask)  # B = 原始 NewTask（非子类实例）
        for head in (c, e):
            self.assertIsInstance(head, ensemble.EnsembleNewTask)
            self.assertIsInstance(head, NewTask)
        self.assertEqual(tuple(h.rank for h in c.residual_heads), (16,))
        self.assertEqual(tuple(h.rank for h in e.residual_heads), (8, 8))
        self.assertEqual(ensemble.ARM_RANKS, {"B": (), "C": (16,), "E": (8, 8)})

    def test_main_head_bit_identical_across_arms(self):
        b = tiny_head("B")
        c = tiny_head("C")
        e = tiny_head("E")
        b_state, c_state, e_state = b.state_dict(), c.state_dict(), e.state_dict()
        for key, value in b_state.items():
            self.assertTrue(torch.equal(value, c_state[key]), key)
            self.assertTrue(torch.equal(value, e_state[key]), key)

    def test_param_budget_control_matches_ensemble_at_tiny_scale(self):
        report_b = ensemble.param_report(tiny_head("B"))
        report_c = ensemble.param_report(tiny_head("C"))
        report_e = ensemble.param_report(tiny_head("E"))
        # B 无残差头
        self.assertEqual(report_b["residual"], 0)
        self.assertEqual(report_b["main_head"], report_b["trainable"])
        # 主头三臂一致
        self.assertEqual(report_c["main_head"], report_b["main_head"])
        self.assertEqual(report_e["main_head"], report_b["main_head"])
        # C/E 可训练参数量相差 ≤5%（本设计下精确相等；与规模无关，tiny 规模同样成立）
        self.assertLessEqual(abs(report_c["trainable"] - report_e["trainable"]) / min(report_c["trainable"], report_e["trainable"]), 0.05)
        self.assertEqual(report_c["trainable"], report_e["trainable"])
        # 不在此处断言「E < 2×完整 NewTask」：该条件与规模相关，tiny fixture（rep_dim=8、极小 tower）
        # 下残差头占可训练参数多数，条件不成立属预期，不是正式配置的参数结论。硬门禁以正式配置在
        # test_param_budget_gate_holds_at_formal_config 验证，正式运行由 run_arm（enforce_budget=True 默认）强制。

    def test_param_budget_gate_holds_at_formal_config(self):
        # 预注册（spec 2026-10-08「公平配对」）：C/E 可训练参数量相差 ≤5% 且 E 总量 < 两个完整 NewTask，
        # 否则正式运行前 NO-GO。该硬门禁必须在正式 64/32/32 配置（protocol 常量）下成立。
        self.assertEqual(protocol.NEWTASK_REP_DIM, protocol.EXPERT_HIDDEN[-1])  # run_arm 的 rep_dim 前提

        def formal_head(arm):
            return ensemble.build_arm_head(
                arm,
                input_size=protocol.INPUT_SIZE,
                rep_dim=protocol.NEWTASK_REP_DIM,
                tower_dnn_hidden_units=protocol.TOWER_HIDDEN,
                reg_dnn=protocol.REG_DNN,
                device=torch.device("cpu"),
            )

        report_b = ensemble.param_report(formal_head("B"))
        report_c = ensemble.param_report(formal_head("C"))
        report_e = ensemble.param_report(formal_head("E"))
        # 残差头参数量为解析公式：C = 64×16+16 = 1040，E = 2×(64×8+8) = 1040
        self.assertEqual(report_c["residual"], ensemble.residual_param_count(protocol.NEWTASK_REP_DIM, ensemble.ARM_RANKS["C"]))
        self.assertEqual(report_e["residual"], ensemble.residual_param_count(protocol.NEWTASK_REP_DIM, ensemble.ARM_RANKS["E"]))
        # 硬门禁不抛错即 GO（enforce 默认 True，与正式运行一致）
        budget = ensemble.budget_check(
            c_trainable=report_c["trainable"],
            e_trainable=report_e["trainable"],
            newtask_trainable=report_b["trainable"],
        )
        self.assertTrue(budget["c_vs_e_within_5pct"])
        self.assertTrue(budget["e_less_than_two_newtasks"])
        self.assertLess(report_e["trainable"], 2 * report_b["trainable"])

    def test_budget_check_boundaries(self):
        ok = ensemble.budget_check(c_trainable=1000, e_trainable=1049, newtask_trainable=1000)
        self.assertTrue(ok["c_vs_e_within_5pct"])
        with self.assertRaises(AssertionError):
            ensemble.budget_check(c_trainable=1000, e_trainable=1051, newtask_trainable=1000)
        with self.assertRaises(AssertionError):
            ensemble.budget_check(c_trainable=1000, e_trainable=1000, newtask_trainable=500)
        # enforce=False：只记录结论、不抛错（tiny 规模专用）；布尔结论必须如实反映两个失败项
        soft = ensemble.budget_check(c_trainable=1000, e_trainable=1051, newtask_trainable=500, enforce=False)
        self.assertFalse(soft["c_vs_e_within_5pct"])
        self.assertFalse(soft["e_less_than_two_newtasks"])


class TestEnsembleForward(unittest.TestCase):
    def test_zero_residual_output_equals_main_head_bitwise(self):
        for arm in ("C", "E"):
            head = tiny_head(arm)
            inputs = tiny_inputs()
            arm_out = head(*inputs)
            main_out = NewTask.forward(head, *inputs)  # 父类原样路径
            self.assertTrue(torch.equal(arm_out, main_out), arm)

    def test_zeroed_residual_after_random_init_equals_main_head(self):
        head = tiny_head("E")
        for residual in head.residual_heads:
            torch.nn.init.normal_(residual.up.weight, std=1e-3)
        inputs = tiny_inputs()
        self.assertFalse(torch.equal(head(*inputs), NewTask.forward(head, *inputs)))
        for residual in head.residual_heads:
            residual.up.weight.data.zero_()
        self.assertTrue(torch.equal(head(*inputs), NewTask.forward(head, *inputs)))

    def test_members_and_ensemble_mean(self):
        head = tiny_head("E")
        torch.nn.init.normal_(head.residual_heads[0].up.weight, std=1e-3)
        torch.nn.init.normal_(head.residual_heads[1].up.weight, std=2e-3)
        inputs = tiny_inputs()
        members = head.member_probabilities(*inputs)
        self.assertEqual(len(members), 2)
        self.assertFalse(torch.equal(members[0], members[1]))
        arm_out = head(*inputs)
        self.assertTrue(torch.equal(arm_out, torch.stack(members, dim=0).mean(dim=0)))
        self.assertEqual(tuple(arm_out.shape), (16,))

    def test_single_head_arm_output_is_that_head(self):
        head = tiny_head("C")
        torch.nn.init.normal_(head.residual_heads[0].up.weight, std=1e-3)
        inputs = tiny_inputs()
        members = head.member_probabilities(*inputs)
        self.assertEqual(len(members), 1)
        self.assertTrue(torch.equal(head(*inputs), members[0]))


class TestTrainingGradients(unittest.TestCase):
    def test_heads_parameters_disjoint(self):
        head = tiny_head("E")
        pointers = []
        for residual in head.residual_heads:
            pointers.extend(p.data_ptr() for p in residual.parameters())
        self.assertEqual(len(set(pointers)), len(pointers))
        head0_params = {id(p) for p in head.residual_heads[0].parameters()}
        head1_params = {id(p) for p in head.residual_heads[1].parameters()}
        self.assertFalse(head0_params & head1_params)

    def test_each_head_gets_own_nonzero_gradients(self):
        head = tiny_head("E")
        loss_func = nn.BCELoss()
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        inputs = tiny_inputs(batch=32, seed=3)
        labels = tiny_labels(batch=32, seed=4).float()

        def step():
            members = head.member_probabilities(*inputs)
            loss = sum(loss_func(member.cpu(), labels) for member in members) / len(members)
            loss = loss + head.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()
            return loss

        # 第一步反传：up 层（零初始化）即有非零梯度（两头各自、互不共享）
        step()
        for residual in head.residual_heads:
            self.assertIsNotNone(residual.up.weight.grad)
            self.assertGreater(float(residual.up.weight.grad.abs().sum()), 0.0)
            self.assertEqual(float(residual.down.weight.grad.abs().sum()), 0.0)  # up=0 ⇒ 第一步 down 梯度为 0
        self.assertIsNot(head.residual_heads[0].up.weight.grad, head.residual_heads[1].up.weight.grad)
        optimizer.step()
        # 第二步反传：down 层梯度恢复非零（两头都活）
        step()
        for residual in head.residual_heads:
            self.assertGreater(float(residual.up.weight.grad.abs().sum()), 0.0)
            self.assertGreater(float(residual.down.weight.grad.abs().sum()), 0.0)

    def test_frozen_backbone_unchanged_by_head_training(self):
        torch.manual_seed(5)
        model = bench.MPTRec(
            num_tasks=protocol.NUM_TASKS,
            feature_vocabulary=TINY_VOCAB,
            embedding_size=2,
            input_size=4,
            expert_dnn_hidden_units=(8, 4),
            tower_dnn_hidden_units=(4,),
            dropout=None,
            reg_embedding=1e-4,
            reg_dnn=7e-6,
            device=torch.device("cpu"),
        )
        protocol.freeze_backbone(model)
        sha_before = protocol.backbone_sha256(model)

        head = tiny_head("E", seed=9, rep_dim=4)
        head.train()
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        loss_func = nn.BCELoss()
        features = {
            "121": torch.tensor([row[3] for row in TRAIN_ROWS]),
            "122": torch.tensor([row[4] for row in TRAIN_ROWS]),
        }
        labels = torch.tensor([row[5] for row in TRAIN_ROWS]).float()
        labels = torch.where(labels == 2, torch.zeros_like(labels), torch.ones_like(labels))

        with torch.no_grad():  # 计算图级冻结
            dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
        members = head.member_probabilities(dnn_input, gen_rep, spec_reps, env_embs)
        loss = sum(loss_func(member.cpu(), labels) for member in members) / len(members) + head.get_l2_reg()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertEqual(protocol.backbone_sha256(model), sha_before)  # 参数/SHA 不变
        protocol.assert_no_grads(model)  # 梯度全为 None
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))


class TestMechanismStats(unittest.TestCase):
    def test_identical_members_flagged_collapsed(self):
        stats = ensemble.mechanism_stats([torch.tensor([0.5, 0.6, 0.7]), torch.tensor([0.5, 0.6, 0.7])])
        self.assertTrue(stats["exactly_identical"])
        self.assertTrue(stats["collapsed"])
        self.assertEqual(stats["mean_abs_diff"], 0.0)

    def test_below_tolerance_flagged_collapsed(self):
        base = torch.tensor([0.5, 0.6, 0.7])
        stats = ensemble.mechanism_stats([base, base + 5e-5])
        self.assertFalse(stats["exactly_identical"])
        self.assertTrue(stats["collapsed"])
        self.assertLess(stats["mean_abs_diff"], ensemble.RESIDUAL_COLLAPSE_TOL)

    def test_distinct_members_not_collapsed_and_stats_fields(self):
        base = torch.linspace(0.1, 0.9, 64)
        other = base + torch.linspace(-2e-3, 2e-3, 64)
        stats = ensemble.mechanism_stats([base, other])
        self.assertFalse(stats["collapsed"])
        self.assertGreaterEqual(stats["mean_abs_diff"], ensemble.RESIDUAL_COLLAPSE_TOL)
        self.assertGreater(stats["max_abs_diff"], stats["mean_abs_diff"])
        self.assertGreater(stats["q90_abs_diff"], stats["q50_abs_diff"])
        self.assertAlmostEqual(stats["corr"], 1.0, places=2)
        self.assertEqual(stats["tol"], ensemble.RESIDUAL_COLLAPSE_TOL)

    def test_single_member_returns_none(self):
        self.assertIsNone(ensemble.mechanism_stats([torch.tensor([0.5, 0.6])]))


class TestScreenVerdict(unittest.TestCase):
    def verdict(self, **overrides):
        kwargs = {
            "delta_eb": 0.002,
            "delta_ec": 0.002,
            "val_direction_consistent": True,
            "collapsed": False,
            "gates_ok": True,
        }
        kwargs.update(overrides)
        return ensemble.screen_verdict(**kwargs)

    def test_positive_boundary_inclusive(self):
        self.assertEqual(ensemble.DELTA_POSITIVE_MIN, 0.001)
        self.assertEqual(self.verdict(delta_eb=0.001)["classification"], "POSITIVE_IMPROVEMENT")

    def test_no_clear_band(self):
        self.assertEqual(self.verdict(delta_eb=0.0009)["classification"], "NO_CLEAR_IMPROVEMENT")
        self.assertEqual(self.verdict(delta_eb=-0.0199)["classification"], "NO_CLEAR_IMPROVEMENT")

    def test_decline_boundary_inclusive(self):
        self.assertEqual(ensemble.DELTA_DECLINE_MAX, -0.02)
        self.assertEqual(self.verdict(delta_eb=-0.02)["classification"], "CLEAR_DECLINE")

    def test_expansion_requires_all_conditions(self):
        self.assertTrue(self.verdict()["expansion_eligible"])
        self.assertFalse(self.verdict(delta_ec=0.0009)["expansion_eligible"])
        self.assertFalse(self.verdict(delta_eb=0.0009)["expansion_eligible"])
        self.assertFalse(self.verdict(val_direction_consistent=False)["expansion_eligible"])
        self.assertFalse(self.verdict(collapsed=True)["expansion_eligible"])
        self.assertFalse(self.verdict(collapsed=None)["expansion_eligible"])  # 机制未知不算未塌缩
        self.assertFalse(self.verdict(gates_ok=False)["expansion_eligible"])


class TestRunIdHelper(unittest.TestCase):
    def test_arm_run_id_suffix(self):
        run_id = ensemble.make_arm_run_id(
            datetime(2026, 10, 8, 14, 30),
            prefix_tag=protocol.PREFIX_TAG,
            model_seed=protocol.MODEL_SEED,
            tag="short",
            commit="abc1234",
            arm="E",
        )
        self.assertEqual(run_id, "20261008-1430-p2M-v500k-t1M-m1688723512-short-abc1234-ens-E")


def fabricate_run_dir(base: Path, arm: str, *, stage1_id: str = DEFAULT_STAGE1_ID,
                      test_labels=None, test_members=None, val_labels=None, val_epoch_members=None,
                      model_seed: int = protocol.MODEL_SEED, prefix_tag: str = protocol.PREFIX_TAG,
                      commit: str = "abc1234", dirty: bool = False,
                      fingerprint_sha256: str = DEFAULT_FP_SHA, backbone_sha256: str = DEFAULT_BB_SHA,
                      budgets=None, epochs: int = 5, patience: int = 2, tag: str = "smoke",
                      tamper=None) -> Path:
    """按 run_arm 的落盘 schema 构造合成 run 目录（用于独立解析器测试）。

    身份字段（seed/前缀/stage1/commit+git/指纹/backbone SHA/预算/epochs/patience/run_id/tag）默认
    在 config/metrics/predictions 三份产物中逐项同值；臂输出由成员概率均值派生（与 run_arm 同式）。
    tamper 模拟篡改：{文档: {字段: 新值或 f(旧值)}}，文档 ∈ config/metrics/pred/gate/pred_test/pred_val。
    """
    run_dir = base / f"run-{arm}"
    run_dir.mkdir(parents=True)
    if test_labels is None:
        test_labels = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0])
    if val_labels is None:
        val_labels = torch.tensor([1, 0, 1, 1, 1, 0])
    if test_members is None:
        base_probs = torch.linspace(0.05, 0.95, len(test_labels))
        test_members = [base_probs - 0.01, base_probs + 0.01] if arm == "E" else [base_probs]
    if val_epoch_members is None:
        flat = torch.full((len(val_labels),), 0.5)
        aligned = torch.tensor([0.9, 0.1, 0.8, 0.7, 0.6, 0.2])[: len(val_labels)]
        val_epoch_members = [([p - 0.01, p + 0.01] if arm == "E" else [p]) for p in (flat, aligned)]
    if budgets is None:
        budgets = {"train": 12, "val": len(val_labels), "test": len(test_labels)}

    def arm_mean(members):
        return torch.stack([m.float() for m in members], dim=0).mean(dim=0)

    val_epoch_records = [
        {"arm": arm_mean(ms), "members": [m.float() for m in ms]} for ms in val_epoch_members
    ]
    predictions = {
        "run_id": run_dir.name,
        "tag": tag,
        "arm": arm,
        "model_seed": model_seed,
        "prefix_tag": prefix_tag,
        "stage1_id": stage1_id,
        "commit": commit,
        "git": {"commit": commit, "dirty": dirty},
        "fingerprint_sha256": fingerprint_sha256,
        "backbone_sha256": backbone_sha256,
        "budgets": dict(budgets),
        "epochs": epochs,
        "patience": patience,
        "test": {"labels": test_labels.long(), "arm": arm_mean(test_members),
                 "members": [m.float() for m in test_members]},
        "val": {"labels": val_labels.long(), "epochs": val_epoch_records},
    }
    val_aucs = [metrics.auc_score(val_labels, record["arm"]) for record in val_epoch_records]
    best_epoch = int(max(range(len(val_aucs)), key=lambda i: val_aucs[i])) + 1
    metrics_doc = {
        "run_id": run_dir.name,
        "tag": tag,
        "arm": arm,
        "stage1_id": stage1_id,
        "prefix_tag": prefix_tag,
        "budgets": dict(budgets),
        "model_seed": model_seed,
        "epochs": epochs,
        "patience": patience,
        "best_epoch": best_epoch,
        "best_val_auc_bsi": max(val_aucs),
        "test_auc_bsi": metrics.auc_score(test_labels, predictions["test"]["arm"]),
        "per_epoch": [{"epoch": i + 1, "val_auc_bsi": auc} for i, auc in enumerate(val_aucs)],
        "backbone_sha256_loaded": backbone_sha256,
        "backbone_sha256_before": backbone_sha256,
        "backbone_sha256_after": backbone_sha256,
        "fingerprint_sha256": fingerprint_sha256,
        "hard_pass": True,
        "commit": commit,
        "git": {"commit": commit, "dirty": dirty},
    }
    config_doc = {
        "run_id": run_dir.name,
        "tag": tag,
        "arm": arm,
        "stage1_id": stage1_id,
        "prefix_tag": prefix_tag,
        "budgets": dict(budgets),
        "model_seed": model_seed,
        "epochs": epochs,
        "patience": patience,
        "fingerprint_sha256": fingerprint_sha256,
        "backbone_sha256": backbone_sha256,
        "commit": commit,
        "git": {"commit": commit, "dirty": dirty},
    }
    gate_doc = {
        "run_id": run_dir.name,
        "tag": tag,
        "arm": arm,
        "enforce_b": False,
        "gates": {"A1": {"verdict": "PASS"}, "A3": {"verdict": "SKIP"}},
        "hard_pass": True,
    }
    for key, doc in (("config", config_doc), ("metrics", metrics_doc), ("pred", predictions), ("gate", gate_doc)):
        for field, value in (tamper or {}).get(key, {}).items():
            doc[field] = value(doc.get(field)) if callable(value) else value
    for key, section in (("pred_test", predictions["test"]), ("pred_val", predictions["val"])):
        for field, value in (tamper or {}).get(key, {}).items():
            section[field] = value(section.get(field)) if callable(value) else value
    torch.save(predictions, run_dir / "predictions.pt")
    (run_dir / "metrics.json").write_text(json.dumps(metrics_doc), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config_doc), encoding="utf-8")
    (run_dir / "gate_report.json").write_text(json.dumps(gate_doc), encoding="utf-8")
    return run_dir


class TestRederiveAndCompare(unittest.TestCase):
    def test_rederive_recomputes_and_cross_checks(self):
        with tempfile.TemporaryDirectory() as td:
            # 与 run_arm 落盘语义一致：arm = 成员均值（(0.5 + 0.6) / 2 = 0.55）
            run_dir = fabricate_run_dir(
                Path(td), "E",
                test_members=[torch.full((8,), 0.5), torch.full((8,), 0.6)],
            )
            report = ensemble.rederive_arm(run_dir)
            self.assertTrue(report["all_ok"])
            self.assertEqual(report["n_test"], 8)
            self.assertEqual(report["arm"], "E")
            expected = metrics.auc_score(torch.tensor([0, 1, 1, 0, 1, 0, 1, 0]), torch.full((8,), 0.55))
            self.assertAlmostEqual(report["auc_test_bsi"], expected, places=12)
            self.assertEqual(report["best_epoch"], 2)
            self.assertIsNotNone(report["mechanism"])

    def test_rederive_detects_tampered_metrics(self):
        cases = {
            "auc_matches_metrics_json": {"metrics": {"test_auc_bsi": lambda v: v + 0.01}},
            "best_epoch_matches": {"metrics": {"best_epoch": lambda v: v + 1}},
            "val_aucs_match": {
                "metrics": {
                    "per_epoch": lambda rows: [dict(rows[0], val_auc_bsi=rows[0]["val_auc_bsi"] + 0.01)] + rows[1:]
                }
            },
        }
        for flag, tamper in cases.items():
            with tempfile.TemporaryDirectory() as td:
                run_dir = fabricate_run_dir(Path(td), "B", tamper=tamper)
                report = ensemble.rederive_arm(run_dir)
                self.assertFalse(report["checks"][flag], flag)
                self.assertFalse(report["all_ok"], flag)

    def test_compare_arms_paired_deltas_and_identity(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            labels = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0])
            b_probs = torch.tensor([0.1, 0.2, 0.8, 0.3, 0.7, 0.2, 0.9, 0.1])
            c_probs = torch.tensor([0.1, 0.3, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2])
            e_probs = torch.tensor([0.1, 0.4, 0.9, 0.2, 0.9, 0.1, 0.95, 0.1])
            dirs = [
                fabricate_run_dir(base, "B", test_labels=labels, test_members=[b_probs]),
                fabricate_run_dir(base, "C", test_labels=labels, test_members=[c_probs]),
                fabricate_run_dir(base, "E", test_labels=labels,
                                  test_members=[e_probs - 0.01, e_probs + 0.01]),
            ]
            report = ensemble.compare_arms(dirs)
            self.assertTrue(report["paired"])
            self.assertTrue(report["stage1_consistent"])
            self.assertTrue(report["identifiers_consistent"])
            self.assertTrue(report["labels_consistent"])
            for left, right in (("E", "B"), ("E", "C"), ("C", "B")):
                delta = report["deltas"][f"{left}-{right}"]["test_auc_bsi"]
                expected = report["arms"][left]["auc_test_bsi"] - report["arms"][right]["auc_test_bsi"]
                self.assertAlmostEqual(delta, expected, places=12)
            self.assertTrue(report["arms"]["B"]["all_ok"])
            self.assertIn("screen", report)

    def test_compare_arms_flags_stage1_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            dirs = [
                fabricate_run_dir(base, "B"),
                fabricate_run_dir(base, "C"),
                fabricate_run_dir(Path(td) / "other", "E", stage1_id="s1-different-m123-e1-cafe0000"),
            ]
            report = ensemble.compare_arms(dirs)
            self.assertFalse(report["stage1_consistent"])
            self.assertFalse(report["identity_checks"]["stage1_id"])
            self.assertFalse(report["identifiers_consistent"])
            self.assertFalse(report["screen"]["expansion_eligible"])


class TestRederiveIdentityGuards(unittest.TestCase):
    """身份字段 config/metrics/predictions 三方互证 + 结构数值守卫，逐项篡改单测。"""

    IDENTITY_CASES = {
        "model_seed": ("model_seed_consistent", 999),
        "prefix_tag": ("prefix_tag_consistent", "other-prefix"),
        "stage1_id": ("stage1_id_consistent", "s1-other-m999-e9-deadbeef"),
        "commit": ("commit_consistent", "fffffff"),
        "fingerprint_sha256": ("fingerprint_sha256_consistent", "0" * 64),
        "backbone_sha256": ("backbone_sha256_consistent", "1" * 64),
        "epochs": ("epochs_consistent", 7),
        "patience": ("patience_consistent", 3),
        "budgets": ("budgets_consistent", {"train": 12, "val": 99, "test": 8}),
    }

    def test_each_doc_tamper_of_identity_fields_detected(self):
        for doc in ("config", "metrics", "pred"):
            for field, (flag, value) in self.IDENTITY_CASES.items():
                key = "backbone_sha256_loaded" if (doc == "metrics" and field == "backbone_sha256") else field
                with tempfile.TemporaryDirectory() as td:
                    run_dir = fabricate_run_dir(Path(td), "B", tamper={doc: {key: value}})
                    report = ensemble.rederive_arm(run_dir)
                    self.assertFalse(report["checks"][flag], f"{doc}:{field}")
                    self.assertFalse(report["all_ok"], f"{doc}:{field}")

    def test_run_id_tag_arm_tamper_in_any_doc_detected(self):
        cases = {
            "run_id": ("run_id_consistent", "other-run"),
            "tag": ("tag_consistent", "other-tag"),
            "arm": ("arm_consistent", "C"),
        }
        for doc in ("config", "metrics", "pred", "gate"):
            for field, (flag, value) in cases.items():
                with tempfile.TemporaryDirectory() as td:
                    run_dir = fabricate_run_dir(Path(td), "B", tamper={doc: {field: value}})
                    report = ensemble.rederive_arm(run_dir)
                    self.assertFalse(report["checks"][flag], f"{doc}:{field}")
                    self.assertFalse(report["all_ok"], f"{doc}:{field}")

    def test_git_and_hard_pass_selfclaims_detected(self):
        cases = [
            ({"metrics": {"git": {"commit": "abc1234", "dirty": True}}}, "git_clean_all"),
            ({"pred": {"git": {"commit": "abc1234", "dirty": True}}}, "git_clean_all"),
            ({"config": {"git": {"commit": "fffffff", "dirty": False}}}, "commit_consistent"),
            ({"pred": {"git": {"commit": "oldcomm", "dirty": False}}}, "commit_consistent"),
            ({"gate": {"hard_pass": False}}, "hard_pass_consistent"),
            ({"metrics": {"hard_pass": False}}, "hard_pass_consistent"),
            ({"gate": {"gates": {"A1": {"verdict": "FAIL"}, "A3": {"verdict": "SKIP"}}}},
             "hard_pass_consistent"),
        ]
        for tamper, flag in cases:
            with tempfile.TemporaryDirectory() as td:
                run_dir = fabricate_run_dir(Path(td), "B", tamper=tamper)
                report = ensemble.rederive_arm(run_dir)
                self.assertFalse(report["checks"][flag], tamper)
                self.assertFalse(report["all_ok"], tamper)

    def test_checks_include_identity_and_payload_flags_on_pristine_run(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = fabricate_run_dir(Path(td), "B")
            report = ensemble.rederive_arm(run_dir)
            for name in ("run_id_consistent", "tag_consistent", "arm_consistent",
                         "model_seed_consistent", "prefix_tag_consistent", "stage1_id_consistent",
                         "commit_consistent", "git_clean_all", "fingerprint_sha256_consistent",
                         "backbone_sha256_consistent", "budgets_consistent", "epochs_consistent",
                         "patience_consistent", "hard_pass_consistent",
                         "n_test_matches_budget", "n_val_matches_budget",
                         "test_labels_binary", "val_labels_binary",
                         "test_arrays_aligned", "val_epochs_arrays_aligned",
                         "test_probs_finite", "val_probs_finite",
                         "test_probs_in_unit_range", "val_probs_in_unit_range",
                         "member_count_matches_arm", "test_arm_equals_member_mean",
                         "val_arms_equal_member_mean", "val_epoch_count_matches_per_epoch"):
                self.assertIn(name, report["checks"], name)
                self.assertTrue(report["checks"][name], name)
            self.assertTrue(report["all_ok"])
            # 无 <root>/runs 布局 ⇒ 锚定检查不适用（None，而非默认放行 True）
            self.assertIsNone(report["artifact_root"])
            self.assertIsNone(report["checks"]["fingerprint_anchor_matches"])
            self.assertIsNone(report["checks"]["stage1_meta_matches"])
            self.assertIsNone(report["checks"]["backbone_recompute_matches"])
            self.assertEqual(report["identity"]["stage1_id"], DEFAULT_STAGE1_ID)
            self.assertEqual(report["identity"]["budgets"], {"train": 12, "val": 6, "test": 8})
            self.assertEqual(report["n_val"], 6)
            self.assertIsNotNone(report["test_labels_sha256"])
            self.assertIsNotNone(report["val_labels_sha256"])


class TestRederiveRawPayloadGuards(unittest.TestCase):
    """raw 载荷守卫：长度截断 / NaN / 非二值标签 / 成员数错配 / 均值不自洽 / val epochs 错位。"""

    def _report(self, td, arm="B", **kwargs):
        return ensemble.rederive_arm(fabricate_run_dir(Path(td), arm, **kwargs))

    def test_truncated_test_labels_detected(self):
        with tempfile.TemporaryDirectory() as td:
            report = self._report(td, tamper={"pred_test": {"labels": torch.tensor([0, 1, 1, 0, 1, 0, 1])}})
            self.assertFalse(report["checks"]["n_test_matches_budget"])
            self.assertFalse(report["checks"]["test_arrays_aligned"])
            self.assertFalse(report["all_ok"])

    def test_truncated_val_labels_detected(self):
        with tempfile.TemporaryDirectory() as td:
            report = self._report(td, tamper={"pred_val": {"labels": torch.tensor([1, 0, 1, 1, 1])}})
            self.assertFalse(report["checks"]["n_val_matches_budget"])
            self.assertFalse(report["checks"]["val_epochs_arrays_aligned"])
            self.assertFalse(report["all_ok"])

    def test_nan_probabilities_detected(self):
        nan = float("nan")
        with tempfile.TemporaryDirectory() as td:
            report = self._report(td, tamper={"pred_test": {"arm": torch.full((8,), nan)}})
            self.assertFalse(report["checks"]["test_probs_finite"])
            self.assertFalse(report["checks"]["test_probs_in_unit_range"])
            self.assertFalse(report["all_ok"])
        with tempfile.TemporaryDirectory() as td:
            report = self._report(td, arm="E", tamper={"pred_test": {"members": [torch.full((8,), nan),
                                                                                torch.full((8,), nan)]}})
            self.assertFalse(report["checks"]["test_probs_finite"])
            self.assertFalse(report["all_ok"])

    def test_non_binary_labels_detected(self):
        with tempfile.TemporaryDirectory() as td:
            report = self._report(td, tamper={"pred_test": {"labels": torch.tensor([0, 1, 1, 0, 1, 2, 1, 0])}})
            self.assertFalse(report["checks"]["test_labels_binary"])
            self.assertFalse(report["all_ok"])

    def test_member_count_and_mean_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as td:
            # E 臂成员数被改成 1（期望 2）
            report = self._report(td, arm="E", tamper={"pred_test": {"members": [torch.full((8,), 0.5)]}})
            self.assertFalse(report["checks"]["member_count_matches_arm"])
            self.assertFalse(report["checks"]["test_arm_equals_member_mean"])
            self.assertFalse(report["all_ok"])
        with tempfile.TemporaryDirectory() as td:
            # 臂输出不再是成员均值（成员未动）
            report = self._report(td, tamper={"pred_test": {"arm": torch.full((8,), 0.2)}})
            self.assertFalse(report["checks"]["test_arm_equals_member_mean"])
            self.assertFalse(report["all_ok"])

    def test_val_epochs_count_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as td:
            report = self._report(td, tamper={"pred_val": {"epochs": lambda rows: rows[:1]}})
            self.assertFalse(report["checks"]["val_epoch_count_matches_per_epoch"])
            self.assertFalse(report["checks"]["val_aucs_match"])
            self.assertFalse(report["all_ok"])

    def test_best_epoch_out_of_range_detected(self):
        with tempfile.TemporaryDirectory() as td:
            report = self._report(td, tamper={"metrics": {"best_epoch": 9}})
            self.assertFalse(report["checks"]["epoch_records_wellformed"])
            self.assertFalse(report["checks"]["best_epoch_matches"])
            self.assertFalse(report["all_ok"])


class TestRederiveArtifactAnchors(unittest.TestCase):
    """产物树锚定：仅信任 raw 文件（指纹自哈希 / Stage-1 meta / backbone.pt 字节重算）。"""

    def _anchor_fixture(self, td):
        root, data_files, budgets = make_tiny_root(td)
        meta = run_tiny_stage1_at(root, data_files, budgets)
        fp = protocol.load_fingerprint(root, "tiny")
        return root, meta, fp, budgets

    def _fabricate(self, root, meta, fp, budgets, arm="B", tamper=None):
        return fabricate_run_dir(
            root / "runs", arm,
            stage1_id=meta["stage1_id"], prefix_tag="tiny", model_seed=123,
            fingerprint_sha256=fp["fingerprint_sha256"], backbone_sha256=meta["backbone_sha256"],
            budgets=budgets, tamper=tamper,
        )

    def test_anchors_live_and_pass_on_pristine_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root, meta, fp, budgets = self._anchor_fixture(td)
            report = ensemble.rederive_arm(self._fabricate(root, meta, fp, budgets))
            self.assertEqual(report["artifact_root"], str(root))
            self.assertTrue(report["checks"]["fingerprint_anchor_matches"])
            self.assertTrue(report["checks"]["stage1_meta_matches"])
            self.assertTrue(report["checks"]["backbone_recompute_matches"])
            self.assertTrue(report["all_ok"])

    def test_consistent_three_way_backbone_lie_caught_by_byte_recompute(self):
        with tempfile.TemporaryDirectory() as td:
            root, meta, fp, budgets = self._anchor_fixture(td)
            lie = "0" * 64
            run_dir = self._fabricate(root, meta, fp, budgets, tamper={
                "config": {"backbone_sha256": lie},
                "metrics": {"backbone_sha256_loaded": lie, "backbone_sha256_before": lie,
                            "backbone_sha256_after": lie},
                "pred": {"backbone_sha256": lie},
            })
            report = ensemble.rederive_arm(run_dir)
            # 三方自述完全一致（互证通过）——但字节级重算不采信自述
            self.assertTrue(report["checks"]["backbone_sha256_consistent"])
            self.assertFalse(report["checks"]["backbone_recompute_matches"])
            self.assertFalse(report["all_ok"])

    def test_perturbed_backbone_bytes_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root, meta, fp, budgets = self._anchor_fixture(td)
            run_dir = self._fabricate(root, meta, fp, budgets)
            self.assertTrue(ensemble.rederive_arm(run_dir)["all_ok"])
            backbone_path = protocol.stage1_dir(root, meta["stage1_id"]) / "backbone.pt"
            state = torch.load(backbone_path, map_location="cpu")
            key = sorted(k for k, v in state.items() if v.dtype.is_floating_point)[0]
            state[key] = state[key] + 1.0
            torch.save(state, backbone_path)
            report = ensemble.rederive_arm(run_dir)
            self.assertFalse(report["checks"]["backbone_recompute_matches"])
            self.assertFalse(report["all_ok"])

    def test_consistent_fingerprint_lie_caught_by_file_anchor(self):
        with tempfile.TemporaryDirectory() as td:
            root, meta, fp, budgets = self._anchor_fixture(td)
            lie = "9" * 64
            run_dir = self._fabricate(root, meta, fp, budgets, tamper={
                "config": {"fingerprint_sha256": lie},
                "metrics": {"fingerprint_sha256": lie},
                "pred": {"fingerprint_sha256": lie},
            })
            report = ensemble.rederive_arm(run_dir)
            self.assertTrue(report["checks"]["fingerprint_sha256_consistent"])
            self.assertFalse(report["checks"]["fingerprint_anchor_matches"])
            self.assertFalse(report["checks"]["stage1_meta_matches"])
            self.assertFalse(report["all_ok"])

    def test_tampered_fingerprint_file_fails_self_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root, meta, fp, budgets = self._anchor_fixture(td)
            run_dir = self._fabricate(root, meta, fp, budgets)
            fp_path = protocol.splits_dir(root, "tiny") / "prefix_fingerprint.json"
            doc = json.loads(fp_path.read_text(encoding="utf-8"))
            doc["label_counts"]["val"]["n"] += 1
            fp_path.write_text(json.dumps(doc), encoding="utf-8")
            report = ensemble.rederive_arm(run_dir)
            self.assertFalse(report["checks"]["fingerprint_anchor_matches"])
            self.assertFalse(report["all_ok"])

    def test_consistent_fake_stage1_id_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root, meta, fp, budgets = self._anchor_fixture(td)
            fake = "s1-fake-m123-e1-00000000"
            run_dir = self._fabricate(root, meta, fp, budgets, tamper={
                "config": {"stage1_id": fake},
                "metrics": {"stage1_id": fake},
                "pred": {"stage1_id": fake},
            })
            report = ensemble.rederive_arm(run_dir)
            # 三方一致指向不存在的 Stage-1 ⇒ 锚定检查（强制项）必须失败
            self.assertTrue(report["checks"]["stage1_id_consistent"])
            self.assertFalse(report["checks"]["stage1_meta_matches"])
            self.assertFalse(report["checks"]["backbone_recompute_matches"])
            self.assertFalse(report["all_ok"])


class TestCompareArmsCrossArmGuards(unittest.TestCase):
    """三臂间共享标识必须逐项一致；test/val 标签必须逐项相同（同长度不同内容也须被识破）。"""

    def _report(self, td, *, e_kwargs=None):
        base = Path(td)
        dirs = [
            fabricate_run_dir(base, "B"),
            fabricate_run_dir(base, "C"),
            fabricate_run_dir(base, "E", **(e_kwargs or {})),
        ]
        return ensemble.compare_arms(dirs)

    def test_identifier_divergence_blocks_expansion(self):
        cases = {
            "model_seed": {"model_seed": 999},
            "prefix_tag": {"prefix_tag": "other-prefix"},
            "fingerprint_sha256": {"fingerprint_sha256": "0" * 64},
            "backbone_sha256": {"backbone_sha256": "1" * 64},
            "epochs": {"epochs": 4},
            "patience": {"patience": 3},
            "commit": {"commit": "fffffff"},
            "stage1_id": {"stage1_id": "s1-other-m123-e1-deadbeef"},
            "budgets": {"budgets": {"train": 12, "val": 5, "test": 8},
                        "val_labels": torch.tensor([1, 0, 1, 1, 1])},
        }
        for field, e_kwargs in cases.items():
            with tempfile.TemporaryDirectory() as td:
                report = self._report(td, e_kwargs=e_kwargs)
                self.assertFalse(report["identity_checks"][field], field)
                self.assertFalse(report["identifiers_consistent"], field)
                self.assertFalse(report["screen"]["expansion_eligible"], field)

    def test_test_label_value_divergence_same_length_detected(self):
        with tempfile.TemporaryDirectory() as td:
            report = self._report(td, e_kwargs={"test_labels": torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])})
            self.assertTrue(report["identifiers_consistent"])
            self.assertFalse(report["label_checks"]["test_labels_identical"])
            self.assertTrue(report["label_checks"]["test_lengths_equal"])
            self.assertFalse(report["labels_consistent"])
            self.assertFalse(report["screen"]["expansion_eligible"])

    def test_val_label_divergence_detected(self):
        with tempfile.TemporaryDirectory() as td:
            report = self._report(td, e_kwargs={"val_labels": torch.tensor([0, 1, 1, 1, 1, 0])})
            self.assertTrue(report["identifiers_consistent"])
            self.assertFalse(report["label_checks"]["val_labels_identical"])
            self.assertFalse(report["labels_consistent"])
            self.assertFalse(report["screen"]["expansion_eligible"])


class TestRunArmGuards(unittest.TestCase):
    def test_run_arm_refuses_dirty_tree_when_required(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = make_tiny_root(td)
            with mock.patch.object(protocol, "git_state", return_value={"commit": "abc1234", "dirty": True}):
                with self.assertRaises(RuntimeError):
                    ensemble.run_arm(
                        root=root, arm="B", stage1_id="s1-nonexistent",
                        data_files=data_files, budgets=budgets, prefix_tag="tiny",
                        model_seed=123, epochs=1, patience=1, tag="smoke",
                        device=torch.device("cpu"), vocab=TINY_VOCAB,
                        expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
                        require_clean=True, log=noop_log,
                    )

    def test_run_arm_rejects_unknown_arm(self):
        with self.assertRaises(KeyError):
            ensemble.run_arm(
                root=Path("unused"), arm="Z", stage1_id="s1-x",
                data_files=protocol.DATA_FILES, budgets={"train": 1, "val": 1, "test": 1},
                prefix_tag="tiny", model_seed=1, epochs=1, patience=1, tag="smoke",
                device=torch.device("cpu"), log=noop_log,
            )


class TestEndToEndArms(unittest.TestCase):
    def _tiny_stage1(self, root, data_files, budgets):
        return run_tiny_stage1_at(root, data_files, budgets)

    def test_all_three_arms_run_end_to_end_and_rederive(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_files, budgets = make_tiny_root(td)
            stage1_meta = self._tiny_stage1(root, data_files, budgets)
            results = {}
            for arm in ("B", "C", "E"):
                # 测试与工作树脏净解耦：git 溯源打桩为已提交状态（dirty 行为另有单测）
                with mock.patch.object(protocol, "git_state", return_value={"commit": "deadbee", "dirty": False}):
                    result = ensemble.run_arm(
                        root=root, arm=arm, stage1_id=stage1_meta["stage1_id"],
                        data_files=data_files, budgets=budgets, prefix_tag="tiny",
                        model_seed=123, epochs=2, patience=2, tag="smoke",
                        device=torch.device("cpu"), vocab=TINY_VOCAB,
                        expert_hidden=(4,), tower_hidden=(4,), embedding_size=2, input_size=4,
                        # tiny fixture 不适用「E < 2×完整 NewTask」检查（规模相关：残差头占比过高）；
                        # 硬门禁在正式配置单测与 run_arm 默认（enforce_budget=True）中保留
                        enforce_budget=False,
                        log=noop_log,
                    )
                results[arm] = result
                run_dir = protocol.run_dir(root, result["run_id"])
                for name in ("config.json", "metrics.json", "gate_report.json", "newtask.pt", "predictions.pt"):
                    self.assertTrue((run_dir / name).exists(), f"{arm}:{name}")
                # A 类硬门禁必须全 PASS（A3 SKIP），smoke 不判 B 类
                gates = result["gates"]
                for gate_id in ("A1", "A2", "A4", "A5", "A6"):
                    self.assertEqual(gates[gate_id]["verdict"], "PASS", f"{arm}:{gate_id}")
                self.assertTrue(result["hard_pass"], arm)
                # 原始预测可独立重算
                rederived = ensemble.rederive_arm(run_dir)
                self.assertTrue(rederived["all_ok"], arm)
                # 三臂同一 stage1
                self.assertEqual(rederived["stage1_id"], stage1_meta["stage1_id"])
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(config["arm"], arm)
                self.assertFalse(config["enforce_budget"])  # tiny 测试显式关闭，须如实记录
                if arm == "B":
                    self.assertIsNone(config["budget_report"])
                else:
                    # 未强制也要如实落盘 tiny 规模下的真实结论（不构成正式配置的参数结论）
                    self.assertTrue(config["budget_report"]["c_vs_e_within_5pct"])
                    self.assertFalse(config["budget_report"]["e_less_than_two_newtasks"])
                metrics_doc = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                self.assertIn("git", metrics_doc)
                self.assertEqual(len(metrics_doc["per_epoch"]), 2)
                # 真实性守卫：身份字段三份产物逐项同值落盘
                predictions_doc = torch.load(run_dir / "predictions.pt", map_location="cpu")
                self.assertEqual(predictions_doc["run_id"], result["run_id"])
                self.assertEqual(predictions_doc["tag"], "smoke")
                self.assertEqual(predictions_doc["model_seed"], 123)
                self.assertEqual(predictions_doc["prefix_tag"], "tiny")
                self.assertEqual(predictions_doc["stage1_id"], stage1_meta["stage1_id"])
                self.assertEqual(predictions_doc["epochs"], 2)
                self.assertEqual(predictions_doc["patience"], 2)
                self.assertEqual(predictions_doc["commit"], "deadbee")
                self.assertEqual(predictions_doc["git"], {"commit": "deadbee", "dirty": False})
                self.assertEqual(predictions_doc["fingerprint_sha256"], config["fingerprint_sha256"])
                self.assertEqual(predictions_doc["backbone_sha256"], metrics_doc["backbone_sha256_after"])
                self.assertEqual(config["commit"], "deadbee")
                self.assertEqual(config["backbone_sha256"], metrics_doc["backbone_sha256_loaded"])
            # E 臂有塌缩判定；B/C 无
            e_metrics = json.loads((protocol.run_dir(root, results["E"]["run_id"]) / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("mechanism", e_metrics)
            self.assertIsInstance(e_metrics["mechanism"]["collapsed"], bool)
            c_metrics = json.loads((protocol.run_dir(root, results["C"]["run_id"]) / "metrics.json").read_text(encoding="utf-8"))
            self.assertIsNone(c_metrics["mechanism"])
            # 不写 SUMMARY（由独立复核后另行处置）
            self.assertFalse((root / "SUMMARY.md").exists())
            # 三臂配对核验可用，且共享标识一致、标签逐项相同
            report = ensemble.compare_arms([protocol.run_dir(root, results[arm]["run_id"]) for arm in ("B", "C", "E")])
            self.assertTrue(report["paired"])
            self.assertTrue(report["stage1_consistent"])
            self.assertTrue(report["identifiers_consistent"])
            self.assertTrue(all(report["identity_checks"].values()))
            self.assertTrue(report["labels_consistent"])
            self.assertTrue(all(report["label_checks"].values()))
            for arm_report in report["arms"].values():
                self.assertTrue(arm_report["all_ok"], arm_report["arm"])
                # 产物树锚定检查在 <root>/runs 布局下为强制项且必须通过
                self.assertTrue(arm_report["checks"]["fingerprint_anchor_matches"])
                self.assertTrue(arm_report["checks"]["stage1_meta_matches"])
                self.assertTrue(arm_report["checks"]["backbone_recompute_matches"])


if __name__ == "__main__":
    unittest.main()
