"""router_probe 前置诊断测试。

唯一事实来源：docs/superpowers/specs/2026-09-30-stage2-cond-scale-bias-design.md
本测试文件在实现之前写成（TDD：先失败），锁定五件事：
  1. 复算式与 `NewTask.forward` 内部的 router W 逐位一致（spec 2.2）
  2. 四项指标的数学定义（spec 3.1）
  3. 预注册门槛 `router_w_std >= 0.02` 含等号（spec 3.2）
  4. 零训练 / 确定性 / 只读 checkpoint / 落盘字段（spec 2.2、4）
  5. 模块源码可被本解释器编译（3.10 下跨物理行的 f-string 是语法错误，import 即崩）

诊断对象是 `--newtask-checkpoint` 指向的**基线 run 训练后的新任务头**（spec 2.1）：构造顺序仍按
stage2 复现，随后 strict 载入；checkpoint 缺失 / 键或形状不符一律拒绝运行，绝不静默换成随机初始化头。
"""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import run_census_benchmark
from census_benchmark import protocol as P
from census_benchmark import router_probe
from multitaskrec.model import NewTask

# 与 test_protocol / test_smoke 同口径：dnn_input 维度 = 2 个特征 × embedding_size 4 = 8
TINY_VOCAB, TINY_INPUT_SIZE, TINY_EMBEDDING, TINY_REP_DIM = {"a": 3, "b": 2}, 8, 4, 4


def assert_stats_close(case: unittest.TestCase, got: dict, want: dict, places: int = 12) -> None:
    """逐字段比较统计结果（浮点用 almost，列表逐元素，其余严格相等）。"""
    case.assertEqual(sorted(got), sorted(want))
    for key, value in want.items():
        if isinstance(value, list):
            for a, b in zip(got[key], value):
                case.assertAlmostEqual(a, b, places=places)
        elif isinstance(value, float):
            case.assertAlmostEqual(got[key], value, places=places)
        else:
            case.assertEqual(got[key], value)


def tiny_backbone(device=None):
    """make_backbone 契约 = 与 run_census_benchmark.build_mptrec 同签名（只吃 device）。"""
    return run_census_benchmark.MPTRec(
        num_tasks=2, feature_vocabulary=dict(TINY_VOCAB), embedding_size=TINY_EMBEDDING,
        input_size=TINY_INPUT_SIZE, expert_dnn_hidden_units=(8, TINY_REP_DIM),
        tower_dnn_hidden_units=(4, 2), device=device)


def tiny_newtask(input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM, device=None):
    """必须与 router_probe.probe_newtask 用同一组超参，否则初始化不可比。"""
    return NewTask(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=list(P.TOWER_HIDDEN),
                   reg_dnn=P.REG_DNN, device=device)


def trained_newtask_state(device, *, offset: float = 0.05, seed: int = 5) -> dict:
    """造一份"训练过"的 NewTask 权重：与 tiny_newtask 同构，且可区分于随机初始化。"""
    torch.manual_seed(seed)
    newtask = tiny_newtask(device=device)
    with torch.no_grad():
        for value in newtask.parameters():
            value.add_(offset)
    return {name: value.detach().clone() for name, value in newtask.state_dict().items()}


def trained_newtask_checkpoint(path, device, **kwargs) -> Path:
    """按基线 run 的布局落盘 `newtask.pt`（torch.save(state_dict)）。"""
    path = Path(path)
    torch.save(trained_newtask_state(device, **kwargs), path)
    return path


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


def tiny_inputs(n_train=64, n_val=16, n_test=16, batch_size=16):
    """train_ds 是全量训练集，test_ds 按 P.SPLIT_SEED 切成 val / test，与真实路径同构。"""
    train_ds, test_ds = TinyCensus(n_train, 1), TinyCensus(n_val + n_test, 2)
    val_idx, test_idx = P.make_split(len(test_ds), P.SPLIT_SEED)
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size),
               "val": DataLoader(Subset(test_ds, val_idx.tolist()), batch_size=batch_size),
               "test": DataLoader(Subset(test_ds, test_idx.tolist()), batch_size=batch_size)}
    return loaders, P.split_stats(val_idx, test_idx, n_train=len(train_ds)), (val_idx, test_idx)


def run_tiny_stage1(root, device):
    """真跑一次 stage1，让 --stage1-dir 指向真实布局的产物（含划分指纹落盘）。"""
    loaders, stats, indices = tiny_inputs()
    stage1 = run_census_benchmark.run_stage1(root, epochs=2, device=device, model=tiny_backbone(device),
                                             loaders=loaders, stats=stats, indices=indices, tag="short")
    return stage1, loaders, stats, indices


def probe_kwargs(stage1, loaders, stats, indices, checkpoint) -> dict:
    """run_router_probe 的公共实参：tiny 夹具必须注入 tiny_backbone（否则会去建真实 vocab 的 backbone），
    并显式给出诊断对象（--newtask-checkpoint）。"""
    return dict(stage1_dir=Path(stage1["dir"]), newtask_checkpoint=Path(checkpoint), loaders=loaders,
                stats=stats, indices=indices, make_backbone=tiny_backbone, input_size=TINY_INPUT_SIZE,
                rep_dim=TINY_REP_DIM)


class TestModuleSourceCompiles(unittest.TestCase):
    """静态守卫：源码必须能被本解释器编译（最便宜的用例，先跑它）。

    venv 是 3.10：非三引号 f-string 不允许跨物理行，一旦跨行 `import` 阶段就 SyntaxError，
    本文件其余用例会整体以 collection error 失败。故不依赖 import 成功——源码路径由本文件位置推出。
    """

    def test_source_compiles(self):
        path = Path(__file__).resolve().parents[1] / "router_probe.py"
        self.assertTrue(path.is_file(), f"未找到被测源码: {path}")
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


class TestRouterWeightsFidelity(unittest.TestCase):
    """复算式必须与 NewTask.forward 里那一次 softmax 完全同源（否则诊断的不是真实 router）。"""

    def test_router_weights_reproduce_newtask_forward(self):
        torch.manual_seed(0)
        newtask = tiny_newtask()
        batch = 5
        dnn_input = torch.randn(batch, TINY_INPUT_SIZE)
        gen_rep = torch.randn(batch, TINY_REP_DIM)
        spec_reps = [torch.randn(batch, TINY_REP_DIM) for _ in range(P.NUM_TASKS)]
        env_embs = [torch.randn(TINY_REP_DIM) for _ in range(P.NUM_TASKS)]    # env 数 = num_tasks

        real_softmax = F.softmax
        seen = []

        def spy(x, *args, **kwargs):
            seen.append(x.detach().clone())
            return real_softmax(x, *args, **kwargs)

        with mock.patch("torch.nn.functional.softmax", side_effect=spy):
            newtask(dnn_input, gen_rep, spec_reps, env_embs)

        # forward 里恰好两次 softmax：先 router（W），后 gate_network → 顺序即身份
        self.assertEqual(len(seen), 2)
        self.assertTrue(torch.allclose(seen[1], newtask.gate_network[0](dnn_input), atol=1e-6))

        with torch.no_grad():
            h_out = newtask.projection_network(dnn_input)
            logits = torch.mm(h_out, torch.stack(env_embs, dim=1)) / newtask.temperature
            expected = real_softmax(logits, dim=-1)
            got = router_probe.router_weights(newtask, dnn_input, env_embs)

        self.assertTrue(torch.allclose(seen[0], logits, atol=1e-6))           # 落点：mm 方向 + 温度
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))
        self.assertTrue(torch.allclose(got, real_softmax(seen[0], dim=-1), atol=1e-6))
        self.assertEqual(tuple(got.shape), (batch, P.NUM_TASKS))
        self.assertAlmostEqual(float(got.sum(dim=1).min()), 1.0, places=6)    # 每行是概率分布

    def test_router_weights_read_temperature_from_newtask(self):
        """温度必须取自 newtask.temperature（协议值 150），不接受外部传入的近似值。"""
        torch.manual_seed(1)
        newtask = tiny_newtask()
        self.assertEqual(newtask.temperature, 150)
        dnn_input = torch.randn(3, TINY_INPUT_SIZE)
        env_embs = [torch.randn(TINY_REP_DIM) for _ in range(P.NUM_TASKS)]
        with torch.no_grad():
            hot = router_probe.router_weights(newtask, dnn_input, env_embs)
            newtask.temperature = 1                                           # 温度越小分布越尖
            sharp = router_probe.router_weights(newtask, dnn_input, env_embs)
        self.assertGreater(float((sharp - 0.5).abs().max()), float((hot - 0.5).abs().max()))


class TestRouterStats(unittest.TestCase):
    """四项指标的数学定义（spec 3.1）。"""

    def test_num_envs_below_two_rejected(self):
        with self.assertRaises(ValueError):
            router_probe.RouterStats(num_envs=1)                              # 归一化熵需要 log(K) > 0

    def test_uniform_routing_zero_std_full_entropy(self):
        stats = router_probe.RouterStats(num_envs=2)
        stats.update(torch.full((4, 2), 0.5, dtype=torch.float64))
        r = stats.result()
        self.assertAlmostEqual(r["router_w_std"], 0.0, places=12)
        self.assertAlmostEqual(r["normalized_entropy"], 1.0, places=12)       # 均匀 → 最大熵
        self.assertAlmostEqual(r["top1_share"], 0.5, places=12)
        self.assertEqual(r["source_share"], [1.0, 0.0])                       # 并列时 argmax 取最小下标
        self.assertEqual(r["n_samples"], 4)
        self.assertEqual(len(r["router_w_std_per_env"]), 2)

    def test_split_one_hot_routing(self):
        stats = router_probe.RouterStats(num_envs=2)
        stats.update(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64))
        r = stats.result()
        self.assertAlmostEqual(r["router_w_std"], 0.5, places=12)
        self.assertAlmostEqual(r["normalized_entropy"], 0.0, places=12)       # entr(0)=0 → 精确 0
        self.assertAlmostEqual(r["top1_share"], 1.0, places=12)
        self.assertEqual(r["source_share"], [0.5, 0.5])

    def test_mean_and_population_std_match_hand_computation(self):
        stats = router_probe.RouterStats(num_envs=2)
        stats.update(torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float64))
        r = stats.result()
        self.assertAlmostEqual(r["router_w_std_per_env"][0], 0.2, places=12)  # 总体 std（ddof=0）
        self.assertAlmostEqual(r["router_w_std_per_env"][1], 0.2, places=12)
        self.assertAlmostEqual(r["router_w_std"], 0.2, places=12)             # = 各 env std 的均值
        # 归一化熵是**逐样本** H(W_b)/log K 的均值：两行都要算（0.2/0.8 行 ≈ 0.7219，0.6/0.4 行 ≈ 0.9710）
        rows = ((0.2, 0.8), (0.6, 0.4))
        expected_entropy = sum(-sum(p * math.log(p) for p in row) for row in rows) / len(rows) / math.log(2)
        self.assertAlmostEqual(r["normalized_entropy"], expected_entropy, places=12)
        self.assertAlmostEqual(r["top1_share"], 0.7, places=12)               # (0.8 + 0.6) / 2
        self.assertEqual(r["source_share"], [0.5, 0.5])

    def test_three_env_source_share_sums_to_one(self):
        stats = router_probe.RouterStats(num_envs=3)
        stats.update(torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7],
                                   [0.2, 0.7, 0.1], [0.6, 0.3, 0.1]], dtype=torch.float64))
        r = stats.result()
        self.assertAlmostEqual(sum(r["source_share"]), 1.0, places=12)
        self.assertEqual(r["source_share"], [0.5, 0.25, 0.25])                # argmax 命中 [0, 2, 1, 0]
        self.assertEqual(len(r["router_w_std_per_env"]), 3)

    def test_streaming_update_equals_single_batch(self):
        gen = torch.Generator().manual_seed(7)
        w = torch.softmax(torch.randn(37, 3, generator=gen, dtype=torch.float64), dim=-1)
        one = router_probe.RouterStats(num_envs=3)
        one.update(w)
        chunked = router_probe.RouterStats(num_envs=3)
        for chunk in w.split([10, 10, 17]):
            chunked.update(chunk)
        assert_stats_close(self, chunked.result(), one.result())              # 流式累加 = 整批（显存 O(1)）

    def test_accumulators_stay_on_cpu_at_float64(self):
        """沿用 fix: keep mechanism accumulators on CPU 的教训，避免 device mismatch。"""
        stats = router_probe.RouterStats(num_envs=2)
        stats.update(torch.tensor([[0.3, 0.7]], dtype=torch.float64))
        self.assertEqual(stats.total.device.type, "cpu")
        self.assertEqual(stats.total.dtype, torch.float64)


class TestGate(unittest.TestCase):
    """预注册门槛（spec 3.2）——只允许在看到结果之前修改。"""

    def test_threshold_value_and_inclusive_boundary(self):
        self.assertAlmostEqual(router_probe.ROUTER_W_STD_MIN, 0.02, places=12)
        self.assertTrue(router_probe.gate_verdict(0.02)["proceed"])           # 含等号
        self.assertTrue(router_probe.gate_verdict(0.021)["proceed"])
        self.assertFalse(router_probe.gate_verdict(0.0199)["proceed"])

    def test_verdict_payload(self):
        verdict = router_probe.gate_verdict(0.004)
        self.assertEqual(verdict["rule"], "router_w_std >= 0.02")
        self.assertAlmostEqual(verdict["threshold"], 0.02, places=12)
        self.assertAlmostEqual(verdict["observed"], 0.004, places=12)
        self.assertFalse(verdict["proceed"])
        self.assertIsInstance(verdict["proceed"], bool)


class TestNewTaskInitFidelity(unittest.TestCase):
    """新任务头必须按 baseline Stage-2 的 model seed 与构造顺序构造，再载入训练后的权重（spec 2.1）。"""

    def test_probe_newtask_follows_stage2_rng_order(self):
        device = torch.device("cpu")
        backbone, newtask = router_probe.probe_newtask(
            device, model_seed=P.MODEL_SEED, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
            make_backbone=tiny_backbone)

        # run_stage2 的顺序：seed_model → 构造 backbone →（to/load/freeze 不耗 RNG）→ 构造 NewTask
        P.seed_model(P.MODEL_SEED)
        reference = tiny_backbone(device)
        P.freeze_backbone(reference)
        expected = tiny_newtask(device=device)

        got, want = newtask.state_dict(), expected.state_dict()
        self.assertEqual(sorted(got), sorted(want))
        for name in sorted(want):
            self.assertTrue(torch.equal(got[name], want[name]), f"NewTask 初始化偏离 stage2 顺序: {name}")

        self.assertFalse(any(p.requires_grad for p in backbone.parameters()))  # 真冻结
        self.assertFalse(newtask.training)                                    # eval，不训练

    def test_wrong_construction_order_changes_init(self):
        """反例：把 NewTask 建在 backbone 之前 → 初始化必然不同（证明顺序是保真条件）。"""
        device = torch.device("cpu")
        _, newtask = router_probe.probe_newtask(device, model_seed=P.MODEL_SEED,
                                                input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                                                make_backbone=tiny_backbone)
        P.seed_model(P.MODEL_SEED)
        wrong = tiny_newtask(device=device)     # 先建 NewTask，再建 backbone = 错误顺序
        tiny_backbone(device)
        self.assertFalse(all(torch.equal(a, b) for a, b in
                             zip(wrong.state_dict().values(), newtask.state_dict().values())))

    def test_probe_newtask_loads_stage1_state_and_freezes(self):
        device = torch.device("cpu")
        P.seed_model(P.MODEL_SEED)
        stage1_model = tiny_backbone(device)
        with torch.no_grad():
            for value in stage1_model.parameters():
                value.add_(0.05)                     # 造一份"训练过"的权重，与随机初始化可区分
        trained_sha = P.backbone_sha256(stage1_model)
        self.assertNotEqual(trained_sha, P.backbone_sha256(tiny_backbone(device)))

        backbone, _ = router_probe.probe_newtask(device, model_seed=P.MODEL_SEED, input_size=TINY_INPUT_SIZE,
                                                 rep_dim=TINY_REP_DIM, make_backbone=tiny_backbone,
                                                 backbone_state=stage1_model.state_dict())
        self.assertEqual(P.backbone_sha256(backbone), trained_sha)            # 载入的确实是 stage1 权重
        self.assertFalse(any(p.requires_grad for p in backbone.parameters()))

    def test_probe_newtask_loads_trained_newtask_state_in_stage2_order(self):
        """诊断对象 = 基线 run 训练后的头：strict 载入 checkpoint，且构造顺序仍与 stage2 逐位一致。"""
        device = torch.device("cpu")
        trained_state = trained_newtask_state(device, seed=11)

        P.seed_model(P.MODEL_SEED)                       # 参照：stage2 顺序（seed → backbone → NewTask）的 RNG 终点
        tiny_backbone(device)
        tiny_newtask(device=device)
        expected_rng = torch.get_rng_state().clone()

        _, newtask = router_probe.probe_newtask(device, model_seed=P.MODEL_SEED, input_size=TINY_INPUT_SIZE,
                                                rep_dim=TINY_REP_DIM, make_backbone=tiny_backbone,
                                                newtask_state=trained_state)

        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))     # 载入不改变构造顺序
        got = newtask.state_dict()
        self.assertEqual(sorted(got), sorted(trained_state))
        for name in sorted(trained_state):
            self.assertTrue(torch.equal(got[name], trained_state[name]), f"未载入 checkpoint: {name}")


class TestNewTaskCheckpoint(unittest.TestCase):
    """诊断对象来自 `--newtask-checkpoint`（基线 run 的 newtask.pt）：只读，缺失 / 不兼容必须报错。"""

    def test_missing_checkpoint_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                router_probe.load_newtask_state(Path(td) / "no_such_newtask.pt")

    def test_load_newtask_state_round_trip_and_non_state_dict_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "newtask.pt"
            torch.save(trained_newtask_state(torch.device("cpu")), path)
            state = router_probe.load_newtask_state(path)
            self.assertEqual(sorted(state), sorted(tiny_newtask().state_dict()))
            self.assertTrue(all(torch.is_tensor(value) for value in state.values()))

            torch.save(torch.zeros(3), path)                              # 不是 state_dict
            with self.assertRaises(ValueError):
                router_probe.load_newtask_state(path)

    def test_probe_newtask_rejects_incompatible_checkpoint(self):
        """缺键 / 多键 / 形状不符都必须报错——绝不静默换成随机初始化的头。"""
        device = torch.device("cpu")
        state = trained_newtask_state(device)
        missing = {name: value for name, value in state.items() if not name.startswith("tower_network")}
        with self.assertRaises(ValueError) as ctx:
            router_probe.probe_newtask(device, model_seed=P.MODEL_SEED, input_size=TINY_INPUT_SIZE,
                                       rep_dim=TINY_REP_DIM, make_backbone=tiny_backbone, newtask_state=missing)
        self.assertIn("tower_network", str(ctx.exception))                # 报错必须点名不一致的键

        for incompatible in (dict(state, **{"projection_network.0.weight": torch.zeros(3, 3)}),   # 形状不符
                             dict(state, **{"extra.weight": torch.zeros(1)})):                     # 多出的键
            with self.assertRaises(ValueError):
                router_probe.probe_newtask(device, model_seed=P.MODEL_SEED, input_size=TINY_INPUT_SIZE,
                                           rep_dim=TINY_REP_DIM, make_backbone=tiny_backbone,
                                           newtask_state=incompatible)


class TestRunRouterProbe(unittest.TestCase):
    def test_probe_is_zero_training_deterministic_and_persisted(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            checkpoint_path = Path(stage1["dir"]) / "backbone.pt"
            checkpoint_bytes = checkpoint_path.read_bytes()
            ckpt = trained_newtask_checkpoint(Path(td) / "newtask.pt", device)
            ckpt_bytes = ckpt.read_bytes()
            kwargs = dict(device=device, **probe_kwargs(stage1, loaders, stats, indices, ckpt))

            first = router_probe.run_router_probe(root, **kwargs)
            second = router_probe.run_router_probe(root, **kwargs)

            self.assertEqual(first["metrics"], second["metrics"])             # 同输入 → 同结果
            self.assertEqual(checkpoint_path.read_bytes(), checkpoint_bytes)  # stage1 checkpoint 只读
            self.assertEqual(ckpt.read_bytes(), ckpt_bytes)                   # newtask checkpoint 只读

            payload = first["payload"]
            self.assertEqual(payload["probe_id"],
                             f"rp-{stage1['stage1_id']}-val-{payload['newtask_state_sha256'][:8]}")
            self.assertEqual(payload["stage1_id"], stage1["stage1_id"])
            self.assertEqual(payload["split"], "val")
            self.assertEqual(payload["trained_steps"], 0)                     # 零训练（spec 2.2）
            self.assertTrue(payload["frozen"])
            self.assertTrue(payload["grads_all_none"])
            self.assertTrue(payload["split_ok"])
            self.assertEqual(payload["backbone_sha256_before"], payload["backbone_sha256_after"])
            self.assertTrue(payload["stage1_backbone_sha256_matches"])
            self.assertEqual(payload["stage1_backbone_sha256"], payload["backbone_sha256_before"])
            self.assertEqual(payload["temperature"], 150)
            self.assertEqual(payload["num_envs"], P.NUM_ENVS)
            self.assertEqual(payload["split_fingerprint_sha256"],
                             P.load_split_fingerprint(root, P.SPLIT_SEED)["fingerprint_sha256"])
            # 诊断对象证据：文件路径、文件哈希、checkpoint 状态哈希、模型内实际参数四方一致（spec 2.2）
            self.assertEqual(payload["newtask_checkpoint"], str(ckpt))
            self.assertEqual(payload["newtask_checkpoint_sha256"], P.sha256_bytes(ckpt_bytes))
            self.assertEqual(payload["newtask_checkpoint_state_sha256"],
                             router_probe.state_dict_sha256(router_probe.load_newtask_state(ckpt)))
            self.assertEqual(payload["newtask_state_sha256"], payload["newtask_checkpoint_state_sha256"])

            for key in ("router_w_std", "normalized_entropy", "source_share", "top1_share", "n_samples"):
                self.assertIn(key, payload["metrics"])                        # 四项必须落盘（spec 4）
            self.assertEqual(payload["metrics"]["n_samples"], stats["n_val"])
            self.assertEqual(len(payload["metrics"]["source_share"]), P.NUM_ENVS)
            self.assertAlmostEqual(sum(payload["metrics"]["source_share"]), 1.0, places=9)
            self.assertEqual(payload["gate"]["proceed"],
                             payload["metrics"]["router_w_std"] >= router_probe.ROUTER_W_STD_MIN)
            self.assertIn(payload["gate"]["rule"], json.dumps(payload))       # 门槛随结果一起留痕

            out_path = root / "probes" / first["probe_id"] / "router_probe.json"
            self.assertEqual(Path(first["dir"]), out_path.parent)
            self.assertTrue(out_path.exists())
            on_disk = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["metrics"], payload["metrics"])
            self.assertEqual(on_disk["gate"], payload["gate"])
            self.assertEqual(on_disk["newtask_checkpoint"], str(ckpt))

    def test_probe_diagnoses_checkpoint_head_not_fresh_init(self):
        """诊断的必须是 checkpoint 里的头：统计 = 复算该头的统计，且与随机初始化头不同。"""
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            ckpt = trained_newtask_checkpoint(Path(td) / "newtask.pt", device)
            out = router_probe.run_router_probe(root, device=device,
                                                **probe_kwargs(stage1, loaders, stats, indices, ckpt))
            backbone_state = P.load_stage1(root, stage1["stage1_id"])["backbone_state"]

            backbone, trained = router_probe.probe_newtask(
                device, model_seed=P.MODEL_SEED, input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM,
                make_backbone=tiny_backbone, backbone_state=backbone_state,
                newtask_state=router_probe.load_newtask_state(ckpt))
            assert_stats_close(self, out["metrics"],
                               router_probe.probe_loader(trained, backbone, loaders["val"], device))

            _, fresh = router_probe.probe_newtask(device, model_seed=P.MODEL_SEED, input_size=TINY_INPUT_SIZE,
                                                  rep_dim=TINY_REP_DIM, make_backbone=tiny_backbone,
                                                  backbone_state=backbone_state)
            fresh_metrics = router_probe.probe_loader(fresh, backbone, loaders["val"], device)
            self.assertNotAlmostEqual(out["metrics"]["router_w_std"], fresh_metrics["router_w_std"], places=6)

    def test_probe_consumes_real_stage2_run_checkpoint(self):
        """端到端口径：--newtask-checkpoint 指向 run_stage2 落盘的 newtask.pt 时必须可直接诊断。"""
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            run = run_census_benchmark.run_stage2(root, stage1_dir=Path(stage1["dir"]), epochs=1, device=device,
                                                  model=tiny_backbone(device), loaders=loaders, stats=stats,
                                                  indices=indices, input_size=TINY_INPUT_SIZE,
                                                  rep_dim=TINY_REP_DIM)
            ckpt = Path(run["run_dir"]) / "newtask.pt"
            out = router_probe.run_router_probe(root, device=device,
                                                **probe_kwargs(stage1, loaders, stats, indices, ckpt))
            payload = out["payload"]
            self.assertEqual(payload["newtask_checkpoint"], str(ckpt))
            self.assertEqual(payload["newtask_state_sha256"],                                  # 头 = run 落盘的头
                             router_probe.state_dict_sha256(router_probe.load_newtask_state(ckpt)))
            self.assertEqual(payload["newtask_run_config"]["run_id"], run["run_id"])            # 旁证 run 口径
            self.assertEqual(payload["newtask_run_config"]["stage1_id"], stage1["stage1_id"])
            self.assertEqual(Path(out["dir"]).parent.name, "probes")

    def test_newtask_run_config_provenance_mismatch_raises(self):
        """checkpoint 旁的 config.json 若指向别的 stage1 / model seed → 拒绝继续（口径归属可查）。"""
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            ckpt = trained_newtask_checkpoint(run_dir / "newtask.pt", device)
            kwargs = dict(device=device, **probe_kwargs(stage1, loaders, stats, indices, ckpt))

            (run_dir / "config.json").write_text(json.dumps(
                {"run_id": "r1", "stage1_id": "s1-other", "model_seed": P.MODEL_SEED}), encoding="utf-8")
            with self.assertRaises(ValueError):
                router_probe.run_router_probe(root, **kwargs)

            (run_dir / "config.json").write_text(json.dumps(
                {"run_id": "r1", "stage1_id": stage1["stage1_id"], "model_seed": P.MODEL_SEED + 1}),
                encoding="utf-8")
            with self.assertRaises(ValueError):
                router_probe.run_router_probe(root, **kwargs)

            (run_dir / "config.json").write_text(json.dumps(
                {"run_id": "r1", "stage1_id": stage1["stage1_id"], "model_seed": P.MODEL_SEED}), encoding="utf-8")
            out = router_probe.run_router_probe(root, **kwargs)
            self.assertEqual(out["payload"]["newtask_run_config"]["run_id"], "r1")              # 一致则记录留痕

    def test_missing_newtask_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            with self.assertRaises(FileNotFoundError):
                router_probe.run_router_probe(root, device=device,
                                              **probe_kwargs(stage1, loaders, stats, indices,
                                                             Path(td) / "no_such_newtask.pt"))

    def test_gate_is_recorded_even_when_it_fails(self):
        """门槛未过也必须落盘（spec 4：不得只报告成功的 run）。"""
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            ckpt = trained_newtask_checkpoint(Path(td) / "newtask.pt", device)
            out = router_probe.run_router_probe(root, device=device,
                                                **probe_kwargs(stage1, loaders, stats, indices, ckpt))
            payload = out["payload"]
            self.assertIsInstance(payload["gate"]["proceed"], bool)
            self.assertAlmostEqual(payload["gate"]["observed"], payload["metrics"]["router_w_std"], places=12)
            self.assertTrue((root / "probes" / out["probe_id"] / "router_probe.json").exists())

    def test_split_probe_uses_test_loader_and_records_test_sha(self):
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            ckpt = trained_newtask_checkpoint(Path(td) / "newtask.pt", device)
            out = router_probe.run_router_probe(root, split="test", device=device,
                                                **probe_kwargs(stage1, loaders, stats, indices, ckpt))
            payload = out["payload"]
            self.assertEqual(payload["split"], "test")
            self.assertEqual(payload["probe_id"],
                             f"rp-{stage1['stage1_id']}-test-{payload['newtask_state_sha256'][:8]}")
            self.assertEqual(payload["metrics"]["n_samples"], stats["n_test"])
            self.assertEqual(payload["split_sha256"], stage1["meta"]["test_sha256"])   # 与 stage1 记录一致

    def test_missing_stage1_dir_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                router_probe.run_router_probe(Path(td) / "artifacts", stage1_dir=Path(td) / "no_such_stage1",
                                              newtask_checkpoint=Path(td) / "newtask.pt",
                                              device=torch.device("cpu"))           # stage1 先于 checkpoint 校验

    def test_stage1_id_mismatch_raises(self):
        """meta 与目录名不一致 → 拒绝继续（防止诊断的不是自己以为的那份 backbone）。"""
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            ckpt = trained_newtask_checkpoint(Path(td) / "newtask.pt", device)
            src = Path(stage1["dir"])
            dst = src.parent / "s1-tampered"
            dst.mkdir()
            for name in ("backbone.pt", "env_ids.pt", "meta.json"):
                (dst / name).write_bytes((src / name).read_bytes())
            kwargs = dict(device=device, loaders=loaders, stats=stats, indices=indices,
                          make_backbone=tiny_backbone, newtask_checkpoint=ckpt,
                          input_size=TINY_INPUT_SIZE, rep_dim=TINY_REP_DIM)
            with self.assertRaises(ValueError):
                router_probe.run_router_probe(root, stage1_dir=dst, **kwargs)

    def test_probe_writes_only_into_probes(self):
        """诊断产物只进 probes/，绝不写 stage1/ 或 runs/（spec 4）。"""
        with tempfile.TemporaryDirectory() as td:
            root, device = Path(td) / "artifacts", torch.device("cpu")
            stage1, loaders, stats, indices = run_tiny_stage1(root, device)
            ckpt = trained_newtask_checkpoint(Path(td) / "newtask.pt", device)
            out = router_probe.run_router_probe(root, device=device,
                                                **probe_kwargs(stage1, loaders, stats, indices, ckpt))
            self.assertEqual(Path(out["dir"]).parent.name, "probes")
            self.assertFalse((root / "runs").exists())
            self.assertEqual([p.name for p in (root / "stage1").iterdir()], [Path(stage1["dir"]).name])


class TestCli(unittest.TestCase):
    def test_cli_accepts_the_preregistered_command(self):
        args = router_probe.build_parser().parse_args(
            ["--stage1-dir", str(P.ARTIFACT_ROOT / "stage1" / "s1-x"),
             "--newtask-checkpoint", str(P.ARTIFACT_ROOT / "runs" / "r1" / "newtask.pt"), "--gpu", "0"])
        self.assertEqual(Path(args.stage1_dir).name, "s1-x")
        self.assertEqual(Path(args.newtask_checkpoint).name, "newtask.pt")
        self.assertEqual(args.gpu, 0)
        self.assertEqual(args.split, "val")                                   # 默认在固定 val 划分上诊断
        self.assertEqual(Path(args.root), Path(P.ARTIFACT_ROOT))

    def test_cli_split_choices(self):
        base = ["--stage1-dir", "x", "--newtask-checkpoint", "r1/newtask.pt"]
        self.assertEqual(router_probe.build_parser().parse_args(base + ["--split", "test"]).split, "test")
        with self.assertRaises(SystemExit):
            router_probe.build_parser().parse_args(base + ["--split", "train"])

    def test_stage1_dir_is_required(self):
        with self.assertRaises(SystemExit):
            router_probe.build_parser().parse_args(["--newtask-checkpoint", "r1/newtask.pt"])

    def test_newtask_checkpoint_is_required(self):
        """CLI 必须强制指定诊断对象：缺 --newtask-checkpoint 直接拒绝运行（不许退回随机初始化头）。"""
        with self.assertRaises(SystemExit):
            router_probe.build_parser().parse_args(["--stage1-dir", "x"])


if __name__ == "__main__":
    unittest.main()
