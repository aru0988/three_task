"""AliCCP 两路径增量效用路由（baseline vs 范数受控残差 Prompt）：核心纯函数（numpy/sklearn）。

预注册（唯一事实来源，先于本实现单独提交 C1=40e1b9d）：
docs/superpowers/specs/2026-10-06-aliccp-incremental-utility-verifier-design.md

- 训练内部三分割 A/B/C（train 前缀行序连续 70/15/15，确定性、无 split 种子）；
- 两臂（baseline 头 / 残差 Prompt 头）只在 A 上训练 ⇒ B/C/test 头级 out-of-sample；
- 配对效用 u = BCE_B − BCE_P；有益标签固定 1[u > 0]（不使用中位数替代）；
- verifier 推理特征零标签（两臂 logits、差异、冻结表示范数，恰 8 维）；
- Ridge 闭式（alpha=1.0 固定）；阈值 / 混合权重 / 随机路由率仅由 C 校准，test 端不调节。
本模块不含 GPU 依赖、不 import 任何模型代码，便于单测与独立复核。
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.metrics import roc_auc_score

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)                        # A/B/C（预注册 §3）
ROUTER_ALPHA = 1.0                                           # Ridge 正则（固定，不搜索）
THRESHOLD_QUANTILES = tuple(i / 20 for i in range(1, 20))    # 0.05..0.95（19 点）
MIX_GRID = tuple(i / 20 for i in range(0, 21))               # 0.00..1.00（21 点）
ROUTER_RANDOM_SALT = 20261006                                # 随机路由 RNG 种子 = model_seed ^ salt
RUN_ID_SUFFIX = "-iuv"
FEATURE_DIM = 8
EPS = 1e-12
DELTA_POSITIVE = 0.001                                       # 用户分类边界（闭开，预注册 §0）
DELTA_DEGRADE = -0.02


def _f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def split_bounds(n: int) -> dict:
    """训练前缀行序三分割边界（A/B/C 连续、互斥、并集=全体）。"""
    if n <= 0:
        raise ValueError(f"样本数必须为正: {n}")
    a_end = int(n * SPLIT_FRACTIONS[0])
    b_end = int(n * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]))
    if not (0 < a_end < b_end < n):
        raise ValueError(f"样本数过小，无法三分割: {n}")
    return {"A": (0, a_end), "B": (a_end, b_end), "C": (b_end, n)}


def split_records(n: int) -> list:
    """分割的 (名称, 起, 止, 大小) 记录（落盘用）。"""
    return [(k, v[0], v[1], v[1] - v[0]) for k, v in split_bounds(n).items()]


def range_sha(a: int, b: int) -> str:
    """连续行区间的索引 sha256（分割完整性落盘证据）。"""
    return hashlib.sha256(np.arange(int(a), int(b), dtype=np.int64).tobytes()).hexdigest()


def bce_per_sample(p, y) -> np.ndarray:
    p = np.clip(_f64(p), EPS, 1.0 - EPS)
    y = _f64(y)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def utility(p_b, p_p, y) -> np.ndarray:
    """配对效用 u = BCE_B − BCE_P（正 = prompt 路径更优 = 路由到 P 的增量效用）。"""
    return bce_per_sample(p_b, y) - bce_per_sample(p_p, y)


def beneficial(u) -> np.ndarray:
    """有益标签固定 1[u > 0]（预注册：禁止用中位数或其他分位替代）。"""
    return (_f64(u) > 0.0).astype(np.int64)


def _row_norms(x) -> np.ndarray:
    x = _f64(x)
    if x.ndim == 1:                     # 常量列（如 env 范数）以 1 维传入
        x = x.reshape(-1, 1)
    return np.sqrt((x * x).sum(axis=1))


def router_features(p_b, p_p, dnn_input, gen_rep, spec_0, spec_1, env_emb) -> np.ndarray:
    """8 维零标签特征：两臂 logits、差异、冻结表示范数。

    任何标签或标签依赖统计不得进入（预注册 §4.1）；签名即守卫（单测断言无形参 y/label）。
    """
    cols = [
        _f64(p_b), _f64(p_p), _f64(p_p) - _f64(p_b),
        _row_norms(dnn_input), _row_norms(gen_rep),
        _row_norms(spec_0), _row_norms(spec_1), _row_norms(env_emb),
    ]
    out = np.column_stack(cols)
    if out.shape[1] != FEATURE_DIM:
        raise AssertionError(f"特征维度必须为 {FEATURE_DIM}")
    return out


def fit_ridge(X, target, alpha: float = ROUTER_ALPHA) -> dict:
    """Ridge 闭式解（标准化特征 + 截距；仅由 B 调用）。"""
    X = _f64(X)
    t = _f64(target)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    Xs = (X - mean) / std
    d = Xs.shape[1]
    w = np.linalg.solve(Xs.T @ Xs + float(alpha) * np.eye(d), Xs.T @ t)
    return {
        "mean": mean.tolist(), "std": std.tolist(), "w": w.tolist(),
        "b": float(t.mean()), "alpha": float(alpha),
    }


def ridge_score(model: dict, X) -> np.ndarray:
    Xs = (_f64(X) - _f64(model["mean"])) / _f64(model["std"])
    return Xs @ _f64(model["w"]) + float(model["b"])


def route_predict(scores, p_b, p_p, thr) -> np.ndarray:
    """二值路由：score > thr 选 P，否则选 B。"""
    return np.where(_f64(scores) > float(thr), _f64(p_p), _f64(p_b))


def select_threshold(scores_C, p_b_C, p_p_C, y_C) -> dict:
    """阈值仅由 C 校准：C-AUC 最大；并列取被路由占比更小；再并列取更早（更低）候选。"""
    scores = _f64(scores_C)
    y = _f64(y_C)
    candidates = [-np.inf] + [float(np.quantile(scores, q)) for q in THRESHOLD_QUANTILES] + [np.inf]
    grid, best = [], None
    for idx, thr in enumerate(candidates):
        pred = route_predict(scores, p_b_C, p_p_C, thr)
        auc = float(roc_auc_score(y, pred))
        pi = float((scores > thr).mean())
        grid.append({"thr": thr, "auc_C_routed": auc, "pi": pi})
        key = (auc, -pi, -idx)
        if best is None or key > best[0]:
            best = (key, thr, pi, auc)
    return {"thr": float(best[1]), "pi_hat": float(best[2]), "auc_C_routed": float(best[3]), "grid": grid}


def select_mix_alpha(p_b_C, p_p_C, y_C) -> dict:
    """固定混合权重仅由 C 校准：C-AUC 最大；并列取更小 alpha。"""
    y = _f64(y_C)
    p_b, p_p = _f64(p_b_C), _f64(p_p_C)
    grid, best = [], None
    for a in MIX_GRID:
        auc = float(roc_auc_score(y, a * p_p + (1.0 - a) * p_b))
        grid.append({"alpha": float(a), "auc_C_mix": auc})
        key = (auc, -float(a))
        if best is None or key > best[0]:
            best = (key, float(a), auc)
    return {"alpha_hat": best[1], "auc_C_mix": best[2], "grid": grid}


def random_route_mask(n: int, pi_hat: float, seed: int) -> np.ndarray:
    """随机路由掩码：路由率固定取 calibration 的 pi_hat（test 端不调节）；RNG 种子钉死。"""
    rng = np.random.default_rng(int(seed))
    return rng.random(int(n)) < float(pi_hat)


def oracle_route(p_b, p_p, u) -> np.ndarray:
    """label-assisted BCE oracle diagnostic（诊断专用，禁止作为部署指标）：按 1[u > 0] 选择路径。

    注意：逐样本 BCE 最小化**不是 AUC 上界**（AUC 为排序度量），禁止表述为"上界"。
    """
    return np.where(_f64(u) > 0.0, _f64(p_p), _f64(p_b))


def expand_eligible(router_value: bool, all_gates_ok: bool, git_dirty: bool) -> bool:
    """扩 seed 机械条件（预注册 §6，v3.1）：三比较 router 价值 ∧ 全部门禁 ∧ 非 dirty。"""
    return bool(router_value) and bool(all_gates_ok) and (not bool(git_dirty))


def auc(y, p) -> float:
    return float(roc_auc_score(_f64(y), _f64(p)))


def classify_delta(delta: float) -> str:
    """用户分类（边界闭开，预注册 §0）。"""
    if delta >= DELTA_POSITIVE:
        return "POSITIVE_IMPROVEMENT"
    if delta <= DELTA_DEGRADE:
        return "CLEAR_DEGRADATION"
    return "NO_CLEAR_IMPROVEMENT"


def router_value_verdict(delta_base: float, delta_prompt: float, delta_mix: float) -> dict:
    """router 价值充分条件（预注册 §0）：三比较同时成立；仅优于较弱一侧不构成证据。"""
    ok = (delta_base >= DELTA_POSITIVE) and (delta_prompt > 0.0) and (delta_mix > 0.0)
    return {
        "classification_by_delta_base": classify_delta(float(delta_base)),
        "delta_base_ge_positive": bool(delta_base >= DELTA_POSITIVE),
        "delta_prompt_gt_0": bool(delta_prompt > 0.0),
        "delta_mix_gt_0": bool(delta_mix > 0.0),
        "router_value": bool(ok),
    }
