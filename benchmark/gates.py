"""门禁引擎：GateOutcome（状态 + 证据）、evaluate 构造、hard_pass 判定与两种输出 schema 渲染。

门禁条件与证据字段是各数据集适配器（datasets/census.py 的 judge、datasets/aliccp.py 的
evaluate_a_gates / evaluate_b_gates），共用此处的状态词表与渲染：
- census schema：{"pass": bool, "detail": <结构化 dict>}（render_pass_detail）；
- aliccp schema：{"verdict": "PASS"/"FAIL"/"SKIP"/"N/A", "detail": <文本>}（render_verdict_detail）。
"""
from __future__ import annotations

from dataclasses import dataclass

PASS, FAIL, SKIP, NA = "PASS", "FAIL", "SKIP", "N/A"
_STATES = (PASS, FAIL, SKIP, NA)


@dataclass(frozen=True)
class GateOutcome:
    """单个门禁结果：state ∈ {PASS, FAIL, SKIP, N/A}，detail 为证据（形态由 renderer 决定）。"""

    gate_id: str
    state: str
    detail: object = None

    def __post_init__(self):
        if self.state not in _STATES:
            raise ValueError(f"非法门禁状态: {self.state}")

    @property
    def passed(self) -> bool:
        return self.state == PASS


def evaluate(items) -> list[GateOutcome]:
    """items: (gate_id, condition, detail) 或现成 GateOutcome（如 SKIP/N/A）的序列 → 同序 GateOutcome。"""
    out = []
    for item in items:
        if isinstance(item, GateOutcome):
            out.append(item)
        else:
            gate_id, condition, detail = item
            out.append(GateOutcome(gate_id, PASS if condition else FAIL, detail))
    return out


def hard_pass(states, *, allowed) -> bool:
    """硬性判定：所有状态都在 allowed 集合内才通过（空集合 → True）。"""
    return all(state in allowed for state in states)


def render_pass_detail(outcomes) -> dict:
    """census schema：{"pass": bool, "detail": 结构化证据}。"""
    return {outcome.gate_id: {"pass": outcome.passed, "detail": outcome.detail} for outcome in outcomes}


def render_verdict_detail(outcomes) -> dict:
    """aliccp schema：{"verdict": 状态字符串, "detail": 文本/结构化证据}。"""
    return {outcome.gate_id: {"verdict": outcome.state, "detail": outcome.detail} for outcome in outcomes}
