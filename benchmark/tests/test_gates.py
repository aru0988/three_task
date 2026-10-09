"""门禁引擎测试：状态校验、evaluate 构造、两种 schema 渲染、hard_pass 真值表。"""
import unittest

from benchmark import gates


class TestGateOutcome(unittest.TestCase):
    def test_state_validation(self):
        for state in (gates.PASS, gates.FAIL, gates.SKIP, gates.NA):
            self.assertEqual(gates.GateOutcome("X", state).state, state)
        with self.assertRaises(ValueError):
            gates.GateOutcome("X", "OK")

    def test_only_pass_counts_as_passed(self):
        self.assertTrue(gates.GateOutcome("X", gates.PASS).passed)
        for state in (gates.FAIL, gates.SKIP, gates.NA):
            self.assertFalse(gates.GateOutcome("X", state).passed, state)


class TestEvaluate(unittest.TestCase):
    def test_conditions_and_literal_outcomes_keep_order(self):
        outcomes = gates.evaluate([
            ("A1", True, {"x": 1}),
            gates.GateOutcome("A3", gates.SKIP, "按需"),
            ("B1", False, "text"),
        ])
        self.assertEqual([o.gate_id for o in outcomes], ["A1", "A3", "B1"])
        self.assertEqual([o.state for o in outcomes], [gates.PASS, gates.SKIP, gates.FAIL])
        self.assertEqual(outcomes[0].detail, {"x": 1})
        self.assertEqual(outcomes[2].detail, "text")


class TestRender(unittest.TestCase):
    def test_pass_detail_schema(self):
        rendered = gates.render_pass_detail(gates.evaluate([("A1", True, {"a": 1})]))
        self.assertEqual(rendered, {"A1": {"pass": True, "detail": {"a": 1}}})

    def test_verdict_detail_schema(self):
        rendered = gates.render_verdict_detail(gates.evaluate([
            ("A1", True, "ok"), ("B4", False, "bad"), gates.GateOutcome("A3", gates.SKIP, "-")]))
        self.assertEqual(rendered, {"A1": {"verdict": "PASS", "detail": "ok"},
                                    "B4": {"verdict": "FAIL", "detail": "bad"},
                                    "A3": {"verdict": "SKIP", "detail": "-"}})


class TestHardPassTruthTable(unittest.TestCase):
    def test_empty_is_true(self):
        self.assertTrue(gates.hard_pass([], allowed=(gates.PASS,)))

    def test_single_state(self):
        self.assertTrue(gates.hard_pass([gates.PASS], allowed=(gates.PASS,)))
        for state in (gates.FAIL, gates.SKIP, gates.NA):
            self.assertFalse(gates.hard_pass([state], allowed=(gates.PASS,)), state)

    def test_a_class_allows_skip_only(self):
        # aliccp A 类：PASS/SKIP 通过；N/A 与 FAIL 不通过
        self.assertTrue(gates.hard_pass([gates.PASS, gates.SKIP], allowed=(gates.PASS, gates.SKIP)))
        self.assertFalse(gates.hard_pass([gates.PASS, gates.NA], allowed=(gates.PASS, gates.SKIP)))
        self.assertFalse(gates.hard_pass([gates.PASS, gates.FAIL], allowed=(gates.PASS, gates.SKIP)))

    def test_b_class_allows_na_only(self):
        # aliccp B 类：PASS/N/A 通过；SKIP 与 FAIL 不通过
        self.assertTrue(gates.hard_pass([gates.PASS, gates.NA], allowed=(gates.PASS, gates.NA)))
        self.assertFalse(gates.hard_pass([gates.PASS, gates.SKIP], allowed=(gates.PASS, gates.NA)))
        self.assertFalse(gates.hard_pass([gates.PASS, gates.FAIL], allowed=(gates.PASS, gates.NA)))

    def test_any_fail_fails_the_whole_set(self):
        self.assertFalse(gates.hard_pass([gates.PASS, gates.SKIP, gates.NA, gates.FAIL],
                                         allowed=(gates.PASS, gates.SKIP, gates.NA)))


if __name__ == "__main__":
    unittest.main()
