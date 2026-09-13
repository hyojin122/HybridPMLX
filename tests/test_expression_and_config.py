import math
import unittest
from pathlib import Path

from hybridpmlx.config import load_config
from hybridpmlx.expressions import SafeExpression


class ExpressionAndConfigTests(unittest.TestCase):
    def test_safe_expression_numeric(self):
        expr = SafeExpression("B + A * exp(-K * IDV)")
        out = expr({"A": 2.0, "B": 1.0, "K": 0.5, "IDV": 2.0}, {"exp": math.exp})
        self.assertAlmostEqual(out, 1 + 2 * math.exp(-1))

    def test_unsafe_expression_rejected(self):
        with self.assertRaises(ValueError):
            SafeExpression("__import__('os').system('echo unsafe')")

    def test_examples_validate(self):
        root = Path(__file__).parents[1]
        for name in ("gh_gompertz.yaml", "dementia_weibull.yaml"):
            cfg = load_config(root / "examples" / name)
            self.assertEqual(cfg["data"]["id_column"], "ID")


if __name__ == "__main__":
    unittest.main()
