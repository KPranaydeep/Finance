import ast
import unittest
from pathlib import Path


class UniversalPreselectionCapUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("portfolio_rebalancer_database.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_ui_control_uses_fifty_candidate_steps(self):
        controls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "number_input"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "Universal candidate shortlist cap"
        ]
        self.assertEqual(len(controls), 1)
        keywords = {item.arg: item.value for item in controls[0].keywords}
        self.assertEqual(ast.literal_eval(keywords["min_value"]), 50)
        self.assertEqual(ast.literal_eval(keywords["max_value"]), 1000)
        self.assertEqual(ast.literal_eval(keywords["step"]), 50)

    def test_selected_cap_is_passed_to_candidate_extension(self):
        calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "extend_allocation_with_universal_candidates"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertEqual(
            ast.unparse(keywords["maximum_candidates"]),
            "universal_preselection_cap",
        )


if __name__ == "__main__":
    unittest.main()
