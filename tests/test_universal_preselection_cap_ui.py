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
        self.assertNotIn("max_value", keywords)
        self.assertEqual(ast.literal_eval(keywords["step"]), 50)

    def test_selected_cap_is_passed_to_candidate_extension(self):
        calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "extend_allocation_with_universal_candidates"
        ]
        supplied_caps = {
            ast.unparse({item.arg: item.value for item in call.keywords}["maximum_candidates"])
            for call in calls
        }
        self.assertEqual(supplied_caps, {"cap", "universal_preselection_cap"})

    def test_advanced_search_uses_adaptive_steps_with_fifty_minimum(self):
        function = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "search_universal_shortlist_caps"
        )
        defaults = {
            argument.arg: default
            for argument, default in zip(function.args.kwonlyargs, function.args.kw_defaults)
            if default is not None
        }
        self.assertEqual(ast.literal_eval(defaults["minimum_trading_days"]), 252)
        self.assertEqual(ast.literal_eval(defaults["step"]), 50)
        self.assertIsNone(ast.literal_eval(defaults["maximum_cap"]))
        self.assertTrue(ast.literal_eval(defaults["adaptive"]))
        self.assertEqual(ast.literal_eval(defaults["runtime_brake_seconds"]), 900)
        source = ast.unparse(function)
        self.assertIn("trading_days < int(minimum_trading_days)", source)
        self.assertIn("minimum_increment = max(int(step), 50)", source)
        self.assertIn("next_adaptive_jump", source)
        self.assertIn("elapsed_seconds >= float(runtime_brake_seconds)", source)
        self.assertIn("cap = min(cap + current_jump, range_ceiling)", source)


if __name__ == "__main__":
    unittest.main()
