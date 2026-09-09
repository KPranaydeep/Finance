import ast
import json
from pathlib import Path
import unittest
from public_cash_withdrawal import suggested_withdrawal,withdrawal_instructions


class CashWithdrawalTests(unittest.TestCase):
    def test_smallest_low_weight_position_after_costs(self):
        plan={"orders":[{"ticker":"A","quantity":10,"planning_price":100},
                        {"ticker":"B","quantity":2,"planning_price":400},
                        {"ticker":"C","quantity":1,"planning_price":10}]}
        target=[{"ticker":"A","target_weight":.01},{"ticker":"B","target_weight":.01},
                {"ticker":"C","target_weight":.98}]
        entry={"minimum_capital_inr":100000,"assumptions":{
            "fixed_cost_per_order":20,"statutory_cost_rate":.0012,"slippage_rate":.001}}
        result=suggested_withdrawal(plan,entry,target)
        self.assertEqual(result["reference_ticker"],"B")
        self.assertAlmostEqual(result["amount_inr"],778.24)
        self.assertEqual(result["reference_quantity"],2)

    def test_no_orders_has_no_invented_estimate(self):
        entry={"minimum_capital_inr":1000,"assumptions":{
            "fixed_cost_per_order":20,"statutory_cost_rate":.0012,"slippage_rate":.001}}
        self.assertIsNone(suggested_withdrawal({"orders":[]},entry,[]))

    def test_amount_validation(self):
        self.assertIn("INR 1.00",withdrawal_instructions(1))
        for amount in [0,-1,float("nan"),float("inf")]:
            with self.assertRaises(ValueError):
                withdrawal_instructions(amount)

    def test_actual_prompt_receives_selected_amount(self):
        source=Path(__file__).resolve().parents[1]/"public_portfolio_performance.py"
        tree=ast.parse(source.read_text(encoding="utf-8"))
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                      and node.name=="build_execution_plan_prompt")
        namespace={"json":json,"withdrawal_instructions":withdrawal_instructions,
                   "EXECUTION_SCENARIOS":{"Raise cash from existing holdings":"Ask for cash amount"}}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),"exec"),namespace)
        current={key:"value" for key in ["basket_id","publication_id","portfolio_fingerprint",
                                        "strategy_version","calculation_version","as_of","published_at"]}
        current.update(portfolio_version=3,cash_weight=0)
        prompt,_=namespace["build_execution_plan_prompt"](
            current,[],{},"Raise cash from existing holdings",withdrawal_amount=1)
        self.assertIn("REQUESTED NET CASH: INR 1.00",prompt)
        self.assertNotIn("Ask for cash amount",prompt)
        self.assertIn("AFTER withdrawal",prompt)


if __name__=="__main__": unittest.main()
