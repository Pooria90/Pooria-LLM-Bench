"""Small tool-use task built from Railtracks function nodes.

The agent gets three deterministic tools (an order database, an exchange-rate
table and a calculator). Every question needs one or more tool calls, because the
data only exists behind the tools. Scoring checks the final number and records
which tools were actually called, so a model that guesses without calling tools
shows up in the details even when it happens to be right.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

import railtracks as rt

from .. import scorers
from ..instrument import CURRENT_TRACE
from .base import BuildContext, Sample, Score, Task

_ORDERS = {
    "A-1001": {"customer": "Lumen Labs", "amount": 1250.00, "currency": "EUR", "status": "shipped"},
    "A-1002": {"customer": "Northwind", "amount": 980.50, "currency": "GBP", "status": "pending"},
    "A-1003": {"customer": "Lumen Labs", "amount": 410.25, "currency": "USD", "status": "shipped"},
    "A-1004": {"customer": "Kestrel Co", "amount": 76000.00, "currency": "JPY", "status": "cancelled"},
}
_RATES_TO_USD = {"USD": 1.0, "EUR": 1.0825, "GBP": 1.2710, "JPY": 0.006650, "CAD": 0.7310}

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(expr: str) -> float:
    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    return ev(ast.parse(expr, mode="eval"))


@rt.function_node
def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID.

    Args:
        order_id (str): The order ID, for example "A-1001".

    Returns:
        The order's customer, amount, currency and status, or an error message.
    """
    order = _ORDERS.get(order_id.strip().upper())
    return dict(order) if order else {"error": f"No order with id {order_id!r}"}


@rt.function_node
def get_exchange_rate(currency: str) -> dict:
    """Get how many US dollars one unit of a currency is worth.

    Args:
        currency (str): ISO currency code, for example "EUR".

    Returns:
        The rate to USD, or an error message for unknown currencies.
    """
    code = currency.strip().upper()
    if code not in _RATES_TO_USD:
        return {"error": f"Unknown currency {currency!r}"}
    return {"currency": code, "usd_per_unit": _RATES_TO_USD[code]}


@rt.function_node
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression with + - * / ** % and parentheses.

    Args:
        expression (str): The expression, for example "1250 * 1.0825".

    Returns:
        The result as a string, or an error message.
    """
    try:
        return repr(_safe_eval(expression))
    except Exception as exc:  # the model should see the error and recover
        return f"error: {exc}"


SYSTEM = (
    "You are an operations assistant. Order data and exchange rates are only "
    "available through your tools; never guess them. Use the calculator for "
    "arithmetic. End your reply with a final line 'Answer: <number>' and nothing after it."
)


def _usd(order_id: str) -> float:
    o = _ORDERS[order_id]
    return o["amount"] * _RATES_TO_USD[o["currency"]]


# (id, question, target, expected tools)
_ITEMS: list[tuple[str, str, float, set[str]]] = [
    ("t1", "What is the amount of order A-1003, in its own currency?", 410.25, {"lookup_order"}),
    ("t2", "What is order A-1001 worth in US dollars? Round to 2 decimals.", round(_usd("A-1001"), 2), {"lookup_order", "get_exchange_rate"}),
    ("t3", "What is order A-1002 worth in US dollars? Round to 2 decimals.", round(_usd("A-1002"), 2), {"lookup_order", "get_exchange_rate"}),
    ("t4", "What is the combined value in US dollars of all *shipped* orders among A-1001, A-1002 and A-1003? Round to 2 decimals.",
     round(_usd("A-1001") + _usd("A-1003"), 2), {"lookup_order", "get_exchange_rate"}),
    ("t5", "How many Canadian dollars is order A-1004 worth? Round to 2 decimals.",
     round(_usd("A-1004") / _RATES_TO_USD["CAD"], 2), {"lookup_order", "get_exchange_rate"}),
    ("t6", "Order A-1009 was mentioned by a customer. What is its amount? If it does not exist, answer 0.", 0.0, {"lookup_order"}),
]


class SmokeTools(Task):
    name = "smoke-tools"
    description = "6 questions that need 1-4 tool calls (order DB, FX rates, calculator)."

    def load(self, limit: int | None = None) -> list[Sample]:
        samples = [
            Sample(id=i, input=q, target=t, metadata={"category": "tools", "expected_tools": sorted(e)})
            for i, q, t, e in _ITEMS
        ]
        return samples[:limit] if limit else samples

    def build(self, ctx: BuildContext):
        return ctx.agent(
            name="OpsAgent",
            system_message=SYSTEM,
            tool_nodes=[lookup_order, get_exchange_rate, calculator],
        )

    def score(self, sample: Sample, output: Any) -> Score:
        score = scorers.numeric_match(str(output), sample.target, rel_tol=1e-3, abs_tol=0.011)
        trace = CURRENT_TRACE.get()
        used = sorted({name for c in (trace.calls if trace else []) for name in c.tool_calls})
        missing = sorted(set(sample.metadata["expected_tools"]) - set(used))
        score.details |= {"tools_used": used, "missing_tools": missing}
        return score
