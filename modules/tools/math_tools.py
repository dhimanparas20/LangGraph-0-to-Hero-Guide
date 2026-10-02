"""Math tools for the agents."""

import re

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluates a math expression, e.g. '12 * (3 + 4) / 2'. Supports + - * / % and parentheses.

    Args:
        expression: the arithmetic expression to evaluate. Only digits and
            + - * / % ( ) . are allowed - anything else is rejected.

    Returns:
        The result as a string, or an "Error: ..." message.
    """
    expression = expression.strip()

    # only allow safe characters: digits, operators, parentheses, dots, spaces
    if not re.fullmatch(r"[0-9+\-*/%(). ]+", expression):
        return "Error: only numbers and + - * / % ( ) are allowed."

    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - chars are whitelisted above
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception:
        return "Error: could not evaluate the expression."
