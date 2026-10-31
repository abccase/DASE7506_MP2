"""Arithmetic verifier used as an inference-time feature extractor (v31).

Reads only the candidate's own text and its officially parsed final answer.
It never looks at references, other candidates, or the network.

The six features are deliberately conservative: an equality is counted only when
both sides reduce to a number with no variable touching an operator. A permissive
parser was tried and rejected -- it started inventing equalities from prose and
the signal disappeared (see report/EXPERIMENTS.md, 2026-10-31).
"""
import ast
import operator
import re
from fractions import Fraction

import numpy as np

BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}
NUMTOK = r"\d+(?:\.\d+)?"
SYM_WINDOW = re.compile(rf"{NUMTOK}\s*[*+\-/]\s*[a-zA-Z]|[a-zA-Z]\s*[*+\-/]\s*{NUMTOK}")
LEAD_NUM = re.compile(r"^\s*[-+]?[\d(][-+*/()\d.\s]*")

COLUMNS = ("n_eq", "n_false", "false_rate", "any_false", "in_rhs", "n_symbolic")


def normalise(text):
    t = re.sub(r"\\boxed\s*\{([^{}]*)\}", r" \1 ", text)
    t = re.sub(r"\\text\s*\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r" ((\1)/(\2)) ", t)
    t = t.replace("\\times", "*").replace("\\cdot", "*").replace("×", "*")
    t = t.replace("\\div", "/").replace("÷", "/")
    t = t.replace("\\%", "/100").replace("%", "/100")
    for junk in ("$", "\\,", "\\;", "\\!", "\\left", "\\right", "**", "\\ "):
        t = t.replace(junk, " ")
    return t.replace(",", "")


def safe_eval(expr):
    expr = expr.strip()
    if not expr or len(expr) > 120:
        return None
    if not re.fullmatch(r"[-+*/()\d.\s]+", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in BIN:
            return BIN[type(n.op)](walk(n.left), walk(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in BIN:
            return BIN[type(n.op)](walk(n.operand))
        raise ValueError("unsupported")

    try:
        v = walk(tree)
    except (ValueError, ZeroDivisionError, TypeError, OverflowError):
        return None
    return v if np.isfinite(v) else None


def lead_value(seg):
    """Longest numeric prefix of a segment, evaluated."""
    seg = seg.strip()
    if not seg or not re.match(r"[-+]?[\d(]", seg):
        return None
    m = LEAD_NUM.match(seg)
    return safe_eval(m.group(0)) if m else None


def extract(text, final):
    """Return the six verifier features for one candidate."""
    t = normalise(text)
    n_eq = n_false = 0
    rhs_values = []
    for chunk in t.split("\n"):
        if "=" not in chunk:
            continue
        parts = chunk.split("=")
        for i in range(len(parts) - 1):
            lhs, rhs = parts[i], parts[i + 1]
            if re.search(r"[A-Za-z]", lhs.strip()):
                continue
            a, b = lead_value(lhs), lead_value(rhs)
            if a is None or b is None:
                continue
            n_eq += 1
            rhs_values.append(b)
            if abs(a - b) > 1e-6 * max(1.0, abs(a), abs(b)):
                n_false += 1
    final_v = None
    if final is not None:
        try:
            final_v = float(Fraction(final))
        except (ValueError, ZeroDivisionError):
            final_v = None
    in_rhs = int(final_v is not None and any(
        abs(v - final_v) <= 1e-6 * max(1.0, abs(final_v)) for v in rhs_values))
    return np.array([
        n_eq,
        n_false,
        n_false / n_eq if n_eq else 0.0,
        int(n_false > 0),
        in_rhs,
        len(SYM_WINDOW.findall(t)),
    ], dtype=np.float64)


def build_rows(bank):
    """[N, 4, 6] verifier features; the last entry of each row is never used."""
    from common import parse_answer
    out = np.zeros((len(bank), 4, len(COLUMNS)), dtype=np.float64)
    for q, row in enumerate(bank):
        if len(row["candidates"]) != 4:
            raise ValueError("Expected four candidates per question")
        for i, candidate in enumerate(row["candidates"]):
            out[q, i] = extract(candidate["text"], parse_answer(candidate["text"]))
    return out
