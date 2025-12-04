#!/usr/bin/env python3
"""
Translator Push-Down LR(1) cu stivă de atribute (Tema 4).
Gramatica conform PDF:
 1. E -> E + T
 2. E -> E - T
 3. E -> T
 4. T -> T * F
 5. T -> T / F
 6. T -> F
 7. F -> ( E )
 8. F -> - ( E )
 9. F -> id

Tabelele de acțiuni/salt sunt cele din expr.output (inclusiv uminus).
Codul nu depinde de alte teme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# Tabela de acțiuni: (stare, terminal) -> di / rj / acc
ACTION_TABLE: Dict[Tuple[int, str], str] = {
    (0, "id"): "d1",
    (0, "-"): "d2",
    (0, "("): "d3",
    (1, "+"): "r9",
    (1, "-"): "r9",
    (1, "*"): "r9",
    (1, "/"): "r9",
    (1, ")"): "r9",
    (1, "$"): "r9",
    (2, "("): "d7",
    (3, "id"): "d1",
    (3, "-"): "d2",
    (3, "("): "d3",
    (4, "+"): "d10",
    (4, "-"): "d11",
    (4, "$"): "d9",
    (5, "+"): "r3",
    (5, "-"): "r3",
    (5, "*"): "d12",
    (5, "/"): "d13",
    (5, ")"): "r3",
    (5, "$"): "r3",
    (6, "+"): "r6",
    (6, "-"): "r6",
    (6, "*"): "r6",
    (6, "/"): "r6",
    (6, ")"): "r6",
    (6, "$"): "r6",
    (7, "id"): "d1",
    (7, "-"): "d2",
    (7, "("): "d3",
    (8, "+"): "d10",
    (8, "-"): "d11",
    (8, ")"): "d15",
    (9, "$"): "acc",
    (10, "id"): "d1",
    (10, "-"): "d2",
    (10, "("): "d3",
    (11, "id"): "d1",
    (11, "-"): "d2",
    (11, "("): "d3",
    (12, "id"): "d1",
    (12, "-"): "d2",
    (12, "("): "d3",
    (13, "id"): "d1",
    (13, "-"): "d2",
    (13, "("): "d3",
    (14, "+"): "d10",
    (14, "-"): "d11",
    (14, ")"): "d20",
    (15, "+"): "r7",
    (15, "-"): "r7",
    (15, "*"): "r7",
    (15, "/"): "r7",
    (15, ")"): "r7",
    (15, "$"): "r7",
    (16, "+"): "r1",
    (16, "-"): "r1",
    (16, "*"): "d12",
    (16, "/"): "d13",
    (16, ")"): "r1",
    (16, "$"): "r1",
    (17, "+"): "r2",
    (17, "-"): "r2",
    (17, "*"): "d12",
    (17, "/"): "d13",
    (17, ")"): "r2",
    (17, "$"): "r2",
    (18, "+"): "r4",
    (18, "-"): "r4",
    (18, "*"): "r4",
    (18, "/"): "r4",
    (18, ")"): "r4",
    (18, "$"): "r4",
    (19, "+"): "r5",
    (19, "-"): "r5",
    (19, "*"): "r5",
    (19, "/"): "r5",
    (19, ")"): "r5",
    (19, "$"): "r5",
    (20, "+"): "r8",
    (20, "-"): "r8",
    (20, "*"): "r8",
    (20, "/"): "r8",
    (20, ")"): "r8",
    (20, "$"): "r8",
}

# Tabela de salt: (stare, neterminal) -> stare
GOTO_TABLE: Dict[Tuple[int, str], int] = {
    (0, "E"): 4,
    (0, "T"): 5,
    (0, "F"): 6,
    (3, "E"): 8,
    (3, "T"): 5,
    (3, "F"): 6,
    (7, "E"): 14,
    (7, "T"): 5,
    (7, "F"): 6,
    (10, "T"): 16,
    (10, "F"): 6,
    (11, "T"): 17,
    (11, "F"): 6,
    (12, "F"): 18,
    (13, "F"): 19,
}

# Productii: idx -> (LHS, RHS)
PRODUCTIONS: Dict[int, Tuple[str, Tuple[str, ...]]] = {
    1: ("E", ("E", "+", "T")),
    2: ("E", ("E", "-", "T")),
    3: ("E", ("T",)),
    4: ("T", ("T", "*", "F")),
    5: ("T", ("T", "/", "F")),
    6: ("T", ("F",)),
    7: ("F", ("(", "E", ")")),
    8: ("F", ("-", "(", "E", ")")),
    9: ("F", ("id",)),
}

ATTR_SYMBOLS = {"E", "T", "F", "id"}
STACK_START = ("$", 0)


@dataclass
class TraceRow:
    stack: str
    attr_stack: str
    remaining_input: str
    action: str
    generated: str | None = None


def tokenize(expr: str) -> List[Tuple[str, str]]:
    if not expr:
        raise ValueError("Expresia nu poate fi vida.")
    tokens: List[Tuple[str, str]] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "+-*/()":
            tokens.append((ch, ch))
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            lexeme = expr[i:j]
            tokens.append(("id", lexeme))
            i = j
            continue
        if ch.isdigit():
            j = i + 1
            while j < len(expr) and expr[j].isdigit():
                j += 1
            lexeme = expr[i:j]
            tokens.append(("id", lexeme))
            i = j
            continue
        raise ValueError(f"Caracter necunoscut: '{ch}'")
    tokens.append(("$", "$"))
    return tokens


def stack_to_string(stack: Sequence[Tuple[str, int]]) -> str:
    return " ".join(f"{sym}{state}" for sym, state in stack)


class TranslatorPushDown:
    def __init__(self) -> None:
        self.code: List[str] = []
        self._temp_counter = 0

    def newtemp(self) -> str:
        self._temp_counter += 1
        return f"t{self._temp_counter}"

    def reduce_semantics(self, lhs: str, rhs: Tuple[str, ...], attr_stack: List[str]) -> str | None:
        vals: List[str] = []
        for sym in reversed(rhs):
            if sym in ATTR_SYMBOLS:
                if not attr_stack:
                    raise ValueError(f"Stiva de atribute goala in reducerea {lhs}->{rhs}")
                vals.append(attr_stack.pop())
        vals.reverse()

        def emit(line: str) -> str:
            self.code.append(line)
            return line

        if lhs == "E" and rhs == ("E", "+", "T"):
            e, t = vals
            temp = self.newtemp()
            attr_stack.append(temp)
            return emit(f"{temp} = {e} + {t}")
        if lhs == "E" and rhs == ("E", "-", "T"):
            e, t = vals
            temp = self.newtemp()
            attr_stack.append(temp)
            return emit(f"{temp} = {e} - {t}")
        if lhs == "E" and rhs == ("T",):
            attr_stack.append(vals[0])
            return None
        if lhs == "T" and rhs == ("T", "*", "F"):
            t, f = vals
            temp = self.newtemp()
            attr_stack.append(temp)
            return emit(f"{temp} = {t} * {f}")
        if lhs == "T" and rhs == ("T", "/", "F"):
            t, f = vals
            temp = self.newtemp()
            attr_stack.append(temp)
            return emit(f"{temp} = {t} / {f}")
        if lhs == "T" and rhs == ("F",):
            attr_stack.append(vals[0])
            return None
        if lhs == "F" and rhs == ("(", "E", ")"):
            attr_stack.append(vals[0])
            return None
        if lhs == "F" and rhs == ("-", "(", "E", ")"):
            e = vals[0]
            temp = self.newtemp()
            attr_stack.append(temp)
            return emit(f"{temp} = - {e}")
        if lhs == "F" and rhs == ("id",):
            attr_stack.append(vals[0])
            return None
        raise ValueError(f"Actiune semantica nedefinita pentru {lhs} -> {' '.join(rhs)}")

    def translate(self, expr: str) -> Tuple[bool, List[TraceRow], List[str]]:
        tokens = tokenize(expr)
        stack: List[Tuple[str, int]] = [STACK_START]
        attr_stack: List[str] = []
        trace: List[TraceRow] = []
        pos = 0
        self.code = []
        self._temp_counter = 0

        while True:
            state = stack[-1][1]
            lookahead = tokens[pos][0] if pos < len(tokens) else "$"
            action = ACTION_TABLE.get((state, lookahead))
            remaining = " ".join(tok[1] for tok in tokens[pos:]) if pos < len(tokens) else "$"
            stack_str = stack_to_string(stack)
            attr_str = " ".join(attr_stack)

            if action is None:
                trace.append(TraceRow(stack_str, attr_str, remaining, "eroare (actiune nedefinita)"))
                return False, trace, self.code

            action_desc = action if action != "acc" else "acceptare"

            if action == "acc":
                trace.append(TraceRow(stack_str, attr_str, remaining, action_desc))
                return True, trace, self.code

            if action.startswith("d"):
                next_state = int(action[1:])
                sym, lexeme = tokens[pos]
                stack.append((sym, next_state))
                if sym == "id":
                    attr_stack.append(lexeme)
                pos += 1
                trace.append(TraceRow(stack_str, attr_str, remaining, action_desc))
                continue

            if action.startswith("r"):
                prod_idx = int(action[1:])
                lhs, rhs = PRODUCTIONS[prod_idx]
                for _ in rhs:
                    stack.pop()
                goto_state = GOTO_TABLE.get((stack[-1][1], lhs))
                generated = self.reduce_semantics(lhs, rhs, attr_stack)
                if goto_state is None:
                    trace.append(
                        TraceRow(stack_to_string(stack), " ".join(attr_stack), remaining, f"eroare TS({stack[-1][1]}, {lhs})", generated)
                    )
                    return False, trace, self.code
                stack.append((lhs, goto_state))
                trace.append(TraceRow(stack_str, attr_str, remaining, f"r{prod_idx}: {lhs} -> {' '.join(rhs)}", generated))
                continue

            trace.append(TraceRow(stack_str, attr_str, remaining, f"actiune necunoscuta: {action}"))
            return False, trace, self.code


def print_trace(rows: Sequence[TraceRow]) -> None:
    headers = ("Stiva", "Stiva atribute", "Cuvant de intrare", "Actiune", "Cod generat")
    widths = [len(h) for h in headers]
    data = []
    for row in rows:
        gen = row.generated or ""
        data.append((row.stack, row.attr_stack, row.remaining_input, row.action, gen))
        widths[0] = max(widths[0], len(row.stack))
        widths[1] = max(widths[1], len(row.attr_stack))
        widths[2] = max(widths[2], len(row.remaining_input))
        widths[3] = max(widths[3], len(row.action))
        widths[4] = max(widths[4], len(gen))

    def fmt_row(values: Tuple[str, str, str, str, str]) -> str:
        return " ".join(val.ljust(widths[i]) for i, val in enumerate(values))

    print(fmt_row(headers))
    for row in data:
        print(fmt_row(row))


def main() -> None:
    translator = TranslatorPushDown()
    default_expr = "a1+a2*a3"
    try:
        raw = input(f"Expresie de analizat [{default_expr}]: ")
    except EOFError:
        raw = ""
    expr = raw.strip() or default_expr
    ok, trace, code = translator.translate(expr)
    print()
    print_trace(trace)
    print()
    print("Cod intermediar generat:")
    if code:
        for line in code:
            print(" ", line)
    else:
        print("  (nu s-a generat cod)")
    print("Rezultat:", "acceptat" if ok else "respins")


if __name__ == "__main__":
    main()
