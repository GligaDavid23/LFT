#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
PROD_FILE = BASE_DIR / "lr1_productions.txt"
ACTION_FILE = BASE_DIR / "lr1_actions.txt"
GOTO_FILE = BASE_DIR / "lr1_goto.txt"

STACK_START = ("$", 0)
ATTR_SYMBOLS = {"id"}


@dataclass
class TraceRow:
    stack: str
    attr_stack: str
    remaining_input: str
    action: str
    generated: str | None = None


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def read_productions(path: Path) -> Dict[int, Tuple[str, Tuple[str, ...]]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Nu gasesc fisierul de productii: {path}. Ruleaza mai intai build_tables.py pentru a genera tabelele din expr.output."
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    raw_prods: List[Tuple[str, Tuple[str, ...]]] = []
    for raw in lines:
        line = strip_comment(raw)
        if not line:
            continue
        if "->" not in line:
            raise ValueError(f"Lipseste '->' in linia: {raw}")
        lhs_part, rhs_part = line.split("->", 1)
        lhs = lhs_part.strip()
        rhs_tokens = rhs_part.strip().split()
        if not lhs or not rhs_tokens:
            raise ValueError(f"Linie de productie invalida: {raw}")
        raw_prods.append((lhs, tuple(rhs_tokens)))
    if not raw_prods:
        raise ValueError("Fisierul de productii este gol.")
    return {idx + 1: prod for idx, prod in enumerate(raw_prods)}


def read_action_table(path: Path) -> Dict[Tuple[int, str], str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Nu gasesc fisierul cu tabela de actiuni: {path}. Ruleaza mai intai build_tables.py pentru a genera tabelele din expr.output."
        )
    rows = [strip_comment(line) for line in path.read_text(encoding="utf-8").splitlines() if strip_comment(line)]
    if len(rows) < 2:
        raise ValueError("Tabela de actiuni trebuie sa aiba antet si cel putin un rand.")
    header = rows[0].split()
    if header[0].lower() != "state":
        raise ValueError("Prima coloana din tabela de actiuni trebuie sa fie 'state'.")
    symbols = header[1:]
    table: Dict[Tuple[int, str], str] = {}
    for line in rows[1:]:
        parts = line.split()
        if not parts:
            continue
        state = int(parts[0])
        cells = parts[1:]
        if len(cells) < len(symbols):
            cells.extend(["-"] * (len(symbols) - len(cells)))
        for sym, cell in zip(symbols, cells):
            val = cell.strip()
            if val in {"", "-", "_"}:
                continue
            table[(state, sym)] = val
    return table


def read_goto_table(path: Path) -> Dict[Tuple[int, str], int]:
    if not path.exists():
        raise FileNotFoundError(
            f"Nu gasesc fisierul cu tabela de salt: {path}. Ruleaza mai intai build_tables.py pentru a genera tabelele din expr.output."
        )
    rows = [strip_comment(line) for line in path.read_text(encoding="utf-8").splitlines() if strip_comment(line)]
    if len(rows) < 2:
        raise ValueError("Tabela de salt trebuie sa aiba antet si cel putin un rand.")
    header = rows[0].split()
    if header[0].lower() != "state":
        raise ValueError("Prima coloana din tabela de salt trebuie sa fie 'state'.")
    nonterminals = header[1:]
    table: Dict[Tuple[int, str], int] = {}
    for line in rows[1:]:
        parts = line.split()
        if not parts:
            continue
        state = int(parts[0])
        cells = parts[1:]
        if len(cells) < len(nonterminals):
            cells.extend(["-"] * (len(nonterminals) - len(cells)))
        for nt, cell in zip(nonterminals, cells):
            val = cell.strip()
            if val in {"", "-", "_"}:
                continue
            table[(state, nt)] = int(val)
    return table


def stack_to_string(stack: Sequence[Tuple[str, int]]) -> str:
    return " ".join(f"{sym}{state}" for sym, state in stack)


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


class Translator:
    def __init__(self) -> None:
        self.productions = read_productions(PROD_FILE)
        self.action_table = read_action_table(ACTION_FILE)
        self.goto_table = read_goto_table(GOTO_FILE)
        self._temp_counter = 0
        self.code: List[str] = []

    def newtemp(self) -> str:
        self._temp_counter += 1
        return f"t{self._temp_counter}"

    def action_description(self, code: str) -> str:
        if code == "acc":
            return "acceptare"
        if code.startswith("d"):
            return f"d{code[1:]}"
        if code.startswith("r"):
            idx = int(code[1:])
            lhs, rhs = self.productions[idx]
            rhs_str = " ".join(rhs)
            return f"r{idx}: {lhs} -> {rhs_str}"
        return code

    def reduce_semantics(self, prod_idx: int, lhs: str, rhs: Tuple[str, ...], attr_stack: List[str]) -> str | None:
        # Folosim pattern-uri pe productie ca sa fim robusti la renumerotari.
        rhs_tuple = tuple(rhs)

        def pop_one() -> str:
            if not attr_stack:
                raise ValueError(f"Stiva de atribute goala la reducerea regulii {prod_idx}: {lhs} -> {' '.join(rhs_tuple)}")
            return attr_stack.pop()

        if lhs == "E" and rhs_tuple == ("E", "+", "T"):
            t_attr = pop_one()
            e_attr = pop_one()
            temp = self.newtemp()
            line = f"{temp} = {e_attr} + {t_attr}"
            self.code.append(line)
            attr_stack.append(temp)
            return line

        if lhs == "E" and rhs_tuple == ("E", "-", "T"):
            t_attr = pop_one()
            e_attr = pop_one()
            temp = self.newtemp()
            line = f"{temp} = {e_attr} - {t_attr}"
            self.code.append(line)
            attr_stack.append(temp)
            return line

        if lhs == "E" and rhs_tuple == ("T",):
            t_attr = pop_one()
            attr_stack.append(t_attr)
            return None

        if lhs == "T" and rhs_tuple == ("T", "*", "F"):
            f_attr = pop_one()
            t_attr = pop_one()
            temp = self.newtemp()
            line = f"{temp} = {t_attr} * {f_attr}"
            self.code.append(line)
            attr_stack.append(temp)
            return line

        if lhs == "T" and rhs_tuple == ("T", "/", "F"):
            f_attr = pop_one()
            t_attr = pop_one()
            temp = self.newtemp()
            line = f"{temp} = {t_attr} / {f_attr}"
            self.code.append(line)
            attr_stack.append(temp)
            return line

        if lhs == "T" and rhs_tuple == ("F",):
            f_attr = pop_one()
            attr_stack.append(f_attr)
            return None

        if lhs == "F" and rhs_tuple == ("(", "E", ")"):
            e_attr = pop_one()
            attr_stack.append(e_attr)
            return None

        if lhs == "F" and rhs_tuple == ("-", "(", "E", ")"):
            e_attr = pop_one()
            temp = self.newtemp()
            line = f"{temp} = - {e_attr}"
            self.code.append(line)
            attr_stack.append(temp)
            return line

        if lhs == "F" and rhs_tuple == ("id",):
            id_attr = pop_one()
            attr_stack.append(id_attr)
            return None

        raise ValueError(f"Actiune semantica nedefinita pentru productia {prod_idx}: {lhs} -> {' '.join(rhs_tuple)}")

    def translate(self, expr: str) -> Tuple[bool, List[TraceRow], List[str]]:
        tokens = tokenize(expr)
        stack: List[Tuple[str, int]] = [STACK_START]
        attr_stack: List[str] = []
        pos = 0
        trace: List[TraceRow] = []
        self.code = []
        self._temp_counter = 0

        while True:
            state = stack[-1][1]
            lookahead = tokens[pos][0]
            action = self.action_table.get((state, lookahead))
            remaining = " ".join(tok[1] for tok in tokens[pos:])
            stack_str = stack_to_string(stack)
            attr_str = " ".join(attr_stack)

            if action is None:
                trace.append(TraceRow(stack_str, attr_str, remaining, "eroare (actiune nedefinita)"))
                return False, trace, self.code

            action_desc = self.action_description(action)

            if action == "acc":
                trace.append(TraceRow(stack_str, attr_str, remaining, action_desc))
                return True, trace, self.code

            if action.startswith("d"):
                next_state = int(action[1:])
                sym, lexeme = tokens[pos]
                stack.append((sym, next_state))
                if sym in ATTR_SYMBOLS:
                    attr_stack.append(lexeme)
                pos += 1
                trace.append(TraceRow(stack_str, attr_str, remaining, action_desc))
                continue

            if action.startswith("r"):
                prod_idx = int(action[1:])
                lhs, rhs = self.productions[prod_idx]
                pop_count = len(rhs) if rhs != ("epsilon",) else 0
                for _ in range(pop_count):
                    stack.pop()
                goto_state = self.goto_table.get((stack[-1][1], lhs))
                generated = self.reduce_semantics(prod_idx, lhs, rhs, attr_stack)
                if goto_state is None:
                    trace.append(
                        TraceRow(stack_to_string(stack), " ".join(attr_stack), remaining, f"eroare TS({stack[-1][1]}, {lhs})", generated)
                    )
                    return False, trace, self.code
                stack.append((lhs, goto_state))
                trace.append(TraceRow(stack_str, attr_str, remaining, action_desc, generated))
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
    translator = Translator()
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
