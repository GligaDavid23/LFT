from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

BASE_DIR = Path(__file__).parent
PROD_FILE = BASE_DIR / "lr1_productions.txt"
ACTION_FILE = BASE_DIR / "lr1_actions.txt"
GOTO_FILE = BASE_DIR / "lr1_goto.txt"

PRODUCTIONS: Dict[int, Tuple[str, Tuple[str, ...]]] = {}
ACTION_TABLE: Dict[Tuple[int, str], str] = {}
GOTO_TABLE: Dict[Tuple[int, str], int] = {}
STACK_START = ("$", 0)


@dataclass
class TraceRow:
    stack: str
    remaining_input: str
    action: str


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def read_productions(path: Path) -> Dict[int, Tuple[str, Tuple[str, ...]]]:
    if not path.exists():
        raise FileNotFoundError(f"Nu gasesc fisierul de productii: {path}")
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
        if lhs and lhs[0].isdigit():
            lhs = lhs.split(".", 1)[-1].strip()
        if not lhs:
            raise ValueError(f"Membru stang invalid in linia: {raw}")
        rhs_tokens = rhs_part.strip().split()
        rhs = tuple(rhs_tokens if rhs_tokens else ("epsilon",))
        raw_prods.append((lhs, rhs))
    if not raw_prods:
        raise ValueError("Fisierul de productii este gol.")
    return {idx + 1: prod for idx, prod in enumerate(raw_prods)}


def read_action_table(path: Path) -> Dict[Tuple[int, str], str]:
    if not path.exists():
        raise FileNotFoundError(f"Nu gasesc fisierul cu tabela de actiuni: {path}")
    lines = [strip_comment(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [line for line in lines if line]
    if len(rows) < 2:
        raise ValueError("Tabela de actiuni trebuie sa aiba antet si cel putin un rand.")
    header = rows[0].split()
    if not header or header[0].lower() != "state":
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
            cell_norm = cell.strip()
            if cell_norm in {"", "-", "_"}:
                continue
            table[(state, sym)] = cell_norm
    return table


def read_goto_table(path: Path) -> Dict[Tuple[int, str], int]:
    if not path.exists():
        raise FileNotFoundError(f"Nu gasesc fisierul cu tabela de salt: {path}")
    lines = [strip_comment(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [line for line in lines if line]
    if len(rows) < 2:
        raise ValueError("Tabela de salt trebuie sa aiba antet si cel putin un rand.")
    header = rows[0].split()
    if not header or header[0].lower() != "state":
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
            cell_norm = cell.strip()
            if cell_norm in {"", "-", "_"}:
                continue
            table[(state, nt)] = int(cell_norm)
    return table


def load_configuration() -> None:
    global PRODUCTIONS, ACTION_TABLE, GOTO_TABLE
    PRODUCTIONS = read_productions(PROD_FILE)
    ACTION_TABLE = read_action_table(ACTION_FILE)
    GOTO_TABLE = read_goto_table(GOTO_FILE)


def tokenize(expr: str) -> List[str]:
    if not expr:
        raise ValueError("Expresia nu poate fi vida.")
    tokens: List[str] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "+-*()":
            tokens.append(ch)
            i += 1
            continue
        if ch == "$":
            raise ValueError("Nu introduce simbolul $ in input; este adaugat automat.")
        if ch.isalpha():
            j = i + 1
            while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            tokens.append("id")
            i = j
            continue
        if ch.isdigit():
            j = i + 1
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append("id")
            i = j
            continue
        raise ValueError(f"Caracter necunoscut: '{ch}'")
    tokens.append("$")
    return tokens


def stack_to_string(stack: Sequence[Tuple[str, int]]) -> str:
    return " ".join(f"{sym}{state}" for sym, state in stack)


def action_description(code: str) -> str:
    if code == "acc":
        return "acceptare"
    if code.startswith("d"):
        return f"d{code[1:]}"
    if code.startswith("r"):
        idx = int(code[1:])
        lhs, rhs = PRODUCTIONS[idx]
        rhs_str = " ".join(rhs)
        return f"r{idx}: {lhs} -> {rhs_str}"
    return code


def lr1_parse(expr: str) -> Tuple[bool, List[TraceRow]]:
    tokens = tokenize(expr)
    stack: List[Tuple[str, int]] = [STACK_START]
    pos = 0
    trace: List[TraceRow] = []

    while True:
        state = stack[-1][1]
        lookahead = tokens[pos] if pos < len(tokens) else "$"
        action = ACTION_TABLE.get((state, lookahead))
        stack_str = stack_to_string(stack)
        remaining = "".join(tokens[pos:]) if pos < len(tokens) else "$"

        if action is None:
            trace.append(TraceRow(stack_str, remaining, "eroare (actiune nedefinita)"))
            return False, trace

        trace.append(TraceRow(stack_str, remaining, action_description(action)))

        if action == "acc":
            return True, trace

        if action.startswith("d"):
            next_state = int(action[1:])
            stack.append((lookahead, next_state))
            pos += 1
            continue

        if action.startswith("r"):
            prod_idx = int(action[1:])
            lhs, rhs = PRODUCTIONS[prod_idx]
            pop_count = len(rhs) if rhs != ("epsilon",) else 0
            for _ in range(pop_count):
                if len(stack) == 1:
                    raise ValueError("Stiva s-a golit neasteptat in timpul reducerii.")
                stack.pop()
            goto_state = GOTO_TABLE.get((stack[-1][1], lhs))
            if goto_state is None:
                trace.append(
                    TraceRow(stack_to_string(stack), remaining, f"eroare TS({stack[-1][1]}, {lhs})")
                )
                return False, trace
            stack.append((lhs, goto_state))
            continue

        raise ValueError(f"Actiune necunoscuta: {action}")


def print_trace(rows: Sequence[TraceRow]) -> None:
    headers = ("Stiva", "Cuvant de intrare", "Actiune")
    widths = [len(h) for h in headers]
    data = [(row.stack, row.remaining_input, row.action) for row in rows]
    for stack_str, remaining, action in data:
        widths[0] = max(widths[0], len(stack_str))
        widths[1] = max(widths[1], len(remaining))
        widths[2] = max(widths[2], len(action))

    def fmt_row(values: Tuple[str, str, str]) -> str:
        return " ".join(val.ljust(widths[i]) for i, val in enumerate(values))

    print(fmt_row(headers))
    for stack_str, remaining, action in data:
        print(fmt_row((stack_str, remaining, action)))


def main() -> None:
    load_configuration()
    default = "id+(id*id)"
    expr = input(f"Expresie de analizat [{default}]: ").strip() or default
    ok, trace = lr1_parse(expr)
    print()
    print_trace(trace)
    print()
    print("Rezultat:", "acceptat" if ok else "respins")


if __name__ == "__main__":
    main()
