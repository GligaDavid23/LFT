from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

BASE_DIR = Path(__file__).parent
TEMA3_DIR = BASE_DIR.parent / "Tema3"

TABLE_DIR = (
    TEMA3_DIR
    if (TEMA3_DIR / "lr1_actions.txt").exists()
    else BASE_DIR
)
PROD_FILE = TABLE_DIR / "lr1_productions.txt"
ACTION_FILE = TABLE_DIR / "lr1_actions.txt"
GOTO_FILE = TABLE_DIR / "lr1_goto.txt"

PRODUCTIONS: Dict[int, Tuple[str, Tuple[str, ...]]] = {}
ACTION_TABLE: Dict[Tuple[int, str], str] = {}
GOTO_TABLE: Dict[Tuple[int, str], int] = {}
STACK_START = ("$", 0, None)


@dataclass
class Token:
    type: str
    lexeme: str


@dataclass
class TraceRow:
    stack: str
    remaining_input: str
    action: str
    attr_stack: str
    semantic: str
    code: str


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def read_productions(path: Path) -> Dict[int, Tuple[str, Tuple[str, ...]]]:
    if not path.exists():
        raise FileNotFoundError(f"Nu gasesc fisierul de productii: {path}")
    prods: Dict[int, Tuple[str, Tuple[str, ...]]] = {}
    auto_idx = 1
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = strip_comment(raw)
        if not line:
            continue
        if "->" not in line:
            raise ValueError(f"Lipseste '->' in linia: {raw}")
        lhs_part, rhs_part = line.split("->", 1)
        lhs_tokens = lhs_part.strip().split()
        if not lhs_tokens:
            raise ValueError(f"Membru stang invalid: {raw}")
        idx = None
        if lhs_tokens[0][0].isdigit():
            idx_token = lhs_tokens[0].rstrip(".")
            if idx_token.isdigit():
                idx = int(idx_token)
                lhs_tokens = lhs_tokens[1:]
        lhs = " ".join(lhs_tokens).strip()
        rhs_tokens = rhs_part.strip().split()
        rhs = tuple(rhs_tokens if rhs_tokens else ("epsilon",))
        key = idx if idx is not None else auto_idx
        prods[key] = (lhs, rhs)
        auto_idx += 1
    return prods


def _read_table(path: Path, cast: bool) -> Dict[Tuple[int, str], int | str]:
    if not path.exists():
        raise FileNotFoundError(f"Nu gasesc fisierul: {path}")
    rows = [strip_comment(line) for line in path.read_text(encoding="utf-8").splitlines() if strip_comment(line)]
    header = rows[0].split()
    symbols = header[1:]
    table: Dict[Tuple[int, str], int | str] = {}
    for line in rows[1:]:
        parts = line.split()
        if not parts:
            continue
        state = int(parts[0])
        cells = parts[1:]
        cells += ["-"] * (len(symbols) - len(cells))
        for sym, cell in zip(symbols, cells):
            cell = cell.strip()
            if cell in {"", "-", "_"}:
                continue
            table[(state, sym)] = int(cell) if cast else cell
    return table


def read_action_table(path: Path) -> Dict[Tuple[int, str], str]:
    return _read_table(path, False)  # type: ignore[return-value]


def read_goto_table(path: Path) -> Dict[Tuple[int, str], int]:
    return _read_table(path, True)  # type: ignore[return-value]


def load_configuration() -> None:
    global PRODUCTIONS, ACTION_TABLE, GOTO_TABLE
    PRODUCTIONS = read_productions(PROD_FILE)
    ACTION_TABLE = read_action_table(ACTION_FILE)
    GOTO_TABLE = read_goto_table(GOTO_FILE)


def tokenize(expr: str) -> List[Token]:
    if not expr:
        raise ValueError("Expresia nu poate fi vida.")
    tokens: List[Token] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "+-*/()":
            tokens.append(Token(ch, ch))
            i += 1
            continue
        if ch == "$":
            raise ValueError("Nu introduce simbolul $ in input; este adaugat automat.")
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            lex = expr[i:j]
            tokens.append(Token("id", lex))
            i = j
            continue
        if ch.isdigit():
            j = i + 1
            while j < len(expr) and expr[j].isdigit():
                j += 1
            lex = expr[i:j]
            tokens.append(Token("id", lex))
            i = j
            continue
        raise ValueError(f"Caracter necunoscut: '{ch}'")
    tokens.append(Token("$", "$"))
    return tokens


def stack_to_string(stack: Sequence[Tuple[str, int, Token | None]]) -> str:
    def display(sym: str, tok: Token | None) -> str:
        if tok and sym in {"+", "-", "*", "/", "(", ")", "id"}:
            return tok.lexeme
        return sym

    return " ".join(f"{display(sym, tok)}{state}" for sym, state, tok in stack)


def remaining_input(tokens: Sequence[Token], pos: int) -> str:
    return "".join(tok.lexeme for tok in tokens[pos:])


class TempGenerator:
    def __init__(self) -> None:
        self.counter = 0

    def new(self) -> str:
        self.counter += 1
        return f"t{self.counter}"


def apply_semantic(
    prod_idx: int,
    popped: List[Tuple[str, int, Token | None]],
    attr_stack: List[str],
    temp_gen: TempGenerator,
) -> Tuple[str, str | None]:
    emitted: str | None = None
    desc: List[str] = []

    def pop_attr(name: str) -> str:
        if not attr_stack:
            raise ValueError(f"Stiva de atribute este goala; nu pot scoate {name}.")
        val = attr_stack.pop()
        desc.append(f"{name}=pop({val})")
        return val

    lhs, rhs = PRODUCTIONS[prod_idx]

    if lhs == "F" and rhs == ("id",):
        token = popped[0][2]
        val = token.lexeme if token else "id"
        desc.append(f"F.p={val}")
        attr_stack.append(val)
        desc.append("push(F.p)")
    elif lhs == "F" and rhs == ("(", "E", ")"):
        e_val = pop_attr("E.p")
        desc.append(f"F.p={e_val}")
        attr_stack.append(e_val)
        desc.append("push(F.p)")
    elif lhs == "F" and rhs == ("-", "(", "E", ")"):
        e_val = pop_attr("E.p")
        temp = temp_gen.new()
        emitted = f"{temp} = uminus {e_val}"
        desc.append(f"F.p={temp}")
        attr_stack.append(temp)
        desc.append("push(F.p)")
    elif lhs == "T" and rhs == ("F",):
        f_val = pop_attr("F.p")
        desc.append(f"T.p={f_val}")
        attr_stack.append(f_val)
        desc.append("push(T.p)")
    elif lhs == "E" and rhs == ("T",):
        t_val = pop_attr("T.p")
        desc.append(f"E.p={t_val}")
        attr_stack.append(t_val)
        desc.append("push(E.p)")
    elif lhs == "T" and rhs == ("T", "*", "F"):
        f_val = pop_attr("F.p")
        t1_val = pop_attr("T1.p")
        temp = temp_gen.new()
        emitted = f"{temp} = {t1_val} * {f_val}"
        desc.append(f"T.p={temp}")
        attr_stack.append(temp)
        desc.append("push(T.p)")
    elif lhs == "T" and rhs == ("T", "/", "F"):
        f_val = pop_attr("F.p")
        t1_val = pop_attr("T1.p")
        temp = temp_gen.new()
        emitted = f"{temp} = {t1_val} / {f_val}"
        desc.append(f"T.p={temp}")
        attr_stack.append(temp)
        desc.append("push(T.p)")
    elif lhs == "E" and rhs == ("E", "+", "T"):
        t_val = pop_attr("T.p")
        e1_val = pop_attr("E1.p")
        temp = temp_gen.new()
        emitted = f"{temp} = {e1_val} + {t_val}"
        desc.append(f"E.p={temp}")
        attr_stack.append(temp)
        desc.append("push(E.p)")
    elif lhs == "E" and rhs == ("E", "-", "T"):
        t_val = pop_attr("T.p")
        e1_val = pop_attr("E1.p")
        temp = temp_gen.new()
        emitted = f"{temp} = {e1_val} - {t_val}"
        desc.append(f"E.p={temp}")
        attr_stack.append(temp)
        desc.append("push(E.p)")
    else:
        raise ValueError(f"Produtie necunoscuta: {lhs} -> {' '.join(rhs)}")

    semantic_text = "; ".join(desc)
    return semantic_text, emitted


def translate(expr: str) -> Tuple[bool, List[TraceRow], List[str]]:
    tokens = tokenize(expr)
    stack: List[Tuple[str, int, Token | None]] = [STACK_START]
    attr_stack: List[str] = []
    code: List[str] = []
    pos = 0
    temp_gen = TempGenerator()
    trace: List[TraceRow] = []

    # inregistram starea initiala
    trace.append(
        TraceRow(
            stack_to_string(stack),
            remaining_input(tokens, pos),
            "-",
            " ".join(attr_stack),
            "",
            "",
        )
    )

    while True:
        state = stack[-1][1]
        lookahead = tokens[pos].type if pos < len(tokens) else "$"
        action = ACTION_TABLE.get((state, lookahead))

        if action is None:
            trace.append(
                TraceRow(
                    stack_to_string(stack),
                    remaining_input(tokens, pos),
                    "eroare",
                    " ".join(attr_stack),
                    "actiune nedefinita",
                    "",
                )
            )
            return False, trace, code

        if action == "acc":
            trace.append(
                TraceRow(
                    stack_to_string(stack),
                    remaining_input(tokens, pos),
                    "acc",
                    " ".join(attr_stack),
                    "",
                    "",
                )
            )
            return True, trace, code

        if action.startswith("d"):
            next_state = int(action[1:])
            stack.append((lookahead, next_state, tokens[pos]))
            pos += 1
            trace.append(
                TraceRow(
                    stack_to_string(stack),
                    remaining_input(tokens, pos),
                    f"d{next_state}",
                    " ".join(attr_stack),
                    "",
                    "",
                )
            )
            continue

        if action.startswith("r"):
            prod_idx = int(action[1:])
            lhs, rhs = PRODUCTIONS[prod_idx]
            pop_count = len(rhs) if rhs != ("epsilon",) else 0
            popped: List[Tuple[str, int, Token | None]] = []
            for _ in range(pop_count):
                if len(stack) == 1:
                    raise ValueError("Stiva s-a golit neasteptat in timpul reducerii.")
                popped.append(stack.pop())
            semantic_text, emitted = apply_semantic(prod_idx, popped, attr_stack, temp_gen)
            goto_state = GOTO_TABLE.get((stack[-1][1], lhs))
            if goto_state is None:
                trace.append(
                    TraceRow(
                        stack_to_string(stack),
                        remaining_input(tokens, pos),
                        f"eroare TS({stack[-1][1]}, {lhs})",
                        " ".join(attr_stack),
                        "",
                        "",
                    )
                )
                return False, trace, code
            stack.append((lhs, goto_state, None))
            if emitted:
                code.append(emitted)
            action_desc = f"r{prod_idx}: {lhs}->{ ' '.join(rhs)}; goto {goto_state}"
            trace.append(
                TraceRow(
                    stack_to_string(stack),
                    remaining_input(tokens, pos),
                    action_desc,
                    " ".join(attr_stack),
                    semantic_text,
                    emitted or "",
                )
            )
            continue

        raise ValueError(f"Actiune necunoscuta: {action}")


def print_trace(rows: Sequence[TraceRow]) -> None:
    headers = ("Stiva APD", "Intrare", "Actiune", "Stiva atribute", "Semantica / Cod")
    widths = [len(h) for h in headers]
    data = [
        (
            row.stack,
            row.remaining_input,
            row.action,
            row.attr_stack,
            (row.semantic + ("; " + row.code if row.code else "")).strip(),
        )
        for row in rows
    ]
    for stack_str, remaining, action, attrs, semantic in data:
        widths[0] = max(widths[0], len(stack_str))
        widths[1] = max(widths[1], len(remaining))
        widths[2] = max(widths[2], len(action))
        widths[3] = max(widths[3], len(attrs))
        widths[4] = max(widths[4], len(semantic))

    def fmt(values: Tuple[str, str, str, str, str]) -> str:
        return " ".join(val.ljust(widths[i]) for i, val in enumerate(values))

    print(fmt(headers))
    for row in data:
        print(fmt(row))


def main() -> None:
    load_configuration()
    default = "a+a*a"
    expr = input(f"Expresie de analizat [{default}]: ").strip() or default
    ok, trace, code = translate(expr)
    print()
    print_trace(trace)
    print()
    if code:
        print("Cod intermediar:")
        for line in code:
            print(" ", line)
    else:
        print("Nu s-a generat cod intermediar.")
    print("Rezultat:", "acceptat" if ok else "respins")


if __name__ == "__main__":
    main()
