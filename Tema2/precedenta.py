from pathlib import Path
from typing import List, Sequence

MARKERS = ("<.", "=.")
PRODS: List[tuple[int, str, tuple[str, ...]]] = []
PRECEDENCE_MATRIX: dict[tuple[str, str], str] = {}
START = ""
BASE_DIR = Path(__file__).parent
PROD_FILE = BASE_DIR / "productions.txt"
TABLE_FILE = BASE_DIR / "precedence.txt"


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def read_productions(path: Path) -> List[tuple[str, tuple[str, ...]]]:
    if not path.exists():
        raise FileNotFoundError(f"Nu gasesc {path}.")
    prods: List[tuple[str, tuple[str, ...]]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
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
            raise ValueError(f"Membru stang invalid: {raw}")
        rhs = tuple(rhs_part.strip().split())
        prods.append((lhs, rhs))
    if not prods:
        raise ValueError("Fisierul de productii este gol.")
    return prods


def normalize_relation(value: str) -> str:
    if not value:
        return ""
    text = value.strip().lower()
    if not text or text in {"-", "_"}:
        return ""
    text = text.strip("()")
    text = text.replace("egal", "=.")
    if "accept" in text:
        return ""
    if "=." in text or text == "=":
        return "=."
    if "<." in text or text == "<":
        return "<."
    if ">." in text or text == ">":
        return ">."
    if "<" in text:
        return "<."
    if ">" in text:
        return ">."
    if "=" in text:
        return "=."
    return ""


def read_precedence_table(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        raise FileNotFoundError(f"Nu gasesc {path}.")
    lines = [strip_comment(raw) for raw in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < 2:
        raise ValueError("Tabela de precedenta trebuie sa aiba antet si cel putin un rand.")
    header = lines[0].split()
    if len(header) < 2:
        raise ValueError("Antetul trebuie sa contina cel putin doua simboluri.")
    columns = header[1:]
    matrix: dict[tuple[str, str], str] = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        row_symbol, cells = parts[0], parts[1:]
        if len(cells) < len(columns):
            cells.extend([""] * (len(columns) - len(cells)))
        for col, cell in zip(columns, cells):
            rel = normalize_relation(cell)
            if rel:
                matrix[(row_symbol, col)] = rel
    if not matrix:
        raise ValueError("Tabela de precedenta nu contine relatii.")
    return matrix


def tokenize(expr: str) -> List[str]:
    if not expr:
        raise ValueError("Expresia nu poate fi vida.")
    out, i = [], 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
        elif ch in "+*()":
            out.append(ch)
            i += 1
        elif ch == "$":
            raise ValueError("Nu adauga simbolul $.")
        elif ch.isalnum():
            j = i + 1
            while j < len(expr) and expr[j].isalnum():
                j += 1
            out.append("a")
            i = j
        else:
            raise ValueError(f"Simbol necunoscut: {ch}")
    out.append("$")
    return out


def top_symbol(stack: Sequence[str]) -> str:
    for s in reversed(stack):
        if s not in MARKERS:
            return s
    return "$"


def compact(stack: Sequence[str]) -> str:
    return "".join(s for s in stack if s not in MARKERS)


def pretty(indices: List[int]) -> str:
    names = [f"r{i}" for i in indices]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return " si ".join(names)
    return ", ".join(names[:-1]) + f" si {names[-1]}"


def reduce_step(stack: List[str]):
    bases = [i for i in range(len(stack) - 1, -1, -1) if stack[i] in MARKERS]
    try:
        bases.append(stack.index("$"))
    except ValueError as exc:
        raise ValueError("Nu gasesc simbolul $.") from exc
    best_len = -1
    best = None
    best_base = None
    choices = set()
    for base in bases:
        window = [s for s in stack[base + 1 :] if s not in MARKERS]
        if not window:
            continue
        hits = []
        for idx, lhs, rhs in PRODS:
            if len(rhs) <= len(window) and list(rhs) == window[-len(rhs) :]:
                hits.append((len(rhs), idx, lhs))
                choices.add(idx)
        if not hits:
            continue
        hits.sort(key=lambda item: (-item[0], item[1]))
        ln, idx, lhs = hits[0]
        if ln > best_len:
            best_len, best, best_base = ln, (idx, lhs), base
    if best is None or best_base is None:
        raise ValueError("Nu exista productie compatibila pentru reducere.")
    return best_base, best, sorted(choices)


def precedence_parse(expr: str):
    tokens = tokenize(expr)
    stack, pos, trace = ["$"], 0, []
    while True:
        a, b = top_symbol(stack), tokens[pos]
        rel = PRECEDENCE_MATRIX.get((a, b))
        if stack == ["$", "=.", START] and b == "$":
            trace.append((compact(stack), "accept", "$", ""))
            return True, trace
        if rel in ("<.", "=."):
            trace.append((compact(stack), rel, "".join(tokens[pos:]), "d"))
            stack.extend((rel, b))
            pos += 1
            continue
        if rel == ">.":  # reduce
            base, (idx, lhs), opts = reduce_step(stack)
            action = f"r{idx}"
            if len(opts) > 1:
                action += f" (ales dintre {pretty(opts)})"
            trace.append((compact(stack), rel, "".join(tokens[pos:]), action))
            del stack[base:]
            stack.extend(("=.", lhs))
            continue
        raise ValueError(f"Relatie de precedenta inexistenta intre '{a}' si '{b}'.")


def print_trace(rows: Sequence[Sequence[str]]):
    headers = [
        "Stiva",
        "Relatia de precedenta",
        "Sir de intrare",
        "Deplasare d / Reducere cu productia r n",
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def fmt(row):
        return " ".join(val.ljust(widths[i]) for i, val in enumerate(row))

    print(fmt(headers))
    for row in rows:
        print(fmt(row))


def load_configuration(prod_path: Path = PROD_FILE, table_path: Path = TABLE_FILE):
    global PRODS, START, PRECEDENCE_MATRIX
    raw_prods = read_productions(prod_path)
    PRODS = [(i + 1, lhs, rhs) for i, (lhs, rhs) in enumerate(raw_prods)]
    START = raw_prods[0][0]
    PRECEDENCE_MATRIX = read_precedence_table(table_path)


def main():
    load_configuration()
    default = "a*(a+a)"
    expr = input(f"Expresie de analizat [{default}]: ").strip() or default
    ok, rows = precedence_parse(expr)
    print()
    print_trace(rows)
    print()
    print("Rezultat:", "acceptat" if ok else "respins")


if __name__ == "__main__":
    main()
