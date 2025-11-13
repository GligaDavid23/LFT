from typing import List, Sequence

TERMINALS = ("a", "+", "*", "(", ")", "$")
NONTERMINALS = ("E", "T", "F")
SYMS = set(TERMINALS) | set(NONTERMINALS) | {"S'"}
MARKERS = ("<.", "=.")
START = "E"
RAW_PRODS = [
    ("E", ("E", "+", "T")),
    ("E", ("T",)),
    ("T", ("T", "*", "F")),
    ("T", ("F",)),
    ("F", ("a",)),
    ("F", ("(", "E", ")")),
]
PRODS = [(i + 1, lhs, rhs) for i, (lhs, rhs) in enumerate(RAW_PRODS)]
AUG_PRODS = RAW_PRODS + [("S'", ("$", START, "$"))]


def propagate(front: bool) -> dict:
    data = {s: set() for s in SYMS}
    for t in TERMINALS:
        data[t].add(t)
    changed = True
    while changed:
        changed = False
        for lhs, rhs in AUG_PRODS:
            if not rhs:
                continue
            key = rhs[0] if front else rhs[-1]
            add = {key} | data[key]
            if add - data[lhs]:
                data[lhs].update(add)
                changed = True
    return data


def build_matrix() -> dict:
    tini, tfin, matrix = propagate(True), propagate(False), {}

    def put(x: str, y: str, val: str) -> None:
        if y not in TERMINALS:
            return
        cur = matrix.get((x, y))
        if cur and cur != val:
            raise ValueError(f"Conflict on {(x, y)}: {cur} vs {val}")
        matrix[(x, y)] = val

    for _, rhs in AUG_PRODS:
        for i in range(len(rhs) - 1):
            x, y = rhs[i], rhs[i + 1]
            put(x, y, "=.")
            if y in NONTERMINALS or y == "S'":
                for sym in tini[y]:
                    if sym in TERMINALS:
                        put(x, sym, "<.")
            if x in NONTERMINALS or x == "S'":
                targets = tini[y] | {y}
                for left in tfin[x]:
                    for sym in targets:
                        if sym in TERMINALS:
                            put(left, sym, ">.")
    return matrix


PRECEDENCE_MATRIX = build_matrix()


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


def main():
    default = "a*(a+a)"
    expr = input(f"Expresie de analizat [{default}]: ").strip() or default
    ok, rows = precedence_parse(expr)
    print()
    print_trace(rows)

    print()
    print("Rezultat:", "acceptat" if ok else "respins")


if __name__ == "__main__":
    main()
