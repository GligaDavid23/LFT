from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Tema1")))
from generator import Grammar, parse_grammar  # type: ignore

END = "$"
EPS = "<eps>"

EXPECTED_NONTERMINALS = {"E", "T", "F"}
EXPECTED_TERMINALS = {"a", "+", "*", "(", ")"}
PRECEDENCE_ORDER = ["E", "T", "F", "a", "+", "*", "(", ")", "$"]
RelationCell = Union[Tuple[str, ...], str]
PRECEDENCE_TABLE: Dict[str, Dict[str, RelationCell]] = {
    "E": {"+": ("=",), ")": ("=",), "$": "accept"},
    "T": {"+": (">",), "*": ("=",), ")": (">",), "$": (">",)},
    "F": {"+": (">",), "*": (">",), ")": (">",), "$": (">",)},
    "a": {"+": (">",), "*": (">",), ")": (">",), "$": (">",)},
    "+": {"T": ("=", "<"), "F": ("<",), "a": ("<",), "(": ("<",)},
    "*": {"E": ("<",), "F": ("=",), "a": ("<",), "(": ("<",)},
    "(": {"E": ("=", "<"), "T": ("<",), "F": ("<",), "a": ("<",), "(": ("<",)},
    ")": {"+": (">",), "*": (">",), ")": (">",), "$": (">",)},
    "$": {"E": ("=", "<"), "T": ("<",), "F": ("<",), "a": ("<",), "(": ("<",)},
}


@dataclass(frozen=True)
class LR1Item:
    head: str
    body: Tuple[str, ...]
    dot: int
    lookahead: str

    def next_symbol(self) -> Optional[str]:
        return self.body[self.dot] if self.dot < len(self.body) else None

    def advance(self) -> "LR1Item":
        return LR1Item(self.head, self.body, self.dot + 1, self.lookahead)

    def complete(self) -> bool:
        return self.dot >= len(self.body)


def augment(grammar: Grammar) -> Tuple[Grammar, str]:
    fresh = grammar.start + "'"
    while fresh in grammar.nonterminals:
        fresh += "'"
    productions = dict(grammar.productions)
    productions.setdefault(fresh, []).insert(0, (grammar.start,))
    return Grammar(
        nonterminals={*grammar.nonterminals, fresh},
        terminals={*grammar.terminals, END},
        start=fresh,
        productions=productions,
    ), fresh


def first_sets(grammar: Grammar) -> Dict[str, Set[str]]:
    first: Dict[str, Set[str]] = {t: {t} for t in grammar.terminals}
    first.update({nt: set() for nt in grammar.nonterminals})
    changed = True
    while changed:
        changed = False
        for head, bodies in grammar.productions.items():
            for body in bodies:
                if not body and EPS not in first[head]:
                    first[head].add(EPS)
                    changed = True
                    continue
                nullable = True
                for symbol in body:
                    current = first.get(symbol, {symbol})
                    before = len(first[head])
                    first[head].update(x for x in current if x != EPS)
                    if len(first[head]) != before:
                        changed = True
                    if EPS not in current:
                        nullable = False
                        break
                if nullable and EPS not in first[head]:
                    first[head].add(EPS)
                    changed = True
    return first


def first_of(sequence: Sequence[str], first: Dict[str, Set[str]]) -> Set[str]:
    if not sequence:
        return {EPS}
    result: Set[str] = set()
    for symbol in sequence:
        current = first.get(symbol, {symbol})
        result.update(x for x in current if x != EPS)
        if EPS not in current:
            break
    else:
        result.add(EPS)
    return result


def closure(items: Iterable[LR1Item], grammar: Grammar, first: Dict[str, Set[str]]) -> Set[LR1Item]:
    result: Set[LR1Item] = set(items)
    queue: deque[LR1Item] = deque(items)
    while queue:
        item = queue.popleft()
        symbol = item.next_symbol()
        if symbol is None or not grammar.is_nonterminal(symbol):
            continue
        lookaheads = first_of(item.body[item.dot + 1 :] + (item.lookahead,), first) - {EPS}
        for production in grammar.productions.get(symbol, []):
            for la in lookaheads:
                candidate = LR1Item(symbol, production, 0, la)
                if candidate not in result:
                    result.add(candidate)
                    queue.append(candidate)
    return result


def goto(items: Iterable[LR1Item], symbol: str, grammar: Grammar, first: Dict[str, Set[str]]) -> Set[LR1Item]:
    moved = {item.advance() for item in items if item.next_symbol() == symbol}
    return closure(moved, grammar, first) if moved else set()


def lr1_automaton(grammar: Grammar) -> Tuple[List[Tuple[str, List[str]]], Dict[int, Dict[str, List[str]]], Dict[int, Dict[str, int]]]:
    augmented, start_prime = augment(grammar)
    first = first_sets(augmented)
    start_item = LR1Item(start_prime, (grammar.start,), 0, END)
    start_state = closure({start_item}, augmented, first)

    states: List[Set[LR1Item]] = []
    order: Dict[frozenset[LR1Item], int] = {}
    edges: Dict[Tuple[int, str], int] = {}
    queue: deque[Set[LR1Item]] = deque()

    def register(state: Set[LR1Item]) -> int:
        key = frozenset(state)
        if key in order:
            return order[key]
        idx = len(states)
        states.append(state)
        order[key] = idx
        queue.append(state)
        return idx

    register(start_state)
    symbols = sorted(augmented.nonterminals | augmented.terminals)

    while queue:
        state = queue.popleft()
        src = order[frozenset(state)]
        for symbol in symbols:
            target = goto(state, symbol, augmented, first)
            if not target:
                continue
            dst = register(target)
            edges[(src, symbol)] = dst

    action: Dict[int, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    goto_table: Dict[int, Dict[str, int]] = defaultdict(dict)

    for idx, state in enumerate(states):
        for symbol in grammar.nonterminals:
            dst = edges.get((idx, symbol))
            if dst is not None:
                goto_table[idx][symbol] = dst
        for item in state:
            if not item.complete():
                symbol = item.next_symbol()
                if symbol in grammar.terminals | {END}:
                    dst = edges.get((idx, symbol))
                    if dst is not None:
                        action[idx][symbol].append(f"shift I{dst}")
            else:
                if item.head == start_prime and item.lookahead == END:
                    action[idx][END].append("accept")
                else:
                    rhs = " ".join(item.body) or "<eps>"
                    action[idx][item.lookahead].append(f"reduce {item.head} -> {rhs}")

    states_view = [
        (f"I{idx}", sorted(_format_item(item) for item in state))
        for idx, state in enumerate(states)
    ]
    return states_view, action, goto_table


def _format_item(item: LR1Item) -> str:
    left = " ".join(item.body[: item.dot])
    right = " ".join(item.body[item.dot :])
    if left and right:
        prod = f"{left} . {right}"
    elif left:
        prod = f"{left} ."
    elif right:
        prod = f". {right}"
    else:
        prod = "."
    return f"{item.head} -> {prod}, [{item.lookahead}]"


def precedence_relations(grammar: Grammar) -> Dict[str, Dict[str, RelationCell]]:
    if grammar.start != "E":
        raise ValueError("Modul de precedenta simpla este definit doar pentru gramatica cu simbol initial E.")
    if not EXPECTED_NONTERMINALS.issubset(grammar.nonterminals):
        raise ValueError(
            "Modul de precedenta simpla necesita neterminalele E, T, F exact ca in materialul din Tema2."
        )
    if not EXPECTED_TERMINALS.issubset(grammar.terminals):
        raise ValueError(
            "Modul de precedenta simpla necesita terminalele a, +, *, (, )."
        )
    return {row: dict(cols) for row, cols in PRECEDENCE_TABLE.items()}


def render_lr1(states: List[Tuple[str, List[str]]], action: Dict[int, Dict[str, List[str]]], goto_table: Dict[int, Dict[str, int]], grammar: Grammar) -> None:
    print("=== APD LR(1) ===")
    for name, items in states:
        idx = int(name[1:])
        print(name)
        print("\n".join(f"  {entry}" for entry in items))
        for symbol, ops in sorted(action[idx].items()):
            print(f"  act {symbol}: {', '.join(sorted(set(ops)))}")
        for symbol, dst in sorted(goto_table[idx].items()):
            print(f"  goto {symbol} -> I{dst}")
        print()


def render_precedence(table: Dict[str, Dict[str, RelationCell]]) -> None:
    print("=== APD precedenta simpla ===")
    header = PRECEDENCE_ORDER
    print("       " + " ".join(f"{symbol:>7}" for symbol in header))
    for row_symbol in header:
        entries = [f"{row_symbol:>7}"]
        for col_symbol in header:
            cell = table.get(row_symbol, {}).get(col_symbol)
            if isinstance(cell, tuple):
                text = " ".join(f"{rel}." for rel in cell)
            elif isinstance(cell, str):
                text = cell
            else:
                text = "."
            entries.append(f"{text:>7}")
        print(" ".join(entries))
    print()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Construieste APD pentru gramatici LR(1) sau de precedenta simpla.")
    parser.add_argument("-g", "--grammar", required=True, help="Fisierul cu gramatica.")
    parser.add_argument("-m", "--mode", choices=["lr1", "precedence"], default="lr1")
    args = parser.parse_args(argv)

    grammar = parse_grammar(args.grammar)
    if args.mode == "lr1":
        states, action, goto_table = lr1_automaton(grammar)
        render_lr1(states, action, goto_table, grammar)
    else:
        table = precedence_relations(grammar)
        render_precedence(table)


if __name__ == "__main__":
    main()
