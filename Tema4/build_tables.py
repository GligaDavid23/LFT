#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

BASE_DIR = Path(__file__).parent
PRODUCTIONS_OUT = BASE_DIR / "lr1_productions.txt"
ACTION_OUT = BASE_DIR / "lr1_actions.txt"
GOTO_OUT = BASE_DIR / "lr1_goto.txt"

TERMINALS = ["id", "+", "-", "*", "/", "(", ")", "$"]
NONTERMINALS = ["E", "T", "F"]
START_SYMBOL = "E"
AUGMENTED_START = "S'"

PRODUCTIONS: Dict[int, Tuple[str, Tuple[str, ...]]] = {
    1: ("E", ("E", "+", "T")),
    11: ("E", ("E", "-", "T")),
    2: ("E", ("T",)),
    3: ("T", ("T", "*", "F")),
    31: ("T", ("T", "/", "F")),
    4: ("T", ("F",)),
    5: ("F", ("(", "E", ")")),
    51: ("F", ("-", "(", "E", ")")),
    6: ("F", ("id",)),
}

PRODUCTION_LIST: List[Tuple[int, str, Tuple[str, ...]]] = [
    (0, AUGMENTED_START, (START_SYMBOL,)),
    *[(idx, lhs, rhs) for idx, (lhs, rhs) in PRODUCTIONS.items()],
]


@dataclass(frozen=True)
class Item:
    lhs: str
    rhs: Tuple[str, ...]
    dot: int
    lookahead: str

    def next_symbol(self) -> str | None:
        return self.rhs[self.dot] if self.dot < len(self.rhs) else None

    def advance(self) -> "Item":
        return Item(self.lhs, self.rhs, self.dot + 1, self.lookahead)


def compute_first_sets() -> Dict[str, Set[str]]:
    first: Dict[str, Set[str]] = {nt: set() for nt in NONTERMINALS}
    changed = True
    while changed:
        changed = False
        for _, lhs, rhs in PRODUCTION_LIST:
            if lhs == AUGMENTED_START:
                continue
            for sym in rhs:
                if sym in TERMINALS:
                    if sym not in first[lhs]:
                        first[lhs].add(sym)
                        changed = True
                    break
                else:
                    before = len(first[lhs])
                    first[lhs].update(first[sym])
                    if len(first[lhs]) != before:
                        changed = True
                    break
    return first


FIRST = compute_first_sets()


def first_of_sequence(seq: Sequence[str], lookahead: str) -> Set[str]:
    if not seq:
        return {lookahead}
    sym = seq[0]
    if sym in TERMINALS:
        return {sym}
    return set(FIRST[sym])


def closure(items: Iterable[Item]) -> Set[Item]:
    closure_set: Set[Item] = set(items)
    added = True
    while added:
        added = False
        new_items: List[Item] = []
        for item in closure_set:
            nxt = item.next_symbol()
            if nxt and nxt in NONTERMINALS:
                beta = item.rhs[item.dot + 1 :]
                lookaheads = first_of_sequence(beta, item.lookahead)
                for prod_idx, lhs, rhs in PRODUCTION_LIST:
                    if lhs != nxt:
                        continue
                    for la in lookaheads:
                        cand = Item(lhs, rhs, 0, la)
                        if cand not in closure_set:
                            new_items.append(cand)
        if new_items:
            closure_set.update(new_items)
            added = True
    return closure_set


def goto(items: Set[Item], symbol: str) -> Set[Item]:
    advanced = [item.advance() for item in items if item.next_symbol() == symbol]
    if not advanced:
        return set()
    return closure(advanced)


def symbol_sort_key(sym: str) -> Tuple[int, str]:
    return (0, sym) if sym in TERMINALS else (1, sym)

def build_states() -> Tuple[List[Set[Item]], Dict[Tuple[int, str], int]]:
    start_item = Item(AUGMENTED_START, (START_SYMBOL,), 0, "$")
    start_state = closure([start_item])
    states: List[Set[Item]] = [start_state]
    index: Dict[frozenset[Item], int] = {frozenset(start_state): 0}
    transitions: Dict[Tuple[int, str], int] = {}
    queue: List[Set[Item]] = [start_state]

    while queue:
        state = queue.pop(0)
        state_idx = index[frozenset(state)]
        symbols = sorted(
            {item.next_symbol() for item in state if item.next_symbol()},
            key=symbol_sort_key,
        )
        for sym in symbols:
            target = goto(state, sym)
            if not target:
                continue
            frozen = frozenset(target)
            if frozen not in index:
                index[frozen] = len(states)
                states.append(target)
                queue.append(target)
            transitions[(state_idx, sym)] = index[frozen]
    return states, transitions


def build_tables(
    states: List[Set[Item]], transitions: Dict[Tuple[int, str], int]
) -> Tuple[Dict[Tuple[int, str], str], Dict[Tuple[int, str], int]]:
    prod_lookup: Dict[Tuple[str, Tuple[str, ...]], int] = {
        (lhs, rhs): idx for idx, lhs, rhs in PRODUCTION_LIST
    }
    action: Dict[Tuple[int, str], str] = {}
    goto_table: Dict[Tuple[int, str], int] = {}

    for (state_idx, sym), target_idx in transitions.items():
        if sym in TERMINALS:
            action[(state_idx, sym)] = f"d{target_idx}"
        else:
            goto_table[(state_idx, sym)] = target_idx

    for idx, state in enumerate(states):
        for item in state:
            if item.next_symbol():
                continue
            if item.lhs == AUGMENTED_START:
                action[(idx, "$")] = "acc"
                continue
            prod_idx = prod_lookup.get((item.lhs, item.rhs))
            if prod_idx is None:
                raise RuntimeError(f"Lipseste productia pentru {item.lhs}->{item.rhs}")
            key = (idx, item.lookahead)
            if key in action:
                raise RuntimeError(f"Conflict in starea {idx} pe {item.lookahead}")
            action[key] = f"r{prod_idx}"

    return action, goto_table


def write_productions(path: Path) -> None:
    lines = [
        f"{idx}. {lhs} -> {' '.join(rhs)}"
        for idx, lhs, rhs in PRODUCTION_LIST
        if idx != 0
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_action_table(states: List[Set[Item]], action: Dict[Tuple[int, str], str]) -> None:
    header = "state " + " ".join(TERMINALS)
    lines = [header]
    for idx in range(len(states)):
        cells = [action.get((idx, t), "-") for t in TERMINALS]
        lines.append(f"{idx:<5} " + " ".join(f"{c:>3}" for c in cells))
    ACTION_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_goto_table(states: List[Set[Item]], goto_table: Dict[Tuple[int, str], int]) -> None:
    header = "state " + " ".join(NONTERMINALS)
    lines = [header]
    for idx in range(len(states)):
        cells = [str(goto_table.get((idx, nt), "-")) for nt in NONTERMINALS]
        lines.append(f"{idx:<5} " + " ".join(f"{c:>3}" for c in cells))
    GOTO_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    states, transitions = build_states()
    action, goto_table = build_tables(states, transitions)
    write_productions(PRODUCTIONS_OUT)
    write_action_table(states, action)
    write_goto_table(states, goto_table)
    print(f"Am generat {len(states)} stari.")
    print(f"Actiuni scrise in {ACTION_OUT.name}, goto in {GOTO_OUT.name}, productii in {PRODUCTIONS_OUT.name}.")


if __name__ == "__main__":
    main()
