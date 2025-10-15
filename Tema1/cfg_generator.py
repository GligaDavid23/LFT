#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple CFG string generator for Tema 1."""

import argparse
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

EPS_SYMBOLS = {"?", "epsilon", "eps", "lambda", "\u03bb"}


@dataclass(frozen=True)
class Grammar:
    nonterminals: Set[str]
    terminals: Set[str]
    start: str
    productions: Dict[str, List[Tuple[str, ...]]]

    def is_nonterminal(self, symbol: str) -> bool:
        return symbol in self.nonterminals

    def is_terminal(self, symbol: str) -> bool:
        return symbol in self.terminals


def parse_grammar(path: str) -> Grammar:
    declared_nonterminals: Set[str] = set()
    declared_terminals: Set[str] = set()
    start_symbol: Optional[str] = None
    productions: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    seen_productions = False

    with open(path, encoding="utf-8") as handler:
        for raw_line in handler:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("p:"):
                seen_productions = True
                continue

            if not seen_productions and "->" not in line:
                key, sep, value = line.partition("=")
                if not sep:
                    continue
                key = key.strip().upper()
                symbols = value.strip().split()
                if key == "N":
                    declared_nonterminals.update(symbols)
                elif key == "T":
                    declared_terminals.update(symbols)
                elif key == "S" and symbols:
                    start_symbol = symbols[0]
                continue

            seen_productions = True
            lhs, sep, rhs_block = line.partition("->")
            if not sep:
                raise ValueError(f"Invalid production line: {line}")
            lhs = lhs.strip()
            if not lhs:
                raise ValueError(f"Missing left-hand side in production: {line}")
            if start_symbol is None:
                start_symbol = lhs

            for alt in rhs_block.split("|"):
                alt = alt.strip()
                if not alt or alt in EPS_SYMBOLS:
                    productions[lhs].append(tuple())
                else:
                    productions[lhs].append(tuple(alt.split()))

    if not productions:
        raise ValueError("Grammar contains no productions.")
    if start_symbol is None:
        raise ValueError("Start symbol not provided and could not be inferred.")

    nonterminals = set(declared_nonterminals) or set(productions.keys())
    nonterminals.update(productions.keys())
    nonterminals.add(start_symbol)

    rhs_symbols = {
        symbol
        for options in productions.values()
        for option in options
        for symbol in option
    }

    terminals = set(declared_terminals)
    inferred_terminals = {
        symbol for symbol in rhs_symbols if symbol not in nonterminals and symbol not in EPS_SYMBOLS
    }
    if terminals:
        terminals.update(inferred_terminals)
    else:
        terminals = inferred_terminals

    unknown = {
        symbol
        for symbol in rhs_symbols
        if symbol not in nonterminals and symbol not in terminals and symbol not in EPS_SYMBOLS
    }
    if unknown:
        raise ValueError(f"Unknown symbols in productions: {unknown}")

    return Grammar(nonterminals, terminals, start_symbol, dict(productions))


def is_terminal_form(form: Tuple[str, ...], grammar: Grammar) -> bool:
    return all(grammar.is_terminal(symbol) for symbol in form)


def terminal_count(form: Tuple[str, ...], grammar: Grammar) -> int:
    return sum(1 for symbol in form if grammar.is_terminal(symbol))


def derive_strings(
    grammar: Grammar,
    *,
    leftmost: bool = True,
    max_len: int = 12,
    max_steps: int = 100_000,
    max_form_len: Optional[int] = None,
    list_all: bool = False,
) -> Iterable[Tuple[str, List[Tuple[str, ...]]]]:
    """Breadth-first enumeration of terminal strings up to `max_len`."""
    if max_form_len is None:
        max_form_len = max_len + 4

    start_form = (grammar.start,)
    queue = deque([(start_form, [start_form])])
    seen_forms: Set[Tuple[str, ...]] = {start_form}
    produced: Set[str] = set()
    steps = 0

    while queue and steps < max_steps:
        current_form, history = queue.popleft()
        steps += 1

        if terminal_count(current_form, grammar) > max_len or len(current_form) > max_form_len:
            continue

        if is_terminal_form(current_form, grammar):
            word = "".join(current_form)
            if len(word) <= max_len and (list_all or word not in produced):
                produced.add(word)
                yield word, history
            continue

        indices = range(len(current_form))
        if not leftmost:
            indices = range(len(current_form) - 1, -1, -1)

        target_index = next((i for i in indices if grammar.is_nonterminal(current_form[i])), None)
        if target_index is None:
            continue

        target_symbol = current_form[target_index]
        for replacement in grammar.productions.get(target_symbol, []):
            next_form = current_form[:target_index] + replacement + current_form[target_index + 1 :]
            if terminal_count(next_form, grammar) > max_len or len(next_form) > max_form_len:
                continue
            if next_form in seen_forms:
                continue
            seen_forms.add(next_form)
            queue.append((next_form, history + [next_form]))


def format_history(history: List[Tuple[str, ...]]) -> str:
    return " => ".join("".join(step) if step else "?" for step in history)


def render_word(word: str) -> str:
    return "?" if word == "" else word


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate strings from a context-free grammar.")
    parser.add_argument("-g", "--grammar", required=True, help="Path to the grammar file.")
    branch = parser.add_mutually_exclusive_group()
    branch.add_argument("--leftmost", action="store_true", help="Use leftmost derivations (default).")
    branch.add_argument("--rightmost", action="store_true", help="Use rightmost derivations.")
    parser.add_argument("--max-len", type=int, default=12, help="Maximum length of generated terminal strings.")
    parser.add_argument("--count", type=int, default=20, help="Number of distinct strings to print before stopping.")
    parser.add_argument("--list-all", action="store_true", help="Do not filter duplicates reachable via other derivations.")
    parser.add_argument("--max-steps", type=int, default=100_000, help="Safety cap on the breadth-first search.")
    parser.add_argument("--max-form-len", type=int, help="Limit on the size of intermediate sentential forms.")
    parser.add_argument("--show-deriv", action="store_true", help="Display one derivation sequence for each string.")
    args = parser.parse_args(argv)

    grammar = parse_grammar(args.grammar)
    leftmost = True
    if args.rightmost:
        leftmost = False

    generator = derive_strings(
        grammar,
        leftmost=leftmost,
        max_len=args.max_len,
        max_steps=args.max_steps,
        max_form_len=args.max_form_len,
        list_all=args.list_all,
    )

    produced = 0
    for word, history in generator:
        produced += 1
        if args.show_deriv:
            print(f"{render_word(word)}   [{format_history(history)}]")
        else:
            print(render_word(word))
        if not args.list_all and produced >= args.count:
            break


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
