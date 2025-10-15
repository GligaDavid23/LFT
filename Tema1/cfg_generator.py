
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFG (Type-2) string generator — Tema 1
Author: ChatGPT

Reads a context-free grammar (N, T, S, P) and generates terminal strings
using leftmost or rightmost derivations, bounded by user-provided limits.

Grammar file format (UTF-8):
  - Lines starting with '#' are comments.
  - Optional sets (can be omitted if you just list productions):
        N = S A B
        T = a b c
        S = S
    You may skip these and the script will infer N, T, and S from productions
    (default start symbol is the LHS of the first production).
  - Productions start after a line that begins with 'P:' or directly,
    one per line, in the form:
        A -> a B c | λ
    Symbols are separated by spaces. Use the Greek letter λ for epsilon
    (empty string). You may also use 'epsilon' or 'eps' as aliases.
  - Example:
        # L = { a^n b^k : n>=1, k>=1 }  (Module examples 1.2)
        N = S A B
        T = a b
        S = S
        P:
        S -> a A
        A -> b B | a A
        B -> b | b B | λ

Usage:
    python cfg_generator.py -g grammar_ex_1_2.txt --leftmost --max-len 8 --count 20
    python cfg_generator.py -g grammar_ex_1_2.txt --rightmost --max-len 8 --list-all
    python cfg_generator.py --help

Notes:
  - Generation uses BFS over sentential forms and is guaranteed to find
    all distinct terminal strings up to --max-len, subject to --max-steps
    (default reasonably high) and pruning rules.
  - To avoid explosive growth, forms are pruned if their terminal prefix
    already exceeds --max-len or if their total symbol length exceeds
    --max-form-len (derived from --max-len unless overridden).
"""

import argparse
import sys
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Set, Optional

EPS_SYMBOLS = {"λ", "epsilon", "eps"}

@dataclass(frozen=True)
class Grammar:
    nonterminals: Set[str]
    terminals: Set[str]
    start: str
    productions: Dict[str, List[Tuple[str, ...]]]  # LHS -> list of RHS tuples (each symbol is str)

    def is_nonterminal(self, sym: str) -> bool:
        return sym in self.nonterminals

    def is_terminal(self, sym: str) -> bool:
        return sym in self.terminals

def parse_grammar(path: str) -> Grammar:
    nonterminals: Set[str] = set()
    terminals: Set[str] = set()
    start_symbol: Optional[str] = None
    productions: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)

    in_prod = False
    prod_lines: List[str] = []

    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("p:"):
                in_prod = True
                continue

            if not in_prod and ("->" not in line):
                # Maybe a set declaration
                if line.startswith("N=") or line.startswith("N ="):
                    rhs = line.split("=", 1)[1].strip()
                    nonterminals.update(rhs.split())
                    continue
                if line.startswith("T=") or line.startswith("T ="):
                    rhs = line.split("=", 1)[1].strip()
                    terminals.update(rhs.split())
                    continue
                if line.startswith("S=") or line.startswith("S ="):
                    start_symbol = line.split("=", 1)[1].strip()
                    continue
                # else: ignore stray lines
            # Otherwise treat as production
            in_prod = True
            prod_lines.append(line)

    # If no productions encountered yet, prod_lines may still be empty
    if not prod_lines:
        raise ValueError("No productions found in the grammar file.")

    # Parse productions
    for line in prod_lines:
        if "->" not in line:
            raise ValueError(f"Invalid production line (missing '->'): {line}")
        lhs, rhs_all = [part.strip() for part in line.split("->", 1)]
        if start_symbol is None:
            start_symbol = lhs
        if lhs not in nonterminals:
            nonterminals.add(lhs)
        for alt in rhs_all.split("|"):
            alt = alt.strip()
            if not alt or alt in EPS_SYMBOLS:
                rhs = tuple()
            else:
                rhs = tuple(alt.split())
            # collect terminals / nonterminals tentatively; we'll infer after reading all
            productions[lhs].append(rhs)

    # Infer terminals as "symbols that never appear on LHS and are not epsilon"
    lhs_syms = set(productions.keys())
    rhs_syms = set(sym for rhss in productions.values() for rhs in rhss for sym in rhs)
    inferred_terms = {s for s in rhs_syms if s not in lhs_syms and s not in EPS_SYMBOLS}
    # Merge with any explicit terminals provided
    if terminals:
        terminals |= inferred_terms
    else:
        terminals = inferred_terms

    # Add any explicitly provided nonterminals and check unknowns
    nonterminals |= lhs_syms

    if start_symbol is None:
        raise ValueError("Start symbol not determined. Provide 'S =' or ensure productions exist.")

    # Sanity: ensure all symbols in productions are recognized
    all_known = nonterminals | terminals
    unknown = {s for s in rhs_syms if s not in all_known and s not in EPS_SYMBOLS}
    if unknown:
        raise ValueError(f"Unknown symbols in RHS (neither terminal nor nonterminal): {unknown}")

    return Grammar(nonterminals, terminals, start_symbol, dict(productions))

def is_all_terminals(form: Tuple[str, ...], G: Grammar) -> bool:
    return all(G.is_terminal(s) for s in form)

def terminal_length(form: Tuple[str, ...], G: Grammar) -> int:
    return sum(1 for s in form if G.is_terminal(s))

def realize_lambda(s: str) -> str:
    # Utility for printing lambda nicely
    return "λ" if s == "" else s

def derive(G: Grammar,
           leftmost: bool = True,
           max_len: int = 12,
           max_steps: int = 100000,
           max_form_len: Optional[int] = None,
           list_all: bool = False) -> Iterable[Tuple[str, List[Tuple[str, ...]]]]:
    """
    Enumerate terminal strings up to max_len using BFS over sentential forms.
    Yields pairs (terminal_string, one_derivation_forms_sequence).
    """
    if max_form_len is None:
        # allow a small margin over terminal length to keep a few nonterminals in forms
        max_form_len = max_len + 4

    start = (G.start,)
    queue = deque()
    queue.append((start, [start]))
    seen_forms: Set[Tuple[str, ...]] = set([start])
    produced: Set[str] = set()

    steps = 0
    while queue and steps < max_steps:
        form, hist = queue.popleft()
        steps += 1

        # Prune if terminal prefix already too long
        tlen = terminal_length(form, G)
        if tlen > max_len or len(form) > max_form_len:
            continue

        if is_all_terminals(form, G):
            s = "".join(form)
            # accept empty string if length constraint allows
            if len(s) <= max_len:
                if list_all or s not in produced:
                    produced.add(s)
                    yield (s, hist)
            # even if terminals, continue? No, terminals can't expand further.
            continue

        # choose expansion position
        idx = None
        if leftmost:
            for i, sym in enumerate(form):
                if G.is_nonterminal(sym):
                    idx = i
                    break
        else:
            for i in range(len(form)-1, -1, -1):
                if G.is_nonterminal(form[i]):
                    idx = i
                    break
        assert idx is not None

        nt = form[idx]
        for rhs in G.productions.get(nt, []):
            # Construct new form
            new_form = form[:idx] + rhs + form[idx+1:]
            # prune obvious overshoot: if terminals exceed max_len already, skip
            if terminal_length(new_form, G) > max_len:
                continue
            if len(new_form) > max_form_len:
                continue
            if new_form not in seen_forms:
                seen_forms.add(new_form)
                queue.append((new_form, hist + [new_form]))

def format_derivation(hist: List[Tuple[str, ...]]) -> str:
    return " ⇒ ".join("".join(step) if step else "λ" for step in hist)

def main(argv=None):
    p = argparse.ArgumentParser(description="Generate strings from a Context-Free Grammar (Type-2).")
    p.add_argument("-g", "--grammar", required=True, help="Path to grammar file (UTF-8).")
    side = p.add_mutually_exclusive_group()
    side.add_argument("--leftmost", action="store_true", help="Use leftmost derivations (default).")
    side.add_argument("--rightmost", action="store_true", help="Use rightmost derivations.")
    p.add_argument("--max-len", type=int, default=12, help="Max length of terminal strings to generate.")
    p.add_argument("--count", type=int, default=20, help="Stop after generating this many DISTINCT strings.")
    p.add_argument("--list-all", action="store_true", help="List all (including duplicates if reachable via different derivations).")
    p.add_argument("--max-steps", type=int, default=100000, help="Safety cap on BFS steps.")
    p.add_argument("--max-form-len", type=int, default=None, help="Max sentential-form length (terminals + nonterminals).")
    p.add_argument("--show-deriv", action="store_true", help="Show one derivation sequence per string.")
    args = p.parse_args(argv)

    G = parse_grammar(args.grammar)
    leftmost = True if (not args.rightmost) else False
    leftmost = True if args.leftmost or (not args.rightmost) else leftmost

    gen = derive(G,
                 leftmost=leftmost,
                 max_len=args.max_len,
                 max_steps=args.max_steps,
                 max_form_len=args.max_form_len,
                 list_all=args.list_all)

    total = 0
    for s, hist in gen:
        total += 1
        if args.show_deriv:
            print(f"{realize_lambda(s)}   [{format_derivation(hist)}]")
        else:
            print(realize_lambda(s))
        if not args.list_all and total >= args.count:
            break

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
