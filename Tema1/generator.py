import argparse
import sys
from collections import defaultdict
import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

EPS_SYMBOLS = {"?", "epsilon", "eps", "lambda", "\u03bb"}

DEFAULT_MAX_LEN = 60
DEFAULT_COUNT = 60
DEFAULT_MAX_STEPS = 200_000

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
                raise ValueError(f"Linie de productie invalida: {line}")
            lhs = lhs.strip()
            if not lhs:
                raise ValueError(f"Lipseste partea stanga in productie: {line}")
            if start_symbol is None:
                start_symbol = lhs

            for alt in rhs_block.split("|"):
                alt = alt.strip()
                if not alt or alt in EPS_SYMBOLS:
                    productions[lhs].append(tuple())
                else:
                    productions[lhs].append(tuple(alt.split()))

    if not productions:
        raise ValueError("Gramatica nu contine productii.")
    if start_symbol is None:
        raise ValueError("Simbolul de start nu a fost furnizat si nu a putut fi dedus.")

    nonterminals = set(declared_nonterminals) | set(productions.keys())
    nonterminals.add(start_symbol)

    rhs_symbols = {
        symbol
        for options in productions.values()
        for option in options
        for symbol in option
    }

    terminals = set(declared_terminals)
    for symbol in rhs_symbols:
        if symbol not in nonterminals and symbol not in EPS_SYMBOLS:
            terminals.add(symbol)

    return Grammar(nonterminals, terminals, start_symbol, dict(productions))


def derive_strings(
    grammar: Grammar,
    *,
    leftmost: bool = True,
    max_len: int = DEFAULT_MAX_LEN,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_form_len: Optional[int] = None,
) -> Iterable[str]:
    max_form_len = max_form_len or max_len + 4

    start_form = (grammar.start,)
    heap: List[Tuple[int, int, int, Tuple[str, ...]]] = []
    seen_forms: Set[Tuple[str, ...]] = set()
    emitted: Set[str] = set()
    steps = 0
    counter = 0

    def enqueue(form: Tuple[str, ...]) -> None:
        nonlocal counter
        if form in seen_forms:
            return
        terminal_total = sum(1 for symbol in form if grammar.is_terminal(symbol))
        if terminal_total > max_len or len(form) > max_form_len:
            return
        seen_forms.add(form)
        heapq.heappush(heap, (terminal_total, len(form), counter, form))
        counter += 1

    enqueue(start_form)

    while heap and steps < max_steps:
        terminal_total, _, _, current_form = heapq.heappop(heap)
        steps += 1

        if terminal_total > max_len or len(current_form) > max_form_len:
            continue

        if all(grammar.is_terminal(symbol) for symbol in current_form):
            word = "".join(current_form)
            if len(word) <= max_len and word not in emitted:
                emitted.add(word)
                yield word
            continue

        indices = range(len(current_form)) if leftmost else range(len(current_form) - 1, -1, -1)

        target_index = next((i for i in indices if grammar.is_nonterminal(current_form[i])), None)
        if target_index is None:
            continue

        target_symbol = current_form[target_index]
        for replacement in grammar.productions.get(target_symbol, []):
            next_form = current_form[:target_index] + replacement + current_form[target_index + 1 :]

            enqueue(next_form)


def render_word(word: str) -> str:
    return "epsilon" if word == "" else word


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Genereaza siruri dintr-o gramatica CFG (alege doar leftmost sau rightmost)."
    )
    parser.add_argument("-g", "--grammar", required=True, help="Calea catre fisierul cu gramatica.")
    branch = parser.add_mutually_exclusive_group()
    branch.add_argument("--leftmost", action="store_true", help="Foloseste derivari leftmost (implicit).")
    branch.add_argument("--rightmost", action="store_true", help="Foloseste derivari rightmost.")
    args = parser.parse_args(argv)

    grammar = parse_grammar(args.grammar)
    leftmost = not args.rightmost

    for produced, word in enumerate(derive_strings(grammar, leftmost=leftmost), start=1):
        print(render_word(word))
        if produced >= DEFAULT_COUNT:
            break


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Eroare: {exc}", file=sys.stderr)
        sys.exit(1)
