#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MSYS2_BASH_CANDIDATES = [Path(r"C:\msys64\usr\bin\bash.exe"), Path(r"C:\msys64\usr\bin\bash")]
TERMINALS = ["id", "+", "-", "*", "(", ")", "$"]
NONTERMINALS = ["E", "T", "F"]
BISON_OUTPUT = SCRIPT_DIR / "expr.output"
GRAMMAR_FILE = SCRIPT_DIR / "expr.y"
ACTIONS_OUT = SCRIPT_DIR / "lr1_actions.txt"
GOTO_OUT = SCRIPT_DIR / "lr1_goto.txt"
PRODUCTIONS_OUT = SCRIPT_DIR / "lr1_productions.txt"
LR1_FILE = SCRIPT_DIR / "LR1.py"


def find_msys2_bash():
    env = os.environ.get("MSYS2_BASH")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    for c in MSYS2_BASH_CANDIDATES:
        if c.exists():
            return c
    return None


def to_msys_path(path: Path) -> str:
    p = path.resolve()
    drive = p.drive.rstrip(":\\/").lower()
    tail = "/".join(p.parts[1:])
    return f"/{drive}/{tail}"


def run_bison():
    if not GRAMMAR_FILE.exists():
        print("Eroare: nu gasesc expr.y langa build_all_lr1.py.")
        sys.exit(1)
    bash = find_msys2_bash()
    if bash:
        cmd = f"cd {to_msys_path(SCRIPT_DIR)} && bison -d -v expr.y"
        print(f">>> Rulez prin MSYS2: {bash} -lc \"{cmd}\"")
        result = subprocess.run([str(bash), "-lc", cmd], check=False, cwd=SCRIPT_DIR)
    else:
        print(">>> MSYS2 nu a fost gasit, rulez direct 'bison -d -v expr.y'")
        try:
            result = subprocess.run(["bison", "-d", "-v", "expr.y"], check=False, cwd=SCRIPT_DIR)
        except FileNotFoundError:
            print("Eroare: nu gasesc executabilul 'bison'.")
            print("Seteaza MSYS2_BASH daca bash.exe este in alta locatie.")
            sys.exit(1)
    if result.returncode != 0:
        print(f"Eroare: bison a esuat cu codul {result.returncode}.")
        sys.exit(1)
    if not BISON_OUTPUT.exists():
        print("Eroare: nu s-a generat expr.output. Verifica expr.y.")
        sys.exit(1)
    print(">>> bison a rulat cu succes, expr.output generat.")


def map_terminal_symbol(sym: str):
    if sym == "ID":
        return "id"
    if sym == "$end":
        return "$"
    if sym.startswith("'") and sym.endswith("'") and len(sym) == 3:
        return sym[1]
    if sym in {"+", "-", "*", "(", ")"}:
        return sym
    return None


def map_grammar_symbol(sym: str) -> str:
    mapped = map_terminal_symbol(sym)
    return mapped if mapped is not None else sym


def parse_productions(path: Path):
    if not path.exists():
        print(f"Eroare: nu gasesc {path} ca sa extrag productiile.")
        sys.exit(1)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_grammar = False
    current_lhs = None
    rules = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not in_grammar:
            if line.strip() == "Grammar":
                in_grammar = True
            continue
        if line.startswith("Terminals"):
            break
        stripped = line.strip()
        if not stripped:
            continue
        m_full = re.match(r"^\s*(\d+)\s+([^\s:]+)\s*:\s*(.*)$", line)
        if m_full:
            rule_no = int(m_full.group(1))
            current_lhs = m_full.group(2)
            rhs_text = m_full.group(3).strip()
            if rule_no != 0:
                rules.append((rule_no, current_lhs, rhs_text))
            continue
        m_alt = re.match(r"^\s*(\d+)\s+\|\s*(.*)$", line)
        if m_alt and current_lhs:
            rule_no = int(m_alt.group(1))
            rhs_text = m_alt.group(2).strip()
            if rule_no != 0:
                rules.append((rule_no, current_lhs, rhs_text))
            continue
    if not rules:
        print("Eroare: nu am gasit nici o productie in sectiunea Grammar din expr.output.")
        sys.exit(1)
    productions = []
    for _, lhs, rhs_text in sorted(rules, key=lambda x: x[0]):
        rhs_tokens = rhs_text.split()
        rhs = tuple(map_grammar_symbol(tok) for tok in rhs_tokens) if rhs_tokens else ("epsilon",)
        productions.append((lhs, rhs))
    return productions


def parse_bison_output(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    state = None
    states = set()
    action_table = {}
    goto_table = {}
    state_re = re.compile(r"^[Ss]tate\s+(\d+)")
    shift_re = re.compile(r"^(\S+)\s+shift, and go to state\s+(\d+)")
    reduce_re = re.compile(r"^(\S+)\s+reduce using rule\s+(\d+)")
    accept_re = re.compile(r"^(\S+)\s+accept")
    goto_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+go to state\s+(\d+)")

    for raw_line in text:
        line = raw_line.strip()
        if not line:
            continue
        m = state_re.match(line)
        if m:
            state = int(m.group(1))
            states.add(state)
            continue
        if state is None:
            continue
        if line.startswith("$default"):
            m_def = re.match(r"^\$default\s+reduce using rule\s+(\d+)", line)
            if m_def:
                prod_no = int(m_def.group(1))
                for t in TERMINALS:
                    action_table.setdefault((state, t), f"r{prod_no}")
                continue
            if "accept" in line:
                action_table.setdefault((state, "$"), "acc")
                continue
        if line.startswith("error"):
            continue
        m = shift_re.match(line)
        if m:
            term = map_terminal_symbol(m.group(1))
            if term in TERMINALS:
                action_table[(state, term)] = f"d{int(m.group(2))}"
            continue
        m = reduce_re.match(line)
        if m:
            term = map_terminal_symbol(m.group(1))
            if term in TERMINALS:
                action_table[(state, term)] = f"r{int(m.group(2))}"
            continue
        m = accept_re.match(line)
        if m:
            term = map_terminal_symbol(m.group(1))
            if term in TERMINALS:
                action_table[(state, term)] = "acc"
            continue
        m = goto_re.match(line)
        if m and m.group(1) in NONTERMINALS:
            goto_table[(state, m.group(1))] = int(m.group(2))
    return states, action_table, goto_table


def write_actions(states, action_table, path: Path):
    lines = ["state " + " ".join(TERMINALS)]
    for s in sorted(states):
        cells = [action_table.get((s, t), "-") for t in TERMINALS]
        lines.append(f"{s:<5} " + " ".join(f"{c:>2}" for c in cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f">>> Scris tabela de actiuni in {path}")


def write_goto(states, goto_table, path: Path):
    lines = ["state " + " ".join(NONTERMINALS)]
    for s in sorted(states):
        cells = [str(goto_table.get((s, nt), "-")) for nt in NONTERMINALS]
        lines.append(f"{s:<5} " + " ".join(f"{c:>2}" for c in cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f">>> Scris tabela de salt in {path}")


def write_productions(productions, path: Path):
    lines = [f"{lhs} -> {' '.join(rhs)}" for lhs, rhs in productions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f">>> Scris productiile in {path}")


def run_lr1():
    if os.environ.get("SKIP_LR1"):
        print("SKIP_LR1 este setat; sar peste rularea LR1.py.")
        return
    if not LR1_FILE.exists():
        print("Atentie: nu gasesc LR1.py langa build_all_lr1.py; sar peste rulare.")
        return
    if not sys.stdin.isatty():
        print("Atentie: stdin nu este interactiv; sar peste rularea LR1.py.")
        return
    subprocess.run([sys.executable, "LR1.py"], cwd=SCRIPT_DIR)


def main():
    run_bison()
    productions = parse_productions(BISON_OUTPUT)
    states, action_table, goto_table = parse_bison_output(BISON_OUTPUT)
    if not states:
        print("Eroare: nu am gasit niciun 'state N' in expr.output.")
        sys.exit(1)
    write_productions(productions, PRODUCTIONS_OUT)
    write_actions(states, action_table, ACTIONS_OUT)
    write_goto(states, goto_table, GOTO_OUT)
    run_lr1()


if __name__ == "__main__":
    main()
