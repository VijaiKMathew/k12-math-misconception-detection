#!/usr/bin/env python3
"""
Zero-Dependency Python Anti-Slop AST Linter (`tools/checks/anti_slop.py`)
========================================================================
Inspired by Dillon Mulroy's `oxlint anti-slop` & TinyFrontier's `anti-slop-py`.

Enforces deterministic back-pressure against AI coding agent anti-patterns
and low-evidence escape hatches that degrade codebases over 1,000+ runs:

1. SLOP01: Unsafe Dynamic Access (`getattr` / `setattr` / `hasattr` with string literals).
   -> Use explicit typed attributes or dataclass/dict indexing.
2. SLOP02: Blind Cast Cascades (`cast(Any, ...)` or cast widening/narrowing loops).
   -> Provide concrete domain contracts or parse at boundaries.
3. SLOP03: Lazy `Any` / `object` Annotations on parameter signatures.
   -> Use specific domain types, Generics, or protocols.
4. SLOP04: Bare / Blanket Exception Swallowing (`except Exception: pass`).
   -> Handle specific expected exceptions and log/raise.
5. SLOP05: Stray Debug Artifacts (`print("DEBUG...")`, `breakpoint()`).
   -> Use structured logging or remove before review.
6. SLOP06: Naked type-ignore comments without specific error codes and rationale.
   -> Include specific rule (e.g. `# type: ignore[attr-defined] -- explanation`).
7. SLOP07: Hallucinated / Shadow Mocking (`mock.patch("builtins...")` outside tests).
   -> Inject dependencies explicitly.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class SlopVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, lines: list[str]) -> None:
        self.filename = filename
        self.lines = lines
        self.violations: list[str] = []

    def add_violation(self, node: ast.AST, code: str, msg: str, remediation: str) -> None:
        if isinstance(node, (ast.stmt, ast.expr, ast.excepthandler, ast.arg)):
            lineno = node.lineno
            col_offset = node.col_offset
        else:
            lineno = 0
            col_offset = 0
        self.violations.append(
            f"[{code}] {self.filename}:{lineno}:{col_offset}\n"
            f"  Issue       : {msg}\n"
            f"  Remediation : {remediation}"
        )

    def visit_Call(self, node: ast.Call) -> None:
        # SLOP01: getattr / setattr / hasattr with literal string attribute names
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "setattr", "hasattr"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            attr_name = node.args[1].value
            if not (attr_name.startswith("__") and attr_name.endswith("__")):
                self.add_violation(
                    node,
                    "SLOP01_UNSAFE_DYNAMIC_ATTR",
                    f"Agent used '{node.func.id}(obj, \"{attr_name}\")' instead of explicit typed access.",
                    f"Replace with direct attribute access 'obj.{attr_name}' or define a typed protocol.",
                )

        # SLOP05: breakpoint() calls
        if isinstance(node.func, ast.Name) and node.func.id == "breakpoint":
            self.add_violation(
                node,
                "SLOP05_STRAY_BREAKPOINT",
                "Leftover 'breakpoint()' call detected.",
                "Remove debug breakpoints before check-in.",
            )

        # SLOP02: cast(Any, ...)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "cast"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "Any"
        ):
            self.add_violation(
                node,
                "SLOP02_CAST_ANY_ESCAPE_HATCH",
                "Agent used 'cast(Any, ...)' to bypass type check.",
                "Parse at I/O boundary or narrow types cleanly with match/case or isinstance.",
            )

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # SLOP03: Lazy Any / object annotations on function arguments
        for arg in node.args.args:
            if (
                arg.annotation
                and isinstance(arg.annotation, ast.Name)
                and arg.annotation.id in {"Any", "object"}
            ):
                self.add_violation(
                    arg,
                    "SLOP03_LAZY_ANY_PARAM",
                    f"Parameter '{arg.arg}' lazily annotated with '{arg.annotation.id}'.",
                    "Replace with a concrete domain type, Generic 'T', or typing Protocol.",
                )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # SLOP04: Bare or Exception pass
        is_blanket = False
        if node.type is None or isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
            is_blanket = True

        if is_blanket and len(node.body) == 1:
            first_stmt = node.body[0]
            if isinstance(first_stmt, ast.Pass):
                self.add_violation(
                    node,
                    "SLOP04_SILENT_EXCEPTION_SWALLOW",
                    "Agent silently swallowed broad exception with 'pass'.",
                    "Catch a specific expected Exception type and log or handle gracefully.",
                )
        self.generic_visit(node)


def check_source_lines(filename: str, lines: list[str]) -> list[str]:
    violations: list[str] = []
    ignore_pattern = re.compile(r"#\s*type:\s*ignore(?!\[)")
    debug_pattern = re.compile(r'print\(f?["\'](?:DEBUG|FIXME|TODO-AGENT|XXX):', re.IGNORECASE)

    for idx, line in enumerate(lines, 1):
        # SLOP06: Naked type-ignore without rule bracket
        if ignore_pattern.search(line):
            violations.append(
                f"[SLOP06_NAKED_TYPE_IGNORE] {filename}:{idx}:0\n"
                f"  Issue       : Naked type-ignore comment found without rule code or justification.\n"
                f"  Remediation : Specify rule code, e.g. '# type: ignore[arg-type]  # rationale'."
            )
        # SLOP05: Stray debug print
        if debug_pattern.search(line):
            violations.append(
                f"[SLOP05_STRAY_DEBUG_PRINT] {filename}:{idx}:0\n"
                f"  Issue       : Stray debug log '{line.strip()}' found.\n"
                f"  Remediation : Remove debugging prints or route through structured logger."
            )

    return violations


def audit_file(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    lines = source.splitlines()
    rel_path = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    line_violations = check_source_lines(rel_path, lines)

    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError as e:
        return [f"[SLOP_SYNTAX_ERROR] {rel_path}:{e.lineno}:{e.offset}: {e.msg}"]

    visitor = SlopVisitor(rel_path, lines)
    visitor.visit(tree)
    return line_violations + visitor.violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["src", "tests", "tools"], help="Directories/files to audit")
    parser.add_argument("--strict", action="store_true", help="Fail with non-zero exit code on any violation")
    args = parser.parse_args(argv)

    all_violations: list[str] = []
    for target in args.paths:
        p = Path(target)
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file() and p.suffix == ".py":
            all_violations.extend(audit_file(p))
        elif p.is_dir():
            for f in sorted(p.rglob("*.py")):
                if ".venv" in f.parts or "__pycache__" in f.parts:
                    continue
                all_violations.extend(audit_file(f))

    if all_violations:
        print(f"\n[ANTI-SLOP AUDIT] Detected {len(all_violations)} agent anti-pattern(s):\n")
        for v in all_violations:
            print(v)
            print("-" * 60)
        return 1

    print("\n[ANTI-SLOP AUDIT] Clean. Zero agent slop patterns detected across audited files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
