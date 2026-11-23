"""Directive 1: First Divergent Step (L_k) & 50k Classmate Consensus Invariant.

Pure-stdlib implementation (compatible with `/usr/bin/python3` and `qaggle.perf`):
  1. Unsupervised modal canonical intermediate state graph (G_q) extraction from
     unlabelled OCR traces without ground-truth labels (ARM-04 compliant).
  2. Vectorized Schwartz-Zippel polynomial identity testing over an 8-point
     transcendental probe grid.
  3. Rigorous step-diff comparator isolating the 1-based index k of the First
     Divergent Step L_k while filtering self-corrected transient slips.
  4. Symbolic error predicate E(L_{k-1}, L_k) mapped deterministically to the
     75-code taxonomy (`taxonomy_v1.csv`).
"""

from __future__ import annotations

import ast
import csv
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from qaggle.perf import configure_high_performance_env

ProbeVec = tuple[float, ...]

NUM_PROBES: int = 8
_PROBE_VARS: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "a",
    "b",
    "c",
    "n",
    "m",
    "p",
    "q",
    "r",
    "t",
    "k",
)
_PROBE_GRID: dict[str, ProbeVec] = {
    v: tuple(
        math.pi ** ((j + 1) / 3.0) + math.e ** ((d + 1) / 5.0)
        for d in range(NUM_PROBES)
    )
    for j, v in enumerate(_PROBE_VARS)
}
_ZERO_VEC: ProbeVec = (0.0,) * NUM_PROBES

_FNV_OFFSET_BASIS: int = 14695981039346656037
_FNV_PRIME: int = 1099511628211

_STRIKETHROUGH_RE = re.compile(
    r"(\[strikethrough\].*?\[/strikethrough\]|\[crossed out[^\]]*\]|\[scribble[^\]]*\]|"
    r"\\cancel\{[^}]*\}|\\sout\{[^}]*\}|~+[^~]+~+)",
    flags=re.IGNORECASE,
)
_BRACKET_NOTE_RE = re.compile(r"\[[^\]]*\]")
_LINE_PREFIX_RE = re.compile(
    r"^\s*(?:L\d+\s*\|\s*|step\s*\d+\s*[:.)-]?\s*|"
    r"(?:jadi|maka|misal(?:kan)?|diketahui|ditanya|jawab(?:an)?|hasil(?:nya)?)\s*[:=]?\s*)+",
    flags=re.IGNORECASE,
)
_COMMA_DECIMAL_RE = re.compile(r"(?<=\d),(?=\d)")
_MIXED_FRAC_RE = re.compile(r"(\d+)\s+(\d+)\s*/\s*(\d+)")
_LATEX_FRAC_RE = re.compile(r"\\(?:d|t)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_UNICODE_SQRT_RE = re.compile(r"√\s*\(([^()]+)\)|√\s*([0-9a-zA-Z]+)")
_IMPLICIT_MUL_NUM_VAR_RE = re.compile(r"(\d)([a-zA-Z(])")
_IMPLICIT_MUL_PAR_RE = re.compile(r"(\))([a-zA-Z0-9(])")
_RELATIONAL_SPLIT_RE = re.compile(r"(<=|>=|!=|\\le|\\ge|\\neq|=|<|>)")


def normalize_math_ocr_line(raw_line: str) -> str:
    """Project a noisy bilingual K-12 OCR line into normalized ASCII math syntax."""
    if not raw_line:
        return ""
    s = _STRIKETHROUGH_RE.sub("", str(raw_line))
    s = _BRACKET_NOTE_RE.sub("", s)
    s = _LINE_PREFIX_RE.sub("", s.strip())
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    s = (
        s.replace("²", "^2")
        .replace("³", "^3")
        .replace("⁴", "^4")
        .replace("⁵", "^5")
        .replace("⁶", "^6")
    )
    for _ in range(4):
        s_next = _LATEX_FRAC_RE.sub(r"((\1)/(\2))", s)
        s_next = _LATEX_SQRT_RE.sub(r"sqrt(\1)", s_next)
        if s_next == s:
            break
        s = s_next
    s = _UNICODE_SQRT_RE.sub(
        lambda m: f"sqrt({m.group(1) or m.group(2)})", s
    )
    s = _MIXED_FRAC_RE.sub(r"(\1+(\2)/(\3))", s)
    s = _COMMA_DECIMAL_RE.sub(".", s)
    s = (
        s.replace("\\times", "*")
        .replace("\\cdot", "*")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("\\div", "/")
        .replace(":", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("^", "**")
    )
    s = _IMPLICIT_MUL_NUM_VAR_RE.sub(r"\1*\2", s)
    s = _IMPLICIT_MUL_PAR_RE.sub(r"\1*\2", s)
    s = re.sub(r"(?<![a-zA-Z])([a-zA-Z])(\()", r"\1*\2", s)
    return re.sub(r"\s+", "", s).lower()


def _eval_ast_node_vectorized(
    node: ast.AST, env: dict[str, ProbeVec]
) -> ProbeVec:
    """Evaluate a Python math AST node over the 8-point probe vector."""
    if isinstance(node, ast.Expression):
        return _eval_ast_node_vectorized(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        val = float(node.value)
        return (val,) * NUM_PROBES
    if isinstance(node, ast.Name):
        v = node.id.lower()
        if v == "pi":
            return (math.pi,) * NUM_PROBES
        if v == "e":
            return (math.e,) * NUM_PROBES
        if v in env:
            return env[v]
        if v in _PROBE_GRID:
            return _PROBE_GRID[v]
        acc = [1.0] * NUM_PROBES
        for ch in v:
            row = _PROBE_GRID.get(ch, _PROBE_GRID["x"])
            for i in range(NUM_PROBES):
                acc[i] *= row[i]
        return tuple(acc)
    if isinstance(node, ast.UnaryOp):
        val_vec = _eval_ast_node_vectorized(node.operand, env)
        if isinstance(node.op, ast.USub):
            return tuple(-x for x in val_vec)
        if isinstance(node.op, ast.UAdd):
            return val_vec
    if isinstance(node, ast.BinOp):
        left = _eval_ast_node_vectorized(node.left, env)
        right = _eval_ast_node_vectorized(node.right, env)
        if isinstance(node.op, ast.Add):
            return tuple(a + b for a, b in zip(left, right))
        if isinstance(node.op, ast.Sub):
            return tuple(a - b for a, b in zip(left, right))
        if isinstance(node.op, ast.Mult):
            return tuple(a * b for a, b in zip(left, right))
        if isinstance(node.op, ast.Div):
            out = []
            for a, b in zip(left, right):
                if abs(b) <= 1e-15:
                    raise ZeroDivisionError("Zero probe denominator")
                out.append(a / b)
            return tuple(out)
        if isinstance(node.op, ast.Pow):
            out = []
            for a, b in zip(left, right):
                exp = max(-12.0, min(12.0, b))
                is_even = abs(exp - round(exp)) < 1e-9 and (round(exp) % 2 == 0)
                sign = 1.0 if is_even else (-1.0 if a < 0 else 1.0)
                out.append(((abs(a) + 1e-18) ** exp) * sign)
            return tuple(out)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn_name = node.func.id.lower()
        if node.args:
            arg0 = _eval_ast_node_vectorized(node.args[0], env)
            if fn_name == "sqrt":
                return tuple(math.sqrt(abs(x)) for x in arg0)
            if fn_name == "abs":
                return tuple(abs(x) for x in arg0)
    raise ValueError(f"Unsupported AST construct: {type(node).__name__}")


@dataclass(frozen=True)
class CanonicalStepState:
    """Canonical algebraic representation of a single student line L_t."""

    raw_text: str
    norm_text: str
    is_relational: bool
    rel_op: str
    lhs_text: str
    rhs_text: str
    probe_vec: ProbeVec
    state_hash: int
    ast_valid: bool


def _vec_std(v: ProbeVec) -> float:
    if not v:
        return 0.0
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def canonicalize_step(
    raw_line: str,
    env: dict[str, ProbeVec] | None = None,
) -> CanonicalStepState:
    """Map raw student line L_t to its Schwartz-Zippel probe vector and 64-bit state ID."""
    norm = normalize_math_ocr_line(raw_line)
    if not norm:
        return CanonicalStepState(
            raw_line, "", False, "", "", "", _ZERO_VEC, 0, False
        )

    env_map = env or {}
    parts = _RELATIONAL_SPLIT_RE.split(norm)
    is_rel = len(parts) >= 3
    rel_op = parts[-2] if is_rel else ""
    lhs_str = parts[-3] if is_rel else norm
    rhs_str = parts[-1] if is_rel else ""

    try:
        if is_rel and lhs_str and rhs_str:
            lhs_val = _eval_ast_node_vectorized(
                ast.parse(lhs_str, mode="eval"), env_map
            )
            rhs_val = _eval_ast_node_vectorized(
                ast.parse(rhs_str, mode="eval"), env_map
            )
            diff = [a - b for a, b in zip(lhs_val, rhs_val)]
            norm_inf = max(abs(x) for x in diff)
            if math.isfinite(norm_inf) and norm_inf > 1e-11:
                scaled = [x / norm_inf for x in diff]
                if rel_op == "=":
                    for x in scaled:
                        if abs(x) > 1e-7:
                            if x < 0:
                                scaled = [-v for v in scaled]
                            break
                vec = tuple(scaled)
            else:
                vec = _ZERO_VEC
        else:
            expr_str = rhs_str if (is_rel and not lhs_str) else lhs_str
            vec = _eval_ast_node_vectorized(
                ast.parse(expr_str, mode="eval"), env_map
            )
        if not all(math.isfinite(x) for x in vec):
            raise ValueError("Non-finite probe evaluation")
        ast_ok = True
    except Exception:  # noqa: BLE001
        fallback = [0.0] * NUM_PROBES
        for idx, ch in enumerate(norm[:32]):
            fallback[idx % NUM_PROBES] += (ord(ch) * (idx + 1)) * 1e-3
        vec = tuple(fallback)
        ast_ok = False

    h = _FNV_OFFSET_BASIS
    tag = 1 if is_rel else 0
    h = ((h ^ tag) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    for val in vec:
        qv = round(val * 1e6)
        h = ((h ^ (qv & 0xFFFFFFFFFFFFFFFF)) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF

    return CanonicalStepState(
        raw_text=raw_line,
        norm_text=norm,
        is_relational=is_rel,
        rel_op=rel_op,
        lhs_text=lhs_str,
        rhs_text=rhs_str,
        probe_vec=vec,
        state_hash=h,
        ast_valid=ast_ok,
    )


def steps_algebraically_equivalent(
    s1: CanonicalStepState,
    s2: CanonicalStepState,
    tol: float = 1e-5,
) -> bool:
    """Check if two canonical steps represent the same mathematical state."""
    if s1.state_hash != 0 and s1.state_hash == s2.state_hash:
        return True
    if not (s1.ast_valid and s2.ast_valid):
        return s1.norm_text != "" and s1.norm_text == s2.norm_text
    if (
        s1.is_relational
        and s2.is_relational
        and s1.rel_op == "="
        and s2.rel_op == "="
    ):
        n1 = math.sqrt(sum(x * x for x in s1.probe_vec))
        n2 = math.sqrt(sum(x * x for x in s2.probe_vec))
        if n1 < tol and n2 < tol:
            return True
        if n1 < tol or n2 < tol:
            return False
        dot = sum(a * b for a, b in zip(s1.probe_vec, s2.probe_vec))
        cos_sim = abs(dot / (n1 * n2))
        return cos_sim >= 1.0 - tol
    return max(abs(a - b) for a, b in zip(s1.probe_vec, s2.probe_vec)) <= tol


@dataclass
class ConsensusGraph:
    """Modal Canonical Correct Intermediate State Graph G_q for a question_id."""

    question_id: str
    answer_state: CanonicalStepState
    answer_scalar: float | None
    silver_trace_count: int
    node_support: dict[int, float]
    node_exemplars: dict[int, CanonicalStepState]
    edge_weights: dict[tuple[int, int], float]
    modal_paths: list[list[int]]
    node_probe_matrix: list[ProbeVec] = field(default_factory=list)
    node_hash_array: list[int] = field(default_factory=list)
    node_support_array: list[float] = field(default_factory=list)


def _extract_numeric_answer(answer_str: str) -> float | None:
    st = canonicalize_step(answer_str)
    if not st.ast_valid:
        return None
    if not st.is_relational and _vec_std(st.probe_vec) < 1e-7:
        return float(st.probe_vec[0])
    if st.is_relational and st.rhs_text:
        rhs_st = canonicalize_step(st.rhs_text)
        if rhs_st.ast_valid and _vec_std(rhs_st.probe_vec) < 1e-7:
            return float(rhs_st.probe_vec[0])
    return None


def build_question_consensus_graph(
    question_id: str,
    correct_answer: str,
    unlabelled_ocr_traces: Sequence[Sequence[str]],
    *,
    max_skip: int = 3,
    skip_gamma: float = 0.75,
    min_node_support: float = 0.04,
    min_edge_weight: float = 0.025,
) -> ConsensusGraph:
    """Extract modal canonical intermediate state graph G_q from unlabelled OCR traces."""
    ans_state = canonicalize_step(correct_answer)
    ans_scalar = _extract_numeric_answer(correct_answer)

    canonical_traces: list[list[CanonicalStepState]] = []
    silver_traces: list[list[CanonicalStepState]] = []

    for raw_lines in unlabelled_ocr_traces:
        if not raw_lines:
            continue
        dedup: list[CanonicalStepState] = []
        for ln in raw_lines:
            st = canonicalize_step(ln)
            if not st.norm_text:
                continue
            if not dedup or st.state_hash != dedup[-1].state_hash:
                dedup.append(st)
        if not dedup:
            continue
        canonical_traces.append(dedup)

        last_st = dedup[-1]
        matches_key = steps_algebraically_equivalent(last_st, ans_state)
        if not matches_key and ans_scalar is not None and last_st.ast_valid:
            if last_st.is_relational and last_st.rhs_text:
                rhs_st = canonicalize_step(last_st.rhs_text)
                if (
                    rhs_st.ast_valid
                    and abs(float(rhs_st.probe_vec[0]) - ans_scalar) <= 1e-5
                ):
                    matches_key = True
            elif (
                not last_st.is_relational
                and abs(float(last_st.probe_vec[0]) - ans_scalar) <= 1e-5
            ):
                matches_key = True
        if matches_key and 1 <= len(dedup) <= 16:
            silver_traces.append(dedup)

    active_cohort = silver_traces if len(silver_traces) >= 2 else canonical_traces
    n_cohort = max(len(active_cohort), 1)
    tau_node = (
        max(min_node_support, 2.0 / n_cohort) if len(active_cohort) >= 10 else 0.0
    )

    node_counts: dict[int, int] = {}
    node_exemplars: dict[int, CanonicalStepState] = {}
    edge_accum: dict[tuple[int, int], float] = {}

    for trace in active_cohort:
        seen_in_trace: set[int] = set()
        for st in trace:
            h = st.state_hash
            if h not in seen_in_trace:
                node_counts[h] = node_counts.get(h, 0) + 1
                seen_in_trace.add(h)
            if h not in node_exemplars:
                node_exemplars[h] = st

        m = len(trace)
        for a in range(m):
            u = trace[a].state_hash
            for b in range(a + 1, min(m, a + 1 + max_skip)):
                v = trace[b].state_hash
                if u != v:
                    pair = (u, v)
                    edge_accum[pair] = edge_accum.get(pair, 0.0) + (
                        skip_gamma ** (b - a - 1)
                    )

    node_support: dict[int, float] = {
        h: cnt / n_cohort
        for h, cnt in node_counts.items()
        if (cnt / n_cohort) >= tau_node
    }
    if ans_state.state_hash != 0:
        node_support[ans_state.state_hash] = max(
            node_support.get(ans_state.state_hash, 0.0), 1.0
        )
        node_exemplars[ans_state.state_hash] = ans_state

    edge_weights: dict[tuple[int, int], float] = {
        (u, v): w / n_cohort
        for (u, v), w in edge_accum.items()
        if u in node_support
        and v in node_support
        and (w / n_cohort) >= min_edge_weight
    }

    modal_paths: list[list[int]] = []
    if node_support:
        ordered_nodes = sorted(
            node_support.keys(), key=lambda h: node_support[h], reverse=True
        )
        modal_paths.append(ordered_nodes[:8])

    valid_hashes = [h for h in node_support if node_exemplars[h].ast_valid]
    probe_mat = [node_exemplars[h].probe_vec for h in valid_hashes]
    sup_arr = [node_support[h] for h in valid_hashes]

    return ConsensusGraph(
        question_id=question_id,
        answer_state=ans_state,
        answer_scalar=ans_scalar,
        silver_trace_count=len(silver_traces),
        node_support=node_support,
        node_exemplars=node_exemplars,
        edge_weights=edge_weights,
        modal_paths=modal_paths,
        node_probe_matrix=probe_mat,
        node_hash_array=valid_hashes,
        node_support_array=sup_arr,
    )


@dataclass
class DivergenceDiagnosis:
    """Output of the step-diff comparator and symbolic transition analyzer."""

    divergent_index_1based: int | None
    l_prev: CanonicalStepState | None
    l_curr: CanonicalStepState | None
    step_validity: list[bool]
    step_consensus_scores: list[float]
    stopped_early_ans01: bool
    matched_codes: list[tuple[str, float, str]]


def locate_first_divergent_step(
    student_lines: Sequence[str],
    consensus_graph: ConsensusGraph,
    question_anchor_expr: str = "",
) -> tuple[int | None, list[CanonicalStepState], list[bool], list[float], bool]:
    """Identify the 1-based index k of the First Divergent Step L_k."""
    _ = question_anchor_expr
    steps = [
        canonicalize_step(ln) for ln in student_lines if normalize_math_ocr_line(ln)
    ]
    m = len(steps)
    if m == 0:
        return None, [], [], [], False

    consensus_scores: list[float] = [0.0] * m
    if consensus_graph.node_probe_matrix:
        for t, st in enumerate(steps):
            best_sim = 0.0
            for node_vec, sup in zip(
                consensus_graph.node_probe_matrix,
                consensus_graph.node_support_array,
            ):
                diff_inf = max(
                    abs(a - b) for a, b in zip(st.probe_vec, node_vec)
                )
                try:
                    exponent = -0.5 * (diff_inf / 1e-4) ** 2
                    if not math.isfinite(exponent):
                        raise OverflowError
                    sim = math.exp(exponent) * (sup ** 0.25)
                except OverflowError:
                    sim = 0.0  # unreachably far node: zero similarity
                best_sim = max(best_sim, sim)
            consensus_scores[t] = best_sim

    ans_scalar = consensus_graph.answer_scalar
    ans_env = (
        {"x": (ans_scalar,) * NUM_PROBES} if ans_scalar is not None else None
    )

    # Only use root-preservation fallback when consensus graph has no intermediate equation nodes
    has_consensus_equations = sum(
        1
        for h in consensus_graph.node_hash_array
        if consensus_graph.node_exemplars[h].is_relational
    ) >= 2

    valid_mask: list[bool] = [False] * m
    for t, st in enumerate(steps):
        if (
            st.state_hash in consensus_graph.node_support
            or consensus_scores[t] >= 0.35
        ):
            valid_mask[t] = True
            continue

        if (
            t > 0
            and valid_mask[t - 1]
            and steps_algebraically_equivalent(steps[t - 1], st)
        ):
            valid_mask[t] = True
            continue

        if (
            not has_consensus_equations
            and ans_env is not None
            and st.is_relational
            and st.lhs_text
            and st.rhs_text
        ):
            try:
                lhs_v = _eval_ast_node_vectorized(
                    ast.parse(st.lhs_text, mode="eval"), ans_env
                )
                rhs_v = _eval_ast_node_vectorized(
                    ast.parse(st.rhs_text, mode="eval"), ans_env
                )
                if max(abs(a - b) for a, b in zip(lhs_v, rhs_v)) <= 1e-5:
                    valid_mask[t] = True
                    continue
            except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
                _ = None

        if steps_algebraically_equivalent(st, consensus_graph.answer_state):
            valid_mask[t] = True

    divergent_k: int | None = None
    for t in range(m):
        if not valid_mask[t]:
            recovers_later = any(valid_mask[j] for j in range(t + 1, m))
            if not recovers_later:
                divergent_k = t + 1
                break

    stopped_early = False
    if divergent_k is None and m > 0:
        last_st = steps[-1]
        matches_final = steps_algebraically_equivalent(
            last_st, consensus_graph.answer_state
        )
        if not matches_final and ans_scalar is not None and last_st.ast_valid:
            if last_st.is_relational and last_st.rhs_text:
                rhs_st = canonicalize_step(last_st.rhs_text)
                matches_final = (
                    rhs_st.ast_valid
                    and abs(float(rhs_st.probe_vec[0]) - ans_scalar) <= 1e-5
                )
            else:
                matches_final = abs(float(last_st.probe_vec[0]) - ans_scalar) <= 1e-5
        if not matches_final:
            stopped_early = True

    return divergent_k, steps, valid_mask, consensus_scores, stopped_early


_PARTIAL_DIST_RE = re.compile(r"([0-9a-z]+)\*\(([0-9a-z*]+)\+([0-9a-z*]+)\)")
_NEG_DIST_RE = re.compile(r"-\(([0-9a-z*]+)\+([0-9a-z*]+)\)")
_FRESHMAN_DREAM_RE = re.compile(r"\(([0-9a-z*]+)\+([0-9a-z*]+)\)\*\*(\d+)")
_EXP_BASE_MUL_RE = re.compile(r"(\d+)\*\*(\d+)")
_EXP_MUL_RE = re.compile(r"([0-9a-z]+)\*\*(\d+)\*\1\*\*(\d+)")
_EXP_POW_RE = re.compile(r"\(([0-9a-z]+)\*\*(\d+)\)\*\*(\d+)")
_FRAC_ADD_RE = re.compile(
    r"\(\((\d+)\)/\((\d+)\)\)\+\(\((\d+)\)/\((\d+)\)\)|(\d+)/(\d+)\+(\d+)/(\d+)"
)
_FRAC_DIV_RE = re.compile(
    r"\(\((\d+)\)/\((\d+)\)\)/\(\((\d+)\)/\((\d+)\)\)|(\d+)/(\d+)/\((\d+)/(\d+)\)"
)
_SQRT_ADD_RE = re.compile(r"sqrt\(([^()]+)\+([^()]+)\)")
_OOO_LEFT_TO_RIGHT_RE = re.compile(r"^(\d+)\+(\d+)\*(\d+)$")


def evaluate_symbolic_error_predicate(
    l_prev: CanonicalStepState | None,
    l_curr: CanonicalStepState | None,
    *,
    divergent_k: int | None,
    stopped_early_ans01: bool,
    topic_name: str = "",
    question_text: str = "",
    valid_taxonomy_codes: set[str] | None = None,
) -> list[tuple[str, float, str]]:
    """Evaluate symbolic error predicate E(L_{k-1}, L_k) against taxonomy_v1.csv."""
    if divergent_k is None:
        if stopped_early_ans01:
            return [
                (
                    "ANS-01",
                    5.80,
                    "Valid intermediate derivation stopping short of final answer key",
                ),
                ("NONE", 2.10, "All written steps algebraically valid"),
                ("CAL-01", 0.50, "Fallback minor terminal slip"),
            ]
        return [
            (
                "NONE",
                6.50,
                "All steps verified against classmate consensus graph and answer key",
            ),
            ("ANS-01", 0.40, "Secondary check for unsimplified form"),
            ("CAL-01", 0.20, "Secondary minor arithmetic check"),
        ]

    if l_curr is None:
        return [("NONE", 1.0, "Empty trace fallback")]

    matches: list[tuple[str, float, str]] = []
    u_norm = l_prev.norm_text if l_prev is not None else ""

    if divergent_k == 1 and (l_prev is None or not u_norm):
        q_nums = [
            float(x)
            for x in re.findall(r"\d+(?:\.\d+)?", normalize_math_ocr_line(question_text))
        ]
        if len(q_nums) >= 2 and l_curr.ast_valid:
            a, b = q_nums[0], q_nums[1]
            v_val = float(l_curr.probe_vec[0])
            wrong_ops = [a + b, a - b, b - a, a * b] + (
                [a / b, b / a] if b != 0 and a != 0 else []
            )
            if any(abs(v_val - w) <= 1e-5 for w in wrong_ops):
                matches.append(
                    (
                        "WOP-01",
                        5.40,
                        "Line L_1 applies wrong binary operation to problem constants",
                    )
                )
        matches.append(
            (
                "FRM-01",
                4.60,
                "Initial formula or equation setup at L_1 diverges from consensus",
            )
        )

    if l_prev is not None and u_norm:
        if _PARTIAL_DIST_RE.search(u_norm):
            buggy = _PARTIAL_DIST_RE.sub(r"(\1*\2+\3)", u_norm)
            if steps_algebraically_equivalent(canonicalize_step(buggy), l_curr):
                matches.append(
                    (
                        "ALG-04",
                        6.80,
                        "Incomplete distribution: a(b + c) rewritten as ab + c",
                    )
                )
        if _NEG_DIST_RE.search(u_norm):
            buggy = _NEG_DIST_RE.sub(r"(-\1+\2)", u_norm)
            if steps_algebraically_equivalent(canonicalize_step(buggy), l_curr):
                matches.append(
                    (
                        "ALG-04",
                        6.70,
                        "Incomplete negative distribution across parentheses: -(b+c) -> -b+c",
                    )
                )

        if (
            l_prev.is_relational
            and l_curr.is_relational
            and l_prev.rel_op == "="
            and l_curr.rel_op == "="
        ):
            try:
                u_diff = _eval_ast_node_vectorized(
                    ast.parse(f"({l_prev.lhs_text})-({l_prev.rhs_text})", mode="eval"),
                    {},
                )
                v_diff = _eval_ast_node_vectorized(
                    ast.parse(f"({l_curr.lhs_text})-({l_curr.rhs_text})", mode="eval"),
                    {},
                )
                shift = tuple(b - a for a, b in zip(u_diff, v_diff))
                if _vec_std(shift) < 1e-6 and abs(shift[0]) > 1e-4:
                    u_consts = [
                        float(c) for c in re.findall(r"\d+(?:\.\d+)?", u_norm)
                    ]
                    if any(
                        abs(abs(shift[0]) - 2.0 * c) <= 1e-4
                        for c in u_consts
                        if c > 0
                    ):
                        matches.append(
                            (
                                "ALG-06",
                                6.60,
                                "Transposition across '=' without inverting sign (+/-2c shift)",
                            )
                        )
            except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError):
                _ = None

        m_coef = re.match(
            r"^([+-]?\d+(?:\.\d+)?)\*([a-z])=([+-]?\d+(?:\.\d+)?)$", u_norm
        )
        if m_coef:
            a_val, var_name, b_val = (
                float(m_coef.group(1)),
                m_coef.group(2),
                float(m_coef.group(3)),
            )
            for cand in (f"{var_name}={b_val - a_val}", f"{var_name}={b_val + a_val}"):
                if steps_algebraically_equivalent(canonicalize_step(cand), l_curr):
                    matches.append(
                        (
                            "ALG-06",
                            6.65,
                            "Factor moved across '=' as subtraction/addition instead of division",
                        )
                    )

        if _EXP_BASE_MUL_RE.search(u_norm):
            buggy = _EXP_BASE_MUL_RE.sub(
                lambda m: str(int(m.group(1)) * int(m.group(2))), u_norm
            )
            if steps_algebraically_equivalent(canonicalize_step(buggy), l_curr):
                matches.append(
                    ("EXP-01", 6.85, "Power evaluated as base * exponent (a^b -> a*b)")
                )

        if _EXP_MUL_RE.search(u_norm):
            buggy = _EXP_MUL_RE.sub(
                lambda m: f"{m.group(1)}**({int(m.group(2))*int(m.group(3))})", u_norm
            )
            if steps_algebraically_equivalent(canonicalize_step(buggy), l_curr):
                matches.append(
                    (
                        "EXP-02",
                        6.85,
                        "Exponents multiplied instead of added in product of powers",
                    )
                )
        if _EXP_POW_RE.search(u_norm):
            buggy = _EXP_POW_RE.sub(
                lambda m: f"{m.group(1)}**({int(m.group(2))+int(m.group(3))})", u_norm
            )
            if steps_algebraically_equivalent(canonicalize_step(buggy), l_curr):
                matches.append(
                    (
                        "EXP-02",
                        6.85,
                        "Power of a power computed by adding exponents: (a^m)^n -> a^(m+n)",
                    )
                )

        if _FRESHMAN_DREAM_RE.search(u_norm):
            buggy = _FRESHMAN_DREAM_RE.sub(r"((\1)**\3+(\2)**\3)", u_norm)
            if steps_algebraically_equivalent(canonicalize_step(buggy), l_curr):
                matches.append(
                    ("EXP-03", 6.90, "Square of a sum expanded as a^2 + b^2")
                )

        if _SQRT_ADD_RE.search(u_norm):
            buggy = _SQRT_ADD_RE.sub(r"(sqrt(\1)+sqrt(\2))", u_norm)
            if steps_algebraically_equivalent(canonicalize_step(buggy), l_curr):
                matches.append(
                    (
                        "EXP-04",
                        6.80,
                        "Radical distributed over addition: sqrt(a+b) -> sqrt(a)+sqrt(b)",
                    )
                )

        m_f = _FRAC_ADD_RE.search(u_norm)
        if m_f:
            g = [x for x in m_f.groups() if x is not None]
            if len(g) == 4:
                a_f, b_f, c_f, d_f = (
                    float(g[0]),
                    float(g[1]),
                    float(g[2]),
                    float(g[3]),
                )
                if b_f != d_f and (b_f + d_f) != 0:
                    buggy_val = (a_f + c_f) / (b_f + d_f)
                    if abs(float(l_curr.probe_vec[0]) - buggy_val) <= 1e-5:
                        matches.append(
                            (
                                "FRA-01",
                                6.90,
                                "Added numerators and denominators directly without common denominator",
                            )
                        )

        m_div = _FRAC_DIV_RE.search(u_norm)
        if m_div:
            g = [x for x in m_div.groups() if x is not None]
            if len(g) == 4:
                a_f, b_f, c_f, d_f = (
                    float(g[0]),
                    float(g[1]),
                    float(g[2]),
                    float(g[3]),
                )
                if (
                    b_f * d_f != 0
                    and abs(float(l_curr.probe_vec[0]) - (a_f * c_f) / (b_f * d_f))
                    <= 1e-5
                ):
                    matches.append(
                        (
                            "FRA-03",
                            6.85,
                            "Fraction division without inverting divisor",
                        )
                    )

        m_ooo = _OOO_LEFT_TO_RIGHT_RE.match(u_norm)
        if m_ooo:
            a_o, b_o, c_o = (
                float(m_ooo.group(1)),
                float(m_ooo.group(2)),
                float(m_ooo.group(3)),
            )
            if abs(float(l_curr.probe_vec[0]) - ((a_o + b_o) * c_o)) <= 1e-5:
                matches.append(
                    (
                        "OOO-01",
                        6.85,
                        "Order of operations ignored: evaluated addition before multiplication",
                    )
                )

    topic_lower = topic_name.lower()
    if not matches:
        if any(
            k in topic_lower
            for k in ("fungsi", "function", "gradien", "garis", "linear")
        ):
            matches.append(
                (
                    "FUN-01",
                    3.60,
                    "Divergent algebraic transition within Functions/Linear topic",
                )
            )
        if any(
            k in topic_lower
            for k in ("perbandingan", "rasio", "peluang", "proport", "probab")
        ):
            matches.append(
                (
                    "PRO-04",
                    3.60,
                    "Divergent transition within Ratio/Proportion/Probability topic",
                )
            )
        matches.append(
            (
                "CAL-01",
                4.20,
                "Unclassified numeric perturbation at L_{k-1} -> L_k (arithmetic/copy slip)",
            )
        )
        matches.append(("ALG-06", 2.80, "Secondary algebraic transposition candidate"))

    if valid_taxonomy_codes:
        allowed = set(valid_taxonomy_codes) | {"NONE"}
        matches = [m for m in matches if m[0] in allowed]

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches


class FirstDivergentStepDetector:
    """Vectorized P4 Classmate Consensus & Symbolic Transition Detector."""

    def __init__(self, taxonomy_csv_path: Path | None = None) -> None:
        configure_high_performance_env()
        self.valid_codes: set[str] = set()
        if taxonomy_csv_path is not None and taxonomy_csv_path.is_file():
            with taxonomy_csv_path.open("r", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    code = (row.get("id") or row.get("code") or "").strip()
                    if code:
                        self.valid_codes.add(code)
        self.graphs: dict[str, ConsensusGraph] = {}

    def fit_unlabelled_corpus(
        self,
        question_metadata: dict[str, dict[str, str]],
        unlabelled_traces_by_qid: dict[str, Sequence[Sequence[str]]],
    ) -> None:
        """Build modal consensus graphs G_q across question_ids in the 50k pool."""
        for qid, traces in unlabelled_traces_by_qid.items():
            meta = question_metadata.get(qid, {})
            ans_key = meta.get("correct_answer", "")
            self.graphs[qid] = build_question_consensus_graph(
                question_id=qid,
                correct_answer=ans_key,
                unlabelled_ocr_traces=traces,
            )

    def diagnose_trace(
        self,
        question_id: str,
        student_ocr_lines: Sequence[str],
        *,
        correct_answer: str = "",
        topic_name: str = "",
        question_text: str = "",
    ) -> DivergenceDiagnosis:
        """Diagnose a single student trace (ARM-04 compliant) and isolate L_k."""
        graph = self.graphs.get(question_id)
        if graph is None:
            graph = build_question_consensus_graph(
                question_id=question_id,
                correct_answer=correct_answer,
                unlabelled_ocr_traces=[],
            )

        k, steps, valid_mask, cons_scores, stopped_early = locate_first_divergent_step(
            student_lines=student_ocr_lines,
            consensus_graph=graph,
            question_anchor_expr=question_text,
        )

        l_prev: CanonicalStepState | None = None
        l_curr: CanonicalStepState | None = None
        if k is not None and 1 <= k <= len(steps):
            l_curr = steps[k - 1]
            l_prev = steps[k - 2] if k >= 2 else None
        elif steps:
            l_curr = steps[-1]
            l_prev = steps[-2] if len(steps) >= 2 else None

        ranked_codes = evaluate_symbolic_error_predicate(
            l_prev=l_prev,
            l_curr=l_curr,
            divergent_k=k,
            stopped_early_ans01=stopped_early,
            topic_name=topic_name,
            question_text=question_text,
            valid_taxonomy_codes=self.valid_codes or None,
        )

        return DivergenceDiagnosis(
            divergent_index_1based=k,
            l_prev=l_prev,
            l_curr=l_curr,
            step_validity=valid_mask,
            step_consensus_scores=cons_scores,
            stopped_early_ans01=stopped_early,
            matched_codes=ranked_codes,
        )
