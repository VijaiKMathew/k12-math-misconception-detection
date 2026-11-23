"""Directive 2: Fine-Grained Sibling Misconception Disambiguation Matrix & 4-Tag Referee.

Implements:
  1. Top 20 Sibling Misconception Collision Pairs mapped strictly to canonical IDs in
     `taxonomy_v1.csv`, with 2-sentence disambiguation rules and symbolic/lexical triggers.
  2. 4-Tag Structured Reasoning Chain prompt formatter and XML parser:
       <divergence_step>L_k</divergence_step>
       <algebraic_error>delta</algebraic_error>
       <secondary_slips>CAL-01, ...</secondary_slips>
       <root_code>CODE</root_code>
  3. Intra-trace ARM-05 Root-Over-Slip ranker and expected MAP@3 gain calculator.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

LETTERS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
SECONDARY_SLIP_CODES: frozenset[str] = frozenset({"CAL-01", "ANS-01", "NOT-01"})
NEVER_ROOT_MINOR_CODES: frozenset[str] = frozenset({"NOT-01"})


@dataclass(frozen=True)
class SiblingRule:
    """Disambiguation rule and trigger patterns for a confused sibling pair."""

    pair_id: int
    code_a: str
    code_b: str
    rule_text: str
    trigger_a_regex: str
    trigger_b_regex: str


# Top 20 Sibling Collision Pairs mapped strictly to valid IDs in taxonomy_v1.csv
SIBLING_DISAMBIGUATION_MATRIX: tuple[SiblingRule, ...] = (
    SiblingRule(
        1,
        "EXP-01",
        "EXP-02",
        "Assign EXP-01 when a power is evaluated as base times exponent (3^4 -> 3*4 = 12). "
        "Assign EXP-02 when the product-of-powers or power-of-power law is misused (a^m * a^n -> a^(m*n) or (a^m)^n -> a^(m+n)).",
        r"\b(\d+)\^(\d+)\s*=\s*\1\s*[*×]\s*\2|\b3\^4\s*=\s*12|\b2\^3\s*=\s*6",
        r"([a-z0-9]+)\^\{?(\d+)\}?\s*[*×·]\s*\1\^\{?(\d+)\}?|\(([a-z0-9]+)\^\{?(\d+)\}?\)\^\{?(\d+)\}?",
    ),
    SiblingRule(
        2,
        "EXP-03",
        "EXP-04",
        "Assign EXP-03 when the square of a binomial drops the middle cross-term ((a+b)^2 -> a^2 + b^2). "
        "Assign EXP-04 when a square root is distributed over addition (sqrt(a+b) -> sqrt(a) + sqrt(b)) or fractional exponents are misread.",
        r"\([0-9a-z]+\s*[+-]\s*[0-9a-z]+\)\s*(?:\^2|²)",
        r"(?:\\sqrt|√)\s*\{?\s*[0-9a-z]+\s*[+-]\s*[0-9a-z]+",
    ),
    SiblingRule(
        3,
        "EXP-04",
        "EXP-07",
        "Assign EXP-04 when radicals are split across addition/subtraction (sqrt(a+b) -> sqrt(a)+sqrt(b)) or fractional exponent definitions fail. "
        "Assign EXP-07 when multiplying radicals (sqrt(a)*sqrt(b)) fails or a resulting surd is left unsimplified.",
        r"(?:\\sqrt|√)\s*([0-9a-z]+)\s*[+-]\s*(?:\\sqrt|√)\s*([0-9a-z]+)",
        r"(?:\\sqrt|√)\s*(\d+)\s*[*×·]\s*(?:\\sqrt|√)\s*(\d+)|(?:\\sqrt|√)\s*(?:4|9|16|25|36|49|64|81|100)\b",
    ),
    SiblingRule(
        4,
        "FRA-01",
        "FRA-06",
        "Assign FRA-01 when adding or subtracting fractions adds numerators and denominators directly (a/b + c/d -> (a+c)/(b+d)). "
        "Assign FRA-06 when addition mechanics (common denominator or cross-multiplying) are applied to a fraction multiplication problem.",
        r"(\d+)/(\d+)\s*[+-]\s*(\d+)/(\d+)",
        r"(\d+)/(\d+)\s*[*×·]\s*(\d+)/(\d+)",
    ),
    SiblingRule(
        5,
        "FRA-03",
        "FRA-04",
        "Assign FRA-03 when fraction division fails to convert division into multiplication by the reciprocal of the divisor. "
        "Assign FRA-04 when simplifying strikes out equal digits or non-factor terms across addition/subtraction.",
        r"(\d+)/(\d+)\s*[:÷/]\s*(\d+)/(\d+)",
        r"cancel|coret|striking out",
    ),
    SiblingRule(
        6,
        "ALG-02",
        "ALG-03",
        "Assign ALG-02 when a constant and variable term are conjoined into a single monomial (2 + 3x -> 5x). "
        "Assign ALG-03 when coefficients of non-like variable terms or distinct powers (e.g. 2x + 3y or 2x^2 + 3x) are combined.",
        r"\b\d+\s*[+-]\s*\d+[a-z]\s*=\s*\d+[a-z]\b",
        r"\b\d+[a-z](?:\^\d+)?\s*[+-]\s*\d+[a-z](?:\^\d+)?",
    ),
    SiblingRule(
        7,
        "ALG-04",
        "ALG-06",
        "Assign ALG-04 when distribution across parentheses is incomplete (2(x+3) -> 2x+3) or factoring is invalid. "
        "Assign ALG-06 when a term is moved across '=' without flipping its sign or a factor is moved as subtraction instead of division.",
        r"\d+\s*\(\s*[0-9a-z]+\s*[+-]\s*[0-9a-z]+\s*\)",
        r"pindah ruas|[0-9a-z]+\s*[+-]\s*\d+\s*=\s*\d+",
    ),
    SiblingRule(
        8,
        "ANG-01",
        "ANG-02",
        "Assign ANG-01 when equal parallel-line angles (corresponding/alternate) are treated as supplementary (180°) or vice versa. "
        "Assign ANG-02 when polygon interior angle-sum totals (180° for triangle, 360° for quadrilateral) are misapplied.",
        r"sehadap|berseberangan|sepihak|parallel",
        r"segitiga|segiempat|triangle|quadrilateral|180|360",
    ),
    SiblingRule(
        9,
        "MEA-01",
        "MEA-04",
        "Assign MEA-01 when perimeter is computed instead of area (or vice versa, e.g. summing side lengths for area). "
        "Assign MEA-04 when another 2D shape's area formula is substituted (e.g. rectangle formula for a trapezoid).",
        r"keliling|perimeter|jumlah sisi",
        r"trapesium|jajargenjang|belah ketupat|layang",
    ),
    SiblingRule(
        10,
        "MEA-04",
        "MEA-05",
        "Assign MEA-04 when using the wrong 2D/3D shape formula within the right measurement concept. "
        "Assign MEA-05 when 3D surface area is confused with volume or with the count of faces/edges.",
        r"rumus|formula|luas",
        r"volume|luas permukaan|kubus|balok|prisma|tabung|sisi|rusuk",
    ),
    SiblingRule(
        11,
        "KPF-01",
        "KPF-02",
        "Assign KPF-01 when KPK (LCM) is computed instead of FPB (GCF) or vice versa despite valid factor mechanics. "
        "Assign KPF-02 when KPK is rightly found for a periodic problem, but the Nth simultaneous occurrence is off by one.",
        r"\b(?:kpk|fpb|lcm|gcd|gcf)\b",
        r"bersama-sama|ke-\d+|pertama kali|tanggal|hari",
    ),
    SiblingRule(
        12,
        "KPF-01",
        "FRM-01",
        "Assign KPF-01 when the student completes the KPK/FPB procedure for the opposite concept (LCM vs GCF). "
        "Assign FRM-01 when the division table or formula variant is recalled incompletely without extracting KPK/FPB.",
        r"\b(?:kpk|fpb)\s*=",
        r"\b1\s*\|\s*1\b",
    ),
    SiblingRule(
        13,
        "WOP-01",
        "WOP-02",
        "Assign WOP-01 when the arithmetic is executed properly but the wrong operation is inferred from misreading the word problem. "
        "Assign WOP-02 when an irrelevant or extraneous distractor number from the problem statement is incorporated into the calculation.",
        r"salah operasi|wrong operation",
        r"distractor|angka tidak relevan|extraneous",
    ),
    SiblingRule(
        14,
        "WOP-01",
        "ANS-01",
        "Assign WOP-01 when an incorrect operation causes the student's steps to diverge from the valid solution path. "
        "Assign ANS-01 when all intermediate steps are valid on the correct solution path, but the work stops before the final quantity.",
        r"diverges|wrong operation",
        r"intermediate|belum selesai|stopped",
    ),
    SiblingRule(
        15,
        "INT-02",
        "INT-03",
        "Assign INT-02 when integer multiplication sign rules fail (negative * negative treated as negative). "
        "Assign INT-03 when subtracting a negative integer a - (-b) is not converted into addition a + b.",
        r"-\s*\d+\s*[*×·]\s*\(?-\s*\d+\)?",
        r"\d+\s*-\s*\(\s*-\s*\d+\s*\)",
    ),
    SiblingRule(
        16,
        "INT-03",
        "INT-04",
        "Assign INT-03 specifically when the double-negative subtraction pattern a - (-b) is not converted to a + b. "
        "Assign INT-04 when a minus sign is silently dropped, -40 - 20 is evaluated as -20, or subtraction order is reversed.",
        r"-\s*\(\s*-\s*\d+\s*\)",
        r"-\s*\d+\s*-\s*\d+\s*=\s*-\s*\d+",
    ),
    SiblingRule(
        17,
        "Dec-01",
        "Dec-02",
        "Assign Dec-01 when comparing decimals by digit count ('longer is larger', e.g. 0.45 > 0.8). "
        "Assign Dec-02 when decimal commas/points are misaligned or shifted to the wrong place value during arithmetic operations.",
        r"0[.,]\d{2,}\s*[><]\s*0[.,]\d\b",
        r"0[.,]\d+\s*[*×+-:/]\s*\d+",
    ),
    SiblingRule(
        18,
        "SEQ-01",
        "SEQ-02",
        "Assign SEQ-01 when arithmetic (common difference) and geometric (common ratio) sequences are conceptually swapped. "
        "Assign SEQ-02 when the n-th term formula U_n = a+(n-1)b or a*r^(n-1) is misapplied or off by one index.",
        r"aritmatika|geometri|rasio|beda",
        r"u_?n|\(\s*n\s*-\s*1\s*\)",
    ),
    SiblingRule(
        19,
        "PRO-01",
        "PRO-04",
        "Assign PRO-01 when joint 'AND' probabilities of independent events are added instead of multiplied (P(A)+P(B)). "
        "Assign PRO-04 when the total number of outcomes in the sample space n(S) is counted erroneously.",
        r"(\d+/\d+)\s*\+\s*(\d+/\d+)",
        r"ruang sampel|sample space|dadu|koin|kombinasi|permutasi",
    ),
    SiblingRule(
        20,
        "ANS-01",
        "ANS-02",
        "Assign ANS-01 when the student halts at a valid intermediate value before reaching the requested quantity. "
        "Assign ANS-02 when the student reaches the target value but continues with unneeded operations (over-answers) or confuses change with total.",
        r"stopped|intermediate|belum selesai",
        r"over-answers|selisih|total|further unneeded",
    ),
)


@dataclass
class ParsedRefereeOutput:
    """Structured output parsed from the 4-tag reasoning chain."""

    divergence_step: str
    algebraic_error: str
    secondary_slips: list[str]
    root_code: str
    selected_letter: str | None = None
    parse_valid: bool = True


def get_sibling_pairs_list() -> list[tuple[str, str]]:
    """Return the 20 canonical sibling collision pairs."""
    return [(r.code_a, r.code_b) for r in SIBLING_DISAMBIGUATION_MATRIX]


def format_sibling_rules_for_candidates(candidates: Sequence[str]) -> str:
    """Select active 2-sentence sibling disambiguation rules for a candidate pool."""
    cand_set = {str(c).strip() for c in candidates}
    both_matched: list[str] = []
    single_matched: list[str] = []
    for rule in SIBLING_DISAMBIGUATION_MATRIX:
        if rule.code_a in cand_set and rule.code_b in cand_set:
            both_matched.append(
                f"- Pair #{rule.pair_id} ({rule.code_a} vs {rule.code_b}): {rule.rule_text}"
            )
        elif rule.code_a in cand_set or rule.code_b in cand_set:
            single_matched.append(
                f"- Pair #{rule.pair_id} ({rule.code_a} vs {rule.code_b}): {rule.rule_text}"
            )
    selected = both_matched if both_matched else single_matched[:6]
    if not selected:
        return "- Prioritize conceptual/procedural root causes over secondary slips (CAL-01, ANS-01, NOT-01)."
    return "\n".join(selected[:8])


def build_four_tag_listwise_prompt(
    question_text: str,
    correct_answer: str,
    ocr_steps: str,
    candidates: Sequence[str],
    taxonomy: dict[str, dict[str, str]],
    exemplars: Sequence[dict[str, str]] | None = None,
) -> tuple[str, dict[str, str]]:
    """Build the [A]..[J] listwise prompt with the 4-tag structured reasoning protocol."""
    capped = list(candidates)[: len(LETTERS)]
    cand_options: list[str] = []
    letter_to_cand: dict[str, str] = {}
    for idx, cand in enumerate(capped):
        letter = LETTERS[idx]
        info = taxonomy.get(cand, {})
        name = info.get("name", cand)
        desc = info.get("description", "")
        cand_options.append(f"[{letter}] {cand}: {name} — {desc}")
        letter_to_cand[letter] = cand

    exemplar_section = ""
    if exemplars:
        ex_lines = [
            f"- Classmate error ({ex['root_code']}): {ex['steps']}" for ex in exemplars
        ]
        exemplar_section = (
            "\nKnown Student Error Patterns for this Exact Question:\n"
            + "\n".join(ex_lines)
            + "\n"
        )

    sibling_section = format_sibling_rules_for_candidates(capped)

    prompt = (
        "You are a master K-12 math diagnostic judge for Indonesian student workings.\n"
        "Analyze the student's step-by-step derivation against the question, the target answer, "
        "and known classmate error patterns to pinpoint the primary ROOT MISCONCEPTION.\n\n"
        f"Question: {question_text}\n"
        f"Target Correct Answer: {correct_answer}\n"
        f"Student Written Steps:\n{ocr_steps}\n"
        f"{exemplar_section}\n"
        "Candidate Misconception Definitions:\n"
        + "\n".join(cand_options)
        + "\n\nActive Sibling Disambiguation Rules:\n"
        + sibling_section
        + "\n\nDiagnostic Protocol to reach Global Optima:\n"
        "1. Locate the FIRST DIVERGENT STEP (L_k): Find the exact line where the student's derivation "
        "first deviates from mathematically valid work or from known classmate consensus steps.\n"
        "2. Differentiate Root Misconception vs Secondary Slips (ARM-05):\n"
        "   - Do NOT select secondary calculation errors (like CAL-01 arithmetic slips, ANS-01 stopping "
        "at an intermediate step, or NOT-01 notation slips) if they are merely consequences of an earlier conceptual breakdown.\n"
        "   - Apply the Active Sibling Disambiguation Rules above to separate closely related codes.\n"
        "3. Confirm Clean-Work Gate: If all written steps are mathematically valid and the final result "
        "matches the target answer, select NONE.\n\n"
        "Emit the 4-tag structured reasoning chain followed by the primary ROOT MISCONCEPTION:\n"
        "<divergence_step>L_k</divergence_step>\n"
        "<algebraic_error>delta</algebraic_error>\n"
        "<secondary_slips>CAL-01, ...</secondary_slips>\n"
        "<root_code>[X] CODE</root_code>"
    )
    return prompt, letter_to_cand


_TAG_PATTERNS: dict[str, re.Pattern[str]] = {
    "divergence_step": re.compile(
        r"<divergence_step>\s*(.*?)\s*</divergence_step>", re.DOTALL | re.IGNORECASE
    ),
    "algebraic_error": re.compile(
        r"<algebraic_error>\s*(.*?)\s*</algebraic_error>", re.DOTALL | re.IGNORECASE
    ),
    "secondary_slips": re.compile(
        r"<secondary_slips>\s*(.*?)\s*</secondary_slips>", re.DOTALL | re.IGNORECASE
    ),
    "root_code": re.compile(
        r"<root_code>\s*(.*?)\s*</root_code>", re.DOTALL | re.IGNORECASE
    ),
}
_CODE_REGEX = re.compile(r"\b(NONE|[A-Z][A-Za-z]{1,2}-\d{2})\b")


def parse_referee_xml(
    raw_text: str,
    candidates: Sequence[str],
) -> ParsedRefereeOutput:
    """Parse the 4-tag XML reasoning chain and resolve the predicted root_code."""
    capped = [str(c).strip() for c in candidates[: len(LETTERS)]]
    letter_to_code = {LETTERS[i]: code for i, code in enumerate(capped)}
    valid_codes = set(capped) | {"NONE"}

    extracted: dict[str, str] = {}
    parse_valid = True
    for tag, pattern in _TAG_PATTERNS.items():
        match = pattern.search(raw_text)
        if match:
            extracted[tag] = match.group(1).strip()
        else:
            extracted[tag] = "NONE"
            parse_valid = False

    slips_raw = extracted["secondary_slips"]
    secondary_slips = [
        m.group(1)
        for m in _CODE_REGEX.finditer(slips_raw)
        if m.group(1) != "NONE"
    ]

    root_raw = extracted["root_code"]
    selected_letter: str | None = None
    letter_match = re.search(r"\[([A-J])\]", root_raw)
    if letter_match:
        selected_letter = letter_match.group(1)

    code_matches = [m.group(1) for m in _CODE_REGEX.finditer(root_raw)]
    resolved_code = "NONE"
    if code_matches and code_matches[0] in valid_codes:
        resolved_code = code_matches[0]
        if selected_letter is None:
            for let, c in letter_to_code.items():
                if c == resolved_code:
                    selected_letter = let
                    break
    elif selected_letter and selected_letter in letter_to_code:
        resolved_code = letter_to_code[selected_letter]
    elif root_raw in letter_to_code:
        selected_letter = root_raw
        resolved_code = letter_to_code[root_raw]
    else:
        parse_valid = False

    if resolved_code in NEVER_ROOT_MINOR_CODES:
        resolved_code = "NONE"

    return ParsedRefereeOutput(
        divergence_step=extracted["divergence_step"],
        algebraic_error=extracted["algebraic_error"],
        secondary_slips=secondary_slips,
        root_code=resolved_code,
        selected_letter=selected_letter,
        parse_valid=parse_valid,
    )


def score_and_rerank_candidates(
    candidates: Sequence[str],
    prior_logits: dict[str, float],
    parsed: ParsedRefereeOutput,
    option_token_logprobs: dict[str, float] | None = None,
    arm05_margin_gamma: float = 1.35,
    llm_top1_boost: float = 2.10,
    sibling_delta_boost: float = 0.85,
) -> tuple[list[str], dict[str, float]]:
    """Combine prior logits, 4-tag chain output, sibling triggers, and ARM-05 slip demotion."""
    cand_codes = [str(c).strip() for c in candidates[: len(LETTERS)]]
    raw_vals = [prior_logits.get(c, -4.0) for c in cand_codes]
    mean_v = sum(raw_vals) / max(len(raw_vals), 1)
    std_v = (
        math.sqrt(sum((v - mean_v) ** 2 for v in raw_vals) / max(len(raw_vals), 1))
        + 1e-6
    )
    scores: dict[str, float] = {
        c: (prior_logits.get(c, -4.0) - mean_v) / std_v for c in cand_codes
    }

    if option_token_logprobs:
        for idx, code in enumerate(cand_codes):
            letter = LETTERS[idx]
            if letter in option_token_logprobs:
                scores[code] = (
                    0.55 * scores[code] + 0.45 * option_token_logprobs[letter]
                )

    if parsed.root_code in scores:
        scores[parsed.root_code] += llm_top1_boost

    delta_text = parsed.algebraic_error
    for rule in SIBLING_DISAMBIGUATION_MATRIX:
        if rule.code_a in scores and rule.code_b in scores:
            hit_a = bool(re.search(rule.trigger_a_regex, delta_text, re.IGNORECASE))
            hit_b = bool(re.search(rule.trigger_b_regex, delta_text, re.IGNORECASE))
            if hit_a and not hit_b:
                scores[rule.code_a] += sibling_delta_boost
                scores[rule.code_b] -= 0.5 * sibling_delta_boost
            elif hit_b and not hit_a:
                scores[rule.code_b] += sibling_delta_boost
                scores[rule.code_a] -= 0.5 * sibling_delta_boost

    structural_codes = [
        c
        for c in cand_codes
        if c not in SECONDARY_SLIP_CODES
        and c != "NONE"
        and c not in NEVER_ROOT_MINOR_CODES
    ]
    max_structural_score = max((scores[c] for c in structural_codes), default=-999.0)

    for slip_code in set(parsed.secondary_slips) | SECONDARY_SLIP_CODES:
        if (
            slip_code in scores
            and slip_code != parsed.root_code
            and max_structural_score > -1.0
        ):
            scores[slip_code] = min(
                scores[slip_code], max_structural_score - arm05_margin_gamma
            )

    if "NOT-01" in scores:
        scores["NOT-01"] = -999.0

    ranked = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    return ranked[:3], scores


def expected_map3_root_over_slip_gain(
    p_root_only: float,
    p_both_cooccur: float,
    p_slip_only: float,
    alpha_root_dominance: float = 0.92,
) -> float:
    """Compute exact expected AP@3 gain of ranking (c_root, c_slip, c_3) over (c_slip, c_root, c_3).

    Implements Theorem 1 of Directive 2:
      Delta E[AP@3] = 0.5 * (P(E_10) + (2*alpha - 1)*P(E_11) - P(E_01))
    """
    p_y_root = p_root_only + alpha_root_dominance * p_both_cooccur
    p_y_slip = p_slip_only + (1.0 - alpha_root_dominance) * p_both_cooccur
    return 0.5 * (p_y_root - p_y_slip)
