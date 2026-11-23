"""Wave-4 counterfactual trace synthesis (textual buggy rewrites, CPU only).

Synthesizes one-line algebraic and conceptual mutations defined by the taxonomy
by applying forward buggy rewrite operators to clean classmate solution lines.
Covers the 38 rare long-tail taxonomy codes (<10 training rows) plus core
algebraic/exponent/fraction misconception operators.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import Optional

Rewrite = Callable[[str], Optional[str]]

_M = r"-?\d+(?:\.\d+)?"
_NUM = r"\d+(?:\.\d+)?"
_NUMSIGNED = r"-?\d+(?:\.\d+)?"


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(_M, text)]


def _fmt(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _close(a: float, b: float) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def _rewrite_exp01(line: str) -> str | None:
    """a^b = R -> a^b = a*b (exponent misread as multiplication).

    Only fires when the stated RHS is the TRUE value of a^b, so a valid step is
    corrupted rather than masking an already-buggy trace.
    """
    m = re.search(
        rf"(?<![\w\d.])({_NUM})\^({_NUMSIGNED})\s*=\s*({_NUMSIGNED})(?![\d.])", line
    )
    if not m:
        return None
    a, b, rhs = float(m.group(1)), float(m.group(2)), float(m.group(3))
    try:
        true_value = float(a) ** float(b)
    except (OverflowError, ValueError):
        return None
    if not _close(true_value, rhs):
        return None
    return line[: m.start(3)] + _fmt(a * b) + line[m.end(3) :]


def _rewrite_exp02(line: str) -> str | None:
    """a^m * a^n = a^(m+n) -> a^(m*n); only when the stated exponent is correct."""
    m = re.search(
        rf"(?<![\w\d.])({_NUM})\^({_NUMSIGNED})\s*\*\s*({_NUM})\^({_NUMSIGNED})"
        rf"\s*=\s*({_NUM})\^({_NUMSIGNED})(?![\d.])",
        line,
    )
    if not m:
        return None
    base_a, base_r = float(m.group(1)), float(m.group(5))
    m_exp, n_exp, rhs_exp = (
        float(m.group(2)),
        float(m.group(4)),
        float(m.group(6)),
    )
    if not _close(base_a, base_r) or not _close(m_exp + n_exp, rhs_exp):
        return None
    span_start, span_end = m.start(6), m.end(6)
    return line[:span_start] + _fmt(m_exp * n_exp) + line[span_end:]


def _rewrite_exp03(line: str) -> str | None:
    """(a+b)^2 expansion -> drop the middle cross term (the 2ab term)."""
    m = re.search(r"\^\s*2\s*=\s*(.+)$", line)
    if not m:
        return None
    terms = [t.strip() for t in re.split(r"\s*\+\s*", m.group(1)) if t.strip()]
    if len(terms) < 3:
        return None
    first, middle = terms[0], terms[1]
    cross_has_var = bool(re.search(r"[a-z]", middle))
    first_is_square = bool(re.search(r"(\^2|²|\*2)", first)) or bool(
        re.fullmatch(rf"{_M}", first.replace(" ", ""))
    )
    if not cross_has_var or not first_is_square:
        return None
    kept = [terms[0], *terms[2:]]
    return line[: m.start(1)] + " + ".join(kept) + line[m.end(1) :]


def _rewrite_fra01(line: str) -> str | None:
    """a/b + c/d = R -> a+c/b+d; only when the stated RHS is the true sum."""
    m = re.search(
        rf"(?<![\w\d.])({_NUM})\s*/\s*({_NUM})\s*\+\s*({_NUM})\s*/\s*({_NUM})"
        rf"\s*=\s*({_NUMSIGNED})\s*/\s*({_NUMSIGNED})(?![\d.])",
        line,
    )
    if not m:
        return None
    a, b, c, d = (float(m.group(i)) for i in range(1, 5))
    rhs_num, rhs_den = float(m.group(5)), float(m.group(6))
    if abs(float(b)) < 1e-9 or abs(float(d)) < 1e-9 or abs(rhs_den) < 1e-9:
        return None
    if not _close(rhs_num * float(b) * float(d), rhs_den * (a * d + c * b)):
        return None
    span_start, span_end = m.start(5), m.end(6)
    result = f"{_fmt(a + c)}/{_fmt(b + d)}"
    return line[:span_start] + result + line[span_end:]


def _rewrite_ooo01(line: str) -> str | None:
    """Left-to-right evaluation before multiplication: a + b*c = (a+b)*c.

    Only fires when the stated RHS is the correctly-ordered value a + b*c.
    """
    m = re.search(
        rf"(?<![\w\d.])({_NUM})\s*\+\s*({_NUM})\s*\*\s*({_NUM})\s*=\s*({_NUMSIGNED})(?![\d.])",
        line,
    )
    if not m:
        return None
    a, b, c, rhs = (
        float(m.group(1)),
        float(m.group(2)),
        float(m.group(3)),
        float(m.group(4)),
    )
    if not _close(a + b * c, rhs):
        return None
    span_start, span_end = m.start(4), m.end(4)
    return line[:span_start] + _fmt((a + b) * c) + line[span_end:]


def _rewrite_ooo02(line: str) -> str | None:
    """a - b + c = R -> a - (b + c); sign associated with wrong term in +/- chain."""
    m = re.search(
        rf"(?<![\w\d.])({_NUM})\s*-\s*({_NUM})\s*\+\s*({_NUM})\s*=\s*({_NUMSIGNED})(?![\d.])",
        line,
    )
    if not m:
        return None
    a, b, c, rhs = (
        float(m.group(1)),
        float(m.group(2)),
        float(m.group(3)),
        float(m.group(4)),
    )
    if c == 0 or not _close(a - b + c, rhs):
        return None
    return line[: m.start(4)] + _fmt(a - (b + c)) + line[m.end(4) :]


def _rewrite_fra03(line: str) -> str | None:
    """(a/b) : (c/d) = R -> (a*c)/(b*d) (division without inverting divisor)."""
    m = re.search(
        rf"(?<![\w\d.])({_NUM})\s*/\s*({_NUM})\s*[:÷]\s*({_NUM})\s*/\s*({_NUM})"
        rf"\s*=\s*({_NUMSIGNED})\s*/\s*({_NUMSIGNED})(?![\d.])",
        line,
    )
    if not m:
        return None
    a, b, c, d = (float(m.group(i)) for i in range(1, 5))
    rhs_num, rhs_den = float(m.group(5)), float(m.group(6))
    if abs(b) < 1e-9 or abs(c) < 1e-9 or abs(d) < 1e-9 or abs(rhs_den) < 1e-9:
        return None
    if not _close(rhs_num * (b * c), rhs_den * (a * d)):
        return None
    if _close(a * c, a * d) and _close(b * d, b * c):
        return None
    return line[: m.start(5)] + f"{_fmt(a * c)}/{_fmt(b * d)}" + line[m.end(6) :]


def _rewrite_fra05(line: str) -> str | None:
    """w n/d = (w*d+n)/d -> (w*n)/d (mixed-to-improper fraction conversion error)."""
    m = re.search(
        r"(?<![\w\d.])(\d+)\s+(\d+)\s*/\s*(\d+)\s*=\s*(\d+)\s*/\s*(\d+)(?![\d.])",
        line,
    )
    if not m:
        return None
    w, n, d, rn, rd = (int(m.group(i)) for i in range(1, 6))
    if d <= 1 or rd != d or rn != w * d + n:
        return None
    buggy_num = w * d * n if w * n == rn else w * n
    return line[: m.start(4)] + f"{buggy_num}/{d}" + line[m.end(5) :]


def _rewrite_exp05(line: str) -> str | None:
    """a^0 = 1 -> a^0 = 0 (zero exponent rule misapplied)."""
    m = re.search(rf"(?<![\w\d.])({_NUM})\^0\s*=\s*1(?![\d.])", line)
    if not m:
        return None
    return line[: m.end(0) - 1] + "0" + line[m.end(0) :]


# All 38 rare long-tail root taxonomy codes (<10 positive root traces in train.csv,
# excluding non-root/meta codes CHK-01, NOT-01, NOT-02, REL-01) plus the 5 classic
# line-rewrite codes.
RARE_38_TAXONOMY_CODES: tuple[str, ...] = (
    "OOO-02",
    "KPF-03",
    "INT-01",
    "FRA-01",
    "FRA-02",
    "FRA-03",
    "FRA-04",
    "FRA-05",
    "Dec-01",
    "EXP-03",
    "RAT-01",
    "RAT-02",
    "ALG-01",
    "ALG-02",
    "ALG-03",
    "ALG-05",
    "MEA-01",
    "MEA-02",
    "MEA-03",
    "MEA-05",
    "GEO-01",
    "ANG-02",
    "COR-01",
    "STA-03",
    "STA-04",
    "STA-05",
    "PRO-01",
    "PRO-02",
    "PRO-03",
    "LIN-01",
    "SIM-01",
    "QUA-02",
    "EXP-05",
    "EXP-06",
    "SEQ-01",
    "SEQ-02",
    "SET-01",
    "UNT-01",
)

SUPPORTED_CODES: tuple[str, ...] = (
    "EXP-01",
    "EXP-02",
    "EXP-03",
    "FRA-01",
    "OOO-01",
    *tuple(c for c in RARE_38_TAXONOMY_CODES if c not in ("EXP-03", "FRA-01")),
)

BUGGY_REWRITES: dict[str, tuple[Rewrite, ...]] = {
    "EXP-01": (_rewrite_exp01,),
    "EXP-02": (_rewrite_exp02,),
    "EXP-03": (_rewrite_exp03,),
    "FRA-01": (_rewrite_fra01,),
    "OOO-01": (_rewrite_ooo01,),
    "OOO-02": (_rewrite_ooo02,),
    "FRA-03": (_rewrite_fra03,),
    "FRA-05": (_rewrite_fra05,),
    "EXP-05": (_rewrite_exp05,),
}

# Topic-anchored single-line counterfactual templates for each of the 38 rare
# taxonomy codes so clean classmate derivations on matching topics can be
# deterministically perturbed at an intermediate step L_k.
RARE_CODE_STEP_PERTURBATIONS: dict[str, tuple[tuple[str, ...], str]] = {
    "OOO-02": (
        ("Operations on Integers", "Fraction operations", "Rational Number Operations"),
        "2/3 + 1/10 - 2/3 = 2/3 - 1/10 = 17/30",
    ),
    "KPF-03": (
        ("GCD", "LCM", "Introduction to LCM and GCD", "Applications of LCM and GCD"),
        "FPB dari 16, 24, dan 32 -> hanya menghitung FPB(16, 24) = 8 (32 diabaikan)",
    ),
    "INT-01": (
        ("Integer Value", "Introduction to Integers", "Operations on Integers"),
        "-5 > -2 karena 5 > 2",
    ),
    "FRA-01": (
        ("Fraction operations", "Rational Number Operations"),
        "2/3 + 1/4 = (2+1)/(3+4) = 3/7",
    ),
    "FRA-02": (
        ("Comparison of Fractions", "Introduction to Rational Numbers", "Fraction Conversion"),
        "1/8 > 1/4 karena penyebut 8 > 4",
    ),
    "FRA-03": (
        ("Fraction operations", "Rational Number Operations"),
        "2/3 : 4/5 = (2*4)/(3*5) = 8/15",
    ),
    "FRA-04": (
        ("Fraction Conversion", "Fraction operations", "Rationalizing the Denominator"),
        "(16 + 4) / 16 = 4 (mencoret 16 tanpa memfaktorkan)",
    ),
    "FRA-05": (
        ("Fraction Conversion", "Fraction operations", "Conversion of Rational Numbers"),
        "2 3/4 = (2*3)/4 = 6/4",
    ),
    "Dec-01": (
        ("Comparison of Fractions", "Conversion of Rational Numbers", "Fraction Conversion"),
        "0,45 > 0,8 karena 45 > 8",
    ),
    "EXP-03": (
        ("Exponents", "Operations of Exponents", "Value and Form of a Function"),
        "(x + 3)^2 = x^2 + 9",
    ),
    "RAT-01": (
        ("Speed", "Data Presentation and Interpretation", "Whole Number Operations up to 10,000"),
        "rasio 3 : 5 dianggap selisih 5 - 3 = 2 (berpikir aditif)",
    ),
    "RAT-02": (
        ("Data Presentation and Interpretation", "Fraction Conversion", "Conversion of Rational Numbers"),
        "25% dari 80 = 25 * 80 = 2000 (persen tidak dibagi 100)",
    ),
    "ALG-01": (
        ("Linear Equation in One Variable", "Applications of Linear Equations in One Variable"),
        "2a + 3 = 5a (variabel a dibaca sebagai label objek)",
    ),
    "ALG-02": (
        ("Value and Form of a Function", "Linear Equation in One Variable", "Function Operations"),
        "2 + 3x = 5x (menggabungkan konstanta dan suku bervariabel)",
    ),
    "ALG-03": (
        ("Function Operations", "Application of Two-Variable Linear Equation Systems"),
        "2x + 3y = 5xy (menjumlahkan koefisien suku tidak sejenis)",
    ),
    "ALG-05": (
        ("Linear Equation in One Variable", "Applications of Linear Equations in One Variable"),
        "2x + 4 = 10 -> 2x + 4 - 4 = 10 -> 2x = 10",
    ),
    "MEA-01": (
        ("Area of Flat Shapes", "Composition of 2D Shapes", "Surface Area of a Geometric Solid"),
        "Luas = jumlah seluruh sisi = s1 + s2 + s3 + s4 (keliling dihitung sebagai luas)",
    ),
    "MEA-02": (
        ("Volume of a Geometric Shape", "Surface Area of a Geometric Solid", "Speed"),
        "1,44 m^2 = 1,44 * 100 = 144 cm^2 (faktor konversi luas salah 100 bukan 10.000)",
    ),
    "MEA-03": (
        ("Volume of a Geometric Shape", "Surface Area of a Geometric Solid", "Area of Flat Shapes"),
        "Volume = 72 cm (satuan pangkat tiga cm^3 diabaikan/salah)",
    ),
    "MEA-05": (
        ("Surface Area of a Geometric Solid", "Introduction to Three-Dimensional Shapes", "Volume of a Geometric Shape"),
        "Luas permukaan balok = p * l * t (rumus volume dipakai untuk luas permukaan)",
    ),
    "GEO-01": (
        ("Area of Flat Shapes", "Composition of 2D Shapes", "Introduction to Three-Dimensional Shapes"),
        "Luas jajargenjang = sisi_miring * alas (sifat persegi panjang diterapkan pada jajargenjang)",
    ),
    "ANG-02": (
        ("Angle", "Area of Flat Shapes", "Linear Equation in One Variable"),
        "Jumlah sudut segitiga = 360° -> x + 60° + 70° = 360°",
    ),
    "COR-01": (
        ("Introduction to Geometric Transformation", "Composition of Geometric Transformation", "Equation of the Regression Line"),
        "Titik (x, y) = (2, -5) dibaca terbalik menjadi (y, x) = (-5, 2)",
    ),
    "STA-03": (
        ("Measures of Central Tendency", "Data Presentation and Interpretation"),
        "Rata-rata (mean) ditentukan dengan mengambil nilai tengah (median) data terurut",
    ),
    "STA-04": (
        ("Measures of Central Tendency", "Measures of Dispersion"),
        "Rata-rata data berkelompok = jumlah titik tengah / banyak kelas (frekuensi f_i diabaikan)",
    ),
    "STA-05": (
        ("Measures of Dispersion", "Measures of Central Tendency"),
        "Kuartil Q1 = n/4 = 20/4 = 5 (posisi indeks kuartil dianggap sebagai nilai kuartil)",
    ),
    "PRO-01": (
        ("Probability of Compound Events", "Probability of an Event"),
        "P(A dan B) = P(A) + P(B) = 1/2 + 1/3 = 5/6 (peluang kejadian saling bebas dijumlahkan)",
    ),
    "PRO-02": (
        ("Probability of Compound Events", "Probability of an Event"),
        "Tanpa pengembalian: P(ambil ke-2) tetap dibagi n(S) awal tanpa dikurangi 1",
    ),
    "PRO-03": (
        ("Probability of an Event", "Probability of Compound Events"),
        "P(A) = 8/5 = 1,6 (nilai peluang > 1 tidak dikenali sebagai mustahil)",
    ),
    "LIN-01": (
        ("Equation of the Regression Line", "Linear Equation in One Variable", "Types of Functions"),
        "Gradien m = (x2 - x1) / (y2 - y1) (rumus perubahan x dibagi perubahan y terbalik)",
    ),
    "SIM-01": (
        ("Application of Two-Variable Linear Equation Systems", "Linear Equation in One Variable"),
        "Eliminasi: (2x + y = 10) - (x - y = 2) -> x = 8 (salah tanda saat mengeliminasi -y)",
    ),
    "QUA-02": (
        ("Value and Form of a Function", "Function Composition", "Function Operations"),
        "x^2 - 5x + 6 = (x + 2)(x + 3) (tanda pemfaktoran kuadrat salah)",
    ),
    "EXP-05": (
        ("Exponents", "Operations of Exponents", "Introduction to Exponents"),
        "5^0 = 0 (pangkat nol dievaluasi menjadi 0 bukan 1)",
    ),
    "EXP-06": (
        ("Operations of Exponents", "Exponents", "Exponential Function", "Exponential Applications (Exponents)"),
        "3^(x+1) = 3^4 -> berhenti tanpa menyamakan eksponen x + 1 = 4",
    ),
    "SEQ-01": (
        ("Arithmetic Sequences and Series", "Applications of geometric sequences and series"),
        "Barisan geometri r = 2 dihitung dengan rumus aritmatika Un = a + (n - 1)b",
    ),
    "SEQ-02": (
        ("Arithmetic Sequences and Series", "Applications of geometric sequences and series"),
        "Un = a + n*b (indeks suku ke-n meleset satu tanpa n - 1)",
    ),
    "SET-01": (
        ("Counting Rules", "Probability of Compound Events", "Types of Functions"),
        "n(A irisan B) dihitung sebagai n(A gabungan B) = n(A) + n(B)",
    ),
    "UNT-01": (
        ("Data Presentation and Interpretation", "Whole Number Operations up to 10,000", "Fraction operations"),
        "15% + 6 siswa = 21 siswa (menjumlahkan persentase langsung dengan jumlah orang)",
    ),
}


def apply_buggy_rewrite(line: str, code: str) -> str | None:
    """Apply the first matching buggy rewrite for ``code``; None when N/A."""
    for rewrite in BUGGY_REWRITES.get(code, ()):
        mutated = rewrite(line)
        if mutated is not None and mutated != line:
            return mutated
    return None


def synthesize_countefactual_line(
    ocr_lines: Sequence[str],
    code: str,
    *,
    avoid_last_line: bool = True,
) -> tuple[int, str] | None:
    """Return (index, mutated_line) for the first mutable line, else None.

    The last line is skipped by default so the final answer line stays intact
    (mutating it changes the graded target rather than the process).
    """
    candidates = range(
        len(ocr_lines) if not avoid_last_line else max(len(ocr_lines) - 1, 0)
    )
    for index in candidates:
        line = ocr_lines[index]
        mutated = apply_buggy_rewrite(line, code)
        if mutated is not None:
            return index, mutated
    return None


def synthesize_topic_counterfactual_line(
    ocr_lines: Sequence[str],
    code: str,
    topic_name: str,
    *,
    avoid_last_line: bool = True,
) -> tuple[int, str] | None:
    """Synthesize a single-line counterfactual step mutation for rare code ``code``.

    First attempts exact algebraic regex rewrites (:func:`synthesize_countefactual_line`).
    When no regex rewrite fires and ``topic_name`` matches the curriculum domain of
    ``code`` in :data:`RARE_CODE_STEP_PERTURBATIONS`, mutates the first intermediate
    step ``L_k`` to inject the canonical buggy state transition while preserving line
    count and the final answer line.
    """
    direct = synthesize_countefactual_line(
        ocr_lines, code, avoid_last_line=avoid_last_line
    )
    if direct is not None:
        return direct

    topic = str(topic_name or "").strip()
    if not topic or code not in RARE_CODE_STEP_PERTURBATIONS:
        return None

    allowed_topics, buggy_transition = RARE_CODE_STEP_PERTURBATIONS[code]
    if topic not in allowed_topics:
        return None

    limit = len(ocr_lines) if not avoid_last_line else max(len(ocr_lines) - 1, 0)
    if limit <= 0:
        return None

    target_idx = 0
    raw_line = ocr_lines[target_idx]
    prefix_match = re.match(r"^(\s*L\d+\s*\|\s*)", raw_line)
    prefix = prefix_match.group(1) if prefix_match else ""
    body = raw_line[len(prefix) :].strip()
    mutated = f"{prefix}{body} -> [{code}] {buggy_transition}"
    if mutated == raw_line:
        return None
    return target_idx, mutated


__all__ = [
    "RARE_38_TAXONOMY_CODES",
    "RARE_CODE_STEP_PERTURBATIONS",
    "SUPPORTED_CODES",
    "apply_buggy_rewrite",
    "synthesize_countefactual_line",
    "synthesize_topic_counterfactual_line",
]
