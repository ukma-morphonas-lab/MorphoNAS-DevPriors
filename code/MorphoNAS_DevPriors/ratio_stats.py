"""
Shared statistics for the developmental-priors ratios.

Two binomial proportions (MorphoNAS competence rate vs matched-random rate) and
their ratio, plus the cross-task strengtheners:

  * wilson_ci / ratio_ci_mover  -- the proven MOVER+Wilson ratio CI (Donner &
    Zou 2012), copied verbatim from the locked control so every script reports
    one interval definition. Stays finite when the random arm has zero hits.
  * grade_vs_reference          -- the graded comparison vs a reference ratio.
  * ror_delta                   -- ratio-of-ratios (RR_a / RR_b) with a
    delta-method CI on the log scale: the cross-task
    interaction, reported on the non-weak bar.
  * logistic_interaction        -- the saturated logistic interaction term
    (ln of the ratio of odds ratios) with a closed-form delta-method SE and a
    two-sided p. Same question as ror_delta, odds-ratio flavoured; no
    statsmodels dependency.

Zero cells (the Acrobot solved arm) are handled by an optional Haldane-Anscombe
+0.5 continuity correction, which the caller is told was applied.
"""

from __future__ import annotations

import math
from typing import Optional

Z95 = 1.959963984540054  # standard normal 0.975 quantile


def wilson_ci(x: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion. Lower bound is exactly 0 when
    x == 0, which keeps the ratio CI well-defined with no random successes."""
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def ratio_ci_mover(x1: int, n1: int, x2: int, n2: int, z: float = Z95) -> dict:
    """95% CI for the ratio (x1/n1) / (x2/n2) of two INDEPENDENT binomial
    proportions, via MOVER combined with Wilson intervals (Donner & Zou 2012).

    Stays finite and correct when the denominator arm has zero successes
    (x2 == 0): the lower bound is sqrt(l1*(2*p1-l1))/u2 and the upper is +inf.
    """
    p1 = x1 / n1 if n1 else 0.0
    p2 = x2 / n2 if n2 else 0.0
    l1, u1 = wilson_ci(x1, n1, z)
    l2, u2 = wilson_ci(x2, n2, z)

    point = (p1 / p2) if p2 > 0 else math.inf

    if p2 == 0:
        lo = math.sqrt(max(l1 * (2 * p1 - l1), 0.0)) / u2 if u2 > 0 else math.inf
    else:
        A = u2 * (2 * p2 - u2)
        disc = max(p1 * p1 * p2 * p2 - l1 * (2 * p1 - l1) * u2 * (2 * p2 - u2), 0.0)
        lo = (p1 * p2 - math.sqrt(disc)) / A

    Au = l2 * (2 * p2 - l2)
    if Au <= 0:
        hi = math.inf
    else:
        disc_u = max(p1 * p1 * p2 * p2 - u1 * (2 * p1 - u1) * l2 * (2 * p2 - l2), 0.0)
        hi = (p1 * p2 + math.sqrt(disc_u)) / Au

    return {
        "point": point,
        "lo": lo,
        "hi": hi if math.isfinite(hi) else None,
        "mn_rate": p1,
        "mn_ci": list(wilson_ci(x1, n1, z)),
        "rand_rate": p2,
        "rand_ci": list(wilson_ci(x2, n2, z)),
        "mn_count": x1,
        "mn_n": n1,
        "rand_count": x2,
        "rand_n": n2,
    }


def grade_vs_reference(
    acro_lo: float,
    acro_hi: Optional[float],
    ref_point: float,
    ref_lo: float,
    ref_hi: float,
) -> dict:
    """Graded comparison. Primary: CI lower bound vs a reference point
    ratio. Secondary, stricter: do the two CIs overlap."""
    hi = acro_hi if acro_hi is not None else math.inf
    if acro_lo > ref_point:
        primary = "grows"
    elif hi >= ref_point:
        primary = "persists"
    else:
        primary = "shrinks"
    cis_overlap = not (acro_lo > ref_hi or hi < ref_lo)
    return {
        "verdict": primary,
        "acrobot_ci_lo_vs_ref_point": acro_lo - ref_point,
        "cis_overlap": cis_overlap,
        "reference_point": ref_point,
        "reference_ci": [ref_lo, ref_hi],
    }


def _log_rr_var(x_mn: int, n_mn: int, x_rand: int, n_rand: int, cc: float) -> tuple[float, float, bool]:
    """ln(RR) and its delta-method variance for one task. RR = (x_mn/n_mn) /
    (x_rand/n_rand). Applies a +cc continuity correction to all four counts if
    any numerator is zero (so the log stays finite); reports whether it fired."""
    corrected = (x_mn == 0) or (x_rand == 0)
    a, b = (x_mn + cc, n_mn + 2 * cc) if corrected else (x_mn, n_mn)
    c, d = (x_rand + cc, n_rand + 2 * cc) if corrected else (x_rand, n_rand)
    ln_rr = math.log((a / b) / (c / d))
    var = 1.0 / a - 1.0 / b + 1.0 / c - 1.0 / d
    return ln_rr, var, corrected


def ror_delta(
    task_a: dict,
    task_b: dict,
    z: float = Z95,
    cc: float = 0.5,
) -> dict:
    """Ratio-of-ratios RoR = RR_a / RR_b with a delta-method CI on the log scale.

    Each task dict carries {x_mn, n_mn, x_rand, n_rand}. Var(ln RoR) is the sum
    of the two per-task Var(ln RR). RoR > 1 with a CI excluding 1 is the
    significant-growth result (task_a harder than task_b). The harder task is
    task_a by convention.
    """
    ln_a, var_a, cc_a = _log_rr_var(
        task_a["x_mn"], task_a["n_mn"], task_a["x_rand"], task_a["n_rand"], cc
    )
    ln_b, var_b, cc_b = _log_rr_var(
        task_b["x_mn"], task_b["n_mn"], task_b["x_rand"], task_b["n_rand"], cc
    )
    ln_ror = ln_a - ln_b
    se = math.sqrt(var_a + var_b)
    lo = math.exp(ln_ror - z * se)
    hi = math.exp(ln_ror + z * se)
    zstat = ln_ror / se if se > 0 else math.inf
    p = math.erfc(abs(zstat) / math.sqrt(2.0))  # two-sided
    return {
        "ror_point": math.exp(ln_ror),
        "ror_ci": [lo, hi],
        "ln_ror": ln_ror,
        "se_ln_ror": se,
        "z": zstat,
        "p_two_sided": p,
        "rr_a": math.exp(ln_a),
        "rr_b": math.exp(ln_b),
        "continuity_correction": cc if (cc_a or cc_b) else 0.0,
        "ci_excludes_1": lo > 1.0 or hi < 1.0,
        "grows": lo > 1.0,
    }


def logistic_interaction(
    task_a: dict,
    task_b: dict,
    cc: float = 0.5,
) -> dict:
    """Saturated logistic interaction (competent ~ encoding * task): the term
    equals ln(OR_a / OR_b), with delta-method SE = sqrt(sum 1/cell) over the
    eight cells. Two-sided p. Odds-ratio analogue of ror_delta."""

    def cells(t):
        x_mn, n_mn = t["x_mn"], t["n_mn"]
        x_rand, n_rand = t["x_rand"], t["n_rand"]
        return [x_mn, n_mn - x_mn, x_rand, n_rand - x_rand]

    raw = cells(task_a) + cells(task_b)
    corrected = any(c == 0 for c in raw)
    adj = [c + cc for c in raw] if corrected else raw
    a1, a0, ar1, ar0, b1, b0, br1, br0 = adj
    ln_or_a = math.log((a1 * ar0) / (a0 * ar1))
    ln_or_b = math.log((b1 * br0) / (b0 * br1))
    ln_int = ln_or_a - ln_or_b
    se = math.sqrt(sum(1.0 / c for c in adj))
    zstat = ln_int / se if se > 0 else math.inf
    p = math.erfc(abs(zstat) / math.sqrt(2.0))
    return {
        "interaction_or": math.exp(ln_int),
        "or_a": math.exp(ln_or_a),
        "or_b": math.exp(ln_or_b),
        "ln_interaction": ln_int,
        "se": se,
        "z": zstat,
        "p_two_sided": p,
        "continuity_correction": cc if corrected else 0.0,
        "significant_growth": math.exp(ln_int - 1.959963984540054 * se) > 1.0,
    }


def fisher_within_task(x_mn: int, n_mn: int, x_rand: int, n_rand: int) -> dict:
    """Two-proportion exact test (Fisher) of MN-competent vs random-competent
    within one task. Table-stakes significance; uses scipy if available."""
    table = [[x_mn, n_mn - x_mn], [x_rand, n_rand - x_rand]]
    try:
        from scipy.stats import fisher_exact

        odds, p = fisher_exact(table, alternative="greater")
        return {"odds_ratio": float(odds), "p_one_sided_greater": float(p), "method": "fisher_exact"}
    except Exception as e:  # pragma: no cover - scipy expected present
        return {"error": str(e), "table": table}
