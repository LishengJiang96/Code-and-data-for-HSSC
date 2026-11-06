"""Dagum Gini coefficient decomposition of the coupling coordination degree.

The script reads ``output/ccd_panel.csv`` (written by ``main_analysis.py``),
computes the overall Gini coefficient for each year, decomposes it into
intra-regional differences, inter-regional differences and the supervariable
density, and writes ``output/dagum_results.csv``.

The formulas follow Eqs. (15)-(18) of the manuscript.  Region labels are
Ganpo (GP), Middle Reaches (MR) and Downstream (DR).

Run from the package root:  python code/dagum_gini.py
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "output" / "ccd_panel.csv"
OUTPUT = BASE / "output" / "dagum_results.csv"

REGIONS = ["Ganpo", "Middle Reaches", "Downstream"]
SHORT = {"Ganpo": "GP", "Middle Reaches": "MR", "Downstream": "DR"}


def gini_overall(values):
    """Overall Gini coefficient (Eq. 15)."""
    values = np.asarray(values, dtype=float)
    n = values.size
    mu = values.mean()
    gaps = np.abs(values[:, None] - values[None, :])
    return gaps.sum() / (2 * n * n * mu)


def gini_within(values):
    """Gini coefficient of one region (Eq. 16, left)."""
    values = np.asarray(values, dtype=float)
    n = values.size
    mu = values.mean()
    gaps = np.abs(values[:, None] - values[None, :])
    return gaps.sum() / (2 * n * n * mu)


def gini_between(a, b):
    """Gini coefficient between two regions (Eq. 16, right)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.abs(a[:, None] - b[None, :]).sum() / (
        a.size * b.size * (a.mean() + b.mean())
    )


def relative_impact(a, b):
    """Relative impact R_jh (Eq. 17).

    Dagum's decomposition orders the two regions so that region j has the
    higher mean; otherwise D_jh can be negative and the between-region term
    loses its interpretation.  The Gini coefficient itself is symmetric, so
    swapping the pair does not affect G_jh or its weight.
    """
    if np.mean(a) < np.mean(b):
        a, b = b, a
    diff = a[:, None] - b[None, :]
    d = np.maximum(diff, 0).mean()
    p = np.maximum(-diff, 0).mean()
    if d + p == 0:
        return 0.0
    return (d - p) / (d + p)


def decompose(year_df):
    """Decompose the Gini coefficient for one year."""
    values = year_df["D"].to_numpy(dtype=float)
    groups = {r: year_df.loc[year_df["Region"] == r, "D"].to_numpy(dtype=float)
              for r in REGIONS}

    n = values.size
    mu = values.mean()
    g_total = gini_overall(values)

    # Within-region Gini coefficients and population/CCD shares.
    g_within = {r: gini_within(g) for r, g in groups.items()}
    p_share = {r: len(g) / n for r, g in groups.items()}
    s_share = {r: len(g) * g.mean() / (n * mu) for r, g in groups.items()}

    g_between = {}
    d_impact = {}
    for r1, r2 in combinations(REGIONS, 2):
        g_between[(r1, r2)] = gini_between(groups[r1], groups[r2])
        d_impact[(r1, r2)] = relative_impact(groups[r1], groups[r2])

    g_w = sum(g_within[r] * p_share[r] * s_share[r] for r in REGIONS)

    g_nb = 0.0
    g_t = 0.0
    for (r1, r2) in combinations(REGIONS, 2):
        weight = p_share[r1] * s_share[r2] + p_share[r2] * s_share[r1]
        g_nb += g_between[(r1, r2)] * weight * d_impact[(r1, r2)]
        g_t += g_between[(r1, r2)] * weight * (1 - d_impact[(r1, r2)])

    return {
        "Total": g_total,
        "G_GP": g_within["Ganpo"],
        "G_MR": g_within["Middle Reaches"],
        "G_DR": g_within["Downstream"],
        "G_GP_DR": g_between[("Ganpo", "Downstream")],
        "G_GP_MR": g_between[("Ganpo", "Middle Reaches")],
        "G_DR_MR": g_between[("Middle Reaches", "Downstream")],
        "Intra": g_w / g_total * 100,
        "Inter": g_nb / g_total * 100,
        "Super": g_t / g_total * 100,
        "Check": (g_w + g_nb + g_t) / g_total * 100,
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Run main_analysis.py first."
        )

    panel = pd.read_csv(INPUT)
    rows = []
    for year, year_df in panel.groupby("Year"):
        record = {"Year": int(year)}
        record.update(decompose(year_df))
        rows.append(record)

    results = pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)
    average = {"Year": "Average"}
    average.update({c: results[c].mean() for c in results.columns if c != "Year"})
    results = pd.concat([results, pd.DataFrame([average])], ignore_index=True)

    results.to_csv(OUTPUT, index=False, float_format="%.4f")

    print(f"Wrote {OUTPUT}")
    first = results.iloc[0]
    last = results.iloc[-2]
    print(f"Overall Gini: {first['Year']} = {first['Total']:.3f}, "
          f"{last['Year']} = {last['Total']:.3f}")
    print(f"Average contributions: intra = {average['Intra']:.2f}%, "
          f"inter = {average['Inter']:.2f}%, "
          f"supervariable = {average['Super']:.2f}%")
    print(f"Additivity check (should be 100): {average['Check']:.2f}%")
    print("Benchmarks in the manuscript: Gini 0.127 (2010) -> 0.077 (2023); "
          "average contributions 35.85% / 19.23% / 44.92%.")


if __name__ == "__main__":
    main()
