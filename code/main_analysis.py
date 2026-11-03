"""Reproduce the food and ecological security assessment.

Workflow
--------
1. Read the city-year panel (two sheets: ecological security and food security).
2. Compute entropy and CRITIC weights and combine them with the uninorm
   operator (Eqs. 1-7 of the manuscript).
3. Fit generalised extreme value distributions to the maximum and minimum
   evaluation gaps and derive the goodness/badness probabilities (Eqs. 8-11).
4. Aggregate the probabilities into a composite evaluation for each city-year
   (Eqs. 10-11).
5. Compute the improved coupling coordination degree (Eqs. 12-16).
6. Write the scores, weights, CCD values and a region-tagged CSV that is used
   by the Dagum decomposition and kernel density scripts.

Run from the package root:  python code/main_analysis.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import genextreme

# Make ``method.py`` importable when the script is run from any directory.
sys.path.append(str(Path(__file__).resolve().parent))
import method  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
DATA_FILE = BASE / "data" / "panel_data.xlsx"
OUTPUT_DIR = BASE / "output"
FIGURE_DIR = BASE / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# Set to True to save the empirical-vs-fitted GEV diagnostic plots.
SAVE_DIAGNOSTICS = False

# Indicator attributes.  The order must match the columns of each sheet after
# the first two columns (city and year).  "cost" indicators are multiplied by
# -1 before the outranking step; "benefit" indicators are kept as they are.
SHEETS = {
    "Ecological security": {
        "label": "Ecological security",
        "types": [
            "cost", "cost", "cost", "cost",           # pressure
            "benefit", "benefit", "benefit",          # state
            "benefit", "benefit", "benefit",          # response
        ],
        "columns": [
            "Population density", "Urbanization rate",
            "Proportion of agricultural water consumption", "GDP growth rate",
            "Arable land per capita", "Per capita park green space",
            "Green space coverage in built-up areas",
            "Fiscal expenditure on agriculture and forestry (100 million CNY)",
            "Effective irrigated area ratio",
            "Environmental protection investment ratio",
        ],
    },
    "Food security": {
        "label": "Food security",
        "types": [
            "benefit", "benefit", "cost",              # quantitative
            "cost", "cost", "cost", "benefit",         # qualitative
            "cost", "benefit", "benefit",              # consumption
            "cost", "cost", "benefit", "benefit",      # resource
        ],
        "columns": [
            "Grain output per unit area", "Multiple cropping index",
            "Fluctuation coefficient of grain production",
            "Fertilizer use intensity", "Agricultural film use intensity",
            "Pesticide use intensity", "Gross power of agricultural machinery",
            "Natural population growth rate", "Grain possession per capita",
            "Grain self-sufficiency rate",
            "Water resource consumption per unit grain",
            "Arable land resource consumption per unit grain",
            "Agricultural labor productivity", "Agricultural land productivity",
        ],
    },
}


def region_of(city):
    """Map a city label such as ``Jiangsu - Nanjing`` to one of the three regions."""
    province = city.split(" - ")[0]
    if province in ("Jiangsu", "Anhui"):
        return "Downstream"
    if province == "Jiangxi":
        return "Ganpo"
    if province in ("Hubei", "Hunan"):
        return "Middle Reaches"
    raise ValueError(f"Unknown province in city label: {city}")


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
score_frames = []
weight_rows = []

for sheet_name, spec in SHEETS.items():
    df = pd.read_excel(DATA_FILE, sheet_name=sheet_name)
    indicator_columns = list(df.columns[2:])

    if indicator_columns != spec["columns"]:
        raise ValueError(
            f"Column order in sheet '{sheet_name}' does not match the "
            "configuration. Expected:\n"
            f"  {spec['columns']}\nFound:\n  {indicator_columns}\n"
            "Update SHEETS in main_analysis.py if the indicator set has changed."
        )

    indicator_types = spec["types"]
    if len(indicator_types) != len(indicator_columns):
        raise ValueError("The number of indicator attributes does not match "
                         "the number of indicator columns.")

    data = df.iloc[:, 2:].to_numpy(dtype=float)
    if np.isnan(data).any():
        raise ValueError(f"Missing values detected in sheet '{sheet_name}'.")

    # --- Weights (entropy + CRITIC + uninorm) -----------------------------
    w_entropy = method.entropy(data)
    w_critic = method.critic(data)
    if np.isnan(w_entropy).any() or np.isnan(w_critic).any():
        raise ValueError("NaN values in the computed weights.")

    e = 1 / len(w_critic)
    w = []
    for a, b in zip(w_critic, w_entropy):
        if 0 < a < e and 0 < b < e:
            weight = 2 * e * a * b / (a * b + (2 * e - a) * (2 * e - b))
        elif e < a < 1 and e < b < 1:
            weight = e + (1 - e) * (
                ((a - 2 * e + 1) * (b - 2 * e + 1) - (1 - a) * (1 - b))
                / ((a - 2 * e + 1) * (b - 2 * e + 1) + (1 - a) * (1 - b))
            )
        else:
            weight = max(a, b)
        w.append(weight)
    w = np.asarray(w)
    w = w / w.sum()

    for name, weight in zip(indicator_columns, w):
        weight_rows.append({"Subsystem": spec["label"], "Indicator": name,
                            "Weight": float(weight)})

    # --- Extreme-value thresholds -----------------------------------------
    data_norm = data.copy()
    cost_mask = np.array(indicator_types) == "cost"
    data_norm[:, cost_mask] *= -1

    set_max = method.max_c(data_norm)
    set_min = -method.min_c(data_norm)

    prob_yes = {}
    prob_no = {}
    for j, indicator in enumerate(indicator_columns):
        params_max = method.fit_gev(set_max[j, :])
        params_min = method.fit_gev(set_min[j, :])

        if SAVE_DIAGNOSTICS:
            method.plot_gev_diagnostics(set_max[j, :], params_max, j, "max",
                                        str(FIGURE_DIR))
            method.plot_gev_diagnostics(set_min[j, :], params_min, j, "min",
                                        str(FIGURE_DIR))

        column = data_norm[:, j]
        difference = column[:, np.newaxis] - column
        prob_yes[j] = genextreme(c=params_max[0], loc=params_max[1],
                                 scale=params_max[2]).cdf(difference)

        difference_min = -np.abs(column[:, np.newaxis] - column)
        prob_no[j] = genextreme(c=params_min[0], loc=params_min[1],
                                scale=params_min[2]).cdf(difference_min)

    n_eval = data.shape[0]
    prob_yes_weighted = np.zeros((n_eval, n_eval))
    prob_no_weighted = np.zeros((n_eval, n_eval))
    for j in range(len(indicator_columns)):
        prob_yes_weighted += prob_yes[j] * w[j]
        prob_no_weighted += prob_no[j] * w[j]

    # --- Composite evaluation --------------------------------------------
    score = prob_yes_weighted.copy()
    discount = prob_no_weighted > prob_yes_weighted
    score = np.where(discount, score * (1 - prob_no_weighted), score)
    np.fill_diagonal(score, 0.0)

    score_pos = score.sum(axis=1)
    score_neg = score.sum(axis=0)
    composite = score_pos / (score_pos + score_neg)

    panel = df.iloc[:, :2].copy()
    panel.columns = ["City", "Year"]
    panel[spec["label"]] = composite
    score_frames.append(panel)

# Merge the two subsystems on city and year.
scores = score_frames[0].merge(score_frames[1], on=["City", "Year"], how="inner")

# --- Coupling coordination degree -----------------------------------------
ccd_rows = []
for i in range(len(scores)):
    c, t, d = method.ccd([
        scores["Ecological security"].iloc[i],
        scores["Food security"].iloc[i],
    ])
    ccd_rows.append((c, t, d))

ccd_df = pd.DataFrame(ccd_rows, columns=["C", "T", "D"])
result = pd.concat([scores.reset_index(drop=True), ccd_df], axis=1)
result["Region"] = result["City"].map(region_of)

# --- Write outputs ---------------------------------------------------------
output_xlsx = OUTPUT_DIR / "score_ccd.xlsx"
weights_df = pd.DataFrame(weight_rows)
with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
    result.to_excel(writer, sheet_name="Score_CCD", index=False)
    weights_df.to_excel(writer, sheet_name="Weight", index=False)

csv_path = OUTPUT_DIR / "ccd_panel.csv"
result[["City", "Year", "Region", "C", "T", "D"]].to_csv(
    csv_path, index=False, encoding="utf-8-sig")

# --- Verification summary -------------------------------------------------
print(f"Wrote {output_xlsx}")
print(f"Wrote {csv_path}")
print(f"Observations: {len(result)}")
for year in (2010, 2023):
    subset = result[result["Year"] == year]
    print(
        f"{year}: ecological = {subset['Ecological security'].mean():.3f}, "
        f"food = {subset['Food security'].mean():.3f}, "
        f"CCD = {subset['D'].mean():.3f}"
    )
print("Benchmarks reported in the manuscript: ecological 0.36 -> 0.59; "
      "CCD 0.50 -> 0.71.")
