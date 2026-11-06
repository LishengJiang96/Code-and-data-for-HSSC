"""Core routines for the food and ecological security assessment.

This module implements the components that are used by ``main_analysis.py``:

* entropy weighting and CRITIC weighting;
* the extreme-value thresholds used by the outranking aggregation;
* the improved coupling coordination degree (CCD) model.

The functions deliberately follow the equations in the manuscript. They are
kept close to their original form so that readers can match each block of code
to a numbered equation.
"""

import os

import numpy as np
from scipy.stats import genextreme

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def entropy(x):
    """Entropy weight of each column of ``x`` (rows = observations).

    The evaluations are min-max standardised, converted to column shares, and
    then used to compute the entropy weight.  The revised form (Eq. 3 of the manuscript) adds 1 / n_indicators before
    renormalising, which prevents extremely small weights.
    """
    n, m = x.shape

    # Min-max standardisation (Eq. 1 of the manuscript).
    p = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

    # Column shares.
    p = p / p.sum(axis=0, keepdims=True)

    # Entropy of each indicator (Eq. 2).
    k = 1 / np.log(n)
    entropy_j = -k * np.sum(p * np.log(p + 1 * 10 ** (-10)), axis=0)

    # Divergence coefficient and entropy weight.
    d = 1 - entropy_j
    w = d / d.sum()

    # Revised entropy weight (Eq. 3).
    w = (w + 1 / m) / np.sum(w + 1 / m)
    return w


def critic(x):
    """CRITIC weight of each column of ``x`` (rows = observations).

    The conflict intensity is measured as the column standard deviation
    multiplied by the sum of (1 - correlation) against the other indicators.
    The revised form (Eq. 5 of the manuscript) adds 1 / n_indicators before renormalising.
    """
    n, m = x.shape

    # Min-max standardisation.
    x_norm = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

    # Standard deviation of each standardised indicator.
    std_j = x_norm.std(axis=0, ddof=0)

    # Correlation matrix (Eq. 4), converted to conflict intensity.
    R = 1 - np.corrcoef(x_norm, rowvar=False)

    # Information content and CRITIC weight (defined after Eq. 4).
    C = std_j * R.sum(axis=0)
    w = C / C.sum()

    # Revised CRITIC weight (Eq. 5).
    w = (w + 1 / m) / np.sum(w + 1 / m)
    return w


def min_c(x):
    """Smallest non-zero absolute gap between each evaluation and the others.

    Returns an array of shape (n_indicators, n_observations).  Each row is the
    set of minimum evaluation gaps ``SG_min`` for one indicator, which is used
    to fit the generalised extreme value distribution of the lower tail.
    """
    num_a, num_c = x.shape
    para_min = np.zeros((num_c, num_a))
    for col in range(num_c):
        arr = x[:, col]
        results = []
        for i in range(num_a):
            current = arr[i]
            others = np.delete(arr, i)
            differences = np.abs(current - others)
            # Drop exact ties; the smallest non-zero gap is the threshold.
            positive = differences[differences != 0]
            results.append(np.min(positive) if positive.size else 0.0)
        para_min[col, :] = results
    return para_min


def max_c(x):
    """Largest signed gap between each evaluation and the others.

    Returns an array of shape (n_indicators, n_observations).  Each row is the
    set of maximum evaluation gaps ``SG_max`` for one indicator, which is used
    to fit the generalised extreme value distribution of the upper tail.
    """
    num_a, num_c = x.shape
    para_max = np.zeros((num_c, num_a))
    for col in range(num_c):
        arr = x[:, col]
        results = []
        for i in range(num_a):
            current = arr[i]
            others = np.delete(arr, i)
            results.append(np.max(current - others))
        para_max[col, :] = results
    return para_max


def fit_gev(values):
    """Maximum-likelihood fit of a GEV distribution (Eq. 8)."""
    return genextreme.fit(values)


def plot_gev_diagnostics(values, params, number, kind, outdir="figures"):
    """Optional diagnostic plots: empirical vs fitted CDF and PDF.

    These plots are not used in the manuscript; they are provided so that the
    tail fits can be inspected.  Set ``SAVE_DIAGNOSTICS = False`` in
    ``main_analysis.py`` to skip them.
    """
    os.makedirs(outdir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False

    # Empirical vs fitted CDF.
    plt.figure(figsize=(5, 5))
    sorted_data = np.sort(values)
    ecdf = np.arange(1, len(values) + 1) / len(values)
    plt.step(sorted_data, ecdf, where="post",
             label="Empirical cumulative distribution function", linewidth=2)
    x = np.linspace(min(values) * 0.95, max(values) * 1.05, 100)
    plt.plot(x, genextreme.cdf(x, *params), "r--",
             label="Fitted cumulative distribution function", linewidth=3)
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(genextreme.ppf(0.5, *params), color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Evaluation difference", fontsize=18)
    plt.ylabel("Cumulative probability", fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{kind}_{number}_cdf.png"), dpi=600,
                bbox_inches="tight")
    plt.close()

    # Empirical vs fitted PDF.
    plt.figure(figsize=(5, 5))
    plt.hist(values, density=True, alpha=0.6, color="skyblue",
             edgecolor="black", label="Data histogram")
    x_pdf = np.linspace(min(values) * 0.9, max(values) * 1.1, 200)
    plt.plot(x_pdf, genextreme.pdf(x_pdf, *params), "r--", linewidth=3,
             label="Fitted probability distribution function")
    plt.xlabel("Evaluation difference", fontsize=18)
    plt.ylabel("Probability density", fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{kind}_{number}_pdf.png"), dpi=600,
                bbox_inches="tight")
    plt.close()


def ccd(values):
    """Improved coupling coordination degree for two subsystems.

    ``values`` is a sequence of subsystem evaluations, each in [0, 1].
    Returns (C, T, D) as defined by Eqs. (11)-(14) of the manuscript.
    """
    values = np.asarray(values, dtype=float)
    noa = len(values)
    w = np.ones((noa, 1)) / noa

    # Comprehensive development index T (Eq. 12).
    T = 0
    for j in range(noa):
        T += w[j] * values[j]

    # Coupling degree C (Eq. 11): dispersion term x product term.
    c1 = 0
    c2 = 1
    max_value = np.max(values)
    for i1 in range(noa):
        c2 *= values[i1] / max_value
        for i2 in range(i1):
            c1 += abs(values[i1] - values[i2])

    m = 0
    for j in range(noa - 1):
        m += j + 1

    c = ((1 - c1 / m) * (c2) ** (1 / (noa - 1))) ** 0.5

    # Coupling coordination degree D (Eq. 14).
    d = (c * T) ** 0.5
    return c, T[0], d[0]
