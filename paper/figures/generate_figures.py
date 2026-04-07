#!/usr/bin/env python3
"""
Generate publication figures for the NovoExpert-2 paper.

Produces:
  - fig_benchmark_deltas.pdf: Δ vs SOTA bar chart for all 22 endpoints
  - fig_cyp2d6_seeds.pdf: Per-seed CYP2D6 scores with SOTA line
  - fig_dili_method_comparison.pdf: CatBoost vs Chemprop on DILI

Usage:
    python generate_figures.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Clean publication style
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.05

COLOR_SOTA = '#2a9d8f'    # teal — beats SOTA
COLOR_BELOW = '#d8d8d8'   # grey — below SOTA
COLOR_DILI = '#e76f51'    # orange — Chemprop win
COLOR_CB = '#264653'      # dark blue — CatBoost
COLOR_CP = '#e9c46a'      # yellow — Chemprop
COLOR_SOTA_LINE = '#e76f51'

HERE = Path(__file__).parent
RESULTS_DIR = HERE.parent.parent / "results" / "json_reports"

# Full endpoint data (endpoint, category, metric, NovoExpert-2 score, SOTA, method)
# Ordered by category, then by delta (best first within category)
ENDPOINT_DATA = [
    # (endpoint, label, metric, score, sota, is_mae_lower_better, method)
    # Metabolism (top — 3 wins)
    ("cyp2d6_veith",                     "CYP2D6 Veith",                "AUPRC",    0.7776, 0.750, False, "CB+MG"),
    ("cyp3a4_veith",                     "CYP3A4 Veith",                "AUPRC",    0.9163, 0.900, False, "CB+MG"),
    ("cyp3a4_substrate_carbonmangels",   "CYP3A4 Sub Carbonmangels",    "AUROC",    0.6600, 0.656, False, "CB+MG"),
    ("cyp2d6_substrate_carbonmangels",   "CYP2D6 Sub Carbonmangels",    "AUPRC",    0.7163, 0.736, False, "CB+MG"),
    ("cyp2c9_veith",                     "CYP2C9 Veith",                "AUPRC",    0.8612, 0.900, False, "CB+MG"),
    ("cyp2c9_substrate_carbonmangels",   "CYP2C9 Sub Carbonmangels",    "AUPRC",    0.3749, 0.441, False, "CB+MG"),
    # Excretion (1 win)
    ("clearance_hepatocyte_az",          "Clearance Hepatocyte AZ",     "Spearman", 0.5112, 0.487, False, "CB+MG"),
    ("clearance_microsome_az",           "Clearance Microsome AZ",      "Spearman", 0.6196, 0.630, False, "CB+MG"),
    ("half_life_obach",                  "Half Life Obach",             "Spearman", 0.4268, 0.562, False, "CB+MG"),
    # Toxicity (1 win — DILI with Chemprop)
    ("dili",                             "DILI",                        "AUROC",    0.9217, 0.916, False, "CP"),
    ("ames",                             "AMES",                        "AUROC",    0.8621, 0.871, False, "CB+MG"),
    ("herg",                             "hERG",                        "AUROC",    0.8596, 0.880, False, "CB+MG"),
    ("ld50_zhu",                         "LD50 Zhu",                    "MAE",      0.618,  0.573, True,  "CB+MG"),
    # Distribution
    ("bbb_martins",                      "BBB Martins",                 "AUROC",    0.9107, 0.916, False, "CB+MG"),
    ("ppbr_az",                          "PPBR AZ",                     "MAE",      7.570,  7.526, True,  "CB+MG"),
    ("vdss_lombardo",                    "VDss Lombardo",               "Spearman", 0.5557, 0.713, False, "CB+MG"),
    # Absorption
    ("hia_hou",                          "HIA Hou",                     "AUROC",    0.9827, 0.989, False, "CB+MG"),
    ("pgp_broccatelli",                  "Pgp Broccatelli",             "AUROC",    0.9283, 0.940, False, "CB+MG"),
    ("bioavailability_ma",               "Bioavailability Ma",          "AUROC",    0.6508, 0.748, False, "CB+MG"),
    ("caco2_wang",                       "Caco2 Wang",                  "MAE",      0.307,  0.276, True,  "CB+MG"),
    ("lipophilicity_astrazeneca",        "Lipophilicity AZ",            "MAE",      0.5019, 0.467, True,  "CB+MG"),
    ("solubility_aqsoldb",               "Solubility AqSolDB",          "MAE",      0.8045, 0.761, True,  "CB+MG"),
]


def compute_delta(score, sota, is_mae_lower):
    """Positive delta = improvement. For MAE (lower is better), flip the sign."""
    if is_mae_lower:
        return sota - score  # lower score is better, so sota-score positive = improvement
    return score - sota


def fig_benchmark_deltas():
    """Horizontal bar chart showing Δ vs SOTA for all 22 endpoints, sorted."""
    data = [(name, compute_delta(s, sota, mae), beats_sota(s, sota, mae), method)
            for _, name, _, s, sota, mae, method in ENDPOINT_DATA]

    # Sort by delta descending
    data.sort(key=lambda x: -x[1])
    names, deltas, beats, methods = zip(*data)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    y_pos = np.arange(len(names))

    colors = []
    for beat, method in zip(beats, methods):
        if beat:
            colors.append(COLOR_DILI if method == "CP" else COLOR_SOTA)
        else:
            colors.append(COLOR_BELOW)

    bars = ax.barh(y_pos, deltas, color=colors, edgecolor='black', linewidth=0.5)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(r'$\Delta$ vs Published SOTA (positive = improvement)')
    ax.set_title('NovoExpert-2: Per-Endpoint Performance vs TDC Leaderboard SOTA',
                 pad=10, weight='bold')

    # Annotate bars
    for i, (bar, delta, method) in enumerate(zip(bars, deltas, methods)):
        x = bar.get_width()
        label = f"{delta:+.3f}" + (f" [{method}]" if beats[i] else "")
        ha = 'left' if delta >= 0 else 'right'
        x_offset = 0.001 if delta >= 0 else -0.001
        ax.text(x + x_offset, i, label, va='center', ha=ha, fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_SOTA, edgecolor='black', label='Beats SOTA (CatBoost+MapLight+GIN)'),
        Patch(facecolor=COLOR_DILI, edgecolor='black', label='Beats SOTA (Chemprop v2)'),
        Patch(facecolor=COLOR_BELOW, edgecolor='black', label='Below SOTA'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95)

    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.savefig(HERE / 'fig_benchmark_deltas.pdf')
    plt.savefig(HERE / 'fig_benchmark_deltas.png')
    plt.close()
    print(f"Saved fig_benchmark_deltas.pdf / .png")


def beats_sota(score, sota, is_mae_lower):
    return (score < sota) if is_mae_lower else (score > sota)


def fig_cyp2d6_seeds():
    """Scatter/bar of per-seed CYP2D6 AUPRC with SOTA reference line."""
    with open(RESULTS_DIR / "cyp2d6_veith.json") as f:
        data = json.load(f)

    seeds = list(range(5))
    scores = data["seed_scores"]
    ensemble = data["ensemble_score"]
    sota = data["published_sota"]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Per-seed bars
    bars = ax.bar(seeds, scores, color=COLOR_SOTA, edgecolor='black', linewidth=0.8,
                  width=0.6, label='Per-seed AUPRC')

    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.002,
                f"{score:.4f}", ha='center', va='bottom', fontsize=9)

    # Ensemble line
    ax.axhline(ensemble, color=COLOR_CB, linewidth=2, linestyle='-',
               label=f'5-seed ensemble: {ensemble:.4f}')

    # SOTA line
    ax.axhline(sota, color=COLOR_SOTA_LINE, linewidth=2, linestyle='--',
               label=f'Published SOTA: {sota:.3f}')

    ax.set_xlabel('Seed')
    ax.set_ylabel('AUPRC')
    ax.set_title('CYP2D6 Veith: Per-Seed and Ensemble Performance', weight='bold')
    ax.set_xticks(seeds)
    ax.set_ylim(0.74, 0.79)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.savefig(HERE / 'fig_cyp2d6_seeds.pdf')
    plt.savefig(HERE / 'fig_cyp2d6_seeds.png')
    plt.close()
    print(f"Saved fig_cyp2d6_seeds.pdf / .png")


def fig_dili_comparison():
    """Method comparison for DILI: CatBoost vs Chemprop per-seed and ensemble."""
    # Per-seed AUROC from close-calls v4 run (verified from logs)
    cb_seeds = [0.9013, 0.9004, 0.8817, 0.8774, 0.8770]
    cp_seeds = [0.9183, 0.9248, 0.9161, 0.8883, 0.8878]

    # Ensemble scores
    cb_ensemble = 0.9057
    cp_ensemble = 0.9217
    sota = 0.916

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(5)
    width = 0.35

    ax.bar(x - width / 2, cb_seeds, width, color=COLOR_CB, edgecolor='black',
           linewidth=0.5, label='CatBoost+MapLight+GIN (per seed)')
    ax.bar(x + width / 2, cp_seeds, width, color=COLOR_CP, edgecolor='black',
           linewidth=0.5, label='Chemprop v2 (per seed)')

    # Ensemble lines
    ax.axhline(cb_ensemble, color=COLOR_CB, linewidth=2, linestyle='-',
               label=f'CatBoost ensemble: {cb_ensemble:.4f}')
    ax.axhline(cp_ensemble, color=COLOR_CP, linewidth=2, linestyle='-',
               label=f'Chemprop ensemble: {cp_ensemble:.4f}')
    ax.axhline(sota, color=COLOR_SOTA_LINE, linewidth=2, linestyle='--',
               label=f'Published SOTA: {sota:.3f}')

    ax.set_xlabel('Seed')
    ax.set_ylabel('AUROC')
    ax.set_title('DILI: CatBoost+MapLight+GIN vs Chemprop v2', weight='bold')
    ax.set_xticks(x)
    ax.set_ylim(0.86, 0.935)
    ax.legend(loc='lower right', framealpha=0.95, fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.savefig(HERE / 'fig_dili_comparison.pdf')
    plt.savefig(HERE / 'fig_dili_comparison.png')
    plt.close()
    print(f"Saved fig_dili_comparison.pdf / .png")


if __name__ == "__main__":
    fig_benchmark_deltas()
    fig_cyp2d6_seeds()
    fig_dili_comparison()
    print("\nAll figures generated.")
