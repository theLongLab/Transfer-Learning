#!/usr/bin/env python
# coding: utf-8


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests


tracks_b=pd.read_excel("SupplementaryTables.xlsx", sheet_name="S1", header=1)
tracks_p=pd.read_excel("SupplementaryTables.xlsx", sheet_name="S2", header=1)
tracks_b_s=tracks_b[['FastQC', 'UniquelyMappedRatio', 'PBC','PeaksFoldChangeAbove10', 'FRiP', 'PeaksUnionDHSRatio','Pearson R between predicted and target (mean value across 1937 test sequences)']]
tracks_p_s=tracks_p[['FastQC', 'UniquelyMappedRatio', 'PBC','PeaksFoldChangeAbove10', 'FRiP', 'PeaksUnionDHSRatio','Pearson R between predicted and target (mean value across 1937 test sequences)']]

col = 'Pearson R between predicted and target (mean value across 1937 test sequences)'
tracks_b_s["cat"] = pd.cut(
    tracks_b_s[col],
    bins=[-np.inf, 0.3, 0.6, np.inf],
    labels=["low", "moderate", "high"]
)

tracks_p_s["cat"] = pd.cut(
    tracks_p_s[col],
    bins=[-np.inf, 0.3, 0.6, np.inf],
    labels=["low", "moderate", "high"]
)

qc_cols = ['FastQC','UniquelyMappedRatio','PBC','PeaksFoldChangeAbove10','FRiP','PeaksUnionDHSRatio']
# Ensure correct order
tracks_b_s["cat"] = pd.Categorical(tracks_b_s["cat"],categories=["low", "moderate", "high"],ordered=True)


tracks_b_s.groupby("cat")["FastQC"].count()
tracks_p_s.groupby("cat")["FastQC"].count()


def plot_qc_boxplots_with_ks(
    tracks_b_s,
    color_list,
    output_tag,
    qc_cols=None,
    figsize=(16, 8),
    palette=None,
    fdr_method="fdr_bh"
):
    """
    Plot QC metric boxplots stratified by performance category
    and add FDR-adjusted KS test p-values in titles.

    Parameters
    ----------
    tracks_b_s : pd.DataFrame
        Must contain column 'cat' with values ['low','moderate','high']
    qc_cols : list
        List of QC metric column names
    figsize : tuple
        Figure size
    palette : dict
        Color dictionary for categories
    fdr_method : str
        Multiple testing correction method (default: fdr_bh)

    Returns
    -------
    ks_table : pd.DataFrame
        DataFrame of FDR-adjusted KS p-values
    """

    if qc_cols is None:
        qc_cols = [
            'FastQC',
            'UniquelyMappedRatio',
            'PBC',
            'PeaksFoldChangeAbove10',
            'FRiP',
            'PeaksUnionDHSRatio'
        ]

    if palette is None:
        palette = {
            "low": color_list[0],
            "moderate": color_list[1],
            "high": color_list[2]
        }

    # Ensure correct category order
    tracks_b_s = tracks_b_s.copy()
    tracks_b_s["cat"] = pd.Categorical(
        tracks_b_s["cat"],
        categories=["low", "moderate", "high"],
        ordered=True
    )

    pairs = [("low", "moderate"), ("low", "high"), ("moderate", "high")]
    pvals = []
    ks_results = {}

    # ---- Compute KS tests ----
    for col in qc_cols:
        ks_results[col] = {}
        for g1, g2 in pairs:
            d1 = tracks_b_s[tracks_b_s["cat"] == g1][col].dropna()
            d2 = tracks_b_s[tracks_b_s["cat"] == g2][col].dropna()

            if len(d1) > 0 and len(d2) > 0:
                _, p = ks_2samp(d1, d2)
            else:
                p = np.nan
            ks_results[col][(g1, g2)] = p

    # ---- Plot ----
    ncols = 3
    nrows = int(np.ceil(len(qc_cols) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, col in enumerate(qc_cols):
        ax = axes[i]

        sns.boxplot(
            data=tracks_b_s,
            x="cat",
            y=col,
            palette=palette,
            ax=ax,
            showfliers=False
        )

        p1 = ks_results[col][("low", "moderate")]
        p2 = ks_results[col][("low", "high")]
        p3 = ks_results[col][("moderate", "high")]

        ax.set_title(
            f"{col}\n"
            f"low–mod: {p1:.0e} | low–high: {p2:.0e}| mod–high: {p3:.0e}",
            fontsize=11
        )

    # Remove unused axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("/data/sbcs/GuoLab/backup/liq17/DTL_revision2/"+output_tag+".pdf", dpi=300)

    # ---- Return results table ----
    ks_table = pd.DataFrame.from_dict(
        {(col, pair): ks_results[col][pair]
         for col in qc_cols for pair in pairs},
        orient="index",
        columns=["P value (Kolmogorov-Smirnov test)"]
    )
    ks_table.index = pd.MultiIndex.from_tuples(
        ks_table.index,
        names=["QC_metric", "Comparison"]
    )


col="FRiP"
pairs = [("low", "high")]
for g1, g2 in pairs:
    d1 = tracks_b_s[tracks_b_s["cat"] == g1][col].dropna()
    d2 = tracks_b_s[tracks_b_s["cat"] == g2][col].dropna()

    if len(d1) > 0 and len(d2) > 0:
        _, p = ks_2samp(d1, d2)
    else:
        p = np.nan
    print(p)


plot_qc_boxplots_with_ks(tracks_b_s, ["#e7aec9","#c67ca2","#a74b86"], "breast")

plot_qc_boxplots_with_ks(tracks_p_s, ["#bbdcbe","#8dc0aa","#5d9d99"], "prostate")
