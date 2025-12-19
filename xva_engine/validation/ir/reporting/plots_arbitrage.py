from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_df_monotonicity_heatmap(freq_time_pillar: np.ndarray, title: str):
    fig, ax = plt.subplots()
    im = ax.imshow(freq_time_pillar.T, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("time index")
    ax.set_ylabel("pillar interval index (k→k+1)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("violation frequency")
    fig.tight_layout()
    return fig


def plot_kink_index_bands(kink: np.ndarray, time_grid_years: np.ndarray, title: str):
    q05 = np.quantile(kink, 0.05, axis=0)
    q50 = np.quantile(kink, 0.50, axis=0)
    q95 = np.quantile(kink, 0.95, axis=0)

    fig, ax = plt.subplots()
    ax.plot(time_grid_years, q50)
    ax.fill_between(time_grid_years, q05, q95, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("time (years)")
    ax.set_ylabel("kink index")
    fig.tight_layout()
    return fig


def plot_wedge_histogram(wedge: np.ndarray, title: str, bins: int = 60):
    fig, ax = plt.subplots()
    ax.hist(wedge, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("wedge")
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def plot_wedge_vs_maturity(maturities_years: np.ndarray, stat: np.ndarray, title: str):
    fig, ax = plt.subplots()
    ax.plot(maturities_years, stat)
    ax.set_title(title)
    ax.set_xlabel("maturity (years)")
    ax.set_ylabel("p95(|wedge|)")
    fig.tight_layout()
    return fig
