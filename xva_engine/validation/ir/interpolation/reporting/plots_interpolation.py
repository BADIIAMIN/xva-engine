from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_time_series_bands(median: np.ndarray, p95: np.ndarray, title: str, outpath: Path) -> None:
    plt.figure()
    x = np.arange(median.size)
    plt.plot(x, median, label="median")
    plt.plot(x, p95, label="p95")
    plt.title(title)
    plt.xlabel("time index")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_histogram(samples: np.ndarray, title: str, outpath: Path, bins: int = 60) -> None:
    plt.figure()
    plt.hist(samples, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_finding2_plots(
    out_dir: str,
    sens: dict,
    rough: dict,
    density: dict,
    prefix: str,
) -> None:
    out = _ensure_dir(Path(out_dir))

    plot_time_series_bands(
        sens["rms_time_med"], sens["rms_time_p95"],
        f"{prefix} interpolation sensitivity (RMS diff): median / p95",
        out / f"{prefix}_interp_sensitivity_rms_bands.png",
    )
    plot_time_series_bands(
        sens["maxabs_time_p95"], sens["maxabs_time_p95"],
        f"{prefix} interpolation sensitivity (max abs diff): p95",
        out / f"{prefix}_interp_sensitivity_maxabs_p95.png",
    )

    # Roughness bands
    rough_lin = rough["rough_lin"]
    rough_logdf = rough["rough_logdf"]
    plot_time_series_bands(
        np.median(rough_lin, axis=0), np.quantile(rough_lin, 0.95, axis=0),
        f"{prefix} forward roughness (linear zero): median / p95",
        out / f"{prefix}_forward_roughness_linear_bands.png",
    )
    plot_time_series_bands(
        np.median(rough_logdf, axis=0), np.quantile(rough_logdf, 0.95, axis=0),
        f"{prefix} forward roughness (logDF): median / p95",
        out / f"{prefix}_forward_roughness_logdf_bands.png",
    )

    # Density stress bands
    plot_time_series_bands(
        density["rms_time_med"], density["rms_time_p95"],
        f"{prefix} pillar density stress (RMS diff): median / p95",
        out / f"{prefix}_pillar_density_rms_bands.png",
    )

    # One representative histogram at final time index
    t_last = sens["rms_pt"].shape[1] - 1
    plot_histogram(
        sens["rms_pt"][:, t_last],
        f"{prefix} interp sensitivity RMS at final time",
        out / f"{prefix}_interp_sensitivity_rms_hist_tlast.png",
    )
