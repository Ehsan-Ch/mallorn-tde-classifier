"""Label-free, per-object features for sparse, irregular six-band light curves.

No fitting across objects, extrapolation across seasons, or use of SpecType.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import extinction
import numpy as np
import pandas as pd
from scipy.signal import lombscargle
from scipy.stats import skew

from .data import BANDS, validate_photometry

# Representative wavelengths, Angstrom. Monochromatic approximation, not full
# throughput/SED integration; retained in the manifest for reproducibility.
WAVELENGTHS = dict(zip(BANDS, [3670., 4825., 6222., 7545., 8691., 9710.]))
PHASE_EDGES = [-np.inf, -60, -30, -10, 0, 10, 30, 60, 120, np.inf]
COLOR_PHASES = [-10, 0, 15, 30, 60]


@dataclass(frozen=True)
class FeatureConfig:
    deredden: bool = True
    max_color_gap_days: float = 60.0
    min_color_snr: float = 2.0

    def metadata(self) -> dict:
        return {**asdict(self), "wavelengths_angstrom": WAVELENGTHS,
                "extinction_law": "Fitzpatrick99; Rv=3.1; monochromatic approximation",
                "redshift_policy": "Use supplied Z; exclude Z_err due train/test missingness shift"}


def dust_scale(ebv: float, band: str) -> float:
    a_lambda = extinction.fitzpatrick99(np.array([WAVELENGTHS[band]], dtype=float), 3.1 * ebv, 3.1)[0]
    scale = float(10 ** (0.4 * a_lambda))
    if not np.isfinite(scale):
        raise ValueError("Extinction correction overflow")
    return scale


def safe_skew(values: np.ndarray) -> float:
    return float(skew(values, bias=False)) if len(values) >= 3 and np.std(values) > 1e-12 else np.nan


def interpolate_observed(t, f, e, query, max_gap, min_snr):
    """Linear interpolation only between confident, nearby detections."""
    if not len(t):
        return np.nan
    j = int(np.searchsorted(t, query))
    if j < len(t) and t[j] == query:
        return float(f[j]) if f[j] / e[j] >= min_snr else np.nan
    if j == 0 or j == len(t) or t[j] - t[j - 1] > max_gap:
        return np.nan
    if min(f[j] / e[j], f[j - 1] / e[j - 1]) < min_snr:
        return np.nan
    return float(np.interp(query, t[j-1:j+1], f[j-1:j+1]))


def band_features(t, f, e, peak, z) -> dict:
    base_names = ["n", "span", "max_gap", "mean", "std", "min", "max", "q10", "q30", "q50", "q90",
                  "skew", "weighted_mean", "chi2_dof", "snr_max", "snr_mean", "negative_fraction",
                  "detected_fraction", "width_snr5", "peak_lag", "q90_q50_scale", "q100_q50_scale",
                  "decay_slope", "decay_r2", "period_rest_days", "period_power"]
    out = {name: np.nan for name in base_names}
    for i in range(len(PHASE_EDGES)-1):
        out.update({f"phase{i}_mean_norm": np.nan, f"phase{i}_snr_mean": np.nan, f"phase{i}_n": 0.})
    out["n"] = float(len(t))
    if not len(t):
        return out
    rest_t = (t - peak) / (1 + z)
    snr = f / e
    scale = max(float(np.max(np.abs(f))), 1e-12)
    weights = 1 / e**2
    weighted_mean = float(np.average(f, weights=weights))
    q10, q30, q50, q90 = np.quantile(f, [.1, .3, .5, .9])
    detected = snr > 5
    out.update(n=float(len(t)), span=float(np.ptp(rest_t)),
               max_gap=float(np.max(np.diff(rest_t))) if len(t) > 1 else np.nan,
               mean=float(np.mean(f)), std=float(np.std(f)), min=float(min(f)), max=float(max(f)),
               q10=q10, q30=q30, q50=q50, q90=q90, skew=safe_skew(f), weighted_mean=weighted_mean,
               chi2_dof=float(np.sum(weights*(f-weighted_mean)**2)/(len(t)-1)) if len(t)>1 else np.nan,
               snr_max=float(max(snr)), snr_mean=float(np.mean(snr)), negative_fraction=float(np.mean(f<0)),
               detected_fraction=float(np.mean(detected)),
               width_snr5=float(np.ptp(rest_t[detected])) if detected.sum()>=2 else np.nan,
               peak_lag=float(rest_t[np.argmax(f)]), q90_q50_scale=float((q90-q50)/scale),
               q100_q50_scale=float((max(f)-q50)/scale))
    for i, (left, right) in enumerate(zip(PHASE_EDGES[:-1], PHASE_EDGES[1:])):
        mask = (rest_t >= left) & (rest_t < right)
        out[f"phase{i}_n"] = float(mask.sum())
        if mask.any():
            out[f"phase{i}_mean_norm"] = float(np.mean(f[mask])/scale)
            out[f"phase{i}_snr_mean"] = float(np.mean(snr[mask]))
    # A descriptive log-log decline statistic, not a fitted physical TDE model.
    mask = (rest_t >= 5) & (snr > 2)
    if mask.sum() >= 3 and np.ptp(rest_t[mask]) > 5:
        x, y = np.log(rest_t[mask]), np.log(f[mask])
        coef = np.polyfit(x, y, 1, w=snr[mask])
        residual = np.sum((y - np.polyval(coef, x))**2)
        total = np.sum((y-y.mean())**2)
        out["decay_slope"] = float(coef[0])
        out["decay_r2"] = float(1-residual/total) if total>1e-12 else np.nan
    if len(t) >= 8 and np.ptp(rest_t) > 20 and np.std(f) > 1e-10:
        # Uneven cadence: use weighted Lomb-Scargle, not an FFT of raw samples.
        periods = np.geomspace(5, max(6, np.ptp(rest_t)), 128)
        power = lombscargle(rest_t, f, 2*np.pi/periods, weights=weights,
                           floating_mean=True, normalize=True)
        if np.isfinite(power).any():
            best = int(np.nanargmax(power))
            out["period_rest_days"], out["period_power"] = float(periods[best]), float(power[best])
    return out


def extract_features(log: pd.DataFrame, photo: pd.DataFrame, config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    if log.object_id.duplicated().any():
        raise ValueError("Duplicate metadata IDs")
    photo = validate_photometry(photo)
    if set(photo.object_id) != set(log.object_id):
        raise ValueError("Feature input object sets must match exactly")
    rows = {}
    metadata = log.set_index("object_id")
    for object_id, lc in photo.groupby("object_id", sort=True):
        row = metadata.loc[object_id]
        z, ebv = float(row.Z), float(row.EBV)
        if not np.isfinite([z, ebv]).all() or min(z, ebv) < 0:
            raise ValueError("Invalid redshift/extinction")
        bands = {}
        for band in BANDS:
            g = lc.loc[lc.Filter.eq(band)].sort_values("Time (MJD)")
            factor = dust_scale(ebv, band) if config.deredden else 1.
            bands[band] = (g["Time (MJD)"].to_numpy(), g.Flux.to_numpy()*factor, g.Flux_err.to_numpy()*factor)
        # Reference peak from highest-SNR observation in g/r/i, or all bands if absent.
        reference = lc[lc.Filter.isin(list("gri"))]
        if reference.empty:
            reference = lc
        peak = float(reference.loc[(reference.Flux/reference.Flux_err).idxmax(), "Time (MJD)"])
        out = {"Z": z, "EBV": ebv, "log1p_Z": float(np.log1p(z)),
               "total_observations": float(len(lc)), "bands_observed": float(lc.Filter.nunique())}
        detected_t = lc.loc[(lc.Flux/lc.Flux_err)>5, "Time (MJD)"].to_numpy()
        out["all_width_snr5"] = float(np.ptp(detected_t)/(1+z)) if len(detected_t)>=2 else np.nan
        for band, (t, f, e) in bands.items():
            out.update({f"{band}__{key}": value for key, value in band_features(t, f, e, peak, z).items()})
        # Common-phase colors avoid ratios of maxima from different dates.
        for a, b in combinations(BANDS, 2):
            colors = []
            for phase in COLOR_PHASES:
                query = peak + phase * (1+z)
                fluxes = [interpolate_observed(*bands[c], query, config.max_color_gap_days, config.min_color_snr) for c in (a,b)]
                color = float(-2.5*np.log10(fluxes[0]/fluxes[1])) if min(fluxes)>0 and np.isfinite(fluxes).all() else np.nan
                out[f"color_{a}_{b}__p{phase}"] = color
                colors.append(color)
            out[f"color_{a}_{b}__evolution_0_30"] = colors[3]-colors[1]
        rows[object_id] = out
    result = pd.DataFrame.from_dict(rows, orient="index").reindex(log.object_id)
    result.index.name = "object_id"
    result = result.reindex(sorted(result.columns), axis=1).astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Infinite engineered features")
    return result


def basic_columns(columns) -> list[str]:
    return [c for c in columns if not any(term in c for term in ("phase", "decay_", "period_", "color_"))]
