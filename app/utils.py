import math
import numpy as np
import pandas as pd

CLIMATE_14 = [
    "temperature_mean_c", "temperature_max_c", "temperature_min_c",
    "rainfall_mm_year", "precip_hours_day", "humidity_pct",
    "sunshine_hrs_day", "shortwave_radiation", "et0_mm_day",
    "vapour_pressure_def", "wind_speed", "soil_moisture_0_7cm",
    "soil_moisture_7_28cm", "soil_temperature",
]


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_kabupaten(lat: float, lon: float, kab_df: pd.DataFrame) -> pd.Series:
    """Return the row of the nearest kabupaten to (lat, lon)."""
    dists = kab_df.apply(
        lambda r: haversine(lat, lon, r["centroid_lat"], r["centroid_lon"]), axis=1
    )
    return kab_df.loc[dists.idxmin()]


def compute_anomalies(
    climate_vals: dict,
    climate_stats: dict,
) -> dict:
    """
    Compute climate anomaly z-scores.
    climate_stats: {col: {"mean": float, "std": float}} for each CLIMATE_14 col.
    Returns dict of {col_anomaly: z_score}.
    When using kabupaten averages (Option 1), all anomalies are 0 by definition.
    """
    anomalies = {}
    for col in CLIMATE_14:
        stats = climate_stats.get(col, {"mean": climate_vals[col], "std": 1.0})
        mean = stats["mean"]
        std  = stats["std"] if stats["std"] > 0 else 1.0
        anomalies[f"{col}_anomaly"] = (climate_vals[col] - mean) / std
    return anomalies


def viability_label(prob: float, low_thresh: float, high_thresh: float) -> tuple[str, str]:
    """Map predicted probability to (label, colour)."""
    if prob >= 0.70:
        return "Great", "green"
    elif prob >= 0.40:
        return "Moderate", "orange"
    else:
        return "Bad", "red"


def estimated_yield(prob: float, low_thresh: float, high_thresh: float) -> float:
    """
    Yield estimate (ton/ha) consistent with viability label boundaries.
      Bad      (prob < 0.40):  yield < low_thresh
      Moderate (0.40–0.70):   low_thresh ≤ yield < high_thresh
      Great    (prob ≥ 0.70):  yield ≥ high_thresh
    Java padi floor ≈ 3.5 ton/ha, ceiling ≈ 7.5 ton/ha.
    """
    MIN_JAVA, MAX_JAVA = 3.5, 7.5
    if prob >= 0.70:
        t = (prob - 0.70) / 0.30
        return round(high_thresh + t * (MAX_JAVA - high_thresh), 2)
    elif prob >= 0.40:
        t = (prob - 0.40) / 0.30
        return round(low_thresh + t * (high_thresh - low_thresh), 2)
    else:
        t = prob / 0.40
        return round(MIN_JAVA + t * (low_thresh - MIN_JAVA), 2)


def economic_estimate(yield_ton_ha: float, area_ha: float = 1.0) -> float:
    """Revenue estimate in IDR. HPP GKP Rp 6,500/kg (Kepbadan No. 14/2025)."""
    hpp_per_kg = 6500
    kg = yield_ton_ha * 1000 * area_ha
    return kg * hpp_per_kg


def build_feature_vector(
    climate_vals: dict,
    kab_stats: dict,
    yield_lag1: float,
    yield_lag2: float,
    anomaly_vals: dict,
    feature_order: list[str],
) -> pd.DataFrame:
    """Assemble a 35-feature DataFrame in the model's expected column order."""
    row = {
        **climate_vals,
        **kab_stats,
        "yield_lag1": yield_lag1,
        "yield_lag2": yield_lag2,
        **anomaly_vals,
    }
    return pd.DataFrame([[row[f] for f in feature_order]], columns=feature_order)
