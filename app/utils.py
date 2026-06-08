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


def rank_kabupaten(model, meta: dict, kab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict padi viability for every kabupaten using its historical climate
    averages (Option 1: anomalies = 0 by definition, lags = kab_mean).

    Returns a DataFrame sorted by descending viability probability with columns:
    kabupaten, province, centroid_lat, centroid_lon, prob, label, colour,
    est_yield, revenue.
    """
    kab_stats_map = meta["kab_stats_map"]
    feature_order = meta["features"]
    low, high     = meta["low_threshold"], meta["high_threshold"]

    rows = []
    for _, r in kab_df.iterrows():
        ks       = kab_stats_map.get(r["kabupaten"], {})
        kab_mean = ks.get("kab_mean", meta["threshold"])
        row = {col: r[col] for col in CLIMATE_14}
        row.update({
            "kab_mean":   kab_mean,
            "kab_std":    ks.get("kab_std",    0.3),
            "kab_median": ks.get("kab_median", meta["threshold"]),
            "kab_min":    ks.get("kab_min",    low),
            "kab_max":    ks.get("kab_max",    high),
            "yield_lag1": kab_mean,
            "yield_lag2": kab_mean,
        })
        for col in CLIMATE_14:
            row[f"{col}_anomaly"] = 0.0
        rows.append(row)

    X = pd.DataFrame([[r[f] for f in feature_order] for r in rows], columns=feature_order)
    probs = model.predict_proba(X)[:, 1]

    out = kab_df[["kabupaten", "province", "centroid_lat", "centroid_lon"]].copy()
    out["prob"]      = probs
    labels_colours   = [viability_label(p, low, high) for p in probs]
    out["label"]     = [lc[0] for lc in labels_colours]
    out["colour"]    = [lc[1] for lc in labels_colours]
    out["est_yield"] = [estimated_yield(p, low, high) for p in probs]
    out["revenue"]   = [economic_estimate(y) for y in out["est_yield"]]
    return out.sort_values("prob", ascending=False).reset_index(drop=True)
