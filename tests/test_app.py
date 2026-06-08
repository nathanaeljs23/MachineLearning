"""26 app utility tests for app/utils.py."""

import math
import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from utils import (
    CLIMATE_14,
    haversine,
    nearest_kabupaten,
    viability_label,
    estimated_yield,
    economic_estimate,
    build_feature_vector,
    compute_anomalies,
)

KAB_STAT_COLS = ["kab_mean", "kab_std", "kab_median", "kab_min", "kab_max"]
ANOMALY_COLS  = [f"{c}_anomaly" for c in CLIMATE_14]
FEATURES      = CLIMATE_14 + KAB_STAT_COLS + ["yield_lag1", "yield_lag2"] + ANOMALY_COLS


# ── haversine ─────────────────────────────────────────────────────────────────

def test_haversine_same_point():
    assert haversine(0, 0, 0, 0) == pytest.approx(0.0)

def test_haversine_known_distance():
    dist = haversine(-6.2, 106.8, -6.9, 107.6)
    assert 100 < dist < 140

def test_haversine_symmetry():
    d1 = haversine(-6.2, 106.8, -7.5, 110.0)
    d2 = haversine(-7.5, 110.0, -6.2, 106.8)
    assert d1 == pytest.approx(d2)

def test_haversine_non_negative():
    assert haversine(-8.0, 112.0, -7.0, 111.0) >= 0


# ── nearest_kabupaten ─────────────────────────────────────────────────────────

@pytest.fixture
def kab_df():
    return pd.DataFrame({
        "kabupaten":   ["Alpha", "Beta", "Gamma"],
        "centroid_lat": [-6.9, -7.5, -8.2],
        "centroid_lon": [107.6, 110.0, 112.5],
        "province":    ["Jawa Barat", "Jawa Tengah", "Jawa Timur"],
    })

def test_nearest_kabupaten_exact(kab_df):
    assert nearest_kabupaten(-6.9, 107.6, kab_df)["kabupaten"] == "Alpha"

def test_nearest_kabupaten_closest(kab_df):
    assert nearest_kabupaten(-7.4, 109.9, kab_df)["kabupaten"] == "Beta"

def test_nearest_kabupaten_returns_series(kab_df):
    assert isinstance(nearest_kabupaten(-7.0, 108.0, kab_df), pd.Series)

def test_nearest_kabupaten_has_province(kab_df):
    assert "province" in nearest_kabupaten(-8.0, 112.0, kab_df).index


# ── compute_anomalies ─────────────────────────────────────────────────────────

def test_anomaly_zero_at_mean():
    climate_vals = {col: 5.0 for col in CLIMATE_14}
    stats = {col: {"mean": 5.0, "std": 1.0} for col in CLIMATE_14}
    result = compute_anomalies(climate_vals, stats)
    for col in CLIMATE_14:
        assert result[f"{col}_anomaly"] == pytest.approx(0.0)

def test_anomaly_positive_above_mean():
    climate_vals = {col: 6.0 for col in CLIMATE_14}
    stats = {col: {"mean": 5.0, "std": 1.0} for col in CLIMATE_14}
    result = compute_anomalies(climate_vals, stats)
    for col in CLIMATE_14:
        assert result[f"{col}_anomaly"] == pytest.approx(1.0)

def test_anomaly_handles_zero_std():
    climate_vals = {col: 5.0 for col in CLIMATE_14}
    stats = {col: {"mean": 5.0, "std": 0.0} for col in CLIMATE_14}
    result = compute_anomalies(climate_vals, stats)
    # std=0 is treated as 1.0 to avoid division by zero
    for col in CLIMATE_14:
        assert result[f"{col}_anomaly"] == pytest.approx(0.0)


# ── viability_label ───────────────────────────────────────────────────────────

def test_viability_great():
    label, colour = viability_label(0.75, 5.31, 5.78)
    assert label == "Great" and colour == "green"

def test_viability_moderate():
    label, colour = viability_label(0.55, 5.31, 5.78)
    assert label == "Moderate" and colour == "orange"

def test_viability_bad():
    label, colour = viability_label(0.20, 5.31, 5.78)
    assert label == "Bad" and colour == "red"

def test_viability_boundary_great():
    assert viability_label(0.70, 5.31, 5.78)[0] == "Great"

def test_viability_boundary_moderate():
    assert viability_label(0.40, 5.31, 5.78)[0] == "Moderate"

def test_viability_boundary_bad():
    assert viability_label(0.39, 5.31, 5.78)[0] == "Bad"


# ── estimated_yield ───────────────────────────────────────────────────────────

def test_estimated_yield_increases_with_prob():
    y_low  = estimated_yield(0.1, 5.31, 5.78)
    y_high = estimated_yield(0.9, 5.31, 5.78)
    assert y_high > y_low

def test_estimated_yield_positive():
    assert estimated_yield(0.5, 5.31, 5.78) > 0

def test_estimated_yield_type():
    assert isinstance(estimated_yield(0.5, 5.31, 5.78), float)


# ── economic_estimate ─────────────────────────────────────────────────────────

def test_economic_estimate_positive():
    assert economic_estimate(5.5) > 0

def test_economic_estimate_scales_with_yield():
    assert economic_estimate(6.0) > economic_estimate(5.0)

def test_economic_estimate_scales_with_area():
    assert economic_estimate(5.5, area_ha=2.0) == pytest.approx(economic_estimate(5.5, area_ha=1.0) * 2)

def test_economic_estimate_hpp_rate():
    assert economic_estimate(1.0, area_ha=1.0) == pytest.approx(6_500_000)


# ── build_feature_vector ──────────────────────────────────────────────────────

@pytest.fixture
def sample_inputs():
    climate = {k: 1.0 for k in CLIMATE_14}
    kab = {k: 1.0 for k in KAB_STAT_COLS}
    anomalies = {k: 0.0 for k in ANOMALY_COLS}
    return climate, kab, anomalies

def test_feature_vector_shape(sample_inputs):
    climate, kab, anomalies = sample_inputs
    X = build_feature_vector(climate, kab, 5.5, 5.5, anomalies, FEATURES)
    assert X.shape == (1, 35)

def test_feature_vector_lag_order(sample_inputs):
    climate, kab, anomalies = sample_inputs
    X = build_feature_vector(climate, kab, 9.9, 8.8, anomalies, FEATURES)
    assert X["yield_lag1"].iloc[0] == pytest.approx(9.9)
    assert X["yield_lag2"].iloc[0] == pytest.approx(8.8)

def test_feature_vector_type(sample_inputs):
    climate, kab, anomalies = sample_inputs
    X = build_feature_vector(climate, kab, 5.5, 5.5, anomalies, FEATURES)
    assert isinstance(X, pd.DataFrame)
