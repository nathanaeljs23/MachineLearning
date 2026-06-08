"""12 model validation tests for app/model.pkl and app/model_meta.json."""

import json
import pytest
import pandas as pd
import joblib

MODEL_PATH = "app/model.pkl"
META_PATH  = "app/model_meta.json"

EXPECTED_FEATURES    = 35   # 14 climate + 5 kab_stats + 2 lags + 14 anomalies
EXPECTED_MIN_ACCURACY = 0.82
EXPECTED_MIN_F1       = 0.80
EXPECTED_MIN_AUC      = 0.88


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def meta():
    with open(META_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sample_X(meta):
    """One realistic 35-feature row (Java average, anomalies = 0)."""
    row = {
        # climate
        "temperature_mean_c": 26.0, "temperature_max_c": 30.5, "temperature_min_c": 22.5,
        "rainfall_mm_year": 2500.0, "precip_hours_day": 7.5, "humidity_pct": 80.0,
        "sunshine_hrs_day": 9.5, "shortwave_radiation": 19.0, "et0_mm_day": 4.0,
        "vapour_pressure_def": 1.5, "wind_speed": 5.5,
        "soil_moisture_0_7cm": 0.32, "soil_moisture_7_28cm": 0.31, "soil_temperature": 26.0,
        # kab stats
        "kab_mean": 5.6, "kab_std": 0.3, "kab_median": 5.6, "kab_min": 4.8, "kab_max": 6.4,
        # lags
        "yield_lag1": 5.6, "yield_lag2": 5.6,
        # anomalies (all zero = average year for this kabupaten)
        "temperature_mean_c_anomaly": 0.0, "temperature_max_c_anomaly": 0.0,
        "temperature_min_c_anomaly": 0.0, "rainfall_mm_year_anomaly": 0.0,
        "precip_hours_day_anomaly": 0.0, "humidity_pct_anomaly": 0.0,
        "sunshine_hrs_day_anomaly": 0.0, "shortwave_radiation_anomaly": 0.0,
        "et0_mm_day_anomaly": 0.0, "vapour_pressure_def_anomaly": 0.0,
        "wind_speed_anomaly": 0.0, "soil_moisture_0_7cm_anomaly": 0.0,
        "soil_moisture_7_28cm_anomaly": 0.0, "soil_temperature_anomaly": 0.0,
    }
    return pd.DataFrame([[row[f] for f in meta["features"]]], columns=meta["features"])


# ── Meta validation ───────────────────────────────────────────────────────────

def test_meta_model_type(meta):
    assert meta["model_type"] == "binary_classification"

def test_meta_feature_count(meta):
    assert len(meta["features"]) == EXPECTED_FEATURES

def test_meta_accuracy(meta):
    assert meta["accuracy"] >= EXPECTED_MIN_ACCURACY

def test_meta_f1_score(meta):
    assert meta["f1_score"] >= EXPECTED_MIN_F1

def test_meta_auc_roc(meta):
    assert meta["auc_roc"] >= EXPECTED_MIN_AUC

def test_meta_kab_yield_map_size(meta):
    assert len(meta["kab_yield_map"]) == 112

def test_meta_kab_stats_map_size(meta):
    assert len(meta["kab_stats_map"]) == 112

def test_meta_thresholds_order(meta):
    assert meta["low_threshold"] < meta["threshold"] < meta["high_threshold"]


# ── Model inference ───────────────────────────────────────────────────────────

def test_model_predict_returns_binary(model, sample_X):
    assert model.predict(sample_X)[0] in (0, 1)

def test_model_predict_proba_shape(model, sample_X):
    assert model.predict_proba(sample_X).shape == (1, 2)

def test_model_proba_sums_to_one(model, sample_X):
    assert abs(model.predict_proba(sample_X)[0].sum() - 1.0) < 1e-6

def test_model_proba_in_range(model, sample_X):
    prob = model.predict_proba(sample_X)[0][1]
    assert 0.0 <= prob <= 1.0
