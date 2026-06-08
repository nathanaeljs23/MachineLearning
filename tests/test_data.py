"""24 dataset validation tests for java_padi_dataset_v3.csv."""

import pytest
import pandas as pd
import numpy as np

DATASET_PATH = "data/processed/java_padi_dataset_v3.csv"
EXPECTED_KABUPATEN = 112
EXPECTED_YEARS = list(range(2018, 2026))
EXPECTED_ROWS = 896
EXPECTED_COLS = 18

CLIMATE_14 = [
    "temperature_mean_c", "temperature_max_c", "temperature_min_c",
    "rainfall_mm_year", "precip_hours_day", "humidity_pct",
    "sunshine_hrs_day", "shortwave_radiation", "et0_mm_day",
    "vapour_pressure_def", "wind_speed", "soil_moisture_0_7cm",
    "soil_moisture_7_28cm", "soil_temperature",
]
PROVINCES = {"Jawa Barat", "Jawa Tengah", "Jawa Timur", "DI Yogyakarta", "Banten"}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATASET_PATH)


# ── Shape & structure ─────────────────────────────────────────────────────────

def test_row_count(df):
    assert len(df) == EXPECTED_ROWS

def test_col_count(df):
    assert df.shape[1] == EXPECTED_COLS

def test_kabupaten_count(df):
    assert df["kabupaten"].nunique() == EXPECTED_KABUPATEN

def test_year_range(df):
    assert set(df["year"].unique()) == set(EXPECTED_YEARS)

def test_kabupaten_year_completeness(df):
    # Every kabupaten should have exactly 8 years
    counts = df.groupby("kabupaten")["year"].count()
    assert (counts == 8).all(), f"Some kabupaten have ≠ 8 rows: {counts[counts != 8]}"

def test_required_columns_present(df):
    required = ["kabupaten", "province", "year", "yield_ton_ha"] + CLIMATE_14
    assert all(c in df.columns for c in required)


# ── Province membership ───────────────────────────────────────────────────────

def test_provinces_correct(df):
    assert set(df["province"].unique()) == PROVINCES

def test_no_dki_jakarta(df):
    assert "DKI Jakarta" not in df["kabupaten"].values
    assert "DKI Jakarta" not in df["province"].values

def test_no_kota_tangerang_selatan(df):
    assert "Kota Tangerang Selatan" not in df["kabupaten"].values


# ── Yield range ───────────────────────────────────────────────────────────────

def test_yield_min(df):
    assert df["yield_ton_ha"].min() >= 3.0

def test_yield_max(df):
    assert df["yield_ton_ha"].max() <= 8.5

def test_yield_no_nulls(df):
    assert df["yield_ton_ha"].notna().all()

def test_yield_positive(df):
    assert (df["yield_ton_ha"] > 0).all()


# ── Climate feature validity ──────────────────────────────────────────────────

def test_temperature_mean_range(df):
    assert df["temperature_mean_c"].between(15, 35).all()

def test_temperature_max_gt_mean(df):
    assert (df["temperature_max_c"] >= df["temperature_mean_c"]).all()

def test_temperature_min_lt_mean(df):
    assert (df["temperature_min_c"] <= df["temperature_mean_c"]).all()

def test_humidity_pct_range(df):
    assert df["humidity_pct"].between(0, 100).all()

def test_rainfall_non_negative(df):
    assert (df["rainfall_mm_year"] >= 0).all()

def test_soil_moisture_range(df):
    assert df["soil_moisture_0_7cm"].between(0, 1).all()
    assert df["soil_moisture_7_28cm"].between(0, 1).all()

def test_sunshine_hrs_positive(df):
    assert (df["sunshine_hrs_day"] > 0).all()

def test_et0_positive(df):
    assert (df["et0_mm_day"] > 0).all()


# ── No duplicate rows ─────────────────────────────────────────────────────────

def test_no_duplicate_kabupaten_year(df):
    dupes = df.duplicated(subset=["kabupaten", "year"])
    assert not dupes.any(), f"{dupes.sum()} duplicate kabupaten-year pairs found"

def test_no_all_null_rows(df):
    assert not df.isnull().all(axis=1).any()
