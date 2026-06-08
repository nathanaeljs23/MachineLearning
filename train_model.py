"""
Train ExtraTreesClassifier on java_padi_dataset_v3.csv (extended 35 features).
Produces app/model.pkl and app/model_meta.json.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH  = "data/processed/java_padi_dataset_v3.csv"
COORDS_PATH = "data/processed/kabupaten_coordinates.csv"

CLIMATE_14 = [
    "temperature_mean_c", "temperature_max_c", "temperature_min_c",
    "rainfall_mm_year", "precip_hours_day", "humidity_pct",
    "sunshine_hrs_day", "shortwave_radiation", "et0_mm_day",
    "vapour_pressure_def", "wind_speed", "soil_moisture_0_7cm",
    "soil_moisture_7_28cm", "soil_temperature",
]
KAB_STAT_COLS = ["kab_mean", "kab_std", "kab_median", "kab_min", "kab_max"]
ANOMALY_COLS  = [f"{c}_anomaly" for c in CLIMATE_14]
FEATURES      = CLIMATE_14 + KAB_STAT_COLS + ["yield_lag1", "yield_lag2"] + ANOMALY_COLS  # 35


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["kabupaten", "year"]).copy()
    df["yield_lag1"] = df.groupby("kabupaten")["yield_ton_ha"].shift(1)
    df["yield_lag2"] = df.groupby("kabupaten")["yield_ton_ha"].shift(2)
    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["kabupaten", "year"]).reset_index(drop=True)
    df = build_lag_features(df)

    # Time-based split
    train = df[df["year"] <= 2024].copy()
    test  = df[df["year"] == 2025].copy()

    # Kab stats from TRAIN ONLY
    kab_stats = train.groupby("kabupaten")["yield_ton_ha"].agg(
        ["mean", "std", "median", "min", "max"]
    ).rename(columns={
        "mean": "kab_mean", "std": "kab_std", "median": "kab_median",
        "min": "kab_min", "max": "kab_max",
    })
    for col in KAB_STAT_COLS:
        train[col] = train["kabupaten"].map(kab_stats[col])
        test[col]  = test["kabupaten"].map(kab_stats[col])

    # Fill lag NaNs with kab_mean
    for split in [train, test]:
        m1 = split["yield_lag1"].isna()
        m2 = split["yield_lag2"].isna()
        split.loc[m1, "yield_lag1"] = split.loc[m1, "kab_mean"]
        split.loc[m2, "yield_lag2"] = split.loc[m2, "kab_mean"]

    # Climate anomaly z-scores — from TRAIN ONLY
    kab_clim_mean = train.groupby("kabupaten")[CLIMATE_14].mean()
    kab_clim_std  = train.groupby("kabupaten")[CLIMATE_14].std().fillna(1.0)
    for split in [train, test]:
        for col in CLIMATE_14:
            means = split["kabupaten"].map(kab_clim_mean[col])
            stds  = split["kabupaten"].map(kab_clim_std[col]).fillna(1.0)
            split[f"{col}_anomaly"] = (split[col] - means) / stds

    # Binary label
    threshold = train["yield_ton_ha"].median()
    train["viable"] = (train["yield_ton_ha"] >= threshold).astype(int)
    test["viable"]  = (test["yield_ton_ha"]  >= threshold).astype(int)

    X_train, y_train = train[FEATURES], train["viable"]
    X_test,  y_test  = test[FEATURES],  test["viable"]

    # Train pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"Test 2025 — Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC-ROC: {auc:.4f}")

    # Viability thresholds
    train_yield = train["yield_ton_ha"]
    low_thresh  = float(np.percentile(train_yield, 33))
    high_thresh = float(np.percentile(train_yield, 67))

    # kab_stats_map
    kab_stats_map = kab_stats.to_dict(orient="index")

    # Province proxy kab_stats (mean of per-kabupaten kab_stats within each province)
    kab_prov = train[["kabupaten", "province"]].drop_duplicates().set_index("kabupaten")
    kab_stats_df = kab_stats.copy()
    kab_stats_df["province"] = kab_stats_df.index.map(kab_prov["province"])
    prov_agg = kab_stats_df.groupby("province")[KAB_STAT_COLS].mean()
    prov_stats_map = {
        prov: {col: round(row[col], 4) for col in KAB_STAT_COLS}
        for prov, row in prov_agg.iterrows()
    }

    # Province climate stats for Option 2 anomaly computation
    train_with_prov = train.copy()
    prov_clim_mean = train_with_prov.groupby("province")[CLIMATE_14].mean()
    prov_clim_std  = train_with_prov.groupby("province")[CLIMATE_14].std().fillna(1.0)
    prov_climate_stats = {
        prov: {
            col: {
                "mean": round(float(prov_clim_mean.loc[prov, col]), 6),
                "std":  round(float(prov_clim_std.loc[prov, col]), 6),
            }
            for col in CLIMATE_14
        }
        for prov in prov_clim_mean.index
    }

    # Per-kabupaten climate stats (for Option 1 anomaly: stored but set to 0 at inference)
    kab_climate_stats = {
        kab: {
            col: {
                "mean": round(float(kab_clim_mean.loc[kab, col]), 6),
                "std":  round(float(kab_clim_std.loc[kab, col]), 6),
            }
            for col in CLIMATE_14
        }
        for kab in kab_clim_mean.index
    }

    meta = {
        "model_type":          "binary_classification",
        "model_name":          "Extra Trees",
        "features":            FEATURES,
        "threshold":           float(threshold),
        "low_threshold":       round(low_thresh, 4),
        "high_threshold":      round(high_thresh, 4),
        "accuracy":            round(acc, 4),
        "f1_score":            round(f1, 4),
        "auc_roc":             round(auc, 4),
        "kab_yield_map":       {k: round(v, 4) for k, v in kab_stats["kab_mean"].to_dict().items()},
        "kab_stats_map":       {k: {sk: round(sv, 4) for sk, sv in v.items()} for k, v in kab_stats_map.items()},
        "prov_stats_map":      prov_stats_map,
        "prov_climate_stats":  prov_climate_stats,
        "kab_climate_stats":   kab_climate_stats,
    }

    joblib.dump(pipeline, "app/model.pkl")
    with open("app/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved app/model.pkl and app/model_meta.json")

    # Rebuild kabupaten_climate_avg.csv from v3
    coords = pd.read_csv(COORDS_PATH)
    climate_avg = df.groupby("kabupaten")[CLIMATE_14].mean().reset_index()
    climate_avg = climate_avg.merge(
        coords[["kabupaten", "centroid_lat", "centroid_lon"]], on="kabupaten", how="left"
    )
    prov_map = df[["kabupaten", "province"]].drop_duplicates()
    climate_avg = climate_avg.merge(prov_map, on="kabupaten", how="left")
    climate_avg.to_csv("app/kabupaten_climate_avg.csv", index=False)
    print(f"Saved app/kabupaten_climate_avg.csv ({len(climate_avg)} rows)")


if __name__ == "__main__":
    main()
