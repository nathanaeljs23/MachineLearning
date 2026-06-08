import json
import os
import sys

import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    CLIMATE_14,
    build_feature_vector,
    compute_anomalies,
    economic_estimate,
    estimated_yield,
    nearest_kabupaten,
    viability_label,
)

# ── Constants ────────────────────────────────────────────────────────────────

APP_DIR = os.path.dirname(os.path.abspath(__file__))

CLIMATE_LABELS = {
    "temperature_mean_c":    ("Mean Temperature",           "°C"),
    "temperature_max_c":     ("Max Temperature",            "°C"),
    "temperature_min_c":     ("Min Temperature",            "°C"),
    "rainfall_mm_year":      ("Annual Rainfall",            "mm/year"),
    "precip_hours_day":      ("Precipitation Hours",        "hrs/day"),
    "humidity_pct":          ("Humidity",                   "%"),
    "sunshine_hrs_day":      ("Sunshine Hours",             "hrs/day"),
    "shortwave_radiation":   ("Shortwave Radiation",        "MJ/m²/day"),
    "et0_mm_day":            ("Evapotranspiration (ET₀)",   "mm/day"),
    "vapour_pressure_def":   ("Vapour Pressure Deficit",    "kPa"),
    "wind_speed":            ("Wind Speed",                 "m/s"),
    "soil_moisture_0_7cm":   ("Soil Moisture 0-7cm",        "m³/m³"),
    "soil_moisture_7_28cm":  ("Soil Moisture 7-28cm",       "m³/m³"),
    "soil_temperature":      ("Soil Temperature",           "°C"),
}

PROVINCES       = ["Banten", "DI Yogyakarta", "Jawa Barat", "Jawa Tengah", "Jawa Timur"]
DEFAULT_PROVINCE = "Jawa Tengah"

_CLIMATE_READABLE = {
    "temperature_mean_c":   "mean temperature",
    "temperature_max_c":    "max temperature",
    "temperature_min_c":    "min temperature",
    "rainfall_mm_year":     "annual rainfall",
    "precip_hours_day":     "precipitation hours",
    "humidity_pct":         "humidity",
    "sunshine_hrs_day":     "sunshine hours",
    "shortwave_radiation":  "solar radiation",
    "et0_mm_day":           "evapotranspiration",
    "vapour_pressure_def":  "vapour pressure deficit",
    "wind_speed":           "wind speed",
    "soil_moisture_0_7cm":  "topsoil moisture",
    "soil_moisture_7_28cm": "subsoil moisture",
    "soil_temperature":     "soil temperature",
}

JAVA_CENTER = [-7.5, 110.0]

DEFAULTS = {
    "temperature_mean_c":   26.0,
    "temperature_max_c":    30.5,
    "temperature_min_c":    22.5,
    "rainfall_mm_year":     2500.0,
    "precip_hours_day":     7.5,
    "humidity_pct":         80.0,
    "sunshine_hrs_day":     9.5,
    "shortwave_radiation":  19.0,
    "et0_mm_day":           4.0,
    "vapour_pressure_def":  1.5,
    "wind_speed":           5.5,
    "soil_moisture_0_7cm":  0.32,
    "soil_moisture_7_28cm": 0.31,
    "soil_temperature":     26.0,
}


# ── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(APP_DIR, "model.pkl"))


@st.cache_data
def load_meta():
    with open(os.path.join(APP_DIR, "model_meta.json")) as f:
        return json.load(f)


@st.cache_data
def load_kab_df():
    return pd.read_csv(os.path.join(APP_DIR, "kabupaten_climate_avg.csv"))


# ── Prediction helper ─────────────────────────────────────────────────────────

def make_explanation(label: str, kab_mean: float, threshold: float,
                     anomaly_vals: dict) -> str:
    scored = []
    for col in CLIMATE_14:
        z = anomaly_vals.get(f"{col}_anomaly", 0.0)
        if abs(z) > 0.6:
            direction = "above-average" if z > 0 else "below-average"
            scored.append((abs(z), direction, _CLIMATE_READABLE[col]))
    scored.sort(reverse=True)
    top = scored[:2]

    km_str = f"<b>{kab_mean:.2f} ton/ha</b>"
    th_str = f"{threshold:.2f} ton/ha"

    if label == "Great":
        base = f"Regional historical yield averages {km_str}, above the viability threshold of {th_str}."
        if top:
            factors = " and ".join(f"{d} {n}" for _, d, n in top)
            base += f" Current climate shows {factors}, supporting strong output."
        else:
            base += " Climate is close to the regional norm — ideal conditions for padi."
    elif label == "Moderate":
        base = f"Regional average yield is {km_str} (threshold: {th_str})."
        if top:
            factors = " and ".join(f"{d} {n}" for _, d, n in top)
            base += f" Notable climate deviations: {factors} — some yield variability expected."
        else:
            base += " Climate is near average; seasonal conditions will determine final output."
    else:
        base = f"Regional average yield of {km_str} falls below the viability threshold of {th_str}."
        if top:
            factors = " and ".join(f"{d} {n}" for _, d, n in top)
            base += f" Climate also shows {factors}, further reducing viability."
        else:
            base += " Low historical yield in this region drives the prediction."
    return base


def predict(model, meta, climate_vals: dict, kab_stats: dict, lag: float,
            anomaly_vals: dict):
    X = build_feature_vector(
        climate_vals=climate_vals,
        kab_stats=kab_stats,
        yield_lag1=lag,
        yield_lag2=lag,
        anomaly_vals=anomaly_vals,
        feature_order=meta["features"],
    )
    prob = model.predict_proba(X)[0][1]
    label, colour = viability_label(prob, meta["low_threshold"], meta["high_threshold"])
    est_yield = estimated_yield(prob, meta["low_threshold"], meta["high_threshold"])
    revenue   = economic_estimate(est_yield)
    return prob, label, colour, est_yield, revenue


# ── Result display ────────────────────────────────────────────────────────────

def show_result(prob, label, colour, est_yield, revenue, title="", explanation=""):
    border_map = {"green": "#16a34a", "orange": "#d97706", "red": "#dc2626"}
    text_map   = {"green": "#14532d", "orange": "#78350f", "red": "#7f1d1d"}
    badge_map  = {"green": "#dcfce7", "orange": "#fef3c7", "red": "#fee2e2"}
    track_map  = {"green": "#bbf7d0", "orange": "#fde68a", "red": "#fecaca"}
    icon_map   = {"green": "🟢", "orange": "🟡", "red": "🔴"}

    border = border_map[colour]
    text   = text_map[colour]
    badge  = badge_map[colour]
    track  = track_map[colour]
    icon   = icon_map[colour]
    bar_pct = int(prob * 100)
    expl_html = (
        f'<div class="viability-row viability-explain-row">'
        f'<span class="viability-label">Why</span>'
        f'<span class="viability-explain">{explanation}</span></div>'
    ) if explanation else ""

    st.markdown(
        f"""
        <style>
        .viability-card {{
            background: #ffffff;
            border: 2px solid {border};
            border-radius: 12px;
            padding: 20px 24px 16px;
            margin: 6px 0 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.09);
        }}
        .viability-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 14px;
        }}
        .viability-badge {{
            background: {badge};
            color: {text};
            font-size: 1.1rem;
            font-weight: 800;
            padding: 5px 16px;
            border-radius: 999px;
            border: 2px solid {border};
            letter-spacing: 0.03em;
        }}
        .viability-location {{
            color: #1f2937;
            font-size: 1rem;
            font-weight: 700;
        }}
        .viability-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 9px 0;
            border-bottom: 1px solid #e5e7eb;
        }}
        .viability-row:last-child {{ border-bottom: none; }}
        .viability-label {{
            color: #4b5563;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            flex-shrink: 0;
            margin-right: 12px;
        }}
        .viability-value {{
            color: #111827;
            font-size: 1.08rem;
            font-weight: 900;
            text-align: right;
        }}
        .viability-sub {{
            color: #6b7280;
            font-size: 0.78rem;
            font-weight: 500;
            margin-left: 4px;
        }}
        .conf-bar-wrap {{
            background: {track};
            border-radius: 999px;
            height: 10px;
            width: 130px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
            margin-left: 10px;
            border: 1px solid {border};
        }}
        .conf-bar-fill {{
            background: {border};
            height: 100%;
            width: {bar_pct}%;
            border-radius: 999px;
        }}
        .viability-explain-row {{
            align-items: flex-start !important;
            padding-top: 10px !important;
        }}
        .viability-explain {{
            color: #374151;
            font-size: 0.9rem;
            font-weight: 500;
            line-height: 1.5;
            text-align: right;
            max-width: 72%;
        }}
        </style>
        <div class="viability-card">
            <div class="viability-header">
                <span style="font-size:1.7rem">{icon}</span>
                <span class="viability-badge">{label} Viability</span>
                {"<span class='viability-location'>" + title + "</span>" if title else ""}
            </div>
            <div class="viability-row">
                <span class="viability-label">Viability Score</span>
                <span class="viability-value">
                    {prob*100:.1f}%
                    <span class="conf-bar-wrap"><span class="conf-bar-fill"></span></span>
                </span>
            </div>
            <div class="viability-row">
                <span class="viability-label">Estimated Yield</span>
                <span class="viability-value">{est_yield}<span class="viability-sub">ton/ha</span></span>
            </div>
            <div class="viability-row">
                <span class="viability-label">Revenue <span class="viability-sub">(1 ha)</span></span>
                <span class="viability-value">Rp {revenue:,.0f}<span class="viability-sub"> · HPP GKP Rp 6,500/kg</span></span>
            </div>
            {expl_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Agri-Smart | Padi Viability Predictor",
        page_icon="🌾",
        layout="wide",
    )

    model  = load_model()
    meta   = load_meta()
    kab_df = load_kab_df()

    st.title("🌾 Agri-Smart — Padi Viability Predictor")
    st.caption(
        f"Extra Trees classifier · Java Island · "
        f"Accuracy {meta['accuracy']*100:.1f}% · "
        f"F1 {meta['f1_score']*100:.1f}% · "
        f"AUC-ROC {meta['auc_roc']*100:.1f}%"
    )

    tab1, tab2 = st.tabs(["🗺️ Map Pin", "✏️ Manual Input"])

    # ════════════════════════════════════════════════════════
    # TAB 1 — Map Pin
    # ════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Pin a location on Java Island")
        st.markdown(
            "Click anywhere on the map. The app finds the nearest kabupaten and "
            "predicts padi viability using its historical climate averages."
        )

        m = folium.Map(location=JAVA_CENTER, zoom_start=7, tiles="CartoDB positron")
        for _, row in kab_df.iterrows():
            folium.CircleMarker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=4, color="#2a6496", fill=True, fill_opacity=0.6,
                tooltip=row["kabupaten"],
            ).add_to(m)

        st.markdown(
            """<style>
            div[data-testid="stCustomComponentV1"] { margin-bottom: -2.5rem !important; }
            </style>""",
            unsafe_allow_html=True,
        )
        map_data = st_folium(m, use_container_width=True, height=460, key="map_pin")

        clicked = map_data.get("last_clicked")
        if clicked:
            lat, lon = clicked["lat"], clicked["lng"]

            kab_row  = nearest_kabupaten(lat, lon, kab_df)
            kab_name = kab_row["kabupaten"]
            province = kab_row["province"]

            climate_vals = {col: kab_row[col] for col in CLIMATE_14}

            ks = meta["kab_stats_map"].get(kab_name, {})
            kab_stats = {
                "kab_mean":   ks.get("kab_mean",   meta["threshold"]),
                "kab_std":    ks.get("kab_std",    0.3),
                "kab_median": ks.get("kab_median", meta["threshold"]),
                "kab_min":    ks.get("kab_min",    meta["low_threshold"]),
                "kab_max":    ks.get("kab_max",    meta["high_threshold"]),
            }
            lag = kab_stats["kab_mean"]

            # Option 1 uses kabupaten climate averages → anomaly = 0 by definition
            anomaly_vals = {f"{col}_anomaly": 0.0 for col in CLIMATE_14}

            prob, label, colour, est_yield, revenue = predict(
                model, meta, climate_vals, kab_stats, lag, anomaly_vals
            )
            expl = make_explanation(label, kab_stats["kab_mean"], meta["threshold"], anomaly_vals)
            st.caption(f"📍 {lat:.4f}, {lon:.4f}  ·  Nearest kabupaten: **{kab_name}** ({province})")
            show_result(prob, label, colour, est_yield, revenue, kab_name, expl)

            with st.expander("Climate features used"):
                display = {
                    CLIMATE_LABELS[k][0]: f"{climate_vals[k]:.3f} {CLIMATE_LABELS[k][1]}"
                    for k in CLIMATE_14
                }
                st.dataframe(pd.Series(display, name="Value"), use_container_width=True)
        else:
            st.markdown("_Click the map to pin a location._")

    # ════════════════════════════════════════════════════════
    # TAB 2 — Manual Input
    # ════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Enter climate values manually")
        st.markdown(
            "Input 14 climate features for your location. "
            "Select a province — its historical yield stats are used as a regional proxy."
        )

        with st.form("manual_form"):
            province_sel = st.selectbox("Province", PROVINCES,
                                        index=PROVINCES.index(DEFAULT_PROVINCE))
            col1, col2 = st.columns(2)
            manual_climate = {}

            for i, key in enumerate(CLIMATE_14):
                label_text, unit = CLIMATE_LABELS[key]
                target_col = col1 if i % 2 == 0 else col2
                with target_col:
                    manual_climate[key] = st.number_input(
                        f"{label_text} ({unit})",
                        value=float(DEFAULTS[key]),
                        format="%.4f",
                        key=f"manual_{key}",
                    )

            st.markdown("""
                <style>
                div[data-testid="stFormSubmitButton"] > button {
                    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
                    color: white;
                    font-size: 1.05rem;
                    font-weight: 700;
                    padding: 0.65rem 1.5rem;
                    border: none;
                    border-radius: 10px;
                    letter-spacing: 0.04em;
                    box-shadow: 0 3px 10px rgba(22,163,74,0.35);
                    transition: transform 0.1s, box-shadow 0.1s;
                    width: 100%;
                    cursor: pointer;
                }
                div[data-testid="stFormSubmitButton"] > button:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 5px 16px rgba(22,163,74,0.45);
                }
                div[data-testid="stFormSubmitButton"] > button:active {
                    transform: translateY(0px);
                    box-shadow: 0 2px 6px rgba(22,163,74,0.3);
                }
                </style>
            """, unsafe_allow_html=True)
            submitted = st.form_submit_button("🌾  Predict Viability", use_container_width=True)

        if submitted:
            prov_stats = meta.get("prov_stats_map", {}).get(province_sel)
            if prov_stats:
                kab_stats = prov_stats
            else:
                kab_stats = {
                    "kab_mean": meta["threshold"], "kab_std": 0.3,
                    "kab_median": meta["threshold"],
                    "kab_min": meta["low_threshold"], "kab_max": meta["high_threshold"],
                }
            lag = kab_stats["kab_mean"]

            # Compute anomaly z-scores using province-level climate reference
            prov_clim = meta.get("prov_climate_stats", {}).get(province_sel)
            if prov_clim:
                anomaly_vals = compute_anomalies(manual_climate, prov_clim)
            else:
                anomaly_vals = {f"{col}_anomaly": 0.0 for col in CLIMATE_14}

            prob, label, colour, est_yield, revenue = predict(
                model, meta, manual_climate, kab_stats, lag, anomaly_vals
            )
            expl = make_explanation(label, kab_stats["kab_mean"], meta["threshold"], anomaly_vals)
            show_result(prob, label, colour, est_yield, revenue, province_sel, expl)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "**Agri-Smart** predicts padi cultivation viability across Java Island "
            "using historical climate data and kabupaten yield patterns.\n\n"
            "**Model:** Extra Trees (35 features)\n\n"
            "**Training:** 2018–2024 · **Test:** 2025\n\n"
            "**Coverage:** 112 kabupaten across 5 provinces"
        )
        st.markdown("---")
        st.markdown("**Viability legend**")
        st.markdown("🟢 **Great** — P ≥ 70%")
        st.markdown("🟡 **Moderate** — P 40–70%")
        st.markdown("🔴 **Bad** — P < 40%")
        st.markdown("---")
        st.markdown(
            "<small>COMP6577001 Machine Learning · BINUS University · "
            "Nathanael Joshua</small>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
