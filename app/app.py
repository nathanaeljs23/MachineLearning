import json
import os
import sys

import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    CLIMATE_14,
    build_feature_vector,
    compute_anomalies,
    economic_estimate,
    estimated_yield,
    haversine,
    nearest_kabupaten,
    rank_kabupaten,
    viability_label,
)
from i18n import (
    CLIMATE_LABELS,
    CLIMATE_READABLE,
    EXPL,
    LANGUAGES,
    UI,
    VIABILITY,
)

# ── Constants ────────────────────────────────────────────────────────────────

APP_DIR = os.path.dirname(os.path.abspath(__file__))

PROVINCES       = ["Banten", "DI Yogyakarta", "Jawa Barat", "Jawa Tengah", "Jawa Timur"]
DEFAULT_PROVINCE = "Jawa Tengah"

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


@st.cache_data
def compute_rankings():
    """Predict viability for all kabupaten once and cache the ranked table."""
    return rank_kabupaten(load_model(), load_meta(), load_kab_df())


COLOUR_HEX = {"green": "#16a34a", "orange": "#d97706", "red": "#dc2626"}


# ── Prediction helper ─────────────────────────────────────────────────────────

def make_explanation(label: str, kab_mean: float, threshold: float,
                     anomaly_vals: dict, lang: str = "en") -> str:
    e  = EXPL[lang]
    rd = CLIMATE_READABLE[lang]

    scored = []
    for col in CLIMATE_14:
        z = anomaly_vals.get(f"{col}_anomaly", 0.0)
        if abs(z) > 0.6:
            direction = e["above"] if z > 0 else e["below"]
            scored.append((abs(z), direction, rd[col]))
    scored.sort(reverse=True)
    top = scored[:2]

    km_str = f"<b>{kab_mean:.2f} ton/ha</b>"
    th_str = f"{threshold:.2f} ton/ha"

    base = e[f"{label}_base"].format(km=km_str, th=th_str)
    if top:
        factors = e["join"].join(e["factor_fmt"].format(d=d, n=n) for _, d, n in top)
        base += e[f"{label}_factors"].format(factors=factors)
    else:
        base += e[f"{label}_none"]
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

def show_result(prob, label, colour, est_yield, revenue, title="", explanation="",
                lang="en"):
    T = UI[lang]
    disp_label = VIABILITY[lang][label]
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
        f'<span class="viability-label">{T["card_why"]}</span>'
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
                <span class="viability-badge">{T["badge_template"].format(label=disp_label)}</span>
                {"<span class='viability-location'>" + title + "</span>" if title else ""}
            </div>
            <div class="viability-row">
                <span class="viability-label">{T["card_score"]}</span>
                <span class="viability-value">
                    {prob*100:.1f}%
                    <span class="conf-bar-wrap"><span class="conf-bar-fill"></span></span>
                </span>
            </div>
            <div class="viability-row">
                <span class="viability-label">{T["card_yield"]}</span>
                <span class="viability-value">{est_yield}<span class="viability-sub">{T["card_yield_unit"]}</span></span>
            </div>
            <div class="viability-row">
                <span class="viability-label">{T["card_revenue"]} <span class="viability-sub">{T["card_revenue_sub"]}</span></span>
                <span class="viability-value">Rp {revenue:,.0f}<span class="viability-sub">{T["card_revenue_note"]}</span></span>
            </div>
            {expl_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Compare card ──────────────────────────────────────────────────────────────

def compare_card(name, province, prob, label, colour, est_yield, revenue, lang="en"):
    """Compact card with only inline styles, so several can render side by side
    without the shared CSS classes in show_result() colliding."""
    T          = UI[lang]
    disp_label = VIABILITY[lang][label]
    border   = COLOUR_HEX[colour]
    badge_bg = {"green": "#dcfce7", "orange": "#fef3c7", "red": "#fee2e2"}[colour]
    text     = {"green": "#14532d", "orange": "#78350f", "red": "#7f1d1d"}[colour]
    icon     = {"green": "🟢", "orange": "🟡", "red": "🔴"}[colour]
    row = (
        '<div style="display:flex;justify-content:space-between;padding:7px 0;'
        'border-bottom:1px solid #eee;">'
        '<span style="color:#6b7280;font-size:0.72rem;font-weight:700;'
        'text-transform:uppercase;letter-spacing:0.05em;">{k}</span>'
        '<span style="font-weight:900;color:#111;">{v}</span></div>'
    )
    st.markdown(
        f"""
        <div style="background:#fff;border:2px solid {border};border-radius:12px;
                    padding:16px 18px;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <div style="font-weight:800;font-size:1.02rem;color:#1f2937;">{name}</div>
            <div style="color:#6b7280;font-size:0.78rem;margin-bottom:10px;">{province}</div>
            <div style="display:inline-block;background:{badge_bg};color:{text};
                        border:2px solid {border};font-weight:800;font-size:0.82rem;
                        padding:3px 12px;border-radius:999px;margin-bottom:10px;">
                {icon} {disp_label}
            </div>
            {row.format(k=T["card_score"],   v=f"{prob*100:.1f}%")}
            {row.format(k=T["card_yield"],   v=f"{est_yield} t/ha")}
            {row.format(k=T["card_revenue"], v=f"Rp {revenue:,.0f}")}
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

    # ── Language selector (top of sidebar) ────────────────────────────────────
    with st.sidebar:
        lang_name = st.radio(
            UI["en"]["lang_label"], list(LANGUAGES.keys()), key="lang"
        )
        st.markdown("---")
    lang    = LANGUAGES[lang_name]
    T       = UI[lang]
    clabels = CLIMATE_LABELS[lang]

    st.title(T["app_title"])
    st.caption(T["metrics_caption"].format(
        acc=meta["accuracy"] * 100,
        f1=meta["f1_score"] * 100,
        auc=meta["auc_roc"] * 100,
    ))

    tab1, tab2, tab_rank = st.tabs([T["tab_map"], T["tab_manual"], T["tab_rank"]])

    # ════════════════════════════════════════════════════════
    # TAB 1 — Map Pin
    # ════════════════════════════════════════════════════════
    with tab1:
        st.subheader(T["map_subheader"])
        st.markdown(T["map_intro"])

        # ── Geolocation button ─────────────────────────────────────────────
        geo_col, geo_hint, _ = st.columns(
            [1, 5, 6], gap="small", vertical_alignment="center"
        )
        with geo_col:
            loc = streamlit_geolocation()
        with geo_hint:
            st.caption(T["map_hint"])

        geo_latlon = None
        if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
            geo_latlon = (float(loc["latitude"]), float(loc["longitude"]))
        geo_just_fetched = geo_latlon is not None and geo_latlon != st.session_state.get("mp_prev_geo")

        m = folium.Map(location=JAVA_CENTER, zoom_start=7, tiles="CartoDB positron")
        for _, row in kab_df.iterrows():
            folium.CircleMarker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=4, color="#2a6496", fill=True, fill_opacity=0.6,
                tooltip=row["kabupaten"],
            ).add_to(m)
        if geo_latlon:
            folium.Marker(
                location=list(geo_latlon),
                tooltip=T["loc_your"],
                icon=folium.Icon(color="red", icon="user", prefix="fa"),
            ).add_to(m)

        st.markdown(
            """<style>
            div[data-testid="stCustomComponentV1"] { margin-bottom: -2.5rem !important; }
            </style>""",
            unsafe_allow_html=True,
        )
        folium_kwargs = dict(use_container_width=True, height=460, key="map_pin")
        if geo_just_fetched:
            folium_kwargs["center"] = list(geo_latlon)
            folium_kwargs["zoom"]   = 9
        map_data = st_folium(m, **folium_kwargs) or {}

        # ── Resolve active location: a fresh action (geo or click) wins ────
        clicked = map_data.get("last_clicked")
        click_latlon = (clicked["lat"], clicked["lng"]) if clicked else None

        if geo_latlon is not None and geo_latlon != st.session_state.get("mp_prev_geo"):
            active, source = geo_latlon, "geo"
        elif click_latlon is not None and click_latlon != st.session_state.get("mp_prev_click"):
            active, source = click_latlon, "click"
        elif click_latlon is not None:
            active, source = click_latlon, "click"
        elif geo_latlon is not None:
            active, source = geo_latlon, "geo"
        else:
            active, source = None, None

        st.session_state["mp_prev_geo"]   = geo_latlon
        st.session_state["mp_prev_click"] = click_latlon

        if active:
            lat, lon = active

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
            expl = make_explanation(label, kab_stats["kab_mean"], meta["threshold"],
                                    anomaly_vals, lang)

            dist_km = haversine(lat, lon, kab_row["centroid_lat"], kab_row["centroid_lon"])
            origin  = T["loc_your"] if source == "geo" else T["loc_pinned"]
            st.caption(T["nearest_caption"].format(
                origin=origin, lat=lat, lon=lon, kab=kab_name, prov=province, dist=dist_km,
            ))
            if dist_km > 75:
                st.warning(T["off_java_warning"])
            show_result(prob, label, colour, est_yield, revenue, kab_name, expl, lang)

            with st.expander(T["climate_features_expander"]):
                display = {
                    clabels[k][0]: f"{climate_vals[k]:.3f} {clabels[k][1]}"
                    for k in CLIMATE_14
                }
                st.dataframe(pd.Series(display, name="Value"), use_container_width=True)
        else:
            st.markdown(T["map_empty"])

    # ════════════════════════════════════════════════════════
    # TAB — Rankings & Compare
    # ════════════════════════════════════════════════════════
    with tab_rank:
        st.subheader(T["rank_subheader"])
        st.markdown(T["rank_intro"])

        ranking = compute_rankings()

        prov_filter = st.multiselect(
            T["filter_province"], PROVINCES, default=PROVINCES, key="rank_prov"
        )
        view = ranking[ranking["province"].isin(prov_filter)].reset_index(drop=True)

        if view.empty:
            st.info(T["no_province"])
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(T["metric_regions"],  len(view))
            c2.metric(T["metric_great"],    int((view["label"] == "Great").sum()))
            c3.metric(T["metric_moderate"], int((view["label"] == "Moderate").sum()))
            c4.metric(T["metric_bad"],      int((view["label"] == "Bad").sum()))

            map_col, table_col = st.columns([1.15, 1])

            with map_col:
                rm = folium.Map(location=JAVA_CENTER, zoom_start=7, tiles="CartoDB positron")
                for _, row in view.iterrows():
                    hexc = COLOUR_HEX[row["colour"]]
                    disp = VIABILITY[lang][row["label"]]
                    folium.CircleMarker(
                        location=[row["centroid_lat"], row["centroid_lon"]],
                        radius=6, color=hexc, weight=1,
                        fill=True, fill_color=hexc, fill_opacity=0.8,
                        tooltip=f"{row['kabupaten']} — {disp} ({row['prob']*100:.0f}%)",
                    ).add_to(rm)
                st_folium(rm, use_container_width=True, height=460, key="map_rank",
                          returned_objects=[])

            with table_col:
                table = view.copy()
                table.insert(0, T["col_rank"], range(1, len(table) + 1))
                table[T["col_viability"]] = table["label"].map(VIABILITY[lang])
                table[T["col_score"]]   = (table["prob"] * 100).round(1).map("{:.1f}%".format)
                table[T["col_yield"]]   = table["est_yield"].map("{:.2f} t/ha".format)
                table[T["col_revenue"]] = table["revenue"].map("Rp {:,.0f}".format)
                show = table[[T["col_rank"], "kabupaten", "province", T["col_viability"],
                              T["col_score"], T["col_yield"], T["col_revenue"]]].rename(
                    columns={"kabupaten": T["col_kabupaten"], "province": T["col_province"]})
                st.dataframe(show, use_container_width=True, hide_index=True, height=460)

            csv = view[["kabupaten", "province", "prob", "label",
                        "est_yield", "revenue"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                T["download_csv"], csv,
                file_name="padi_viability_rankings.csv", mime="text/csv",
            )

        st.markdown("---")
        st.markdown(T["compare_header"])
        compare = st.multiselect(
            T["compare_select"],
            ranking["kabupaten"].tolist(), max_selections=4, key="compare_sel",
        )
        if len(compare) >= 2:
            cols = st.columns(len(compare))
            for col, name in zip(cols, compare):
                r = ranking[ranking["kabupaten"] == name].iloc[0]
                with col:
                    compare_card(name, r["province"], r["prob"], r["label"],
                                 r["colour"], r["est_yield"], r["revenue"], lang)
        elif len(compare) == 1:
            st.info(T["compare_one_more"])
        else:
            st.markdown(T["compare_empty"])

    # ════════════════════════════════════════════════════════
    # TAB 2 — Manual Input
    # ════════════════════════════════════════════════════════
    with tab2:
        st.subheader(T["manual_subheader"])
        st.markdown(T["manual_intro"])

        with st.form("manual_form"):
            province_sel = st.selectbox(T["province_label"], PROVINCES,
                                        index=PROVINCES.index(DEFAULT_PROVINCE))
            col1, col2 = st.columns(2)
            manual_climate = {}

            for i, key in enumerate(CLIMATE_14):
                label_text, unit = clabels[key]
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
            submitted = st.form_submit_button(T["predict_button"], use_container_width=True)

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
            expl = make_explanation(label, kab_stats["kab_mean"], meta["threshold"],
                                    anomaly_vals, lang)
            show_result(prob, label, colour, est_yield, revenue, province_sel, expl, lang)

    # ── Sidebar (below the language selector) ─────────────────────────────────
    with st.sidebar:
        st.markdown(T["about_header"])
        st.markdown(T["about_body"])
        st.markdown("---")
        st.markdown(T["legend_header"])
        st.markdown(T["legend_great"])
        st.markdown(T["legend_moderate"])
        st.markdown(T["legend_bad"])
        st.markdown("---")
        st.markdown(T["footer"], unsafe_allow_html=True)


if __name__ == "__main__":
    main()
