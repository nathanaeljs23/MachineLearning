"""
Internationalisation for Agri-Smart — English (en) and Bahasa Indonesia (id).

Only *display* text lives here. The model's canonical viability labels
("Great" / "Moderate" / "Bad") stay in English everywhere in the logic and
data; they are translated to the selected language at render time via VIABILITY.
"""

# Display name → internal language code. Order = order shown in the selector.
LANGUAGES = {"English": "en", "Bahasa Indonesia": "id"}

# ── UI strings ─────────────────────────────────────────────────────────────────
UI = {
    "en": {
        "lang_label":   "🌐 Language / Bahasa",
        "app_title":    "🌾 Agri-Smart — Padi Viability Predictor",
        "metrics_caption": "Extra Trees classifier · Java Island · "
                           "Accuracy {acc:.1f}% · F1 {f1:.1f}% · AUC-ROC {auc:.1f}%",
        "tab_map":      "🗺️ Map Pin",
        "tab_manual":   "✏️ Manual Input",
        "tab_rank":     "📊 Rankings",

        # Map Pin tab
        "map_subheader": "Pin a location on Java Island",
        "map_intro":    "Click anywhere on the map, or use the **locate button** "
                        "(the ⊕ crosshair icon below) to pin your current location. "
                        "The app finds the nearest kabupaten and predicts padi "
                        "viability using its historical climate averages.",
        "map_hint":     "👈 Click the ⊕ crosshair button and allow location access "
                        "to pin where you are.",
        "loc_your":     "Your location",
        "loc_pinned":   "Pinned",
        "nearest_caption": "📍 {origin}: {lat:.4f}, {lon:.4f}  ·  "
                           "Nearest kabupaten: **{kab}** ({prov}) · {dist:.0f} km away",
        "off_java_warning": "This point is far from any kabupaten in the dataset — "
                            "it may be outside Java. The prediction uses the nearest "
                            "available region.",
        "climate_features_expander": "Climate features used",
        "map_empty":    "_Click the map or use the ⊕ button to pin a location._",

        # Manual Input tab
        "manual_subheader": "Enter climate values manually",
        "manual_intro": "Input 14 climate features for your location. Select a "
                        "province — its historical yield stats are used as a "
                        "regional proxy.",
        "province_label": "Province",
        "predict_button": "🌾  Predict Viability",

        # Rankings tab
        "rank_subheader": "Kabupaten viability rankings",
        "rank_intro":   "All 112 kabupaten ranked by predicted padi viability, "
                        "computed from each region's historical climate averages.",
        "filter_province": "Filter by province",
        "no_province":  "Select at least one province to see rankings.",
        "metric_regions":  "Regions",
        "metric_great":    "🟢 Great",
        "metric_moderate": "🟡 Moderate",
        "metric_bad":      "🔴 Bad",
        "col_rank":      "Rank",
        "col_kabupaten": "Kabupaten",
        "col_province":  "Province",
        "col_viability": "Viability",
        "col_score":     "Score",
        "col_yield":     "Yield",
        "col_revenue":   "Revenue",
        "download_csv":  "⬇️ Download rankings (CSV)",
        "compare_header": "#### ⚖️ Compare kabupaten",
        "compare_select": "Select 2–4 kabupaten to compare side by side",
        "compare_one_more": "Pick at least one more kabupaten to compare.",
        "compare_empty": "_Select kabupaten above to compare them side by side._",

        # Result card
        "badge_template":   "{label} Viability",
        "card_score":       "Viability Score",
        "card_yield":       "Estimated Yield",
        "card_yield_unit":  "ton/ha",
        "card_revenue":     "Revenue",
        "card_revenue_sub": "(1 ha)",
        "card_revenue_note": " · HPP GKP Rp 6,500/kg",
        "card_why":         "Why",

        # Sidebar
        "about_header": "### About",
        "about_body":   "**Agri-Smart** predicts padi cultivation viability across "
                        "Java Island using historical climate data and kabupaten "
                        "yield patterns.\n\n"
                        "**Model:** Extra Trees (35 features)\n\n"
                        "**Training:** 2018–2024 · **Test:** 2025\n\n"
                        "**Coverage:** 112 kabupaten across 5 provinces",
        "legend_header":   "**Viability legend**",
        "legend_great":    "🟢 **Great** — P ≥ 70%",
        "legend_moderate": "🟡 **Moderate** — P 40–70%",
        "legend_bad":      "🔴 **Bad** — P < 40%",
        "footer": "<small>COMP6577001 Machine Learning · BINUS University · "
                  "Nathanael Joshua</small>",
    },
    "id": {
        "lang_label":   "🌐 Language / Bahasa",
        "app_title":    "🌾 Agri-Smart — Prediktor Kelayakan Padi",
        "metrics_caption": "Klasifikasi Extra Trees · Pulau Jawa · "
                           "Akurasi {acc:.1f}% · F1 {f1:.1f}% · AUC-ROC {auc:.1f}%",
        "tab_map":      "🗺️ Pin Peta",
        "tab_manual":   "✏️ Input Manual",
        "tab_rank":     "📊 Peringkat",

        # Map Pin tab
        "map_subheader": "Tandai lokasi di Pulau Jawa",
        "map_intro":    "Klik di mana saja pada peta, atau gunakan **tombol lokasi** "
                        "(ikon ⊕ di bawah) untuk menandai lokasi Anda saat ini. "
                        "Aplikasi akan mencari kabupaten terdekat dan memprediksi "
                        "kelayakan padi menggunakan rata-rata iklim historisnya.",
        "map_hint":     "👈 Klik tombol ⊕ dan izinkan akses lokasi untuk menandai "
                        "posisi Anda.",
        "loc_your":     "Lokasi Anda",
        "loc_pinned":   "Ditandai",
        "nearest_caption": "📍 {origin}: {lat:.4f}, {lon:.4f}  ·  "
                           "Kabupaten terdekat: **{kab}** ({prov}) · berjarak {dist:.0f} km",
        "off_java_warning": "Titik ini jauh dari kabupaten mana pun dalam dataset — "
                            "kemungkinan di luar Pulau Jawa. Prediksi menggunakan "
                            "wilayah terdekat yang tersedia.",
        "climate_features_expander": "Fitur iklim yang digunakan",
        "map_empty":    "_Klik peta atau gunakan tombol ⊕ untuk menandai lokasi._",

        # Manual Input tab
        "manual_subheader": "Masukkan nilai iklim secara manual",
        "manual_intro": "Masukkan 14 fitur iklim untuk lokasi Anda. Pilih provinsi — "
                        "statistik hasil panen historisnya digunakan sebagai proksi "
                        "wilayah.",
        "province_label": "Provinsi",
        "predict_button": "🌾  Prediksi Kelayakan",

        # Rankings tab
        "rank_subheader": "Peringkat kelayakan kabupaten",
        "rank_intro":   "Seluruh 112 kabupaten diperingkat berdasarkan prediksi "
                        "kelayakan padi, dihitung dari rata-rata iklim historis "
                        "tiap wilayah.",
        "filter_province": "Saring berdasarkan provinsi",
        "no_province":  "Pilih minimal satu provinsi untuk melihat peringkat.",
        "metric_regions":  "Wilayah",
        "metric_great":    "🟢 Sangat Baik",
        "metric_moderate": "🟡 Sedang",
        "metric_bad":      "🔴 Kurang",
        "col_rank":      "Peringkat",
        "col_kabupaten": "Kabupaten",
        "col_province":  "Provinsi",
        "col_viability": "Kelayakan",
        "col_score":     "Skor",
        "col_yield":     "Hasil",
        "col_revenue":   "Pendapatan",
        "download_csv":  "⬇️ Unduh peringkat (CSV)",
        "compare_header": "#### ⚖️ Bandingkan kabupaten",
        "compare_select": "Pilih 2–4 kabupaten untuk dibandingkan berdampingan",
        "compare_one_more": "Pilih minimal satu kabupaten lagi untuk dibandingkan.",
        "compare_empty": "_Pilih kabupaten di atas untuk membandingkannya berdampingan._",

        # Result card
        "badge_template":   "Kelayakan {label}",
        "card_score":       "Skor Kelayakan",
        "card_yield":       "Perkiraan Hasil",
        "card_yield_unit":  "ton/ha",
        "card_revenue":     "Pendapatan",
        "card_revenue_sub": "(1 ha)",
        "card_revenue_note": " · HPP GKP Rp 6.500/kg",
        "card_why":         "Alasan",

        # Sidebar
        "about_header": "### Tentang",
        "about_body":   "**Agri-Smart** memprediksi kelayakan budidaya padi di "
                        "seluruh Pulau Jawa menggunakan data iklim historis dan "
                        "pola hasil panen kabupaten.\n\n"
                        "**Model:** Extra Trees (35 fitur)\n\n"
                        "**Pelatihan:** 2018–2024 · **Uji:** 2025\n\n"
                        "**Cakupan:** 112 kabupaten di 5 provinsi",
        "legend_header":   "**Keterangan kelayakan**",
        "legend_great":    "🟢 **Sangat Baik** — P ≥ 70%",
        "legend_moderate": "🟡 **Sedang** — P 40–70%",
        "legend_bad":      "🔴 **Kurang** — P < 40%",
        "footer": "<small>COMP6577001 Machine Learning · BINUS University · "
                  "Nathanael Joshua</small>",
    },
}

# ── Viability display names (canonical English label → localised) ───────────────
VIABILITY = {
    "en": {"Great": "Great",       "Moderate": "Moderate", "Bad": "Bad"},
    "id": {"Great": "Sangat Baik", "Moderate": "Sedang",   "Bad": "Kurang"},
}

# ── Climate feature labels (name, unit) for inputs and the feature table ────────
CLIMATE_LABELS = {
    "en": {
        "temperature_mean_c":   ("Mean Temperature",         "°C"),
        "temperature_max_c":    ("Max Temperature",          "°C"),
        "temperature_min_c":    ("Min Temperature",          "°C"),
        "rainfall_mm_year":     ("Annual Rainfall",          "mm/year"),
        "precip_hours_day":     ("Precipitation Hours",      "hrs/day"),
        "humidity_pct":         ("Humidity",                 "%"),
        "sunshine_hrs_day":     ("Sunshine Hours",           "hrs/day"),
        "shortwave_radiation":  ("Shortwave Radiation",      "MJ/m²/day"),
        "et0_mm_day":           ("Evapotranspiration (ET₀)", "mm/day"),
        "vapour_pressure_def":  ("Vapour Pressure Deficit",  "kPa"),
        "wind_speed":           ("Wind Speed",               "m/s"),
        "soil_moisture_0_7cm":  ("Soil Moisture 0-7cm",      "m³/m³"),
        "soil_moisture_7_28cm": ("Soil Moisture 7-28cm",     "m³/m³"),
        "soil_temperature":     ("Soil Temperature",         "°C"),
    },
    "id": {
        "temperature_mean_c":   ("Suhu Rata-rata",            "°C"),
        "temperature_max_c":    ("Suhu Maksimum",             "°C"),
        "temperature_min_c":    ("Suhu Minimum",              "°C"),
        "rainfall_mm_year":     ("Curah Hujan Tahunan",       "mm/tahun"),
        "precip_hours_day":     ("Jam Hujan",                 "jam/hari"),
        "humidity_pct":         ("Kelembapan",                "%"),
        "sunshine_hrs_day":     ("Jam Penyinaran Matahari",   "jam/hari"),
        "shortwave_radiation":  ("Radiasi Gelombang Pendek",  "MJ/m²/hari"),
        "et0_mm_day":           ("Evapotranspirasi (ET₀)",    "mm/hari"),
        "vapour_pressure_def":  ("Defisit Tekanan Uap",       "kPa"),
        "wind_speed":           ("Kecepatan Angin",           "m/s"),
        "soil_moisture_0_7cm":  ("Kelembapan Tanah 0-7cm",    "m³/m³"),
        "soil_moisture_7_28cm": ("Kelembapan Tanah 7-28cm",   "m³/m³"),
        "soil_temperature":     ("Suhu Tanah",                "°C"),
    },
}

# ── Readable climate names used inside generated explanations ───────────────────
CLIMATE_READABLE = {
    "en": {
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
    },
    "id": {
        "temperature_mean_c":   "suhu rata-rata",
        "temperature_max_c":    "suhu maksimum",
        "temperature_min_c":    "suhu minimum",
        "rainfall_mm_year":     "curah hujan tahunan",
        "precip_hours_day":     "jam hujan",
        "humidity_pct":         "kelembapan",
        "sunshine_hrs_day":     "jam penyinaran matahari",
        "shortwave_radiation":  "radiasi matahari",
        "et0_mm_day":           "evapotranspirasi",
        "vapour_pressure_def":  "defisit tekanan uap",
        "wind_speed":           "kecepatan angin",
        "soil_moisture_0_7cm":  "kelembapan tanah atas",
        "soil_moisture_7_28cm": "kelembapan tanah bawah",
        "soil_temperature":     "suhu tanah",
    },
}

# ── Explanation sentence templates ─────────────────────────────────────────────
# factor_fmt controls word order of "<direction> <feature>": English puts the
# direction first ("above-average rainfall"), Indonesian puts it after the noun
# ("curah hujan tahunan di atas rata-rata").
EXPL = {
    "en": {
        "above": "above-average",
        "below": "below-average",
        "join":  " and ",
        "factor_fmt": "{d} {n}",
        "Great_base":     "Regional historical yield averages {km}, above the "
                          "viability threshold of {th}.",
        "Great_factors":  " Current climate shows {factors}, supporting strong output.",
        "Great_none":     " Climate is close to the regional norm — ideal conditions for padi.",
        "Moderate_base":  "Regional average yield is {km} (threshold: {th}).",
        "Moderate_factors": " Notable climate deviations: {factors} — some yield "
                            "variability expected.",
        "Moderate_none":  " Climate is near average; seasonal conditions will "
                          "determine final output.",
        "Bad_base":       "Regional average yield of {km} falls below the viability "
                          "threshold of {th}.",
        "Bad_factors":    " Climate also shows {factors}, further reducing viability.",
        "Bad_none":       " Low historical yield in this region drives the prediction.",
    },
    "id": {
        "above": "di atas rata-rata",
        "below": "di bawah rata-rata",
        "join":  " dan ",
        "factor_fmt": "{n} {d}",
        "Great_base":     "Rata-rata hasil panen historis wilayah ini {km}, di atas "
                          "ambang kelayakan {th}.",
        "Great_factors":  " Kondisi iklim saat ini menunjukkan {factors}, mendukung "
                          "hasil yang tinggi.",
        "Great_none":     " Iklim mendekati norma wilayah — kondisi ideal untuk padi.",
        "Moderate_base":  "Rata-rata hasil panen wilayah ini {km} (ambang: {th}).",
        "Moderate_factors": " Penyimpangan iklim yang menonjol: {factors} — variasi "
                            "hasil mungkin terjadi.",
        "Moderate_none":  " Iklim mendekati rata-rata; kondisi musiman akan "
                          "menentukan hasil akhir.",
        "Bad_base":       "Rata-rata hasil panen wilayah {km} berada di bawah ambang "
                          "kelayakan {th}.",
        "Bad_factors":    " Iklim juga menunjukkan {factors}, semakin menurunkan kelayakan.",
        "Bad_none":       " Hasil panen historis yang rendah di wilayah ini mendorong prediksi.",
    },
}
