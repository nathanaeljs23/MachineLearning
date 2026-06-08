# Agri-Smart — PPT Slide Summary
**Course:** COMP6577001 Machine Learning · BINUS University  
**Student:** Nathanael Joshua

---

## 1. Cover
- **Title:** Agri-Smart: Padi Viability Predictor
- **Subtitle:** Predicting rice cultivation viability across Java Island using climate data and machine learning
- Course, student name, BINUS University logo

---

## 2. Problem
- Indonesia is the 3rd largest rice producer globally — Java Island is its core production zone
- Farmers lack data-driven tools to assess whether their land and climate conditions will yield a viable harvest
- **Goal:** Build a web app that takes climate inputs for any location on Java and predicts whether padi cultivation is **viable or not**, with an estimated yield and revenue output
- **Coverage:** 112 kabupaten (regencies) across 5 provinces — Jawa Barat, Jawa Tengah, Jawa Timur, Banten, DI Yogyakarta (DKI Jakarta excluded — urban; Kota Tangerang Selatan dropped — insufficient data)

---

## 3. Dataset
- **File:** `java_padi_dataset_v3.csv`
- **Shape:** 896 rows × 18 columns (112 kabupaten × 8 years: 2018–2025)
- **Sources:**
  - Climate (14 features): Open-Meteo ERA5-Land API — CC-BY 4.0, free
  - Yield target: BPS Indonesia provincial portals — official government statistics
- **14 Climate Features:** Mean/Max/Min temperature, Annual rainfall, Precipitation hours, Humidity, Sunshine hours, Shortwave radiation, ET₀, Vapour pressure deficit, Wind speed, Soil moisture (0–7cm, 7–28cm), Soil temperature
- **Target:** `yield_ton_ha` — range 3.56–7.53 ton/ha
- No synthetic data. All sources are real and legally sourced.

---

## 4. EDA
- Yield distribution across 112 kabupaten shows clear regional patterns — Jawa Timur highest, Banten/DI Yogyakarta lowest
- Strong correlation between kab_mean yield and current year yield (autocorrelation)
- Climate features vary significantly by province — justifies using province-level anomaly z-scores
- Yield is autocorrelated year-over-year → supports adding lag features
- Key finding: kabupaten historical average yield is the strongest single predictor (+9% F1 vs. climate-only baseline)
- *(Use EDA PNGs from `data/processed/` folder)*

---

## 5. ML Formulation
- **Task:** Binary Classification — Viable (1) vs Not Viable (0)
- **Label definition:** Viable if `yield_ton_ha ≥ median of training data` (threshold = 5.5675 ton/ha, computed from 2018–2024 only — no data leakage)
- **Train/Test Split:** Time-based (not random)
  - Train: 2018–2024 → 784 rows
  - Test: 2025 → 112 rows (held-out future year)
- **Feature Engineering (35 total):**
  - 14 raw climate features
  - 5 kabupaten yield statistics (mean, std, median, min, max) — from train only
  - 2 lag features (yield_lag1, yield_lag2 — previous 1 and 2 years per kabupaten)
  - 14 climate anomaly z-scores (deviation from kabupaten historical climate mean)
- Why binary over regression: Binary F1=0.808 vs Regression R²=0.71; clearer actionable output for farmers

---

## 6. Model Selection
Models evaluated on Test 2025 (112 rows):

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 82.14% | 80.77% | 90.24% |
| Random Forest | 79.46% | 79.28% | 89.86% |
| SVM (RBF) | 75.00% | 73.08% | 85.01% |
| Gradient Boosting | 76.79% | 75.93% | 87.97% |
| KNN | 78.57% | 78.18% | 87.43% |
| Decision Tree | 75.89% | 76.11% | 83.14% |
| **Extra Trees (Final)** | **83.04%** | **82.57%** | **90.72%** |

- Extra Trees selected: best F1 and AUC with extended 35-feature set
- All classical ML — no deep learning (course requirement)
- 5-fold cross-validation used during selection

---

## 7. Training / Validation
- **Final model:** Extra Trees Classifier
  - `n_estimators=300`, `min_samples_leaf=2`, `random_state=42`
  - Wrapped in `Pipeline([StandardScaler, ExtraTreesClassifier])`
- Kab stats computed from train only → mapped to test (no leakage)
- Anomaly z-scores computed from train climate mean/std → applied to test
- Lag features: `yield_lag1` = previous year yield per kabupaten; NaNs filled with `kab_mean`
- Viability threshold = training median = **5.5675 ton/ha**
- 5-fold CV confirmed Extra Trees most stable across folds

---

## 8. Evaluation
**Test set: 2025 (112 unseen rows)**

| Metric | Score |
|---|---|
| Accuracy | **83.04%** |
| F1-Score | **82.57%** |
| AUC-ROC | **90.72%** |

- Confusion matrix, ROC curve, and feature importance plot available in `notebooks/02_model_training.ipynb`
- Top features: `kab_mean`, `yield_lag1`, `yield_lag2`, temperature features, soil moisture
- Viability output brackets:
  - Great — P ≥ 70% → yield ≥ 5.78 ton/ha
  - Moderate — P 40–70% → yield 5.31–5.78 ton/ha
  - Bad — P < 40% → yield < 5.31 ton/ha
- Economic estimate: `yield × 1000 kg × Rp 6,500` (HPP GKP Kepbadan No. 14/2025)

---

## 9. Deployment
- **Framework:** Streamlit web app
- **Hosted:** Streamlit Community Cloud (public URL)
- **Two input modes:**
  - **Map Pin:** User clicks anywhere on a Folium interactive map → app finds nearest kabupaten via Haversine distance → uses stored climate averages → predicts viability
  - **Manual Input:** User types 14 climate values + selects province → app computes anomaly z-scores against province climate reference → predicts viability
- **Latency:** < 100ms (Extra Trees inference is near-instant)
- **Stack:** Python 3.9, scikit-learn, pandas, Streamlit, Folium, streamlit-folium

---

## 10. Screenshots
- App homepage with map tab
- Example of clicking a location and seeing Great/Moderate/Bad result card
- Manual input form with 14 fields
- Result card showing: Viability Score, Estimated Yield, Revenue Estimate, Why explanation

---

## 11. User Testing Design
- **Target users:** Farmers, agricultural students, extension workers
- **Method:** Likert scale (1–5) + qualitative open-ended questions
- **Metrics measured:**
  - Ease of use
  - Clarity of output
  - Usefulness of result
  - Trust in prediction
- **Minimum:** 5 real users (required for full grade)
- Testing done on live deployed app URL

---

## 12. User Testing Results
*(Fill in after conducting tests with at least 5 users)*
- Summary of Likert scores per question
- Notable qualitative feedback
- Overall satisfaction score

---

## 13. Analysis
- Model performs well on 2025 (unseen future year) — generalizes beyond training period
- Key driver: kabupaten historical yield encoding captures long-term productivity patterns that climate alone cannot
- Lag features capture year-to-year momentum in yield
- Climate anomaly z-scores add contextual deviation signal that improves discrimination
- Limitation of Manual Input: uses province-level proxy for kab stats — less precise than map pin mode

---

## 14. Limitations
- Coverage: Java Island only — not applicable to Sumatra, Kalimantan, etc.
- Kab stats from 2018–2024 only — does not yet reflect 2025 updates
- Manual input mode uses province mean as proxy for kabupaten encoding — adds uncertainty
- No soil type or irrigation data (SoilGrids API returned only 25/112 kabupaten — unreliable, dropped)
- Model retrained annually required to stay current

---

## 15. Conclusion
- Agri-Smart successfully predicts padi viability across 112 Java kabupaten with **83% accuracy and 90.7% AUC-ROC** on held-out 2025 data
- Extra Trees + kabupaten encoding + lag features + anomaly z-scores is the strongest combination tested
- Two-mode Streamlit app makes it accessible to both location-aware and data-aware users
- Satisfies all AOL requirements: classical ML, scikit-learn, Streamlit deployment, TDD (62 tests), GitHub

---

## 16. Appendix (optional)
- Full model comparison table
- Feature importance chart
- Dataset column descriptions
- Data sources and licenses
- Test suite summary (62 tests: 27 app utils, 24 data validation, 12 model validation — all passing)
