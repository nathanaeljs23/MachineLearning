"""Translation-parity and explanation tests for app/i18n.py + app.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from i18n import (
    CLIMATE_LABELS,
    CLIMATE_READABLE,
    EXPL,
    LANGUAGES,
    UI,
    VIABILITY,
)
from utils import CLIMATE_14
from app import make_explanation

LANGS = list(LANGUAGES.values())  # ["en", "id"]


# ── Every dict covers exactly the same languages ──────────────────────────────

@pytest.mark.parametrize("table", [UI, CLIMATE_LABELS, CLIMATE_READABLE, VIABILITY, EXPL])
def test_languages_present(table):
    assert set(table.keys()) == set(LANGS)


# ── Key parity across languages ───────────────────────────────────────────────

@pytest.mark.parametrize("table", [UI, CLIMATE_LABELS, CLIMATE_READABLE, VIABILITY, EXPL])
def test_key_parity(table):
    en_keys = set(table["en"].keys())
    for lang in LANGS:
        assert set(table[lang].keys()) == en_keys, f"{lang} keys differ from en"


# ── Climate dicts cover all 14 features ───────────────────────────────────────

@pytest.mark.parametrize("lang", LANGS)
def test_climate_labels_cover_14(lang):
    assert set(CLIMATE_LABELS[lang].keys()) == set(CLIMATE_14)
    assert all(len(v) == 2 for v in CLIMATE_LABELS[lang].values())  # (name, unit)

@pytest.mark.parametrize("lang", LANGS)
def test_climate_readable_cover_14(lang):
    assert set(CLIMATE_READABLE[lang].keys()) == set(CLIMATE_14)


# ── Viability maps the three canonical labels ─────────────────────────────────

@pytest.mark.parametrize("lang", LANGS)
def test_viability_canonical_labels(lang):
    assert set(VIABILITY[lang].keys()) == {"Great", "Moderate", "Bad"}


# ── No leftover format placeholders after rendering an explanation ────────────

@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("label", ["Great", "Moderate", "Bad"])
def test_make_explanation_renders(lang, label):
    # One strong anomaly so the "_factors" branch is exercised.
    anomalies = {f"{c}_anomaly": 0.0 for c in CLIMATE_14}
    anomalies["rainfall_mm_year_anomaly"] = 1.5
    text = make_explanation(label, 5.6, 5.3, anomalies, lang)
    assert text and "{" not in text and "}" not in text
    # The localised readable feature name appears in the sentence.
    assert CLIMATE_READABLE[lang]["rainfall_mm_year"] in text

@pytest.mark.parametrize("lang", LANGS)
def test_make_explanation_no_factor_branch(lang):
    # All anomalies below threshold → the "_none" branch (no factors listed).
    anomalies = {f"{c}_anomaly": 0.0 for c in CLIMATE_14}
    text = make_explanation("Great", 5.6, 5.3, anomalies, lang)
    assert text and "{" not in text
