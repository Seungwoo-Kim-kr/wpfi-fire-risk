# 🔥 WPFI — Power Facility Fire Risk Prioritization System

> **Explainable GeoAI model for identifying and prioritizing wildfire-prone power infrastructure**
> Built for the **2026 KMA Weather Big Data Contest** — Topic 1: Weather-Based Fire Risk Analysis of Power Facilities

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-AUC%200.737%2F0.806-brightgreen)](https://lightgbm.readthedocs.io/)
[![Weather](https://img.shields.io/badge/Weather%20Contribution-94.7%25-blue)](https://github.com/Seungwoo-Kim-kr/wpfi-fire-risk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**WPFI\_v2** (Weather-driven Power Facility Fire-risk Index v2) integrates KMA ASOS meteorological data, Canadian FWI fire weather indices, and fire department electrical fire records to compute a **multi-hazard composite risk score** for each power pole — enabling weather-driven prioritized field inspection.

| Attribute | Value |
|-----------|-------|
| Coverage | 1,387,831 power poles — Gangwon Province, South Korea |
| Analysis period | 2022–2024 (3 years of daily observations) |
| Formula | `WPFI_v2 = 0.5 × P_dry + 0.3 × P_heat + 0.2 × P_light` |
| Dry model AUC | **0.737** |
| Heat model AUC | **0.806** |
| Weather contribution | **94.7–94.8%** (Top-5 features all weather; zero calendar dependency) |

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🗺️ **Interactive Risk Map** | Folium map with WPFI\_v2-based 4-tier risk grades for 1.38M facilities |
| 📋 **Priority Inspection List** | Top-K% facilities sorted by WPFI\_v2 score with CSV export |
| 🔍 **Facility Detail + RAG AI** | SHAP waterfall + RAG-enhanced GPT-4o-mini explanation |
| 📊 **Model Performance** | AUC, Recall@K, weather contribution %, zero calendar dependency |
| 🌦️ **Multi-Hazard Weather Simulator** | 8 weather conditions (rain/storm/heatwave/tropical night/drought...) + direct sliders |
| 📅 **Trend Analysis** | Weather hazard trends + FWI seasonal pattern |
| 📡 **3-Day Forecast** | KMA API → LightGBM inference → fire risk prediction |
| 🧮 **Methodology** | FWI formula, WPFI\_v2 label design, cascade risk, RAG system |
| 🌐 **Bilingual** | Korean / English toggle |

---

## What's New — v4.0 (2026-06)

### Multi-Hazard 3-Component Model

**WPFI\_v2 = 0.5 × P\_dry + 0.3 × P\_heat + 0.2 × P\_light**

| Component | Weight | Method | AUC | Weather Contribution |
|-----------|--------|--------|-----|---------------------|
| P\_dry (Dryness) | 50% | LightGBM | **0.737** | **94.8%** |
| P\_heat (Heat) | 30% | LightGBM | **0.806** | **94.7%** |
| P\_light (Lightning) | 20% | Rule-based | — | 100% |

### Weather-Condition Label Design

Instead of cause-code classification from fire department records (poor geocoding precision), labels are defined by **actual weather conditions on fire occurrence dates**.

| | v3.1 (Single Model) | **v4.0 (Multi-Hazard)** |
|-|---------------------|------------------------|
| Label basis | Wildfire proximity | **Weather condition on fire date** |
| Threshold | — | Dry: P75 (dry\_streak≥5, fwi≥2.7) / Heat: temp≥27.2°C |
| Time window | 7 days | **3 days** (higher signal density) |
| Positive rate | 1.87% | dry 3.15% / heat 8.41% |
| Weather contribution | 33% | **94.7–94.8%** |
| Calendar dependency | month/season included | **0%** (fully removed) |

### Weather Signal Separation

| Feature | Dry model (label=1 vs 0) | Heat model (label=1 vs 0) |
|---------|--------------------------|---------------------------|
| fwi\_score | **+118.9%** ✅ | −21.4% |
| dryness\_score | +47.8% ✅ | −26.9% ✅ |
| heat\_score | −10.5% ✅ | **+70.7%** ✅ |
| eff\_humidity | −23.6% ✅ | +7.5% |
| wind\_score | +39.8% | −16.5% ✅ |

Both models learn perfectly **opposite** weather patterns — physically sound.

### Seasonal Pattern Validation

| Season | Dry model positive rate | Heat model positive rate |
|--------|------------------------|--------------------------|
| Spring | 4.0% | 3.4% |
| **Summer** | 1.9% | **27.7%** |
| Autumn | 1.5% | 4.7% |
| **Winter** | **5.2%** | **0.0%** |

> Dry model peaks in winter/spring (dry season). Heat model peaks in summer (heatwave season). Physically validated.

### Lightning Rule v2 — P\_light

```python
# Before: P_light = elevation / 1033  (correlation 1.0000 — elevation copy)
# v4.0: weather signal added
P_light = 0.50 × elevation_norm
        + 0.30 × (summer_temp_range / max)   # temp amplitude → atmospheric instability
        + 0.20 × (summer_rh / max)            # summer humidity
# Result: correlation 1.0000 → 0.9818 (weather signal enters)
```

### Improved Weather Simulator

8 weather condition types with direct parameter sliders:
- **Weather types**: ☀️ Clear / 🌧 Rain / 🌦 Heavy Rain / ⛈ Thunderstorm / 🌡 Heatwave / 🌙 Tropical Night / 🏜 Extreme Drought / 🌬 Strong Wind
- **Direct sliders**: Temperature (±15°C), Humidity (±40%), Precipitation (mm/day), FWI multiplier (×0.1–5.0)
- **Real-time preview**: P\_dry / P\_heat / P\_light component change (%) displayed live

### Structured AI Summaries

```python
# Before: free-form text
st.info("Gangwon Province analysis shows 69,174 (5.0%) Very High risk...")

# v4.0: response_format=json_object enforced
{
  "headline": "Very High 69,174 (5.0%) — Wonju area highest risk",
  "situation": "1,387,831 facilities analyzed...",
  "risk_factor": "Dry/FWI conditions, high-altitude lightning exposure",
  "action": "Immediate inspection for Very High; pre-check insulation in dry-alert zones"
}
```

### Risk Grade Distribution (v4.0)

| Grade | Threshold | Count | Ratio |
|-------|-----------|-------|-------|
| 🔴 Very High | WPFI\_v2 ≥ 36.9 | 69,174 | 5.0% |
| 🟠 High | 30.2 ~ 36.9 | 139,025 | 10.0% |
| 🟡 Moderate | 16.9 ~ 30.2 | 345,391 | 24.9% |
| 🟢 Low | < 16.9 | 834,241 | 60.1% |

---

## Architecture (v4.0)

```
KMA ASOS Daily Weather (11 stations, Gangwon 2022–2024)
        │
        ▼  FWI calculation / Rolling window / Alert flags
   Weather Features (34)   ←── Canadian FWI (FFMC/DMC/DC/ISI/BUI)
        │                       Accumulated risk / Dryness / Heat scores
        │                       Calendar features REMOVED (month, season)
        │
ESA WorldCover + Copernicus DEM  → Spatial Features
Power Facility Locations          → Facility Features (cascade_risk 300m BallTree)
        │
NFDS Electrical Fire Records 1,758 cases (Gangwon 2022–2024)
        ▼  Weather-condition classification (P75 threshold + 3-day window)
   label_dry  (3.15%): dry_streak≥5 AND fwi≥2.7
   label_heat (8.41%): temp_max≥27.2°C
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│   Model_dry  (LightGBM) — AUC 0.737, Weather 94.8%     │
│   Model_heat (LightGBM) — AUC 0.806, Weather 94.7%     │
│   Rule_light (elevation + summer instability + humidity) │
└─────────────────────────────────────────────────────────┘
        │
        ▼  WPFI_v2 = 0.5×P_dry + 0.3×P_heat + 0.2×P_light
┌─────────────────────────────────────────────────────────┐
│      hazard_combined (0–1) → final_risk_wpfi (0–100)    │
│      Grade: VH≥36.9 / High≥30.2 / Moderate≥16.9       │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│             Streamlit Dashboard v4.0                     │
│  8 tabs · WPFI_v2 Simulator · JSON AI Summaries         │
└─────────────────────────────────────────────────────────┘
```

### WPFI\_v2 Components

| Component | Weight | Method | Top Features | Weather % |
|-----------|--------|--------|-------------|-----------|
| P\_dry | 50% | LightGBM | fwi\_dry\_streak (33%), fwi\_isi (7%), eff\_humidity (7%) | **94.8%** |
| P\_heat | 30% | LightGBM | heat\_score (53%), accumulated\_risk (5%), temp\_max (4%) | **94.7%** |
| P\_light | 20% | Rule-based | elevation (50%), summer temp range (30%), summer humidity (20%) | 100% |

---

## Model Performance

### WPFI\_v2 Multi-Hazard

| Model | AUC | Recall@Top10% | Weather Contribution | Calendar Dependency |
|-------|-----|--------------|---------------------|---------------------|
| P\_dry (Dryness) | **0.737** | 0.320 | **94.8%** | **0%** |
| P\_heat (Heat) | **0.806** | 0.285 | **94.7%** | **0%** |

### Improvement Journey

| Version | Model | AUC | Weather |
|---------|-------|-----|---------|
| v1 (Geographic label) | Single LightGBM | 0.998 | 14% (overfitting) |
| v3.1 (Temporal label) | Single LightGBM | 0.957 | 33% |
| **v4.0 (Multi-Hazard)** | P\_dry + P\_heat + P\_light | **0.737/0.806** | **94.7–94.8%** |

> The AUC drop from v3.1 to v4.0 is expected: prior models were dominated by spatial features (facility\_density).
> v4.0 achieves **genuine weather-driven prediction**, directly aligned with the Weather Big Data Contest objective.

### Regional Risk Profiles (Top-5)

| Municipality | WPFI\_v2 | P\_dry | P\_heat | P\_light |
|-------------|---------|-------|--------|---------|
| Wonju | 0.343 | 0.462 | 0.053 | 0.479 |
| Taebaek | 0.327 | 0.316 | 0.007 | 0.834 |
| Inje | 0.222 | 0.078 | 0.108 | 0.753 |
| Samcheok | 0.219 | 0.184 | 0.006 | 0.629 |
| Hoengseong | 0.214 | 0.120 | 0.124 | 0.584 |

---

## Quick Start

### Prerequisites
- Python 3.12+
- ~4 GB disk space for processed data

### Installation

```bash
git clone https://github.com/Seungwoo-Kim-kr/wpfi-fire-risk.git
cd wpfi-fire-risk

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
# KMA API Hub (apihub.kma.go.kr) — for 3-day forecast tab
KMA_API_KEY=your_kma_api_key_here

# OpenAI API — for RAG-enhanced AI explanations
OPENAI_API_KEY=your_openai_api_key_here
```

### Run

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

> **Note:** Processed data files (`data/processed/`) are required.
> To reproduce labels from scratch: `python rebuild_labels_v4.py`
> To apply model improvements: `python apply_improvements.py`
> To regenerate figures: `python regen_figures.py`

---

## Project Structure

```
wpfi-fire-risk/
├── app.py                    # Streamlit dashboard v4.0 (8 tabs, WPFI_v2, JSON AI)
├── config.py                 # Paths, weights, API keys
├── forecast_engine.py        # KMA forecast API + LightGBM inference
├── generate_report.py        # PDF report generator (ReportLab)
├── regen_labels.py           # Temporal+Spatial label regeneration
├── rebuild_labels_v4.py      # WPFI_v2 weather-condition label pipeline
├── apply_improvements.py     # Lightning rule v2 + month removal retraining
├── improve_labels.py         # P75 threshold + 3-day window quality improvement
├── regen_figures.py          # Multi-hazard visualization regeneration
├── requirements.txt
│
├── .streamlit/
│   └── config.toml           # Dark theme + server configuration
│
├── notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_spatial_preprocessing.ipynb
│   ├── 03_buffer_feature_engineering.ipynb
│   ├── 04_weather_feature_engineering.ipynb
│   ├── 05_risk_index.ipynb
│   ├── 06_modeling.ipynb
│   ├── 07_explainability.ipynb
│   ├── 08_validation.ipynb
│   ├── 09_mapping_outputs.ipynb
│   └── 10_llm_explanation.ipynb
│
├── data/
│   ├── raw/                  # Contest data (not in repo)
│   ├── processed/            # Parquet/pkl/gpkg files (not in repo)
│   └── external/             # Public datasets
│
├── outputs/
│   ├── figures/              # PNG charts (fig_multihazard_*.png)
│   ├── maps/                 # Folium HTML maps
│   └── tables/               # CSV result tables
│
└── reports/
    └── WPFI_analysis_report_v4.0.pdf
```

---

## Canadian FWI System

International standard adopted by KMA and forestry agencies worldwide:

| Code | Name | Inputs | Meaning |
|------|------|--------|---------|
| FFMC | Fine Fuel Moisture Code | Temp, RH, Wind, Precip | Fine fuel moisture |
| DMC | Duff Moisture Code | Temp, RH, Precip (14-day) | Duff layer moisture |
| DC | Drought Code | Temp, Precip (30-day) | Deep soil dryness |
| ISI | Initial Spread Index | Wind + FFMC | Fire spread speed |
| BUI | Buildup Index | DMC + DC | Fuel accumulation |
| **FWI** | **Fire Weather Index** | ISI + BUI | **Final fire danger** |

---

## Weather Simulation Scenarios

| Condition | P\_dry mult | P\_heat mult | P\_light mult | Description |
|-----------|------------|-------------|--------------|-------------|
| ☀️ Clear/Normal | ×1.0 | ×1.0 | ×1.0 | Baseline |
| 🌧 Rain (Moderate) | ×0.35 | ×0.88 | ×0.85 | 10–30 mm/day |
| 🌦 Heavy Rain | ×0.10 | ×0.80 | ×0.70 | 50mm+ |
| ⛈ Thunderstorm | ×0.45 | ×1.10 | ×1.80 | Strong wind + rain + lightning |
| 🌡 Heatwave | ×1.55 | ×2.20 | ×1.15 | Temp 35°C+ |
| 🌙 Tropical Night | ×1.20 | ×1.75 | ×1.05 | Night temp ≥25°C |
| 🏜 Extreme Drought | ×2.80 | ×1.30 | ×1.00 | 14-day no-rain, eff. humidity <25% |
| 🌬 Strong Wind | ×1.10 | ×1.05 | ×1.30 | Gust 15 m/s+ |

---

## Data Sources

| Dataset | Source | Purpose |
|---------|--------|---------|
| Power facility locations (Gangwon) | KMA Weather Big Data Contest (날씨마루) ✓ Approved | Base facility data |
| Daily weather observations (ASOS) | KMA / 날씨마루 | Weather + FWI features |
| Electrical fire records 2022–2024 | National Fire Data System (NFDS) | Weather-condition label design |
| Wildfire statistics 2022–2024 | Korea Forest Service | Supplementary label validation |
| ESA WorldCover 10m | [AWS Open Data](https://registry.opendata.aws/esa-worldcover/) | Forest / land cover features |
| Copernicus DEM 30m | [AWS Open Data](https://registry.opendata.aws/copernicus-dem/) | Elevation + terrain |
| Admin boundaries | KOSTAT | Visualization |

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| ML / XAI | LightGBM, scikit-learn, SHAP |
| Spatial | GeoPandas, Shapely, scikit-learn BallTree |
| Visualization | Matplotlib, Folium, streamlit-folium |
| Web App | Streamlit 1.58 (dark theme, Glassmorphism CSS) |
| Data | Pandas 3.x, NumPy 2.x, PyArrow |
| LLM + RAG | OpenAI GPT-4o-mini + TF-IDF (106 documents), JSON structured output |
| Fire Index | Canadian FWI System (FFMC/DMC/DC/ISI/BUI) |
| Weather API | KMA API Hub |
| Reports | ReportLab PDF |

---

## Future Work

| Priority | Item | Expected Benefit |
|----------|------|-----------------|
| Short-term | KMA lightning observation data → P\_light ML model | Improved lightning prediction accuracy |
| Short-term | 30-day accumulated precipitation feature | Dry model AUC 0.75+ target |
| Medium-term | WPFI\_v2 weight optimization (scipy) | Data-driven 0.5/0.3/0.2 validation |
| Medium-term | Precision GPS electrical fire data | Spatial separation → AUC 0.85+ possible |
| Long-term | Real-time streaming pipeline (AWS/GCP) | Operational deployment |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **Korea Meteorological Administration (KMA)** — weather data & 날씨마루 platform
- **Korea Forest Service (KFS)** — wildfire statistics
- **National Fire Data System (NFDS)** — electrical fire records for label design
- **Natural Resources Canada** — Canadian FWI System methodology
- **ESA / Copernicus** — WorldCover 10m and DEM via AWS Open Data
