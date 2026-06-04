# 🔥 WPFI — Power Facility Fire Risk Prioritization System

> **Explainable GeoAI model for identifying and prioritizing wildfire-prone power infrastructure**  
> Built for the **2026 KMA Weather Big Data Contest** — Topic 1: Weather-Based Fire Risk Analysis of Power Facilities

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-AUC%200.9574-brightgreen)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

WPFI (**W**eather-Spatial **P**ower **F**acility Fire-risk **I**ndex) integrates meteorological data, Canadian FWI fire weather indices, spatial information (forest cover, terrain), power facility attributes, and historical fire records to compute a composite risk score for each power pole — enabling prioritized field inspection.

**Coverage:** 1,387,831 power poles in Gangwon Province, South Korea  
**Analysis period:** 2022–2024 (3 years of daily data)  
**Model AUC:** 0.9574 (LightGBM Model C — FWI + Accumulated Risk + Cascade)  
**Recall@Top10%:** 82.8% — inspecting top 10% captures 83% of actual fire-risk events

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🗺️ **Interactive Risk Map** | Folium map with ML probability-based 4-tier risk grades for 1.38M facilities |
| 📋 **Priority Inspection List** | Top-K% facilities sorted by combined ML+WPFI score with CSV export |
| 🔍 **Facility Detail + RAG AI** | SHAP waterfall + RAG-enhanced GPT-4o-mini explanation (wildfire cases + KMA standards) |
| 📊 **Model Performance** | AUC, Recall@K, Ablation (A→B→C journey), Region CV (9 stations) |
| 📈 **Spatial Distribution** | Station-level risk heatmap + scatter map + Region CV chart |
| 📅 **Trend Analysis** | Weather hazard trends + FWI seasonal pattern (spring peak visualization) |
| 📡 **3-Day Weather Forecast** | KMA short-term API → LightGBM inference → fire risk prediction |
| 🧮 **Methodology** | FWI formula (FFMC/DMC/DC/ISI/BUI), label design, cascade risk, RAG system |
| 🌦️ **What-if Simulation** | 6 scenario presets with real-time grade recalculation |
| 🌐 **Bilingual** | Korean / English toggle |

---

## 🆕 v3.1 Improvements (2026-06)

### 🤖 Model C — Physics-based Fire Risk

**Label redesign (core contribution):**

| | Old (Geographic) | New (Temporal + Spatial) |
|-|-----------------|--------------------------|
| label=1 condition | Near any historical fire = ALL dates | Within ±14 days of wildfire + 500m radius |
| label=1 ratio | 9.1% (278,636 rows) | 1.9% (57,036 rows) |
| Weather dependency | None | Yes — dry/windy days concentrated |
| AUC | 0.9982 (overfitting suspected) | **0.9574** (reliable) |
| Weather SHAP contribution | ~14% | **33%** |

**4 new features added:**

| Feature | Description |
|---------|-------------|
| `fwi_score / fwi_bui / fwi_isi` | Canadian FWI — physically-based fire danger |
| `accumulated_risk_norm` | 14-day exponential decay accumulated risk |
| `cascade_risk` | 300m BallTree network → cascade failure risk |
| `terrain_fire_risk` | Forest(40%) + elevation(30%) + land cover(30%) |

### 📐 Canadian FWI System

International standard adopted by KMA and forestry agencies worldwide:

| Code | Name | Inputs | Meaning |
|------|------|--------|---------|
| FFMC | Fine Fuel Moisture Code | Temp, RH, Wind, Precip | Fine fuel moisture |
| DMC | Duff Moisture Code | Temp, RH, Precip (14-day) | Duff layer moisture |
| DC | Drought Code | Temp, Precip (30-day) | Deep soil dryness |
| ISI | Initial Spread Index | Wind + FFMC | Fire spread speed |
| BUI | Buildup Index | DMC + DC | Fuel accumulation |
| FWI | Fire Weather Index | ISI + BUI | Final fire danger |

Peak recorded: Daegwallyeong station, 2022-03-05 → **FWI 29.1** (Very High)

### ⚡ Risk Grade Redesign

ML probability absolute-value thresholds (not arbitrary percentile slicing):

| Grade | ML Probability | Count | Meaning |
|-------|--------------|-------|---------|
| 🔴 Very High | ≥ 0.20 | 5,635 (0.41%) | Actual wildfire conditions |
| 🟠 High | 0.05 ~ 0.20 | 7,847 (0.57%) | Elevated weather risk |
| 🟡 Moderate | 0.01 ~ 0.05 | 20,422 (1.47%) | Above-average caution |
| 🟢 Low | < 0.01 | 1,353,927 (97.56%) | Normal range |

### 🔮 RAG-Enhanced AI Explanations

```
Facility selected
    → FWI / accumulated_risk / cascade_risk → query string
    → TF-IDF search over 106 documents
        ├── 100 wildfire cases (Gangwon, Korea Forest Service 2022-2024)
        ├── KMA weather warning standards (dry/wind alert thresholds)
        └── Domain knowledge (FWI interpretation, cascade failure, etc.)
    → Retrieved context injected into GPT-4o-mini prompt
    → Evidence-based, grade-differentiated explanation
```

### 🎨 UI Overhaul

- **Dark theme** via `.streamlit/config.toml`
- **Glassmorphism KPI cards** — Very High / High / Low counts + AUC + Recall@Top10%
- **Hero header** — gradient background + badge tags
- **Sidebar** — WPFI logo branding + progress bar risk breakdown + model status card
- **8 tabs** (was 7) — 🧮 산출방식 / Methodology tab added with FWI formulas (`st.latex`)

---

## 🏗️ Architecture

```
KMA ASOS Daily Weather (11 stations, 2022-2024)
        │
        ▼  FWI calculation / Rolling window / Alert flags
   Weather Features (18)   ←── Canadian FWI (FFMC/DMC/DC/ISI/BUI)
        │                       Accumulated risk (14-day decay)
        │
ESA WorldCover + Copernicus DEM  → Spatial Features (4)
Power Facility Locations          → Facility Features (4)
                                       cascade_risk (300m BallTree)
Wildfire records (KFS 2022-2024)  → Temporal+Spatial Label
   (±14 days + 500m radius)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│            LightGBM Model C (39 features)               │
│  AUC 0.9574 | Recall@Top10% 82.8%                      │
│  Weather+FWI SHAP: 33% | Region CV avg: 0.977           │
└─────────────────────────────────────────────────────────┘
        │
        ▼  ML probability × Rule-based WPFI (60:40 blend)
┌─────────────────────────────────────────────────────────┐
│           Combined Risk Score (0-100) + Grade           │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│             Streamlit Dashboard v3.1                     │
│  8 tabs · Dark theme · Glassmorphism KPI · RAG AI       │
└─────────────────────────────────────────────────────────┘
```

### Risk Score Components

| Layer | Weight | Key Variables |
|-------|--------|---------------|
| Weather Hazard | **35%** | FWI, dryness score, wind score, accumulated risk, effective humidity |
| Spatial Exposure | **30%** | Elevation, terrain fire exposure, forest proximity |
| Facility Exposure | **25%** | Facility density, cascade risk (300m BallTree network) |
| Historical Prior | **10%** | Past wildfire count (Korea Forest Service) |

---

## 🚀 Quick Start

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

> **Note:** Processed data files (`data/processed/`) are required. To reproduce from scratch, run notebooks 01–09 in order after obtaining contest data from [날씨마루](https://bd.kma.go.kr). To regenerate labels only: `python regen_labels.py`

---

## 📁 Project Structure

```
wpfi-fire-risk/
├── app.py                    # Streamlit dashboard v3.1 (8 tabs, dark theme, RAG)
├── config.py                 # Paths, weights, API keys
├── forecast_engine.py        # KMA forecast API + LightGBM inference
├── regen_labels.py           # Temporal+Spatial label regeneration script
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
│   ├── 05_risk_index.ipynb               # 4-layer WPFI index
│   ├── 06_modeling.ipynb                 # Model C: LightGBM + scale_pos_weight
│   ├── 07_explainability.ipynb           # SHAP analysis
│   ├── 08_validation.ipynb               # Ablation, Region CV, external val.
│   ├── 09_mapping_outputs.ipynb          # Folium maps + figures
│   └── 10_llm_explanation.ipynb          # GPT-4o-mini + RAG integration
│
├── data/
│   ├── raw/                  # Contest data (not in repo)
│   ├── processed/            # Parquet/pkl/gpkg files (not in repo)
│   └── external/             # Public datasets (see Data Sources)
│
└── outputs/
    ├── figures/              # PNG charts (Apple SD Gothic Neo Korean font)
    ├── maps/                 # Folium HTML maps
    └── tables/               # CSV result tables
```

---

## 📊 Model Performance

| Metric | Value | Note |
|--------|-------|------|
| AUC | **0.9574** | LightGBM Model C |
| Recall@Top5% | 0.628 | 5% inspection → 63% fire events caught |
| Recall@Top10% | **0.828** | 10% inspection → 83% caught |
| Recall@Top20% | 0.996 | 20% inspection → ~100% caught |
| Region CV AUC | **0.977 avg** | 9 stations, all ≥ 0.91 — no regional bias |
| Weather+FWI SHAP | **33%** | vs. 14% in old geographic-label model |
| Weather-only AUC | 0.659 | FWI alone has independent predictive power |

### Ablation Study

| Step | Features | AUC |
|------|----------|-----|
| M1 | Weather + FWI only | 0.659 |
| M2 | + Spatial (terrain, forest) | 0.848 |
| M3 | + Facility (density, cascade) | 0.903 |
| **M4 (adopted)** | All 39 features | **0.957** |

---

## 📊 Data Sources

| Dataset | Source | Purpose |
|---------|--------|---------|
| Power facility locations (Gangwon) | 날씨마루 (KMA contest) | Base facility data |
| Daily weather observations (ASOS) | KMA / 날씨마루 | Weather + FWI features |
| Wildfire statistics 2022–2024 | Korea Forest Service | Temporal+Spatial label design |
| Electrical fire records 2020–2024 | [NFDS](https://www.nfds.go.kr) | Supplementary validation |
| ESA WorldCover 10m | [AWS Open Data](https://registry.opendata.aws/esa-worldcover/) | Forest/land cover features |
| Copernicus DEM 30m | [AWS Open Data](https://registry.opendata.aws/copernicus-dem/) | Elevation + terrain |
| Admin boundaries | [KOSTAT](https://sgis.kostat.go.kr) | Visualization |

---

## 🌦️ Weather Simulation Scenarios

| Scenario | WH Multiplier | Conditions |
|----------|--------------|------------|
| Baseline | ×1.00 | Actual observed data |
| Spring Dry-Windy | ×1.35 | Low humidity + strong wind |
| Summer Heatwave | ×1.20 | High temperature |
| Autumn Dry | ×1.40 | 14-day no-rain |
| Winter Strong Wind | ×1.15 | Cold + gusty |
| Worst-Case Composite | ×1.70 | Extreme drought + wind + heat |

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| ML / XAI | LightGBM, scikit-learn, SHAP |
| Spatial | GeoPandas, Shapely, scikit-learn BallTree |
| Visualization | Matplotlib (Apple SD Gothic Neo), Folium, streamlit-folium |
| Web App | Streamlit 1.58 (dark theme, custom CSS Glassmorphism) |
| Data | Pandas 3.x, NumPy 2.x, PyArrow |
| LLM + RAG | OpenAI GPT-4o-mini + TF-IDF retrieval (106 documents) |
| Fire Index | Canadian FWI System (FFMC/DMC/DC/ISI/BUI) |
| Weather API | KMA API Hub |
| Report | ReportLab PDF |

---

## 🗺️ Future Work

| Priority | Item | Expected Benefit |
|----------|------|-----------------|
| Short-term | Wind direction data → fire propagation direction feature | Dynamic directional risk |
| Short-term | 날씨마루 official data integration → full pipeline re-run | Final contest performance |
| Mid-term | Knowledge Graph → power network topology cascade analysis | KEPCO grid-level impact |
| Mid-term | RAG corpus expansion (KMA alert history) | Stronger AI explanation evidence |
| Long-term | Real-time streaming pipeline (AWS/GCP) | Operational deployment |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **Korea Meteorological Administration (KMA)** — weather data & 날씨마루 platform
- **Korea Forest Service (KFS)** — wildfire statistics for label design
- **Natural Resources Canada** — Canadian FWI System methodology
- **ESA / Copernicus** — WorldCover 10m and DEM via AWS Open Data
- **National Fire Data System** — electrical fire records
