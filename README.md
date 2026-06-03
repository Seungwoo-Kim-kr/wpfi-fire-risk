# 🔥 WPFI — Power Facility Fire Risk Prioritization System

> **Explainable GeoAI model for identifying and prioritizing wildfire-prone power infrastructure**  
> Built for the **2026 KMA Weather Big Data Contest** — Topic 1: Weather-Based Fire Risk Analysis of Power Facilities

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-AUC%200.9982-brightgreen)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

WPFI (**W**eather-Spatial **P**ower **F**acility Fire-risk **I**ndex) integrates meteorological data, spatial information (forest cover, terrain), power facility attributes, and historical fire records to compute a 0–100 risk score for each power pole, enabling prioritized field inspection.

**Coverage:** 1,387,831 power poles in Gangwon Province, South Korea  
**Analysis period:** 2022–2024 (3 years of daily data)  
**Model AUC:** 0.9982

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🗺️ **Interactive Risk Map** | Folium-based map with 4-tier risk color coding for 1.38M facilities |
| 📋 **Priority Inspection List** | Top-K% facilities sorted by risk score with CSV export |
| 🔍 **Facility Detail Analysis** | SHAP waterfall chart + AI-generated risk explanation per facility |
| 📊 **Model Performance** | AUC, Recall@K, Ablation Study, Region CV with plain-language summaries |
| 📈 **Spatial Distribution** | Station-level risk heatmap and grade distribution charts |
| 📅 **Trend Analysis** | Monthly/seasonal weather hazard trends with station × month heatmap |
| 📡 **3-Day Weather Forecast** | KMA short-term forecast API → LightGBM inference → fire risk prediction |
| 🌦️ **What-if Simulation** | 6 scenario presets (spring dry wind, summer heatwave, worst-case, etc.) |
| 🤖 **AI Summaries** | OpenAI GPT-4o-mini generates natural-language explanations throughout |
| 🌐 **Bilingual** | Korean / English toggle (top of sidebar) |

---

## 🏗️ Architecture

```
Weather Data (KMA ASOS)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│               4-Layer Feature Engineering            │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │Weather Hazard│  │Spatial Exposure│               │
│  │  (35 weight) │  │  (30 weight) │                 │
│  └──────────────┘  └──────────────┘                 │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │  Facility    │  │ Historical   │                 │
│  │  Exposure    │  │    Prior     │                 │
│  │  (25 weight) │  │  (10 weight) │                 │
│  └──────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────┐      ┌──────────────────┐
│  LightGBM Model  │ ───► │  WPFI Risk Score  │
│  (AUC: 0.9982)   │      │    (0 – 100)      │
└──────────────────┘      └──────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│          Streamlit Dashboard             │
│  Risk Map · Inspection List · Forecast   │
│  SHAP Analysis · AI Explanations         │
└──────────────────────────────────────────┘
```

### Risk Score Components

| Layer | Weight | Key Variables |
|-------|--------|---------------|
| Weather Hazard | 35% | Dryness score, wind score, heat score, no-rain days, weather alert flags |
| Spatial Exposure | 30% | Forest ratio (ESA WorldCover 10m), slope/elevation (DEM 30m), fire count |
| Facility Exposure | 25% | Facility density, distance to nearest facility |
| Historical Prior | 10% | Past wildfire count, past electrical fire count |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- ~4GB disk space for processed data
- API keys (see [Configuration](#configuration))

### Installation

```bash
git clone https://github.com/Seungwoo-Kim-kr/wpfi-fire-risk.git
cd wpfi-fire-risk

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
# Korea Meteorological Administration API (apihub.kma.go.kr)
# Required for the 3-day weather forecast tab
KMA_API_KEY=your_kma_api_key_here

# OpenAI API (optional — enables AI explanations throughout the app)
OPENAI_API_KEY=your_openai_api_key_here
```

**Getting API keys:**

| Key | Where to get | Required? |
|-----|-------------|-----------|
| `KMA_API_KEY` | [KMA API Hub](https://apihub.kma.go.kr) → Sign up → Apply for *단기예보 격자 API* | For forecast tab |
| `OPENAI_API_KEY` | [OpenAI Platform](https://platform.openai.com) | For AI summaries |

### Run the App

```bash
# Make sure .venv is activated
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** The processed data files (`data/processed/`) are required to run the app. If you are reproducing results from scratch, run notebooks `01` through `09` in order after obtaining the contest data from [날씨마루](https://bd.kma.go.kr).

---

## 📁 Project Structure

```
wpfi-fire-risk/
├── app.py                    # Streamlit dashboard (main entry point)
├── config.py                 # Paths, API keys, model parameters
├── forecast_engine.py        # KMA forecast API + LightGBM inference pipeline
├── collect_weather.py        # KMA ASOS weather data collection
├── fix_and_merge_weather.py  # Weather data preprocessing
├── generate_report.py        # PDF report generator (ReportLab)
├── requirements.txt
│
├── notebooks/
│   ├── 01_data_check.ipynb               # Data exploration + Hive connection
│   ├── 02_spatial_preprocessing.ipynb    # CRS unification + spatial matching
│   ├── 03_buffer_feature_engineering.ipynb  # Multi-scale buffer features
│   ├── 04_weather_feature_engineering.ipynb # Weather features + hazard scores
│   ├── 05_risk_index.ipynb               # 4-layer risk index + grading
│   ├── 06_modeling.ipynb                 # LightGBM training + SHAP
│   ├── 07_explainability.ipynb           # SHAP analysis
│   ├── 08_validation.ipynb               # Ablation, sensitivity, external val.
│   ├── 09_mapping_outputs.ipynb          # Folium maps + figure export
│   └── 10_llm_explanation.ipynb          # GPT-4o-mini integration
│
├── data/
│   ├── raw/                  # Contest data (not in repo — see Data Sources)
│   ├── processed/            # Generated parquet/pkl/gpkg files (not in repo)
│   └── external/             # Publicly downloadable datasets (see below)
│
└── outputs/
    ├── figures/              # PNG charts for reports
    ├── maps/                 # Folium HTML maps
    └── tables/               # CSV result tables (included in repo)
```

---

## 📊 Data Sources

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| Power facility locations (Gangwon) | 날씨마루 (KMA contest platform) | ~350MB | Facility lat/lon + attributes |
| Daily weather observations (ASOS) | KMA / 날씨마루 | ~2MB | Weather hazard features |
| Electrical fire records 2020–2024 | National Fire Data System ([NFDS](https://www.nfds.go.kr)) | 22MB | Historical prior features |
| Wildfire statistics 2022–2024 | Korea Forest Service | 191KB | Historical prior features |
| ESA WorldCover 10m (land cover) | [AWS Open Data](https://registry.opendata.aws/esa-worldcover/) | 146MB | Forest ratio features |
| Copernicus DEM 30m | [AWS Open Data](https://registry.opendata.aws/copernicus-dem/) | 767MB | Slope / elevation features |
| Admin boundaries (sido/sigungu) | [KOSTAT](https://sgis.kostat.go.kr) | 79MB | Visualization + geocoding |

> **Note:** The contest data (power facility locations) is only available through the 날씨마루 platform to registered participants. All other datasets are publicly available at the links above.

---

## 🌦️ Weather Simulation Scenarios

| Scenario | Weather Hazard Multiplier | Conditions |
|----------|--------------------------|------------|
| Current Data (Baseline) | ×1.00 | Actual observed data |
| Spring Dry-Windy | ×1.35 | Humidity −20%, Wind +5 m/s |
| Summer Heatwave | ×1.20 | Temperature +8°C, Humidity −10% |
| Autumn Dry | ×1.40 | 14-day no-rain, Humidity −25% |
| Winter Strong Wind | ×1.15 | Wind +8 m/s, Temperature −10°C |
| Worst-Case Composite | ×1.70 | Extreme drought + strong wind + heat |

Risk grades are recalculated dynamically based on the original data's percentile thresholds.

---

## 📡 3-Day Forecast Pipeline

```
KMA API Hub (apihub.kma.go.kr)
    ↓  Grid forecast API (nph-dfs_shrt_grd)
    ↓  5 variables × 3 days × 3 time slots = 45 parallel calls
    ↓  ThreadPoolExecutor(max_workers=8) → ~20 seconds
Daily Aggregation (temp_max, rh_min, ws_max, precip_sum)
    ↓
Feature Computation (dryness_score, wind_score, heat_score, ...)
    ↓
LightGBM Inference (1.2M facilities × static features merged)
    ↓
Forecast Risk Score + Grade + OpenAI Recommendation
```

---

## 🔬 Model Performance

| Metric | Value |
|--------|-------|
| AUC (ROC) | **0.9982** |
| Recall@Top5% | See app → Model Performance tab |
| Region CV AUC (avg) | See app → Model Performance tab |
| External Validation | Spatial overlap with actual fire events |

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| ML / XAI | LightGBM, scikit-learn, SHAP |
| Spatial | GeoPandas, Shapely, PyProj, Rasterio |
| Visualization | Matplotlib, Folium, streamlit-folium |
| Web App | Streamlit |
| Data | Pandas, NumPy, PyArrow |
| LLM | OpenAI GPT-4o-mini |
| Weather API | KMA API Hub (apihub.kma.go.kr) |

---

## 📝 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **Korea Meteorological Administration (KMA)** for weather data and the 날씨마루 platform
- **Korea Electric Power Corporation (KEPCO)** for power facility data (contest-only)
- **ESA** for WorldCover 10m global land cover data
- **Copernicus / ESA** for DEM data via AWS Open Data
- **National Fire Data System** for fire occurrence statistics
