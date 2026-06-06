# 🔥 WPFI — Power Facility Fire Risk Prioritization System

> **Explainable GeoAI model for identifying and prioritizing wildfire-prone power infrastructure**  
> Built for the **2026 KMA Weather Big Data Contest** — Topic 1: Weather-Based Fire Risk Analysis of Power Facilities

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-AUC%200.737%2F0.806-brightgreen)](https://lightgbm.readthedocs.io/)
[![Weather](https://img.shields.io/badge/Weather%20Contribution-94.7%25-blue)](https://github.com/Seungwoo-Kim-kr/wpfi-fire-risk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

WPFI\_v2 (**W**eather-driven **P**ower **F**acility Fire-risk **I**ndex v2) integrates KMA ASOS meteorological data, Canadian FWI fire weather indices, and fire department electrical fire records to compute a **multi-hazard composite risk score** for each power pole — enabling weather-driven prioritized field inspection.

**Coverage:** 1,387,831 power poles in Gangwon Province, South Korea  
**Analysis period:** 2022–2024 (3 years of daily data)  
**Model AUC:** Dry 0.737 / Heat 0.806 (WPFI_v2 Multi-Hazard)  
**Weather Contribution:** 94.7–94.8% (Top-5 features all weather; zero calendar dependency)  
**Formula:** `WPFI_v2 = 0.5 × P_dry + 0.3 × P_heat + 0.2 × P_light`

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

## 🆕 v4.0 — WPFI_v2 Multi-Hazard Model (2026-06)

### 🎯 핵심 혁신: 날씨 조건 기반 멀티해저드 3성분

| 구성요소 | 가중치 | 학습 방식 | AUC | **날씨기여** |
|----------|--------|----------|-----|-----------|
| **P_dry** (건조형) | **50%** | LightGBM | 0.737 | **94.8%** |
| **P_heat** (고온형) | **30%** | LightGBM | 0.806 | **94.7%** |
| **P_light** (낙뢰형) | **20%** | Rule-based | — | 100% |

```
WPFI_v2 = 0.5 × P_dry + 0.3 × P_heat + 0.2 × P_light
```

### 📐 Label 재설계 — 날씨 조건 기반 분류

소방청 발화원인 코드(지오코딩 정밀도 부족) 대신 **실제 화재 발생일 기상 조건**으로 label을 정의합니다.

| | v3.1 (단일 모델) | **v4.0 (멀티해저드)** |
|-|----------------|----------------------|
| Label 기준 | 산불 발생 위치 근접 | **화재 발생일 날씨 조건** |
| 임계값 | — | dry: P75(dry_streak≥5, fwi≥2.7) / heat: temp≥27.2°C |
| 시간 윈도우 | 7일 | **3일** (신호 농도 향상) |
| 양성률 | 1.87% | dry 3.15% / heat 8.41% |
| 날씨기여 | 33% | **94.7–94.8%** |
| 달력 의존도 | month/season 포함 | **0%** (완전 제거) |

### 🌤 날씨 신호 분리도

| 피처 | 건조형 (label=1 vs 0) | 고온형 (label=1 vs 0) |
|------|----------------------|----------------------|
| fwi_score | **+118.9%** ✅ | -21.4% |
| dryness_score | +47.8% ✅ | -26.9% ✅ |
| heat_score | -10.5% ✅ | **+70.7%** ✅ |
| eff_humidity | -23.6% ✅ | +7.5% |
| wind_score | +39.8% | -16.5% ✅ |

> 두 모델이 완벽히 반대 방향의 기상 패턴을 학습 — 물리적으로 타당

### ⚡ 낙뢰 rule v2 — 기상 신호 추가

```python
# 기존: P_light = elevation / 1033  (상관계수 1.0000 — 고도 복사본)
# v4.0: 여름 대기불안정 + 습도 결합
P_light = 0.50 × elevation_norm
        + 0.30 × (summer_temp_range / max)   # 여름 일교차 → 대기불안정
        + 0.20 × (summer_rh / max)            # 여름 습도
# 결과: 상관계수 1.0000 → 0.9818 (날씨 신호 진입)
```

### 🤖 AI 요약 JSON 구조화

```python
# 기존: 자유 형식 텍스트 → 매번 다른 구조
st.info("강원도 전력설비 1,387,831개를 분석한 결과...")

# v4.0: response_format=json_object 강제 + 구조화 렌더러
{
  "headline": "매우높음 69,174개(5.0%) — 원주시 권역 최고 위험",
  "situation": "...",
  "risk_factor": "...",
  "action": "..."
}
```

### 📊 등급 분포 (v4.0)

| 등급 | 임계값 | 설비 수 | 비율 |
|------|--------|---------|------|
| 🔴 Very High | WPFI\_v2 ≥ 36.9 | 69,174개 | 5.0% |
| 🟠 High | 30.2 ~ 36.9 | 139,025개 | 10.0% |
| 🟡 Moderate | 16.9 ~ 30.2 | 345,391개 | 24.9% |
| 🟢 Low | < 16.9 | 834,241개 | 60.1% |

---

## 📋 v3.1 Improvements (2026-06)

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

## 🏗️ Architecture (v4.0)

```
KMA ASOS Daily Weather (11 stations, 2022-2024)
        │
        ▼  FWI calculation / Rolling window / Alert flags
   Weather Features (34)   ←── Canadian FWI (FFMC/DMC/DC/ISI/BUI)
        │                       Accumulated risk / Dryness / Heat scores
        │
ESA WorldCover + Copernicus DEM  → Spatial Features
Power Facility Locations          → Facility Features (cascade_risk)
        │
소방청 전기화재 1,758건 (강원 2022-2024)
        ▼  날씨 조건 기반 분류 (P75 임계값 + 3일 윈도우)
   label_dry  (3.15%): dry_streak≥5 AND fwi≥2.7
   label_heat (8.41%): temp_max≥27.2°C
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│   Model_dry  (LightGBM) — AUC 0.737, 날씨기여 94.8%    │
│   Model_heat (LightGBM) — AUC 0.806, 날씨기여 94.7%    │
│   Rule_light (고도+여름일교차+여름습도)                  │
└─────────────────────────────────────────────────────────┘
        │
        ▼  WPFI_v2 = 0.5×P_dry + 0.3×P_heat + 0.2×P_light
┌─────────────────────────────────────────────────────────┐
│      hazard_combined (0~1) → final_risk_wpfi (0~100)    │
│      Grade: VH≥36.9 / Hi≥30.2 / Mo≥16.9               │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│             Streamlit Dashboard v4.0                     │
│  8 tabs · WPFI_v2 구성요소 차트 · JSON 구조화 AI 요약   │
└─────────────────────────────────────────────────────────┘
```

### WPFI_v2 구성요소

| 구성요소 | 가중치 | 학습 | 주요 변수 | 날씨기여 |
|----------|--------|------|---------|---------|
| P_dry (건조형) | **50%** | LightGBM | fwi_dry_streak(33%), fwi_isi(7%), eff_humidity(7%) | **94.8%** |
| P_heat (고온형) | **30%** | LightGBM | heat_score(53%), accumulated_risk(5%), temp_max(4%) | **94.7%** |
| P_light (낙뢰형) | **20%** | Rule-based | 고도(50%), 여름일교차(30%), 여름습도(20%) | 100% |

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
├── app.py                    # Streamlit dashboard v4.0 (WPFI_v2, JSON AI, 8 tabs)
├── config.py                 # Paths, weights, API keys
├── forecast_engine.py        # KMA forecast API + LightGBM inference
├── regen_labels.py           # Temporal+Spatial label regeneration
├── rebuild_labels_v4.py      # WPFI_v2 날씨 조건 기반 label 생성 파이프라인
├── apply_improvements.py     # 낙뢰 rule v2 + month 제거 재학습
├── improve_labels.py         # P75 임계값 + 3일 윈도우 label 품질 개선
├── regen_figures.py          # 멀티해저드 시각화 재생성
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

## 📊 Model Performance (v4.0)

### WPFI_v2 Multi-Hazard

| 모델 | AUC | Recall@Top10% | 날씨기여 | 달력기여 |
|------|-----|--------------|---------|---------|
| P_dry (건조형) | **0.737** | 0.320 | **94.8%** | **0%** |
| P_heat (고온형) | **0.806** | 0.285 | **94.7%** | **0%** |

### 개선 여정

| 버전 | 모델 | AUC | 날씨기여 |
|------|------|-----|---------|
| v1 (Geographic) | 단일 LightGBM | 0.998 | 14% (과적합) |
| v3.1 (Temporal) | 단일 LightGBM | 0.957 | 33% |
| **v4.0 (Multi-Hazard)** | P_dry + P_heat + P_light | **0.737/0.806** | **94.7~94.8%** |

> AUC가 낮아진 것은 과거 모델이 공간 피처에 과의존했기 때문입니다.  
> v4.0은 **순수 날씨 기반 예측**으로, 날씨 빅데이터 콘테스트 목적에 직접 부합합니다.

### 계절 패턴 (물리적 타당성)

| 계절 | 건조형 label=1 비율 | 고온형 label=1 비율 |
|------|-------------------|-------------------|
| 봄 | 4.0% | 3.4% |
| **여름** | 1.9% | **27.7%** |
| 가을 | 1.5% | 4.7% |
| **겨울** | **5.2%** | 0.0% |

> 건조형은 겨울·봄 건조 시즌, 고온형은 여름 폭염 시즌에 집중 — 물리적으로 타당

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

| 우선순위 | 항목 | 기대 효과 |
|---------|------|---------|
| 단기 | KMA 낙뢰 관측 데이터 수집 → P_light ML화 | 낙뢰 예측 정확도 향상 |
| 단기 | 30일 누적 강수량 피처 추가 | 건조형 AUC 0.75+ 목표 |
| 중기 | WPFI_v2 가중치 scipy 최적화 | 데이터 기반 0.5/0.3/0.2 검증 |
| 중기 | 정밀 GPS 전기화재 데이터 확보 | 공간 분리로 AUC 0.85+ 가능 |
| 장기 | 실시간 스트리밍 파이프라인 (AWS/GCP) | 운영 배포 |

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
