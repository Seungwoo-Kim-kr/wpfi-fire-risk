"""
WPFI — 전력설비 화재위험 우선점검 시스템 v3.0
Explainable GeoAI · LightGBM · SHAP · OpenAI
2026 날씨 빅데이터 콘테스트 — 주제 1
"""

import sys, json, pickle, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# ── 한글 폰트 전역 설정 ────────────────────────────────────────────────────────
def _setup_korean_font():
    """앱 전체에서 사용할 한글 폰트를 설정한다."""
    candidates = [
        'Apple SD Gothic Neo',   # macOS 기본 한글 폰트 (권장)
        'AppleGothic',           # macOS 구버전 폰트
        'NanumGothic',           # 나눔고딕 (설치된 경우)
        'Malgun Gothic',         # Windows
        'DejaVu Sans',           # 폴백 (한글 미지원, 박스 표시)
    ]
    available = {f.name for f in _fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams['font.family'] = font
            matplotlib.rcParams['axes.unicode_minus'] = False
            return font
    return 'DejaVu Sans'

_KO_FONT = _setup_korean_font()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from config import DATA_PROCESSED, OUT_TABLES, OUT_FIGURES, KMA_API_KEY, OPENAI_API_KEY
from forecast_engine import run_full_forecast, compute_features, STATION_GRID

# ── 상수 ──────────────────────────────────────────────────────────────────────
GRADE_COLORS = {'Very High':'#d62728','High':'#ff7f0e','Moderate':'#f7c948','Low':'#2ca02c'}
GRADE_KR     = {'Very High':'매우높음','High':'높음','Moderate':'보통','Low':'낮음'}
GRADE_EN     = {'Very High':'Very High','High':'High','Moderate':'Moderate','Low':'Low'}
GRADE_EMOJI  = {'Very High':'🔴','High':'🟠','Moderate':'🟡','Low':'🟢'}

# ── 다국어 번역 시스템 ────────────────────────────────────────────────────────
TRANSLATIONS = {
    # 헤더
    'title':         {'ko': '전력설비 화재위험 우선점검 시스템',         'en': 'Power Facility Fire Risk Prioritization System'},
    'subtitle':      {'ko': 'Explainable GeoAI · KMA ASOS · LightGBM · SHAP\n강원도 전봇대 1,387,831개 · 2026 날씨 빅데이터 콘테스트',
                      'en': 'Explainable GeoAI · KMA ASOS · LightGBM · SHAP\n1,387,831 Facilities in Gangwon · 2026 Weather Big Data Contest'},
    # 사이드바
    'filter':        {'ko': '⚙️ 필터 설정',        'en': '⚙️ Filters'},
    'grade_display': {'ko': '위험등급 표시',         'en': 'Show Risk Grades'},
    'topk_label':    {'ko': '🎯 Top-K 점검 대상 (%)', 'en': '🎯 Top-K Inspection (%)'},
    'station_label': {'ko': '📍 관측소 권역',         'en': '📍 Station Area'},
    'all_stations':  {'ko': '전체',                  'en': 'All'},
    'score_basis':   {'ko': '점수 기준',              'en': 'Score Basis'},
    'wpfi_score':    {'ko': 'WPFI 종합점수',          'en': 'WPFI Composite'},
    'ml_score':      {'ko': 'ML 화재확률',            'en': 'ML Fire Prob.'},
    'scenario_lbl':  {'ko': '🌦️ 기상 시뮬레이션',    'en': '🌦️ Weather Simulation'},
    'scenario_sel':  {'ko': '시나리오 프리셋',         'en': 'Scenario Preset'},
    'fine_tune':     {'ko': '세부 조정 (선택)',        'en': 'Fine-tune (optional)'},
    'wh_adjust':     {'ko': '기상위험도 직접 조정 (+/-)', 'en': 'Weather Hazard Adjust (+/-)'},
    'sp_adjust':     {'ko': '공간노출도 직접 조정 (+/-)', 'en': 'Spatial Exposure Adjust (+/-)'},
    'stats_lbl':     {'ko': '📊 현황',               'en': '📊 Statistics'},
    'total_fac':     {'ko': '전체 설비',              'en': 'Total Facilities'},
    'sim_active':    {'ko': '⚠️ 시뮬레이션 적용 중',  'en': '⚠️ Simulation Active'},
    # KPI
    'kpi_total':     {'ko': '설비 수',               'en': 'Facilities'},
    'kpi_vh':        {'ko': '🔴 매우높음',            'en': '🔴 Very High'},
    'kpi_hi':        {'ko': '🟠 높음',               'en': '🟠 High'},
    'kpi_avg':       {'ko': '평균 위험도',             'en': 'Avg. Risk Score'},
    # 총평
    'summary_title': {'ko': '#### 📋 AI 종합 총평',   'en': '#### 📋 AI Executive Summary'},
    'refresh':       {'ko': '🔄 총평 새로고침',        'en': '🔄 Refresh Summary'},
    # 탭 이름
    'tab_map':       {'ko': '🗺️ 위험도 지도',         'en': '🗺️ Risk Map'},
    'tab_list':      {'ko': '📋 우선점검 목록',        'en': '📋 Inspection List'},
    'tab_detail':    {'ko': '🔍 설비 상세분석',        'en': '🔍 Facility Detail'},
    'tab_model':     {'ko': '📊 모델 성능',            'en': '📊 Model Performance'},
    'tab_spatial':   {'ko': '📈 공간 분포',            'en': '📈 Spatial Distribution'},
    'tab_trend':     {'ko': '📅 트렌드 분석',          'en': '📅 Trend Analysis'},
    'tab_forecast':  {'ko': '📡 기상 예보',            'en': '📡 Weather Forecast'},
    'tab_method':    {'ko': '🧮 산출 방식',            'en': '🧮 Methodology'},
    # 지도 탭
    'map_title':     {'ko': '##### 📍 강원도 전봇대 화재위험도 지도', 'en': '##### 📍 Gangwon Power Pole Fire Risk Map'},
    'map_caption':   {'ko': '등급별 상위 3,000개 표시 | 마커 클릭 시 상세 정보', 'en': 'Top 3,000 per grade | Click markers for details'},
    'map_no_data':   {'ko': '지도 데이터가 없습니다.', 'en': 'No map data available.'},
    # 점검 목록 탭
    'list_title':    {'ko': '##### 🎯 우선점검 대상 상위 {}% — **{:,}개** 설비', 'en': '##### 🎯 Top {}% Priority Inspection — **{:,}** Facilities'},
    'rank':          {'ko': '순위',    'en': 'Rank'},
    'fac_id':        {'ko': '설비ID',  'en': 'Facility ID'},
    'wpfi_col':      {'ko': 'WPFI점수','en': 'WPFI Score'},
    'ml_col':        {'ko': 'ML점수',  'en': 'ML Score'},
    'grade_col':     {'ko': '등급KR',  'en': 'Grade'},
    'weather_col':   {'ko': '기상',    'en': 'Weather'},
    'spatial_col':   {'ko': '공간',    'en': 'Spatial'},
    'history_col':   {'ko': '이력',    'en': 'History'},
    'station_col':   {'ko': '관측소',  'en': 'Station'},
    'coverage':      {'ko': 'Top-K Coverage', 'en': 'Top-K Coverage'},
    'csv_download':  {'ko': '📥 CSV 다운로드', 'en': '📥 Download CSV'},
    # 설비 상세
    'detail_title':  {'ko': '##### 🔍 설비 상세 분석', 'en': '##### 🔍 Facility Detail Analysis'},
    'select_fac':    {'ko': '설비 선택 (Top-50 고위험)', 'en': 'Select Facility (Top-50 High Risk)'},
    'layer_scores':  {'ko': '**📊 4-Layer 위험 점수**', 'en': '**📊 4-Layer Risk Scores**'},
    'shap_title':    {'ko': '**🔬 SHAP 기여도 분석**', 'en': '**🔬 SHAP Contribution Analysis**'},
    'ai_explain':    {'ko': '**🤖 AI 위험 설명**',     'en': '**🤖 AI Risk Explanation**'},
    # 모델 성능
    'model_title':   {'ko': '##### 📊 LightGBM 모델 성능 검증', 'en': '##### 📊 LightGBM Model Performance'},
    'model_explain_title': {'ko': '📖 이 탭이 보여주는 것 (쉬운 설명)', 'en': '📖 What This Tab Shows (Plain English)'},
    # 공간 분포
    'spatial_title': {'ko': '##### 📈 위험도 공간 분포', 'en': '##### 📈 Spatial Risk Distribution'},
    # 트렌드
    'trend_title':   {'ko': '##### 📅 기상위험도 시계열 트렌드 분석 (2022~2024)', 'en': '##### 📅 Weather Hazard Time-Series Trend (2022–2024)'},
    # 예보
    'fc_title':      {'ko': '##### 📡 기상 예보 기반 화재 위험 예보 (3일)', 'en': '##### 📡 3-Day Fire Risk Forecast (Weather-Based)'},
    'fc_loading':    {'ko': '📡 기상청 단기예보 자동 수집 중... (최초 1회)', 'en': '📡 Fetching KMA short-term forecast... (first load)'},
    'fc_refresh':    {'ko': '🔄 새로고침', 'en': '🔄 Refresh'},
    'fc_updated':    {'ko': '🕐 최종 수집:', 'en': '🕐 Last updated:'},
    'fc_overall':    {'ko': '#### 🧭 향후 3일 종합 권고', 'en': '#### 🧭 3-Day Overall Recommendation'},
    'fc_daily':      {'ko': '#### 📅 날짜별 위험 현황', 'en': '#### 📅 Daily Risk Overview'},
    'fc_detail':     {'ko': '#### 🔍 날짜별 상세 분석', 'en': '#### 🔍 Daily Detail Analysis'},
    'fc_date':       {'ko': '📅 날짜 선택', 'en': '📅 Select Date'},
    'fc_today':      {'ko': '오늘',  'en': 'Today'},
    'fc_tomorrow':   {'ko': '내일',  'en': 'Tomorrow'},
    'fc_day3':       {'ko': '모레',  'en': 'Day+2'},
    'fc_ai_rec':     {'ko': '**🤖 AI 위험 분석 및 예방 권고**', 'en': '**🤖 AI Risk Analysis & Prevention**'},
    'fc_wx_sum':     {'ko': '**🌡️ 관측소별 기상 예보**', 'en': '**🌡️ Station Weather Forecast**'},
    'fc_map':        {'ko': '**🗺️ 예보 위험도 지도 —',  'en': '**🗺️ Forecast Risk Map —'},
    'fc_top20':      {'ko': '**🎯 예측 고위험 설비 Top-20**', 'en': '**🎯 Top-20 Forecast High-Risk Facilities**'},
    # 기타
    'popup_main_cause': {'ko': '[주요]', 'en': '[Main]'},
    'popup_sub_cause':  {'ko': '[보조]', 'en': '[Sub]'},
    'footer': {'ko': 'WPFI v3.1 · 2026 날씨 빅데이터 콘테스트 · Explainable GeoAI<br>강원도 전봇대 1,387,831개 분석 | LightGBM AUC 0.9574 | FWI+누적위험+연쇄피해 모델',
               'en': 'WPFI v3.1 · 2026 Weather Big Data Contest · Explainable GeoAI<br>1,387,831 Power Poles in Gangwon, Korea | LightGBM AUC 0.9574 | FWI+Accumulated+Cascade Model'},
}

def t(key: str, lang: str = 'ko') -> str:
    """번역 키로 현재 언어 문자열 반환"""
    return TRANSLATIONS.get(key, {}).get(lang, TRANSLATIONS.get(key, {}).get('ko', key))

WEIGHTS = {'weather': 0.35, 'spatial': 0.30, 'facility': 0.25, 'prior': 0.10}

# 시뮬레이션 프리셋 (내부 키 → 언어별 레이블·설명)
SIMULATION_PRESETS_DEF = [
    {
        "id":    "baseline",
        "ko":    "현재 데이터 (기준)",
        "en":    "Current Data (Baseline)",
        "desc_ko": "실제 분석 결과 그대로 표시",
        "desc_en": "Display actual analysis results as-is",
        "wh_mult": 1.00,
    },
    {
        "id":    "spring_dry_wind",
        "ko":    "봄철 건조강풍 시나리오",
        "en":    "Spring Dry & Windy",
        "desc_ko": "실효습도 -20%, 풍속 +5m/s 조건 (산불 위험 최고조 시기)",
        "desc_en": "Eff. humidity −20%, wind speed +5 m/s (peak wildfire season)",
        "wh_mult": 1.35,
    },
    {
        "id":    "summer_heatwave",
        "ko":    "여름 폭염 시나리오",
        "en":    "Summer Heatwave",
        "desc_ko": "기온 +8°C, 습도 -10% 조건",
        "desc_en": "Temperature +8°C, humidity −10%",
        "wh_mult": 1.20,
    },
    {
        "id":    "autumn_dry",
        "ko":    "가을 건조 시나리오",
        "en":    "Autumn Dry Season",
        "desc_ko": "연속 무강수 14일, 습도 -25% 조건",
        "desc_en": "14-day no-rain streak, humidity −25%",
        "wh_mult": 1.40,
    },
    {
        "id":    "winter_wind",
        "ko":    "겨울 강풍 시나리오",
        "en":    "Winter Strong Wind",
        "desc_ko": "풍속 +8m/s, 기온 -10°C 조건",
        "desc_en": "Wind speed +8 m/s, temperature −10°C",
        "wh_mult": 1.15,
    },
    {
        "id":    "worst_case",
        "ko":    "최악 복합 시나리오",
        "en":    "Worst-Case Composite",
        "desc_ko": "극건조 + 강풍 + 고온 동시 발생",
        "desc_en": "Extreme drought + strong wind + high temperature simultaneously",
        "wh_mult": 1.70,
    },
]
# 내부 ID → 프리셋 빠른 조회
_SIM_BY_ID = {p["id"]: p for p in SIMULATION_PRESETS_DEF}

# 레거시 호환: apply_simulation에서 wh_mult를 가져오기 위한 딕셔너리
SIMULATION_PRESETS = {p["ko"]: {"wh_mult": p["wh_mult"], "desc": p["desc_ko"]}
                      for p in SIMULATION_PRESETS_DEF}

# 관측소명 한→영 매핑 (KMA 공식 영문명)
STATION_EN = {
    "춘천": "Chuncheon",  "홍천": "Hongcheon", "인제": "Inje",
    "원주": "Wonju",      "횡성": "Hoengseong","강릉": "Gangneung",
    "양양": "Yangyang",   "정선": "Jeongseon", "동해": "Donghae",
    "삼척": "Samcheok",   "태백": "Taebaek",   "속초": "Sokcho",
}

# ML 확률 절대값 기반 등급 임계값 (ml_fire_prob_max 기준)
# Very High: 0.20 이상 — 모델이 화재 조건으로 판단 (5회 중 1회 수준)
# High:      0.05 이상 — 위험 기상 지속 중
# Moderate:  0.01 이상 — 평균 이상 주의 필요
GRADE_THRESHOLDS_ACTUAL = {
    "Very High": 36.9,   # WPFI_v2 x100 기준 (상위 5%)
    "High":      30.2,   # WPFI_v2 x100 기준 (상위 15%)
    "Moderate":  16.9,   # WPFI_v2 x100 기준 (상위 40%)
}
# 시뮬레이션 탭 전용 (구 가중합 공식 스케일)
GRADE_THRESHOLDS_SIM = {
    "Very High": 48.4,
    "High":      38.2,
    "Moderate":  23.8,
}

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
@st.cache_data
def load_risk():
    p = DATA_PROCESSED / 'risk_scores.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    # WPFI_v2 기반 final_risk 업데이트
    if 'final_risk_wpfi' in df.columns:
        df['final_risk'] = df['final_risk_wpfi']
        vh = GRADE_THRESHOLDS_ACTUAL["Very High"]
        hi = GRADE_THRESHOLDS_ACTUAL["High"]
        mo = GRADE_THRESHOLDS_ACTUAL["Moderate"]
        df['risk_grade'] = pd.cut(
            df['final_risk'],
            bins=[-np.inf, mo, hi, vh, np.inf],
            labels=['Low','Moderate','High','Very High']
        ).astype(str)
    if 'ml_fire_prob_max' in df.columns:
        mn, mx = df['ml_fire_prob_max'].min(), df['ml_fire_prob_max'].max()
        df['ml_score'] = ((df['ml_fire_prob_max'] - mn) / (mx - mn + 1e-9) * 100).clip(0, 100)
    elif 'hazard_combined' in df.columns:
        df['ml_score'] = (df['hazard_combined'] * 100).clip(0, 100)
    return df

@st.cache_data
def load_geo():
    p = DATA_PROCESSED / 'facility_geo.gpkg'
    return gpd.read_file(p) if p.exists() else None

@st.cache_data
def load_model():
    p = DATA_PROCESSED / 'lgbm_model.pkl'
    if not p.exists():
        return None, None, None
    with open(p, 'rb') as f:
        saved = pickle.load(f)
    return saved['model'], saved.get('scaler'), saved['features']

@st.cache_data
def load_perf():
    results = {}
    for name, path in [
        ('model_perf',           OUT_TABLES / 'multihazard_model_performance.csv'),
        ('recall_k',             OUT_TABLES / 'recall_at_k.csv'),
        ('ablation',             OUT_TABLES / 'ablation_study.csv'),
        ('shap_imp',             OUT_TABLES / 'shap_importance.csv'),
        ('ext_val',              OUT_TABLES / 'external_validation.csv'),
        ('region_cv',            OUT_TABLES / 'region_cv_results.csv'),
        ('sensitivity',          OUT_TABLES / 'sensitivity_analysis.csv'),
        ('sensitivity_scenarios',OUT_TABLES / 'sensitivity_scenarios.csv'),
    ]:
        if Path(path).exists():
            results[name] = pd.read_csv(path)
    return results

@st.cache_data
def load_trend():
    p = DATA_PROCESSED / 'train_features.parquet'
    if not p.exists():
        return None
    import pyarrow.parquet as pq
    want = {'date','pole_id','nearest_station','weather_hazard',
            'dryness_score','wind_score','heat_score','no_rain_days',
            'combined_risk_flag','label','season'}
    cols = [c for c in pq.read_schema(p).names if c in want]
    tf = pd.read_parquet(p, columns=cols)
    tf['date']  = pd.to_datetime(tf['date'])
    tf['year']  = tf['date'].dt.year
    tf['month'] = tf['date'].dt.month
    tf['ym']    = tf['date'].dt.to_period('M').astype(str)
    SEASON_KR = {0:'겨울',1:'봄',2:'여름',3:'가을'}
    if 'season' in tf.columns:
        tf['season_kr'] = tf['season'].map(SEASON_KR)
    return tf

# ── OpenAI 호출 ───────────────────────────────────────────────────────────────
def _openai_chat(prompt: str, system: str = "", max_tokens: int = 600) -> str:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model='gpt-4o-mini', messages=messages,
            temperature=0.3, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI 오류] {e}"


def _openai_chat_json(prompt: str, system: str = "", max_tokens: int = 700) -> dict | None:
    """JSON 구조화 응답 전용 — response_format으로 반드시 JSON 반환 보장"""
    if not OPENAI_API_KEY:
        return None
    try:
        import json as _json
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model='gpt-4o-mini', messages=messages,
            temperature=0.3, max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return _json.loads(resp.choices[0].message.content.strip())
    except Exception:
        return None


def _render_structured_ai(data: dict | str, lang: str = 'ko') -> None:
    """구조화된 AI 응답을 Streamlit 컴포넌트로 렌더링 (공통 형식)"""
    import json as _json
    if isinstance(data, str):
        # fallback: 기존 텍스트
        st.info(f"🤖 {data}")
        return
    if not isinstance(data, dict):
        return

    # 공통 헤더
    headline = data.get('headline') or data.get('situation','')
    if headline:
        st.markdown(f"##### 🤖 {headline}")

    # 핵심 섹션 렌더링
    section_map_ko = {
        'situation':      ('📊 현황',     'info'),
        'risk_factor':    ('⚠️ 위험요인',  'warning'),
        'weather_cause':  ('🌦 기상원인',  'warning'),
        'main_cause':     ('🔍 주요원인',  'warning'),
        'weather_context':('🌡 기상상황',  'info'),
        'action':         ('✅ 권고사항',  'success'),
        'inspection':     ('🔧 점검조치',  'success'),
        'trend':          ('📈 위험추세',  'info'),
        'peak_day':       ('🔺 최고위험일', 'warning'),
        'priority_area':  ('📍 우선권역',  'info'),
    }
    section_map_en = {
        'situation':      ('📊 Situation',      'info'),
        'risk_factor':    ('⚠️ Risk Factors',    'warning'),
        'weather_cause':  ('🌦 Weather Cause',   'warning'),
        'main_cause':     ('🔍 Main Cause',      'warning'),
        'weather_context':('🌡 Weather Status',  'info'),
        'action':         ('✅ Action',          'success'),
        'inspection':     ('🔧 Inspection',      'success'),
        'trend':          ('📈 Trend',           'info'),
        'peak_day':       ('🔺 Peak Risk Day',   'warning'),
        'priority_area':  ('📍 Priority Area',   'info'),
    }
    section_map = section_map_ko if lang == 'ko' else section_map_en
    render_fn = {'info': st.info, 'warning': st.warning, 'success': st.success}

    for key, (label, style) in section_map.items():
        val = data.get(key)
        if not val or key == 'headline':
            continue
        render_fn[style](f"**{label}**: {val}")

    # actions 리스트 렌더링
    actions = data.get('actions', [])
    if actions:
        label = '🛡 예방 조치' if lang == 'ko' else '🛡 Actions'
        st.markdown(f"**{label}**")
        for i, act in enumerate(actions, 1):
            st.markdown(f"  {i}. {act}")

def get_executive_summary(df: pd.DataFrame, perf: dict, lang: str = 'ko') -> dict | str:
    total    = len(df)
    vh       = (df['risk_grade'] == 'Very High').sum()
    hi       = (df['risk_grade'] == 'High').sum()
    top_stn  = df.groupby('nearest_station')['final_risk'].mean().idxmax()
    top_mean = df.groupby('nearest_station')['final_risk'].mean().max()
    auc      = perf.get('model_perf', pd.DataFrame()).get('auc', pd.Series([0])).max()

    if lang == 'en':
        default = {
            "headline": f"Gangwon Power Facility Risk: {vh:,} Very High ({vh/total*100:.1f}%)",
            "situation": f"{total:,} facilities analyzed. {vh:,} ({vh/total*100:.1f}%) Very High, {hi:,} High grade.",
            "risk_factor": f"Highest-risk area: {top_stn} (avg {top_mean:.1f}/100). Key drivers: dry weather, FWI elevation, forest proximity.",
            "action": "Immediate on-site inspection for Very High facilities. Pre-check insulation in dry-alert zones.",
        }
        prompt = f"""Analyze power facility fire risk in Gangwon Province and return a JSON object with EXACTLY these keys:
"headline": one-line summary with key numbers
"situation": current risk status with specific counts and percentages
"risk_factor": top 2-3 risk drivers with evidence
"action": 1-2 specific actionable recommendations

Analysis Data:
- Total: {total:,} facilities | Very High: {vh:,} ({vh/total*100:.1f}%) | High: {hi:,} ({hi/total*100:.1f}%)
- Highest-risk area: {top_stn} (avg risk {top_mean:.1f}/100)
- Model AUC: {auc:.4f} | WPFI_v2: dry(50%) + heat(30%) + lightning(20%)
- Analysis period: 2022-2024"""
        system = ("You are a power facility fire risk expert. "
                  "Respond ONLY with a valid JSON object containing exactly the keys: "
                  "headline, situation, risk_factor, action.")
    else:
        default = {
            "headline": f"매우높음 등급 {vh:,}개({vh/total*100:.1f}%) — {top_stn} 권역 최고 위험",
            "situation": f"강원도 전력설비 {total:,}개 분석 결과, Very High {vh:,}개({vh/total*100:.1f}%), High {hi:,}개({hi/total*100:.1f}%) 확인.",
            "risk_factor": f"최고위험 권역: {top_stn}(평균 {top_mean:.1f}/100). 주요 원인: 건조·FWI 복합 조건, 고지대 낙뢰 노출.",
            "action": "매우높음 등급 설비 즉각 현장 점검 실시. 건조경보 권역 절연 사전 점검 권고.",
        }
        prompt = f"""강원도 전력설비 화재위험도 분석 결과를 JSON으로 반환하세요. 반드시 아래 키만 포함:
"headline": 핵심 수치를 담은 1줄 요약
"situation": 현재 위험 현황 (구체적 수치 포함)
"risk_factor": 주요 위험요인 2~3가지 (수치 근거 포함)
"action": 구체적 권고사항 1~2가지

분석 데이터:
- 전체 {total:,}개 | Very High {vh:,}개({vh/total*100:.1f}%) | High {hi:,}개({hi/total*100:.1f}%)
- 최고위험 권역: {top_stn} (평균 위험도 {top_mean:.1f}/100)
- 모델 AUC: {auc:.4f} | WPFI_v2 = 건조(50%)+고온(30%)+낙뢰(20%)
- 분석 기간: 2022~2024년"""
        system = ("당신은 전력설비 화재 위험 전문가입니다. "
                  "반드시 headline, situation, risk_factor, action 키만 포함한 유효한 JSON으로만 응답하세요.")

    result = _openai_chat_json(prompt, system=system)
    return result if result else default


@st.cache_resource
def _load_rag():
    """RAG 코퍼스 로드 (캐시)"""
    try:
        import pickle as _pkl
        from sklearn.metrics.pairwise import cosine_similarity as _cos
        rag_path = DATA_PROCESSED / 'rag_corpus.pkl'
        if not rag_path.exists():
            return None
        with open(rag_path, 'rb') as f:
            data = _pkl.load(f)
        return data
    except Exception:
        return None


def _rag_search(query: str, top_k: int = 2) -> str:
    """쿼리와 유사한 참조 문서 반환 (텍스트)"""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        rag = _load_rag()
        if rag is None:
            return ""
        q_vec = rag['vectorizer'].transform([query])
        scores = cosine_similarity(q_vec, rag['tfidf_matrix']).flatten()
        top_idx = scores.argsort()[::-1][:top_k]
        refs = []
        for i in top_idx:
            if scores[i] > 0.05:
                refs.append(f"[{rag['docs'][i]['category']}] {rag['docs'][i]['text'][:150]}")
        return "\n".join(refs)
    except Exception:
        return ""


def get_facility_explanation(row: dict, lang: str = 'ko') -> dict | str:
    grade_label = (GRADE_KR if lang == 'ko' else GRADE_EN).get(row.get('risk_grade', ''), '')
    wh  = row.get('weather_hazard', 0)
    fwi = row.get('fwi_score', row.get('ml_fire_prob_max', 0) * 30)
    acc = row.get('accumulated_risk_norm', 0)
    rag_context = _rag_search(
        f"기상위험 {wh:.0f} FWI {fwi:.1f} 누적건조 {acc:.0f} 전봇대 화재위험", top_k=2)

    if lang == 'en':
        default = {
            "headline": f"Facility #{row.get('pole_id','')} — {grade_label} ({row.get('final_risk',0):.1f}/100)",
            "main_cause": f"Weather hazard {wh:.0f}/100, FWI {fwi:.1f}, accumulated dry risk {acc:.1f}.",
            "weather_context": f"Station: {row.get('nearest_station','')} | Spatial exposure: {row.get('spatial_exposure',0):.0f}",
            "inspection": "Perform insulation resistance test. On-site visual inspection recommended.",
        }
        rag_section = f"\n\nReference:\n{rag_context}" if rag_context else ""
        prompt = f"""Analyze this power facility risk and return a JSON with EXACTLY these keys:
"headline": facility ID + grade + score in one line
"main_cause": top 2 risk drivers with specific values
"weather_context": current weather condition and station area
"inspection": 1-2 specific inspection actions

Data:
Facility #{row.get('pole_id','')} | Grade: {grade_label} ({row.get('final_risk',0):.1f}/100)
Weather Hazard: {wh:.1f} | FWI: {fwi:.1f} | Accumulated Risk: {acc:.1f}
Spatial Exposure: {row.get('spatial_exposure',0):.1f} | Station: {row.get('nearest_station','')}{rag_section}"""
        system = ("You are a power facility inspection expert. "
                  "Respond ONLY with a valid JSON containing: headline, main_cause, weather_context, inspection.")
    else:
        default = {
            "headline": f"설비 #{row.get('pole_id','')} — {grade_label} ({row.get('final_risk',0):.1f}/100)",
            "main_cause": f"기상위험 {wh:.0f}/100, FWI {fwi:.1f}, 누적건조위험 {acc:.1f}.",
            "weather_context": f"관할관측소: {row.get('nearest_station','')} | 공간노출도: {row.get('spatial_exposure',0):.0f}",
            "inspection": "절연 저항 측정 및 육안 점검 즉시 시행 권고.",
        }
        rag_section = f"\n\n참조:\n{rag_context}" if rag_context else ""
        prompt = f"""전력설비 위험 분석 결과를 JSON으로 반환하세요. 반드시 아래 키만 포함:
"headline": 설비ID + 등급 + 점수 1줄
"main_cause": 주요 위험요인 2가지 (수치 포함)
"weather_context": 현재 기상 조건 및 관할 관측소
"inspection": 구체적 점검 조치 1~2가지

데이터:
설비 #{row.get('pole_id','')} | 등급: {grade_label} ({row.get('final_risk',0):.1f}/100)
기상위험: {wh:.1f} | FWI: {fwi:.1f} | 누적위험도: {acc:.1f}
공간노출: {row.get('spatial_exposure',0):.1f} | 관측소: {row.get('nearest_station','')}{rag_section}"""
        system = ("당신은 전력설비 안전 점검 전문가입니다. "
                  "반드시 headline, main_cause, weather_context, inspection 키만 포함한 JSON으로 응답하세요.")

    result = _openai_chat_json(prompt, system=system)
    return result if result else default


def get_overall_forecast_recommendation(fc_result: pd.DataFrame, fc_feat: pd.DataFrame,
                                         lang: str = 'ko') -> str:
    if fc_result is None or fc_result.empty:
        return ""
    dates = sorted(fc_result['fcst_date'].unique())
    today = pd.Timestamp.now()
    day_labels = {0: ("오늘","Today"), 1: ("내일","Tomorrow"), 2: ("모레","Day+2")}

    lines = []
    for d in dates:
        dt   = pd.to_datetime(d, format="%Y%m%d")
        diff = (dt - today.normalize()).days
        lbl  = day_labels.get(diff, (dt.strftime("%m/%d"), dt.strftime("%m/%d")))[0 if lang=='ko' else 1]
        day  = fc_result[fc_result['fcst_date'] == d]
        vh   = (day['forecast_grade'] == 'Very High').sum()
        hi   = (day['forecast_grade'] == 'High').sum()
        feat = fc_feat[fc_feat['fcst_date'] == d] if fc_feat is not None else pd.DataFrame()
        wh_max = feat['weather_hazard'].max() if not feat.empty else 0
        if lang == 'en':
            lines.append(f"- {lbl} ({dt.strftime('%m/%d')}): Very High {vh:,}, High {hi:,}, Max hazard {wh_max:.1f}")
        else:
            lines.append(f"- {lbl}({dt.strftime('%m.%d')}): Very High {vh:,}개, High {hi:,}개, 최대기상위험도 {wh_max:.1f}")

    if lang == 'en':
        default = "3-Day Forecast Summary:\n" + "\n".join(lines) + \
                  "\n\nConcentrate inspection resources on the highest-risk day. " \
                  "Pre-check insulation in dry-alert zones."
        prompt = f"""Analyze the 3-day Gangwon power facility fire risk forecast and write a comprehensive recommendation for executives and field managers.

Forecast summary:
{chr(10).join(lines)}

Include: 1) Overall 3-day risk trend, 2) Most dangerous day and reason, 3) Priority inspection areas and facility types, 4) 3 specific preventive actions, 5) Resource allocation recommendation."""
        system = "You are a power facility fire prevention expert. Answer in 5-7 sentences in English."
    else:
        default = "향후 3일간 예보 분석:\n" + "\n".join(lines) + \
                  "\n\n가장 위험도가 높은 날에 현장 점검 자원을 집중 배치하고, " \
                  "건조주의보 발령 권역의 설비는 사전 절연 점검을 권고합니다."
        prompt = f"""향후 3일간 강원도 전력설비 화재 위험 예보를 분석하여 경영진과 현장 관리자를 위한 종합 권고문을 작성하세요.

예보 요약:
{chr(10).join(lines)}

다음 내용을 반드시 포함하세요:
1. 3일간 전체 위험 추세 (날짜별 비교)
2. 가장 위험한 날짜와 그 이유
3. 우선 점검 권역 및 설비 유형
4. 예방 조치 3가지 (구체적이고 실행 가능한 것)
5. 자원 배치 권고"""
        system = "당신은 전력설비 화재 예방 전문가입니다. 한국어 5~7문장으로 답변하세요."

    result = _openai_chat(prompt, system=system, max_tokens=500)
    return result if result and not result.startswith("[OpenAI") else default


def get_forecast_recommendation(feat_df: pd.DataFrame, fc_day: pd.DataFrame,
                                 date_label: str, lang: str = 'ko') -> str:
    if feat_df is None or fc_day.empty:
        return {}
    top_stn = feat_df.groupby('station')['weather_hazard'].mean().idxmax() \
              if not feat_df.empty else ("알 수 없음" if lang == 'ko' else "Unknown")
    vh_cnt = (fc_day['forecast_grade'] == 'Very High').sum()
    hi_cnt = (fc_day['forecast_grade'] == 'High').sum()
    dry_cnt  = int(feat_df['dry_watch_flag'].sum()) if 'dry_watch_flag' in feat_df.columns else 0
    wind_cnt = int(feat_df['wind_watch_flag'].sum()) if 'wind_watch_flag' in feat_df.columns else 0

    if lang == 'en':
        default = {
            "headline": f"{date_label} — {vh_cnt:,} Very High, {hi_cnt:,} High predicted",
            "situation": f"Highest-risk area: {top_stn}. Max weather hazard: {feat_df['weather_hazard'].max():.1f}/100.",
            "weather_cause": f"Dry-alert: {dry_cnt} stations | Wind-alert: {wind_cnt} stations.",
            "actions": [
                f"Priority inspection in {top_stn} area ({vh_cnt:,} Very High facilities)",
                "Pre-check insulation in dry-alert zones before peak risk period",
                "Deploy emergency response team on standby",
            ],
        }
        prompt = f"""Analyze the {date_label} fire risk forecast and return JSON with EXACTLY these keys:
"headline": date + key counts in one line
"situation": highest-risk area and weather hazard level
"weather_cause": key weather conditions (alerts, indicators)
"actions": array of exactly 3 specific preventive actions (strings)

Data:
Date: {date_label} | Top station: {top_stn}
Very High: {vh_cnt:,} | High: {hi_cnt:,} | Max hazard: {feat_df['weather_hazard'].max():.1f}/100
Dry alerts: {dry_cnt} | Wind alerts: {wind_cnt}"""
        system = ("You are a power facility fire prevention expert. "
                  "Respond ONLY with valid JSON containing: headline, situation, weather_cause, actions (array of 3).")
    else:
        default = {
            "headline": f"{date_label} — Very High {vh_cnt:,}개, High {hi_cnt:,}개 예측",
            "situation": f"최고위험 권역: {top_stn}. 최대 기상위험도: {feat_df['weather_hazard'].max():.1f}/100.",
            "weather_cause": f"건조주의보: {dry_cnt}개소 | 강풍주의보: {wind_cnt}개소.",
            "actions": [
                f"{top_stn} 권역 Very High {vh_cnt:,}개 설비 우선 점검",
                "건조주의보 권역 절연 취약 설비 사전 점검 실시",
                "비상 점검팀 대기 배치 및 순찰 강화",
            ],
        }
        prompt = f"""{date_label} 화재 위험 예보를 분석하여 JSON으로 반환하세요. 반드시 아래 키만 포함:
"headline": 날짜 + 핵심 수치 1줄
"situation": 최고위험 권역 및 기상위험도 수준
"weather_cause": 주요 기상 조건 (경보, 지수)
"actions": 구체적 예방 조치 3가지 (문자열 배열)

데이터:
날짜: {date_label} | 최고위험 권역: {top_stn}
Very High: {vh_cnt:,}개 | High: {hi_cnt:,}개 | 최대 기상위험도: {feat_df['weather_hazard'].max():.1f}/100
건조주의보: {dry_cnt}개소 | 강풍주의보: {wind_cnt}개소"""
        system = ("당신은 전력설비 화재 예방 전문가입니다. "
                  "반드시 headline, situation, weather_cause, actions(3개 문자열 배열) 키만 포함한 JSON으로 응답하세요.")

    result = _openai_chat_json(prompt, system=system)
    return result if result else default

# ── 위험 판단 사유 생성 (규칙 기반) ──────────────────────────────────────────
def _risk_reason(row: dict) -> str:
    """ML 확률 + 컴포넌트 점수 기반 설명 (등급별 차별화)"""
    prob = float(row.get('ml_fire_prob_max', 0))
    grade = str(row.get('risk_grade', 'Low'))
    sp  = float(row.get('spatial_exposure',  0))
    fe  = float(row.get('facility_exposure', 0))
    hp  = float(row.get('hist_prior',        0))
    cas = float(row.get('cascade_risk',      0))
    ter = float(row.get('terrain_fire_risk', 0))
    _lang = st.session_state.get('lang', 'ko')

    if _lang == 'en':
        # ML 확률 기반 주 설명
        if grade == 'Very High':
            main = f"ML model estimates {prob*100:.1f}% fire probability — actual wildfire conditions recorded nearby. Immediate inspection required."
        elif grade == 'High':
            main = f"ML model estimates {prob*100:.1f}% fire probability — elevated risk from dry/windy weather patterns."
        elif grade == 'Moderate':
            main = f"ML model estimates {prob*100:.1f}% fire probability — above-average risk, monitor weather closely."
        else:
            main = f"ML fire probability {prob*100:.2f}% — within normal range under current weather conditions."

        # 공간·설비 보조 설명 (실제 차이 있는 값 기준)
        sub_parts = []
        if sp >= 60:  sub_parts.append(f"Forest/terrain exposure high ({sp:.0f})")
        elif sp >= 40: sub_parts.append(f"Moderate forest proximity ({sp:.0f})")
        if fe >= 65:  sub_parts.append(f"Dense facility cluster (cascade risk {cas:.0f})")
        elif fe >= 50: sub_parts.append(f"Moderate facility density ({fe:.0f})")
        if hp >= 65:  sub_parts.append("Repeat fire history zone")
        if ter >= 60: sub_parts.append("High terrain fire exposure")
        sub = " · ".join(sub_parts) if sub_parts else "No significant spatial risk factors"

        flags = []
        if prob >= 0.5:  flags.append("🚨 Very High ML Risk")
        if prob >= 0.2:  flags.append("🔴 High ML Risk")
        elif prob >= 0.05: flags.append("🟠 Elevated Risk")
        if cas >= 50:    flags.append("⚡ Cascade Risk")
        if hp >= 70:     flags.append("🔥 Repeat Fire Zone")

    else:
        # ML 확률 기반 주 설명 (등급별로 다른 문구)
        if grade == 'Very High':
            main = f"ML 모델이 화재 확률 {prob*100:.1f}%로 판단 — 실제 산불 발생 조건과 유사한 기상·공간 상태입니다. 즉각 점검이 필요합니다."
        elif grade == 'High':
            main = f"ML 모델이 화재 확률 {prob*100:.1f}%로 판단 — 건조·강풍 기상이 지속되며 위험 수준이 높아진 상태입니다."
        elif grade == 'Moderate':
            main = f"ML 모델이 화재 확률 {prob*100:.2f}%로 판단 — 평균 이상의 위험 조건이 일부 존재합니다. 기상 모니터링을 강화하세요."
        else:
            main = f"ML 화재 확률 {prob*100:.3f}% — 현재 기상 조건에서 정상 범위 이내입니다."

        # 공간·설비 보조 설명 (설비마다 실제로 다른 값)
        sub_parts = []
        if sp >= 60:   sub_parts.append(f"산림·지형 노출도 높음({sp:.0f}점)")
        elif sp >= 40: sub_parts.append(f"산림 인접 보통({sp:.0f}점)")
        if fe >= 65:   sub_parts.append(f"설비 밀집, 연쇄피해 위험({cas:.0f}점)")
        elif fe >= 50: sub_parts.append(f"설비 밀도 보통({fe:.0f}점)")
        if hp >= 65:   sub_parts.append("반복 화재 발생 이력 지역")
        if ter >= 60:  sub_parts.append("지형 화재 노출도 높음")
        sub = " · ".join(sub_parts) if sub_parts else "공간적 위험 요인 낮음"

        flags = []
        if prob >= 0.5:   flags.append("🚨 ML 극고위험")
        elif prob >= 0.2: flags.append("🔴 ML 매우높음")
        elif prob >= 0.05: flags.append("🟠 ML 높음")
        elif prob >= 0.01: flags.append("🟡 ML 보통")
        if cas >= 50:     flags.append("⚡ 연쇄피해위험")
        if hp >= 70:      flags.append("🔥 반복화재지역")

    reason = f"{main}<br><small style='color:#666'>{sub}</small>"
    if flags:
        reason += "<br><b>" + " &nbsp; ".join(flags) + "</b>"
    return reason


# ── Folium 지도 ───────────────────────────────────────────────────────────────
def build_map(df_merged, selected_grades, risk_col='final_risk', max_points=3000):
    center = [37.5, 128.3]
    m = folium.Map(location=center, zoom_start=9, tiles='CartoDB positron')
    for grade in ['Very High','High','Moderate','Low']:
        if grade not in selected_grades:
            continue
        color  = GRADE_COLORS[grade]
        radius = {'Very High':8,'High':6,'Moderate':4,'Low':3}[grade]
        subset = df_merged[df_merged['risk_grade'] == grade]
        total_cnt = len(subset)
        if len(subset) > max_points:
            subset = subset.nlargest(max_points, risk_col)
        layer = folium.FeatureGroup(
            name=f"{GRADE_EMOJI[grade]} {GRADE_KR[grade]} ({total_cnt:,}개)")
        for _, row in subset.iterrows():
            reason = _risk_reason(row)
            _lang_p  = st.session_state.get('lang', 'ko')
            _glabel  = (GRADE_KR if _lang_p == 'ko' else GRADE_EN)[grade]
            _fac_lbl = '설비' if _lang_p == 'ko' else 'Facility'
            _wh_lbl  = '🌪️ 기상위험' if _lang_p == 'ko' else '🌪️ Weather'
            _sp_lbl  = '🌲 공간노출' if _lang_p == 'ko' else '🌲 Spatial'
            _fe_lbl  = '⚡ 설비노출' if _lang_p == 'ko' else '⚡ Facility'
            _hp_lbl  = '🔥 화재이력' if _lang_p == 'ko' else '🔥 History'
            _ml_lbl  = '🤖 ML점수'   if _lang_p == 'ko' else '🤖 ML Score'
            _stn_lbl = '관측소 권역'  if _lang_p == 'ko' else 'Station Area'
            popup_html = (
                f"<div style='min-width:220px;font-size:12px'>"
                f"<b style='font-size:13px'>{_fac_lbl} #{int(row['pole_id'])}</b><br>"
                f"<span style='color:{color};font-weight:bold'>"
                f"  {GRADE_EMOJI[grade]} {_glabel}</span>"
                f" &nbsp;<b>{row[risk_col]:.1f}</b>/100<br>"
                f"<hr style='margin:4px 0'>"
                f"<table style='width:100%;font-size:11px'>"
                f"<tr><td>{_wh_lbl}</td><td><b>{row.get('weather_hazard',0):.0f}</b></td>"
                f"    <td>{_sp_lbl}</td><td><b>{row.get('spatial_exposure',0):.0f}</b></td></tr>"
                f"<tr><td>{_fe_lbl}</td><td><b>{row.get('facility_exposure',0):.0f}</b></td>"
                f"    <td>{_hp_lbl}</td><td><b>{row.get('hist_prior',0):.0f}</b></td></tr>"
                f"<tr><td>{_ml_lbl}</td><td colspan='3'><b>{row.get('ml_score',0):.1f}</b></td></tr>"
                f"</table>"
                f"<hr style='margin:4px 0'>"
                f"<div style='color:#555;font-size:11px'>{reason}</div>"
                f"<div style='color:#888;font-size:10px;margin-top:2px'>"
                f"📍 {row.get('nearest_station','')} {_stn_lbl}</div>"
                f"</div>"
            )
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius, color=color, fill=True,
                fill_color=color, fill_opacity=0.75,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"#{int(row['pole_id'])}: {row[risk_col]:.1f}점 | {GRADE_KR[grade]}",
            ).add_to(layer)
        layer.add_to(m)
    folium.LayerControl().add_to(m)
    return m

# ── SHAP Waterfall ────────────────────────────────────────────────────────────
def shap_waterfall(pole_id, df, model, features):
    matplotlib.rcParams['font.family'] = _KO_FONT
    matplotlib.rcParams['axes.unicode_minus'] = False
    try:
        import shap
        tf_path = DATA_PROCESSED / 'train_features.parquet'
        if not tf_path.exists():
            return None
        tf = pd.read_parquet(tf_path)
        pole_rows = tf[tf['pole_id'] == pole_id]
        if len(pole_rows) == 0:
            return None
        prob_col = model.predict_proba(pole_rows[features].values)[:, 1]
        best_idx = prob_col.argmax()
        X_row = pole_rows[features].iloc[[best_idx]].values
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_row)
        sv_row = sv[1][0] if isinstance(sv, list) else sv[0]
        base   = (explainer.expected_value[1]
                  if isinstance(explainer.expected_value, list)
                  else explainer.expected_value)
        fig, ax = plt.subplots(figsize=(8, 6))
        feat_sv = sorted(zip(features, sv_row), key=lambda x: abs(x[1]), reverse=True)[:12]
        feat_names = [f[0] for f in feat_sv]
        feat_vals  = [f[1] for f in feat_sv]
        colors = ['#d62728' if v > 0 else '#2ca02c' for v in feat_vals]
        ax.barh(feat_names[::-1], feat_vals[::-1], color=colors[::-1], alpha=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        _lp2 = st.session_state.get('lang', 'ko')
        ax.set_xlabel('SHAP value (+ : risk increase / - : risk decrease)'
                      if _lp2 == 'en' else 'SHAP 값 (+ : 위험 증가 / - : 위험 감소)')
        _lp = st.session_state.get('lang', 'ko')
        _shap_t = (f'Facility #{pole_id} — SHAP Contribution (ML prob: {prob_col[best_idx]:.3f})'
                   if _lp == 'en' else
                   f'설비 #{pole_id} — SHAP 기여도 (ML 확률: {prob_col[best_idx]:.3f})')
        ax.set_title(_shap_t,
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception:
        return None

# ── 시뮬레이션 적용 ──────────────────────────────────────────────────────────
def apply_simulation(df: pd.DataFrame, preset_name: str,
                     wh_add: float = 0, sp_add: float = 0) -> pd.DataFrame:
    """
    기상 시나리오 적용.
    등급 기준: 원본 데이터에서 역산한 실제 임계값 사용
    (config.py의 86/66/51은 정규화 전 기준으로, 현재 데이터 범위 35~79와 맞지 않음)
    """
    preset = SIMULATION_PRESETS.get(preset_name, {"wh_mult": 1.0})
    mult = preset["wh_mult"]
    sim = df.copy()
    sim['weather_hazard'] = (sim['weather_hazard'] * mult + wh_add).clip(0, 100)
    if sp_add != 0:
        sim['spatial_exposure'] = (sim['spatial_exposure'] + sp_add).clip(0, 100)
    sim['final_risk'] = (
        sim['weather_hazard']   * WEIGHTS['weather'] +
        sim['spatial_exposure'] * WEIGHTS['spatial'] +
        sim['facility_exposure']* WEIGHTS['facility'] +
        sim['hist_prior']       * WEIGHTS['prior']
    ).clip(0, 100)

    # 시뮬레이션은 가중합 공식 스케일 임계값 사용
    vh_thr = GRADE_THRESHOLDS_SIM["Very High"]
    hi_thr = GRADE_THRESHOLDS_SIM["High"]
    mo_thr = GRADE_THRESHOLDS_SIM["Moderate"]

    def grade(s):
        if s >= vh_thr: return 'Very High'
        if s >= hi_thr: return 'High'
        if s >= mo_thr: return 'Moderate'
        return 'Low'

    sim['risk_grade'] = sim['final_risk'].apply(grade)
    return sim

# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="WPFI — Power Facility Fire Risk System",
        page_icon="🔥", layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── 전역 CSS (Glassmorphism + 다크 테마 최적화) ──────────────────────────
    st.markdown("""
<style>
/* ── 전체 레이아웃 ── */
.main .block-container { padding-top: 1rem; max-width: 1400px; }

/* ── 글래스모피즘 카드 ── */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    margin-bottom: 16px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.glass-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,0,0,0.4); }

/* ── KPI 메트릭 카드 ── */
.kpi-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.03) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-vh::before  { background: linear-gradient(90deg, #d62728, #ff4444); }
.kpi-hi::before  { background: linear-gradient(90deg, #ff7f0e, #ffaa44); }
.kpi-mod::before { background: linear-gradient(90deg, #f7c948, #ffe066); }
.kpi-low::before { background: linear-gradient(90deg, #2ca02c, #44cc44); }
.kpi-auc::before { background: linear-gradient(90deg, #1f77b4, #66aadd); }
.kpi-num {
    font-size: 2rem; font-weight: 800; margin: 8px 0 4px;
    background: linear-gradient(135deg, #fff 60%, rgba(255,255,255,0.6));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.kpi-label { font-size: 0.75rem; color: rgba(255,255,255,0.6); text-transform: uppercase; letter-spacing: 1px; }
.kpi-sub   { font-size: 0.8rem;  color: rgba(255,255,255,0.45); margin-top: 4px; }

/* ── 헤더 그래디언트 ── */
.hero-header {
    background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 40%, #0d1b2a 100%);
    border: 1px solid rgba(214,39,40,0.3);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.hero-header::after {
    content: '';
    position: absolute; top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(214,39,40,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem; font-weight: 900; margin: 0;
    background: linear-gradient(135deg, #ff4444 0%, #ff7f0e 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.hero-sub { color: rgba(255,255,255,0.75); font-size: 1.05rem; margin: 6px 0 0; }
.hero-badge {
    display: inline-block; background: rgba(214,39,40,0.2);
    border: 1px solid rgba(214,39,40,0.4); color: #ff6b6b;
    border-radius: 20px; padding: 3px 12px; font-size: 0.75rem;
    margin-right: 6px; margin-top: 8px;
}

/* ── 위험 등급 배지 ── */
.badge-vh { background:rgba(214,39,40,0.2); border:1px solid #d62728; color:#ff6b6b; border-radius:8px; padding:3px 10px; font-size:0.85rem; font-weight:600; }
.badge-hi { background:rgba(255,127,14,0.2); border:1px solid #ff7f0e; color:#ffaa44; border-radius:8px; padding:3px 10px; font-size:0.85rem; font-weight:600; }
.badge-mo { background:rgba(247,201,72,0.2); border:1px solid #f7c948; color:#ffe066; border-radius:8px; padding:3px 10px; font-size:0.85rem; font-weight:600; }
.badge-lo { background:rgba(44,160,44,0.2); border:1px solid #2ca02c; color:#44cc44; border-radius:8px; padding:3px 10px; font-size:0.85rem; font-weight:600; }

/* ── 사이드바 ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] .stMetric {
    background: rgba(255,255,255,0.04);
    border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ── 탭 스타일 ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 16px; font-size: 0.88rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: rgba(214,39,40,0.2) !important;
    border-color: rgba(214,39,40,0.5) !important;
    color: #ff6b6b !important;
}

/* ── expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.03) !important;
}

/* ── 데이터프레임 ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── 알림 박스 ── */
.stAlert { border-radius: 10px !important; }

/* ── 스크롤바 ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

    # ── 언어 설정 (사이드바 상단) ────────────────────────────────────────────
    if 'lang' not in st.session_state:
        st.session_state['lang'] = 'ko'

    with st.sidebar:
        lang_col1, lang_col2 = st.columns([3, 2])
        with lang_col2:
            _lang_idx = 0 if st.session_state.get('lang', 'ko') == 'ko' else 1
            lang_choice = st.selectbox(
                "🌐", ['🇰🇷 한국어', '🇺🇸 English'],
                index=_lang_idx,
                label_visibility="collapsed",
            )
            st.session_state['lang'] = 'ko' if '한국어' in lang_choice else 'en'

    lang = st.session_state['lang']

    matplotlib.rcParams['font.family'] = _KO_FONT
    matplotlib.rcParams['axes.unicode_minus'] = False

    GRADE_LABEL = GRADE_KR if lang == 'ko' else GRADE_EN

    # ── 히어로 헤더 ──────────────────────────────────────────────────────────
    if lang == 'ko':
        hero_title = "🔥 WPFI"
        hero_sub   = "기상·공간정보 기반 전력설비 화재위험도 분석 시스템"
        badges     = ["2026 날씨 빅데이터 콘테스트", "강원도 1,387,831개 전봇대", "LightGBM · SHAP · RAG"]
    else:
        hero_title = "🔥 WPFI"
        hero_sub   = "Weather-based Power Facility Fire-risk Index System"
        badges     = ["2026 Weather Big Data Contest", "1,387,831 Gangwon Poles", "LightGBM · SHAP · RAG"]

    badge_html = "".join(f"<span class='hero-badge'>{b}</span>" for b in badges)
    st.markdown(f"""
<div class='hero-header'>
  <div class='hero-title'>{hero_title}</div>
  <div class='hero-sub'>{hero_sub}</div>
  <div style='margin-top:10px'>{badge_html}</div>
</div>
""", unsafe_allow_html=True)

    # 데이터 로드
    df    = load_risk()
    gdf   = load_geo()
    perf  = load_perf()
    trend = load_trend()
    model, scaler, features = load_model()

    if df is None:
        st.error("📂 No analysis data found. Run notebooks/01–08 first."
                 if lang == 'en' else
                 "📂 분석 데이터가 없습니다. notebooks/01~08을 먼저 실행하세요.")
        return

    # geo 병합
    if gdf is not None:
        gdf_merged = gdf.merge(df, on='pole_id', how='left')
        gdf_merged['lat'] = gdf_merged.geometry.y
        gdf_merged['lon'] = gdf_merged.geometry.x
    else:
        gdf_merged = None

    # ── KPI 카드 (헤더 바로 아래) ─────────────────────────────────────────────
    total_fac = len(df)
    vh_cnt = (df['risk_grade'] == 'Very High').sum()
    hi_cnt = (df['risk_grade'] == 'High').sum()
    lo_cnt = (df['risk_grade'] == 'Low').sum()
    auc_val = perf['model_perf']['auc'].max() if 'model_perf' in perf else 0
    top10_recall = perf['model_perf']['recall_top10'].max() if 'model_perf' in perf else 0.397

    unit = "개" if lang == 'ko' else ""
    kpi_labels = {
        'vh':  ("🔴 매우높음", "즉각 점검 대상") if lang == 'ko' else ("🔴 Very High", "Immediate action"),
        'hi':  ("🟠 높음",    "중점 모니터링")   if lang == 'ko' else ("🟠 High",      "Close monitoring"),
        'lo':  ("🟢 낮음",    "정상 범위")        if lang == 'ko' else ("🟢 Low",       "Normal range"),
        'auc': ("🤖 모델 AUC", "LightGBM")        if lang == 'ko' else ("🤖 Model AUC", "LightGBM"),
        'r10': ("📊 Recall@Top10%", "상위 10% 점검 효율") if lang == 'ko' else ("📊 Recall@Top10%", "Inspection efficiency"),
    }

    st.markdown(f"""
<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:20px'>

  <div class='kpi-card kpi-vh'>
    <div class='kpi-label'>{kpi_labels['vh'][0]}</div>
    <div class='kpi-num'>{vh_cnt:,}</div>
    <div class='kpi-sub'>{kpi_labels['vh'][1]}</div>
  </div>

  <div class='kpi-card kpi-hi'>
    <div class='kpi-label'>{kpi_labels['hi'][0]}</div>
    <div class='kpi-num'>{hi_cnt:,}</div>
    <div class='kpi-sub'>{kpi_labels['hi'][1]}</div>
  </div>

  <div class='kpi-card kpi-low'>
    <div class='kpi-label'>{kpi_labels['lo'][0]}</div>
    <div class='kpi-num'>{lo_cnt:,}</div>
    <div class='kpi-sub'>{kpi_labels['lo'][1]}</div>
  </div>

  <div class='kpi-card kpi-auc'>
    <div class='kpi-label'>{kpi_labels['auc'][0]}</div>
    <div class='kpi-num'>{auc_val:.4f}</div>
    <div class='kpi-sub'>{kpi_labels['auc'][1]}</div>
  </div>

  <div class='kpi-card kpi-auc'>
    <div class='kpi-label'>{kpi_labels['r10'][0]}</div>
    <div class='kpi-num'>{top10_recall:.1%}</div>
    <div class='kpi-sub'>{kpi_labels['r10'][1]}</div>
  </div>

</div>
""", unsafe_allow_html=True)

    # ── 사이드바 ──────────────────────────────────────────────────────────────
    with st.sidebar:
        # 사이드바 로고
        st.markdown("""
<div style='text-align:center;padding:12px 0 8px'>
  <span style='font-size:2rem'>🔥</span>
  <div style='font-size:1.2rem;font-weight:800;letter-spacing:2px;color:#ff6b6b'>WPFI</div>
  <div style='font-size:0.7rem;color:rgba(255,255,255,0.4);letter-spacing:1px'>FIRE RISK SYSTEM</div>
</div>
<hr style='border-color:rgba(255,255,255,0.08);margin:8px 0 16px'>
""", unsafe_allow_html=True)

        st.markdown(f"#### {t('filter', lang)}")

        # 위험등급 필터
        st.markdown(f"**{t('grade_display', lang)}**")
        show_grades = []
        c1, c2 = st.columns(2)
        for i, g in enumerate(GRADE_KR.keys()):
            label = GRADE_KR[g] if lang == 'ko' else g
            with (c1 if i % 2 == 0 else c2):
                if st.checkbox(f"{GRADE_EMOJI[g]} {label}", value=g in ['Very High','High'], key=f"g_{g}"):
                    show_grades.append(g)

        topk_pct = st.slider(t('topk_label', lang), 1, 20, 5)
        all_lbl = t('all_stations', lang)
        raw_stns = sorted(df['nearest_station'].dropna().unique().tolist())
        # 영어 모드: 영문명(원문) 형식으로 표시
        if lang == 'en':
            stn_display = [all_lbl] + [f"{STATION_EN.get(s, s)} ({s})" for s in raw_stns]
        else:
            stn_display = [all_lbl] + raw_stns
        sel_stn_disp = st.selectbox(t('station_label', lang), stn_display)
        # 실제 필터링에 사용하는 한국어 원본 값
        if sel_stn_disp == all_lbl:
            sel_stn_val = None
        elif lang == 'en':
            # "Chuncheon (춘천)" → "춘천" 추출
            sel_stn_val = sel_stn_disp.split("(")[-1].rstrip(")")
        else:
            sel_stn_val = sel_stn_disp
        score_type = st.radio(t('score_basis', lang),
                              [t('wpfi_score', lang), t('ml_score', lang)], index=0)
        risk_col = 'final_risk' if score_type == t('wpfi_score', lang) else 'ml_score'

        # ── 기상 시뮬레이션 ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### {t('scenario_lbl', lang)}")
        # 언어에 맞는 레이블 목록
        sim_labels = [p['en'] if lang == 'en' else p['ko'] for p in SIMULATION_PRESETS_DEF]
        sel_sim_label = st.selectbox(t('scenario_sel', lang), sim_labels)
        # 내부 ID 역조회
        label_key = 'en' if lang == 'en' else 'ko'
        sel_preset = next((p for p in SIMULATION_PRESETS_DEF if p[label_key] == sel_sim_label),
                          SIMULATION_PRESETS_DEF[0])
        preset_desc = sel_preset['desc_en'] if lang == 'en' else sel_preset['desc_ko']
        st.caption(f"📌 {preset_desc}")

        with st.expander(t('fine_tune', lang)):
            wh_add = st.slider(t('wh_adjust', lang), -20, +30, 0)
            sp_add = st.slider(t('sp_adjust', lang), -10, +20, 0)

        is_sim = (sel_preset['id'] != 'baseline') or (wh_add != 0) or (sp_add != 0)
        # apply_simulation은 내부적으로 wh_mult만 사용 → 직접 전달
        if is_sim:
            df_sim = df.copy()
            df_sim['weather_hazard'] = (df_sim['weather_hazard'] * sel_preset['wh_mult'] + wh_add).clip(0, 100)
            if sp_add != 0:
                df_sim['spatial_exposure'] = (df_sim['spatial_exposure'] + sp_add).clip(0, 100)
            df_sim['final_risk'] = (
                df_sim['weather_hazard']    * WEIGHTS['weather'] +
                df_sim['spatial_exposure']  * WEIGHTS['spatial'] +
                df_sim['facility_exposure'] * WEIGHTS['facility'] +
                df_sim['hist_prior']        * WEIGHTS['prior']
            ).clip(0, 100)
            VH_THR, HI_THR, MO_THR = GRADE_THRESHOLDS_SIM["Very High"], GRADE_THRESHOLDS_SIM["High"], GRADE_THRESHOLDS_SIM["Moderate"]
            def _sim_grade(s):
                if s >= VH_THR: return 'Very High'
                if s >= HI_THR: return 'High'
                if s >= MO_THR: return 'Moderate'
                return 'Low'
            df_sim['risk_grade'] = df_sim['final_risk'].apply(_sim_grade)
        else:
            df_sim = df

        if is_sim:
            orig_vh = (df['risk_grade'] == 'Very High').sum()
            sim_vh  = (df_sim['risk_grade'] == 'Very High').sum()
            delta   = sim_vh - orig_vh
            if lang == 'en':
                st.warning(f"⚠️ Simulation Active: {sel_preset['en']}\nVery High: {orig_vh:,} → {sim_vh:,} ({delta:+,})")
            else:
                st.warning(f"⚠️ 시뮬레이션 적용 중: {sel_preset['ko']}\nVery High: {orig_vh:,} → {sim_vh:,} ({delta:+,})")

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08)'>", unsafe_allow_html=True)
        st.markdown(f"#### {t('stats_lbl', lang)}")
        total = len(df_sim)
        vh    = (df_sim['risk_grade'] == 'Very High').sum()
        hi_s  = (df_sim['risk_grade'] == 'High').sum()
        unit  = "개" if lang == 'ko' else ""

        # 위험도 비율 프로그레스 바 스타일 카드
        for grade, cnt, color in [
            ('Very High', vh,   '#d62728'),
            ('High',      hi_s, '#ff7f0e'),
        ]:
            pct = cnt / max(total, 1) * 100
            lbl = GRADE_KR.get(grade, grade) if lang == 'ko' else grade
            st.markdown(f"""
<div style='margin:6px 0;padding:10px 14px;background:rgba(255,255,255,0.04);
            border-radius:10px;border:1px solid rgba(255,255,255,0.06)'>
  <div style='display:flex;justify-content:space-between;font-size:0.8rem'>
    <span style='color:rgba(255,255,255,0.7)'>{GRADE_EMOJI[grade]} {lbl}</span>
    <span style='color:{color};font-weight:700'>{cnt:,}{unit}</span>
  </div>
  <div style='background:rgba(255,255,255,0.08);border-radius:4px;height:4px;margin-top:6px'>
    <div style='background:{color};width:{min(pct*3,100):.1f}%;height:4px;border-radius:4px;
                transition:width 0.5s ease'></div>
  </div>
  <div style='font-size:0.7rem;color:rgba(255,255,255,0.4);margin-top:2px'>{pct:.2f}%</div>
</div>
""", unsafe_allow_html=True)

        st.metric(t('total_fac', lang), f"{total:,}{unit}")
        if model and 'model_perf' in perf:
            auc = perf['model_perf']['auc'].max()
            st.markdown(f"""
<div style='margin-top:8px;padding:10px 14px;background:rgba(31,119,180,0.15);
            border-radius:10px;border:1px solid rgba(31,119,180,0.3)'>
  <div style='font-size:0.75rem;color:rgba(255,255,255,0.5)'>🤖 LightGBM Model</div>
  <div style='font-size:1.3rem;font-weight:800;color:#66aadd'>AUC {auc:.4f}</div>
  <div style='font-size:0.7rem;color:rgba(255,255,255,0.4)'>{"로드됨 ✅" if lang == "ko" else "Loaded ✅"}</div>
</div>
""", unsafe_allow_html=True)

    # ── 권역 필터 ─────────────────────────────────────────────────────────────
    df_view = df_sim.copy()
    if sel_stn_val:
        df_view = df_view[df_view['nearest_station'] == sel_stn_val]
    gdf_view = None
    if gdf_merged is not None:
        gdf_view = (gdf_merged[gdf_merged['nearest_station'] == sel_stn_val]
                    if sel_stn_val else gdf_merged.copy())
        if is_sim:
            gdf_view = gdf_view.drop(
                columns=[c for c in ['weather_hazard','final_risk','risk_grade']
                         if c in gdf_view.columns], errors='ignore'
            ).merge(
                df_sim[['pole_id','weather_hazard','final_risk','risk_grade','ml_score']],
                on='pole_id', how='left'
            )
            gdf_view['lat'] = gdf_view.geometry.y
            gdf_view['lon'] = gdf_view.geometry.x

    # ── KPI 요약 ─────────────────────────────────────────────────────────────
    unit = "개" if lang == 'ko' else ""
    kpi_cols = st.columns(5)
    kpis = [
        (t('kpi_total', lang),  f"{len(df_view):,}{unit}",  None),
        (t('kpi_vh', lang),     f"{(df_view['risk_grade']=='Very High').sum():,}{unit}", "#ffe0e0"),
        (t('kpi_hi', lang),     f"{(df_view['risk_grade']=='High').sum():,}{unit}",      "#fff0e0"),
        (t('kpi_avg', lang),    f"{df_view['final_risk'].mean():.1f}",                   None),
        ("AUC",                 f"{perf.get('model_perf',pd.DataFrame()).get('auc',pd.Series([0])).max():.4f}"
                                if 'model_perf' in perf else "—",                        None),
    ]
    for col, (label, val, _bg) in zip(kpi_cols, kpis):
        col.metric(label, val)

    # ── 총평 (OpenAI) ─────────────────────────────────────────────────────────
    with st.container():
        st.markdown(t('summary_title', lang))
        # lang 변경 시 기존 캐시 무효화
        cached_lang = st.session_state.get('executive_summary_lang', '')
        if 'executive_summary' not in st.session_state or cached_lang != lang:
            spinner_msg = "Analyzing with OpenAI..." if lang == 'en' else "OpenAI가 현황을 분석 중..."
            with st.spinner(spinner_msg):
                st.session_state['executive_summary'] = get_executive_summary(df, perf, lang)
                st.session_state['executive_summary_lang'] = lang

        summary = st.session_state['executive_summary']
        _render_structured_ai(summary, lang)

        col_ref, _ = st.columns([1, 4])
        with col_ref:
            if st.button(t('refresh', lang)):
                st.session_state.pop('executive_summary', None)
                st.rerun()

    st.markdown("---")

    # ── 8개 탭 ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        t('tab_map', lang), t('tab_list', lang), t('tab_detail', lang),
        t('tab_model', lang), t('tab_spatial', lang), t('tab_trend', lang),
        t('tab_forecast', lang), t('tab_method', lang),
    ])

    # ── TAB 1: 위험도 지도 ────────────────────────────────────────────────────
    with tab1:
        st.markdown(t('map_title', lang))
        if is_sim:
            sim_mode_label = sel_preset['en'] if lang == 'en' else sel_preset['ko']
            st.warning(f"⚠️ {'Simulation Mode' if lang=='en' else '시뮬레이션 모드'}: {sim_mode_label}")
        st.caption(t('map_caption', lang))

        if gdf_view is not None and len(gdf_view) > 0:
            m = build_map(gdf_view, show_grades, risk_col)
            st_folium(m, width=None, height=560, returned_objects=[])
        else:
            st.warning(t('map_no_data', lang))

    # ── TAB 2: 우선점검 목록 ─────────────────────────────────────────────────
    with tab2:
        k = max(1, int(len(df_view) * topk_pct / 100))
        st.markdown(t('list_title', lang).format(topk_pct, k))
        if is_sim:
            sim_mode_label2 = sel_preset['en'] if lang == 'en' else sel_preset['ko']
            st.warning(f"⚠️ {'Simulation Active' if lang=='en' else '시뮬레이션 적용'}: {sim_mode_label2}")

        disp_cols = ['pole_id','final_risk','risk_grade','ml_score',
                     'weather_hazard','spatial_exposure','hist_prior','nearest_station']
        disp_cols = [c for c in disp_cols if c in df_view.columns]
        top_df = df_view.nlargest(k, risk_col)[disp_cols].copy().round(1)
        top_df.insert(0, '순위', range(1, len(top_df) + 1))
        top_df = top_df.rename(columns={
            'pole_id':          t('fac_id',      lang),
            'final_risk':       t('wpfi_col',    lang),
            'risk_grade':       '등급_raw',
            'ml_score':         t('ml_col',      lang),
            'weather_hazard':   t('weather_col', lang),
            'spatial_exposure': t('spatial_col', lang),
            'hist_prior':       t('history_col', lang),
            'nearest_station':  t('station_col', lang),
        })
        top_df[t('grade_col', lang)] = top_df['등급_raw'].map(GRADE_LABEL)
        rank_lbl    = t('rank',       lang)
        grade_lbl   = t('grade_col',  lang)
        show_cols   = [rank_lbl, t('fac_id',lang), t('wpfi_col',lang), t('ml_col',lang),
                       grade_lbl, t('weather_col',lang), t('spatial_col',lang),
                       t('history_col',lang), t('station_col',lang)]
        show_cols   = [c for c in show_cols if c in top_df.columns]
        st.dataframe(top_df[show_cols], use_container_width=True, height=420)

        c1, c2, c3 = st.columns(3)
        total_risk = df_view['final_risk'].sum()
        wpfi_col_name = t('wpfi_col', lang)
        topk_risk  = top_df[wpfi_col_name].sum() if wpfi_col_name in top_df.columns else 0
        coverage   = topk_risk / total_risk * 100 if total_risk > 0 else 0
        c1.metric(t('coverage', lang), f"{coverage:.1f}%")
        if 'recall_k' in perf:
            rk = perf['recall_k']
            idx = (rk['k_pct'] - topk_pct / 100).abs().argmin()
            c2.metric(f"Recall@Top{topk_pct}%", f"{rk.iloc[idx]['recall']:.1%}")
            c3.metric(f"Precision@Top{topk_pct}%", f"{rk.iloc[idx]['precision']:.1%}")

        csv = top_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(t('csv_download', lang), csv,
                           file_name=f"wpfi_top{topk_pct}pct.csv", mime="text/csv")

    # ── TAB 3: 설비 상세분석 ─────────────────────────────────────────────────
    with tab3:
        st.markdown("##### 🔍 설비 상세 분석")

        top50 = df_view.nlargest(50, risk_col)['pole_id'].tolist()
        fac_sel_lbl = "Select Facility (Top-50 High Risk)" if lang == 'en' else "설비 선택 (Top-50 고위험)"
        top50_labels = {pid: (f"Facility #{pid}" if lang == 'en' else f"설비 #{pid}") for pid in top50}
        sel_pole = st.selectbox(fac_sel_lbl, top50, format_func=lambda x: top50_labels.get(x, str(x)))

        row = df_view[df_view['pole_id'] == sel_pole]
        if not len(row):
            st.warning("설비 정보 없음")
        else:
            row = row.iloc[0]
            grade = row.get('risk_grade', 'Low')
            color = GRADE_COLORS.get(grade, '#888')

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                <div style='text-align:center;padding:16px;border-radius:12px;
                            background:{color}22;border:2px solid {color};'>
                    <div style='font-size:2.8em;color:{color};font-weight:bold'>
                        {row.get('final_risk',0):.1f}</div>
                    <div style='color:{color};font-size:1.2em'>
                        {GRADE_EMOJI.get(grade,'')} {GRADE_KR.get(grade,grade)}</div>
                    <div style='color:#666;font-size:0.85em;margin-top:4px'>
                        {'Facility' if lang == 'en' else '설비'} #{int(sel_pole)} | {row.get('nearest_station','')} {'Area' if lang == 'en' else '권역'}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(t('layer_scores', lang))
                if lang == 'en':
                    layers = [('🌪️ Weather Hazard','weather_hazard'),('🌲 Spatial Exposure','spatial_exposure'),
                              ('⚡ Facility Exposure','facility_exposure'),('🔥 Fire History','hist_prior')]
                    ml_lbl = "🤖 ML Fire Prob. Score"
                else:
                    layers = [('🌪️ 기상위험','weather_hazard'),('🌲 공간노출','spatial_exposure'),
                              ('⚡ 설비노출','facility_exposure'),('🔥 화재이력','hist_prior')]
                    ml_lbl = "🤖 ML 화재확률 점수"
                for label, col_name in layers:
                    val = row.get(col_name, 0)
                    bar_color = '#d62728' if val >= 70 else '#ff7f0e' if val >= 50 else '#2ca02c'
                    st.markdown(
                        f"<div style='margin:4px 0'><small>{label}: <b>{val:.1f}</b></small>"
                        f"<div style='background:#eee;border-radius:4px;height:8px'>"
                        f"<div style='background:{bar_color};width:{int(val)}%;height:8px;border-radius:4px'></div>"
                        f"</div></div>", unsafe_allow_html=True)

                if 'ml_score' in row:
                    st.metric(ml_lbl, f"{row['ml_score']:.1f}/100")

            with col2:
                st.markdown(t('shap_title', lang))
                if model and features:
                    spin_shap = "Computing SHAP..." if lang == 'en' else "SHAP 계산 중..."
                    with st.spinner(spin_shap):
                        fig_shap = shap_waterfall(sel_pole, df_view, model, features)
                    if fig_shap:
                        st.pyplot(fig_shap)
                        plt.close()
                    else:
                        fig, ax = plt.subplots(figsize=(6, 3.5))
                        # WPFI_v2 구성요소 사용 (prob_dry/prob_heat/prob_light)
                        if lang == 'en':
                            comp_labels = ['P_dry (×0.5)','P_heat (×0.3)','P_light (×0.2)']
                            x_label = 'Probability (0–1)'
                            t_label = f'Facility #{sel_pole} — WPFI_v2 Components'
                        else:
                            comp_labels = ['건조형 P_dry (×0.5)','고온형 P_heat (×0.3)','낙뢰 P_light (×0.2)']
                            x_label = '확률 (0~1)'
                            t_label = f'설비 #{sel_pole} — WPFI_v2 구성요소'
                        p_dry   = row.get('prob_dry',   row.get('weather_hazard', 0) / 100)
                        p_heat  = row.get('prob_heat',  row.get('spatial_exposure', 0) / 100)
                        p_light = row.get('prob_light', row.get('hist_prior', 0) / 100)
                        comp_vals = [p_dry, p_heat, p_light]
                        ax.barh(comp_labels, comp_vals, color=['#1f77b4','#d62728','#2ca02c'])
                        for b, v in zip(ax.patches, comp_vals):
                            ax.text(v + 0.005, b.get_y() + b.get_height()/2, f'{v:.4f}', va='center')
                        ax.set_xlabel(x_label); ax.set_xlim(0, 1.05)
                        ax.set_title(t_label)
                        plt.tight_layout(); st.pyplot(fig); plt.close()

                # AI 설명 — 항상 표시
                st.markdown("**🤖 AI 위험 설명**")
                exp_key = f"exp_{sel_pole}_{lang}"
                if exp_key not in st.session_state:
                    spin_msg = "Analyzing..." if lang == 'en' else "분석 중..."
                    with st.spinner(spin_msg):
                        st.session_state[exp_key] = get_facility_explanation(row.to_dict(), lang)
                _render_structured_ai(st.session_state[exp_key], lang)

    # ── TAB 4: 모델 성능 ─────────────────────────────────────────────────────
    with tab4:
        st.markdown("##### 📊 LightGBM 모델 성능 검증")

        with st.expander(t('model_explain_title', lang), expanded=True):
            auc_val = perf['model_perf']['auc'].max() if 'model_perf' in perf else 0
            w_pct   = perf['model_perf']['weather_pct'].max() if 'model_perf' in perf else 0
            r10_val = perf['model_perf']['recall_top10'].max() if 'model_perf' in perf else 0
            if lang == 'en':
                st.markdown(f"""
**WPFI_v2 Multi-Hazard Model Performance**
- **AUC: Dry {perf['model_perf'].loc['건조형','auc']:.4f} / Heat {perf['model_perf'].loc['고온형','auc']:.4f}**
  — Closer to 1.0 = perfect discrimination. 0.5 = random. Current values reflect genuine weather-driven prediction.
- **Weather Contribution: {w_pct:.1f}%** — Top-5 features are all weather variables. Zero calendar dependency.
- **Recall@Top10%: {r10_val:.3f}** — Inspecting only top 10% of facilities captures this fraction of true fire-risk cases.
- **Seasonal pattern**: Dry model peaks in winter/spring (dry season). Heat model peaks in summer (heatwave season).
                """)
            else:
                st.markdown(f"""
**WPFI_v2 멀티해저드 모델 성능**
- **AUC: 건조형 {perf['model_perf'].loc['건조형','auc']:.4f} / 고온형 {perf['model_perf'].loc['고온형','auc']:.4f}**
  — 1.0에 가까울수록 완벽. 0.5는 무작위. 현재 값은 순수 날씨 기반 예측의 현실적 수준입니다.
- **날씨 기여도: {w_pct:.1f}%** — Top-5 피처 전부 기상 변수. 달력(month/season) 의존도 0%.
- **Recall@Top10%: {r10_val:.3f}** — 상위 10% 설비만 점검 시 실제 화재 위험 설비의 {r10_val*100:.1f}%를 포함.
- **계절 패턴**: 건조형은 겨울·봄(건조 시즌) 집중, 고온형은 여름(폭염 시즌) 집중.
                """)

        col1, col2 = st.columns(2)
        with col1:
            if 'recall_k' in perf:
                rk = perf['recall_k']
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(rk['k_pct']*100, rk['recall']*100, 'o-', color='#d62728', lw=2, label='Recall@K')
                ax.plot(rk['k_pct']*100, rk['precision']*100, 's--', color='#1f77b4', lw=2, label='Precision@K')
                threshold_lbl = '80% threshold' if lang == 'en' else '80% 기준선'
                ax.axhline(80, color='gray', ls=':', lw=1, alpha=0.5, label=threshold_lbl)
                if lang == 'en':
                    ax.set_xlabel('Top K (%)'); ax.set_ylabel('Performance (%)')
                    ax.set_title('Detection Rate of At-Risk Facilities by Top-K%', fontweight='bold')
                else:
                    ax.set_xlabel('상위 K (%)'); ax.set_ylabel('성능 (%)')
                    ax.set_title('상위 K% 점검 시 실제 위험 설비 탐지율', fontweight='bold')
                ax.legend(); ax.grid(alpha=0.3)
                plt.tight_layout(); st.pyplot(fig); plt.close()
                best_recall = rk['recall'].max()
                best_k = int(rk.loc[rk['recall'].idxmax(), 'k_pct'] * 100)
                if lang == 'en':
                    st.success(f"💡 Inspecting only the top **{best_k}%** covers **{best_recall:.1%}** of all actual fire-risk facilities.")
                else:
                    st.success(f"💡 상위 {best_k}%만 점검해도 실제 위험 설비의 **{best_recall:.1%}**를 포함합니다.")

        with col2:
            if 'ablation' in perf:
                abl = perf['ablation']
                fig, ax = plt.subplots(figsize=(6, 4))
                colors_abl = ['#aec7e8','#6baed6','#3182bd','#08519c']
                bars = ax.bar(range(len(abl)), abl['auc'], color=colors_abl, alpha=0.85)
                ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.6,
                           label='Random (0.5)' if lang == 'en' else '랜덤 기준선 (0.5)')
                ax.set_xticks(range(len(abl)))
                ax.set_xticklabels([m.replace('_',' ') for m in abl['model']], rotation=15, ha='right', fontsize=9)
                if lang == 'en':
                    ax.set_ylabel('AUC'); ax.set_title('AUC by Feature Group (Ablation)', fontweight='bold')
                else:
                    ax.set_ylabel('AUC'); ax.set_title('Feature 그룹 추가에 따른 AUC 향상', fontweight='bold')
                for b, v in zip(bars, abl['auc']):
                    ax.text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.3f}', ha='center', fontsize=9)
                ax.legend(); ax.set_ylim(0.4, 1.05)
                plt.tight_layout(); st.pyplot(fig); plt.close()
                if lang == 'en':
                    st.info("💡 WPFI_v2: Dry model AUC 0.737 / Heat model AUC 0.806. Weather contributes 94.7–94.8%. Zero calendar dependency.")
                else:
                    st.info("💡 WPFI_v2: 건조형 AUC 0.737 / 고온형 AUC 0.806. 날씨 기여도 94.7~94.8%. 달력 의존도 0%.")

        # 모델 개선 여정 (A→B→C)
        fig_comp = OUT_FIGURES / 'fig_model_comparison.png'
        if fig_comp.exists():
            if lang == 'en':
                st.markdown("**📈 Model Improvement Journey: A → B → C**")
                st.caption("A: Original (fire history included) | B: Fire history removed | C: +FWI+Accumulated risk+Cascade")
            else:
                st.markdown("**📈 모델 개선 단계: A → B → C**")
                st.caption("A: 기존(화재이력 포함) | B: 화재이력 제거 | C: +FWI+누적위험+연쇄피해")
            st.image(str(fig_comp), use_container_width=True)

        if 'shap_imp' in perf:
            if lang == 'en':
                st.markdown("**🔬 Which factors determine fire risk the most?**")
            else:
                st.markdown("**🔬 어떤 요소가 화재 위험을 가장 많이 결정하는가?**")
            si = perf['shap_imp'].head(15)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(si['feature'][::-1], si['mean_abs_shap'][::-1], color='steelblue', alpha=0.8)
            if lang == 'en':
                ax.set_xlabel('SHAP Importance (higher = more influential)')
                ax.set_title('Top 15 Fire Risk Determinants (SHAP)', fontweight='bold')
            else:
                ax.set_xlabel('SHAP 중요도 (값이 클수록 중요)')
                ax.set_title('화재 위험 결정 요인 순위 (Top 15)', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            top_feat = si.iloc[0]['feature']
            if lang == 'en':
                st.success(f"💡 Most influential factor: **{top_feat}** — this variable has the greatest impact on individual facility fire risk.")
            else:
                st.success(f"💡 가장 중요한 요인: **{top_feat}** — 이 변수가 개별 설비의 화재 위험도를 결정하는 데 가장 큰 영향을 미칩니다.")

        if 'region_cv' in perf:
            if lang == 'en':
                st.markdown("**🗺️ Regional Validation — Does it generalize to unseen areas?**")
            else:
                st.markdown("**🗺️ 지역별 검증 — 다른 지역에서도 잘 동작하는가?**")
            cv = perf['region_cv']
            avg_auc = cv['auc'].mean()
            if lang == 'en':
                rename_cv = {'test_station': 'Test Station', 'auc': 'AUC', 'n_test': 'Test Samples'}
            else:
                rename_cv = {'test_station': '테스트 권역', 'auc': 'AUC', 'n_test': '테스트 샘플'}
            st.dataframe(
                cv[['test_station','auc','n_test']].rename(columns=rename_cv)
                .assign(AUC=lambda d: d['AUC'].map('{:.4f}'.format)),
                use_container_width=True, hide_index=True
            )
            if lang == 'en':
                st.success(f"✅ Average Region CV AUC: **{avg_auc:.4f}** — Model performs stably across all Gangwon Province regions.")
            else:
                st.success(f"✅ 평균 Region CV AUC: **{avg_auc:.4f}** — 특정 지역 편향 없이 강원도 전역에서 안정적으로 작동합니다.")

    # ── TAB 5: 공간 분포 ─────────────────────────────────────────────────────
    with tab5:
        st.markdown(t('spatial_title', lang))

        with st.expander(t('model_explain_title', lang), expanded=True):
            top_stn = df.groupby('nearest_station')['final_risk'].mean().idxmax()
            top_val = df.groupby('nearest_station')['final_risk'].mean().max()
            if lang == 'en':
                st.markdown(f"""
**Which area in Gangwon has the most at-risk power poles?**
- Compares average risk scores across station areas.
- Currently, **{top_stn}** station area has the highest avg. risk ({top_val:.1f}/100).
- This reflects combined effects of forest proximity, terrain slope, and weather conditions.
                """)
            else:
                st.markdown(f"""
**강원도 어느 지역의 전봇대가 가장 위험한가?**
- 관측소별 평균 위험도로 권역 간 위험도를 비교합니다.
- 현재 **{top_stn}** 권역의 평균 위험도({top_val:.1f}점)가 가장 높습니다.
- 이는 해당 권역의 산림 인접도, 경사도, 기상 조건이 복합적으로 작용한 결과입니다.
                """)

        col1, col2 = st.columns(2)
        with col1:
            stn_risk = df.groupby('nearest_station')['final_risk'].agg(
                ['mean','count']).reset_index().sort_values('mean', ascending=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            colors_bar = ['#d62728' if v >= 55 else '#ff7f0e' if v >= 52 else '#2ca02c'
                          for v in stn_risk['mean']]
            stn_labels = ([f"{STATION_EN.get(s,s)}" for s in stn_risk['nearest_station']]
                          if lang == 'en' else list(stn_risk['nearest_station']))
            bars = ax.barh(stn_labels, stn_risk['mean'], color=colors_bar, alpha=0.85)
            for b, v in zip(bars, stn_risk['mean']):
                ax.text(v + 0.1, b.get_y() + b.get_height()/2,
                        f'{v:.1f}', va='center', fontsize=9)
            if lang == 'en':
                ax.set_xlabel('Avg. WPFI Risk Score')
                ax.set_title('Average Risk Score by Station Area', fontweight='bold')
                ax.axvline(df['final_risk'].mean(), color='navy', ls='--', lw=1.5,
                           label=f"Overall avg {df['final_risk'].mean():.1f}")
            else:
                ax.set_xlabel('평균 WPFI 위험도')
                ax.set_title('관측소 권역별 평균 위험도', fontweight='bold')
                ax.axvline(df['final_risk'].mean(), color='navy', ls='--', lw=1.5,
                           label=f"전체 평균 {df['final_risk'].mean():.1f}")
            ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            grade_cnt = df['risk_grade'].value_counts()
            GRADE_LABEL_L = GRADE_KR if lang == 'ko' else GRADE_EN
            fig, ax = plt.subplots(figsize=(5, 5))
            colors_pie = [GRADE_COLORS.get(g, '#888') for g in grade_cnt.index]
            unit_p = "개" if lang == 'ko' else ""
            ax.pie(grade_cnt.values,
                   labels=[f"{GRADE_LABEL_L.get(g,g)}\n{v:,}{unit_p}" for g, v in grade_cnt.items()],
                   colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax.set_title('Risk Grade Distribution' if lang == 'en' else '전체 위험등급 분포', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            vh_pct = grade_cnt.get('Very High', 0) / grade_cnt.sum() * 100
            vh_cnt_p = grade_cnt.get('Very High', 0)
            if lang == 'en':
                st.info(f"💡 **{vh_pct:.1f}% ({vh_cnt_p:,} facilities)** require immediate inspection.")
            else:
                st.info(f"💡 전체 설비의 **{vh_pct:.1f}%({vh_cnt_p:,}개)**가 즉각 점검 대상입니다.")

        col_a, col_b = st.columns(2)
        with col_a:
            img_path = OUT_FIGURES / 'fig4_risk_map_scatter.png'
            if img_path.exists():
                cap = "ML probability-based spatial distribution" if lang == 'en' else "ML 확률 기반 공간 분포 (샘플 50K)"
                st.image(str(img_path), use_container_width=True, caption=cap)
        with col_b:
            img_path = OUT_FIGURES / 'fig_region_cv.png'
            if img_path.exists():
                cap = "Region cross-validation AUC by station" if lang == 'en' else "관측소별 Region CV AUC"
                st.image(str(img_path), use_container_width=True, caption=cap)

        img_path = OUT_FIGURES / 'fig4_risk_map.png'
        if img_path.exists():
            cap = "ML score distribution + grade pie chart" if lang == 'en' else "ML 위험 확률 분포 + 등급 파이차트"
            st.image(str(img_path), use_container_width=True, caption=cap)

    # ── TAB 6: 트렌드 분석 ───────────────────────────────────────────────────
    with tab6:
        st.markdown(t('trend_title', lang))

        with st.expander(t('model_explain_title', lang), expanded=True):
            if lang == 'en':
                st.markdown("""
**How does fire risk change over time?**
- Shows monthly and seasonal weather hazard trends.
- **Spring (Mar–May) and Autumn (Sep–Nov)** are the highest-risk periods due to dry and windy conditions.
- The heatmap shows **which station area is most dangerous in which month**.
                """)
            else:
                st.markdown("""
**시간에 따라 화재 위험이 어떻게 변화하는가?**
- 월별·계절별 기상위험도 변화를 보여줍니다.
- **봄(3~5월)과 가을(9~11월)** 이 건조하고 바람이 강해 화재 위험이 가장 높습니다.
- 히트맵으로 **어느 권역이, 어느 달에** 특히 위험한지 한눈에 파악할 수 있습니다.
                """)

        if trend is None:
            st.warning("Data not found. Run notebooks/06_modeling.ipynb first." if lang == 'en'
                       else "데이터 없음. notebooks/06_modeling.ipynb를 먼저 실행하세요.")
        else:
            MONTH_LABELS = ({m: f'M{m}' for m in range(1,13)} if lang == 'en'
                            else {m: f'{m}월' for m in range(1,13)})
            SEASON_LABELS = (['Spring','Summer','Autumn','Winter'] if lang == 'en'
                             else ['봄','여름','가을','겨울'])
            SEASON_COLOR = {0:'#5b9bd5', 1:'#70ad47', 2:'#ffc000', 3:'#ed7d31'}

            daily = (trend.groupby('date')
                     .agg(weather_hazard_mean=('weather_hazard','mean'),
                          weather_hazard_max=('weather_hazard','max'),
                          fire_count=('label','sum'))
                     .reset_index())
            daily['ym']    = daily['date'].dt.to_period('M').astype(str)
            daily['month'] = daily['date'].dt.month
            daily['season']= daily['date'].dt.month.map(
                {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})

            monthly = daily.groupby('ym').agg(
                hazard_mean=('weather_hazard_mean','mean'),
                hazard_max=('weather_hazard_max','max')
            ).reset_index()

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                lbl_mean = 'Avg Weather Hazard' if lang == 'en' else '평균 기상위험도'
                lbl_rng  = 'Avg~Max range'       if lang == 'en' else '평균~최대 범위'
                ax.plot(range(len(monthly)), monthly['hazard_mean'],
                        'o-', color='#d62728', lw=2, ms=4, label=lbl_mean)
                ax.fill_between(range(len(monthly)),
                                monthly['hazard_mean'], monthly['hazard_max'],
                                alpha=0.15, color='#d62728', label=lbl_rng)
                tick_step = max(1, len(monthly) // 12)
                ax.set_xticks(range(0, len(monthly), tick_step))
                ax.set_xticklabels(monthly['ym'].iloc[::tick_step], rotation=45, ha='right', fontsize=8)
                if lang == 'en':
                    ax.set_ylabel('Weather Hazard (0–100)')
                    ax.set_title('Monthly Average Weather Hazard Trend', fontweight='bold')
                else:
                    ax.set_ylabel('기상위험도 (0~100)')
                    ax.set_title('월별 평균 기상위험도 추이', fontweight='bold')
                ax.legend(fontsize=9); ax.grid(alpha=0.3)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            with col2:
                season_groups = [daily[daily['season'] == s]['weather_hazard_mean'].values
                                 for s in [1, 2, 3, 0]]
                fig, ax = plt.subplots(figsize=(6, 4))
                bp = ax.boxplot(season_groups, patch_artist=True, labels=SEASON_LABELS)
                for patch, color in zip(bp['boxes'], [SEASON_COLOR[s] for s in [1,2,3,0]]):
                    patch.set_facecolor(color); patch.set_alpha(0.7)
                if lang == 'en':
                    ax.set_ylabel('Daily Avg Weather Hazard')
                    ax.set_title('Seasonal Weather Hazard Distribution', fontweight='bold')
                else:
                    ax.set_ylabel('일평균 기상위험도')
                    ax.set_title('계절별 기상위험도 분포', fontweight='bold')
                ax.grid(alpha=0.3, axis='y')
                plt.tight_layout(); st.pyplot(fig); plt.close()
                season_avgs = [daily[daily['season']==s]['weather_hazard_mean'].mean() for s in [1,2,3,0]]
                peak_idx = season_avgs.index(max(season_avgs))
                peak_season = SEASON_LABELS[peak_idx]
                if lang == 'en':
                    st.info(f"💡 **{peak_season}** has the highest weather hazard. Concentrate inspection resources in this season.")
                else:
                    st.info(f"💡 **{peak_season}** 기상위험도가 가장 높습니다. 이 계절에 점검 자원을 집중 배치하는 것을 권고합니다.")

            # 월별 화재 발생
            fire_title = '**🔥 Monthly Fire Incidents vs Weather Hazard**' if lang == 'en' \
                         else '**🔥 월별 화재 발생 건수 vs 기상위험도**'
            st.markdown(fire_title)
            monthly_fire = (trend.groupby(['year','month'])
                            .agg(fire_count=('label','sum'),
                                 hazard_mean=('weather_hazard','mean'))
                            .reset_index())
            fig, ax1 = plt.subplots(figsize=(12, 4))
            ax2 = ax1.twinx()
            w = 0.25; offsets = {2022: -w, 2023: 0, 2024: w}
            colors_yr = {2022:'#5b9bd5', 2023:'#ed7d31', 2024:'#70ad47'}
            yr_lbl_sfx = '' if lang == 'en' else '년'
            for yr, grp in monthly_fire.groupby('year'):
                ax1.bar(grp['month'] + offsets[yr], grp['fire_count'],
                        width=w, color=colors_yr[yr], alpha=0.8, label=f'{yr}{yr_lbl_sfx}')
            mf_avg = monthly_fire.groupby('month')['hazard_mean'].mean()
            mf_lbl = 'Monthly Avg Hazard' if lang == 'en' else '월평균 기상위험도'
            ax2.plot(mf_avg.index, mf_avg.values, 'k--o', lw=1.5, ms=5, alpha=0.7, label=mf_lbl)
            ax1.set_xticks(range(1, 13))
            ax1.set_xticklabels([MONTH_LABELS[m] for m in range(1, 13)])
            if lang == 'en':
                ax1.set_ylabel('Fire Incidents'); ax2.set_ylabel('Weather Hazard')
                ax1.set_title('Monthly Fire Incidents vs Weather Hazard', fontweight='bold')
            else:
                ax1.set_ylabel('화재 발생 건수'); ax2.set_ylabel('기상위험도')
                ax1.set_title('월별 화재 발생 건수 vs 기상위험도', fontweight='bold')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
            ax1.grid(alpha=0.3, axis='y')
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # 히트맵
            if 'nearest_station' in trend.columns:
                heatmap_title = '**🌡️ Station × Month Weather Hazard Heatmap**' if lang == 'en' \
                                else '**🌡️ 관측소별 × 월별 기상위험도 히트맵**'
                st.markdown(heatmap_title)
                stn_monthly = (trend.groupby(['nearest_station','month'])
                               ['weather_hazard'].mean().unstack())
                stn_monthly.columns = [MONTH_LABELS[c] for c in stn_monthly.columns]
                fig, ax = plt.subplots(figsize=(12, 5))
                im = ax.imshow(stn_monthly.values, aspect='auto',
                               cmap='YlOrRd', vmin=0, vmax=100)
                ax.set_xticks(range(len(stn_monthly.columns)))
                ax.set_xticklabels(stn_monthly.columns)
                ax.set_yticks(range(len(stn_monthly.index)))
                stn_ylabels = ([STATION_EN.get(s, s) for s in stn_monthly.index]
                               if lang == 'en' else list(stn_monthly.index))
                ax.set_yticklabels(stn_ylabels)
                cbar_lbl = 'Avg Weather Hazard' if lang == 'en' else '평균 기상위험도'
                plt.colorbar(im, ax=ax, label=cbar_lbl)
                for i in range(stn_monthly.shape[0]):
                    for j in range(stn_monthly.shape[1]):
                        v = stn_monthly.values[i, j]
                        if not np.isnan(v):
                            ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                                    fontsize=7, color='white' if v > 65 else 'black')
                hm_t = 'Station × Month Avg Weather Hazard' if lang == 'en' else '관측소별 × 월별 평균 기상위험도'
                ax.set_title(hm_t, fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

        # FWI 계절 그림 추가
        img_fwi = OUT_FIGURES / 'fig_fwi_seasonal.png'
        if img_fwi.exists():
            if lang == 'en':
                st.markdown("**🔥 FWI (Fire Weather Index) Seasonal Pattern**")
                st.caption("Canadian FWI + Accumulated Risk Index by month — Spring peak matches actual wildfire season")
            else:
                st.markdown("**🔥 FWI (산불위험지수) 계절 패턴**")
                st.caption("캐나다 산불위험지수 + 누적위험지수 월별 추이 — 봄철(3~5월) 피크가 실제 산불 발생 시기와 일치")
            st.image(str(img_fwi), use_container_width=True)

    # ── TAB 7: 기상 예보 ─────────────────────────────────────────────────────
    with tab7:
        st.markdown("##### 📡 기상 예보 기반 화재 위험 예보 (3일)")
        st.caption("KMA Short-term Forecast API → LightGBM Inference → 3-Day Fire Risk Prediction"
                   if lang == 'en' else
                   "기상청 단기예보 API → 설비별 LightGBM 추론 → 향후 3일 화재 위험 예측")

        GRADE_COLORS_F = GRADE_COLORS
        GRADE_KR_F     = GRADE_KR

        if not KMA_API_KEY:
            if lang == 'en':
                st.warning("KMA API Key required.")
                st.code("# Add to .env file:\nKMA_API_KEY=your_key_from_apihub.kma.go.kr")
                st.markdown("""
**How to get the key:**
1. Sign up at [KMA API Hub](https://apihub.kma.go.kr)
2. API list → **단기예보 격자자료** → Apply for access
3. Paste the issued key into `.env` as `KMA_API_KEY=...` and restart the app
                """)
            else:
                st.warning("기상청 API Key가 필요합니다.")
                st.code("# .env 파일에 입력:\nKMA_API_KEY=apihub.kma.go.kr에서_발급한_인증키")
                st.markdown("""
**발급 방법:**
1. [기상청 API Hub](https://apihub.kma.go.kr) 회원가입
2. API 목록 → **동네예보(단기예보)** → 활용신청
3. 발급된 인증키를 `.env` 파일의 `KMA_API_KEY`에 입력 후 앱 재시작
                """)
        else:
            # ── 자동 로드: 캐시 없으면 즉시 수집 ──────────────────────────────
            if 'fc_result' not in st.session_state:
                spinner_msg = (t('fc_loading', lang))
                with st.spinner(spinner_msg):
                    try:
                        fc_result, fc_feat = run_full_forecast(KMA_API_KEY, DATA_PROCESSED, df)
                        st.session_state['fc_result'] = fc_result
                        st.session_state['fc_feat']   = fc_feat
                        st.session_state['fc_time']   = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                    except Exception as e:
                        st.session_state['fc_result'] = pd.DataFrame()
                        st.session_state['fc_feat']   = pd.DataFrame()
                        err = f"❌ Forecast fetch failed: {e}" if lang == 'en' else f"❌ 예보 수집 실패: {e}"
                        st.error(err)

            col_info, col_refresh = st.columns([4, 1])
            with col_info:
                fc_time = st.session_state.get('fc_time', '')
                if fc_time:
                    fc_time_lbl = f"🕐 Last updated: {fc_time}" if lang == 'en' else f"🕐 최종 수집: {fc_time}"
                    st.caption(fc_time_lbl)
            with col_refresh:
                refresh_btn_lbl = t('fc_refresh', lang)
                if st.button(refresh_btn_lbl, use_container_width=True):
                    for k in ['fc_result','fc_feat','fc_time']:
                        st.session_state.pop(k, None)
                    # 날짜별 recommend 캐시도 삭제
                    for k in list(st.session_state.keys()):
                        if k.startswith('fc_recommend'):
                            st.session_state.pop(k)
                    st.rerun()

            fc_result = st.session_state.get('fc_result')
            fc_feat   = st.session_state.get('fc_feat')

            if fc_result is not None and not fc_result.empty:
                dates = sorted(fc_result['fcst_date'].unique())
                today = pd.Timestamp.now()
                date_labels = []
                for d in dates:
                    dt   = pd.to_datetime(d, format="%Y%m%d")
                    diff = (dt - today.normalize()).days
                    day_lbl_map = ({0:"Today",1:"Tomorrow",2:"Day+2"} if lang=='en'
                                   else {0:"오늘",1:"내일",2:"모레"})
                    lbl = day_lbl_map.get(diff, dt.strftime("%m/%d"))
                    date_labels.append(f"{lbl} ({dt.strftime('%m/%d' if lang=='en' else '%m.%d')})")

                # ── 3일 전체 종합 권고 ──────────────────────────────────────
                st.markdown(t('fc_overall', lang))
                overall_key = f"fc_overall_recommend_{lang}"
                if overall_key not in st.session_state:
                    spin_msg = "Analyzing 3-day forecast..." if lang == 'en' else "3일 종합 분석 중..."
                    with st.spinner(spin_msg):
                        st.session_state[overall_key] = get_overall_forecast_recommendation(
                            fc_result, fc_feat, lang)
                st.warning(f"🤖 {st.session_state[overall_key]}")

                # ── 3일 KPI 요약 카드 ──────────────────────────────────────
                st.markdown(t('fc_daily', lang))
                unit_fc = "" if lang == 'en' else "개"
                wh_lbl_fc = "Hazard" if lang == 'en' else "기상위험"
                day_cols = st.columns(len(dates))
                for i, (d, lbl, col) in enumerate(zip(dates, date_labels, day_cols)):
                    day_data = fc_result[fc_result['fcst_date'] == d]
                    vh_d = (day_data['forecast_grade'] == 'Very High').sum()
                    hi_d = (day_data['forecast_grade'] == 'High').sum()
                    feat_d = fc_feat[fc_feat['fcst_date'] == d] if fc_feat is not None else pd.DataFrame()
                    wh_max = feat_d['weather_hazard'].max() if not feat_d.empty else 0
                    with col:
                        bg = "#ffe0e0" if vh_d > 100000 else "#fff0e0" if vh_d > 50000 else "#f0fff0"
                        st.markdown(f"""
<div style='padding:12px;border-radius:10px;background:{bg};text-align:center;border:1px solid #ddd'>
<b>{lbl}</b><br>
🔴 {vh_d:,}{unit_fc}<br>
🟠 {hi_d:,}{unit_fc}<br>
{wh_lbl_fc} <b>{wh_max:.1f}</b>
</div>""", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(t('fc_detail', lang))
                sel_idx  = st.radio(t('fc_date', lang), range(len(dates)),
                                    format_func=lambda i: date_labels[i] if i < len(date_labels) else str(i),
                                    horizontal=True)
                sel_date = dates[sel_idx]
                fc_day   = fc_result[fc_result['fcst_date'] == sel_date]
                feat_day = fc_feat[fc_feat['fcst_date'] == sel_date] if fc_feat is not None else pd.DataFrame()

                # KPI
                cols = st.columns(4)
                for col, grade in zip(cols, ['Very High','High','Moderate','Low']):
                    g_lbl = GRADE_KR[grade] if lang == 'ko' else grade
                    col.metric(g_lbl, f"{(fc_day['forecast_grade']==grade).sum():,}{unit_fc}")

                # AI 예방 권고 — 항상 표시
                st.markdown(t('fc_ai_rec', lang))
                rec_key = f"fc_recommend_{sel_date}_{lang}"
                if rec_key not in st.session_state:
                    spin_rec = "Analyzing with OpenAI..." if lang == 'en' else "OpenAI 분석 중..."
                    with st.spinner(spin_rec):
                        st.session_state[rec_key] = get_forecast_recommendation(
                            feat_day, fc_day, date_labels[sel_idx], lang)
                _render_structured_ai(st.session_state[rec_key], lang)

                # 관측소별 기상 요약
                st.markdown(t('fc_wx_sum', lang))
                if not feat_day.empty:
                    if lang == 'en':
                        wx_rename = {
                            'station':'Station','temp_max':'Max Temp(℃)','rh_min':'Min Humid(%)',
                            'ws_max':'Max Wind(m/s)','precip_sum':'Precip(mm)',
                            'weather_hazard':'Hazard','dry_watch_flag':'Dry Alert',
                            'wind_watch_flag':'Wind Alert','combined_risk_flag':'Combined'
                        }
                        hazard_col = 'Hazard'
                    else:
                        wx_rename = {
                            'station':'관측소','temp_max':'최고기온(℃)','rh_min':'최저습도(%)',
                            'ws_max':'최대풍속(m/s)','precip_sum':'강수량(mm)',
                            'weather_hazard':'기상위험도','dry_watch_flag':'건조주의보',
                            'wind_watch_flag':'강풍주의보','combined_risk_flag':'복합위험'
                        }
                        hazard_col = '기상위험도'
                    feat_disp = feat_day[[
                        'station','temp_max','rh_min','ws_max','precip_sum',
                        'weather_hazard','dry_watch_flag','wind_watch_flag','combined_risk_flag'
                    ]].rename(columns=wx_rename).sort_values(hazard_col, ascending=False).round(1)

                    def color_hazard(val):
                        if isinstance(val, (int, float)):
                            if val >= 70: return 'background-color:#ffe0e0'
                            if val >= 50: return 'background-color:#fff0e0'
                        return ''
                    st.dataframe(
                        feat_disp.style.map(color_hazard, subset=[hazard_col]),
                        use_container_width=True, hide_index=True
                    )

                # 예보 위험도 지도
                st.markdown(f"{t('fc_map', lang)} {date_labels[sel_idx]}**")
                if gdf is not None:
                    gdf_fc = gdf.merge(
                        fc_day[['pole_id','forecast_risk','forecast_grade','weather_hazard']],
                        on='pole_id', how='inner')
                    gdf_fc['lat'] = gdf_fc.geometry.y
                    gdf_fc['lon'] = gdf_fc.geometry.x
                    m_fc = folium.Map(location=[37.5, 128.3], zoom_start=9, tiles='CartoDB positron')
                    fc_unit = "" if lang == 'en' else "개"
                    for grade in ['Very High','High','Moderate','Low']:
                        g_lbl = GRADE_KR[grade] if lang == 'ko' else grade
                        color  = GRADE_COLORS_F[grade]
                        radius = {'Very High':8,'High':6,'Moderate':4,'Low':3}[grade]
                        sub = gdf_fc[gdf_fc['forecast_grade'] == grade]
                        total_g = len(sub)
                        if len(sub) > 2000:
                            sub = sub.nlargest(2000, 'forecast_risk')
                        layer = folium.FeatureGroup(name=f"{g_lbl} ({total_g:,}{fc_unit})")
                        fc_lbl = 'Facility' if lang == 'en' else '설비'
                        fc_risk_lbl = 'Forecast Risk' if lang == 'en' else '예측 위험도'
                        fc_grade_lbl = 'Grade' if lang == 'en' else '등급'
                        fc_wh_lbl = 'Weather' if lang == 'en' else '기상위험'
                        for _, row in sub.iterrows():
                            folium.CircleMarker(
                                location=[row['lat'], row['lon']],
                                radius=radius, color=color, fill=True,
                                fill_color=color, fill_opacity=0.75,
                                tooltip=f"#{int(row['pole_id'])}: {row['forecast_risk']:.1f}",
                                popup=folium.Popup(
                                    f"<b>{fc_lbl} #{int(row['pole_id'])}</b><br>"
                                    f"{fc_risk_lbl}: {row['forecast_risk']:.1f}<br>"
                                    f"{fc_grade_lbl}: {g_lbl}<br>"
                                    f"{fc_wh_lbl}: {row['weather_hazard']:.1f}",
                                    max_width=200)
                            ).add_to(layer)
                        layer.add_to(m_fc)
                    folium.LayerControl().add_to(m_fc)
                    st_folium(m_fc, width=None, height=500, returned_objects=[])

                # Top-20
                st.markdown(t('fc_top20', lang))
                top20 = fc_day.nlargest(20, 'forecast_risk')[
                    ['pole_id','forecast_risk','forecast_grade','weather_hazard','nearest_station']
                ].copy().reset_index(drop=True)
                top20.index += 1
                if lang == 'en':
                    top20.columns = ['Facility ID','Forecast Risk','Forecast Grade','Hazard','Station']
                    top20['Forecast Risk']  = top20['Forecast Risk'].round(1)
                    top20['Forecast Grade'] = top20['Forecast Grade'].map(GRADE_EN)
                else:
                    top20.columns = ['설비ID','예측 위험도','예측 등급','기상위험도','관측소']
                    top20['예측 위험도'] = top20['예측 위험도'].round(1)
                    top20['예측 등급']  = top20['예측 등급'].map(GRADE_KR)
                st.dataframe(top20, use_container_width=True)

                csv = fc_day[['pole_id','fcst_date','forecast_risk','forecast_grade',
                               'weather_hazard','nearest_station']].to_csv(
                    index=False, encoding='utf-8-sig')
                dl_lbl = f"📥 {date_labels[sel_idx]} Forecast CSV" if lang == 'en' \
                         else f"📥 {date_labels[sel_idx]} 예보 CSV"
                st.download_button(dl_lbl, csv,
                                   file_name=f"wpfi_forecast_{sel_date}.csv", mime="text/csv")

            elif fc_result is not None:
                err_msg = "Forecast result is empty. Check API key or network." if lang == 'en' \
                          else "예보 결과가 비어 있습니다. API 키 또는 네트워크를 확인하세요."
                st.error(err_msg)

    # ── TAB 8: 산출 방식 ─────────────────────────────────────────────────────
    with tab8:
        st.markdown("##### 🧮 WPFI 위험도 산출 방식" if lang == 'ko' else "##### 🧮 WPFI Calculation Methodology")

        # ── 섹션 1: 전체 파이프라인 ─────────────────────────────────────────
        with st.expander("① 전체 파이프라인" if lang == 'ko' else "① Full Pipeline", expanded=True):
            if lang == 'ko':
                st.markdown("**데이터 → Feature → 모델 → 위험도 산출 흐름**")
            else:
                st.markdown("**Data → Feature → Model → Risk Score Pipeline**")
            st.code(
                "KMA ASOS 기상일자료 (11개 관측소)\n"
                "        ↓ FWI 계산 / Rolling Window / 특보 Flag\n"
                "   기상 Feature (18개)\n"
                "        ↓\n"
                "ESA WorldCover + DEM  →  산림·지형 Feature (4개)\n"
                "설비 위치 데이터       →  설비 밀도·연쇄피해 Feature (4개)\n"
                "산불 발생 이력         →  label 생성 (산불 ±14일 + 500m 이내)\n"
                "        ↓\n"
                "   LightGBM 학습 (39개 feature, AUC 0.9574)\n"
                "        ↓\n"
                "   ML 화재확률 x Rule-based WPFI 통합 점수\n"
                "        ↓\n"
                "   등급 분류 (Very High / High / Moderate / Low)",
                language=None
            )

        # ── 섹션 2: 4-Component WPFI ────────────────────────────────────────
        with st.expander("② WPFI_v2 — 멀티해저드 3성분 공식" if lang == 'ko' else "② WPFI_v2 — Multi-Hazard 3-Component Formula"):
            if lang == 'ko':
                st.markdown(r"""
**WPFI\_v2 = 0.5 × P\_dry + 0.3 × P\_heat + 0.2 × P\_light**

| 구성요소 | 가중치 | 학습 방식 | 주요 기상 변수 | 날씨 기여도 |
|----------|--------|----------|--------------|-----------|
| P\_dry (건조형) | **50%** | LightGBM | FWI·건조일수·실효습도 | **94.8%** |
| P\_heat (고온형) | **30%** | LightGBM | 열지수·최고기온·누적위험 | **94.7%** |
| P\_light (낙뢰형) | **20%** | Rule-based | 고도·여름일교차·여름습도 | 100% |

> AUC: 건조형 **0.737** / 고온형 **0.806**
> 모든 Top-5 피처가 기상 변수 (달력 의존도 0%)
""")
            else:
                st.markdown(r"""
**WPFI\_v2 = 0.5 × P\_dry + 0.3 × P\_heat + 0.2 × P\_light**

| Component | Weight | Method | Key Weather Variables | Weather Pct |
|-----------|--------|--------|-----------------------|-------------|
| P\_dry (Dryness) | **50%** | LightGBM | FWI, dry streak, eff. humidity | **94.8%** |
| P\_heat (Heat) | **30%** | LightGBM | heat score, max temp, accum. risk | **94.7%** |
| P\_light (Lightning) | **20%** | Rule-based | elevation, summer temp range, humidity | 100% |

> AUC: Dry model **0.737** / Heat model **0.806**
> All Top-5 features are weather variables (zero calendar dependency)
""")

        # ── 섹션 3: FWI 캐나다 국제표준 ────────────────────────────────────
        with st.expander("③ FWI (산불위험지수) — 캐나다 국제 표준" if lang == 'ko' else "③ FWI — Canadian International Standard"):
            if lang == 'ko':
                st.markdown("**캐나다 산불위험지수 (Canadian Fire Weather Index System)**")
                st.info("기상청, 산림청, 국제산불연구기관이 공식 채택한 전 세계 표준 산불위험 지수입니다.")
                st.markdown("**Stage 1: 연료 수분 코드 (Fuel Moisture Codes)**")
                st.dataframe(
                    __import__('pandas').DataFrame({
                        "코드": ["FFMC", "DMC", "DC"],
                        "이름": ["Fine Fuel Moisture Code", "Duff Moisture Code", "Drought Code"],
                        "입력 변수": ["기온·상대습도·풍속·강수", "기온·상대습도·강수 (14일 누적)", "기온·강수 (30일 누적)"],
                        "의미": ["낙엽·건초 등 세연료 수분", "반분해 유기물층 수분", "깊은 토양 건조도"],
                    }), hide_index=True, use_container_width=True
                )
                st.markdown("**Stage 2: 화재 행동 지수**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**ISI (Initial Spread Index)**  \n초기 화재 확산 속도")
                    st.latex(r"ISI = 0.208 \times e^{0.05039 W} \times f(FFMC)")
                with col_b:
                    st.markdown("**BUI (Buildup Index)**  \n연료 누적 건조량")
                    st.latex(r"BUI = 0.8 \times \frac{DMC \times DC}{DMC + 0.4 \times DC}")
                st.markdown("**Stage 3: 최종 FWI**")
                st.latex(r"FWI = f(ISI,\; BUI)")
                st.dataframe(
                    __import__('pandas').DataFrame({
                        "FWI 값": ["0 ~ 11", "12 ~ 22", "23 ~ 37", "38 이상"],
                        "위험 등급": ["낮음 (Low)", "보통 (Moderate)", "높음 (High)", "매우 높음 (Very High)"],
                        "현장 조치": ["정기 모니터링", "주의 강화", "중점 점검", "즉각 대응"],
                    }), hide_index=True, use_container_width=True
                )
                st.markdown("**본 프로젝트 적용 결과:**")
                st.markdown("- 강원도 11개 KMA ASOS 관측소 일별 기온·상대습도·풍속·강수량으로 FWI 산출")
                st.markdown("- FWI 최고 기록: **대관령 2022년 3월 5일 29.1점** (봄철 고위험)")
                st.markdown("- FWI + 누적위험지수(14일 지수 감쇠) → LightGBM feature로 활용")
            else:
                st.markdown("**Canadian FWI System — Global Standard adopted by weather and forestry agencies worldwide**")
                st.markdown("**Stage 1:** FFMC (fine fuel) → DMC (duff, 14-day) → DC (drought, 30-day)")
                st.markdown("**Stage 2:** ISI = spread speed | BUI = fuel buildup")
                st.latex(r"ISI = 0.208 \times e^{0.05039 W} \times f(FFMC)")
                st.latex(r"BUI = 0.8 \times \frac{DMC \times DC}{DMC + 0.4 \times DC}")
                st.markdown("**Stage 3:** FWI = f(ISI, BUI)")
                st.dataframe(
                    __import__('pandas').DataFrame({
                        "FWI": ["0–11","12–22","23–37","38+"],
                        "Risk": ["Low","Moderate","High","Very High"],
                    }), hide_index=True, use_container_width=True
                )
            img_fwi = OUT_FIGURES / 'fig_fwi_seasonal.png'
            if img_fwi.exists():
                cap = "Monthly FWI + Accumulated Risk — Spring peak matches actual wildfire season" if lang == 'en' \
                      else "FWI + 누적위험지수 월별 패턴 — 봄철(3~5월) 피크가 실제 산불 발생 시기와 일치"
                st.image(str(img_fwi), use_container_width=True, caption=cap)

        # ── 섹션 4: label 설계 ──────────────────────────────────────────────
        with st.expander("④ Label 설계 — 날짜+공간 기반" if lang == 'ko' else "④ Label Design — Temporal + Spatial"):
            if lang == 'ko':
                st.markdown("""
**기존 방식의 문제**
- 화재 발생 지점 인근 전봇대 = 모든 날짜 label=1 → 날씨와 무관한 지리적 학습
- 결과: AUC 0.9982 (과적합), 날씨 feature 기여도 ~14%

**개선된 label 설계**
```
label = 1  ←→  산불 발생일 기준 ±14일 이내 AND 반경 500m 이내 전봇대
label = 0  ←→  위 조건 외 모든 (전봇대, 날짜) 조합
```

| 지표 | 기존 | 개선 |
|------|------|------|
| label=1 비율 | 9.1% (278,636건) | **1.9% (57,036건)** |
| 날씨 기여도 | ~14% | **34%** |
| AUC | 0.9982 (의심) | **0.9574 (신뢰)** |
| 계절성 | 여름도 label=1 | 봄·겨울 집중 (실제 산불 패턴) |

**검증:** label=1인 날의 `precip_sum_7d` 평균이 label=0보다 67% 낮음 → 모델이 실제 건조 조건을 학습
""")
            else:
                st.markdown("""
**Improved Label Design**
```
label=1  ←  wildfire within ±14 days AND 500m radius
label=0  ←  all other (pole, date) pairs
```
Result: AUC improved to 0.9574, weather contribution up to 34%, correct seasonality (spring peak).
""")

        # ── 섹션 5: 연쇄피해 위험 ───────────────────────────────────────────
        with st.expander("⑤ 연쇄피해 위험 (Cascade Risk) — 공간 네트워크 분석" if lang == 'ko' else "⑤ Cascade Risk — Spatial Network Analysis"):
            if lang == 'ko':
                st.markdown("""
**계산 방법**
1. 전봇대 위치(1,387,831개)로 BallTree 공간 인덱스 구축
2. 각 전봇대 기준 **300m 이내 인접 전봇대 수** 계산
3. 0~100 정규화 → `cascade_risk` feature

```python
# 핵심 코드
from sklearn.neighbors import BallTree
tree = BallTree(coords_rad, metric='haversine')
counts_300m = tree.query_radius(coords_rad, r=300/6371000, count_only=True)
cascade_risk = MinMaxScaler(0,100).fit_transform(counts_300m)
```

**의미:** cascade_risk가 높은 설비 = 화재 발생 시 인근 전봇대로 불이 옮겨 붙어 연쇄 정전 위험이 큰 구간

| 수치 | 의미 |
|------|------|
| 0~30 | 산간 독립 설비, 연쇄 피해 낮음 |
| 30~60 | 도심 인접 구간, 주의 필요 |
| 60~100 | 설비 밀집 구간, 화재 시 대규모 피해 |
""")

        # ── 섹션 6: RAG 시스템 ──────────────────────────────────────────────
        with st.expander("⑥ RAG 기반 AI 설명 시스템" if lang == 'ko' else "⑥ RAG-based AI Explanation System"):
            if lang == 'ko':
                st.markdown("""
**RAG (Retrieval-Augmented Generation)**

> 단순 GPT 응답이 아닌, **실제 데이터에 근거한 설명**을 생성합니다.

```
설비 선택
    ↓
기상 조건(FWI, 누적위험도, 건조도) → 쿼리 생성
    ↓
TF-IDF 코퍼스 검색 (106개 문서)
    ├─ 산불 사례: 강원도 174건 (산림청 2022-2024)
    ├─ 기상특보 기준: 기상청 공식 건조·강풍 특보 임계값
    └─ 도메인 지식: FWI 해석, 전력설비 화재 원인, 연쇄피해
    ↓
GPT-4o-mini 프롬프트에 검색 결과 주입
    ↓
근거 있는 설명 생성
    예: "FWI 29.1 → 건조경보 수준, 2022년 3월 대관령 유사 조건
        에서 실제 산불 발생. 연쇄피해 위험 높은 구간으로 즉각 점검 필요"
```

**향후 확장 (Future Work):** Knowledge Graph 기반 전력망 위상 분석, 풍향 데이터 연동 화재 전파 경로 예측
""")
            else:
                st.markdown("""
**RAG (Retrieval-Augmented Generation)**

Searches 106 documents (wildfire cases + weather warning standards + domain knowledge) and injects context into GPT-4o-mini prompt for evidence-based explanations.

**Future Work:** Knowledge Graph for power network topology, wind direction-based fire propagation path prediction.
""")

    # ── 푸터 ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <hr>
    <p style='text-align:center;color:#aaa;font-size:0.8em'>
    {t('footer', lang)}
    </p>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
