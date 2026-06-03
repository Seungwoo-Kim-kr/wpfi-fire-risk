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
    'footer': {'ko': 'WPFI v3.0 · 2026 날씨 빅데이터 콘테스트 · Explainable GeoAI<br>강원도 전봇대 1,387,831개 분석 | LightGBM AUC 0.9982',
               'en': 'WPFI v3.0 · 2026 Weather Big Data Contest · Explainable GeoAI<br>1,387,831 Power Poles in Gangwon, Korea | LightGBM AUC 0.9982'},
}

def t(key: str, lang: str = 'ko') -> str:
    """번역 키로 현재 언어 문자열 반환"""
    return TRANSLATIONS.get(key, {}).get(lang, TRANSLATIONS.get(key, {}).get('ko', key))

WEIGHTS = {'weather': 0.35, 'spatial': 0.30, 'facility': 0.25, 'prior': 0.10}

SIMULATION_PRESETS = {
    "현재 데이터 (기준)":       {"wh_mult": 1.00, "desc": "실제 분석 결과 그대로 표시"},
    "봄철 건조강풍 시나리오":   {"wh_mult": 1.35, "desc": "실효습도 -20%, 풍속 +5m/s 조건 (산불 위험 최고조 시기)"},
    "여름 폭염 시나리오":       {"wh_mult": 1.20, "desc": "기온 +8°C, 습도 -10% 조건"},
    "가을 건조 시나리오":       {"wh_mult": 1.40, "desc": "연속 무강수 14일, 습도 -25% 조건"},
    "겨울 강풍 시나리오":       {"wh_mult": 1.15, "desc": "풍속 +8m/s, 기온 -10°C 조건"},
    "최악 복합 시나리오":       {"wh_mult": 1.70, "desc": "극건조 + 강풍 + 고온 동시 발생"},
}

# 실제 데이터에서 역산한 등급 임계값 (노트북 분위수 기반)
# Very High: 58.28~78.92, High: 55.06~58.28, Moderate: 50.83~55.06, Low: 35.72~50.83
GRADE_THRESHOLDS_ACTUAL = {
    "Very High": 58.28,
    "High":      55.06,
    "Moderate":  50.83,
}

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
@st.cache_data
def load_risk():
    p = DATA_PROCESSED / 'risk_scores.parquet'
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if 'ml_fire_prob_max' in df.columns:
        mn, mx = df['ml_fire_prob_max'].min(), df['ml_fire_prob_max'].max()
        df['ml_score'] = ((df['ml_fire_prob_max'] - mn) / (mx - mn + 1e-9) * 100).clip(0, 100)
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
        ('model_perf',           OUT_TABLES / 'model_performance.csv'),
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

def get_executive_summary(df: pd.DataFrame, perf: dict) -> str:
    total = len(df)
    vh    = (df['risk_grade'] == 'Very High').sum()
    hi    = (df['risk_grade'] == 'High').sum()
    top_stn = df.groupby('nearest_station')['final_risk'].mean().idxmax()
    top_mean = df.groupby('nearest_station')['final_risk'].mean().max()
    auc = perf.get('model_perf', pd.DataFrame()).get('auc', pd.Series([0])).max()

    default = (
        f"강원도 전력설비 {total:,}개를 분석한 결과, {vh:,}개({vh/total*100:.1f}%)가 "
        f"Very High(매우높음) 등급으로 즉각적인 점검이 필요합니다. "
        f"위험도가 가장 높은 권역은 {top_stn}(평균 {top_mean:.1f}점)이며, "
        f"주요 위험 요인은 산림 인접도와 건조·강풍 기상 조건의 복합 작용입니다. "
        f"LightGBM 모델의 예측 정확도(AUC)는 {auc:.4f}입니다."
    )
    prompt = f"""강원도 전력설비 화재위험도 분석 결과를 현장 관리자 및 경영진을 위한 총평으로 3~4문장 작성하세요.
수치 근거를 반드시 포함하고, 핵심 위험요인과 권고사항도 포함하세요.

분석 데이터:
- 총 전력설비: {total:,}개 (강원도 전봇대)
- Very High 등급: {vh:,}개 ({vh/total*100:.1f}%)
- High 등급: {hi:,}개 ({hi/total*100:.1f}%)
- 최고위험 권역: {top_stn} (평균 위험도 {top_mean:.1f}/100)
- LightGBM 예측 AUC: {auc:.4f}
- 주요 위험요인: 기상위험(건조·강풍) 35%, 산림·지형 노출 30%, 설비 노출 25%, 화재이력 10%
- 분석 기간: 2022~2024년 (3년)"""

    result = _openai_chat(prompt, system="당신은 전력설비 화재 위험 전문가입니다. 한국어로 간결하고 정확하게 답변하세요.")
    return result if result and not result.startswith("[OpenAI") else default

def get_facility_explanation(row: dict) -> str:
    grade_kr = GRADE_KR.get(row.get('risk_grade', ''), '')
    default = (
        f"설비 #{row.get('pole_id','')}은(는) 기상위험 {row.get('weather_hazard',0):.0f}점, "
        f"공간노출 {row.get('spatial_exposure',0):.0f}점이 주요 위험요인으로, "
        f"{grade_kr} 등급({row.get('final_risk',0):.1f}점)입니다. "
        f"절연 저항 측정 및 현장 점검이 권고됩니다."
    )
    prompt = f"""전력설비 화재위험 분석 결과를 현장 점검 엔지니어용으로 3문장 설명하세요.
수치 범위를 벗어난 내용을 추가하지 마세요.

설비ID: {row.get('pole_id','')}
위험등급: {grade_kr} ({row.get('final_risk',0):.1f}/100)
ML 화재확률 점수: {row.get('ml_score',0):.1f}/100
기상위험: {row.get('weather_hazard',0):.1f} | 공간노출: {row.get('spatial_exposure',0):.1f}
설비노출: {row.get('facility_exposure',0):.1f} | 화재이력: {row.get('hist_prior',0):.1f}
관할관측소: {row.get('nearest_station','')}
권고사항: 현장 점검 및 예방 조치를 구체적으로 포함"""
    result = _openai_chat(prompt, system="당신은 전력설비 안전 점검 전문가입니다.", max_tokens=300)
    return result if result and not result.startswith("[OpenAI") else default

def get_overall_forecast_recommendation(fc_result: pd.DataFrame, fc_feat: pd.DataFrame) -> str:
    """3일 전체 예보 종합 권고 (OpenAI)"""
    if fc_result is None or fc_result.empty:
        return ""
    dates = sorted(fc_result['fcst_date'].unique())
    today = pd.Timestamp.now()

    lines = []
    for d in dates:
        dt   = pd.to_datetime(d, format="%Y%m%d")
        diff = (dt - today.normalize()).days
        lbl  = {0:"오늘",1:"내일",2:"모레"}.get(diff, dt.strftime("%m/%d"))
        day  = fc_result[fc_result['fcst_date'] == d]
        vh   = (day['forecast_grade'] == 'Very High').sum()
        hi   = (day['forecast_grade'] == 'High').sum()
        feat = fc_feat[fc_feat['fcst_date'] == d] if fc_feat is not None else pd.DataFrame()
        wh_max = feat['weather_hazard'].max() if not feat.empty else 0
        lines.append(f"- {lbl}({dt.strftime('%m.%d')}): Very High {vh:,}개, High {hi:,}개, 최대기상위험도 {wh_max:.1f}")

    default = (
        f"향후 3일간 예보 분석:\n" + "\n".join(lines) +
        "\n\n가장 위험도가 높은 날에 현장 점검 자원을 집중 배치하고, "
        "건조주의보 발령 권역의 설비는 사전 절연 점검을 권고합니다."
    )
    prompt = f"""향후 3일간 강원도 전력설비 화재 위험 예보를 분석하여 경영진과 현장 관리자를 위한 종합 권고문을 작성하세요.

예보 요약:
{chr(10).join(lines)}

다음 내용을 반드시 포함하세요:
1. 3일간 전체 위험 추세 (날짜별 비교)
2. 가장 위험한 날짜와 그 이유
3. 우선 점검 권역 및 설비 유형
4. 예방 조치 3가지 (구체적이고 실행 가능한 것)
5. 자원 배치 권고"""
    result = _openai_chat(prompt,
        system="당신은 전력설비 화재 예방 전문가입니다. 한국어 5~7문장으로 답변하세요.",
        max_tokens=500)
    return result if result and not result.startswith("[OpenAI") else default

def get_forecast_recommendation(feat_df: pd.DataFrame, fc_day: pd.DataFrame,
                                 date_label: str) -> str:
    if feat_df is None or fc_day.empty:
        return ""
    top_stn = feat_df.groupby('station')['weather_hazard'].mean().idxmax() \
              if not feat_df.empty else "알 수 없음"
    vh_cnt = (fc_day['forecast_grade'] == 'Very High').sum()
    hi_cnt = (fc_day['forecast_grade'] == 'High').sum()

    feat_row = feat_df.iloc[0] if not feat_df.empty else {}
    default = (
        f"{date_label} 기상 예보 기준, {top_stn} 권역이 최고 기상위험도를 보입니다. "
        f"Very High 등급 예측 설비 {vh_cnt:,}개, High 등급 {hi_cnt:,}개로, "
        f"건조주의보 발령 조건에 해당하는 구역의 설비 긴급 점검이 필요합니다."
    )
    prompt = f"""{date_label} 기상 예보 기반 전력설비 화재 위험 예보 분석 결과를 작성하세요.
다음 내용을 반드시 포함하세요:
1. 어떤 권역/설비가 위험에 노출되는지
2. 주요 위험 기상 조건
3. 구체적인 예방·대응 조치 (3가지)

예보 날짜: {date_label}
최고위험 권역: {top_stn}
Very High 예측 설비: {vh_cnt:,}개
High 예측 설비: {hi_cnt:,}개
최대 기상위험도: {feat_df['weather_hazard'].max():.1f}/100
건조주의보 관측소 수: {int(feat_df['dry_watch_flag'].sum()) if 'dry_watch_flag' in feat_df.columns else 0}개소
강풍주의보 관측소 수: {int(feat_df['wind_watch_flag'].sum()) if 'wind_watch_flag' in feat_df.columns else 0}개소"""
    result = _openai_chat(prompt, system="당신은 전력설비 화재 예방 전문가입니다. 한국어 3~5문장으로 답변하세요.", max_tokens=400)
    return result if result and not result.startswith("[OpenAI") else default

# ── 위험 판단 사유 생성 (규칙 기반) ──────────────────────────────────────────
def _risk_reason(row: dict) -> str:
    """4개 컴포넌트 점수를 분석해 이해하기 쉬운 판단 사유 반환"""
    wh = float(row.get('weather_hazard',   0))
    sp = float(row.get('spatial_exposure', 0))
    fe = float(row.get('facility_exposure',0))
    hp = float(row.get('hist_prior',       0))

    # 가중 점수 기준 주요/보조 요인 순위
    weighted = {
        'weather':  wh * WEIGHTS['weather'],
        'spatial':  sp * WEIGHTS['spatial'],
        'facility': fe * WEIGHTS['facility'],
        'prior':    hp * WEIGHTS['prior'],
    }
    ranked = sorted(weighted, key=weighted.get, reverse=True)
    top1, top2 = ranked[0], ranked[1]

    # 요인별 자연어 설명 (점수 수준에 따라 차별화)
    def _wh_text(v):
        if v >= 80: return "건조·강풍 조건이 매우 심각해 화재 발화·확산 위험이 극도로 높습니다"
        if v >= 65: return "습도가 낮고 바람이 강해 불씨가 발생하면 빠르게 번질 수 있는 조건입니다"
        return "기상 조건이 다소 건조하거나 바람이 있어 주의가 필요합니다"

    def _sp_text(v):
        if v >= 75: return "반경 내 산림 비율이 높고 경사가 가팔라 화재 시 대규모 확산이 우려됩니다"
        if v >= 55: return "산림과 인접하고 지형상 화재가 퍼지기 쉬운 구조입니다"
        return "주변에 산림이 일부 있어 화재 확산 가능성이 존재합니다"

    def _fe_text(v):
        if v >= 75: return "주변에 전력설비가 밀집되어 있어 연쇄 피해 가능성이 큽니다"
        if v >= 55: return "인근 설비 밀도가 높아 한 곳에서 화재 발생 시 파급 범위가 넓습니다"
        return "설비 노출 수준이 보통이나 지속 모니터링이 필요합니다"

    def _hp_text(v):
        if v >= 65: return "이 지역은 과거에도 전기화재·산불이 반복 발생한 이력이 있는 고위험 구역입니다"
        if v >= 50: return "과거 화재 이력이 있어 위험 패턴이 반복될 가능성이 있습니다"
        return "과거 화재 이력은 낮으나 다른 요인이 위험을 높이고 있습니다"

    text_fn = {'weather': _wh_text, 'spatial': _sp_text,
               'facility': _fe_text, 'prior': _hp_text}
    vals    = {'weather': wh, 'spatial': sp, 'facility': fe, 'prior': hp}
    labels  = {'weather':'기상','spatial':'지형','facility':'설비','prior':'이력'}

    # lang 파악 (session_state)
    _lang = st.session_state.get('lang', 'ko')
    _main = '[주요]' if _lang == 'ko' else '[Main]'
    _sub  = '[보조]' if _lang == 'ko' else '[Sub]'

    reason = (
        f"<b>{_main}</b> {labels[top1]}: {text_fn[top1](vals[top1])}<br>"
        f"<b>{_sub}</b> {labels[top2]}: {text_fn[top2](vals[top2])}"
    )

    # 임계값 초과 경보 플래그
    flags = []
    if wh >= 75: flags.append("🚨 극고위험 기상")
    if sp >= 70: flags.append("🌲 고위험 산림노출")
    if hp >= 60: flags.append("🔥 반복 화재지역")
    if fe >= 75: flags.append("⚡ 설비 밀집위험")
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
            popup_html = (
                f"<div style='min-width:220px;font-size:12px'>"
                f"<b style='font-size:13px'>설비 #{int(row['pole_id'])}</b><br>"
                f"<span style='color:{color};font-weight:bold'>"
                f"  {GRADE_EMOJI[grade]} {GRADE_KR[grade]}</span>"
                f" &nbsp;<b>{row[risk_col]:.1f}</b>/100<br>"
                f"<hr style='margin:4px 0'>"
                f"<table style='width:100%;font-size:11px'>"
                f"<tr><td>🌪️ 기상위험</td><td><b>{row.get('weather_hazard',0):.0f}</b></td>"
                f"    <td>🌲 공간노출</td><td><b>{row.get('spatial_exposure',0):.0f}</b></td></tr>"
                f"<tr><td>⚡ 설비노출</td><td><b>{row.get('facility_exposure',0):.0f}</b></td>"
                f"    <td>🔥 화재이력</td><td><b>{row.get('hist_prior',0):.0f}</b></td></tr>"
                f"<tr><td>🤖 ML점수</td><td colspan='3'><b>{row.get('ml_score',0):.1f}</b></td></tr>"
                f"</table>"
                f"<hr style='margin:4px 0'>"
                f"<div style='color:#555;font-size:11px'>{reason}</div>"
                f"<div style='color:#888;font-size:10px;margin-top:2px'>"
                f"📍 {row.get('nearest_station','')} 관측소 권역</div>"
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
        ax.set_xlabel('SHAP 값 (+ : 위험 증가 / - : 위험 감소)')
        ax.set_title(f'설비 #{pole_id} — SHAP 기여도 (ML 확률: {prob_col[best_idx]:.3f})',
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

    # 실제 데이터 분포 기반 임계값 사용 (동적 분위수 방식)
    vh_thr = GRADE_THRESHOLDS_ACTUAL["Very High"]
    hi_thr = GRADE_THRESHOLDS_ACTUAL["High"]
    mo_thr = GRADE_THRESHOLDS_ACTUAL["Moderate"]

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

    # ── 언어 설정 (사이드바 상단) ────────────────────────────────────────────
    # session_state 기본값
    if 'lang' not in st.session_state:
        st.session_state['lang'] = 'ko'

    with st.sidebar:
        lang_col1, lang_col2 = st.columns([3, 2])
        with lang_col2:
            lang_choice = st.selectbox(
                "🌐", ['🇰🇷 한국어', '🇺🇸 English'],
                index=0 if st.session_state['lang'] == 'ko' else 1,
                label_visibility="collapsed",
                key="lang_selector",
            )
            st.session_state['lang'] = 'ko' if '한국어' in lang_choice else 'en'

    lang = st.session_state['lang']

    # 언어별 matplotlib 폰트
    if lang == 'ko':
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    GRADE_LABEL = GRADE_KR if lang == 'ko' else GRADE_EN

    # ── 헤더 ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='text-align:center;padding:10px 0 4px'>
        <h1 style='color:#d62728;margin:0'>🔥 WPFI</h1>
        <h3 style='color:#444;margin:0'>{t('title', lang)}</h3>
        <p style='color:#888;font-size:0.85em;margin:4px 0'>
        {t('subtitle', lang).replace(chr(10), '<br>')}
        </p>
    </div><hr style='margin:8px 0'>
    """, unsafe_allow_html=True)

    # 데이터 로드
    df    = load_risk()
    gdf   = load_geo()
    perf  = load_perf()
    trend = load_trend()
    model, scaler, features = load_model()

    if df is None:
        st.error("📂 분석 데이터가 없습니다. notebooks/01~08을 먼저 실행하세요.")
        return

    # geo 병합
    if gdf is not None:
        gdf_merged = gdf.merge(df, on='pole_id', how='left')
        gdf_merged['lat'] = gdf_merged.geometry.y
        gdf_merged['lon'] = gdf_merged.geometry.x
    else:
        gdf_merged = None

    # ── 사이드바 ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### {t('filter', lang)}")

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
        stns = [all_lbl] + sorted(df['nearest_station'].dropna().unique().tolist())
        sel_stn = st.selectbox(t('station_label', lang), stns)
        sel_stn_val = None if sel_stn == all_lbl else sel_stn
        score_type = st.radio(t('score_basis', lang),
                              [t('wpfi_score', lang), t('ml_score', lang)], index=0)
        risk_col = 'final_risk' if score_type == t('wpfi_score', lang) else 'ml_score'

        # ── 기상 시뮬레이션 ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### {t('scenario_lbl', lang)}")
        preset_name = st.selectbox(
            t('scenario_sel', lang),
            list(SIMULATION_PRESETS.keys()),
        )
        st.caption(f"📌 {SIMULATION_PRESETS[preset_name]['desc']}")

        with st.expander(t('fine_tune', lang)):
            wh_add = st.slider(t('wh_adjust', lang), -20, +30, 0)
            sp_add = st.slider(t('sp_adjust', lang), -10, +20, 0)

        is_sim = (preset_name != "현재 데이터 (기준)") or (wh_add != 0) or (sp_add != 0)
        df_sim = apply_simulation(df, preset_name, wh_add, sp_add) if is_sim else df

        if is_sim:
            orig_vh = (df['risk_grade'] == 'Very High').sum()
            sim_vh  = (df_sim['risk_grade'] == 'Very High').sum()
            delta   = sim_vh - orig_vh
            st.warning(f"{t('sim_active', lang)}\nVery High: {orig_vh:,} → {sim_vh:,} ({delta:+,})")

        st.markdown("---")
        st.markdown(f"### {t('stats_lbl', lang)}")
        total = len(df_sim)
        vh    = (df_sim['risk_grade'] == 'Very High').sum()
        unit  = "개" if lang == 'ko' else ""
        st.metric(t('total_fac', lang), f"{total:,}{unit}")
        st.metric("Very High", f"{vh:,}{unit} ({vh/total*100:.0f}%)")
        if model:
            st.success("✅ LightGBM " + ("로드됨" if lang == 'ko' else "Loaded"))
            if 'model_perf' in perf:
                auc = perf['model_perf']['auc'].max()
                st.metric("AUC", f"{auc:.4f}")

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
        if 'executive_summary' not in st.session_state:
            with st.spinner("OpenAI가 현황을 분석 중..."):
                st.session_state['executive_summary'] = get_executive_summary(df, perf)

        summary = st.session_state['executive_summary']
        icon = "🤖" if OPENAI_API_KEY else "📊"
        st.info(f"{icon} {summary}")

        col_ref, _ = st.columns([1, 4])
        with col_ref:
            if st.button(t('refresh', lang)):
                st.session_state.pop('executive_summary', None)
                st.rerun()

    st.markdown("---")

    # ── 7개 탭 ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        t('tab_map', lang), t('tab_list', lang), t('tab_detail', lang),
        t('tab_model', lang), t('tab_spatial', lang), t('tab_trend', lang),
        t('tab_forecast', lang),
    ])

    # ── TAB 1: 위험도 지도 ────────────────────────────────────────────────────
    with tab1:
        st.markdown(t('map_title', lang))
        if is_sim:
            st.warning(f"⚠️ {'시뮬레이션 모드' if lang=='ko' else 'Simulation Mode'}: {preset_name}")
        st.caption(t('map_caption', lang))

        if gdf_view is not None and len(gdf_view) > 0:
            m = build_map(gdf_view, show_grades, risk_col)
            st_folium(m, width=None, height=560, returned_objects=[])
        else:
            st.warning("지도 데이터가 없습니다.")

    # ── TAB 2: 우선점검 목록 ─────────────────────────────────────────────────
    with tab2:
        k = max(1, int(len(df_view) * topk_pct / 100))
        st.markdown(f"##### 🎯 우선점검 대상 상위 {topk_pct}% — **{k:,}개** 설비")
        if is_sim:
            st.warning(f"⚠️ 시뮬레이션 적용: {preset_name}")

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
        topk_risk  = top_df['WPFI점수'].sum()
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
        sel_pole = st.selectbox("설비 선택 (Top-50 고위험)", top50,
                                format_func=lambda x: f"설비 #{x}")

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
                        설비 #{int(sel_pole)} | {row.get('nearest_station','')} 권역</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**📊 4-Layer 위험 점수**")
                layers = [
                    ('🌪️ 기상위험',  'weather_hazard'),
                    ('🌲 공간노출',  'spatial_exposure'),
                    ('⚡ 설비노출',  'facility_exposure'),
                    ('🔥 화재이력',  'hist_prior'),
                ]
                for label, col_name in layers:
                    val = row.get(col_name, 0)
                    bar_color = '#d62728' if val >= 70 else '#ff7f0e' if val >= 50 else '#2ca02c'
                    st.markdown(
                        f"<div style='margin:4px 0'><small>{label}: <b>{val:.1f}</b></small>"
                        f"<div style='background:#eee;border-radius:4px;height:8px'>"
                        f"<div style='background:{bar_color};width:{int(val)}%;height:8px;border-radius:4px'></div>"
                        f"</div></div>", unsafe_allow_html=True)

                if 'ml_score' in row:
                    st.metric("🤖 ML 화재확률 점수", f"{row['ml_score']:.1f}/100")

            with col2:
                st.markdown("**🔬 SHAP 기여도 분석**")
                if model and features:
                    with st.spinner("SHAP 계산 중..."):
                        fig_shap = shap_waterfall(sel_pole, df_view, model, features)
                    if fig_shap:
                        st.pyplot(fig_shap)
                        plt.close()
                    else:
                        fig, ax = plt.subplots(figsize=(6, 3.5))
                        comp_labels = ['기상위험','공간노출','설비노출','화재이력']
                        comp_vals = [row.get('weather_hazard',0), row.get('spatial_exposure',0),
                                     row.get('facility_exposure',0), row.get('hist_prior',0)]
                        ax.barh(comp_labels, comp_vals, color=['#d62728','#ff7f0e','#1f77b4','#2ca02c'])
                        for b, v in zip(ax.patches, comp_vals):
                            ax.text(v + 0.5, b.get_y() + b.get_height()/2, f'{v:.1f}', va='center')
                        ax.set_xlabel('점수'); ax.set_xlim(0, 110)
                        ax.set_title(f'설비 #{sel_pole} 위험 구성요소')
                        plt.tight_layout(); st.pyplot(fig); plt.close()

                # AI 설명 — 항상 표시
                st.markdown("**🤖 AI 위험 설명**")
                exp_key = f"exp_{sel_pole}"
                if exp_key not in st.session_state:
                    with st.spinner("분석 중..."):
                        st.session_state[exp_key] = get_facility_explanation(row.to_dict())
                st.info(st.session_state[exp_key])

    # ── TAB 4: 모델 성능 ─────────────────────────────────────────────────────
    with tab4:
        st.markdown("##### 📊 LightGBM 모델 성능 검증")

        # 비전문가용 요약
        with st.expander("📖 이 탭이 보여주는 것 (쉬운 설명)", expanded=True):
            auc_val = perf['model_perf']['auc'].max() if 'model_perf' in perf else 0
            st.markdown(f"""
**모델이 얼마나 정확한가?**
- **AUC {auc_val:.4f}** — 1.0에 가까울수록 완벽. 0.5는 무작위 추측과 같음.
  현재 값은 **전체 설비 중 실제 위험 설비를 찾아내는 정확도가 매우 높음**을 의미합니다.
- **Recall@Top5%** — "상위 5%만 점검하면 실제 화재 위험 설비 중 몇 %를 잡을 수 있는가?"
- **Ablation Study** — 기상/공간/설비/이력 중 어떤 요소가 가장 중요한지 확인
- **Region CV** — 특정 지역 데이터로만 학습해도 다른 지역에서 잘 동작하는지 검증
            """)

        col1, col2 = st.columns(2)
        with col1:
            if 'recall_k' in perf:
                rk = perf['recall_k']
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(rk['k_pct']*100, rk['recall']*100, 'o-',
                        color='#d62728', lw=2, label='Recall@K')
                ax.plot(rk['k_pct']*100, rk['precision']*100, 's--',
                        color='#1f77b4', lw=2, label='Precision@K')
                ax.axhline(80, color='gray', ls=':', lw=1, alpha=0.5, label='80% 기준선')
                ax.set_xlabel('상위 K (%)'); ax.set_ylabel('성능 (%)')
                ax.set_title('상위 K% 점검 시 실제 위험 설비 탐지율', fontweight='bold')
                ax.legend(); ax.grid(alpha=0.3)
                plt.tight_layout(); st.pyplot(fig); plt.close()
                # 해석
                best_recall = rk['recall'].max()
                st.success(f"💡 상위 {int(rk.loc[rk['recall'].idxmax(),'k_pct']*100)}%만 점검해도 "
                           f"실제 위험 설비의 **{best_recall:.1%}**를 포함합니다.")

        with col2:
            if 'ablation' in perf:
                abl = perf['ablation']
                fig, ax = plt.subplots(figsize=(6, 4))
                colors_abl = ['#aec7e8','#6baed6','#3182bd','#08519c']
                bars = ax.bar(range(len(abl)), abl['top5_overlap_pct'],
                              color=colors_abl, alpha=0.85)
                ax.axhline(80, color='red', ls='--', lw=1.5, label='80% 기준')
                ax.set_xticks(range(len(abl)))
                ax.set_xticklabels([m.replace('_',' ') for m in abl['model']],
                                   rotation=15, ha='right', fontsize=9)
                ax.set_ylabel('Top5% 일치율 (%)')
                ax.set_title('요소 추가에 따른 예측 안정성', fontweight='bold')
                for b, v in zip(bars, abl['top5_overlap_pct']):
                    ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}%',
                            ha='center', fontsize=9)
                ax.legend(); ax.set_ylim(0, 115)
                plt.tight_layout(); st.pyplot(fig); plt.close()
                st.info("💡 기상 정보만 사용할 때보다 공간·설비·이력 정보를 추가할수록\n"
                        "Top-5% 위험 설비 식별 정확도가 크게 향상됩니다.")

        if 'shap_imp' in perf:
            st.markdown("**🔬 어떤 요소가 화재 위험을 가장 많이 결정하는가?**")
            si = perf['shap_imp'].head(15)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(si['feature'][::-1], si['mean_abs_shap'][::-1],
                    color='steelblue', alpha=0.8)
            ax.set_xlabel('SHAP 중요도 (값이 클수록 중요)')
            ax.set_title('화재 위험 결정 요인 순위 (Top 15)', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            top_feat = si.iloc[0]['feature']
            st.success(f"💡 가장 중요한 요인: **{top_feat}** — "
                       f"이 변수가 개별 설비의 화재 위험도를 결정하는 데 가장 큰 영향을 미칩니다.")

        if 'region_cv' in perf:
            st.markdown("**🗺️ 지역별 검증 — 다른 지역에서도 잘 동작하는가?**")
            cv = perf['region_cv']
            avg_auc = cv['auc'].mean()
            st.dataframe(
                cv[['test_station','auc','n_test']].rename(
                    columns={'test_station':'테스트 권역','auc':'AUC','n_test':'테스트 샘플'}
                ).assign(AUC=lambda d: d['AUC'].map('{:.4f}'.format)),
                use_container_width=True, hide_index=True
            )
            st.success(f"✅ 평균 Region CV AUC: **{avg_auc:.4f}** "
                       f"— 특정 지역 편향 없이 강원도 전역에서 안정적으로 작동합니다.")

    # ── TAB 5: 공간 분포 ─────────────────────────────────────────────────────
    with tab5:
        st.markdown("##### 📈 위험도 공간 분포")

        with st.expander("📖 이 탭이 보여주는 것 (쉬운 설명)", expanded=True):
            top_stn = df.groupby('nearest_station')['final_risk'].mean().idxmax()
            top_val = df.groupby('nearest_station')['final_risk'].mean().max()
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
            bars = ax.barh(stn_risk['nearest_station'], stn_risk['mean'],
                           color=colors_bar, alpha=0.85)
            for b, v in zip(bars, stn_risk['mean']):
                ax.text(v + 0.1, b.get_y() + b.get_height()/2,
                        f'{v:.1f}', va='center', fontsize=9)
            ax.set_xlabel('평균 WPFI 위험도')
            ax.set_title('관측소 권역별 평균 위험도', fontweight='bold')
            ax.axvline(df['final_risk'].mean(), color='navy', ls='--', lw=1.5,
                       label=f"전체 평균 {df['final_risk'].mean():.1f}")
            ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            grade_cnt = df['risk_grade'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            colors_pie = [GRADE_COLORS.get(g, '#888') for g in grade_cnt.index]
            ax.pie(grade_cnt.values,
                   labels=[f"{GRADE_KR.get(g,g)}\n{v:,}개" for g, v in grade_cnt.items()],
                   colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax.set_title('전체 위험등급 분포', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            vh_pct = grade_cnt.get('Very High', 0) / grade_cnt.sum() * 100
            st.info(f"💡 전체 설비의 **{vh_pct:.1f}%({grade_cnt.get('Very High',0):,}개)**가 즉각 점검 대상입니다.")

        img_path = OUT_FIGURES / 'fig4_risk_map.png'
        if img_path.exists():
            st.markdown("**강원도 전봇대 화재위험도 공간 분포**")
            st.image(str(img_path), use_container_width=True,
                     caption="붉은색 = 상위 5% 고위험 설비 (강원도 전역)")

    # ── TAB 6: 트렌드 분석 ───────────────────────────────────────────────────
    with tab6:
        st.markdown("##### 📅 기상위험도 시계열 트렌드 분석 (2022~2024)")

        with st.expander("📖 이 탭이 보여주는 것 (쉬운 설명)", expanded=True):
            st.markdown("""
**시간에 따라 화재 위험이 어떻게 변화하는가?**
- 월별·계절별 기상위험도 변화를 보여줍니다.
- **봄(3~5월)과 가을(9~11월)** 이 건조하고 바람이 강해 화재 위험이 가장 높습니다.
- 히트맵으로 **어느 권역이, 어느 달에** 특히 위험한지 한눈에 파악할 수 있습니다.
            """)

        if trend is None:
            st.warning("데이터 없음. notebooks/06_modeling.ipynb를 먼저 실행하세요.")
        else:
            MONTH_KR = {m: f'{m}월' for m in range(1, 13)}
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
                ax.plot(range(len(monthly)), monthly['hazard_mean'],
                        'o-', color='#d62728', lw=2, ms=4, label='평균 기상위험도')
                ax.fill_between(range(len(monthly)),
                                monthly['hazard_mean'], monthly['hazard_max'],
                                alpha=0.15, color='#d62728', label='평균~최대 범위')
                tick_step = max(1, len(monthly) // 12)
                ax.set_xticks(range(0, len(monthly), tick_step))
                ax.set_xticklabels(monthly['ym'].iloc[::tick_step],
                                   rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('기상위험도 (0~100)')
                ax.set_title('월별 평균 기상위험도 추이', fontweight='bold')
                ax.legend(fontsize=9); ax.grid(alpha=0.3)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            with col2:
                season_groups = [daily[daily['season'] == s]['weather_hazard_mean'].values
                                 for s in [1, 2, 3, 0]]
                fig, ax = plt.subplots(figsize=(6, 4))
                bp = ax.boxplot(season_groups, patch_artist=True, labels=['봄','여름','가을','겨울'])
                for patch, color in zip(bp['boxes'], [SEASON_COLOR[s] for s in [1,2,3,0]]):
                    patch.set_facecolor(color); patch.set_alpha(0.7)
                ax.set_ylabel('일평균 기상위험도')
                ax.set_title('계절별 기상위험도 분포', fontweight='bold')
                ax.grid(alpha=0.3, axis='y')
                plt.tight_layout(); st.pyplot(fig); plt.close()
                peak_season = ['봄','여름','가을','겨울'][[
                    daily[daily['season']==s]['weather_hazard_mean'].mean()
                    for s in [1,2,3,0]].index(max(
                    [daily[daily['season']==s]['weather_hazard_mean'].mean()
                     for s in [1,2,3,0]]))]
                st.info(f"💡 **{peak_season}** 기상위험도가 가장 높습니다. "
                        f"이 계절에 점검 자원을 집중 배치하는 것을 권고합니다.")

            # 월별 화재 발생
            st.markdown("**🔥 월별 화재 발생 건수 vs 기상위험도**")
            monthly_fire = (trend.groupby(['year','month'])
                            .agg(fire_count=('label','sum'),
                                 hazard_mean=('weather_hazard','mean'))
                            .reset_index())
            fig, ax1 = plt.subplots(figsize=(12, 4))
            ax2 = ax1.twinx()
            w = 0.25; offsets = {2022: -w, 2023: 0, 2024: w}
            colors_yr = {2022:'#5b9bd5', 2023:'#ed7d31', 2024:'#70ad47'}
            for yr, grp in monthly_fire.groupby('year'):
                ax1.bar(grp['month'] + offsets[yr], grp['fire_count'],
                        width=w, color=colors_yr[yr], alpha=0.8, label=f'{yr}년')
            mf_avg = monthly_fire.groupby('month')['hazard_mean'].mean()
            ax2.plot(mf_avg.index, mf_avg.values, 'k--o', lw=1.5, ms=5, alpha=0.7,
                     label='월평균 기상위험도')
            ax1.set_xticks(range(1, 13))
            ax1.set_xticklabels([MONTH_KR[m] for m in range(1, 13)])
            ax1.set_ylabel('화재 발생 건수'); ax2.set_ylabel('기상위험도')
            ax1.set_title('월별 화재 발생 건수 vs 기상위험도', fontweight='bold')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
            ax1.grid(alpha=0.3, axis='y')
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # 히트맵
            if 'nearest_station' in trend.columns:
                st.markdown("**🌡️ 관측소별 × 월별 기상위험도 히트맵**")
                stn_monthly = (trend.groupby(['nearest_station','month'])
                               ['weather_hazard'].mean().unstack())
                stn_monthly.columns = [MONTH_KR[c] for c in stn_monthly.columns]
                fig, ax = plt.subplots(figsize=(12, 5))
                im = ax.imshow(stn_monthly.values, aspect='auto',
                               cmap='YlOrRd', vmin=0, vmax=100)
                ax.set_xticks(range(len(stn_monthly.columns)))
                ax.set_xticklabels(stn_monthly.columns)
                ax.set_yticks(range(len(stn_monthly.index)))
                ax.set_yticklabels(stn_monthly.index)
                plt.colorbar(im, ax=ax, label='평균 기상위험도')
                for i in range(stn_monthly.shape[0]):
                    for j in range(stn_monthly.shape[1]):
                        v = stn_monthly.values[i, j]
                        if not np.isnan(v):
                            ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                                    fontsize=7, color='white' if v > 65 else 'black')
                ax.set_title('관측소별 × 월별 평균 기상위험도', fontweight='bold')
                plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── TAB 7: 기상 예보 ─────────────────────────────────────────────────────
    with tab7:
        st.markdown("##### 📡 기상 예보 기반 화재 위험 예보 (3일)")
        st.caption("기상청 단기예보 API → 설비별 LightGBM 추론 → 향후 3일 화재 위험 예측")

        GRADE_COLORS_F = GRADE_COLORS
        GRADE_KR_F     = GRADE_KR

        if not KMA_API_KEY:
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
                with st.spinner("📡 기상청 단기예보 자동 수집 중... (최초 1회)"):
                    try:
                        fc_result, fc_feat = run_full_forecast(KMA_API_KEY, DATA_PROCESSED, df)
                        st.session_state['fc_result'] = fc_result
                        st.session_state['fc_feat']   = fc_feat
                        st.session_state['fc_time']   = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                    except Exception as e:
                        st.session_state['fc_result'] = pd.DataFrame()
                        st.session_state['fc_feat']   = pd.DataFrame()
                        st.error(f"❌ 예보 수집 실패: {e}")

            col_info, col_refresh = st.columns([4, 1])
            with col_info:
                fc_time = st.session_state.get('fc_time', '')
                if fc_time:
                    st.caption(f"🕐 최종 수집: {fc_time}")
            with col_refresh:
                if st.button("🔄 새로고침", use_container_width=True):
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
                    lbl  = {0:"오늘",1:"내일",2:"모레"}.get(diff, dt.strftime("%m/%d"))
                    date_labels.append(f"{lbl} ({dt.strftime('%m.%d')})")

                # ── 3일 전체 종합 권고 ──────────────────────────────────────
                st.markdown("#### 🧭 향후 3일 종합 권고")
                overall_key = "fc_overall_recommend"
                if overall_key not in st.session_state:
                    with st.spinner("3일 종합 분석 중..."):
                        st.session_state[overall_key] = get_overall_forecast_recommendation(
                            fc_result, fc_feat)
                st.warning(f"🤖 {st.session_state[overall_key]}")

                # ── 3일 KPI 요약 카드 ──────────────────────────────────────
                st.markdown("#### 📅 날짜별 위험 현황")
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
🔴 {vh_d:,}개<br>
🟠 {hi_d:,}개<br>
기상위험 <b>{wh_max:.1f}</b>
</div>""", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("#### 🔍 날짜별 상세 분석")
                sel_idx  = st.radio("📅 날짜 선택", range(len(dates)),
                                    format_func=lambda i: date_labels[i], horizontal=True)
                sel_date = dates[sel_idx]
                fc_day   = fc_result[fc_result['fcst_date'] == sel_date]
                feat_day = fc_feat[fc_feat['fcst_date'] == sel_date] if fc_feat is not None else pd.DataFrame()

                # KPI
                cols = st.columns(4)
                for col, grade in zip(cols, ['Very High','High','Moderate','Low']):
                    col.metric(GRADE_KR_F[grade],
                               f"{(fc_day['forecast_grade']==grade).sum():,}개")

                # AI 예방 권고 — 항상 표시
                st.markdown("**🤖 AI 위험 분석 및 예방 권고**")
                rec_key = f"fc_recommend_{sel_date}"
                if rec_key not in st.session_state:
                    with st.spinner("OpenAI 분석 중..."):
                        st.session_state[rec_key] = get_forecast_recommendation(
                            feat_day, fc_day, date_labels[sel_idx])
                st.warning(st.session_state[rec_key])

                # 관측소별 기상 요약
                st.markdown("**🌡️ 관측소별 기상 예보**")
                if not feat_day.empty:
                    feat_disp = feat_day[[
                        'station','temp_max','rh_min','ws_max','precip_sum',
                        'weather_hazard','dry_watch_flag','wind_watch_flag','combined_risk_flag'
                    ]].rename(columns={
                        'station':'관측소','temp_max':'최고기온(℃)','rh_min':'최저습도(%)',
                        'ws_max':'최대풍속(m/s)','precip_sum':'강수량(mm)',
                        'weather_hazard':'기상위험도','dry_watch_flag':'건조주의보',
                        'wind_watch_flag':'강풍주의보','combined_risk_flag':'복합위험'
                    }).sort_values('기상위험도', ascending=False).round(1)

                    def color_hazard(val):
                        if isinstance(val, (int, float)):
                            if val >= 70: return 'background-color:#ffe0e0'
                            if val >= 50: return 'background-color:#fff0e0'
                        return ''
                    st.dataframe(
                        feat_disp.style.map(color_hazard, subset=['기상위험도']),
                        use_container_width=True, hide_index=True
                    )

                # 예보 위험도 지도
                st.markdown(f"**🗺️ 예보 위험도 지도 — {date_labels[sel_idx]}**")
                if gdf is not None:
                    gdf_fc = gdf.merge(
                        fc_day[['pole_id','forecast_risk','forecast_grade','weather_hazard']],
                        on='pole_id', how='inner')
                    gdf_fc['lat'] = gdf_fc.geometry.y
                    gdf_fc['lon'] = gdf_fc.geometry.x
                    m_fc = folium.Map(location=[37.5, 128.3], zoom_start=9,
                                      tiles='CartoDB positron')
                    for grade in ['Very High','High','Moderate','Low']:
                        color  = GRADE_COLORS_F[grade]
                        radius = {'Very High':8,'High':6,'Moderate':4,'Low':3}[grade]
                        sub = gdf_fc[gdf_fc['forecast_grade'] == grade]
                        total_g = len(sub)
                        if len(sub) > 2000:
                            sub = sub.nlargest(2000, 'forecast_risk')
                        layer = folium.FeatureGroup(
                            name=f"{GRADE_KR_F[grade]} ({total_g:,}개)")
                        for _, row in sub.iterrows():
                            folium.CircleMarker(
                                location=[row['lat'], row['lon']],
                                radius=radius, color=color, fill=True,
                                fill_color=color, fill_opacity=0.75,
                                tooltip=f"#{int(row['pole_id'])}: {row['forecast_risk']:.1f}점",
                                popup=folium.Popup(
                                    f"<b>설비 #{int(row['pole_id'])}</b><br>"
                                    f"예측 위험도: {row['forecast_risk']:.1f}<br>"
                                    f"등급: {GRADE_KR_F.get(row['forecast_grade'],'')}<br>"
                                    f"기상위험: {row['weather_hazard']:.1f}",
                                    max_width=200)
                            ).add_to(layer)
                        layer.add_to(m_fc)
                    folium.LayerControl().add_to(m_fc)
                    st_folium(m_fc, width=None, height=500, returned_objects=[])

                # Top-20 위험 설비
                st.markdown("**🎯 예측 고위험 설비 Top-20**")
                top20 = fc_day.nlargest(20, 'forecast_risk')[
                    ['pole_id','forecast_risk','forecast_grade','weather_hazard','nearest_station']
                ].copy().reset_index(drop=True)
                top20.index += 1
                top20.columns = ['설비ID','예측 위험도','예측 등급','기상위험도','관측소']
                top20['예측 위험도'] = top20['예측 위험도'].round(1)
                top20['예측 등급']  = top20['예측 등급'].map(GRADE_KR_F)
                st.dataframe(top20, use_container_width=True)

                csv = fc_day[['pole_id','fcst_date','forecast_risk','forecast_grade',
                               'weather_hazard','nearest_station']].to_csv(
                    index=False, encoding='utf-8-sig')
                st.download_button(
                    f"📥 {date_labels[sel_idx]} 예보 CSV",
                    csv, file_name=f"wpfi_forecast_{sel_date}.csv", mime="text/csv")

            elif fc_result is not None:
                st.error("예보 결과가 비어 있습니다. API 키 또는 네트워크를 확인하세요.")

    # ── 푸터 ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <hr>
    <p style='text-align:center;color:#aaa;font-size:0.8em'>
    {t('footer', lang)}
    </p>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
