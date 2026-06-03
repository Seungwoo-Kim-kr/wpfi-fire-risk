"""
KMA API허브 ASOS 일자료 수집 스크립트
신청한 API: 지상(종관·ASOS) 일자료(기간조회) — kma_sfcdd3.php

WPFI 필요 필드 매핑:
  KMA 필드      → WPFI 변수
  TA_MAX        → temp_max    (heat_score)
  TA_MIN        → temp_min
  TA_AVG        → temp_mean
  HM_MIN        → rh_min      (dryness_score ★ 건조주의보 기준)
  HM_AVG        → rh_mean     (실효습도 계산)
  WS_MAX        → ws_max      (wind_score ★)
  WS_INS        → ws_gust_max (강풍경보 기준: 26m/s)
  WS_AVG        → ws_mean
  RN_DAY        → precip      (no_rain_score)
  SS_DAY        → sunshine    (일조 — 건조 보조)
"""

import requests
import pandas as pd
import time
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────────────────────────
API_KEY = "여기에_발급받은_인증키_입력"   # ← API 키 입력
BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"
OUT_PATH = Path("data/raw/weather_gangwon_kma.csv")

# 강원도 KMA 관측소 코드
GANGWON_STATIONS = {
    101: "춘천", 114: "원주", 105: "강릉", 106: "동해",
     90: "속초", 177: "양양", 100: "대관령", 216: "태백",
    217: "삼척", 211: "정선", 212: "홍천", 313: "횡성",
    211: "인제"   # 인제는 별도 코드 확인 필요
}

# WPFI에 필요한 KMA 필드
NEEDED_COLS = [
    'TM', 'STN',
    'TA_AVG', 'TA_MAX', 'TA_MIN',   # 기온
    'HM_AVG', 'HM_MIN',              # 습도 ★
    'WS_AVG', 'WS_MAX', 'WS_INS',   # 풍속 ★
    'RN_DAY',                         # 강수 ★
    'SS_DAY',                         # 일조
]

# ── 데이터 수집 ────────────────────────────────────────────────────────────
def fetch_station(stn_code, tm1="20220101", tm2="20241231"):
    params = {
        "tm1": tm1,
        "tm2": tm2,
        "stn": stn_code,
        "disp": 0,
        "help": 1,          # 컬럼명 포함
        "authKey": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    if resp.status_code != 200:
        print(f"  ❌ 오류: {stn_code} → {resp.status_code}")
        return None

    lines = resp.text.strip().split('\n')
    # help=1 이면 첫 줄이 컬럼명
    data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
    if len(data_lines) < 2:
        print(f"  ⚠ 데이터 없음: {stn_code}")
        return None

    # 첫 줄: 컬럼명, 나머지: 데이터
    cols = data_lines[0].strip().split()
    rows = [l.strip().split() for l in data_lines[1:]]
    df = pd.DataFrame(rows, columns=cols)
    return df

all_dfs = []
for stn_code, stn_name in GANGWON_STATIONS.items():
    print(f"수집 중: {stn_name} ({stn_code})...")
    df = fetch_station(stn_code)
    if df is not None:
        df['station_name'] = stn_name
        all_dfs.append(df)
        print(f"  ✅ {len(df)}행")
    time.sleep(0.5)   # API 호출 간격

if not all_dfs:
    print("❌ 수집된 데이터 없음 — API 키 확인 필요")
else:
    df_all = pd.concat(all_dfs, ignore_index=True)

    # 필요 컬럼만 선택 (없는 컬럼 제외)
    exist_cols = [c for c in NEEDED_COLS if c in df_all.columns]
    df_sel = df_all[exist_cols + ['station_name']].copy()

    # 숫자 변환 (-9, -99, -999 → NaN 처리)
    for col in exist_cols[2:]:  # TM, STN 제외
        df_sel[col] = pd.to_numeric(df_sel[col], errors='coerce')
        df_sel.loc[df_sel[col] < -9, col] = None

    # 컬럼명 WPFI 형식으로 변환
    rename_map = {
        'TM': 'date', 'STN': 'stn_code', 'station_name': 'station',
        'TA_AVG': 'temp_mean', 'TA_MAX': 'temp_max', 'TA_MIN': 'temp_min',
        'HM_AVG': 'rh_mean',   'HM_MIN': 'rh_min',
        'WS_AVG': 'ws_mean',   'WS_MAX': 'ws_max', 'WS_INS': 'ws_gust_max',
        'RN_DAY': 'precip',    'SS_DAY': 'sunshine',
    }
    df_sel = df_sel.rename(columns={k: v for k, v in rename_map.items() if k in df_sel.columns})

    # 실효습도 계산 (건조주의보 핵심 변수)
    # EFF_RH = 0.7×RH(n) + 0.2×RH(n-1) + 0.1×RH(n-2)
    df_sel = df_sel.sort_values(['station', 'date']).reset_index(drop=True)
    df_sel['eff_humidity'] = (
        0.7 * df_sel['rh_mean'] +
        0.2 * df_sel.groupby('station')['rh_mean'].shift(1) +
        0.1 * df_sel.groupby('station')['rh_mean'].shift(2)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_sel.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장 완료: {OUT_PATH}")
    print(f"   총 {len(df_sel):,}행 × {len(df_sel.columns)}개 컬럼")
    print(f"   기간: {df_sel['date'].min()} ~ {df_sel['date'].max()}")
    print(f"   관측소: {df_sel['station'].nunique()}개")
    print(f"\n컬럼 목록: {list(df_sel.columns)}")
