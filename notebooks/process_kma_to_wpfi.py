"""
KMA ASOS 데이터 → WPFI weather features 변환
수집된 weather_kma_raw.csv → weather_gangwon_kma.csv → 파이프라인 재실행
"""
import pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from pathlib import Path
from config import DATA_RAW, DATA_PROCESSED

RAW   = DATA_RAW / 'weather_kma_raw.csv'
OUT   = DATA_RAW / 'weather_gangwon_kma.csv'

print("KMA 원시 데이터 로드 중...")
df = pd.read_csv(RAW, dtype=str)
print(f"  원시: {df.shape}")

# 숫자 변환 (-9, -99, -999 → NaN)
num_cols = ['WS_AVG','WS_MAX','WS_INS','TA_AVG','TA_MAX','TA_MIN',
            'HM_AVG','HM_MIN','RN_DAY','SS_DAY','TD_AVG','SI_DAY']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df.loc[df[c] < -9, c] = np.nan

# 날짜 정제
df['date'] = pd.to_datetime(df['TM'].str.strip(), format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['date'])

# 강원도 관측소만
gangwon = ['속초','대관령','춘천','강릉','동해','원주','양양',
           '정선','홍천','태백','삼척','횡성','고성','인제']
df = df[df['station'].isin(gangwon)]

# WPFI 컬럼명으로 rename
df = df.rename(columns={
    'TA_AVG':'temp_mean', 'TA_MAX':'temp_max', 'TA_MIN':'temp_min',
    'HM_AVG':'rh_mean',   'HM_MIN':'rh_min',
    'WS_AVG':'ws_mean',   'WS_MAX':'ws_max',   'WS_INS':'ws_gust_max',
    'RN_DAY':'precip',    'SS_DAY':'sunshine',  'TD_AVG':'dewpoint',
})

# 실효습도 계산 (기상청 공식)
# EFF_RH(n) = 0.7×RH(n) + 0.2×RH(n-1) + 0.1×RH(n-2)
df = df.sort_values(['station','date']).reset_index(drop=True)
df['eff_humidity'] = (
    0.7 * df['rh_mean'] +
    0.2 * df.groupby('station')['rh_mean'].shift(1) +
    0.1 * df.groupby('station')['rh_mean'].shift(2)
)

# 최종 컬럼 선택
final_cols = ['station','date','temp_max','temp_min','temp_mean',
              'rh_max','rh_min','rh_mean','eff_humidity',
              'ws_max','ws_gust_max','ws_mean','precip','sunshine']
# rh_max는 없을 수 있으므로 처리
if 'rh_max' not in df.columns:
    df['rh_max'] = df['rh_mean']  # 근사값

exist = [c for c in final_cols if c in df.columns]
df_out = df[exist].copy()
df_out.to_csv(OUT, index=False, encoding='utf-8-sig')

print(f"  변환 완료: {df_out.shape}")
print(f"  기간: {df_out['date'].min().date()} ~ {df_out['date'].max().date()}")
print(f"  관측소: {sorted(df_out['station'].unique())}")
print(f"  저장: {OUT}")
print(f"  NaN 현황:\n{df_out.isnull().sum()}")
