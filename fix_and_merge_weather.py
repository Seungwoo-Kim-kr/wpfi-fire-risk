"""
KMA 기상 데이터 정리 + 결측 재수집 + 파이프라인 재실행
- 날짜 타입 충돌 버그 수정
- -9.0 sentinel 값 처리
- 삼척/횡성/속초/태백 결측분 재수집
"""
import requests, pandas as pd, numpy as np, time, sys
from pathlib import Path
from datetime import date, timedelta

API_KEY = "uFB2eXBeSOOQdnlwXljjDQ"
BASE    = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd.php"
CSV     = Path("data/raw/weather_gangwon_kma.csv")

# ── STEP 1. 기존 데이터 로드 + sentinel 처리 ────────────────────────────
print("=" * 50)
print("STEP 1. 기존 데이터 로드 및 이상값 처리")
print("=" * 50)

df = pd.read_csv(CSV)
df['date'] = pd.to_datetime(df['date'])   # ← 타입 통일 (핵심 수정)

before = len(df)
# -9.0 → 강수 없음(0) or 결측(NaN)
df['precip'] = df['precip'].apply(lambda x: 0.0 if pd.notna(x) and x < 0 else x)
for c in ['rh_mean','rh_min','ws_max','ws_mean','ws_gust_max','sunshine']:
    if c in df.columns:
        df.loc[df[c] < 0, c] = np.nan

print(f"  기존: {before}행, sentinel 처리 완료")
print(f"  관측소: {sorted(df['station'].unique())}")

# ── STEP 2. 결측 날짜 파악 ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2. 결측 현황 파악")
print("=" * 50)

all_dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
expected  = len(all_dates)

STNS = {90:'속초', 100:'대관령', 101:'춘천', 105:'강릉',
        106:'동해', 114:'원주',  177:'양양', 211:'정선',
        212:'홍천', 216:'태백',  217:'삼척', 313:'횡성'}

need_map = {}
for code, name in STNS.items():
    existing = set(df[df['station']==name]['date']) if name in df['station'].values else set()
    need = [d for d in all_dates if d not in existing]
    pct  = (expected - len(need)) / expected * 100
    st   = "✅" if len(need) == 0 else ("⚠" if len(need) < 30 else "❌")
    print(f"  {st} {name:<6}: {expected-len(need)}/{expected}일 ({pct:.0f}%) — 결측 {len(need)}일")
    if need:
        need_map[code] = (name, need)

total_need = sum(len(v[1]) for v in need_map.values())
print(f"\n  총 재수집: {total_need}건")

if total_need == 0:
    print("  모든 데이터 완전! 재수집 불필요")
else:
    # ── STEP 3. 결측분 재수집 ─────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 3. 결측분 재수집")
    print("=" * 50)

    all_new, done = [], 0
    for stn_code, (stn_name, need) in need_map.items():
        print(f"\n  [{stn_name}] {len(need)}일 수집 시작...")
        for i, d in enumerate(need):
            tm = d.strftime('%Y%m%d')
            try:
                r = requests.get(BASE, params={
                    "tm":tm,"stn":stn_code,"disp":1,"help":0,"authKey":API_KEY
                }, timeout=5)
                for line in r.text.split('\n'):
                    if (line.strip() and not line.startswith('#')
                            and not line.startswith('777') and ',' in line):
                        p = line.strip().split(',')
                        if len(p) >= 39:
                            pv = float(p[38]) if p[38].strip() else 0.0
                            all_new.append({
                                'station':    stn_name,
                                'date':       d,       # pd.Timestamp 타입
                                'temp_max':   float(p[11]) if p[11].strip() else np.nan,
                                'temp_min':   float(p[13]) if p[13].strip() else np.nan,
                                'temp_mean':  float(p[10]) if p[10].strip() else np.nan,
                                'rh_mean':    float(p[18]) if p[18].strip() else np.nan,
                                'rh_min':     float(p[19]) if p[19].strip() else np.nan,
                                'ws_max':     float(p[5])  if p[5].strip()  else np.nan,
                                'ws_gust_max':float(p[8])  if p[8].strip()  else np.nan,
                                'ws_mean':    float(p[2])  if p[2].strip()  else np.nan,
                                'precip':     max(0.0, pv) if pv >= 0 else 0.0,
                                'sunshine':   float(p[32]) if len(p)>32 and p[32].strip() else np.nan,
                                'dewpoint':   float(p[15]) if len(p)>15 and p[15].strip() else np.nan,
                            })
            except:
                pass

            done += 1
            total_pct = done / total_need * 100
            stn_pct   = (i+1) / len(need) * 100
            bar = '#' * int(total_pct/5) + '-' * (20 - int(total_pct/5))
            sys.stdout.write(
                f"\r  [{bar}] 전체 {total_pct:4.1f}% | "
                f"[{stn_name}] {stn_pct:4.1f}% ({i+1}/{len(need)}) | {tm}"
            )
            sys.stdout.flush()
            time.sleep(0.12)

        print(f"\n  [{stn_name}] 완료!")

    # ── STEP 4. 병합 ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("STEP 4. 병합 및 저장")
    print("=" * 50)

    df_new = pd.DataFrame(all_new)
    df_new['date'] = pd.to_datetime(df_new['date'])   # 타입 통일
    df = pd.concat([df, df_new], ignore_index=True)
    print(f"  신규 {len(df_new)}행 추가")

# ── STEP 5. 최종 정리 + 저장 ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5. 최종 정리 및 저장")
print("=" * 50)

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['station','date']).drop_duplicates(['station','date']).reset_index(drop=True)

# 실효습도 재계산
df['eff_humidity'] = (
    0.7 * df['rh_mean'] +
    0.2 * df.groupby('station')['rh_mean'].shift(1) +
    0.1 * df.groupby('station')['rh_mean'].shift(2)
)

df.to_csv(CSV, index=False, encoding='utf-8-sig')
print(f"  저장: {CSV} ({CSV.stat().st_size//1024}KB)")

# ── 최종 품질 리포트 ──────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("최종 품질 리포트")
print("=" * 50)
print(f"{'관측소':<8} {'행수':>5} {'충족률':>6} {'상태'}")
for stn in sorted(df['station'].unique()):
    n   = len(df[df['station']==stn])
    pct = n / expected * 100
    st  = "✅" if pct >= 99 else ("⚠" if pct >= 90 else "❌")
    print(f"  {st} {stn:<6}: {n:>4}/{expected} ({pct:.0f}%)")

print(f"\n  총 {len(df):,}행 | {df['station'].nunique()}개 관측소")
print(f"  기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"\n완료! 다음 단계: Notebook 04 재실행")
