"""KMA ASOS 일자료 수집 - 진행률 표시 버전"""
import requests, pandas as pd, time, sys
from pathlib import Path
from datetime import date, timedelta

API_KEY = "uFB2eXBeSOOQdnlwXljjDQ"
BASE    = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd.php"

STNS = {
    90:'속초', 100:'대관령', 101:'춘천', 105:'강릉',
    106:'동해', 114:'원주',  177:'양양', 211:'정선',
    212:'홍천', 216:'태백',  217:'삼척', 313:'횡성'
}

# 2022~2024 날짜 목록
start, end = date(2022,1,1), date(2024,12,31)
dates, d = [], start
while d <= end:
    dates.append(d.strftime('%Y%m%d'))
    d += timedelta(days=1)

TOTAL = len(STNS) * len(dates)
print(f"수집 대상: {len(STNS)}개 관측소 x {len(dates)}일 = {TOTAL:,}건")
print(f"예상 시간: 약 {TOTAL*0.15/60:.0f}분\n")

all_rows = []
done = 0

for stn_code, stn_name in STNS.items():
    stn_rows = 0
    for i, tm in enumerate(dates):
        try:
            r = requests.get(BASE, params={
                "tm":tm, "stn":stn_code, "disp":1, "help":0, "authKey":API_KEY
            }, timeout=10)

            for line in r.text.split('\n'):
                if line.strip() and not line.startswith('#') and not line.startswith('777') and ',' in line:
                    p = line.strip().split(',')
                    if len(p) >= 40:
                        all_rows.append({
                            'station': stn_name,
                            'date':    tm,
                            'temp_max':  p[11].strip(),
                            'temp_min':  p[13].strip(),
                            'temp_mean': p[10].strip(),
                            'rh_mean':   p[18].strip(),
                            'rh_min':    p[19].strip(),
                            'ws_max':    p[5].strip(),
                            'ws_gust_max': p[8].strip(),
                            'ws_mean':   p[2].strip(),
                            'precip':    p[38].strip(),
                            'sunshine':  p[32].strip() if len(p)>32 else '',
                            'dewpoint':  p[15].strip() if len(p)>15 else '',
                        })
                        stn_rows += 1
        except Exception as e:
            pass

        done += 1
        pct = done / TOTAL * 100

        # 진행률 출력 (같은 줄 덮어쓰기)
        bar = '#' * int(pct/5) + '-' * (20 - int(pct/5))
        sys.stdout.write(f"\r[{bar}] {pct:5.1f}% | {stn_name} {tm} | 누적 {len(all_rows):,}행")
        sys.stdout.flush()

        time.sleep(0.12)

    print(f"\n  [{stn_name}] 완료 — {stn_rows}행")

print(f"\n\n총 {len(all_rows):,}행 수집 완료. 저장 중...")

df = pd.DataFrame(all_rows)

# 숫자 변환 및 이상값 처리
num_cols = ['temp_max','temp_min','temp_mean','rh_mean','rh_min',
            'ws_max','ws_gust_max','ws_mean','precip','sunshine']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    df.loc[df[c] < -90, c] = None

# 날짜 형식 변환
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

# 실효습도 계산 (기상청 공식)
df = df.sort_values(['station','date']).reset_index(drop=True)
df['eff_humidity'] = (
    0.7 * df['rh_mean'] +
    0.2 * df.groupby('station')['rh_mean'].shift(1) +
    0.1 * df.groupby('station')['rh_mean'].shift(2)
)

OUT = Path("data/raw/weather_gangwon_kma.csv")
df.to_csv(OUT, index=False, encoding='utf-8-sig')

print(f"저장 완료: {OUT}")
print(f"  크기: {OUT.stat().st_size//1024}KB")
print(f"  행수: {len(df):,}")
print(f"  기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"  관측소: {sorted(df['station'].unique())}")
