"""
날짜 기반 label 재생성 스크립트
- 기존: 화재 발생 지점 인근 pole → 모든 날짜 label=1 (geographic)
- 변경: 산불 발생일 기준 [-7, 0]일 이내 + 3km 이내 pole → label=1 (temporal+spatial)

소방청 전기화재(연 580건/강원) → 산림청 산불(연 58건/강원) 으로 데이터 교체:
- 산불은 날씨(건조+강풍)에 직접적으로 영향받는 이벤트
- 발생 빈도가 낮아 화재일 vs 비화재일 대비가 명확
- 전력설비 주변 산불이 설비 손상의 핵심 위험
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
from pathlib import Path

ROOT = Path(__file__).parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EXTERNAL  = ROOT / "data" / "external"

RADIUS_M   = 3000   # 공간 반경 (m) — 산불은 더 넓은 범위에서 전파
WINDOW_PRE = 7      # 산불 발생 며칠 전부터 label=1

print("=" * 60)
print("날짜 기반 label 재생성 (산불 데이터 기반)")
print(f"  공간 반경: {RADIUS_M}m, 시간 창: [-{WINDOW_PRE}일, 0일]")
print("=" * 60)

# ── 1. 강원 산불 데이터 로드 (산림청) ─────────────────────────────────────
print("\n[1] 산불 데이터 로드")
forest = pd.read_csv(DATA_EXTERNAL / "fire_history_geocoded.csv")
forest["fire_date"] = pd.to_datetime(forest["fire_date"]).dt.normalize()

fires = forest[
    forest["발생장소_시도"].str.contains("강원", na=False)
    & (forest["fire_date"] >= "2022-01-01")
    & (forest["fire_date"] <= "2024-12-31")
    & forest["lon"].notna()
    & forest["lat"].notna()
].copy().reset_index(drop=True)

print(f"  강원 산불 건수: {len(fires):,}")
print(f"  날짜 범위: {fires['fire_date'].min().date()} ~ {fires['fire_date'].max().date()}")
print(f"  화재 발생 unique 날짜 수: {fires['fire_date'].nunique():,}")

# ── 2. 전봇대 위치 로드 ──────────────────────────────────────────────────
print("\n[2] 전봇대 위치 로드")
tf = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
fac_all = gpd.read_file(DATA_PROCESSED / "facility_geo.gpkg")
fac = fac_all[fac_all["pole_id"].isin(tf["pole_id"].unique())].copy()
fac = fac.reset_index(drop=True)
print(f"  train_features 포함 poles: {len(fac):,}")

# ── 3. 공간 인덱스 구축 (BallTree, haversine) ─────────────────────────────
print("\n[3] 공간 인덱스 구축")
EARTH_R = 6371000.0
pole_coords_rad = np.radians(fac[["lat", "lon"]].values)
tree = BallTree(pole_coords_rad, metric="haversine")
print("  BallTree 완료")

# ── 4. 화재별 인근 poles 찾기 ────────────────────────────────────────────
print("\n[4] 화재별 1km 이내 poles + 날짜 창 생성")
label1_records = []
radius_rad = RADIUS_M / EARTH_R

for i, fire in fires.iterrows():
    fire_coord = np.radians([[fire["lat"], fire["lon"]]])
    idx = tree.query_radius(fire_coord, r=radius_rad)[0]
    if len(idx) == 0:
        continue
    nearby_pole_ids = fac.iloc[idx]["pole_id"].values
    fire_dt = fire["fire_date"]
    for d in range(WINDOW_PRE + 1):  # -7일 ~ 당일
        date = fire_dt - pd.Timedelta(days=d)
        for pid in nearby_pole_ids:
            label1_records.append((pid, date))

    if (i + 1) % 200 == 0:
        print(f"  진행: {i+1}/{len(fires)} fires — 현재까지 {len(label1_records):,}건")

print(f"  생성된 (pole, date) label=1 쌍: {len(label1_records):,}")

# ── 5. 중복 제거 후 set 변환 ─────────────────────────────────────────────
print("\n[5] 중복 제거")
label1_set = set(label1_records)
print(f"  중복 제거 후: {len(label1_set):,}")

# 날짜별 화재-인근 dates
unique_fire_dates = set(dt for _, dt in label1_set)
print(f"  화재 창에 포함된 unique dates: {len(unique_fire_dates):,}")
unique_fire_poles = set(pid for pid, _ in label1_set)
print(f"  화재 창에 포함된 unique poles: {len(unique_fire_poles):,}")

# ── 6. train_features label 재할당 ───────────────────────────────────────
print("\n[6] train_features label 재할당")
tf["date"] = pd.to_datetime(tf["date"]).dt.normalize()

# 벡터화된 lookup: (pole_id, date) 튜플 → label
tf_tuples = list(zip(tf["pole_id"].values, tf["date"].values))
new_label = np.array([1 if (int(p), pd.Timestamp(d)) in label1_set else 0
                      for p, d in zip(tf["pole_id"].values, tf["date"].values)], dtype=np.int8)

print(f"  기존 label=1: {tf['label'].sum():,}")
print(f"  새로운 label=1: {new_label.sum():,}")
print(f"  새로운 label=1 비율: {new_label.mean()*100:.2f}%")

tf["label"] = new_label.astype(int)

# ── 7. 저장 ──────────────────────────────────────────────────────────────
print("\n[7] 파일 저장")
tf.to_parquet(DATA_PROCESSED / "train_features.parquet", index=False)
print("  ✓ train_features.parquet 저장")

labels_new = tf[["pole_id", "date", "label"]].copy()
labels_new.to_parquet(DATA_PROCESSED / "labels.parquet", index=False)
print("  ✓ labels.parquet 저장")

# ── 8. column_map.json 업데이트 ──────────────────────────────────────────
print("\n[8] column_map.json 업데이트")
with open(DATA_PROCESSED / "column_map.json", encoding="utf-8") as f:
    cmap = json.load(f)

cmap["flags"]["has_label"]     = True
cmap["facility"]["label"]      = "label"

with open(DATA_PROCESSED / "column_map.json", "w", encoding="utf-8") as f:
    json.dump(cmap, f, ensure_ascii=False, indent=2)
print("  ✓ column_map.json: has_label=True, label='label'")

# ── 9. 결과 요약 ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("완료 요약")
print("=" * 60)
print(f"  label=0: {(tf['label']==0).sum():,}")
print(f"  label=1: {(tf['label']==1).sum():,}")
print(f"  비율 (1:0): 1:{(tf['label']==0).sum()/max(1,(tf['label']==1).sum()):.1f}")
print(f"  화재 근접 poles: {tf[tf['label']==1]['pole_id'].nunique():,}")
print()
print("다음 단계:")
print("  1. jupyter nbconvert --to notebook --execute notebooks/06_modeling.ipynb")
print("  2. jupyter nbconvert --to notebook --execute notebooks/07_explainability.ipynb")
print("  3. jupyter nbconvert --to notebook --execute notebooks/08_validation.ipynb")
