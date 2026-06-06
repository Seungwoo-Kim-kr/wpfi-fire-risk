"""
날씨 조건 기반 label 재정의 v4 — 직접 join 방식
핵심: train_features에 sgg_name 추가 후, (sgg_name, date_str) set으로 벡터 조건 처리
중간 DataFrame 팽창 없음
"""
import pandas as pd, numpy as np, pickle, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from config import DATA_PROCESSED, OUT_TABLES, SEED

# ── 1. 화재 + 날씨 join → 건조/고온 이벤트 분류 ──────────────────────
print("[1] 화재 이벤트 분류")
elec = pd.read_csv('data/external/fire_history_elec_geocoded.csv')
elec['fire_date'] = pd.to_datetime(elec['fire_date']).dt.normalize()
gangwon = elec[
    elec['시도'].str.contains('강원', na=False) &
    (elec['fire_date'] >= '2022-01-01') & (elec['fire_date'] <= '2024-12-31')
].copy()

wf = pd.read_parquet(DATA_PROCESSED / 'weather_fwi.parquet')
wf['date'] = pd.to_datetime(wf['date']).dt.normalize()

pole_sgg = pd.read_parquet(DATA_PROCESSED / 'pole_sigungu_map.parquet')
tf_stn   = pd.read_parquet(DATA_PROCESSED / 'train_features.parquet',
                             columns=['pole_id','nearest_station']).drop_duplicates('pole_id')
tf_stn   = tf_stn.merge(pole_sgg, on='pole_id', how='left')
sgg_stn  = tf_stn.groupby('sgg_name')['nearest_station'].agg(
    lambda x: x.value_counts().index[0]).to_dict()

gangwon['station'] = gangwon['시군구_norm'].map(sgg_stn)
fire_w = gangwon.merge(
    wf[['station','date','dry_streak','fwi_score','temp_max']],
    left_on=['station','fire_date'], right_on=['station','date'], how='left'
).dropna(subset=['fwi_score'])

dry_med = fire_w['dry_streak'].median()
fwi_med = fire_w['fwi_score'].median()
heat_th = 25.0
fire_w['is_dry']  = (fire_w['dry_streak'] >= dry_med) & (fire_w['fwi_score'] >= fwi_med)
fire_w['is_heat'] = fire_w['temp_max'] >= heat_th

dry_fires  = fire_w[fire_w['is_dry']][['시군구_norm','fire_date']].drop_duplicates()
heat_fires = fire_w[fire_w['is_heat']][['시군구_norm','fire_date']].drop_duplicates()
print(f"   건조형: {len(dry_fires)}건  고온형: {len(heat_fires)}건")
print(f"   임계값: dry_streak≥{dry_med:.0f}, fwi≥{fwi_med:.1f}, temp≥{heat_th}°C")

# ── 2. (sgg_name, date_str) frozenset 생성 ────────────────────────────
print("[2] 날짜 윈도우 셋 생성")
WINDOW = 7
def make_sgg_date_set(fires_df):
    s = set()
    for _, r in fires_df.iterrows():
        for d in range(WINDOW + 1):
            s.add((r['시군구_norm'],
                   (r['fire_date'] - pd.Timedelta(days=d)).strftime('%Y-%m-%d')))
    return s

dry_set  = make_sgg_date_set(dry_fires)
heat_set = make_sgg_date_set(heat_fires)
print(f"   건조형 (sgg,date) 쌍: {len(dry_set):,}")
print(f"   고온형 (sgg,date) 쌍: {len(heat_set):,}")

# ── 3. train_features에 sgg_name 추가 후 조건 체크 ─────────────────────
print("[3] train_features 로드 + label 할당")
tf = pd.read_parquet(DATA_PROCESSED / 'train_features.parquet')
tf['date'] = pd.to_datetime(tf['date']).dt.normalize()

# sgg_name 추가
tf = tf.merge(pole_sgg[['pole_id','sgg_name']], on='pole_id', how='left')

# date_str 컬럼 생성 (벡터화)
tf['date_str'] = tf['date'].dt.strftime('%Y-%m-%d')

# (sgg_name, date_str) 튜플 컬럼 → isin 체크
print("   튜플 키 생성 중...")
keys = list(zip(tf['sgg_name'].fillna('').values, tf['date_str'].values))

print("   label_dry 할당 중...")
tf['label_dry']  = np.array([1 if k in dry_set  else 0 for k in keys], dtype=np.int8)
print("   label_heat 할당 중...")
tf['label_heat'] = np.array([1 if k in heat_set else 0 for k in keys], dtype=np.int8)

tf = tf.drop(columns=['sgg_name','date_str'])
tf.to_parquet(DATA_PROCESSED / 'train_features.parquet', index=False)

dry_pos  = int(tf['label_dry'].sum())
heat_pos = int(tf['label_heat'].sum())
both_pos = int(((tf['label_dry']==1) & (tf['label_heat']==1)).sum())
print(f"\n✅ label_dry : {dry_pos:,}건 ({dry_pos/len(tf)*100:.2f}%)")
print(f"✅ label_heat: {heat_pos:,}건 ({heat_pos/len(tf)*100:.2f}%)")
print(f"   공존(고온건조): {both_pos:,}건")

# ── 4. 날씨 신호 검증 ─────────────────────────────────────────────────
print("\n[4] 날씨 신호 검증")
for lbl, name in [('label_dry','건조형'), ('label_heat','고온형')]:
    pos = tf[tf[lbl]==1]; neg = tf[tf[lbl]==0]
    print(f"\n  [{name}]")
    for col in ['dryness_score','heat_score','wind_score','fwi_score','eff_humidity']:
        if col not in tf.columns: continue
        p = pos[col].mean(); n = neg[col].mean()
        diff = (p-n)/max(abs(n),0.001)*100
        mark = '✅' if abs(diff) > 5 else ''
        print(f"    {col:28s}: neg={n:.3f}  pos={p:.3f}  ({diff:+.1f}%) {mark}")

# ── 5. 모델 학습 ──────────────────────────────────────────────────────
print("\n[5] 모델 학습")
FIRE_HIST = {'fire_count_3000m','fire_count_5000m',
             'elec_fire_count_1000m','elec_fire_count_3000m','elec_x_forest'}
EXCLUDE   = {'pole_id','date','label','label_arc','label_mech',
             'label_dry','label_heat','nearest_station'}
FEAT = [c for c in tf.columns if c not in EXCLUDE and c not in FIRE_HIST
        and tf[c].dtype in [np.float64,np.float32,np.int64,np.int32,np.int8]]

dates  = tf['date'].sort_values().unique()
cutoff = dates[int(len(dates)*0.8)]
tr = tf['date'] < cutoff; te = tf['date'] >= cutoff
X  = tf[FEAT].fillna(tf[FEAT].median())
X_tr, X_te = X[tr], X[te]

weather_kw = ['score','hazard','humidity','fwi','accumulated','precip',
              'temp','rh','ws','dry','hot','heat','warn','watch','flag','wind']
results = {}; models = {}

for lbl, name in [('label_dry','건조형'), ('label_heat','고온형')]:
    y   = tf[lbl].astype(int)
    spw = int((y==0).sum()/max(y.sum(),1))
    print(f"\n  [{name}] pos={y.sum():,}, spw={spw}")
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                            num_leaves=63, random_state=SEED,
                            scale_pos_weight=spw, verbose=-1)
    m.fit(X_tr, y[tr])
    prob = m.predict_proba(X_te)[:,1]
    auc  = roc_auc_score(y[te], prob)
    r10  = y[te].values[np.argsort(prob)[::-1][:int(len(y[te])*0.1)]].sum()/max(y[te].sum(),1)
    shap = pd.DataFrame({'feature':FEAT,
                          'gain':m.booster_.feature_importance('gain')
                         }).sort_values('gain', ascending=False)
    total = shap['gain'].sum()
    w_pct = shap[shap['feature'].apply(
        lambda f: any(k in f for k in weather_kw))]['gain'].sum()/total*100
    print(f"  AUC={auc:.4f}  R@10%={r10:.3f}  날씨기여={w_pct:.1f}%")
    for _, r in shap.head(5).iterrows():
        tag = '🌤' if any(k in r['feature'] for k in weather_kw) else '🗺'
        print(f"    {tag} {r['feature']:30s}: {r['gain']/total*100:.1f}%")
    results[name] = {'auc':auc,'recall_top10':r10,'weather_pct':w_pct}
    models[lbl]   = {'model':m,'feature_cols':FEAT}

with open(DATA_PROCESSED/'model_dry.pkl',  'wb') as f: pickle.dump(models['label_dry'],  f)
with open(DATA_PROCESSED/'model_heat.pkl', 'wb') as f: pickle.dump(models['label_heat'], f)
pd.DataFrame(results).T.to_csv(OUT_TABLES/'multihazard_model_performance.csv')

# ── 6. risk_scores 업데이트 ───────────────────────────────────────────
print("\n[6] risk_scores 업데이트")
rs = pd.read_parquet(DATA_PROCESSED/'risk_scores.parquet')
meds = {c: tf[c].median() for c in FEAT if c in tf.columns}
for label_key in ['label_dry','label_heat']:
    fc = models[label_key]['feature_cols']
    X_rs = pd.DataFrame({c: rs[c].fillna(meds.get(c,0)) if c in rs.columns
                          else meds.get(c,0) for c in fc})
    col = 'prob_dry' if label_key=='label_dry' else 'prob_heat'
    rs[col] = models[label_key]['model'].predict_proba(X_rs)[:,1]

lightning = pd.read_parquet(DATA_PROCESSED/'lightning_risk.parquet')
rs = rs.drop(columns=[c for c in ['lightning_risk_rule','prob_light','hazard_combined',
                                    'prob_arc','prob_mech'] if c in rs.columns])
rs = rs.merge(lightning, on='pole_id', how='left')
rs['prob_light']      = rs['lightning_risk_rule'].fillna(0) / 100
rs['hazard_combined'] = (0.50*rs['prob_dry'] + 0.30*rs['prob_heat'] + 0.20*rs['prob_light']).clip(0,1)
rs.to_parquet(DATA_PROCESSED/'risk_scores.parquet', index=False)

print("\n" + "="*55)
print("WPFI_v2 = 0.5×P_dry + 0.3×P_heat + 0.2×P_light")
print(f"{'모델':20s} {'AUC':>7} {'R@10%':>7} {'날씨기여':>8}")
for name, r in results.items():
    print(f"  {name:20s} {r['auc']:>7.4f} {r['recall_top10']:>7.3f} {r['weather_pct']:>7.1f}%")
print("✅ 완료")
