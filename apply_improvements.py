"""
방안 A: 낙뢰 rule v2 — 고도 + 여름 대기불안정 + 여름 습도
방안 B: month/season 계절 컬럼 제거 후 모델 재학습
"""
import pandas as pd, numpy as np, pickle, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from pathlib import Path
from config import DATA_PROCESSED, OUT_TABLES, SEED

# ══════════════════════════════════════════════════════
# 방안 A: 낙뢰 rule v2
# ══════════════════════════════════════════════════════
print("=" * 55)
print("방안 A: 낙뢰 rule v2 (고도 + 대기불안정 + 습도)")
print("=" * 55)

# 여름(6~8월) 관측소별 평균 일교차 / 습도
wf = pd.read_parquet(DATA_PROCESSED / 'weather_fwi.parquet')
wf['date'] = pd.to_datetime(wf['date'])
wf['month'] = wf['date'].dt.month
wf['temp_range'] = wf['temp_max'] - wf['temp_min']
summer = wf[wf['month'].isin([6, 7, 8])]
stn_stats = summer.groupby('station').agg(
    instability=('temp_range', 'mean'),   # 일교차 클수록 대기불안정
    rh_summer=('rh_mean', 'mean'),        # 습도 높을수록 뇌우 가능성
).reset_index()
print(f"관측소별 여름 통계:\n{stn_stats.set_index('station').round(2)}\n")

# pole → sgg → station 매핑
pole_sgg = pd.read_parquet(DATA_PROCESSED / 'pole_sigungu_map.parquet')
tf_stn   = pd.read_parquet(DATA_PROCESSED / 'train_features.parquet',
                             columns=['pole_id','nearest_station']).drop_duplicates('pole_id')
tf_stn   = tf_stn.merge(pole_sgg, on='pole_id', how='left')
sgg_stn  = tf_stn.groupby('sgg_name')['nearest_station'].agg(
    lambda x: x.value_counts().index[0]).to_dict()
pole_station = tf_stn[['pole_id','sgg_name']].copy()
pole_station['station'] = pole_station['sgg_name'].map(sgg_stn)

# 고도 데이터
buf = pd.read_parquet(DATA_PROCESSED / 'buffer_features_v2.parquet',
                      columns=['pole_id','elevation_m'])

# 통합
lightning_df = buf.merge(pole_station[['pole_id','station']], on='pole_id', how='left')
lightning_df = lightning_df.merge(stn_stats, on='station', how='left')

# 정규화
ELEV_MAX = buf['elevation_m'].quantile(0.99)          # 1033m
INST_MAX = stn_stats['instability'].max()              # 최대 일교차
RH_MAX   = stn_stats['rh_summer'].max()               # 최대 여름 습도

lightning_df['elev_norm']  = (lightning_df['elevation_m'].clip(0, ELEV_MAX) / ELEV_MAX).fillna(0)
lightning_df['inst_norm']  = (lightning_df['instability'].fillna(stn_stats['instability'].mean()) / INST_MAX)
lightning_df['rh_norm']    = (lightning_df['rh_summer'].fillna(stn_stats['rh_summer'].mean()) / RH_MAX)

# 가중 합산 (0~100 스케일)
lightning_df['lightning_risk_rule'] = (
    0.50 * lightning_df['elev_norm'] +
    0.30 * lightning_df['inst_norm'] +
    0.20 * lightning_df['rh_norm']
) * 100

# 상관계수 비교
from scipy.stats import pearsonr
old_lr = pd.read_parquet(DATA_PROCESSED / 'lightning_risk.parquet')
merged_check = lightning_df.merge(old_lr.rename(columns={'lightning_risk_rule':'old'}), on='pole_id')
corr_old, _ = pearsonr(merged_check['elev_norm']*100, merged_check['old'])
corr_new, _ = pearsonr(merged_check['elev_norm']*100, merged_check['lightning_risk_rule'])
print(f"기존 rule vs 고도 상관계수: {corr_old:.4f}")
print(f"신규 rule vs 고도 상관계수: {corr_new:.4f}  ← 낮을수록 개선")

lightning_out = lightning_df[['pole_id','lightning_risk_rule']].copy()
lightning_out.to_parquet(DATA_PROCESSED / 'lightning_risk.parquet', index=False)
print(f"\n✅ lightning_risk.parquet 저장")
print(f"   평균: {lightning_out['lightning_risk_rule'].mean():.2f}")
print(f"   max:  {lightning_out['lightning_risk_rule'].max():.2f}")

# ══════════════════════════════════════════════════════
# 방안 B: 계절 컬럼 제거 + 모델 재학습
# ══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("방안 B: 계절 컬럼 제거 + 모델 재학습")
print("=" * 55)

tf = pd.read_parquet(DATA_PROCESSED / 'train_features.parquet')
tf['date'] = pd.to_datetime(tf['date'])

FIRE_HIST    = {'fire_count_3000m','fire_count_5000m',
                'elec_fire_count_1000m','elec_fire_count_3000m','elec_x_forest'}
SEASONAL     = {'month','season','is_spring','is_autumn','risk_season'}   # 제거
EXCLUDE      = {'pole_id','date','label','label_arc','label_mech',
                'label_dry','label_heat','nearest_station'} | SEASONAL
FEAT = [c for c in tf.columns
        if c not in EXCLUDE and c not in FIRE_HIST
        and tf[c].dtype in [np.float64,np.float32,np.int64,np.int32,np.int8]]

print(f"Feature 수: {len(FEAT)}개 (계절 컬럼 {len(SEASONAL)}개 제거됨)")

dates  = tf['date'].sort_values().unique()
cutoff = dates[int(len(dates) * 0.8)]
tr = tf['date'] < cutoff
te = tf['date'] >= cutoff
X  = tf[FEAT].fillna(tf[FEAT].median())
X_tr, X_te = X[tr], X[te]

weather_kw = ['score','hazard','humidity','fwi','accumulated','precip',
              'temp','rh','ws','dry','hot','heat','warn','watch','flag','wind']
results = {}; models = {}

for lbl, name in [('label_dry','건조형'), ('label_heat','고온형')]:
    y   = tf[lbl].astype(int)
    spw = int((y == 0).sum() / max(y.sum(), 1))
    print(f"\n  [{name}] pos={y.sum():,}, spw={spw}")
    m = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        num_leaves=63, random_state=SEED,
        scale_pos_weight=spw, verbose=-1)
    m.fit(X_tr, y[tr])
    prob = m.predict_proba(X_te)[:, 1]
    auc  = roc_auc_score(y[te], prob)
    r10  = y[te].values[np.argsort(prob)[::-1][:int(len(y[te]) * 0.1)]].sum() / max(y[te].sum(), 1)
    shap = pd.DataFrame({'feature': FEAT,
                          'gain': m.booster_.feature_importance('gain')
                         }).sort_values('gain', ascending=False)
    total = shap['gain'].sum()
    w_pct = shap[shap['feature'].apply(
        lambda f: any(k in f for k in weather_kw))]['gain'].sum() / total * 100
    # month 관련 피처 기여도
    cal_pct = shap[shap['feature'].isin(SEASONAL)]['gain'].sum() / total * 100
    print(f"  AUC={auc:.4f}  R@10%={r10:.3f}  날씨기여={w_pct:.1f}%  달력기여={cal_pct:.1f}%")
    print("  Top-5:")
    for _, r in shap.head(5).iterrows():
        tag = '🌤' if any(k in r['feature'] for k in weather_kw) else '🗺'
        print(f"    {tag} {r['feature']:30s}: {r['gain']/total*100:.1f}%")
    results[name] = {'auc': auc, 'recall_top10': r10, 'weather_pct': w_pct}
    models[lbl]   = {'model': m, 'feature_cols': FEAT}

with open(DATA_PROCESSED / 'model_dry.pkl',  'wb') as f: pickle.dump(models['label_dry'],  f)
with open(DATA_PROCESSED / 'model_heat.pkl', 'wb') as f: pickle.dump(models['label_heat'], f)
pd.DataFrame(results).T.to_csv(OUT_TABLES / 'multihazard_model_performance.csv')
print("\n✅ model_dry.pkl, model_heat.pkl 저장")

# ══════════════════════════════════════════════════════
# risk_scores 업데이트 (A + B 통합)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("risk_scores 업데이트 (A + B 반영)")
print("=" * 55)
rs = pd.read_parquet(DATA_PROCESSED / 'risk_scores.parquet')
meds = {c: tf[c].median() for c in FEAT if c in tf.columns}

for lbl_key in ['label_dry', 'label_heat']:
    fc  = models[lbl_key]['feature_cols']
    X_rs = pd.DataFrame({
        c: rs[c].fillna(meds.get(c, 0)) if c in rs.columns else meds.get(c, 0)
        for c in fc
    })
    col = 'prob_dry' if lbl_key == 'label_dry' else 'prob_heat'
    rs[col] = models[lbl_key]['model'].predict_proba(X_rs)[:, 1]

# 낙뢰 v2 반영
rs = rs.drop(columns=[c for c in ['lightning_risk_rule','prob_light','hazard_combined',
                                    'final_risk_wpfi'] if c in rs.columns])
rs = rs.merge(lightning_out, on='pole_id', how='left')
rs['prob_light']      = rs['lightning_risk_rule'].fillna(0) / 100
rs['hazard_combined'] = (
    0.50 * rs['prob_dry'] +
    0.30 * rs['prob_heat'] +
    0.20 * rs['prob_light']
).clip(0, 1)
rs['final_risk_wpfi'] = (rs['hazard_combined'] * 100).clip(0, 100)
rs.to_parquet(DATA_PROCESSED / 'risk_scores.parquet', index=False)

# ══════════════════════════════════════════════════════
# 최종 결과 출력
# ══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("최종 결과")
print("=" * 55)

# 낙뢰 상관계수 재확인
rs2 = pd.read_parquet(DATA_PROCESSED / 'risk_scores.parquet',
                       columns=['pole_id','prob_light'])
merged2 = buf.merge(rs2, on='pole_id', how='left')
corr_final, _ = pearsonr(merged2['elevation_m'].fillna(0),
                          merged2['prob_light'].fillna(0))
print(f"\n[낙뢰 v2] prob_light vs elevation 상관계수: {corr_final:.4f}")
print(f"  (기존 0.9991 → {corr_final:.4f}, 낮을수록 개선)")

print(f"\n[모델 성능 비교]")
print(f"{'모델':12s} {'AUC':>7} {'R@10%':>7} {'날씨기여':>8}")
for name, r in results.items():
    print(f"  {name:12s} {r['auc']:>7.4f} {r['recall_top10']:>7.3f} {r['weather_pct']:>7.1f}%")

print(f"\n[WPFI_v2 통계]")
print(f"  mean={rs['hazard_combined'].mean():.4f}  max={rs['hazard_combined'].max():.4f}")
print("\n✅ 방안 A + B 적용 완료")
