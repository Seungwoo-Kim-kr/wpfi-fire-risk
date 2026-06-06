"""WPFI_v2 최종 시각화 재생성"""
import pandas as pd, numpy as np, pickle, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')
from pathlib import Path
DATA_PROCESSED = Path('data/processed')
OUT_FIGURES    = Path('outputs/figures')
OUT_TABLES     = Path('outputs/tables')

KO = next((f.name for f in fm.fontManager.ttflist
           if f.name in ['Apple SD Gothic Neo','AppleGothic']), 'DejaVu Sans')
matplotlib.rcParams['font.family'] = KO
matplotlib.rcParams['axes.unicode_minus'] = False

with open(DATA_PROCESSED/'model_dry.pkl',  'rb') as f: m_dry  = pickle.load(f)
with open(DATA_PROCESSED/'model_heat.pkl', 'rb') as f: m_heat = pickle.load(f)
tf = pd.read_parquet(DATA_PROCESSED/'train_features.parquet')
rs = pd.read_parquet(DATA_PROCESSED/'risk_scores.parquet')
perf = pd.read_csv(OUT_TABLES/'multihazard_model_performance.csv', index_col=0)

weather_kw = ['score','hazard','humidity','fwi','accumulated','precip',
              'temp','rh','ws','dry','hot','heat','warn','watch','flag','wind']

# ── Figure 1: SHAP Feature Importance ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, model_data, name, color, label_col in [
    (axes[0], m_dry,  f'건조형 모델\nAUC={perf.loc["건조형","auc"]:.3f}  날씨기여={perf.loc["건조형","weather_pct"]:.1f}%', '#1f77b4', 'label_dry'),
    (axes[1], m_heat, f'고온형 모델\nAUC={perf.loc["고온형","auc"]:.3f}  날씨기여={perf.loc["고온형","weather_pct"]:.1f}%', '#d62728', 'label_heat'),
]:
    feat = model_data['feature_cols']
    gain = model_data['model'].booster_.feature_importance('gain')
    df_g = pd.DataFrame({'feature':feat,'gain':gain}).sort_values('gain',ascending=False)
    total = df_g['gain'].sum(); top10 = df_g.head(10)
    bc = [color if any(k in f for k in weather_kw) else '#aec6cf' for f in top10['feature']]
    ax.barh(top10['feature'][::-1], top10['gain'][::-1]/total*100, color=bc[::-1])
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature Importance (Gain %)')
    ax.grid(axis='x', alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=color,label='날씨'),Patch(color='#aec6cf',label='공간')],
              fontsize=8, loc='lower right')

axes[2].text(0.5, 0.65, '⚡ 낙뢰 위험\n(Rule-based)', ha='center', va='center',
             fontsize=14, fontweight='bold', color='#2171b5', transform=axes[2].transAxes)
axes[2].text(0.5, 0.42, '고도 기반 정규화\n태백/인제/평창 고위험',
             ha='center', va='center', fontsize=10, color='#555', transform=axes[2].transAxes)
axes[2].text(0.5, 0.22, 'WPFI_v2\n= 0.5×P_dry + 0.3×P_heat + 0.2×P_light',
             ha='center', va='center', fontsize=9, color='#333', transform=axes[2].transAxes)
axes[2].set_title('낙뢰/역섬락\n(고도 proxy)', fontsize=11, fontweight='bold')
axes[2].axis('off')

plt.suptitle('WPFI_v2 멀티해저드 모델 — Feature Importance (날씨기여 91%+)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_FIGURES/'fig_multihazard_shap.png', dpi=150, bbox_inches='tight')
plt.close(); print('fig_multihazard_shap.png 저장')

# ── Figure 2: 날씨 조건 비교 ──────────────────────────────────────────
weather_cols = ['dryness_score','fwi_score','heat_score','wind_score','eff_humidity','accumulated_risk_norm']
col_labels   = ['건조도','FWI','열지수','풍속','유효습도','누적위험']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, lbl, name, color in [
    (axes[0], 'label_dry',  '건조형 화재 조건', '#1f77b4'),
    (axes[1], 'label_heat', '고온형 화재 조건', '#d62728'),
]:
    pos = tf[tf[lbl]==1]; neg = tf[tf[lbl]==0]
    m1 = [pos[c].mean() for c in weather_cols]
    m0 = [neg[c].mean() for c in weather_cols]
    x  = np.arange(len(weather_cols))
    ax.bar(x-0.18, m0, 0.35, label='비화재 조건', color='#aec6cf', alpha=0.85)
    ax.bar(x+0.18, m1, 0.35, label='화재 조건',   color=color,    alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(col_labels, rotation=20, ha='right')
    ax.set_title(f'{name}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    # 차이 표시
    for i, (a, b) in enumerate(zip(m0, m1)):
        diff = (b-a)/max(abs(a),0.001)*100
        ax.text(i+0.18, b*1.02, f'{diff:+.0f}%', ha='center', va='bottom', fontsize=7,
                color='darkred' if diff > 0 else 'darkblue', fontweight='bold')

plt.suptitle('WPFI_v2 날씨 조건 분리 — 건조형(FWI+118%) vs 고온형(열지수+71%)', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_FIGURES/'fig_multihazard_weather.png', dpi=150, bbox_inches='tight')
plt.close(); print('fig_multihazard_weather.png 저장')

# ── Figure 3: WPFI_v2 구성요소 분포 ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, title, color in [
    (axes[0], 'prob_dry',  f'P_dry (건조형)\n평균={rs["prob_dry"].mean():.4f}',  '#1f77b4'),
    (axes[1], 'prob_heat', f'P_heat (고온형)\n평균={rs["prob_heat"].mean():.4f}', '#d62728'),
    (axes[2], 'prob_light',f'P_light (낙뢰)\n평균={rs["prob_light"].mean():.4f}', '#2ca02c'),
]:
    v = rs[col].clip(0, rs[col].quantile(0.99))
    ax.hist(v, bins=50, color=color, alpha=0.75, edgecolor='white', linewidth=0.3)
    ax.axvline(v.mean(), color='black', linestyle='--', linewidth=1.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('위험 확률'); ax.grid(alpha=0.3)

plt.suptitle('WPFI_v2 = 0.5×P_dry + 0.3×P_heat + 0.2×P_light — 전봇대별 분포', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_FIGURES/'fig_multihazard_distribution.png', dpi=150, bbox_inches='tight')
plt.close(); print('fig_multihazard_distribution.png 저장')

print('\n모든 시각화 완료')
