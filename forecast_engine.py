"""
WPFI 기상 예보 엔진 v3
apihub.kma.go.kr 단기예보 격자 API → feature 계산 → LightGBM 추론
- 격자 API(nph-dfs_shrt_grd): 전국 격자 동시 반환 → 12개소 한번에 추출
- ThreadPoolExecutor 병렬 호출로 속도 단축
"""

import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── 강원도 12개 관측소 격자 좌표 (nx, ny) ────────────────────────────────────
STATION_GRID = {
    "춘천": {"nx": 73,  "ny": 134},
    "홍천": {"nx": 75,  "ny": 130},
    "인제": {"nx": 80,  "ny": 138},
    "원주": {"nx": 76,  "ny": 122},
    "횡성": {"nx": 77,  "ny": 126},
    "강릉": {"nx": 92,  "ny": 131},
    "양양": {"nx": 88,  "ny": 138},
    "정선": {"nx": 84,  "ny": 126},
    "동해": {"nx": 97,  "ny": 127},
    "삼척": {"nx": 98,  "ny": 123},
    "태백": {"nx": 95,  "ny": 119},
    "속초": {"nx": 87,  "ny": 141},
}

# KMA 단기예보 격자 크기 (한반도 전역)
GRID_NX, GRID_NY = 149, 253
GRID_TOTAL = GRID_NX * GRID_NY  # 37,697

# 예보 변수 목록
FCST_VARS = ["TMP", "REH", "WSD", "PCP", "PTY"]

# 학습 데이터 기반 percentile 경계
PCTL = {
    "temp_max": {"lo": 4.10,  "hi": 33.40},
    "rh_min":   {"lo": 14.00, "hi": 59.00},
    "ws_max":   {"lo": 2.00,  "hi": 6.10},
    "no_rain":  {"lo": 0.00,  "hi": 13.00},
    "eff_hum":  {"lo": 42.87, "hi": 86.64},
}

# weather_hazard 가중치 (R²=1.00 역추적)
W_HAZARD = {
    "dryness":   0.3492,
    "wind":      0.3002,
    "heat":      0.2000,
    "no_rain":   0.1504,
    "dry_watch": 0.0850,
    "combined":  0.0243,
}

WEIGHTS = {"weather": 0.35, "spatial": 0.30, "facility": 0.25, "prior": 0.10}

# ── API 설정 ──────────────────────────────────────────────────────────────────
GRID_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd"
_ISSUE_SLOTS = [2, 5, 8, 11, 14, 17, 20, 23]


def _latest_tmfc(now: datetime) -> str:
    """현재 시각 → 최근 발표시각 YYYYMMDDhh"""
    valid = [s for s in _ISSUE_SLOTS if s <= now.hour]
    slot = valid[-1] if valid else 23
    if not valid:
        now = now - timedelta(days=1)
    return now.strftime("%Y%m%d") + f"{slot:02d}"


def _parse_grid(text: str) -> np.ndarray | None:
    """
    격자 API 응답(CSV 텍스트) → numpy 2D 배열 (shape: NY×NX)
    전국 격자: 253행 × 149열 = 37,697셀
    """
    vals = []
    for ln in text.strip().splitlines():
        for v in ln.split(","):
            v = v.strip()
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
    if len(vals) != GRID_TOTAL:
        return None
    return np.array(vals, dtype=np.float32).reshape(GRID_NY, GRID_NX)


def _extract_stations(grid: np.ndarray) -> dict[str, float]:
    """격자에서 12개 관측소 위치값 추출. nx, ny는 1-indexed."""
    result = {}
    for stn, g in STATION_GRID.items():
        ny, nx = g["ny"] - 1, g["nx"] - 1  # 0-indexed
        if 0 <= ny < GRID_NY and 0 <= nx < GRID_NX:
            val = float(grid[ny, nx])
            result[stn] = None if val <= -98 else val
        else:
            result[stn] = None
    return result


# ── 단일 API 호출 (변수 + 시각) ───────────────────────────────────────────────
def _fetch_one(api_key: str, var: str, tmfc: str, tmef: str) -> dict:
    """격자 API 1회 호출 → 관측소별 값 반환"""
    r = requests.get(
        GRID_URL,
        params={"tmfc": tmfc, "tmef": tmef, "vars": var, "authKey": api_key},
        timeout=20,
    )
    r.raise_for_status()
    grid = _parse_grid(r.text)
    if grid is None:
        return {"var": var, "tmef": tmef, "stations": {stn: None for stn in STATION_GRID}}
    return {"var": var, "tmef": tmef, "stations": _extract_stations(grid)}


# ── 3일 예보 수집 (병렬) ─────────────────────────────────────────────────────
def fetch_forecast_grid(api_key: str) -> pd.DataFrame:
    """
    향후 3일 × 3시각(09, 15, 21시) × 5변수 = 45 API 호출
    ThreadPoolExecutor로 병렬 처리 → 약 10~20초
    """
    now    = datetime.now()
    tmfc   = _latest_tmfc(now)

    # 예보 시각: 내일~3일 후 09시, 15시, 21시
    forecast_times = []
    for d in range(1, 4):
        base = (now + timedelta(days=d)).replace(minute=0, second=0, microsecond=0)
        for h in [9, 15, 21]:
            forecast_times.append(base.replace(hour=h))

    tasks = [
        (var, tmfc, ft.strftime("%Y%m%d%H"))
        for ft in forecast_times
        for var in FCST_VARS
    ]

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_fetch_one, api_key, var, tmfc, tmef): (var, tmef)
            for var, tmfc, tmef in tasks
        }
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception:
                var, tmef = futures[fut]
                results.append({
                    "var": var, "tmef": tmef,
                    "stations": {stn: None for stn in STATION_GRID},
                })

    # long 형식으로 변환
    rows = []
    for res in results:
        tmef_dt = datetime.strptime(res["tmef"], "%Y%m%d%H")
        for stn, val in res["stations"].items():
            rows.append({
                "station":   stn,
                "fcst_date": tmef_dt.strftime("%Y%m%d"),
                "fcst_hour": tmef_dt.hour,
                "var":       res["var"],
                "value":     val,
            })
    return pd.DataFrame(rows)


# ── 일별 집계 ─────────────────────────────────────────────────────────────────
def compute_daily_weather(raw: pd.DataFrame) -> pd.DataFrame:
    """시간별 raw → 날짜 × 관측소 일별 집계"""
    pivot = raw.pivot_table(
        index=["station", "fcst_date", "fcst_hour"],
        columns="var", values="value", aggfunc="first",
    ).reset_index()

    defaults = {"TMP": 22.0, "REH": 50.0, "WSD": 3.0, "PCP": 0.0, "PTY": 0.0}
    for col, val in defaults.items():
        if col not in pivot.columns:
            pivot[col] = val
        else:
            pivot[col] = pivot[col].fillna(val)

    # 강수 처리
    if "PCP" in pivot.columns:
        pivot["PCP"] = pd.to_numeric(
            pivot["PCP"].astype(str).replace({"강수없음": "0", "1mm 미만": "0.5"}),
            errors="coerce",
        ).fillna(0)

    daily = (
        pivot.groupby(["station", "fcst_date"])
        .agg(
            temp_max=("TMP", "max"),
            rh_min=("REH", "min"),
            ws_max=("WSD", "max"),
            precip_sum=("PCP", "sum"),
            has_rain=("PTY", lambda x: int((x > 0).any())),
        )
        .reset_index()
    )
    return daily


# ── Feature 계산 ──────────────────────────────────────────────────────────────
def _pct_score(val: float, lo: float, hi: float, invert: bool = False) -> float:
    s = float(np.clip((val - lo) / (hi - lo + 1e-9) * 100, 0, 100))
    return 100 - s if invert else s


def compute_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy().sort_values(["station", "fcst_date"]).reset_index(drop=True)

    df["no_rain_days"] = df.groupby("station")["has_rain"].transform(
        lambda s: s.eq(0).cumsum()
    )
    df["eff_hum"] = df["rh_min"]

    df["dryness_score"]    = df["rh_min"].apply(
        lambda v: _pct_score(v, PCTL["rh_min"]["lo"], PCTL["rh_min"]["hi"], invert=True))
    df["wind_score"]       = df["ws_max"].apply(
        lambda v: _pct_score(v, PCTL["ws_max"]["lo"], PCTL["ws_max"]["hi"]))
    df["heat_score"]       = df["temp_max"].apply(
        lambda v: _pct_score(v, PCTL["temp_max"]["lo"], PCTL["temp_max"]["hi"]))
    df["no_rain_score"]    = df["no_rain_days"].apply(
        lambda v: _pct_score(v, PCTL["no_rain"]["lo"], PCTL["no_rain"]["hi"]))
    df["eff_humidity"]     = df["eff_hum"]
    df["eff_humidity_score"] = df["eff_hum"].apply(
        lambda v: _pct_score(v, PCTL["eff_hum"]["lo"], PCTL["eff_hum"]["hi"], invert=True))

    df["dry_wind_interaction"] = df["dryness_score"] * df["wind_score"] / 100
    df["hot_dry_interaction"]  = df["heat_score"]    * df["dryness_score"] / 100

    df["dry_watch_flag"]     = (df["eff_hum"] <= 35).astype(int)
    df["dry_warn_flag"]      = (df["eff_hum"] <= 25).astype(int)
    df["wind_watch_flag"]    = (df["ws_max"] >= 14).astype(int)
    df["wind_warn_flag"]     = (df["ws_max"] >= 21).astype(int)
    df["combined_risk_flag"] = (
        (df["dry_watch_flag"] == 1) & (df["wind_watch_flag"] == 1)
    ).astype(int)

    df["weather_hazard"] = (
          df["dryness_score"]      * W_HAZARD["dryness"]
        + df["wind_score"]         * W_HAZARD["wind"]
        + df["heat_score"]         * W_HAZARD["heat"]
        + df["no_rain_score"]      * W_HAZARD["no_rain"]
        + df["dry_watch_flag"]     * W_HAZARD["dry_watch"] * 100
        + df["combined_risk_flag"] * W_HAZARD["combined"]  * 100
    ).clip(0, 100)

    df["fcst_date_dt"] = pd.to_datetime(df["fcst_date"], format="%Y%m%d")
    df["month"]      = df["fcst_date_dt"].dt.month
    df["season"]     = df["month"].map(
        {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3})
    df["is_spring"]  = (df["season"] == 1).astype(int)
    df["is_autumn"]  = (df["season"] == 3).astype(int)
    df["risk_season"]= df["season"].isin([1, 3]).astype(int)

    df["temp_max_3d"]   = df.groupby("station")["temp_max"].transform(
        lambda s: s.rolling(3, min_periods=1).max())
    df["rh_min_3d"]     = df.groupby("station")["rh_min"].transform(
        lambda s: s.rolling(3, min_periods=1).min())
    df["ws_max_1d"]     = df["ws_max"]
    df["precip_sum_7d"] = df.groupby("station")["precip_sum"].transform(
        lambda s: s.rolling(7, min_periods=1).sum())

    return df


# ── 정적 feature 로드 ─────────────────────────────────────────────────────────
def load_static_features(data_processed: Path) -> pd.DataFrame:
    cache = data_processed / "static_features_cache.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    want = [
        "pole_id", "nearest_station",
        "nearest_facility_dist_m", "facility_density_500m", "facility_density_1000m",
        "landcover_class", "is_near_forest", "elevation_m",
        "elec_fire_count_1000m", "elec_fire_count_3000m",
        "fire_count_3000m", "fire_count_5000m", "elec_x_forest",
    ]
    tf = pd.read_parquet(data_processed / "train_features.parquet", columns=want)
    static = tf.groupby("pole_id").first().reset_index()
    static.to_parquet(cache, index=False)
    return static


# ── LightGBM 추론 ─────────────────────────────────────────────────────────────
def run_inference(
    weather_df: pd.DataFrame,
    static: pd.DataFrame,
    model_path: Path,
    risk_scores: pd.DataFrame,
) -> pd.DataFrame:
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model    = saved["model"]
    features = saved["features"]

    results = []
    for _, wrow in weather_df.iterrows():
        stn   = wrow["station"]
        poles = static[static["nearest_station"] == stn].copy()
        if poles.empty:
            continue
        for col in weather_df.columns:
            if col not in ("station", "fcst_date", "fcst_date_dt"):
                poles[col] = wrow[col]
        poles["dry_x_forest"] = poles["dryness_score"] * poles["is_near_forest"]
        for f in features:
            if f not in poles.columns:
                poles[f] = 0
        X     = poles[features].astype(float).fillna(0)
        probs = model.predict_proba(X)[:, 1]
        poles["ml_prob"]      = probs
        poles["fcst_date"]    = wrow["fcst_date"]
        poles["fcst_date_dt"] = wrow["fcst_date_dt"]
        results.append(
            poles[["pole_id", "nearest_station", "fcst_date",
                   "fcst_date_dt", "ml_prob", "weather_hazard"]]
        )

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    mn, mx = out["ml_prob"].min(), out["ml_prob"].max()
    out["forecast_risk"] = ((out["ml_prob"] - mn) / (mx - mn + 1e-9) * 100).clip(0, 100)

    available = [c for c in ["pole_id","spatial_exposure","facility_exposure",
                              "hist_prior","final_risk","risk_grade"]
                 if c in risk_scores.columns]
    out = out.merge(risk_scores[available], on="pole_id", how="left")

    def grade(s):
        if s >= 80: return "Very High"
        if s >= 60: return "High"
        if s >= 40: return "Moderate"
        return "Low"

    out["forecast_grade"] = out["forecast_risk"].apply(grade)
    return out


# ── 메인 파이프라인 ────────────────────────────────────────────────────────────
def run_full_forecast(api_key: str, data_processed: Path, risk_scores: pd.DataFrame):
    """
    전체 파이프라인:
    격자 API 병렬 수집 → feature 계산 → LightGBM 추론
    예상 소요: 15~25초
    """
    raw     = fetch_forecast_grid(api_key)
    daily   = compute_daily_weather(raw)
    feat_df = compute_features(daily)
    static  = load_static_features(data_processed)
    result  = run_inference(feat_df, static, data_processed / "lgbm_model.pkl", risk_scores)
    return result, feat_df
