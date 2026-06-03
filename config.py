"""
WPFI Project Configuration
Explainable GeoAI Model for Power Facility Fire-Risk Prioritisation
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── API Keys (from .env) ───────────────────────────────────────────────────────
KMA_API_KEY    = os.getenv("KMA_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
DATA_EXTERNAL   = ROOT / "data" / "external"
OUT_MAPS        = ROOT / "outputs" / "maps"
OUT_FIGURES     = ROOT / "outputs" / "figures"
OUT_TABLES      = ROOT / "outputs" / "tables"
REPORTS         = ROOT / "reports"

# ── 날씨마루 Hive 연결 정보 ────────────────────────────────────────────────────
# 날씨마루 접속 후 실제 값으로 교체
HIVE_HOST     = "hive.bd.kma.go.kr"   # 실제 호스트로 교체
HIVE_PORT     = 10000
HIVE_DATABASE = "default"             # 실제 DB명으로 교체
HIVE_USERNAME = "fo.swkim@gmail.com"  # 날씨마루 계정
HIVE_PASSWORD = "Tmddnt606!"          # 날씨마루 비밀번호

# 날씨마루 Hive 테이블명 (접속 후 실제 테이블명으로 교체)
HIVE_TABLES = {
    "weather":  "db_sfc_obs_day",     # 지상 기상 일자료 (추정)
    "facility": "kepco_facility",     # 전력설비 데이터 (추정)
    "fire":     "fire_event",         # 화재 발생 데이터 (추정)
}

# ── 외부 공공데이터 활용 설정 ──────────────────────────────────────────────────
# 대회 "기타 데이터" 페이지에서 공식 링크된 포털 → 사용 가능 확정
ALLOW_EXTERNAL_SPATIAL = True

# 외부 데이터 소스 URL (참고용)
EXTERNAL_SOURCES = {
    "spatial":       "https://www.nsdi.go.kr",             # 국가공간정보포털 (임상도/DEM)
    "forest_map":    "https://map.forest.go.kr",           # 산림공간정보서비스 (임상도 신청)
    "forest_trade":  "https://www.kofpi.or.kr",            # 산림빅데이터거래소
    "disaster":      "https://www.safetydata.go.kr",       # 재난안전데이터포털
    "data_go":       "https://www.data.go.kr",             # 공공데이터포털
    "sgis":          "https://sgis.kostat.go.kr",          # 통계지리정보서비스
    "fire_station":  "https://www.nfds.go.kr",             # 소방청 국가화재정보시스템
    "weather_warn":  "https://www.data.go.kr",             # 기상청 기상특보
}

# ── 외부 데이터 파일 경로 ───────────────────────────────────────────────────
EXTERNAL_FILES = {
    # L1 — 핵심 feature 생성용
    "forest_gpkg":        DATA_EXTERNAL / "forest.gpkg",          # 임상도 (산림청, 신청 필요)
    "worldcover_tif":     DATA_EXTERNAL / "worldcover_korea.tif", # ESA WorldCover 10m (임상도 대체)
    "dem_tif":            DATA_EXTERNAL / "dem.tif",               # 수치표고모델 5m
    "fire_history_csv":          DATA_EXTERNAL / "fire_history.csv",              # 산불발생통계 (산림청)
    "fire_history_geocoded":     DATA_EXTERNAL / "fire_history_geocoded.csv",     # 산불통계 + 위경도
    "fire_sos_geocoded":         DATA_EXTERNAL / "fire_history_sos_geocoded.csv", # 소방청 전체 + 위경도
    "fire_elec_geocoded":        DATA_EXTERNAL / "fire_history_elec_geocoded.csv",# 소방청 전기화재만
    "fire_sos_csv":       DATA_EXTERNAL / "fire_history_sos.csv",  # 소방청 화재통계
    "admin_sido":         DATA_EXTERNAL / "admin_boundary_sido.geojson",
    "admin_sigungu":      DATA_EXTERNAL / "admin_boundary_sigungu.geojson",
    # L2 — 검증 및 보조용
    "weather_warn_csv":   DATA_EXTERNAL / "weather_warning.csv",   # 기상특보 이력
    "landuse_gpkg":       DATA_EXTERNAL / "landuse.gpkg",          # 토지이용현황도
}

# ── CRS ────────────────────────────────────────────────────────────────────────
CRS_GEO     = "EPSG:4326"    # WGS84 — 지도 시각화용
CRS_PROJ    = "EPSG:5179"    # Korea 2000 / UTM-K — 거리·buffer 계산용

# ── Buffer radii (metres) ──────────────────────────────────────────────────────
BUFFER_RADII = [250, 500, 1000, 3000]

# ── Risk Index weights (Base scenario) ────────────────────────────────────────
# 민감도 분석에서 시나리오별로 교체하여 사용
WEIGHTS = {
    # slope/임상도 확정으로 Spatial 0.25→0.30 격상, Weather 0.40→0.35 조정
    "base":             {"weather": 0.35, "spatial": 0.30, "facility": 0.25, "prior": 0.10},
    "weather_heavy":    {"weather": 0.50, "spatial": 0.20, "facility": 0.20, "prior": 0.10},
    "spatial_heavy":    {"weather": 0.25, "spatial": 0.45, "facility": 0.20, "prior": 0.10},
    "facility_heavy":   {"weather": 0.30, "spatial": 0.25, "facility": 0.35, "prior": 0.10},
    "no_prior":         {"weather": 0.40, "spatial": 0.35, "facility": 0.25, "prior": 0.00},
    "equal":            {"weather": 0.30, "spatial": 0.30, "facility": 0.30, "prior": 0.10},
}

# ── Risk grade thresholds ──────────────────────────────────────────────────────
# 고정 임계값 방식 (분위수 방식과 비교하여 최종 선택)
GRADE_THRESHOLDS = {
    "Very High": (86, 100),
    "High":      (66,  85),
    "Moderate":  (51,  65),
    "Low":       ( 0,  50),
}

# ── Top-K inspection targets ───────────────────────────────────────────────────
TOP_K_PCTS = [0.05, 0.10, 0.20]   # 상위 5% / 10% / 20%

# ── Weather feature windows (days) ────────────────────────────────────────────
ROLL_WINDOWS = [1, 3, 7, 14]

# ── Interaction feature pairs ──────────────────────────────────────────────────
# (score_A, score_B) → A × B / 100
INTERACTION_PAIRS = [
    ("dryness_score", "wind_score"),      # dry_wind_interaction
    ("heat_score",    "dryness_score"),   # hot_dry_interaction
    ("no_rain_score", "wind_score"),      # no_rain_wind_interaction
]

# ── Spatial matching method ────────────────────────────────────────────────────
SPATIAL_MATCH_METHOD = "nearest"   # "nearest" | "idw"
IDW_POWER = 2                       # inverse distance weighting exponent

# ── Model mode (auto-detected in 06_modeling) ─────────────────────────────────
# "supervised"  — label 존재 시 XGBoost / LightGBM
# "unsupervised" — label 없음, risk prioritisation + clustering
# "supervised"  — label 존재 시 (화재 발생 데이터 제공 확인됨 — FAQ Q.49)
# "unsupervised" — label 없을 경우 fallback
MODEL_MODE = "supervised"

# ── Random seed ───────────────────────────────────────────────────────────────
SEED = 42
