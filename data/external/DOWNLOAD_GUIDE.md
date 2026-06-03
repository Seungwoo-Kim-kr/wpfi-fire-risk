# 수동 다운로드 필요 데이터

## ① 소방청 화재발생현황 → `fire_history_sos.csv`

1. https://www.data.go.kr/data/15044003/fileData.do 접속
2. 페이지 하단 **파일 목록** → CSV 다운로드 클릭 (로그인 불필요)
3. 다운로드 파일을 `data/external/fire_history_sos.csv` 로 저장

**필요 컬럼**: 화재발생년월일, 주소(시도/시군구), 발화요인(전기적), 위도, 경도

---

## ② 산림청 산불발생통계 → `fire_history.csv`

1. https://www.data.go.kr/data/15121380/fileData.do 접속
2. **파일 다운로드** 클릭 (로그인 불필요)
3. `data/external/fire_history.csv` 로 저장

**필요 컬럼**: 발생일시, 발생지역(행정구역), 발생원인, 위도, 경도

---

## ③ 기상특보 이력 → `weather_warning.csv` (선택)

> ⚠️ 기상 데이터에서 직접 파생 가능하므로 우선순위 낮음

1. https://data.kma.go.kr/data/weatherReport/wsrList.do?pgmNo=647 접속
2. 기간: 2020-01-01 ~ 현재
3. CSV 다운로드
4. `data/external/weather_warning.csv` 로 저장

**필요 컬럼**: 발표시각, 지역, 특보종류(건조/강풍)

---

## ④ 산림청 임상도 → `forest.gpkg`

1. https://map.forest.go.kr/ 접속 → 산림공간정보 서비스
2. 임상도 레이어 다운로드 신청 (1~2일 소요)
3. SHP → GeoPackage 변환 후 `data/external/forest.gpkg` 저장

변환 명령:
```bash
.venv/bin/python -c "
import geopandas as gpd
gdf = gpd.read_file('임상도.shp', encoding='euc-kr')
gdf.to_file('data/external/forest.gpkg', driver='GPKG')
"
```

---

## ✅ 자동 수집 완료된 데이터

| 파일 | 크기 | 출처 |
|------|------|------|
| `dem.tif` | 767MB | Copernicus DEM 30m |
| `admin_boundary_sido.geojson` | - | KOSTAT/southkorea-maps |
| `admin_boundary_sigungu.geojson` | - | KOSTAT/southkorea-maps |
