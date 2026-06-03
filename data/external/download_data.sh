#!/bin/bash
# 외부 공공데이터 수집 스크립트
# 실행: bash /Users/seungwookim/Desktop/wpfi_project/data/external/download_data.sh

BASE_DIR="/Users/seungwookim/Desktop/wpfi_project/data/external"
cd "$BASE_DIR"

echo "====================================================="
echo "공공데이터 수집 시작: $(date)"
echo "====================================================="

# -----------------------------------------------------------
# 1. 행정경계 GeoJSON - 시도 (southkorea-maps GitHub)
# -----------------------------------------------------------
echo ""
echo "[1/3] 시도 행정경계 GeoJSON 다운로드..."
curl -L \
  "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_provinces_geo.json" \
  -o "$BASE_DIR/admin_boundary_sido.geojson" \
  --max-time 120 \
  -w "완료: HTTP %{http_code}, 크기 %{size_download} bytes\n"

# -----------------------------------------------------------
# 2. 행정경계 GeoJSON - 시군구 (southkorea-maps GitHub)
# -----------------------------------------------------------
echo ""
echo "[2/3] 시군구 행정경계 GeoJSON 다운로드..."
curl -L \
  "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_municipalities_geo.json" \
  -o "$BASE_DIR/admin_boundary_sigungu.geojson" \
  --max-time 120 \
  -w "완료: HTTP %{http_code}, 크기 %{size_download} bytes\n"

# -----------------------------------------------------------
# 3. 산불통계 CSV - 공공데이터포털 파일데이터
# 주의: 로그인 없이 직접 다운로드 시도 (실패 시 수동 다운로드 필요)
# -----------------------------------------------------------
echo ""
echo "[3/3] 산불통계 CSV 다운로드 시도..."
# 공공데이터포털은 로그인 필요할 수 있으므로 forest.go.kr 직접 시도
curl -L \
  "https://fd.forest.go.kr/ffas/pubConn/movePage/sub3.do" \
  -o "$BASE_DIR/fire_history_check.html" \
  --max-time 30 \
  -w "HTTP %{http_code}\n" 2>&1 | head -5

echo ""
echo "====================================================="
echo "다운로드 완료. 파일 목록:"
ls -lh "$BASE_DIR"
echo "====================================================="
echo ""
echo "산불 CSV 수동 다운로드 방법:"
echo "  1. https://www.data.go.kr/data/15121380/fileData.do 접속"
echo "  2. 로그인 후 '다운로드' 버튼 클릭"
echo "  3. 다운로드된 파일을 $BASE_DIR/fire_history.csv 로 이름 변경"
