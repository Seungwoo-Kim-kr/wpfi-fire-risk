"""
산림청 산불발생통계 수집 스크립트
---------------------------------
공공데이터포털(data.go.kr)에서 직접 다운로드:
  https://www.data.go.kr/data/15121380/fileData.do
  → '파일 다운로드' 버튼 클릭 → CSV 저장 → fire_history.csv 로 이름 변경

또는 API 키 발급 후 아래 코드로 자동 수집 가능.
"""
import requests, pandas as pd
from pathlib import Path

SAVE_DIR = Path(__file__).parent

# ── 방법 A: 공공데이터포털 API (키 필요) ──────────────────────────────────
API_KEY = ""   # data.go.kr 에서 발급 후 입력

def fetch_via_api(api_key, num_rows=1000):
    url = "https://apis.data.go.kr/1400377/forestFire/forestFireList"
    params = {
        "serviceKey": api_key,
        "numOfRows":  num_rows,
        "pageNo":     1,
        "type":       "json",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    return pd.DataFrame(items)

# ── 방법 B: 수동 다운로드 후 경로 지정 ───────────────────────────────────
def load_manual_csv(path="fire_history.csv"):
    return pd.read_csv(SAVE_DIR / path, encoding="utf-8-sig")

# ── 실행 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if API_KEY:
        df = fetch_via_api(API_KEY)
        df.to_csv(SAVE_DIR / "fire_history.csv", index=False, encoding="utf-8-sig")
        print(f"API 수집 완료: {len(df)}건 → fire_history.csv")
    else:
        print("수동 다운로드 방법:")
        print("  1. https://www.data.go.kr/data/15121380/fileData.do 접속")
        print("  2. '파일 다운로드' 버튼 클릭")
        print("  3. 다운로드한 CSV → data/external/fire_history.csv 로 저장")
        print()
        print("또는 API_KEY 변수에 data.go.kr 발급 키를 입력하세요.")
