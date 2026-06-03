"""
날씨마루 Hive 데이터 탐색 스크립트
이 파일을 날씨마루 Jupyter 노트북에 복붙해서 실행하세요.
결과를 캡처하거나 CSV로 저장해서 공유해주세요.
"""

from pyhive import hive
import pandas as pd
import json

# ── 연결 ────────────────────────────────────────────────────────────────────
conn = hive.Connection(
    host="localhost",   # 날씨마루 내부에서는 localhost 또는 별도 호스트
    port=10000,
    auth="NOSASL",      # 내부 접속은 인증 없을 수 있음
)
cur = conn.cursor()

print("=" * 60)
print("STEP 1. 데이터베이스 목록")
print("=" * 60)
cur.execute("SHOW DATABASES")
dbs = [r[0] for r in cur.fetchall()]
for db in dbs:
    print(f"  {db}")

print("\n" + "=" * 60)
print("STEP 2. 전체 테이블 목록")
print("=" * 60)
all_tables = {}
for db in dbs:
    try:
        cur.execute(f"USE {db}")
        cur.execute("SHOW TABLES")
        tables = [r[0] for r in cur.fetchall()]
        all_tables[db] = tables
        if tables:
            print(f"\n  [{db}]")
            for t in tables:
                print(f"    - {t}")
    except Exception as e:
        print(f"  [{db}] 접근 오류: {e}")

print("\n" + "=" * 60)
print("STEP 3. 주요 키워드 테이블 찾기")
print("=" * 60)
keywords = ['fire', 'hwa', '화재', 'weather', 'obs', 'asos', 'aws',
            'kepco', 'facility', 'pole', 'power', 'elec']
found = {}
for db, tables in all_tables.items():
    for t in tables:
        for kw in keywords:
            if kw.lower() in t.lower():
                found.setdefault(kw, []).append(f"{db}.{t}")

for kw, hits in found.items():
    print(f"  [{kw}]: {hits}")

print("\n" + "=" * 60)
print("STEP 4. 기상 테이블 컬럼 확인 (후보 테이블명 수정 필요)")
print("=" * 60)
# 실제 기상 테이블명으로 변경
weather_table_candidates = [
    t for db, tables in all_tables.items()
    for t in tables
    if any(k in t.lower() for k in ['obs', 'asos', 'aws', 'sfc', 'weather', 'kma'])
]
print(f"  기상 테이블 후보: {weather_table_candidates[:5]}")

for tbl in weather_table_candidates[:3]:
    try:
        cur.execute(f"DESCRIBE {tbl}")
        cols = cur.fetchall()
        print(f"\n  {tbl}:")
        for c in cols:
            print(f"    {c[0]:30s} {c[1]}")
        # 샘플 데이터
        cur.execute(f"SELECT * FROM {tbl} LIMIT 3")
        rows = cur.fetchall()
        print(f"  샘플 ({len(rows)}행): {rows[0] if rows else '없음'}")
    except Exception as e:
        print(f"  {tbl} 오류: {e}")

print("\n" + "=" * 60)
print("STEP 5. 화재 테이블 컬럼 확인")
print("=" * 60)
fire_candidates = [
    t for db, tables in all_tables.items()
    for t in tables
    if any(k in t.lower() for k in ['fire', 'hwa', '화재', 'incident'])
]
print(f"  화재 테이블 후보: {fire_candidates}")

for tbl in fire_candidates[:3]:
    try:
        cur.execute(f"DESCRIBE {tbl}")
        cols = cur.fetchall()
        print(f"\n  {tbl}:")
        for c in cols:
            print(f"    {c[0]:30s} {c[1]}")
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        cnt = cur.fetchone()[0]
        print(f"  총 행 수: {cnt:,}")
        cur.execute(f"SELECT * FROM {tbl} LIMIT 3")
        rows = cur.fetchall()
        if rows:
            print(f"  샘플: {rows[0]}")
    except Exception as e:
        print(f"  {tbl} 오류: {e}")

print("\n" + "=" * 60)
print("STEP 6. 설비 테이블 컬럼 확인")
print("=" * 60)
facility_candidates = [
    t for db, tables in all_tables.items()
    for t in tables
    if any(k in t.lower() for k in ['kepco', 'facility', 'pole', 'power', 'elec', 'hanjeon'])
]
print(f"  설비 테이블 후보: {facility_candidates}")

for tbl in facility_candidates[:3]:
    try:
        cur.execute(f"DESCRIBE {tbl}")
        cols = cur.fetchall()
        print(f"\n  {tbl}:")
        for c in cols:
            print(f"    {c[0]:30s} {c[1]}")
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        cnt = cur.fetchone()[0]
        print(f"  총 행 수: {cnt:,}")
        cur.execute(f"SELECT * FROM {tbl} LIMIT 2")
        rows = cur.fetchall()
        if rows:
            print(f"  샘플: {rows[0]}")
    except Exception as e:
        print(f"  {tbl} 오류: {e}")

conn.close()
print("\n\n✅ 탐색 완료 — 위 결과를 캡처해서 공유해주세요!")
