"""
WPFI 프로젝트 종합 문서 PDF 생성기
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── 폰트 등록 ────────────────────────────────────────────────────────────────
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"

pdfmetrics.registerFont(TTFont("Korean", FONT_PATH))
pdfmetrics.registerFont(TTFont("KoreanB", FONT_BOLD_PATH))

# ── 색상 팔레트 ───────────────────────────────────────────────────────────────
RED      = colors.HexColor("#C0392B")
DARKRED  = colors.HexColor("#922B21")
ORANGE   = colors.HexColor("#E67E22")
BLUE     = colors.HexColor("#1A5276")
LBLUE    = colors.HexColor("#2E86C1")
GREEN    = colors.HexColor("#1E8449")
GRAY     = colors.HexColor("#566573")
LGRAY    = colors.HexColor("#F2F3F4")
DGRAY    = colors.HexColor("#2C3E50")
WHITE    = colors.white
BLACK    = colors.black

# ── 스타일 정의 ───────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    styles = {
        "cover_title": ParagraphStyle("cover_title", fontName="KoreanB",
            fontSize=28, textColor=WHITE, alignment=1, leading=38,
            spaceAfter=12),
        "cover_sub": ParagraphStyle("cover_sub", fontName="Korean",
            fontSize=14, textColor=colors.HexColor("#BDC3C7"),
            alignment=1, leading=22, spaceAfter=6),
        "cover_meta": ParagraphStyle("cover_meta", fontName="Korean",
            fontSize=11, textColor=colors.HexColor("#85929E"),
            alignment=1, leading=18),

        "h1": ParagraphStyle("h1", fontName="KoreanB", fontSize=18,
            textColor=DARKRED, spaceBefore=20, spaceAfter=10,
            borderPad=4, leading=24),
        "h2": ParagraphStyle("h2", fontName="KoreanB", fontSize=14,
            textColor=BLUE, spaceBefore=14, spaceAfter=6, leading=20),
        "h3": ParagraphStyle("h3", fontName="KoreanB", fontSize=12,
            textColor=DGRAY, spaceBefore=10, spaceAfter=4, leading=18),
        "body": ParagraphStyle("body", fontName="Korean", fontSize=10,
            textColor=DGRAY, leading=17, spaceAfter=5),
        "body_small": ParagraphStyle("body_small", fontName="Korean",
            fontSize=9, textColor=GRAY, leading=14, spaceAfter=3),
        "bullet": ParagraphStyle("bullet", fontName="Korean", fontSize=10,
            textColor=DGRAY, leading=16, leftIndent=16,
            firstLineIndent=-10, spaceAfter=3),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8.5,
            textColor=colors.HexColor("#1A1A1A"),
            backColor=colors.HexColor("#F8F9FA"),
            borderPad=6, leading=13, leftIndent=10),
        "highlight": ParagraphStyle("highlight", fontName="Korean",
            fontSize=10, textColor=DARKRED,
            backColor=colors.HexColor("#FDEDEC"),
            borderPad=6, leading=16, spaceAfter=6),
        "note": ParagraphStyle("note", fontName="Korean", fontSize=9,
            textColor=GRAY, backColor=LGRAY,
            borderPad=5, leading=14, leftIndent=8),
        "toc": ParagraphStyle("toc", fontName="Korean", fontSize=10,
            textColor=BLUE, leading=18, leftIndent=8),
        "toc_sub": ParagraphStyle("toc_sub", fontName="Korean", fontSize=9,
            textColor=GRAY, leading=16, leftIndent=20),
    }
    return styles

# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────
def b(text):
    return f"<b>{text}</b>"

def c(text, color="#C0392B"):
    return f'<font color="{color}">{text}</font>'

def table(data, col_widths, header_bg=BLUE, stripe=True):
    t = Table(data, colWidths=col_widths)
    n_rows = len(data)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "KoreanB"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Korean"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
        ("ROUNDEDCORNERS", [4], ),
    ]
    if stripe:
        for i in range(2, n_rows, 2):
            style.append(("BACKGROUND", (0, i), (-1, i), LGRAY))
    t.setStyle(TableStyle(style))
    return t

def section_line(color=RED):
    return HRFlowable(width="100%", thickness=2, color=color,
                      spaceAfter=8, spaceBefore=4)

def divider():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#D5D8DC"),
                      spaceAfter=6, spaceBefore=6)

# ── PDF 생성 ──────────────────────────────────────────────────────────────────
def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
        title="WPFI 프로젝트 종합 문서",
        author="WPFI Team",
        subject="전력설비 화재위험 우선점검 시스템",
    )
    W = A4[0] - 4*cm  # 유효 너비

    S = make_styles()
    story = []

    # ═══════════════════════════════════════════════════════════
    # 표지
    # ═══════════════════════════════════════════════════════════
    cover_data = [[
        Paragraph("WPFI", S["cover_title"]),
    ]]
    cover = Table(cover_data, colWidths=[W])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARKRED),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 40),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 40),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.append(cover)
    story.append(Spacer(1, 0.4*cm))

    for line in [
        Paragraph("Explainable GeoAI 기반 전력설비 화재위험 우선점검 시스템", S["h2"]),
        Paragraph("2026 날씨 빅데이터 콘테스트 — 주제 1", S["body"]),
        Spacer(1, 0.3*cm),
        Paragraph("프로젝트 종합 문서 v1.0", S["body_small"]),
        Paragraph("작성일: 2026년 5월 30일", S["body_small"]),
    ]:
        story.append(line)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 목차
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("목  차", S["h1"]))
    story.append(section_line())
    toc = [
        ("1.", "프로젝트 개요", ""),
        ("2.", "대회 정보 및 규정", ""),
        ("3.", "문제 정의 및 접근 전략", ""),
        ("4.", "데이터셋 전체 목록", ""),
        ("  4.1", "대회 제공 데이터 (날씨마루)", ""),
        ("  4.2", "외부 공공데이터 (수집 완료)", ""),
        ("5.", "전체 파이프라인 아키텍처", ""),
        ("6.", "단계별 상세 설명", ""),
        ("  6.1", "공간 전처리 및 좌표계 통일", ""),
        ("  6.2", "Multi-scale Buffer Feature Engineering", ""),
        ("  6.3", "기상 Feature Engineering", ""),
        ("  6.4", "Risk Index Baseline 산출", ""),
        ("  6.5", "ML 모델링 (Supervised / Unsupervised 분기)", ""),
        ("  6.6", "SHAP Explainability 분석", ""),
        ("  6.7", "검증 전략", ""),
        ("  6.8", "지도 시각화 및 출력", ""),
        ("7.", "Feature 전체 목록 (42개+)", ""),
        ("8.", "기술 스택", ""),
        ("9.", "OpenAI LLM 자연어 설명 모듈", ""),
        ("10.", "Streamlit GUI 앱", ""),
        ("11.", "관련 연구 논문 비교 분석", ""),
        ("12.", "우리 모델의 강점과 한계", ""),
        ("13.", "현재 진행 상태 및 향후 계획", ""),
        ("14.", "APA 7 참고문헌", ""),
    ]
    for num, title, _ in toc:
        indent = S["toc_sub"] if num.startswith("  ") else S["toc"]
        story.append(Paragraph(f"{num}  {title}", indent))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 1. 프로젝트 개요
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("1. 프로젝트 개요", S["h1"]))
    story.append(section_line())
    story.append(Paragraph(
        "WPFI(Weather-Spatial Power Facility Fire-risk Index)는 기상 데이터, 전력설비 위치 정보, "
        "산림·지형 공간 정보, 과거 화재 이력을 통합하여 개별 전력설비의 화재위험도를 "
        "0~100점으로 산정하고, 고위험 설비를 우선점검 대상으로 제시하는 "
        "Explainable GeoAI 기반 의사결정 지원 시스템입니다.", S["body"]))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("핵심 산출물", S["h3"]))
    outputs = [
        ["산출물", "설명"],
        ["전력설비별 위험도 점수", "각 설비의 화재위험을 0~100점으로 정량화"],
        ["위험등급 분류", "Very High / High / Moderate / Low 4단계"],
        ["고위험 설비 Top-K 목록", "상위 5~10% 우선 점검 대상 리스트"],
        ["위험도 지도", "인터랙티브 Folium 기반 지도"],
        ["SHAP 설명 차트", "설비별 '왜 위험한지' 수치 기여도"],
        ["GPT-4o-mini 자연어 설명", "현장 엔지니어용 한국어 설명문"],
    ]
    story.append(table(outputs, [5*cm, 11*cm]))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("모델명", S["h3"]))
    story.append(Paragraph(
        "영문: <b>Explainable GeoAI Model for Power Facility Fire-Risk Prioritisation</b><br/>"
        "한글: <b>Explainable GeoAI 기반 전력설비 화재위험 우선점검 모델</b>", S["body"]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 2. 대회 정보
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("2. 대회 정보 및 규정", S["h1"]))
    story.append(section_line())

    contest_data = [
        ["항목", "내용"],
        ["대회명", "2026 날씨 빅데이터 콘테스트"],
        ["주최", "기상청 (기상청 날씨마루)"],
        ["주제 1", "기상 데이터 및 공간정보 기반 전력설비 인근 화재 위험도 분석"],
        ["참여기관", "한국전력공사 (데이터 제공 및 특별상 후원)"],
        ["참가 접수", "2026년 5월 1일 ~ 5월 31일"],
        ["공모작 제출", "2026년 6월 1일 ~ 6월 26일"],
        ["1차 심사", "2026년 7월 8일 (결과 발표: 7월 13일)"],
        ["최종 시상", "2026년 8월 5일"],
        ["최우수상", "부장관상 + 200만원"],
        ["우수상", "기상청장상 + 150만원"],
        ["특별상 (한전)", "기관장상 + 150만원"],
        ["제출 형식", "분석 보고서 (한글 hwpx, 최대 6페이지) + 코드 ZIP"],
        ["제공 데이터", "기상 데이터 (기상청) + 전력설비 데이터 (한국전력공사)"],
        ["분석 플랫폼", "날씨마루 (기상청 클라우드 Hive 분석 환경)"],
    ]
    story.append(table(contest_data, [4.5*cm, 11.5*cm]))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        "※ 심사 기준: 과제 부합성, 데이터 분석능력, 모델 예측 정확성, 창의성, 합리성, 활용성 (6개 항목)",
        S["note"]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 3. 문제 정의 및 접근 전략
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("3. 문제 정의 및 접근 전략", S["h1"]))
    story.append(section_line())

    story.append(Paragraph("3.1 문제 배경", S["h2"]))
    story.append(Paragraph(
        "2025년 경북·강원 대형 산불 이후 전력설비 인근 화재위험이 사회적 이슈로 부각되었습니다. "
        "전력설비는 건조하고 바람이 강한 기상 조건에서 화재 발생 위험이 높아지며, "
        "특히 산림 인접 설비는 화재 확산 피해가 극심합니다. "
        "그러나 현재 기상청/산림청의 산불위험지수는 지역 단위(격자/행정구역)로만 제공되어 "
        "'어떤 설비를 먼저 점검해야 하는가'라는 현장 의사결정에 직접 활용하기 어렵습니다.",
        S["body"]))

    story.append(Paragraph("3.2 핵심 질문", S["h2"]))
    story.append(Paragraph(
        c("기상이 악화되는 날, 전국 수만 개 전력설비 중 어떤 설비가, 왜, 얼마나 위험한가?",
          "#C0392B"), S["highlight"]))

    story.append(Paragraph("3.3 접근 전략", S["h2"]))
    strategy = [
        ("① 분석 단위", "설비 × 날짜 (facility × date): 설비별 일별 위험도 산정"),
        ("② 4-Layer 구조", "기상위험 + 공간노출 + 설비노출 + 과거화재이력 결합"),
        ("③ Multi-scale Buffer", "설비 중심 250m/500m/1km/3km 반경 공간 특성 추출"),
        ("④ Explainability", "SHAP으로 '왜 위험한지' 수치 근거 제공"),
        ("⑤ 의사결정 연결", "Top-K 우선점검 리스트로 현장 직접 활용 가능"),
    ]
    for num, desc in strategy:
        story.append(Paragraph(f"<b>{num}</b>: {desc}", S["bullet"]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 4. 데이터셋
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("4. 데이터셋 전체 목록", S["h1"]))
    story.append(section_line())

    story.append(Paragraph("4.1 대회 제공 데이터 (날씨마루 Hive — 승인 대기)", S["h2"]))
    core_data = [
        ["데이터", "제공처", "내용", "파이프라인 역할"],
        ["기상 일자료", "기상청", "기온·습도·풍속·강수·실효습도 일별 관측값", "Weather Hazard Score"],
        ["전력설비 데이터", "한국전력공사", "설비 위치·유형·전압·설치연도", "Facility Exposure Score"],
        ["화재 발생 시간자료", "기상청 (FAQ 확인)", "화재 발생 일시·위치 (시간 단위)", "Supervised ML Label"],
    ]
    story.append(table(core_data, [3.5*cm, 3.5*cm, 5.5*cm, 3.5*cm]))

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("4.2 외부 공공데이터 (수집 완료)", S["h2"]))
    ext_data = [
        ["데이터", "출처", "크기", "내용", "파이프라인 역할"],
        ["소방청 화재발생현황", "data.go.kr", "22MB", "191,510건 (2020~2024)\n전기적 원인 49,759건", "전기화재 Prior feature"],
        ["산림청 산불발생통계", "data.go.kr", "191KB", "2,020건 (2022~2024)", "산불 Prior feature"],
        ["Copernicus DEM 30m", "AWS S3 (공개)", "767MB", "한국 전체 수치표고모델", "slope·elevation·aspect"],
        ["ESA WorldCover 10m", "AWS S3 (공개)", "146MB", "토지피복 (산림 비율 71.9%)", "forest_ratio feature"],
        ["행정경계 시도·시군구", "KOSTAT/GitHub", "79MB", "시도 17개, 시군구 251개", "지도 시각화·지오코딩"],
        ["화재 지오코딩 결과", "자체 처리", "-", "시군구→위경도 100% 완료", "공간 매핑"],
    ]
    story.append(table(ext_data, [3.5*cm, 2.8*cm, 1.5*cm, 4.2*cm, 4*cm]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "※ ESA WorldCover는 수종 구분(침엽수/활엽수)이 불가하여 임상도 대체 사용. "
        "임상도(산림청) 신청 시 conifer_ratio feature 추가 가능.",
        S["note"]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 5. 파이프라인 아키텍처
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("5. 전체 파이프라인 아키텍처", S["h1"]))
    story.append(section_line())
    story.append(Paragraph(
        "총 14단계로 구성되며, label 유무에 따라 06단계에서 분기됩니다.", S["body"]))
    story.append(Spacer(1, 0.3*cm))

    pipeline = [
        ["단계", "Notebook", "주요 작업"],
        ["01 데이터 확인", "01_data_check", "Hive 연결, 컬럼 확인, label 유무 판별, column_map.json 생성"],
        ["02 공간 전처리", "02_spatial_preprocessing", "좌표계 통일 (WGS84↔UTM-K), 설비 GeoDataFrame 생성, 기상-설비 공간 매칭"],
        ["03 Buffer Feature", "03_buffer_feature_engineering", "Multi-scale buffer (250m~3km), 산림·지형·전기화재 feature"],
        ["04 기상 Feature", "04_weather_feature_engineering", "Rolling window (1/3/7/14일), 단위 위험 점수, 기상특보 파생 flag 5종"],
        ["05 Risk Index", "05_risk_index", "4-layer 가중합, 0~100 정규화, 4단계 등급, 민감도 분석 5 시나리오"],
        ["06 모델링", "06_modeling", "Case A (Supervised: LightGBM+XGBoost) / Case B (Rule-based+Clustering) 분기"],
        ["07 설명가능성", "07_explainability", "SHAP (TreeExplainer / LinearExplainer / KernelExplainer), Score Decomposition"],
        ["08 검증", "08_validation", "Ablation Study, Sensitivity Analysis, External Validation"],
        ["09 시각화", "09_mapping_outputs", "Folium 위험도 지도, Top-K 설비 지도, 보고서용 Figure"],
        ["10 LLM 설명", "10_llm_explanation", "GPT-4o-mini로 SHAP 수치 → 한국어 자연어 설명 변환"],
        ["GUI", "app.py (Streamlit)", "4탭 인터랙티브 앱 (지도·목록·상세분석·트렌드)"],
    ]
    story.append(table(pipeline, [3.5*cm, 4*cm, 8.5*cm], header_bg=DARKRED))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("▶ Case A/B 분기 기준: 날씨마루 화재 데이터에 label(발생 여부/위치)이 있으면 Case A, 없으면 Case B", S["note"]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 6. 단계별 상세 설명
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("6. 단계별 상세 설명", S["h1"]))
    story.append(section_line())

    # 6.1
    story.append(Paragraph("6.1 공간 전처리 및 좌표계 통일", S["h2"]))
    story.append(Paragraph(
        "기상 데이터, 전력설비 데이터, 외부 GIS 데이터가 서로 다른 좌표계를 사용하므로 통일이 필요합니다.",
        S["body"]))
    crs_data = [
        ["목적", "좌표계", "EPSG"],
        ["지도 시각화", "WGS84", "EPSG:4326"],
        ["거리·면적·Buffer 계산", "Korea 2000 / UTM-K", "EPSG:5179"],
    ]
    story.append(table(crs_data, [5*cm, 6*cm, 5*cm], header_bg=LBLUE))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "기상 관측소와 전력설비를 공간적으로 연결하는 방법 두 가지를 비교합니다:", S["body"]))
    for txt in ["최근접 관측소 매칭 (Nearest Station): 기본 방식, 가장 가까운 관측소 연결",
                "역거리 가중 보간 (IDW): 민감도 분석용, k=5개 관측소 가중 평균"]:
        story.append(Paragraph(f"• {txt}", S["bullet"]))

    # 6.2
    story.append(Paragraph("6.2 Multi-scale Buffer Feature Engineering", S["h2"]))
    story.append(Paragraph(
        "이 파이프라인의 핵심입니다. 전력설비를 중심으로 여러 반경의 buffer를 생성하여 "
        "직접 인접 위험과 지역적 위험을 동시에 반영합니다.", S["body"]))
    buf_data = [
        ["반경", "주요 포착 내용"],
        ["250m", "설비 직접 인접 산림 노출, 즉각적 화재 위험"],
        ["500m", "근거리 산림/지형 노출도, 기상특보 파생 기준"],
        ["1km", "국지적 설비 밀도, 지형 영향권"],
        ["3km", "지역적 화재 이력 Prior, 공간 군집성"],
    ]
    story.append(table(buf_data, [3*cm, 13*cm], header_bg=GREEN))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Buffer 내부에서 계산하는 feature: 산림 비율(WorldCover), 경사도(DEM slope), "
        "고도(DEM elevation), 사면 방향 위험도(aspect risk), "
        "설비 밀도, 최근접 설비 거리, 과거 산불 발생 건수, 과거 전기화재 발생 건수",
        S["body"]))

    story.append(Paragraph("6.3 기상 Feature Engineering", S["h2"]))
    wx_data = [
        ["Feature 그룹", "변수 예시", "의미"],
        ["Rolling Window", "temp_max_3d, rh_min_7d, ws_max_1d", "1/3/7/14일 창 기상 통계"],
        ["단위 위험 점수", "dryness_score, wind_score, heat_score", "0~100 백분위 변환"],
        ["상호작용 Feature", "dry_wind_interaction, hot_dry_interaction", "복합 위험 조합"],
        ["기상특보 파생 Flag", "dry_watch_flag, wind_warn_flag, combined_risk_flag", "기상청 특보 기준 직접 계산"],
        ["누적 지표", "no_rain_days, dry_watch_days, combined_risk_days", "연속 위험 지속 기간"],
    ]
    story.append(table(wx_data, [4.5*cm, 5.5*cm, 6*cm], header_bg=LBLUE))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "기상특보 파생 Flag: 기상청 공식 특보 발령 기준(건조주의보: 실효습도 35% 이하, "
        "강풍주의보: 풍속 14m/s 이상)을 직접 계산하여 실제 특보 데이터 없이도 동일한 정보 생성",
        S["note"]))

    story.append(Paragraph("6.4 Risk Index Baseline 산출", S["h2"]))
    story.append(Paragraph(
        "4개 레이어를 가중합으로 결합합니다. 가중치는 5개 시나리오를 비교하여 안정성을 검증합니다.",
        S["body"]))
    risk_formula = [
        ["레이어", "가중치 (Base)", "주요 변수"],
        ["Weather Hazard", "0.35", "dryness·wind·heat·no_rain score + 특보 flag"],
        ["Spatial Exposure", "0.30", "forest_ratio + slope(★SHAP 1위) + elec_fire_count"],
        ["Facility Exposure", "0.25", "facility_density + nearest_dist + facility_age"],
        ["Historical Prior", "0.10", "fire_count + elec_fire_count (산불+전기화재 6:4)"],
    ]
    story.append(table(risk_formula, [4*cm, 3*cm, 9*cm], header_bg=RED))
    story.append(Spacer(1, 0.3*cm))
    grade_data = [
        ["점수 범위", "등급", "의미", "조치"],
        ["86 ~ 100", "Very High", "매우높음", "긴급 점검 (즉시)"],
        ["66 ~ 85", "High", "높음", "우선 점검 (이번 순회 포함)"],
        ["51 ~ 65", "Moderate", "보통", "모니터링 강화"],
        ["0 ~ 50", "Low", "낮음", "일반 관리"],
    ]
    story.append(table(grade_data, [3*cm, 3*cm, 3*cm, 7*cm], header_bg=DGRAY))
    story.append(PageBreak())

    story.append(Paragraph("6.5 ML 모델링 (Case A / Case B 분기)", S["h2"]))
    story.append(Paragraph(b("Case A — Supervised ML (label 있을 때)"), S["h3"]))
    story.append(Paragraph(
        "날씨마루 화재 데이터에 발생 여부/위치 정보가 있을 때 실행됩니다.", S["body"]))
    ml_data = [
        ["모델", "특징", "역할"],
        ["Logistic Regression", "선형, 해석 용이", "Baseline"],
        ["Random Forest", "비선형 앙상블", "Feature importance 참고"],
        ["LightGBM", "빠른 학습, categorical 처리", "Main 후보"],
        ["XGBoost", "높은 성능, 안정적", "Main 후보"],
    ]
    story.append(table(ml_data, [4*cm, 6*cm, 6*cm], header_bg=GREEN))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("핵심 평가 지표:", S["body"]))
    for txt in [
        "AUC — 전체 순위화 성능",
        "Recall@Top-K — 상위 K% 설비에 실제 위험 설비가 몇 % 포함되는가 (실용적 핵심 지표)",
        "Precision@Top-K — 점검 자원 효율성",
        "Region-based Split — 공간 데이터 과대평가 방지 검증",
    ]:
        story.append(Paragraph(f"• {txt}", S["bullet"]))

    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(b("Case B — Risk Prioritisation (label 없을 때)"), S["h3"]))
    story.append(Paragraph(
        "화재 label이 없을 경우 Rule-based Risk Index + HDBSCAN 클러스터링 + KernelExplainer SHAP을 적용합니다. "
        "모델 표현은 '예측 모델'이 아닌 '위험도 산정 및 우선순위화 모델'로 설명합니다.", S["body"]))

    story.append(Paragraph("6.6 SHAP Explainability 분석", S["h2"]))
    story.append(Paragraph(
        "SHAP(SHapley Additive exPlanations)는 게임 이론의 Shapley value를 머신러닝에 적용한 "
        "수학적으로 증명된 feature 기여도 분해 방법입니다. "
        "모든 SHAP 값의 합 = 예측값 - 기준값이 항상 보장됩니다.", S["body"]))
    shap_data = [
        ["Explainer 종류", "적용 조건", "특징"],
        ["TreeExplainer", "LightGBM·XGBoost·RF (Case A)", "가장 빠르고 정확, 트리 구조 최적화"],
        ["LinearExplainer", "Logistic Regression (Case A)", "선형 모델 전용"],
        ["KernelExplainer", "Rule-based Risk Index (Case B)", "어떤 함수에도 적용 가능, 느림"],
    ]
    story.append(table(shap_data, [4*cm, 5*cm, 7*cm], header_bg=LBLUE))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("SHAP 출력물 두 종류:", S["body"]))
    for txt in [
        "Global Explanation: 전체 모델에서 어떤 feature가 가장 중요한지 (SHAP Summary Plot)",
        "Local Explanation: 특정 설비가 왜 고위험인지 feature별 기여도 (Waterfall Plot)",
    ]:
        story.append(Paragraph(f"• {txt}", S["bullet"]))

    story.append(Paragraph("6.7 검증 전략", S["h2"]))
    val_data = [
        ["검증 방법", "내용"],
        ["Ablation Study", "M1(기상만)~M4(전체) 4개 모델 비교. GeoAI 레이어 기여 수치 검증"],
        ["Sensitivity Analysis", "가중치 5개 시나리오. Top-K overlap ≥ 80% 안정성 확인"],
        ["External Validation", "산불발생통계/소방청 화재와 고위험 설비 군집 일치율 비교"],
        ["Region-based CV", "공간 자기상관 Leakage 방지. 지역별 Train/Test 분리"],
    ]
    story.append(table(val_data, [5*cm, 11*cm], header_bg=DGRAY))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 7. Feature 전체 목록
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("7. Feature 전체 목록 (42개+)", S["h1"]))
    story.append(section_line())

    story.append(Paragraph("Weather Hazard Layer — 19개", S["h3"]))
    wf = [
        ["Feature", "설명"],
        ["temp_max/mean_{1,3,7,14}d", "일최대·평균기온 rolling window"],
        ["rh_min/mean_{1,3,7,14}d", "최저·평균 상대습도 rolling window"],
        ["ws_max/mean_{1,3,7,14}d", "최대·평균 풍속 rolling window"],
        ["precip_sum_{1,3,7,14}d", "누적 강수량 rolling window"],
        ["no_rain_days", "연속 무강수일수"],
        ["dryness_score, wind_score, heat_score, no_rain_score", "0~100 백분위 단위 위험 점수"],
        ["dry_wind_interaction, hot_dry_interaction, no_rain_wind_interaction", "복합 상호작용 feature"],
        ["dry_watch_flag, dry_warn_flag", "건조주의보·경보 기준 flag (기상청 공식 기준)"],
        ["wind_watch_flag, wind_warn_flag", "강풍주의보·경보 기준 flag"],
        ["combined_risk_flag, combined_risk_days, dry_watch_days", "복합위험 및 연속일수"],
        ["weather_hazard", "최종 Weather Hazard Score (0~100)"],
    ]
    story.append(table(wf, [7*cm, 9*cm], header_bg=RED))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Spatial Exposure Layer — 13개", S["h3"]))
    sf = [
        ["Feature", "데이터 소스", "설명"],
        ["forest_ratio_{250m,500m,1000m}", "ESA WorldCover 10m", "반경 내 산림(Tree cover) 비율"],
        ["conifer_ratio_{250m,500m,1000m}", "임상도 (신청 시)", "침엽수 비율 (화재 취약)"],
        ["distance_to_forest_m", "ESA WorldCover", "최근접 산림까지 거리"],
        ["slope_mean_{500m,1000m}", "Copernicus DEM 30m", "평균 경사도 ★SHAP 연구 1위 변수"],
        ["elevation_mean_{500m,1000m}", "Copernicus DEM 30m", "평균 고도"],
        ["aspect_risk_{500m,1000m}", "Copernicus DEM 30m", "사면 방향 위험도 (남향=고위험)"],
    ]
    story.append(table(sf, [5*cm, 4*cm, 7*cm], header_bg=GREEN))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Facility Exposure Layer — 5개+", S["h3"]))
    ff = [
        ["Feature", "설명"],
        ["facility_density_{500m,1000m}", "반경 내 다른 설비 개수"],
        ["nearest_facility_dist_m", "가장 가까운 설비까지 거리"],
        ["facility_age", "설비 노후도 (설치연도 기반, 날씨마루 확인 후)"],
        ["voltage_level, facility_type", "전압 등급, 설비 유형 (날씨마루 확인 후)"],
    ]
    story.append(table(ff, [5.5*cm, 10.5*cm], header_bg=LBLUE))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Historical Prior Layer — 5개", S["h3"]))
    pf = [
        ["Feature", "데이터 소스", "설명"],
        ["fire_count_{1000m,3000m,5000m}", "산림청 산불발생통계 (지오코딩 완료)", "반경 내 과거 산불 발생 건수"],
        ["elec_fire_count_{500m,1000m}", "소방청 화재통계 (전기 원인, 지오코딩 완료)", "반경 내 과거 전기화재 발생 건수"],
    ]
    story.append(table(pf, [4.5*cm, 5.5*cm, 6*cm], header_bg=DGRAY))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 8. 기술 스택
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("8. 기술 스택", S["h1"]))
    story.append(section_line())

    tech_data = [
        ["분류", "라이브러리/도구", "용도"],
        ["공간 분석", "GeoPandas, Shapely, PyProj, Rasterio", "GeoDataFrame, CRS 변환, Raster 처리"],
        ["ML 모델", "LightGBM, XGBoost, scikit-learn", "Supervised / Unsupervised 모델"],
        ["Explainability", "SHAP (TreeExplainer, LinearExplainer, KernelExplainer)", "Feature 기여도 분석"],
        ["클러스터링", "HDBSCAN", "고위험 설비군 공간 탐지 (Case B)"],
        ["시각화", "Matplotlib, Seaborn, Folium, Contextily", "정적/동적 지도, 차트"],
        ["GUI", "Streamlit, streamlit-folium", "인터랙티브 웹 앱"],
        ["LLM", "OpenAI GPT-4o-mini", "SHAP 수치 → 자연어 설명"],
        ["데이터 처리", "Pandas, NumPy, SciPy", "Feature engineering, 통계"],
        ["데이터 접근", "PyHive (Hive JDBC)", "날씨마루 Hive 연결"],
        ["개발 환경", "Python 3.12, Jupyter Notebook, .venv", "분석 및 코드 개발"],
    ]
    story.append(table(tech_data, [3.5*cm, 6*cm, 6.5*cm]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 9. OpenAI LLM 모듈
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("9. OpenAI LLM 자연어 설명 모듈", S["h1"]))
    story.append(section_line())
    story.append(Paragraph(
        "SHAP 수치를 현장 엔지니어가 이해할 수 있는 한국어 설명으로 변환합니다. "
        "LLM은 SHAP의 '번역기' 역할만 수행하며, 수치를 벗어난 생성을 엄격히 제한합니다.",
        S["body"]))

    story.append(Paragraph("역할 구분", S["h3"]))
    role_data = [
        ["구성 요소", "역할", "수학적 보장"],
        ["SHAP", "실제 Explainable AI (feature 기여도 분해)", "✓ 항상 보장 (Shapley value)"],
        ["GPT-4o-mini", "자연어 리포팅 도구 (SHAP 번역)", "✗ 보장 없음 (프롬프트로 제어)"],
    ]
    story.append(table(role_data, [4*cm, 7*cm, 5*cm], header_bg=LBLUE))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("출력 예시", S["h3"]))
    story.append(Paragraph(
        '"설비 F-1023은 최근 3일 상대습도가 18%로 매우 낮고 최대 풍속이 17m/s에 달하는 강풍 조건이 '
        '겹쳐 Very High 등급으로 분류됩니다. 반경 500m 내 산림 비율이 68%에 달해 화재 발생 시 '
        '급속한 확산이 우려되며, 과거 5년간 인근 1km 내 전기화재가 3건 발생한 이력이 있어 '
        '긴급 점검이 권고됩니다."',
        S["highlight"]))
    story.append(Paragraph(
        "캐시 기능: 동일 설비 재호출 시 API 비용 절약 (explanation_cache.json)", S["note"]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 10. Streamlit GUI
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("10. Streamlit GUI 앱 (app.py)", S["h1"]))
    story.append(section_line())
    story.append(Paragraph(
        "날씨마루 데이터 연동 후 즉시 실행 가능한 인터랙티브 웹 앱입니다. "
        "데이터 없을 때는 UI 미리보기 모드로 동작합니다.", S["body"]))

    gui_data = [
        ["탭", "주요 기능"],
        ["🗺️ 위험도 지도", "Folium 인터랙티브 지도, 4단계 등급별 색상, 설비 클릭 시 팝업 (위험도·기여도)"],
        ["📋 우선점검 목록", "Top-K 설비 테이블, Top-K Coverage 수치 (상위 X%가 전체 위험의 Y% 커버)"],
        ["📊 설비 상세 분석", "기여도 Bar Chart, SHAP Waterfall, GPT-4o-mini 자연어 설명 버튼"],
        ["📈 트렌드 분석", "날짜별 평균·최대 위험도 추이, 등급별 설비 수 변화 Stack 차트"],
    ]
    story.append(table(gui_data, [4*cm, 12*cm]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("실행 명령: cd wpfi_project && .venv/bin/streamlit run app.py", S["code"]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 11. 관련 연구 비교
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("11. 관련 연구 논문 비교 분석", S["h1"]))
    story.append(section_line())
    story.append(Paragraph(
        "10편의 최신 연구(2020~2026)를 분석한 결과, 세 가지 연구 흐름이 존재합니다:", S["body"]))

    for txt in [
        "전력설비 단위 위험 연구 (Yao et al. 2022 등): SHAP 없음, Feeder 단위",
        "SHAP 기반 산불 예측 연구 (Liao et al. 2025, Suheb et al. 2026 등): 전력설비 없음, 격자 단위",
        "한국 산불 연구 (Heo et al. 2026, Lee et al. 2025): 전력설비 없음, 행정구역 단위",
    ]:
        story.append(Paragraph(f"• {txt}", S["bullet"]))
    story.append(Spacer(1, 0.3*cm))

    compare_data = [
        ["비교 항목", "기존 연구 최선", "WPFI"],
        ["분석 단위", "Feeder 단위 (Yao 2022)", "설비 × 날짜 단위"],
        ["SHAP 적용", "격자 단위 (Liao 2025)", "설비 단위 SHAP"],
        ["전력설비+기상+산림+이력 결합", "부분적 결합만 존재", "4-layer 완전 결합"],
        ["전기화재 이력 활용", "없음", "소방청 49,759건 포함"],
        ["기상특보 파생 Flag", "없음", "5종 flag 직접 계산"],
        ["Multi-scale Buffer", "없음", "250m/500m/1km/3km"],
        ["0~100 위험 점수 체계", "이진 분류 또는 확률만", "0~100 + 4등급"],
        ["LLM 자연어 설명", "없음", "GPT-4o-mini 한국어"],
        ["한국 전력설비 맥락", "없음", "KEPCO 데이터 직접 활용"],
    ]
    story.append(table(compare_data, [5*cm, 6*cm, 5*cm], header_bg=DARKRED))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Research Gap: 전력설비 단위로 SHAP 기반 설명가능성을 제공하면서, "
        "기상·산림·지형·과거화재 이력을 다중 스케일 공간 버퍼로 결합한 연구는 현재까지 존재하지 않음. "
        "특히 한국 맥락의 전력설비 화재위험 ML 연구는 전무함.",
        S["highlight"]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 12. 강점과 한계
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("12. 우리 모델의 강점과 한계", S["h1"]))
    story.append(section_line())

    story.append(Paragraph("강점", S["h2"]))
    strengths = [
        ("과제 부합성", "전력설비 단위 화재위험 → 점검 우선순위 직접 연결. 대회 주제와 완벽 일치"),
        ("데이터 차별성", "소방청 전기화재 49,759건, ESA WorldCover 10m, DEM 30m, 산불 지오코딩 100%"),
        ("기술 차별성", "Multi-scale Buffer + SHAP + 기상특보 파생 flag 조합은 기존 연구에 없음"),
        ("활용성", "Top-K 우선점검 목록, Coverage 수치, GPT 설명 → 한전 현장 즉시 활용 가능"),
        ("합리성", "가중치 5개 시나리오 민감도 분석, Ablation Study로 근거 제시"),
        ("시의성", "2025년 경북·강원 대형 산불 이후 사회적 관심 최고조"),
    ]
    for title, desc in strengths:
        story.append(Paragraph(f"• <b>{title}</b>: {desc}", S["bullet"]))

    story.append(Paragraph("한계 및 주의사항", S["h2"]))
    limits = [
        ("분석 단위 불확실성", "날씨마루 화재 데이터의 위치 정보 포함 여부에 따라 설비 단위 label 정확도 변동"),
        ("WorldCover 한계", "수종 구분(침엽수/활엽수) 불가. 임상도 확보 시 conifer_ratio 추가 가능"),
        ("소방청 화재 좌표", "시군구 중심점으로 근사(±수km 오차). Prior로만 사용 권장"),
        ("LLM 신뢰성", "GPT 출력이 SHAP 수치와 불일치할 수 있음. 프롬프트 엄격 제한으로 완화"),
        ("전력망 연결 정보", "네트워크 토폴로지(from/to) 비공개. KG 구현은 KEPCO 데이터 확인 후 결정"),
    ]
    for title, desc in limits:
        story.append(Paragraph(f"• <b>{title}</b>: {desc}", S["bullet"]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 13. 현재 상태 및 계획
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("13. 현재 진행 상태 및 향후 계획", S["h1"]))
    story.append(section_line())

    status_data = [
        ["작업", "상태", "비고"],
        ["Python 환경 설치 (.venv)", "✓ 완료", "numpy<2, 모든 패키지 정상"],
        ["config.py 전역 설정", "✓ 완료", "가중치·경로·Hive 설정"],
        ["외부 데이터 수집", "✓ 완료", "DEM 767MB, WorldCover 146MB"],
        ["소방청 화재통계 수집·지오코딩", "✓ 완료", "191,510건, 99.7% 좌표 매핑"],
        ["산불발생통계 수집·지오코딩", "✓ 완료", "2,020건, 100% 좌표 매핑"],
        ["notebooks 01~09 작성", "✓ 완료", "SHAP 버그 수정 포함"],
        ["10_llm_explanation.ipynb", "✓ 완료", "GPT-4o-mini 연동, 캐시"],
        ["Streamlit app.py", "✓ 완료", "4탭, Folium 지도, 미리보기 모드"],
        ["날씨마루 접속 신청", "✓ 완료", "분석 플랫폼 '사용' 신청 완료"],
        ["날씨마루 데이터 승인", "⏳ 대기", "승인 즉시 01번 notebook 실행"],
        ["01_data_check 실행", "⏳ 승인 후", "Hive 테이블명, label 유무 확인"],
        ["02~09 notebook 실행", "⏳ 승인 후", "당일 실험 가능 상태"],
        ["임상도 신청 (산림청)", "⏳ 선택", "신청 시 1~2일 소요"],
        ["보고서 작성 (6페이지 hwpx)", "⏳ 실험 후", "제출 마감: 6월 26일"],
    ]
    story.append(table(status_data, [6.5*cm, 2.5*cm, 7*cm], header_bg=DGRAY))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("날씨마루 승인 직후 실행 순서", S["h3"]))
    for i, txt in enumerate([
        "01_data_check.ipynb: DESCRIBE 쿼리 → 실제 컬럼명 확인, column_map.json 완성",
        "02~05: 공간 전처리 → buffer feature → 기상 feature → risk index",
        "06: label 확인 → Case A (LightGBM + SHAP) 또는 Case B 자동 분기",
        "07~09: SHAP 설명 → 검증 → 지도 출력",
        "10: OpenAI API 키 입력 → 상위 설비 자연어 설명 생성",
        "app.py: streamlit run app.py → GUI 실시간 확인",
        "보고서: 결과 기반 6페이지 hwpx 작성 (제출: 6월 26일)",
    ], 1):
        story.append(Paragraph(f"{i}. {txt}", S["bullet"]))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 14. 참고문헌
    # ═══════════════════════════════════════════════════════════
    story.append(Paragraph("14. APA 7 참고문헌", S["h1"]))
    story.append(section_line())
    story.append(Paragraph(
        "본 프로젝트 설계 및 방법론 수립에 참고한 주요 연구 논문 목록입니다.", S["body"]))
    story.append(Spacer(1, 0.3*cm))

    refs = [
        ("[1]", "Yao, J., Bharadwaj, S., Zhang, M., Jin, Z., & Callaway, D. (2022). Predicting electricity infrastructure "
         "induced wildfire risk in California. arXiv preprint. https://arxiv.org/abs/2206.02930",
         "설비(Feeder) 단위 전력 인프라 화재위험 예측. XGBoost 계열. AUC 0.776~0.824. "
         "SHAP 미적용. 가장 근접한 선행 연구."),
        ("[2]", "Xu, J., Fang, C., & Cheng, Y. (2025). Wildfire risk assessment to overhead transmission-line "
         "based on improved analytic hierarchy process. Fire and Materials, 49(5), 523-535. "
         "https://doi.org/10.1002/fam.3267",
         "AHP 기반 송전선 화재위험 평가. 전문가 판단 기반 가중치. ML/SHAP 미적용."),
        ("[3]", "Liao, B., Zhou, T., Liu, Y., Li, M., & Zhang, T. (2025). Tackling the wildfire prediction challenge: "
         "An XAI model combining XGBoost with SHAP for enhanced interpretability and accuracy. "
         "Forests, 16(4), Article 689. https://doi.org/10.3390/f16040689",
         "XGBoost + SHAP 산불 예측. AUC 0.983. 격자 단위. 전력설비 미포함."),
        ("[4]", "Yang, X., Hao, Y., Ding, H., Yu, C., Liu, J., Li, L., & Chen, J. (2025). XAI framework using "
         "XGBoost and SHAP for urban fire risk based on spatial distribution features. "
         "International Journal of Disaster Risk Reduction, 129, Article 105798. "
         "https://doi.org/10.1016/j.ijdrr.2025.105798",
         "도시 화재 위험 XAI. 공간 분포 feature 활용 방식 참고."),
        ("[5]", "Heo, S., Ahn, S., Lee, Y.-E., Jung, S.-C., & Jang, M. (2026). Bridging climate and "
         "socio-environmental vulnerability for wildfire risk assessment using explainable machine learning: "
         "Evidence from the 2025 wildfire in Korea. Forests, 17(2), Article 182. "
         "https://doi.org/10.3390/f17020182",
         "2025년 한국 산불 Explainable ML. 격자 단위. 전력설비 미포함. 한국 맥락 참고."),
        ("[6]", "Suheb, M. N., Ali, M. S., Rashid, M. M., Naqvi, D. F., Qaisar, H., Sicard, P., "
         "Karuppannan, S., & Naqvi, H. R. (2026). An explainable GeoAI framework for spatial "
         "assessment of wildfire susceptibility in the Upper Ravi sub-basin, Indian Himalaya. "
         "Scientific Reports, 16(1), Article 11662. https://doi.org/10.1038/s41598-026-46924-w",
         "GeoAI + SHAP + 앙상블 스태킹. AUC 0.95. 방법론 구조가 WPFI와 가장 유사."),
        ("[7]", "Lee, C., Choi, E. H., Han, Y., & Lee, Y. (2025). Year-round daily wildfire prediction "
         "and key factor analysis using machine learning: A case study of Gangwon State, South Korea. "
         "Scientific Reports, 15(1), Article 29910. https://doi.org/10.1038/s41598-025-15508-5",
         "강원도 일별 산불 예측. SHAP 적용. DEM slope가 핵심 변수 확인. 한국 맥락."),
    ]

    for num, ref, note in refs:
        story.append(KeepTogether([
            Paragraph(f"<b>{num}</b> {ref}", S["body_small"]),
            Paragraph(f"→ {note}", S["note"]),
            Spacer(1, 0.2*cm),
        ]))

    story.append(Spacer(1, 0.5*cm))
    story.append(divider())
    story.append(Paragraph(
        "본 문서는 WPFI 프로젝트의 모든 설계 내용, 수행 작업, 데이터 현황을 담고 있습니다. "
        "날씨마루 승인 후 실험 결과가 추가되면 v2.0으로 업데이트할 예정입니다.",
        S["note"]))

    doc.build(story)
    print(f"PDF 생성 완료: {output_path}")


if __name__ == "__main__":
    out = "/Users/seungwookim/Desktop/WPFI_프로젝트_종합문서.pdf"
    build_pdf(out)
    import os
    size = os.path.getsize(out) / 1024
    print(f"파일 크기: {size:.0f} KB")
