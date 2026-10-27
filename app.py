import os
import re
import base64
import streamlit as st
import pandas as pd
import google.generativeai as genai
import PyPDF2
from google.api_core import exceptions as gexc

# =====================================================================
# 1. 기본 설정 및 스타일
# =====================================================================
st.set_page_config(
    page_title="승강기 설계심사 에이전트",
    page_icon="🛗",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSET_DIR = "assets"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

.block-container { padding-top: 3.5rem; padding-bottom: 2rem;
    padding-left: 2.5rem; padding-right: 2.5rem; max-width: 1560px; }

div[data-testid="stVerticalBlock"]:has(.st-key-qa_section) {
    display: flex; flex-direction: column; min-height: calc(100vh - 6rem); }
.st-key-qa_section { margin-top: auto !important; }
.st-key-result_section { width: 100%; }

.kh-topbar { height: 6px; border-radius: 6px 6px 0 0;
    background: linear-gradient(90deg, #F47920 0%, #FDB913 35%, #03BFD7 70%, #0A7FC1 100%); }
.kh-header { display: flex; align-items: center; gap: 20px; background: #ffffff; color: #1d2b3a;
    padding: 16px 24px; border-radius: 0 0 16px 16px; box-shadow: 0 8px 24px rgba(10,61,98,.10);
    margin-bottom: 22px; }
.kh-header img { height: 46px; }
.kh-header .kh-title { font-size: 1.45rem; font-weight: 800; color: #0A3D62; }
.kh-header .kh-sub { font-size: 0.86rem; color: #6d6e71; margin-top: 2px; }

.kh-hero { display: flex; align-items: center; gap: 28px; margin-bottom: 26px;
    background: linear-gradient(120deg, #E6F9FC 0%, #FFFFFF 55%, #FFF6E0 100%);
    border: 1px solid #DDF3F8; border-radius: 22px; padding: 22px 30px;
    box-shadow: 0 10px 30px rgba(3,191,215,.12); }
.kh-hero .frame { background: #ffffff; border-radius: 16px; padding: 8px;
    box-shadow: 0 6px 16px rgba(10,61,98,.10); flex-shrink: 0; }
.kh-hero .frame img { height: 130px; display: block; }
.kh-hero h2 { margin: 0; font-size: 1.5rem; font-weight: 800; color: #0A3D62; line-height: 1.35; }
.kh-hero p { margin: 10px 0 0 0; color: #4a4a4a; font-size: 0.96rem; line-height: 1.65; }
.kh-hero .kh-chip { display: inline-block; font-size: 0.75rem; font-weight: 700; color: #0A7FC1;
    background: #ffffff; border: 1px solid #BFE9F2; border-radius: 999px; padding: 3px 12px; margin-bottom: 10px; }

.kh-pill { display: inline-flex; align-items: center; gap: 8px; font-size: 1.05rem; font-weight: 800;
    color: #0A3D62; background: #ffffff; border: 1px solid #DDEFF5; border-radius: 999px;
    padding: 8px 18px; box-shadow: 0 4px 12px rgba(10,61,98,.06); margin: 10px 0 16px 0; }

div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {
    gap: 0.9rem; }
div[data-testid="stMetric"] { background: linear-gradient(135deg, #F3FBFD, #FFFFFF);
    border: 1px solid #DDEFF5; border-left: 4px solid #03BFD7;
    border-radius: 14px; padding: 0.8rem 1rem; }
div[data-testid="stMetric"] label p { color: #6d6e71 !important; font-weight: 700; }

div[data-testid="stAlert"] { border-radius: 14px; }

div[data-testid="column"] button[kind="secondary"] {
    border-radius: 999px; border: 1px solid #BFE9F2; color: #0A7FC1; font-weight: 700;
    background: #F7FDFE; min-height: 2.6rem; }
div[data-testid="column"] button[kind="secondary"]:hover {
    border-color: #03BFD7; background: #E6F9FC; color: #0A3D62; }

div[data-testid="stChatMessage"] {
    background: #ffffff; border: 1px solid #EAF1F6; border-radius: 18px;
    padding: 12px 18px; box-shadow: 0 4px 14px rgba(10,61,98,.06); margin-bottom: 12px; }
div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] h1, div[data-testid="stChatMessage"] h2,
div[data-testid="stChatMessage"] h3, div[data-testid="stChatMessage"] strong {
    color: #1d2b3a !important; }

.side-title { font-size: 1.1rem; font-weight: 800; color: #0A7FC1;
    border-left: 5px solid #FDB913; padding-left: 10px; margin: 0.4rem 0 0.6rem 0; }
.side-note { font-size: 0.78rem; opacity: 0.75; line-height: 1.55; margin: 0.3rem 0 1rem 0; }
div[data-testid="stExpander"] { border-radius: 14px; border: 1px solid #CDEFF6;
    box-shadow: 0 2px 8px rgba(3,191,215,.08); margin-bottom: 8px; }
div[data-testid="stExpander"] summary p { font-weight: 800; color: #0A3D62; }

button[kind="primary"] { background: linear-gradient(90deg, #F47920, #FDB913) !important;
    border: none !important; color: #ffffff !important; border-radius: 12px;
    font-weight: 800; height: 3rem; box-shadow: 0 6px 16px rgba(244,121,32,.30); }
button[kind="primary"]:hover { filter: brightness(1.05); }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. 이미지 도우미
# =====================================================================
@st.cache_data(show_spinner=False)
def img_uri(name):
    p = os.path.join(ASSET_DIR, name)
    if not os.path.exists(p):
        return None
    mime = "image/png" if p.lower().endswith(".png") else "image/jpeg"
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"

def asset_path(name):
    p = os.path.join(ASSET_DIR, name)
    return p if os.path.exists(p) else None

logo_uri = img_uri("koelsa_logo.png")
logo_tag = f'<img src="{logo_uri}">' if logo_uri else ""
st.markdown(
    '<div class="kh-topbar"></div>'
    '<div class="kh-header">' + logo_tag +
    '<div><div class="kh-title">승강기 설계심사 에이전트</div>'
    '<div class="kh-sub">승강기 안전인증 설계심사 부적합 사항 예측 및 안전 기준 질의</div></div></div>',
    unsafe_allow_html=True,
)

hello_uri = img_uri("char_hello.png")
hello_frame = f'<div class="frame"><img src="{hello_uri}"></div>' if hello_uri else ""
st.markdown(
    '<div class="kh-hero">' + hello_frame +
    '<div><span class="kh-chip">AI 설계심사 도우미</span>'
    '<h2>입력한 제원으로 부적합 사항을 미리 확인하세요</h2>'
    '<p>좌측에서 세부제원 선택 후 <b>예측 분석 실행</b>을 누르면, '
    '과거 개별인증 사례에서 유사하게 나온 부적합 사항을 정리해 드립니다.</p></div></div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "submit_from_button" not in st.session_state:
    st.session_state.submit_from_button = False
if "llm_cache" not in st.session_state:
    st.session_state.llm_cache = {}

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"
model = genai.GenerativeModel(MODEL_NAME)

AI_TIMEOUT = 120             # AI 응답 제한 시간(초)
ANALYSIS_MAX_TOKENS = 4096   # 묶음 요약 + 추천 질문을 담는 출력 상한
CHAT_MAX_TOKENS = 3000       # 질의응답 출력 상한

# =====================================================================
# 3. 기준 문서(PDF) 로드 및 조각내기
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_pdf_chunks():
    texts = []
    for f in os.listdir("."):
        if f.lower().endswith(".pdf"):
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                texts.append(page.extract_text() or "")
    full = "\n".join(texts)
    chunks = [full[i:i + 1200] for i in range(0, len(full), 1000)]
    return [c for c in chunks if c.strip()]

pdf_error = None
try:
    PDF_CHUNKS = load_pdf_chunks()
    if not PDF_CHUNKS:
        pdf_error = "기준 문서(PDF)를 찾지 못했거나 내용이 비어 있습니다. 파일 위치를 확인해 주세요."
except Exception as e:
    PDF_CHUNKS = []
    pdf_error = f"기준 문서 학습 중 오류가 발생했습니다: {e}"

if pdf_error:
    st.sidebar.error(pdf_error)

def retrieve_chunks(query, chunks, k=5):
    tokens = re.findall(r"[가-힣A-Za-z0-9\.]+", query)
    grams = set()
    for t in tokens:
        for i in range(len(t) - 1):
            grams.add(t[i:i + 2])
    scored = []
    for ch in chunks:
        s = sum(1 for g in grams if g in ch)
        scored.append((s, ch))
    scored.sort(key=lambda x: -x[0])
    picked = [c for s, c in scored[:k] if s > 0]
    return picked or chunks[:2]

# =====================================================================
# 4. 데이터 로드
# =====================================================================
DATA_FILE = "개별설계심사현황 1회차(접수일자20230701_20260630)_최종 완성 자료(마스킹 완료)_최종.csv"
TYPES = ["일반형", "전기식 소형화물용", "경사형"]

@st.cache_resource
def load_data():
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df['매다는 장치 [종류]'] = df['매다는 장치 [종류]'].replace('로프', '와이어로프')
    return df

df = load_data()

# =====================================================================
# 5. 입력 항목 정의
# =====================================================================
TOP_FIELDS = ['안전인증유형', '승강기종류', '구분1']

NUMERIC_COLS = [
    '구동기 용량 [kW]', '정격하중 [kg]', '견인하중 [kg]', '정원 [인승]', '정격속도 [m/s]',
    '최대 운행거리 [m]', '매다는 장치 직경 [mm, (호)]', '매다는 장치 [가닥]', '주 도르래 직경 [mm]',
    '권부각도 α [ º ]', '인버터 용량 [kW]', '카 자중 [kg]', '카 내부(폭) 크기 [m]',
    '카 내부(깊이) 크기 [m]', '카 출입구(폭) 크기  [m]', '카 출입구(높이) 크기  [m]',
    '승강장 출입구(폭) 크기  [m]', '승강장 출입구(높이) 크기  [m]', '균형추/평형추 총하중 [kg]',
    '균형추/평형추 도르래 직경 [mm]', '매다는 장치 파단하중 [kN]', '과속조절기 로프 직경  [mm]',
    '과속조절기 도르래 직경  [mm]', '카 주행안내 레일 브라켓 최대간격 [mm]',
    '균형추/평형추 주행안내 레일 브라켓 최대간격 [mm]',
]

BASIC_REST = ['신청자명', '제조업체', '업체구분', '모델명', '심사원']

DRIVE_GEN = ['구동기 모델', '구동기 용량 [kW]', '브레이크 모델', '정격하중 [kg]', '견인하중 [kg]',
             '정원 [인승]', '정격속도 [m/s]', '최대 운행거리 [m]', '매다는 장치 [종류]',
             '매다는 장치 직경 [mm, (호)]', '매다는 장치 [가닥]', '보상수단 [종류] (선택사양)',
             '로핑방식', '로핑 구조', '주 도르래 직경 [mm]', '권부각도 α [ º ]', '주 도르래 형상']
CTRL_GEN = ['제어반 모델', '주 전원 [V]', '카 조명 전원 [V]', '회생전력 처리 형식', '인버터 용량 [kW]',
            '안전회로 전원 [V]', '제어회로 전원 [V]', '개문 착상 운전 (선택사양)',
            '개문 재착상 운전 (선택사양)', '주제어 기판', 'UCMP 감지부 모델', 'UCMP 제어부 모델',
            'UCMP 정지부 모델', '안전회로기판 모델', '구출운전방식', '강제감속운전 (선택사양)']
CAR_GEN = ['카 자중 [kg]', '카 내부(폭) 크기 [m]', '카 내부(깊이) 크기 [m]', '카 출입구(폭) 크기  [m]',
           '카 출입구(높이) 크기  [m]', '승강장 출입구(폭) 크기  [m]', '승강장 출입구(높이) 크기  [m]',
           '출입문 형식', '2개 이상의 카 출입문 (선택사양)', '문 개폐방식', '카 문 잠금장치 (선택사양)',
           '문닫힘 안전장치']
HOIST_GEN = ['구동기 지지대 규격  [mm]', '오버밸런스율 [%]', '균형추/평형추 총하중 [kg]',
             '균형추/평형추 도르래 직경 [mm]', '균형추 추락방지안전장치 (선택사양)',
             '매다는 장치 체결방식(카)', '매다는 장치 체결방식(균형추/평형추)', '매다는 장치 파단하중 [kN]',
             '과부하감지장치 종류(선택사양)', '과부하감지장치 위치 (선택사양)',
             '과속조절기 로프 직경  [mm]', '과속조절기 도르래 직경  [mm]',
             '카 주행안내 레일 규격 [K, (T)]', '균형추/평형추 주행안내 레일 규격  [K, (T)]',
             '카 주행안내 레일 브라켓 최대간격 [mm]', '균형추/평형추 주행안내 레일 브라켓 최대간격 [mm]']

DRIVE_SMALL = ['구동기 모델', '구동기 용량 [kW]', '브레이크 모델', '정격하중 [kg]', '견인하중 [kg]',
               '정격속도 [m/s]', '최대 운행거리 [m]', '매다는 장치 [종류]', '매다는 장치 직경 [mm, (호)]',
               '매다는 장치 [가닥]', '로핑방식', '로핑 구조', '주 도르래 직경 [mm]', '권부각도 α [ º ]',
               '주 도르래 형상']
CTRL_SMALL = ['제어반 모델', '주 전원 [V]', '카 조명 전원 [V]', '회생전력 처리 형식', '인버터 용량 [kW]',
              '안전회로 전원 [V]', '제어회로 전원 [V]', '개문 착상 운전 (선택사양)',
              '개문 재착상 운전 (선택사양)', '주제어 기판', 'UCMP 감지부 모델', 'UCMP 제어부 모델',
              'UCMP 정지부 모델', '안전회로기판 모델', '비상 운전방식']

DRIVE_SLOPE = ['구동기 모델', '구동기 용량 [kW]', '브레이크 모델', '정격하중 [kg]', '견인하중 [kg]',
               '정원 [인승]', '최대 경사각도 [ º ]', '정격속도 [m/s]', '최대 운행거리 [m]',
               '매다는 장치 [종류]', '매다는 장치 직경 [mm, (호)]', '매다는 장치 [가닥]',
               '보상수단 [종류] (선택사양)', '로핑방식', '로핑 구조', '주 도르래 직경 [mm]',
               '권부각도 α [ º ]', '주 도르래 형상']
CTRL_SLOPE = ['제어반 모델', '주 전원 [V]', '카 조명 전원 [V]', '회생전력 처리 형식', '인버터 용량 [kW]',
              '안전회로 전원 [V]', '제어회로 전원 [V]', '개문 착상 운전 (선택사양)',
              '개문 재착상 운전 (선택사양)', '주제어 기판', 'UCMP 감지부 모델', 'UCMP 정지부 모델',
              'UCMP 제어부 모델', '안전회로기판 모델', '구출운전방식']
CAR_SLOPE = CAR_GEN + ['과부하감지장치 종류(선택사양)']
HOIST_SLOPE = ['구동기 지지대 규격  [mm]', '오버밸런스율 [%]', '균형추/평형추 총하중 [kg]',
               '균형추/평형추 도르래 직경 [mm]', '매다는 장치 체결방식(카)',
               '매다는 장치 체결방식(균형추/평형추)', '매다는 장치 파단하중 [kN]',
               '과부하감지장치 위치 (선택사양)', '과속조절기 로프 직경  [mm]',
               '과속조절기 도르래 직경  [mm]', '카 주행안내 레일 규격 [K, (T)]',
               '균형추/평형추 주행안내 레일 규격  [K, (T)]', '주행트랙/주행안내 레일 간격 [mm]',
               '카 주행안내 레일 브라켓 최대간격 [mm]', '균형추/평형추 주행안내 레일 브라켓 최대간격 [mm]']

LAYOUTS = {
    "일반형": [
        ("구동 시스템", DRIVE_GEN),
        ("제어 시스템", CTRL_GEN),
        ("카 구조", CAR_GEN),
        ("승강로 구조", HOIST_GEN),
    ],
    "전기식 소형화물용": [
        ("구동 시스템", DRIVE_SMALL),
        ("제어 시스템", CTRL_SMALL),
        ("카 구조", CAR_GEN),
        ("승강로 구조", HOIST_GEN),
    ],
    "경사형": [
        ("구동 시스템", DRIVE_SLOPE),
        ("제어 시스템", CTRL_SLOPE),
        ("카 구조", CAR_SLOPE),
        ("승강로 구조", HOIST_SLOPE),
    ],
}

# =====================================================================
# 6. 선택지 및 입력 파싱
# =====================================================================
@st.cache_data
def cat_options_all(col):
    return sorted(df[col].dropna().astype(str).unique().tolist())

@st.cache_data
def cat_options(gubun, col):
    sub = df[df["구분2"].astype(str) == gubun]
    return sorted(sub[col].dropna().astype(str).unique().tolist())

def parse_num(text):
    t = (text or "").strip().replace(",", "")
    if t == "":
        return True, None
    try:
        return True, float(t)
    except ValueError:
        return False, None

# =====================================================================
# 7. 사이드바 (입력)
# =====================================================================
btn_slot = st.sidebar.container()

st.sidebar.markdown('<div class="side-title">📋 세부 제원 입력</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="side-note">\'선택 안 함\'은 조건에서 제외됩니다. '
    '숫자 항목은 입력한 값 그대로 사용되며, 비워 두면 제외됩니다.</div>',
    unsafe_allow_html=True,
)

user_inputs = {}

for col in TOP_FIELDS:
    choice = st.sidebar.selectbox(col, ["선택 안 함"] + cat_options_all(col), key=f"top|{col}")
    user_inputs[col] = ("cat", choice) if choice != "선택 안 함" else None

gubun = st.sidebar.selectbox("구분2 (승강기 유형)", TYPES, index=0, key="gubun")

with st.sidebar.expander(f"기본 정보 ({len(BASIC_REST)})", expanded=False):
    for col in BASIC_REST:
        choice = st.selectbox(col, ["선택 안 함"] + cat_options(gubun, col), key=f"cat|{gubun}|{col}")
        user_inputs[col] = ("cat", choice) if choice != "선택 안 함" else None

for group_name, cols in LAYOUTS[gubun]:
    with st.sidebar.expander(f"{group_name} ({len(cols)})", expanded=False):
        for col in cols:
            if col in NUMERIC_COLS:
                raw = st.text_input(col, key=f"txt|{gubun}|{col}", placeholder="숫자 입력 (비우면 제외)")
                ok, num = parse_num(raw)
                if not ok:
                    st.warning(f"'{col}': 숫자만 입력해 주세요.")
                user_inputs[col] = ("num", num) if (ok and num is not None) else None
            else:
                choice = st.selectbox(col, ["선택 안 함"] + cat_options(gubun, col), key=f"cat|{gubun}|{col}")
                user_inputs[col] = ("cat", choice) if choice != "선택 안 함" else None

run = btn_slot.button("🔍 예측 분석 실행", type="primary", use_container_width=True)

# =====================================================================
# 8. 연관 사례 찾기 + 중복 정리(파이썬) + 기준 원문 검색 + AI 묶기·요약 1회
# =====================================================================
SAMPLE_N = 200            # 연관도 상위 몇 건을 볼지
SEND_N = 150              # AI에 보낼 고유 문장 최대 개수
REF_K = 8                 # 안전기준 원문에서 가져올 조각 수
IGNORE_KEYWORDS = ["안내사항", "개별인증 신청 시 기술원 홈페이지", "인증 신청 시 작성하는", "최신 설계심사 목록표를 참조하여"]

@st.cache_data(show_spinner=False)
def find_matches(gubun, active_items):
    base = df[df["구분2"].astype(str) == gubun]
    score = pd.Series(0.0, index=base.index)

    for col, kind, val in active_items:
        if kind == "cat":
            score += (base[col].astype(str) == val).astype(float)
        else:
            vals = pd.to_numeric(base[col], errors="coerce")
            rng = vals.max() - vals.min()
            if pd.isna(rng) or rng == 0:
                rng = 1.0
            score += (1 - ((vals - val).abs() / rng).clip(0, 1)).fillna(0)

    tmp = pd.DataFrame({"부적합내용": base["부적합내용"], "유사도 점수": score})
    top = tmp.nlargest(SAMPLE_N, "유사도 점수")
    return len(base), top

def normalize_for_dedupe(s):
    s = str(s)
    s = re.sub(r"\(\s*p\.?\s*\d+[^)]*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bp\.?\s*\d+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w가-힣]", "", s)
    return s.lower()

def unique_issues(texts):
    """글자가 완전히 같은 문장(공백·문장부호·페이지 표기 무시)만 하나로 모읍니다. 건수 순서로 반환합니다."""
    groups = {}
    for t in texts:
        t = str(t).strip()
        key = normalize_for_dedupe(t)
        if not key:
            continue
        groups.setdefault(key, {})
        groups[key][t] = groups[key].get(t, 0) + 1

    result = []
    for key, forms in groups.items():
        rep = max(forms, key=forms.get)
        result.append((rep, sum(forms.values())))
    result.sort(key=lambda x: -x[1])
    return result

@st.cache_data(show_spinner=False)
def compute_issues(gubun, active_items):
    base_n, top = find_matches(gubun, active_items)
    raw = top["부적합내용"].dropna().astype(str).str.strip()
    raw = raw[~raw.apply(lambda x: any(k in x for k in IGNORE_KEYWORDS))]
    uniq = unique_issues(raw.tolist())
    return base_n, len(top), len(raw), uniq

def get_text(resp):
    try:
        return resp.text or ""
    except Exception:
        return ""

def is_truncated(resp):
    try:
        reason = resp.candidates[0].finish_reason
        name = getattr(reason, "name", str(reason))
        return "MAX_TOKENS" in str(name).upper()
    except Exception:
        return False

def call_gemini(prompt, max_tokens):
    """(응답 텍스트, 끝까지 생성됐는지) 반환."""
    config = {"max_output_tokens": max_tokens, "temperature": 0.2}
    resp = model.generate_content(
        prompt,
        generation_config=config,
        request_options={"timeout": AI_TIMEOUT},
    )
    return get_text(resp), not is_truncated(resp)

def build_reference(uniq_items):
    """부적합 문장의 핵심 단어로 안전기준 원문 조각을 찾아 하나의 텍스트로 만듭니다."""
    if not PDF_CHUNKS:
        return ""
    query = " ".join(txt for txt, _ in uniq_items[:15])
    picked = retrieve_chunks(query, PDF_CHUNKS, k=REF_K)
    return "\n\n---\n\n".join(picked)

def build_analysis_prompt(uniq_items, reference):
    lines = "\n".join(f"- ({cnt}건) {txt[:200]}" for txt, cnt in uniq_items[:SEND_N])
    return f"""
    다음은 승강기 설계심사에서 입력 조건과 유사한 과거 개별인증 사례의 부적합 사항 목록입니다.
    괄호 안 숫자는 같은 문장이 나온 건수입니다. 표현만 다르고 뜻이 같은 문장끼리는 스스로 한 묶음으로 합치고, 그 묶음의 건수를 모두 더해 주세요.

    [참고 안전기준 원문]
    아래 원문은 안전기준과 설계심사 가이드에서 찾은 발췌입니다. 조항 번호와 제목은 반드시 이 원문에 실제로 적혀 있는 것만 사용하세요.
    원문에서 해당 조항을 확인할 수 없으면 조항 번호 자리에 "조항 확인 필요"라고만 쓰고, 번호를 지어내지 마세요.

    {reference}

    합쳐진 묶음 중에서 실제로 자주 반복되는 주요 부적합 유형을 최대 5개까지 골라 아래 형식으로 작성해 주세요.
    헤더의 대괄호 안에는 실제 조항 번호와 제목을 넣으세요. 예: 6.1.2 카 주행안내 레일 브라켓 최대간격
    억지로 개수를 맞추지 말고, 목록에 없는 내용은 지어내지 마세요.
    - 부적합 내용: 어떤 설계서·도면·계산서가 빠졌거나 어떤 값이 기준과 맞지 않는지 2~3문장으로 구체적으로 설명하세요.
    - 엔지니어링 조언: 설계 단계에서 무엇을 어떤 기준으로 확인하고, 도면이나 계산서에 무엇을 반영해야 하는지 2~3문장으로 구체적으로 쓰세요.

    ### [조항 번호 제목] 핵심 부적합 사유
    * **부적합 내용**: (2~3문장)
    * **유사 발생 건수**: 약 O건 (합쳐진 묶음의 건수 합계)
    * **엔지니어링 조언**: (2~3문장)

    모든 항목을 쓴 뒤, 마지막 줄부터 아래 형식의 줄 3개만 덧붙이세요.
    질문은 위에서 쓴 부적합 유형의 원인, 확인 기준, 계산 방법을 묻는 자연스러운 한국어 질문이어야 합니다.
    페이지 번호나 원문 문장을 그대로 넣지 마세요.
    [추천질문] (질문 1)
    [추천질문] (질문 2)
    [추천질문] (질문 3)

    부적합 목록:
    {lines}
    """

def split_questions(raw):
    body, questions = [], []
    for line in raw.split("\n"):
        if line.strip().startswith("[추천질문]"):
            q = line.strip().replace("[추천질문]", "").strip()
            if q:
                questions.append(q)
        else:
            body.append(line)
    return "\n".join(body).strip(), questions[:3]

def render_ai_analysis(slot, uniq_items, key):
    """AI 묶기·요약과 추천 질문을 조건당 한 번만 만들고 캐시에 저장합니다. 추천 질문 목록을 반환합니다."""
    cache = st.session_state.llm_cache

    if not uniq_items:
        slot.info("연관 사례는 있으나 분석할 부적합 내용이 없습니다.")
        return []
    if key in cache:
        slot.markdown(cache[key]["text"])
        return cache[key]["questions"]
    if st.session_state.get("ai_error_key") == key:
        slot.error(st.session_state.ai_error_msg)
        return []

    with slot:
        with st.spinner(f"기준 원문을 찾고 AI가 유사 항목을 정리하는 중... (최대 {AI_TIMEOUT}초)"):
            try:
                reference = build_reference(uniq_items)
                raw, complete = call_gemini(
                    build_analysis_prompt(uniq_items, reference),
                    max_tokens=ANALYSIS_MAX_TOKENS,
                )
                st.session_state.pop("ai_error_key", None)
            except gexc.ResourceExhausted:
                st.session_state.ai_error_key = key
                st.session_state.ai_error_msg = "AI 사용량 한도에 도달했습니다. 잠시 후 다시 실행해 주세요."
                slot.error(st.session_state.ai_error_msg)
                return []
            except Exception as e:
                st.session_state.ai_error_key = key
                st.session_state.ai_error_msg = f"AI 요약이 완료되지 않았습니다(시간 초과 또는 오류): {e}"
                slot.error(st.session_state.ai_error_msg)
                return []

    text, questions = split_questions(raw)
    if complete:
        cache[key] = {"text": text, "questions": questions}
    else:
        slot.warning("AI 응답이 길이 제한에 걸려 끊겼습니다. 같은 조건으로 다시 실행하면 새로 생성됩니다.")
    slot.markdown(text)
    return questions

if run:
    active = {c: v for c, v in user_inputs.items() if v is not None}

    if not active:
        st.session_state.pop("result", None)
        st.session_state.pop("ai_key", None)
        st.warning("제원을 하나 이상 선택하거나 입력해 주세요.")
    else:
        active_items = tuple(sorted(((c, k, v) for c, (k, v) in active.items()), key=lambda x: x[0]))

        with st.spinner("유사 사례를 찾는 중..."):
            base_n, top_n, raw_n, uniq = compute_issues(gubun, active_items)

        # 먼저 ai_key를 저장한 뒤 result를 저장합니다
        st.session_state.ai_key = (gubun, active_items)
        st.session_state.result = {
            "gubun": gubun,
            "base_n": base_n,
            "top_n": top_n,
            "active_n": len(active),
            "unique_n": len(uniq),
            "uniq_items": uniq[:SEND_N],
        }

# =====================================================================
# 9. 분석 결과 (환영 배너와 질의응답 사이에 표시)
# =====================================================================
result_section = st.container(key="result_section")
with result_section:
    if "result" in st.session_state:
        r = st.session_state.result

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("유형", r["gubun"])
            c2.metric("분석 대상", f"{r['top_n']:,}건")
            c3.metric("입력 항목", f"{r['active_n']}개")
            st.caption(
                f"전체 {r['base_n']:,}건 중 입력 조건과 가장 가까운 {r['top_n']:,}건을 찾았고, "
                f"같은 뜻의 문장을 정리해 고유 부적합 {r['unique_n']:,}개를 AI에 전달했습니다."
            )

            st.warning("내용이 모호하거나 확실하지 않은 경우, 반드시 관련 안전기준 원문 항목을 직접 확인하시기 바랍니다.")

            ai_slot = st.empty()
            ai_key = st.session_state.get("ai_key")
            if ai_key is None:
                st.session_state.pop("result", None)
                st.info("결과가 중간에 끊겼습니다. 예측 분석 실행을 다시 눌러 주세요.")
                questions = []
            else:
                questions = render_ai_analysis(ai_slot, r["uniq_items"], ai_key)

        if questions:
            st.markdown('<div class="kh-pill">💡 관련 추천 질문</div>', unsafe_allow_html=True)
            q_cols = st.columns(len(questions))
            for i, q in enumerate(questions):
                if q_cols[i].button(q, key=f"rq_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.session_state.submit_from_button = True
                    st.rerun()

# =====================================================================
# 10. 질의응답 (화면 하단에 고정되는 블록)
# =====================================================================
AI_AVATAR = asset_path("elri.png") or "🛗"
USER_AVATAR = "🧑‍💼"

qa = st.container(key="qa_section")
with qa:
    st.markdown('<div class="kh-pill">💬 안전기준 및 공학 추론 질의응답</div>', unsafe_allow_html=True)
    st.caption("질문과 관련된 안전기준을 찾아 답변합니다.")

    chat_box = st.container()
    with chat_box:
        for message in st.session_state.messages:
            avatar = USER_AVATAR if message["role"] == "user" else AI_AVATAR
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        prompt_input = st.chat_input("질문을 입력하세요. (예: 매다는 장치 안전율 계산 관련 안전기준을 알려줘)")

trigger = False

if prompt_input:
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    trigger = True
elif st.session_state.submit_from_button:
    trigger = True
    st.session_state.submit_from_button = False

if trigger:
    latest_user_msg = st.session_state.messages[-1]["content"]
    with chat_box:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(latest_user_msg)

        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.spinner("관련 기준을 찾고 답변을 준비하는 중..."):
                related = "\n\n".join(retrieve_chunks(latest_user_msg, PDF_CHUNKS, k=5))
                system_prompt = f"""
                당신은 한국 승강기안전공단의 설계심사 최고 권위자이자 기계공학 전문가입니다.
                사용자의 질문에 대해 기계공학적 배경지식과, 아래 [참고 기준 문서]에 근거하여 전문가적인 어조로 답변하세요.
                문서에 없는 공학적 계산(예: 오일러 좌굴 하중 등)은 물리/기계공학 지식을 활용해 수식과 함께 설명하세요.
                출처는 페이지 번호 대신 문서 내 조항 번호나 항목 제목으로 명시하십시오.
                문서에서 근거를 찾을 수 없으면 그렇다고 말하고, 일반 공학 지식으로 답하고 있음을 밝히세요.

                [참고 기준 문서 내용]
                {related}

                사용자 질문: {latest_user_msg}
                """
                try:
                    answer, complete = call_gemini(system_prompt, max_tokens=CHAT_MAX_TOKENS)
                    st.markdown(answer)
                    if not complete:
                        st.warning("답변이 길이 제한에 걸려 끊겼습니다. 질문을 나눠서 다시 물어보시면 좋습니다.")
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except gexc.ResourceExhausted:
                    st.error("AI 사용량 한도에 도달했습니다. 잠시 후 다시 질문해 주세요.")
                except Exception as e:
                    st.error(f"답변 생성 중 오류가 발생했습니다: {e}")