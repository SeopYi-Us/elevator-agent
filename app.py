import os
import streamlit as st
import pandas as pd
import google.generativeai as genai
import PyPDF2

# =====================================================================
# 1. 기본 설정 및 엔진 초기화
# =====================================================================
st.set_page_config(page_title="승강기 설계심사 에이전트", layout="wide")

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.8-flash"  # 실제 사용 가능한 모델명인지 Google AI Studio에서 확인하세요
model = genai.GenerativeModel(MODEL_NAME)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "submit_from_button" not in st.session_state:
    st.session_state.submit_from_button = False

st.title("🔍 승강기 설계심사 전문 에이전트")
st.markdown("입력된 제원을 바탕으로 보완 사항을 예측하고, 안전기준에 근거한 공학적 대화를 지원합니다.")

# =====================================================================
# 2. PDF 규정/가이드라인 자동 학습 (문제가 있을 때만 알림 표시)
# =====================================================================
@st.cache_data(show_spinner=False)
def load_pdf_knowledge():
    folder_path = "."
    extracted_text = ""
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            reader = PyPDF2.PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
    return extracted_text

if "pdf_context" not in st.session_state:
    with st.spinner("승강기 안전기준과 설계 가이드를 백그라운드에서 학습 중입니다..."):
        try:
            st.session_state.pdf_context = load_pdf_knowledge()
            st.session_state.pdf_error = None
            if not st.session_state.pdf_context.strip():
                st.session_state.pdf_error = "기준 문서(PDF)에서 읽어 온 내용이 없습니다. 파일 위치를 확인해 주세요."
        except Exception as e:
            st.session_state.pdf_context = ""
            st.session_state.pdf_error = f"기준 문서 학습 중 오류가 발생했습니다: {e}"

# 문제가 있을 때만 표시
if st.session_state.get("pdf_error"):
    st.sidebar.error(f"📚 {st.session_state.pdf_error}")

# =====================================================================
# 3. 데이터 로드
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
# 4. 입력 항목 정의
# =====================================================================
# 사이드바 상단에 올리는 항목 (순서 중요)
TOP_FIELDS = ['안전인증유형', '승강기종류', '구분1']

# 숫자로 직접 입력받는 항목
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

# 기본 정보 중 상단으로 올리지 않은 나머지
BASIC_REST = ['신청자명', '제조업체', '업체구분', '모델명', '현장명', '심사원']

# ---- 일반형 ----
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

# ---- 전기식 소형화물용 ----
DRIVE_SMALL = ['구동기 모델', '구동기 용량 [kW]', '브레이크 모델', '정격하중 [kg]', '견인하중 [kg]',
               '정격속도 [m/s]', '최대 운행거리 [m]', '매다는 장치 [종류]', '매다는 장치 직경 [mm, (호)]',
               '매다는 장치 [가닥]', '로핑방식', '로핑 구조', '주 도르래 직경 [mm]', '권부각도 α [ º ]',
               '주 도르래 형상']
CTRL_SMALL = ['제어반 모델', '주 전원 [V]', '카 조명 전원 [V]', '회생전력 처리 형식', '인버터 용량 [kW]',
              '안전회로 전원 [V]', '제어회로 전원 [V]', '개문 착상 운전 (선택사양)',
              '개문 재착상 운전 (선택사양)', '주제어 기판', 'UCMP 감지부 모델', 'UCMP 제어부 모델',
              'UCMP 정지부 모델', '안전회로기판 모델', '비상 운전방식']

# ---- 경사형 ----
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
# 5. 선택지 및 입력 파싱 유틸
# =====================================================================
@st.cache_data
def cat_options_all(col):
    """승강기종류, 구분1 등 상단 항목은 전체 데이터 기준으로 선택지를 만듭니다."""
    return sorted(df[col].dropna().astype(str).unique().tolist())

@st.cache_data
def cat_options(gubun, col):
    """구분2 유형별 데이터 기준으로 선택지를 만듭니다."""
    sub = df[df["구분2"].astype(str) == gubun]
    return sorted(sub[col].dropna().astype(str).unique().tolist())

def parse_num(text):
    """
    입력한 문자열을 그대로 존중해서 숫자로 변환합니다.
    반환: (정상 여부, 값). 비어 있으면 (True, None).
    None이 들어와도 빈 문자열로 처리합니다.
    """
    t = (text or "").strip().replace(",", "")
    if t == "":
        return True, None
    try:
        return True, float(t)
    except ValueError:
        return False, None

# =====================================================================
# 6. 사이드바
# =====================================================================
st.sidebar.header("📋 세부 제원 입력")

# (1) 예측 분석 버튼: 가장 상단에 자리를 먼저 확보
btn_slot = st.sidebar.container()
st.sidebar.caption("'선택 안 함'은 조건에서 제외됩니다. 숫자 항목은 입력한 값 그대로 사용되며, 비워 두면 제외됩니다.")

user_inputs = {}

# (2) 최상단 고정 항목: 안전인증유형 → 승강기종류 → 구분1 → 구분2
for col in TOP_FIELDS:
    choice = st.sidebar.selectbox(col, ["선택 안 함"] + cat_options_all(col), key=f"top|{col}")
    user_inputs[col] = ("cat", choice) if choice != "선택 안 함" else None

gubun = st.sidebar.selectbox("구분2 (승강기 유형)", TYPES, index=0, key="gubun")

# (3) 나머지 기본 정보
with st.sidebar.expander(f"기본 정보 ({len(BASIC_REST)})", expanded=True):
    for col in BASIC_REST:
        choice = st.selectbox(col, ["선택 안 함"] + cat_options(gubun, col), key=f"cat|{gubun}|{col}")
        user_inputs[col] = ("cat", choice) if choice != "선택 안 함" else None

# (4) 유형별 그룹
for group_name, cols in LAYOUTS[gubun]:
    with st.sidebar.expander(f"{group_name} ({len(cols)})", expanded=False):
        for col in cols:
            if col in NUMERIC_COLS:
                raw = st.text_input(
                    col,
                    key=f"txt|{gubun}|{col}",
                    placeholder="숫자 입력 (비우면 제외)",
                )
                ok, num = parse_num(raw)
                if not ok:
                    st.warning(f"'{col}': 숫자만 입력해 주세요.")
                user_inputs[col] = ("num", num) if (ok and num is not None) else None
            else:
                choice = st.selectbox(col, ["선택 안 함"] + cat_options(gubun, col), key=f"cat|{gubun}|{col}")
                user_inputs[col] = ("cat", choice) if choice != "선택 안 함" else None

# 버튼은 위에서 만든 자리에 그려지므로 화면 최상단에 나타납니다
run = btn_slot.button("예측 분석 실행", type="primary", use_container_width=True)

# =====================================================================
# 7. 연관도 계산 및 부적합 예측
# =====================================================================
TOP_N = 200

if run:
    active = {c: v for c, v in user_inputs.items() if v is not None}
    base = df[df["구분2"].astype(str) == gubun]

    if not active:
        st.warning("제원을 하나 이상 선택하거나 입력해 주세요.")
    elif base.empty:
        st.warning(f"'{gubun}' 유형의 과거 데이터가 없습니다.")
    else:
        result_df = base.copy()
        result_df['유사도 점수'] = 0.0

        for col, (kind, val) in active.items():
            if kind == "cat":
                # 범주형: 정확히 일치하면 1점
                result_df.loc[result_df[col].astype(str) == val, '유사도 점수'] += 1
            else:
                # 숫자형: 입력값과 가까울수록 1점에 가깝게 (범위 기준 정규화)
                vals = pd.to_numeric(result_df[col], errors='coerce')
                rng = vals.max() - vals.min()
                if pd.isna(rng) or rng == 0:
                    rng = 1.0
                closeness = 1 - ((vals - val).abs() / rng).clip(0, 1)
                result_df['유사도 점수'] += closeness.fillna(0)

        top_matches = result_df.sort_values(by='유사도 점수', ascending=False).head(TOP_N)
        st.caption(f"'{gubun}' 데이터 {len(base):,}건 중 연관도 상위 {len(top_matches)}건으로 분석했습니다.")

        issues = top_matches['부적합내용'].dropna().tolist()
        ignore_keywords = ["안내사항", "개별인증 신청 시 기술원 홈페이지", "인증 신청 시 작성하는", "최신 설계심사 목록표를 참조하여"]
        filtered_issues = [str(i) for i in issues if not any(k in str(i) for k in ignore_keywords)]

        if filtered_issues:
            with st.spinner("사례를 분석하여 핵심 기준 위주로 정리하고 있습니다..."):
                grouping_prompt = f"""
                다음은 승강기 설계심사에서 발생한 부적합 사항 리스트입니다.

                억지로 개수를 맞추지 말고, 데이터에서 유의미하게 반복되는 주요 부적합 유형들만 추출하여 아래 형식을 엄격히 지켜 작성해 주세요.
                업체 고유의 내용이나 심사자만 아는 불확실한 부분은 과도하게 해석하지 말고 객관적으로 2~3문장 이내로만 요약하세요.

                중요: 안전기준 조항은 대분류(예: 6항, 9항)가 아닌, 부적합 내용과 직접적으로 매칭되는 세부 조항(예: 6.1.2항, 8.3.2.1항)까지 최대한 구체적으로 파악하여 기재해 주세요.

                ### [안전기준 세부 조항 번호 및 제목] 핵심 부적합 사유
                * **부적합 내용**: (객관적 요약 2~3문장)
                * **유사 발생 건수 예측**: 약 O건
                * **엔지니어링 조언**: (설계 시 확인해야 할 조언)

                위 분석 내용이 모두 끝난 후, 맨 마지막 줄에 반드시 '위에서 분석된 예상 부적합 사항'을 기반으로 사용자가 챗봇에 이어서 물어볼 만한 심층적인 '추천 질문' 딱 3개를 다음 형식으로만 적어주세요.
                [추천질문] (분석된 주요 부적합 사항의 구체적 원인이나 계산식, 규정 세부 내용을 묻는 질문)
                [추천질문] (분석된 내용 관련 질문 2)
                [추천질문] (분석된 내용 관련 질문 3)

                부적합 데이터: {filtered_issues[:200]}
                """
                response = model.generate_content(grouping_prompt)

                main_content, recommended_questions = [], []
                for line in response.text.split('\n'):
                    if line.startswith('[추천질문]'):
                        recommended_questions.append(line.replace('[추천질문]', '').strip())
                    else:
                        main_content.append(line)

                st.session_state.analysis_result = '\n'.join(main_content)
                st.session_state.recommended_questions = recommended_questions
        else:
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("recommended_questions", None)
            st.info("연관 데이터는 있으나 분석할 부적합 내용이 없습니다.")

# 분석 결과 출력 (세션 저장값 사용 - 추천 질문 버튼을 눌러도 유지)
if "analysis_result" in st.session_state:
    st.subheader("🧠 예상 주요 부적합 사항 (의미 기반 분석)")
    st.markdown("### ⚠️ 주의: 내용이 모호하거나 확실하지 않은 경우, 반드시 관련 안전기준 원문 항목을 직접 확인하시기 바랍니다.")
    st.markdown(st.session_state.analysis_result)

    if st.session_state.get("recommended_questions"):
        st.markdown("💡 **관련 추천 질문 (클릭 시 자동 질문)**")
        cols = st.columns(len(st.session_state.recommended_questions))
        for i, q in enumerate(st.session_state.recommended_questions):
            if cols[i].button(q, key=f"rq_{i}"):
                st.session_state.messages.append({"role": "user", "content": q})
                st.session_state.submit_from_button = True
                st.rerun()

st.divider()

# =====================================================================
# 8. 대화형 어시스턴트
# =====================================================================
st.subheader("💬 설계심사 규정 및 공학 추론 어시스턴트")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("질문을 입력하세요. (예: 매다는 장치 안전율 계산 관련 안전기준을 알려줘)")
trigger = False

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    trigger = True
elif st.session_state.submit_from_button:
    trigger = True
    st.session_state.submit_from_button = False

if trigger:
    latest_user_msg = st.session_state.messages[-1]["content"]
    with st.chat_message("user"):
        st.markdown(latest_user_msg)

    with st.chat_message("assistant"):
        with st.spinner("규정 문서를 탐색하고 공학적 추론을 진행 중입니다..."):
            system_prompt = f"""
            당신은 한국 승강기안전공단의 설계심사 최고 권위자이자 기계공학 전문가입니다.
            사용자의 질문에 대해 철저히 기계공학적 배경지식과, 제공된 [참고 기준 문서]에 근거하여 전문가적인 어조로 상세히 답변하세요.
            문서에 없는 공학적 계산(예: 오일러 좌굴 하중 등)을 물어보면 당신의 물리/기계공학 지식을 활용하여 수식과 함께 설명하세요.

            추가 지시사항:
            답변 시 출처를 명시할 때 파싱 과정에서 오차가 발생할 수 있는 '페이지 번호(예: p.24 등)'를 언급하는 것을 최대한 지양하고, 대신 문서 내의 명확한 '조항 번호(예: 6.1.8.1 등)'나 '항목 제목'을 기준으로 근거를 명시하십시오.

            [참고 기준 문서 내용]
            {st.session_state.pdf_context}

            사용자 질문: {latest_user_msg}
            """
            chat_response = model.generate_content(system_prompt)
            st.markdown(chat_response.text)
            st.session_state.messages.append({"role": "assistant", "content": chat_response.text})