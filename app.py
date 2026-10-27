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
# 2. PDF 규정/가이드라인 자동 학습
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

with st.spinner("승강기 안전기준과 설계 가이드를 백그라운드에서 학습 중입니다..."):
    if "pdf_context" not in st.session_state:
        st.session_state.pdf_context = load_pdf_knowledge()

st.sidebar.success("📚 기준 문서 자동 학습 완료!")

# =====================================================================
# 3. 제원 데이터 로드 및 입력 그룹 정의
# =====================================================================
DATA_FILE = "개별설계심사현황 1회차(접수일자20230701_20260630)_최종 완성 자료(마스킹 완료)_최종.csv"

# (그룹명, 시작 컬럼명, 끝 컬럼명) - 시작~끝 사이의 컬럼이 해당 그룹에 들어갑니다
INPUT_GROUPS = [
    ("기본 정보", "안전인증유형", "심사원"),
    ("구동 시스템", "구동기 모델", "권부각도 α [ º ]"),
    ("제어 시스템", "제어반 모델", "강제감속운전 (선택사양)"),
    ("카 구조", "카 자중 [kg]", "문닫힘 안전장치"),
    ("승강로 구조", "구동기 지지대 규격  [mm]", "균형추/평형추 주행안내 레일 브라켓 최대간격 [mm]"),
]

EXCLUDE_COLS = ['접수일자', '접수번호', '부적합내용', '비고', '항목별 제출서류', '항목',
                'Unnamed: 79', '신청업체', '모델']

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, low_memory=False)
    # '로프'를 '와이어로프'로 묶어주는 처리
    df['매다는 장치 [종류]'] = df['매다는 장치 [종류]'].replace('로프', '와이어로프')
    return df

@st.cache_data
def get_options(df):
    """각 컬럼의 선택지를 한 번만 계산해 두어 재실행 시 속도를 높입니다."""
    return {col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in df.columns}

df = load_data()
options_map = get_options(df)

input_cols = [col for col in df.columns if col not in EXCLUDE_COLS and options_map.get(col)]

def get_group_cols(start, end):
    """시작 컬럼~끝 컬럼 사이의 컬럼 목록을 반환합니다 (입력 대상만)."""
    start_idx = df.columns.get_loc(start)
    end_idx = df.columns.get_loc(end)
    return [c for c in df.columns[start_idx:end_idx + 1] if c in input_cols]

# 그룹에 할당된 컬럼 집합 (나머지는 '기타'로 묶음)
grouped_cols = set()
resolved_groups = []
for group_name, start, end in INPUT_GROUPS:
    cols = get_group_cols(start, end)
    grouped_cols.update(cols)
    resolved_groups.append((group_name, cols))

other_cols = [c for c in input_cols if c not in grouped_cols]
if other_cols:
    resolved_groups.append(("기타", other_cols))

# =====================================================================
# 4. 사이드바: 예측 버튼을 상단에 배치 + 그룹별 접이식 제원 입력
# =====================================================================
st.sidebar.header("📋 세부 제원 입력")

# 버튼 자리를 먼저 확보해 두면, 아래에서 입력을 다 받아도 버튼은 위쪽에 그려집니다
btn_slot = st.sidebar.container()
st.sidebar.caption("그룹을 펼쳐서 제원을 선택하세요. '선택 안 함'은 조건에서 제외됩니다.")

user_inputs = {}
for group_name, cols in resolved_groups:
    with st.sidebar.expander(f"{group_name} ({len(cols)})", expanded=(group_name == "기본 정보")):
        for col in cols:
            options = ["선택 안 함"] + options_map[col]
            user_inputs[col] = st.selectbox(col, options, key=f"sel_{col}")

run = btn_slot.button("예측 분석 실행", type="primary", use_container_width=True)

# =====================================================================
# 5. 부적합 예측 및 의미 기반 분석
# =====================================================================
if run:
    result_df = df.copy()
    result_df['유사도 점수'] = 0

    for col, selected_val in user_inputs.items():
        if selected_val != "선택 안 함":
            result_df.loc[result_df[col].astype(str) == selected_val, '유사도 점수'] += 1

    top_matches = result_df[result_df['유사도 점수'] > 0].sort_values(by='유사도 점수', ascending=False)

    if len(top_matches) == 0:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("recommended_questions", None)
        st.warning("일치하는 과거 데이터가 없습니다.")
    else:
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
            st.info("일치하는 데이터는 있으나 분석할 부적합 내용이 없습니다.")

# 분석 결과 출력 (세션에 저장된 값 사용 - 버튼을 다시 눌러도 유지됨)
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
# 6. 대화형 어시스턴트
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