import streamlit as st
import pandas as pd
import google.generativeai as genai
import PyPDF2
import os

# =====================================================================
# 1. 기본 설정 및 엔진 초기화
# =====================================================================
st.set_page_config(page_title="승강기 설계심사 에이전트", layout="wide")

# 💡 외부 배포용으로 st.secrets를 활용하여 API 키 보호 적용 완료
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-3.8-flash')

st.title("🔍 승강기 설계심사 전문 에이전트")
st.markdown("입력된 제원을 바탕으로 보완 사항을 예측하고, 안전기준에 근거한 공학적 대화를 지원합니다.")

# =====================================================================
# 2. PDF 규정/가이드라인 자동 학습
# =====================================================================
@st.cache_data(show_spinner=False)
def load_pdf_knowledge():
    # 💡 폴더 경로를 현재 폴더를 의미하는 상대 경로(".")로 수정 완료
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
# 3. 좌측 사이드바: 제원 입력
# =====================================================================
st.sidebar.header("📋 세부 제원 입력")

@st.cache_data
def load_data():
    file_path = "개별설계심사현황 1회차(접수일자20230701_20260630)_최종 완성 자료(마스킹 완료)_최종.csv"
    df = pd.read_csv(file_path)
    
    # 💡 '로프'를 '와이어로프'로 묶어주는 코드 추가
    df['매다는 장치 [종류]'] = df['매다는 장치 [종류]'].replace('로프', '와이어로프')
    
    return df

df = load_data()
exclude_cols = ['접수일자', '접수번호', '부적합내용', '비고', '항목별 제출서류', '항목', 'Unnamed: 79', '신청자명', '모델']
input_cols = [col for col in df.columns if col not in exclude_cols]

user_inputs = {}
for col in input_cols:
    unique_vals = df[col].dropna().astype(str).unique().tolist()
    if unique_vals:
        options = ["선택 안 함"] + sorted(unique_vals)
        user_inputs[col] = st.sidebar.selectbox(f"{col}", options)

# =====================================================================
# 4. 부적합 예측 및 의미 기반 분석 (상태 저장 최적화)
# =====================================================================
if st.sidebar.button("예측 분석 실행", type="primary"):
    result_df = df.copy()
    result_df['유사도 점수'] = 0
    
    for col, selected_val in user_inputs.items():
        if selected_val != "선택 안 함":
            result_df.loc[result_df[col].astype(str) == selected_val, '유사도 점수'] += 1
            
    top_matches = result_df[result_df['유사도 점수'] > 0].sort_values(by='유사도 점수', ascending=False)
    
    if len(top_matches) == 0:
        st.warning("일치하는 과거 데이터가 없습니다.")
    else:
        issues = top_matches['부적합내용'].dropna().tolist()
        ignore_keywords = ["안내사항", "개별인증 신청 시 기술원 홈페이지", "인증 신청 시 작성하는", "최신 설계심사 목록표를 참조하여"]
        filtered_issues = [str(i) for i in issues if not any(keyword in str(i) for keyword in ignore_keywords)]
        
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
                
                # 텍스트와 추천 질문 분리 파싱 및 세션 저장 (API 중복 호출 방지)
                response_text = response.text
                main_content = []
                recommended_questions = []
                
                for line in response_text.split('\n'):
                    if line.startswith('[추천질문]'):
                        recommended_questions.append(line.replace('[추천질문]', '').strip())
                    else:
                        main_content.append(line)
                
                st.session_state.analysis_result = '\n'.join(main_content)
                st.session_state.recommended_questions = recommended_questions

# 분석 결과 출력 및 버튼 로직 (세션 저장된 데이터 활용)
if "analysis_result" in st.session_state:
    st.subheader("🧠 예상 주요 부적합 사항 (의미 기반 분석)")
    st.markdown("### ⚠️ 주의: 내용이 모호하거나 확실하지 않은 경우, 반드시 관련 안전기준 원문 항목을 직접 확인하시기 바랍니다.")
    st.markdown(st.session_state.analysis_result)
    
    if st.session_state.recommended_questions:
        st.markdown("💡 **관련 추천 질문 (클릭 시 자동 질문)**")
        cols = st.columns(len(st.session_state.recommended_questions))
        for i, q in enumerate(st.session_state.recommended_questions):
            if cols[i].button(q):
                st.session_state.messages.append({"role": "user", "content": q})
                st.session_state.submit_from_button = True
                st.rerun()

st.divider()

# =====================================================================
# 5. 승강기 공학 및 안전기준 특화 대화형 어시스턴트
# =====================================================================
st.subheader("💬 설계심사 규정 및 공학 추론 어시스턴트")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("질문을 입력하세요. (예: 매다는 장치 안전율 계산 관련 안전기준을 알려줘)")
trigger = False

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    trigger = True
elif st.session_state.get("submit_from_button"):
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