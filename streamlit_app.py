import streamlit as st
import pandas as pd
import plotly.express as px
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Streamlit 페이지 설정
st.set_page_config(page_title="Job Skillset Extractor", layout="wide")

# OpenAI API 키 설정 (Streamlit Cloud의 Secrets & Variables 사용)
api_key = st.secrets["OPENAI_API_KEY"]

if not api_key:
    st.error("⚠️ OpenAI API 키가 설정되지 않았습니다. Streamlit Cloud의 Secrets & Variables 설정을 확인하세요.")
    st.stop()

# OpenAI LLM 인스턴스 생성
try:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=api_key)
    st.sidebar.success("✅ OpenAI API 연결 성공!")
except Exception as e:
    st.error(f"❌ OpenAI API 연결 실패: {e}")
    st.stop()

# 샘플 잡공고 데이터프레임 불러오기 또는 업로드
st.sidebar.header("Upload Job Postings CSV")
file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.sidebar.success("File Uploaded Successfully!")
else:
    # 샘플 데이터 생성 (Job Description 포함)
    data = {
        "Job Title": ["Data Scientist", "Software Engineer", "Product Manager"],
        "Job Description": [
            "Experience with Python, machine learning frameworks, and SQL. Must be familiar with data visualization tools like Tableau.",
            "Strong knowledge of Java, JavaScript, React, and cloud platforms such as AWS or GCP.",
            "Ability to define product strategy, work with engineering teams, and analyze market trends. Proficiency in Agile methodologies."
        ]
    }
    df = pd.DataFrame(data)

st.sidebar.header("Prompt Engineering")

# 사용자 정의 Prompt 입력
default_prompt = """
Extract a list of key skills from the following job description:

{job_description}

Provide the skills in a comma-separated format.
"""

prompt_template = st.sidebar.text_area("Customize the Prompt", value=default_prompt, height=150)

# 결과 저장을 위한 DataFrame
if "skillset_df" not in st.session_state:
    st.session_state["skillset_df"] = None

# 버튼 클릭 시 실행
if st.sidebar.button("Extract Skills"):
    skillset_results = []
    
    for _, row in df.iterrows():
        job_desc = row["Job Description"]
        prompt = PromptTemplate(input_variables=["job_description"], template=prompt_template)
        formatted_prompt = prompt.format(job_description=job_desc)
        
        try:
            response = llm.predict(formatted_prompt)
            skills = response.strip()
        except Exception as e:
            st.error(f"❌ OpenAI API 요청 실패: {e}")
            skills = "Error in extraction"
        
        skillset_results.append({"Job Title": row["Job Title"], "Extracted Skills": skills})
    
    # 결과 데이터프레임 생성 및 저장
    extracted_df = pd.DataFrame(skillset_results)
    st.session_state["skillset_df"] = extracted_df
    st.success("Skill extraction completed!")

# 결과 표시
st.header("Job Skillset Extraction Results")
if st.session_state["skillset_df"] is not None:
    st.dataframe(st.session_state["skillset_df"], use_container_width=True)
    
    # 스킬별 빈도수 시각화
    all_skills = ", ".join(st.session_state["skillset_df"]["Extracted Skills"]).split(", ")
    skill_counts = pd.Series(all_skills).value_counts().reset_index()
    skill_counts.columns = ["Skill", "Count"]
    
    st.subheader("Skill Frequency Analysis")
    fig = px.bar(skill_counts, x="Skill", y="Count", title="Extracted Skill Frequency", text="Count")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload a file and run skill extraction to see results.")
