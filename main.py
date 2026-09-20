import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# 1. 페이지 기본 설정
#    - page_title: 브라우저 탭에 표시되는 제목
#    - page_icon: 브라우저 탭 아이콘 (이모지 사용 가능)
#    - layout: 화면을 넓게 쓰도록 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# 2. 화면 맨 위 제목
# -----------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.write("뇌졸중 데이터를 살펴보고 예측 모델을 만들어보는 실습 공간입니다.")

st.divider()

# -----------------------------
# 3. 데이터 불러오기
#    - @st.cache_data : 데이터를 한 번만 불러오고 캐시에 저장해서
#      앱이 새로고침될 때마다 매번 다시 불러오지 않도록 함
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="UTF-8")
    return df

df = load_data()

# -----------------------------
# 4. 데이터 소개 화면
# -----------------------------
st.header("📌 데이터 소개")
st.write("이 데이터는 환자들의 여러 건강 정보를 바탕으로 뇌졸중 발생 여부를 기록한 자료입니다.")

# ---- 4-1. 큰 숫자 카드 네 개 ----
total_people = len(df)
total_columns = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = stroke_count / total_people * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 사람 수", f"{total_people:,} 명")
col2.metric("열 개수", f"{total_columns} 개")
col3.metric("뇌졸중 발생 수", f"{stroke_count:,} 명")
col4.metric("뇌졸중 비율", f"{stroke_ratio:.2f} %")

st.divider()

# ---- 4-2. 열 설명 표 ----
st.subheader("📋 열(컬럼) 설명")
st.write("‘우리말 뜻’ 칸은 비어 있어요. 교재를 참고해서 직접 채워보세요!")

# 각 열의 정보를 정리한 표 만들기
column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": ["" for _ in df.columns],   # 학생이 직접 채울 빈 칸
    "값의 종류": [str(df[col].dropna().unique()[:5]) + (" ..." if df[col].nunique() > 5 else "")
                for col in df.columns],
    "빈 값 개수": df.isnull().sum().values
})

# data_editor를 사용하면 '우리말 뜻' 칸을 화면에서 직접 입력할 수 있어요
edited_info = st.data_editor(
    column_info,
    use_container_width=True,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],  # 이 열들은 수정 못 하게 막기
    key="column_info_editor"
)

st.divider()

# ---- 4-3. 데이터 미리보기 (처음 5줄) ----
st.subheader("👀 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(), use_container_width=True)

st.divider()

# ---- 4-4. 데이터 출처 ----
st.subheader("📚 데이터 출처")
st.text_area(
    "아래에 교재를 참고하여 데이터 출처를 적어보세요.",
    placeholder="여기에 데이터 출처를 입력하세요...",
    height=100
)
