import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# 1. 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 탐색",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 데이터 탐색")
st.write("뇌졸중 데이터를 다양한 관점에서 살펴봅니다.")

st.divider()

# -----------------------------
# 2. 데이터 불러오기 (첫 화면과 동일)
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="UTF-8")
    return df

df = load_data()

# -----------------------------
# 3. 나이 / 평균 혈당 분포 히스토그램
# -----------------------------
st.header("1️⃣ 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df, x="age", nbins=30,
        title="나이 분포",
        labels={"age": "나이"}
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df, x="avg_glucose_level", nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose, use_container_width=True)

st.divider()

# -----------------------------
# 4. 뇌졸중 여부에 따른 나이 / 평균 혈당 상자그림
# -----------------------------
st.header("2️⃣ 뇌졸중 여부에 따른 나이·평균 혈당 비교")

# stroke 값(0, 1)을 보기 좋은 문자로 바꾼 열 추가
df["뇌졸중_여부"] = df["stroke"].map({0: "겪지 않음", 1: "겪음"})

col3, col4 = st.columns(2)

with col3:
    fig_box_age = px.box(
        df, x="뇌졸중_여부", y="age",
        title="뇌졸중 여부별 나이 분포",
        labels={"뇌졸중_여부": "뇌졸중 여부", "age": "나이"}
    )
    st.plotly_chart(fig_box_age, use_container_width=True)

with col4:
    fig_box_glucose = px.box(
        df, x="뇌졸중_여부", y="avg_glucose_level",
        title="뇌졸중 여부별 평균 혈당 분포",
        labels={"뇌졸중_여부": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_box_glucose, use_container_width=True)

# 두 그룹의 평균값 표
st.subheader("📊 두 그룹의 평균값")
mean_table = df.groupby("뇌졸중_여부")[["age", "avg_glucose_level"]].mean().reset_index()
mean_table.columns = ["뇌졸중 여부", "평균 나이", "평균 혈당"]
st.dataframe(mean_table, use_container_width=True)

st.divider()

# -----------------------------
# 5. 고혈압 / 심장병 유무에 따른 뇌졸중 비율
# -----------------------------
st.header("3️⃣ 고혈압·심장병 유무에 따른 뇌졸중 비율")

col5, col6 = st.columns(2)

with col5:
    # 고혈압 유무별 뇌졸중 비율(평균) 계산
    hyper_ratio = df.groupby("hypertension")["stroke"].mean().reset_index()
    hyper_ratio["hypertension"] = hyper_ratio["hypertension"].map({0: "없음", 1: "있음"})
    hyper_ratio.columns = ["고혈압 여부", "뇌졸중 비율"]

    fig_hyper = px.bar(
        hyper_ratio, x="고혈압 여부", y="뇌졸중 비율",
        title="고혈압 유무별 뇌졸중 비율",
        text_auto=".2%"
    )
    fig_hyper.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_hyper, use_container_width=True)

with col6:
    # 심장병 유무별 뇌졸중 비율(평균) 계산
    heart_ratio = df.groupby("heart_disease")["stroke"].mean().reset_index()
    heart_ratio["heart_disease"] = heart_ratio["heart_disease"].map({0: "없음", 1: "있음"})
    heart_ratio.columns = ["심장병 여부", "뇌졸중 비율"]

    fig_heart = px.bar(
        heart_ratio, x="심장병 여부", y="뇌졸중 비율",
        title="심장병 유무별 뇌졸중 비율",
        text_auto=".2%"
    )
    fig_heart.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# -----------------------------
# 6. bmi 결측치 그룹의 뇌졸중 비율 비교
# -----------------------------
st.header("4️⃣ 체질량지수(bmi) 결측치와 뇌졸중 비율")

bmi_missing_count = df["bmi"].isnull().sum()
bmi_missing_stroke_ratio = df[df["bmi"].isnull()]["stroke"].mean()
overall_stroke_ratio = df["stroke"].mean()

bmi_compare_table = pd.DataFrame({
    "구분": ["bmi 결측치 그룹", "전체 데이터"],
    "사람 수": [bmi_missing_count, len(df)],
    "뇌졸중 비율": [bmi_missing_stroke_ratio, overall_stroke_ratio]
})

# 비율을 % 형식 문자열로 보기 좋게 변환
bmi_compare_table["뇌졸중 비율"] = bmi_compare_table["뇌졸중 비율"].apply(lambda x: f"{x*100:.2f}%")

st.dataframe(bmi_compare_table, use_container_width=True)

st.divider()

# -----------------------------
# 7. 흡연 상태별 사람 수
# -----------------------------
st.header("5️⃣ 흡연 상태별 사람 수")

smoking_count = df["smoking_status"].value_counts().reset_index()
smoking_count.columns = ["흡연 상태", "사람 수"]

st.dataframe(smoking_count, use_container_width=True)
