import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 분류 모델 만들기")
st.write("나이, 혈당 등의 정보를 이용해 뇌졸중 여부를 예측하는 모델을 만들어봅니다.")

st.divider()

# -----------------------------
# 2. 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="UTF-8")
    return df

df = load_data()

# 열 이름 <-> 우리말 이름 대응표
name_map = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
reverse_name_map = {v: k for k, v in name_map.items()}

all_features_kor = list(name_map.values())
default_features_kor = ["나이", "평균 혈당", "고혈압", "심장병"]  # bmi 제외한 넷

# -----------------------------
# 3. 속성 선택
# -----------------------------
st.header("1️⃣ 입력 속성 고르기")

selected_kor = st.multiselect(
    "모델의 입력으로 사용할 속성을 골라주세요.",
    options=all_features_kor,
    default=default_features_kor
)

if len(selected_kor) < 2:
    st.warning("⚠️ 속성을 두 개 이상 골라주세요.")
    st.stop()

selected_cols = [reverse_name_map[k] for k in selected_kor]
use_bmi = "bmi" in selected_cols

st.divider()

# -----------------------------
# 4. 데이터 준비: id 순 정렬 후 10명씩 묶어 앞 3명 테스트용
# -----------------------------
st.header("2️⃣ 데이터 나누기")

df_sorted = df.sort_values("id").reset_index(drop=True)

# 10명씩 묶었을 때, 그룹 안에서의 위치(0~9)를 계산
position_in_group = df_sorted.index % 10
is_test = position_in_group < 3  # 앞 3명은 테스트용

train_df = df_sorted[~is_test].copy()
test_df = df_sorted[is_test].copy()

st.write(f"- 학습용 사람 수: **{len(train_df):,} 명**")
st.write(f"- 테스트용 사람 수: **{len(test_df):,} 명**")

# -----------------------------
# 5. 결측치(bmi) 처리 : 훈련용 중앙값으로 채우기
# -----------------------------
if use_bmi:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)
    st.write(f"- 체질량지수(bmi)의 빈 값은 훈련용 중앙값 **{bmi_median:.2f}** 로 채웠습니다.")

X_train = train_df[selected_cols]
y_train = train_df["stroke"]
X_test = test_df[selected_cols]
y_test = test_df["stroke"]

st.divider()

# -----------------------------
# 6. 모델 학습
# -----------------------------
st.header("3️⃣ 모델 학습과 정확도")

# 로지스틱 회귀
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# 의사결정트리 (질문 3번까지, 마지막 마디 5명 미만이면 그만)
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)
tree_model.fit(X_train, y_train)

# 더미 모델 (다수결로만 답하는 모델)
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)

# 정확도 계산 함수
def get_accuracies(model):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return train_acc, test_acc

log_train_acc, log_test_acc = get_accuracies(log_model)
tree_train_acc, tree_test_acc = get_accuracies(tree_model)
dummy_train_acc, dummy_test_acc = get_accuracies(dummy_model)

# -----------------------------
# 7. 정확도 카드 세 개
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("로지스틱 회귀(확률로 답하는 모델)\n테스트 정확도", f"{log_test_acc*100:.2f}%")
    st.caption(f"훈련 정확도: {log_train_acc*100:.2f}%  /  테스트 정확도: {log_test_acc*100:.2f}%")

with col2:
    st.metric("의사결정트리(질문으로 답하는 모델)\n테스트 정확도", f"{tree_test_acc*100:.2f}%")
    st.caption(f"훈련 정확도: {tree_train_acc*100:.2f}%  /  테스트 정확도: {tree_test_acc*100:.2f}%")

with col3:
    st.metric("다수결 모델(입력을 보지 않는 모델)\n테스트 정확도", f"{dummy_test_acc*100:.2f}%")
    st.caption(f"훈련 정확도: {dummy_train_acc*100:.2f}%  /  테스트 정확도: {dummy_test_acc*100:.2f}%")

st.divider()

# -----------------------------
# 8. 산점도 그리기 (두 속성 선택)
# -----------------------------
st.header("4️⃣ 산점도로 살펴보기")

col_x, col_y = st.columns(2)
with col_x:
    x_kor = st.selectbox("가로축으로 쓸 속성", selected_kor, index=0)
with col_y:
    remaining = [c for c in selected_kor if c != x_kor]
    y_kor = st.selectbox("세로축으로 쓸 속성", remaining, index=0)

x_col = reverse_name_map[x_kor]
y_col = reverse_name_map[y_kor]

# 두 축이 아닌 나머지 속성들은 테스트 데이터의 중앙값으로 고정
other_cols = [c for c in selected_cols if c not in [x_col, y_col]]
fixed_values = {}
for c in other_cols:
    fixed_values[c] = X_test[c].median()

if fixed_values:
    fixed_text = ", ".join([f"{name_map[c]} = {v:.2f}" for c, v in fixed_values.items()])
    st.write(f"📌 두 축이 아닌 속성은 테스트 데이터의 중앙값으로 고정했습니다: **{fixed_text}**")
else:
    st.write("📌 선택한 속성이 두 개뿐이라 고정할 다른 속성이 없습니다.")

# 산점도용 데이터프레임 (실제 뇌졸중 여부 포함)
plot_df = X_test.copy()
plot_df["실제_뇌졸중"] = y_test.map({0: "겪지 않음", 1: "겪음"}).values

fig = go.Figure()

for label, color in [("겪지 않음", "royalblue"), ("겪음", "crimson")]:
    subset = plot_df[plot_df["실제_뇌졸중"] == label]
    fig.add_trace(go.Scatter(
        x=subset[x_col], y=subset[y_col],
        mode="markers",
        name=label,
        marker=dict(color=color, size=7, opacity=0.7)
    ))

# ---- 결정 경계용 격자 만들기 ----
x_min, x_max = X_test[x_col].min(), X_test[x_col].max()
y_min, y_max = X_test[y_col].min(), X_test[y_col].max()

x_range = np.linspace(x_min, x_max, 200)
y_range = np.linspace(y_min, y_max, 200)
xx, yy = np.meshgrid(x_range, y_range)

# 격자 위의 모든 점에 대해 나머지 속성은 고정값으로 채운 입력 데이터 생성
grid_df = pd.DataFrame({x_col: xx.ravel(), y_col: yy.ravel()})
for c in other_cols:
    grid_df[c] = fixed_values[c]
grid_df = grid_df[selected_cols]  # 학습할 때와 같은 열 순서로 맞추기

# ---- 의사결정트리 배경 칠하기 ----
tree_pred_grid = tree_model.predict(grid_df).reshape(xx.shape)
fig.add_trace(go.Contour(
    x=x_range, y=y_range, z=tree_pred_grid,
    showscale=False,
    opacity=0.25,
    colorscale=[[0, "royalblue"], [1, "crimson"]],
    contours=dict(coloring="fill"),
    line=dict(width=0),
    name="의사결정트리 영역",
    hoverinfo="skip"
))

# ---- 로지스틱 회귀 0.5 확률 경계선 그리기 ----
proba_grid = log_model.predict_proba(grid_df)[:, 1].reshape(xx.shape)

# 0.5를 지나는 등고선을 찾아서 그 좌표를 선으로 표시
boundary_fig = go.Figure(data=go.Contour(
    x=x_range, y=y_range, z=proba_grid,
    contours=dict(start=0.5, end=0.5, size=0.1)
))
# plotly의 contour 결과에서 실제 좌표를 뽑기 어려우므로, 대신 산점도 위에 등고선을 겹쳐 그림
fig.add_trace(go.Contour(
    x=x_range, y=y_range, z=proba_grid,
    showscale=False,
    contours=dict(
        start=0.5, end=0.5, size=0.1,
        coloring="lines"
    ),
    line=dict(width=3, color="black"),
    name="로지스틱 회귀 경계선(0.5)",
    hoverinfo="skip"
))

# 경계선이 화면 안에 존재하는지 확인
boundary_exists = (proba_grid.min() <= 0.5 <= proba_grid.max())
if not boundary_exists:
    st.info("ℹ️ 로지스틱 회귀의 0.5 경계선이 이 그림 범위 안에 나타나지 않습니다.")

fig.update_layout(
    title=f"{x_kor} vs {y_kor} 산점도와 결정 경계",
    xaxis_title=x_kor,
    yaxis_title=y_kor,
    legend_title="실제 뇌졸중 여부"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------
# 9. 의사결정트리 가지 그림 (Graphviz DOT)
# -----------------------------
st.header("5️⃣ 의사결정트리 뜯어보기")

tree = tree_model.tree_
feature_names = selected_cols

def build_dot(tree, feature_names, y_train):
    dot = "digraph Tree {\n"
    dot += 'node [shape=box, style="filled, rounded", fontname="malgun gothic"];\n'
    dot += 'edge [fontname="malgun gothic"];\n'

    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value  # 각 노드의 클래스별 샘플 수

    for i in range(n_nodes):
        n_samples = int(value[i].sum())
        n_stroke = int(value[i][0][1])  # stroke=1인 샘플 수
        ratio = n_stroke / n_samples if n_samples > 0 else 0

        is_leaf = children_left[i] == children_right[i]

        if is_leaf:
            # 답을 내는 마디: 다수결로 예/아니오 결정
            predicted = "겪음" if n_stroke > n_samples - n_stroke else "겪지 않음"
            color = "mistyrose" if predicted == "겪음" else "lightblue"
            label = f"샘플 {n_samples}명\\n뇌졸중 {n_stroke}명\\n비율 {ratio*100:.1f}%\\n답: {predicted}"
            dot += f'{i} [label="{label}", fillcolor="{color}"];\n'
        else:
            feat_kor = name_map[feature_names[feature[i]]]
            thresh = threshold[i]
            label = f"{feat_kor} <= {thresh:.2f} ?\\n샘플 {n_samples}명\\n뇌졸중 {n_stroke}명\\n비율 {ratio*100:.1f}%"
            dot += f'{i} [label="{label}", fillcolor="lightyellow"];\n'

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:
            dot += f'{i} -> {children_left[i]} [label="예"];\n'
            dot += f'{i} -> {children_right[i]} [label="아니요"];\n'

    dot += "}\n"
    return dot

dot_string = build_dot(tree, feature_names, y_train)
st.graphviz_chart(dot_string)

# -----------------------------
# 10. 트리 요약 정보
# -----------------------------
st.subheader("📋 트리 요약")

children_left = tree.children_left
children_right = tree.children_right
feature = tree.feature
value = tree.value

leaf_indices = [i for i in range(tree.node_count) if children_left[i] == children_right[i]]
n_leaves = len(leaf_indices)

n_no_leaves = 0
for i in leaf_indices:
    n_samples = int(value[i].sum())
    n_stroke = int(value[i][0][1])
    predicted = "겪음" if n_stroke > n_samples - n_stroke else "겪지 않음"
    if predicted == "겪지 않음":
        n_no_leaves += 1

st.write(f"- 답을 내는 마디(잎)는 모두 **{n_leaves}개**이고, 그중 **{n_no_leaves}개**가 '아님'이라고 답합니다.")

# 실제로 트리가 사용한 속성 찾기 (잎이 아닌 노드에서 사용된 feature)
used_feature_indices = set(feature[i] for i in range(tree.node_count) if children_left[i] != children_right[i])
used_features_kor = [name_map[feature_names[idx]] for idx in used_feature_indices]

if used_features_kor:
    st.write(f"- 고른 속성 중 이 나무가 실제로 물어본 것: **{', '.join(used_features_kor)}**")
else:
    st.write("- 이 나무는 어떤 속성도 실제로 묻지 않았습니다 (뿌리가 곧 잎입니다).")
