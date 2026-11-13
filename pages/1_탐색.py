import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------------
# 1. 데이터 설정: main.py와 동일한 UCI 유방암 진단 데이터
# ------------------------------------------------------------
DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
)

BASE_FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave points",
    "symmetry",
    "fractal dimension",
]

FEATURE_COLUMNS = (
    [f"{feature}_mean" for feature in BASE_FEATURES]
    + [f"{feature}_se" for feature in BASE_FEATURES]
    + [f"{feature}_worst" for feature in BASE_FEATURES]
)

COLUMNS = ["ID", "diagnosis"] + FEATURE_COLUMNS


@st.cache_data
def load_data():
    """UCI 원본 파일을 UTF-8로 읽고 열 이름을 지정합니다."""
    data = pd.read_csv(
        DATA_URL,
        header=None,
        names=COLUMNS,
        encoding="utf-8",
    )

    data["diagnosis"] = (
        data["diagnosis"].astype(str).str.strip().str.upper()
    )

    # 진단 코드의 표시용 이름
    data["진단 결과"] = data["diagnosis"].map(
        {"B": "양성(B)", "M": "악성(M)"}
    )
    return data


# ------------------------------------------------------------
# 2. 페이지 제목 및 데이터 로딩
# ------------------------------------------------------------
st.title("🔎 데이터 탐색")
st.write(
    "유방 종양의 진단 결과에 따라 세포핵의 형태학적 특징이 "
    "어떻게 달라지는지 그래프와 표로 살펴봅니다."
)

try:
    df = load_data()
except Exception as error:
    st.error("UCI 데이터 파일을 불러오지 못했습니다.")
    st.write("인터넷 연결과 데이터 주소를 확인한 뒤 다시 실행해 주세요.")
    st.exception(error)
    st.stop()

# 데이터의 기본 정보
total_count = len(df)
benign_count = int((df["diagnosis"] == "B").sum())
malignant_count = int((df["diagnosis"] == "M").sum())

st.subheader("📌 데이터 요약")
metric1, metric2, metric3 = st.columns(3)
metric1.metric("전체 종양 사례", f"{total_count:,}건")
metric2.metric("양성(B)", f"{benign_count:,}건")
metric3.metric("악성(M)", f"{malignant_count:,}건")

st.info(
    "이 유방암 데이터에는 나이, 평균 혈당, 고혈압, 심장병, BMI, "
    "흡연 상태 열이 없습니다. 따라서 아래에서는 실제 데이터에 포함된 "
    "세포핵의 반지름과 면적 등의 특징을 이용해 분석합니다."
)


# ------------------------------------------------------------
# 3. 세포핵 특징 분포: 히스토그램 2개
# ------------------------------------------------------------
st.divider()
st.header("1. 세포핵 특징의 분포")
st.write(
    "전체 종양 사례에서 세포핵의 평균 반지름과 평균 면적이 "
    "어떤 범위에 분포하는지 확인합니다."
)

left, right = st.columns(2)

with left:
    radius_hist = px.histogram(
        df,
        x="radius_mean",
        nbins=30,
        title="세포핵 평균 반지름 분포",
        labels={
            "radius_mean": "세포핵 평균 반지름",
            "count": "사례 수",
        },
        color_discrete_sequence=["#4C78A8"],
    )
    radius_hist.update_layout(
        xaxis_title="세포핵 평균 반지름",
        yaxis_title="사례 수",
        bargap=0.08,
    )
    st.plotly_chart(radius_hist, use_container_width=True)

with right:
    area_hist = px.histogram(
        df,
        x="area_mean",
        nbins=30,
        title="세포핵 평균 면적 분포",
        labels={
            "area_mean": "세포핵 평균 면적",
            "count": "사례 수",
        },
        color_discrete_sequence=["#F58518"],
    )
    area_hist.update_layout(
        xaxis_title="세포핵 평균 면적",
        yaxis_title="사례 수",
        bargap=0.08,
    )
    st.plotly_chart(area_hist, use_container_width=True)


# ------------------------------------------------------------
# 4. 양성/악성별 상자그림 2개와 평균값 표
# ------------------------------------------------------------
st.divider()
st.header("2. 양성 종양과 악성 종양의 특징 비교")
st.write(
    "양성(B)과 악성(M) 종양으로 나누어 세포핵의 평균 반지름과 "
    "평균 면적을 상자그림으로 비교합니다."
)

left, right = st.columns(2)

with left:
    radius_box = px.box(
        df,
        x="진단 결과",
        y="radius_mean",
        color="진단 결과",
        category_orders={"진단 결과": ["양성(B)", "악성(M)"]},
        title="진단 결과별 세포핵 평균 반지름",
        labels={
            "진단 결과": "종양 진단 결과",
            "radius_mean": "세포핵 평균 반지름",
        },
        color_discrete_map={
            "양성(B)": "#4C78A8",
            "악성(M)": "#E45756",
        },
    )
    radius_box.update_layout(showlegend=False)
    st.plotly_chart(radius_box, use_container_width=True)

with right:
    area_box = px.box(
        df,
        x="진단 결과",
        y="area_mean",
        color="진단 결과",
        category_orders={"진단 결과": ["양성(B)", "악성(M)"]},
        title="진단 결과별 세포핵 평균 면적",
        labels={
            "진단 결과": "종양 진단 결과",
            "area_mean": "세포핵 평균 면적",
        },
        color_discrete_map={
            "양성(B)": "#4C78A8",
            "악성(M)": "#E45756",
        },
    )
    area_box.update_layout(showlegend=False)
    st.plotly_chart(area_box, use_container_width=True)

mean_table = (
    df.groupby("진단 결과")[["radius_mean", "area_mean"]]
    .mean()
    .reindex(["양성(B)", "악성(M)"])
    .rename(
        columns={
            "radius_mean": "세포핵 평균 반지름의 평균",
            "area_mean": "세포핵 평균 면적의 평균",
        }
    )
    .round(3)
)

st.markdown("**양성·악성 그룹의 평균값**")
st.dataframe(mean_table, use_container_width=True)


# ------------------------------------------------------------
# 5. 세포핵 특징 수준별 악성 비율 막대그래프 2개
# ------------------------------------------------------------
st.divider()
st.header("3. 세포핵 특징 수준에 따른 악성 비율")
st.write(
    "고혈압과 심장병 여부 대신, 세포핵의 평균 반지름과 평균 면적을 "
    "각각 낮음·중간·높음 세 그룹으로 나누어 악성 비율을 비교합니다. "
    "그룹 구분은 각 특징의 값 분포를 기준으로 합니다."
)


def calculate_malignant_rate(data, feature):
    """특징값을 3개 그룹으로 나누고 각 그룹의 악성 비율을 계산합니다."""
    temp = data[[feature, "diagnosis"]].copy()

    temp["특징 수준"] = pd.qcut(
        temp[feature],
        q=3,
        labels=["낮음", "중간", "높음"],
        duplicates="drop",
    )

    result = (
        temp.groupby("특징 수준", observed=False)
        .agg(
            전체 사례 수=("diagnosis", "size"),
            악성 사례 수=(
                "diagnosis",
                lambda values: int((values == "M").sum()),
            ),
        )
        .reset_index()
    )

    result["악성 비율(%)"] = (
        result["악성 사례 수"] / result["전체 사례 수"] * 100
    )
    return result


left, right = st.columns(2)

with left:
    radius_rates = calculate_malignant_rate(df, "radius_mean")
    radius_rate_chart = px.bar(
        radius_rates,
        x="특징 수준",
        y="악성 비율(%)",
        text=radius_rates["악성 비율(%)"].map(
            lambda value: f"{value:.1f}%"
        ),
        title="평균 반지름 수준별 악성 비율",
        labels={"특징 수준": "세포핵 평균 반지름 수준"},
        color="특징 수준",
        category_orders={"특징 수준": ["낮음", "중간", "높음"]},
    )
    radius_rate_chart.update_layout(
        showlegend=False,
        yaxis_range=[0, 100],
        yaxis_title="악성 비율(%)",
    )
    st.plotly_chart(radius_rate_chart, use_container_width=True)

with right:
    area_rates = calculate_malignant_rate(df, "area_mean")
    area_rate_chart = px.bar(
        area_rates,
        x="특징 수준",
        y="악성 비율(%)",
        text=area_rates["악성 비율(%)"].map(
            lambda value: f"{value:.1f}%"
        ),
        title="평균 면적 수준별 악성 비율",
        labels={"특징 수준": "세포핵 평균 면적 수준"},
        color="특징 수준",
        category_orders={"특징 수준": ["낮음", "중간", "높음"]},
    )
    area_rate_chart.update_layout(
        showlegend=False,
        yaxis_range=[0, 100],
        yaxis_title="악성 비율(%)",
    )
    st.plotly_chart(area_rate_chart, use_container_width=True)


# ------------------------------------------------------------
# 6. 결측값이 있는 사례와 전체 사례 비교
# ------------------------------------------------------------
st.divider()
st.header("4. 결측값이 있는 사례 확인")
st.write(
    "이 데이터에는 BMI 열이 없으므로, 실제로 제공되는 30개 세포핵 특징 "
    "중 하나라도 비어 있는 사례를 찾아 전체 사례와 악성 비율을 비교합니다."
)

missing_rows = df[FEATURE_COLUMNS].isna().any(axis=1)
missing_df = df.loc[missing_rows]

overall_malignant_rate = (df["diagnosis"] == "M").mean() * 100

if len(missing_df) > 0:
    missing_malignant_rate = (
        (missing_df["diagnosis"] == "M").mean() * 100
    )
    missing_rate_text = round(missing_malignant_rate, 2)
else:
    missing_rate_text = "해당 사례 없음"

missing_comparison = pd.DataFrame(
    [
        {
            "구분": "결측값이 하나 이상 있는 사례",
            "사례 수": len(missing_df),
            "악성 사례 수": int((missing_df["diagnosis"] == "M").sum()),
            "악성 비율(%)": missing_rate_text,
        },
        {
            "구분": "전체 사례",
            "사례 수": len(df),
            "악성 사례 수": malignant_count,
            "악성 비율(%)": round(overall_malignant_rate, 2),
        },
    ]
)

st.dataframe(missing_comparison, use_container_width=True)

if missing_df.empty:
    st.success(
        "이 데이터의 세포핵 특징 열에는 결측값이 없습니다. "
        "따라서 결측 사례의 악성 비율은 계산할 수 없습니다."
    )


# ------------------------------------------------------------
# 7. 진단 결과별 사례 수 표
# ------------------------------------------------------------
st.divider()
st.header("5. 진단 결과별 사례 수")
st.write(
    "흡연 상태 열은 이 데이터에 없으므로, 대신 유방암 데이터의 "
    "진단 결과인 양성(B)과 악성(M)별 사례 수를 집계합니다."
)

diagnosis_table = (
    df["진단 결과"]
    .value_counts()
    .reindex(["양성(B)", "악성(M)"], fill_value=0)
    .rename_axis("진단 결과")
    .reset_index(name="사례 수")
)

diagnosis_table["비율(%)"] = (
    diagnosis_table["사례 수"] / total_count * 100
).round(2)

st.dataframe(diagnosis_table, use_container_width=True)

st.caption(
    "이 분석은 데이터의 분포와 특징 간 차이를 탐색하기 위한 것으로, "
    "개별 환자의 진단이나 의학적 판단을 대신하지 않습니다."
)
