import json
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
APP_TITLE = "유방암 악성 & 양성 분류 실습실"
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔬",
    layout="wide",
)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
)

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "데이터_소개_기록.json"

# UCI 원본 wdbc.data에는 열 이름이 없으므로 직접 지정합니다.
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

FEATURE_NAMES = (
    [f"{name}_mean" for name in BASE_FEATURES]
    + [f"{name}_se" for name in BASE_FEATURES]
    + [f"{name}_worst" for name in BASE_FEATURES]
)

COLUMNS = ["ID", "diagnosis"] + FEATURE_NAMES


# --------------------------------------------------
# 기록 불러오기 / 저장하기
# --------------------------------------------------
def load_record():
    """저장된 우리말 뜻과 데이터 출처를 JSON에서 불러옵니다."""
    if not RECORD_FILE.exists():
        return {}, ""

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        return saved.get("meanings", {}), saved.get("source_text", "")
    except (json.JSONDecodeError, OSError):
        return {}, ""


def save_record(meanings, source_text):
    """우리말 뜻과 출처 입력 내용을 JSON 파일로 저장합니다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "meanings": meanings,
        "source_text": source_text,
    }
    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=4)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(
        DATA_URL,
        header=None,
        names=COLUMNS,
        encoding="utf-8",
    )
    return data


# --------------------------------------------------
# 화면
# --------------------------------------------------
st.title(f"🔬 {APP_TITLE}")
st.caption(
    "세포핵의 형태학적 특징을 살펴보고 유방암 진단 데이터를 이해하는 공간입니다."
)

try:
    df = load_data()
except Exception as error:
    st.error("UCI 저장소에서 데이터를 불러오지 못했습니다.")
    st.write(
        "인터넷 연결을 확인한 뒤 다시 실행해 주세요. "
        "학교 네트워크에서 원본 주소 접속이 차단될 수도 있습니다."
    )
    st.exception(error)
    st.stop()

# 실제 데이터의 진단값은 M(악성), B(양성)입니다.
diagnosis = df["diagnosis"].astype(str).str.strip().str.upper()
total_people = len(df)
column_count = len(df.columns)
malignant_count = int((diagnosis == "M").sum())
malignant_ratio = malignant_count / total_people * 100 if total_people else 0

st.subheader("📊 데이터 한눈에 보기")
card1, card2, card3, card4 = st.columns(4)

card1.metric("전체 사람 수", f"{total_people:,}명")
card2.metric("전체 열 개수", f"{column_count:,}개")
card3.metric("악성(M)인 사람 수", f"{malignant_count:,}명")
card4.metric("악성(M) 비율", f"{malignant_ratio:.1f}%")

st.info(
    "이 데이터의 정답 열은 `diagnosis`이며, M은 악성(Malignant), "
    "B는 양성(Benign)을 뜻합니다. 따라서 원래 요청에 적힌 "
    "`stroke`와 `bmi` 열은 이 유방암 진단 데이터에 존재하지 않습니다."
)

st.divider()
st.subheader("📚 데이터 열 살펴보기")
st.write(
    "각 열의 우리말 뜻은 교재를 참고해 직접 입력할 수 있습니다. "
    "입력한 내용은 저장 버튼을 누르면 JSON 파일에 기록됩니다."
)

saved_meanings, saved_source = load_record()

# ID와 diagnosis를 포함해 실제 데이터의 모든 열을 표에 표시합니다.
table_rows = []
for column in df.columns:
    if column == "ID":
        value_type = "정수형"
    elif column == "diagnosis":
        value_type = "범주형"
    else:
        value_type = "실수형"

    table_rows.append(
        {
            "열 이름": column,
            "우리말 뜻": saved_meanings.get(column, ""),
            "값의 종류": value_type,
            "빈 값 개수": int(df[column].isna().sum()),
        }
    )

meaning_df = pd.DataFrame(table_rows)

with st.form("meaning_form"):
    edited_meaning_df = st.data_editor(
        meaning_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "열 이름": st.column_config.TextColumn(
                "열 이름",
                disabled=True,
            ),
            "우리말 뜻": st.column_config.TextColumn(
                "우리말 뜻",
                help="교재를 참고해 각 열의 뜻을 직접 입력하세요.",
            ),
            "값의 종류": st.column_config.TextColumn(
                "값의 종류",
                disabled=True,
            ),
            "빈 값 개수": st.column_config.NumberColumn(
                "빈 값 개수",
                disabled=True,
                format="%d",
            ),
        },
    )

    st.markdown("**데이터 출처 기록**")
    source_text = st.text_area(
        "교재에 나온 데이터 출처를 입력하세요.",
        value=saved_source,
        height=110,
        placeholder="예: 교재에 제시된 데이터 출처를 여기에 입력",
    )

    save_clicked = st.form_submit_button(
        "💾 우리말 뜻과 출처 저장",
        use_container_width=True,
    )

if save_clicked:
    meanings_to_save = {}
    for _, row in edited_meaning_df.iterrows():
        meanings_to_save[str(row["열 이름"])] = str(row["우리말 뜻"]).strip()

    try:
        save_record(meanings_to_save, source_text)
        st.success(f"입력 내용이 `{RECORD_FILE}` 파일에 저장되었습니다.")
        st.rerun()
    except OSError as error:
        st.error("기록 파일을 저장하지 못했습니다.")
        st.exception(error)

st.caption(f"저장 위치: {RECORD_FILE.as_posix()}")

st.divider()
st.subheader("🔎 데이터 미리보기")
st.write("아래는 원본 데이터의 처음 다섯 행입니다.")
st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("🌐 데이터 출처")
st.write(
    "UCI Machine Learning Repository — "
    "Breast Cancer Wisconsin (Diagnostic)"
)
st.markdown(
    "[UCI 데이터셋 페이지 열기]"
    "(https://archive.ics.uci.edu/dataset/17/"
    "breast+cancer+wisconsin+diagnostic)"
)
st.caption(
    "참고: 원본 데이터는 569개 사례와 30개 수치형 특징으로 구성되며, "
    "진단값은 M(악성)과 B(양성)입니다."
)
