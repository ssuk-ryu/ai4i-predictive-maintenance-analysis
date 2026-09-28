import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_data(file_path):
    df = pd.read_csv(file_path)

    print("[데이터 크기]")
    print(df.shape)

    print("\n[결측치]")
    print(df.isnull().sum())

    print("\n[Machine failure]")
    print(df["Machine failure"].value_counts())

    print("\n[고장 유형별 발생 건수]")
    print(df[TARGET_COLS].sum())

    return df


# 교수 피드백 추가
# Machine failure과 5개 고장 유형 존재 여부가 일치하는지 확인
def check_label_consistency(df):
    failure_any = (df[TARGET_COLS].sum(axis=1) > 0).astype(int)

    cross_table = pd.crosstab(
        df["Machine failure"],
        failure_any,
        rownames=["Machine failure"],
        colnames=["Failure type exists"],
    )

    print("\n[Machine failure × 고장 유형 존재 여부]")
    print(cross_table)

    mismatch = df[df["Machine failure"] != failure_any]

    print("\n[불일치 데이터 수]")
    print(len(mismatch))

    print("\n[불일치 데이터]")
    print(mismatch[["UDI", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]])

    return cross_table, mismatch


# 교수 피드백 추가
# 파생변수를 넣지 않은 원본 Feature 구성
def make_raw_x(df):
    feature_cols = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    X = df[feature_cols].copy()

    X = pd.get_dummies(X, columns=["Type"])

    return X


# 기존 함수
# 파생변수가 추가된 df를 받아 최종 X, y 구성
def make_xy(df):
    feature_cols = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temperature difference [K]",
        "Power [W]",
        "Wear torque",
    ]

    X = df[feature_cols].copy()
    y = df[TARGET_COLS].copy()

    X = pd.get_dummies(X, columns=["Type"])

    print("\n[X shape]")
    print(X.shape)

    print("\n[y shape]")
    print(y.shape)

    return X, y


# 기존 80:20 분할을
# Train 60 / Validation 20 / Test 20으로 수정
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print("\n[데이터 분할]")
    print("Train :", X_train.shape)
    print("Validation :", X_val.shape)
    print("Test :", X_test.shape)

    print("\n[Train 고장 건수]")
    print(y_train.sum())

    print("\n[Validation 고장 건수]")
    print(y_val.sum())

    print("\n[Test 고장 건수]")
    print(y_test.sum())

    return (X_train, X_val, X_test, y_train, y_val, y_test)
