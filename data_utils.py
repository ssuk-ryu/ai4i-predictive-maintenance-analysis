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

    X = pd.get_dummies(X, columns=["Type"], dtype=int)

    return X, y


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n[Train/Test]")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("\n[Train 고장 건수]")
    print(y_train.sum())

    print("\n[Test 고장 건수]")
    print(y_test.sum())

    return X_train, X_test, y_train, y_test
