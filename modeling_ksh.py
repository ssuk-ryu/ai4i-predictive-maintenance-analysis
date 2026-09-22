import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ==================================
# 1. 데이터 불러오기
# ==================================

df = pd.read_csv("ai4i2020.csv")


# ==================================
# 2. 고장 유형 설정
# ==================================

labels = ["TWF", "HDF", "PWF", "OSF", "RNF"]


print("고장 유형별 발생 개수")
print(df[labels].sum())


print("\n동시 발생 고장 개수")

failure_count = df[labels].sum(axis=1)

print(failure_count.value_counts().sort_index())

print("\n2개 이상 고장이 동시에 발생한 데이터:", (failure_count >= 2).sum())


# ==================================
# 3. 입력 변수 설정
# ==================================

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


X = df[features]

y = df[labels]


# ==================================
# 4. Type 숫자 변환
# ==================================

X = pd.get_dummies(X, columns=["Type"], dtype=int)


# ==================================
# 5. Train / Test 분리
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================================
# 6. Baseline
# ==================================

baseline = MultiOutputClassifier(DummyClassifier(strategy="most_frequent"))


# ==================================
# 7. Logistic Regression
# ==================================

logistic = MultiOutputClassifier(
    Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )
)


# ==================================
# 8. Decision Tree
# ==================================

tree = MultiOutputClassifier(
    DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
)


# ==================================
# 9. Random Forest
# ==================================

forest = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
)


# ==================================
# 10. 모델 학습
# ==================================

models = {
    "Baseline": baseline,
    "Logistic Regression": logistic,
    "Decision Tree": tree,
    "Random Forest": forest,
}


results = []


for model_name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    # ==============================
    # 고장 유형별 성능 평가
    # ==============================

    for i, label in enumerate(labels):

        precision = precision_score(y_test.iloc[:, i], pred[:, i], zero_division=0)

        recall = recall_score(y_test.iloc[:, i], pred[:, i], zero_division=0)

        f1 = f1_score(y_test.iloc[:, i], pred[:, i], zero_division=0)

        results.append([model_name, label, precision, recall, f1])

    # ==============================
    # Confusion Matrix
    # ==============================

    print("\n============================")
    print(model_name)
    print("============================")

    for i, label in enumerate(labels):

        print("\n", label)

        print(confusion_matrix(y_test.iloc[:, i], pred[:, i]))


# ==================================
# 11. 성능 비교표
# ==================================

result_df = pd.DataFrame(
    results, columns=["Model", "Failure Type", "Precision", "Recall", "F1-score"]
)


print("\n============================")
print("최종 모델 성능 비교")
print("============================")

print(result_df.round(3))
