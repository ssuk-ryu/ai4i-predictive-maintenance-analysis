import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from feature_utils import add_features

# --------------------------------------------------
# Random Forest 학습
# --------------------------------------------------


def train_random_forest(X_train, y_train):
    models = {}

    for target in y_train.columns:
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )

        model.fit(X_train, y_train[target])

        models[target] = model

    return models


# --------------------------------------------------
# Random Forest 평가
# --------------------------------------------------


def evaluate_models(models, X_test, y_test, threshold=0.5):
    rows = []

    predictions = {}
    probabilities = {}

    for target in y_test.columns:
        model = models[target]

        probability = model.predict_proba(X_test)[:, 1]

        pred = (probability >= threshold).astype(int)

        predictions[target] = pred
        probabilities[target] = probability

        tn, fp, fn, tp = confusion_matrix(y_test[target], pred, labels=[0, 1]).ravel()

        precision = precision_score(y_test[target], pred, zero_division=0)

        recall = recall_score(y_test[target], pred, zero_division=0)

        f1 = f1_score(y_test[target], pred, zero_division=0)

        rows.append([target, precision, recall, f1, tn, fp, fn, tp])

    result = pd.DataFrame(
        rows,
        columns=["Failure Type", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP"],
    )

    return (result, predictions, probabilities)


# --------------------------------------------------
# 기존 Z-score Baseline
# --------------------------------------------------


def zscore_baseline(X_train, X_test, y_train, y_test, threshold=3):
    feature_map = {
        "TWF": ["Tool wear [min]"],
        "HDF": ["Temperature difference [K]", "Rotational speed [rpm]"],
        "PWF": ["Power [W]"],
        "OSF": ["Wear torque"],
        "RNF": [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Temperature difference [K]",
            "Power [W]",
            "Wear torque",
        ],
    }

    rows = []

    for target in y_train.columns:
        features = feature_map[target]

        # 해당 고장 유형이 발생하지 않은 Train 데이터
        normal_train = X_train.loc[y_train[target] == 0, features]

        mean = normal_train.mean()
        std = normal_train.std()

        z = ((X_test[features] - mean) / std).abs()

        pred = (z >= threshold).any(axis=1).astype(int)

        precision = precision_score(y_test[target], pred, zero_division=0)

        recall = recall_score(y_test[target], pred, zero_division=0)

        f1 = f1_score(y_test[target], pred, zero_division=0)

        rows.append([target, precision, recall, f1])

    result = pd.DataFrame(rows, columns=["Failure Type", "Precision", "Recall", "F1"])

    return result


# --------------------------------------------------
# 교수 피드백 추가
# Dummy Classifier Baseline
# --------------------------------------------------


def dummy_baseline(X_train, X_test, y_train, y_test):
    rows = []

    for target in y_train.columns:
        model = DummyClassifier(strategy="most_frequent")

        model.fit(X_train, y_train[target])

        pred = model.predict(X_test)

        precision = precision_score(y_test[target], pred, zero_division=0)

        recall = recall_score(y_test[target], pred, zero_division=0)

        f1 = f1_score(y_test[target], pred, zero_division=0)

        rows.append([target, precision, recall, f1])

    result = pd.DataFrame(rows, columns=["Failure Type", "Precision", "Recall", "F1"])

    return result


# --------------------------------------------------
# 교수 피드백 추가
# 원본 Feature RF vs 파생변수 추가 RF
# --------------------------------------------------


def compare_feature_engineering(
    X_train_raw, X_test_raw, X_train_feature, X_test_feature, y_train, y_test
):
    rows = []

    for target in y_train.columns:

        # 원본 Feature만 사용
        raw_model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )

        raw_model.fit(X_train_raw, y_train[target])

        raw_pred = raw_model.predict(X_test_raw)

        raw_f1 = f1_score(y_test[target], raw_pred, zero_division=0)

        # 파생변수 포함
        feature_model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )

        feature_model.fit(X_train_feature, y_train[target])

        feature_pred = feature_model.predict(X_test_feature)

        feature_f1 = f1_score(y_test[target], feature_pred, zero_division=0)

        rows.append([target, raw_f1, feature_f1, feature_f1 - raw_f1])

    result = pd.DataFrame(
        rows, columns=["Failure Type", "Raw F1", "Engineered F1", "F1 Change"]
    )

    return result


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------


def get_feature_importance(models, feature_columns):
    results = {}

    # 실제 판별 성능이 확보된 고장 유형만 해석
    for target in ["HDF", "PWF", "OSF"]:
        model = models[target]

        importance = pd.DataFrame(
            {"Feature": feature_columns, "Importance": model.feature_importances_}
        )

        importance = importance.sort_values("Importance", ascending=False)

        results[target] = importance

    return results


# --------------------------------------------------
# 새로운 데이터의 고장 유형 판별
# --------------------------------------------------


def predict_failure_type(new_data, models, thresholds, feature_columns):
    data = add_features(new_data)

    data = pd.get_dummies(data, columns=["Type"])

    data = data.reindex(columns=feature_columns, fill_value=0)

    result = {}

    for target, model in models.items():

        probability = model.predict_proba(data)[:, 1]

        threshold = thresholds.get(target, 0.5)

        warning = (probability >= threshold).astype(int)

        result[target] = {
            "Probability": probability[0],
            "Threshold": threshold,
            "Warning": warning[0],
        }

    return pd.DataFrame(result).T
