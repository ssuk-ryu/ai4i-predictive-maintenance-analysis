import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from feature_utils import add_features

TARGET_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def train_random_forest(X_train, y_train):

    models = {}

    for target in TARGET_COLS:

        model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )

        model.fit(X_train, y_train[target])

        models[target] = model

    return models


def evaluate_models(models, X_test, y_test, threshold=0.5):

    rows = []
    predictions = {}
    probabilities = {}

    for target, model in models.items():

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)

        precision = precision_score(y_test[target], pred, zero_division=0)

        recall = recall_score(y_test[target], pred, zero_division=0)

        f1 = f1_score(y_test[target], pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_test[target], pred, labels=[0, 1]).ravel()

        rows.append([target, precision, recall, f1, tn, fp, fn, tp])

        predictions[target] = pred
        probabilities[target] = proba

    result = pd.DataFrame(
        rows,
        columns=["Failure type", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP"],
    )

    return result, predictions, probabilities


def zscore_baseline(X_train, X_test, y_train, y_test):

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

    for target in TARGET_COLS:

        features = feature_map[target]
        # X_train에서 '정상 데이터의 센서값만'가져오기 위해
        # y_train을 조건으로 씀
        normal_train = X_train.loc[y_train[target] == 0, features]

        mean = normal_train.mean()
        std = normal_train.std().replace(0, 1)

        z = abs((X_test[features] - mean) / std)

        pred = (z >= 3).any(axis=1).astype(int)

        precision = precision_score(y_test[target], pred, zero_division=0)

        recall = recall_score(y_test[target], pred, zero_division=0)

        f1 = f1_score(y_test[target], pred, zero_division=0)

        rows.append([target, precision, recall, f1])

    return pd.DataFrame(rows, columns=["Failure type", "Precision", "Recall", "F1"])


def get_feature_importance(models, X_train):

    result = {}

    for target, model in models.items():

        importance = pd.DataFrame(
            {"Feature": X_train.columns, "Importance": model.feature_importances_}
        )

        importance = importance.sort_values("Importance", ascending=False)

        result[target] = importance

    return result


def predict_failure_type(new_data, models, thresholds, feature_columns):

    data = add_features(new_data)

    data = pd.get_dummies(data, columns=["Type"], dtype=int)

    data = data.reindex(columns=feature_columns, fill_value=0)

    rows = []

    for target, model in models.items():

        probability = model.predict_proba(data)[:, 1][0]

        threshold = thresholds[target]

        warning = int(probability >= threshold)

        rows.append([target, probability, threshold, warning])

    return pd.DataFrame(
        rows, columns=["Failure type", "Probability", "Threshold", "Warning"]
    )
