import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# ==================================================
# 1. 클래스 불균형
# ==================================================


def plot_class_imbalance(df):
    targets = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    counts = df[targets].sum()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values)

    plt.title("클래스 불균형")
    plt.xlabel("고장 유형")
    plt.ylabel("발생 건수")

    plt.tight_layout()
    plt.show()


# ==================================================
# 2. 고장 유형별 Boxplot
# ==================================================


def plot_failure_boxplots(df):
    configs = [
        ("TWF", "Tool wear [min]"),
        ("HDF", "Rotational speed [rpm]"),
        ("HDF", "Torque [Nm]"),
        ("OSF", "Tool wear [min]"),
        ("OSF", "Torque [Nm]"),
    ]

    for target, feature in configs:

        non_failure = df.loc[df[target] == 0, feature]

        failure = df.loc[df[target] == 1, feature]

        plt.figure(figsize=(7, 5))

        plt.boxplot([non_failure, failure], tick_labels=[f"비{target}", target])

        plt.title("고장 유형별 Boxplot")
        plt.xlabel(target)
        plt.ylabel(feature)

        plt.tight_layout()
        plt.show()


# ==================================================
# 3. PWF Rotational speed × Torque
# ==================================================


def plot_pwf_scatter(df):
    normal = df[df["PWF"] == 0]
    failure = df[df["PWF"] == 1]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        normal["Rotational speed [rpm]"],
        normal["Torque [Nm]"],
        alpha=0.3,
        label="비PWF",
    )

    plt.scatter(
        failure["Rotational speed [rpm]"],
        failure["Torque [Nm]"],
        alpha=0.8,
        label="PWF",
    )

    plt.title("Rotational speed × Torque 산점도")

    plt.xlabel("Rotational speed [rpm]")

    plt.ylabel("Torque [Nm]")

    plt.legend()
    plt.tight_layout()
    plt.show()


# ==================================================
# 4. 파생변수 분포
# ==================================================


def plot_engineered_features(df):
    configs = [
        ("HDF", "Temperature difference [K]"),
        ("PWF", "Power [W]"),
        ("OSF", "Wear torque"),
    ]

    for target, feature in configs:

        non_failure = df.loc[df[target] == 0, feature]

        failure = df.loc[df[target] == 1, feature]

        plt.figure(figsize=(7, 5))

        plt.boxplot([non_failure, failure], tick_labels=[f"비{target}", target])

        plt.title("파생변수 분포")
        plt.xlabel(target)
        plt.ylabel(feature)

        plt.tight_layout()
        plt.show()


# ==================================================
# 5. Z-score / Dummy / RF 비교
# 교수 피드백 반영
# ==================================================


def plot_model_comparison(zscore_result, dummy_result, rf_result):
    targets = rf_result["Failure Type"].tolist()

    zscore_f1 = zscore_result["F1"].values

    dummy_f1 = dummy_result["F1"].values

    rf_f1 = rf_result["F1"].values

    x = np.arange(len(targets))
    width = 0.25

    plt.figure(figsize=(9, 5))

    plt.bar(x - width, zscore_f1, width, label="Z-score")

    plt.bar(x, dummy_f1, width, label="Dummy")

    plt.bar(x + width, rf_f1, width, label="Random Forest")

    plt.xticks(x, targets)

    plt.ylim(0, 1.1)

    plt.xlabel("고장 유형")
    plt.ylabel("F1-score")

    plt.title("Baseline vs Random Forest F1-score")

    plt.legend()

    plt.tight_layout()
    plt.show()


# ==================================================
# 6. 원본 Feature vs 파생변수 추가
# 교수 피드백 반영
# ==================================================


def plot_feature_engineering_comparison(feature_compare):
    targets = feature_compare["Failure Type"].tolist()

    raw = feature_compare["Raw F1"].values

    engineered = feature_compare["Engineered F1"].values

    x = np.arange(len(targets))
    width = 0.35

    plt.figure(figsize=(9, 5))

    plt.bar(x - width / 2, raw, width, label="원본 Feature")

    plt.bar(x + width / 2, engineered, width, label="파생변수 추가")

    plt.xticks(x, targets)

    plt.ylim(0, 1.1)

    plt.xlabel("고장 유형")
    plt.ylabel("F1-score")

    plt.title("파생변수 추가 전후 F1-score")

    plt.legend()

    plt.tight_layout()
    plt.show()


# ==================================================
# 7. Confusion Matrix
# ==================================================


def plot_confusion_matrices(y_test, predictions):
    for target in y_test.columns:

        cm = confusion_matrix(y_test[target], predictions[target], labels=[0, 1])

        plt.figure(figsize=(5, 4))

        plt.imshow(cm)

        plt.title(f"Confusion Matrix - {target}")

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.xticks([0, 1], ["0", "1"])

        plt.yticks([0, 1], ["0", "1"])

        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center")

        plt.tight_layout()
        plt.show()


# ==================================================
# 8. Feature Importance
# HDF / PWF / OSF
# ==================================================


def plot_feature_importance(importance_results):
    for target, importance in importance_results.items():

        data = importance.head(8)

        plt.figure(figsize=(8, 5))

        plt.barh(data["Feature"], data["Importance"])

        plt.gca().invert_yaxis()

        plt.xlabel("Importance")

        plt.title(f"Feature Importance - {target}")

        plt.tight_layout()
        plt.show()


# ==================================================
# 9. HDF 오류 사례
# 새 Test 데이터 기준으로 자동 탐색
# ==================================================


def plot_hdf_errors(df, X_test, y_test, predictions):
    hdf_pred = pd.Series(predictions["HDF"], index=y_test.index)

    fp_index = y_test.index[(y_test["HDF"] == 0) & (hdf_pred == 1)]

    fn_index = y_test.index[(y_test["HDF"] == 1) & (hdf_pred == 0)]

    plt.figure(figsize=(8, 6))

    non_hdf = df.loc[y_test.index[y_test["HDF"] == 0]]

    hdf = df.loc[y_test.index[y_test["HDF"] == 1]]

    plt.scatter(
        non_hdf["Temperature difference [K]"],
        non_hdf["Rotational speed [rpm]"],
        alpha=0.3,
        label="비HDF",
    )

    plt.scatter(
        hdf["Temperature difference [K]"],
        hdf["Rotational speed [rpm]"],
        alpha=0.8,
        label="HDF",
    )

    if len(fp_index) > 0:
        plt.scatter(
            df.loc[fp_index, "Temperature difference [K]"],
            df.loc[fp_index, "Rotational speed [rpm]"],
            marker="X",
            s=150,
            label="FP",
        )

    if len(fn_index) > 0:
        plt.scatter(
            df.loc[fn_index, "Temperature difference [K]"],
            df.loc[fn_index, "Rotational speed [rpm]"],
            marker="*",
            s=180,
            label="FN",
        )

    plt.xlabel("Temperature difference [K]")

    plt.ylabel("Rotational speed [rpm]")

    plt.title("HDF 오류 사례 산점도")

    plt.legend()

    plt.tight_layout()
    plt.show()


# ==================================================
# 10. TWF Validation Threshold 성능
# 새로 추가
# ==================================================


def plot_threshold_performance(threshold_result):
    plt.figure(figsize=(8, 5))

    plt.plot(
        threshold_result["Threshold"],
        threshold_result["Precision"],
        marker="o",
        label="Precision",
    )

    plt.plot(
        threshold_result["Threshold"],
        threshold_result["Recall"],
        marker="o",
        label="Recall",
    )

    plt.plot(
        threshold_result["Threshold"], threshold_result["F1"], marker="o", label="F1"
    )

    plt.xlabel("Threshold")
    plt.ylabel("Score")

    plt.ylim(0, 1)

    plt.title("TWF Validation Threshold별 성능")

    plt.legend()

    plt.tight_layout()
    plt.show()


# ==================================================
# 11. 비용 민감도
# Validation 데이터 기준
# ==================================================


def plot_threshold_cost(cost_result):
    plt.figure(figsize=(8, 5))

    for ratio in sorted(cost_result["FN Cost Ratio"].unique()):
        data = cost_result[cost_result["FN Cost Ratio"] == ratio]

        plt.plot(
            data["Threshold"], data["Cost"], marker="o", label=f"FN 비용 {ratio}배"
        )

    plt.xlabel("Threshold")
    plt.ylabel("상대 비용")

    plt.title("비용 민감도 그래프")

    plt.legend()

    plt.tight_layout()
    plt.show()
