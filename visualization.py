import matplotlib.pyplot as plt
import pandas as pd

TARGET_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


# 시각화 1
def plot_class_imbalance(df):

    counts = df[TARGET_COLS].sum()

    plt.figure(figsize=(8, 5))

    counts.plot(kind="bar")

    plt.title("클래스 불균형")
    plt.xlabel("고장 유형")
    plt.ylabel("발생 건수")

    plt.tight_layout()
    plt.show()


# 시각화 2
def plot_failure_boxplots(df):

    configs = [
        ("TWF", "Tool wear [min]"),
        ("HDF", "Rotational speed [rpm]"),
        ("HDF", "Torque [Nm]"),
        ("OSF", "Tool wear [min]"),
        ("OSF", "Torque [Nm]"),
    ]

    for target, feature in configs:

        normal = df.loc[df[target] == 0, feature]

        failure = df.loc[df[target] == 1, feature]

        plt.figure(figsize=(6, 5))

        plt.boxplot([normal, failure], tick_labels=["비고장", target])

        plt.title("고장 유형별 Boxplot")
        plt.ylabel(feature)

        plt.tight_layout()
        plt.show()


# 시각화 3
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

    plt.scatter(failure["Rotational speed [rpm]"], failure["Torque [Nm]"], label="PWF")

    plt.title("Rotational speed × Torque 산점도")
    plt.xlabel("Rotational speed [rpm]")
    plt.ylabel("Torque [Nm]")

    plt.legend()
    plt.tight_layout()
    plt.show()


# 시각화 4
def plot_engineered_features(df):

    configs = [
        ("HDF", "Temperature difference [K]"),
        ("PWF", "Power [W]"),
        ("OSF", "Wear torque"),
    ]

    for target, feature in configs:

        normal = df.loc[df[target] == 0, feature]

        failure = df.loc[df[target] == 1, feature]

        plt.figure(figsize=(6, 5))

        plt.boxplot([normal, failure], tick_labels=[f"비{target}", target])

        plt.title("파생변수 분포")
        plt.ylabel(feature)

        plt.tight_layout()
        plt.show()


# 시각화 5
def plot_model_comparison(zscore_result, rf_result):

    merged = pd.merge(
        zscore_result[["Failure type", "F1"]],
        rf_result[["Failure type", "F1"]],
        on="Failure type",
        suffixes=("_Zscore", "_RF"),
    )

    merged = merged.set_index("Failure type")

    merged.plot(kind="bar", figsize=(8, 5))

    plt.title("Z-score vs Random Forest F1-score")
    plt.xlabel("고장 유형")
    plt.ylabel("F1-score")

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# 시각화 6
def plot_confusion_matrices(rf_result):

    for _, row in rf_result.iterrows():

        matrix = [[row["TN"], row["FP"]], [row["FN"], row["TP"]]]

        fig, ax = plt.subplots(figsize=(5, 4))

        image = ax.imshow(matrix)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(matrix[i][j]), ha="center", va="center")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(["정상 예측", "고장 예측"])

        ax.set_yticklabels(["실제 정상", "실제 고장"])

        plt.title("Confusion Matrix")

        plt.colorbar(image)
        plt.tight_layout()
        plt.show()


# 시각화 7
def plot_feature_importance(importance_result, top_n=8):

    for target in ["HDF", "PWF", "OSF"]:

        importance = importance_result[target].head(top_n).sort_values("Importance")

        plt.figure(figsize=(8, 5))

        plt.barh(importance["Feature"], importance["Importance"])

        plt.title("Feature Importance")
        plt.xlabel("Importance")

        plt.tight_layout()
        plt.show()


# 시각화 8
def plot_hdf_errors(df, X_test, y_test, predictions):

    test_data = df.loc[X_test.index].copy()

    test_data["Actual HDF"] = y_test["HDF"]

    test_data["Predicted HDF"] = predictions["HDF"]

    normal = test_data[test_data["Actual HDF"] == 0]

    hdf = test_data[test_data["Actual HDF"] == 1]

    fp = test_data[(test_data["Actual HDF"] == 0) & (test_data["Predicted HDF"] == 1)]

    fn = test_data[(test_data["Actual HDF"] == 1) & (test_data["Predicted HDF"] == 0)]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        normal["Temperature difference [K]"],
        normal["Rotational speed [rpm]"],
        alpha=0.2,
        label="비HDF",
    )

    plt.scatter(
        hdf["Temperature difference [K]"], hdf["Rotational speed [rpm]"], label="HDF"
    )

    plt.scatter(
        fp["Temperature difference [K]"],
        fp["Rotational speed [rpm]"],
        marker="X",
        s=150,
        label="FP",
    )

    plt.scatter(
        fn["Temperature difference [K]"],
        fn["Rotational speed [rpm]"],
        marker="*",
        s=180,
        label="FN",
    )

    plt.title("HDF 오류 사례 산점도")
    plt.xlabel("Temperature difference [K]")
    plt.ylabel("Rotational speed [rpm]")

    plt.legend()
    plt.tight_layout()
    plt.show()


# 시각화 9
def plot_threshold_cost(cost_result, cost_ratios):

    plt.figure(figsize=(8, 5))

    for ratio in cost_ratios:

        plt.plot(
            cost_result["Threshold"],
            cost_result[f"Cost_FN_{ratio}"],
            marker="o",
            label=f"FN 비용 {ratio}배",
        )

    plt.title("비용 민감도 그래프")
    plt.xlabel("Threshold")
    plt.ylabel("Normalized Cost")

    plt.legend()
    plt.tight_layout()
    plt.show()
