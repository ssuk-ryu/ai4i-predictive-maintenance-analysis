import pandas as pd

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# --------------------------------------------------
# Threshold별 성능 분석
# --------------------------------------------------


def threshold_analysis(y_true, probability, thresholds):
    rows = []

    for threshold in thresholds:
        pred = (probability >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

        precision = precision_score(y_true, pred, zero_division=0)

        recall = recall_score(y_true, pred, zero_division=0)

        f1 = f1_score(y_true, pred, zero_division=0)

        rows.append([threshold, precision, recall, f1, tn, fp, fn, tp])

    result = pd.DataFrame(
        rows, columns=["Threshold", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP"]
    )

    return result


# --------------------------------------------------
# FP / FN 비용 민감도
# --------------------------------------------------


def cost_analysis(threshold_result, cost_ratios):
    rows = []

    for ratio in cost_ratios:

        for _, row in threshold_result.iterrows():

            cost = row["FP"] + ratio * row["FN"]

            rows.append([ratio, row["Threshold"], row["FP"], row["FN"], cost])

    result = pd.DataFrame(
        rows, columns=["FN Cost Ratio", "Threshold", "FP", "FN", "Cost"]
    )

    return result


# --------------------------------------------------
# 비용이 가장 작은 Threshold
# --------------------------------------------------


def find_best_threshold(cost_result, cost_ratio):
    data = cost_result[cost_result["FN Cost Ratio"] == cost_ratio]

    best = data.loc[data["Cost"].idxmin()]

    return best


# --------------------------------------------------
# 교수 피드백 추가
# Validation에서 선택한 Threshold를
# Test에서 최종 평가
# --------------------------------------------------


def evaluate_selected_threshold(y_true, probability, threshold):
    pred = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, pred, zero_division=0)

    recall = recall_score(y_true, pred, zero_division=0)

    f1 = f1_score(y_true, pred, zero_division=0)

    result = pd.DataFrame(
        [[threshold, precision, recall, f1, tn, fp, fn, tp]],
        columns=["Threshold", "Precision", "Recall", "F1", "TN", "FP", "FN", "TP"],
    )

    return result
