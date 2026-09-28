import pandas as pd

from sklearn.metrics import confusion_matrix


def threshold_analysis(y_true, probability, thresholds):

    rows = []

    for threshold in thresholds:

        pred = (probability >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

        rows.append([threshold, tn, fp, fn, tp])

    return pd.DataFrame(rows, columns=["Threshold", "TN", "FP", "FN", "TP"])


def cost_analysis(threshold_result, cost_ratios):

    result = threshold_result.copy()

    for ratio in cost_ratios:

        result[f"Cost_FN_{ratio}"] = result["FP"] + ratio * result["FN"]

    return result


def find_best_threshold(cost_result, cost_ratios):

    rows = []

    for ratio in cost_ratios:

        column = f"Cost_FN_{ratio}"

        idx = cost_result[column].idxmin()

        rows.append(
            [ratio, cost_result.loc[idx, "Threshold"], cost_result.loc[idx, column]]
        )

    return pd.DataFrame(
        rows, columns=["FN/FP Cost Ratio", "Best Threshold", "Minimum Cost"]
    )
