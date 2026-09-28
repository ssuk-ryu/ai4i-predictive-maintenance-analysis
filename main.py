from data_utils import load_data, make_xy, split_data

from feature_utils import add_features

from model_utils import (
    train_random_forest,
    evaluate_models,
    zscore_baseline,
    get_feature_importance,
    predict_failure_type,
)

from threshold_utils import threshold_analysis, cost_analysis, find_best_threshold

from visualization import (
    plot_class_imbalance,
    plot_failure_boxplots,
    plot_pwf_scatter,
    plot_engineered_features,
    plot_model_comparison,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_hdf_errors,
    plot_threshold_cost,
)
from korean_font import set_korean_font

set_korean_font()
# =========================
# 1. 데이터 불러오기
# =========================

df = load_data("data/ai4i2020.csv")


# =========================
# 2. 클래스 불균형
# =========================

plot_class_imbalance(df)


# =========================
# 3. EDA
# =========================

plot_failure_boxplots(df)

plot_pwf_scatter(df)


# =========================
# 4. Feature Engineering
# =========================

df = add_features(df)

plot_engineered_features(df)


# =========================
# 5. X / y 구성
# =========================

X, y = make_xy(df)

X_train, X_test, y_train, y_test = split_data(X, y)


# =========================
# 6. Z-score Baseline
# =========================

zscore_result = zscore_baseline(X_train, X_test, y_train, y_test)

print("\n[Z-score Baseline]")
print(zscore_result)


# =========================
# 7. Random Forest
# =========================

models = train_random_forest(X_train, y_train)

rf_result, predictions, probabilities = evaluate_models(
    models, X_test, y_test, threshold=0.5
)

print("\n[Random Forest]")
print(rf_result)


# =========================
# 8. 모델 성능 비교
# =========================

plot_model_comparison(zscore_result, rf_result)

plot_confusion_matrices(rf_result)


# =========================
# 9. Feature Importance
# =========================

importance_result = get_feature_importance(models, X_train)

for target in ["HDF", "PWF", "OSF"]:

    print(f"\n[{target} Feature Importance]")
    print(importance_result[target].head(10))

plot_feature_importance(importance_result)


# =========================
# 10. FP / FN 분석
# =========================

plot_hdf_errors(df, X_test, y_test, predictions)


# =========================
# 11. TWF Threshold 분석
# =========================

thresholds_to_test = [0.05, 0.10, 0.15, 0.20, 0.30]

threshold_result = threshold_analysis(
    y_test["TWF"], probabilities["TWF"], thresholds_to_test
)

print("\n[TWF Threshold]")
print(threshold_result)


# =========================
# 12. 비용 분석
# =========================

cost_ratios = [5, 10, 20, 50]

cost_result = cost_analysis(threshold_result, cost_ratios)

print("\n[Threshold 비용 분석]")
print(cost_result)

best_threshold = find_best_threshold(cost_result, cost_ratios)

print("\n[비용별 최적 Threshold]")
print(best_threshold)

plot_threshold_cost(cost_result, cost_ratios)


# =========================
# 13. 운영 Threshold 설정
# =========================

# 실제 현장 비용정보가 없으므로
# 기본 Threshold는 0.5로 설정한다.
# TWF는 현장 비용비율에 따라
# 비용분석 결과의 Threshold로 변경 가능하다.

operating_thresholds = {"TWF": 0.50, "HDF": 0.50, "PWF": 0.50, "OSF": 0.50, "RNF": 0.50}


# =========================
# 14. 새로운 설비 데이터 판별
# =========================

new_data = df.loc[
    [X_test.index[0]],
    [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ],
].copy()


prediction_result = predict_failure_type(
    new_data, models, operating_thresholds, X.columns
)

print("\n[새로운 설비 데이터]")
print(new_data)

print("\n[고장 유형별 판별 결과]")
print(prediction_result)

print("끝")
