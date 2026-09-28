from data_utils import (
    load_data,
    check_label_consistency,
    make_raw_x,
    make_xy,
    split_data,
)

from feature_utils import add_features

from model_utils import (
    train_random_forest,
    evaluate_models,
    zscore_baseline,
    dummy_baseline,
    compare_feature_engineering,
    get_feature_importance,
    predict_failure_type,
)

from threshold_utils import (
    threshold_analysis,
    cost_analysis,
    find_best_threshold,
    evaluate_selected_threshold,
)

from visualization import (
    plot_class_imbalance,
    plot_failure_boxplots,
    plot_pwf_scatter,
    plot_engineered_features,
    plot_model_comparison,
    plot_feature_engineering_comparison,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_hdf_errors,
    plot_threshold_performance,
    plot_threshold_cost,
)

from korean_font import set_korean_font

# ==================================================
# 1. 한글 폰트 설정
# ==================================================

set_korean_font()


# ==================================================
# 2. 데이터 로드
# ==================================================

df = load_data("data/ai4i2020.csv")


# ==================================================
# 3. 라벨 일관성 확인
# ==================================================

label_table, mismatch = check_label_consistency(df)


# ==================================================
# 4. 기본 EDA 시각화
# ==================================================

plot_class_imbalance(df)

plot_failure_boxplots(df)

plot_pwf_scatter(df)


# ==================================================
# 5. 원본 Feature 구성
# 파생변수 비교용
# ==================================================

X_raw = make_raw_x(df)


# ==================================================
# 6. Feature Engineering
# ==================================================

df_feature = add_features(df)


# 파생변수 분포 시각화
plot_engineered_features(df_feature)


# ==================================================
# 7. 최종 X / y 구성
# ==================================================

X, y = make_xy(df_feature)


# ==================================================
# 8. Train / Validation / Test 분할
#
# Train      : 모델 학습
# Validation : Threshold 선택
# Test       : 최종 평가
# ==================================================

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)


# 원본 Feature 역시
# 동일한 index를 이용하여 분할
X_raw_train = X_raw.loc[X_train.index]

X_raw_val = X_raw.loc[X_val.index]

X_raw_test = X_raw.loc[X_test.index]


# ==================================================
# 9. Z-score Baseline
# ==================================================

zscore_result = zscore_baseline(X_train, X_test, y_train, y_test)

print("\n[Z-score Baseline]")

print(zscore_result)


# ==================================================
# 10. Dummy Baseline
# ==================================================

dummy_result = dummy_baseline(X_train, X_test, y_train, y_test)

print("\n[Dummy Baseline]")

print(dummy_result)


# ==================================================
# 11. Random Forest 학습
# ==================================================

models = train_random_forest(X_train, y_train)


# ==================================================
# 12. Random Forest 최종 Test 평가
# 기본 Threshold = 0.5
# ==================================================

rf_result, predictions, probabilities = evaluate_models(
    models, X_test, y_test, threshold=0.5
)

print("\n[Random Forest]")

print(rf_result)


# ==================================================
# 13. Baseline vs Random Forest 시각화
# ==================================================

plot_model_comparison(zscore_result, dummy_result, rf_result)


# ==================================================
# 14. 원본 Feature vs 파생변수 비교
#
# 동일 Train/Test index를 사용하여
# 파생변수의 효과를 비교
# ==================================================

feature_compare = compare_feature_engineering(
    X_raw_train, X_raw_test, X_train, X_test, y_train, y_test
)

print("\n[원본 Feature vs 파생변수 추가]")

print(feature_compare)


# 비교 시각화
plot_feature_engineering_comparison(feature_compare)


# ==================================================
# 15. Confusion Matrix
# ==================================================

plot_confusion_matrices(y_test, predictions)


# ==================================================
# 16. Feature Importance
#
# 실제 판별 성능이 확보된
# HDF / PWF / OSF 중심
# ==================================================

importance_results = get_feature_importance(models, X_train.columns)

for target, importance in importance_results.items():
    print(f"\n[{target} Feature Importance]")

    print(importance.head(10))


plot_feature_importance(importance_results)


# ==================================================
# 17. HDF FP / FN 오류 사례
#
# 새 Test 데이터에서
# FP와 FN을 자동으로 찾아 시각화
# ==================================================

plot_hdf_errors(df_feature, X_test, y_test, predictions)


# ==================================================
# 18. TWF Validation Threshold 분석
#
# 중요:
# Threshold 선택에는 Test를 사용하지 않는다.
# ==================================================

twf_model = models["TWF"]

twf_val_probability = twf_model.predict_proba(X_val)[:, 1]


threshold_candidates = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


twf_threshold_result = threshold_analysis(
    y_val["TWF"], twf_val_probability, threshold_candidates
)


print("\n[TWF Validation Threshold 분석]")

print(twf_threshold_result)


# Threshold에 따른
# Precision / Recall / F1 변화
plot_threshold_performance(twf_threshold_result)


# ==================================================
# 19. TWF 비용 민감도 분석
#
# 실제 금액이 아니라
# FP 비용 = 1을 기준으로 한 상대 비용
# ==================================================

cost_ratios = [5, 10, 20, 50]


twf_cost_result = cost_analysis(twf_threshold_result, cost_ratios)


print("\n[TWF 비용 민감도]")

print(twf_cost_result)


plot_threshold_cost(twf_cost_result)


# ==================================================
# 20. 예시 비용 시나리오
#
# FN 비용을 FP 비용의 10배로
# 가정했을 때 Validation에서
# 최소 비용 Threshold 선택
#
# 실제 현장 최적값이라는 의미는 아님
# ==================================================

best_threshold_info = find_best_threshold(twf_cost_result, cost_ratio=10)


selected_twf_threshold = best_threshold_info["Threshold"]


print("\n[TWF 선택 Threshold]")

print(selected_twf_threshold)


# ==================================================
# 21. 선택된 Threshold를
# Test에서 최종 평가
# ==================================================

twf_test_probability = twf_model.predict_proba(X_test)[:, 1]


twf_final_test = evaluate_selected_threshold(
    y_test["TWF"], twf_test_probability, selected_twf_threshold
)


print("\n[TWF 최종 Test 평가]")

print(twf_final_test)


# ==================================================
# 22. 최종 운영 Threshold 예시
#
# TWF는 비용비율 10배를 가정한
# 시나리오의 Threshold 사용
#
# 나머지는 기본 0.5
# ==================================================

thresholds_final = {
    "TWF": selected_twf_threshold,
    "HDF": 0.5,
    "PWF": 0.5,
    "OSF": 0.5,
    "RNF": 0.5,
}


# ==================================================
# 23. 실제 HDF 고장 사례를 이용한
# 최종 판별 함수 동작 예시
# ==================================================

hdf_indices = y_test[y_test["HDF"] == 1].index


sample_index = hdf_indices[0]


sample = df.loc[
    [sample_index],
    [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ],
]


prediction_result = predict_failure_type(
    sample, models, thresholds_final, X_train.columns
)


print("\n[예시 데이터 UDI]")

print(df.loc[sample_index, "UDI"])


print("\n[실제 고장 라벨]")

print(y_test.loc[sample_index])


print("\n[고장 유형 판별 결과]")

print(prediction_result)
