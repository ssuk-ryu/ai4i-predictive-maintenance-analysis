import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import platform
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, average_precision_score

# ==========================================================================

DATA_FILEB = "c:/Users/coco/Desktop/Class_materials/k뉴딜 팀플/ai4i2020.csv"
df = pd.read_csv(DATA_FILEB)


def 줄바꿈():
    print()
    print("=" * 80)
    print()


# ==========================================================================

failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# 각 고장 유형에서 값이 1인 데이터의 개수를 구했다
failure_count = df[failure_cols].sum()

# 전체 데이터에서 각 고장 유형이 차지하는 비율을 구했다
failure_ratio = (failure_count / len(df)) * 100

print("고장 유형별 발생 개수")
print(failure_count)
# 고장 유형별 발생 개수
# TWF     46
# HDF    115
# PWF     95
# OSF     98
# RNF     19
# dtype: int64

print("\n고장 유형별 발생 비율(%)")
print(failure_ratio.round(2))
# 고장 유형별 발생 비율(%)
# TWF    0.46
# HDF    1.15
# PWF    0.95
# OSF    0.98
# RNF    0.19
# dtype: float64

줄바꿈()

# ===============================================================

# 각 고장 유형별 발생 건수를 막대그래프로 나타냈다
# failure_count.plot(kind="bar")

# plt.title("Failure Type Distribution")
# plt.xlabel("Failure Type")
# plt.ylabel("Count")
# plt.xticks(rotation=0)
# plt.show()

# ===============================================================

# 각 행에서 동시에 발생한 고장 유형의 개수를 구했다
failure_num = df[failure_cols].sum(axis=1)

# 동시에 발생한 고장 유형 개수별 데이터 수를 구했다
print(failure_num.value_counts().sort_index())
# 0    9652
# 1     324
# 2      23
# 3       1
줄바꿈()

# ==============================================================

sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# 전체 데이터에서 센서별 기초 통계값을 구했다
sensor_stats = df[sensor_cols].describe().T

print(sensor_stats)
#                           count        mean         std     min     25%     50%     75%     max
# Air temperature [K]      10000.0   300.00493    2.000259   295.3   298.3   300.1   301.5   304.5
# Process temperature [K]  10000.0   310.00556    1.483734   305.7   308.8   310.1   311.1   313.8
# Rotational speed [rpm]   10000.0  1538.77610  179.284096  1168.0  1423.0  1503.0  1612.0  2886.0
# Torque [Nm]              10000.0    39.98691    9.968934     3.8    33.2    40.1    46.8    76.6
# Tool wear [min]          10000.0   107.95100   63.654147     0.0    53.0   108.0   162.0   253.0
줄바꿈()

# ===============================================================

# 전체 데이터에서 센서별 값의 분포를 확인했다
# for col in sensor_cols:
#     plt.figure(figsize=(8, 5))

#     plt.hist(df[col], bins=30, edgecolor="black")

#     plt.title(f"Distribution of {col}")
#     plt.xlabel(col)
#     plt.ylabel("Count")

#     plt.show()

# ==========================================================

# 센서별로 일반적인 범위를 크게 벗어나는 값이 있는지 확인했다
# for col in sensor_cols:
#     plt.figure(figsize=(8, 4))

#     plt.boxplot(
#         df[col],
#         vert=False
#     )

#     plt.title(f"Boxplot of {col}")
#     plt.xlabel(col)

#     plt.show()

# ===========================================================

# Machine failure을 기준으로 정상과 고장 데이터를 나눴다
normal = df[df["Machine failure"] == 0]
failure = df[df["Machine failure"] == 1]

print("정상 데이터:", len(normal))
# 정상 데이터: 9661
print("고장 데이터:", len(failure))
# 고장 데이터: 339

줄바꿈()

# =============================================================

sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# 정상과 고장에서 각 센서의 평균값을 구했다
sensor_mean = df.groupby("Machine failure")[sensor_cols].mean().T

# 고장 평균에서 정상 평균을 빼서 차이를 구했다
sensor_mean["difference"] = sensor_mean[1] - sensor_mean[0]

# 정상 상태와 비교했을 때 고장에서 몇 % 변했는지 구했다
sensor_mean["change_percent"] = sensor_mean["difference"] / sensor_mean[0] * 100

print(sensor_mean.round(3))
# Machine failure                 0         1  difference  change_percent
# Air temperature [K]       299.974   300.886       0.912           0.304
# Process temperature [K]   309.996   310.290       0.295           0.095
# Rotational speed [rpm]   1540.260  1496.487     -43.773          -2.842
# Torque [Nm]                39.630    50.168      10.538          26.592
# Tool wear [min]           106.694   143.782      37.088          34.761

줄바꿈()

# ============================================================

# 정상과 고장에서 각 센서의 중앙값을 구했다
sensor_median = df.groupby("Machine failure")[sensor_cols].median().T

print(sensor_median)
# Machine failure               0       1
# Air temperature [K]       300.0   301.6
# Process temperature [K]   310.0   310.4
# Rotational speed [rpm]   1507.0  1365.0
# Torque [Nm]                39.9    53.7
# Tool wear [min]           107.0   165.0

줄바꿈()

# ==============================================================

# 각 센서에서 정상과 고장의 전체적인 값 분포를 비교했다
# for col in sensor_cols:

#     normal_values = df.loc[df["Machine failure"] == 0, col]

#     failure_values = df.loc[df["Machine failure"] == 1, col]

#     plt.figure(figsize=(8, 5))

#     plt.boxplot([normal_values, failure_values], tick_labels=["Normal", "Failure"])

#     plt.title(f"Normal vs Failure - {col}")
#     plt.xlabel("Machine Status")
#     plt.ylabel(col)

#     plt.show()

# ==============================================================

sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# 전체 데이터에서 센서 변수끼리의 상관계수를 구했다
sensor_corr = df[sensor_cols].corr()

print(sensor_corr.round(3))
#                          Air temperature [K]  Process temperature [K]  Rotational speed [rpm]  Torque [Nm]  Tool wear [min]
# Air temperature [K]                    1.000                    0.876                   0.023       -0.014            0.014
# Process temperature [K]                0.876                    1.000                   0.019       -0.014            0.013
# Rotational speed [rpm]                 0.023                    0.019                   1.000       -0.875            0.000
# Torque [Nm]                           -0.014                   -0.014                  -0.875        1.000           -0.003
# Tool wear [min]                        0.014                    0.013                   0.000       -0.003            1.000

줄바꿈()

# =======================================================

# 센서 변수끼리의 상관관계를 히트맵으로 나타냈다
# plt.figure(figsize=(9, 7))

# sns.heatmap(sensor_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)

# plt.title("Correlation Between Sensor Variables")
# plt.show()

# ======================================================

# 각 센서와 Machine failure의 상관계수를 구했다
failure_corr = (
    df[sensor_cols + ["Machine failure"]]
    .corr()["Machine failure"]
    .drop("Machine failure")
)

print(failure_corr.sort_values(ascending=False).round(3))
# Torque [Nm]                0.191
# Tool wear [min]            0.105
# Air temperature [K]        0.083
# Process temperature [K]    0.036
# Rotational speed [rpm]    -0.044
# Name: Machine failure, dtype: float64

줄바꿈()

# ========================================================

failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# 각 센서와 각각의 고장 유형 사이의 상관계수를 구했다
sensor_failure_corr = (
    df[sensor_cols + failure_cols].corr().loc[sensor_cols, failure_cols]
)

print(sensor_failure_corr.round(3))
#                            TWF    HDF    PWF    OSF    RNF
# Air temperature [K]      0.010  0.138  0.003  0.002  0.018
# Process temperature [K]  0.007  0.057 -0.003  0.005  0.022
# Rotational speed [rpm]   0.010 -0.121  0.123 -0.105 -0.013
# Torque [Nm]             -0.015  0.143  0.084  0.183  0.016
# Tool wear [min]          0.116 -0.001 -0.009  0.156  0.011

줄바꿈()

# ==============================================================

# 고장 유형 두 개가 같은 행에서 동시에 발생한 횟수를 구했다
cooccurrence = pd.DataFrame(index=failure_cols, columns=failure_cols)

for failure_a in failure_cols:
    for failure_b in failure_cols:

        cooccurrence.loc[failure_a, failure_b] = (
            (df[failure_a] == 1) & (df[failure_b] == 1)
        ).sum()

print(cooccurrence)
#     TWF  HDF PWF OSF RNF
# TWF  46    0   1   3   1
# HDF   0  115   3   6   0
# PWF   1    3  95  12   0
# OSF   3    6  12  98   0
# RNF   1    0   0   0  19

줄바꿈()

# ===============================================


# 한 행에서 동시에 발생한 고장 유형의 이름을 하나의 문자열로 만들었다
def get_failure_combination(row):

    active_failures = [col for col in failure_cols if row[col] == 1]

    if len(active_failures) == 0:
        return "None"

    return "+".join(active_failures)


# 전체 데이터에서 실제 고장 조합을 만들었다
df["failure_combination"] = df[failure_cols].apply(get_failure_combination, axis=1)

# 실제로 존재하는 고장 조합별 데이터 수를 구했다
combination_count = df["failure_combination"].value_counts()

print(combination_count)
# failure_combination
# None           9652
# HDF             106
# PWF              80
# OSF              78
# TWF              42
# RNF              18
# PWF+OSF          11
# HDF+OSF           6
# HDF+PWF           3
# TWF+OSF           2
# TWF+RNF           1
# TWF+PWF+OSF       1

줄바꿈()

# ====================================================

sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Machine failure이 0인 정상 데이터의 센서 평균을 구했다
normal_mean = df[df["Machine failure"] == 0][sensor_cols].mean()

# 각 고장이 발생했을 때 센서 평균을 구했다
failure_sensor_mean = pd.DataFrame()

for failure in failure_cols:
    failure_sensor_mean[failure] = df[df[failure] == 1][sensor_cols].mean()

print("정상 센서 평균")
print(normal_mean.round(3))
# 정상 센서 평균
# Air temperature [K]         299.974
# Process temperature [K]     309.996
# Rotational speed [rpm]     1540.260
# Torque [Nm]                  39.630
# Tool wear [min]             106.694

print("\n고장 유형별 센서 평균")
print(failure_sensor_mean.round(3))
# 고장 유형별 센서 평균
#                               TWF       HDF       PWF       OSF       RNF
# Air temperature [K]       300.298   302.561   300.076   300.045   300.816
# Process temperature [K]   310.165   310.789   309.955   310.073   310.763
# Rotational speed [rpm]   1566.174  1337.261  1763.968  1350.327  1485.000
# Torque [Nm]                37.837    53.167    48.515    58.370    43.674
# Tool wear [min]           216.370   107.191   101.884   207.694   124.474

줄바꿈()

# =========================================

# 각 행에서 동시에 발생한 고장 유형의 개수를 구했다
df["failure_count"] = df[failure_cols].sum(axis=1)

single_failure_mean = {}

for failure in failure_cols:

    # 해당 고장만 단독으로 발생한 데이터를 구했다
    single_data = df[(df[failure] == 1) & (df["failure_count"] == 1)]

    # 단독 고장에서 센서 평균을 구했다
    single_failure_mean[failure] = single_data[sensor_cols].mean()

single_failure_mean = pd.DataFrame(single_failure_mean).T

print(single_failure_mean.round(3))
#      Air temperature [K]  Process temperature [K]  Rotational speed [rpm]  Torque [Nm]  Tool wear [min]
# TWF              300.269                  310.129                1583.238       35.990          216.214
# HDF              302.560                  310.802                1340.708       52.363          101.226
# PWF              300.023                  309.961                1843.238       45.000           87.625
# OSF              299.868                  310.051                1354.244       56.878          208.218
# RNF              300.767                  310.756                1489.444       43.522          119.889

줄바꿈()

# =======================================================================


# 한 행에서 발생한 고장 유형을 조합 이름으로 만들었다
def make_failure_combination(row):

    active = [failure for failure in failure_cols if row[failure] == 1]

    if len(active) == 0:
        return "None"

    return "+".join(active)


df["failure_combination"] = df.apply(make_failure_combination, axis=1)

# 두 개 이상의 고장이 동시에 발생한 데이터만 구했다
multiple_failure = df[df["failure_count"] >= 2]

# 고장 조합별 센서 평균을 구했다
multiple_sensor_mean = multiple_failure.groupby("failure_combination")[
    sensor_cols
].mean()

print(multiple_sensor_mean.round(3))

#                      Air temperature [K]  Process temperature [K]  Rotational speed [rpm]  Torque [Nm]  Tool wear [min]
# failure_combination
# HDF+OSF                            302.7                  310.750                1289.500       60.117          215.333
# HDF+PWF                            302.3                  310.400                1311.000       67.667          101.667
# PWF+OSF                            299.8                  309.764                1347.364       67.327          196.000
# TWF+OSF                            300.0                  310.550                1389.500       58.600          228.500
# TWF+PWF+OSF                        300.7                  310.200                1364.000       65.300          208.000
# TWF+RNF                            301.7                  310.900                1405.000       46.400          207.000

줄바꿈()

# ============================================================

# 각 고장 유형의 센서 평균을 행 방향으로 정리했다
failure_mean = pd.DataFrame()

for failure in failure_cols:

    failure_mean.loc[failure, sensor_cols] = df[df[failure] == 1][sensor_cols].mean()

# 정상 평균을 기준으로 고장 센서값의 변화율을 구했다
change_percent = (failure_mean - normal_mean) / normal_mean * 100

print(change_percent.round(2))
#     Air temperature [K]  Process temperature [K]  Rotational speed [rpm]  Torque [Nm]  Tool wear [min]
# TWF                 0.11                     0.05                    1.68        -4.52           102.80
# HDF                 0.86                     0.26                  -13.18        34.16             0.47
# PWF                 0.03                    -0.01                   14.52        22.42            -4.51
# OSF                 0.02                     0.03                  -12.33        47.29            94.66
# RNF                 0.28                     0.25                   -3.59        10.20            16.66

줄바꿈()

# =====================================================

y = df["Machine failure"]
# 0 = 정상
# 1 = 고장

feature_cols = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# 설비 상태를 예측하는 데 사용할 입력 데이터를 만들었다
X = df[feature_cols].copy()

# 정상과 고장 여부를 모델의 정답으로 만들었다
y = df["Machine failure"].copy()

print("X shape:", X.shape)
print("y shape:", y.shape)
# X shape: (10000, 6)
# y shape: (10000,)

print("\n입력 변수")
print(X.columns.tolist())
# 입력 변수
# ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

print("\n정답 분포")
print(y.value_counts())
# 정답 분포
# Machine failure
# 0    9661
# 1     339

줄바꿈()

# =================================================================

feature_cols = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# 모델 입력 변수와 정답에서 결측치 개수를 확인했다
missing_count = df[feature_cols + ["Machine failure"]].isnull().sum()

print(missing_count)
# Type                       0
# Air temperature [K]        0
# Process temperature [K]    0
# Rotational speed [rpm]     0
# Torque [Nm]                0
# Tool wear [min]            0
# Machine failure            0

줄바꿈()

# =========================================

duplicate_cols = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
]

# 식별자를 제외하고 동일한 데이터가 중복되어 있는지 확인했다
duplicate_count = df.duplicated(subset=duplicate_cols).sum()

print("중복 데이터 수:", duplicate_count)
# 중복 데이터 수: 0

줄바꿈()

# ===================================================

# Type에 어떤 값이 존재하는지 확인했다
print(df["Type"].value_counts())
# Type
# L    6000
# M    2997
# H    1003

줄바꿈()

# ==================================================

# Type을 모델이 사용할 수 있도록 One-Hot Encoding했다
X = pd.get_dummies(df[feature_cols], columns=["Type"], dtype=int)

print(X.head())
#    Air temperature [K]  Process temperature [K]  Rotational speed [rpm]  Torque [Nm]  Tool wear [min]  Type_H  Type_L  Type_M
# 0                298.1                    308.6                    1551         42.8                0       0       0       1
# 1                298.2                    308.7                    1408         46.3                3       0       1       0
# 2                298.1                    308.5                    1498         49.4                5       0       1       0
# 3                298.2                    308.6                    1433         39.5                7       0       1       0
# 4                298.2                    308.7                    1408         40.0                9       0       1       0
print("\n변환 후 X shape:", X.shape)
# 변환 후 X shape: (10000, 8)

print("\n입력 변수:")
print(X.columns.tolist())
# 입력 변수:
# ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type_H', 'Type_L', 'Type_M']

줄바꿈()

# ===================================================================

# 정상과 고장 여부를 모델의 정답으로 분리했다
y = df["Machine failure"].copy()

print(y.value_counts())
# Machine failure
# 0    9661
# 1     339

줄바꿈()

# ==================================================================

# 전체 데이터를 Train 80%, Test 20%로 분리했다
# stratify=y를 사용해 정상/고장 비율이 양쪽에 비슷하게 유지되도록 했다
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("X_train:", X_train.shape)
# X_train: (8000, 8)

print("X_test :", X_test.shape)
# X_test : (2000, 8)

print("y_train:", y_train.shape)
# y_train: (8000,)

print("y_test :", y_test.shape)
# y_test : (2000,)

줄바꿈()

# ============================================

# Train 데이터에서 정상과 고장의 개수를 확인했다
print("Train 정답 개수")
print(y_train.value_counts())
# Train 정답 개수
# Machine failure
# 0    7729
# 1     271

# Train 데이터에서 정상과 고장의 비율을 확인했다
print("\nTrain 정답 비율(%)")
print((y_train.value_counts(normalize=True) * 100).round(2))
# Train 정답 비율(%)
# Machine failure
# 0    96.61
# 1     3.39

줄바꿈()

# ===============================================================

# Test 데이터에서 정상과 고장의 개수를 확인했다
print("Test 정답 개수")
print(y_test.value_counts())
# Test 정답 개수
# Machine failure
# 0    1932
# 1      68

# Test 데이터에서 정상과 고장의 비율을 확인했다
print("\nTest 정답 비율(%)")
print((y_test.value_counts(normalize=True) * 100).round(2))
# Test 정답 비율(%)
# Machine failure
# 0    96.6
# 1     3.4

줄바꿈()

# ================================================

# Scaling과 Logistic Regression을 하나의 과정으로 만들었다
logistic_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

# Train 데이터로 정상과 고장 패턴을 학습시켰다
logistic_model.fit(X_train, y_train)

print("Logistic Regression 학습 완료")

줄바꿈()

# ===================================================================

# Decision Tree 모델을 만들었다
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Train 데이터로 정상과 고장 패턴을 학습시켰다
decision_tree_model.fit(X_train, y_train)

print("Decision Tree 학습 완료")

줄바꿈()

# ==============================================================

# 여러 Decision Tree를 사용하는 Random Forest 모델을 만들었다
random_forest_model = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)

# Train 데이터로 정상과 고장 패턴을 학습시켰다
random_forest_model.fit(X_train, y_train)

print("Random Forest 학습 완료")

줄바꿈()

# ===========================================================

# Logistic Regression으로 Test 데이터를 예측했다
logistic_pred = logistic_model.predict(X_test)

# Decision Tree로 Test 데이터를 예측했다
tree_pred = decision_tree_model.predict(X_test)

# Random Forest로 Test 데이터를 예측했다
forest_pred = random_forest_model.predict(X_test)

# =============================================================

# Random Forest가 각 Test 데이터를 고장이라고 판단한 확률을 구했다
failure_probability = random_forest_model.predict_proba(X_test)[:, 1]

print(failure_probability[:10])
# [0.11 0.   0.02 0.   0.05 0.   0.   0.   0.   0.01]

줄바꿈()

# ============================================================

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

results = []

for threshold in thresholds:

    # 현재 임계값보다 고장 확률이 높으면 1로 판단했다
    y_pred_threshold = (failure_probability >= threshold).astype(int)

    # 실제값과 예측값으로 혼동행렬을 구했다
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

    # Precision, Recall, F1-score를 구했다
    precision = precision_score(y_test, y_pred_threshold, zero_division=0)

    recall = recall_score(y_test, y_pred_threshold)

    f1 = f1_score(y_test, y_pred_threshold)

    results.append([threshold, tn, fp, fn, tp, precision, recall, f1])

# ======================================================================

threshold_result = pd.DataFrame(
    results, columns=["Threshold", "TN", "FP", "FN", "TP", "Precision", "Recall", "F1"]
)

print(threshold_result.round(3))
#    Threshold    TN  FP  FN  TP  Precision  Recall     F1
# 0        0.2  1889  43  12  56      0.566   0.824  0.671
# 1        0.3  1912  20  17  51      0.718   0.750  0.734
# 2        0.4  1923   9  25  43      0.827   0.632  0.717
# 3        0.5  1928   4  33  35      0.897   0.515  0.654
# 4        0.6  1929   3  39  29      0.906   0.426  0.580
# 5        0.7  1932   0  50  18      1.000   0.265  0.419

줄바꿈()

# ===================================================

# 현재는 분석 방법 확인을 위한 가상 비용을 설정했다
FP_COST = 100000
FN_COST = 1000000

# 각 임계값에서 발생하는 예상 비용을 계산했다
threshold_result["Cost"] = (
    threshold_result["FP"] * FP_COST + threshold_result["FN"] * FN_COST
)

print(
    threshold_result[
        ["Threshold", "FP", "FN", "Precision", "Recall", "F1", "Cost"]
    ].round(3)
)

#    Threshold  FP  FN  Precision  Recall     F1      Cost
# 0        0.2  43  12      0.566   0.824  0.671  16300000
# 1        0.3  20  17      0.718   0.750  0.734  19000000
# 2        0.4   9  25      0.827   0.632  0.717  25900000
# 3        0.5   4  33      0.897   0.515  0.654  33400000
# 4        0.6   3  39      0.906   0.426  0.580  39300000
# 5        0.7   0  50      1.000   0.265  0.419  50000000

줄바꿈()

# ========================================================

# 현재 설정한 비용 기준에서 총 비용이 가장 낮은 임계값을 찾았다
best_row = threshold_result.loc[threshold_result["Cost"].idxmin()]

print("비용 기준 최적 임계값")
print(best_row)
# 비용 기준 최적 임계값
# Threshold    2.000000e-01
# TN           1.889000e+03
# FP           4.300000e+01
# FN           1.200000e+01
# TP           5.600000e+01
# Precision    5.656566e-01
# Recall       8.235294e-01
# F1           6.706587e-01
# Cost         1.630000e+07

줄바꿈()

# ==================================================

logistic_probability = logistic_model.predict_proba(X_test)[:, 1]

forest_probability = random_forest_model.predict_proba(X_test)[:, 1]

# ================================================

# 모델 성능평가 및 최종 모델 선정
# 각 모델이 Test 데이터의 정상/고장을 예측하게 했다
logistic_pred = logistic_model.predict(X_test)
tree_pred = decision_tree_model.predict(X_test)
forest_pred = random_forest_model.predict(X_test)
print("Logistic Regression 예측 완료")
print("Decision Tree 예측 완료")
print("Random Forest 예측 완료")

models_pred = {
    "Logistic Regression": logistic_pred,
    "Decision Tree": tree_pred,
    "Random Forest": forest_pred,
}
for name, pred in models_pred.items():
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"\n{name}")
    print("TN :", tn)
    print("FP :", fp)
    print("FN :", fn)
    print("TP :", tp)

줄바꿈()

for name, pred in models_pred.items():
    accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
print(f"\n{name}")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall :", round(recall, 4))
print("F1-score :", round(f1, 4))

줄바꿈()

evaluation_results = []
for name, pred in models_pred.items():
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    evaluation_results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        }
    )
evaluation_df = pd.DataFrame(evaluation_results)
print(evaluation_df.round(3))
#                  Model  Accuracy  Precision  Recall     F1    TN  FP  FN  TP
# 0  Logistic Regression     0.968      0.636   0.103  0.177  1928   4  61   7
# 1        Decision Tree     0.978      0.687   0.676  0.681  1911  21  22  46
# 2        Random Forest     0.981      0.895   0.500  0.642  1928   4  34  34

줄바꿈()

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
}
auc_results = []
for name, model in models.items():
    # 각 모델이 Test 데이터의 고장 확률을 계산했다
    probability = model.predict_proba(X_test)[:, 1]

    # ROC-AUC와 PR-AUC를 계산했다
    roc_auc = roc_auc_score(y_test, probability)
    pr_auc = average_precision_score(y_test, probability)
    auc_results.append({"Model": name, "ROC-AUC": roc_auc, "PR-AUC": pr_auc})
auc_df = pd.DataFrame(auc_results)
print(auc_df.round(3))
