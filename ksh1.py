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


# 폰트전환 함수
def set_korean_font():
    font = "Malgun Gothic" if platform.system() == "windows" else "AppleGothic"
    plt.rc("font", family=font)
    plt.rcParams["axes.unicode_minus"] = False


# 테마 지정 화이트그리드 스타일
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt
from korean_font import set_korean_font

# set_korean_font()


def 줄바꿈():
    print()
    print("===" * 30)
    print()


DATA_FILEB = "k뉴딜 팀플/ai4i2020.csv"
df = pd.read_csv(DATA_FILEB)

# =============================================
# 1. 추세
cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

for col in cols:
    plt.figure(figsize=(12, 4))

    plt.plot(df["UDI"], df[col])

    plt.title(f"{col} 의 추세")
    plt.xlabel("UDI")
    plt.ylabel(col)

    plt.show()

# ========================================================
# 평균 추세
for col in cols:

    trend = df[col].rolling(window=100).mean()

    plt.figure(figsize=(12, 4))

    plt.plot(df["UDI"], df[col], alpha=0.3, label="Original")
    plt.plot(df["UDI"], trend, linewidth=2, label="Trend")

    plt.title(f"{col}의 평균 추세")
    plt.xlabel("UDI")
    plt.ylabel(col)

    plt.legend()
    plt.show()

# 추세 분석 결과 :
# Air temperature와 Process temperature에서는 UDI 진행에 따라 구간별 장기 상승·하락 패턴이 뚜렷하게 관찰되었다.
# 반면 Rotational speed와 Torque는 순간적인 변동 폭은 크지만 이동평균이 비교적 일정한 수준을 유지하여 뚜렷한 장기 추세는 확인되지 않았다.
# Tool wear는 전체적인 단방향 추세보다는 값이 누적된 후 다시 낮은 값으로 초기화되는 반복적인 구조가 관찰되었다.
# 따라서 온도 계열은 추세 관련 특성을 검토하고,
# 회전속도와 토크는 변동성 및 급격한 변화 특성에서 추가 분석하며,
# Tool wear는 주기성 분석에서 별도로 확인한다.
# ============================================
# 2. 변동성
sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# 전체 관측값 기준 표준편차
print("=== 전체 센서 변동성 ===")
print(f"전체 관측값 기준 표준편차 : \n{df[sensor_cols].std()}")

# 고장 데이터
failure = df[df["Machine failure"] == 1]

# 전체 10,000개 관측값 시각화
for col in sensor_cols:

    plt.figure(figsize=(15, 4))

    plt.plot(df["UDI"], df[col], label="Sensor Value")

    plt.scatter(failure["UDI"], failure[col], label="Machine Failure")

    plt.xlabel("UDI")
    plt.ylabel(col)
    plt.title(col + "의 10,000개 관측값 시각화")

    plt.legend()
    plt.show()

# 변동성 분석 결과 :
# 5개 센서의 전체 10,000개 관측값에 대해 표준편차와 시계열 패턴을 확인하였다.
# Air temperature와 Process temperature는 급격한 변동보다는 장기적인 수준 변화가 관찰되었다.
# Rotational speed와 Torque는 전체 구간에서 단기적인 변동과 극단값이 상대적으로 자주 관찰되었다.
# Tool wear는 값이 증가한 후 다시 낮아지는 반복적인 구조가 확인되었다.
# 센서별 단위와 값의 범위가 다르므로 표준편차의 절대적인 크기를 센서 간 직접 비교하지 않고,
# 각 센서 내부의 변동 특성을 파악하는 기초 지표로 활용하였다.

# ==============================================================
# 3. 변화량
sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# 1. 직전 관측값 대비 변화량 생성
for col in sensor_cols:
    df[col + "_diff"] = df[col].diff()

# 2. 변화량의 절댓값 생성
for col in sensor_cols:
    df[col + "_abs_diff"] = df[col + "_diff"].abs()

# 3. 변화량 기본 통계 확인
diff_cols = [col + "_diff" for col in sensor_cols]

print("=== 센서 변화량 통계 ===")
print(df[diff_cols].describe())
# === 센서 변화량 통계 ===
#        Air temperature [K]_diff  Process temperature [K]_diff  Rotational speed [rpm]_diff  Torque [Nm]_diff  Tool wear [min]_diff
# count               9999.000000                   9999.000000                  9999.000000       9999.000000           9999.000000
# mean                   0.000090                      0.000010                    -0.005101         -0.000260              0.003000
# std                    0.068308                      0.079914                   252.584472         14.060764             23.714499
# min                   -0.300000                     -0.300000                 -1478.000000        -50.500000           -253.000000
# 25%                    0.000000                     -0.100000                  -133.000000         -9.300000              2.000000
# 50%                    0.000000                      0.000000                    -1.000000          0.000000              2.000000
# 75%                    0.000000                      0.100000                   134.000000          9.300000              3.000000
# max                    0.200000                      0.300000                  1449.000000         52.200000              5.000000

# 4. 고장 데이터
failure = df[df["Machine failure"] == 1]

# 5. 전체 관측값 변화량 + 고장 위치 시각화
for col in sensor_cols:

    plt.figure(figsize=(15, 4))

    plt.plot(df["UDI"], df[col + "_diff"], label="Difference")

    plt.scatter(failure["UDI"], failure[col + "_diff"], label="Machine Failure")

    #     # 변화량 0 기준선
    plt.axhline(0)

    plt.xlabel("UDI")
    plt.ylabel("Difference")
    plt.title(col + " - Difference & Failure")

    plt.legend()
    plt.show()

# 변화량 분석 결과 :
# 전체 10,000개 관측값을 순서대로 배치하고 직전 관측값과의 차이(diff)를 계산하였다.
# Air temperature와 Process temperature는 대부분 작은 범위에서 변화하였다.
# 반면 Rotational speed와 Torque에서는 상대적으로 큰 양·음의 변화가 반복적으로 관찰되었으며 일부 고장 관측값도 큰 변화 영역에 위치하였다.
# 이에 따라 변화 방향을 나타내는 diff와 방향에 관계없이 변화 크기를 나타내는 abs_diff를 파생 Feature 후보로 선정하였다.
# Tool wear에서는 작은 증가와 큰 음의 변화가 반복되는 구조가 나타났으므로 단순 이상 변화로 해석하지 않고 주기성 분석에서 추가 확인한다.

# ===========================================================
# 4. 주기성
sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# # 그래프에 표시할 한글 이름
sensor_names = {
    "Air temperature [K]": "공기 온도",
    "Process temperature [K]": "공정 온도",
    "Rotational speed [rpm]": "회전 속도",
    "Torque [Nm]": "토크",
    "Tool wear [min]": "공구 마모",
}


# ---------------------------------
# 1. 5개 센서 자기상관 분석
# ---------------------------------

max_lag = 500

for col in sensor_cols:

    autocorr_values = []

    for lag in range(1, max_lag + 1):

        autocorr_values.append(df[col].autocorr(lag=lag))

    plt.figure(figsize=(15, 4))

    plt.plot(range(1, max_lag + 1), autocorr_values)

    plt.axhline(0)

    plt.xlabel("관측값 간격(Lag)")
    plt.ylabel("자기상관계수")
    plt.title(sensor_names[col] + " - 자기상관 분석")

    plt.show()


# ---------------------------------
# 2. 공구 마모 초기화 위치 찾기
# ---------------------------------

tool_diff = df["Tool wear [min]"].diff()

reset_points = df.loc[tool_diff < -100, "UDI"]

print("=== 공구 마모 초기화 위치 ===")
print(reset_points)
# === 공구 마모 초기화 위치 ===
# 78        79
# 162      163
# 250      251
# 332      333
# 418      419
#         ...
# 9672    9673
# 9759    9760
# 9834    9835
# 9908    9909
# 9989    9990
# Name: UDI, Length: 119, dtype: int64

# ---------------------------------
# 3. 초기화 사이의 관측값 간격
# ---------------------------------

reset_intervals = reset_points.diff()

print("\n=== 공구 마모 초기화 사이의 간격 ===")
print(reset_intervals)
# === 공구 마모 초기화 사이의 간격 ===
# 78       NaN
# 162     84.0
# 250     88.0
# 332     82.0
# 418     86.0
#         ...
# 9672    95.0
# 9759    87.0
# 9834    75.0
# 9908    74.0
# 9989    81.0
# Name: UDI, Length: 119, dtype: float64

print("\n평균 초기화 간격:")
print(reset_intervals.mean())
# 평균 초기화 간격:
# 83.99152542372882

# ---------------------------------
# 4. 초기화 간격 시각화
# ---------------------------------

plt.figure(figsize=(12, 4))

plt.plot(reset_intervals.values, marker="o")

plt.xlabel("초기화 순번")
plt.ylabel("초기화 사이 관측값 개수")
plt.title("공구 마모 초기화 간격")

plt.show()


# ---------------------------------
# 5. 공구 마모 원본 + 초기화 위치
# ---------------------------------

plt.figure(figsize=(15, 4))

plt.plot(df["UDI"], df["Tool wear [min]"], label="공구 마모")

plt.scatter(
    reset_points, df.loc[reset_points.index, "Tool wear [min]"], label="초기화 위치"
)

plt.xlabel("UDI")
plt.ylabel("공구 마모 [분]")
plt.title("공구 마모 반복 및 초기화 패턴")

plt.legend()
plt.show()

# 주기성 분석 결과 :
# 5개 센서를 대상으로 1~500 관측값 간격의 자기상관을 분석하였다.
# 공기 온도와 공정 온도는 자기상관이 높은 상태에서 완만하게 감소하였으나
# 일정 간격의 반복적인 봉우리는 나타나지 않아 주기성보다는 장기적인 추세의 영향이 큰 것으로 판단하였다.
# 회전 속도와 토크는 자기상관계수가 대부분 0 부근에서 불규칙하게 나타나 뚜렷한 주기성이 확인되지 않았다.
# 반면 공구 마모는 자기상관 그래프에서 반복적인 파동이 나타났으며,
# 공구 마모가 크게 감소하는 초기화 지점 119개를 확인하였다.
# 초기화 사이의 평균 간격은 약 83.99개 관측값으로 나타나 공구 마모에 반복적인 초기화 구조가 존재함을 확인하였다.
# 이를 바탕으로 tool_wear_reset, 초기화 이후 경과 관측값 등의 파생 Feature를 후보로 활용한다.

# ============================================================
# 5. 시계열 파생 Feature 생성

# 1. 온도 추세
df["air_temp_mean_10"] = df["Air temperature [K]"].rolling(window=10).mean()

df["process_temp_mean_10"] = df["Process temperature [K]"].rolling(window=10).mean()

# 2. 센서 간 온도 차이
df["temp_gap"] = df["Process temperature [K]"] - df["Air temperature [K]"]

# 3. 회전속도 변화량
df["speed_diff"] = df["Rotational speed [rpm]"].diff()

df["speed_abs_diff"] = df["speed_diff"].abs()

# 4. 토크 변화량
df["torque_diff"] = df["Torque [Nm]"].diff()

df["torque_abs_diff"] = df["torque_diff"].abs()

# 5. 회전속도와 토크의 최근 변동성
df["speed_std_10"] = df["Rotational speed [rpm]"].rolling(window=10).std()

df["torque_std_10"] = df["Torque [Nm]"].rolling(window=10).std()

# 6. 공구 마모 초기화 여부
df["tool_wear_reset"] = (df["Tool wear [min]"].diff() < -100).astype(int)

# 7. 공구 마모 초기화 후 경과 관측값
reset_group = df["tool_wear_reset"].cumsum()

df["since_tool_reset"] = df.groupby(reset_group).cumcount()

# 8. 회전속도 × 토크
df["speed_torque"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"]

# Feature 목록
feature_cols = [
    "air_temp_mean_10",
    "process_temp_mean_10",
    "temp_gap",
    "speed_diff",
    "speed_abs_diff",
    "torque_diff",
    "torque_abs_diff",
    "speed_std_10",
    "torque_std_10",
    "tool_wear_reset",
    "since_tool_reset",
    "speed_torque",
]

# print(df[feature_cols].head(15))

#    air_temp_mean_10  process_temp_mean_10  temp_gap  speed_diff  speed_abs_diff  ...  speed_std_10  torque_std_10  tool_wear_reset  since_tool_reset  speed_torque
# 0                NaN                   NaN      10.5         NaN             NaN  ...           NaN            NaN                0                 0       66382.8
# 1                NaN                   NaN      10.5      -143.0           143.0  ...           NaN            NaN                0                 1       65190.4
# 2                NaN                   NaN      10.4        90.0            90.0  ...           NaN            NaN                0                 2       74001.2
# 3                NaN                   NaN      10.4       -65.0            65.0  ...           NaN            NaN                0                 3       56603.5
# 4                NaN                   NaN      10.5       -25.0            25.0  ...           NaN            NaN                0                 4       56320.0
# 5                NaN                   NaN      10.5        17.0            17.0  ...           NaN            NaN                0                 5       59707.5
# 6                NaN                   NaN      10.5       133.0           133.0  ...           NaN            NaN                0                 6       66059.2
# 7                NaN                   NaN      10.5       -31.0            31.0  ...           NaN            NaN                0                 7       61385.4
# 8                NaN                   NaN      10.4       140.0           140.0  ...           NaN            NaN                0                 8       47676.2
# 9             298.19                308.66      10.5        74.0            74.0  ...    113.060652       6.826655                0                 9       48748.0
# 10            298.22                308.69      10.5        41.0            41.0  ...    140.100004       8.377722                0                10       42589.8
# 11            298.26                308.73      10.5      -359.0           359.0  ...    138.545460       8.179622                0                11       63038.9
# 12            298.31                308.79      10.5       -84.0            84.0  ...    153.055582       8.459899                0                12       68422.9
# 13            298.35                308.85      10.6       403.0           403.0  ...    162.150684       8.798131                0                13       52260.0
# 14            298.39                308.90      10.6       293.0           293.0  ...    210.380317      10.276402                0                14       39886.0

# [15 rows x 12 columns]

# =========================================================
# 6. 정상 / 고장 Feature 비교

feature_cols = [
    "air_temp_mean_10",
    "process_temp_mean_10",
    "temp_gap",
    "speed_diff",
    "speed_abs_diff",
    "torque_diff",
    "torque_abs_diff",
    "speed_std_10",
    "torque_std_10",
    "tool_wear_reset",
    "since_tool_reset",
    "speed_torque",
]

feature_names = {
    "air_temp_mean_10": "공기 온도 최근 평균",
    "process_temp_mean_10": "공정 온도 최근 평균",
    "temp_gap": "공정 온도 - 공기 온도",
    "speed_diff": "회전 속도 변화량",
    "speed_abs_diff": "회전 속도 절대 변화량",
    "torque_diff": "토크 변화량",
    "torque_abs_diff": "토크 절대 변화량",
    "speed_std_10": "회전 속도 최근 변동성",
    "torque_std_10": "토크 최근 변동성",
    "tool_wear_reset": "공구 마모 초기화 여부",
    "since_tool_reset": "공구 초기화 후 경과 관측값",
    "speed_torque": "회전 속도 × 토크",
}


# 1. 정상 / 고장 개수
print("=== 정상 / 고장 데이터 개수 ===")
print(df["Machine failure"].value_counts())
# Machine failure
# 0    9661
# 1     339
# Name: count, dtype: int64

print("\n=== 정상 / 고장 비율(%) ===")
print(df["Machine failure"].value_counts(normalize=True) * 100)
# Machine failure
# 0    96.61
# 1     3.39
# Name: proportion, dtype: float64

# 2. Feature 평균 비교
comparison = df.groupby("Machine failure")[feature_cols].mean()

normal_mean = comparison.loc[0]
failure_mean = comparison.loc[1]

difference_percent = (failure_mean - normal_mean) / normal_mean.abs() * 100

result = pd.DataFrame(
    {
        "정상 평균": normal_mean,
        "고장 평균": failure_mean,
        "차이 비율(%)": difference_percent,
    }
)

print("\n=== 정상 / 고장 Feature 평균 비교 ===")
print(result.round(2))
# === 정상 / 고장 Feature 평균 비교 ===
#                          정상 평균     고장 평균  차이 비율(%)
# air_temp_mean_10        299.98    300.88      0.30
# process_temp_mean_10    310.00    310.28      0.09
# temp_gap                 10.02      9.40     -6.16
# speed_diff                1.45    -41.47  -2959.93
# speed_abs_diff          177.21    276.37     55.95
# torque_diff              -0.35      9.92   2947.36
# torque_abs_diff          10.91     17.69     62.10
# speed_std_10            163.66    187.84     14.78
# torque_std_10             9.63     11.00     14.23
# tool_wear_reset           0.01      0.01    -26.30
# since_tool_reset         41.09     55.41     34.87
# speed_torque          59631.04  69545.80     16.63

# 3. Feature별 박스플롯
for col in feature_cols:

    plt.figure(figsize=(7, 5))

    df.boxplot(column=col, by="Machine failure")

    plt.title(feature_names[col] + " - 정상/고장 비교")

    plt.suptitle("")

    plt.xlabel("설비 상태 (0=정상, 1=고장)")
    plt.ylabel(feature_names[col])

    plt.show()


# 4. 공구 초기화 발생 비율
reset_compare = df.groupby("Machine failure")["tool_wear_reset"].mean() * 100

print("\n=== 정상/고장 공구 초기화 발생 비율(%) ===")
print(reset_compare.round(2))
# === 정상/고장 공구 초기화 발생 비율(%) ===
# Machine failure
# 0    1.20
# 1    0.88
# Name: tool_wear_reset, dtype: float64

# 5. 원본 센서 평균도 비교
sensor_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

sensor_comparison = df.groupby("Machine failure")[sensor_cols].mean().T

print("\n=== 원본 센서 정상/고장 평균 비교 ===")
print(sensor_comparison.round(2))
# === 원본 센서 정상/고장 평균 비교 ===
# Machine failure                0        1
# Air temperature [K]       299.97   300.89
# Process temperature [K]   310.00   310.29
# Rotational speed [rpm]   1540.26  1496.49
# Torque [Nm]                39.63    50.17
# Tool wear [min]           106.69   143.78

# 정상/고장 비교 결과 :
# 전체 10,000개 관측값 중 정상은 9,661개(96.61%), 고장은 339개(3.39%)로 클래스 불균형이 확인되었다.
# 고장 관측값에서는 회전속도 절대 변화량이 177.21에서 276.37로,
# 토크 절대 변화량이 10.91에서 17.69로 증가하여 센서의 급격한 변화가 나타나는 경향을 보였다.
# 또한 회전속도 및 토크의 최근 변동성도 고장에서 증가하였다.
# 공구 초기화 후 경과 관측값은 41.09에서 55.41로 증가했으며,
# 회전속도와 토크의 결합 특성 역시 고장에서 높은 값을 보였다.
# 반면 온도의 최근 평균과 공구 초기화 여부는 정상과 고장 간 차이가 상대적으로 작았다.
# 따라서 변화량, 변동성, 공구 사용 진행도 및 센서 결합 특성을 최종 Feature 후보로 검토한다.

# ======================================================================
# 최종 특성 선정
feature_cols = [
    "air_temp_mean_10",
    "process_temp_mean_10",
    "temp_gap",
    "speed_diff",
    "speed_abs_diff",
    "torque_diff",
    "torque_abs_diff",
    "speed_std_10",
    "torque_std_10",
    "tool_wear_reset",
    "since_tool_reset",
    "speed_torque",
]


# Feature 간 상관관계
feature_corr = df[feature_cols].corr()

print("=== Feature 간 상관계수 ===")
print(feature_corr.round(2))
# === Feature 간 상관계수 ===
#   air_temp_mean_10  process_temp_mean_10  temp_gap  speed_diff  ...  torque_std_10  tool_wear_reset  since_tool_reset  speed_torque
# air_temp_mean_10                  1.00                  0.88     -0.70        0.00  ...           0.06            -0.00              0.02         -0.01
# process_temp_mean_10              0.88                  1.00     -0.27        0.00  ...           0.02            -0.00              0.02         -0.01
# temp_gap                         -0.70                 -0.27      1.00        0.00  ...          -0.08            -0.00             -0.02          0.01
# speed_diff                        0.00                  0.00      0.00        1.00  ...           0.00            -0.01              0.00         -0.57
# speed_abs_diff                    0.03                  0.02     -0.04        0.00  ...           0.29             0.01             -0.02         -0.28
# torque_diff                       0.00                 -0.00     -0.00       -0.88  ...          -0.00             0.01             -0.00          0.69
# torque_abs_diff                   0.03                  0.01     -0.04        0.01  ...           0.31             0.01             -0.02         -0.02
# speed_std_10                      0.06                  0.03     -0.07        0.00  ...           0.72            -0.00             -0.03         -0.13
# torque_std_10                     0.06                  0.02     -0.08        0.00  ...           1.00            -0.01             -0.01         -0.01
# tool_wear_reset                  -0.00                 -0.00     -0.00       -0.01  ...          -0.01             1.00             -0.19          0.01
# since_tool_reset                  0.02                  0.02     -0.02        0.00  ...          -0.01            -0.19              1.00         -0.00
# speed_torque                     -0.01                 -0.01      0.01       -0.57  ...          -0.01             0.01             -0.00          1.00

# [12 rows x 12 columns]

# 높은 상관관계 Feature 자동 탐색
print("\n=== 상관계수 절댓값 0.8 이상 Feature 조합 ===")

for i in range(len(feature_cols)):
    for j in range(i + 1, len(feature_cols)):

        corr_value = feature_corr.iloc[i, j]

        if abs(corr_value) >= 0.8:

            print(feature_cols[i], "<->", feature_cols[j], ":", round(corr_value, 2))

# === 상관계수 절댓값 0.8 이상 Feature 조합 ===
# air_temp_mean_10 <-> process_temp_mean_10 : 0.88
# speed_diff <-> torque_diff : -0.88

# 상관관계 히트맵
# plt.figure(figsize=(12, 10))

# plt.imshow(feature_corr, cmap="coolwarm", vmin=-1, vmax=1)

# plt.colorbar(label="상관계수")

# plt.xticks(range(len(feature_cols)), feature_cols, rotation=90)

# plt.yticks(range(len(feature_cols)), feature_cols)

# for i in range(len(feature_cols)):
#     for j in range(len(feature_cols)):

#         plt.text(
#             j, i, f"{feature_corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8
#         )

# plt.title("파생 Feature 간 상관관계")
# plt.tight_layout()
# plt.show()

# ==========================================
# 최종 시계열 Feature 생성
# ==========================================

# 1. 회전속도 절대 변화량
df["speed_abs_diff"] = df["Rotational speed [rpm]"].diff().abs()

# 2. 토크 절대 변화량
df["torque_abs_diff"] = df["Torque [Nm]"].diff().abs()

# 3. 최근 10개 회전속도 변동성
df["speed_std_10"] = df["Rotational speed [rpm]"].rolling(window=10).std()

# 4. 최근 10개 토크 변동성
df["torque_std_10"] = df["Torque [Nm]"].rolling(window=10).std()

# 5. 공구 초기화 탐지
tool_reset = df["Tool wear [min]"].diff() < -100

# 초기화 이후 경과 관측값
reset_group = tool_reset.cumsum()

df["since_tool_reset"] = df.groupby(reset_group).cumcount()

# 6. 회전속도 × 토크
df["speed_torque"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"]

# 7. 공정온도 - 공기온도
df["temp_gap"] = df["Process temperature [K]"] - df["Air temperature [K]"]


final_features = [
    "speed_abs_diff",
    "torque_abs_diff",
    "speed_std_10",
    "torque_std_10",
    "since_tool_reset",
    "speed_torque",
    "temp_gap",
]

print(df[final_features].head(15))
#     speed_abs_diff  torque_abs_diff  speed_std_10  torque_std_10  since_tool_reset  speed_torque  temp_gap
# 0              NaN              NaN           NaN            NaN                 0       66382.8      10.5
# 1            143.0              3.5           NaN            NaN                 1       65190.4      10.5
# 2             90.0              3.1           NaN            NaN                 2       74001.2      10.4
# 3             65.0              9.9           NaN            NaN                 3       56603.5      10.4
# 4             25.0              0.5           NaN            NaN                 4       56320.0      10.5
# 5             17.0              1.9           NaN            NaN                 5       59707.5      10.5
# 6            133.0              0.5           NaN            NaN                 6       66059.2      10.5
# 7             31.0              2.2           NaN            NaN                 7       61385.4      10.5
# 8            140.0             11.6           NaN            NaN                 8       47676.2      10.4
# 9             74.0              0.6    113.060652       6.826655                 9       48748.0      10.5
# 10            41.0              4.1    140.100004       8.377722                10       42589.8      10.5
# 11           359.0             20.4    138.545460       8.179622                11       63038.9      10.5
# 12            84.0              6.8    153.055582       8.459899                12       68422.9      10.5
# 13           403.0             21.1    162.150684       8.798131                13       52260.0      10.6
# 14           293.0             10.4    210.380317      10.276402                14       39886.0      10.6

# ===============================================
# 1. 추세 분석
# 공기온도와 공정온도는 장기적인 변화 흐름을 보였으나 정상/고장 평균 차이는 크지 않았다.

# 2. 변동성 분석
# 고장 데이터에서 회전속도와 토크의 최근 변동성이 정상보다 증가했다.

# 3. 변화량 분석
# 회전속도 절대 변화량은 정상 대비 약 56%, 토크 절대 변화량은 약 62% 증가해 고장 상태에서 센서 변화 폭이 커지는 경향이 확인됐다.

# 4. 주기성 분석
# 공구 마모는 평균 약 84개 관측값 간격으로 초기화되는 반복 패턴을 보였으며, 고장 데이터에서는 초기화 이후 경과 관측값이 더 높았다.

# 5. 센서 간 관계
# 회전속도 변화량과 토크 변화량 사이에서 -0.88의 강한 음의 상관관계가 확인됐으며, 회전속도와 토크를 결합한 Feature 역시 고장에서 높은 값을 보였다.

# 6. 최종 결과
# 변화량·변동성·공구 사용 진행도·센서 간 결합 정보를 중심으로 7개의 최종 파생 Feature 후보를 선정했다.

# 그리고 아주 중요한 한 가지 현재 데이터는
# 정상 96.61% / 고장 3.39% 로 상당히 불균형해.
# 따라서 결론은 “이 7개가 고장을 예측한다고 증명했다”가 아니라,
# “정상/고장 간 분포 차이와 시계열적 의미를 근거로 모델링 후보 Feature 7개를 선정했다”가 정확해.
