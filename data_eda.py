import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from visualization import plot_failure_boxplot

df = pd.read_csv("ai4i2020.csv")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# print(df[["Machine failure"]].value_counts().to_dict())

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

normal = df[df["Machine failure"] == 0]
failure = df[df["Machine failure"] == 1]

normal_corr = normal[features].corr()
failure_corr = failure[features].corr()

# print("[정상 상관관계]")
# print(normal_corr)

# print("\n[고장 상관관계]")
# print(failure_corr)


# plt.figure(figsize=(8, 6))

# sns.heatmap(normal_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)

# plt.title("Normal Correlation")
# plt.show()

# plt.figure(figsize=(8, 6))

# sns.heatmap(failure_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)

# plt.title("Failure Correlation")
# plt.show()

corr_diff = failure_corr - normal_corr

# print(corr_diff)

failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# print(df[failure_types].sum())

df["failure_type_count"] = df[failure_types].sum(axis=1)

# print(df["failure_type_count"].value_counts().sort_index())

# print(pd.crosstab(df["Machine failure"], df["failure_type_count"]))

sensor_mean = pd.DataFrame()

sensor_mean["Normal"] = df[df["Machine failure"] == 0][features].mean()

for failure_type in failure_types:
    sensor_mean[failure_type] = df[df[failure_type] == 1][features].mean()

# print(sensor_mean.round(2))

#### 고장 유형별 센서 분포 분석 ####
# plot_failure_boxplot(df, "TWF", "Tool wear [min]")

# plot_failure_boxplot(df, "HDF", "Rotational speed [rpm]")
# plot_failure_boxplot(df, "HDF", "Torque [Nm]")

# plot_failure_boxplot(df, "PWF", "Rotational speed [rpm]")
# plot_failure_boxplot(df, "PWF", "Torque [Nm]")

# plot_failure_boxplot(df, "OSF", "Rotational speed [rpm]")
# plot_failure_boxplot(df, "OSF", "Torque [Nm]")
# plot_failure_boxplot(df, "OSF", "Tool wear [min]")

pwf_data = df[(df["Machine failure"] == 0) | (df["PWF"] == 1)]

# plt.figure(figsize=(8, 6))

# sns.scatterplot(
#     data=pwf_data, x="Rotational speed [rpm]", y="Torque [Nm]", hue="PWF", alpha=0.6
# )

# plt.title("Normal vs PWF - Rotational Speed & Torque")
# plt.show()

df["Power [W]"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * 3.141592 / 60

# print(df["Power [W]"])

# print(df[df["PWF"] == 0]["Power [W]"].describe())
# print()
# print(df[df["PWF"] == 1]["Power [W]"].describe())

pwf_data = df[(df["Machine failure"] == 0) | (df["PWF"] == 1)]

# sns.boxplot(data=pwf_data, x="PWF", y="Power [W]")

# plt.title("Normal vs PWF - Power")
# plt.show()

# HDF 후보
# Heat Dissipation Failure니까 단순히 Air temperature 하나보다
# 공정온도와 외기온도의 차이가 열 방출 상태를 더 잘 나타낼 가능성 있음
df["Temperature difference [K]"] = (
    df["Process temperature [K]"] - df["Air temperature [K]"]
)

# boxplot에서 Tool wear ↑, Torque ↑가 동시에 아주 뚜렷함
# 두 값의 누적 부하 성격을 한 변수로 표현할 수 있는지
df["Wear torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]

# print(df[df["HDF"] == 0]["Temperature difference [K]"].describe())
# print(df[df["HDF"] == 1]["Temperature difference [K]"].describe())

# print()

# print(df[df["OSF"] == 0]["Wear torque"].describe())
# print(df[df["OSF"] == 1]["Wear torque"].describe())

failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# # Type별 전체 데이터 수
# print("[Type별 데이터 수]")
# print(df["Type"].value_counts())

# # Type별 고장 유형 발생 건수
# type_failure_count = df.groupby("Type")[failure_types].sum()

# print("\n[Type별 고장 유형 발생 건수]")
# print(type_failure_count)

# # Type별 고장 유형 발생률
# type_failure_rate = df.groupby("Type")[failure_types].mean() * 100

# print("\n[Type별 고장 유형 발생률(%)]")
# print(type_failure_rate.round(2))

# =========================
# 모델링 데이터 구성
# =========================

failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature difference [K]",
    "Power [W]",
    "Wear torque",
]

X = df[features]
y = df[failure_types]

# print("[X]")
# print(X.head())

# print("\n[y]")
# print(y.head())

# print("\nX shape:", X.shape)
# print("y shape:", y.shape)

# print("\n고장 유형별 개수")
# print(y.sum())

# X = pd.get_dummies(X, columns=["Type"], dtype=int)

# print(X.head())
# print(X.columns)
# print(X.shape)
