# 반복 사용 시각화
import matplotlib.pyplot as plt
import seaborn as sns


def plot_failure_boxplot(df, failure_type, feature):
    data = df[(df["Machine failure"] == 0) | (df[failure_type] == 1)]

    plt.figure(figsize=(6, 4))

    sns.boxplot(data=data, x=failure_type, y=feature)

    plt.title(f"Normal vs {failure_type} - {feature}")
    plt.show()
