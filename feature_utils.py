# 파생변수 생성
import numpy as np


def add_features(df):

    data = df.copy()

    data["Temperature difference [K]"] = (
        data["Process temperature [K]"] - data["Air temperature [K]"]
    )

    data["Power [W]"] = (
        data["Torque [Nm]"] * data["Rotational speed [rpm]"] * 2 * np.pi / 60
    )

    data["Wear torque"] = data["Tool wear [min]"] * data["Torque [Nm]"]

    return data
