import numpy as np


def add_features(df):
    data = df.copy()

    # HDF 관련
    data["Temperature difference [K]"] = (
        data["Process temperature [K]"] - data["Air temperature [K]"]
    )

    # PWF 관련
    data["Power [W]"] = (
        data["Torque [Nm]"] * data["Rotational speed [rpm]"] * 2 * np.pi / 60
    )

    # OSF 관련
    data["Wear torque"] = data["Tool wear [min]"] * data["Torque [Nm]"]

    return data
