import matplotlib.pyplot as plt
import platform


def set_korean_font():
    font = "Malgun Gothic" if platform.system() == "Windows" else "AppleGothic"
    plt.rc("font", family=font)
    plt.rcParams["axes.unicode_minus"] = False
