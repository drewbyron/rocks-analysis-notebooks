# Author: Drew Byron
# Date: 02/22/2023
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np


def plot_experimental_ratio(ratio_exp, ax, label=None):

    if label is None:
        label = f"Ratio, experiment"
    ax.errorbar(
        ratio_exp.index,
        ratio_exp.Ratio,
        yerr=ratio_exp["sRatio"],
        label=label,
        marker="o",
        ms=4,
        ls="None",
    )

    return None


def plot_predicted_ratio(ratio_pre, ax, label = None):

    if label is None:
        label = f"Ratio, predicted"
    ax.plot(ratio_pre.index, ratio_pre.Ratio, label=label, marker="o", ms=6)

    return None
