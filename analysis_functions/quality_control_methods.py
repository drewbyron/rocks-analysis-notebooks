# Author: Drew Byron
# Date: 02/22/2023
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np


def events_summary(events, title=""):
    """
    Prints a summary of what run_ids belong to what set field and
    """

    print(f"\n {title}: List of run_ids vs set_field in events df")
    print(events.groupby(["set_field"]).run_id.unique().reset_index())
    print("=================\n")

    fig, ax = plt.subplots(figsize=(12, 4))
    events.groupby(["run_id"]).file_id.count().to_frame("counts").reset_index().plot(
        x="run_id",
        title=f"{title}: event count vs run_id",
        ax=ax,
        ls="dotted",
        marker="o",
    )

    return None


def root_files_summary(root_files, title=""):
    """
    Prints a summary of what run_ids belong to what set field and
    """

    print(f"\n {title}: List of run_ids vs set_field in root_files df")
    print(root_files.groupby(["set_field"]).run_id.unique().reset_index())
    print("=================\n")

    return None


def viz_events(events, file_id_max=1, set_fields=[1, 2, 3], title=""):

    events["normed_power"] = (events.mMeanSNR - events.mMeanSNR.quantile(0.01)) / (
        events.mMeanSNR.quantile(0.99) - events.mMeanSNR.quantile(0.01)
    )

    events["normed_power"] = events["normed_power"].clip(0, 1)

    grouped = events.groupby(["run_id", "file_id"])

    for i, (name, group) in enumerate(grouped):

        set_field = group.set_field.mean()

        if (name[1] < file_id_max) and (set_field in set_fields):

            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ["b", "r", "g", "c", "m", "k"]

            for row_index, row in group.iterrows():

                time_coor = np.array([row["EventStartTime"], row["EventEndTime"]])
                freq_coor = np.array([row["EventStartFreq"], row["EventEndFreq"]])

                ax.plot(
                    time_coor,
                    freq_coor,
                    color=str(row.normed_power),
                    alpha=1,
                    label="EventID = {}".format(row["EventID"]),
                    marker="o",
                    markerfacecolor="red",
                    markersize=4,
                )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1200e6)

            ax.set_ylabel("Freq (Hz)")
            ax.set_xlabel("Time (s)")
            plt.title(f"{title}. {set_field} T. rid: {name[0]}, fid: {name[1]}. events: {len(group)}")
            plt.show()

    return None
