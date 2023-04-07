# Author: Drew Byron
# Date: 02/22/2023
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def cluster_and_clean_events(events, clust_params={}, diagnostics=False):

    if diagnostics:

        # Take stock of what events were like before the clustering.
        pre_clust_counts = events.groupby("set_field").file_id.count()
        pre_clust_summary_mean = events.groupby("set_field").mean()
        pre_clust_summary_std = events.groupby("set_field").std()
    
    # cluster
    events = cluster_events(events, clust_params=clust_params)

    # cleanup
    events = update_event_info(events)
    events = build_events(events)
    
    if diagnostics:

        # Take stock of what events were like after the clustering.
        post_clust_counts = events.groupby("set_field").file_id.count()
        post_clust_summary_mean = events.groupby("set_field").mean()
        post_clust_summary_std = events.groupby("set_field").std()

        print("Summary of clustring: \n")
        print(
            f"\nFractional reduction in counts from clustering:",
            post_clust_counts / pre_clust_counts,
        )
        print("\nPre-clustering means:")
        display(pre_clust_summary_mean)
        print("\nPre-clustering stds:")
        display(pre_clust_summary_std)
        print("\nPost-clustering means:")
        display(post_clust_summary_mean)
        print("\nPost-clustering stds:")
        display(post_clust_summary_std)

    return events


def cluster_events(events, clust_params={}):
    """Notes:
    * This is really clustering events not track segments.
    * Default up to 1/24/23 was .003 up to now.
    * On 1/24/23 1600, Drew is testing how .005 performs.
    """

    events_copy = events.copy()
    events_copy["event_label"] = np.NaN

    for i, (name, group) in enumerate(events_copy.groupby(["run_id", "file_id"])):

        set_field = group.set_field.mean()

        condition = (events_copy.run_id == name[0]) & (events_copy.file_id == name[1])

        events_copy.loc[condition, "event_label"] = dbscan_clustering(
            events_copy[condition],
            features=clust_params[set_field]["features"],
            eps=clust_params[set_field]["eps"],
            min_samples=1,
        )
    events_copy["EventID"] = events_copy["event_label"] + 1

    return events_copy


def dbscan_clustering(df, features: list, eps: float, min_samples: int):

    # Previously (incorrectly) used the standardscaler but
    # This meant there was a different normalization on each file!
    # X_norm = StandardScaler().fit_transform(df[features])

    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(df[features])
    labels = db.labels_

    return labels


def update_event_info(events_in: pd.DataFrame) -> pd.DataFrame:

    events = events_in.copy()

    events["EventStartTime"] = events.groupby(["run_id", "file_id", "EventID"])[
        "EventStartTime"
    ].transform("min")
    events["EventEndTime"] = events.groupby(["run_id", "file_id", "EventID"])[
        "EventEndTime"
    ].transform("max")

    events["EventStartFreq"] = events.groupby(["run_id", "file_id", "EventID"])[
        "EventStartFreq"
    ].transform("min")
    events["EventEndFreq"] = events.groupby(["run_id", "file_id", "EventID"])[
        "EventEndFreq"
    ].transform("max")

    events["EventTimeLength"] = events["EventEndTime"] - events["EventStartTime"]
    events["EventFreqLength"] = events["EventEndFreq"] - events["EventStartFreq"]
    events["EventNBins"] = events.groupby(["run_id", "file_id", "EventID"])[
        "EventNBins"
    ].transform("sum")

    events["EventSlope"] = events["EventFreqLength"] / events["EventTimeLength"]

    cols_to_average_over = [
        "EventTrackCoverage",
        "EventTrackTot",
        "EventFreqIntc",
        "EventTimeIntc",
        "mMeanSNR",
        "sMeanSNR",
        "mTotalSNR",
        "sTotalSNR",
        "mMaxSNR",
        "sMaxSNR",
        "mTotalNUP",
        "sTotalNUP",
        "mTotalPower",
        "sTotalPower",
        "field",
        "set_field",
        "monitor_rate",
    ]
    for col in cols_to_average_over:

        events[col] = events.groupby(["run_id", "file_id", "EventID"])[col].transform(
            "mean"
        )

    return events


def build_events(events: pd.DataFrame) -> pd.DataFrame:

    event_cols = [
        "run_id",
        "file_id",
        "EventID",
        "EventStartTime",
        "EventEndTime",
        "EventStartFreq",
        "EventEndFreq",
        "EventTimeLength",
        "EventFreqLength",
        "EventTrackCoverage",
        "EventSlope",
        "EventNBins",
        "EventTrackTot",
        "EventFreqIntc",
        "EventTimeIntc",
        "mMeanSNR",
        "sMeanSNR",
        "mTotalSNR",
        "sTotalSNR",
        "mMaxSNR",
        "sMaxSNR",
        "mTotalNUP",
        "sTotalNUP",
        "mTotalPower",
        "sTotalPower",
        "field",
        "set_field",
        "monitor_rate",
    ]
    events = (
        events.groupby(["run_id", "file_id", "EventID"])
        .first()
        .reset_index()[event_cols]
    )

    return events
