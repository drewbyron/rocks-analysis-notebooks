# Author: Drew Byron
# Date: 2/2/23

import sys
import pandas as pd
import numpy as np


def cut_df(events, cuts):

    events_cut = events.copy()

    for column, cut in cuts.items():
        events_cut = events_cut[
            (events_cut[column] >= cut[0]) & (events_cut[column] <= cut[1])
        ]

    return events_cut


def build_normalization(root_files):

    condition = root_files.root_file_exists == True
    normalization = (
        root_files[condition]
        .groupby(["run_id", "file_id"])[["monitor_rate", "set_field"]]
        .first()
        .reset_index()
    )

    return normalization


def build_spectrum(events, root_files, cuts):
    """Builds the field-wise normalized spectrum from the events and root_files dfs.
    Example cuts: cuts = {"EventStartFreq": (100e6, 1200e6), "EventNBins": (0, np.inf)}
    Document right asap.

    TODO: Should make the cuts specific to the two isotopes.
    """

    # Collect all valid events.
    valid_events = cut_df(events, cuts)

    # Build a normalization df that contains the monitor rate for each existing root file.
    normalization = build_normalization(root_files)

    # Count the number of events per file.
    event_counts = (
        valid_events.groupby(["run_id", "file_id"])
        .EventID.count()
        .to_frame("event_count")
        .reset_index()
    )

    # Merge the above with the normalization df.
    spec1 = pd.merge(
        normalization,
        event_counts,
        how="left",
        left_on=["run_id", "file_id"],
        right_on=["run_id", "file_id"],
    )
    spec1.fillna(0, inplace=True)

    # Now sum over set field, getting total counts per field and the total monitor_rate.
    # Note that this DOES correctly account for files with zero events, as their monitor_rate is still summed.
    spec2 = (
        spec1.groupby(["set_field"])[["event_count", "monitor_rate"]]
        .sum()
        .reset_index()
    )

    # Now that the summation has been done, rename the monitor rate column.
    spec2 = spec2.rename(columns={"monitor_rate": "tot_monitor_rate"})

    # Now also count the number of total files you have for each field.
    spec3 = (
        spec1.groupby(["set_field"])
        .event_count.count()
        .to_frame("seconds_of_data")
        .reset_index()
    )

    # Merge those two dfs
    spec4 = pd.merge(
        spec2, spec3, how="inner", left_on=["set_field"], right_on=["set_field"]
    )

    # Define the normed (mean = 1) tot_mon_rate.
    spec4["normed_tot_monitor_rate"] = (
        spec4["tot_monitor_rate"] / spec4["tot_monitor_rate"].mean()
    )

    # Use this norm 1 array to adjust the counts seen.
    spec4["mon_adjusted_count"] = (
        spec4["event_count"] / spec4["normed_tot_monitor_rate"]
    )

    # Assign a root N uncertainty to the array using the actual number of observed events.
    spec4["mon_adjusted_count_uncert"] = spec4["event_count"] ** 0.5

    return spec4


def build_ratio(ne_spectrum, he_spectrum):
    """Builds the field-wise normalized spectra and the ratio, with undertainties."""

    ratio = pd.DataFrame()
    ratio["Ne19"] = (
        ne_spectrum["mon_adjusted_count"] / ne_spectrum["mon_adjusted_count"].sum()
    )
    ratio["He6"] = (
        he_spectrum["mon_adjusted_count"] / he_spectrum["mon_adjusted_count"].sum()
    )

    ratio["sNe19"] = (
        ne_spectrum["mon_adjusted_count_uncert"]
        / ne_spectrum["mon_adjusted_count"].sum()
    )
    ratio["sHe6"] = (
        he_spectrum["mon_adjusted_count_uncert"]
        / he_spectrum["mon_adjusted_count"].sum()
    )

    ratio["Ratio"] = ratio["Ne19"] / ratio["He6"]

    ratio["sRatio"] = (
        ratio["Ratio"]
        * (
            1 / ne_spectrum["mon_adjusted_count"]
            + 1 / he_spectrum["mon_adjusted_count"]
        )
        ** 0.5
    )

    ratio["set_field"] = ne_spectrum["set_field"]
    ratio.set_index("set_field", inplace=True)

    return ratio


def build_ratio_altnorm(ne_spectrum, he_spectrum):
    """Builds the field-wise normalized spectra and the ratio, with undertainties. 
    Use the alternate way of normalizing where you don't set mean to 1."""

    ratio = pd.DataFrame()
    ratio["Ne19"] = (
        ne_spectrum["event_count"] / ne_spectrum["tot_monitor_rate"]
    )
    ratio["He6"] = (
        he_spectrum["event_count"] / he_spectrum["tot_monitor_rate"]
    )

    ratio["Ratio"] = ratio["Ne19"] / ratio["He6"]

    ratio["sRatio"] = (
        ratio["Ratio"]
        * (
            1 / ne_spectrum["event_count"]
            + 1 / he_spectrum["event_count"]
        )
        ** 0.5
    )

    ratio["set_field"] = ne_spectrum["set_field"]
    ratio.set_index("set_field", inplace=True)

    return ratio