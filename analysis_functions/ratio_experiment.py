# Author: Drew Byron
# Date: 2/2/23
# Description: Module containing the machinery necessary to build the
# experimental ratio.
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def build_spectrum(
    events,
    root_files,
    cuts,
    normed_cols=["mMeanSNR", "EventTimeLength", "uniform_rand"],
    diagnostics=False,
):
    """Builds extra features (specified by normed_cols),
    then applies cuts, then builds the field-wise normalized spectrum
    using the events and root_files dfs.

    Args:
    events (pd.DataFrame): set of events as output by rocks_analysis_pipeline
        post-processor (written to events.csv).
    root_files (pd.DataFrame): set of rootfiles as output by rocks_analysis_pipeline
        post-processor (written to root_files.csv). This is needed for the
        monitor normalization.
    cuts (dict): Cuts to be applied to the events before building the spectra.
        The format of the cuts follows "{col_name}": (min_val, max_val).
        Instead of making more complicated cuts try (if possible) to make
        a new feature such that the cuts can still follow this simple rule.
        Example: cuts = {"EventStartFreq": (100e6, 1200e6),
                         "EventNBins": (50, np.inf)}
    normed_cols (List(str)): A list of column names (must be present in
        events df) for which to add field_wise_percentage and field_wise_norm
        features columns for. Note that cuts can be made on these features.
    diagnostics (bool): whether or not to print diagnostics that tell the
        user about the effect of the cuts applied.
    """

    # Add composite or normalized features.
    events = add_detectability(events)
    events = add_uniform_rand(events)
    events = add_field_wise_percentage(events, cols=normed_cols)
    events = add_field_wise_norm(events, cols=normed_cols)

    # Collect all valid events.
    valid_events = cut_df(events, cuts)

    if diagnostics:
        # Take stock of what events were like before the cuts.
        pre_cuts_counts = events.groupby("set_field").file_id.count()
        # Take stock of what events were like after the cuts.
        post_cuts_counts = valid_events.groupby("set_field").file_id.count()

        print("Fractional change in counts:")
        print(post_cuts_counts / pre_cuts_counts)

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

    return spec4


def add_detectability(events, slew_time=35e-3):
    """Adds a new feature that is the normalized event "length" (this
    is the max possible hypotenous for our slew cycle) times the mMeanSNR.
    This should be a proxy for how detectable a track is but use at your
    own risk.

    Args:
    events (pd.DataFrame): set of events as output by rocks_analysis_pipeline
        post-processor (written to events.csv).
    """
    events["detectability"] = (
        (events["EventTimeLength"] / slew_time) ** 2
        + (events["EventFreqLength"] / 1200e6) ** 2
    ) ** 0.5 * events["mMeanSNR"]

    return events


def add_uniform_rand(events, seed=12345):
    """Adds a uniform random (0,1) feature to the events df. This is useful
    for making a purely statistical cut that preserves all other disributions.
    Used to compare a cut's efficacy to a random cut.

    Args:
    events (pd.DataFrame): set of events as output by rocks_analysis_pipeline
        post-processor (written to events.csv).
    """
    events_copy = events.copy()
    rand_col = "uniform_rand"
    rng = np.random.default_rng(seed=seed)
    events_copy[rand_col] = rng.uniform(0, 1, events_copy.shape[0])

    return events_copy


def add_field_wise_percentage(events, cols=["mMeanSNR", "uniform_rand"]):
    """Adds a new set of features (specified by cols) that ranks each
    event by the field-wise percentile according to the given column
    and writes that to a new column: "{col}_p".

    Args:
    events (pd.DataFrame): set of events as output by rocks_analysis_pipeline
        post-processor (written to events.csv).
    cols (List(str)): list of columns for which to make this additional
        feature.
    """
    events_copy = events.copy()
    rand_col = "uniform_rand"
    events_copy[rand_col] = np.random.uniform(0, 1, events_copy.shape[0])

    for col in cols:
        col_perc = col + "_p"
        events_copy[col_perc] = np.NaN
        for name, group in events_copy.groupby(["set_field"]):
            cond = events_copy.set_field == name
            sz = group[col].size - 1
            events_copy.loc[cond, (col_perc)] = (
                events_copy.loc[cond, (col)]
                .rank(method="max")
                .apply(lambda x: (x - 1) / sz)
            )

    return events_copy


def add_field_wise_norm(events, cols=["mMeanSNR", "uniform_rand"]):
    """Adds a new set of features (specified by cols) that normalizes
    the given column to be in (0,1) and writes that to a new column:
    "{col}_n".

    Args:
    events (pd.DataFrame): set of events as output by rocks_analysis_pipeline
        post-processor (written to events.csv).
    cols (List(str)): list of columns for which to make this additional
        feature.
    """
    events_copy = events.copy()

    for col in cols:
        col_mn = col + "_n"
        events_copy[col_mn] = np.NaN
        for name, group in events_copy.groupby(["set_field"]):

            # Note that the 99% and 1% are used instead of max and min
            # in order to reduce the effect of outliers.
            cond = events_copy.set_field == name
            col_vals = events_copy.loc[cond, (col)].copy()
            events_copy.loc[cond, (col_mn)] = (col_vals - col_vals.quantile(0.01)) / (
                col_vals.quantile(0.99) - col_vals.quantile(0.01)
            )

    return events_copy


def cut_df(events, cuts, diagnostics=False):

    # Make a copy of events to alter and return.
    events_cut = events.copy()
    cuts = cuts.copy()

    if diagnostics:

        # Take stock of what events were like before the clustering.
        pre_clust_counts = events_cut.groupby("set_field").file_id.count()
        pre_clust_summary_mean = events_cut.groupby("set_field").mean()

    # Cut first
    for column, cut in cuts.items():

        events_cut = events_cut[
            (events_cut[column] >= cut[0]) & (events_cut[column] <= cut[1])
        ]
    if diagnostics:

        # Take stock of what events were like after the clustering.
        post_clust_counts = events_cut.groupby("set_field").file_id.count()
        post_clust_summary_mean = events_cut.groupby("set_field").mean()

        print("Summary of cut: \n")
        print(
            f"\nFractional reduction in counts from cut:",
            post_clust_counts / pre_clust_counts,
        )
        print("\nPre-cut means:")
        display(pre_clust_summary_mean)
        print("\nPost-cut means:")
        display(post_clust_summary_mean)

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



def build_ratio(ne_spectrum, he_spectrum):
    """Builds the ratio based on the ne and he spectra as output by the
    build_spectrum() function above. This normalization scheme needs to
    be used in conjunction with a normalization factor C obtained from a 
    chisq fit to the predicted spectrum. 

    Args:
    ne_spectrum (pd.DataFrame): Neon spectrum df containing the total number of 
        events (post cuts) and total associated monitor rate for Neon. 
    he_spectrum (pd.DataFrame): Helium spectrum df containing the total number of 
        events (post cuts) and total associated monitor rate for Helium.
    """

    ratio = pd.DataFrame()
    ratio["Ne19"] = ne_spectrum["event_count"] / ne_spectrum["tot_monitor_rate"]
    ratio["He6"] = he_spectrum["event_count"] / he_spectrum["tot_monitor_rate"]

    ratio["Ratio"] = ratio["Ne19"] / ratio["He6"]

    # NEW (as of 4/10/23) definition of the uncertainty, adding in mon err.
    ratio["sRatio"] = (
        ratio["Ratio"]
        * (
            1 / ne_spectrum["event_count"]
            + 1 / he_spectrum["event_count"]
            + 1 / ne_spectrum["tot_monitor_rate"]
            + 1 / ne_spectrum["tot_monitor_rate"]
        )
        ** 0.5
    )
    ratio["set_field"] = ne_spectrum["set_field"]
    ratio.set_index("set_field", inplace=True)

    return ratio

def build_ratio_altnorm(ne_spectrum, he_spectrum):
    """Builds the ratio based on the ne and he spectra as output by the
    build_spectrum() function above. This normalization scheme normalizes
    all arrays to mean = 1 which doesn't work very well. Could be thought
    more about. 

    Args:
    ne_spectrum (pd.DataFrame): Neon spectrum df containing the total number of 
        events (post cuts) and total associated monitor rate for Neon. 
    he_spectrum (pd.DataFrame): Helium spectrum df containing the total number of 
        events (post cuts) and total associated monitor rate for Helium.
    """
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

# ===== OLD CODE FOR FUTURE REFERENCE ======

# Code for field-wise cuts for future reference (can add into cuts method above)
"""
# Note that the None prevents a key error if you have no field-wise cuts.
field_wise_cuts = cuts.pop("field_wise", None)

if field_wise_cuts is not None:
    for column, cut in field_wise_cuts.items():

        events_cut["cut_cond"] = events.groupby(["set_field"])[column].transform(
            cut[0]
        )
        if plot_fw_cuts:
            events_cut["cut_cond"].plot()
            plt.show()

        cond = (events_cut[column] >= cut[1][0] * events_cut["cut_cond"]) & (
            events_cut[column] <= cut[1][1] * events_cut["cut_cond"]
        )

        events_cut = events_cut[cond]
"""

# Following was part of the build_spectrum
# # Define the normed (mean = 1) tot_mon_rate.
# spec4["normed_tot_monitor_rate"] = (
#     spec4["tot_monitor_rate"] / spec4["tot_monitor_rate"].mean()
# )

# # Use this norm 1 array to adjust the counts seen.
# spec4["mon_adjusted_count"] = (
#     spec4["event_count"] / spec4["normed_tot_monitor_rate"]
# )

# # Assign a root N uncertainty to the array using the actual number of observed events.
# spec4["mon_adjusted_count_uncert"] = spec4["event_count"] ** 0.5
