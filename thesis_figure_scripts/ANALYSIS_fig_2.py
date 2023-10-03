# Author: Drew Byron
# Date: 04/07/2023
"""
Description: This module makes a ratio plot that aggregates the results
and errors of the SNR study. Here we use the "from-below" (fb)
monte carlo, so we don't attempt to cut events coming from below. We
fit to all three free parameters (C, slew_time, b). Alternatively we 
could fix one or more of those parameters by changing the vary argument 
of the lmfit parameter object to false.

"""
# Imports.
import sys
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
from pathlib import Path

# Path to local imports.
sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")
sys.path.append("/home/drew/He6CRES/rocks_analysis_notebooks/")
sys.path.append("/home/drew/He6CRES/rocks_analysis_pipeline/")


# Local imports from rocks_analysis_pipeline.
from results import ExperimentResults
from rocks_utility import he6cres_db_query

# Local imports from he6-cres-spec-sims.
import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc
import he6_cres_spec_sims.experiment as exp
import he6_cres_spec_sims.spec_tools.beta_source.beta_spectrum as bs

# Local imports from rocks_analysis_notebooks.
import analysis_functions.ratio_experiment as re
import analysis_functions.ratio_prediction as rp
import analysis_functions.quality_control_methods as qc
import analysis_functions.plotting_methods as pm
import analysis_functions.event_clustering as clust
import analysis_functions.snr_tests as snr_tests

import mc_functions.simple_mc as mc
import mc_functions.wall_effect as we
import mc_functions.energy_domain as ed
import mc_functions.from_below as fb

import thesis_figure_scripts.data_loaders as dl

# Set plot parameters.
params = {
    "axes.titlesize": 20,
    "legend.fontsize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}
plt.rcParams.update(params)

# 10th Percentile
def q10(x):
    return x.quantile(0.1)


# 50th Percentile
def q50(x):
    return x.quantile(0.5)


# 90th Percentile
def q90(x):
    return x.quantile(0.9)


def make_viz_0(fig_path, snr_study):

    # Impose SNR Cut and stuff:
    snr = 9
    events_ne = snr_study["ne"][snr].tracks
    events_he = snr_study["he"][snr].tracks

    print(events_ne.columns)

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(
        events_ne.StartFrequency,
        events_ne.MeanTrackSNR,
        marker="o",
        ls="None",
        ms=1.5,
        color="g",
        # marker = "o",
        alpha=1,
        label="Ne19",
    )

    plt.plot(
        events_he.StartFrequency,
        events_he.MeanTrackSNR,
        marker="x",
        ls="None",
        ms=2,
        color="r",
        alpha=0.1,
        # marker = "s",
        label="He6",
    )

    # Add bigger legend markers
    # Create dummy Line2D objects for legend
    h1 = plt.Line2D([0], [0], marker="o", markersize=4, color="g", linestyle="None")
    h2 = plt.Line2D([0], [0], marker="o", markersize=4, color="r", linestyle="None")

    # Plot legend.
    ax.legend(
        [h1, h2],
        ["Ne19", "He6"],
        loc="upper right",
    )
    ax.set_ylabel("Mean Track Segment SNR")
    ax.set_xlabel("Frequency (MHz)")
    # Save and display the figure.
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    return None


def make_viz_1(fig_path, snr_study):

    # Impose SNR Cut and stuff:
    snr = 9
    events_ne = snr_study["ne"][snr].events
    events_he = snr_study["he"][snr].events

    # aggs = [q10, q50, q90]
    # col_names = ["10", "50", "90"]

    aggs = [q50]
    col_names = ["50"]

    snr_cuts = np.arange(8, 11, 1)

    features = ["mMeanSNR"]
    for column in features:

        fig0, ax0 = plt.subplots(figsize=(16, 8))
        for i, cut in enumerate(snr_cuts):

            events_ne = snr_study["ne"][cut].events
            events_he = snr_study["he"][cut].events

            # events_ne = re.add_detectability(events_ne)
            # events_he = re.add_detectability(events_he)

            # # print(events_ne.EventStartFreq.min())
            # events_cut_ne = re.cut_df(events_ne, ne_cuts)
            # events_cut_he = re.cut_df(events_he, he_cuts)
            # print(events_cut_ne.EventStartFreq.min())
            ne_col_names = [col_name + f"_ne_{cut}" for col_name in col_names]
            he_col_names = [col_name + f"_he_{cut}" for col_name in col_names]
            orig_col_names = ["q" + col_name for col_name in col_names]

            events_ne.groupby("set_field")[column].agg(aggs).rename(
                columns=dict(zip(orig_col_names, ne_col_names))
            ).plot(ax=ax0, marker=f"{i+1}", ms=10)
            events_he.groupby("set_field")[column].agg(aggs).rename(
                columns=dict(zip(orig_col_names, he_col_names))
            ).plot(ax=ax0, ls="dotted", marker=f"{i+1}", ms=10)
        plt.title(f"snr_cuts = {snr_cuts}. {column}")
        plt.legend()

    # Save and display the figure.
    fig0.savefig(fig_path, bbox_inches="tight", dpi=300)
    return None


def make_viz_2(fig_path, snr_study):

    # Impose SNR Cut and stuff:
    snr = 9
    events_ne = snr_study["ne"][snr].events
    events_he = snr_study["he"][snr].events

    field_err_ne = events_ne.groupby("set_field")["field"].mean()
    field_err_ne = (field_err_ne.index - field_err_ne) / field_err_ne.index
    field_err_ne_std = events_ne.groupby("set_field")["field"].std() / field_err_ne.index

    field_err_he = events_he.groupby("set_field")["field"].mean()
    field_err_he = (field_err_he.index - field_err_he) / field_err_he.index
    field_err_he_std = events_he.groupby("set_field")["field"].std() / field_err_he.index

    print(field_err_he)
    print(field_err_he_std)
    # print(field_err_he.std())

    fig0, ax0 = plt.subplots(figsize=(12, 6))

    ax0.errorbar(
    field_err_ne.index,
    field_err_ne.values * 1e6,
    yerr=field_err_ne_std.values * 1e6,
    fmt="o",
    color="g",
    capsize=5,
    label="Ne19",)

    ax0.errorbar(
    field_err_he.index,
    field_err_he.values * 1e6,
    yerr=field_err_he_std.values * 1e6,
    fmt="o",
    color="c",
    capsize=5,
    label="He6",)
    # ax0.plot(
    #     field_err_ne.index,
    #     field_err_ne.values * 1e6,
    #     label="Ne19",
    #     marker="o",
    #     ls="None",
    #     ms=8,
    #     color="g",
    # )
    # ax0.plot(
    #     field_err_he.index,
    #     field_err_he.values * 1e6,
    #     label="He6",
    #     marker="o",
    #     ls="None",
    #     ms=8,
    #     color="c",
    # )
    ax0.set_ylim(-255, 155)

    ax0.set_ylabel("Main Field Error (ppm)")
    ax0.set_xlabel("Set Field (T)")
    ax0.grid(ls="-", linewidth=1)
    ax0.legend(loc="upper right")

    # Save and display the figure.
    fig0.savefig(fig_path, bbox_inches="tight", dpi=300)
    return None


# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/measurements/data_analysis")

# Load the snr study.
snr_study = dl.load_snr_study()

fig_name = "ANALYSIS_2_test_{}.png"
fig_path = fig_dir / fig_name

make_viz_0(fig_dir / Path(fig_name.format(0)), snr_study)
# make_viz_1(fig_dir / Path(fig_name.format(1)), snr_study)
make_viz_2(fig_dir / Path(fig_name.format(2)), snr_study)

plt.show()
