# Author: Drew Byron
# Date: 04/07/2023
"""
Description:   

"""
# Imports.
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy import stats

pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    "axes.titlesize": 15,
    "legend.fontsize": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}
plt.rcParams.update(params)


def scatter_grid_by_field(
    events_ne,
    events_he,
    feature1,
    feature2,
    cuts,
    normed_cols,
    nrows,
    yscale,
    bins,
    base_path="",
):

    fig, axs = plt.subplots(
        figsize=(12, 8),
        nrows=2,
        ncols=2,  # fix as above
        gridspec_kw=dict(hspace=0.4),
    )

    # Settings:
    ms = 0.5
    alpha = 0.3

    extra_cut = {
        "EventStartFreq": (200 * 1e6, 1200 * 1e6),
        "mMeanSNR_n": (0.5, np.inf),
    }

    extra_cuts = {**cuts, **extra_cut}

    # Without cuts
    events_ne = re.prep_events(events_ne, cuts, normed_cols)
    events_he = re.prep_events(events_he, cuts, normed_cols)

    events_ne = events_ne[events_ne.set_field > 1.1]
    events_he = events_he[events_he.set_field > 1.1]

    # With "from-below" cuts
    events_ne_cut = re.prep_events(events_ne, extra_cuts, normed_cols)
    events_he_cut = re.prep_events(events_he, extra_cuts, normed_cols)

    events_ne_cut = events_ne_cut[events_ne_cut.set_field > 1.1]
    events_he_cut = events_he_cut[events_he_cut.set_field > 1.1]

    ax_flat = axs.flatten()

    ax_flat[0].plot(
        events_ne[feature1],
        events_ne[feature2],
        "o",
        markersize=ms,
        alpha=alpha,
        label="Ne",
        color="g",
    )
    ax_flat[0].plot(
        events_he[feature1],
        events_he[feature2],
        "o",
        markersize=ms,
        alpha=alpha,
        label="He",
        color="c",
    )

    ax_flat[1].plot(
        events_ne_cut[feature1],
        events_ne_cut[feature2],
        "o",
        markersize=ms,
        alpha=alpha,
        label="Ne",
        color="g",
    )
    ax_flat[1].plot(
        events_he_cut[feature1],
        events_he_cut[feature2],
        "o",
        markersize=ms,
        alpha=alpha,
        label="He",
        color="c",
    )

    bins = 100

    ax_flat[2].hist(
        events_ne[feature1],
        bins=bins,
        histtype="step",
        density=True,
        label="Ne",
        color="g",
    )

    ax_flat[2].hist(
        events_he[feature1],
        bins=bins,
        histtype="step",
        density=True,
        label="He",
        color="c",
    )

    ax_flat[3].hist(
        events_ne_cut[feature1],
        bins=bins,
        histtype="step",
        density=True,
        label="Ne",
        color="g",
    )

    ax_flat[3].hist(
        events_he_cut[feature1],
        bins=bins,
        histtype="step",
        density=True,
        label="He",
        color="c",
    )
    # Add in horiz and vertical lines to represent the cut
    b = ax_flat[0].axhline(y = 0.5, xmin = 0, xmax = 1, color="r", linestyle="-", alpha = .5 )
    a = ax_flat[0].axvline(.2e9, 0,1.5, color="r", linestyle="-", label = "cuts", alpha = .5 )
    b = ax_flat[1].axhline(y = 0.5, xmin = 0, xmax = 1, color="r", linestyle="-", alpha = .5 )
    a = ax_flat[1].axvline(.2e9, 0,1.5, color="r", linestyle="-", label = "cuts", alpha = .5 )
    # Build legend for plots:

    # Create dummy Line2D objects for legend
    h1 = Line2D([0], [0], marker="o", markersize=4, color="g", linestyle="None")
    h2 = Line2D([0], [0], marker="o", markersize=4, color="c", linestyle="None")

    # Plot legend.
    ax_flat[0].legend(
        [h1, h2, a],
        ["Ne", "He", "cuts"],
        loc="upper right",
    )
    ax_flat[1].legend(
        [h1, h2, a],
        ["Ne", "He", "cuts"],
        loc="upper right",
    )

    ax_flat[2].legend()
    ax_flat[3].legend()
    ax_flat[2].set_yscale("log")
    ax_flat[3].set_yscale("log")

    ax_flat[0].set_xlabel("Start Freq (Hz)")
    ax_flat[1].set_xlabel("Start Freq (Hz)")
    ax_flat[2].set_xlabel("Start Freq (Hz)")
    ax_flat[3].set_xlabel("Start Freq (Hz)")

    ax_flat[0].set_ylabel("SNR (normalized)")

    ax_flat[0].set_ylim(-0.2, 1.5)
    ax_flat[1].set_ylim(-0.2, 1.5)

    ax_flat[0].set_xlim(0.08e9, 1.22e9)
    ax_flat[1].set_xlim(0.08e9, 1.22e9)
    ax_flat[2].set_xlim(0.08e9, 1.22e9)
    ax_flat[3].set_xlim(0.08e9, 1.22e9)

    fig_path = base_path + f"motivating_cuts_2_by_2_{feature1}_{feature2}.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)

    # =========Field-wise scatter plots===========

    # Field-wise plots
    grouped_ne = events_ne.groupby("set_field")
    grouped_he = events_he.groupby("set_field")
    # rowlength =
    fig, axs = plt.subplots(
        figsize=(12, 12),
        nrows=3,
        ncols=3,  # fix as above
        gridspec_kw=dict(hspace=0.8),
    )  # Much control of gridspec

    #
    group_keys = list(grouped_ne.groups.keys())
    print(group_keys)
    targets = zip(group_keys, axs.flatten())

    for i, (key, ax) in enumerate(targets):

        ne_field_1 = grouped_ne.get_group(key)[feature1]
        he_field_1 = grouped_he.get_group(key)[feature1]

        ne_field_2 = grouped_ne.get_group(key)[feature2]
        he_field_2 = grouped_he.get_group(key)[feature2]

        # Scatter Plots
        ax.plot(
            ne_field_1,
            ne_field_2,
            "o",
            markersize=ms,
            alpha=alpha,
            label="Ne",
            color="g",
        )
        ax.plot(
            he_field_1,
            he_field_2,
            "o",
            markersize=ms,
            alpha=alpha,
            label="He",
            color="c",
        )

        b = ax.axhline(y = 0.5, xmin = 0, xmax = 1, color="r", linestyle="-", alpha = .5 )
        a = ax.axvline(.2e9, 0,1.5, color="r", linestyle="-", label = "cuts", alpha = .5 )

        ax.set_title(f"Field: {key} T")
        ax.set_yscale(yscale)
        ax.set_xlabel("Start Freq (Hz)")

        if i % 3 == 0:
            ax.set_ylabel("SNR (normalized)")

        # Set axes limits
        # plt.gca().set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.5)

        # Create dummy Line2D objects for legend
        h1 = Line2D([0], [0], marker="o", markersize=4, color="g", linestyle="None")
        h2 = Line2D([0], [0], marker="o", markersize=4, color="c", linestyle="None")

        # Plot legend.
        if i == 2: 
            ax.legend(
                [h1, h2, a],
                ["Ne", "He", "cuts"],
                loc="upper right",
            )

    fig_path = base_path + f"scatt_field_wise_{feature1}_{feature2}.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)

    return None


# ----- Loop through all features and make a hist plot with and without cuts ----
event_cuts = {
    "uniform_rand": (0, 1),
    "EventTimeLength_n": (-np.inf, np.inf),
    "EventNBins": (0, np.inf),
    "EventStartTime": (0.0001, np.inf),
    "EventTimeLength": (0, 0.05),
    "EventTimeIntc": (-0.5e7, np.inf),
    "EventFreqIntc": (-2e11, np.inf),
    "EventSlope": (0.01e10, np.inf),
    "EventTrackCoverage": (0, np.inf),
    "mTotalSNR": (0, np.inf),
    "mTotalPower": (0, np.inf),
    "mTotalNUP": (0, np.inf),
    "mMaxSNR": (0, np.inf),
    "detectability": (0, np.inf),
    "mMeanSNR": (0, np.inf),
}

# Load the snr study.
snr_study = dl.load_snr_study()

# Can add to these lists to generate more plots.
# snr_cuts = [ .4]
# start_freqs = [200]

snr_cuts = [-np.inf]
start_freqs = [100]

# Make all possible pairs of the above cuts:
specific_cuts = np.array(np.meshgrid(snr_cuts, start_freqs)).T.reshape(-1, 2)
print(specific_cuts)

# Set parameters of plot.

nrows = 3
bins = 60
yscale = "linear"
# add_time_length_pdf = False

# Set fig path.
base_path = (
    "/media/drew/T7 Shield/thesis_figures/measurements/event_feature_scatterplots/"
)
fig_base_name = "MEAS_fig_4_"


features1 = ["EventStartFreq"]
features2 = ["mMeanSNR_n"]

normed_cols = ["mMeanSNR", "EventTimeLength", "uniform_rand"]

for feature1, feature2 in zip(features1, features2):

    for snr_cut, start_freq in specific_cuts:

        freq_cut = {
            "EventStartFreq": ((start_freq) * 1e6, (1200) * 1e6),
            "mMeanSNR_n": (snr_cut, np.inf),
        }

        spec_cuts = {**event_cuts, **freq_cut}

        print(f"Building hist for {feature1}_{feature2}.")

        fig_path = base_path + f"scatt_{feature1}_{feature2}.png"

        snr_threshold = 9

        events_ne = snr_study["ne"][snr_threshold].events
        events_he = snr_study["he"][snr_threshold].events

        scatter_grid_by_field(
            events_ne,
            events_he,
            feature1,
            feature2,
            spec_cuts,
            normed_cols,
            nrows,
            yscale,
            bins,
            base_path=base_path,
        )

plt.show()
