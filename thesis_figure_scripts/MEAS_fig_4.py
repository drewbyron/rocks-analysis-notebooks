# Author: Drew Byron
# Date: 04/07/2023
"""
Description: This module makes the field-wise histogram that is used to 
argue that the .75 T and 1.25 T fields should be dropped. The event 
length distributions are notably different.  

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


def hist_grid_by_field(
    events_ne,
    events_he,
    feature,
    cuts,
    normed_cols,
    nrows,
    yscale,
    bins,
    fig_path="",
    add_time_length_pdf=False,
):

    events_ne = re.prep_events(events_ne, cuts, normed_cols)
    events_he = re.prep_events(events_he, cuts, normed_cols)

    grouped_ne = events_ne.groupby("set_field")
    grouped_he = events_he.groupby("set_field")
    rowlength = int(np.ceil(grouped_ne.ngroups / 4))
    fig, axs = plt.subplots(
        figsize=(12, 12),
        nrows=4,
        ncols=3,  # fix as above
        gridspec_kw=dict(hspace=0.8),
    )  # Much control of gridspec

    targets = zip(grouped_ne.groups.keys(), axs.flatten())

    for i, (key, ax) in enumerate(targets):


        ne_field = grouped_ne.get_group(key)[feature]
        he_field = grouped_he.get_group(key)[feature]

        if feature == "EventTimeLength" and add_time_length_pdf:
            freq_BW = cuts["EventStartFreq"][1] - cuts["EventStartFreq"][0]
            print(freq_BW)
            ls, pdf = track_len_pdf_field(field=key, slew_cycle=35e-3, freq_BW=freq_BW)
            ax.plot(ls, pdf, color=str(0), label="pdf")

        # Calculate KS test for similarity of samples.
        ks = stats.kstest(ne_field, he_field)

        ax.hist(
            ne_field, bins=bins, histtype="step", density=True, label="Ne", color="g"
        )
        ax.hist(
            he_field, bins=bins, histtype="step", density=True, label="He", color="c"
        )
        ax.set_title(f"Field: {key} T")
        ax.set_yscale(yscale)
        ax.set_xlabel("Event Length (s)")

        if i in [0,1]: ax.legend(loc = "upper right")
        else: ax.legend(loc = "lower left")

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
start_freqs = [0]

# Make all possible pairs of the above cuts:
specific_cuts = np.array(np.meshgrid(snr_cuts, start_freqs)).T.reshape(-1, 2)
print(specific_cuts)

# Set parameters of plot.

nrows = 3
bins = 60
yscale = "log"
add_time_length_pdf = False

# Set fig path.
base_path = (
    "/media/drew/T7 Shield/thesis_figures/measurements/event_feature_histograms/"
)
fig_base_name = "MEAS_fig_4_"

features = ["EventTimeLength"]
# features = [ "EventStartFreq"]
normed_cols = ["mMeanSNR", "EventTimeLength", "uniform_rand"]

for feature in features:

    for snr_cut, start_freq in specific_cuts:

        freq_cut = {
            "EventStartFreq": ((start_freq) * 1e6, (1200) * 1e6),
            "mMeanSNR_n": (snr_cut, np.inf),
        }

        spec_cuts = {**event_cuts, **freq_cut}

        print(f"Building hist for {feature}. cut = {snr_cut}.")

        fig_path = base_path + f"events_hist_{feature}_cutting_low_fields.png"

        snr_threshold = 9

        events_ne = snr_study["ne"][snr_threshold].events
        events_he = snr_study["he"][snr_threshold].events

        hist_grid_by_field(
            events_ne,
            events_he,
            feature,
            spec_cuts,
            normed_cols,
            nrows,
            yscale,
            bins,
            fig_path=fig_path,
            add_time_length_pdf=add_time_length_pdf,
        )

plt.show()
