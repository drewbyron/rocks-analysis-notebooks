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


def make_viz_0(fig_path):

    # Impose SNR Cut and stuff:
    # snr = 9
    events_ne = pd.read_csv("/media/drew/T7 Shield/rocks_analysis/saved_experiments/ne19_full_0_aid_2/events.csv")
    events_he = pd.read_csv("/media/drew/T7 Shield/rocks_analysis/saved_experiments/he6_full_0_aid_1/events.csv")

    # filter bad files
    events_he["bad_file"] = False
    events_he.loc[events_he["run_id"].isin([380,381,377]) , "bad_file"] = True
    events_he = events_he[events_he["bad_file"] == False]

    print(events_ne.columns)

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(
        events_ne.EventTimeLength*1000,
        events_ne.EventSlope/1e9,
        marker="o",
        ls="None",
        ms=.5,
        color="g",
        # marker = "o",
        alpha=0.7,
        label="Ne19",
    )

    plt.plot(
        events_he.EventTimeLength*1000,
        events_he.EventSlope/1e9,
        marker="o",
        ls="None",
        ms=.5,
        color="r",
        alpha=0.3,
        # marker = "s",
        label="He6",
    )

    # Make the time_max. 
    n_pts = 1000
    slope = np.linspace(0,220e9, n_pts)
    BW = 1.1e9
    time_max = BW/ slope

    a = plt.plot(time_max*1000, slope/1e9, marker="None", markersize=4, color="black", linestyle="dotted")
    
    # Add bigger legend markers
    # Create dummy Line2D objects for legend
    h1 = plt.Line2D([0], [0], marker="o", markersize=4, color="g", linestyle="None")
    h2 = plt.Line2D([0], [0], marker="o", markersize=4, color="r", linestyle="None")
    h3 = plt.Line2D([0], [0], marker="None", markersize=4, color="black", linestyle="dotted")


    # Plot legend.
    ax.legend(
        [h1, h2, h3],
        ["Ne19", "He6", r"$t_{max}$"],
        loc="upper right",
    )
    ax.set_ylabel("Slope (GHz/s)")
    ax.set_xlabel("Event Length (ms)")

    ax.set_ylim(-8,220)
    ax.set_xlim(0,41)

    # Save and display the figure.
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    return None


# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/initial_observations")

# Load the snr study.
# snr_study = dl.load_snr_study()

fig_name = "INITIAL_OBS_fig_0.png"
fig_path = fig_dir / Path(fig_name)

make_viz_0(fig_path)


plt.show()
