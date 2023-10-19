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
    "axes.labelsize": 25,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}
plt.rcParams.update(params)



def make_viz_0(fig_path, config, exp_results):


    # Impose SNR Cut and stuff:
    condition = (
        # (
        #     print.events.EventMeanSNR
        #     > exp_results.events.EventMeanSNR.max() * 0.35
        # ) &
        (exp_results.events.EventStartFreq > 100e6)
        & (exp_results.events.EventNBins > 10)
    )
    exp_results.events = exp_results.events[condition]

    # Specific run_id and file_id.
    run_id = 393
    file_id = 0

    viz_settings = {
            "figsize": (8, 8),
            "colors": ["b", "r", "g", "c", "m", "k"],
        }
    fig, ax = exp_results.visualize(run_id, file_id, config, viz_settings = viz_settings)

    ax.set_xlim(0.55, 0.57)
    ax.set_ylim(80, 1200)
    ax.set_title("")


    # Save and display the figure.
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    return None


# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/measurements/data_analysis")

# sparses = [True, False, False]
trackss = [True, False]
eventss = [False, True]
sparse = False

event_alpha = 0.8
frac_pts = 1.0
ss_alpha = 0.5
mrk_sz = 0.3

event_ids = np.arange(0, 9).tolist()
print(event_ids)
event_ids = [11,12,13]

# Load the snr study.
snr_study = dl.load_snr_study()

for i, snr in enumerate(np.arange(8,12)):
    if i == 3: event_ids == [11]

    for i, (tracks, events) in enumerate(zip(trackss, eventss)):

        fig_name = Path(f"ANALYSIS_sideband_fig_{snr}_{i}.png")
        fig_path = fig_dir / fig_name


        config = {
            "tracks": {"show": tracks, "alpha": event_alpha, "EventIDs": []},
            "events": {
                "show": events,
                "alpha": event_alpha,
                "cuts": {},
                "EventIDs": event_ids,
            },
            "sparse_spec": {
                "show": sparse,
                "frac_pts": frac_pts,
                "alpha": ss_alpha,
                "mrk_sz": mrk_sz,
            },
        }

        exp_results = snr_study["he"][snr]

        make_viz_0(fig_path, config, exp_results)

    # plt.show()
