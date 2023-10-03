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
    "axes.titlesize": 15,
    "legend.fontsize": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
}
plt.rcParams.update(params)


def make_viz_0(fig_path, config):

    local_dir = "/media/drew/T7 Shield/rocks_analysis/saved_experiments"
    experiment_name = "ne19_full_0"
    analysis_id = 2
    include_root_files = True
    max_root_files_to_grab = 5
    rebuild_experiment_dir = False

    exp_results_demo = ExperimentResults(
        local_dir=local_dir,
        experiment_name=experiment_name,
        analysis_id=analysis_id,
        include_root_files=include_root_files,
        max_root_files_to_grab=max_root_files_to_grab,
        rebuild_experiment_dir=rebuild_experiment_dir,
    )
    # Impose SNR Cut and stuff:
    condition = (
        (
            exp_results_demo.events.EventMeanSNR
            > exp_results_demo.events.EventMeanSNR.max() * 0.35
        )
        & (exp_results_demo.events.EventStartFreq > 100e6)
        & (exp_results_demo.events.EventNBins > 10)
    )
    exp_results_demo.events = exp_results_demo.events[condition]

    # Specific run_id and file_id.
    run_id = 532
    file_id = 2

    viz_settings = {
            "figsize": (12, 4),
            "colors": ["b", "r", "g", "c", "m", "k"],
        }
    fig, ax = exp_results_demo.visualize(run_id, file_id, config, viz_settings = viz_settings)

    ax.set_xlim(0, 0.86)
    ax.set_ylim(80, 1200)

    ax.set_title("")


    # Save and display the figure.
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    return None


# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/measurements/data_analysis")

sparses = [True, False, False]
trackss = [False, True, False]
eventss = [False, False, True]

event_alpha = 0.8
frac_pts = 1.0
ss_alpha = 0.5
mrk_sz = 0.3

event_ids = np.arange(0, 9).tolist()
print(event_ids)


for i, (tracks, events, sparse) in enumerate(zip(trackss, eventss, sparses)):

    fig_name = Path(f"ANALYSIS_fig_{i}.png")
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

    make_viz_0(fig_path, config)

# plt.show()
