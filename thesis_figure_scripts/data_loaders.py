# Author: Drew Byron
# Date: 04/07/2023

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

# Set plot parameters.
params = {
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
plt.rcParams.update(params)


def load_snr_study():
    # Load data
    local_dir = "/media/drew/T7 Shield/rocks_analysis/saved_experiments"
    include_root_files = False
    max_root_files_to_grab = 0
    rebuild_experiment_dir = False

    snrs = [8, 9, 10, 11]
    aids = [25, 26, 27, 29]
    n_files = 100
    experiment_names = [f"ne_snr{snr}_fn{n_files}" for snr in snrs]
    print(experiment_names)

    ne_snr_results = [
        ExperimentResults(
            local_dir=local_dir,
            experiment_name=experiment_name,
            analysis_id=aid,
            include_root_files=include_root_files,
            max_root_files_to_grab=max_root_files_to_grab,
            rebuild_experiment_dir=rebuild_experiment_dir,
        )
        for (experiment_name, aid) in zip(experiment_names, aids)
    ]
    local_dir = "/media/drew/T7 Shield/rocks_analysis/saved_experiments"
    include_root_files = False
    max_root_files_to_grab = 0
    rebuild_experiment_dir = False

    snrs = [8, 9, 10, 11]
    aids = [21, 22, 23, 25]
    n_files = 100
    experiment_names = [f"he_snr{snr}_fn{n_files}" for snr in snrs]
    print(experiment_names)

    he_snr_results = [
        ExperimentResults(
            local_dir=local_dir,
            experiment_name=experiment_name,
            analysis_id=aid,
            include_root_files=include_root_files,
            max_root_files_to_grab=max_root_files_to_grab,
            rebuild_experiment_dir=rebuild_experiment_dir,
        )
        for (experiment_name, aid) in zip(experiment_names, aids)
    ]
    # Aggregate the results.
    snrs = np.array([8, 9, 10, 11])
    snr_study = {
        "ne": dict(zip(snrs, ne_snr_results)),
        "he": dict(zip(snrs, he_snr_results)),
    }

    drop_rid_list = [548, 549, 522, 495, 377]

    cluster = False

    initial_cuts = {
        "EventStartFreq": (0e6, 1200e6),
    }

    for isotope in snr_study.keys():
        for snr in snrs:

            # Drop bad rids.
            snr_study[isotope][snr].events = snr_study[isotope][snr].events[
                ~snr_study[isotope][snr].events.run_id.isin(drop_rid_list)
            ]
            snr_study[isotope][snr].tracks = snr_study[isotope][snr].tracks[
                ~snr_study[isotope][snr].tracks.run_id.isin(drop_rid_list)
            ]
            snr_study[isotope][snr].root_files = snr_study[isotope][snr].root_files[
                ~snr_study[isotope][snr].root_files.run_id.isin(drop_rid_list)
            ]

            # Make initial cuts. Need to do this before clustering or we cluster good events into bad.
            events = snr_study[isotope][snr].events
            print(f"{isotope}, {snr}.\npre cuts events: {len(events)}")

            # snr_study[isotope][cut].events = re.cut_df(events, both_cuts)
            events_cut = re.cut_df(events, initial_cuts)
            snr_study[isotope][snr].events = events_cut
            print(f"\npost cuts events: {len(events_cut)}")
    return snr_study

