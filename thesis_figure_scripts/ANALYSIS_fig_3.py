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

import uproot4

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

def construct_GV_array(rootfile_path, gainvar_num_spectra):
    # Open rootfile.
    rootfile = uproot4.open(rootfile_path)
    GV_array = rootfile["histGV_{}_0;1".format(gainvar_num_spectra)]._bases[1]._data
    return GV_array


def make_viz_0(fig_path):

    freq = np.linspace(0,1200,302)

    he_root_file = "/media/drew/T7 Shield/rocks_analysis/saved_experiments/he6_full_0_aid_1/root_files/Freq_data_2022-08-17-20-29-17_001.root"
    ne_root_file = "/media/drew/T7 Shield/rocks_analysis/saved_experiments/ne19_full_0_aid_2/root_files/Freq_data_2022-10-05-19-27-55_002.root"

    he_noise = construct_GV_array(he_root_file, gainvar_num_spectra = 50000)
    ne_noise = construct_GV_array(ne_root_file, gainvar_num_spectra = 50000)

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(freq, ne_noise, label = "Ne19", color="g",)
    plt.plot(freq, he_noise, label = "He6", color="c",)
    ax.set_ylabel("Mean Noise (arb.)")
    ax.set_xlabel("Frequency (MHz)")
    plt.legend()
    
    # Save and display the figure.
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    plt.show()
    return None


# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/measurements/data_analysis")

# Load the snr study.
# snr_study = dl.load_snr_study()

fig_name = "ANALYSIS_3_test_{}.png"
fig_path = fig_dir / fig_name

make_viz_0(fig_dir / Path(fig_name.format(0)))

plt.show()
