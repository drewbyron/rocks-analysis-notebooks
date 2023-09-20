# Author: Drew Byron
# Date: 04/07/2023
"""
Description: This module makes a ratio plot(s) that aggregates the results
and errors of the SNR study. Here we attempt to cut all events born
below the visible bandwidth and then we use the naive Monte Carlo that 
just looks at the ratio of the spectral densities (Ne/He). 

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
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}
plt.rcParams.update(params)


def make_snr_test_ratio_plot(snr_study, spec_cuts, fig_path):
    """
    This function makes a ratio plot for each ratio obtained in the SNR
    study (8,9,10,11).
    """
    set_fields = np.arange(1.25, 3.5, 0.25)
    freq_BW = np.array([18.0e9, 19.1e9])
    freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

    normed_cols = ["mMeanSNR", "EventTimeLength", "uniform_rand", "detectability"]
    snrs = np.array([8, 9, 10, 11])
    snr_study_results = snr_tests.process_snr_study(
        snr_study, spec_cuts, snrs, normed_cols=normed_cols
    )

    # Make the experimental ratio and fit to predicted.
    f, (ax0, ax1) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [2.5, 1]}, figsize=(12, 7)
    )
    
    ne_counts = []
    he_counts = []

    for snr in snrs:

        ratio_exp = snr_study_results["ratio"][snr]

        # Make the predicted ratio.
        freq_BW = np.array(
            [
                17.9e9 + spec_cuts["EventStartFreq"][0],
                17.9e9 + spec_cuts["EventStartFreq"][1],
            ]
        )
        print(
            "Spectrum cut: {} MHz".format(np.array(spec_cuts["EventStartFreq"]) / 1e6)
        )
        print(f"BW used for predicted ratio:{freq_BW}")

        freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

        ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b=0, plot=False)

        # Only take the fields you want:
        ratio_pred = ratio_pred[ratio_pred.index.isin(set_fields)]
        ratio_exp = ratio_exp[ratio_exp.index.isin(set_fields)]

        my_pars = Parameters()
        my_pars.add("C", value=1, min=0, max=10, vary=True)
        my_pars.add("b", value=0, min=0, max=10, vary=False)

        # Fit to just C, leave b fixed.
        result = minimize(
            mc.objfunc_chisq, my_pars, args=(freq_BWs, set_fields, ratio_exp)
        )
        print(f"reduced chisq: {result.redchi}\n")
        C = result.params["C"].value
        ratio_exp["Ratio"] = C * ratio_exp["Ratio"]
        ratio_exp["sRatio"] = C * ratio_exp["sRatio"]

        pm.plot_experimental_ratio(ratio_exp, ax0, label=f"SNR cut: {snr}")

        # Plot predicted ratio
        if snr == 9:
            pm.plot_predicted_ratio(ratio_pred, ax0)

        ne_counts.append(
            snr_study_results["ne_spectrum"][snr].event_count.sum().round(-2) / 1000
        )
        he_counts.append(
            snr_study_results["he_spectrum"][snr].event_count.sum().round(-2) / 1000
        )
        count_ratio = np.array(ne_counts) / np.array(he_counts)
        ax1.plot(
            ratio_pred.index,
            (ratio_exp.Ratio - ratio_pred.Ratio) / ratio_exp.sRatio,
            label=f"residuals",
            marker="o",
            ls="None",
            ms=6,
        )

    print(f"\nne_counts = {ne_counts}\n")
    print(f"he_counts = {he_counts}\n")

    ax0.set_ylabel("ratio")
    ax0.set_xlabel("Set Field (T)")
    ax0.set_title(
        f"C = {C:.2f}., Ne_counts:{ne_counts} k, He_counts:{he_counts} k, count ratio: {count_ratio}"
    )

    ax0.legend()

    f.savefig(fig_path, bbox_inches="tight", dpi=300)

    return None


def make_snr_combined_ratio_plot(snr_study, spec_cuts, fig_path):

    set_fields = np.arange(1.25, 3.5, 0.25)
    freq_BW = np.array([18.0e9, 19.1e9])
    freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

    normed_cols = ["mMeanSNR", "EventTimeLength", "uniform_rand", "detectability"]
    snrs = np.array([8, 9, 10, 11])
    ratio_exp_combined = snr_tests.combine_ratios(
        snr_study, spec_cuts, snrs, normed_cols=normed_cols
    )

    # Make the experimental ratio and fit to predicted.
    f, (ax0, ax1) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [2.5, 1]}, figsize=(12, 7)
    )

    # Make the predicted ratio.
    freq_BW = np.array(
        [
            17.9e9 + spec_cuts["EventStartFreq"][0],
            17.9e9 + spec_cuts["EventStartFreq"][1],
        ]
    )
    print("Spectrum cut: {} MHz".format(np.array(spec_cuts["EventStartFreq"]) / 1e6))
    print(f"BW used for predicted ratio:{freq_BW}")

    freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

    ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b=0, plot=False)

    # Only take the fields you want:
    ratio_pred = ratio_pred[ratio_pred.index.isin(set_fields)]
    ratio_exp = ratio_exp_combined[ratio_exp_combined.index.isin(set_fields)]

    my_pars = Parameters()
    my_pars.add("C", value=1, min=0, max=10, vary=True)
    my_pars.add("b", value=0, min=-1, max=1, vary=False)

    # Fit to just C, leave b fixed.
    result = minimize(mc.objfunc_chisq, my_pars, args=(freq_BWs, set_fields, ratio_exp))

    # Print fit results
    print(f"reduced chisq: {result.redchi}")
    print(fit_report(result.params))

    C = result.params["C"].value
    ratio_exp["Ratio"] = C * ratio_exp["Ratio"]
    ratio_exp["sRatio"] = C * ratio_exp["sRatio"]

    # pm.plot_experimental_ratio(ratio_exp, ax0, label=f"ratio exp (combined)")

    ax0.errorbar(
        ratio_exp.index,
        ratio_exp.Ratio,
        yerr=ratio_exp["sRatio"],
        label="Data",
        marker="o",
        ls="None",
        ms=5,
        alpha=1,
        color="black",
    )

    # Plot predicted ratio
    ax0.plot(
        ratio_pred.index,
        ratio_pred.Ratio,
        label=r"Predicted" "\n" r"($b=0$)",
        color="#1f77b4",
        alpha=1,
    )

    # Plot residuals
    ax1.plot(
        ratio_pred.index,
        (ratio_exp.Ratio - ratio_pred.Ratio) / ratio_exp.sRatio,
        marker="o",
        ls="None",
        ms=6,
        color="black",
    )

    ax1.axhline(y=0, color="#1f77b4", linestyle="-")

    ax0.legend()

    ax0.set_ylabel("$N(^{19}$Ne$)/N(^{6}$He$)$")
    ax1.set_xlabel("Field (T)")
    ax1.set_ylabel(r"Residuals ($\sigma$)")

    f.savefig(fig_path, bbox_inches="tight", dpi=300)

    return None


# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/measurements")
fig_base_name = "MEAS_fig_2_"
fig_suffix = ".png"

event_cuts = {
    "uniform_rand": (0, 1),
    "EventTimeLength_n": (-np.inf, np.inf),
    "detectability_n": (0, 1),
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
snr_cuts = [ .5]
start_freqs = [200]

# Make all possible pairs of the above cuts:
specific_cuts = np.array(np.meshgrid(snr_cuts, start_freqs)).T.reshape(-1, 2)
print(specific_cuts)

for snr_cut, start_freq in specific_cuts:

    freq_cut = {
        "EventStartFreq": ((start_freq) * 1e6, (1200) * 1e6),
        "mMeanSNR_n": (snr_cut, np.inf),
    }

    spec_cuts = {**event_cuts, **freq_cut}
    fig_path_full = fig_dir / Path(
        fig_base_name + f"full_freq_{snr_cut}_snr_{start_freq}" + fig_suffix
    )
    fig_path_combined = fig_dir / Path(
        fig_base_name + f"combined_freq_{snr_cut}_snr_{start_freq}" + fig_suffix
    )

    # Run the above functions to make the two desired plots.
    make_snr_test_ratio_plot(snr_study, spec_cuts, fig_path_full)
    make_snr_combined_ratio_plot(snr_study, spec_cuts, fig_path_combined)


plt.show()
