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

import thesis_figure_scripts.data_loaders as dl

# Set plot parameters.
params = {
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
plt.rcParams.update(params)


def make_fb_full_snr_test_plot(snr_study, fig_path):

    event_cuts = {
        "EventStartFreq": (100e6, 1200e6),
        "uniform_rand": (0, 1),
        "mMeanSNR_n": (-np.inf, np.inf),
        "EventTimeLength_n": (-np.inf, np.inf),
        "detectability_n": (-np.inf, np.inf),
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

    normed_cols = ["mMeanSNR", "EventTimeLength", "uniform_rand", "detectability"]

    snrs = np.array([8, 9, 10, 11])
    snr_study_results = snr_tests.process_snr_study(
        snr_study, event_cuts, snrs, normed_cols=normed_cols
    )

    plt.rcParams.update({"font.size": 15})
    f, (ax0, ax1) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(12, 8)
    )

    ne_counts = []
    he_counts = []
    count_ratios = []
    # Set the fields to use and the BW to use.
    set_fields = np.arange(1.25, 3.5, 0.25)
    freq_BW = np.array([18.0e9, 19.1e9])
    freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

    for snr in snrs:

        # Grab the experimental ratio for this cut, only take fields we are looking at.
        ratio_exp = snr_study_results["ratio"][snr]
        ratio_exp = ratio_exp.loc[ratio_exp.index.isin(set_fields), :]

        #  Define the fit params.
        my_pars = Parameters()
        my_pars.add("C", value=0.32322636, min=0.3, max=0.4, vary=True)
        my_pars.add("slew_time", value=0.0391, min=0.03, max=0.05, vary=True)
        my_pars.add("b", value=0.0, min=-1, max=1, vary=False)

        # Run the fit.
        result = minimize(
            fb.objfunc_chisq_slewfree,
            my_pars,
            args=(ratio_exp, set_fields, freq_BW),
            epsfcn=5e-3,
            max_nfev=100,
        )

        # Print fit results
        print(f"reduced chisq: {result.redchi}")
        print(fit_report(result.params))

        # Grab best fit parameters.
        C = result.params["C"].value
        b = result.params["b"].value
        slew_time = result.params["slew_time"].value

        # Build the predicted ratio according to these best fit values.
        ratio_pred = fb.build_from_below_ratio_pred(set_fields, freq_BW, slew_time, b)
        ratio_exp["Ratio"] = C * ratio_exp.copy()["Ratio"]
        ratio_exp["sRatio"] = C * ratio_exp.copy()["sRatio"]

        # Plot the experimental ratio.
        pm.plot_experimental_ratio(ratio_exp, ax0, label=f"SNR cut: {snr}")

        # Plot the predicted ratio (but only once).
        if snr == 9:
            ax0.plot(
                ratio_pred.index,
                ratio_pred.Ratio,
                color="g",
                label="ratio pred (from below)",
                marker="o",
                ms=6,
            )

        ne_counts.append(
            snr_study_results["ne_spectrum"][snr].event_count.sum().round(-2) / 1000
        )
        he_counts.append(
            snr_study_results["he_spectrum"][snr].event_count.sum().round(-2) / 1000
        )
        count_ratios.append(
            (
                snr_study_results["ne_spectrum"][snr].event_count.sum()
                / snr_study_results["he_spectrum"][snr].event_count.sum()
            ).round(3)
        )

        ax1.plot(
            ratio_pred.index,
            (ratio_exp.Ratio - ratio_pred.Ratio) / ratio_exp.sRatio,
            label=f"snr cut: {snr}",
            marker="o",
            ls="None",
            ms=6,
        )

    # ax0.set_yscale("log")
    ax0.set_ylabel("ratio")
    ax1.set_xlabel("Field (T)")
    ax0.set_title(
        f"C = {C:.2f}., Ne_counts:{ne_counts} k, He_counts:{he_counts} k, count ratio: {count_ratios}"
    )
    ax0.legend()
    ax1.legend()

    # Save and display the figure.
    f.savefig(fig_path, bbox_inches="tight", dpi=300)

    return None


plt.show()


# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/measurements")
fig_name = Path("MC_fig_0_fb_full_snr_test.png")
fig_path = fig_dir / fig_name

snr_study = dl.load_snr_study()
make_fb_full_snr_test_plot(snr_study, fig_path)

plt.show()
