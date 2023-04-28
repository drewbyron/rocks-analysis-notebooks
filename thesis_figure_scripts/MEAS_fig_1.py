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


def make_fb_combined_snr_test_plot(snr_study, fig_path):

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
    # snr_study_results = snr_tests.process_snr_study(
    #     snr_study, event_cuts, snrs, normed_cols=normed_cols
    # )

    ratio_exp_combined = snr_tests.combine_ratios(
        snr_study, event_cuts, snrs, normed_cols=normed_cols
    )
    # display(ratio_exp_combined)

    # ne_counts = []
    # he_counts = []
    # count_ratios = []
    # Set the fields to use and the BW to use.
    set_fields = np.arange(1.25, 3.5, 0.25)
    freq_BW = np.array([18.0e9, 19.0e9])
    freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

    # for snr in snrs:

    # Grab the experimental ratio for this cut, only take fields we are looking at.
    ratio_exp = ratio_exp_combined
    ratio_exp = ratio_exp.loc[ratio_exp.index.isin(set_fields), :]
    print(ratio_exp)
    #  Define the fit params.
    my_pars = Parameters()
    my_pars.add("C", value=0.32322636, min=0.31, max=0.34, vary=False, brute_step=0.01)
    my_pars.add(
        "slew_time",
        value=0.03948916,
        min=0.035,
        max=0.042,
        vary=False,
        brute_step=0.0005,
    )
    # my_pars.add('slew_time', value=0.035, min=.035, max = .042, vary = True, brute_step=0.0005)
    my_pars.add("b", value=0, min=-0.1, max=0.1, vary=False, brute_step=0.01)

    # Run the fit.
    # result = minimize(fb.objfunc_chisq_slewfree, my_pars, args = (ratio_exp, set_fields, freq_BW),method = 'brute')
    result = minimize(
        fb.objfunc_chisq_slewfree,
        my_pars,
        args=(ratio_exp, set_fields, freq_BW),
        method="leastsq",
        epsfcn=1e-3,
    )
    # Print fit results
    print(f"reduced chisq: {result.redchi}")
    print(fit_report(result.params))

    # Grab best fit parameters.
    C = result.params["C"].value
    b = result.params["b"].value
    slew_time = result.params["slew_time"].value

    # Build the predicted ratio according to these best fit values.
    ratio_pred_b0 = fb.build_from_below_ratio_pred(set_fields, freq_BW, slew_time, b)
    ratio_pred_b1p = fb.build_from_below_ratio_pred(
        set_fields, freq_BW, slew_time, b=+1
    )
    ratio_pred_b1n = fb.build_from_below_ratio_pred(
        set_fields, freq_BW, slew_time, b=-1
    )
    ratio_exp.loc[:, "Ratio"] = C * ratio_exp.loc[:, "Ratio"]
    ratio_exp.loc[:, "sRatio"] = C * ratio_exp.loc[:, "sRatio"]
    plt.rcParams.update({"font.size": 15})
    f, (ax0, ax1) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(12, 8)
    )

    # Plot the experimental ratio.
    pm.plot_experimental_ratio(ratio_exp, ax0, label=f"data")

    # Plot the predicted ratio (but only once).
    # if snr == 9:
    ax0.plot(
        ratio_pred_b0.index,
        ratio_pred_b0.Ratio,
        color="tab:orange",
        label="predicted (b=0)",
        marker="o",
        ms=6,
        linestyle="solid",
    )
    ax0.plot(
        ratio_pred_b1p.index,
        ratio_pred_b1p.Ratio,
        color="tab:green",
        label=r"predicted (b=$\pm$1)",
        marker="o",
        ms=6,
        linestyle="dashed",
    )
    ax0.plot(
        ratio_pred_b1n.index,
        ratio_pred_b1n.Ratio,
        color="tab:green",
        marker="o",
        ms=6,
        linestyle="dashed",
    )

    ax1.plot(
        ratio_pred_b0.index,
        (ratio_exp.Ratio - ratio_pred_b0.Ratio) / ratio_exp.sRatio,
        label=f"residuals",
        marker="o",
        ls="None",
        color="tab:blue",
        ms=6,
    )

    ax0.set_ylabel("ratio")
    ax1.set_xlabel("Set Field (T)")
    ax0.set_title(f"Ratio Measurement (including from below)")
    ax0.legend()
    ax1.legend()
    ax0.set_ylim(0, 3.5)
    ax1.set_ylim(-2.5, 2.5)

    # Save and display the figure.
    f.savefig(fig_path, bbox_inches="tight", dpi=300)

    return None


# plt.show()


# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/measurements")
fig_name = Path("MC_fig_1_fb_combined_snr_test.png")
fig_path = fig_dir / fig_name

snr_study = dl.load_snr_study()
make_fb_combined_snr_test_plot(snr_study, fig_path)

plt.show()
