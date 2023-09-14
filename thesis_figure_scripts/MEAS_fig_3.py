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
    "legend.fontsize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}
plt.rcParams.update(params)


def make_energy_domain_plot(snr_study, spec_cuts, freq_chunk, fig_path):
    
    f, (ax0, ax1) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(12, 7)
    )

    set_fields = np.arange(1.25, 3.5, 0.25)
    mixer_freq = 17.9e9
    b = 0

    b_fits = []
    b_errs = []

    # Feed the info dict to the BetaSpectrum class.
    isotopes = {"Ne19": {"b": -0.7204 * b}, "He6": {"b": b}}
    bspec = bs.BetaSpectrum("Ne19", b=0)

    Ws = np.linspace(1.001, bspec.W0 - 0.001, 300)
    pdf = bspec.dNdE(Ws)

    freq_BW_full = np.array([18.1e9, 19.1e9])
    freq_BW_tot = freq_BW_full[1] - freq_BW_full[0]

    n_chunks = int(np.ceil((freq_BW_tot / freq_chunk)))

    b_fits = []
    b_errs = []
    red_chi = []
    for i, chunk in enumerate(range(n_chunks)):

        set_fields = np.arange(1.25, 3.5, 0.25)
        freq_BW = np.array([18.0e9, 19.1e9])
        freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

        normed_cols = ["mMeanSNR", "EventTimeLength", "uniform_rand", "detectability"]
        snrs = np.array([8, 9, 10, 11])

        freq_BW = np.clip(
            np.array(
                [
                    freq_BW_full[0] + freq_chunk * chunk,
                    freq_BW_full[0] + freq_chunk * (chunk + 1),
                ]
            ),
            0,
            freq_BW_full.max(),
        )

        chunk_cuts = {
            "EventStartFreq": (freq_BW[0] - mixer_freq, freq_BW[1] - mixer_freq)
        }
        tot_cuts = {**spec_cuts, **chunk_cuts}

        ratio_exp_combined = snr_tests.combine_ratios(
            snr_study, tot_cuts, snrs, normed_cols=normed_cols
        )

        ratio_exp = ratio_exp_combined

        # Make the predicted ratio.
        freq_BW = np.array(
            [
                mixer_freq + tot_cuts["EventStartFreq"][0],
                mixer_freq + tot_cuts["EventStartFreq"][1],
            ]
        )
        print(freq_BW)
        freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

        ratio_exp = ratio_exp[ratio_exp.index.isin(set_fields)]

        # Conduct fit.
        my_pars = Parameters()
        my_pars.add("C", value=1, min=-100, max=100, vary=True)
        my_pars.add("b", value=0, min=-10, max=10, vary=False)

        result = minimize(
            we.objfunc_chisq,
            my_pars,
            args=(freq_BWs, set_fields, ratio_exp),
            method="leastsq",
            epsfcn=1e-1,
        )

        # Print fit results
        print(f"reduced chisq: {result.redchi}")
        print(fit_report(result.params))

        red_chi.append(result.redchi)

        C = result.params["C"].value
        b = result.params["b"].value

        # Get the SM prediction.
        ratio_pred = we.AUC_expectation_we(
            set_fields, freq_BWs, b=b, plot=False, wall_effect=False
        )

        ratio_corr = ratio_exp.copy()
        ratio_corr["Ne19_corr"] = C * ratio_pred["He6"] * ratio_exp["Ratio"]
        ratio_corr["sNe19_corr"] = C * ratio_pred["He6"] * ratio_exp["sRatio"]

        (
            gamma_acceptances,
            gamma_widths,
            gamma_heights,
            gamma_height_errs,
            SM_heights,
        ) = ed.freq_to_energy_domain(set_fields, freq_BWs, ratio_corr, ratio_pred)

        label_bool = i == 0
        ed.energy_domain_plot(
            ax0,
            ax1,
            gamma_acceptances,
            gamma_widths,
            gamma_heights,
            gamma_height_errs,
            SM_heights,
            ratio_corr,
            ratio_pred,
            label=label_bool,
        )

        # Now add to the list of b_normed

        b_fits.append(result.params["b"].value)
        b_errs.append(result.params["b"].stderr)

    print("reduced chisq's: ", red_chi)
    ax0.plot(
        Ws,
        pdf,
        label="pdf",
        color="#1f77b4",
        alpha=1,
    )

    ax0.legend()
    ax0.set_xlim(1.5, 5.5)
    ax1.set_xlim(1.5, 5.5)

    # Save and display the figure.
    f.savefig(fig_path, bbox_inches="tight", dpi=300)

    return None


# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/measurements")
fig_base_name = "MEAS_fig_3_"
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

snr_cuts = [0.5]
start_freqs = [200]
freq_chunks = [500e6, 1000e6]


# Make all possible pairs of the above cuts:
specific_cuts = np.array(np.meshgrid(snr_cuts, start_freqs, freq_chunks)).T.reshape(
    -1, 3
)
print(specific_cuts)

for snr_cut, start_freq, freq_chunk in specific_cuts:

    freq_cut = {
        "EventStartFreq": ((start_freq) * 1e6, (1200) * 1e6),
        "mMeanSNR_n": (snr_cut, np.inf),
    }

    spec_cuts = {**event_cuts, **freq_cut}

    fig_path = fig_dir / Path(
        fig_base_name
        + f"ed_freq_{snr_cut}_snr_{start_freq}_chunk_{freq_chunk/1e6}"
        + fig_suffix
    )

    make_energy_domain_plot(snr_study, spec_cuts, freq_chunk, fig_path)


plt.show()
