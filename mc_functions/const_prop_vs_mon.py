import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from lmfit import minimize, Parameters, fit_report
from pathlib import Path

# Local imports for plotting ratios and such.
import analysis_functions.ratio_experiment as re
import analysis_functions.ratio_prediction as rp
import analysis_functions.plotting_methods as pm
import mc_functions.simple_mc as mc


def run_mon_const_prop_test(set_fields, freq_BW, exp_max=12, trial_max=20):
    """
    Note that this is looking at the total number of mon counts over the 
    entire isotope (split evening among the different set fields).
    """
    # Tile freq_BW.
    freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

    # C, relationship between he and ne monitor.
    # C_exp = np.random.uniform(0.25, 1.75)
    C_exp = 1

    # count rate tot:
    N = 10**15
    # Set little b.
    b = 0

    bs_uncert = {}
    bs = {}

    for exp in np.arange(4, exp_max, 1):

        
        # Number of mon counts per field
        mon = int((10**exp)/len(set_fields))

        bs_uncert[exp] = []
        bs[exp] = []

        for trial in range(trial_max):
            print(f"mon counts per field = 10**{exp}, trial = {trial}")

            # Simulate simple experiment. Don't poisson vary monitor.
            ratio_exp, spectra_ne_exp, spectra_he_exp = mc.simple_MC(
                set_fields,
                freq_BWs,
                C_exp,
                b,
                counts_per_isotope=N,
                monitor_rate=mon,
                counts_pois=False,
                mon_pois=True,
            )

            ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b=b, plot=False)

            # Conduct fit.
            my_pars = Parameters()
            my_pars.add("C", value=1, min=0, max=10, vary=True)
            my_pars.add("b", value=0.001, min=-10, max=10, vary=True)

            result = minimize(
                mc.objfunc_chisq, my_pars, args=(freq_BWs, set_fields, ratio_exp)
            )

            bs_uncert[exp].append(result.params["b"].stderr)
            bs[exp].append(result.params["b"].value)

    bs_uncert = pd.DataFrame(bs_uncert)
    bs = pd.DataFrame(bs)

    return bs, bs_uncert


def plot_mon_const_prop_test(bs_uncert, path):

    # ------ Set Thesis Plot Parameters -----------
    params = {
        "axes.titlesize": 15,
        "legend.fontsize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
    plt.rcParams.update(params)
    figsize = (10, 6)

    # Plot results.
    fig0, ax0 = plt.subplots(figsize=figsize)
    N = 10 ** bs_uncert.mean().index.astype(int).values
    Const = bs_uncert.mean().values * np.sqrt(N)
    Const_err = bs_uncert.std().values * np.sqrt(N)
    plt.errorbar(N, Const, yerr=Const_err)

    # recalc without breaking up into orders of mag. 
    Const = np.nanmean((bs_uncert * np.sqrt(N)).values)
    Const_err = np.nanstd((bs_uncert * np.sqrt(N)).values)
    print(f"X_mon = {Const:.3f}+- {Const_err:.3f}")

    ax0.set_xscale("log")
    ax0.set_ylabel("Proportionality constant (unitless)")
    ax0.set_xlabel("N per isotope (counts)")
    ax0.set_title(
        f"Sensitivity to monitor counts per field per isotope. Const = {Const:.3f} +-{Const:.3f}"
    )

    # Save the figure.
    if path is not None:
        plt.savefig(path, bbox_inches="tight", dpi=300)

    return None
