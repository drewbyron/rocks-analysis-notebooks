# Author: Drew Byron
# Date: 2/2/23

import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from lmfit import minimize, Parameters, fit_report
from pathlib import Path
from scipy import integrate

# Path to local imports. Alter to match your machine.
sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")

# Local imports.
import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc
import he6_cres_spec_sims.spec_tools.beta_source.beta_spectrum as bs

# Local imports for plotting ratios and such.
import analysis_functions.ratio_experiment as re
import analysis_functions.ratio_prediction as rp
import analysis_functions.plotting_methods as pm


def freq_to_energy_domain(set_fields, freq_BWs, ratio_corr, ratio_pred):

    integrate_vect = np.vectorize(integrate.quad)

    energy_acceptances_high = sc.freq_to_energy(freq_BWs[:, 0], set_fields)
    energy_acceptances_low = sc.freq_to_energy(freq_BWs[:, 1], set_fields)
    energy_acceptances = np.stack(
        (energy_acceptances_low, energy_acceptances_high), axis=-1
    )
    gamma_acceptances = sc.gamma(energy_acceptances)

    gamma_widths = gamma_acceptances[:, 1] - gamma_acceptances[:, 0]

    gamma_heights = ratio_corr["Ne19_corr"] / gamma_widths
    gamma_height_errs = ratio_corr["sNe19_corr"] / gamma_widths

    # SM heights:
    SM_heights = ratio_pred["Ne19"] / gamma_widths

    return gamma_acceptances, gamma_widths, gamma_heights, gamma_height_errs, SM_heights


def energy_domain_plot(
    ax0,
    ax1,
    gamma_acceptances,
    gamma_widths,
    gamma_heights,
    gamma_height_errs,
    SM_heights,
    ratio_corr,
    ratio_pred,
    label=True,
    label_contents="data (corrected)",
):
    if label:
        ax0.bar(
            gamma_acceptances[:, 0],
            gamma_heights.values,
            width=gamma_widths,
            yerr=gamma_height_errs.values,
            color="r",
            align="edge",
            label=label_contents,
        )

        # ax0.bar(gamma_acceptances[:,0], SM_heights.values, width=gamma_widths, align = 'edge', edgecolor='black', color='none', label = "SM",linewidth = .1)

        ax1.plot(
            (gamma_acceptances[:, 0] + gamma_widths / 2),
            (ratio_corr["Ne19_corr"] - ratio_pred["Ne19"]) / ratio_corr["sNe19_corr"],
            label=f"residuals",
            marker="o",
            ls="None",
            color="r",
            ms=4,
        )
    else:
        ax0.bar(
            gamma_acceptances[:, 0],
            gamma_heights.values,
            width=gamma_widths,
            yerr=gamma_height_errs.values,
            color="r",
            align="edge",
        )

        ax1.plot(
            (gamma_acceptances[:, 0] + gamma_widths / 2),
            (ratio_corr["Ne19_corr"] - ratio_pred["Ne19"]) / ratio_corr["sNe19_corr"],
            marker="o",
            ls="None",
            color="r",
            ms=4,
        )

    ax1.axhline(y=0, color="b", linestyle="-")

    # Make plot labels and titles.
    # ax0.set_title("Ne19 Corrected Spectra")
    # ax0.set_xlabel(r"$\gamma$")
    ax1.set_xlabel(r"$\gamma$")
    ax0.set_ylabel(r"$\frac{dN}{d\gamma}$")
    ax1.set_ylabel(r"$\sigma$")

    return None
