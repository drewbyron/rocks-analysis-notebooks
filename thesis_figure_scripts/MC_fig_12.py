# Author: Drew Byron
# Date: 04/07/2023
"""
Description: FILL IN.  
"""

# Imports.
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import minimize, Parameters, fit_report
from pathlib import Path
from pprint import pprint
from scipy import integrate

# Path to local imports.
sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")
sys.path.append("/home/drew/He6CRES/rocks_analysis_notebooks/")

# Local imports.
import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc
import he6_cres_spec_sims.experiment as exp
import he6_cres_spec_sims.spec_tools.beta_source.beta_spectrum as bs

# Local imports for plotting ratios and such.
import analysis_functions.ratio_experiment as re
import analysis_functions.ratio_prediction as rp
import analysis_functions.plotting_methods as pm
import mc_functions.simple_mc as mc
import mc_functions.const_prop_vs_counts as cpc
import mc_functions.const_prop_vs_mon as cpm
import mc_functions.B_sensitivity as btest
import mc_functions.mon_drift_sensitivity as montest
import mc_functions.wall_effect as we
import mc_functions.energy_domain as ed
import mc_functions.from_below as fb

# Set plot parameters.
params = {
    "axes.titlesize": 15,
    "legend.fontsize": 14,
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}
plt.rcParams.update(params)

# Set fig path.
dir_path = Path("/media/drew/T7 Shield/thesis_figures/monte_carlo")


def freq_to_energy_domain(set_fields, freq_BWs, ratio, isotope):

    integrate_vect = np.vectorize(integrate.quad)

    energy_acceptances_high = sc.freq_to_energy(freq_BWs[:, 0], set_fields)
    energy_acceptances_low = sc.freq_to_energy(freq_BWs[:, 1], set_fields)
    energy_acceptances = np.stack(
        (energy_acceptances_low, energy_acceptances_high), axis=-1
    )
    gamma_acceptances = sc.gamma(energy_acceptances)

    gamma_widths = gamma_acceptances[:, 1] - gamma_acceptances[:, 0]

    gamma_heights = ratio[isotope] / gamma_widths
    gamma_height_errs = ratio[isotope] / gamma_widths

    return gamma_acceptances, gamma_widths, gamma_heights


def energy_domain_plot(
    ax0,
    gamma_acceptances,
    gamma_widths,
    gamma_heights,
    isotope,
    label="styff",
    color="r",
):

    return None


isotopes = ["He6", "Ne19"]

set_fields = np.arange(0.75, 3.5, 0.5)

# Freq_BW
freq_BW = np.array([18.0e9, 19.1e9])

# Build baseline ratio
freq_BWs_bl = np.tile(freq_BW, (len(set_fields), 1))

# Build extended BW ratio, from slope.

# Trap Slew timing.
slew_time = 35e-3

powers = sc.power_larmor(set_fields, freq_BW.mean())
energys = sc.freq_to_energy(freq_BW.mean(), set_fields)
slopes = sc.df_dt(energys, set_fields, powers)

freq_BW_extenstion = slopes * slew_time
freq_BWs_ext = freq_BWs_bl
freq_BWs_ext[:, 0] = freq_BWs_bl[:, 0] - freq_BW_extenstion

freq_BWs_bl = np.tile(freq_BW, (len(set_fields), 1))

# Get expected ratios
ratio_bl = rp.AUC_expectation(set_fields, freq_BWs_bl, b=0, plot=False)
ratio_ext = rp.AUC_expectation(set_fields, freq_BWs_ext, b=0, plot=False)


for isotope in isotopes:

    fig0, ax0 = plt.subplots(figsize=(12, 6))

    gamma_acceptances_bl, gamma_widths_bl, gamma_heights_bl = freq_to_energy_domain(
        set_fields, freq_BWs_bl, ratio_bl, isotope
    )

    ax0.bar(
        gamma_acceptances_bl[:, 0],
        gamma_heights_bl.values,
        width=gamma_widths_bl,
        align="edge",
        fill=True,
        color="b",
        alpha=0.2,
        edgecolor="b",
        label="18.0-19.1 GHz",
    )

    gamma_acceptances_ext, gamma_widths_ext, gamma_heights_ext = freq_to_energy_domain(
        set_fields, freq_BWs_ext, ratio_ext, isotope
    )

    ax0.bar(
        gamma_acceptances_ext[:, 0],
        gamma_heights_bl.values,
        width=gamma_widths_ext,
        align="edge",
        fill=False,
        edgecolor="r",
        label="Extended Bandwidth",
    )

    bspec = bs.BetaSpectrum(isotope, b=0)

    Ws = np.linspace(1.001, bspec.W0 - 0.001, 300)
    pdf = bspec.dNdE(Ws)
    ax0.plot(Ws, pdf, label=f"{isotope}", color = "b")

    # Make plot labels and titles.
    ax0.set_xlabel(r"$\gamma$")
    ax0.set_ylabel(r"$\frac{dN}{d\gamma}$")

    ax0.set_xlim(0.85, Ws.max() + 0.25)

    ax0.legend()

    # Define necessary paths:
    fig_path = dir_path / Path(f"MC_12_extented_bw_viz_{isotope}.png")
    fig0.savefig(fig_path, bbox_inches="tight", dpi=300)

plt.show()

# xxx End of file. xxx
