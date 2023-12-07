# Author: Drew Byron
# Date: 04/07/2023
"""
Description: This module runs and plots a study of the wall effect and its
mitigation via finer binning of the BW. 
"""
# Imports.
import sys
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
from pathlib import Path
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
    "legend.fontsize": 15,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
}
plt.rcParams.update(params)

# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/final_exam")
fig_base_name = "EXAM_DEMO_"
fig_suffix = ".png"

# Utility function.
def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


# ------ Plot 1-2: gammas, r_cycl, efficiencies -----------
def wall_efficiency(gamma, field, trap_radius=0.578e-2):

    energy = sc.energy(gamma)
    r = sc.cyc_radius(sc.energy(gamma), field, pitch_angle=90)
    e = (trap_radius - r) ** 2 / trap_radius**2

    return e


def plots1_2_3(set_fields, freq_BW, b=0):

    freq_BWs = np.tile(freq_BW, (len(set_fields), 1))
    integrate_vect = np.vectorize(integrate.quad)

    energy_acceptances_high = sc.freq_to_energy(freq_BWs[:, 0], set_fields)
    energy_acceptances_low = sc.freq_to_energy(freq_BWs[:, 1], set_fields)
    energy_acceptances = np.stack(
        (energy_acceptances_low, energy_acceptances_high), axis=-1
    )

    # Empty dict to house relative rates.
    rates = {}

    isotopes = {"Ne19": {"b": -0.7204 * b}, "He6": {"b": b}}
    colors = {"Ne19": "tab:blue", "He6": "tab:orange"}
    field_cmap = get_cmap(len(set_fields))

    # This delta_W prevents a zero argument in a log. Doesn't effect pdfs
    delta_W = 10**-10

    pdf_plot_pts = 10**2
    f0, ax0 = plt.subplots(1, figsize=(12, 6))
    f1, ax1 = plt.subplots(1, figsize=(12, 6))
    # f2, ax2 = plt.subplots(1, figsize=(12, 6))

    # Plot first without any reference to the 
    # bs_ne = bs.BetaSpectrum("Ne19")
    # bs_he = bs.BetaSpectrum("He6")

    eps = .01
    W_ne = np.linspace(1+ eps, 5.337377690802349-eps, 100)
    W_he = np.linspace(1+ eps, 5.337377690802349-eps, 100)

    
    for b in [0,1]:

        
        bs_ne = bs.BetaSpectrum("Ne19", b = -0.7204 * b)
        bs_he = bs.BetaSpectrum("He6", b = b)

        pdf_ne =bs_ne.dNdE(W_ne)
        pdf_he =bs_he.dNdE(W_he)

        if b ==0: 
            alpha = .75
            ax1.plot(W_ne, pdf_ne, color =colors["Ne19"], label=r"Ne19, $b_{fierz}$ = "+f"{b}", )
            ax1.plot(W_he, pdf_he, color =colors["He6"], label=r"He6, $b_{fierz}$ = "+f"{b}", )
        if b ==1: 
            alpha =.5

            ax1.fill_between(
                            W_ne,
                            0,
                            pdf_ne,
                            alpha=alpha,
                            color=colors["Ne19"],
                            label=r"Ne19, $b_{fierz}$ = "+f"{b}",
                        )
            ax1.fill_between(
                            W_he,
                            0,
                            pdf_he,
                            alpha=alpha,
                            color=colors["He6"],
                            label=r"He6, $b_{fierz}$ = "+f"{b}",
                    )
    ax1.legend()
    ax1.set_xlabel(r"$\gamma$")
    ax1.set_ylabel(r"$\frac{dN}{d\gamma}$")
    # plt.show()
    for j, (isotope_name, isotope_info) in enumerate(isotopes.items()):

        W_low = sc.gamma(energy_acceptances[:, 0]) + delta_W
        W_high = sc.gamma(energy_acceptances[:, 1]) - delta_W

        # Feed the info dict to the BetaSpectrum class.
        bspec = bs.BetaSpectrum(isotope_name, b=isotope_info["b"])




        def dNdE_wall_effect(gamma, field):
            return bspec.dNdE(gamma) * wall_efficiency(gamma, field)

        fraction_of_spectrum, err = integrate_vect(
            dNdE_wall_effect, W_low, W_high, args=(set_fields), epsrel=1e-12
        )

        Ws = np.linspace(W_low, W_high, pdf_plot_pts)
        pdf = bspec.dNdE(Ws)


        for i, (W, p) in enumerate(zip(Ws.T, pdf.T)):
            if i == 0:
                ax0.fill_between(
                    W,
                    0,
                    p,
                    alpha=0.5,
                    color=colors[isotope_name],
                    label=f"{isotope_name}",
                )
            else:
                ax0.fill_between(W, 0, p, alpha=0.5, color=colors[isotope_name])
        ax0.legend()
        ax0.set_xlabel(r"$\gamma$")
        ax0.set_ylabel(r"$\frac{dN}{d\gamma}$")

        # if j == 0:
        #     rs = sc.cyc_radius(sc.energy(Ws), set_fields, pitch_angle=90)
        #     effs = wall_efficiency(Ws, set_fields)
        #     for i, (W, r, eff) in enumerate(zip(Ws.T, rs.T, effs.T)):
        #         label = f"{set_fields[i]:.2f} T"
        #         lw = 3
        #         ax1.plot(W, r * 1e3, label=label, color=field_cmap(i), linewidth=lw)
        #         ax2.plot(W, eff, label=label, color=field_cmap(i), linewidth=lw)

        #     ax1.legend()
        #     ax2.legend()

        #     ax1.set_xlabel(r"$\gamma$")
        #     ax2.set_xlabel(r"$\gamma$")

        #     ax1.set_ylabel(r"$r_{cycl}$ (mm)")
        #     ax2.set_ylabel(r"Efficiency ($\epsilon_{e}(E)_{we}$)")

    # Save the figures to disk.
    f0_path = fig_dir / Path(fig_base_name + "0" + fig_suffix)
    f1_path = fig_dir / Path(fig_base_name + "1" + fig_suffix)
    # f2_path = fig_dir / Path(fig_base_name + "2" + fig_suffix)
    f0.savefig(f0_path, bbox_inches="tight", dpi=300)
    f1.savefig(f1_path, bbox_inches="tight", dpi=300)
    # f2.savefig(f2_path, bbox_inches="tight", dpi=300)

    # plt.show()

    return None

######################################
######################################
# Run the above functions to build thesis plots.


# Select set fields.
set_fields = np.arange(0.75, 3.5, 0.25)

# Full Freq BW.
freq_BW = np.array([18.1e9, 19.1e9])

# Make basic first three plots illustrating the size of the effect.
plots1_2_3(set_fields, freq_BW)

# # Select set fields.
# set_fields = np.arange(0.75, 3.25, 0.01)

# # Freq BW.
# freq_BW = np.array([18.1e9, 19.1e9])
# wall_effect_correction(set_fields, freq_BW, freq_chunk=1000e6)

# # Freq BW.
# freq_BW = np.array([18.1e9, 19.1e9])
# wall_effect_correction(set_fields, freq_BW, freq_chunk=200e6)

# # Select set fields.
# set_fields = np.arange(0.75, 3.25, 0.25)
# # Full Freq BW.
# freq_BW = np.array([18.1e9, 19.1e9])
# corrected_spectrum(
#     set_fields,
#     freq_BW,
#     freq_chunk=1000e6,
#     corrections_off=False,
#     MC_label="simulated data (corrected)",
# )
# corrected_spectrum(
#     set_fields,
#     freq_BW,
#     freq_chunk=1000e6,
#     corrections_off=True,
#     MC_label="simulated data (uncorrected)",
# )

# # Full Freq BW.
# freq_BW = np.array([18.1e9, 19.1e9])
# corrected_spectrum(
#     set_fields,
#     freq_BW,
#     freq_chunk=500e6,
#     corrections_off=False,
#     MC_label="simulated data (corrected)",
# )

# corrected_spectrum(
#     set_fields,
#     freq_BW,
#     freq_chunk=200e6,
#     corrections_off=False,
#     MC_label="simulated data (corrected)",
# )

# corrected_spectrum(
#     set_fields,
#     freq_BW,
#     freq_chunk=100e6,
#     corrections_off=False,
#     MC_label="simulated data (corrected)",
# )
plt.show()
