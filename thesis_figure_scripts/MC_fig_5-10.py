# Author: Drew Byron
# Date: 04/07/2023

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
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
plt.rcParams.update(params)

# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/monte_carlo")
fig_base_name = "MC_wall_effect_"
fi_suffix = ".png"


# ------ Plot 1-2: gammas, r_cycl, efficiencies -----------


def wall_efficiency(gamma, field, trap_radius=0.578e-2):

    energy = sc.energy(gamma)
    r = sc.cyc_radius(sc.energy(gamma), field, pitch_angle=90)
    e = (trap_radius - r) ** 2 / trap_radius**2

    return e


def plots1_2_3(set_fields, freq_BWs, b=0):

    integrate_vect = np.vectorize(integrate.quad)

    energy_acceptances_high = sc.freq_to_energy(freq_BWs[:, 0], set_fields)
    energy_acceptances_low = sc.freq_to_energy(freq_BWs[:, 1], set_fields)
    energy_acceptances = np.stack(
        (energy_acceptances_low, energy_acceptances_high), axis=-1
    )

    # Empty dict to house relative rates.
    rates = {}

    isotopes = {"Ne19": {"b": -0.7204 * b}, "He6": {"b": b}}

    # This delta_W prevents a zero argument in a log. Doesn't effect pdfs
    delta_W = 10**-10

    pdf_plot_pts = 10**2
    f0, ax0 = plt.subplots(1, figsize=(10, 5))
    f1, ax1 = plt.subplots(1, figsize=(10, 5))
    f2, ax2 = plt.subplots(1, figsize=(10, 5))

    for isotope_name, isotope_info in isotopes.items():

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
        ax0.plot(Ws, pdf)

        r = sc.cyc_radius(sc.energy(Ws), set_fields, pitch_angle=90)
        ax1.plot(Ws, r, label=set_fields)
        ax1.legend()

        eff = wall_efficiency(Ws, set_fields)
        ax2.plot(Ws, eff)
    plt.show()

    return None


# Select set fields.
set_fields = np.arange(0.75, 3.5, 0.25)

# Full Freq BW.
freq_BW = np.array([18.0e9, 19.1e9])
freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

plots1_2_3(set_fields, freq_BWs)


# if plot:
#     ax0.set_xlabel(r"$\gamma$")
#     ax0.set_title("He6 and Ne19 Spectra (pdf)")
#     ax1.set_xlabel(r"$\gamma$")
#     ax1.set_title("radius (m)")
#     ax2.set_xlabel(r"$\gamma$")
#     ax2.set_title("wall effect (efficiency)")
#     plt.show()

# for isotope in rates:
#     rates[isotope] = np.array(rates[isotope])

# rates["set_fields"] = set_fields
# rates = pd.DataFrame(rates)

# # Make the ratio a column of the df
# rates["Ratio"] = rates["Ne19"] / rates["He6"]

# rates.set_index("set_fields", inplace=True)
# return rates


## OLD DELETE

# # Plotting functions.
# def plot_sim_exp_ratio(ratio_exp, ax):

#     label = f"Monte Carlo"
#     ax.errorbar(
#         ratio_exp.index,
#         ratio_exp.Ratio,
#         yerr=ratio_exp["sRatio"],
#         label=label,
#         marker="o",
#         ms=6,
#         color="tab:blue",
#     )

#     return None


# def plot_predicted_ratio(ratio_pre, ax, label=None):

#     if label is None:
#         label = f"Prediction"
#     ax.plot(
#         ratio_pre.index,
#         ratio_pre.Ratio,
#         label=label,
#         marker="o",
#         ms=6,
#         color="tab:orange",
#     )

#     return None


# # Select set fields.
# set_fields = np.arange(0.75, 3.5, 0.25)
# # Freq BW.
# freq_BW = np.array([19.0e9, 19.1e9])
# # Tile freq_BW.
# freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

# # C, relationship between he and ne monitor.
# C_exp = np.random.uniform(0.5, 1.5)

# # Number of counts:
# N = 10**4
# # monitor rate tot:
# mon = 10**8
# # Set little b.
# b = 0

# # Simulate simple experiment.
# ratio_exp, spectra_ne_exp, spectra_he_exp = mc.simple_MC(
#     set_fields,
#     freq_BWs,
#     C_exp,
#     b,
#     counts_per_isotope=N,
#     monitor_rate=mon,
#     counts_pois=True,
#     mon_pois=True,
# )

# ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b=b, plot=False)

# # Conduct fit.
# my_pars = Parameters()
# my_pars.add("C", value=1, min=0, max=10, vary=True)
# my_pars.add("b", value=0.1, min=-10, max=10, vary=True)

# result = minimize(mc.objfunc_chisq, my_pars, args=(freq_BWs, set_fields, ratio_exp))

# # Fit report.
# print(fit_report(result.params))

# # Plot results.
# f, (ax0, ax1) = plt.subplots(
#     2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(12, 7)
# )

# C = result.params["C"].value

# ratio_exp_cp = ratio_exp.copy()
# ratio_exp_cp["Ratio"] = C * ratio_exp_cp["Ratio"]
# ratio_exp_cp["sRatio"] = C * ratio_exp_cp["sRatio"]

# plot_sim_exp_ratio(ratio_exp_cp, ax0)
# plot_predicted_ratio(ratio_pred, ax0)

# # ax0.set_yscale("log")
# ax0.set_ylabel("ratio")
# ax1.set_ylabel(r"$\sigma$")
# ax0.set_title(f"Simulated Experiment. Counts per isotope: 10^4")
# ax0.legend()

# ax0.set_ylabel("ratio")
# ax1.set_xlabel("Set Field (T)")
# ax1.set_ylim(-2,2)


# ax1.plot(
#     ratio_pred.index,
#     (ratio_exp_cp.Ratio - ratio_pred.Ratio) / ratio_exp_cp.sRatio,
#     label=f"residuals",
#     marker="o",
#     ls="None",
#     ms=6,
#     color="tab:blue",
# )
# ax1.legend()

# # Save and display the figure.
# plt.savefig(fig_path, bbox_inches="tight", dpi=300)
# plt.show()
