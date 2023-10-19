# Author: Drew Byron
# Date: 04/07/2023
"""
Description: This module makes a simulated ratio plot in the same style 
as the prl ratio plot. 
"""
# Imports.
import sys
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
from pathlib import Path

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
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}
plt.rcParams.update(params)


# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/monte_carlo")
fig_name = Path("MC_fig_0_simple_illustration.png")
fig_path = fig_dir / fig_name
print(fig_path)


# Plotting functions.
def plot_sim_exp_ratio(ratio_exp, ax):

    label = f"Monte Carlo"
    ax.errorbar(
        ratio_exp.index,
        ratio_exp.Ratio,
        yerr=ratio_exp["sRatio"],
        label=label,
        marker="o",
        ms=6,
        color="tab:blue",
    )

    return None


def plot_predicted_ratio(ratio_pre, ax, label=None):

    if label is None:
        label = f"Prediction"
    ax.plot(
        ratio_pre.index,
        ratio_pre.Ratio,
        label=label,
        marker="o",
        ms=6,
        color="tab:orange",
    )

    return None


# Select set fields.
set_fields = np.arange(0.75, 3.5, 0.25)
# Freq BW.
freq_BW = np.array([19.0e9, 19.1e9])
# Tile freq_BW.
freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

# C, relationship between he and ne monitor.
C_exp = np.random.uniform(0.5, 1.5)

# Number of counts:
N = 10**4
# monitor rate tot:
mon = 10**8
# Set little b.
b = 0

# Simulate simple experiment.
ratio_exp, spectra_ne_exp, spectra_he_exp = mc.simple_MC(
    set_fields,
    freq_BWs,
    C_exp,
    b,
    counts_per_isotope=N,
    monitor_rate=mon,
    counts_pois=True,
    mon_pois=True,
)

ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b=b, plot=False)

ratio_pred_b1n = rp.AUC_expectation(set_fields, freq_BWs, b=-1, plot=False)
ratio_pred_b1p = rp.AUC_expectation(set_fields, freq_BWs, b=+1, plot=False)
print(ratio_exp, ratio_pred)

# Conduct fit.
my_pars = Parameters()
my_pars.add("C", value=1, min=0, max=10, vary=True)
my_pars.add("b", value=0.1, min=-10, max=10, vary=True)

result = minimize(mc.objfunc_chisq, my_pars, args=(freq_BWs, set_fields, ratio_exp))

# Fit report.
print(fit_report(result.params))

# Plot results.
f, (ax0, ax1) = plt.subplots(
    2, 1, gridspec_kw={"height_ratios": [2.5, 1]}, figsize=(12, 7)
)

C = result.params["C"].value

ratio_exp_cp = ratio_exp.copy()
ratio_exp_cp["Ratio"] = C * ratio_exp_cp["Ratio"]
ratio_exp_cp["sRatio"] = C * ratio_exp_cp["sRatio"]

# plot_sim_exp_ratio(ratio_exp_cp, ax0)
# plot_predicted_ratio(ratio_pred, ax0)

# ax0.set_yscale("log")
# ax0.set_ylabel("ratio")
# ax1.set_ylabel(r"$\sigma$")
# ax0.set_title(f"Simulated Experiment. Counts per isotope: 10^4")
ax0.legend()

ax0.set_ylabel("ratio")
ax1.set_xlabel("Set Field (T)")
ax1.set_ylim(-2,2)
ax0.set_ylim(-.5, 5)

# Plot the experimental ratio.
ax0.errorbar(
    ratio_exp_cp.index,
    ratio_exp_cp.Ratio,
    yerr=ratio_exp_cp["sRatio"],
    label = "Data (simulated)", 
    marker="o",
    ls="None",
    ms=5,
    alpha=1,
    color="black",
)

ax0.plot(
    ratio_pred.index,
    ratio_pred.Ratio,
    label=r"Predicted" "\n" r"($b=0$)",
    color="#1f77b4",
    alpha=1,
)

ax0.plot(
    ratio_pred_b1p.index,
    ratio_pred_b1p.Ratio,
    label=r"Predicted" "\n" r"($b=\pm1$)",
    color="green",
    linestyle="--",
    alpha=0.8,
)

ax0.plot(
    ratio_pred_b1n.index,
    ratio_pred_b1n.Ratio,
    color="green",
    linestyle="--",
    alpha=0.8,
)
ax1.plot(
    ratio_pred.index,
    (ratio_exp_cp.Ratio - ratio_pred.Ratio) / ratio_exp_cp.sRatio,
    marker="o",
    ls="None",
    ms=6,
    color="black",
)

ax1.axhline(y=0, color="#1f77b4", linestyle="-")

ax0.set_ylabel("$N(^{19}$Ne$)/N(^{6}$He$)$")
ax1.set_xlabel("Field (T)")
ax1.set_ylabel(r"Residuals ($\sigma$)")

ax0.legend()

# Save and display the figure.
plt.savefig(fig_path, bbox_inches="tight", dpi=300)
plt.show()
