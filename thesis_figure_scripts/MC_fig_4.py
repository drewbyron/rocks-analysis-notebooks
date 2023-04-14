# Author: Drew Byron
# Date: 04/07/2023

# Imports.
import sys
import numpy as np
import pandas as pd
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
    "legend.fontsize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
plt.rcParams.update(params)

# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/monte_carlo")
fig_name = Path("MC_fig_4_mon_drift.png")
fig_path = fig_dir / fig_name

# CSV paths.
mon_drift_path = fig_dir / Path("mon_drift.csv")

# Comment out appropriately if you've already run the simulation.

# Simulate the monitor drift.
trials = 1
n_drifts = 20
# Use a logspace here.
mon_drifts = np.logspace(-4, -1, n_drifts)
print(mon_drifts)

rerun = False
if rerun:
    mon_drift_df = montest.run_mon_sensitivity_test(mon_drifts, trial_max=trials)

    # Write the results to a csv so you don't have to rerun every time.
    mon_drift_df.to_csv(mon_drift_path)


# Read csv.
mon_drift_df = pd.read_csv(mon_drift_path, index_col=0)


# --- Making fig 4 for thesis MC section.
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    mon_drift_df.columns.to_series().astype(float).round(5),
    mon_drift_df.mean(),
    "o",
    ms=5,
    color="tab:orange",
    label=r"Linear monitor drift ($\Delta_{mon}$)",
)


ax.set_xscale("log")
ax.legend()
ax.set_xlabel(r"Linear monitor drift ($\Delta_{mon}$)")
ax.set_ylabel(r"Fit value for $b_{Fierz}$")
# Save and display the figure.
plt.savefig(fig_path, bbox_inches="tight", dpi=300)
plt.show()


# make a plot for assessing the size of the effect: 
# Make sure errors are right here.
x = mon_drift_df.columns.to_series().astype(float).round(5)
y = mon_drift_df.mean() / mon_drift_df.columns.to_series().astype(float)
yerr = y * mon_drift_df.std()
plt.errorbar(
    x,
    y,
    yerr=yerr,
    fmt="o",
    capsize=5,
    label="linear offset he",
)
plt.title(f"{y[3:].mean()} +- {y[3:].std()}")
plt.legend()
plt.show()