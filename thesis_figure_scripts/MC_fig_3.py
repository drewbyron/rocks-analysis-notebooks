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
fig_name_bfield = Path("MC_fig_3_bfield_sensitivity.png")
fig_path_bfield = fig_dir / fig_name_bfield

# CSV paths.
field_err_csv_path = fig_dir / Path("field_err.csv")
field_offset_ne_csv_path = fig_dir / Path("field_offset_ne.csv")
field_offset_he_csv_path = fig_dir / Path("field_offset_he.csv")

rerun = False
# --- Run B-field sensitivity experiment. ---
if rerun:
    N_errs = 10
    trials = 50
    field_errs = np.logspace(-5, -2, N_errs)
    field_offsets = np.logspace(-5, -2, N_errs)
    print(field_errs, field_offsets)
    (
        field_err_test,
        field_offset_test_ne,
        field_offset_test_he,
    ) = btest.run_B_sensitivity_test(field_errs, field_offsets, trial_max=trials)

    # Write the results to a csv so you don't have to rerun every time.
    field_err_test.to_csv(field_err_csv_path)
    field_offset_test_ne.to_csv(field_offset_ne_csv_path)
    field_offset_test_he.to_csv(field_offset_he_csv_path)

field_err_test = pd.read_csv(field_err_csv_path, index_col=0)
field_offset_test_ne = pd.read_csv(field_offset_ne_csv_path, index_col=0)
field_offset_test_he = pd.read_csv(field_offset_he_csv_path, index_col=0)

# Need to work on how I'm going to present this data.
# Don't really pin this down until you are actually writing this section.
# print(field_err_test)
# print(field_offset_test)

# field_err_test.hist(bins=10)
# plt.show()


# --- Making fig 3 for thesis MC section.
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(
    field_err_test.columns.to_series().astype(float).round(5),
    field_err_test.mean(),
    yerr=field_err_test.std(),
    fmt="o",
    color="tab:blue",
    capsize=5,
    label=r"Gaussian spread ($\sigma_B$)",
)
ax.plot(
    field_offset_test_ne.columns.to_series().astype(float).round(5),
    field_offset_test_ne.mean(),
    "o",
    ms=5,
    color="tab:orange",
    label=r"Linear offset for Neon ($\Delta B$)",
)
ax.plot(
    field_offset_test_he.columns.to_series().astype(float).round(5),
    field_offset_test_he.mean(),
    "o",
    ms=5,
    color="tab:green",
    label=r"Linear offset for Helium ($\Delta B$)",
)

ax.set_xscale("log")
ax.legend()
ax.set_xlabel(r"Field error ($\sigma_B$ or $\Delta B$)")
ax.set_ylabel(r"Fit value for $b_{Fierz}$")
plt.savefig(fig_path_bfield, bbox_inches="tight", dpi=300)
plt.show()

# Make sure errors are right here.
x = field_err_test.columns.to_series().astype(float).round(5)
y = field_err_test.std() / field_err_test.columns.to_series().astype(float)
yerr = y * field_err_test.std()
plt.errorbar(
    x,
    y,
    yerr=yerr,
    fmt="o",
    capsize=5,
    label="gauss",
)
plt.title(f"{y.mean()} +- {y.std()}")
plt.legend()
plt.show()


# Make sure errors are right here.
x = field_offset_test_ne.columns.to_series().astype(float).round(5)
y = field_offset_test_ne.mean() / field_offset_test_ne.columns.to_series().astype(float)
yerr = y * field_offset_test_ne.std()
plt.errorbar(
    x,
    y,
    yerr=yerr,
    fmt="o",
    capsize=5,
    label="linear offset ne",
)
plt.title(f"{y.mean()} +- {y.std()}")
plt.legend()
plt.show()

# Make sure errors are right here.
x = field_offset_test_he.columns.to_series().astype(float).round(5)
y = field_offset_test_he.mean() / field_offset_test_he.columns.to_series().astype(float)
yerr = y * field_offset_test_he.std()
plt.errorbar(
    x,
    y,
    yerr=yerr,
    fmt="o",
    capsize=5,
    label="linear offset he",
)
plt.title(f"{y.mean()} +- {y.std()}")
plt.legend()
plt.show()
# plt.errplot(field_err_test.mean())
# plt.plot(field_offset_test.mean())


# Save and display the figure.

# plt.show()
