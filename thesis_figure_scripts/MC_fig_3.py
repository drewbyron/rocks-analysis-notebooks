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
    "legend.fontsize": 12,
    "axes.labelsize": 12,
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
field_offset_csv_path = fig_dir / Path("field_offset.csv")

# Comment out appropriately if you've already run the simulation. 
# --- Run B-field sensitivity experiment. ---

N_fields = 2
trials = 2
field_errs = np.linspace(10**-5, 2 * 10**-2, N_fields)
field_offsets = np.linspace(10**-5, 2 * 10**-2, N_fields)

field_err_test, field_offset_test = btest.run_B_sensitivity_test(
    field_errs, field_offsets, trial_max=trials
)

# Write the results to a csv so you don't have to rerun every time.
field_err_test.to_csv(field_err_csv_path)
field_offset_test.to_csv(field_offset_csv_path)


field_err_test = pd.read_csv(field_err_csv_path, index_col=0)
field_offset_test = pd.read_csv(field_offset_csv_path, index_col=0)

# Need to work on how I'm going to present this data.
# Don't really pin this down until you are actually writing this section. 
print(field_err_test)
print(field_offset_test)

field_err_test.hist(bins=10)
plt.show()

plt.plot(field_err_test.mean())
plt.plot(field_offset_test.mean())


# Save and display the figure.
plt.savefig(fig_path_bfield, bbox_inches="tight", dpi=300)
plt.show()
