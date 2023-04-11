# Author: Drew Byron
# Date: 04/07/2023

# Imports.
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
fig_name_counts = Path("MC_fig_1aa_counts.png")
fig_name_mon = Path("MC_fig_2aa_mon_.png")
fig_path_counts = fig_dir / fig_name_counts
fig_path_mon = fig_dir / fig_name_mon

# CSV paths.
N_bs_csv_path = fig_dir / Path("N_bs_a.csv")
N_bs_uncert_csv_path = fig_dir / Path("N_bs_uncert_a.csv")
mon_bs_csv_path = fig_dir / Path("mon_bs_a.csv")
mon_bs_uncert_csv_path = fig_dir / Path("mon_bs_uncert_a.csv")


# Whether or not to rerun the experiment.
rerun = True

# Simulation parameters.
exp_max = 12
trial_max = 100


# --- Run counts experiment. ---

# Select set fields.
set_fields = np.arange(0.75, 3.5, 0.25)

# Freq BW.
freq_BW = np.array([18.0e9, 19.1e9])

# COMMENT OUT IF ALREADY RUN. (between xxx's)
if rerun:
    # Run N const, MC experiments.
    N_bs, N_bs_uncert = cpc.run_N_const_prop_test(
        set_fields, freq_BW, exp_max=exp_max, trial_max=trial_max
    )

    N_bs.to_csv(N_bs_csv_path)
    N_bs_uncert.to_csv(N_bs_uncert_csv_path)


N_bs = pd.read_csv(N_bs_csv_path, index_col=0)
N_bs_uncert = pd.read_csv(N_bs_uncert_csv_path, index_col=0)

fig0, ax0 = plt.subplots(figsize=(10, 6))
N_bs_uncert.mean().plot(ax=ax0, label="estimated")
N_bs.std().plot(ax=ax0, label="true")
ax0.legend()
# plt.show()


cpc.plot_N_const_prop_test(N_bs_uncert, fig_path_counts)

# COMMENT OUT IF ALREADY RUN. (between xxx's)
if rerun:
    # --- Run monitor counts experiment. ---
    mon_bs, mon_bs_uncert = cpm.run_mon_const_prop_test(
        set_fields, freq_BW, exp_max=exp_max, trial_max=trial_max
    )

    mon_bs.to_csv(mon_bs_csv_path)
    mon_bs_uncert.to_csv(mon_bs_uncert_csv_path)


mon_bs = pd.read_csv(mon_bs_csv_path, index_col=0)
mon_bs_uncert = pd.read_csv(mon_bs_uncert_csv_path, index_col=0)

fig0, ax0 = plt.subplots(figsize=(10, 6))
mon_bs_uncert.mean().plot(ax=ax0, label="estimated")
mon_bs.std().plot(ax=ax0, label="true")
ax0.legend()
plt.show()
cpm.plot_mon_const_prop_test(mon_bs_uncert, fig_path_mon)

# xxx End of file. xxx
