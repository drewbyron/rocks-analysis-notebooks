# Author: Drew Byron
# Date: 04/07/2023

# Imports.
import numpy as np
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
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
plt.rcParams.update(params)

# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/monte_carlo")
fig_name = Path("MC_fig_0_simple_illustration.png")
fig_path = fig_dir / fig_name
print(fig_path)

# Select set fields. 
set_fields = np.arange(.75,3.5,.25)
# Freq BW.
freq_BW = np.array([19.0e9 ,  19.1e9])
# Tile freq_BW.
freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

# C, relationship between he and ne monitor.
C_exp = np.random.uniform(.5,1.5)

# Number of counts: 
N = 10**4
# monitor rate tot: 
mon = 10**8
# Set little b.
b = 0

# Simulate simple experiment.
ratio_exp, spectra_ne_exp, spectra_he_exp = mc.simple_MC(set_fields, 
                                                         freq_BWs, 
                                                         C_exp, 
                                                         b, 
                                                         counts_per_isotope = N, 
                                                         monitor_rate = mon,
                                                         counts_pois = True, 
                                                         mon_pois = True)

ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b = b, plot = False)

# Conduct fit. 
my_pars = Parameters()
my_pars.add('C', value=1, min=0, max = 10, vary =True)
my_pars.add('b', value=.1, min=-10, max = 10, vary =True)

result = minimize(mc.objfunc_chisq, my_pars, args = (freq_BWs, set_fields, ratio_exp))

# Fit report.
print(fit_report(result.params))

# Plot results.
fig0, ax0 = plt.subplots(figsize=(12,6))

C = result.params["C"].value

ratio_exp_cp = ratio_exp.copy()
ratio_exp_cp["Ratio"] = C*ratio_exp_cp["Ratio"]
ratio_exp_cp["sRatio"] = C*ratio_exp_cp["sRatio"]

pm.plot_experimental_ratio(ratio_exp_cp, ax0, label= f"exp (sim)")
pm.plot_predicted_ratio(ratio_pred, ax0)

# ax0.set_yscale("log")
ax0.set_ylabel('ratio')
ax0.set_xlabel('Set Field (T)')
ax0.set_title(f"Simulated Experiment")
ax0.legend()
plt.show()