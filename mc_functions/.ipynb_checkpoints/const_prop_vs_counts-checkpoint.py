import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np
import uproot4
from lmfit import minimize, Parameters, fit_report
from scipy import integrate
from pathlib import Path

# # Path to local imports.
# sys.path.append("/home/drew/He6CRES/rocks_analysis_pipeline/")
# from results import ExperimentResults
# from rocks_utility import he6cres_db_query

# # Path to local imports.
# sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")

# # Local imports.
# import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc
# import he6_cres_spec_sims.experiment as exp
# import he6_cres_spec_sims.spec_tools.beta_source.beta_spectrum as bs

# Local imports for plotting ratios and such. 
import analysis_functions.ratio_experiment as re
import analysis_functions.ratio_prediction as rp
import analysis_functions.plotting_methods as pm
import mc_functions.simple_mc as mc

# Plot font size
plt.rcParams.update({'font.size': 20})

def run_N_const_prop_test(exp_max = 12, trial_max = 20 ):
    
    # Select set fields. 
    set_fields = np.arange(.75,3.5,.25)
    # Freq BW.
    freq_BW = np.array([19.0e9 ,  19.1e9])
    # Tile freq_BW.
    freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

    # C, relationship between he and ne monitor.
    C_exp = np.random.uniform(.25,1.75)

    # monitor rate tot: 
    mon = 10**10
    # Set little b.
    b = 0

    b_uncert = {}
    for exp in np.arange(3,exp_max,1):
        # Number of counts: 
        N = int(10**exp)

        b_uncert[exp] = []

        for trial in range(trial_max): 
            print(f"N = 10**{exp}, trial = {trial}")

            # Simulate simple experiment. Don't poisson vary monitor.
            ratio_exp, spectra_ne_exp, spectra_he_exp = mc.simple_MC(set_fields, 
                                                                     freq_BWs, 
                                                                     C_exp, 
                                                                     b, 
                                                                     counts_per_isotope = N, 
                                                                     monitor_rate = mon,
                                                                     counts_pois = True, 
                                                                     mon_pois = False)

            ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b = b, plot = False)

            # Conduct fit. 
            my_pars = Parameters()
            my_pars.add('C', value=1, min=0, max = 10, vary =True)
            my_pars.add('b', value=.001, min=-10, max = 10, vary =True)

            result = minimize(mc.objfunc_chisq, my_pars, args = (freq_BWs, set_fields, ratio_exp, b))

            b_uncert[exp].append(result.params["b"].stderr)

    b_uncert = pd.DataFrame(b_uncert)
    
    return b_uncert

def plot_N_const_prop_test(b_uncert):
    
    # Plot results.
    fig0, ax0 = plt.subplots(figsize=(12,6))
    N = 10**b_uncert.mean().index.values
    Const = b_uncert.mean().values * np.sqrt(N)
    Const_err = b_uncert.std().values * np.sqrt(N)
    plt.errorbar(N, Const, yerr =  Const_err)

    ax0.set_xscale("log")
    ax0.set_ylabel('Proportionality constant (unitless)')
    ax0.set_xlabel('N per isotope (counts)')
    ax0.set_title(f"Sensitivity to number of counts per isotope ")


    # ------ Set Thesis Plot Parameters -----------

    params = {
        "axes.titlesize": 15,
        "legend.fontsize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
    plt.rcParams.update(params)
    figsize = (10,6)

    # Save the figure.
    figures_path = Path("/home/drew/He6CRES/rocks_analysis_notebooks/saved_plots")
    plt.savefig(figures_path / Path(f"Const_prop.png"), bbox_inches="tight", dpi=300)
    plt.show()


    return None