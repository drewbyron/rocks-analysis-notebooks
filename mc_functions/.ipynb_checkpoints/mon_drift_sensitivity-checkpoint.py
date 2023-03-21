# Author: Drew Byron
# Date: 2/2/23

import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from lmfit import minimize, Parameters, fit_report
from pathlib import Path
from scipy import integrate
# Path to local imports. Alter to match your machine. 
sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")

# Local imports.
import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc
import he6_cres_spec_sims.spec_tools.beta_source.beta_spectrum as bs

# Local imports for plotting ratios and such. 
import analysis_functions.ratio_experiment as re
import analysis_functions.ratio_prediction as rp
import analysis_functions.plotting_methods as pm


def mon_drift_MC(set_fields,
                  freq_BWs, 
                  C_exp, 
                  b, 
                  mon_drift_ne = 10**-2,
                  mon_drift_he = -10**-2,
                  counts_per_isotope = 10**4, 
                  monitor_rate = 10**5,                                  
                  counts_pois = True,                           
                  mon_pois = True ): 
    
    # Simulate data.
    ratio_pred = ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b = b, plot = False)

    # Simulate data that provides the "spectra" df for both ne and he.
    spectra_ne_exp = pd.DataFrame()
    spectra_he_exp = pd.DataFrame()

    spectra_ne_exp["set_field"] = ratio_pred.index
    spectra_he_exp["set_field"] = ratio_pred.index

    spectra_ne_exp.index = ratio_pred.index
    spectra_he_exp.index = ratio_pred.index

    spectra_ne_exp["event_count"] = ratio_pred["Ne19"]*counts_per_isotope/ratio_pred["Ne19"].sum()
    spectra_he_exp["event_count"] = ratio_pred["He6"]*counts_per_isotope/ratio_pred["He6"].sum()
    
    # Here introduce a linear monitor drift. 
    spectra_ne_exp["tot_monitor_rate"] = C_exp*monitor_rate*(1+spectra_ne_exp.reset_index().index*mon_drift_ne/spectra_ne_exp.reset_index().index.max())
    spectra_he_exp["tot_monitor_rate"] = monitor_rate*(1+spectra_ne_exp.reset_index().index*mon_drift_he/spectra_ne_exp.reset_index().index.max())
    
    if mon_pois:
        # Apply a poisson statistic with the given mean for the event counts. 
        spectra_ne_exp["tot_monitor_rate"] = np.random.poisson(spectra_ne_exp["tot_monitor_rate"])
        spectra_he_exp["tot_monitor_rate"] = np.random.poisson(spectra_he_exp["tot_monitor_rate"])
    
    if counts_pois:
        # Apply a poisson statistic with the given mean for the event counts.  
        spectra_ne_exp["event_count"] = np.random.poisson(spectra_ne_exp["event_count"])
        spectra_he_exp["event_count"] = np.random.poisson(spectra_he_exp["event_count"])

    ratio_exp = re.build_ratio_altnorm(spectra_ne_exp, spectra_he_exp)
    
    return ratio_exp, spectra_ne_exp, spectra_he_exp




def objfunc_chisq(my_pars, freq_BWs, set_fields, ratio_exp, b): 

    C =my_pars["C"].value
    b =my_pars["b"].value

    ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b = b, plot = False)
    chisq_gauss = (ratio_pred["Ratio"] - C*ratio_exp["Ratio"])/ (C*ratio_exp["sRatio"])

    return chisq_gauss


def run_mon_sensitivity_test( mon_drifts, trial_max = 20 ):
        
    seed = 1234
    rng = np.random.default_rng(seed=seed)
    
    # Select set fields. 
    set_fields = np.arange(.75,3.5,.25)
    # Number of counts: 
    N = 10**10
    # monitor rate tot: 
    mon = 10**10
    # Set little b.
    b = 0
    
    # Freq BW.
    freq_BW = np.array([18.0e9 ,  19.1e9])
    # Tile freq_BW.
    freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

    # C, relationship between he and ne monitor.
    C_exp = np.random.uniform(.5,1.5)
    
    mon_drift_test = {}
    mon_drift_he = 0
    # first run field_errs test: 
    for mon_drift in mon_drifts: 
        
        mon_drift_test[mon_drift] = []
        
        for trial in range(trial_max):
            print(f"Monitor drift: {mon_drift}, trial: {trial}")
            # Add in some uncertainty in the field that's different for ne and he. 

            # Simulate simple experiment.
            ratio_exp, spectra_ne_exp, spectra_he_exp = mon_drift_MC(set_fields,
                                                                     freq_BWs, 
                                                                     C_exp, 
                                                                     b, 
                                                                     mon_drift_ne = mon_drift,
                                                                     mon_drift_he = mon_drift_he,
                                                                     counts_per_isotope = N, 
                                                                     monitor_rate = mon,
                                                                     counts_pois = False, 
                                                                     mon_pois = False)

            # Conduct fit. 
            my_pars = Parameters()
            my_pars.add('C', value=1, min=0, max = 10, vary =True)
            my_pars.add('b', value=.1, min=-10, max = 10, vary =True)

            result = minimize(objfunc_chisq, my_pars, args = (freq_BWs, set_fields, ratio_exp, b))

            mon_drift_test[mon_drift].append(result.params["b"].value)
            
    
    mon_drift_test = pd.DataFrame(mon_drift_test)
    
    return mon_drift_test



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
    ax0.set_title(f"Sensitivity to CRES counts per isotope. Mean = {Const.mean()}")


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
    plt.savefig(figures_path / Path(f"N_const_prop.png"), bbox_inches="tight", dpi=300)
    plt.show()

    return None