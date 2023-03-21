# Author: Drew Byron
# Date: 2/2/23

import sys
import pandas as pd
import numpy as np

# Local imports for plotting ratios and such. 
import analysis_functions.ratio_experiment as re
import analysis_functions.ratio_prediction as rp
import analysis_functions.plotting_methods as pm


def simple_MC(set_fields, 
              freq_BWs, 
              C_exp, 
              b, 
              counts_per_isotope = 10**4, 
              monitor_rate = 10**5,                                  
              counts_pois = True,                           
              mon_pois = True ): 
    

    # Simulate data.
    ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b = b, plot = False)

    # Simulate data that provides the "spectra" df for both ne and he.
    spectra_ne_exp = pd.DataFrame()
    spectra_he_exp = pd.DataFrame()

    spectra_ne_exp["set_field"] = ratio_pred.index
    spectra_he_exp["set_field"] = ratio_pred.index

    spectra_ne_exp.index = ratio_pred.index
    spectra_he_exp.index = ratio_pred.index

    spectra_ne_exp["event_count"] = ratio_pred["Ne19"]*counts_per_isotope/ratio_pred["Ne19"].sum()
    spectra_he_exp["event_count"] = ratio_pred["He6"]*counts_per_isotope/ratio_pred["He6"].sum()

    spectra_ne_exp["tot_monitor_rate"] = C_exp*monitor_rate
    spectra_he_exp["tot_monitor_rate"] = monitor_rate
    
    if mon_pois:
        # Apply a poisson statistic with the given mean for the event counts. 
        spectra_ne_exp["tot_monitor_rate"] = np.random.poisson(spectra_ne_exp["tot_monitor_rate"])
        spectra_he_exp["tot_monitor_rate"] = np.random.poisson(spectra_he_exp["tot_monitor_rate"])
    
    if counts_pois:
        # Apply a poisson statistic with the given mean for the event counts.  
        spectra_ne_exp["event_count"] = np.random.poisson(spectra_ne_exp["event_count"])
        spectra_he_exp["event_count"] = np.random.poisson(spectra_he_exp["event_count"])
    
    # Be careful to use this (correct but alternate) normalization.
    ratio_exp = re.build_ratio_altnorm(spectra_ne_exp, spectra_he_exp)
    
    return ratio_exp, spectra_ne_exp, spectra_he_exp




def objfunc_chisq(my_pars, freq_BWs, set_fields, ratio_exp, b): 

    C =my_pars["C"].value
    b =my_pars["b"].value

    ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b = b, plot = False)
    chisq_gauss = (ratio_pred["Ratio"] - C*ratio_exp["Ratio"])/ (C*ratio_exp["sRatio"])

    return chisq_gauss

