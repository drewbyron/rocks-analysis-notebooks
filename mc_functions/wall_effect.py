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


def wall_efficiency(gamma, field, trap_radius=0.578e-2): 
    
    energy = sc.energy(gamma)
    # print(energy)
    r = sc.cyc_radius(sc.energy(gamma), field, pitch_angle = 90)
    # print(r)
    e = (trap_radius - r)**2 / trap_radius**2
    # print(field)
    return e

def AUC_expectation_we(set_fields, freq_BWs, b = 0, plot = False, wall_effect = True): 
    
    integrate_vect = np.vectorize(integrate.quad)
    
    energy_acceptances_high = sc.freq_to_energy(
        freq_BWs[:,0], set_fields
    )
    energy_acceptances_low = sc.freq_to_energy(
        freq_BWs[:,1], set_fields
    )
    energy_acceptances = np.stack((energy_acceptances_low, energy_acceptances_high), axis=-1)
    
    # Empty dict to house relative rates. 
    rates = {}

    isotopes = {"Ne19": {"b": -0.7204 * b}, "He6": {"b": b}}

    # This delta_W prevents a zero argument in a log. Doesn't effect pdfs
    delta_W = 10**-10
    
    if plot: 
        pdf_plot_pts = 10**2
        f0, ax0 = plt.subplots(1, figsize=(10,5))
        f1, ax1 = plt.subplots(1, figsize=(10,5))
        f2, ax2 = plt.subplots(1, figsize=(10,5))

    for isotope_name, isotope_info in isotopes.items(): 

        W_low = sc.gamma(energy_acceptances[:, 0]) + delta_W
        W_high = sc.gamma(energy_acceptances[:, 1])-delta_W

        # Feed the info dict to the BetaSpectrum class. 
        bspec = bs.BetaSpectrum(isotope_name, b = isotope_info["b"] )
        
        if wall_effect: 
            
            def dNdE_wall_effect(gamma, field): 
                return bspec.dNdE(gamma)*wall_efficiency(gamma, field)

            fraction_of_spectrum, err = integrate_vect(
                    dNdE_wall_effect,
                    W_low,
                    W_high,
                    args = (set_fields)
                )
        else: 
        
            fraction_of_spectrum, err = integrate_vect(
                    bspec.dNdE,
                    W_low,
                    W_high,
                )
        rates[isotope_name] = fraction_of_spectrum
        
        if plot:
            Ws = np.linspace(W_low, W_high, pdf_plot_pts)
            pdf = bspec.dNdE(Ws)
            ax0.plot(Ws, pdf)
            
            r = sc.cyc_radius(sc.energy(Ws), set_fields, pitch_angle = 90)
            ax1.plot(Ws, r)
            
            eff = wall_efficiency(Ws, set_fields)
            ax2.plot(Ws, eff)

    if plot: 
        ax0.set_xlabel(r"$\gamma$")
        ax0.set_title("He6 and Ne19 Spectra (pdf)")
        ax1.set_xlabel(r"$\gamma$")
        ax1.set_title("radius (m)")
        ax2.set_xlabel(r"$\gamma$")
        ax2.set_title("wall effect (efficiency)")
        plt.show()


    for isotope in rates: 
        rates[isotope] = np.array(rates[isotope]) 

    rates["set_fields"] = set_fields
    rates = pd.DataFrame(rates)
    
    # Make the ratio a column of the df
    rates["Ratio"] = rates["Ne19"] / rates["He6"]
    
    rates.set_index("set_fields", inplace = True)
    return rates


def we_simple_MC(set_fields, 
              freq_BWs, 
              C_exp, 
              b, 
              counts_per_isotope = 10**4, 
              monitor_rate = 10**5,                                  
              counts_pois = True,                           
              mon_pois = True,
              wall_effect = True): 
    
    # Simulate data.
    ratio_pred = AUC_expectation_we(set_fields, freq_BWs, b = b, plot = False, wall_effect = wall_effect)

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

    # ratio_exp = re.build_ratio_altnorm(spectra_ne_exp, spectra_he_exp)
    
    return spectra_ne_exp, spectra_he_exp

# def objfunc_chisq_ne(my_pars, freq_BWs, set_fields, ratio_exp): 

#     D =my_pars["D"].value
#     b =my_pars["b"].value
    
#     ratio_pred = AUC_expectation_we(set_fields, freq_BWs, b = b, plot = False, wall_effect = False)

#     ratio_exp["Ne19_corr"] = ratio_pred["He6"]*ratio_exp["Ratio"]
#     ratio_exp["sNe19_corr"] = ratio_pred["He6"]*ratio_exp["sRatio"]

#     chisq_gauss = (ratio_pred["Ne19"] - D*ratio_exp["Ne19_corr"])/ (D*ratio_exp["sNe19_corr"])

#     return chisq_gauss

def objfunc_chisq(my_pars, freq_BWs, set_fields, ratio_exp): 

    C =my_pars["C"].value
    b =my_pars["b"].value

    ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b = b, plot = False)
    chisq_gauss = (ratio_pred["Ratio"] - C*ratio_exp["Ratio"])/ (C*ratio_exp["sRatio"])

    return chisq_gauss