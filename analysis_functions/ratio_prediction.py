# Author: Drew Byron
# Date: 2/2/23

import sys
import matplotlib.pyplot as plt

# import matplotlib.ticker as ticker
import pandas as pd

# import seaborn as sns
# from datetime import datetime
import numpy as np
from scipy import integrate


# Path to local imports. Alter to match your machine.
sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")

# Local imports.
import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc
import he6_cres_spec_sims.spec_tools.beta_source.beta_spectrum as bs

# import he6_cres_spec_sims.spec_tools.beta_source.beta_spectrum as bs


def AUC_expectation(set_fields, freq_BWs, b=0, plot=False):

    integrate_vect = np.vectorize(integrate.quad)

    energy_acceptances_high = sc.freq_to_energy(freq_BWs[:, 0], set_fields)
    energy_acceptances_low = sc.freq_to_energy(freq_BWs[:, 1], set_fields)
    energy_acceptances = np.stack(
        (energy_acceptances_low, energy_acceptances_high), axis=-1
    )

    # Empty dict to house relative rates.
    rates = {}
    isotopes = {"Ne19": {"b": -0.7204 * b}, "He6": {"b": b}}

    # This delta_W prevents a zero argument in a log. Doesn't effect pdfs
    delta_W = 10**-10

    if plot:
        pdf_plot_pts = 10**2
        f, ax = plt.subplots(1, figsize=(10, 5))

    for isotope_name, isotope_info in isotopes.items():

        W_low = sc.gamma(energy_acceptances[:, 0]) + delta_W
        W_high = sc.gamma(energy_acceptances[:, 1]) - delta_W

        # Feed the info dict to the BetaSpectrum class.
        bspec = bs.BetaSpectrum(isotope_name, b = isotope_info["b"] )

        fraction_of_spectrum, err = integrate_vect(
            bspec.dNdE,
            W_low,
            W_high,
        )
        rates[isotope_name] = fraction_of_spectrum

        if plot:
            Ws = np.linspace(W_low, W_high, pdf_plot_pts)
            pdf = bspec.dNdE(Ws)
            ax.plot(Ws, pdf)

    if plot:
        ax.set_xlabel(r"$\gamma$")
        plt.title("He6 and Ne19 Spectra (pdf)")
        plt.show()

    for isotope in rates:
        rates[isotope] = np.array(rates[isotope])

    rates["set_fields"] = set_fields
    rates = pd.DataFrame(rates)

    # Make the ratio a column of the df
    rates["Ratio"] = rates["Ne19"] / rates["He6"]

    rates.set_index("set_fields", inplace=True)
    return rates


def AUC_expectation_old(set_fields, freq_BWs, b=0, plot=False):
    """
    Need to use the old beta spectrum class (based on Alejandro's mathematica nb)
    to use this version.

    """
    integrate_vect = np.vectorize(integrate.quad)

    energy_acceptances_high = sc.freq_to_energy(freq_BWs[:, 0], set_fields)
    energy_acceptances_low = sc.freq_to_energy(freq_BWs[:, 1], set_fields)
    energy_acceptances = np.stack(
        (energy_acceptances_low, energy_acceptances_high), axis=-1
    )

    # Empty dict to house relative rates.
    rates = {}
    isotopes = {
        "Ne19": {
            "W0": 5.339539,
            "Z": 10,
            "A": 19,
            "beta_type": "Mixed",
            "decay_type": "+",
            "mixing_ratio": 2.22,
            "R": 2.9,
            "bAc": 0,
            "dAc": 0,
            "Lambda": 0,
            "b": -0.7204 * b,
        },
        "He6": {
            "W0": 7.859525,
            "Z": 2,
            "A": 6,
            "beta_type": "Gamow-Teller",
            "decay_type": "-",
            "mixing_ratio": None,
            "R": 1.6,
            "bAc": 0,
            "dAc": 0,
            "Lambda": 0,
            "b": b,
        },
    }

    # This delta_W prevents a zero argument in a log. Doesn't effect pdfs
    delta_W = 10**-10

    if plot:
        pdf_plot_pts = 10**2
        f, ax = plt.subplots(1, figsize=(10, 5))

    for isotope_name, isotope_info in isotopes.items():

        W_low = sc.gamma(energy_acceptances[:, 0]) + delta_W
        W_high = sc.gamma(energy_acceptances[:, 1]) - delta_W

        # Feed the info dict to the BetaSpectrum class.
        bspec = bs.BetaSpectrum(isotope_info)

        fraction_of_spectrum, err = integrate_vect(
            bspec.dNdE,
            W_low,
            W_high,
        )
        rates[isotope_name] = fraction_of_spectrum

        if plot:
            Ws = np.linspace(W_low, W_high, pdf_plot_pts)
            pdf = bspec.dNdE(Ws)
            ax.plot(Ws, pdf)

    if plot:
        ax.set_xlabel(r"$\gamma$")
        plt.title("He6 and Ne19 Spectra (pdf)")
        plt.show()

    for isotope in rates:
        rates[isotope] = np.array(rates[isotope])

    rates["set_fields"] = set_fields
    rates = pd.DataFrame(rates)

    # Make the ratio a column of the df
    rates["Ratio"] = rates["Ne19"] / rates["He6"]

    rates.set_index("set_fields", inplace=True)
    return rates
