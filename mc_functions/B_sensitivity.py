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
import mc_functions.simple_mc as mc


def AUC_expectation_alt(
    set_fields_ne, set_fields_he, set_fields_niave, freq_BWs, b=0, plot=False
):

    integrate_vect = np.vectorize(integrate.quad)

    energy_acceptances_high_ne = sc.freq_to_energy(freq_BWs[:, 0], set_fields_ne)
    energy_acceptances_low_ne = sc.freq_to_energy(freq_BWs[:, 1], set_fields_ne)
    energy_acceptances_ne = np.stack(
        (energy_acceptances_low_ne, energy_acceptances_high_ne), axis=-1
    )

    energy_acceptances_high_he = sc.freq_to_energy(freq_BWs[:, 0], set_fields_ne)
    energy_acceptances_low_he = sc.freq_to_energy(freq_BWs[:, 1], set_fields_ne)
    energy_acceptances_he = np.stack(
        (energy_acceptances_low_he, energy_acceptances_high_he), axis=-1
    )

    # Now write these (in principle different) energy acceptances to a dict.
    energy_acceptances = {"Ne19": energy_acceptances_ne, "He6": energy_acceptances_he}
    # Empty dict to house relative rates.
    rates = {}
    isotopes = {"Ne19": {"b": -0.7204 * b}, "He6": {"b": b}}

    # This delta_W prevents a zero argument in a log. Doesn't effect pdfs
    delta_W = 10**-10

    if plot:
        pdf_plot_pts = 10**2
        f, ax = plt.subplots(1, figsize=(10, 5))

    for isotope_name, isotope_info in isotopes.items():

        W_low = sc.gamma(energy_acceptances[isotope_name][:, 0]) + delta_W
        W_high = sc.gamma(energy_acceptances[isotope_name][:, 1]) - delta_W

        # Feed the info dict to the BetaSpectrum class.
        bspec = bs.BetaSpectrum(isotope_name, b=isotope_info["b"])

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

    rates["set_fields"] = set_fields_niave
    rates = pd.DataFrame(rates)

    # Make the ratio a column of the df
    rates["Ratio"] = rates["Ne19"] / rates["He6"]

    rates.set_index("set_fields", inplace=True)
    return rates


def field_spread_MC(
    set_fields_ne,
    set_fields_he,
    set_fields_niave,
    freq_BWs,
    C_exp,
    b,
    counts_per_isotope=10**4,
    monitor_rate=10**5,
    counts_pois=True,
    mon_pois=True,
):

    # Simulate data.
    ratio_pred = AUC_expectation_alt(
        set_fields_ne, set_fields_he, set_fields_niave, freq_BWs, b=b, plot=False
    )

    # Simulate data that provides the "spectra" df for both ne and he.
    spectra_ne_exp = pd.DataFrame()
    spectra_he_exp = pd.DataFrame()

    spectra_ne_exp["set_field"] = ratio_pred.index
    spectra_he_exp["set_field"] = ratio_pred.index

    spectra_ne_exp.index = ratio_pred.index
    spectra_he_exp.index = ratio_pred.index

    spectra_ne_exp["event_count"] = (
        ratio_pred["Ne19"] * counts_per_isotope / ratio_pred["Ne19"].sum()
    )
    spectra_he_exp["event_count"] = (
        ratio_pred["He6"] * counts_per_isotope / ratio_pred["He6"].sum()
    )

    spectra_ne_exp["tot_monitor_rate"] = C_exp * monitor_rate
    spectra_he_exp["tot_monitor_rate"] = monitor_rate

    if mon_pois:
        # Apply a poisson statistic with the given mean for the event counts.
        spectra_ne_exp["tot_monitor_rate"] = np.random.poisson(
            spectra_ne_exp["tot_monitor_rate"]
        )
        spectra_he_exp["tot_monitor_rate"] = np.random.poisson(
            spectra_he_exp["tot_monitor_rate"]
        )

    if counts_pois:
        # Apply a poisson statistic with the given mean for the event counts.
        spectra_ne_exp["event_count"] = np.random.poisson(spectra_ne_exp["event_count"])
        spectra_he_exp["event_count"] = np.random.poisson(spectra_he_exp["event_count"])

    ratio_exp = re.build_ratio_altnorm(spectra_ne_exp, spectra_he_exp)

    return ratio_exp, spectra_ne_exp, spectra_he_exp


# def objfunc_chisq(my_pars, freq_BWs, set_fields, ratio_exp, b):

#     C =my_pars["C"].value
#     b =my_pars["b"].value

#     ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b = b, plot = False)
#     chisq_gauss = (ratio_pred["Ratio"] - C*ratio_exp["Ratio"])/ (C*ratio_exp["sRatio"])

#     return chisq_gauss


def run_B_sensitivity_test(field_errs, field_offsets, trial_max=20):

    seed = 1234
    rng = np.random.default_rng(seed=seed)

    # Select set fields.
    set_fields_niave = np.arange(0.75, 3.5, 0.25)
    # Number of counts:
    N = 10**10
    # monitor rate tot:
    mon = 10**10
    # Set little b.
    b = 0

    # Freq BW.
    freq_BW = np.array([18.0e9, 19.1e9])
    # Tile freq_BW.
    freq_BWs = np.tile(freq_BW, (len(set_fields_niave), 1))

    # C, relationship between he and ne monitor.
    C_exp = np.random.uniform(0.5, 1.5)

    field_err_test = {}
    field_offset_test = {}

    # first run field_errs test:
    for field_err in field_errs:
        field_offset = 10**-10
        field_err_test[field_err] = []

        for trial in range(trial_max):
            print(f"Field err: {field_err}, trial: {trial}")
            # Add in some uncertainty in the field that's different for ne and he.

            set_fields_ne = rng.normal(
                set_fields_niave + set_fields_niave * field_offset,
                set_fields_niave * field_err,
            )
            set_fields_he = rng.normal(set_fields_niave, set_fields_niave * field_err)

            # Simulate simple experiment.
            ratio_exp, spectra_ne_exp, spectra_he_exp = field_spread_MC(
                set_fields_ne,
                set_fields_he,
                set_fields_niave,
                freq_BWs,
                C_exp,
                b,
                counts_per_isotope=N,
                monitor_rate=mon,
                counts_pois=False,
                mon_pois=False,
            )

            # Conduct fit.
            my_pars = Parameters()
            my_pars.add("C", value=1, min=0, max=10, vary=True)
            my_pars.add("b", value=0.1, min=-10, max=10, vary=True)

            result = minimize(
                mc.objfunc_chisq, my_pars, args=(freq_BWs, set_fields_niave, ratio_exp)
            )

            field_err_test[field_err].append(result.params["b"].value)

    # Second run field_offset test:
    for field_offset in field_offsets:
        field_err = 10**-10
        field_offset_test[field_offset] = []

        for trial in range(trial_max):
            print(f"Field offset: {field_offset}, trial: {trial}")

            # Add in some uncertainty in the field that's different for ne and he.

            set_fields_ne = rng.normal(
                set_fields_niave + set_fields_niave * field_offset,
                set_fields_niave * field_err,
            )
            set_fields_he = rng.normal(set_fields_niave, set_fields_niave * field_err)

            # Simulate simple experiment.
            ratio_exp, spectra_ne_exp, spectra_he_exp = field_spread_MC(
                set_fields_ne,
                set_fields_he,
                set_fields_niave,
                freq_BWs,
                C_exp,
                b,
                counts_per_isotope=N,
                monitor_rate=mon,
                counts_pois=False,
                mon_pois=False,
            )

            # Conduct fit.
            my_pars = Parameters()
            my_pars.add("C", value=1, min=0, max=10, vary=True)
            my_pars.add("b", value=0.1, min=-10, max=10, vary=True)

            result = minimize(
                mc.objfunc_chisq, my_pars, args=(freq_BWs, set_fields_niave, ratio_exp)
            )

            field_offset_test[field_offset].append(result.params["b"].value)

    field_err_test = pd.DataFrame(field_err_test)
    field_offset_test = pd.DataFrame(field_offset_test)

    return field_err_test, field_offset_test


#     # Tile freq_BW.
#     freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

#     # C, relationship between he and ne monitor.
#     C_exp = np.random.uniform(.25,1.75)

#     # monitor rate tot:
#     mon = 10**10
#     # Set little b.
#     b = 0


#     b_uncert = {}
#     for exp in np.arange(3,exp_max,1):
#         # Number of counts:
#         N = int(10**exp)

#         b_uncert[exp] = []

#         for trial in range(trial_max):
#             print(f"N = 10**{exp}, trial = {trial}")

#             # Simulate simple experiment. Don't poisson vary monitor.
#             ratio_exp, spectra_ne_exp, spectra_he_exp = mc.simple_MC(set_fields,
#                                                                      freq_BWs,
#                                                                      C_exp,
#                                                                      b,
#                                                                      counts_per_isotope = N,
#                                                                      monitor_rate = mon,
#                                                                      counts_pois = True,
#                                                                      mon_pois = False)

#             ratio_pred = rp.AUC_expectation(set_fields, freq_BWs, b = b, plot = False)

#             # Conduct fit.
#             my_pars = Parameters()
#             my_pars.add('C', value=1, min=0, max = 10, vary =True)
#             my_pars.add('b', value=.001, min=-10, max = 10, vary =True)

#             result = minimize(mc.objfunc_chisq, my_pars, args = (freq_BWs, set_fields, ratio_exp, b))

#             b_uncert[exp].append(result.params["b"].stderr)

#     b_uncert = pd.DataFrame(b_uncert)

#     return b_uncert


def plot_N_const_prop_test(b_uncert):

    # Plot results.
    fig0, ax0 = plt.subplots(figsize=(12, 6))
    N = 10 ** b_uncert.mean().index.values
    Const = b_uncert.mean().values * np.sqrt(N)
    Const_err = b_uncert.std().values * np.sqrt(N)
    plt.errorbar(N, Const, yerr=Const_err)

    ax0.set_xscale("log")
    ax0.set_ylabel("Proportionality constant (unitless)")
    ax0.set_xlabel("N per isotope (counts)")
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
    figsize = (10, 6)

    # Save the figure.
    figures_path = Path("/home/drew/He6CRES/rocks_analysis_notebooks/saved_plots")
    plt.savefig(figures_path / Path(f"N_const_prop.png"), bbox_inches="tight", dpi=300)
    plt.show()

    return None
