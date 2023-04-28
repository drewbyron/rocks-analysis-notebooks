# Author: Drew Byron
# Date: 04/07/2023

# Imports.
import sys
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
from pathlib import Path
from scipy import integrate

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
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
}
plt.rcParams.update(params)

# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/monte_carlo")
fig_base_name = "MC_we_"
fig_suffix = ".png"

# Utility function.
def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


# ------ Plot 1-2: gammas, r_cycl, efficiencies -----------
def wall_efficiency(gamma, field, trap_radius=0.578e-2):

    energy = sc.energy(gamma)
    r = sc.cyc_radius(sc.energy(gamma), field, pitch_angle=90)
    e = (trap_radius - r) ** 2 / trap_radius**2

    return e


def plots1_2_3(set_fields, freq_BW, b=0):

    freq_BWs = np.tile(freq_BW, (len(set_fields), 1))
    integrate_vect = np.vectorize(integrate.quad)

    energy_acceptances_high = sc.freq_to_energy(freq_BWs[:, 0], set_fields)
    energy_acceptances_low = sc.freq_to_energy(freq_BWs[:, 1], set_fields)
    energy_acceptances = np.stack(
        (energy_acceptances_low, energy_acceptances_high), axis=-1
    )

    # Empty dict to house relative rates.
    rates = {}

    isotopes = {"Ne19": {"b": -0.7204 * b}, "He6": {"b": b}}
    colors = {"Ne19": "tab:blue", "He6": "tab:orange"}
    field_cmap = get_cmap(len(set_fields))

    # This delta_W prevents a zero argument in a log. Doesn't effect pdfs
    delta_W = 10**-10

    pdf_plot_pts = 10**2
    f0, ax0 = plt.subplots(1, figsize=(12, 6))
    f1, ax1 = plt.subplots(1, figsize=(12, 6))
    f2, ax2 = plt.subplots(1, figsize=(12, 6))

    for j, (isotope_name, isotope_info) in enumerate(isotopes.items()):

        W_low = sc.gamma(energy_acceptances[:, 0]) + delta_W
        W_high = sc.gamma(energy_acceptances[:, 1]) - delta_W

        # Feed the info dict to the BetaSpectrum class.
        bspec = bs.BetaSpectrum(isotope_name, b=isotope_info["b"])

        def dNdE_wall_effect(gamma, field):
            return bspec.dNdE(gamma) * wall_efficiency(gamma, field)

        fraction_of_spectrum, err = integrate_vect(
            dNdE_wall_effect, W_low, W_high, args=(set_fields), epsrel=1e-12
        )

        Ws = np.linspace(W_low, W_high, pdf_plot_pts)
        pdf = bspec.dNdE(Ws)
        for i, (W, p) in enumerate(zip(Ws.T, pdf.T)):
            if i == 0:
                ax0.fill_between(
                    W,
                    0,
                    p,
                    alpha=0.5,
                    color=colors[isotope_name],
                    label=f"{isotope_name}",
                )
            else:
                ax0.fill_between(W, 0, p, alpha=0.5, color=colors[isotope_name])
        ax0.legend()
        ax0.set_xlabel(r"$\gamma$")
        ax0.set_ylabel(r"$\frac{dN}{dE}$")

        if j == 0:
            rs = sc.cyc_radius(sc.energy(Ws), set_fields, pitch_angle=90)
            effs = wall_efficiency(Ws, set_fields)
            for i, (W, r, eff) in enumerate(zip(Ws.T, rs.T, effs.T)):
                label = f"{set_fields[i]:.2f} T"
                lw = 3
                ax1.plot(W, r * 1e3, label=label, color=field_cmap(i), linewidth=lw)
                ax2.plot(W, eff, label=label, color=field_cmap(i), linewidth=lw)

            ax1.legend()
            ax2.legend()

            ax1.set_xlabel(r"$\gamma$")
            ax2.set_xlabel(r"$\gamma$")

            ax1.set_ylabel(r"$r_{cycl}$ (mm)")
            ax2.set_ylabel(r"Efficiency ($\epsilon_{e}(E)_{we}$)")

    # Save the figures to disk.
    f0_path = fig_dir / Path(fig_base_name + "0" + fig_suffix)
    f1_path = fig_dir / Path(fig_base_name + "1" + fig_suffix)
    f2_path = fig_dir / Path(fig_base_name + "2" + fig_suffix)
    f0.savefig(f0_path, bbox_inches="tight", dpi=300)
    f1.savefig(f1_path, bbox_inches="tight", dpi=300)
    f2.savefig(f2_path, bbox_inches="tight", dpi=300)

    # plt.show()

    return None


def wall_effect_correction(set_fields, freq_BW, freq_chunk=1.1e9):
    print(f"Wall effect correction to ratio.")
    f3, ax3 = plt.subplots(1, figsize=(10, 5))

    freq_BW_tot = freq_BW[1] - freq_BW[0]
    n_chunks = int(np.ceil((freq_BW_tot / freq_chunk)))
    print(f"n_chunks: {n_chunks}")

    for i, chunk in enumerate(range(n_chunks)):
        freq_BW_chunk = np.clip(
            np.array(
                [freq_BW[0] + freq_chunk * chunk, freq_BW[0] + freq_chunk * (chunk + 1)]
            ),
            0,
            freq_BW.max(),
        )
        freq_BW_chunk_tot = freq_BW_chunk[1] - freq_BW_chunk[0]
        freq_BWs = np.tile(freq_BW_chunk, (len(set_fields), 1))

        ratio_wall = we.AUC_expectation_we(
            set_fields, freq_BWs, b=0, plot=False, wall_effect=True
        )

        ratio_0 = we.AUC_expectation_we(
            set_fields, freq_BWs, b=0, plot=False, wall_effect=False
        )

        wall_effect = (ratio_wall / ratio_0 - 1)["Ratio"]
        ax3.plot(
            set_fields,
            wall_effect.abs().values,
            label=f"chunk {chunk}: {freq_BWs[0,0]*1e-9:.1f}-{freq_BWs[0,1]*1e-9:.1f} GHz",
        )

    ax3.set_yscale("log")
    ax3.set_xlabel(r"Field (T)")
    ax3.set_ylabel(r"$abs\left(\frac{R_{we}}{R_0} - 1 \right)$ ")
    ax3.legend(loc="lower right")

    f3_path = fig_dir / Path(fig_base_name + f"ratio_n_chunks_{n_chunks}" + fig_suffix)
    f3.savefig(f3_path, bbox_inches="tight", dpi=300)

    return None


def corrected_spectrum(
    set_fields,
    freq_BW,
    freq_chunk=1100e6,
    corrections_off=False,
    MC_label="simulated data(corrected)",
):

    print(f"Corrected spectrum with wall effect.")
    C_exp = 0.75
    b = 0

    N_per_isotope = 10**5
    mon_rate = 10**14

    freq_BW_tot = freq_BW[1] - freq_BW[0]
    n_chunks = int(np.ceil((freq_BW_tot / freq_chunk)))
    print(f"n_chunks: {n_chunks}")

    f4, (ax0, ax1) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(12, 7.5)
    )

    b_fits = []
    b_errs = []

    for i, chunk in enumerate(range(n_chunks)):
        freq_BW_chunk = np.clip(
            np.array(
                [freq_BW[0] + freq_chunk * chunk, freq_BW[0] + freq_chunk * (chunk + 1)]
            ),
            0,
            freq_BW.max(),
        )
        freq_BW_chunk_tot = freq_BW_chunk[1] - freq_BW_chunk[0]
        N_per_isotope_in_chunk = int(N_per_isotope * freq_BW_chunk_tot / freq_BW_tot)
        mon_rate_per_isotope_in_chunk = mon_rate
        freq_BWs = np.tile(freq_BW_chunk, (len(set_fields), 1))

        spectra_ne_we, spectra_he_we = we.we_simple_MC(
            set_fields,
            freq_BWs,
            C_exp,
            b,
            counts_per_isotope=N_per_isotope_in_chunk,
            monitor_rate=mon_rate_per_isotope_in_chunk,
            counts_pois=False,
            mon_pois=False,
            wall_effect=True,
        )

        ratio_exp = re.build_ratio(spectra_ne_we, spectra_he_we)
        # Conduct fit.
        my_pars = Parameters()
        my_pars.add("C", value=1, min=-100, max=100, vary=True)
        my_pars.add("b", value=0, min=-10, max=10, vary=True)

        result = minimize(
            we.objfunc_chisq, my_pars, args=(freq_BWs, set_fields, ratio_exp)
        )

        # Fit report.
        # print(fit_report(result.params))

        C = result.params["C"].value
        b = result.params["b"].value

        # Get the SM prediction.
        ratio_pred = we.AUC_expectation_we(
            set_fields, freq_BWs, b=b, plot=False, wall_effect=False
        )

        ratio_corr = ratio_exp.copy()
        ratio_corr["Ne19_corr"] = C * ratio_pred["He6"] * ratio_exp["Ratio"]
        ratio_corr["sNe19_corr"] = C * ratio_pred["He6"] * ratio_exp["sRatio"]

        if corrections_off:
            # Here we are just showing how bad the fit would be without the corrections.
            ratio_corr["Ne19_corr"] = (
                spectra_ne_we["event_count"]
                * ratio_corr["Ne19_corr"].mean()
                / spectra_ne_we["event_count"].mean()
            )
            ratio_corr["sNe19_corr"] = C * ratio_pred["He6"] * ratio_exp["sRatio"]

        (
            gamma_acceptances,
            gamma_widths,
            gamma_heights,
            gamma_height_errs,
            SM_heights,
        ) = ed.freq_to_energy_domain(set_fields, freq_BWs, ratio_corr, ratio_pred)

        label_bool = i == 0
        ed.energy_domain_plot(
            ax0,
            ax1,
            gamma_acceptances,
            gamma_widths,
            gamma_heights,
            gamma_height_errs,
            SM_heights,
            ratio_corr,
            ratio_pred,
            label=label_bool,
            label_contents=MC_label,
        )

        # Now add to the list of b_normed

        b_fits.append(result.params["b"].value)
        b_errs.append(result.params["b"].stderr)

    b_fits = np.array(b_fits)
    b_errs = np.array(b_errs)

    b_fit = np.average(b_fits, weights=b_errs)
    b_err = np.sqrt(np.sum(b_errs**2))
    # print(b_fits)
    # print(b_errs)
    print(f"\n\nFinal Result: b = {b_fit}+- {b_err}")
    # NOw use the b we got from the fit in making the spectrum?? THINK ABOUT THAT.
    isotopes = {"Ne19": {"b": -0.7204 * b}, "He6": {"b": b}}
    # Feed the info dict to the BetaSpectrum class.
    bspec = bs.BetaSpectrum("Ne19")

    Ws = np.linspace(1.001, bspec.W0 - 0.001, 300)
    pdf = bspec.dNdE(Ws)
    ax0.plot(Ws, pdf, label="Ne19 pdf")

    ax0.legend()
    ax1.legend()

    ax0.set_xlim(0.85, 5.5)
    ax1.set_xlim(0.85, 5.5)

    f4_path = fig_dir / Path(
        fig_base_name
        + f"corrected_spec_n_chunks_{n_chunks}_{corrections_off}"
        + fig_suffix
    )
    f4.savefig(f4_path, bbox_inches="tight", dpi=300)

    return None


######################################
######################################
# Run the above functions to build thesis plots.


# Select set fields.
set_fields = np.arange(0.75, 3.5, 0.25)

# Full Freq BW.
freq_BW = np.array([18.1e9, 19.1e9])

# Make basic first three plots illustrating the size of the effect.
plots1_2_3(set_fields, freq_BW)

# Select set fields.
set_fields = np.arange(0.75, 3.25, 0.01)

# Freq BW.
freq_BW = np.array([18.1e9, 19.1e9])
wall_effect_correction(set_fields, freq_BW, freq_chunk=1000e6)

# Freq BW.
freq_BW = np.array([18.1e9, 19.1e9])
wall_effect_correction(set_fields, freq_BW, freq_chunk=200e6)

# Select set fields.
set_fields = np.arange(0.75, 3.25, 0.25)
# Full Freq BW.
freq_BW = np.array([18.1e9, 19.1e9])
corrected_spectrum(
    set_fields,
    freq_BW,
    freq_chunk=1000e6,
    corrections_off=False,
    MC_label="simulated data (corrected)",
)
corrected_spectrum(
    set_fields,
    freq_BW,
    freq_chunk=1000e6,
    corrections_off=True,
    MC_label="simulated data (uncorrected)",
)

# Full Freq BW.
freq_BW = np.array([18.1e9, 19.1e9])
corrected_spectrum(
    set_fields,
    freq_BW,
    freq_chunk=500e6,
    corrections_off=False,
    MC_label="simulated data (corrected)",
)

corrected_spectrum(
    set_fields,
    freq_BW,
    freq_chunk=200e6,
    corrections_off=False,
    MC_label="simulated data (corrected)",
)

corrected_spectrum(
    set_fields,
    freq_BW,
    freq_chunk=100e6,
    corrections_off=False,
    MC_label="simulated data (corrected)",
)
plt.show()
