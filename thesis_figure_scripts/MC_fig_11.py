# Author: Drew Byron
# Date: 04/07/2023
"""
Description: Here we use the "from-below" monte carlo to generate 
"""

# Imports.
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import minimize, Parameters, fit_report
from pathlib import Path
from pprint import pprint

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
    "legend.fontsize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}
plt.rcParams.update(params)

# Set fig path.
dir_path = Path("/media/drew/T7 Shield/thesis_figures/monte_carlo")


# fig_name_counts = Path("MC_fig_1aa_counts.png")
# fig_name_mon = Path("MC_fig_2aa_mon_.png")
# fig_path_counts = fig_dir / fig_name_counts
# fig_path_mon = fig_dir / fig_name_mon

# # CSV paths.
# N_bs_csv_path = fig_dir / Path("N_bs_a.csv")
# N_bs_uncert_csv_path = fig_dir / Path("N_bs_uncert_a.csv")
# mon_bs_csv_path = fig_dir / Path("mon_bs_a.csv")
# mon_bs_uncert_csv_path = fig_dir / Path("mon_bs_uncert_a.csv")


def make_from_below_fraction_plot(slew_time, dir_path, n_tot=1e6, rerun_sims=False):

    # Define necessary paths:
    fig_path = dir_path / Path(f"MC_11_frac_from_below_{slew_time}.png")
    events_he_path = dir_path / Path(f"events_he_slew_{slew_time}.csv")
    events_ne_path = dir_path / Path(f"events_ne_slew_{slew_time}.csv")

    # Setting up a "from-below" experiment
    isotopes = ["Ne19", "He6"]
    seed = 12345

    # Set fields to use.
    set_fields = np.arange(0.75, 3.5, 0.25)

    # Hardware freq_BW
    freq_BW = np.array([18.0e9, 19.1e9])
    freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

    if rerun_sims:

        events_ne = fb.build_canditate_events(
            freq_BW,
            set_fields,
            isotopes[0],
            cut_events=True,
            fix_slopes=False,
            slew_time=slew_time,
            n=n_tot,
            rng_seed=seed,
        )

        events_he = fb.build_canditate_events(
            freq_BW,
            set_fields,
            isotopes[1],
            cut_events=True,
            fix_slopes=False,
            slew_time=slew_time,
            n=n_tot,
            rng_seed=seed,
        )

        events_he.to_csv(events_he_path)
        events_ne.to_csv(events_ne_path)

    events_he = pd.read_csv(events_he_path)
    events_ne = pd.read_csv(events_ne_path)

    # Define cuts. Note that the start frequency of all events originating
    # from below is assigned to be exactly 18.0 GHz.
    below_cut = {"EventStartFreq": (17.99e9, 18.0001e9)}
    within_cut = {"EventStartFreq": (18.001e9, 19.1e9)}

    spectra_ne_below, spectra_he_below = fb.build_MC_spectra(
        events_ne, events_he, below_cut, monitor_rate=10**5
    )
    spectra_ne_within, spectra_he_within = fb.build_MC_spectra(
        events_ne, events_he, within_cut, monitor_rate=10**5
    )

    # Here we calculate the percent of events originating below the visible bandwidth
    ne_from_below = (
        spectra_ne_below / (spectra_ne_below + spectra_ne_within)
    ).event_count.values * 100
    he_from_below = (
        spectra_he_below / (spectra_he_below + spectra_he_within)
    ).event_count.values * 100

    # Plot results.
    fig0, ax0 = plt.subplots(figsize=(12, 6))

    plt.plot(set_fields, ne_from_below, marker="o", ms=5, color="g", label="Ne")
    plt.plot(set_fields, he_from_below, marker="o", ms=5, color="c", label="He")

    # ax0.set_yscale("log")
    ax0.set_ylabel(r"Events originating below 18.0 GHz ($\%$)")
    ax0.set_xlabel("Field (T)")
    ax0.legend(
        loc="upper left",
        title=r"$t_{trap}$= " + f"{slew_time*1000} ms",
        fontsize=15,
        title_fontsize=15,
        fancybox=False,
    )

    fig0.savefig(fig_path, bbox_inches="tight", dpi=300)

    # Now look at the percent change of the
    return None


def make_change_in_ratio_plot(slew_times, dir_path):

    # Define necessary paths:
    fig_path = dir_path / Path(f"MC_11_change_in_ratio.png")


    # Build the experimental ratio.
    within_cut = {"EventStartFreq": (18.001e9, 19.1e9)}
    everything_cut = {"EventStartFreq": (17.99e9, 19.1e9)}

    cuts = [within_cut, everything_cut]

    # Plot results.
    fig0, ax0 = plt.subplots(figsize=(12, 6))

    for slew_time in slew_times:

        # Note you need to have already made these results with the first function
        events_he_path = dir_path / Path(f"events_he_slew_{slew_time}.csv")
        events_ne_path = dir_path / Path(f"events_ne_slew_{slew_time}.csv")
        events_he = pd.read_csv(events_he_path)
        events_ne = pd.read_csv(events_ne_path)

        ratios = []
        for cut in cuts:
            spectra_ne, spectra_he = fb.build_MC_spectra(
                events_ne, events_he, cut, monitor_rate=10**7
            )

            ratio_exp = re.build_ratio(spectra_ne, spectra_he)
            # pprint("ratio_exp\n", ratio_exp)
            ratios.append(ratio_exp)

        ratio_change = 100*(ratios[1].Ratio - ratios[0].Ratio) / ratios[0].Ratio

        pprint(ratio_change)
        ax0.plot(
            ratio_exp.index,
            ratio_change,
            label=r"$t_{trap}$= " + f"{slew_time*1000} ms",
            marker="o",
            ms=6,
        )

    ax0.set_ylabel(r"Ratio relative difference ($\%$)")
    ax0.set_xlabel("Field (T)")
    ax0.legend()

    fig0.savefig(fig_path, bbox_inches="tight", dpi=300)

    return None


slew_times = [35e-3, 10e-3]
n_tot = 1e7
rerun_sims = False

for slew_time in slew_times:

    make_from_below_fraction_plot(
        slew_time, dir_path, n_tot=n_tot, rerun_sims=rerun_sims
    )

make_change_in_ratio_plot(slew_times, dir_path)

plt.show()

# xxx End of file. xxx
