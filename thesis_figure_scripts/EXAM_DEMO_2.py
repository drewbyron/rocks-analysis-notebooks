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
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}
plt.rcParams.update(params)


# Set fig path.
fig_dir = Path("/media/drew/T7 Shield/thesis_figures/final_exam")
fig_base_name = "EXAM_DEMO_frombelow_"
fig_suffix = ".png"
f1_path = fig_dir / Path(fig_base_name + "1" + fig_suffix)
f2_path = fig_dir / Path(fig_base_name + "2" + fig_suffix)
    

def viz_mc_events(ax, events, set_field, color="r", lw= 1):

    events_sf = events.groupby(["set_field"]).get_group(set_field)

    for row_index, row in events_sf.iterrows():

        time_coor = np.array([row["EventStartTime"], row["EventEndTime"]])
        freq_coor = np.array([row["EventStartFreq"], row["EventEndFreq"]])


        ax.plot(
            time_coor*1e3,
            freq_coor*1e-9,
            color=color,
            alpha=0.5,
            marker="o",
            markerfacecolor="red",
            markersize=1,
            lw = lw
        ) 

    ax.set_ylabel("Freq (GHz)")
    ax.set_xlabel("Time (ms)")

    return None

n_tot=.5*1e3
rerun_sims=True
slew_time = 35e-3
# Setting up a "from-below" experiment
isotopes = ["Ne19", "He6"]
seed = 12345

# Set fields to use.
set_fields = np.arange(0.75, 3.5, 0.25)

# Hardware freq_BW
freq_BW = np.array([18.0e9, 19.1e9])
freq_BWs = np.tile(freq_BW, (len(set_fields), 1))

f1, ax1 = plt.subplots(figsize=(12, 6))
f2, ax2= plt.subplots(figsize=(12, 6))

if rerun_sims:
    for cut, color, lw in zip( [False, True], ["b", "r"] ,[1, 2]): 

        events_ne = fb.build_canditate_events(
            freq_BW,
            set_fields,
            isotopes[0],
            cut_events=cut,
            fix_slopes=False,
            slew_time=slew_time,
            n=n_tot,
            rng_seed=seed,
        )

        events_he = fb.build_canditate_events(
            freq_BW,
            set_fields,
            isotopes[1],
            cut_events=cut,
            fix_slopes=False,
            slew_time=slew_time,
            n=n_tot,
            rng_seed=seed,
        )
        
        viz_mc_events(ax1, events_ne, 3.0, color=color, lw = lw)
        viz_mc_events(ax2, events_he, 3.0, color=color, lw = lw)
ax1.set_ylim(1.6e10*1e-9, 1.91e10*1e-9)
ax2.set_ylim(1.6e10*1e-9, 1.91e10*1e-9)

f1.savefig(f1_path, bbox_inches="tight", dpi=300)
f2.savefig(f2_path, bbox_inches="tight", dpi=300)

plt.show()