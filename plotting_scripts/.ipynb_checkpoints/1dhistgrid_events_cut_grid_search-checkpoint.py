import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np
from scipy import stats
from pathlib import Path

import itertools


# ------ Set Plot Parameters -----------

params = {
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
plt.rcParams.update(params)
# ------ Load events for he and ne ------

# read in He-6 data
he_events_path = (
    "/media/drew/T7 Shield/rocks_analysis/saved_experiments/he6_full_0_aid_1/events.csv"
)
he = pd.read_csv(he_events_path, index_col=0)

# read in Ne-19 data
ne_events_path = "/media/drew/T7 Shield/rocks_analysis/saved_experiments/ne19_full_0_aid_2/events.csv"
ne = pd.read_csv(ne_events_path, index_col=0)

# ------ Make base cuts and drop currupt files -------

he = he[~he.run_id.isin([377, 380, 381])]
ne = ne[~ne.run_id.isin([522])]

# -------- Plotting Grid of Histograms ------------


def hist_grid_by_field(
    ne,
    he,
    feature,
    nrows,
    yscale,
    bins,
    fig_path,
    apply_cut=True,
    cuts={"EventStartFreq": 150e6, "EventMeanSNR": 12},
):

    if apply_cut:
        # Apply cuts.
        he_c = he[
            (he["EventStartFreq"] > cuts["EventStartFreq"])
            & (he["EventMeanSNR"] > cuts["EventMeanSNR"])
        ]
        ne_c = ne[
            (ne["EventStartFreq"] > cuts["EventStartFreq"])
            & (ne["EventMeanSNR"] > cuts["EventMeanSNR"])
        ]

    else:
        he_c = he
        ne_c = ne

    grouped_ne = ne_c.groupby("set_field")
    grouped_he = he_c.groupby("set_field")
    rowlength = int(np.ceil(grouped_ne.ngroups / 3))
    fig, axs = plt.subplots(
        figsize=(25, 12),
        nrows=3,
        ncols=rowlength,  # fix as above
        gridspec_kw=dict(hspace=0.8),
    )  # Much control of gridspec

    targets = zip(grouped_ne.groups.keys(), axs.flatten())

    for i, (key, ax) in enumerate(targets):

        ne_field = grouped_ne.get_group(key)[feature]
        he_field = grouped_he.get_group(key)[feature]

        # Calculate KS test for similarity of samples.
        ks = stats.kstest(ne_field, he_field)

        ax.hist(ne_field, bins=bins, histtype="step", density=True, label="Ne")
        ax.hist(he_field, bins=bins, histtype="step", density=True, label="He")
        ax.set_title(f"field: {key} T. KS pval = {ks.pvalue:.4f}")
        ax.set_yscale(yscale)
        ax.set_xlabel(feature)
        ax.legend()

    fig.suptitle(f"{feature}. Cuts = {cuts}.", fontsize=25)
    plt.savefig(fig_path, bbox_inches="tight", dpi=200)

    return None


# ----- Loop through all features and make a hist plot with and without cuts ----

# Set parameters of plot.

nrows = 3
bins = 80
yscale = "log"
base_path = "/media/drew/T7 Shield/rocks_analysis/paper_02_figures/ratio_cuts_12152022"

# Cuts to make for each set of plots:


# ne.columns.to_list()
features = [
    "EventStartTime",
    "EventStartFreq",
    "EventEndFreq",
    "EventTimeLength",
    "EventFreqLength",
    "EventTrackCoverage",
    "EventMeanSNR",
    "EventSlope",
    "EventNBins",
    "EventTrackTot",
    "EventFreqIntc",
    "EventTimeIntc",
]


freq = np.arange(100, 600, 50)
snr = np.arange(10, 20, 2)

freq_cuts, snr_cuts = np.meshgrid(freq, snr)
freq_cuts = freq_cuts.flatten()
snr_cuts = snr_cuts.flatten()

# Breaking this into blocks and then running a few times because the process 
# keeps getting killed. 
cuts = list(zip(freq_cuts, snr_cuts))
len_cuts = len(cuts)
cut_chunks = [cuts[x:x+10] for x in range(0, len(cuts), 10)]

print(len(cut_chunks))

i = 4
for freq_cut, snr_cut in cut_chunks[i]:

    dir_path = Path(base_path) / Path(f"freq_snr__{freq_cut:.0f}_{snr_cut:.0f}")

    if not dir_path.exists(): 
        dir_path.mkdir()

    for feature in features:
     
        cuts = {"EventStartFreq": freq_cut*1e6, "EventMeanSNR": snr_cut}

        print(f"Building hist for {feature}. cuts = {cuts}.")

        fig_path = dir_path / Path(f"{feature}_event_hist.png")

        hist_grid_by_field(ne, he, feature, nrows, yscale, bins, fig_path, apply_cut = True, cuts=cuts)
