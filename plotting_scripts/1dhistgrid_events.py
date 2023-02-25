import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np
from scipy import stats


# ------ Set Plot Parameters -----------

params = {'axes.titlesize': 12,
          'legend.fontsize': 10,
          'axes.labelsize': 10,
          'xtick.labelsize':10,
          'ytick.labelsize':10}
plt.rcParams.update(params)
# ------ Load events for he and ne ------

#read in He-6 data
he_events_path = "/media/drew/T7 Shield/rocks_analysis/saved_experiments/he6_full_0_aid_1/events.csv"
he = pd.read_csv(he_events_path, index_col =0)

#read in Ne-19 data
ne_events_path = "/media/drew/T7 Shield/rocks_analysis/saved_experiments/ne19_full_0_aid_2/events.csv"
ne = pd.read_csv(ne_events_path, index_col =0)

# ------ Make base cuts and drop currupt files -------

he = he[~he.run_id.isin([377,380, 381])]
ne = ne[~ne.run_id.isin([522])]

# -------- Plotting Grid of Histograms ------------




def hist_grid_by_field(ne, he, feature, nrows, yscale, bins, fig_path, cuts = True): 

    if cuts: 
        # Apply cuts. 
        he_c = he[(he["EventStartFreq"] > 150e6)& (he["EventMeanSNR"]>12) ]
        ne_c = ne[(ne["EventStartFreq"] > 150e6)& (ne["EventMeanSNR"]>12)]

    else: 
        he_c = he
        ne_c = ne

    grouped_ne = ne_c.groupby('set_field')
    grouped_he = he_c.groupby('set_field')
    rowlength = int(np.ceil(grouped_ne.ngroups/3))                    
    fig, axs = plt.subplots(figsize=(25,12), 
                            nrows=3, ncols=rowlength,     # fix as above
                            gridspec_kw=dict(hspace=0.8)) # Much control of gridspec

    targets = zip(grouped_ne.groups.keys(),  axs.flatten())

    for i, (key, ax) in enumerate(targets):

        
        ne_field = grouped_ne.get_group(key)[feature]
        he_field = grouped_he.get_group(key)[feature]

        # Calculate KS test for similarity of samples.
        ks = stats.kstest(ne_field, he_field)

        ax.hist(ne_field, bins=bins, 
                       histtype=u'step', density=True, label = "Ne")
        ax.hist(he_field, bins=bins, 
                       histtype=u'step', density=True, label = "He")
        ax.set_title(f"field: {key} T. KS pval = {ks.pvalue:.4f}")
        ax.set_yscale(yscale)
        ax.set_xlabel(feature)
        ax.legend()

    fig.suptitle(f"{feature}. Cuts = {cuts}.", fontsize = 25)
    plt.savefig(fig_path, bbox_inches='tight', dpi = 200)

    return None


# ----- Loop through all features and make a hist plot with and without cuts ---- 

# Set parameters of plot. 

nrows = 3
bins = 40
yscale = "log"
base_path = "/media/drew/T7 Shield/rocks_analysis/paper_02_figures/hist_grids/"


features = ne.columns.to_list()
features.remove('set_field')

for feature in features: 
    for cuts in True, False: 

        print(f"Building hist for {feature}. cuts = {cuts}.")

        fig_path = base_path + f"events_hist_{feature}_cuts_{cuts}.png"

        hist_grid_by_field(ne, he, feature, nrows, yscale, bins, fig_path, cuts = cuts)

