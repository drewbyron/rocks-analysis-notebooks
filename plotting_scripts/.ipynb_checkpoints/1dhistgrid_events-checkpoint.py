%autoreload 2
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np

# Jupyter Lab imports.
import ipywidgets as widgets
from ipywidgets import interact, interact_manual, fixed

# Path to local imports.
sys.path.append("/home/drew/He6CRES/rocks_analysis_pipeline/")
from results import ExperimentResults
from rocks_utility import he6cres_db_query

# Path to local imports.
sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")

# Local imports.
import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc
import he6_cres_spec_sims.experiment as exp

#read in Ne-19 data
ne_data = open("/media/heather/T7/Experiments/saved_experiments/ne19_full_0_aid_2/events.csv", "r")
df = pd.read_csv(dataFile, sep=',')

#read in He-6 data and filter bad files
dataFileh = open("/media/heather/T7/Experiments/saved_experiments/he6_full_0_aid_1/events.csv", "r")
dh = pd.read_csv(dataFileh, sep=',')
#filter bad files
dh["bad_file"] = False
dh.loc[(dh["run_id"] == 380) & (dh["file_id"]%3 == 0), "bad_file"] = True 
dh.loc[(dh["run_id"] == 381), "bad_file"] = True                             
dh = dh[dh["bad_file"] == False]

#Make a new df for those that satisfy cuts in start time, start freq, and SNR
#events = df[(df["EventStartFreq"] > 170e6) & (df["EventMeanSNR"]>8) & (df["set_field"] == 3)]
df_c = df[(df["EventStartFreq"] > 150e6)]
dh_c = dh[(dh["EventStartFreq"] > 150e6)]

df_c2 = df[(df["EventStartFreq"] > 150e6) & (df["EventMeanSNR"]>12)]
dh_c2 = dh[(dh["EventStartFreq"] > 150e6) & (dh["EventMeanSNR"]>12)]

# -------- Plotting ------------

fig_path = path + f"/hist_{feature}_{plot_index}.png"

grouped_ne = ne.events.groupby('set_field')
grouped_he = he.events.groupby('set_field')
rowlength = int(np.ceil(grouped.ngroups/3))                     # fix up if odd number of groups
fig, axs = plt.subplots(figsize=(16,8), 
                        nrows=3, ncols=rowlength,     # fix as above
                        gridspec_kw=dict(hspace=0.4)) # Much control of gridspec

targets = zip(grouped.groups.keys(),  axs.flatten())

for i, (key, ax) in enumerate(targets):

    ax.hist(grouped_ne.get_group(key)[feature], bins=bins, 
                   histtype=u'step', density=True, label = "Ne")
    ax.hist(grouped_he.get_group(key)[feature], bins=bins, 
                   histtype=u'step', density=True, label = "He")
    ax.set_title(f"set field: {key} T")
    ax.set_yscale(yscale)
    ax.legend()
    # ax.tight_layout()

fig_path = 
plt.savefig(fig_path_1, bbox_inches='tight', dpi = 300)
plt.show()