import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np
from scipy import stats
import uproot4


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


def construct_GV_array(rootfile_path, gainvar_num_spectra):
    # Open rootfile.
    rootfile = uproot4.open(rootfile_path)
    GV_array = rootfile["histGV_{}_0;1".format(gainvar_num_spectra)]._bases[1]._data
    return GV_array


he_root_file = "/media/drew/T7 Shield/rocks_analysis/saved_experiments/he6_full_0_aid_1/root_files/Freq_data_2022-08-17-20-29-17_001.root"
ne_root_file = "/media/drew/T7 Shield/rocks_analysis/saved_experiments/ne19_full_0_aid_2/root_files/Freq_data_2022-10-05-19-27-55_002.root"

he_noise = construct_GV_array(he_root_file, gainvar_num_spectra = 50000)
ne_noise = construct_GV_array(ne_root_file, gainvar_num_spectra = 50000)

base_path = "/media/drew/T7 Shield/rocks_analysis/paper_02_figures/noise_plots"
fig_path = base_path + "/noise_floor_experiment.png"

fig, axs = plt.subplots(1, figsize=(12, 6))
plt.plot(ne_noise, label = "ne")
plt.plot(he_noise, label = "he")
plt.title("Noise floors during experiment")
plt.legend()
plt.savefig(fig_path, bbox_inches="tight", dpi=200)

plt.show()

