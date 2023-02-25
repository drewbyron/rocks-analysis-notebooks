import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from datetime import datetime
import numpy as np
from scipy import stats
import uproot4

from pathlib import Path


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


# ------ Make a plot with all the noise floors ------

root_file_dir_ne = Path("/media/drew/T7 Shield/rocks_analysis/saved_experiments/new_event_features_test_aid_7/root_files")
root_file_dir_he = Path("/media/drew/T7 Shield/rocks_analysis/saved_experiments/he_noise_floors_test_aid_10/root_files")

# root_file_dir_ne = Path(root_file_dir)

root_files_ne = list(root_file_dir_ne.glob("*.root"))
root_files_he = list(root_file_dir_he.glob("*.root"))
noise_floors_ne = []
noise_floors_he = []

for root_file in root_files_ne:
    noise_floors_ne.append(construct_GV_array(root_file, gainvar_num_spectra=50000))

for root_file in root_files_he:
    noise_floors_he.append(construct_GV_array(root_file, gainvar_num_spectra=50000))

    # Also want to extract the time and add that to the plot

freqs = np.linspace(0, 1200, len(noise_floors_ne[0]))


# ---- Make plots ------
base_path = "/media/drew/T7 Shield/rocks_analysis/paper_02_figures/noise_plots/noise_floors_over_time"
fig_path = base_path + "/ne_noise_floors.png"

fig, axs = plt.subplots(1, figsize=(12, 6))

for i, noise_floor in enumerate(noise_floors_ne):
    if i == 0:
    	plt.plot(freqs, noise_floor, color = "b", label = "Ne19")
    else: 
    	plt.plot(freqs, noise_floor, color = "b")
    # else:
    #     plt.plot(freqs, noise_floor)
for i, noise_floor in enumerate(noise_floors_he):
    if i == 0:
    	plt.plot(freqs, noise_floor, color = "r", label = "He6")
    else:
    	plt.plot(freqs, noise_floor, color = "r")	
plt.title(
    f"Neon and Helium Noise Floors (one per run_id)."
)
# plt.title(
#     f"Helium noise floors ({len(noise_floors_ne)} run_ids, spanning 24 hrs. 8/17 - 8/18/2022)"
# )
plt.legend()
plt.xlabel("Frequency (MHz)")
plt.ylabel("(arb.)")
plt.savefig(fig_path, bbox_inches="tight", dpi=200)

plt.show()
