# Author: Drew Byron
# Date: 4/10/23
# Description: Module containing functions specific to an analysis that
# contains different SNR thresholds in katydid.
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Local imports.
import analysis_functions.ratio_experiment as re


def process_snr_study(
    snr_study,
    event_cuts,
    snrs,
    normed_cols=["mMeanSNR", "EventTimeLength", "uniform_rand"],
):

    ne_spectra = []
    he_spectra = []
    ratios = []

    for snr in snrs:

        ne = snr_study["ne"][snr]
        ne_spectrum = re.build_spectrum(
            ne.events.copy(deep=True), ne.root_files.copy(deep=True), event_cuts, normed_cols=normed_cols
        )
        ne_spectra.append(ne_spectrum)

        he = snr_study["he"][snr]
        he_spectrum = re.build_spectrum(
            he.events.copy(deep=True), he.root_files.copy(deep=True), event_cuts, normed_cols=normed_cols
        )
        he_spectra.append(he_spectrum)

        ratios.append(re.build_ratio(ne_spectrum, he_spectrum))

    snr_study_results = {
        "ne_spectrum": dict(zip(snrs, ne_spectra)),
        "he_spectrum": dict(zip(snrs, he_spectra)),
        "ratio": dict(zip(snrs, ratios)),
    }
    return snr_study_results

def combine_ratios(snr_study, event_cuts, snrs, normed_cols):
    
    snr_study_results = process_snr_study(snr_study, event_cuts, snrs, normed_cols=normed_cols)
    # snr_study_results = snr_tests.process_snr_study(snr_study, event_cuts,snrs, normed_cols=normed_cols)
    print(snr_study_results["ratio"][9])
    # print(snr_study_results["ratio"][9])
    
    Ratio_all = np.array([snr_study_results["ratio"][snr]["Ratio"] for snr in snrs])
    sRatio_all = np.array([snr_study_results["ratio"][snr]["sRatio"] for snr in snrs])
    
    ratio  = Ratio_all.mean(axis = 0)
    ratio_sys = Ratio_all.max(axis = 0) - Ratio_all.min(axis = 0)
    ratio_stat = sRatio_all.mean(axis = 0)
    
    print(ratio_sys)
    print(ratio_stat)
    ratio_combined = pd.DataFrame()
    ratio_combined.index = snr_study_results["ratio"][9].index
    
    ratio_combined["Ratio"] = ratio
    ratio_combined["sRatio"] = (ratio_sys**2 + ratio_stat**2)**.5
    
    # print(ratio_combined)
    
    return ratio_combined