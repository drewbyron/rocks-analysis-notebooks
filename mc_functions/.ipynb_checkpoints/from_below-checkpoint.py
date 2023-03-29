# Author: Drew Byron
# Date: 3/28/23

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to local imports. Alter to match your machine. 
sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")

# Local imports.
import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc
import he6_cres_spec_sims.spec_tools.beta_source.beta_spectrum as bs

def build_canditate_events(freq_BW_visible, set_fields,  
                           isotope, cut_events = True, n = 1e3, slew_time = .035,
                           fix_slopes = True, rng_seed=12345) -> pd.DataFrame:
    
    # Make number of events to simulate an int. 
    n = int(n)
    
    # Initialize rng with given seed for repeatability
    rng = np.random.default_rng(rng_seed)
    
    # Feed the info dict to the BetaSpectrum class.
    bspec = bs.BetaSpectrum(isotope, b = 0 )
    Ws = np.linspace(1.00, bspec.W0, int(1e4))
    pdf = bspec.dNdE(Ws)
    
    # Sample n gammas from full pdf.
    gammas = np.random.choice(Ws, size=n, p=pdf/pdf.sum()) 
    
    # Generate expanded BW
    freq_BWs_generate = build_extended_freqBW(freq_BW_visible, set_fields, slew_time)
    
    energy_acceptances_high = sc.freq_to_energy(
        freq_BWs_generate[:,0], set_fields
    )
    energy_acceptances_low = sc.freq_to_energy(
        freq_BWs_generate[:,1], set_fields
    )
    energy_acceptances = np.stack((energy_acceptances_low, energy_acceptances_high), axis=-1)
    gamma_acceptances = sc.gamma(energy_acceptances)
    
    # Build events dict.
    events = {}
    events["set_field"] = []
    events["gamma_start"] = [] 

       
    # Figure out how to vectorize this if it has performance issues. Otherwise, fine. 
    for index, (gamma_acceptance, field) in enumerate(zip(gamma_acceptances,set_fields)): 

        gamma_field = gammas[np.logical_and(gammas>=gamma_acceptance[0], gammas<=gamma_acceptance[1])]
        field_field = np.array([field]*len(gamma_field))

        events["set_field"].append(field_field)
        events["gamma_start"].append(gamma_field)

    events["set_field"] = np.concatenate(events["set_field"])
    events["gamma_start"] = np.concatenate(events["gamma_start"])
    
    
    events = pd.DataFrame(events)
    
    # Select start times uniformly in the observation/slew window. 
    events["EventStartTime"] = rng.uniform(0, slew_time, size = len(events["set_field"]))
    events["EventStartFreq"] = sc.energy_to_freq(sc.energy(events["gamma_start"]), events["set_field"])
    events["energy_start"] = sc.energy(events["gamma_start"])

    events["EventEndTime"] = slew_time
    events["TimeLength"] = events["EventEndTime"] - events["EventStartTime"]
    
    # Trying to make slopes work faster.
    events = get_slope_info(events, freq_BW_visible, fix_slopes = fix_slopes)
    
    if cut_events: 
        events = cut_events_fn(events, freq_BW_visible )

    return events

def cut_events_fn(events, freq_BW_visible ):
    """This function is responsible for cutting the events off at the 
    top and bottom of the visible bandwidth. """
    
    events["EventEndFreq"] = np.minimum(events["EventEndFreq"], freq_BW_visible[1])
    
    events["EventEndTime"] = np.minimum(events["EventEndTime"],
                                        events["EventStartTime"] + (freq_BW_visible[1] -events["EventStartFreq"])/events["Slope"])
    
    cut_condition = (events["EventEndFreq"] < freq_BW_visible[0]) 
    events= events.copy()[~cut_condition]
    
    # Note that the order of the below two operations is crucial for it to work. 
    events["EventStartTime"] = np.maximum(events["EventStartTime"],
                            events["EventStartTime"] + 
                                          (freq_BW_visible[0] -events["EventStartFreq"])/events["Slope"])
    events["EventStartFreq"] = np.maximum(events["EventStartFreq"], freq_BW_visible[0])
    
    # Recalculate things now. Should I make different names??  
    events["TimeLength"] = events["EventEndTime"] - events["EventStartTime"]
    events["FreqLength"] = events["EventEndFreq"] - events["EventStartFreq"]
    
    return events
           
               
def build_extended_freqBW(freq_BW_visible, set_fields, slew_time): 
    """Uses the max slope within the visible BW to find window of possible 
    start frequencies that could plausibly make it to the visible BW. Note
    that this doesn't account for the changing slope w/ energy but should be 
    approx correct. """

    powers = sc.power_larmor(set_fields, freq_BW_visible.min())
    energys = sc.freq_to_energy(freq_BW_visible.min(), set_fields)
    slopes = sc.df_dt(energys, set_fields, powers)

    freq_BW_extenstion = slopes*slew_time
    
    
    freq_BWs = np.tile(freq_BW_visible, (len(set_fields), 1))
    freq_BWs[:,0] = freq_BWs[:,0] - freq_BW_extenstion
    
    # Add in a min frequency (otherwise could become negative)
    return np.clip(freq_BWs, 100e6, np.inf)


def get_slope_info(events, freq_BW_visible, fix_slopes): 
    
    if fix_slopes:
        # Fix slope to be the max slope within the visible BW. 
        power_larmor = sc.power_larmor(events["set_field"], freq_BW_visible.min())
        energy_fixed = sc.freq_to_energy(freq_BW_visible.min(), events["set_field"])
        events["power_start"] = power_larmor
        events["Slope"] = sc.df_dt(energy_fixed, events["set_field"], events["power_start"])
        
    else: 
        # Fix slope to be the slope it's born with (could be larger or smaller than the above). 
        # But generally larger. 
        power_larmor = sc.power_larmor_e(events["set_field"], events["energy_start"])

        events["power_start"] = power_larmor
        events["Slope"] = sc.df_dt(events["energy_start"], events["set_field"], events["power_start"])
    
    events["energy_end"] = events["energy_start"] - power_larmor*events["TimeLength"]*sc.J_TO_EV
    events["gamma_end"] = np.clip(sc.gamma(events["energy_end"]),1, np.inf)
    
    events["power_end"] = sc.power_larmor_e(events["set_field"], events["energy_end"])
    events["SlopeEnd"] = sc.df_dt(events["energy_end"], events["set_field"], events["power_end"])
    
    events["EventEndFreq"] = events["EventStartFreq"] + events["TimeLength"]* events["Slope"]
    
    events["FreqLength"] = events["EventEndFreq"] - events["EventStartFreq"]
    
    return events

def df_dt_naive(events_row): 
    """Not currently being used. Didn't work amazingly. Should redo at some point. """
    energy = events_row["energy_start"]
    power = sc.power_larmor_e(events_row["set_field"], energy)
    events_row["power_start"] = power
    energy -= power*events_row["TimeLength"]*sc.J_TO_EV
    power = sc.power_larmor_e(events_row["set_field"], energy)
    events_row["energy_end"] = energy
    events_row["power_end"] = power
    
    return events_row

def df_dt_naive_alt(events_row): 
    """Not currently being used. Didn't work amazingly. Should redo at some point. """
    dt = .005
    if events_row["TimeLength"]<=dt: 
        energy = events_row["energy_start"]
        power = sc.power_larmor_e(events_row["set_field"], energy)
        events_row["power_start"] = power
        energy -= power*events_row["TimeLength"]*sc.J_TO_EV
        power = sc.power_larmor_e(events_row["set_field"], energy)
        events_row["energy_end"] = energy
        events_row["power_end"] = power
        return events_row
    
    else: 
        ts = np.arange(0, events_row["TimeLength"], dt)

        energy = events_row["energy_start"]
        power = sc.power_larmor_e(events_row["set_field"], energy)
        events_row["power_start"] = power
        for t in ts: 

            energy -= power*dt*sc.J_TO_EV
            power = sc.power_larmor_e(events_row["set_field"], energy)

        events_row["energy_end"] = energy
        events_row["power_end"] = power

        return events_row
          

        
def viz_mc_events(ax, events, set_field,  color = "r"):

    events_sf = events.groupby(["set_field"]).get_group(set_field)

    for row_index, row in events_sf.iterrows():

        time_coor = np.array([row["EventStartTime"], row["EventEndTime"]])
        freq_coor = np.array([row["EventStartFreq"], row["EventEndFreq"]])

        ax.plot(
            time_coor,
            freq_coor,
            color=color,
            alpha=.5,
            marker="o",
            markerfacecolor="red",
            markersize=4,
        )

    ax.set_ylabel("Freq (Hz)")
    ax.set_xlabel("Time (s)")
    plt.title(f" {set_field} T.")

    return None


# Below are functions for building the spectra to be fit to our predicted spectra. 


def cut_df(events, cuts):

    events_cut = events.copy()

    for column, cut in cuts.items():
        events_cut = events_cut[
            (events_cut[column] >= cut[0]) & (events_cut[column] <= cut[1])
        ]

    return events_cut

def build_MC_spectra(events_ne, events_he, cuts, monitor_rate = 10**8, mon_pois = True): 
    
    # Simulate data that provides the "spectra" df for both ne and he.
    spectra_ne_exp = pd.DataFrame()
    spectra_he_exp = pd.DataFrame()
    
    ne_events_cut = cut_df(events_ne, cuts)
    he_events_cut = cut_df(events_he, cuts)

    spectra_ne_exp["set_field"] = ne_events_cut.set_field.unique()
    spectra_he_exp["set_field"] = he_events_cut.set_field.unique()


    spectra_ne_exp["event_count"] = ne_events_cut.groupby("set_field").EventStartFreq.count().values
    spectra_he_exp["event_count"] = he_events_cut.groupby("set_field").EventStartFreq.count().values

    spectra_ne_exp["tot_monitor_rate"] = monitor_rate
    spectra_he_exp["tot_monitor_rate"] = monitor_rate
    
    if mon_pois:
        # Apply a poisson statistic with the given mean for the event counts. 
        spectra_ne_exp["tot_monitor_rate"] = np.random.poisson(spectra_ne_exp["tot_monitor_rate"])
        spectra_he_exp["tot_monitor_rate"] = np.random.poisson(spectra_he_exp["tot_monitor_rate"])
    
    return spectra_ne_exp, spectra_he_exp


def build_MC_spectra_alt(events_ne, events_he, cuts, monitor_rate = 10**8, mon_pois = True): 
    
    # Simulate data that provides the "spectra" df for both ne and he.
    spectra_ne_exp = pd.DataFrame()
    spectra_he_exp = pd.DataFrame()
    
    ne_events_cut = cut_df(events_ne, cuts)
    he_events_cut = cut_df(events_he, cuts)

    spectra_ne_exp["set_field"] = ne_events_cut.set_field.unique()
    spectra_he_exp["set_field"] = he_events_cut.set_field.unique()


    spectra_ne_exp["event_count"] = ne_events_cut.groupby("set_field").EventStartFreq.count().values
    spectra_he_exp["event_count"] = he_events_cut.groupby("set_field").EventStartFreq.count().values
    
    spectra_ne_exp["event_count"]*= ne_events_cut.groupby("set_field").detection_prob.mean().values
    spectra_he_exp["event_count"]*= he_events_cut.groupby("set_field").detection_prob.mean().values
    
    
    spectra_ne_exp["tot_monitor_rate"] = monitor_rate
    spectra_he_exp["tot_monitor_rate"] = monitor_rate
    
    if mon_pois:
        # Apply a poisson statistic with the given mean for the event counts. 
        spectra_ne_exp["tot_monitor_rate"] = np.random.poisson(spectra_ne_exp["tot_monitor_rate"])
        spectra_he_exp["tot_monitor_rate"] = np.random.poisson(spectra_he_exp["tot_monitor_rate"])
    
    return spectra_ne_exp, spectra_he_exp