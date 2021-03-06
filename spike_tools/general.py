import numpy as np 
import pandas as pd 
import h5py 
import s3fs
import os
import time 

HUMAN_LFP_DIR = 'human-lfp'
NHP_DIR = 'nhp-lfp'
NHP_WCST_DIR = 'nhp-lfp/wcst-preprocessed/'

def get_spike_path(subject, session):
    return os.path.join(NHP_WCST_DIR, "rawdata", "sub-" + str(subject), "sess-" + str(session), "ephys/")

def get_behavior_path(subject, session):
    return os.path.join(NHP_WCST_DIR, "rawdata", "sub-" + str(subject), "sess-" + str(session), "behavior", "sub-" + str(subject) + "_sess-" + str(session) + "_object_features.csv")

def get_channels_path(subject, session):
    return os.path.join(NHP_WCST_DIR, "rawdata", "sub-" + str(subject), "sess-" + str(session), "channellocations",\
            "sub-" + str(subject) + "_sess-" + str(session) + "_channellocations.csv") 

def list_session_units(fs, subject, session):
    spike_path = get_spike_path(subject, session)
    unit_files = fs.glob(spike_path + "*spiketimes.mat")
    channels = [x.split("chan-")[-1].split("_")[0] for x in unit_files]
    units = [int(x.split("unit-")[-1].split("_")[0]) for x in unit_files]
    unit_ids = list(range(1, len(units)+1))
    unit_info = pd.DataFrame(np.array([unit_ids, channels, units, unit_files]).T, columns=["UnitID", "Channel", "Unit", "SpikeTimesFile"])
    return unit_info

def get_spike_times(fs, subject, session, channel=None, unit=None):
    all_unit_info = list_session_units(fs, subject, session)

    if channel == None:
        filter_rows = all_unit_info
    elif channel != None and unit == None:
        filter_rows = all_unit_info[all_unit_info.Channel == str(channel)]
    elif channel != None and unit != None:
        filter_rows = all_unit_info[(all_unit_info.Channel == str(channel)) & (all_unit_info.Unit == str(unit))]
    
    spike_times = filter_rows.apply(lambda x: np.squeeze(h5py.File(fs.open(x.SpikeTimesFile)).get('timestamps')).astype(int), axis=1).explode()
    spike_times_df = pd.DataFrame(spike_times, index=spike_times.index.values, columns=["SpikeTime"])
    spike_times_df["UnitID"] = spike_times_df.index.values
    return spike_times_df

def get_spike_times_by_trial(fs, subject, session, trial=[], channel=None, unit=None, start_field="TrialStart", end_field="TrialEnd"):
    trial_file = get_behavior_path(subject, session)
    trial_data = pd.read_csv(fs.open(trial_file)) 
    if len(trial) > 0:   
        trial_data = trial_data[trial_data.TrialNumber.isin(trial)] 
    trial_data.set_index("TrialNumber")

    #startTime = time.time()
    spike_data = get_spike_times(fs, subject, session, channel, unit).sort_values('SpikeTime')
    #endTime = time.time()
    #print("Getting Spike Data ", endTime-startTime)
    
    #startTime = time.time()
    trial_spikes = trial_data.apply(lambda x: spike_data[spike_data.SpikeTime.searchsorted(x[start_field], side="left"):\
                                    spike_data.SpikeTime.searchsorted(x[end_field], side="right")].assign(TrialNumber = [x.TrialNumber] *\
                                    (spike_data.SpikeTime.searchsorted(x[end_field], side="right")-spike_data.SpikeTime.searchsorted(x[start_field], side="left"))), axis=1)
    trial_spikes = pd.concat(trial_spikes.tolist())
    #endTime = time.time()
    #print("Filtering spike data: ", endTime-startTime)
    
    #startTime = time.time()
    spikes_by_trial = trial_spikes.merge(trial_data, on="TrialNumber", how="left")
    spikes_by_trial["SpikeTimeFromStart"] = spikes_by_trial.SpikeTime - spikes_by_trial[start_field]
    #endTime = time.time()
    #print("Merging trial and spike data: ", endTime-startTime)
    return spikes_by_trial
