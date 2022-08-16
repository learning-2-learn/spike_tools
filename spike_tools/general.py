import numpy as np 
import pandas as pd 
import h5py 
import s3fs
import os
import time 

## S3 Paths
DATAJOINT_BUCKET = "/l2l.datajoint.data/" 
GENERATED_DATA = DATAJOINT_BUCKET + "processed/"

HUMAN_LFP_DIR = 'human-lfp'
NHP_LFP_DIR = 'nhp-lfp'
NHP_WCST_DIR = 'nhp-lfp/wcst-preprocessed/'

## DataJoint URLs
DATAJOINT_URL = "http://u19-db.cch9uqmmvxno.us-west-2.rds.amazonaws.com/"

def get_subject_session_string(subject, session):
    return "sub-" + subject + "_sess-" + str(session)

def get_spike_path(subject, session):
    return os.path.join(NHP_WCST_DIR, "rawdata", "sub-" + str(subject), "sess-" + str(session), "spikes/")

def get_behavior_path(subject, session):
    return os.path.join(NHP_WCST_DIR, "rawdata", "sub-" + str(subject), "sess-" + str(session), "behavior", "sub-" + str(subject) + "_sess-" + str(session) + "_object_features.csv")

def get_eye_path(subject, session):
    return os.path.join(NHP_WCST_DIR, "rawdata", "sub-" + str(subject), "sess-" + str(session),
                        "eye")

def get_channels_path(subject, session):
    return os.path.join(NHP_WCST_DIR, "rawdata", "sub-" + str(subject), "sess-" + str(session), "channellocations",\
            "sub-" + str(subject) + "_sess-" + str(session) + "_channellocations.csv") 

def list_session_units(fs, subject, session):
    spike_path = get_spike_path(subject, session)
    unit_files = fs.glob(spike_path + "*spiketimes.mat")
    channels = [x.split("chan-")[-1].split("_")[0] for x in unit_files]
    units = [int(x.split("unit-")[-1].split("_")[0]) for x in unit_files]
    unit_ids = list(range(0, len(units)))
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
    spike_data = get_spike_times(fs, subject, session, channel, unit).sort_values('SpikeTime')
    trial_spikes = trial_data.apply(lambda x: spike_data[spike_data.SpikeTime.searchsorted(x[start_field], side="left"):\
                                    spike_data.SpikeTime.searchsorted(x[end_field], side="right")].assign(TrialNumber = [x.TrialNumber] *\
                                    (spike_data.SpikeTime.searchsorted(x[end_field], side="right")-spike_data.SpikeTime.searchsorted(x[start_field], side="left"))), axis=1)
    trial_spikes = pd.concat(trial_spikes.tolist())
    spikes_by_trial = trial_spikes.merge(trial_data, on="TrialNumber", how="left")
    spikes_by_trial["SpikeTimeFromStart"] = spikes_by_trial.SpikeTime - spikes_by_trial[start_field]
    return spikes_by_trial

"""
def get_spike_times_by_interval(fs, subject, session, start_time, end_time, channel=None, unit=None):
    spike_data = get_spike_times(fs, subject, session, channel, unit).sort_values('SpikeTime')
    interval_spikes = spike_data[spike_data.SpikeTime.searchsorted(start_time, side="left"):\
                                    spike_data.SpikeTime.searchsorted(end_time, side="right")].assign(TrialNumber = [x.TrialNumber] *\
                                    (spike_data.SpikeTime.searchsorted(end_time, side="right")-spike_data.SpikeTime.searchsorted(start_time, side="left"))), axis=1)
    trial_spikes = pd.concat(trial_spikes.tolist())
    spikes_by_trial = trial_spikes.merge(trial_data, on="TrialNumber", how="left")
    spikes_by_trial["SpikeTimeFromStart"] = spikes_by_trial.SpikeTime - start_time
    return spikes_by_trial
"""