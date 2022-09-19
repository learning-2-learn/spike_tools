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

def get_spike_path(subject, session, species_dir=NHP_WCST_DIR):
    return os.path.join(NHP_WCST_DIR, "rawdata", "sub-" + str(subject), "sess-" + str(session), "spikes/")

def get_behavior_path(subject, session, species_dir=NHP_WCST_DIR):
    return os.path.join(NHP_WCST_DIR, "rawdata", "sub-" + str(subject), "sess-" + str(session), "behavior", "sub-" + str(subject) + "_sess-" + str(session) + "_object_features.csv")

def get_eye_path(subject, session, species_dir=NHP_WCST_DIR):
    return os.path.join(NHP_WCST_DIR, "rawdata", "sub-" + str(subject), "sess-" + str(session),
                        "eye")

def get_channels_path(subject, session, species_dir=NHP_WCST_DIR):
    return os.path.join(NHP_WCST_DIR, "rawdata", "sub-" + str(subject), "sess-" + str(session), "channellocations",\
            "sub-" + str(subject) + "_sess-" + str(session) + "_channellocations.csv") 

def get_channels_list(fs, subject, session, region="all"):
    channel_file = get_channels_path(subject, session)
    channel_data = pd.read_csv(fs.open(channel_file))
    all_channels = channel_data['1'].to_numpy()
    if region.lower() == "all":
        return all_channels
    elif region.lower() == "hippocampus":
        hippocampus_channels = np.array([x for x in all_channels if x.find('a') < 0])
        return hippocampus_channels
    elif region.lower() == "pfc":
        pfc_channels = np.array([x for x in all_channels if x.find('a') >= 0])
        return pfc_channels
    
def list_session_units(fs, subject, session):
    spike_path = get_spike_path(subject, session)
    unit_files = fs.glob(spike_path + "*spiketimes.mat")
    channels = [x.split("chan-")[-1].split("_")[0] for x in unit_files]
    units = [int(x.split("unit-")[-1].split("_")[0]) for x in unit_files]
    unit_ids = list(range(0, len(units)))
    unit_info = pd.DataFrame(np.array([unit_ids, channels, units, unit_files]).T, columns=["UnitID", "Channel", "Unit", "SpikeTimesFile"])
    return unit_info

def get_spike_times(fs, subject, session, channels=[], units=[]):
    all_unit_info = list_session_units(fs, subject, session)
    
    if len(channels) == 0:
        filter_rows = all_unit_info
    elif len(channels) > 0 and len(units) == 0:
        filter_rows = all_unit_info[all_unit_info.Channel.isin(channels)]
    elif len(channels) > 0 and len(units) > 0:
        filter_rows = all_unit_info[(all_unit_info.Channel.isin(channels)) & (all_unit_info.Unit.isin(units))]
    spike_times = filter_rows.apply(lambda x: np.squeeze(h5py.File(fs.open(x.SpikeTimesFile)).get('timestamps')).astype(int), axis=1).explode()
    spike_times_df = pd.DataFrame(spike_times, index=spike_times.index.values, columns=["SpikeTime"])
    spike_times_df["UnitID"] = spike_times_df.index.values
    return spike_times_df

def get_spike_times_by_trial(fs, subject, session, trials=[], channels=[], units=[], start_field="TrialStart", end_field="TrialEnd", pre_start=0, post_end=0):
    trial_file = get_behavior_path(subject, session)
    trial_data = pd.read_csv(fs.open(trial_file)) 
    if len(trials) > 0:   
        trial_data = trial_data[trial_data.TrialNumber.isin(trials)] 
    trial_data.set_index("TrialNumber")
    spike_data = get_spike_times(fs, subject, session, channels, units).sort_values('SpikeTime')
    trial_spikes = trial_data.apply(lambda x: spike_data[spike_data.SpikeTime.searchsorted(x[start_field] - pre_start, side="left"):\
                                    spike_data.SpikeTime.searchsorted(x[end_field] + post_end, side="right")].assign(TrialNumber = [x.TrialNumber] *\
                                    (spike_data.SpikeTime.searchsorted(x[end_field] + post_end, side="right")-spike_data.SpikeTime.searchsorted(x[start_field] - pre_start, side="left"))), axis=1)
    trial_spikes = pd.concat(trial_spikes.tolist())
    spikes_by_trial = trial_spikes.merge(trial_data, on="TrialNumber", how="left")
    spikes_by_trial["SpikeTimeFromStart"] = spikes_by_trial.SpikeTime - spikes_by_trial[start_field]
    return spikes_by_trial

# def get_spike_times_by_interval(fs, subject, session, start_time, end_time, channel=None, unit=None):
#     trial_file = get_behavior_path(subject, session)
#     trial_data = pd.read_csv(fs.open(trial_file)) 
#     if len(trials) > 0:   
#         trial_data = trial_data[trial_data.TrialNumber.isin(trials)] 
#     trial_data.set_index("TrialNumber")
#     spike_data = get_spike_times(fs, subject, session, channel, unit).sort_values('SpikeTime')
#     interval_spikes = trial_data.apply(lambda x: spike_data[spike_data.SpikeTime.searchsorted(start_time, side="left"):\
#                                     spike_data.SpikeTime.searchsorted(end_time, side="right")].assign(TrialNumber = [x.TrialNumber] *\
#                                     (spike_data.SpikeTime.searchsorted(end_time, side="right")-spike_data.SpikeTime.searchsorted(start_time, side="left"))), axis=1)
#     trial_spikes = pd.concat(trial_spikes.tolist())
#     spikes_by_trial = trial_spikes.merge(trial_data, on="TrialNumber", how="left")
#     spikes_by_trial["SpikeTimeFromStart"] = spikes_by_trial.SpikeTime - start_time
#     return spikes_by_trial
