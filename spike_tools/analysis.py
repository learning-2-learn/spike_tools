import numpy as np 
import pandas as pd 
from scipy.ndimage import gaussian_filter1d

def firing_rate(spData, channelData, bins, smoothing):
    # spData is pandas dataframe with at least TrialNumber, UnitId, and SpikeTimeFromStart columns
    trial_unit_index = pd.MultiIndex.from_product([np.unique(spData.TrialNumber), np.unique(channelData.UnitID).astype(int), bins[:-1]], names=["TrialNumber", "UnitID", "TimeBins"]).to_frame()
    trial_unit_index = trial_unit_index.droplevel(2).drop(columns=["TrialNumber", "UnitID"]).reset_index()
    
    groupedData = spData.groupby(["TrialNumber", "UnitID"])

    fr_DF = groupedData.apply(lambda x: pd.DataFrame(\
                            {"SpikeCounts": np.histogram(x.SpikeTimeFromStart/1000, bins)[0],\
                             "FiringRate": gaussian_filter1d(np.histogram(x.SpikeTimeFromStart/1000, bins)[0].astype(float), smoothing),\
                             "TimeBins": bins[:-1]}))
    #print("Trial", np.unique(trial_unit_index.UnitID))
    #print("FR", np.unique(fr_DF.droplevel(2).reset_index().UnitID))
    all_units_df = trial_unit_index.merge(fr_DF.droplevel(2).reset_index(), how='outer', on=["TrialNumber", "UnitID", "TimeBins"])
    #for unit in np.unique(all_units_df.UnitID):
    #    unit_df = all_units_df[all_units_df.UnitID == unit]
    #    print(unit_df)
    #    print(unit, len(unit_df))
    all_units_df.FiringRate = all_units_df.FiringRate.fillna(0.0)
    all_units_df.SpikeCounts = all_units_df.SpikeCounts.fillna(0)
    return all_units_df
