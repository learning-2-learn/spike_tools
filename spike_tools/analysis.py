import numpy as np 
import pandas as pd 
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt 
from scipy import ndimage

def firing_rate(spData, channelData, bins, smoothing, trials=None):
    bin_size = np.abs(np.diff(bins)[0])
    # spData is pandas dataframe with at least TrialNumber, UnitId, and SpikeTimeFromStart columns
    if trials is None: 
        # spData is pandas dataframe with at least IntervalID, UnitId, and SpikeTimeFromStart columns
        trial_unit_index = pd.MultiIndex.from_product([np.unique(spData.TrialNumber), np.unique(channelData.UnitID).astype(int), bins[:-1]], names=["TrialNumber", "UnitID", "TimeBins"]).to_frame()
    else:
        trial_unit_index = pd.MultiIndex.from_product([trials, np.unique(channelData.UnitID).astype(int), bins[:-1]], names=["TrialNumber", "UnitID", "TimeBins"]).to_frame()
    trial_unit_index = trial_unit_index.droplevel(2).drop(columns=["TrialNumber", "UnitID"]).reset_index()
    
    groupedData = spData.groupby(["TrialNumber", "UnitID"])

    fr_DF = groupedData.apply(lambda x: pd.DataFrame(\
                            {"SpikeCounts": np.histogram(x.SpikeTimeFromStart/1000, bins)[0],\
                             "FiringRate": gaussian_filter1d(np.histogram(x.SpikeTimeFromStart/1000, bins)[0].astype(float)/bin_size, smoothing),\
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

# unit_data is trials x timepoints x neurons (units)
def run_pca_across_neurons(unit_data, labels, folds=10, on_all_trials=False):
    if on_all_trials == True:
        unit_data_flat = np.concatenate(unit_data)
        pca = PCA()
        pcs = pca.fit_transform(unit_data_flat)
        return (pcs, pca.components_, np.cumsum(pca.explained_variance_ratio_))
    
    if folds > unit_data.shape[0]:
        raise AssertionError("Cannot divide " + str(unit_data.shape[0]) + " trials into " + str(folds) + " folds")
    
    unit_data_flat = np.concatenate(unit_data)
    labels_time = np.tile(np.array(labels).reshape((-1, 1)), unit_data.shape[1])
    labels_flat = np.concatenate(labels_time)
    #print(unit_data_flat.shape, labels_flat.shape)
    
    kf = KFold(n_splits=folds)
    
    transformed_pcs = np.zeros(shape=unit_data.shape)
    variance_all = np.zeros(shape=(folds, unit_data.shape[-1]))
    pc_axes_all = np.zeros(shape=(folds, unit_data.shape[-1], unit_data.shape[-1]))
    fold = 0
    for train_idx, test_idx in kf.split(unit_data):
        training_data = unit_data[train_idx, :, :]
        test_data = unit_data[test_idx, :, :]
        
        training_labels = labels[train_idx]
        test_labels = labels[test_idx]
        
        pca = PCA()
        pcs_train_flat = pca.fit_transform(np.concatenate(training_data, axis=0))
        pcs_train = pcs_train_flat.reshape(training_data.shape)
        pcs_test_flat = pca.transform(np.concatenate(test_data, axis=0))
        pcs_test = pcs_test_flat.reshape(test_data.shape)
        
        transformed_pcs[test_idx, :, :] = pcs_test 
        variance_all[fold, :] = np.cumsum(pca.explained_variance_ratio_)
        pc_axes_all[fold, :, :] = pca.components_
        fold = fold+1
        print("Fold ", fold, "... Done")
    return (transformed_pcs, pc_axes_all, variance_all)

# variance_ratios_all_fold is folds x number of components
def plot_variance_explained(variance_ratios_all_folds, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if variance_ratios_all_folds.ndim == 1:
        ax.plot(np.arange(1, variance_ratios_all_folds.size + 1), variance_ratios_all_folds, marker='.')
    else:
        ax.errorbar(np.arange(1, variance_ratios_all_folds.shape[1]+1), np.mean(variance_ratios_all_folds, axis=0),\
                    yerr=np.std(variance_ratios_all_folds, axis=0), marker='.')
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained")
    if show == True:
        plt.show()
    return fig

# pc_axes is components x neurons, unit_data is trials x timepoints x neurons
def plot_pc_projections(pc_axes, num_components, unit_data, plot_axes=[0, 1, 2], show=True):
    pc_space = pc_axes[plot_axes, :].reshape((len(plot_axes), -1))

    fig = plt.figure()
    if len(plot_axes) == 3:
        ax = fig.add_subplot(projection='3d')
    elif len(plot_axes) == 1 or len(plot_axes) == 2:
        ax = fig.add_subplot(111)
    else:
        raise ValueError("Cannot plot in more than 3 dimensions")
        
    for trial in unit_data:
        pc_transformed = pc_space @ trial.T
        if len(plot_axes) == 3:
            ax.plot(pc_transformed[0, :], pc_transformed[1, :], pc_transformed[2, :])
        elif len(plot_axes) == 2:
            ax.plot(pc_transformed[0, :], pc_transformed[1, :])
        elif len(plot_axes) == 1:
            ax.plot(np.arange(0, pc_transformed[0,:].size), pc_transformed[0, :])        
         
    if show == True:
        plt.show()
    return fig
    