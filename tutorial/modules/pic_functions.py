#Final Functions used for the PIC project

#|----- Imports -----|
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from scipy.optimize import curve_fit
from nlb_tools.nwb_interface import NWBDataset

#|----- Dataset Functions -----|
def trial_datasetMaker(dataset, trials):
    '''
    Returns the dataset for only one trial as a DataFrame
    
    INPUT
    dataset: (pd.Dataframe) The pandas data frame from the original dataset
    trials: A one length array with the required trial
    '''
    if(len(trials) == 1):
        trials_possible = trials_present(dataset)
        check_mask = np.isin(trials, trials_possible)[0]
    
        if(check_mask):
            columns = dataset.columns
            matrix = np.asarray(dataset)
            
            trial_ids = np.array(dataset['trial_id'])
            mask = trial_ids == trials[0]
            matrix = matrix[mask]
            trial_dataset = pd.DataFrame(matrix, columns=columns)
            return trial_dataset
        else:
            print('Trial ', trials[0], ' not in the dataset')

def generate_sets(dataset, ratio, folds, unique=True, start_time='move_onset_time', end_time='reach_time', seed=2003):
    '''
    Divides the original training set into train/val/test sets with a fixed seed for reproducibility.

    INPUT
    dataset: (pd.dataframe) The pandas dataframe from the original dataset
    ratio: (float) The percentage of the train set is used for validation
    folds: (int) Number of different train and val sets generated with the given ratio

    OPTIONAL INPUT
    unique: (bool) If true guarantees that all validation sets have different trials 
    start_time: (string): The tag for the timestamp used for the start of the recordings used
    end_time: (string): The tag for the timestamp used for the end of the recordings used
    seed: (int) value to guarantee val/train break reproducibility\

    OUTPUT
    train_sets: (array of pd.dataframe) All train datasets grouped in a array
    val_sets: (array of pd.dataframe) All validation datasets grouped in a array
    test_sets: The dataset provided test set
    '''
    
    def generate_subset(dataset, subset_name, start_time='move_onset_time', end_time='reach_time'):
        mask = np.all(dataset.trial_info[['split']] == subset_name, axis=1)
        return dataset.make_trial_data(start_field=start_time, end_field=end_time, ignored_trials=~mask)
    
    # Set the random seed if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Data Unpacking
    ref_dataset = generate_subset(dataset, 'train')
    
    trials = trials_present(dataset.trial_info)
    selectable_trials = trials_present(ref_dataset)
    full_train_trials = selectable_trials
    val_length = int(len(trials_present(ref_dataset)) * 0.1)
    
    # Train and Val set making
    train_sets = []
    val_sets = []
    val_selectable_trials = selectable_trials
    for k in range(folds):
        val_trials = np.sort(np.random.choice(selectable_trials, val_length, replace=False))
        train_trials = np.setdiff1d(full_train_trials, val_trials)
        
        val_mask = np.isin(trials, val_trials)
        train_mask = np.isin(trials, train_trials)
        val_dataset = dataset.make_trial_data(start_field=start_time, end_field=end_time, ignored_trials=(~val_mask))
        train_dataset = dataset.make_trial_data(start_field=start_time, end_field=end_time, ignored_trials=(~train_mask))
        if unique:
            selectable_trials = train_trials
        val_sets.append(val_dataset)
        train_sets.append(train_dataset)
    
    # Test set making
    test_dataset = generate_subset(dataset, 'val')

    return train_sets, val_sets, test_dataset

def trials_present(dataset):
    '''
    Returns the unique trials present in one dataset
    '''
    trial_dataset = np.asarray(dataset['trial_id'], dtype='int64')
    return np.unique(trial_dataset)

def add_reachTime(dataset):
    '''
    Adds to the trial info dataset the time of the reach (also adds the target position)
    '''
    actual_dataset = dataset.make_trial_data(start_field='move_onset_time')
    
    cursor_pos = np.asarray(actual_dataset['cursor_pos'])
    trials_id = np.asarray(actual_dataset['trial_id'])
    mask = np.concatenate(([True], trials_id[1:] != trials_id[:-1]))
    targets = np.asarray(dataset.trial_info['target_pos'])
    active_id = np.asarray(dataset.trial_info['active_target'])
    target_pos = [target[active_id[i]] for i, target in enumerate(targets)]
    target_pos = np.array(target_pos, dtype='int64')
    started_time = np.array(actual_dataset['align_time'])
    move_time = np.array(dataset.trial_info['move_onset_time'])
    
    diffs = []
    for i in range(len(cursor_pos)):
        diffs += [np.sqrt(np.power(target_pos[trials_id[i]][0] - cursor_pos[i][0], 2)+
                          np.power(target_pos[trials_id[i]][1] - cursor_pos[i][1], 2))]
    
    diffs = np.split(diffs, np.where(mask)[0])[1:]
    started_time = np.split(started_time, np.where(mask)[0])[1:]
    reach_time = []
    
    for i in range(len(target_pos)):
        reach_time += [started_time[i][np.argmin(diffs[i])]+move_time[i]]
        
    dataset.trial_info['reach_time'] = reach_time
    dataset.trial_info['active_pos_x'] = target_pos[:,0]
    dataset.trial_info['active_pos_y'] = target_pos[:,1]

#|----- Processing Functions -----|
def firing_rate(dataset, window_size, bin_time=5, panda=True):
    '''
    Returns the firing rate of a dataset's spikes with a specific window size moving average

    INPUT
    dataset: (pd.dataframe) The dataset with spike data to be converted into firing rate
    window_size: (int) The lenght of the window used for the moving average

    OPTION INPUT
    bin_time: (float) Duration of each bin in ms
    panda: (bool) If true the original dataset is updated

    OUTPUT
    firing_matrix: (np.array) Matrix with the firing rate per entry and neuron
    trial_ids: (array) An array with the trial ID od each entry
    '''
    size = len(dataset)
    neurons_names = dataset['spikes'].columns
    nr_neurons = len(neurons_names)
    spikes_data = np.asarray(dataset['spikes'], dtype='float64')
    trial_ids = np.asarray(dataset['trial_id'], dtype='int64')
    firing_matrix = []
    trials = np.unique(trial_ids)
    kernel = (np.ones((window_size, 1)) / window_size)
    
    for trial in trials:
        mask = trial_ids == trial
        matrix = spikes_data[mask]
        convolved = (convolve2d(matrix, kernel, mode='same')/5)*1000
        firing_matrix.append(convolved)
        
    firing_matrix = np.concatenate(firing_matrix, axis=0)
    if(panda): 
        #dataset['spikes'] = pd.DataFrame(firing_matrix, columns=np.asarray(neurons_names, dtype='int'))
        dataset['spikes'] = firing_matrix
    else: return firing_matrix, trial_ids

def cross_pearson(array1, array2):
    '''
    Returns the cross-correlation with Pearson correlation values
    '''
    array1 = np.asarray(array1, dtype='float64')
    array2 = np.asarray(array2, dtype='float64')
    
    corr_p = np.zeros(len(array2) + len(array1) - 1)
    size = len(array2) - 1
    #left pad
    for index, i in enumerate(range(size, 0, -1)):
        x = np.pad(array2,(i), mode='constant')
        x = x[i*2:]
        corr_p[index] = np.corrcoef(array1,x)[0,1]
    #righ pad
    for index, i in enumerate(range(0, size+1)):
        index = index+size
        x = np.pad(array2,(i), mode='constant')
        x = x[:len(x)-i*2]
        if(i == 0):
            if(np.all(x == x[0])):
                noise = np.random.uniform(low=-0.0001, high=0.0001, size=x.shape)
                x = x + noise
        corr_p[index] = np.corrcoef(array1,x)[0,1]
    return corr_p

def maximaze_corrParameters(cross_corr, lags):
    '''
    Returns the max correlation value and the corresponding lag

    INPUT
    cross_corr: (np.array) Array with all correlation values of a trial
    lags: (np.array) Array with the lag values used to generate the cross_corr array

    OUTPUT
    max_corr: (float) Maximum correlation value
    max_corr_lag: (float) Lag of max correlation
    
    '''
    max_corr_index = np.argmax(cross_corr)
    min_corr_index = np.argmin(cross_corr)
    max_corr_lag = lags[max_corr_index]
    min_corr_lag = lags[min_corr_index]
    max_corr = round(cross_corr[max_corr_index],2)
    min_corr = round(cross_corr[min_corr_index],2)
    if(abs(max_corr)<abs(min_corr)): 
        max_corr = min_corr
        max_corr_lag = min_corr_lag
    return max_corr, max_corr_lag

def get_pearson(dataset, metric, extra_metrics=False):
    '''
    Computes the mean Pearson correlation of all trials (and standard deviation) at 0

    INPUT
    dataset: (pd.Dataframe) The dataset desired to get the correlation values
    metric: (array of strings) An array with the name of the true output column and with the name of the predicted output column
    
    OPTION INPUT
    extra_metrics: (bool): If true it will return all Pearson values per trial

    OUTPUT
    mean_pearson (float): The mean Pearson correlation over all trials
    std_pearson (float): The standard deviation of the Pearson correlation over all trials
    mean_r2 (float): The mean R2 correlation over all trials
    std_r2 (float): The standard deviation of the R2 correlation over all trials

    OPTION OUTPUT
    pearson_matrix (np.array): All Pearson correlation values of all trials
    '''
    #Data Extraction
    trial_ids = np.array(dataset['trial_id'])
    trials = np.unique(trial_ids)
    nr_trials = len(trials)
    color_array = np.round(np.linspace(0, 255, nr_trials))
    
    dataset_matrix = np.asarray(dataset)
    columns = np.array(dataset.columns)
    pred_index = next(i for i, col in enumerate(columns) if col[0] == metric[1])
    obs_index = next(i for i, col in enumerate(columns) if col[0] == metric[0])
    
    #Get the trial_datasets
    trial_datasets = np.asarray([dataset_matrix[trial_ids == trial] for trial in trials], dtype='object')
    sep_preds = [trial[:, pred_index] for trial in trial_datasets]
    sep_obs = [trial[:, obs_index] for trial in trial_datasets]
    
    #Initializes the arrays
    pearson_matrix = []
    
    #Computes the crosss and lags
    for i in range(nr_trials): 
        obs = np.array(sep_obs[i], dtype='float64')
        pred = np.array(sep_preds[i], dtype='float64')
        if(np.all(pred == pred[0])):
            noise = np.random.uniform(low=-0.0001, high=0.0001, size=pred.shape)
            pred = pred + noise
            print(f'-> Noise added to trial {trials[i]}')
        pearson = np.corrcoef(obs, pred)[0,1]
        if(np.isnan(pearson).any()): 
            print(f'Trial {trials[i]} as NaN Pearson')
            print(f'Observed has Nan? {np.isnan(obs).any()}')
            print(obs)
            print(f'Predicted has Nan? {np.isnan(pred).any()}')
            print(pred)
        pearson_matrix.append(pearson)
    pearson_matrix = np.asarray(pearson_matrix, dtype='float64')
    
    #Gets the mean and the standard deviation
    r2 = np.power(pearson_matrix, 2)
    mean_pearson = np.mean(pearson_matrix)
    std_pearson = np.std(pearson_matrix)
    mean_r2 = np.mean(r2)
    std_r2 = np.std(r2)

    if(extra_metrics): return mean_pearson, std_pearson, mean_r2, std_r2, pearson_matrix
    else: return mean_pearson, std_pearson, mean_r2, std_r2

def linear_regression_model(rates, vel, loss='sse', lambaRidge=2):
    '''
    Linear regression with a given loss function and given regularization that returns the weights

    INPUT
    rates: (np.array) Training input 
    vel: (np.array) Training output

    OPTIONAL INPUT
    loss: (string) Loss function to be used
        - 'sse' (default): Sum of Square Errors
        - 'ridge': Sum of Square Errors with Ridge Regularization
    lambaRidge: (float) Value used for the Ridge Regularization

    OUTPUT
    w: (np.array) Predicted weights
    '''
    
    #Loss Function Selection
    if(loss == 'sse'):
        rates = np.insert(rates, 0, 1, axis=1)
        penrose = np.linalg.pinv(rates.astype(np.float32))
        w = np.matmul(penrose,vel)
        predicted = np.matmul(rates,w)
    elif(loss == 'ridge'):
        rates = np.insert(rates, 0, 1, axis=1)
        ratesT = np.transpose(rates)
        prod = np.matmul(ratesT,rates)
        identity = np.identity(len(prod))
        helper = prod + lambaRidge * identity
        inverse = np.linalg.inv(helper)
        penrose = np.matmul(inverse,ratesT)
        w = np.matmul(penrose,vel)
        predicted = np.matmul(rates,w)

    return w

#|----- NB Functions -----|

def nb_remove_neurons(datasets, neurons, predicted, observed, grids, method='depth', threshold=None, remove_fromDataset=True):
    '''
    Removes the neurons according to a certain threshold for the NB decoder

    INPUT
    datasets: (np.array) Array of datasets to be updated (train, val and test)
    neurons: (array) List of all neuron names
    predicted: (np.array) matrix with all the fitted tuning surface parameters (amplitude, std_x, std_y)
    observed: (np.array) matrix with all the observed tuning surface values
    grids: (np.array) array with the array grid_x and grid_y used for the matrix appearance of the surfaces
    method: (string) Mode to choose what removing method to apply to the neuron listing
        - 'depth' (default): removes neurons which have tuning surfaces with max-min distances smaller than the threshold value
        - 'std': removes neurons which have tuning surfaces with standard deviations bigger than the threshold value
        - 'diff': removes neurons which have predicted tuning surfaces with a distance to the observed curve bigger than the threshold value
    

    OPTION INPUT
    threshold: (float) Value used for the thresholding
    remove_fromDataset: (bool) If true it will update the dataset by removing the neurons
    
    OUTPUT
    acepted_neurons: (np.array) List of neurons which pass the threshold
    acepted_predicted: (np.array) Updated predicted values (removed neurons are not present)
    
    '''
    #Finding the neurons
    if(method == 'depth'):
        amplitude = predicted[:,0]
        mask = np.where(amplitude >= threshold, True, False)
    elif(method == 'std'):
        x_std = np.array(predicted[:,3])
        y_std = np.array(predicted[:,4])
        mean_std = (x_std + y_std)/2
        mask = np.where(mean_std <= threshold, True, False)
    elif(method == 'diff'):
        # Convert grids to the appropriate format
        grids = np.array(grids)
        x_grid = grids[0]
        y_grid = grids[1]
        
        # Initialize an array to store differences
        difference_matrix = np.zeros((len(observed), x_grid.shape[0], y_grid.shape[1]))
        mask = ~np.isnan(observed[0])
        
        # Loop over all observed instances
        for i in range(len(observed)):
            # Calculate the predicted Gaussian surface for the entire grid
            pred = ex_twoD_Gaussian2((x_grid, y_grid), *predicted[i])
        
            # Calculate the difference where observed is not NaN
            difference_matrix[i][mask] = pred[mask] - observed[i][mask]
        
        # Flatten the differences for each instance
        difference_matrix = difference_matrix.reshape(len(observed), -1)
        distance_matrix = np.sqrt(np.sum(np.power(difference_matrix, 2), axis=1))

        mask = np.where(distance_matrix <= threshold, True, False)

    #Removing the neurons
    neurons = np.array(neurons)
    removed_neurons = neurons[~mask]
    acepted_neurons = neurons[mask]
    acepted_predicted = predicted[mask]
    
    if(remove_fromDataset):
        for j, dataset in enumerate(datasets):
            col_toRemove = np.empty((len(removed_neurons)), dtype=tuple)
            for i, neuron in enumerate(removed_neurons):
                col_toRemove[i] = ('spikes', neuron)
            dataset.drop(columns=col_toRemove, inplace=True)
    return acepted_neurons, acepted_predicted

def get_tune_distance(grids, observed, predicted):
    '''
    Gets the distance between the predicted and observed tunning surfaces

    INPUT
    predicted: (np.array) matrix with all the fitted tuning surface parameters (amplitude, std_x, std_y)
    observed: (np.array) matrix with all the observed tuning surface values
    grids: (np.array) array with the array grid_x and grid_y used for the matrix appearance of the surfaces

    OUTPUT
    distance_matrix: (np.array) Returns the euclidean distance of every neurons curves
    '''
    grids = np.array(grids)
    x_grid = grids[0]
    y_grid = grids[1]
    
    # Initialize an array to store differences
    difference_matrix = np.zeros((len(observed), x_grid.shape[0], y_grid.shape[1]))
    mask = ~np.isnan(observed[0])
    
    # Loop over all observed instances
    for i in range(len(observed)):
        # Calculate the predicted Gaussian surface for the entire grid
        pred = ex_twoD_Gaussian2((x_grid, y_grid), *predicted[i])
    
        # Calculate the difference where observed is not NaN
        difference_matrix[i][mask] = pred[mask] - observed[i][mask]
    
    # Flatten the differences for each instance
    difference_matrix = difference_matrix.reshape(len(observed), -1)
    distance_matrix = np.sqrt(np.sum(np.power(difference_matrix, 2), axis=1))

    return distance_matrix

#|----- Plotting Functions ----|

def raster_plot(dataset, trial, neurons=None, align=None, ratio=6, save=False, save_name='raster.png'):
    """
    Generates a raster plot for a given trial from the dataset.
    
    Parameters:
        dataset (object): The dataset containing trial information and spike data.
        neurons (list, optional): List of neurons to include in the raster plot. If None, all neurons are included. Default is None.
        trial (int, optional): The trial number to plot. Default is 0.
        align (tuple, optional): Alignment range for the trial data. If None, no alignment is applied. Default is None.
        ratio (int, optional): Ratio for the figure size. Default is 6.
        save (bool, optional): Whether to save the plot as an image file. Default is False.
        save_name (str, optional): The name of the file to save the plot. Default is 'raster.png'.
    """

    mask = np.all(dataset.trial_info[['trial_id']] == trial, axis=1)

    if align is None: 
        trial_data = dataset.make_trial_data(ignored_trials=(~mask))
    else:
        trial_data = dataset.make_trial_data(start_field='move_onset_time', end_field='reach_time', align_range=align, ignored_trials=(~mask))

    if neurons is None:
        neurons = trial_data['spikes'].columns.tolist()  # Gets the name of all neurons
    
    neuron_spikes = []
    for neuron in neurons:
        neuron_spikes.append(trial_data['spikes'][neuron])

    # Initialize a list to store spike times for each neuron
    spike_times = [[] for _ in range(len(neuron_spikes))]
    
    # Iterate through the spike data and extract spike times for each neuron
    for neuron_idx, neuron in enumerate(neuron_spikes):
        for time_idx, spike in enumerate(neuron):
            if spike == 1:
                spike_times[neuron_idx].append(time_idx * 5)  # Multiply by 5 to convert bins to ms

    #Get the color
    color_array = np.round(np.linspace(0, 255, 5))
    color_template = 'turbo'
    
    # Gets the tot timestamp in milliseconds
    tot_str = dataset.trial_info.loc[trial, 'target_on_time']
    gct_str = dataset.trial_info.loc[trial, 'go_cue_time']
    mot_str = dataset.trial_info.loc[trial, 'move_onset_time']
    rt_str = dataset.trial_info.loc[trial, 'reach_time']
    tot = pd.to_timedelta(tot_str).total_seconds() * 1000  # Convert to milliseconds
    gct = pd.to_timedelta(gct_str).total_seconds() * 1000  # Convert to milliseconds
    mot = pd.to_timedelta(mot_str).total_seconds() * 1000  # Convert to milliseconds
    rt = pd.to_timedelta(rt_str).total_seconds() * 1000  # Convert to milliseconds
    
    fig, ax = plt.subplots(figsize=(1*ratio, 3/4*ratio))
    for i, spikes in enumerate(spike_times, start=1):
        ax.eventplot(spikes, colors='k', lineoffsets=i, linelengths=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neurons')
    ax.set_title(f'Raster Plot of Trial {trial}')
    
    # Set x-ticks to show time in ms
    num_bins = trial_data['spikes'].shape[0]  # Total number of bins
    max_time_ms = num_bins * 5  # Convert total bins to ms
    ax.set_xlim(0, max_time_ms)
    xticks = np.arange(0, max_time_ms + 1, 500)  # Set x-ticks every 500 ms
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    
    # Add a vertical line at the 'tot' timestamp
    ax.axvline(tot, color=plt.colormaps.get_cmap(color_template)(int(color_array[0])), linestyle='--', linewidth=1, 
               label=f'Target On Time: {tot/1000:.3f} sec')
    ax.axvline(gct, color=plt.colormaps.get_cmap(color_template)(int(color_array[1])), linestyle='--', linewidth=1, 
               label=f'Go Cue Time: {gct/1000:.3f} sec')
    ax.axvline(mot, color=plt.colormaps.get_cmap(color_template)(int(color_array[2])), linestyle='--', linewidth=1, 
               label=f'Move Onset Time: {mot/1000:.3f} sec')
    ax.axvline(rt, color=plt.colormaps.get_cmap(color_template)(int(color_array[3])), linestyle='--', linewidth=1, 
               label=f'Reach Time: {rt/1000:.3f} sec')
    ax.legend()

    if save:
        plt.savefig(save_name)
    plt.show()

def plot_allTrajectoriesS(dataset, original_dataset, nr_cols=4, save=False, save_name='plot.png'):
    '''
    Plots all trajectories separate (with trial duration in ms)

    INPUT
    dataset: (pd.dataframe) The dataset to be used for every time entry
    original_dataset: (pd.dataframe) The dataset to be used for every trial information

    OPTION INPUT
    nr_cols: (int) The desired number of columns to be plotted. The number of rows will come by consequence
    save: (bool) If True, the plot will be stored in the directory with the save_name name
    save_name: (string) The desired file name
    
    '''
    #Gets the hand positions (trua and pred)
    cursor_pos = np.asarray(dataset['cursor_pos'], dtype='float64')
    true_pos = np.asarray(dataset['hand_pos'], dtype='float64')
    predicted_x = np.asarray(dataset['pred_X'], dtype='float64')
    predicted_y = np.asarray(dataset['pred_Y'], dtype='float64')
    true_x = true_pos[:,0]
    true_y = true_pos[:,1]
    cursor_x = cursor_pos[:,0]
    cursor_y = cursor_pos[:,1]
    
    #Gets the trials and time
    trial_ids = np.asarray(dataset['trial_id'], dtype='int64')
    mask = np.concatenate(([True], trial_ids[1:] != trial_ids[:-1]))
    split_indices = np.where(mask)[0]
    align_time = np.asarray(dataset['align_time'])
    
    #Gets the target positions per trial
    trials = np.unique(trial_ids)
    target_pos = np.asarray(original_dataset.trial_info[['active_pos_x', 'active_pos_y']], dtype='int64')[trials]
    barrier_pos = np.asarray(original_dataset.trial_info['barrier_pos'])[trials]
    barrier_lengths = np.array([len(inner_array) for inner_array in barrier_pos])
    
    
    #Splits all arrays acording to the trials
    predicted_x = np.split(predicted_x, split_indices)[1:]
    predicted_y = np.split(predicted_y, split_indices)[1:]
    align_time = np.split(align_time, split_indices)[1:]
    true_x = np.split(true_x, split_indices)[1:]
    true_y = np.split(true_y, split_indices)[1:]
    cursor_x = np.split(cursor_x, split_indices)[1:]
    cursor_y = np.split(cursor_y, split_indices)[1:]
    
    nr_trials = len(trials)
    #The ploting starts
    if(nr_trials%nr_cols == 0): nr_rows = nr_trials//nr_cols
    else: nr_rows = nr_trials//nr_cols + 1
    
    fig, axs = plt.subplots(nrows=nr_rows, ncols=nr_cols, figsize=(nr_cols*4, nr_rows*4))
    axs = axs.flatten()
    
    #Plotting
    for i in range(nr_trials):
        ax = axs[i]
        x_coords = true_x[i]
        y_coords = true_y[i]
        x_pred = predicted_x[i]
        y_pred = predicted_y[i]
        x_cursor = cursor_x[i]
        y_cursor = cursor_y[i]
        target_x = target_pos[i,0]
        target_y = target_pos[i,1]
        duration = int(align_time[i][-1])/1000000000
        distance = round(np.sqrt(np.power(x_pred[-1]-x_coords[-1],2) + np.power(y_pred[-1]-y_coords[-1],2)),3)
        barriers = barrier_pos[i]
    
        #Gets the limit
        x_lim = np.max([np.max(x_coords), np.max(x_pred), np.max(x_cursor), np.abs(np.min(x_coords)), np.abs(np.min(x_pred)), 
                        np.abs(np.min(x_cursor))])
        y_lim = np.max([np.max(y_coords), np.max(y_pred), np.max(y_cursor), np.abs(np.min(y_coords)), np.abs(np.min(y_pred)), 
                        np.abs(np.min(y_cursor))])
        lim_value = np.max([x_lim,y_lim])+15
        
        #Plots the barriers
        for j in range(barrier_lengths[i]):
            x, y, half_width, half_height = barrier_pos[i][j]
            left = x - half_width
            bottom = y - half_height
            width = 2 * half_width
            height = 2 * half_height
            ax.add_patch(plt.Rectangle((left, bottom), width, height, facecolor='gray'))
    
        ax.plot(x_cursor, y_cursor, label='Cursor Path')
        ax.plot(x_pred, y_pred, label='Predicted Path')
        ax.plot(x_coords, y_coords, label='True Path')
        ax.scatter(target_x, target_y, label='Target Position', color='red')
        ax.scatter(x_coords[0], y_coords[0], label='Start Point', color='black')
        ax.set_title(f'Trial {trials[i]}\nLasted {duration} seconds\nDistance Final: {distance}')
        ax.set_xlim(-lim_value, lim_value)
        ax.set_ylim(-lim_value, lim_value)
        ax.legend()
        ax.grid(True)
    
    plt.suptitle(f'Trial Trajectories', y=1, fontsize=20)
    plt.tight_layout()
    if(save): plt.savefig(save_name)
    plt.show()

def plot_ccTogether(dataset, trials, metric, window_size=150, correlation='pearson', bin_size=5, focus_trial=None, compare_axis=True, 
                    color_template='Greys', save=False, save_name='all_corrs.png'):
    '''
    Plots the correlation for each trial in a dataset in the same plot.
    
    INPUT
    dataset: (pandas.Dataframe) The required dataset with at least two columns named and with the specified metrics
    trials: (array_like) The list of trials to be plotted (must be present in the dataset)
    metric: (array_like of strings) The list with the column name of the actual metric and the predicted metric
    
    OPTION INPUT
    window_size: (integer) The size of the visualization window in the x-axis (-window_size+1, window_size)
    correlation: (string) Define the correlation metric applied
        - 'cross' (default)
            Uses the normalized cross-correlation
        - 'pearson'
            Uses the Pearson correlation
    bin_size: (float) Specifies what is the size of each bin in ms
    focus_trial: {None, int, string} Defines a plotted line that will receive emphasis
        - ''None'' (default)
            No trial receives emphasis
        - ''int''
            The specified trial receives emphasis (needs to be in the trials array)
        - 'max'
            The trial with a higher correlation receives an emphasis
        - 'min'
            The trial with a lower correlation receives an emphasis
        - 'best'
            The trial with the best correlation receives emphasis (positive or negative)
        - 'worst'
            The trial with the worst correlation receives emphasis (closer to 0)
        - 'best_lag'
            The trial with the lowest lag for the highest correlation receives emphasis (closer to 0)
        - 'worst_lag'
            The trial with the highest lag for the highest correlation receives emphasis (positive or negative)
        - 'mean'
            A new plot line with the mean and the standard error shaded will receive emphasis with its maximum correlation 
            for a given lag pointed out
    compare_axis: (bool) If true the y-axis will have the full correlation scope (-1 - margin, 1 + margin) 
    color_template: (string) Defines the plt.colormaps used for the plot lines colour (uses the full spectrum)
    save: (bool) If true the plot will be saved in the same folder
    save_name: (string) If save true will define the file name and type

    OUTPUT
    corr_matrix: (np.array) matrix with all correlation and lags for each trial
        
    '''
    #Data Extraction
    nr_trials = len(trials)
    color_array = np.round(np.linspace(0, 255, nr_trials))
    bin_window = window_size//bin_size
    dataset_matrix = np.asarray(dataset)
    columns = np.array(dataset.columns)
    pred_index = next(i for i, col in enumerate(columns) if col[0] == metric[1])
    obs_index = next(i for i, col in enumerate(columns) if col[0] == metric[0])
    trial_ids = np.array(dataset['trial_id'])
    trials_possible = trials_present(dataset)
    lags = np.arange(-window_size + 1, window_size, bin_size)
    
    #Check if any trial is not there
    check_mask = np.isin(trials, trials_possible)
    if(not check_mask.all()):
        print('Some trial not in the dataset')
    
    #Initializes the arrays
    corr_matrix = []
    max_matrix = []
    max_correlations = []
    max_correlationsAbs = []
    max_lags = []
    max_lagsAbs = []
    
    #Get the trial_datasets
    trial_datasets = np.asarray([dataset_matrix[trial_ids == trial] for trial in trials], dtype='object')
    sep_preds = [trial[:, pred_index] for trial in trial_datasets]
    sep_obs = [trial[:, obs_index] for trial in trial_datasets]
    
    #Computes the cross and lags
    for i in range(nr_trials): 
        obs = np.array(sep_obs[i], dtype='float64')
        pred = np.array(sep_preds[i], dtype='float64')
        if(correlation == 'pearson'):
            cross_corr = cross_pearson(obs, pred)
        elif(correlation == 'cross'):
            k = np.sqrt((np.sum(np.power(obs,2)))*(np.sum(np.power(pred,2))))
            cross_corr = np.correlate(obs, pred, mode='full')/k
        cross_corr = cross_corr[len(cross_corr)//2-bin_window:len(cross_corr)//2+bin_window]
    
        #Stores the information
        max_corr, max_lag = maximaze_corrParameters(cross_corr, lags)
        max_correlations += [max_corr]
        max_correlationsAbs += [np.abs(max_corr)]
        max_lags += [max_corr]
        max_lagsAbs += [np.abs(max_lag)]
        corr_matrix += [[cross_corr, lags]]
        max_matrix += [[max_corr, max_lag]]
    corr_matrix = np.asarray(corr_matrix, dtype='float64')
    
    #Defines the focus_trial
    if(focus_trial == 'max'):
        focus_trial = trials[np.argmax(max_correlations)]
    elif(focus_trial == 'min'):
        focus_trial = trials[np.argmin(max_correlations)]
    elif(focus_trial == 'best'):
        focus_trial = trials[np.argmax(max_correlationsAbs)]
    elif(focus_trial == 'worst'):
        focus_trial = trials[np.argmin(max_correlationsAbs)]
    elif(focus_trial == 'best_lag'):
        focus_trial = trials[np.argmin(max_lagsAbs)]
    elif(focus_trial == 'worst_lag'):
        focus_trial = trials[np.argmax(max_lagsAbs)]
    
    #Plot function
    plt.figure(figsize=(15,10))
    if(nr_trials > 30): no_label = True
    else: no_label = False
    
    for i in range(nr_trials):
        cross_corr = corr_matrix[i][0]
        lags = corr_matrix[i][1]
        max_corr = max_matrix[i][0]
        max_lag = max_matrix[i][1]
        if(focus_trial == None):
            if(not no_label): plt.plot(lags, cross_corr, color=plt.colormaps.get_cmap(color_template)(int(color_array[i])), 
                                       label=f'Trial {trials[i]}', alpha=1)
            else: plt.plot(lags, cross_corr, color=plt.colormaps.get_cmap(color_template)(int(color_array[i])), alpha=1)
        elif(focus_trial == trials[i]):
            plt.plot(lags, cross_corr, color=plt.colormaps.get_cmap(color_template)(int(color_array[i])), label=f'Trial {trials[i]}', 
                     alpha=1)
            plt.scatter(max_lag, max_corr, color='red', label=f'Corr: {round(max_corr,2)}\nLag: {max_lag} ms')
        else:
            if(not no_label): plt.plot(lags, cross_corr, color=plt.colormaps.get_cmap(color_template)(int(color_array[i])), 
                                       label=f'Trial {trials[i]}', alpha=0.25)
            else: plt.plot(lags, cross_corr, color=plt.colormaps.get_cmap(color_template)(int(color_array[i])), alpha=0.1)

    #In case the focus trial is the mean, its the last to be applied
    if(focus_trial == 'mean'):
        mean_vector = np.mean(corr_matrix, axis=0)
        ste_vector = np.std(corr_matrix, axis=0)
        ste_values = ste_vector[0] / np.sqrt(nr_trials)
        cross_corr = mean_vector[0]
        lags = mean_vector[1]
        max_corr, max_lag = maximaze_corrParameters(cross_corr, lags)
        plt.plot(lags, cross_corr, color='black', label='Mean Correlation', alpha=1)
        plt.fill_between(lags, cross_corr-ste_values, cross_corr+ste_values, color='black', label='Standard Error', alpha=0.25)
        plt.scatter(max_lag, max_corr, color='red', label=f'Corr: {round(max_corr,2)}\nLag: {max_lag} ms')
        plt.scatter(0, max_corr, color='red', label=f'Corr: {round(max_corr,2)}\nLag: {max_lag} ms')
    #Labeling and Formating
    plt.xlabel('Lag (ms)')
    if(correlation == 'cross'):
        plt.ylabel('Cross-Correlation')
        plt.title(f'Cross-Correlation of All Trials ({metric[0]})')
    if(correlation == 'pearson'):
        plt.ylabel('Pearson\'s Correlation')
        plt.title(f'Pearson\'s Correlation of All Trials ({metric[0]})')
    if(compare_axis): plt.ylim(-1.1,1.1)
    plt.grid(True)
    if(no_label): plt.legend(loc='lower left')
    if(save): plt.savefig(save_name)
    plt.show()
    
    return corr_matrix

def compare_models(parameters, all_entries, model_names, eval='Pearson', title_name='title', label_name='label', 
                   save=False, save_name='plot.png'):
    '''
    Visualizes the different models of Pearson's performance

    INPUT
    parameters: (np.array) Matrix with all models performance (mean and std pearson and R2 values)
    all_entries: (np.array) Matrix with all the trials per model performance (just used to get the number of trials
    model_names: (np.array) With all the model names (used for labelling)

    OPTIONAL INPUT
    eval: (string) Choses the evaluation metric used ('Pearson' or 'R2')
    title_name: (string) Used for giving a title to the plot
    label_name: (string) Used for providing a label to the plot's x-axis
    save: (bool) If true the plot will be saved in the same folder
    save_name: (string) If save true will define the file name and type
    '''
    # Data extraction
    nr_trials = len(all_entries[0, 0])
    if(eval == 'Pearson'):
        x_std = parameters[:, 0, 1]
        y_std = parameters[:, 1, 1]
        x_mean = parameters[:, 0, 0]
        y_mean = parameters[:, 1, 0]
        label_x = 'Mean Pearson for Mx'
        label_y = 'Mean Pearson for My'
    elif(eval == 'R2'):
        x_std = parameters[:, 0, 3]
        y_std = parameters[:, 1, 3]
        x_mean = parameters[:, 0, 2]
        y_mean = parameters[:, 1, 2]
        label_x = 'Mean R2 for Mx'
        label_y = 'Mean R2 for My'
    
    results = {
        label_x: x_mean,
        label_y: y_mean
    }
    cov_results = {
        'Coefficient of Variation for X': x_std / x_mean,
        'Coefficient of Variation for Y': y_std / y_mean,
        'Mean Coefficient of Variation': np.mean([x_std / x_mean, y_std / y_mean], axis=0)
    }
    
    # Extra Variables
    max_mean_x = np.max(x_mean)
    max_mean_y = np.max(y_mean)
    max_x_model = np.argmax(x_mean)
    max_y_model = np.argmax(y_mean)
    deviation = np.array([x_std, y_std])
    error = deviation / np.sqrt(nr_trials)
    
    #Begin Plotting
    fig, ax1 = plt.subplots(figsize=(15, 5), layout='constrained')
    ax2 = ax1.twinx()
    model_x = np.arange(len(model_names))  # the label locations
    width = 0.25  # the width of the bars
    colors = ['tomato', 'dodgerblue', 'tomato', 'dodgerblue', 'black']
    
    # Plot CoV on the right y-axis
    for i, (attribute, measurement) in enumerate(cov_results.items()):
        offset = width * (i + 1)
        alpha_value = 0.15 if i != 2 else 0.3
        ax2.plot(model_x + offset, measurement, color=colors[i + 2], alpha=alpha_value, label=attribute)
    
    # Plot Pearson values on the left y-axis
    for i, (attribute, measurement) in enumerate(results.items()):
        offset = width * (i + 1)
        ax1.errorbar(model_x + offset, measurement, error[i], linewidth=width * 5, ls='none', capsize=5, label=attribute, color=colors[i])
    
    # Highlight preferred model
    mean_cov = np.mean([x_std / x_mean, y_std / y_mean], axis=0)
    min_cov_index = np.argmin(mean_cov)
    t1_x = model_x[min_cov_index]
    t2_x = t1_x + offset + width
    ax1.vlines(t1_x, 0, 1, color='grey', linestyle='--', linewidth=0.5)
    ax1.vlines(t2_x, 0, 1, color='grey', linestyle='--', linewidth=0.5)
    ax1.fill_betweenx([0, 1], t1_x, t2_x, color='yellow', alpha=0.3, 
                      label=f'Best Parameter: {model_names[min_cov_index]} (x: {round(x_mean[min_cov_index],2)} y: {round(y_mean[min_cov_index],2)})')
    
    # Add labels, title, and custom x-axis tick labels for the left y-axis
    ax1.set_ylabel(eval)
    ax1.set_xlabel(label_name)
    ax1.set_title(title_name)
    ax1.set_xticks(model_x + width)
    ax1.set_xticklabels(np.array(model_names, dtype=int), fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left', ncols=1)
    ax1.grid(True)
    
    # Add label for the right y-axis
    ax2.set_ylabel('Coefficient of Variation')
    ax2.legend(loc='upper right', ncols=1)
    
    # Save figure if required
    if save: plt.savefig(save_name)
    
    # Show plot
    plt.show()


#|----- Warnings ----|
#To remove warnings from the dataset import

import warnings
warnings.filterwarnings('ignore')