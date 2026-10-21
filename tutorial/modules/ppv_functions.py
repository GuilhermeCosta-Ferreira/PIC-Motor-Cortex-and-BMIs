#FUNCTIONS USED FOR THE PPV MODEL

#|---------- Import ----------|
import numpy as np
import matplotlib.pyplot as plt

from modules.pic_functions import *

#|---------- Main Functions ----------|
def tune_neurons(dataset, decoder, nr_directions):
    '''
    Produces the tune curve data for a set number of angles for the observed firing rate and the predicted firing rate

    INPUT
    dataset: (pd.Dataframe) Required dataset
    decoder: (np.array) Matrix with all the parameters obtained from a PPV decoder
    nr_directions: (int) Number of desired bins used for the angle analysis

    OUTPUT
    mean_matrix: (np.array) Matrix with the mean firing rate of each neuron (row) for each different angle (column)
    prediced_matrix: (np.array) Matrix with all the fitted firing rates of each neuron (row) for each different angle (column)
    target_angles: (np.array) List of all binned angle values
    '''
    rate_data = np.asarray(dataset['spikes'], dtype='float64')
    angle_data = np.asarray(dataset['angle'], dtype='float64')
    target_angles = np.linspace(-np.pi, np.pi, nr_directions, endpoint=False)
    
    closest_angle_idxs = np.argmin(np.abs(target_angles[:, np.newaxis] - angle_data), axis=0)
    angle_data = target_angles[closest_angle_idxs]
    
    #Gets the observed curve
    mean_matrix = []
    for target_angle in (target_angles):
        mask = (angle_data == target_angle)
        relevant_rates = rate_data[mask]
        mean_rate = np.mean(relevant_rates, axis=0)
        mean_matrix.append(mean_rate)
    mean_matrix = np.asarray(mean_matrix, dtype='float64')

    #Gets the predicted curve
    bias = decoder[:, 3]
    k = decoder[:, 5]
    angle = decoder[:, 4]
    prediced_matrix = bias[:, np.newaxis] + k[:, np.newaxis] * np.cos(target_angles - angle[:, np.newaxis])
    prediced_matrix = np.asarray(prediced_matrix, dtype='float64')
    return mean_matrix, prediced_matrix, target_angles
    
def angle_decoder(dataset, window_size, neurons, bin_time=5):
    '''
    Builds decoder by fitting the firing rate to cosines. This function does not support delay

    INPUT
    dataset: (pd.Dataframe) Required dataset
    window_size: (int) The lenght of the window used for the moving average

    OPTION INPUT
    bin_time: (float) Duration of each bin in ms

    OUTPUT
    decoder: (np.array) Matrix like array with all the neurons (rows) parameters (columns) for decoding
    '''
    #Change spikes to firing rates in ms (moving average)
    rate_data, trials_vector = firing_rate(dataset, window_size, bin_time=bin_time, panda=False)
    train_size = len(rate_data)
    nr_neurons = len(neurons)
    
    #Initiates the vectors for calculating the vector M
    hand_pos = np.asarray(dataset['hand_pos'], dtype='float64')
    cosines_matrix = np.zeros((train_size, 2))
    angle = np.zeros(train_size)
    
    #Adds the column vectorM with the cosines components to the train_dataset
    for i in range(train_size):
        if (i == 0 or ((trials_vector[i] != trials_vector[i-1]))):
            preX = 0
            preY = 0
        else:
            preX = hand_pos[i-1,0]
            preY = hand_pos[i-1,1]
        x = hand_pos[i,0]
        y = hand_pos[i,1]
        angle_temp = np.arctan2((y - preY), (x - preX))
        cosX = np.cos(angle_temp)
        cosY = np.sin(angle_temp)
        cosines_matrix[i,0] = cosX
        cosines_matrix[i,1] = cosY
        angle[i] = angle_temp
    
    #Trains the decoder fiting the data to a cos
    decoder = []
    for i in range(nr_neurons):
        neuron_data = rate_data[:,i]
        weights = linear_regression_model(cosines_matrix, neuron_data, loss='ridge')
        k = np.sqrt(np.power(weights[1],2) + np.power(weights[2],2))
        if(k == 0): 
            print(f'neuron {neurons[i]} did not fire')
            cx = 1
            cy = 0
        else:
            cx = weights[1]/k
            cy = weights[2]/k
        angleC = np.arctan2(cy, cx)
        decoder.append([neurons[i], cx, cy, weights[0], angleC, k])
    dataset['spikes'] = rate_data
    dataset['angle'] = angle
    return np.array(decoder)

def remove_neurons(predicted_matrix, observed_matrix, dataset, decoder, neurons, threshold, metric='depth', remove_fromDataset=True):
    '''
    Removes the neurons from the PPV decoder and prepares the val_dataset for predictions

    INPUT
    predicted_matrix: (np.array) matrix with all the fitted tuning curve values
    observed_matrix: (np.array) matrix with all the observed tuning curve values
    dataset: (pd.Dataframe) dataset used
    decoder: (np.array) Matrix-like array with all the decoding values (bs, C, and preferred angle)
    neurons: (array) List of all neuron names
    threshold: (float) Value used for the thresholding

    OPTION INPUT
    metric: (string) Mode to choose what removing method to apply to the neuron listing
        - 'depth' (default): removes neurons which have tuning curves with max-min distances smaller than the threshold value
        - 'difference': removes neurons which have predicted tuning curves with a distance to the observed curve bigger than the threshold value
    remove_fromDataset: (bool) If true it will update the dataset by removing the neurons
    
    OUTPUT
    new_decoder: (np.array) Updated version of the decoder
    acepted_neurons: (np.array) List of neurons which pass the threshold
    
    '''
    #Gets the values from the metric used (depth)
    if(metric == 'depth'):
        max_vector = np.max(predicted_matrix, axis=1)
        min_vector = np.min(predicted_matrix, axis=1)
        depth_vector = max_vector-min_vector
        acepted_neurons = np.where(depth_vector>=threshold, neurons, None)
        rejected_neurons = np.where(depth_vector<threshold, neurons, None)
    elif(metric == 'difference'):
        difference_vector = np.sqrt(np.sum(np.power(predicted_matrix - observed_matrix.T, 2), axis=1))
        acepted_neurons = np.where(difference_vector<=threshold, neurons, None)
        rejected_neurons = np.where(difference_vector>threshold, neurons, None)
    
    #Builds arrays with the rejected and acepted neurons
    acepted_neurons = np.asarray(np.ma.masked_equal(acepted_neurons, None).compressed(), dtype='int64')
    
    #Removes from the decoder
    new_decoder = decoder[rejected_neurons == None]
    
    #Removes those columns from the dataset
    rejected_neurons = np.asarray(np.ma.masked_equal(rejected_neurons, None).compressed(), dtype='int64')
    if(remove_fromDataset):
        col_toRemove = np.empty((len(rejected_neurons)), dtype=tuple)
        for i, neuron in enumerate(rejected_neurons):
            col_toRemove[i] = ('spikes', neuron)
        dataset.drop(columns=col_toRemove, inplace=True)
    return new_decoder, acepted_neurons


#|---------- Ploting Functions ----------|
def plot_tuning(dataset, nr_directions, neurons, full_neurons, decoder=None, nr_cols=5, ratio=4, threshold=0, save=False, save_name='plot.png'):
    '''
    Plots the tuning curves

    INPUT
    dataset: (pd.Dataframe) dataset used
    nr_directions: (int) Number of desired bins used for the angle analysis
    neurons: (np.array) List of all neuron names present
    full_neurons: (np.array) List of all neurons without removal

    OPTIONAL INPUT
    decoder: (np.array) Matrix with all the parameters obtained from a PPV decoder
    nr_cols: (int) The desired number of columns to be plotted. The number of rows will come by consequence
    ratio: (int) Ratio to define the size of the plot
    threshold: (float) Threshold value used fro the neuron removal (visualize removed neurons)
    save: (bool) If True, the plot will be stored in the directory with the save_name name
    save_name: (string) The desired file name
    '''
    mean_matrix, predicted_matrix, target_angles = tune_neurons(dataset, decoder, nr_directions)
    max_rate = np.max([np.max(mean_matrix), np.max(predicted_matrix)])+1
    if(len(mean_matrix[0])%nr_cols == 0): nr_rows = len(neurons)//nr_cols
    else: nr_rows = len(neurons)//nr_cols + 1
    
    # Create a figure and subplots
    fig, axs = plt.subplots(nrows=nr_rows, ncols=nr_cols, figsize=(nr_cols*ratio, nr_rows*ratio))
    
    # Flatten the axs array for easier iteration
    axs = axs.flatten()
    
    # Plot each array
    for i, neuron in enumerate(neurons):
        ax = axs[i]
        observed = mean_matrix[:,neuron]
        predicted = predicted_matrix[neuron]
        max_angle = target_angles[np.argmax(predicted)]
        height = round(np.max(predicted_matrix[neuron])-np.min(predicted_matrix[neuron]),3)
        if(height > threshold):
            ax.plot(target_angles, observed, label='Observed Tuning Curve')
            if(i==1):ax.plot(target_angles, predicted, label=f'Fitted Tuning Curve\nPreferred Angle: {np.round(max_angle,2)} rads')
            if(i==0):ax.set_title(f'Tuning curve of neuron {full_neurons[neuron]}', fontsize=10)
            else:ax.set_title(f'Fitted tuning curve of neuron {full_neurons[neuron]}', fontsize=10)
        else: 
            ax.plot(target_angles, observed, label='Observed Tuning Curve', alpha=0.3)
            ax.plot(target_angles, predicted, label=f'Predicted Tuning Curve\nPreferred Angle: {np.round(max_angle,2)} rads', alpha=0.3)
            ax.set_title(f'Tuning curve of neuron {full_neurons[neuron]}', fontsize=10)
        ax.set_xlabel('Angle (rads)')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_ylim(0, max_rate)
        ax.legend(loc='lower left')
        ax.grid(True)
    
    for ax in fig.get_axes():
        ax.label_outer()
        
    #plt.suptitle(f'Tuning Curves\n with {nr_directions} angles and threshold at {threshold}', y=1, fontsize=15)
    plt.tight_layout()
    if(save): plt.savefig(save_name)
    print((nr_cols*ratio, nr_rows*ratio))
    plt.show()