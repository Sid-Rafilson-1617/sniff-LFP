"""
snifflocklfp library

This library contains functions for reading in local field potential and sniff data, preprocessing signals, and further analysis including aligning neural data to inhalation times.

By: Sid Rafilson
Orginal MATLAB functions written by: Nate Hess

"""

import math
import numpy as np
from scipy import signal
from scipy import stats
import matplotlib.pyplot as plt


def load_sniff(sniff_file: str, num_samples: int, nchannels = 8, ch = 8) -> np.array:
    '''
    Load a specific channel from a binary sniff data file into a NumPy array.

    This function reads a binary file containing multi-channel sniff data, processes it, 
    and extracts data from a specified channel.

    Parameters:
    sniff_file (str): The path to the binary file containing sniff data.
    num_samples (int): The number of samples to read from the file.
    nchannels (int, optional): The number of channels in the sniff data. Defaults to 8.
    ch (int, optional): The channel number to extract. Defaults to 8. Channels are
                        indexed starting from 1.

    Returns:
    np.array: A NumPy array containing the data from the specified channel.

    Notes:
    - The number of samples specified is dynamic for debugging but can easily fit the entire data set with num_samples = -1.
    - Channel numbers start from 1. For instance, ch = 1 will extract the first channel.
    '''
    
    # reading in binary data
    num_samples = num_samples * nchannels
    sniff_bin = np.fromfile(sniff_file, dtype=np.uint16, count=num_samples)

    # ensuring equal samples from each channel
    num_complete_sets = len(sniff_bin) // nchannels
    sniff_bin = sniff_bin[:num_complete_sets * nchannels]

    # reshaping data and extracting channel which corresponds to sniff voltage
    sniff_bin = np.reshape(sniff_bin, (nchannels, -1), order = 'F')
    ch_index = ch - 1
    sniff = sniff_bin[ch_index, :]
    return sniff

    
def load_ephys(ephys_file: str, num_samples: int, nchannels = 16) -> np.array:
    '''
    Load and reshape binary electrophysiology data into a NumPy array.

    This function is designed to read binary files containing electrophysiology 
    (ephys) data. It loads the specified number of samples from the file and 
    reshapes them into a 2D NumPy array, where each row represents a channel.

    Parameters:
    ephys_file (str): Path to the binary file containing electrophysiology data.
    num_samples (int): Number of samples to read from the file.
    nchannels (int, optional): Number of channels in the ephys data. Defaults to 16.

    Returns:
    np.array: A 2D NumPy array of the electrophysiology data, reshaped into
              (nchannels, number_of_samples_per_channel).
    '''
  
    # reading in binary data
    num_samples = num_samples * nchannels
    ephys_bin = np.fromfile(ephys_file, dtype=np.uint16, count = num_samples)
    
    # ensuring equal samples from each channel
    num_complete_sets = len(ephys_bin) // nchannels
    ephys_bin = ephys_bin[:num_complete_sets * nchannels]

    # reshape 1d array into nchannels x num_samples NumPy array
    ephys_data = np.reshape(ephys_bin, (nchannels, -1), order='F')
    return(ephys_data)


def plot_ephys(ephys, nchannels = 16):
    if nchannels == 16:
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        for ch in range(nchannels):
            ax = axs[ch // 4, ch % 4]
            cax = ax.plot(ephys[ch, :])
            ax.set_title(f'Channel {ch + 1}')
        plt.show()
    elif nchannels == 64:
        fig, axs = plt.subplots(8, 8, figsize=(12, 12))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        for ch in range(nchannels):
            ax = axs[ch // 8, ch % 8]
            cax = ax.plot(ephys[ch, :])
            ax.set_title(f'Channel {ch + 1}')
        plt.show()


def remove_jumps_sniff(sniff: np.array, threshold = 40000, remove = 65520) -> np.array:
    '''
    Removes a common artifact from the signal which typically causes 65520mV jumps
    
    Parameters:
    sniff (np.array): The sniff signal to be processed, represented as a NumPy array.
    threshold (float, optional): The threshold for declaring that the data point is altered by the artifact. Default is 40000.
    remove (float, optional): The magnitude of the artifact to remove from the data point. Default is 65520.

    Returns:
    np.array: The signal with artifacts removed
    '''
  
    jump_indices = sniff > threshold
    sniff[jump_indices] -= remove
    return sniff


def remove_jumps_ephys(ephys: np.array, nchannels = 16, threshold = 40000, remove = 65520) -> np.array:
    '''
    Removes a common artifact from the signal which typically causes 65520mV jumps
    
    Parameters:
    ephys (np.array): The local field potential signal to be processed, represented as a NumPy array.
    nchannels (int, optional): number of ephys channels recorded. Defaults to 16.
    threshold (float, optional): The threshold for declaring that the data point is altered by the artifact. Default is 40000.
    remove (float, optional): The magnitude of the artifact to remove from the data point. Default is 65520.

    Returns:
    mp.array: The signal with artifacts removed
    '''
    for ch in range(nchannels):
        jump_indices = ephys[ch, :] > threshold
        ephys[ch, jump_indices] -= remove
    return ephys


def resample_sniff(sniff: np.array, original_rate = 30000, target_rate = 1000) -> np.array:
    '''
    Resample a sniff signal from an original rate to a target rate.

    This function applies a decimation process to a sniff signal, which is useful in 
    situations where lower sampling rates are sufficient or desired for analysis. The 
    decimation is performed using a Finite Impulse Response (FIR) filter with a Hamming 
    window to reduce aliasing effects.

    Parameters:
    sniff (np.array): The sniff signal to be resampled, represented as a NumPy array.
    original_rate (int, optional): The original sampling rate of the signal in Hz. Defaults to 30000 Hz.
    target_rate (int, optional): The desired sampling rate of the signal in Hz. Defaults to 1000 Hz.

    Returns:
    np.array: The resampled sniff signal.
    '''

    resample_factor = original_rate // target_rate
    resampled_sniff = signal.decimate(sniff, resample_factor, ftype = 'fir')
    return resampled_sniff

    
def resample_ephys(ephys: np.array, nchannels = 16, original_rate = 30000, target_rate = 1000) -> np.array:
    '''
    Resample multi-channel electrophysiology (ephys) data from an original sampling rate to a target rate.

    This function applies a decimation process to each channel of a multi-channel ephys signal. 
    It uses a 30-point Finite Impulse Response (FIR) filter with a Hamming window to mitigate 
    aliasing effects during resampling.

    Parameters:
    ephys (np.array): A 2D NumPy array representing the ephys data, with shape (nchannels, number_of_samples).
    nchannels (int, optional): Number of channels in the ephys data. Defaults to 16.
    original_rate (int, optional): Original sampling rate of the ephys data in Hz. Defaults to 30000 Hz.
    target_rate (int, optional): Target sampling rate in Hz. Defaults to 1000 Hz.

    Returns:
    np.array: A 2D NumPy array of the resampled ephys data, with the same number of channels 
              and a reduced number of samples per channel.
    '''

    resample_factor = original_rate // target_rate
    if 0 == ephys.shape[1] % resample_factor:
        new_length = ephys.shape[1]//resample_factor
    else:
        print(f'Cannot resample from length {ephys.shape[1]}')
        return 0
    resampled_ephys = np.zeros((nchannels, new_length))
    for ch in range(nchannels):
        resampled_ephys[ch, :] = signal.decimate(ephys[ch,:], resample_factor, ftype = 'fir')

    return resampled_ephys
             

def find_inhales(sniff: np.array, window_length = 101, polyorder = 9, min_peak_prominance = 50, show = False) -> np.array:
    '''
    Smooth a sniff signal using the Savitzky-Golay method and locate inhalation times using peak finding.

    This function first applies a Savitzky-Golay filter to smooth the input sniff signal. 
    It then uses a peak finding algorithm to identify the times of inhalations, which are 
    indicated by prominent peaks in the smoothed signal. Optionally, the function can also 
    plot the original and smoothed signals along with the identified peaks.

    Parameters:
    sniff (np.array): The sniff signal to be processed, represented as a NumPy array.
    window_length (int, optional): The length of the filter window. Defaults to 101.
    polyorder (int, optional): The order of the polynomial used to fit the samples. Defaults to 9.
    min_peak_prominance (int, optional): The minimum prominence of a peak to be considered 
                                         an inhalation. Defaults to 50.
    show (bool, optional): If True, display a plot of the original and smoothed signals with peaks. 
                           Defaults to False.

    Returns:
    tuple: A tuple containing two elements:
           - locs (np.array): An array of indices where inhalation peaks are located.
           - smoothed_sniff (np.array): The smoothed sniff signal.
    '''


    smoothed_sniff = signal.savgol_filter(sniff, window_length, polyorder)
    locs, _ = signal.find_peaks(smoothed_sniff, None, None, min_peak_prominance)
    if show == True:
        plt.figure(figsize=(10, 6))
        plt.plot(sniff, label='Original Sniff Signal')
        plt.plot(smoothed_sniff, label='Smoothed Sniff Signal')
        plt.plot(locs, smoothed_sniff[locs], 'x', label='Peaks')
        plt.title('Sniff Signal and Identified Peaks')
        plt.xlabel('Sample')
        plt.ylabel('Signal Amplitude')
        plt.legend()
        plt.show()
    return locs, smoothed_sniff


def peak_finder_validation(sniff: np.array):
    '''visual validation for peak finder. Plots raw and smoothed signal and with peaks'''
    inhales, smoothed_sniff = find_inhales(sniff)
    plt.plot(sniff)
    plt.plot(smoothed_sniff)
    plt.plot(inhales, sniff[inhales], 'x')
    plt.show()


def sniff_lock_lfp(locs: np.array, ephys: np.array, window_size = 1000, nsniffs = 512, beg = 3000) -> np.array:
    '''
    Aligns local field potential (LFP) signals with sniff inhalation times and constructs a 3D array of z-scored LFP activity.

    This function identifies segments of LFP signals corresponding to inhalation times (specified by 'locs') and 
    standardizes these segments across channels. The output is a 3D array where each 'slice' corresponds to the LFP 
    activity surrounding a single sniff event, with data from all channels.

    Parameters:
    locs (np.array): Array of sniff inhalation times (indices).
    ephys (np.array): 2D array of electrophysiological data with shape (nchannels, number_of_samples).
    nchannels (int, optional): Number of channels in the ephys data. Defaults to 16.
    window_size (int, optional): The size of the window around each sniff event to consider for LFP activity. Defaults to 1000.
    nsniffs (int, optional): Number of sniff events to process. Defaults to 512.
    beg (int, optional): Starting index to begin looking for sniff events. Defaults to 3000.

    Returns:
    np.array: A 3D NumPy array with shape (nsniffs, window_size, nchannels). Each 'slice' of this array 
              represents the z-scored LFP activity around a single sniff event for all channels.

    Raises:
    ValueError: If the 'locs' array does not contain enough data after the specified 'beg' index for the required number of sniffs.
    '''

    print(beg)
    print(nsniffs)
    # finding number of channels
    nchannels = ephys.shape[0]

    # finds nsniffs consecutive inhalation times starting at beg, saving these times to loc_set
    first_loc = np.argmax(locs >= beg)
    print(f'first inhale is #{first_loc}')
    loc_set = locs[first_loc: first_loc + nsniffs]
    print(f'last inhale is #{len(loc_set) + first_loc}')
    if len(loc_set) < nsniffs:
        raise ValueError("locs array does not have enough data for the specified range.")
    
    # propogates an nx2 array containing times half the window size in both directions from inhalation times
    windows = np.zeros((nsniffs, 2), dtype=int)
    for ii in range(nsniffs):
        win_beg = loc_set[ii] - round(window_size/2)
        win_end = loc_set[ii] + round(window_size/2)
        windows[ii] = [win_beg, win_end]

    # finds and saves zscored ephys data from each channel for each inhalaion locked time window
    sniff_activity = np.zeros((nchannels, nsniffs, window_size))
    for ii in range(nsniffs):
        for ch in range(nchannels):
            win_beg, win_end = windows[ii]
            data = ephys[ch, win_beg:win_end]
            data_mean = np.mean(data)
            data_std = np.std(data)
            zscore_data = (data - data_mean) / data_std
            sniff_activity[ch,ii,:] = zscore_data
    return sniff_activity, loc_set


def sort_lfp(sniff_activity, locs):
    '''sorts the sniff locked lfp trace by sniff frequency'''

    nchannels = sniff_activity.shape[0]
    nsniffs = sniff_activity.shape[1]
    window_size = sniff_activity.shape[2]
    
    sorted_activity = np.zeros((nchannels, nsniffs-1, window_size))
    
    # finding sniff frequencies by inhalation time differences (we lose the last sniff)
    freqs = np.diff(locs)

    # sorting the ephys data according to these times
    sort_indices = np.argsort(freqs)
    sorted_activity[:, :, :] = sniff_activity[:, sort_indices, :]
    return sorted_activity


def sort_lfp_test(sniff_activity, locs):
    '''sorts the sniff locked lfp trace by sniff frequency'''

    nchannels = sniff_activity.shape[0]
    nsniffs = sniff_activity.shape[1]
    window_size = sniff_activity.shape[2]
    
    sorted_activity = np.zeros((nchannels, nsniffs-1, window_size))
    
    # finding sniff frequencies by inhalation time differences (we lose the last sniff)
    freqs = np.diff(locs)

    first = sniff_activity[:-2, :, :]
    last = sniff_activity[:,:,:-2]

    print(len(first[0]))
    print(len(last[2]))
    sorted_activity[:, :, :] = first[0], freqs, last[0]
    return sorted_activity


def plot_snifflocked_lfp(lfp, show_y = False, subtitle: str = ''):
    '''
    Plots the sniff-locked LFP signal as a heatmap where each inhalation is a unit 
    on the y-axis, time-lag from inhalation time on the x-axis, and the strength of 
    the LFP represented by color.

    Parameters
    ----------
    lfp : np.array
        A 3D numpy array of LFP data. The first dimension represents channels, the 
        second dimension represents sniffs, and the third dimension represents time lags.
    '''

    # finding size for subplot layout
    nchannels = lfp.shape[0]

    # Create subplots based on the number of channels
    if nchannels == 16:
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
    
    elif nchannels == 32:
        fig, axs = plt.subplots(6, 6, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)
    elif nchannels == 64:
        fig, axs = plt.subplots(8, 8, figsize=(6, 6))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    # title and x-axis size
    fig.suptitle("Sniff-Locked LFP", fontsize=16)

    fig.text(0.5, 0.95, subtitle, ha='center', va='top', fontsize=12)

    window_size = lfp.shape[2]
    x_middle = window_size // 2

    # extracting subset of values for y tick labels
    y_ticks = np.linspace(0, lfp.shape[1] - 1, num = 5, dtype = int)
    y_ticks_labels = [lfp[0, y, 0] for y in y_ticks]

    for ch in range(nchannels):
        if nchannels == 16:
            ax = axs[ch // 4, ch % 4]
        elif nchannels == 32:
            ax = axs[ch // 6, ch % 6]
        elif nchannels ==64:
            ax = axs[ch // 8, ch % 8]

        # Plotting each channel
        cax = ax.imshow(lfp[ch, :, :], aspect='auto', cmap = 'seismic')
        ax.set_title(f'Channel {ch + 1}')

        # x-axis
        ax.set_xticks([x_middle - window_size/2, x_middle, x_middle + window_size/2])
        ax.set_xticklabels([-window_size/2, '0', window_size/2])

        # y-axis
        if show_y:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks_labels)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
        
        #colorbar
        fig.colorbar(cax, ax=ax)

    plt.show()


def combine_channels(lfp: np.array) -> np.array:
    '''
    Combine multiple channels of LFP (Local Field Potential) data by averaging.

    This function processes an array of LFP data, where the data is assumed to be
    organized with one dimension for channels, one for "sniffs" or time points, 
    and one for the sampling window within each sniff. The function averages the 
    LFP data across all channels for each time point within each sniff, resulting 
    in a 2D array with dimensions corresponding to sniffs and window time points.

    Parameters:
    lfp : np.array
        A 3D numpy array of LFP data. The dimensions are expected to be in the 
        order of (channels, sniffs, window time points).

    Returns:
    np.array
        A 2D numpy array where each element is the average of the LFP data across 
        all channels for a given sniff and time point. The dimensions are (sniffs, 
        window time points).
        '''


    # finding data shape
    nsniffs = lfp.shape[1]
    window_size = lfp.shape[2]

    # averaging across channels
    combined_lfp = np.zeros((nsniffs, window_size))
    for ii in range(window_size):
        for sniff in range(nsniffs):
            combined_lfp[sniff, ii] = np.mean(lfp[:,sniff,ii])
    return combined_lfp


def avg_sniff_locked_lfp(lfp, nchannels = 16, window_size = 1000):
    '''averages the lfp strength across sniffs'''
    avg_lfp = np.zeros((nchannels, window_size))
    for ch in range(nchannels):
        avg_lfp[ch,:] = np.mean(lfp[ch,:,:], axis = 0)
    return avg_lfp


def plot_avg_lfp(avg_lfp, nchannels = 16, window_size = 1000):
    '''plots the averaged lfp signal'''
    if nchannels == 16:
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

    if nchannels == 32:
        fig, axs = plt.subplots(6, 6, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)

    x_middle = avg_lfp.shape[0] // 2
    for ii in range(nchannels):
        if nchannels == 16:
            ax = axs[ii // 4, ii % 4]
        if nchannels == 32:
            ax = axs[ii // 6, ii % 6]
        cax = ax.plot(avg_lfp[ii,:])
        ax.set_title(f'Channel {ii + 1}')
        ax.set_xticks([x_middle - window_size/2, x_middle, x_middle + window_size/2])
        ax.set_xticklabels([-window_size/2, '0', window_size/2])
        ax.set_yticklabels([])
        ax.set_yticks([])

    plt.show()


def sniff_lock_lfp_infreq(locs: np.array, ephys: np.array, freq_bin = [6,8], nchannels = 16, window_size = 1000, maxsniffs = 512, beg = 2000, end = 3000) -> np.array:
    '''aligns local field potential signal with sniff inhalation times whithin frequency bins and propogates a 3D array describing LFP activity around each sniff at each channel'''
    loc_set = locs[beg:end]
    sniff_activity = np.zeros((len(loc_set), window_size, nchannels))
    windows = np.zeros((end-beg, 2), dtype=int)
    for ii in range(len(loc_set)):
        win_beg = loc_set[ii] - round(window_size/2)
        win_end = loc_set[ii] + round(window_size/2)
        windows[ii] = [win_beg, win_end]
    for ii in range(len(loc_set)):
        for ch in range(nchannels):
            win_beg, win_end = windows[ii]
            data = ephys[ch, win_beg:win_end]
            data_mean = np.mean(data)
            data_std = np.std(data)
            zscore_data = (data - data_mean) / data_std
            sniff_activity[ii,:,ch] = zscore_data
    for ch in range(nchannels):
        npeaks = []
        count = 0
        while count <= maxsniffs and count <= sniff_activity.shape[0]:
            win_beg, win_end = windows[count]
            inhales = [index for index, value in enumerate(locs) if win_beg <= value <= win_end]
            if len(inhales) in range(freq_bin[0], freq_bin[1] + 1):
                npeaks.append(count) 
            count += 1   
        if ch == 0:
            infreq_activity = np.zeros((len(npeaks), window_size, nchannels))
        infreq_activity[:, :, ch] = sniff_activity[npeaks, :, ch]
    return infreq_activity
    

def plot_single_lfp(lfp: np.array):
    '''
    Plots averaged LFP data from all channels with a colorbar.

    This function visualizes a 2D numpy array of LFP data using a heatmap.
    The plot includes a title, labeled axes, custom tick labels for the x-axis, 
    and a colorbar with a label to interpret the average z-scored LFP values.

    Parameters
    ----------
    lfp : np.array
        A 2D numpy array of LFP data to be visualized. The first dimension 
        represents different sniffs, and the second dimension represents time lags.
    '''
    window_size = lfp.shape[1]
    x_middle = window_size // 2

    im = plt.imshow(lfp, aspect='auto')

    plt.title('Averaged LFP from all Channels')
    plt.xlabel('Time Lag')
    plt.ylabel('Sniff')

    ax = plt.gca()
    ax.set_xticks([x_middle - window_size/2, x_middle, x_middle + window_size/2])
    ax.set_xticklabels([-window_size/2, '0', window_size/2])

    plt.colorbar(im, label='Avg Z-scored LFP Value')
    plt.show()


def circular_shift(ephys: np.array, nshifts: int = 1000, method: str = 'sample') -> np.array:
    '''
    Perform circular shifts on an electrophysiological signal array.

    This function applies circular shifts to a 2D numpy array representing electrophysiological (ephys) data. 
    It supports two methods of shifting: 'sample' and 'random'. 
    For 'sample', the function creates evenly spaced shifts if the number of columns in `ephys` is divisible by `nshifts`.
    If not, it recursively calls itself with one less shift until this condition is met.
    For 'random', the function applies a random shift for each of the `nshifts`.
    
    The function preallocates an array `circ_ephys` to store the shifted arrays.

    Parameters:
    ephys (np.array): A 2D numpy array representing electrophysiological data. 
                      The first dimension corresponds to different signals or channels, 
                      and the second dimension corresponds to time points.
    nshifts (int, optional): The number of shifts to be applied. Default is 1000.
    method (str, optional): The method of shifting to be used. 
                            Can be 'sample' for evenly spaced shifts or 'random' for random shifts. 
                            Default is 'sample'.

    Returns:
    np.array: A 3D numpy array where each 'slice' (along the third dimension) 
              is the `ephys` array after a circular shift.

    '''

    # preallocating an array to hold the ephys signal after all nshifts
    nchannels = ephys.shape[0]   
    signal_length = ephys.shape[1]
    circ_ephys = np.zeros((nchannels, signal_length, nshifts))

    if method == 'sample':
        # shifting the ephys with evenly spaced shifts
        if ephys.shape[1] % nshifts == 0:
            print(f'performing circular shift with {nshifts} shifts')
            jump = ephys.shape[1] // nshifts
            for ii in range(nshifts):
                circ_ephys[:,:,ii] = np.roll(ephys, ii * jump, axis = 1)
        else:
            circular_shift(ephys, nshifts - 1, method = method)

    if method == 'random':
        # shifting the ephys signal with nshifts random shifts
        for ii in range(nshifts):
            circ_ephys[:,:,ii] = np.roll(ephys, np.random.randint(0, nshifts), axis = 1)

    print(circ_ephys.shape)

    return circ_ephys


def create_circular_null(circ_ephys: np.array, locs: np.array, nsniffs: int = 200, window_size: int = 1000, beg: int = 3000, sort = True) -> np.array:

    '''
    Create a circular null distribution of LFP voltages across different channels, sniffs, and time-lags.

    This function generates a four-dimensional array representing the circularly shifted lfp signals, locked to inhalation times,
    which serves as a null distribution for further analysis. The array contains the distributions of local field
    potential (LFP) voltages for each channel, each sniff, and each time-lag, as a function of the number of shifts.

    Parameters:
    - circ_ephys (np.array): A 3D numpy array representing circularly shifted electrophysiological data.
                             The dimensions are expected to be [nchannels, _, nshifts].
    - locs (np.array): Array of inhalation times to be used in the sniff-locking process.
    - nsniffs (int, optional): Number of sniffs to consider. Defaults to 200.
    - window_size (int, optional): The size of the window for analysis. Defaults to 1000.
    - beg (int, optional): The beginning index for the window of analysis. Defaults to 3000.
    - sort (bool, optional): Flag to determine whether to sort the data. Defaults to True.

    Returns:
    - np.array: A 4D numpy array with dimensions [nchannels, nsniffs, window_size, nshifts],
                representing the null distribution of LFP voltages.
    '''

    # preallocating an array to hold the distributions of lfp voltages at each channel, each sniff, and each time-lag as a function of the nshifts
    nchannels = circ_ephys.shape[0]
    nshifts = circ_ephys.shape[2]
    circular_null = np.zeros((nchannels, nsniffs, window_size, nshifts), dtype = np.float64)
    print(beg)

    # propogating a 4d array containing the circul shifted sniff locked signals i.e. the null distribution
    for shift in range(nshifts):
        circular_null[:,:,:,shift], locs = sniff_lock_lfp(locs, circ_ephys[:,:,shift], nsniffs = nsniffs, beg = beg)

    return circular_null
        

def check_normality(circular_null: np.array, alpha: float = 0.05, plot: bool = True, nsamples: int = 4, KStest: bool = True) -> float:
    """
    Check the normality of randomly selected samples from a 4D array using histograms and the Kolmogorov-Smirnov test.

    This function randomly selects a specified number of samples from a 4D array (circular_null) and performs two checks for normality:
    1. Plots histograms of these samples if the 'plot' parameter is True.
    2. Performs the Kolmogorov-Smirnov test on these samples to compare them with a normal distribution if 'KStest' is True.

    Parameters:
    - circular_null (np.array): A 4D numpy array from which samples are drawn.
    - alpha (float, optional): alpha value criteria for signficance. Defaults to 0.05.
    - plot (bool, optional): If True, histograms of the selected samples are plotted. Defaults to True.
    - nsamples (int, optional): The number of random samples to draw for normality checking. Defaults to 4.
    - KStest (bool, optional): If True, performs the Kolmogorov-Smirnov test on the selected samples. Defaults to True.

    Returns:
    - float: The ratio of samples that did not significantly deviate from a normal distribution (p > alpha) in the Kolmogorov-Smirnov test.

    Note:
    - The function returns the ratio of samples with p-values greater than alpha in the Kolmogorov-Smirnov test, indicating non-significant deviation from normality.
    - The function plots histograms for visual inspection of normality if 'plot' is set to True.
    """

    nchannels, nsniffs, window_size, _ = circular_null.shape

    # Randomly select nsamples combinations of channels, sniffs, and times
    random_combinations = [(np.random.randint(nchannels), np.random.randint(nsniffs), np.random.randint(window_size))
                               for _ in range(nsamples)]

    if plot:

        # Determine the subplot grid dimensions
        sqrt_samples = math.ceil(math.sqrt(nsamples))
        fig, axs = plt.subplots(sqrt_samples, sqrt_samples, figsize=(12, 12))
        axs = axs.flatten()  # Flatten the array of axes for easier iteration

        # Create a histogram for each random combination
        for i, (ch, sniff, t) in enumerate(random_combinations):
            data = circular_null[ch, sniff, t, :].flatten()
            axs[i].hist(data, bins=20, alpha=0.7)
            axs[i].set_title(f'Ch {ch}, Sniff {sniff}, Time {t}')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Frequency')

            # Hide unused subplots if any
            if i >= len(axs):
                axs[i].set_visible(False)

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    if KStest:
        count = 0
        for i, (ch, sniff, t) in enumerate(random_combinations):
            data = circular_null[ch, sniff, t, :].flatten()
            KStest_stat = stats.kstest(data, stats.norm.cdf)
            p = KStest_stat.pvalue
            if p > alpha:
                count += 1

    # calculate the ratio of signficant to non-signficant distributions           
    ratio = count / nsamples

    return ratio


def plot_normality(circular_null: np.array, niters: int = 1000, step_size: int = 1, plot: bool = True) -> np.array:
    """
    Assess and plot the fraction of Gaussian distributions in a 4D array over a range of sample sizes.

    This function iteratively increases the number of samples drawn from a 4D array (circular_null) to check for Gaussianity. 
    At each iteration, it calculates the fraction of these samples that are Gaussian based on a normality check function 
    (assumed to be `check_normality`). The results are plotted against the number of distributions sampled if 'plot' is True.

    Parameters:
    - circular_null (np.array): A 4D numpy array to be analyzed for normality.
    - niters (int, optional): The number of iterations for which the normality check is run. Defaults to 1000.
    - step_size (int, optional): The increment in the number of samples checked for normality at each iteration. Defaults to 1.
    - plot (bool, optional): If True, the function plots the fraction of Gaussian distributions over the iterations. Defaults to True.

    Returns:
    - np.array: An array of the fraction of Gaussian distributions for each iteration.

    Note:
    - The function assumes the existence of a `check_normality` function that returns the fraction of distributions that are Gaussian for a given number of samples.
    - The plot, if enabled, displays the trend in the fraction of Gaussian distributions as the sample size increases.
    """

    # finding fraction of significant distributions at each iteration of number of samples
    sig_ratios = np.zeros((niters))
    for i in range(niters):
        nsamples = i * step_size + 1
        sig_ratios[i] = check_normality(circular_null, plot = False, nsamples = nsamples)

    # plotting
    if plot:
        fig, ax = plt.subplots()
        ax.plot(sig_ratios)
        ax.set(xlabel='number of distributions', ylabel='fraction',
        title='Fraction of Gaussian Distributions', )
        ax.grid()

        plt.show()

    return sig_ratios


def sniff_lock_std(locs: np.array, circular_null: np.array, beg: int = 3000, window_size: int = 1000, nsniffs: int = 200) -> np.array:
    """
    Calculates the z-scores of sniff-locked LFP values with respect to a null distribution.

    This function processes Local Field Potential (LFP) data to find the z-score of LFP values at each channel
    and each moment around each inhalation, relative to a null distribution. It identifies inhalation times from the `locs` array,
    creates windows around these times, and computes z-scores for LFP values in these windows based on the mean and standard deviation
    of the null distribution.

    Parameters:
    ----------
    locs : np.array
        Array of inhalation times.
    circular_null : np.array
        A 4D numpy array representing the null distribution of LFP data. The dimensions are expected to be [nchannels, _, _, nshuffles].
    beg : int, optional
        The beginning index for selecting inhalation times. Defaults to 3000.
    window_size : int, optional
        Size of the window to consider around each inhalation time. Defaults to 1000.
    nsniffs : int, optional
        Number of consecutive sniffs to consider starting from the 'beg' index. Defaults to 200.

    Returns:
    -------
    tuple: (sniff_activity, loc_set)
        sniff_activity : np.array
            A 3D numpy array where each element is the z-score of the LFP value for a specific channel, sniff, and time relative to the null distribution.
        loc_set : np.array
            Array of selected inhalation times based on 'beg' and 'nsniffs'.

    Raises:
    ------
    ValueError
        If the 'locs' array does not have enough data for the specified range.

    Notes:
    -----
    - The function assumes the 'circular_null' array is structured as [channels, sniffs, time-lags, shuffles].
    - The z-scores are calculated for each channel, sniff, and time-lag using the corresponding mean and standard deviation from the null distribution.
    """
    # Function implementation...


    # finding number of channels
    nchannels = circular_null.shape[0]

    # finding nsniffs consecutive inhalation times starting at beg, saving these times to loc_set
    first_loc = np.argmax(locs >= beg)
    print(f'first inhale is #{first_loc}')
    loc_set = locs[first_loc: first_loc + nsniffs]
    print(f'last inhale is #{len(loc_set) + first_loc}')
    if len(loc_set) < nsniffs:
        raise ValueError("locs array does not have enough data for the specified range.")
    
    # propogates an nx2 array containing times half the window size in both directions from inhalation times
    windows = np.zeros((nsniffs, 2), dtype=int)
    for ii in range(nsniffs):
        win_beg = loc_set[ii] - round(window_size/2)
        win_end = loc_set[ii] + round(window_size/2)
        windows[ii] = [win_beg, win_end]

    # finds and saves lfp zscore from null distribution in each channel for each moment around each inhalaion
    sniff_activity = np.zeros((nchannels, nsniffs, window_size))
    for sniff in range(nsniffs):
        for ch in range(nchannels):
            for t in range(window_size):
                win_beg, win_end = windows[ii]

                # sniff locked value
                value = circular_null[ch, sniff, t, 0]

                # mean of distribution
                mu = np.mean(circular_null[ch, sniff, t, :].flatten())

                # standard deviation
                sigma = np.std(circular_null[ch, sniff, t, :].flatten())

                # calculating the z-score with respect to the null distribution and saving to sniff_activity
                z = (value - mu) / sigma
                sniff_activity[ch, sniff, t] = z

    return sniff_activity, loc_set
