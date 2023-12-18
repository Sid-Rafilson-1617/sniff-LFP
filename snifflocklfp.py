"""
snifflocklfp library

This library contains functions for reading in local field potential and sniff data, preprocessing signals, and further analysis including aligning neural data to inhalation times.

By: Sid Rafilson
Orginal MATLAB functions written by: Nate Hess

"""

import numpy as np
import pandas as pd
from scipy import signal
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
    adcx = np.fromfile(sniff_file, dtype=np.uint16, count=num_samples)
    num_samples = num_samples * nchannels
    num_complete_sets = len(adcx) // nchannels
    adcx = adcx[:num_complete_sets * nchannels]
    adcx = np.reshape(adcx, (nchannels, -1), order = 'F')
    ch_index = ch - 1
    sniff = adcx[ch_index, :]
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
    num_samples = num_samples * nchannels
    ephys = np.fromfile(ephys_file, dtype=np.uint16, count = num_samples)
    ephys_data = np.reshape(ephys, (nchannels, -1), order='F')
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


def sniff_lock_lfp(locs: np.array, ephys: np.array, nchannels = 16, window_size = 1000, nsniffs = 512, beg = 3000) -> np.array:
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
        


def plot_snifflocked_lfp(lfp, nchannels = 16, window_size = 1000):
    '''plots the sniff locked lfp signal as heatmap where each inhalation is a unit on the y axis, time-lag from inhalation time on x-axis, and the strength of the lfp represented by color'''

    if nchannels == 16:
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
    
    elif nchannels == 32:
        fig, axs = plt.subplots(6, 6, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)
    elif nchannels == 64:
        fig, axs = plt.subplots(8, 8, figsize=(6, 6))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
           
    x_middle = lfp.shape[2] // 2
    for ch in range(nchannels):
        if nchannels == 16:
            ax = axs[ch // 4, ch % 4]
        elif nchannels == 32:
            ax = axs[ch // 6, ch % 6]
        elif nchannels ==64:
            ax = axs[ch // 8, ch % 8]
        cax = ax.imshow(lfp[ch, :, :], aspect='auto')
        ax.set_title(f'Channel {ch + 1}')
        ax.set_xticks([x_middle - window_size/2, x_middle, x_middle + window_size/2])
        ax.set_xticklabels([-window_size/2, '0', window_size/2])
        ax.set_yticklabels([])
        ax.set_yticks([])
        fig.colorbar(cax, ax=ax)
    plt.show()


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
    



