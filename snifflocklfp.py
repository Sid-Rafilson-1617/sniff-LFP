"""snifflocklfp library

This library contains functions for reading in local field potential and sniff data, preprocessing signals, and further analysis including aligning neural data to inhalation times.

By: Sid Rafilson
Orginal MATLAB functions written by: Nate Hess

"""

import numpy as np

import pandas as pd

from scipy import signal

import matplotlib.pyplot as plt



def load_sniff(sniff_file: str, num_samples: int) -> np.array:
    '''Loads binary sniff file into numpy array'''
    adcx = np.fromfile(sniff_file, dtype=np.uint16, count=num_samples)
    return adcx


def get_sniff(adcx: np.array, nchannels = 8, ch = 8) -> np.array:
    '''extracts the sniff signal from the sniff binary array and returns numpy array.
        Typically, there are 8 channels in the array, and sniff is in channel number 8'''
    num_complete_sets = len(adcx) // nchannels
    adcx = adcx[:num_complete_sets * nchannels]
    adcx = np.reshape(adcx, (nchannels, -1), order = 'F')
    ch_index = ch - 1
    sniff = adcx[ch_index, :]
    return sniff


def load_ephys(ephys_file: str, num_samples: int) -> np.array:
    '''Loads binary electrophysiology signal into numpy array'''
    ephys = np.fromfile(ephys_file, dtype=np.uint16, count = num_samples)
    return ephys


def reshape_ephys(ephys: np.array, nchannels = 16) -> np.array:
    '''reshapes binary ephys data into N channel matrix'''
    ephys_data = np.reshape(ephys, (nchannels, -1), order='F')
    return(ephys_data)


def remove_jumps_sniff(sniff: np.array) -> np.array:
    '''Removing an artifact from the signal which causes 65520mV jumps'''
    jump_indices = sniff > 40000
    sniff[jump_indices] -= 65520
    return sniff


def remove_jumps_ephys(ephys: np.array, nchannels = 16) -> np.array:
    '''Removing an artifact from the signal which causes 65520mV jumps'''
    jump_indices = ephys > 40000
    ephys[jump_indices] -= 65520
    return ephys


def resample_sniff(sniff: np.array, original_rate = 30000, target_rate = 1000) -> np.array:
    '''resample sniff signal from original rate to target rate using 30pt FIR filter with Hamming window'''
    try:
        resample_factor = original_rate // target_rate
        cached_resampled_sniff = signal.decimate(sniff, resample_factor, ftype = 'fir')
        return cached_resampled_sniff

    except ValueError as e:
        print(f'Error in resampling: {e}')
        return None
    

def resample_ephys(ephys: np.array, nchannels = 16, original_rate = 30000, target_rate = 1000) -> np.array:
    '''resample ephys signal from original rate to target rate using 30pt FIR filter with Hamming window'''
    try:
        resample_factor = original_rate // target_rate
        resampled_ephys = np.zeros((nchannels, ephys.shape[1]//resample_factor))
        for ch in range(nchannels):
            resampled_ephys[ch, :] = signal.decimate(ephys[ch, :], resample_factor, ftype = 'fir')
        return resampled_ephys
             

    except ValueError as e:
        print(f'Error in resampling: {e}')
        return None
    
def find_inhales(sniff: np.array, window_length = 101, polyorder = 9, min_peak_prominance = 50, show = False) -> np.array:
    '''smoothes sniff signal using savitsky golay method and uses peak finder function to locate inhalations '''
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
    inhales, smoothed_sniff = find_inhales(sniff)
    plt.plot(sniff)
    plt.plot(smoothed_sniff)
    plt.plot(inhales, sniff[inhales], 'x')
    plt.show()


def sniff_lock_lfp(locs: np.array, ephys: np.array, nchannels = 16, window_size = 1000, nsniffs = 512, beg = 3000) -> np.array:
    '''aligns local field potential signal with sniff inhalation times and propogates a 3D array describing LFP activity around each sniff at each channel'''
    if len(locs) < beg + nsniffs:
        raise ValueError("locs array does not have enough data for the specified range.")
    loc_set = locs[beg:nsniffs+beg]
    windows = np.zeros((nsniffs, 2), dtype=int)
    for ii in range(nsniffs):
        win_beg = loc_set[ii] - round(window_size/2)
        win_end = loc_set[ii] + round(window_size/2)
        windows[ii] = [win_beg, win_end]
    sniff_activity = np.zeros((nsniffs, window_size, nchannels))
    for ii in range(nsniffs):
        for ch in range(nchannels):
            win_beg, win_end = windows[ii]
            data = ephys[ch, win_beg:win_end]
            data_mean = np.mean(data)
            data_std = np.std(data)
            zscore_data = (data - data_mean) / data_std
            sniff_activity[ii,:,ch] = zscore_data
    return sniff_activity


def sort_lfp(sniff_activity, locs, nsniffs = 512, window_size = 1000, nchannels = 16, beg = 3000):
    '''sorts the sniff locked lfp trace by sniff frequency'''
    windows = np.zeros((nsniffs, 2), dtype=int)
    sorted_activity = np.zeros_like(sniff_activity)
    loc_set = locs[beg:nsniffs+beg]
    for ii in range(nsniffs):
        win_beg = loc_set[ii] - round(window_size/2)
        win_end = loc_set[ii] + round(window_size/2)
        windows[ii] = [win_beg, win_end]
    for ch in range(nchannels):
        npeaks = np.zeros(nsniffs)
        for ii in range(nsniffs):
            win_beg, win_end = windows[ii]
            npeaks[ii] = len([index for index, value in enumerate(locs) if win_beg <= value <= win_end])
        sort_indices = np.argsort(npeaks)
        sorted_activity[:, :, ch] = sniff_activity[sort_indices, :, ch]
    return sorted_activity


def plot_snifflocked_lfp(lfp, nchannels = 16, window_size = 1000):
    '''plots the sniff locked lfp signal as heatmap where each inhalation is a unit on the y axis, time-lag from inhalation time on x-axis, and the strength of the lfp represented by color'''

    if nchannels == 16:
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
    
    if nchannels == 32:
        fig, axs = plt.subplots(6, 6, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)
           
    x_middle = lfp.shape[1] // 2
    for ii in range(nchannels):
        if nchannels == 16:
            ax = axs[ii // 4, ii % 4]
        if nchannels == 32:
            ax = axs[ii // 6, ii % 6]
        cax = ax.imshow(lfp[:, :, ii], aspect='auto')
        ax.set_title(f'Channel {ii + 1}')
        ax.set_xticks([x_middle - window_size/2, x_middle, x_middle + window_size/2])
        ax.set_xticklabels([-window_size/2, '0', window_size/2])
        ax.set_yticklabels([])
        ax.set_yticks([])

    plt.show()


def avg_sniff_locked_lfp(lfp, nchannels = 16, window_size = 1000):
    '''averages the lfp strength across sniffs'''
    avg_lfp = np.zeros((window_size, nchannels))
    for ch in range(nchannels):
        avg_lfp[:,ch] = np.mean(lfp[:,:,ch], axis = 0)
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
        cax = ax.plot(avg_lfp[:,ii])
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
    


main()
