"""
core library for local field potential project.

This library contains functions for reading in local field potential, sniff and behavioral data,
preprocessing signals, and further analysis including aligning neural data to inhalation times, dimensionallity reduction, etc.

By: Sid Rafilson
Contributors: Nate Hess
Primary Investigator: Matt Smear
"""

import os

import math
import numpy as np
import pandas as pd

from scipy import signal
from scipy import stats
from scipy.fft import fft, ifft
from scipy.io import loadmat

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable
import seaborn as sns

from kneed import KneeLocator

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA



import umap



#_________________________________________________________________________________________________LOADING DATA________________________________________________________________________________________________________________________________#


def load_sniff(sniff_file: str, num_samples: int = -1, start = 0, stop = 0, nchannels: int = 8, ch: int = 8) -> np.array:
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

    # removing start seconds from beggining, and stop from end of signal; default is 0
    start = start * 30000
    stop = stop * 30000
    if stop == 0:
        sniff = sniff[start:]  
    else:
        sniff = sniff[start: -stop]

    return sniff
    

def load_ephys(ephys_file: str, num_samples: int = -1, start = 0, stop = 0, nchannels: int = 16) -> np.array:
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

    # removing start seconds from beggining, and stop from end of signal; default is 0
    start = start * 30000
    stop = stop * 30000

    if stop == 0:
        ephys = ephys_data[:, start:]
    else:
        ephys = ephys_data[:, start: -stop]

    return ephys


def load_tracking(tracking_file: str, num_samples = -1, start = 0, stop = 0) -> np.array:

    if num_samples != -1:
        num_samples = num_samples // (30000 // 100)
    tracking = np.loadtxt(tracking_file, delimiter = ',', dtype = float, max_rows = num_samples)
    if stop == 0:
        tracking = tracking[start:, :].T
    else:
        tracking = tracking[start: -stop, :].T
    return tracking


def get_sniff_signal_MAT(sniff_file: str):
    '''
    Function to extract sniff signal from MATLAB file
    '''
    sniff_data = loadmat(sniff_file)
    sniff_signal = sniff_data['sniff']
    return sniff_signal

def load_sniff_MATLAB(file: str) -> np.array:
    '''
    Loads a MATLAB file containing sniff data and returns a numpy array
    '''

    mat = loadmat(file)
    sniff_params = mat['sniff_params']

    # loading sniff parameters
    inhalation_times = sniff_params[:, 0]
    inhalation_voltage = sniff_params[:, 1]
    exhalation_times = sniff_params[:, 2]
    exhalation_voltage = sniff_params[:, 3]

    # bad sniffs are indicated by 0 value in exhalation_times
    bad_indices = np.where(exhalation_times == 0)


    # removing bad sniffs
    inhalation_times = np.delete(inhalation_times, bad_indices)
    inhalation_voltage = np.delete(inhalation_voltage, bad_indices)
    exhalation_times = np.delete(exhalation_times, bad_indices)
    exhalation_voltage = np.delete(exhalation_voltage, bad_indices)

    return inhalation_times.astype(np.int32), inhalation_voltage, exhalation_times.astype(np.int32), exhalation_voltage






#_________________________________________________________________________________________________PLOTTING________________________________________________________________________________________________________________________________#


def plot_ephys(ephys, plot_range = (0, -1), method = 'overlay'):
    '''plots the ephys signal'''

    # defining function to convert x-axis to seconds
    def ms_to_s(x, pos):
        '''converts milliseconds to seconds'''
        return f'{(x + plot_range[0]) * 1e-3 :.0f}'
    
    # converting x-axis to seconds
    formatter = ticker.FuncFormatter(ms_to_s)

    # finding number of channels
    nchannels = ephys.shape[0]
    if method == 'seperate':
        if nchannels == 16:
            fig, axs = plt.subplots(4, 4, figsize=(12, 12))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)

            for ch in range(nchannels):
                ax = axs[ch // 4, ch % 4]
                cax = ax.plot(ephys[ch, plot_range[0]:plot_range[1]])
                ax.set_title(f'Channel {ch + 1}')
                plt.gca().xaxis.set_major_formatter(formatter)
        elif nchannels == 64:
            fig, axs = plt.subplots(8, 8, figsize=(12, 12))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)

            for ch in range(nchannels):
                ax = axs[ch // 8, ch % 8]
                cax = ax.plot(ephys[ch, plot_range[0]:plot_range[1]])
                ax.set_title(f'Channel {ch + 1}')
                plt.gca().xaxis.set_major_formatter(formatter)

    # plotting ephys signal
    elif method == 'overlay':
        fig, ax = plt.subplots(figsize=(12, 12))
        for ch in range(nchannels):
            ax.plot(ephys[ch, plot_range[0]:plot_range[1]])
        ax.set_title('Ephys Signal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage')
        ax.legend([f'Channel {ch + 1}' for ch in range(nchannels)])
        plt.gca().xaxis.set_major_formatter(formatter)

    plt.show()

def plot_ephyssniff_new(ephys, sniff, inh, exh, plot_range = (0, -1)):

    path = r"E:\Sid_LFP\figs\Poster"

    nchannels = ephys.shape[0]

    sns.set_style('white')
    #sns.set_context('poster')

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 6))

    # finding inhales and exhales in plot_range
    inh = inh[(inh >= plot_range[0]) & (inh <= plot_range[1])]
    exh = exh[(exh >= plot_range[0]) & (exh <= plot_range[1])]





    colors = ['firebrick', 'forestgreen', 'steelblue']
    for ch in range(nchannels):
        sns.lineplot(x = np.arange(ephys.shape[1])[plot_range[0]:plot_range[1]] / 1_000, y = ephys[ch, plot_range[0]:plot_range[1]], ax = axs[0], color = colors[ch])
    axs[0].set_title('LFP Signal')
    axs[0].set_xlabel('Time (seconds)')
    axs[0].set_ylabel('Voltage (mV)')

    sns.lineplot(x = np.arange(sniff.shape[0])[plot_range[0]:plot_range[1]] / 1_000, y = sniff[plot_range[0]:plot_range[1]], ax = axs[1], color = 'black')
    sns.scatterplot(x = inh / 1_000, y = sniff[inh], ax = axs[1], color = 'grey', s = 100)
    sns.scatterplot(x = exh / 1_000, y = sniff[exh], ax = axs[1], color = 'grey', s = 100)
    axs[1].set_title('Thermistor Signal')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('Temperature (a.u.)')
  
    # only show whole number x axis ticks
    for ax in axs:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    sns.despine()

    # making font black
    plt.setp(axs[0].xaxis.get_majorticklabels(), color='black')

    # increasing image quality
    plt.savefig(os.path.join(path, 'ephys_sniff.png'), dpi = 300)
    plt.close()




def plot_ephyssniff(ephys, sniff, plot_range = (0, -1)):

    # defining function to convert x-axis to seconds
    def ms_to_s(x, pos):
        '''converts milliseconds to seconds'''
        return f'{(x + plot_range[0]) * 1e-3 :.0f}'
    
    # converting x-axis to seconds
    formatter = ticker.FuncFormatter(ms_to_s)

    # finding number of channels
    nchannels = ephys.shape[0]

    # Create subplots based on the number of channels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), layout = 'constrained')

    colors = ['firebrick', 'olivedrab', 'steelblue']
    print('plotting ephys signal')
    # plotting ephys signal
    for ch in range(nchannels):
        color = colors[ch % len(colors)]
        ax1.plot(ephys[ch, plot_range[0]:plot_range[1]], color = color)
        ax1.set_title('LFP (1-15 Hz)')
        ax1.set_ylabel('Voltage (au)')
        ax1.legend(['Channel 1','Channel 3', 'Channel 5'])
        #ax1.legend([f'Channel {ch + 1}' for ch in range(nchannels)])
    print('plotting sniff signal')

    # plotting sniff signal    
    ax2.plot(sniff[plot_range[0]:plot_range[1]] * 1000, 'k')
    ax2.set_title('Thermistor')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voltage (mV)')

    # formatting x-axis
    ax1.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)
    print('showing plot')
    plt.show()


def plot_sniff_hist(freqs: np.array):
    '''plots a histogram of the sniff frequencies'''
    counts, bins = np.histogram(freqs)
    plt.stairs(counts, bins, fill = True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count')
    plt.title('Distribution of Sniff Frequencies')
    plt.show()


def plot_snifflocked_lfp_single(lfp: np.array, freqs: np.array, save_path: str, ch = 1, show_y = True, show_x = True):
    window_size = lfp.shape[2]
    x_middle = window_size // 2

    y_ticks = np.linspace(0, lfp.shape[1] - 1, num = 5, dtype = int)
    y_ticks_labels = [f'{freqs[i]:.1f}' for i in y_ticks]

    
    plt.figure()
    im = plt.imshow(lfp[ch-1, :, :], aspect='auto', cmap = 'seismic')


    plt.title(f'LFP Channel {ch}')
    plt.xlabel('Time Lag')
    plt.ylabel('Sniff')

    ax = plt.gca()

    # x-axis
    ax.set_xticks([x_middle - window_size/2, x_middle, x_middle + window_size/2])
    ax.set_xticklabels([-window_size/2, '0', window_size/2])

    if show_x:
        ax.set_xlabel('ms')
    ax.set_ylabel('Sn/s')

    # y-axis
    if show_y:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks_labels)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    plt.colorbar(im)
    plt.savefig(save_path + f'Channel_{ch}_raster.png')
    plt.close()
    plt.cla()


def plot_snifflocked_lfp_3d(lfp: np.array, freqs: np.array, show_y = False, subtitle: str = ''):
    # [existing setup code...]

    nchannels = lfp.shape[0]

    # Create coordinate grids for X (time) and Y (frequencies)
    time = np.arange(lfp.shape[2])  # Replace with actual time values if available
    X, Y = np.meshgrid(time, freqs)

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
    y_ticks_labels = [f'{freqs[i]:.1f}' for i in y_ticks]

    for ch in range(nchannels):
        # Create a 3D subplot
        if nchannels == 16:
            ax = fig.add_subplot(4, 4, ch+1, projection='3d')
        elif nchannels == 32:
            ax = fig.add_subplot(6, 6, ch+1, projection='3d')
        elif nchannels == 64:
            ax = fig.add_subplot(8, 8, ch+1, projection='3d')

        # Plotting each channel as a 3D surface
        cax = ax.plot_surface(X, Y, lfp[ch, :, :], cmap='seismic')

        # Titles and labels
        ax.set_title(f'Channel {ch + 1}')

        # [You can adjust view angle, limits, etc. here]

    # [remaining code...]

    plt.show()


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


def plot_snifflocked_lfp(lfp: np.array, freqs: np.array, show_y = True, show_x = True, subtitle: str = ''):
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
    y_ticks_labels = [f'{freqs[i]:.1f}' for i in y_ticks]

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

        if show_x:
            ax.set_xlabel('ms')
        
        ax.set_ylabel('Sn/s')

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


def plot_signal_hist(ephysdf: pd.DataFrame, channel = 'all', nbins = 15):
    """
    Plots histograms of electrophysiological data from a specified channel or all channels.

    This function takes a pandas DataFrame containing electrophysiological data, 
    a channel number, and the number of bins for the histogram. It plots histograms 
    for either a single specified channel or all channels in the DataFrame. 

    Parameters:
    -----------
    ephysdf : pd.DataFrame
        A pandas DataFrame containing the electrophysiological data. Each column 
        in the DataFrame should represent a different channel, named as 'channel_X' 
        where X is the channel number.

    channel : int or 'all', optional
        The specific channel to plot the histogram for. If set to 'all', the function 
        plots histograms for all channels. Defaults to 'all'.

    nbins : int, optional
        The number of bins to use for the histogram. Defaults to 15.

    Returns:
    --------
    None
        The function does not return anything but plots the histograms directly.

    Raises:
    -------
    ValueError
        If the specified channel is not found in the DataFrame, or if an invalid 
        channel specification is provided.
    """

    if channel == 'all':
        # plot historgrams for all channels

        nchannels = len(ephysdf.columns)

        if nchannels == 16:
            fig, axs = plt.subplots(4, 4, figsize=(10, 10))
        elif nchannels == 32:
            fig, axs = plt.subplots(6, 6, figsize=(8, 8))
        else:
            fig, axs = plt.subplots(nchannels, 1, figsize=(6, 3 * nchannels))

        ephysdf.hist(bins=nbins, ax=axs.ravel())
        fig.suptitle('Histograms for all channels')

    elif isinstance(channel, int):
        # Plot histogram for a specific channel
        col_name = 'channel_' + str(channel)
        if col_name in ephysdf.columns:
            ephysdf[col_name].hist(bins=nbins)

        else:
            raise ValueError(f'Channel {channel} not found in DataFrame.')

    else:
        raise ValueError('Invalid channel specification')

    # Show the plot
    plt.show()


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


def plot_avg_infreq_lfp(avg_activity: np.array, save_path: str, freq_range: tuple, ch = 1, yaxis = 'voltage') -> None:

    window_size = avg_activity.shape[1]
    x_middle = window_size // 2

    fig, ax = plt.subplots()
    ax.plot(avg_activity[ch - 1, :], color = 'black')
    ax.set_title(f'Average LFP for {freq_range[0]} - {freq_range[1]} Hz sniff \n Channel {ch}')
    ax.set_xlabel('Time (ms)')
    if yaxis == 'voltage':
        ax.set_ylabel('Voltage (mV)')
    elif yaxis == 'zscore':
        ax.set_ylabel('Z-scored Voltage')
    else:
        raise ValueError('yaxis must be either "voltage" or "zscore"')

    ax.set_xticks([x_middle - window_size/2, x_middle, x_middle + window_size/2])
    ax.set_xticklabels([-window_size/2, '0', window_size/2])

    plt.savefig(os.path.join(save_path, f'Channel_{ch}_avg_lfp.png'))
    plt.close()


def plot_binned_reduced_ephys(reduced_df: pd.DataFrame, n_bins = 10):
    """
    Plots a heatmap of binned reduced electrophysiology data.

    This function takes a DataFrame of reduced electrophysiology (ephys) data and bins it
    into specified number of bins along two components.
    It then calculates the mean frequencies for each bin and plots a heatmap representing these frequencies.
    The bins represent different segments of the reduced components,
    and the heatmap intensity reflects the average frequency in each bin.

    Parameters:
    reduced_df (pd.DataFrame): A DataFrame containing reduced ephys data with components labeled as 'componant_1' and 'componant_2', and a 'frequencies' column.
    n_bins (int, optional): The number of bins to divide each component into for the heatmap. Default is 10.

    Returns:
    None: The function directly plots a heatmap and does not return any value.
    """

    # binning the data
    reduced_df['comp1_bins'] = pd.cut(reduced_df['componant_1'], n_bins, labels = False)
    reduced_df['comp2_bins'] = pd.cut(reduced_df['componant_2'], n_bins, labels = False)

    # grouping and finding mean along frequencies dimension
    grouped = reduced_df.groupby(['comp1_bins', 'comp2_bins'])['frequencies'].mean().reset_index()

    # creating heatmap data with pivot table
    heatmap_data = grouped.pivot(index = 'comp2_bins', columns = 'comp1_bins', values = 'frequencies')

    # plotting heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(heatmap_data, cmap='cubehelix')
    plt.gca().invert_yaxis()
    plt.title('Frequency Heatmap')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


def plot_correlations(correlation_matrix, ephysdf):

    ncolumns = len(correlation_matrix)

    fig, ax = plt.subplots()
    im = ax.imshow(correlation_matrix, cmap = 'seismic')

    ax.set_xticks(np.arange(ncolumns), labels = np.linspace(1, ncolumns, ncolumns, dtype = int))
    ax.set_yticks(np.arange(ncolumns), labels = np.linspace(1, ncolumns, ncolumns, dtype = int))

    for i in range(ncolumns):
        for j in range(ncolumns):
            text = ax.text(j, i, int(correlation_matrix.iloc[i, j] * 100),
                       ha="center", va="center", color="k")
    
    ax.set_title('Channel Correlations ($ \\times 100 $)')
    fig.tight_layout()
    plt.show()






#____________________________________________________________________________________________________RESAMPLING________________________________________________________________________________________________________________________________

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

    # calculating resample factor
    resample_factor = original_rate // target_rate

    # calculating new length of resampled signal
    if 0 != sniff.shape[0] % resample_factor:
        sniff = sniff[:-(sniff.shape[0] % resample_factor)]

    # calculating new length of resampled signal
    print(f'Resampling from length {sniff.shape[0]}')
    new_length = sniff.shape[0]//resample_factor

    # applying decimation to the signal
    resampled_sniff = np.zeros(new_length)
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

    # calculating resample factor
    resample_factor = original_rate // target_rate

    # removing samples from the end of the signal if the length is not divisible by the resample factor
    if 0 != ephys.shape[1] % resample_factor:
        ephys = ephys[:, :-(ephys.shape[1] % resample_factor)]

    # calculating new length of resampled signal
    print(f'Resampling Ephys from length {ephys.shape[1]}')
    new_length = ephys.shape[1]//resample_factor

    # applying decimation to the signal
    resampled_ephys = np.zeros((nchannels, new_length))
    for ch in range(nchannels):
        resampled_ephys[ch, :] = signal.decimate(ephys[ch,:], resample_factor, ftype = 'fir')

    return resampled_ephys


def resample_tracking(tracking:np.array, original_rate = 100, target_rate = 1000) -> np.array:
    '''
    Resample tracking data from an original rate to a target rate using linear interpolation.

    '''

    
    if target_rate % original_rate != 0:
        raise ValueError("Target rate must be a factor of the original rate.")
        return None
    
    # calculating resample factor
    resample_factor = target_rate // original_rate

    # calculating new length of resampled signal
    resampled_tracking = np.zeros((tracking.shape[0], tracking.shape[1] * resample_factor), dtype=np.uint16)

    # applying linear interpolation to the signal
    for i in range(tracking.shape[0]):
        resampled_tracking[i, :] = np.interp(np.arange(0, tracking.shape[1] * resample_factor), np.arange(0, tracking.shape[1]), tracking[i, :])

    return resampled_tracking






#____________________________________________________________________________________________PREPROCESSING SNIFF AND EPHYS________________________________________________________________________________________________________________________________

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


def remove_trend_boxcar(data: np.array, window_size: int = 1000) -> np.array:
    '''
    Removes the trend from a signal using a boxcar filter.

    This function applies a boxcar filter to a signal to remove the trend. The boxcar 
    filter is a simple moving average filter that replaces each point with the average 
    of the points in a window around it. The window size is specified by the user.

    Parameters:
    signal (np.array): The signal to be processed, represented as a NumPy array.
    window_size (int, optional): The size of the window used for the boxcar filter. 
                                 Defaults to 1000.

    Returns:
    np.array: The signal with the trend removed.
    '''

    # applying boxcar filter
    boxcar = np.ones((window_size))/window_size
    smoothed_signal = np.convolve(data, boxcar, mode='same')

    # removing trend from signal
    detrended_signal = data - smoothed_signal

    return detrended_signal


def remove_trend_zscore(data: np.array) -> np.array:
    detrended = stats.zscore(data)
    return detrended


def remove_trend_bandpass(data: np.array, lowcut: float = 0.1, highcut: float = 20, order: int = 5, f = 1000) -> np.array:

    sos_high = signal.butter(order, lowcut, btype='highpass', output = 'sos', fs = f)
    highpassed = signal.sosfiltfilt(sos_high, data)

    sos_low = signal.butter(order, highcut, btype='lowpass', output = 'sos', fs = f)
    lowpassed = signal.sosfiltfilt(sos_low, data)

    bandpassed = signal.sosfiltfilt(sos_high, lowpassed)

    return bandpassed, lowpassed, highpassed


def smooth_waveform(waveform: np.array):
    '''smooths a waveform using a Savitzky-Golay filter'''

    smoothed_waveform = signal.savgol_filter(waveform, 51, 3)
    return smoothed_waveform

def find_condition_mask_inhales(events, len_sniff: int):

    # finding start and end times of free moving
    freemoving_start = events[0,3]
    if events[1,2] != 0:
        freemoving_end = events[1,2]
    else:
        freemoving_end = len_sniff - 1

    # finding floor flip times
    floorflip_start = events[2,2]
    floorflip_end = events[2,3]

    # removing floor flip times from freemoving times if applicable
    if floorflip_start != 0:
        freemoving_mask = np.concatenate((np.arange(freemoving_start, floorflip_start), np.arange(floorflip_end, freemoving_end))).astype(int)
    else:
        freemoving_mask = np.arange(freemoving_start, freemoving_end).astype(int)
    
    # finding initial headfixed start and end times
    headfix_i_start = 0
    if events[0,2] != 0:
        headfix_i_end = events[0,2]
    else:
        headfix_i_end = (events[0,3] - 1000) # subtracting 10 seconds when mouse is headfixed outside arena
    headfix_i_mask = np.arange(headfix_i_start, headfix_i_end)

    # finding final headfixed start and end times
    headfix_f_end = len_sniff - 1
    if events[1,3] != 0:
        headfix_f_start = events[1,3]
    else:
        if events[1,2] != 0:
            headfix_f_start = (events[1,2] + 100)# adding 10 seconds when mouse is headfixed outside arena
        else:
            headfix_f_start = len_sniff - 1


    headfix_f_mask = np.arange(headfix_f_start, headfix_f_end)

    if len(headfix_f_mask) < 4 * 60 * 100:
        headfix_f_mask = []

    # defining headfixed mask
    headfixed_mask = np.round(np.concatenate((headfix_i_mask, headfix_f_mask))).astype(int)



    return freemoving_mask, headfixed_mask



def find_condition_mask(events, sniff, get_bool = True):

    sniff = sniff.flatten()
    len_sniff = sniff.shape[0]

    # finding start and end times of free moving
    freemoving_start = events[0,3]
    if events[1,2] != 0:
        freemoving_end = events[1,2]
    else:
        freemoving_end = len_sniff - 1

    # finding floor flip times
    floorflip_start = events[2,2]
    floorflip_end = events[2,3]

    # removing floor flip times from freemoving times if applicable
    if floorflip_start != 0:
        freemoving_mask = np.concatenate((np.arange(freemoving_start, floorflip_start), np.arange(floorflip_end, freemoving_end))).astype(int)
    else:
        freemoving_mask = np.arange(freemoving_start, freemoving_end).astype(int)
    
    # finding initial headfixed start and end times
    headfix_i_start = 0
    if events[0,2] != 0:
        headfix_i_end = events[0,2]
    else:
        headfix_i_end = (events[0,3] - 1000) # subtracting 10 seconds when mouse is headfixed outside arena
    headfix_i_mask = np.arange(headfix_i_start, headfix_i_end)

    # finding final headfixed start and end times
    headfix_f_end = len_sniff - 1
    if events[1,3] != 0:
        headfix_f_start = events[1,3]
    else:
        if events[1,2] != 0:
            headfix_f_start = (events[1,2] + 100)# adding 10 seconds when mouse is headfixed outside arena
        else:
            headfix_f_start = len_sniff - 1


    headfix_f_mask = np.arange(headfix_f_start, headfix_f_end)

    if len(headfix_f_mask) < 4 * 60 * 100:
        headfix_f_mask = []

    # defining headfixed mask
    headfixed_mask = np.round(np.concatenate((headfix_i_mask, headfix_f_mask))).astype(int)

    if get_bool:
        # definings boolean masks
        bool_length = np.zeros(len(sniff), dtype=bool)
        if freemoving_mask.size != 0:
            bool_length[freemoving_mask] = True
        freemoving_bool = bool_length

        bool_length = np.zeros(len(sniff), dtype=bool)
        if headfixed_mask.size != 0:
            bool_length[headfixed_mask] = True
        headfixed_bool = bool_length

        return freemoving_mask, headfixed_mask, freemoving_bool, headfixed_bool

    else:
        return freemoving_mask, headfixed_mask
#_________________________________________________________________________________________________PREPROCESSING TRACKING________________________________________________________________________________________________________________________________

def find_velocity(tracking: np.array, keypoint = 'centroid', returns = 'magnitude') -> np.array:

    # extracting x and y coordinates
    if keypoint == 'centroid':
        x = tracking[1, :]
        y = tracking[3, :]

    elif keypoint == 'head':
        x = tracking[0,:]
        y = tracking[2,:]

    else:
        raise ValueError("keypoint must be either 'centroid' or 'head'.")
        return None
    
    # calculating velocities
    velocities_x = np.gradient(x)
    velocities_y = np.gradient(y)

    # calculating magnitude of velocities
    velocity = np.sqrt(velocities_x**2 + velocities_y**2)

    # returning desired values
    if returns == 'magnitude':
        return velocity
    elif returns == 'directional':
        return velocities_x, velocities_y
    

def find_acceleration(tracking: np.array, keypoint = 'centroid', returns = 'magnitude') -> np.array:

    # extracting x and y coordinates
    if keypoint == 'centroid':
        x = tracking[1, :]
        y = tracking[3, :]

    elif keypoint == 'head':
        x = tracking[0,:]
        y = tracking[2,:]

    else:
        raise ValueError("keypoint must be either 'centroid' or 'head'.")
    
    # calculating velocities
    accelerations_x = np.gradient(x, 2)
    accelerations_y = np.gradient(y, 2)

    # calculating magnitude of accelerations
    acceleration = np.sqrt(accelerations_x**2 + accelerations_y**2)

    # returning desired values
    if returns == 'magnitude':
        return acceleration
    elif returns == 'directional':
        return accelerations_x, accelerations_y
    

def find_angular_velocity(tracking: np.array) -> np.array:

    body = np.array([tracking[1, :], tracking[3, :]])
    head = np.array([tracking[0, :], tracking[2, :]])

    # finding velocity vectors
    velocity_head = np.gradient(head, axis = 1)
    velocity_body = np.gradient(body, axis = 1)

    # calculating velocity vectors of head with respect to origin
    head_velocity_wrt_body = velocity_head - velocity_body

    # calculating radius vectors
    radii = head - body

    # calculating radial velocity component
    rad_velocity_component = (np.einsum('ij,ij->j', head_velocity_wrt_body, radii) / np.linalg.norm(radii, axis = 0)) * (radii / np.linalg.norm(radii, axis = 0))

    # calculating tangential velocity component
    tan_velocity = head_velocity_wrt_body - rad_velocity_component

    # calculating angular velocity
    cross = np.cross(radii.T, tan_velocity.T)
    radius_squared = np.linalg.norm(radii, axis = 0)**2
    with np.errstate(divide='ignore', invalid='ignore'):
        omega = np.where(radius_squared != 0, cross / radius_squared, np.nan)

    return omega
    
    


#________________________________________________________________________________________________________ALIGNMENT_________________________________________________________________________________________________________________________________________
def find_inhales_2(data: np.array, window_length = 51, polyorder = 5, min_peak_prominance = 0.3, distance = 50, show = False, signal_type = 'sniff', save_figs = False, name = None, f = 1000) -> np.array:

    inhale_times = []

    # smoothing the signal
    data_smooth = signal.savgol_filter(data, window_length, polyorder)
    
    # defining windows to work in
    windows = np.round(np.arange(0, len(data_smooth), (5*f))).astype(int)

    # preallocating zscored signal array
    zsniff = np.zeros(len(windows))

    # looping through windows
    for scan in range(1, len(windows)):

        # defining time stamp
        time_stamp = ((windows[scan - 1]), windows[scan])

        # zscoring the signal
        zsniff = stats.zscore(data_smooth[time_stamp[0]:time_stamp[1]])

        # finding peaks
        [locs, properties] = signal.find_peaks(zsniff, height = (None, None), prominence = min_peak_prominance, distance = distance)

        # finding inhale times and appending to list
        for peak in range(len(locs)):
            inhale = locs[peak]
            if inhale:
                inhale_times.append(inhale + time_stamp[0])

    # plotting
    if show == True:
        plt.figure(figsize=(10, 6))
        plt.plot(inhale_times, data_smooth[inhale_times], 'x', label='Peaks')

        if signal_type == 'sniff':
            plt.plot(data, label='Original Sniff Signal')
            plt.plot(data_smooth, label='Smoothed Sniff Signal')
            plt.title('Sniff Signal and Identified Peaks')

        elif signal_type == 'lfp':
            plt.plot(data, label='Original LFP Signal')
            plt.plot(data, label='Smoothed LFP Signal')
            plt.title('LFP Signal and Identified Peaks\n')
            
        plt.xlabel('Sample')
        plt.ylabel('Signal Amplitude')
        plt.legend()
        plt.show()

    return inhale_times


def find_inhales(data: np.array, window_length = 51, polyorder = 5, min_peak_prominance = 75, distance = 70, show = False, signal_type = 'sniff', save_figs = False, save_path: str = '', name = None) -> np.array:
    '''
    Smooth a  signal using the Savitzky-Golay method and locate inhalation times using peak finding.

    This function first applies a Savitzky-Golay filter to smooth the input signal. 
    It then uses a peak finding algorithm to identify the times of inhalations, which are 
    indicated by prominent peaks in the smoothed signal. Optionally, the function can also 
    plot the original and smoothed signals along with the identified peaks.

    Parameters:
    data (np.array): The signal to be processed, represented as a NumPy array.
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

    sns.set_style('darkgrid')
    smoothed_data = signal.savgol_filter(data, window_length, polyorder)
    locs, properties = signal.find_peaks(smoothed_data, height = (None, None), prominence = min_peak_prominance, distance = distance)
    
    plt.figure(figsize=(10, 6))
    plt.plot(locs, smoothed_data[locs], 'x', label='Peaks')

    if signal_type == 'sniff':
        plt.plot(data, label='Original Sniff Signal')
        plt.plot(smoothed_data, label='Smoothed Sniff Signal')
        plt.title('Sniff Signal and Identified Peaks')

    elif signal_type == 'lfp':
        plt.plot(data, label='Original LFP Signal')
        plt.plot(smoothed_data, label='Smoothed LFP Signal')
        plt.title('LFP Signal and Identified Peaks\n')
        
    plt.xlabel('Sample')
    plt.ylabel('Signal Amplitude')
    plt.legend()

    if show == True:
        plt.show()

    if save_figs == True:
        name = name + '.png'
        plt.savefig(os.path.join(save_path, name))

    plt.close()
    return locs, smoothed_data, properties


def sniff_lock_lfp(locs: np.array, ephys: np.array, window_size = 1000, nsniffs = 512, beg = 3000, method = 'zscore') -> np.array:
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
    sniff_activity (np.array): A 3D NumPy array with shape (nsniffs, window_size, nchannels). Each 'slice' of this array 
              represents the z-scored LFP activity around a single sniff event for all channels.
    loc_set (np.array): An array of indices where inhalation peaks are located.

    Raises:
    ValueError: If the 'locs' array does not contain enough data after the specified 'beg' index for the required number of sniffs.
    '''


    # finding number of channels
    nchannels = ephys.shape[0]


    # finding the set of inhalation times to use
    if nsniffs == 'all':
        loc_set = locs[50:-50]
        nsniffs = len(loc_set)
    elif isinstance(nsniffs, int):
        first_loc = np.argmax(locs >= beg)
        loc_set = locs[first_loc: first_loc + nsniffs]
    else:
        raise ValueError("nsniffs must be either 'all' or an integer.")

    # checking if locs array has enough data for the specified range
    if isinstance(nsniffs, int):
        if len(loc_set) < nsniffs:
            raise ValueError("locs array does not have enough data for the specified range.")
        
    # propogates an nx2 array containing times half the window size in both directions from inhalation times
    windows = np.zeros((nsniffs, 2), dtype=int)
    for ii in range(nsniffs):
        win_beg = loc_set[ii] - round(window_size/2)
        win_end = loc_set[ii] + round(window_size/2)
        windows[ii] = [win_beg, win_end]

    if method == 'zscore':
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

    elif method == 'none':
        sniff_activity = np.zeros((nchannels, nsniffs, window_size))
        for ii in range(nsniffs):
            for ch in range(nchannels):
                win_beg, win_end = windows[ii]
                data = ephys[ch, win_beg:win_end]
                if len(data) < window_size:
                    data = np.pad(data, (0, window_size - len(data)), mode = 'constant', constant_values = 0)
                    print('!!! padding !!!')
                sniff_activity[ch,ii,:] = data

    return sniff_activity, loc_set


def peak_finder_validation(sniff: np.array):
    '''visual validation for peak finder. Plots raw and smoothed signal and with peaks'''
    inhales, smoothed_sniff = find_inhales(sniff)
    plt.plot(sniff)
    plt.plot(smoothed_sniff)
    plt.plot(inhales, sniff[inhales], 'x')
    plt.show()






#_________________________________________________________________________________________________________SORTING & COMBINING_____________________________________________________________________________________________________________________________________________
    


def sort_lfp(sniff_activity, locs):
    '''sorts the sniff locked lfp trace by sniff frequency'''

    # finding data shape
    nchannels = sniff_activity.shape[0]
    nsniffs = sniff_activity.shape[1]
    window_size = sniff_activity.shape[2]
    
    sorted_activity = np.zeros((nchannels, nsniffs-1, window_size))
    
    # finding sniff frequencies by inhalation time differences (we lose the last sniff)
    freqs = np.diff(locs)

    # sorting the ephys data and frequency values according to these times
    sort_indices = np.argsort(freqs)
    sorted_activity[:, :, :] = sniff_activity[:, sort_indices, :]
    sorted_freqs = freqs[sort_indices]
    sorted_freqs = 1 / (sorted_freqs / 1000)

    return sorted_activity, sorted_freqs


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
    


def collect_snifflocked_ephys(ephys: np.array, inhales: np.array) -> np.array:
    '''collects the ephys voltage at each channel at each inhalation'''

    # finding data shape
    nchannels = ephys.shape[0]
    nsniffs = inhales.shape[0]

    # propogating an nxm array containing the ephys voltage at each channel at each inhalation
    sniff_activity = np.zeros((nchannels, nsniffs - 1))
    for ii in range(nsniffs - 1):
        sniff_activity[:,ii] = ephys[:, inhales[ii]]
    
    # propogating an array containing the inter-sniff time
    inter_sniff_time = np.diff(inhales)

    return sniff_activity , inter_sniff_time



def bin_sniff_times(inter_sniff_time: np.array, bin_size = 200) -> np.array:
    '''bins inter-sniff times into bins of size bin_size'''

    # finding number of bins
    nsniffs = inter_sniff_time.shape[0]

    # propogating an array containing the bin number for each inter-sniff time
    binned_sniff_times = np.floor(inter_sniff_time / bin_size).astype(int)

    return binned_sniff_times



def pull_infreq_lfp(sniff_activity: np.array, freqs: np.array, freq_range: tuple):
    '''pulls the ephys voltage at each channel at each inhalation for sniffs with frequencies in freq_range'''

    # finding data shape
    nchannels = sniff_activity.shape[0]
    nsniffs = sniff_activity.shape[1]
    window_size = sniff_activity.shape[2]

    # Propagating an nxm array containing the ephys voltage at each channel at each inhalation
    infreq_sniff_activity = np.full((nchannels, nsniffs, window_size), np.nan)

    # Creating a mask for valid sniffs based on frequency range
    valid_sniffs_mask = np.array([freq_range[0] <= freq <= freq_range[1] for freq in freqs])

    for ii in range(nsniffs):
        if valid_sniffs_mask[ii]:  # Using the mask to check frequency
            infreq_sniff_activity[:, ii, :] = sniff_activity[:, ii, :]

    # Filter out columns where all values are NaN
    valid_columns_mask = ~np.isnan(infreq_sniff_activity).all(axis=(0, 2))

    # Applying the column mask to infreq_sniff_activity and the sniffs mask to freqs
    infreq_sniff_activity = infreq_sniff_activity[:, valid_columns_mask, :]
    valid_freqs = freqs[valid_sniffs_mask]

    return infreq_sniff_activity, valid_freqs



def avg_infreq_lfp(infreq_sniff_activity: np.array) -> np.array:

    nchannels = infreq_sniff_activity.shape[0]
    window_size = infreq_sniff_activity.shape[2]

    avg_activity = np.zeros((nchannels, window_size))
    avg_activity = np.mean(infreq_sniff_activity, axis = 1)

    return avg_activity



#_____________________________________________________________________________________________________FREQUENCY ANALYSIS________________________________________________________________________________________________________________________________



def PSD_sniff(sniff: np.array, show: bool = False) -> np.array:
    '''
    Power Spectral Density transform on sniff signal
    '''
        

    frequencies, PSD = signal.welch(sniff, 1000)
    
    if show:
        fig, ax = plt.subplots()
        ax.plot(frequencies, PSD)
        plt.gca().set_yscale('log')
        ax.set(xlabel='frequency (Hz)', ylabel='Power', title='Power Spectral Density',
               xlim=[0, 50])
        plt.show()

    return frequencies, PSD



def PSD_ephys(ephys: np.array, show = False) -> np.array:

    # finding number of channels
    nchannels = ephys.shape[0]

    # calculating PSD for each channel
    frequencies = None
    PSDs = None
    for ch in range(nchannels):
        f, p = signal.welch(ephys[ch,:], 1000, nperseg = 100_000, noverlap = 10_000)

        # saving frequencies and PSDs for each channel
        if frequencies is None:
            num_freqs = len(f)
            frequencies = np.zeros((nchannels, num_freqs))
            PSDs = np.zeros((nchannels, num_freqs))
        frequencies[ch,:] = f
        PSDs[ch,:] = p

    
    if show:
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
        elif nchannels == 1:
            fig, axs = plt.subplots(1, 1, figsize=(6, 6))
            fig.subplots_adjust(hspace=0.1, wspace=0.1)
        else:
            raise ValueError("Number of channels must be 16, 32, or 64.")
        
        fig.suptitle('LFP Power Spectrum', fontsize=16)
        
        for ch in range(nchannels):
            if nchannels == 16:
                ax = axs[ch // 4, ch % 4]
            elif nchannels == 32:
                ax = axs[ch // 6, ch % 6]
            elif nchannels ==64:
                ax = axs[ch // 8, ch % 8]
            elif nchannels == 1:
                ax = axs

            # Plotting each channel
            cax = ax.plot(frequencies[ch, :], PSDs[ch, :])
            
            ax.set_title(f'Channel {ch + 1}')
            ax.set_xlabel('frequency (Hz)')
            ax.set_ylabel('Power')
            ax.set_xlim([0, 200])

        plt.gca().set_yscale('log')
        plt.show()

                
        

    return frequencies, PSDs
        

def multitaper_PSD_ephys(data: np.array, frequency_range: int = 500, sampling_rate = 1000, single_taper_duration = 0.4, overlap = 0.1, ntapers = 5) -> np.array:

    import multitaper as mt

    # calculate sampling interval
    sampling_interval = 1 / sampling_rate

    # Estimating PSD for each channel using Thomson's multitaper method
    f, t, quad, MT = mt.mtspec.spectrogram(data, sampling_interval, single_taper_duration, nw = 3.5, olap = overlap, kspec = ntapers, fmax = frequency_range)

    return f, t, quad, MT


def multitaper_cross_spectrum(data1: np.array, data2: np.array, sampling_rate = 1000, ntapers = 7) -> np.array:
    
    import multitaper as mt

    # calculate sampling interval
    sampling_interval = 1 / sampling_rate

    mt_cross = mt.MTCross(data1, data2, nw = 3.5, kspec = ntapers, dt = sampling_interval, nfft = 500)

    cross = mt_cross.Sxy
    xspec = mt_cross.Sxx
    yspec = mt_cross.Syy
    freqs = mt_cross.freq

    cross = np.abs(cross)
    xspec = np.abs(xspec)
    yspec = np.abs(yspec)

    return cross, xspec, yspec, freqs


#_________________________________________________________________________________________________________CIRCULAR SHIFT AND NULL DISTRIBUTIONS_____________________________________________________________________________________________________________________________________________


def circular_shift(ephys: np.array, nshifts: int = 1000, method: str = 'sample', min_shift = 1000) -> np.array:
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
            random_shift = np.random.randint(min_shift, signal_length)
            current_roll = np.roll(ephys, random_shift, axis = 1)
            circ_ephys[:,:,ii] = current_roll

    return circ_ephys


def create_circular_null(circ_ephys: np.array, locs: np.array, window_size: int = 1000, beg: int = 3000, sort = True) -> np.array:

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
    nsniffs = circ_ephys.shape[1]


    circular_null = np.zeros((nchannels, nsniffs, window_size, nshifts), dtype = np.float64)

    # propogating a 4d array containing the circul shifted sniff locked signals i.e. the null distribution
    for shift in range(nshifts):
        circular_null[:,:,:,shift], _ = sniff_lock_lfp(locs, circ_ephys[:,:,shift], nsniffs = nsniffs, beg = beg)

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


def sniff_lock_std(locs: np.array, circular_null: np.array, beg: int = 3000) -> np.array:
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


    # finding number of channels
    nchannels = circular_null.shape[0]
    nsniffs = circular_null.shape[1]
    window_size = circular_null.shape[2]

    # finding first inhalation time
    first_loc = np.argmax(locs >= beg)
    print(f'first inhale is #{first_loc}')

    # finding the set of inhalation times to use
    if nsniffs == 'all':
        loc_set = locs[first_loc:]
        nsniffs = len(loc_set)
    elif isinstance(nsniffs, int):
        loc_set = locs[first_loc: first_loc + nsniffs]
    else:
        raise ValueError("nsniffs must be either 'all' or an integer.")
    print(f'last inhale is #{len(loc_set) + first_loc}')

    # checking if locs array has enough data for the specified range
    if isinstance(nsniffs, int):
        if len(loc_set) < nsniffs:
            raise ValueError("locs array does not have enough data for the specified range.")
    
    
    # propogates an nx2 array containing times half the window size in both directions from inhalation times
    windows = np.zeros((nsniffs, 2), dtype=int)
    for ii in range(nsniffs):
        win_beg = loc_set[ii] - round(window_size/2)
        win_end = loc_set[ii] + round(window_size/2)
        windows[ii] = [win_beg, win_end]

    # finds and saves lfp zscore from null distribution in each channel for each moment around each inhalaion
    print(circular_null.shape)
    sniff_activity = np.zeros((nchannels, nsniffs, window_size))
    print(sniff_activity.shape)
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


def create_circular_signal_distributions(circ_ephys: np.array, show = False) -> np.array:
    
    nchannels = circ_ephys.shape[0]
    signal_length = circ_ephys.shape[1]

    if show:
        plt.figure(figsize=(10, 6))
        for i in range(3):
            plt.plot(circ_ephys[0,:,i])
        plt.title('shifted Signals')
        plt.xlabel('Sample')
        plt.ylabel('Signal Amplitude')
        plt.show()

    signal_distributions = np.zeros((nchannels, signal_length))
    for ch in range(nchannels):
        signal_distributions[ch,:] = np.mean(circ_ephys[ch,:,:], axis = 1)

    if show:
        plt.figure(figsize=(10, 6))
        plt.plot(signal_distributions[0,:])
        plt.title('Circularly Shifted Signal Distributions')
        plt.xlabel('Sample')
        plt.ylabel('Signal Amplitude')
        plt.show()
        

#_________________________________________________________________________________________________________PANDAS_____________________________________________________________________________________________________________________________________________


def ephys2pandas(ephys: np.array) -> pd.DataFrame:

    # finding column names
    nchannels = ephys.shape[0]
    column_names = []
    for ch in range(nchannels):
        column_names.append('channel_' + str(ch))

    ephysdf = pd.DataFrame(ephys.T, columns = column_names)

    return ephysdf


def show_correlations(ephysdf: pd.DataFrame, channel):
    correlation_matrix = ephysdf.corr()

    if channel == 'all':
        return correlation_matrix
    
    elif isinstance(channel, int):
        return(correlation_matrix['channel_' + str(channel)])





#_________________________________________________________________________________________________________CLUSTERING_________________________________________________________________________________________________________________________________________



def kmeans_cluster_learn(plot = True):
    features, true_labels = make_blobs(n_samples = 200, centers = 3, cluster_std = 2.75, random_state = 42)

    scalar = StandardScaler()
    scaled_features = scalar.fit_transform(features)

    # finding appropriate number of clusters 

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42,}

    # elbow method
    sse = []

    for k in range(1,11):
        kmeans = KMeans(n_clusters = k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)
    
    if plot:
        plt.style.use('fivethirtyeight')
        plt.plot(range(1,11), sse)
        plt.xticks(range(1,11))
        plt.xlabel('Number of Clusters')
        plt.ylabel('SSE')
        plt.show()
    
    kl = KneeLocator(range(1,11), sse, curve='convex', direction='decreasing')
    print(f'elbow at k = ',  kl.elbow)

    # silhouette method
    silhouette_coeff = []

    for k in range(2,11):
        kmeans = KMeans(n_clusters = k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_coeff.append(score)
    
    if plot:
        plt.style.use('fivethirtyeight')
        plt.plot(range(2,11), silhouette_coeff)
        plt.xticks(range(2,11))
        plt.xlabel('Number of Clusters')
        plt.ylabel("Silhouette Coefficient")
        plt.show()



def cluster_KMeans(ephys: np.array, binned_freqs: np.array, n_clusters = 3, nPCs = 2, assume_norm = False, reduce_first = True, plot = True, n_init = 50, max_iter = 500):

    # reshape ephys
    ephys = ephys.T
    
    # preprocessing rescales values from each channel. if reduce_first is True, reduces to nPCs principle componants,
    # otherwise, only rescales values
    scaler = StandardScaler() if assume_norm else MinMaxScaler()
    if reduce_first:
        preprocessor = Pipeline(
            [
                ('scaler', scaler),
                ('pca', PCA(n_components = nPCs))
            ]
        )
    else:
        preprocessor = Pipeline(
                [
                    ('scaler', scaler),
                ]
            )

    # cluster implements sklearn KMeans algorithm for n_clusters (k value)
    clusterer = Pipeline([
                ('kmeans', KMeans(
                n_clusters = n_clusters,
                init = 'k-means++',
                n_init=n_init,
                max_iter = max_iter,),),])

    # Combining preprocessing and clusterer into single pipeline
    pipe = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('clusterer', clusterer)
        ]
    )

    # fitting pipeline to data
    pipe.fit(ephys)

    preprocessed_data = pipe['preprocessor'].transform(ephys)

    predicted_centroids = pipe['clusterer']['kmeans'].labels_

    # calculating silhouette score
    sil_score = silhouette_score(preprocessed_data, predicted_centroids)

    # calculating sum of squared errors
    sse = pipe['clusterer']['kmeans'].inertia_

    # calculating adjusted rand index
    ari = adjusted_rand_score(binned_freqs, predicted_centroids)
    
    # creating pandas DataFrame for visualization
    if reduce_first:
        column_names = []
        for ii in range(nPCs):
            column_names.append('componant_' + str(ii + 1))
        pcadf = pd.DataFrame(
            preprocessed_data, 
            columns = column_names
        )
    else:
        # reducing data to 2 principle componants
        reduced_data = PCA(n_components=2).fit_transform(preprocessed_data)
        # creating DataFrame
        pcadf = pd.DataFrame(
            reduced_data,
            columns = ['componant_1', 'componant_2']
        )


    pcadf['predicted_centroids'] = pipe['clusterer']['kmeans'].labels_
    if plot:
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(8,8))

        scat = sns.scatterplot(
            x = 'componant_1', 
            y = 'componant_2',
            alpha = 0.5,
            s=50,
            data=pcadf,
            hue = 'predicted_centroids',
            palette = 'Set2'
        )

        scat.set_title('Clustering Results \nPC space')

        plt.show()

    return pcadf, sil_score, sse, ari



def kmeans_find_nPCs(ephys: np.array, binned_freqs, n_clusters = 3, assume_norm = False, plot = False, nPCs = 10):
    
    sil_score = []
    sse = []
    ari = []
    for PCs in range(nPCs):
        print('kmeans with {} principle componants'.format(PCs + 1))
        _, sil, s, a = cluster_KMeans(ephys, binned_freqs, nPCs = PCs + 1, n_clusters = n_clusters, assume_norm=assume_norm, plot = plot)
        sil_score.append(sil)
        sse.append(s)
        ari.append(a)

    # plot silhouette score, sse, and ari
        
    componant_range = range(1, nPCs + 1)
    plt.style.use('fivethirtyeight')

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    # plot silhouette score
    ax1.plot(componant_range, sil_score)
    ax1.set_xlabel('Number of componants')
    ax1.set_ylabel('Silhouette Score')

    # plot Sum of Squared Errors
    ax2.plot(componant_range, sse)
    ax2.set_xlabel('Number of componants')
    ax2.set_ylabel('SSE')

    # plot Adjusted Rand Index
    ax3.plot(componant_range, ari)
    ax3.set_xlabel('Number of componants')
    ax3.set_ylabel('ARI')
    

    plt.suptitle('KMeans Clustering Performance')

    plt.show()

    return


def kmeans_find_k(ephys: np.array, binned_freqs: np.array, max_k =10, reduce_first = True, assume_norm = False, plot = False):
    '''finds the optimal number of clusters for kmeans clustering'''

    # define arrays to hold silhouette score and SSE
    sil_score = []
    sse = []
    ari = []

    # run kmeans for k = 2 to k = max_k
    for k in range(2, max_k):
        print('kmeans with {} clusters'.format(k))
        _, sil, s, a = cluster_KMeans(ephys, binned_freqs, n_clusters=k, assume_norm=assume_norm, reduce_first = reduce_first, plot = plot)
        sil_score.append(sil)
        sse.append(s)
        ari.append(a)

    # plot silhouette score and SSE
    cluster_range = range(2, max_k)
    plt.style.use('fivethirtyeight')

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    # plot silhouette score
    ax1.plot(cluster_range, sil_score)
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Silhouette Score')

    # plot Sum of Squared Errors
    ax2.plot(cluster_range, sse)
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('SSE')

    # plot Adjusted Rand Index
    ax3.plot(cluster_range, ari)
    ax3.set_xlabel('Number of clusters')
    ax3.set_ylabel('ARI')
    

    plt.suptitle('KMeans Clustering Performance')

    plt.show()



#_________________________________________________________________________________________________________DIMENSIONALLITY REDUCTION_____________________________________________________________________________________________________________________________________________


def UMAP_reduce_scatter(ephys: np.array):
    '''
    reduces ephys data to 2 componants using Uniform Manifold Aproximation Projection (UMAP) and plots the results
    '''

    # reshape ephys
    ephys = ephys.T
    
    # normalizing data
    scaled_ephys = StandardScaler().fit_transform(ephys)

    # reducing data to 2 componants
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(scaled_ephys)

    # creating pandas DataFrame for visualization
    pUMAP = pd.DataFrame(
        embedding, 
        columns = ['componant_1', 'componant_2']
    )

    # plotting the DataFrame
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8,8))

    scat = sns.scatterplot(
        x = 'componant_1', 
        y = 'componant_2',
        alpha = 0.4,
        s=50,
        data=pUMAP,
    )

    scat.set_title('Clustering Results \nUMAP space')

    plt.show()



def reduce_heat(ephys: np.array, method = 'PCA'):
    '''
    reduces ephys data to 2 componants using Uniform Manifold Aproximation Projection (UMAP) and plots the results as a heatmap
    '''

    # reshape ephys
    ephys = ephys.T

    # normalizing data
    scaled_ephys = StandardScaler().fit_transform(ephys)

    # reducing data to 2 componants
    if method == 'UMAP':
        reducer = umap.UMAP()
    elif method == 'PCA':
        reducer = PCA(n_components=2)
    else:
        raise ValueError('invalid method')
    
    embedding = reducer.fit_transform(scaled_ephys)

    # creating pandas DataFrame for visualization
    pred = pd.DataFrame(
        embedding, 
        columns = ['componant_1', 'componant_2']
    )

    # plotting the DataFrame with continuous density heatmap
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8,8))

    sns.kdeplot(
        x= pred['componant_1'],
        y = pred['componant_2'],
        cmap = 'RdPu',
        fill = True,
        gridsize = 1000,
        bw_adjust=0.5,
        cut = 4,
        clip = (-10, 10),
        cbar = True,
        )
    
    if method == 'UMAP':
        plt.title('Clustering Results \nUMAP space')
    elif method == 'PCA':
        plt.title('Clustering Results \nPCA space')
    else:
        raise ValueError('invalid method')

    plt.xlabel('componant 1')
    plt.ylabel('componant 2')

    plt.show()



def reduce_snifflocked_lfp(sniff_activity: np.array, frequencies: np.array, reduce_method = 'PCA',
        assume_norm = True, plot_method = 'scatter', cbar_type = 'continuous', clip = 14, cbar_scale = 'log'):
    '''
    Performs dimensionality reduction on sniff-locked LFP (Local Field Potential) data using PCA 
    (Principal Component Analysis) or UMAP (Uniform Manifold Approximation and Projection), visualizes 
    the results, and returns the reduced data.

    Parameters:
    sniff_activity (np.array): Multidimensional array of sniff-locked LFP data.
    frequencies (np.array): Array of frequency data corresponding to `sniff_activity`.
    reduce_method (str): Method for dimensionality reduction, either 'PCA' or 'UMAP'. Defaults to 'PCA'.
    assume_norm (bool): If True, data will be normalized using StandardScaler, otherwise MinMaxScaler is used. Defaults to True.
    plot_method (str): Type of plot for visualization, either 'scatter' or 'heatmap'. Defaults to 'scatter'.
    cbar_type (str): Type of colorbar to use, either 'continuous' or 'discrete'. Defaults to 'continuous'.

    Returns:
    pandas.DataFrame: A DataFrame `reduced_df` containing the reduced components from the LFP data 
    along with the corresponding frequencies.

    The function transforms the LFP data using the specified dimensionality reduction method and 
    normalizes it if required. It then creates a scatter plot or heatmap based on the `plot_method`. 
    The `cbar_type` determines the style of the colorbar representing the `frequencies`. The transformed
    and reduced data is returned as a pandas DataFrame.
    '''

    #reshape ephys
    ephys = sniff_activity.T

    # normaling ephys data
    if assume_norm:
        scaled_ephys = StandardScaler().fit_transform(ephys)
    else:
        scaled_ephys = MinMaxScaler().fit_transform(ephys)

    
    #redcuing ephys data using PCA or UMAP
    if reduce_method == 'PCA':
        reducer = PCA(n_components=2)
    elif reduce_method == 'UMAP':
        reducer = umap.UMAP()
    else:
        raise ValueError('invalid method')
    reduced_ephys = reducer.fit_transform(scaled_ephys)

    
    # creating pandas DataFrame for visualization
    column_names = ['componant_1', 'componant_2']
    reduced_df = pd.DataFrame(
        reduced_ephys,
        columns = column_names
    )

    # adding frequencies to DataFrame
    frequencies = np.clip(frequencies, 1, clip)
    reduced_df['frequencies'] = frequencies

    # finding unique bins
    unique_bins = np.sort(reduced_df['frequencies'].unique())

    # plotting the DataFrame
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8,8))

    # setting colorbar scale
    if cbar_scale == 'log':
        norm = LogNorm(vmin=reduced_df['frequencies'].min(), vmax=reduced_df['frequencies'].max())
    elif cbar_scale == 'linear':
        norm = Normalize(vmin=reduced_df['frequencies'].min(), vmax=reduced_df['frequencies'].max())
    else:
        raise ValueError('invalid cbar_scale')

    # plotting scatterplot with colors associated with sniff frequency
    if cbar_type == 'continuous':
        cmap = 'cubehelix'
        if plot_method == 'scatter':
            ax = sns.scatterplot(
            x = 'componant_1', 
            y = 'componant_2',
            hue = 'frequencies',
            palette = sns.color_palette(cmap, as_cmap=True),
            alpha = 0.9,
            s=50,
            data=reduced_df,
            legend = False,
        )

        # adding colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, ticks=[reduced_df['frequencies'].min(), reduced_df['frequencies'].max()])
        cbar.set_label('sniffs per second (log)')
        

    elif cbar_type == 'discrete':
        cmap = plt.cm.get_cmap('viridis', len(unique_bins))
        colors = cmap(np.linspace(0, 1, len(unique_bins)))
        color_dict = dict(zip(unique_bins, colors))
        if plot_method == 'scatter':
            ax = sns.scatterplot(
            x = 'componant_1', 
            y = 'componant_2',
            hue = 'frequencies',
            palette = color_dict,
            alpha = 0.9,
            s=50,
            data=reduced_df,
            legend = False,
        )
        boundaries = np.arange(len(unique_bins) + 1) - 0.5
        norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(len(unique_bins)))
        cbar.set_ticklabels(unique_bins)
        cbar.set_label('Frequency bin')

    # plotting heatmap with continuous density
    elif plot_method == 'heatmap':
        ax = sns.kdeplot(
            x= reduced_df['componant_1'],
            y = reduced_df['componant_2'],
            cmap = 'inferno',
            fill = True)
        
    else: 
        raise ValueError('invalid method')
    

    
    # setting title and labels
    if reduce_method == 'PCA':
        plt.title('Clustering Results in PCA space')
    elif reduce_method == 'UMAP':
        plt.title('Clustering Results in UMAP space')
    plt.xlabel('componant_1')
    plt.ylabel('componant_2')
    plt.show()
    
    return reduced_df
