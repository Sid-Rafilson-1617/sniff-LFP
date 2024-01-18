"""
Higher order functions for analysis of local field potential, thermistor sniff, and behavioral data. 

Requires core library.

By: Sid Rafilson
Contributors: Nate Hess
Primary Investigator: Matt Smear
"""

from core import *
import numpy as np


#____________________________________________________________PREPROCESS___________________________________________________________

def preprocess(sniff_file, ephys_file, tracking_file, num_samples = -1, start = 0, stop = 0, nchannels = 16, channels = None, remove_artifacts = False, resample = True, no_load = 'tracking'):
    '''
    Preprocesses sniff and electrophysiology (ephys) data for further analysis.

    This function involves loading data, converting data types, optionally removing artifacts, 
    and optionally resampling the data.

    Parameters:
    sniff_file (str): File path to the sniff data file.
    ephys_file (str): File path to the electrophysiology data file.
    num_samples (int): Number of samples to load from the files. If -1, all samples are loaded. Defaults to -1.
    nchannels (int): Number of channels in the ephys data. Defaults to 16.
    remove_artifacts (bool): If True, artifacts will be removed from the data. Defaults to False.
    resample (bool): If True, the data will be resampled. Defaults to True.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - sniff (np.array): The preprocessed sniff data.
        - ephys (np.array): The preprocessed electrophysiology data.

    The function performs the following steps:
    1. Loading data from the specified files.
    2. Converting the data to 32-bit integers.
    3. Optionally removing artifacts from the data if 'remove_artifacts' is True.
    4. Optionally resampling the data for consistency if 'resample' is True.
    '''

    print('loading data...')
    sniff = load_sniff(sniff_file, num_samples=num_samples, start = start, stop = stop)
    ephys = load_ephys(ephys_file, num_samples=num_samples, start = start, stop = stop, nchannels= nchannels)
    if no_load != 'tracking':
        tracking = load_tracking(tracking_file, num_samples=num_samples, start = start, stop = stop)
        tracking = tracking.astype(np.float32)
        tracking = np.round(tracking, 0)
        tracking = resample_tracking(tracking, original_rate = 100, target_rate = 1000)
    
    

    print('converting data type...')
    sniff = sniff.astype(np.int32)
    ephys = ephys.astype(np.int32)
    

    

    if remove_artifacts:
        print('removing artifact...')
        ephys = remove_jumps_ephys(ephys)
        sniff = remove_jumps_sniff(sniff)

    if resample:
        print('resampling...')
        sniff = resample_sniff(sniff)
        ephys = resample_ephys(ephys, nchannels=16)

    if isinstance(channels, tuple):
            ephys = ephys[channels, :]
        
    if no_load != 'tracking':
        return ephys, sniff, tracking
    else:
        return ephys, sniff
    


#____________________________________________________________SNIFF RASTER___________________________________________________________
    
def make_sniff_raster(ephys: np.array, sniff: np.array, beg: int = 1000, nsniffs: int = 200, window_size: int = 1000, method: str = 'simple', plot: bool = False, shifts: int = 10):

    '''
    Analyzes and optionally plots sniff aligned local field potential data using specified methods.

    The function involves identifying inhale events in sniff data, processing LFP data 
    based on the chosen method (simple or circular shift), and optionally plotting the 
    results.

    Parameters:
    sniff (np.array): Sniff signal to be analyzed.
    ephys (np.array): Electrophysiology data corresponding to the sniff data.
    beg (int): Beginning time index for analysis. Defaults to 1000.
    nsniffs (int): Number of sniffs to consider for the analysis. Defaults to 200.
    window_size (int): Size of the window for analysis around each sniff event. Defaults to 1000.
    method (str): Method for analysis, either 'simple' or 'circular_shift'. Defaults to 'simple'.
    plot (bool): If True, the results will be plotted. Defaults to True.
    shifts (int): Number of shifts to use in the circular shift method. Defaults to 10.

    Returns:
    tuple: A tuple containing two elements:
        - sorted_activity (array-like): The sorted LFP activity based on the sniff data.
        - freqs (array-like): Frequencies corresponding to the sorted LFP activity.

    The function first finds inhale peaks in the sniff data. Depending on the 'method' parameter,
    it either processes the data using a simple z-scoring of signal segments or creates null 
    distributions using a circular shift approach. Finally, it optionally plots the sorted LFP 
    data if 'plot' is True.
    '''

    inhales, _, _ = find_inhales(sniff, show = False)

    # simple method z-scores each 1000ms signal segment, independent from other segments
    if method == 'simple':
        sniff_activity, loc_set = sniff_lock_lfp(inhales, ephys, window_size=window_size, beg = beg, nsniffs = nsniffs)
        sorted_activity, freqs = sort_lfp(sniff_activity, loc_set)
    
    # circular shift method creates null distributions at each time-lag, for each sniff, in each channel, by circularly shifting the signal
    elif method == 'circular_shift':
        circular_ephys = circular_shift(ephys, shifts)
        nulls = create_circular_null(circular_ephys, inhales, nsniffs = nsniffs, window_size = window_size, beg = beg)
        sniff_activity, loc_set = sniff_lock_std(inhales, nulls, beg = beg)
        sorted_activity, freqs = sort_lfp(sniff_activity, loc_set)

    elif method == 'circular_shift_simple':
        circular_ephys = circular_shift(ephys, shifts, method='random')
        signal_distributions = create_circular_signal_distributions(circular_ephys, show = False)

    elif method == 'none':
        sniff_activity, loc_set = sniff_lock_lfp(inhales, ephys, window_size=window_size, beg = beg, nsniffs = nsniffs, method = 'none')
        sorted_activity, freqs = sort_lfp(sniff_activity, loc_set)

    else:
        raise ValueError('invalid method specified')

    # plotting
    if plot:
        plot_snifflocked_lfp(sorted_activity, freqs, show_y=True, show_x= False)

    return sorted_activity, freqs



#____________________________________________________________DIMENSIONALLITY REDUCTION___________________________________________________________


def visualize_reduced(ephys: np.array, sniff: np.array, reduce_method: str = 'UMAP', assume_norm: bool = False):
    """
    Visualizes reduced electrophysiology (ephys) data aligned with inhalation times.

    This function first identifies inhalation times from sniff data. It then selects ephys data aligned with these inhalation times.
    The inter-sniff times are converted to frequencies, and the ephys data is reduced to 2 dimensions
    using a specified dimensionality reduction method (default is UMAP).
    The function returns a DataFrame of the reduced ephys data.

    Parameters:
    ephys (np.array): Array of electrophysiology data.
    sniff (np.array): Array of sniff data, used to synchronize with ephys data.
    reduce_method (str, optional): Method used for dimensionality reduction. Default is 'UMAP'.
    assume_norm (bool, optional): If True, assume data is normalized before reduction. Default is False.

    Returns:
    pd.DataFrame: A DataFrame containing the 2-dimensional reduced ephys data.
    """

    # finding inhalation times
    inhales, _ = find_inhales(sniff, show = False)

    # finding only ephys data which is aligned with inhalation times
    sniff_activity, inter_sniff_time = collect_snifflocked_ephys(ephys, inhales)

    # converting inter-sniff time to frequency
    freqs = 1000/inter_sniff_time

    # reducing ephys data to 2 dimensions
    reduced_df = reduce_snifflocked_lfp(sniff_activity, freqs, reduce_method= reduce_method, assume_norm = assume_norm, plot_method= 'scatter')

    return reduced_df


def plot_sniff_freq_binned_heatmap(ephys, sniff, n_bins = 25, reduce_method= 'UMAP', assume_norm = False):
    """
    Plots a heatmap of binned frequencies based on reduced electrophysiology (ephys) data aligned with sniff data.

    This function integrates the processes of visualizing reduced ephys data (aligned with sniff data) and plotting a heatmap
    of the binned frequencies. It first reduces the ephys data to two dimensions using a specified dimensionality reduction method
    and then bins the reduced data into a specified number of bins.
    A heatmap is plotted to show the distribution of frequencies across the binned reduced data components.

    Parameters:
    ephys: Array-like or pd.DataFrame containing electrophysiology data.
    sniff: Array-like or pd.DataFrame containing sniff data, used to align with ephys data.
    n_bins (int, optional): The number of bins for dividing the reduced data components in the heatmap. Default is 25.
    reduce_method (str, optional): Method used for dimensionality reduction. Default is 'UMAP'.
    assume_norm (bool, optional): If True, assumes that the data is normalized before reduction. Default is False.

    Returns:
    None: The function directly plots a heatmap and does not return any value.
    """

    reduced_df = visualize_reduced(ephys, sniff, reduce_method= reduce_method, assume_norm = assume_norm)
    plot_binned_reduced_ephys(reduced_df, n_bins = n_bins)



#____________________________________________________________KMEANS CLUSTERING___________________________________________________________

def kmeans_explore(sniff_file, ephys_file):
    """
    Explores k-means clustering of sniff-locked electrophysiology (ephys) data.

    This function preprocesses the sniff and ephys data, finds inhalation times, and then explores k-means clustering
    parameter choices (number of principle componants and number of clusters). Plots are generated to visualize the
    silhouette score, summed squared error, and adjusted random index for different parameter choices.
    
    """

    # preprocessing data
    ephys, sniff = preprocess(
        sniff_file, ephys_file, clip = 250, num_samples = -1, nchannels=16, remove_artifacts = True, resample = True)

    # finding inhalation times
    inhales, _ = find_inhales(sniff, show = False)

    # finding only ephys data which is aligned with inhalation times
    sniff_activity, lags = collect_snifflocked_ephys(ephys, inhales)

    # converting inter-sniff time to frequency
    freqs = 1000/lags

    # binning frequencies
    binned_freqs = bin_sniff_times(freqs, bin_size = 4)


    kmeans_find_k(sniff_activity, binned_freqs, max_k = 12, plot = False)

    kmeans_find_nPCs(sniff_activity, binned_freqs, nPCs = 15, n_clusters = 7, plot = False)

    return
    
    

#____________________________________________________________LFP ANALYSIS___________________________________________________________

# auxillary functions
def find_avg_lfp(ephys: np.array, sniff: np.array, freq_range = (4.5,5), window_size = 1000):
     
    """
    Computes the average inhale-aligned local field potential (LFP) activity within a specified sniff frequency range.

    This function performs several steps to calculate the average LFP. It first creates a sniff-aligned raster of ephys data,
    then filters this data to include only the LFP activity within a given frequency range.
    Finally, it averages the LFP activity across this filtered range.

    Parameters:
    ephys: Numpy Array containing electrophysiology data.
    sniff: Numpy Array containing sniff data, used to align with ephys data.
    freq_range (tuple, optional): The frequency range (in Hz) to filter the ephys data. Default is (4.5, 5).
    window_size (int, optional): Size of the window (in ms) for creating sniff-aligned rasters. Default is 1000.

    Returns:
    tuple: A tuple containing the averaged LFP activity (np.array) and the frequencies within the specified range (np.array).
    """
    # make sniff aligned raster
    sorted_activity, freqs = make_sniff_raster(ephys, sniff, beg = 1000, nsniffs = 'all', window_size = window_size, method = 'none', plot = False)

    # pulling in only ephys from sniff activity in frequency range
    infreq_sniff_activity, infreqs = pull_infreq_lfp(sorted_activity, freqs, freq_range)

    # averaging the ephys activity
    avg_activity = avg_infreq_lfp(infreq_sniff_activity)

    return avg_activity, infreqs


def make_null_distribution_avg_lfp(ephys, sniff, nshifts, freq_range = (4.5,5), window_size = 1000):
    """
    Generates a null distribution of average local field potential activity by circularly shifting electrophysiology (ephys) data.

    This function circularly shifts the ephys data a specified number of times and computes the average LFP activity
    for each shift within a given frequency range. The process involves creating a sniff-aligned raster for each circular shift
    of the ephys data, filtering the data to the specified frequency range, and then averaging the LFP activity.
    The result is a null distribution of average LFP activity for each channel over the specified window size.

    Parameters:
    ephys: Array-like or pd.DataFrame containing electrophysiology data.
    sniff: Array-like or pd.DataFrame containing sniff data, used for alignment.
    nshifts (int): The number of circular shifts to perform on the ephys data.
    freq_range (tuple, optional): The frequency range (in Hz) to filter the ephys data. Default is (4.5, 5).
    window_size (int, optional): Size of the window (in ms) for creating sniff-aligned rasters. Default is 1000.

    Returns:
    np.array: A 3D array where each slice along the third dimension represents the average LFP activity distribution for a particular circular shift.
    """

    # finding number of channels
    nchannels = ephys.shape[0]

    # circularly shifting ephys data
    circ_ephys = circular_shift(ephys, nshifts = nshifts, method = 'random', min_shift = 1000)

    # initializing array to store null distributions
    avg_activity_distributions = np.zeros((nchannels, window_size, nshifts))

    for shift in range(nshifts):

        # make sniff aligned raster for each shift
        sorted_activity, freqs = make_sniff_raster(circ_ephys[:,:,shift], sniff, beg = 1000, nsniffs = 'all', window_size = window_size, method = 'none', plot = False)
        
        # pulling in frequency sniff activity
        infreq_sniff_activity, _ = pull_infreq_lfp(sorted_activity, freqs, freq_range)

        # averaging the ephys activity
        avg_activity_distributions[:, :, shift] = avg_infreq_lfp(infreq_sniff_activity)


    return avg_activity_distributions

        
def find_zscores_from_null(aligned_activity, avg_activity_distributions):
    """
    Calculates the z-scores of inhale-aligned local field potential data against a null distribution.

    This function computes z-scores for each channel and time window of the aligned activity data.
    It does this by comparing the data to the mean and standard deviation of the corresponding average activity distributions
    generated through a null process. The z-score is a measure of how many standard deviations an element is
    from the mean of the null distribution.

    Parameters:
    aligned_activity (np.array): A 2D array containing the aligned activity data. The shape should be (number of channels, window size).
    avg_activity_distributions (np.array): A 3D array containing the null distributions of average activity for each channel and time window. The shape should be (number of channels, window size, number of shifts).

    Returns:
    np.array: A 2D array of z-scores for each channel and time window in the aligned activity data. The shape is (number of channels, window size).
    """

    # finding number of channels and window size
    nchannels = aligned_activity.shape[0]
    window_size = aligned_activity.shape[1]

    # initializing array to store z-scores
    z_scores = np.zeros((nchannels, window_size))

    # calculating z-scores
    for channel in range(nchannels):
        for window in range(window_size):
            z_scores[channel, window] = (aligned_activity[channel, window] - np.mean(avg_activity_distributions[channel, window, :]))/np.std(avg_activity_distributions[channel, window, :])

    return z_scores


def avg_lfp_infreq(ephys: np.array, sniff: np.array, freq_range: tuple = (7,8), window_size: int = 1000, channel: int = 1, plot: bool = True):
    """
    Calculates and optionally plots the z-scores of average local field potential (LFP) activity in a specified frequency range.

    This function first builds a null distribution of average LFP activity, then finds the average LFP activity aligned
    with sniff data within the given frequency range. It computes the z-scores of this aligned activity against the null distribution.
    If plotting is enabled, it visualizes these z-scores for a specified channel.

    Parameters:
    ephys (np.array): Array containing electrophysiology data.
    sniff (np.array): Array containing sniff data, used to align with ephys data.
    freq_range (tuple, optional): Frequency range (in Hz) for analyzing LFP activity. Default is (7, 8).
    window_size (int, optional): Size of the window (in ms) for analyzing LFP activity. Default is 1000.
    channel (int, optional): The channel number to plot. Default is 1.
    plot (bool, optional): If True, plots the z-scores of LFP activity. Default is True.

    Returns:
    tuple: A tuple containing the z-scores (np.array) of the LFP activity and the frequencies (np.array) within the specified range.
    """
    
    # building null distributions
    avg_activity_distributions = make_null_distribution_avg_lfp(ephys, sniff, nshifts = 100, freq_range = freq_range, window_size = window_size)

    # finding aligned sniff activity
    aligned_activity, freqs = find_avg_lfp(ephys, sniff, window_size = window_size, freq_range = freq_range)

    # finding z-scores
    z_scores = find_zscores_from_null(aligned_activity, avg_activity_distributions)

    # plotting
    if plot:
        plot_avg_infreq_lfp(z_scores, freq_range, ch = channel, yaxis = 'zscore')

    return z_scores, freqs
    

# main function
def avg_aligned_lfp_main(ephys, sniff, freq_range = (4.5,5), window_size = 1000, channel = 1, show_peakfinder: bool = False):

    
    #ephys_lowpassed = lowpass_ephys(ephys)

    zscores, freqs = avg_lfp_infreq(ephys, sniff, freq_range = freq_range, window_size = window_size, channel = channel)

   

    # finding peaks
    peaks, smoothed_signal, properties_peaks = find_inhales(zscores[channel,:], window_length = 100, polyorder = 7, min_peak_prominance = 3, show = show_peakfinder, signal_type = 'lfp')
    troughs, _, properties_troughs = find_inhales(-zscores[channel,:], window_length = 100, polyorder = 7, min_peak_prominance = 3, show = show_peakfinder, signal_type = 'lfp')

    # excluding peaks and troughs not outside 3 standard deviations
    peaks = peaks[np.where(properties_peaks['peak_heights'] > 1)]
    troughs = troughs[np.where(properties_troughs['peak_heights'] > 1)]

    # excluding peaks and trough before or around 0
    peaks = peaks[np.where(peaks > window_size//2)]
    troughs = troughs[np.where(troughs > window_size//2 + 20)]

    print(f'frequncy range: {freq_range}')
    print(f'peaks: {peaks}')
    print(f'troughs: {troughs}')

    # calculate frequency
    peak2peak = 1000/(peaks[1] - peaks[0])
    peak2trough = 500/(troughs[0] - peaks[0])
    print(f'peak2peak: {peak2peak}')
    print(f'peak2trough: {peak2trough}\n')

    return peak2peak, peak2trough


#____________________________________________________________FILTERING___________________________________________________________

def filter_explore(data: np.array, window_size: int = 500, plot: bool = False):
    """
    Applies various filtering techniques to a data array and optionally plots the results.

    This function demonstrates different methods of trend removal from a given dataset. It applies a boxcar moving average for detrending, as well as bandpass, lowpass, and highpass filters. The user can choose to visualize the effects of these filters on the original data through a plot.

    Parameters:
    data (np.array): The input data array on which filtering techniques will be applied.
    window_size (int, optional): The window size for the boxcar moving average. Default is 500.
    plot (bool, optional): If True, the function plots the original data and its filtered versions. Default is False.

    Returns:
    None: This function does not return a value but rather plots the original and filtered data for comparison.
    """
    # Removing trend with boxcar moving avg
    boxcar_detrended_data = remove_trend_boxcar(data, window_size = 500)

    # Removing trend with bandpass, lowpass, highpass
    bandpassed, lowpassed, highpassed = remove_trend_bandpass(data, 1, 15, 1000)

    # Plotting
    if plot:
        fig, ax = plt.subplots(figsize = (12,12))
        ax.plot(data, label = 'original')
        ax.plot(bandpassed, label = 'bandpassed')
        ax.plot(lowpassed, label = 'lowpassed')
        ax.plot(highpassed, label = 'highpassed')
        ax.plot(boxcar_detrended_data, label = 'boxcar detrended')
        ax.legend()
        plt.show()
     

def highpass_ephys(ephys, cutoff = 15, order = 3, fs = 1000):

    nchannels = ephys.shape[0]
    ephys_highpassed = np.zeros(ephys.shape)

    for ch in range(nchannels):
        _, _, highpassed = remove_trend_bandpass(ephys[ch, :], 1, cutoff, fs)
        ephys_highpassed[ch, :] = highpassed

    return ephys_highpassed


def lowpass_ephys(ephys, cutoff = 1, order = 3, fs = 1000):

    nchannels = ephys.shape[0]
    ephys_lowpassed = np.zeros(ephys.shape)

    for ch in range(nchannels):
        _, lowpassed, _ = remove_trend_bandpass(ephys[ch, :], cutoff, 1000, fs)
        ephys_lowpassed[ch, :] = lowpassed

    return ephys_lowpassed

