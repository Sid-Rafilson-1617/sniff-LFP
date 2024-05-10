"""
Library Name: OB LFP and Sniff Data Analysis
Author: Sid Rafilson
Description: A Python library for analyzing time-domain signals from olfactory bulb local field potentials (LFPs) and respiration data.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
from scipy import stats
from scipy.signal import butter, sosfiltfilt


#______________________________________________________________________________CORE FUNCTIONS______________________________________________________________________________#


def load_sniff_MATLAB(file: str) -> np.array:
    '''
    Loads a MATLAB file containing sniff data and returns a numpy array

    This function loads a MATLAB file containing sniff data and returns a numpy array containing the sniff data.

    Parameters:
    file (str): The path to the MATLAB file containing sniff data.

    Returns:
    np.array: A 2D numpy array containing the sniff data.

    Examples:
    To load sniff data from a MATLAB file:
    >>> sniff_data = load_sniff_MATLAB('path/to/file.mat')
    '''

    mat = scipy.io.loadmat(file)
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

    Examples:
    To perform circular shifts on an electrophysiological signal array:
    >>> shifted_ephys = circular_shift(ephys, nshifts = 1000, method = 'sample')
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



def find_condition_mask_inhales(events: np.array, len_sniff: int):

    """
    Extracts indices representing free-moving and headfixed conditions from event timings.

    This function processes an array of event data to identify intervals during which the mouse
    is freely moving and intervals during which it is headfixed. The function handles scenarios
    where events may overlap or need adjustment based on specific experimental criteria, such
    as removing times during a floor flip from the free-moving period.

    Parameters:
    - events (np.ndarray): A 2D numpy array where each row represents a specific event and relevant
                           timestamps. Expected to have at least three columns for start and end times.
    - len_sniff (int): The total number of samples in the sniff signal, used for boundary conditions.

    Returns:
    - tuple: A tuple containing two numpy arrays:
        - freemoving_mask (np.ndarray): Indices representing the intervals during which the mouse
                                        is freely moving, excluding any intervals that overlap with
                                        floor flip times.
        - headfixed_mask (np.ndarray): Indices representing the intervals during which the mouse
                                       is headfixed. Adjustments are made to account for the mouse
                                       potentially being headfixed outside the experimental arena.

    Notes:
    - The function assumes the presence of certain specific events in the `events` array, including
      free-moving starts and ends, and floor flips. It also handles cases where certain timestamps
      may be missing or zero by substituting default values.
    - Adjustments for headfixed intervals include hardcoded offsets to represent typical experimental
      setups (e.g., adding or subtracting seconds to/from timestamps).
    """

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





#______________________________________________________________________________HELPER FUNCTIONS______________________________________________________________________________#


       
def find_zscores_from_null(aligned_activity, avg_activity_distributions):
    """
    Calculates the z-scores of sniff_time-aligned local field potential data against a null distribution.

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



def normalize_epochs(epochs, method = 'none', show_plot = False):
    """
    Normalize the epochs data based on the specified method and optionally display plots for visual inspection.

    Parameters:
    - epochs (numpy.ndarray): A 3D numpy array with dimensions (channels, sniffs, samples) representing the
                              data to be normalized.
    - method (str, optional): Method of normalization to apply. Valid methods are 'none', 'linear', 'affine', and 'zscore'.
                              Defaults to 'none'.
    - show_plot (bool, optional): If True, plots the original and normalized data for the first channel at every 10th sniff.
                                  Defaults to False.

    Returns:
    - numpy.ndarray: A 3D numpy array of the same shape as `epochs`, containing the normalized data.

    Raises:
    - ValueError: If an invalid normalization method is specified.

    Examples:
    - To normalize without any modification: normalize_epochs(data, method='none')
    - To normalize using linear scaling: normalize_epochs(data, method='linear')
    - To normalize using affine scaling: normalize_epochs(data, method='affine')
    - To normalize using z-score: normalize_epochs(data, method='zscore')
    """

    # preallocating array to hold normalized epochs
    normalized_epochs = np.zeros_like(epochs)

    # looping through channels and sniffs
    for channel in range(epochs.shape[0]):
        for sniff in range(epochs.shape[1]):
            epoch = epochs[channel,sniff,:]

            # normalizing epoch based on method
            if method == 'none':
                normalized_epoch = epoch
            elif method == 'linear':
                a = np.min(epoch)
                b = np.max(epoch)
                normalized_epoch = (epoch - a) / (b - a)
            elif method == 'affine':
                a = np.min(epoch)
                b = np.max(epoch)
                normalized_epoch = 2 * ((epoch - a) / (b - a)) - 1
            elif method == 'zscore':
                normalized_epoch = stats.zscore(epoch)
            else:
                raise ValueError('Invalid normalization method!')
            
            # storing normalized epoch in array
            normalized_epochs[channel, sniff, :] = normalized_epoch
            
            # displaying plots if specified
            if show_plot:
                if channel == 1:
                    if sniff % 10 == 0:

                        fig, ax = plt.subplots(2, 1, figsize = (10, 6))
                        ax[0].plot(epoch)
                        ax[1].plot(normalized_epoch)
                        plt.show()

        

    return normalized_epochs



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
        loc_set = locs[5:-5]
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
                if len(data) < window_size:
                    data = np.pad(data, (0, window_size - len(data)), mode = 'constant', constant_values = 0)
                    print('!!! padding !!!')
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



def sort_lfp(sniff_activity, locs):
    '''sorts the sniff locked lfp trace by sniff frequency

    Parameters:
    sniff_activity (np.array): A 3D numpy array of sniff locked LFP data with shape (nchannels, nsniffs, window_size).
    locs (np.array): An array of sniff inhalation times (indices).

    Returns:
    sorted_activity (np.array): A 3D numpy array of sniff locked LFP data sorted by sniff frequency.
    sorted_freqs (np.array): An array of sniff frequencies corresponding to the sorted LFP data.

    Raises:
    ValueError: If the 'locs' array does not contain enough data for the required number of sniffs.

    Examples:
    To sort sniff locked LFP data by sniff frequency:
    >>> sorted_activity, sorted_freqs = sort_lfp(sniff_activity, locs)
    
    '''

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



def build_binned_raster(LFP: np.array, sniff_times: np.array, events: np.array, filter = None, cutoff = 24, nbins = 10, freq_range = (2,12), window_size = 1_000, nshifts = 100, normalize = 'none', f = 1_000):
    """
    Processes local field potential (LFP) data to build a binned raster of z-scores by analyzing data in specified frequency bins
    and conditions (freemoving and headfixed). This function supports optional signal filtering, circular shifting for null distributions,
    and data normalization.

    Parameters:
    - LFP (numpy.ndarray): An array of shape (n_channels, n_samples) representing the LFP data.
    - sniff_times (numpy.ndarray): An array of timestamps indicating sniff times.
    - events (numpy.ndarray): An array containing event markers and conditions.
    - filter (str, optional): Type of filter to apply to the LFP data. Supported values are 'lowpass', 'highpass', and 'bandpass'. Default is None.
    - cutoff (int or tuple, optional): Cutoff frequency/frequencies for the filter. If 'bandpass', a tuple of (low, high) is expected. Default is 24.
    - nbins (int, optional): Number of bins to divide the frequency range into. Default is 10.
    - freq_range (tuple, optional): Tuple indicating the frequency range to analyze, given as (low, high) in Hz. Default is (2, 12).
    - window_size (int, optional): Size of the window for epoch extraction in milliseconds. Default is 1000.
    - nshifts (int, optional): Number of circular shifts to apply for generating null distributions. Default is 100.
    - normalize (str, optional): Method of normalization to apply to the epochs. Default is 'none'.
    - f (int, optional): Sampling frequency of the LFP data in Hz. Default is 1000.

    Returns:
    - numpy.ndarray: A 4D array of z-scores with dimensions corresponding to (condition, channels, window size, bins).

    Raises:
    - ValueError: If an invalid filter type is provided.

    Examples:
    - To build a binned raster without filtering:
    >>> z_scores = build_binned_raster(LFP_data, sniff_timestamps, event_markers)

    - To apply a lowpass filter:
    >>> z_scores = build_binned_raster(LFP_data, sniff_timestamps, event_markers, filter='lowpass', cutoff=24)
    """

    # getting number of channels
    nchannels = LFP.shape[0]

    # filtering signal
    order = 5
    if filter == 'lowpass':
        sos = butter(order, cutoff, 'low', fs = f, output = 'sos')
        signal = sosfiltfilt(sos, LFP, axis = 1)
    elif filter == 'highpass':
        sos = butter(order, cutoff, 'high', fs = f, output = 'sos')
        signal = sosfiltfilt(sos, LFP, axis = 1)
    elif filter == 'bandpass':
        sos = butter(order, cutoff, 'band', fs = f, output = 'sos')
        signal = sosfiltfilt(sos, LFP, axis = 1)
    else:
        signal = LFP



    # circularly shifting signal for building null distributions
    print('Shifting signal...')
    circ_LFP = circular_shift(signal, nshifts = nshifts, method = 'random', min_shift = 1_000)
    print('Signal shifted!')



    # getting masks for freemoving and headfixed conditions
    free_mask, head_mask = find_condition_mask_inhales(events, np.max(sniff_times))



    # calculating instantaneous frequencies
    freqs = f / np.diff(sniff_times)
    sniff_times = sniff_times[:-1]

    # getting length (l) of frequency bins
    l = freq_range[1] - freq_range[0]

    # preallocating arrays to hold zscores
    all_zscores = np.zeros((2, nchannels, window_size, nbins))



    # looping through conditions (freemoving and headfixed)
    for col, cond in enumerate(['free', 'fixed']):
        print(f'Working on condition {cond}...')

        # getting sniff times and frequencies for current condition
        if cond == 'free':
            mask = free_mask
        else:
            mask = head_mask

        current_sniff_times = sniff_times[np.isin(sniff_times, mask)]
        current_freqs = freqs[np.isin(sniff_times, mask)]


        # looping through bins
        for bin in range(nbins):
            print(f'Working on bin {bin}...')

            # getting current frequency range
            current_range = (freq_range[0] + bin * (l / nbins), freq_range[0] + (bin + 1) * (l / nbins))

            # getting sniff times for current frequency range
            current_infreq_sniff_times = current_sniff_times[np.logical_and(current_freqs >= current_range[0], current_freqs < current_range[1])]

            num_sniffs = len(current_infreq_sniff_times)
            print(f'Number of sniffs in current frequency range: {num_sniffs}\n')

            if num_sniffs < 11:
                print(f'Not enough sniffs in frequency range {current_range} for condition {cond} and bin {bin}!\n')
                continue



            # building null distribution for current frequency range
            avg_activity_distributions = np.zeros((nchannels, window_size, nshifts))
            for shift in range(nshifts):

                # making sniff_time aligned matrix of LFP epochs
                epochs, loc_set = sniff_lock_lfp(current_infreq_sniff_times, circ_LFP[:,:,shift], window_size = window_size, beg = 500, nsniffs = 'all', method = 'none')
                sorted_epochs, _ = sort_lfp(epochs, loc_set)

                
                # normalizing epochs between -1 and 1
                normalized_epochs = normalize_epochs(sorted_epochs, normalize)


                # getting average epoch for current shift in current frequency range
                avg_activity_distributions[:,:,shift] = np.mean(normalized_epochs, axis = 1)


            
            # getting aligned sniff activity
            epochs, loc_set = sniff_lock_lfp(current_infreq_sniff_times, signal, window_size = window_size, beg = 1_000, nsniffs = 'all', method = 'none')
            sorted_epochs, sorted_freqs = sort_lfp(epochs, loc_set)

            # normalizing epochs between -1 and 1
            normalized_epochs = normalize_epochs(sorted_epochs, 'zscore')

            norm_avg_activity = np.mean(normalized_epochs, axis = 1)
            avg_activity = np.mean(sorted_epochs, axis = 1)


            # finding z-scores for current frequency range
            z_scores = find_zscores_from_null(norm_avg_activity, avg_activity_distributions)
            all_zscores[col,:,:,bin] = z_scores



    return all_zscores











#______________________________________________________________________________MAIN FUNCTIONS_______________________________________________________________________________#



def build_all_rasters(data_dir: str, save_dir: str, window_size = 1_000, nshifts = 100, mice = ['1410', '1412', '4122', '4127', '4131', '4138']):

    """
    Main function for processing and analyzing LFP and sniff data across multiple mice and sessions.

    This function orchestrates the workflow for analyzing olfactory bulb local field potentials (LFPs) and sniff data.
    It reads data from the specified directory, performs specified analyses, and saves the results. The function handles
    multiple mice and sessions, applies various filters, calculates z-scores, and organizes the output data into a structured format.

    Parameters:
    - data_dir (str): Path to the directory containing the data files.
    - save_dir (str): Path to the directory where the results will be saved.
    - window_size (int, optional): Size of the window for analysis in milliseconds. Defaults to 1000.
    - nshifts (int, optional): Number of circular shifts used for generating null distributions in z-score calculations. Defaults to 100.
    - mice (list of str, optional): List of mouse IDs to include in the analysis. If None, all mice in the directory will be processed.

    Examples:
    - To run analysis for specific mice with custom settings:
        >>> LFP_sniff_analysis('path/to/data', 'path/to/save', window_size=500, nshifts=50, mice=['1410', '1412'])

    Notes:
    - The function expects each mouse's data to be in a separate subdirectory named after the mouse ID within `data_dir`.
    - Each session within a mouse's directory should contain 'LFP.npy', 'sniff_params.mat', and 'events.mat' files.
    - The results are saved in a structured directory format, preserving the organization of mice and sessions.
    """

    files = os.listdir(data_dir)
    for file in files:
        if file in mice:
            mouse_dir = os.path.join(data_dir, file)
            sessions = os.listdir(mouse_dir)
            for session in sessions:
                session_dir = os.path.join(mouse_dir, session)

                # checking neccessary files exist
                required_files = ['LFP.npy', 'sniff_params.mat', 'events.mat']
                if not all(file in os.listdir(session_dir) for file in required_files):
                    print(f'skipping mouse {file} session {session} due to missing files')
                    continue

                # loading data
                LFP = np.load(os.path.join(session_dir, 'LFP.npy'))
                inh, _, inh_end, _ = load_sniff_MATLAB(os.path.join(session_dir, 'sniff_params.mat'))
                events = scipy.io.loadmat(os.path.join(session_dir, 'events.mat'))['events']

                # creating save directory
                save_path = os.path.join(save_dir, file, session)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # building binned LFP raster for each filter type
                filters_cutoffs = {'lowpass': 24, 'highpass': 24, 'bandpass': (0.1, 24)}
                for filter in filters_cutoffs:
                    zscores = build_binned_raster(LFP, inh, events, filter = filter, cutoff = filters_cutoffs[filter], normalize = 'affine', window_size = window_size, nshifts = nshifts)
                    
                    # saving zscores
                    np.save(os.path.join(save_path, f'{filter}_z_scores.npy'), zscores)
      


def plot_and_save_rasters(data_dir: str, save_dir: str, filters = ['lowpass', 'highpass', 'bandpass'], mice = ['1410', '1412', '4122', '4127', '4131', '4138']):



    sns.set_context('talk')
    for mouse in mice:
        mouse_dir = os.path.join(data_dir, mouse)
        if not os.path.exists(mouse_dir):
            print(f'skipping mouse {mouse} due to missing directory')
            continue
        sessions = os.listdir(mouse_dir)
        for session in sessions:
            session_dir = os.path.join(mouse_dir, session)

            # checking neccessary files exist
            required_files = ['lowpass_z_scores.npy', 'highpass_z_scores.npy', 'bandpass_z_scores.npy']
            if not all(file in os.listdir(session_dir) for file in required_files):
                print(f'skipping mouse {mouse} session {session} due to missing files')
                continue

            # loading data
            for filter in filters:
                zscores = np.load(os.path.join(session_dir, f'{filter}_z_scores.npy'))

                for condition in range(zscores.shape[0]):
                    for channel in range(zscores.shape[1]):

                        if condition == 0:
                            condition_name = 'freemoving'
                        else:
                            condition_name = 'headfixed'

                        title = f'{mouse} session {session} {filter} {condition_name} channel {channel}'

                        save_path = os.path.join(save_dir, mouse, session, filter)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        data_use = zscores[condition, channel, :, :].T
                        bounds = np.max(np.abs(data_use))
                        window_size = data_use.shape[1]
                        nbins = data_use.shape[0]

                        plt.figure(figsize = (12, 7))
                        sns.heatmap(data_use, cmap = 'seismic', center = 0, robust = True, vmin = -bounds, vmax = bounds)
                        plt.gca().invert_yaxis()

                        # plotting vertical line in middle of window
                        plt.axvline(window_size // 2, color = 'black', linestyle = '--')

                        # getting next inhalation time from frequency
                        next_inhale = np.zeros((nbins + 1))
                        for i in range(nbins + 1):
                            next_inhale[i] = (window_size / 2) + (1000 // (2 + i))

                        x = next_inhale
                        y = np.arange(0, nbins + 1)
                        sns.lineplot(x = x, y = y, linestyle = '--', alpha = 0.8, color = 'black')

                        # setting x ticks so middle tick is 0
                        xticks = [0, 1000, 2000]
                        xtick_labels = [-1, 0, 1]
                        plt.xticks(xticks, xtick_labels)
                        plt.xlabel('Time (s)')

                        plt.tick_params(axis = 'x', labelrotation = 0)

                        # setting y ticks
                        yticks =[0, 2, 4, 6, 8, 10]
                        ytick_labels = [2, 4, 6, 8, 10, 12]
                        plt.yticks(yticks, ytick_labels)
                        plt.ylabel('Frequency (Hz)')

                        # setting title
                        plt.title(title)

                        name = f'{condition_name}_channel_{channel}'

                        plt.savefig(os.path.join(save_path, f'{name}.png'), dpi = 300)
                        plt.close()




