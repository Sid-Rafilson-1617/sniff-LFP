"""
Higher order functions for analysis of local field potential, thermistor sniff, and behavioral data. 

Requires core library.

By: Sid Rafilson
Contributors: Nate Hess
Primary Investigator: Matt Smear
"""

from core import *
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.io
import os



#____________________________________________________________PREPROCESS___________________________________________________________

def preprocess(sniff_file, ephys_file, tracking_file, num_samples = -1, start = 0, stop = 0, nchannels = 16, channels = None, remove_artifacts = False, resample = True, tracking_ = False, ephys_ = False, sniff_ = False):
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

    if sniff_:
        sniff = load_sniff(sniff_file, num_samples=num_samples, start = start, stop = stop)
        sniff = sniff.astype(np.int32)
        if remove_artifacts:
            sniff = remove_jumps_sniff(sniff)
        if resample:
            sniff = resample_sniff(sniff)

    if ephys_:
        ephys = load_ephys(ephys_file, num_samples=num_samples, start = start, stop = stop, nchannels= nchannels)
        ephys = ephys.astype(np.int32)
        if remove_artifacts:
            ephys = remove_jumps_ephys(ephys)
        if resample:
            ephys = resample_ephys(ephys, nchannels=nchannels)
        if isinstance(channels, tuple):
            ephys = ephys[channels, :]

    if tracking_:
        tracking = load_tracking(tracking_file, num_samples=num_samples, start = start, stop = stop)
        tracking = tracking.astype(np.float32)
        tracking = np.round(tracking, 0)
        tracking = resample_tracking(tracking, original_rate = 100, target_rate = 1000)
    
    print('data loaded')

    if sniff_ and ephys_ and tracking_:
        return ephys, sniff, tracking
    elif sniff_ and ephys_:
        return ephys, sniff
    elif sniff_ and tracking_:
        return sniff, tracking
    elif ephys_ and tracking_:
        return ephys, tracking
    elif sniff_:
        return sniff
    elif ephys_:
        return ephys
    elif tracking_:
        return tracking
    else:
        raise ValueError('no data loaded')



def get_ephys_rnp():

    '''
    scans DATA_MATT_DRIVE for sniff and ephys files, preprocesses, plots the first few seconds, and saves them to a specified directory. 

    
    '''
    mice = ['4127', '4131', '4138']

        
    #___________4127____________________
    no_artifact_4127_sniff = ['2', '3,' '5', '7', '8', '10', '13', '14', '16']
    artifact_4127_sniff = []
    no_artifact_4127_ephys = []
    artifact_4127_ephys = ['10', '2', '3', '5', '13', '14', '16', '7', '8']


    #___________4131____________________
    no_artifact_4131_sniff = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '13', '19']
    artifact_4131_sniff = ['15', '16', '17']
    no_artifact_4131_ephys = ['19']
    artifact_4131_ephys = ['15', '16', '17', '2', '3', '4', '5', '6', '7', '8', '9', '10', '13', '19']



    #___________4138____________________
    no_artifact_4138_sniff = ['1', '2', '3', '4', '5', '6', '7']
    artifact_4138_sniff = []
    no_artifact_4138_ephys = []
    artifact_4138_ephys = ['1', '2', '3', '4', '5', '6', '7']



    #___________4122____________________
    no_artifact_4122_sniff = ['4', '5', '6']
    artifact_4122_sniff = []
    no_artifact_4122_ephys = []
    artifact_4122_ephys = ['4', '5', '6']


    sniff_file = ''
    tracking_file = ''

    events_dir = r"E:\Sid_LFP\Sid_data\events"
    check_figs_dir = r"E:\Sid_LFP\Sid_data\check_ephys_rnp\\"
    save_dir = r"E:\Sid_LFP\Sid_data\rnp_final"


    for mouse in mice:
            if mouse == '4127':
                nchannels = 16
            elif mouse == '4122':
                nchannels = 16
            elif mouse == '4131':
                nchannels = 64
            elif mouse == '4138':
                nchannels = 64
            else:
                print('Mouse number not recognized by function')
                break

            # loop through sessions
            print('Processing mouse:', mouse)
            mouse_path = os.path.join(dir, mouse)
            sessions = os.listdir(mouse_path)
            for session in sessions:
                print('Processing session:', session)
                data_file_path = os.path.join(mouse_path, session)
                ephys_file = [file for file in os.listdir(data_file_path) if file.endswith('nch16.bin') or file.endswith('nch64.bin') or file.endswith('nch64.bin_cropped.bin') or file.endswith('nch16.bin_cropped.bin') or file.endswith('nch64_cropped.bin')]

                # ensure sniff file is non-empty and check if artifact is present
                if ephys_file != []:
                    ephys_file = os.path.join(data_file_path, ephys_file[0])

                   
                    # ephys session with no artifacts
                    if (mouse == '4127' and session in no_artifact_4127_ephys) or (mouse == '4131' and session in no_artifact_4131_ephys) or (mouse == '4122' and session in no_artifact_4122_ephys) or (mouse == '4138' and session in no_artifact_4138_ephys):
                        ephys = preprocess(
                            sniff_file, ephys_file, tracking_file,
                            num_samples = -1, start = 0,
                            stop = 0, nchannels = nchannels, remove_artifacts = False, resample = True, sniff_ = False, ephys_ = True)
                        
                    # ephys session with artifacts
                    elif (mouse == '4127' and session in artifact_4127_ephys) or (mouse == '4131' and session in artifact_4131_ephys) or (mouse == '4122' and session in artifact_4122_ephys) or (mouse == '4138' and session in artifact_4138_ephys):
                        ephys = preprocess(
                            sniff_file, ephys_file, tracking_file,
                            num_samples = -1, start = 0,
                            stop = 0, nchannels = nchannels, remove_artifacts = True, resample = True, sniff_ = False, ephys_ = True)
                    else:
                        continue

                    
                    
                    save_path = save_dir + '\\' + mouse + '\\' + session + '\\'
                    sns.set_style('darkgrid')
                    sns.lineplot(ephys[0, :5000])
                    plt.savefig(check_figs_dir + '\\' + mouse + '\\' + session + '_LFP.png')
                    plt.close()

                    np.save(save_path + 'LFP.npy', ephys)

                else:
                    print('No sniff or ephys file found for session:', session)



def get_sniff_and_ephys_conditions(condition = True):

    '''
    scans DATA_MATT_DRIVE for sniff and ephys files, preprocesses, plots the first few seconds, and saves them to a specified directory. 

    
    '''

        
    #___________4127____________________
    no_artifact_4127_sniff = ['2', '3,' '5', '7', '8', '10', '13', '14', '16']
    artifact_4127_sniff = []
    no_artifact_4127_ephys = []
    artifact_4127_ephys = ['10', '2', '3', '5', '13', '14', '16', '7', '8']


    #___________4131____________________
    no_artifact_4131_sniff = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '13', '19']
    artifact_4131_sniff = ['15', '16', '17']
    no_artifact_4131_ephys = ['19']
    artifact_4131_ephys = ['15', '16', '17', '2', '3', '4', '5', '6', '7', '8', '9', '10', '13', '19']


    #___________4138____________________
    no_artifact_4138_sniff = ['1', '2', '3', '4', '5', '6', '7']
    artifact_4138_sniff = []
    no_artifact_4138_ephys = []
    artifact_4138_ephys = ['1', '2', '3', '4', '5', '6', '7']



    #___________4122____________________
    no_artifact_4122_sniff = ['4', '5', '6']
    artifact_4122_sniff = []
    no_artifact_4122_ephys = []
    artifact_4122_ephys = ['4', '5', '6']

    tracking_file = r"E:\Sid_LFP\Sid_data\4131\10\10_keypoints.csv"
    events_dir = r"E:\Sid_LFP\Sid_data\events"
    check_figs_dir = r"E:\Sid_LFP\Sid_data\check_ephys_rnp\\"
    save_dir = r"E:\Sid_LFP\Sid_data\rnp_final"

    mice = ['4127', '4131', '4138']

    for mouse in mice:
            if mouse == '4127':
                nchannels = 16
            elif mouse == '4122':
                nchannels = 16
            elif mouse == '4131':
                nchannels = 64
            elif mouse == '4138':
                nchannels = 64
            else:
                print('Mouse number not recognized by function')
                break

            # loop through sessions
            print('Processing mouse:', mouse)
            mouse_path = os.path.join(dir, mouse)
            sessions = os.listdir(mouse_path)
            for session in sessions:
                print('Processing session:', session)
                data_file_path = os.path.join(mouse_path, session)
                sniff_file = [file for file in os.listdir(data_file_path) if file.endswith('ADC.bin') or file.endswith('nch8.bin')]
                ephys_file = [file for file in os.listdir(data_file_path) if file.endswith('nch16.bin') or file.endswith('nch64.bin')]

                # ensure sniff file is non-empty and check if artifact is present
                if sniff_file != [] and ephys_file != []:
                    sniff_file = os.path.join(data_file_path, sniff_file[0])
                    ephys_file = os.path.join(data_file_path, ephys_file[0])

                    # sniff session with no artifacts 
                    if (mouse == '4127' and session in no_artifact_4127_sniff) or (mouse == '4131' and session in no_artifact_4131_sniff) or (mouse == '4122' and session in no_artifact_4122_sniff) or (mouse == '4138' and session in no_artifact_4138_sniff):
                        sniff = preprocess(
                            sniff_file, ephys_file, tracking_file,
                            num_samples = -1, start = 0,
                            stop = 0, nchannels = nchannels, remove_artifacts = False, resample = True, sniff_ = True, ephys_ = False)
                        
                    # sniff session with artifacts
                    elif (mouse == '4127' and session in artifact_4127_sniff) or (mouse == '4131' and session in artifact_4131_sniff) or (mouse == '4122' and session in artifact_4122_sniff) or (mouse == '4138' and session in artifact_4138_sniff):
                        sniff = preprocess(
                            sniff_file, ephys_file, tracking_file,
                            num_samples = -1, start = 0,
                            stop = 0, nchannels = nchannels, remove_artifacts = True, resample = True, sniff_ = True, ephys_ = False)
                    else:
                        continue

                    # ephys session with no artifacts
                    if (mouse == '4127' and session in no_artifact_4127_ephys) or (mouse == '4131' and session in no_artifact_4131_ephys) or (mouse == '4122' and session in no_artifact_4122_ephys) or (mouse == '4138' and session in no_artifact_4138_ephys):
                        ephys = preprocess(
                            sniff_file, ephys_file, tracking_file,
                            num_samples = -1, start = 0,
                            stop = 0, nchannels = nchannels, remove_artifacts = False, resample = True, sniff_ = False, ephys_ = True)
                        
                    # ephys session with artifacts
                    elif (mouse == '4127' and session in artifact_4127_ephys) or (mouse == '4131' and session in artifact_4131_ephys) or (mouse == '4122' and session in artifact_4122_ephys) or (mouse == '4138' and session in artifact_4138_ephys):
                        ephys = preprocess(
                            sniff_file, ephys_file, tracking_file,
                            num_samples = -1, start = 0,
                            stop = 0, nchannels = nchannels, remove_artifacts = True, resample = True, sniff_ = False, ephys_ = True)
                    else:
                        continue

                    
                    if condition == True:

                        # load events file containing trial timestamp information
                        events_file = os.path.join(events_dir, 'events_' + mouse + '_' + session + '.mat')
                        events_mat = scipy.io.loadmat(events_file)
                        events = events_mat['events']

                        # load mask files
                        _, _, freemoving_bool, headfixed_bool = find_condition_mask(events, sniff)

                        ephys_transposed = ephys.transpose()

                        
                        # defining headfixed and freemoving sniff and ephys signals
                        freemoving_sniff = np.where(freemoving_bool, sniff, 0)
                        freemoving_ephys_transposed = np.where(freemoving_bool[:,None], ephys_transposed, 0)
                        freemoving_ephys = freemoving_ephys_transposed.transpose()

                        headfixed_sniff = np.where(headfixed_bool, sniff, 0)
                        headfixed_ephys_transposed = np.where(headfixed_bool[:,None], ephys_transposed, 0)
                        headfixed_ephys = headfixed_ephys_transposed.transpose()

                        plt.plot(freemoving_ephys[0, 1000:5000])
                        plt.savefig(check_figs_dir + mouse + session + '_freemoving_ephys.png')
                        plt.close()

                        plt.plot(headfixed_ephys[0, 1000:5000])
                        plt.savefig(check_figs_dir + mouse + session + '_headfixed_ephys.png')
                        plt.close()
                        
                        plt.plot(freemoving_sniff[1000:5000])
                        plt.savefig(check_figs_dir + mouse + session + '_freemoving_sniff.png')
                        plt.close()
                        
                        plt.plot(headfixed_sniff[1000:5000])
                        plt.savefig(check_figs_dir + mouse + session + '_headfixed_sniff.png')
                        plt.close()

                        save_path = save_dir + '\\' + mouse + '\\'
                        np.save(save_path + session + '_freemoving_sniff.npy', freemoving_sniff)
                        np.save(save_path + session + '_freemoving_ephys.npy', freemoving_ephys)
                        np.save(save_path + session + '_headfixed_sniff.npy', headfixed_sniff)
                        np.save(save_path + session + '_headfixed_ephys.npy', headfixed_ephys)
                    
                    elif condition == False:
                        save_path = save_dir + '\\' + mouse + '\\' + session + '\\'
                        sns.set_style('darkgrid')
                        sns.lineplot(ephys[0, :5000])
                        plt.savefig(check_figs_dir + '\\' + mouse + '\\' + session + '_LFP.png')
                        plt.close()

                        np.save(save_path + 'LFP.npy', ephys)

                else:
                    print('No sniff or ephys file found for session:', session)



    


#____________________________________________________________SNIFF Analysis___________________________________________________________
    
def make_sniff_raster(ephys: np.array, inhales: np.array, beg: int = 1000, nsniffs: int = 200, window_size: int = 1000, method: str = 'simple', plot: bool = False, shifts: int = 10):

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


def probability_distribution_analysis(sniff, mask = None, plot_show = False, mouse = None, session = None):

    if mask is not None:
        sniff = sniff[mask]
    
    # find inhalation times
    inhalation_times = find_inhales_2(sniff, show = False)

    # finding sniff frequency
    inter_sniff_times = np.diff(inhalation_times)

    # finding and removing outliers
    outlier_indexes = np.where(inter_sniff_times < 67)
    inhalation_times = np.delete(inhalation_times, outlier_indexes)
    inter_sniff_times = np.delete(inter_sniff_times, outlier_indexes)
    
    # converting to frequency
    non_zero_sniff_time = inter_sniff_times[inter_sniff_times != 0]
    sniff_freqs = 1000 / non_zero_sniff_time

    # plotting histogram
    n, bins, patches = plt.hist(sniff_freqs, bins=40, density = True, color='c', edgecolor='black')


    def fivemodal_dist(params, x, y):
        weight_1, skew_1, mu1, sigma1, weight_2, skew_2, mu2, sigma2, weight_3, skew_3, mu3, sigma3, weight_4, skew_4, mu4, sigma4, weight_5, skew_5, mu5, sigma5 = params
        gauss1 = weight_1 * skewnorm.pdf(x, skew_1, mu1, sigma1)
        gauss2 = weight_2 * skewnorm.pdf(x, skew_2, mu2, sigma2)
        gauss3 = weight_3 * skewnorm.pdf(x, skew_3, mu3, sigma3)
        gauss4 = weight_4 * skewnorm.pdf(x, skew_4, mu4, sigma4)
        gauss5 = weight_5 * skewnorm.pdf(x, skew_5, mu5, sigma5)

        dist = gauss1 + gauss2 + gauss3 + gauss4 + gauss5
        
        return dist

    def fourmodal_dist(params, x, y):
        weight_1, skew_1, mu1, sigma1, weight_2, skew_2, mu2, sigma2, weight_3, skew_3, mu3, sigma3, weight_4, skew_4, mu4, sigma4 = params
        gauss1 = weight_1 * skewnorm.pdf(x, skew_1, mu1, sigma1)
        gauss2 = weight_2 * skewnorm.pdf(x, skew_2, mu2, sigma2)
        gauss3 = weight_3 * skewnorm.pdf(x, skew_3, mu3, sigma3)
        gauss4 = weight_4 * skewnorm.pdf(x, skew_4, mu4, sigma4)

        dist = gauss1 + gauss2 + gauss3 + gauss4
        
        return dist

    def trimodal_dist(params, x, y):
        weight_1, skew_1, mu1, sigma1, weight_2, skew_2, mu2, sigma2, weight_3, skew_3, mu3, sigma3 = params
        gauss1 = weight_1 * skewnorm.pdf(x, skew_1, mu1, sigma1)
        gauss2 = weight_2 * skewnorm.pdf(x, skew_2, mu2, sigma2)
        gauss3 = weight_3 * skewnorm.pdf(x, skew_3, mu3, sigma3)

        dist = gauss1 + gauss2 + gauss3
        
        return dist
    
    def bimodal_dist(params, x, y):
        weight_1, skew_1, mu1, sigma1, weight_2, skew_2, mu2, sigma2 = params
        gauss1 = weight_1 * skewnorm.pdf(x, skew_1, mu1, sigma1)
        gauss2 = weight_2 * skewnorm.pdf(x, skew_2, mu2, sigma2)

        dist = gauss1 + gauss2
        
        return dist
    
    def unimodal_dist(params, x, y):
        weight_1, skew_1, mu1, sigma1 = params
        gauss1 = weight_1 * skewnorm.pdf(x, skew_1, mu1, sigma1)

        dist = gauss1
        
        return dist

    def sum_of_squares(params, x, y, modes):
        '''
        Sum of squares
        '''

        if modes == 1:
            model = unimodal_dist(params, x, y)
        elif modes == 2:
            model = bimodal_dist(params, x, y)
        elif modes == 3:
            model = trimodal_dist(params, x, y)
        elif modes == 4:
            model = fourmodal_dist(params, x, y)
        elif modes == 5:
            model = fivemodal_dist(params, x, y)
        else:
            raise ValueError('Invalid number of modes')
        return np.sum((y - model) ** 2)
    
    def weight_constraint(params):
        return np.sum(params[::4]) - 1

    def fit_distribution(n, bins, modes = None, plot_show = False):
        '''
        Fits the distibutions to the data
        '''
        # fit distribution
        bin_midpoints = (bins[1:] + bins[:-1]) / 2
        weight = 0.5
        initial_guess = [weight, 0, 3, 1, weight, 0, 10, 1, weight, 0, 5, 0.5, weight, 0, 8, 1, weight, 0, 6, 1]
        bounds = [(1e-6, 1), (-10, 10), (1e-6, 14), (1e-6, 10)] * modes
        if modes == 1:
            model = unimodal_dist
        elif modes == 2:
            model = bimodal_dist
        elif modes == 3:
            model = trimodal_dist
        elif modes == 4:
            model = fourmodal_dist
        elif modes == 5:
            model = fivemodal_dist
        else:
            raise ValueError('Invalid number of modes')
        
        
        cons = ({'type': 'eq', 'fun': weight_constraint})

        result = minimize(sum_of_squares, initial_guess[:4*modes], method = 'SLSQP', args = (bin_midpoints, n, modes), bounds=bounds, options={'maxiter': 10000000, 'ftol': 1e-7}, constraints = cons)

        if result.success:
            fitted_params = result.x
            #print("Fitted parameters:", fitted_params)
        else:
            raise ValueError(result.message)

        
        # compute sum of squares
        sums = np.sum((n - model(fitted_params, bin_midpoints, 0)) ** 2)

        # compute kolmogorov-smirnov statistic
        d, p_value = stats.ks_2samp(n, model(fitted_params, bin_midpoints, 0))

        if plot_show:
            plt.figure()
            xvals = np.linspace(bins[0], bins[-1], 1000)
            if modes == 3:
                plt.plot(xvals, model(fitted_params, xvals, 0), label = f' Modes: {modes}, LSS < {"{:.2f}".format(0.01 * math.floor(100 * np.log(sums)))}', linewidth = 3, color = 'k')
            else:
                plt.plot(xvals, model(fitted_params, xvals, 0), label = f' Modes: {modes},  LSS < {"{:.2f}".format(0.01 * math.floor(100 * np.log(sums)))}')


        return fitted_params , bin_midpoints, d, p_value, sums

    p_vals = []
    sums = []
    for modes in range(1,6):
        _, bin_midpoints, d, p_value, sum = fit_distribution(n, bins, modes = modes, plot_show = plot_show)
        print("Number of modes:", modes)
        print("KS statistic:", d)
        print("P value:", p_value)
        print("Sum of squares:", sum)
        print("\n")
        p_vals.append(p_value)
        sums.append(sum)

        if modes == 5:
            title = 'Sniff frequency distribution \n' + 'mouse ' + mouse + ' ' + 'session '+ session
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Probability')
            plt.title(title)
            plt.legend()
            plt.savefig(r"E:\Sid_LFP\figs\concatenated_signals_distributions\\" + 'mouse_' + mouse + '_' + 'session_' + session + '.png')
            plt.clf()


    return p_vals, sums



def probability_distribution_analysis_updated(freqs: np.array, savepath: str, method: str = 'log-normal', condition: str = 'headfixed', niters: int = 100, nmodes = 5, mouse = None, session = None):
    
    sns.set_style('darkgrid')

    # creating histogram

    bin_edges = np.logspace(start = np.log10(1), stop = np.log10(15), num = 40)
    n, bins, _ = plt.hist(freqs, bins = bin_edges, density = True, color = 'c', edgecolor = 'black', alpha = 0.7)
    plt.gca().set_xscale('log')
    
    # setting x axis range and tic labs
    plt.xlim(1, 16)

    # Setting custom ticks (this will ensure ticks are placed at these positions, but labels are formatted by the function above)
    plt.xticks([1, 2, 5, 10, 15], ['1', '2', '5', '10', '15'])





    # defining the base log-normal distribution
    def log_normal(x, mu, sigma):
        return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x) - mu) ** 2 / (2 * sigma ** 2))


    # defining the mixture distribution
    def mixture(params, x, n_distributions):
        distribution = 0
        for i in range(n_distributions):
            distribution += params[(3 * i)] * log_normal(x, params[(3 * i) + 1], params[(3 * i) + 2])
        return distribution
    
    
    # defining the sum of squares function
    def sum_of_squares(params, x, data, n_distributions):
        return np.sum((mixture(params, x, n_distributions) - data) ** 2)
    
    
    # defining a weight constraint function to maintain valid pdf
    def weight_constraint(params):
        return np.sum(params[::3]) - 1
    


    
    # defining the fitting function
    def fit_distribution(n, bins, n_distributions, plot_show = False):

        # fiting to histogram bin midpoints
        bin_midpoints = (bins[1:] + bins[:-1]) / 2

        # initial guess and bounds for model params
        weight = 1 / n_distributions

        best_sums = np.inf
        for i in range(0, niters):
            initial_guess = [weight,  np.random.lognormal(0, 0.25) * np.log(3),  np.random.lognormal(0, 0.25) * (0.25), weight, np.random.lognormal(0, 0.25) * np.log(10),  np.random.lognormal(0, 0.25) * (0.25), weight, np.random.lognormal(0, 0.25) * np.log(5),  np.random.lognormal(0, 0.25) * (0.25), weight, np.random.lognormal(0, 0.25) * np.log(8),  np.random.lognormal(0, 0.25) * (0.25), weight, np.random.lognormal(0, 0.25) * np.log(6),  np.random.lognormal(0, 0.25) * (0.25)] * 2


            bounds = [(1e-6, 1), (1e-6, 14), (1e-6, 10)] * n_distributions
            constraints = {'type': 'eq', 'fun': weight_constraint}

            result = minimize(sum_of_squares, initial_guess[:3*n_distributions],
                            method = 'SLSQP', args = (bin_midpoints, n, n_distributions),
                            bounds = bounds, options={'maxiter': 10000000, 'ftol': 1e-7}, constraints = constraints)

            if result.success:
                fitted_params = result.x
                #print('fitted parameters:', fitted_params)
            else:
                print(result.message)
                continue
            
            # compute sum of squares
            sums = np.sum((n - mixture(fitted_params, bin_midpoints, n_distributions)) ** 2)
            if sums < best_sums:
                best_sums = sums
                best_params = fitted_params



        fitted_params = best_params
        sums = best_sums
            

        # compute KS statistic
        d, p_val = stats.ks_2samp(n, mixture(fitted_params, bin_midpoints, n_distributions))

        if plot_show:
            xvals = np.linspace(bins[0], bins[-1], 1000)
            if n_distributions == 3:
                plt.plot(xvals, mixture(fitted_params, xvals, n_distributions), label = f' Modes: {n_distributions}', linewidth = 3, color = 'k')
            else:
                plt.plot(xvals, mixture(fitted_params, xvals, n_distributions))


        return fitted_params , bin_midpoints, d, p_val, sums

    p_vals = []
    sums = []
    params = []
    for complexity in range(1, nmodes + 1):
        fitted_params, bin_midpoints, d, p_value, sum = fit_distribution(n, bins, complexity, plot_show = True)
        print("Number of modes:", complexity)
        print("KS statistic:", d)
        print("P value:", p_value)
        print("Sum of squares:", sum)
        print("\n")
        p_vals.append(p_value)
        sums.append(sum)
        params.append(fitted_params)

        if complexity == nmodes:
            if session is not None:
                title = 'Sniff frequency distribution \n' + 'mouse ' + mouse + ' ' + 'session: '+ session + ' ' + 'condition: ' + condition
            else:
                title = 'Sniff frequency distribution \n' + 'mouse ' + mouse + ' ' + 'condition: ' + condition

            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Probability')
            plt.title(title)
            plt.legend()
            #plt.show()
            if session is not None:
                plt.savefig(savepath + '\\mouse_' + mouse + '_' + 'session_' + session + 'condition_' + condition + '.png')
            else:
                plt.savefig(savepath + '\\mouse_' + mouse + '_' + 'condition_' + condition + '.png')
            plt.clf()


    return p_vals, sums, params



def concatenation_function():

    data_path = r"E:\\Sid_LFP\\Sid_data\\"
    mice = ['4122', '4127', '4131', '4138']
    data_arrays = {
        "freemoving_4127": [],
        "headfixed_4127": [],
        "freemoving_4131": [],
        "headfixed_4131": [],
        "freemoving_4138": [],
        "headfixed_4138": [],
        "freemoving_4122": [],
        "headfixed_4122": [],
    }

    for mouse in mice:
        print(f"Processing mouse {mouse}")
        mouse_path = os.path.join(data_path, mouse)
        for condition in ['freemoving', 'headfixed']:
            file_pattern = f"{condition}_sniff.npy"
            sniff_files = [file for file in os.listdir(mouse_path) if file.endswith(file_pattern)]
            for sniff_file in sniff_files:
                data = np.load(os.path.join(mouse_path, sniff_file))
                key = f"{condition}_{mouse}"
                if key in data_arrays:
                    if len(data_arrays[key]) == 0:
                        data_arrays[key] = data
                    else:
                        data_arrays[key] = np.concatenate((data_arrays[key], data))
                else:
                    print(f"Key {key} not found in data_arrays")



def save_snifffreq_figs():

    sns.set_style('darkgrid')

    mice = ['4122', '4127', '4131', '4138']

    path = r"E:\Sid_LFP\Sid_data\rnp_final"
    plot_save_path = r"E:\Sid_LFP\figs\sniff_behavior_complete"
    


    for mouse in mice:
        mouse_path = path + '\\' + mouse
        sessions = os.listdir(mouse_path)
        for session in sessions:
            session_path = os.path.join(mouse_path, session)
            sniff_file = [file for file in os.listdir(session_path) if file.endswith('sniff_params.mat')]
            if sniff_file:

                # creating folders to hold plots
                session_plot_save_path = plot_save_path + '\\' + mouse + '\\' + session + '\\'
                if not os.path.exists(session_plot_save_path):
                    os.makedirs(session_plot_save_path)



                # extracting inhalation times and computing frequency
                inhalation_times, _, exhalation_times, _ = load_sniff_MATLAB(os.path.join(session_path, sniff_file[0]))
                inhalation_freqs = 1000 / np.diff(inhalation_times)


                # finding conditions
                events_file = [file for file in os.listdir(session_path) if file.endswith('events.mat')]
                events_file = os.path.join(session_path, events_file[0])
                events_mat = scipy.io.loadmat(events_file)
                events = events_mat['events']
                print(events)

                # load mask files
                freemoving_mask, headfixed_mask, = find_condition_mask_inhales(events, np.max(inhalation_times))

                freemoving_inhalation_times = inhalation_times[np.isin(inhalation_times, freemoving_mask)]
                freemoving_inhalation_freqs = inhalation_freqs[np.isin(inhalation_times[:-1], freemoving_mask)]
                headfixed_inhalation_times = inhalation_times[np.isin(inhalation_times, headfixed_mask)]
                headfixed_inhalation_freqs = inhalation_freqs[np.isin(inhalation_times[:-1], headfixed_mask)]

                # removing last entry if necessary
                if len(freemoving_inhalation_times) != len(freemoving_inhalation_freqs):
                    freemoving_inhalation_times = freemoving_inhalation_times[:-1]
                elif len(headfixed_inhalation_times) != len(headfixed_inhalation_freqs):
                    headfixed_inhalation_times = headfixed_inhalation_times[:-1]


                plt.figure(figsize = (5 * np.pi, 5))
                sns.scatterplot(x = freemoving_inhalation_times * 0.00001667, y = freemoving_inhalation_freqs, alpha = 0.8, marker = '.', s = 12, color = 'firebrick')
                sns.scatterplot(x = headfixed_inhalation_times * 0.00001667, y = headfixed_inhalation_freqs, alpha = 0.8, marker = '.', s = 12, color = 'slategray')
                plt.title(f"Respiratory Frequency Time Series \n (mouse {mouse}, Session {session})")
                plt.xlabel('Time (min)')
                plt.ylabel('Frequency (Hz)')

                custom_legend_markers = [
                Line2D([0], [0], color='firebrick', marker='o', linestyle='None', markersize=6, label='Freemoving'),
                Line2D([0], [0], color='slategray', marker='o', linestyle='None', markersize=6, label='Headfixed')]
                plt.legend(handles=custom_legend_markers)

                plt.tight_layout()
                plt.savefig(session_plot_save_path + 'sniff_freq_scatter.png')
                plt.close()


                sns.histplot(freemoving_inhalation_freqs, stat = 'density', color = 'firebrick', bins = 40)
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Probability')
                plt.title(f"Sniff Frequency Histogram: Free Moving \n (Mouse {mouse}, Session {session})")
                plt.savefig(session_plot_save_path + 'freemoving_hist.png')
                plt.close()
               

                sns.histplot(headfixed_inhalation_freqs, stat = 'density', color = 'slategray', bins = 40)
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Probability')
                plt.title(f"Sniff Frequency Histogram: Head Fixed \n (Mouse {mouse}, Session {session})")
                plt.savefig(session_plot_save_path + 'headfixed_hist.png')
                plt.close()



def concatenate_conditions(condition = 'freemoving', mouse = '4127'):

    '''concatenate sniff signals for all sessions of a given condition'''

    concatenated_signal = np.array([])


    no_artifact_4127 = ['2', '5', '6', '7', '8', '10', '13', '16']
    artifact_4127 = []
    no_artifact_4131 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '13', '19']
    artifact_4131 = ['15', '16', '17']
    no_artifact_4138 = ['1', '2', '3']

    ephys_file = ""
    tracking_file = ""

    dir = r"\\F-MOVING-DATA\EphysData\DATA_MATT_DRIVE\fromData_Restrx_Ob_Final"
    events_dir = r"E:\Sid_LFP\Sid_data\events"


    # loop through sessions
    print('Processing mouse:', mouse)
    mouse_path = os.path.join(dir, mouse)
    sessions = os.listdir(mouse_path)
    for session in sessions:
        print('Processing session:', session)
        data_file_path = os.path.join(mouse_path, session)
        sniff_file = [file for file in os.listdir(data_file_path) if file.endswith('ADC_int16_med0_nch8.bin') or file.endswith('ADC.bin') or file.endswith('nch8.bin')]

        # ensure sniff file is non-empty and check if artifact is present
        if sniff_file != []:
            if (mouse == '4127' and session in no_artifact_4127) or (mouse == '4131' and session in no_artifact_4131) or (mouse == '4138' and session in no_artifact_4138):
                sniff_file = os.path.join(data_file_path, sniff_file[0])

                sniff = preprocess(
                    sniff_file, ephys_file, tracking_file,
                    num_samples = -1, start = 0, channels = (1,2),
                    stop = 0, nchannels = 16, remove_artifacts = False, resample = True, sniff_ = True)
                
            elif (mouse == '4127' and session in artifact_4127) or (mouse == '4131' and session in artifact_4131):
                sniff_file = os.path.join(data_file_path, sniff_file[0])

                sniff = preprocess(
                    sniff_file, ephys_file, tracking_file,
                    num_samples = -1, start = 0, channels = (1,2),
                    stop = 0, nchannels = 16, remove_artifacts = True, resample = True, sniff_ = True)
            else:
                continue

            # load events file containing trial timestamp information
            events_file = os.path.join(events_dir, 'events_' + mouse + '_' + session + '.mat')
            events_mat = scipy.io.loadmat(events_file)
            events = events_mat['events']

            # load mask files
            freemoving_mask, headfix_mask, freemoving_bool, headfixed_bool = find_condition_mask(events, sniff)

            


            # find condition mask
            freemoving_mask, headfixed_mask, freemoving_bool, headfixed_bool = find_condition_mask(events, sniff)


    
            # defining headfixed and freemoving sniff signals
            freemoving_sniff = np.where(freemoving_bool, sniff, 0)
            headfixed_sniff = np.where(headfixed_bool, sniff, 0)

            if condition == 'headfixed':
                sniff_use = headfixed_sniff
            elif condition == 'freemoving':
                sniff_use = freemoving_sniff
            else:
                sniff_use = None
            
            # concatenate signals
            if concatenated_signal.size == 0:
                concatenated_signal = sniff_use
            else:
                concatenated_signal = np.concatenate((concatenated_signal, sniff_use))

    return concatenated_signal
                
                

def show_timeseries_conditions(sniff_file, ephys_file, tracking_file, events_dir):
    '''plots the sniff frequency time series for headfixed and freemoving conditions for a given session'''



    mouse = '4127'
    session = '13'

    sniff = preprocess(sniff_file, ephys_file, tracking_file, num_samples = -1, start = 0, stop = 0, nchannels = 16, remove_artifacts = False, resample = True, sniff_ = True)

    

    # find inhalation times
    inhalation_times, _, _ = find_inhales(sniff, show = False)


    # find exhalation times
    exhalation_times, _, _ = find_inhales(-sniff)

    # finding sniff frequency
    inter_sniff_times = np.diff(inhalation_times)

    # finding and removing outliers
    outlier_indexes = np.where(inter_sniff_times < 67)
    
    inhalation_times = np.delete(inhalation_times, outlier_indexes)
    inter_sniff_times = np.delete(inter_sniff_times, outlier_indexes)
    


    # converting to frequency
    sniff_freqs = 1000 / inter_sniff_times



    # load events file containing trial timestamp information
    events_file = os.path.join(events_dir, 'events_' + mouse + '_' + session + '.mat')
    events_mat = scipy.io.loadmat(events_file)
    events = events_mat['events']

    # find condition mask
    freemoving_mask, headfixed_mask, freemoving_bool, headfixed_bool = find_condition_mask(events, sniff)
    
    # defining headfixed and freemoving sniff signals
    freemoving_sniff = np.where(freemoving_bool, sniff, 0)
    headfixed_sniff = np.where(headfixed_bool, sniff, 0)

    # find inhalation times
    inhalation_times, _, _ = find_inhales(sniff, show = False)
    free_inhalation_times, _, _ = find_inhales(freemoving_sniff, show = False)
    fixed_inhalation_times, _, _ = find_inhales(headfixed_sniff, show = False)

    # finding sniff frequency
    inter_sniff_times = np.diff(inhalation_times)
    free_inter_sniff_times = np.diff(free_inhalation_times)
    fixed_inter_sniff_times = np.diff(fixed_inhalation_times)

    # finding and removing outliers
    outlier_indexes = np.where(inter_sniff_times < 67)
    free_outlier_indexes = np.where(free_inter_sniff_times < 67)
    fixed_outlier_indexes = np.where(fixed_inter_sniff_times < 67)
    
    inhalation_times = np.delete(inhalation_times, outlier_indexes)
    inter_sniff_times = np.delete(inter_sniff_times, outlier_indexes)
    free_inhalation_times = np.delete(free_inhalation_times, free_outlier_indexes)
    free_inter_sniff_times = np.delete(free_inter_sniff_times, free_outlier_indexes)
    fixed_inhalation_times = np.delete(fixed_inhalation_times, fixed_outlier_indexes)
    fixed_inter_sniff_times = np.delete(fixed_inter_sniff_times, fixed_outlier_indexes)

    


    # converting to frequency
    sniff_freqs = 1000 / inter_sniff_times
    freemoving_sniff_freqs = 1000 / free_inter_sniff_times
    headfixed_sniff_freqs = 1000 / fixed_inter_sniff_times

    # plotting histogram
    n, bins, patches = plt.hist(sniff_freqs, bins=50, density = True, color='c', edgecolor='black')

    # plotting repiratory frequenmcy time series

    plt.scatter(free_inhalation_times[:-1] / 1000, freemoving_sniff_freqs, marker = '.', s = 0.6, alpha = 0.9, color = 'brown')
    plt.scatter(fixed_inhalation_times[:-1] / 1000, headfixed_sniff_freqs, marker = '.', s = 0.6, alpha = 0.9, color = 'darkslategrey')
    plt.title('Respiratory Frequency Time Series \n (Mouse 4127, Session 13)')
    plt.xlabel('Time (s)')
    plt.ylabel('Respiration Frequency (Hz)')
    plt.show()

    return n, bins




def build_sniff_rasters(channel = 6):

    sns.set_style('darkgrid')

    sniff = np.load(r"E:\Sid_LFP\Sid_data\4127\7_headfixed_sniff.npy")
    ephys = np.load(r"E:\Sid_LFP\Sid_data\4127\7_headfixed_ephys.npy")

    ephys_highpassed = highpass_ephys(ephys, 15, order = 10)

    sorted_activity, freqs = make_sniff_raster(ephys_highpassed, sniff, nsniffs = 'all', method = 'simple', window_size = 1000, plot = True)

    # finding size for subplot layout
    nchannels = sorted_activity.shape[0]

    window_size = sorted_activity.shape[2]
    x_middle = window_size // 2

    # extracting subset of values for y tick labels
    y_ticks = np.linspace(0, sorted_activity.shape[1] - 1, num = 5, dtype = int)
    y_ticks_labels = [f'{freqs[i]:.1f}' for i in y_ticks]

    

    ax = sns.heatmap(sorted_activity[channel, :, :], cmap = 'seismic', cbar = True, robust = True)
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Mouse 4127, Session 2 \n Head-Fixed, Channel 4')

    # x-axis
    ax.set_xticks([x_middle - window_size/2, x_middle, x_middle + window_size/2])
    ax.set_xticklabels([-window_size/2, '0', window_size/2])

    # y-axis
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_labels)

    plt.show()



def saving_sniffrasters_check():

    save_dir = r"E:\Sid_LFP\figs\sniff_rasters_complete"
    data_dir = r"E:\Sid_LFP\Sid_data\rnp_final"

    mice = ['4127', '4131', '4138', '4122']


    for mouse in mice:
        mouse_path = os.path.join(data_dir, mouse)
        sessions = os.listdir(mouse_path)
        for session in sessions:
            save_path = os.path.join(save_dir, mouse) + '\\' + session + '\\'
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            session_path = os.path.join(mouse_path, session)
            lfp_files = [file for file in os.listdir(session_path) if file.endswith('LFP.npy')]
            sniff_files = [file for file in os.listdir(session_path) if file.endswith('sniff_params.mat')]
            if lfp_files and sniff_files:
                lfp_file = os.path.join(session_path, lfp_files[0])
                sniff_file = os.path.join(session_path, sniff_files[0])
                inhalation_times, _, exhalation_times, _ = load_sniff_MATLAB(sniff_file)
                lfp = np.load(lfp_file)

                sniff_activity, loc_set = sniff_lock_lfp(inhalation_times, lfp)
                sorted_activity, freqs = sort_lfp(sniff_activity, loc_set)
                for i in range(sorted_activity.shape[0]):
                    plot_snifflocked_lfp_single(sorted_activity, freqs, save_path, i+1)
                    plt.close()



def histogram_analysis(condition = 'headfixed'):


    #global no_artifact_4127, artifact_4127, no_artifact_4131, artifact_4131, no_artifact_4138
    no_artifact_4127 = ['2', '5', '6', '7', '8', '10', '13', '16']
    artifact_4127 = []
    no_artifact_4131 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '13', '19']
    artifact_4131 = ['15', '16', '17']
    no_artifact_4138 = ['1', '2', '3']


    sniff_file = r"E:\Sid_LFP\data\052921_4127_session4_ADC_int16_med0_nch8.bin"
    ephys_file = r"E:\Sid_LFP\data\052921_4127_session4_Ephys_int16_med0_nch16.bin"
    tracking_file = r"\\F-MOVING-DATA\EphysData\rhythmReTrackISH\keypoint_processing\KeypointMatrix\4127\4_keypoints.csv"

    dir = r"\\F-MOVING-DATA\EphysData\DATA_MATT_DRIVE\fromData_Restrx_Ob_Final"
    events_dir = r"E:\Sid_LFP\Sid_data\events"

    # define mice and number of modes
    mice = ['4127', '4131', '4138']
    nmodes = 5

    # preallocate lists for statistics
    all_pvals =[]
    all_sums = []
    all_sig_modes = []
    mouse_sessions = []

    # loop through mice and sessions
    for mouse in mice:
        print('Processing mouse:', mouse)
        mouse_path = os.path.join(dir, mouse)
        sessions = os.listdir(mouse_path)
        for session in sessions:
            print('Processing session:', session)
            data_file_path = os.path.join(mouse_path, session)
            sniff_file = [file for file in os.listdir(data_file_path) if file.endswith('ADC_int16_med0_nch8.bin') or file.endswith('ADC.bin') or file.endswith('nch8.bin')]

            # ensure sniff file is non-empty and check if artifact is present
            if sniff_file != []:
                if (mouse == '4127' and session in no_artifact_4127) or (mouse == '4131' and session in no_artifact_4131) or (mouse == '4138' and session in no_artifact_4138):
                    sniff_file = os.path.join(data_file_path, sniff_file[0])

                    sniff = preprocess(
                        sniff_file, ephys_file, tracking_file,
                        num_samples = -1, start = 0, channels = (1,2),
                        stop = 0, nchannels = 16, remove_artifacts = False, resample = True, sniff_ = True)
                    
                elif (mouse == '4127' and session in artifact_4127) or (mouse == '4131' and session in artifact_4131):
                    sniff_file = os.path.join(data_file_path, sniff_file[0])

                    sniff = preprocess(
                        sniff_file, ephys_file, tracking_file,
                        num_samples = -1, start = 0, channels = (1,2),
                        stop = 0, nchannels = 16, remove_artifacts = True, resample = True, sniff_ = True)
                else:
                    continue

                # load events file containing trial timestamp information
                events_file = os.path.join(events_dir, 'events_' + mouse + '_' + session + '.mat')
                events_mat = scipy.io.loadmat(events_file)
                events = events_mat['events']

                # load mask files
                freemoving_mask, headfix_mask, freemoving_bool, headfixed_bool = find_condition_mask(events, sniff)

                if condition == 'headfixed':
                    mask_use = headfix_mask
                elif condition == 'freemoving':
                    mask_use = freemoving_mask
                else:
                    mask_use = None


                if np.any(mask_use):
                    pvals, Sumerror = probability_distribution_analysis(sniff, mask = mask_use, plot_show = True, mouse = mouse, session = session)

                    # finding index where pval is greater than 0.95
                    for i in range(len(pvals)):
                        if pvals[i] > 0.95:
                            all_sig_modes.append(i+1)
                            break

                    mouse_sessions.append(mouse + '_' + session)
                    
                            

                    all_sums.extend(Sumerror)
                    all_pvals.extend(pvals)

            else:
                print('No sniff file found')
                continue

            


    num_rows = len(all_pvals) // nmodes
    sumerror = np.array(all_sums).reshape(num_rows, nmodes)
    pvals = np.array(all_pvals).reshape(num_rows, nmodes)








    plt.style.use('fivethirtyeight')
    plt.boxplot(sumerror, patch_artist=True, showmeans=True, showfliers=True)
    plt.xlabel('Number of modes')
    plt.ylabel('sum of squares')
    plt.title('Sum of squares for model complexity')
    plt.show()

    plt.style.use('fivethirtyeight')
    plt.boxplot(pvals, patch_artist=True, showmeans=True)
    plt.xlabel('Number of modes')
    plt.ylabel('p value')
    plt.title('KS p values for model complexity')
    plt.show()

    plt.style.use('fivethirtyeight')
    plt.boxplot(sumerror, patch_artist=True, showmeans=True)
    plt.xlabel('Number of modes')
    plt.ylabel('sum of squares')
    plt.gca().set_yscale('log')
    plt.title('Sum of squares for model complexity')
    plt.show()

    plt.style.use('fivethirtyeight')
    for i in range(num_rows):
        plt.plot(np.arange(1, nmodes+1), sumerror[i,:], label = 'Session: ' + str(i+1))
    plt.xlabel('Number of modes')
    plt.ylabel('sum of squares')
    plt.title('Sum of squares for model complexity')
    plt.gca().set_yscale('log')
    plt.legend()
    plt.show()

    plt.style.use('fivethirtyeight')
    plt.hist(all_sig_modes)
    plt.xlabel('Number of modes')
    plt.ylabel('# of significant distributions')
    plt.title('Number of significant distributions at minimal model complexity')
    plt.show()


def concatenated_signal_hist_analysis():
    
    concatenated_signals_free = ['freemoving_4127', 'freemoving_4131']
    concatenated_signals_fixed = ['headfixed_4127', 'headfixed_4131']

    nmodes = 5

    all_sums = []
    for sig in concatenated_signals_free:
        data = np.load('E:\\Sid_LFP\\concatenated_signals\\' + sig + '.npy')


        _, sums = probability_distribution_analysis(data, mask = None, plot_show=True, mouse = sig, session = 'concatenated')

        
        all_sums.extend(sums)
    num_rows = len(all_sums) // nmodes
    sumerror = np.array(all_sums).reshape(num_rows, nmodes)


    plt.style.use('fivethirtyeight')
    plt.boxplot(sumerror, patch_artist=True, showmeans=True, showfliers=True)
    plt.xlabel('Number of modes')
    plt.ylabel('sum of squares')
    plt.title('Sum of squares for model complexity \n Free moving')
    plt.show()

    plt.boxplot(sumerror, patch_artist=True, showmeans=True)
    plt.xlabel('Number of modes')
    plt.ylabel('log sum of squares')
    plt.gca().set_yscale('log')
    plt.title('Sum of squares for model complexity \n Free moving')
    plt.show()


    for i in range(num_rows):
        plt.plot(np.arange(1, nmodes+1), sumerror[i,:], label = concatenated_signals_free[i])
    plt.xlabel('Number of modes')
    plt.ylabel('log sum of squares')
    plt.title('Sum of squares for model complexity \n Free moving')
    plt.gca().set_yscale('log')
    plt.legend()
    plt.show()
    plt.cla()
    plt.clf()

    plt.style.use('default')



    all_sums = []
    for sig in concatenated_signals_fixed:
        data = np.load('E:\\Sid_LFP\\concatenated_signals\\' + sig + '.npy')


        _, sums = probability_distribution_analysis(data, mask = None, plot_show=True, mouse = sig, session = 'concatenated')


        
        all_sums.extend(sums)
    num_rows = len(all_sums) // nmodes
    sumerror = np.array(all_sums).reshape(num_rows, nmodes)


    plt.style.use('fivethirtyeight')
    plt.boxplot(sumerror, patch_artist=True, showmeans=True, showfliers=True)
    plt.xlabel('Number of modes')
    plt.ylabel('sum of squares')
    plt.title('Sum of squares for model complexity \n Head fixed')
    plt.show()


    plt.boxplot(sumerror, patch_artist=True, showmeans=True)
    plt.xlabel('Number of modes')
    plt.ylabel('log sum of squares')
    plt.gca().set_yscale('log')
    plt.title('Sum of squares for model complexity \n Head fixed')
    plt.show()

    for i in range(num_rows):
        plt.plot(np.arange(1, nmodes+1), sumerror[i,:], label =  concatenated_signals_fixed[i])
    plt.xlabel('Number of modes')
    plt.ylabel('log sum of squares')
    plt.title('Sum of squares for model complexity \n Head fixed')
    plt.gca().set_yscale('log')
    plt.legend()
    plt.show()



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
    


#_____________________________________________________________Frequency Analysis__________________________________________________________


def coherence_analysis(sniff, ephys, show_intermediary_plots = False):


  sniff = sniff.flatten()

  # bandpass filtering sniff data
  sniff = lowpass_sniff(sniff, 15)
  #sniff = highpass_sniff(sniff, 1)

  # bandpass filtering ephys data
  ephys = lowpass_ephys(ephys, 15, 1)
  ephys = highpass_ephys(ephys, 1)

  # only using one channel of ephys data
  ephys = ephys[0, 120000:]
  sniff = sniff[120000:]

  plt.figure()
  plt.plot(sniff)
  plt.plot(ephys)
  plt.show()

  # getting inhalation times
  inhales, _, _ = find_inhales(sniff)
  sniff_freqs = 1000 / np.diff(inhales)
  inhales = inhales[:-1]

  plt.figure()
  plt.scatter(inhales, sniff_freqs, c = 'k')
  plt.ylim(0, 15)

  plt.show()


  # defining length and window size for spectral analysis
  signal_length = ephys.shape[0]
  step = 200
  window_size = 4000

  
  # computing number of windows
  num_windows = np.floor((signal_length - window_size) / step).astype(int)

  # computing mean time in each window
  time = np.arange(0, signal_length, step)

  # prealocating lists to hold potentially unequal length arrays
  cross_list = []
  xspec_list = []
  yspec_list = []

  freqs = None

  # computing cross-spectrum and PSDs for each time window for both ephys and sniff
  for i in range(num_windows):
    ephys_window = ephys[i*step:i*step + window_size]
    sniff_window = sniff[i*step:i*step + window_size]

    current_cross, current_xspec, current_yspec, current_freqs = multitaper_cross_spectrum(ephys_window, sniff_window)

    cross_list.append(current_cross)
    xspec_list.append(current_xspec)
    yspec_list.append(current_yspec)

    if freqs is None:
      freqs = current_freqs

  # standardizing lengths of arrays
  max_len_cross = max(len(x) for x in cross_list)
  max_len_xspec = max(len(x) for x in xspec_list)
  max_len_yspec = max(len(x) for x in yspec_list)

  # Flatten and pad each array in cross_list
  cross_standardized = [np.pad(np.ravel(x), (0, max_len_cross - len(np.ravel(x))), 'constant', constant_values=0) for x in cross_list]
  xspec_standardized = [np.pad(np.ravel(x), (0, max_len_xspec - len(np.ravel(x))), 'constant', constant_values=0) for x in xspec_list]
  yspec_standardized = [np.pad(np.ravel(x), (0, max_len_yspec - len(np.ravel(x))), 'constant', constant_values=0) for x in yspec_list]

  # Convert to NumPy array
  mt_cross = np.vstack(cross_standardized)
  mt_xspec = np.vstack(xspec_standardized)
  mt_yspec = np.vstack(yspec_standardized)

  # removing extra dimensions
  freqs = freqs.reshape(-1)
  mt_cross = mt_cross.squeeze()
  mt_xspec = mt_xspec.squeeze()
  mt_yspec = mt_yspec.squeeze()


  neg_indexes = np.where((freqs < 0) | (freqs > 15))
  freqs = np.delete(freqs, neg_indexes)
  mt_cross = np.delete(mt_cross, neg_indexes, axis = 1)
  mt_xspec = np.delete(mt_xspec, neg_indexes, axis = 1)
  mt_yspec = np.delete(mt_yspec, neg_indexes, axis = 1)

  plt.figure()
  plt.plot(freqs, mt_cross[0,:])
  plt.plot(freqs, mt_xspec[0,:])
  plt.plot(freqs, mt_yspec[0,:])
  plt.gca().set_yscale('log')
  plt.legend(['Cross', 'Xspec', 'Yspec'])
  plt.show()


  # computing coherence
  coherence = np.zeros((num_windows, mt_cross.shape[1]))
  for i in range(num_windows):
    coherence[i,:] = (mt_cross[i,:]) ** 2 / (mt_xspec[i,:] * mt_yspec[i,:])

  freqs_display = np.linspace(0,len(freqs), 10, endpoint = False).astype(int)
  time_display = np.linspace(0, len(time), 10, endpoint = False).astype(int)
  # plotting coherence
  plt.figure()
  plt.imshow(np.transpose(np.abs(coherence)),
              aspect = 'auto', cmap = 'jet',
              origin = 'lower')
  plt.colorbar()
  plt.ylabel('Frequency (Hz)')
  plt.xlabel('Time (s)')
  plt.yticks(freqs_display, np.round(freqs[freqs_display]))
  plt.xticks(time_display, np.round(time[time_display] // 1000))
  plt.title('Coherence between Ephys and Sniff')
  plt.show()


def spectrogram_analysis(sniff, ephys, spec_file_path, channel = 0, step = 200, window_size = 4000, plot_raw = False, plot_freqs = False, plot_spectrogram = True, plot_time_spectrograms = True, plot_coherence = True, condition = 'freemoving'):
    
    sns.set_style('darkgrid')

    # only using one channel of ephys data
    if ephys.ndim == 2:
        ephys = ephys[channel, :]
    else:
        ephys = ephys.flatten()

    sniff = sniff.flatten()

    print(len(sniff))
    print(len(ephys))




    print('filtering data...')

    # bandpass filtering sniff data
    sniff = lowpass_sniff(sniff, 20, 3)
    sniff = highpass_sniff(sniff, 1, 3)

    # bandpass filtering ephys data
    ephys = lowpass_sniff(ephys, 20, 3)
    ephys = highpass_sniff(ephys, 1, 3)

    print('data filtered')


    # plotting sniff and ephys data, ensuring the data looks correct
    if plot_raw:
        sns.lineplot(sniff)
        sns.lineplot(ephys)
        plt.legend(['Sniff', 'Ephys'])
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (au)')
        plt.show()

    # getting inhalation times
    inhales = find_inhales_2(sniff)
    sniff_freqs = 1000 / np.diff(inhales)
    inhales = inhales[:-1]

    # creating dataframe to hold inhalation times and sniff frequencies
    inhalation_freqs_df = pd.DataFrame({
    'Inhale Times': inhales,
    'Sniff Frequencies': sniff_freqs
})

    # plotting sniff frequency time series
    if plot_freqs:
        sns.scatterplot(inhalation_freqs_df, x='Inhale Times', y='Sniff Frequencies', marker = '.', s = 10, alpha = 0.9)
        plt.ylim(0, 18)
        plt.title('Sniff Frequency Time Series')
        plt.show()

    
    # computing PSD for entire signal
    cross, xspec, yspec, freqs = multitaper_cross_spectrum(ephys, sniff)

    # removing frequencies outside of range
    highcut = 16
    neg_indexes = np.where((freqs < 0) | (freqs > highcut))
    freqs = np.delete(freqs, neg_indexes)
    cross = np.delete(cross, neg_indexes)
    xspec = np.delete(xspec, neg_indexes)
    yspec = np.delete(yspec, neg_indexes)

    # plotting cross-spectrum and PSDs
    if plot_spectrogram:
        print('plotting cross-spectrum and PSDs...')
        plt.figure()
        sns.lineplot(x = freqs, y = xspec, label = 'Ephys')
        sns.lineplot(x = freqs, y = yspec, label = 'Sniff')
        sns.lineplot(x = freqs, y = cross, label = 'Cross')
        plt.gca().set_yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.legend()
        plt.savefig(spec_file_path + '\\spectrogram.png')
        plt.close()
        


    # computing number of windows
    signal_length = ephys.shape[0]
    num_windows = np.floor((signal_length - window_size) / step).astype(int)

    # computing mean time in each window
    time = np.arange(0, signal_length, step)

    # prealocating lists to hold potentially unequal length arrays
    cross_list = []
    xspec_list = []
    yspec_list = []
    freqs_array = np.array([])

    freqs = None

    # computing cross-spectrum and PSDs for each time window for both ephys and sniff
    print('computing cross-spectrum and PSDs...')
    print('num_windows:', num_windows)
    for i in range(num_windows):

        if i % 100 == 0:
            print('window:', i)

        ephys_window = ephys[i*step:i*step + window_size]

        sniff_window = sniff[i*step:i*step + window_size]
        #sniff window for avg snif frequency taken from middle 1 second of window
        sniff_window_inhales = sniff[i*step + 1500:i*step + 2500]


        locs, _, _ = find_inhales(sniff_window_inhales)
        current_freqs = 1000 / np.diff(locs)
        sniff_freq = np.mean(current_freqs)

        current_cross, current_xspec, current_yspec, current_freqs = multitaper_cross_spectrum(ephys_window, sniff_window)

        cross_list.append(current_cross)
        xspec_list.append(current_xspec)
        yspec_list.append(current_yspec)
        if freqs_array.size == 0:
            freqs_array = np.array([sniff_freq])
        else:
            freqs_array = np.append(freqs_array, sniff_freq)

        if freqs is None:
            freqs = current_freqs
    
    print('cross-spectrum and PSDs computed')

    # standardizing lengths of arrays
    max_len_cross = max(len(x) for x in cross_list)
    max_len_xspec = max(len(x) for x in xspec_list)
    max_len_yspec = max(len(x) for x in yspec_list)

    # Flatten and pad each array in cross_list
    cross_standardized = [np.pad(np.ravel(x), (0, max_len_cross - len(np.ravel(x))), 'constant', constant_values=0) for x in cross_list]
    xspec_standardized = [np.pad(np.ravel(x), (0, max_len_xspec - len(np.ravel(x))), 'constant', constant_values=0) for x in xspec_list]
    yspec_standardized = [np.pad(np.ravel(x), (0, max_len_yspec - len(np.ravel(x))), 'constant', constant_values=0) for x in yspec_list]

    # Convert to NumPy array
    mt_cross = np.vstack(cross_standardized)
    mt_xspec = np.vstack(xspec_standardized)
    mt_yspec = np.vstack(yspec_standardized)

    # removing extra dimensions
    freqs = freqs.reshape(-1)
    mt_cross = mt_cross.squeeze()
    mt_xspec = mt_xspec.squeeze()
    mt_yspec = mt_yspec.squeeze()

    high_cut = 16
    neg_indexes = np.where((freqs < 0) | (freqs > high_cut))
    freqs = np.delete(freqs, neg_indexes)
    mt_cross = np.delete(mt_cross, neg_indexes, axis = 1)
    mt_xspec = np.delete(mt_xspec, neg_indexes, axis = 1)
    mt_yspec = np.delete(mt_yspec, neg_indexes, axis = 1)


   


    # computing coherence
    coherence = np.zeros((num_windows, mt_cross.shape[1]))
    for i in range(num_windows):
        coherence[i,:] = (mt_cross[i,:]) ** 2 / (mt_xspec[i,:] * mt_yspec[i,:])

    freqs_display = np.linspace(0,len(freqs), 10, endpoint = False).astype(int)
    time_display = np.linspace(0, len(time), 10, endpoint = False).astype(int)


    # finding y position corresponding to your data
    y_height = mt_xspec.shape[1]

    y_position= (y_height * freqs_array / high_cut).astype(int)

    # Example x positions corresponding to your data
    x_positions = np.arange(len(freqs_array))

    sns.set_context('talk')
    
    # plottinng spectrograms across time as heatmaps
    if plot_time_spectrograms:
        plt.figure(figsize=(60,20), dpi = 300)
        sns.heatmap(np.transpose(np.abs(mt_xspec)), cmap = 'cubehelix', cbar = True, vmin=0, vmax=np.percentile(np.abs(mt_xspec), 95))
        sns.scatterplot(x = x_positions, y = y_position, size=25, marker = 'o', c='dodgerblue')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.yticks(freqs_display, np.round(freqs[freqs_display]))
        plt.xticks(time_display, np.round(time[time_display] // 1000))
        plt.title('Ephys Spectrogram')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(spec_file_path + '\\ephys_spectrogram.png', dpi = 300)
        plt.close() 




    
    # plotting coherence
    if plot_coherence:
        plt.figure(figsize=(60,20), dpi = 300)
        sns.heatmap(np.transpose(np.abs(coherence)), cmap = 'cubehelix', cbar = True)
        sns.scatterplot(x = x_positions, y = y_position, size=25, marker = 'o', c='dodgerblue')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.yticks(freqs_display, np.round(freqs[freqs_display]))
        plt.xticks(time_display, np.round(time[time_display] // 1000))
        plt.title('Coherence between Ephys and Sniff')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(spec_file_path + '\\coherence.png', dpi = 300)
        plt.close()

    return mt_xspec, mt_yspec, mt_cross, x_positions, y_position



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


def avg_lfp_infreq(ephys: np.array, sniff: np.array, save_path: str, freq_range: tuple = (7,8), window_size: int = 1000, nshifts = 100, channel: int = 1, plot: bool = False):
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
    print('building null distributions...')
    avg_activity_distributions = make_null_distribution_avg_lfp(ephys, sniff, nshifts = nshifts, freq_range = freq_range, window_size = window_size)

    # finding aligned sniff activity
    aligned_activity, freqs = find_avg_lfp(ephys, sniff, window_size = window_size, freq_range = freq_range)

    # finding z-scores
    z_scores = find_zscores_from_null(aligned_activity, avg_activity_distributions)

    # plotting
    if plot:
        plot_avg_infreq_lfp(z_scores, save_path, freq_range, ch = channel, yaxis = 'zscore')

    return z_scores, freqs
    

# main function
def avg_aligned_lfp_main(ephys, sniff, save_path: str, freq_range = (4.5,5), window_size = 1000, nshifts = 100, channel = 1, show_peakfinder: bool = False, save_figs: bool = False):

    
    #ephys_lowpassed = lowpass_ephys(ephys)
    print('computing z-scores and aligning LFP activity')
    zscores, freqs = avg_lfp_infreq(ephys, sniff, freq_range = freq_range, save_path = save_path, window_size = window_size, channel = channel, nshifts = nshifts, plot = True)

   

    # finding peaks
    print('finding peaks')
    peaks, smoothed_signal, properties_peaks = find_inhales(zscores[channel,:], window_length = 100, polyorder = 7, min_peak_prominance = 3, show = show_peakfinder, signal_type = 'lfp', save_figs = save_figs, name = str(freq_range[0]) + '-' + str(freq_range[1]))
    troughs, _, properties_troughs = find_inhales(-zscores[channel,:], window_length = 100, polyorder = 7, min_peak_prominance = 3, show = show_peakfinder, signal_type = 'lfp')

    # excluding peaks and troughs not outside 3 standard deviations
    peaks = peaks[np.where(properties_peaks['peak_heights'] > 1)]
    hights = hights[np.where(properties_peaks['peak_heights'] > 1)]
    
    troughs = troughs[np.where(properties_troughs['peak_heights'] > 1)]
    dips = dips[np.where(properties_troughs['peak_heights'] > 1)]

    # excluding peaks and trough before or around 0
    
    peaks = peaks[np.where(peaks > window_size//2)]
    hights = hights[np.where(peaks > window_size//2)]
    troughs = troughs[np.where(troughs > peaks[0])]
    dips = dips[np.where(troughs > peaks[0])]

    print(f'frequncy range: {freq_range}')
    print(f'peaks: {peaks}')
    print(f'troughs: {troughs}')

    # calculate frequency
    peak2peak = 1000/(peaks[1] - peaks[0])
    peak2trough = 500/(troughs[0] - peaks[0])
    print(f'peak2peak: {peak2peak}')
    print(f'peak2trough: {peak2trough}\n')
    print(f'max hight of peaks: {np.max(np.abs(hights))}')
    print(f'max hight of dips: {np.max(np.abs(dips))}')

    mean_freq = np.mean(freqs)


    return peak2peak, peak2trough, np.max(hights), np.max(dips), zscores[channel - 1, :], mean_freq


# analysis function for sweeping through frequency ranges
def avg_lfp_infrequency_analysis(sniff, ephys, save_path, name, nshifts = 100):


    window_size = 2000
    nbins = 20
    sniff_freqs = np.zeros((nbins))
    ephys_freqs_p2p = np.zeros((nbins))
    ephys_freqs_p2t = np.zeros((nbins))
    hights = np.zeros((nbins))
    dips = np.zeros((nbins))
    zscore = np.zeros((nbins, window_size))
    freqs = np.zeros((nbins))
  
    for i in range(nbins):
        save_folder = os.path.join(save_path, str(i))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        freq_range = (2 + i/2, 2.5 + i/2)
        sniff_freqs[i] = 9/4 + i/2
        ephys_freqs_p2p[i], ephys_freqs_p2t[i], hights[i], dips[i], zscore[i,:], freqs[i] = avg_aligned_lfp_main(ephys, sniff, save_folder, freq_range = freq_range, nshifts = nshifts, window_size = window_size, channel = 1, show_peakfinder = False, save_figs = True)

    np.save(os.path.join(save_path, name + '_zscores.npy'), zscore)
    np.save(os.path.join(save_path, name + '_freqs.npy'), freqs)

    width = 0.15

    plt.figure()
    sns.heatmap(zscore, cmap = 'cubehelix', cbar = True)
    plt.gca().invert_yaxis()
    plt.yticks(freqs)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.savefig(os.path.join(save_path, name + '_zscore.png'))

    fig, ax = plt.subplots(layout = 'constrained')
    for i in range(nbins):
        p2p_bar = ax.bar(sniff_freqs[i] - 0.5 * width, ephys_freqs_p2p[i], width, color = 'tab:orange')
        p2t_bar = ax.bar(sniff_freqs[i] + 0.5 * width, ephys_freqs_p2t[i], width, color = 'tab:blue')
    


    ax.set_xlabel('median sniff frequency (Hz)')
    ax.set_ylabel('ephys frequency (Hz)')
    ax.set_title('ephys frequency vs sniff frequency')

    plt.legend([p2p_bar, p2t_bar], ['peak2peak', 'peak2trough'])
    name = '4127_session7'
    path = save_path
    name = name + '.png'
    plt.savefig(os.path.join(path, name))
    plt.close()

    fig, ax = plt.subplots(layout = 'constrained')
    for i in range(nbins):
        hights_bar = ax.bar(sniff_freqs[i] - 0.5 * width, hights[i], width, color = 'tab:orange', alpha = 0.5)
        dips_bar = ax.bar(sniff_freqs[i] + 0.5 * width, dips[i], width, color = 'tab:blue', alpha = 0.5)

    ax.set_xlabel('median sniff frequency (Hz)')
    ax.set_ylabel('z-scored magnitude')
    ax.set_title('z-score of peaks and trough magnitude vs sniff frequency')

    plt.legend([hights_bar, dips_bar], ['peak hight', 'trough dip'])

    path = save_path
    name = name + '_hights_dips.png'
    plt.savefig(os.path.join(path, name))
    plt.close()


    print(f'sniff freq {sniff_freqs}')
    print(f'peak2peak: {ephys_freqs_p2p}')
    print(f'peak2trough: {ephys_freqs_p2t}')






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
    boxcar_detrended_data = remove_trend_boxcar(data, window_size = window_size)

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
        _, _, highpassed = remove_trend_bandpass(ephys[ch, :], lowcut = cutoff, f = fs, order = order)
        ephys_highpassed[ch, :] = highpassed

    return ephys_highpassed

def highpass_sniff(sniff, cutoff = 1, order = 3, fs = 1000):
    
        _, _, highpassed = remove_trend_bandpass(sniff, lowcut = cutoff, f = fs, order = order)
        return highpassed

def lowpass_sniff(sniff, cutoff = 1, order = 3, fs = 1000):
        
        _, lowpassed, _ = remove_trend_bandpass(sniff, highcut = cutoff, f = fs)
        return lowpassed

def lowpass_ephys(ephys, cutoff = 1, order = 3, fs = 1000):

    nchannels = ephys.shape[0]
    ephys_lowpassed = np.zeros(ephys.shape)

    for ch in range(nchannels):
        _, lowpassed, _ = remove_trend_bandpass(ephys[ch, :], highcut = cutoff, f = fs)
        ephys_lowpassed[ch, :] = lowpassed

    return ephys_lowpassed


#______________________________________________________________Replications_____________________________________________________________


def panel1(sniff_file, ephys_file, tracking_file):
  
  def sniff_statistics(sniff_file, ephys_file, tracking_file: str):

    show = True

    # Loading and preprocessing sniff data
    sniff = preprocess(
        sniff_file, ephys_file, tracking_file,
          num_samples = -1, start = 0, channels = (1,2),
            stop = 0, nchannels = 16, remove_artifacts = True, resample = True, sniff_ = True)

    # find inhalation times
    inhalation_times, _, _ = find_inhales(sniff, show = False)


    # find exhalation times
    exhalation_times, _, _ = find_inhales(-sniff)

    # finding sniff frequency
    inter_sniff_times = np.diff(inhalation_times)

    # finding and removing outliers
    outlier_indexes = np.where(inter_sniff_times < 67)
    
    inhalation_times = np.delete(inhalation_times, outlier_indexes)
    inter_sniff_times = np.delete(inter_sniff_times, outlier_indexes)
    


    # converting to frequency
    sniff_freqs = 1000 / inter_sniff_times


    # plotting histogram
    n, bins, patches = plt.hist(sniff_freqs, bins=50, density = True, color='c', edgecolor='black')

    # plotting repiratory frequenmcy time series

    plt.scatter(inhalation_times[:-1] / 1000, sniff_freqs, marker = '.')
    plt.title('Respiratory Frequency Time Series \n (Mouse 4127, Session 4)')
    plt.xlabel('Time (s)')
    plt.ylabel('Respiration Frequency (Hz)')
    plt.show()

    return n, bins



  def bimodal_dist(params, x, y):
    '''
    Bimodal distribution
    '''
    weight_1, skew_1, mu1, sigma1, weight_2, skew_2, mu2, sigma2, weight_3, skew_3, mu3, sigma3 = params
    gauss1 = weight_1 * skewnorm.pdf(x, skew_1, mu1, sigma1)
    gauss2 = weight_2 * skewnorm.pdf(x, skew_2, mu2, sigma2)
    gauss3 = weight_3 * skewnorm.pdf(x, skew_3, mu3, sigma3)

    dist = gauss1 + gauss2 + gauss3
    
    return dist


  def sum_of_squares(params, x, y):
    '''
    Sum of squares
    '''

    model = bimodal_dist(params, x, y)
    return np.sum((y - model) ** 2)



  def fit_distribution(n, bins):
    '''
    Fits the distibution to the data
    '''
    # fit distribution
    bin_midpoints = (bins[1:] + bins[:-1]) / 2
    weight = 0.5
    initial_guess = [weight, 0, 3, 1, weight, 0, 11, 3, weight, 0, 6, 1]
    bounds = [(None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None)]


    result = minimize(sum_of_squares, initial_guess, method = 'Nelder-Mead', args = (bin_midpoints, n), bounds = bounds, options = {'maxiter': 50000})

    if result.success:
      fitted_params = result.x
      print("Fitted parameters:", fitted_params)
    else:
        raise ValueError(result.message)


    x = np.linspace(min(bin_midpoints), max(bin_midpoints), 500)
    plt.plot(x, bimodal_dist(fitted_params, x, 0), color='red', lw=3)
    plt.hist(bin_midpoints, bins, weights=n, density=True, color='c', edgecolor='black')
    plt.yticks([0, 0.1, 0.2])
    plt.xticks([2,4,6,8,10,12,14])

    plt.show()

  
    plt.plot(x, bimodal_dist(fitted_params, x, 0), color='red', lw=3)
    plt.show()

    return fitted_params
      

  n, bins = sniff_statistics(sniff_file, ephys_file, tracking_file)

  fit_distribution(n, bins)

