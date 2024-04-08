'''
Main run file for local field potential project

By: Sid Rafilson
'''

print("Importing libraries...")
from analysis import *
from core import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import os

from scipy.optimize import curve_fit
from kneed import KneeLocator
from scipy.stats import norm, pearsonr, chi2_contingency
from scipy.signal import correlate
print("Libraries imported successfully")






#__________________________________________________________________COMPLETED ANALYSIS FUNCTIONS______________________________________________________________________________________


def histogram_analysis_current(save_path: str, concatenate_session: bool = True, fitting_iters: int = 100, nmodes: int = 5, data_path = r"E:\Sid_LFP\Sid_data\rnp_final", mice = ['4122', '4127', '4131', '4138']):
    
    '''
    Function to perform histogram analysis on sniff data.
    This function will load the sniff data, find the inhalation times, and compute the frequency of inhalations. It will then perform a histogram analysis on the data, 
    fitting a log-normal distribution to the data. The function will then plot the histogram and the fitted distribution, and save the figure to the specified directory.
    '''

    path = data_path


    for condition in ['freemoving', 'headfixed']:



        # preallocating list to hold sum of squares for each mouse
        if concatenate_session:
            all_sums = []

            # preallocating a list of lists to hold the parameters for each mouse. list_of_lists_of_params[i][j] are the parameters for the jth mode of the ith mouse
            # params are ordered as follows: [weight, mu, sigma] for each mode
            list_of_lists_of_params = []

        else:
            all_sums_4122 = []
            all_sums_4127 = []
            all_sums_4131 = []
            all_sums_4138 = []

            list_of_lists_of_params_4122 = []
            list_of_lists_of_params_4127 = []
            list_of_lists_of_params_4131 = []
            list_of_lists_of_params_4138 = []


        for mouse in mice:
            print(f"Processing mouse {mouse}")
            mouse_path = path + '\\' + mouse
            sessions = os.listdir(mouse_path)
            if concatenate_session:
                concatenated_signal = np.array([])
            for session in sessions:
                print(f"Processing session {session}")
                session_path = os.path.join(mouse_path, session)
                sniff_file = [file for file in os.listdir(session_path) if file.endswith('sniff_params.mat')]
                if sniff_file:

                    # extracting inhalation times and computing frequency
                    inhalation_times, _, exhalation_times, _ = load_sniff_MATLAB(os.path.join(session_path, sniff_file[0]))
                    inhalation_freqs = 1000 / np.diff(inhalation_times)
                    inhalation_times = inhalation_times[:-1]


                    # finding conditions
                    events_file = [file for file in os.listdir(session_path) if file.endswith('events.mat')]
                    events_file = os.path.join(session_path, events_file[0])
                    events_mat = scipy.io.loadmat(events_file)
                    events = events_mat['events']

                    # load mask files
                    freemoving_mask, headfixed_mask, = find_condition_mask_inhales(events, np.max(inhalation_times))

                    # getting the inhalation times and frequencies for the current condition
                    if condition == 'freemoving':
                        freqs = inhalation_freqs[np.isin(inhalation_times, freemoving_mask)]
                        inhalation_times = inhalation_times[np.isin(inhalation_times, freemoving_mask)]
                        
                    elif condition == 'headfixed':
                        freqs = inhalation_freqs[np.isin(inhalation_times, headfixed_mask)]
                        inhalation_times = inhalation_times[np.isin(inhalation_times, headfixed_mask)]
                        
                    
                    # concatenating signal if necessary
                    if concatenate_session:
                        if concatenated_signal.size == 0:
                            concatenated_signal = freqs
                        else:
                            concatenated_signal = np.concatenate((concatenated_signal, freqs))

                    # if not concatenating, perform analysis for each session
                    else:
                        if freqs.size == 0:
                            continue

                        p_vals, sums, params = probability_distribution_analysis_updated(freqs, savepath = save_path, method = 'log-normal', nmodes = nmodes, niters = fitting_iters, condition = condition, mouse = mouse, session = session)
                        if mouse == '4122':
                            all_sums_4122.extend(sums)
                            list_of_lists_of_params_4122.append(params)
                        elif mouse == '4127':
                            all_sums_4127.extend(sums)
                            list_of_lists_of_params_4127.append(params)
                        elif mouse == '4131':
                            all_sums_4131.extend(sums)
                            list_of_lists_of_params_4131.append(params)
                        elif mouse == '4138':
                            all_sums_4138.extend(sums)
                            list_of_lists_of_params_4138.append(params)

            if concatenate_session:
                p_vals, sums, params = probability_distribution_analysis_updated(concatenated_signal, savepath = save_path, method = 'log-normal', nmodes = nmodes, niters = fitting_iters, condition = condition, mouse = mouse)
                all_sums.extend(sums)
                list_of_lists_of_params.append(params)

        if concatenate_session:

            num_rows = len(all_sums) // nmodes
            sumerror = np.array(all_sums).reshape(num_rows, nmodes)

            parameter_dict = {}
            for i, sublist in enumerate(list_of_lists_of_params):
                for j, arr in enumerate(sublist):
                    key = f'parameters_mouse_{mice[i]}_modes_{j + 1}'
                    parameter_dict[key] = arr
            
            np.savez(os.path.join(save_path, f"{condition}_parameters.npz"), **parameter_dict)


            # plotting boxplot of sum of squares
            plt.figure()
            sns.boxplot(sumerror, color = 'dodgerblue', patch_artist=True, showfliers=True)
            plt.xlabel('Number of modes')
            plt.xticks(np.arange(0, nmodes), np.arange(1, nmodes+1))
            plt.ylabel('sum of squares')
            plt.title(f'Sum of squares for model complexity \n {condition}')
            plt.savefig(os.path.join(save_path, f"{condition}_sums.png"))
            plt.close()

            plt.figure()
            sns.boxplot(sumerror, color = 'dodgerblue', patch_artist=True)
            plt.xlabel('Number of modes')
            plt.xticks(np.arange(0, nmodes), np.arange(1, nmodes+1))
            plt.ylabel('log sum of squares')
            plt.gca().set_yscale('log')
            plt.title(f'Sum of squares for model complexity \n {condition}')
            plt.savefig(os.path.join(save_path, f"{condition}_log_sums.png"))
            plt.close()


            # plotting individual sum of squares lineplot
            plt.figure()
            for i in range(num_rows):
                sns.lineplot(x = np.arange(1, nmodes+1), y = sumerror[i,:], label = f"Mouse {mice[i]}")
                print(f"Mouse {mice[i]}: {sumerror[i,:]}")
            plt.xlabel('Number of modes')
            plt.ylabel('log sum of squares')
            plt.title(f'Sum of squares for model complexity \n {condition}')
            plt.gca().set_yscale('log')
            plt.legend()
            plt.savefig(os.path.join(save_path, f"{condition}_log_sums_indiv.png"))
            plt.close()

            # fitting each line to exponential decay
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            def exp_decay_jacobian(x, a, b, c):
                da = np.exp(-b * x)
                db = -a * x * np.exp(-b * x)
                dc = np.ones_like(x)
                return np.vstack([da, db, dc]).T
            
            
            all_knees = []
            for i in range(num_rows):
                x_data = np.arange(1, nmodes+1)
                x_data_extended = np.linspace(1, nmodes, 100)
                y_data = sumerror[i,:]

                params, cov = curve_fit(exp_decay, x_data, y_data, p0 = [0.5, 2, 0.001], jac = exp_decay_jacobian)

                a, b, c = params
                
                

                #finding knee points
                knee_locator = KneeLocator(x_data_extended, exp_decay(x_data_extended, a, b, c), curve = 'convex', direction = 'decreasing')
                knee_locator_raw = KneeLocator(x_data, y_data, curve = 'convex', direction = 'decreasing')
                all_knees.append(knee_locator.knee)

                plt.figure()
                sns.lineplot(x = x_data, y = y_data, label = f"Mouse {mice[i]}")
                sns.lineplot(x = x_data_extended, y = exp_decay(x_data_extended, a, b, c), label = r"$y = {:.2f}e^{{-{:.2f}x}} + {:.3f}$".format(a, b, c))
                plt.axvline(knee_locator.knee, color = 'red', linestyle = '--', label = f"Knee fit: {np.round(knee_locator.knee, 3)}")
                plt.axvline(knee_locator_raw.knee, color = 'green', linestyle = '--', label = f"Knee raw: {np.round(knee_locator_raw.knee, 3)}")
                plt.xlabel('Number of modes')
                plt.ylabel('log sum of squares')
                plt.title(f' error function of model complexity with knee points \n {condition}')
                plt.gca().set_yscale('log')
                plt.legend()
                plt.savefig(os.path.join(save_path, f"{condition}_log_sums_indiv_exp_fit_{mice[i]}.png"))
                plt.close()


            # plotting histogram of knee points
            bins = np.arange(0.5, nmodes+1.5, 1)
            plt.figure()
            sns.histplot(all_knees, color = 'dodgerblue', bins = bins)
            plt.xlim(0, 5)
            plt.xlabel('Knee point (number of modes)')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of knee points for model complexity \n {condition}')
            plt.savefig(os.path.join(save_path, f"{condition}_knees.png"))
            plt.close()


        else:

            all_sums = all_sums_4122 + all_sums_4127 + all_sums_4131 + all_sums_4138
            num_rows = len(all_sums) // nmodes
            sumerror = np.array(all_sums).reshape(num_rows, nmodes)





            

            parameter_dict = {}
            for i, sublist in enumerate(list_of_lists_of_params_4122):
                for j, arr in enumerate(sublist):
                    key = f'parameters_mouse_4122_modes_{j + 1}'
                    parameter_dict[key] = arr
            
            np.savez(os.path.join(save_path, f"4122_{condition}_parameters.npz"), **parameter_dict)

            parameter_dict = {}
            for i, sublist in enumerate(list_of_lists_of_params_4127):
                for j, arr in enumerate(sublist):
                    key = f'parameters_mouse_4127_modes_{j + 1}'
                    parameter_dict[key] = arr
            
            np.savez(os.path.join(save_path, f"4127_{condition}_parameters.npz"), **parameter_dict)

            parameter_dict = {}
            for i, sublist in enumerate(list_of_lists_of_params_4131):
                for j, arr in enumerate(sublist):
                    key = f'parameters_mouse_4131_modes_{j + 1}'
                    parameter_dict[key] = arr
            
            np.savez(os.path.join(save_path, f"4131_{condition}_parameters.npz"), **parameter_dict)

            parameter_dict = {}
            for i, sublist in enumerate(list_of_lists_of_params_4138):
                for j, arr in enumerate(sublist):
                    key = f'parameters_mouse_4138_modes_{j + 1}'
                    parameter_dict[key] = arr
            
            np.savez(os.path.join(save_path, f"4138_{condition}_parameters.npz"), **parameter_dict)

            plt.figure()
            sns.boxplot(sumerror, color = 'dodgerblue', patch_artist=True, showfliers=True)
            plt.xlabel('Number of modes')
            plt.xticks(np.arange(0, nmodes), np.arange(1, nmodes+1))
            plt.ylabel('sum of squares')
            plt.title(f'Sum of squares for model complexity \n {condition}')
            plt.savefig(os.path.join(save_path, f"{condition}_sums.png"))
            plt.close()

            plt.figure()
            sns.boxplot(sumerror, color = 'dodgerblue', patch_artist=True)
            plt.xlabel('Number of modes')
            plt.xticks(np.arange(0, nmodes), np.arange(1, nmodes+1))
            plt.ylabel('log sum of squares')
            plt.gca().set_yscale('log')
            plt.title(f'Sum of squares for model complexity \n {condition}')
            plt.savefig(os.path.join(save_path, f"{condition}_log_sums.png"))
            plt.close()

            plt.figure()
            for i in range(num_rows):
                sns.lineplot(x = np.arange(1, nmodes+1), y = sumerror[i,:])
            plt.xlabel('Number of modes')
            plt.ylabel('log sum of squares')
            plt.title(f'Sum of squares for model complexity \n {condition}')
            plt.gca().set_yscale('log')
            plt.legend()
            plt.savefig(os.path.join(save_path, f"{condition}_log_sums_indiv.png"))
            plt.close()


            # fitting each line to exponential decay
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            def exp_decay_jacobian(x, a, b, c):
                da = np.exp(-b * x)
                db = -a * x * np.exp(-b * x)
                dc = np.ones_like(x)
                return np.vstack([da, db, dc]).T
            
            
            all_knees = []

            for i in range(num_rows):
                x_data = np.arange(1, nmodes+1)
                x_data_extended = np.linspace(1, nmodes, 100)
                y_data = sumerror[i,:]

                params, cov = curve_fit(exp_decay, x_data, y_data, p0 = [0.5, 2, 0.001], jac = exp_decay_jacobian)

                a, b, c = params
                
                

                #computing second derrivative of fitted curve
                knee_locator = KneeLocator(x_data_extended, exp_decay(x_data_extended, a, b, c), curve = 'convex', direction = 'decreasing')
                knee_locator_raw = KneeLocator(x_data, y_data, curve = 'convex', direction = 'decreasing')
                all_knees.append(knee_locator.knee)


                plt.figure()
                sns.lineplot(x = x_data, y = y_data)
                sns.lineplot(x = x_data_extended, y = exp_decay(x_data_extended, a, b, c))
                plt.axvline(knee_locator.knee, color = 'red', linestyle = '--', label = f"Knee fit: {np.round(knee_locator.knee, 3)}")
                plt.axvline(knee_locator_raw.knee, color = 'green', linestyle = '--', label = f"Knee raw: {np.round(knee_locator_raw.knee, 3)}")
                plt.xlabel('Number of modes')
                plt.ylabel('log sum of squares')
                plt.title(f' error function of model complexity with elbow points \n {condition}')
                plt.gca().set_yscale('log')
                plt.legend()
                plt.savefig(os.path.join(save_path, f"{condition}_log_sums_indiv_exp_fit_{i}.png"))
                plt.close()


            # plotting histogram of knee points
            bins = np.arange(0.5, nmodes+1.5, 1)
            plt.figure()
            sns.histplot(all_knees, color = 'dodgerblue', bins = bins)
            plt.xlabel('Knee point (number of modes)')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of knee points for model complexity \n {condition}')
            plt.savefig(os.path.join(save_path, f"{condition}_knees.png"))
            plt.close()


def low_freq_analysis_zelano(ephys: np.array, sniff_signal: np.array, sniff_times: np.array, ch: int = 0, window_size: int = 1000, null_size: int = 100, exclude_edges: int = 25, show_plot = True):
    


    # filtering low frequency data
    ephys = ephys[ch,:]
    sniff = sniff_signal.flatten()
    ephys = lowpass_sniff(ephys, 20, 3)
    sniff = lowpass_sniff(sniff, 20, 3)

    # aligning to sniff times
    lfp_epochs = np.zeros((len(sniff_times) - (2 * exclude_edges), window_size))
    sniff_epochs = np.zeros((len(sniff_times) - (2 * exclude_edges), window_size))

    # removing first and last few sniff times to avoid edge effects
    for i, time in enumerate(sniff_times[exclude_edges:-exclude_edges]):
        lfp_epochs[i,:] = ephys[time - (window_size // 2) : time + (window_size // 2)]
        sniff_epochs[i,:] = sniff[time - (window_size // 2) : time + (window_size // 2)]

    # computing average
    mean_lfp = np.mean(lfp_epochs, axis = 0)
    mean_sniff = np.mean(sniff_epochs, axis = 0)

    # compute linear correlation
    corr, p_val = pearsonr(mean_lfp, mean_sniff)
    correlation = correlate(mean_lfp, mean_sniff, mode = 'full')

    # Fisher z-transform
    fz = np.arctanh(corr)




    #___Building Null Distribution___

    # preallocating arrays to hold lfp epochs and null distribution
    null_corr_dist = np.zeros((null_size, window_size * 2 - 1))
    null_rho_dist = np.zeros((null_size))
    null_epoch = np.zeros((len(sniff_times) - 2 * exclude_edges, window_size))
   

    # building null distribution
    a = sniff_times[exclude_edges]
    b = sniff_times[-exclude_edges]
    for i in range(null_size):
        if i % 10 == 0:
            print(f"Building null distribution: {i}/{null_size}")
        null_points = np.random.randint(a, b, size = len(sniff_times) - 2 * exclude_edges)
        for j, time in enumerate(null_points):
            null_epoch[j,:] = ephys[time - (window_size // 2) : time + (window_size // 2)]
        null_mean = np.mean(null_epoch, axis = 0)

        # computing linear correlation
        null_rho_dist[i], null_p = pearsonr(null_mean, mean_sniff)
        null_corr_dist[i, :] = correlate(null_mean, mean_sniff, mode = 'full')

    # computing Fisher z-transform
    null_fz_dist = np.arctanh(null_rho_dist)





    #___Hypothesis Testing___

    # computing p-value such that the effect is significant if the correlation is greater than 95% of the null distribution
    p_fz = 1 - (null_fz_dist > fz).sum() / null_size


    print(f"Correlation: {corr}, p-value from null: {p_fz}, p-value from correlation: {p_val}")








    if show_plot:

        sns.set_style('darkgrid')

        fig, axs = plt.subplots(2, 2, figsize = (15, 10))

        sns.lineplot(x = np.arange(-window_size // 2, window_size // 2), y = mean_lfp, ax = axs[0,0], label = 'LFP')
        sns.lineplot(x = np.arange(-window_size // 2, window_size // 2), y = mean_sniff, ax = axs[0,0], label = 'Sniff')
        axs[0,0].set_title('Mean LFP and Sniff Signal')
        axs[0,0].set_xlabel('Time (ms)')
        axs[0,0].set_ylabel('Amplitude')

        lags = np.arange(-window_size + 1, window_size).astype(int)

        sns.lineplot(x = lags, y = correlation, ax = axs[0,1])
        axs[0,1].set_title('Cross-correlation')
        axs[0,1].set_xlabel('Lag')
        axs[0,1].set_ylabel('rho')


        sns.heatmap(null_corr_dist, ax = axs[1,0])
        axs[1,0].set_title('Null cross-correlation Distribution')
        axs[1,0].set_xlabel('Lag')
        axs[1,0].set_ylabel('iteration')
        axs[1,0].set_xticks([0, window_size - 1, 2 * window_size - 2])
        axs[1,0].set_xticklabels([-window_size // 2, 0, window_size // 2])
        axs[1,0].set_yticks([])


        sns.lineplot(x = np.arange(null_size), y = null_fz_dist, ax = axs[1,1])
        axs[1,1].set_title('Null Fisher z-transform Distribution')
        axs[1,1].set_xlabel('iteration')
        axs[1,1].set_ylabel('Z')


        plt.tight_layout()
        plt.show()

    return mean_lfp, fz, p_fz

        
def build_spectrograms(mice = ['4122', '4127', '4138', '4131'], data_dir = r"E:\Sid_LFP\Sid_data\rnp_final", save_dir = r"E:\Sid_LFP\figs\spectrograms_complete"):
    for mouse in mice:
        mouse_dir = os.path.join(data_dir, mouse)
        sessions = os.listdir(mouse_dir)
        for session in sessions:
            session_dir = os.path.join(mouse_dir, session)
            files = os.listdir(session_dir)
            if 'LFP.npy' in files and 'sniff_signal.mat' in files:
                sniff_signal = get_sniff_signal_MAT(os.path.join(session_dir, 'sniff_signal.mat'))
                ephys_signal = np.load(os.path.join(session_dir, 'LFP.npy'))
                condition_file = scipy.io.loadmat(os.path.join(session_dir, 'events.mat'))
                events = condition_file['events']
                print(f"Processing mouse {mouse}, session {session}")

                #creating directory to save spec_file
                spec_file_path = os.path.join(save_dir, mouse, session)
                if not os.path.exists(spec_file_path):
                    os.makedirs(spec_file_path)

                spec_values_path = os.path.join(spec_file_path, 'values')
                if not os.path.exists(spec_values_path):
                    os.makedirs(spec_values_path)



                # getting conditions
                sniff_signal = sniff_signal.flatten()
                _, _, freemoving_bool, headfixed_bool = find_condition_mask(events, sniff_signal)
                freemoving_sniff = sniff_signal[freemoving_bool]
                headfixed_sniff = sniff_signal[headfixed_bool]
                for i in range(0, ephys_signal.shape[0], 4):
                    print(f"Processing channel {i}")
                    save_path = os.path.join(spec_file_path, f"channel_{i}")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    freemoving_ephys = ephys_signal[i, freemoving_bool]
                    freemoving_ephys = freemoving_ephys.flatten()
                    headfixed_ephys = ephys_signal[i, headfixed_bool]
                    headfixed_ephys = headfixed_ephys.flatten()

                    # computing, plotting and saving spectrograms
                    ephys_spec, sniff_spec, cross_spec, x_positions, y_positions = spectrogram_analysis(
                        sniff_signal[:], ephys_signal[:,:], spec_file_path = save_path, channel = i
                    )

                    np.savez(os.path.join(spec_values_path, f"channel_{i}_values.npz"),
                              ephys_spec = ephys_spec, sniff_spec = sniff_spec, cross_spec = cross_spec, x_positions = x_positions, y_positions = y_positions)
            else:
                print(f"Skipping mouse {mouse}, session {session}")


def avg_lfp_analysis(ephys: np.array, sniff: np.array, save_path: str, channel: int = 0, window_size: int = 1000, nbins: int = 10, freq_range: tuple = (2,16), nshifts: int = 100, plot_save = True, show_peakfinder = True):

    """
    Analyzes the local field potential (LFP) data by computing z-scored LFP amplitudes, detecting peaks and troughs, 
    and comparing LFP frequencies to sniff frequencies. The analysis is performed within specified frequency bins 
    across a given channel. Results include peak-to-peak and peak-to-trough frequencies, heights of peaks, depths of troughs, 
    and z-scored LFP signal visualization. 

    Parameters:
    - ephys (np.array): The LFP data array with dimensions (nchannels, time).
    - sniff (np.array): The sniff signal array (unused in this function, but typically part of the analysis pipeline).
    - save_path (str): The directory path where results and figures will be saved.
    - channel (int, optional): The channel index to analyze. Defaults to 0.
    - window_size (int, optional): The size of the analysis window in milliseconds. Defaults to 1000.
    - nbins (int, optional): The number of frequency bins to divide the `freq_range` into for analysis. Defaults to 10.
    - freq_range (tuple, optional): The frequency range (in Hz) within which to analyze the LFP data. Defaults to (2, 16).
    - nshifts (int, optional): The number of shifts used for computing the null distribution of z-scored LFP amplitudes. Defaults to 100.
    - show_peakfinder (bool, optional): If True, generates and saves plots of the peak and trough detection process. Defaults to True.
    
    Returns:
    - None: This function does not return a value but saves several files to `save_path` including npz files with analysis results
      and PNG images of plots visualizing these results.

    Notes:
    - This function relies on external functions `avg_lfp_infreq` for computing z-scored LFP amplitudes within frequency bins,
      and `find_inhales` for detecting peaks and troughs in the LFP signal.
    - Results are saved in various formats, including numpy arrays for numerical data and PNG images for visualizations.
    - The function prints progress and result summaries to the console during execution.
    """

    nchannels = ephys.shape[0]
    
    l = freq_range[1] - freq_range[0]
    print(f"Computing average lfp analysis for channel {channel} \nFrequency range: {freq_range} Hz \nWindow size: {window_size} ms \nNumber of bins: {nbins} \nNumber of shifts: {nshifts}\n\n")


    # preallocating arrays to hold results
    ephys_freqs_p2p = np.zeros((nbins))
    ephys_freqs_p2t = np.zeros((nbins))
    zscore = np.zeros((nbins, nchannels, window_size))
    freqs = np.zeros((nbins))
    hights = np.zeros((nbins))
    dips = np.zeros((nbins))
    mid_peak = np.zeros((nbins))
    mid_trough = np.zeros((nbins))

    # looping through frequency bins
    for i in range(nbins):
        print('\n')
        current_range = (freq_range[0] + i * (l / nbins), freq_range[0] + (i + 1) * (l / nbins))
        print(f"Processing bin {i} of {nbins} \ncurrent range: {current_range} Hz")
        # computhing z-scored lfp amplitude from null distirbution
        z, f = avg_lfp_infreq(ephys, sniff, save_path, freq_range = current_range, window_size = window_size, nshifts = nshifts, channel = channel)

        # finding peaks in zscored lfp signal
        peaks, smoothed_signal, _ = find_inhales(z[channel,:], window_length = 100, polyorder = 7, min_peak_prominance = 3, save_figs = False, signal_type = 'lfp')
        troughs, _, _ = find_inhales(-z[channel,:], window_length = 100, polyorder = 7, min_peak_prominance = 3, save_figs = False, signal_type = 'lfp')

        peak_heights = []
        trough_dips = []
        peak_location = []
        trough_location = []

        # finding peak and trough heights by scanning in small window around peak of smoothed signal
        for peak in peaks:
            start_peak = max(peak - 20, 0)
            end_peak = min(peak + 20, window_size)
            window_peak = z[channel, start_peak:end_peak]
            max_height = np.max(window_peak)
            peak_heights.append(max_height)
            peak_location.append(np.where(z[channel,:] == max_height)[0][0])
        
        for trough in troughs:
            start_trough = max(trough - 20, 0)
            end_trough = min(trough + 20, window_size)
            window_trough = z[channel, start_trough:end_trough]
            min_height = np.min(window_trough)
            trough_dips.append(min_height)
            trough_location.append(np.where(z[channel,:] == min_height)[0][0])

        #excluding peaks and troughs before or around time-lag 0
        
        #finding peak nearest to middle of window
        if len(peak_location) > 0:
            middle_peak = np.argmin(np.abs(np.array(peak_location) - window_size // 2))
        else:
            middle_peak = 0

        if len(trough_location) > 0:
            pos_troughs = np.where(np.array(trough_location) > window_size // 2)[0]
            if len(pos_troughs) > 0:
                middle_trough = np.argmin(np.array(pos_troughs))
            else:
                middle_trough = 0
        else:
            middle_trough = 0


        
        


        peaks = [peak for peak in peak_location if peak > window_size // 2]
        troughs = [trough for trough in trough_location if trough > window_size // 2]
        peak_heights = [height for i, height in enumerate(peak_heights) if peak_location[i] > window_size // 2]
        trough_dips = [dip for i, dip in enumerate(trough_dips) if trough_location[i] > window_size // 2]
        




        # plotting peaks and troughs
        if show_peakfinder:
            plt.figure(figsize=(10,6))
            sns.lineplot(x = np.arange(window_size), y = z[channel,:], label = 'z-scored lfp', color = 'dodgerblue')
            sns.lineplot(x = np.arange(window_size), y = smoothed_signal, label = 'smoothed lfp', color = 'crimson')
            sns.scatterplot(x = peaks, y = peak_heights, label = 'peaks', color = 'black')
            sns.scatterplot(x = troughs, y = trough_dips, color = 'black')
            plt.xticks(np.arange(0, window_size, 250), np.arange(-window_size // 2, window_size // 2, 250))
            plt.axvline(window_size // 2, color = 'grey', linestyle = '--', label = 'inhalation', alpha = 0.8)
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude')
            plt.title(f'Peak and Trough Detection \n frequencies: {current_range} Hz')
            plt.legend()
            plt.savefig(os.path.join(save_path, f"channel_{channel}_freqs_{current_range}_peaks_troughs.png"))
            plt.close()


        
        # calculating instantaneous lfp frequency from peak and trough times
        if len(peaks) > 1:
            peak2peak = 1000 / (peaks[1] - peaks[0])
        else:
            peak2peak = 0

        if len(troughs) > 1:
            peak2trough = 500 / (troughs[0] - peaks[0])
        else:
            peak2trough = 0

        
        # handling edge cases
        if len(peak_heights) == 0:
            peak_heights = [0]
        if len(trough_dips) == 0:
            trough_dips = [0]


        # saving results
        ephys_freqs_p2p[i] = np.abs(peak2peak)
        ephys_freqs_p2t[i] = np.abs(peak2trough)
        hights[i] = np.max(peak_heights)
        dips[i] = np.abs(np.min(trough_dips))
        zscore[i,:, :] = z
        freqs[i] = np.mean(f)
        mid_peak[i] = middle_peak
        mid_trough[i] = middle_trough



    # building pandas dataframe to hold results
    results = pd.DataFrame({'ephys_freqs_p2p': ephys_freqs_p2p, 'ephys_freqs_p2t': ephys_freqs_p2t, 'freqs': freqs, 'heights': hights, 'dips': dips, 'mid_peak': mid_peak, 'mid_trough': mid_trough})
    results_melted_freqs = pd.melt(results, id_vars=['freqs'], value_vars=['ephys_freqs_p2p', 'ephys_freqs_p2t'], var_name='measurement', value_name='frequency')
    results_melted_heights = pd.melt(results, id_vars=['freqs'], value_vars=['heights', 'dips'], var_name='measurement', value_name='amplitude')

    if plot_save:
        # saving results
        np.savez(os.path.join(save_path, f"channel_{channel}_results.npz"), ephys_freqs_p2p = ephys_freqs_p2p, ephys_freqs_p2t = ephys_freqs_p2t, zscore = zscore, freqs = freqs, hights = hights, dips = dips)

        # plotting heatmap of zscored lfp signal
        plt.figure(figsize=(10,6))
        sns.heatmap(
            zscore[:, channel, :], cmap = 'coolwarm', cbar = True)
        plt.gca().invert_yaxis()
        ytick_labels = np.arange(freq_range[0], freq_range[1], 1)
        ytick_positions = np.linspace(0, zscore.shape[0]-1, len(ytick_labels))
        plt.yticks(ticks = ytick_positions, labels = ytick_labels)
        xtick_labels = np.arange(-window_size // 2, window_size // 2 + 1, 250)
        xtick_positions = np.linspace(0, zscore.shape[2]-1, len(xtick_labels))
        plt.xticks(ticks = xtick_positions, labels= xtick_labels)
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Z-scored LFP signal')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"channel_{channel}_zscore.png"))
        plt.close()


        # plotting peak and trough frequencies
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_melted_freqs, x='freqs', y='frequency', hue = 'measurement', palette = ['red', 'blue'])
        plt.xticks(ticks = ytick_positions, labels = ytick_labels)
        plt.xlabel('Sniff Frequency (Hz)')
        plt.ylabel('LFP Frequency (Hz)')
        plt.title('LFP Frequency vs Sniff Frequency')
        plt.legend()
        plt.savefig(os.path.join(save_path, f"channel_{channel}_freqs.png"))
        plt.close()



        # plotting peak and trough heights
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_melted_heights, x='freqs', y='amplitude', hue = 'measurement', palette = ['red', 'blue'])
        plt.xticks(ticks = ytick_positions, labels = ytick_labels)
        plt.xlabel('Sniff Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('Peak and Trough Amplitudes')
        plt.legend()
        plt.savefig(os.path.join(save_path, f"channel_{channel}_amplitudes.png"))
        plt.close()

    return results



def concatenate_for_lfp_analysis(data_dir: str, save_path: str, mice: list = ['4122', '4127', '4131', '4138']):


    # concatenating lfp and sniff signal across sessions
    for mouse in mice:
        mouse_dir = os.path.join(data_dir, mouse)
        sessions = os.listdir(mouse_dir)

        # preallocating arrays to hold concatenated data
        concatenated_ephys = np.array([])
        concatenated_inh_times = np.array([])
        concatenated_exh_times = np.array([])

        # setting start time to 0 for first session (this will be used to align sniff times across sessions)
        start_time = 0

        # looping through sessions
        for session in sessions:
            session_dir = os.path.join(mouse_dir, session)
            files = os.listdir(session_dir)

            # checking if session has lfp and sniff signal
            if 'LFP.npy' in files and 'sniff_params.mat' in files:

                # loading lfp and sniff signal
                ephys_signal = np.load(os.path.join(session_dir, 'LFP.npy'))
                concatenated_ephys = np.concatenate((concatenated_ephys, ephys_signal), axis = 1)
                inh_times, _, exh_times, _ = load_sniff_MATLAB(os.path.join(session_dir, 'sniff_params.mat'))

                # aligning sniff times across sessions
                inh_times = inh_times + start_time
                exh_times = exh_times + start_time
                concatenated_inh_times = np.concatenate((concatenated_inh_times, inh_times))
                concatenated_exh_times = np.concatenate((concatenated_exh_times, exh_times))
                start_time += ephys_signal.shape[1]

        # saving concatenated data
        np.savez(os.path.join(save_path, f"{mouse}_concatenated_data.npz"), ephys = concatenated_ephys, inh_times = concatenated_inh_times)


def avg_lfp_analysis_conditions(
        ephys_path: str, sniff_path: str, events_path: str, save_dir: str,
        channel: int = 0, window_size: int = 1000, nbins: int = 10,
        freq_range: tuple = (2,16), nshifts: int = 100, show_peakfinder = True):
    
    # loading data
    print()
    ephys = np.load(ephys_path)
    inh_times, _, exh_times, _ = load_sniff_MATLAB(sniff_path)
    events_mat = scipy.io.loadmat(events_path)
    events = events_mat['events']

    # Computing average lfp analysis for each sniff event type
    for sniff_event in ['exhales', 'inhales']:
        if sniff_event == 'inhales':
            sniff_times = inh_times
        else:
            sniff_times = exh_times

    
        # computing sniff frequencies
        freqs = 1000 / np.diff(sniff_times)
        sniff_times = sniff_times[:-1]
        

        # finding condition masks
        freemoving_mask, headfixed_mask, = find_condition_mask_inhales(events, np.max(sniff_times))

        # computing average lfp analysis for each condition
        for condition in ['freemoving', 'headfixed']:
            if condition == 'freemoving':
                condition_mask = freemoving_mask
            else:
                condition_mask = headfixed_mask


            # creating save path folder
            save_path = os.path.join(save_dir, sniff_event, condition)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x = sniff_times / 60_000, y = freqs, color = 'crimson', label = 'all')


            # getting sniff times for the condition
            freqs_condition = freqs[np.isin(sniff_times, condition_mask)]
            sniff_times_condition = sniff_times[np.isin(sniff_times, condition_mask)]


            # plotting sniff frequency scatterplot
            sns.scatterplot(x = sniff_times_condition / 60_000, y = freqs_condition, color = 'dodgerblue', label = 'condition')
            plt.xlabel('Time (min)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'Sniff Frequency Scatterplot for {sniff_event} {condition}')
            plt.savefig(os.path.join(save_path, f"sniff_freq_scatter.png"))
            plt.legend()
            plt.close()

            # plotting sniff frequency histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(freqs_condition, color = 'dodgerblue', bins = 40)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Frequency')
            plt.title(f'Sniff Frequency Histogram for {sniff_event} {condition}')
            plt.savefig(os.path.join(save_path, f"sniff_freq_hist.png"))
            plt.close()


            # setting LFP from outside of condition to NaN
            ephys_condition = ephys.copy()
            ephys_condition[:, ~np.isin(np.arange(ephys.shape[1]), condition_mask)] = 0

            # plotting lfp signal and sniff times
            plt.figure(figsize=(10, 6))
            sns.lineplot(x = np.arange(0, ephys.shape[1]), y = ephys[channel,:], color = 'dodgerblue', label = 'LFP')
            sns.lineplot(x = np.arange(0, ephys_condition.shape[1]), y = ephys_condition[channel, :], color = 'crimson', label = 'condition')
            sns.scatterplot(x = sniff_times_condition, y = ephys[channel, sniff_times_condition], color = 'red', label = 'Sniff times', s = 40)
            plt.title(f'LFP signal and Sniff Times overlay for {sniff_event} {condition}')
            plt.savefig(os.path.join(save_path, f"channel_{channel}_ephys_sniff.png"))
            plt.close()

            plt.figure(figsize=(10, 6))
            sns.lineplot(x = np.arange(0, ephys_condition.shape[1]), y = ephys_condition[channel,:], color = 'dodgerblue', label = 'LFP')
            sns.scatterplot(x = sniff_times_condition, y = freqs_condition * 2_000, color = 'red', label = 'Sniff times', s = 10, zorder = 10)
            plt.title(f'LFP signal and Sniff Times for {sniff_event} {condition}')
            plt.savefig(os.path.join(save_path, f"channel_{channel}_ephys_sniff_onlycondition.png"))

            # computing average lfp analysis
            print(f"Computing average lfp analysis for {sniff_event} {condition}")
            avg_lfp_analysis(ephys_condition[channel:channel + 1, :], sniff_times_condition, save_path, channel = channel, window_size = window_size, nbins = nbins, freq_range = freq_range, nshifts = nshifts, show_peakfinder = show_peakfinder)



def build_hist_analysis_panel():

    save_path = r"E:\Sid_LFP\figs\Poster"

    nmodes = 10

    sns.set_style('white')


    fig, axs = plt.subplots(2, 2, figsize = (10, 10), sharey = True, sharex = False)
    sns.despine()

    fig.text(0.06, 0.5, 'Probability Density', ha='center', va='center', rotation='vertical', fontsize=20, color = 'black')
    fig.text(0.5, 0.04, 'Instantaneous Sniff Frequency (Hz)', ha='center', va='center', fontsize=20, color = 'black')

    

    # loading parameters from mixture distributions
    freemoving_params = np.load(r"E:\Sid_LFP\figs\histogram_analysis_\freemoving_parameters.npz")
    headfixed_params = np.load(r"E:\Sid_LFP\figs\histogram_analysis_\headfixed_parameters.npz")

    # defining log-spaces bins for histogram
    bin_edges = np.logspace(start = np.log10(1), stop = np.log10(15), num = 40)


    # defining the base log-normal distribution
    def log_normal(x, mu, sigma):
        return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x) - mu) ** 2 / (2 * sigma ** 2))


    # defining the mixture distribution
    def mixture(params, x, n_distributions):
        distribution = 0
        for i in range(n_distributions):
            distribution += params[(3 * i)] * log_normal(x, params[(3 * i) + 1], params[(3 * i) + 2])
        return distribution


    # defining x values for plotting
    x_vals = np.linspace(0.01, 20, 1000)




    print(freemoving_params.files)

    all_all_sums = []

    conditions = ['freemoving', 'headfixed']
    mice = ['4131', '4138', '4127', '4122']
    path = r"E:\Sid_LFP\Sid_data\rnp_final"
    for condition in conditions:

        all_sums = []


        for mouse in mice:
            print(f"Processing mouse {mouse}")
            mouse_path = path + '\\' + mouse
            sessions = os.listdir(mouse_path)
            concatenated_signal = np.array([])
            for session in sessions:
                print(f"Processing session {session}")
                session_path = os.path.join(mouse_path, session)
                sniff_file = [file for file in os.listdir(session_path) if file.endswith('sniff_params.mat')]
                if sniff_file:

                    # extracting inhalation times and computing frequency
                    inhalation_times, _, exhalation_times, _ = load_sniff_MATLAB(os.path.join(session_path, sniff_file[0]))
                    inhalation_freqs = 1000 / np.diff(inhalation_times)
                    inhalation_times = inhalation_times[:-1]


                    # finding conditions
                    events_file = [file for file in os.listdir(session_path) if file.endswith('events.mat')]
                    events_file = os.path.join(session_path, events_file[0])
                    events_mat = scipy.io.loadmat(events_file)
                    events = events_mat['events']

                    # load mask files
                    freemoving_mask, headfixed_mask, = find_condition_mask_inhales(events, np.max(inhalation_times))

                    # getting the inhalation times and frequencies for the current condition
                    if condition == 'freemoving':
                        freqs = inhalation_freqs[np.isin(inhalation_times, freemoving_mask)]
                        inhalation_times = inhalation_times[np.isin(inhalation_times, freemoving_mask)]
                        
                    elif condition == 'headfixed':
                        freqs = inhalation_freqs[np.isin(inhalation_times, headfixed_mask)]
                        inhalation_times = inhalation_times[np.isin(inhalation_times, headfixed_mask)]
                        
                    
                    # concatenating signal 
                    if concatenated_signal.size == 0:
                        concatenated_signal = freqs
                    else:
                        concatenated_signal = np.concatenate((concatenated_signal, freqs))
                    

        
            # plotting the mixture distribution
            if condition == 'freemoving':
                params = freemoving_params
                row = 0
            else:
                params = headfixed_params
                row = 1

            if mouse == '4131':
                col = 0
            elif mouse == '4138':
                col = 1
            else:
                col = 2

            if col != 2:

                colors = ['dodgerblue', 'crimson']
                # plotting histogram
                sns.histplot(concatenated_signal, bins = bin_edges, color = colors[row], ax=axs[row, col], stat='density')

                #enlarging y ticks
                axs[row, col].set_yticks(np.arange(0, 0.5, 0.1), ['0', '0.1', '0.2', '0.3', '0.4'], fontsize = 18)

                # plotting mixture distributions
                for i in range(1,11):
                    y_vals = mixture(params[f'parameters_mouse_{mouse}_modes_{i}'], x_vals, i)
                    sns.lineplot(x = x_vals, y = y_vals, color = 'grey', ax=axs[row, col], alpha = 0.5)


                axs[row, col].set_ylabel('')
                axs[row, col].set_xscale('log')
                axs[row, col].set_xlim([1, 16])
                if row == 1:
                    axs[row, col].set_xticks([1, 2, 4, 8, 16], ['1', '2', '4', '8', '16'], fontsize = 18)
                else:
                    axs[row, col].set_xticks([1,2,4,8,16], ['' for i in range(5)])
                




            n, bins = np.histogram(concatenated_signal, bins = bin_edges, density = True)

            # getting the middle value of each bin
            bin_midpoints = (bins[1:] + bins[:-1]) / 2

            for i in range(1,11):
                current_sum = np.sum((n - mixture(params[f'parameters_mouse_{mouse}_modes_{i}'], bin_midpoints, i)) ** 2)
                all_sums.append(current_sum)

        num_rows = len(all_sums) // nmodes
        sumerror = np.array(all_sums).reshape(num_rows, nmodes)

        all_all_sums.append(sumerror)

    
    legend_elements = [Rectangle((0, 0), 1, 1, color='crimson', label='Head Fixation', alpha=0.7),
                    Rectangle((0, 0), 1, 1, color='dodgerblue', label='Freely Exploring', alpha=0.8)]


    axs[0,1].legend(handles=legend_elements, loc = 'upper right', fontsize = 18)





    plt.savefig(os.path.join(save_path, 'histogram_panel.png'))
    plt.close()





    # locating knee points

    all_knees = []

    fig, axs = plt.subplots(2,1, figsize = (8, 10), sharex = True, sharey = True)

    fig.text(0.5, 0.04, 'Model Complexity', ha='center', va='center', fontsize=20, color = 'black')
    fig.text(0.06, 0.5, 'Log Sum of Squared Errors', ha='center', va='center', rotation='vertical', fontsize=20, color = 'black')

    for con in range(2):
        con_knees = []

        sumerror = all_all_sums[con]
        sumerror = np.log(sumerror)

        # plotting individual sum of squares lineplot
        for i in range(num_rows):
            sns.lineplot(x = np.arange(1, nmodes+1), y = sumerror[i,:], ax = axs[con], color = colors[con])

            #undoing log transformation
            sumerror[i,:] = np.exp(sumerror[i,:])

            # finding knee point
            knee = KneeLocator(np.arange(1, nmodes+1), sumerror[i,:], curve='convex', direction='decreasing')
            con_knees.append(knee.knee)

            sns.scatterplot(x = [knee.knee], y = np.log(sumerror[i, knee.knee - 1]), ax = axs[con], color = 'grey', s = 50)

        plt.xticks(np.arange(1, nmodes+1), np.arange(1, nmodes+1), fontsize = 18)
        axs[con].set_yticks(np.arange(-8, -2), ['-8', '-7', '-6', '-5', '-4', '-3'], fontsize = 18)


        all_knees.append(con_knees)
    
    plt.savefig(os.path.join(save_path, 'knee_points.png'))
    plt.close()


    knee_point_df = pd.DataFrame({'freemoving': all_knees[0], 'headfixed': all_knees[1]})




    # plotting barchart of knee points that show counts of each knee point color coded by condition
    knee_point_df = knee_point_df.melt(var_name = 'condition', value_name = 'knee')
    fig, ax = plt.subplots(figsize = (8, 10))

    fig.text(0.5, 0.04, 'Model Complexity at Elbow Point', ha='center', va='center', fontsize=20, color = 'black')
    fig.text(0.06, 0.5, 'Count', ha='center', va='center', rotation='vertical', fontsize=20, color = 'black')

    sns.countplot(data = knee_point_df, x = 'knee', hue = 'condition', palette = ['dodgerblue', 'crimson'])

    #removing legend and axis titles
    ax.get_legend().remove()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks(np.arange(0, 6, 1), ['0', '1', '2', '3', '4', '5'], fontsize = 18)
    ax.set_xticks([0, 1], [2,3], fontsize = 18)


    # computing chi squared test
    chi2, p, _, _ = chi2_contingency(knee_point_df.groupby(['condition', 'knee']).size().unstack().fillna(0))
    print(f"Chi squared test: {chi2}, p-value: {p}")


    plt.savefig(os.path.join(save_path, 'knee_point_barchart.png'))
    plt.close()


def build_sniff_raster_panel():
    save_path = r"E:\Sid_LFP\figs\Poster"

    ephys_file = r"E:\Sid_LFP\Sid_data\rnp_final\4127\4\LFP.npy"
    sniff_file = r"E:\Sid_LFP\Sid_data\rnp_final\4127\4\sniff_params.mat"

    inh, _, exh, _ = load_sniff_MATLAB(sniff_file)
    ephys = np.load(ephys_file)

 

    


    fig, axs = plt.subplots(1, 3, figsize = (22, 8), sharex = True, sharey = True)

    channel = 0

    for i, filter in enumerate(['none', 'low', 'high']):
        print(f"Processing filter {filter}")

        if filter == 'low':
            ephys = lowpass_ephys(ephys, 16, 5)
        elif filter == 'high':
            ephys = highpass_ephys(ephys, 12, 5)

        act, loc = sniff_lock_lfp(inh, ephys[:5], nsniffs = 'all', method = 'zscore', window_size = 2_000)
        sort, freqs = sort_lfp(act, loc)

        neg_indexes = np.where(freqs < 2)
        sort = np.delete(sort, neg_indexes, axis = 1)
        freqs = np.delete(freqs, neg_indexes)


        # plotting heatmap
        if i == 2:
            sns.heatmap(sort[channel,:,:], cmap = 'seismic', robust = True, ax = axs[i], cbar = True)
        else:
            sns.heatmap(sort[channel,:,:], cmap = 'seismic', robust = True, ax = axs[i], cbar = False)

        # plotting a vertical line in middle of each figure
        axs[i].axvline(sort.shape[2] // 2, color = 'darkgrey', linestyle = '--', linewidth = 4, alpha = 0.8)
        

        x_middle = sort.shape[2] // 2
        window_size = sort.shape[2]


        if i == 0:
            axs[i].set_ylabel('Instantaneous Sniff Frequency (Hz)', fontsize = 20)
        yticks = np.linspace(0, sort.shape[1] - 1, num = 5, dtype = int)
        yticks_labels = [f'{freqs[i]:.1f}' for i in yticks]
        axs[i].set_yticks(yticks)
        axs[i].set_yticklabels(yticks_labels, fontsize = 18)
        axs[i].set_xticks([x_middle - window_size/2, x_middle, x_middle + window_size/2])
        axs[i].set_xticklabels([-window_size/2, '0', window_size/2], fontsize = 18)

    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, 'sniff_raster_inhales.png'), dpi = 300)
        
def build_time_lags_panel():

    sns.set_style('white')

    channel = 1
    mice = ['4122', '4127', '4131', '4138']
    save_path = r"E:\Sid_LFP\figs\Poster"
    data_path = r"E:\Sid_LFP\Sid_data\rnp_final"

    df = pd.DataFrame(columns = ['mouse', 'session', 'ephys_freqs_p2p', 'ephys_freqs_p2t', 'freqs', 'heights', 'dips', 'mid_peak', 'mid_trough'])

    for mouse in mice:
        mouse_path = os.path.join(data_path, mouse)
        sessions = os.listdir(mouse_path)
        for session in sessions:
            session_path = os.path.join(mouse_path, session)

            # checking to ensure file exists
            if not os.path.exists(os.path.join(session_path, 'LFP.npy')):
                print(f"Skipping mouse {mouse}, session {session} due to missing LFP file")
                continue
            
            if not os.path.exists(os.path.join(session_path, 'sniff_params.mat')):
                print(f"Skipping mouse {mouse}, session {session} due to missing sniff file")
                continue

            ephys_file = os.path.join(session_path, 'LFP.npy')
            sniff_file = os.path.join(session_path, 'sniff_params.mat')

            inh, _, exh, _ = load_sniff_MATLAB(sniff_file)
            ephys = np.load(ephys_file)

            print(f"\nProcessing mouse {mouse}, session {session}\n")

            if ephys.shape[1] < inh[-1]:
                print(f"Skipping mouse {mouse}, session {session} due to mismatched data")
                continue

            results_df = avg_lfp_analysis(ephys[channel:channel + 1], inh, save_path, channel = 0, window_size = 100, nbins = 20, freq_range = (2, 12), nshifts = 1000, show_peakfinder = False, plot_save = False)

            results_df['mouse'] = mouse
            results_df['session'] = session

            df = pd.concat([df, results_df], axis = 0)

    # saving results
    df.to_csv(os.path.join(save_path, 'time_lags_results.csv'), index = False)




def plot_timelags():

    # read in pandas dataframe
    df_path = r"E:\Sid_LFP\figs\Poster\time_lags_results.csv"
    save_path = r"E:\Sid_LFP\figs\Poster"
    df = pd.read_csv(df_path)

    # setting zero values to NaN
    df['ephys_freqs_p2t'] = df['ephys_freqs_p2t'].replace(0, np.nan)
    df['ephys_freqs_p2p'] = df['ephys_freqs_p2p'].replace(0, np.nan)
    df['dips'] = df['dips'].replace(0, np.nan)
    df['heights'] = df['heights'].replace(0, np.nan)
    
    # scaling values 
    df['ephys_freqs_p2t'] = df['ephys_freqs_p2t'] / 2

    clean_df = df.dropna()


    sns.set_style('white')
    
    # plotting ephys peak to peak frequencies and peak to trough frequencies
    fig, ax = plt.subplots(2, 2, figsize = (10, 6), sharex = True)

    sns.despine()
    sns.scatterplot(data = df, x = 'freqs', y = 'ephys_freqs_p2t', hue = 'mouse', palette = 'viridis', s = 10, ax = ax[0,0])
    sns.scatterplot(data = df, x = 'freqs', y = 'ephys_freqs_p2p', hue = 'mouse', palette = 'viridis', s = 10, ax = ax[0,1])
    sns.scatterplot(data = df, x = 'freqs', y = 'dips', hue = 'mouse', palette = 'viridis', s = 10, ax = ax[1,0])
    sns.scatterplot(data = df, x = 'freqs', y = 'heights', hue = 'mouse', palette = 'viridis', s = 10, ax = ax[1,1])



    #fitting linear regressions
    x = clean_df['freqs']
    y = clean_df['ephys_freqs_p2t']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x = np.linspace(4, 12, 100)
    print(slope, intercept)
    y = slope * x + intercept
    sns.lineplot(x = x, y = y, ax = ax[0,0], color = 'dodgerblue')
    legend_elements = [Line2D([0], [0], marker = None, color='w', lw=0, label=f'y = {slope:.2f}x + {intercept:.2f}'),
                       Line2D([0], [0], marker=None, color='w', markersize=0, label=f' r = {r_value:.2f}, p < {np.ceil(100 * p_value) / 100}')]
    ax[0,0].legend(handles=legend_elements, fontsize = 12, frameon = False)


    x = clean_df['freqs']
    y = clean_df['ephys_freqs_p2p']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x = np.linspace(2, 12, 100)
    y = slope * x + intercept
    sns.lineplot(x = x, y = y, ax = ax[0,1], color = 'dodgerblue')
    legend_elements = [Line2D([0], [0], marker = None, color='w', lw=0, label=f'y = {slope:.2f}x + {intercept:.2f}'),
                       Line2D([0], [0], marker=None, color='w', markersize=0, label=f' r = {r_value:.2f}, p < {np.ceil(100 * p_value) / 100}')]
    ax[0,1].legend(handles=legend_elements, fontsize = 12, frameon = False)




    x = clean_df['freqs']
    y = clean_df['dips']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x = np.linspace(2, 12, 100)
    y = slope * x + intercept
    sns.lineplot(x = x, y = y, ax = ax[1,0], color = 'dodgerblue')
    legend_elements = [Line2D([0], [0], marker = None, color='w', lw=0, label=f'y = {slope:.2f}x + {intercept:.2f}'),
                       Line2D([0], [0], marker=None, color='w', markersize=0, label=f' r = {r_value:.2f}, p < {np.ceil(100 * p_value) / 100}')]
    ax[1,0].legend(handles=legend_elements, fontsize = 12, frameon = False)


    x = clean_df['freqs']
    y = clean_df['heights']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x = np.linspace(2, 12, 100)
    y = slope * x + intercept
    sns.lineplot(x = x, y = y, ax = ax[1,1], color = 'dodgerblue')
    legend_elements = [Line2D([0], [0], marker = None, color='w', lw=0, label=f'y = {slope:.2f}x + {intercept:.2f}'),
                       Line2D([0], [0], marker=None, color='w', markersize=0, label=f' r = {r_value:.2f}, p < {np.ceil(100 * p_value) / 100}')]
    ax[1,1].legend(handles=legend_elements, fontsize = 12, frameon = False)


 


    ax[0,0].set_ylabel('Peak to Trough Time Lag (Hz)')
    ax[0,1].set_ylabel('Peak to Peak Time Lag (Hz)')
    ax[1,0].set_ylabel('Trough Dip Amplitude (|mV|)')
    ax[1,1].set_ylabel('Peak Height (|mV|)')

    ax[1,1].set_xlabel('Instantaneous Sniff Frequency (Hz)')
    ax[1,0].set_xlabel('Instantaneous Sniff Frequency (Hz)')


    plt.savefig(os.path.join(save_path, 'time_lags_panel.png'), dpi = 300)
    

def extract_HMM_states():

    data_path = r"E:\Sid_LFP\Sid_data\neurohmm1\4127_6.csv"

    hmm_states = np.genfromtxt(data_path, delimiter = ',')




    return




               


                    







if __name__ == '__main__':
    plot_timelags()


    
   
