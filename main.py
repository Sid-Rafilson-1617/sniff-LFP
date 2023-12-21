'''
Main run file for local field potential project

By: Sid Rafilson
'''

from snifflocklfp import *

sniff_file = r"\\F-MOVING-DATA\EphysData\DATA_MATT_DRIVE\fromData_Restrx_Ob_Final\4122\5\121620_4122_hf2fm_ADC_int16_med0_nch8.bin"
ephys_file = r"\\F-MOVING-DATA\EphysData\DATA_MATT_DRIVE\fromData_Restrx_Ob_Final\4122\5\121620_4122_hf2fm_Ephys_int16_med0_nch16.bin"

def main():

    print('loading data...')
    sniff = load_sniff(sniff_file, 30000000)
    ephys = load_ephys(ephys_file, 30000000, nchannels= 16)

    print('converting data type...')
    sniff = sniff.astype(np.int32)
    ephys = ephys.astype(np.int32)

    print('removing artifact...')
    ephys = remove_jumps_ephys(ephys)
    sniff = remove_jumps_sniff(sniff)

    print('resampling...')
    sniff = resample_sniff(sniff)
    ephys = resample_ephys(ephys, nchannels=16)

    print('finding peaks...')
    inhales, _ = find_inhales(sniff)

    print('aligning/sorting...')
    beg = 20000
    nsniffs = 200
    window_size = 1000
    sniff_activity, locs_set = sniff_lock_lfp(inhales, ephys, window_size=window_size, beg = beg, nsniffs = nsniffs)
    sorted_activity =  sort_lfp(sniff_activity, locs_set)

    print('creating null distributions...')
    shifts = 100
    circular_ephys = circular_shift(ephys, shifts)
    null = create_circular_null(circular_ephys, inhales, nsniffs = nsniffs, window_size = window_size, beg = beg)

    print('finding z-scores from null')
    sniff_activity_shift, locs_set_1 = sniff_lock_std(inhales, null, window_size=window_size, beg = beg, nsniffs = nsniffs)
    sorted_activity_shift = sort_lfp(sniff_activity_shift, locs_set_1)


    print('plotting...')
    plot_snifflocked_lfp(sniff_activity)
    plot_snifflocked_lfp(sorted_activity, show_y = False)
    plot_snifflocked_lfp(sniff_activity_shift)
    plot_snifflocked_lfp(sorted_activity_shift)
    

    
    



main()
