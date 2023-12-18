'''
Main run file for local field potential project

By: Sid Rafilson
'''

from snifflocklfp import *

sniff_file = r"\\F-MOVING-DATA\EphysData\DATA_MATT_DRIVE\fromData_Restrx_Ob_Final\4122\5\121620_4122_hf2fm_ADC_int16_med0_nch8.bin"
ephys_file = r"\\F-MOVING-DATA\EphysData\DATA_MATT_DRIVE\fromData_Restrx_Ob_Final\4122\5\121620_4122_hf2fm_Ephys_int16_med0_nch16.bin"

def main():

    print('loading data...')

    sniff = load_sniff(sniff_file, 18000000)
    ephys = load_ephys(ephys_file, 18000000, nchannels= 16)

    print('converting data type...')
    sniff = sniff.astype(np.int32)
    ephys = ephys.astype(np.int32)

    print('removing artifact...')
    ephys = remove_jumps_ephys(ephys)
    sniff = remove_jumps_sniff(sniff)

    print('resampling...')
    sniff = resample_sniff(sniff)
    ephys = resample_ephys(ephys, nchannels=16)

    plot_ephys(ephys, nchannels=16)
    peak_finder_validation(sniff)

    print('finding peaks...')
    inhales, _ = find_inhales(sniff)

    print(f'number of inhales in loaded data = {inhales.shape}')

    print('aligning...')
    sniff_activity, locs = sniff_lock_lfp(inhales, ephys, beg =20000, nsniffs = 200, nchannels=16)
    sorted_activity =  sort_lfp(sniff_activity, locs)

    plot_snifflocked_lfp(sniff_activity, nchannels=16)
    plot_snifflocked_lfp(sorted_activity, nchannels=16)
   
main()
