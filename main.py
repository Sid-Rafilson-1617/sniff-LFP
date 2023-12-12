'''
Main run file for local field potential project

By: Sid Rafilson
'''

from snifflocklfp import *

sniff_file = r"\\F-moving-data\shnk3 (a)\080321_4131_session3_ADC.bin"
ephys_file = r"\\F-moving-data\shnk3 (a)\080321_4131_session3_Ephys.bin"

def main():
    print('loading data...')
    adcx = load_sniff(sniff_file, 1200000000)
    sniff = get_sniff(adcx)
    ephys = load_ephys(ephys_file, 1200000000)
    ephys = reshape_ephys(ephys)
    ephys = remove_jumps_ephys(ephys)
    print('resampling...')
    sniff = resample_sniff(sniff)
    ephys = resample_ephys(ephys)
    print('finding peaks...')
    inhales, smoothed_sniff = find_inhales(sniff)
    print('aligning...')
    print(inhales.shape)
    sniff_activity = sniff_lock_lfp(inhales, ephys, beg = 4000)
    sorted_lfp = sort_lfp(sniff_activity)
    plot_snifflocked_lfp(sorted_lfp, 16, 1000)
    

main()