'''
Main run file for local field potential project

By: Sid Rafilson
'''

from snifflocklfp import *

sniff_file = r"\\F-moving-data\shnk3 (a)\080321_4131_session3_ADC.bin"
ephys_file = r"\\F-moving-data\shnk3 (a)\080321_4131_session3_Ephys.bin"

def main():

    print('loading data...')
    adcx = load_sniff(sniff_file, 600000000)
    sniff = get_sniff(adcx)
    ephys = load_ephys(ephys_file, 600000000)
    ephys = reshape_ephys(ephys)

    print('resampling...')
    sniff = resample_sniff(sniff)
    ephys = resample_ephys(ephys)

    print('finding peaks...')
    inhales, smoothed_sniff = find_inhales(sniff)

    print('aligning...')
    sniff_activity = sniff_lock_lfp(inhales, ephys, beg = 3000)
    sorted_lfp = sort_lfp(sniff_activity, inhales)
    plot_snifflocked_lfp(sorted_lfp, 16, 1000)
    avg_lfp = avg_sniff_locked_lfp(sorted_lfp)
    plot_avg_lfp(avg_lfp)
    lfp = sniff_lock_lfp_infreq(inhales, ephys, [6, 8])
    plot_snifflocked_lfp(lfp)
    

main()
