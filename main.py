'''
Main run file for local field potential project

By: Sid Rafilson
Primary Investigator: Matt Smear
'''

from analysis import *
import numpy as np



# file paths

sniff_file = r"E:\Sid_LFP\data\052921_4127_session4_ADC_int16_med0_nch8.bin"
ephys_file = r"E:\Sid_LFP\data\052921_4127_session4_Ephys_int16_med0_nch16.bin"
tracking_file = r"\\F-MOVING-DATA\EphysData\rhythmReTrackISH\keypoint_processing\KeypointMatrix\4127\4_keypoints.csv"


if __name__ == '__main__':

    ephys, sniff = preprocess(
        sniff_file, ephys_file, tracking_file,
          num_samples = -1, start = 0, channels = (1,2),
            stop = 0, nchannels = 16, remove_artifacts = True, resample = True, no_load = 'tracking')
