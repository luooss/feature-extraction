import os
import numpy as np
import matplotlib.pyplot as plt
import mne

cnt_file = r"D:\bcmi\exp\eeg_cnt_file\leixiaoting_20210324.cnt"

EOG_channels = ['VEO', 'HEO']
unused_channels = ['M1', 'M2']

raw = mne.io.read_raw_cnt(cnt_file, eog=EOG_channels, preload=True)
raw.info['bads'].extend(EOG_channels)
raw.info['bads'].extend(unused_channels)

