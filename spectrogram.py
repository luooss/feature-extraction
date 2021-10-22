import os
import numpy as np
import matplotlib.pyplot as plt
import mne

uiow = ['chenyi_20210322.cnt', 'huangwenjing_20210324.cnt', 'huangxingbao_20210322.cnt', 'huatong_20210323.cnt', 'leixiaoting_20210324.cnt', 'wuwenrui_20210325.cnt', 'yinhao_20210323.cnt']
uoiw = ['chenyi.csv', 'huangwenjing.csv', 'huangxingbao.csv', 'huatong.csv', 'leixiaojiao.csv', 'wuwenrui.csv', 'yinhao.csv']

u = 3

cnt_file = r"D:\bcmi\exp\eeg_cnt_file" + '\\' + uiow[u]

subj_name = cnt_file[cnt_file.rindex('\\')+1 : cnt_file.rindex('_')]
print(subj_name)

EOG_channels = ['VEO', 'HEO']
unused_channels = ['M1', 'M2']

raw = mne.io.read_raw_cnt(cnt_file, eog=EOG_channels)
raw.info['bads'].extend(EOG_channels)
raw.info['bads'].extend(unused_channels)