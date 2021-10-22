import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import csv
import math
import scipy.io as sio
from scipy.fftpack import fft,ifft
import simdkalman
from tqdm import tqdm

uiow = ['chenyi_20210322.cnt', 'huangwenjing_20210324.cnt', 'huangxingbao_20210322.cnt', 'huatong_20210323.cnt', 'wuwenrui_20210325.cnt', 'yinhao_20210323.cnt']
uoiw = ['chenyi.csv', 'huangwenjing.csv', 'huangxingbao.csv', 'huatong.csv', 'wuwenrui.csv', 'yinhao.csv']

EOG_channels = ['VEO', 'HEO']
unused_channels = ['M1', 'M2']

bands = ['all', 'delta', 'theta', 'alpha', 'beta', 'gamma']
lows = [1, 1, 4, 8, 14, 31]
highs = [75, 4, 8, 14, 31, 50]

def DE_PSD(data,stft_para):
    '''
    compute DE and PSD
    --------
    input:  data [n*m]          n electrodes, m time points
            stft_para.stftn     frequency domain sampling rate
            stft_para.fStart    start frequency of each frequency band
            stft_para.fEnd      end frequency of each frequency band
            stft_para.window    window length of each sample point(seconds)
            stft_para.fs        original frequency
    output: psd,DE [n*l*k]        n electrodes, l windows, k frequency bands
    '''
    #initialize the parameters
    STFTN=stft_para['stftn']
    fStart=stft_para['fStart']
    fEnd=stft_para['fEnd']
    fs=stft_para['fs']
    window=stft_para['window']

    WindowPoints=fs*window

    fStartNum=np.zeros([len(fStart)],dtype=int)
    fEndNum=np.zeros([len(fEnd)],dtype=int)
    for i in range(0,len(stft_para['fStart'])):
        fStartNum[i]=int(fStart[i]/fs*STFTN)
        fEndNum[i]=int(fEnd[i]/fs*STFTN)

    #print(fStartNum[0],fEndNum[0])
    n=data.shape[0]
    m=data.shape[1]

    l = math.floor(m / WindowPoints)

    psd = np.zeros([n, l, len(fStart)])
    de = np.zeros([n, l, len(fStart)])
    #Hanning window
    Hwindow= np.hanning(WindowPoints)

    for j in range(n):
        for i in range(l):
            temp = data[j, WindowPoints*i : WindowPoints*(i+1)]
            Hdata = temp * Hwindow
            FFTdata = fft(Hdata, STFTN)
            magFFTdata = abs(FFTdata[0:int(STFTN/2)])
            for p in range(len(fStart)):
                E = 0
                for p0 in range(fStartNum[p], fEndNum[p]+1):
                    E = E + magFFTdata[p0] * magFFTdata[p0]
                
                E = E/(fEndNum[p]-fStartNum[p]+1)
                psd[j][i][p] = E
                de[j][i][p] = math.log(100*E,2)
    
    return psd,de


for u in range(6):
    cnt_file = r"D:\bcmi\exp\eeg_cnt_file" + '\\' + uiow[u]

    subj_name = cnt_file[cnt_file.rindex('\\')+1 : cnt_file.rindex('_')]
    print(subj_name)

    raw = mne.io.read_raw_cnt(cnt_file, eog=EOG_channels)
    raw.info['bads'].extend(EOG_channels)
    raw.info['bads'].extend(unused_channels)

    raw.load_data()
    # Downsample the data and events at the same time
    raw = raw.filter(l_freq=1, h_freq=75).resample(200)

    events, event_id = mne.events_from_annotations(raw)
    # only the start trigger is needed
    choice = np.ones(180)
    for i in range(1, 180, 2):
        choice[i] = 0

    choice = (choice == 1)
    events = events[choice]

    useful_channels = raw.ch_names[:]
    for ch in unused_channels:
        useful_channels.remove(ch)

    for ch in EOG_channels:
        useful_channels.remove(ch)
    
    epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=20, picks=useful_channels)

    # process labels
    labels = {}
    psyfile = r"D:\bcmi\exp\psychopy_export" + '\\' + uoiw[u]

    with open(psyfile, 'r', newline='') as psyf:
        reader = csv.DictReader(psyf)
        for row in reader:
            img_name = row['imageName']
            img_no = int(img_name[:img_name.rindex('.')])
            if row[' category'] == '负向':
                emotion_label = 0
            elif row[' category'] == '中性':
                emotion_label = 1
            elif row[' category'] == '正向':
                emotion_label = 2
            else:
                print('error')
            
            labels[img_no] = emotion_label

    epochs.load_data()
    # 6 * (1, 360, 62, 1000)
    datas = []
    for b in range(6):
        # 90 * (4, 62, 1000)
        for_this_band = []
        filtered_epochs = epochs.copy().filter(l_freq=lows[b], h_freq=highs[b])
        for img in range(1, 91):
            # 4 * (1, 62, 1000)
            slices = []
            # (1, 62, 4000)
            img_data = filtered_epochs[str(img)].get_data()[:, :, :4000]
            for s in range(4):
                slices.append(img_data[:, :, 1000*s : 1000*(s+1)])
            
            for_this_band.append(np.concatenate(slices, axis=0))
        
        datas.append(np.expand_dims(np.concatenate(for_this_band, axis=0), axis=0))

    subj_data = np.concatenate(datas, axis=0)

    # 360
    labs = []
    for img in range(1, 91):
        for s in range(4):
            labs.append(labels[img])

    subj_label = np.array(labs)

    print(subj_data.shape, subj_label.shape)
    print(subj_data.dtype, subj_label.dtype)

    out_dir = './npydata_withoutproc'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.save(out_dir+'/{}_data.npy'.format(subj_name), subj_data)
    np.save(out_dir+'/{}_label.npy'.format(subj_name), subj_label)

    print('feature, smooth...')

    kf = simdkalman.KalmanFilter(
        state_transition = np.array([[1,1],[0,1]]),
        process_noise = np.diag([0.1, 0.01]),
        observation_model = np.array([[1,0]]),
        observation_noise = 1.0)

    data_all = subj_data

    # 6 * (1, 360, 62, 5, 5)
    data_psd = []
    data_de = []
    for b in tqdm(range(6)):
        data = data_all[b]
        # print(data.shape)
        stft_para = {'stftn':512, 'fStart':[1, 4, 8, 14, 31], 'fEnd':[4, 8, 14, 31, 50], 'fs':200, 'window':1}
        # 360 * (1, 62, 5, 5)
        data_psd_b = []
        data_de_b = []
        for case in tqdm(range(360)):
            psd, de = DE_PSD(data[case], stft_para)
            for c in range(62):
                psd[c] = psd[c].T
                de[c] = de[c].T
                for fb in range(5):
                    psd[c, fb] = kf.smooth(psd[c, fb]).observations.mean
                    de[c, fb] = kf.smooth(de[c, fb]).observations.mean
                
            data_psd_b.append(np.expand_dims(psd, axis=0))
            data_de_b.append(np.expand_dims(de, axis=0))

        data_psd_b = np.concatenate(data_psd_b, axis=0)
        data_de_b = np.concatenate(data_de_b, axis=0)

        data_psd.append(np.expand_dims(data_psd_b, axis=0))
        data_de.append(np.expand_dims(data_de_b, axis=0))

    data_psd = np.concatenate(data_psd, axis=0)
    data_de = np.concatenate(data_de, axis=0)

    print(data_psd.shape)
    print(data_de.shape)

    np.save(out_dir+'/{}_data_psd.npy'.format(subj_name), data_psd)
    np.save(out_dir+'/{}_data_de.npy'.format(subj_name), data_de)
    
