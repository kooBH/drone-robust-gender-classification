import os
import librosa

import numpy as np
import torch

def stft(wav):
    window = torch.hann_window(window_length=512, periodic=True, dtype=None, layout=torch.strided, device=None,
                               requires_grad=False)
    data_wav = torch.from_numpy(wav)
    
    spec_noi1 = torch.stft(data_wav, window=window, n_fft=512, hop_length=128,
                                                  win_length=512)
    input_wav_real1 = spec_noi1[:, :, :, 0]
    input_wav_imag1 = spec_noi1[:, :, :, 1]
    phase = torch.atan(input_wav_imag1 / (input_wav_real1 + 1e-8))
    input_wav_magnitude = torch.sqrt(input_wav_real1 ** 2 + input_wav_imag1 ** 2)
    return input_wav_magnitude, phase


def vad(total_audio):
    data = {}

    frame_size = 512
    hop_size = 128
    #RISING_TERM = 30
    #LEAST_TERM = 100
    RISING_TERM = 60
    LEAST_TERM = 60
    power_list = []
    save_wav_count = 0
    eps = 1e-5
    
    time_list = []

    power_list = []
    active_cnt = 0
    tmp_dummy_frame = 0
    dummy_frame = 0
    inactive_cnt = 0
    state = 0
    i = 0
    num = 0
    fs = 16000
    #thr = 0.005
    thr = 0.001

    frame_idx_list = range(0, total_audio.shape[1] - hop_size + 1, hop_size)
    input_wav_mag, phase = stft(total_audio)

    #mean_power = abs(input_wav_mag[:, 64:256, :]).mean()
    #thre = mean_power / 2

    for frame_idx in frame_idx_list:
        num += 1

        if abs(input_wav_mag[:, 30:200, frame_idx // hop_size]).mean() > thr:
            if state == 0:
                active_cnt = 1
                tmp_dummy_frame = 1
                rising_idx = frame_idx
                state = 1

            elif state == 1:
                active_cnt += 1
                tmp_dummy_frame += 1
                if active_cnt == RISING_TERM:
                    state = 2

            elif state == 2:
                active_cnt += 1

            elif state == 3:
                inactive_cnt = 0
                active_cnt += 1
                state = 2

            elif state == 4:
                active_cnt = 1
                tmp_dummy_frame = 1
                rising_idx = frame_idx
                state = 1
        else:
            if state == 0:
                dummy_frame += 1
                state = 0

            elif state == 1:
                active_cnt = 0
                dummy_frame += tmp_dummy_frame
                tmp_dummy_frame = 0
                state = 0

            elif state == 2:
                inactive_cnt = 1
                active_cnt += 1
                state = 3

            elif state == 3:
                inactive_cnt += 1
                active_cnt += 1
                if inactive_cnt == LEAST_TERM:
                    state = 4

            elif state == 4:
                dummy_frame = 1
                state = 0

        # save VAD chunk here in wav
        if state == 4 or (num == len(frame_idx_list) and active_cnt > RISING_TERM):
            falling_idx = frame_idx
            if rising_idx - hop_size < 0:
                rising_idx = 128
            rising_idx = (rising_idx - hop_size)
            if state == 4:
                falling_idx = (falling_idx - (LEAST_TERM - 2) * hop_size)
            else:
                falling_idx = (falling_idx - (inactive_cnt - 2) * hop_size)

            rising_idx = rising_idx / fs
            falling_idx = falling_idx / fs
            time_list.append([rising_idx, falling_idx])
            save_wav_count += 1

            # save chunk for another channel
            i += 1
            state = 4
            active_cnt = 0
            inactive_cnt = 0
            tmp_dummy_frame = 0
            dummy_frame = 0

    return time_list
