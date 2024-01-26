import os
import pandas as pd
import numpy as np
import configparser
import librosa
from pathlib import Path
from scipy.signal import medfilt, convolve, butter, filtfilt
from scipy.signal.windows import gaussian
from scipy.interpolate import interp1d

from calcEnergy2018 import calc_energy_2018
from segEKProbability2018 import seg_ek_probability_2018
from utils import s1


def main():
    
    config = configparser.ConfigParser()
    config.read('defaults.ini')

    working_dir = Path(os.path.join(config['Folders']['working_dir']))

    separator = config['Format']['sep']
    score_tag = config['Format']['score']
    segmentation_tag = config['Format']['segmentation']
    
    labels = dict({ 
        0 : config['Labels']['solo'],
        1 : config['Labels']['choir'],
        2 : config['Labels']['instr'],
        3 : config['Labels']['speech']
        })

    mp3_files = working_dir.rglob('*.mp3')

    for mp3_file in mp3_files:
        
        score_file = mp3_file.with_name(f'{mp3_file.name}.{score_tag}.csv')
        segmentation_file = mp3_file.with_name(f'{mp3_file.name}.{segmentation_tag}.csv')
        if not score_file.is_file() or segmentation_file.is_file():
            continue

        print(str(mp3_file))

        seg_classes = pd.read_csv(score_file, sep=separator, header=None, engine='python').values
        seg_times, seg_labels = segment_recording_deep(mp3_file, seg_classes)

        max_labels = seg_labels.argmax(axis=1)
        seg_times_df = pd.DataFrame({
            'Start Time': seg_times[:-1],
            'End Time': seg_times[1:],
            'Label': [labels[label] for label in max_labels]
        })

        seg_times_df.to_csv(segmentation_file, sep=separator, index=False)


def segment_recording_deep(filename, seg_classes):
    """
    segment_recording_deep Segment a recording and returns segment times (seconds) and labels.

    Parameters:
    - filename: string, audio file to segment.
    - seg_classes: 2D array, probabilities of classification of the file contents with 1/samplerate of statStepInSec.

    Returns:
    - seg_times: segment times.
    - seg_labels: segment labels.
    - seg_classes: probabilities of classification of the file contents with 1/samplerate of statStepInSec.
    - seg_class_sr: 1/statStepInSec
    """
    
    sample_rate = 22050
    energy_win_in_sec = 1024 / sample_rate
    energy_median_in_sec = 1
    stat_step_in_sec = 0.5
    stat_win_in_sec = 2
    kl_win_size_in_sec = 5
    kl_gauss_win_in_sec = 5
    kl_peak_thresh = 4

    # read file
    w, sr = librosa.load(filename, sr=sample_rate) 
    w = s1(w)

    # calculate energy
    energy, et, feature_sr = calc_energy_2018(w, sr, energy_win_in_sec, stat_win_in_sec, stat_step_in_sec)

    seg_classes = np.vstack([seg_classes, np.tile(seg_classes[-1, :], (len(energy) - len(seg_classes), 1))])

    et_m = medfilt(et + 0.0, (int(energy_median_in_sec * feature_sr), 1))
    et_stat = interp1d(np.arange(len(et)) / feature_sr, et_m.T, kind='linear', fill_value='extrapolate')(np.arange(len(energy)) * stat_step_in_sec)[0, :]

    # calculate transitions with KL on seg_classes
    kl_c = np.zeros(len(seg_classes))
    kl_win_size = round(kl_win_size_in_sec / stat_step_in_sec)

    for st in range(round(kl_win_size / 2), len(seg_classes) - round(kl_win_size / 2)):
        rng = np.arange(max(st - kl_win_size * 2 + 1, 0), st + 1)
        et_stat_rng = et_stat[rng] <= 0.5
        X = seg_classes[rng, :]
        X = X[et_stat_rng, :]
        Xe = np.sqrt(energy[rng, :])
        Xe = Xe[et_stat_rng]

        rng = np.arange(st + 1, min(len(seg_classes), st + kl_win_size * 2))
        et_stat_rng = et_stat[rng] <= 0.5
        Y = seg_classes[rng, :]
        Y = Y[et_stat_rng, :]
        Ye = np.sqrt(energy[rng, :])
        Ye = Ye[et_stat_rng]

        X = X[max(0, X.shape[0] - kl_win_size):, :]
        Xe = Xe[max(0, len(Xe) - kl_win_size):]
        Y = Y[:min(Y.shape[0], kl_win_size), :]
        Ye = Ye[:min(len(Ye), kl_win_size)]

        h1 = np.maximum(np.sum(X * Xe, axis=0) / (np.sum(Xe) + np.finfo(float).eps), 1e-3)
        h2 = np.maximum(np.sum(Y * Ye, axis=0) / (np.sum(Ye) + np.finfo(float).eps), 1e-3)
        kl = np.sum(h1 * np.log2(h1) - h1 * np.log2(h2)) + np.sum(h2 * np.log2(h2) - h2 * np.log2(h1))

        kl_c[st] = kl

    kl_gauss_win = int(kl_gauss_win_in_sec / stat_step_in_sec)
    kl_gauss_win += (kl_gauss_win % 2 == 0)
    t = gaussian(kl_gauss_win, ((kl_gauss_win - 1) / 5))
    t /= np.sum(t)
    kl_cg = convolve(kl_c, t, mode='same')

    # calculate silence and transition likelihoods
    b, a = butter(2, 0.1 * stat_step_in_sec * 2)

    # weigh energy according to class
    et_shift = np.atleast_2d(np.array([1.2, 1, 0.9, 1.5]).dot(seg_classes.T))
    energy_curve = np.clip(filtfilt(b, a, et_stat * et_shift + (1 - et_shift)), np.finfo(float).eps, 1)
    kl_curve = np.clip(filtfilt(b, a, (np.atleast_2d(kl_cg) > kl_peak_thresh).astype(float)), np.finfo(float).eps, 1)

    # do the probabilistic segmentation
    p2 = [0, 0, 10 / stat_step_in_sec, 1, 1]
    loc = seg_ek_probability_2018(energy_curve, kl_curve, seg_classes, p2, stat_step_in_sec)

    # calculate segment labels
    seg_times = loc
    seg_labels = np.zeros((len(seg_times) - 1, seg_classes.shape[1]))
    for i in range(len(seg_times) - 1):
        seg_labels[i, :] = np.sum(np.multiply(seg_classes[seg_times[i]:seg_times[i + 1], :].T, 
                                              np.sqrt(energy[seg_times[i]:seg_times[i + 1]])[0]
                                              * np.hamming(seg_times[i + 1] - seg_times[i])).T, 0)
        seg_labels[i, :] /= (np.sum(seg_labels[i, :]) + np.finfo(float).eps)

    seg_times = stat_step_in_sec * seg_times
    seg_times[1:-1] = seg_times[1:-1] + stat_win_in_sec / 2

    return seg_times, seg_labels


if __name__ == '__main__':
    main()
