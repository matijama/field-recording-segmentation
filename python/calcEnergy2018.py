import numpy as np
from scipy.signal import medfilt

from utils import rank_order_filter


def calc_energy_2018(w, sr, win_size_in_sec=0.05, stat_win_size_in_sec=2, stat_step_in_sec=0.5):
    thresh = 10**(-6)  # -60 dB

    win_size = 2 ** int(np.floor(np.log2(win_size_in_sec * sr)))
    step = win_size // 2
    feature_sr = sr / step
    stat_win_size = round(stat_win_size_in_sec / (step / sr))

    # RMS energy
    energy = np.zeros((int(np.ceil(len(w) / step)), 1))
    j = 0
    issubdtype_int = np.issubdtype(w.dtype, np.integer)
    for i in range(0, len(w) - win_size, step):
        t = w[i:i + win_size]
        if issubdtype_int:
            t /= 32768.0
        energy[j] = np.sum(t ** 2) / win_size
        j += 1
    energy = energy[:j]

    # Global noise floor
    noise_floor = rank_order_filter(energy, round(20 * feature_sr), 1)**0.9
    kernel_size = round(feature_sr)
    kernel_size += kernel_size % 2 - 1
    energy_f = medfilt(energy, (kernel_size, 1))

    # Local noise floor
    kernel_size = round(12 * feature_sr)
    kernel_size += kernel_size % 2 - 1
    e_l = medfilt(energy_f, (kernel_size, 1))
    energy_flt = 10**(-3) * e_l**0.5

    # Reset global noise floor if it's too high
    nf_reset = (noise_floor > energy_flt) & (noise_floor > 1e-4) & (noise_floor < 0.3 * e_l**0.9)
    noise_floor[nf_reset] = energy_flt[nf_reset]
    noise_floor[noise_floor < thresh] = thresh
    
    # when are we under noise floor
    et = (energy_f < noise_floor) | (energy_f < energy_flt)

    # Average the output over a window
    energy_out = np.zeros((int (np.ceil(step / sr * len(energy) / stat_step_in_sec) + 1), 1))
    j = 0
    idx = 0
    while idx < len(energy):
        rng = slice(idx, min(idx + int(stat_win_size), len(energy)))
        t = energy[rng]
        energy_out[j] = np.mean(t)
        idx = round(j * stat_step_in_sec * sr / step + 1)
        j += 1
    energy_out = energy_out[:j]

    return energy_out, et, feature_sr
