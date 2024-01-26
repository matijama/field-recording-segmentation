import numpy as np

from utils import get_peaks, get_peaks_v2, normpdfF


def seg_ek_probability_2018(ec, kl, class_table, params, stat_step_in_sec):
    # For caching pdfs (speed)
    pdf_dist = np.full((30000, class_table.shape[1]), -1.0)

    # Segment lengths in seconds according to content type (gaussian means and stds)
    # Solo, choir, instrumental, speech
    seg_length_distribution = np.array([[65.9401, 107.2199, 101.6937, 140.3333],
                                       [52.2381, 70.4295, 99.0742, 234.2829]]) 
    seg_length_distribution /= stat_step_in_sec

    if len(params) < 4:
        min_length = 10 / stat_step_in_sec  # Segments shorter than this are not allowed
        b_len = 1
    else:
        min_length = params[2]
        b_len = params[4]

    partial_sum = np.zeros_like(class_table)
    partial_sum[0, :] = np.sum(class_table[:80, :], axis=0)

    for i in range(1, class_table.shape[0] - 80 + 1):
        partial_sum[i, :] = partial_sum[i - 1, :] - class_table[i - 1, :] + class_table[i + 80 - 1, :]

    # Boundary candidates
    candidates = np.union1d(np.union1d(np.ravel(get_peaks(ec, 0)).nonzero(), np.ravel(get_peaks(kl, 0)).nonzero()), np.ravel(get_peaks(ec + kl, 0)).nonzero())
    if len(candidates) == 0:
        return np.array([1, len(ec)]), np.array([])

    ec = np.atleast_2d(ec.flatten()).T
    kl = np.atleast_2d(kl.flatten()).T
    ekl_t = np.hstack((ec[candidates], kl[candidates]))
    
    if candidates[0] > 0:
        candidates = np.hstack((0, candidates))
        ekl_t = np.vstack(([0.5, 0.5], ekl_t))
    
    ekl_t_max = np.max(ekl_t, axis=1)

    # limit prob to 0.01 & 1
    ekl_t_max = np.clip(ekl_t_max, 0.01, 1 - np.finfo(float).eps)
    
    c_val_pos = -np.log(ekl_t_max)
    c_val_neg = -np.log(1 - ekl_t_max)

    tbl = np.zeros((len(c_val_pos), len(c_val_pos)))
    bp = np.zeros(len(candidates))-1

    tbl[0, 0] = c_val_pos[0]

    # Do DP 
    for i in range(1, len(tbl)):
        for j in range(i - 1):  # (until i-1, all have no boundary at i-1);
            tbl[i, j] = tbl[i - 1, j] + c_val_neg[i - 1]

        tbl[i, i] = np.inf

        for j in range(i):
            length = candidates[i] - candidates[j]

            if length > 0 and pdf_dist[length, 0] == -1:
                pdf = normpdfF(length * np.ones((1, class_table.shape[1])),
                                               seg_length_distribution[0, :], seg_length_distribution[1, :], 1)
                pdf_dist[length, :] = pdf

            p_len = get_seg_len_probability(i, j, candidates, class_table, partial_sum, min_length, pdf_dist[length, :])
            t = tbl[i, j] + tbl[j, j] + c_val_pos[i] + b_len * p_len
            if t < tbl[i, i]:
                tbl[i, i] = t
                bp[i] = j

    # Decode best path
    if len(bp[bp > -1]) != 0:
        best_path = np.array([int(bp[bp > -1][-1]), len(candidates) - 1])
        while best_path[0] != -1:
            best_path = np.hstack((int(bp[best_path[0]]), best_path))

        segs = candidates[best_path[best_path > -1]]
    else:
        segs = np.array([0, len(ec)])

    return segs #, ekl_t


def get_seg_len_probability(to, from_, candidates, class_table, partial_sum, min_length, pdf_dist):
    # Returns probability of segment length given parameters

    en = candidates[to]
    st = candidates[from_]
    length = en - st

    if length <= min_length:
        p_len = np.inf
    else:
        if length > 80:
            t = np.arange(st, en, 80)
            prof = np.sum(partial_sum[t[:-1], :], axis=0)
            prof += np.sum(class_table[t[-1]:en, :], axis=0)
            prof /= (length + 1)
        else:
            prof = np.mean(class_table[st:en, :], axis=0)

        prof /= np.sum(prof)
        p_len = -np.log(np.sum(pdf_dist * prof))

    return p_len
