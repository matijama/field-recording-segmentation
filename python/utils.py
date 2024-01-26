import numpy as np
from scipy.ndimage import percentile_filter


def rank_order_filter(x, N, p):
    """
    RankOrderFilter Rank order filter for 1D or 2D signals.

    Parameters:
    - x: 1D or 2D array, input signal.
    - N: int, window size for rank-order filtering.
    - p: float, percentile value between 0 and 100.

    Returns:
    - y: Rank-order filtered signal.
    """

    k = (N - 1) // 2

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    y = percentile_filter(x, p, size=(k, 1))

    return y


def s1(inp, rowcol=0):
    """
    res = s1(inp, rowcol) scales the input vector so that its maximum positive value is one.
    If inp is a matrix, then rowcol==0 scales the entire matrix to one, rowcol==1 scales
    each row to one, and rowcol==2 scales each column to one. Default rowcol=0.

    Parameters:
    - inp: 1D or 2D array, input vector or matrix.
    - rowcol: int, specifies how to scale the matrix (0 for entire matrix, 1 for each row, 2 for each column).
              Default is 0.

    Returns:
    - res: Scaled array.
    """

    if rowcol == 0:
        res = inp / np.max(inp)
    elif rowcol == 1:
        m = np.max(inp, axis=1)
        res = inp / m[:, np.newaxis]
    elif rowcol == 2:
        m = np.max(inp, axis=0)
        res = inp / m

    return res

def normpdfF(x, MU, SIGMA, noscale=0):
    """
    y = normpdfF(x, MU, SIGMA, noscale)

    Parameters:
    - x: Input array.
    - MU: Mean value or array of mean values.
    - SIGMA: Standard deviation or array of standard deviations.
    - noscale: Optional, default is 0. If noscale is 0, the result is normalized; otherwise, it is not normalized.

    Returns:
    - y: Probability density function values.
    """
    if len(MU) == 1:
        xn = (x - MU) / SIGMA
    else:
        xn = (x - MU) / SIGMA

    if noscale == 0:
        y = np.exp(-0.5 * xn**2) / (np.sqrt(2 * np.pi) * SIGMA)
    else:
        y = np.exp(-0.5 * xn**2)

    return y

def get_peaks(q, threshold, keep=0):
    """
    get_peaks(q, threshold, keep=0) returns peaks of vector q that are larger than the threshold value.
    If keep is set to 1, it keeps the peak values from one peak to the next.

    Parameters:
    - q: 1D or 2D numpy array, input vector or matrix.
    - threshold: Threshold value for detecting peaks.
    - keep: Optional, default is 0. If set to 1, it keeps the peak values from one peak to the next.

    Returns:
    - ret: Array with peaks, zeros elsewhere.
    """
    if q.ndim == 1:
        q = q.reshape(1, -1)

    ret = np.zeros_like(q)

    for jj in range(q.shape[0]):
        w = np.concatenate((q[jj, :], [q[jj, -1]]))
        k = 0
        l = 0
        rise = 0
        prev = 0

        if threshold != 0:
            if threshold > 0:
                trsh = np.max(w) / threshold
            else:
                trsh = -threshold
        else:
            trsh = 0

        for i in range(len(w) - 1):
            if w[i] < w[i + 1]:
                if k != 0:
                    k = 0
                w[i] = 0
                rise = 1
                if keep == 1:
                    ret[jj, i] = prev
            elif w[i] == w[i + 1]:
                rise = 1
                if k == 0:
                    k = i
                l = i + 1
                if keep == 1:
                    ret[jj, i] = prev
            else:
                if k != 0:
                    if rise != 0:
                        for ii in range(k, l):
                            if ii == (k + l) // 2:
                                ret[jj, ii] = w[ii] * (w[ii] > trsh)
                                prev = ret[jj, ii]
                            elif keep == 1:
                                ret[jj, ii] = prev
                        rise = 0
                    elif keep == 1:
                        ret[jj, i] = prev
                    k = 0

                if rise == 1:
                    rise = 0
                    ret[jj, i] = w[i] * (w[i] > trsh)
                    prev = ret[jj, i]
                else:
                    w[i] = 0
                    if keep == 1:
                        ret[jj, i] = prev

        if k != 0:
            if rise != 0:
                for ii in range(k, l):
                    if ii == (k + l) // 2:
                        ret[jj, ii] = w[ii] * (w[ii] > trsh)
                        prev = ret[jj, ii]
                    elif keep == 1:
                        ret[jj, ii] = prev

    return ret

def get_peaks_v2(q, threshold, keep=0):

    if q.ndim == 1:
        q = q.reshape(1, -1)

    ret = np.zeros_like(q)

    for jj in range(q.shape[0]):
        w = np.concatenate((q[jj, :], [q[jj, -1]]))
        rise_mask = w[:-1] < w[1:]
        peak_indices = np.where(rise_mask)[0]

        if keep == 1:
            peak_values = np.where(rise_mask, w[:-1], 0)
            prev = 0

        if threshold != 0:
            trsh = np.max(w) / threshold
            w[w <= trsh] = 0

        for i in peak_indices:
            if keep == 1:
                ret[jj, i] = prev
                prev = peak_values[i]
            else:
                ret[jj, i] = w[i]

    return ret