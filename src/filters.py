import numpy as np


def moving_average(x: np.ndarray, window: int = 5, center: bool = True) -> np.ndarray:
    x = np.asarray(x, float)
    if window < 1:
        raise ValueError("window должен быть >= 1")
    if window == 1:
        return x.copy()

    if center:
        half = window // 2
        padded = np.pad(x, (half, half), mode="reflect")
        kernel = np.ones(window) / window
        out = np.convolve(padded, kernel, mode="valid")
        return out[: len(x)]
    else:
        out = np.empty_like(x)
        for i in range(len(x)):
            lo = max(0, i - window + 1)
            out[i] = x[lo:i + 1].mean()
        return out


def exponential_smoothing(x: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    x = np.asarray(x, float)
    if not 0 < alpha <= 1:
        raise ValueError("alpha должен быть в диапазоне (0, 1]")
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1 - alpha) * out[t - 1]
    return out


class KalmanFilter1D:

    def __init__(self, q: float = 1e-3, r: float = 1.0,
                 x0: float | None = None, p0: float = 1.0):
        self.q = q
        self.r = r
        self.x0 = x0
        self.p0 = p0

    def filter(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)
        n = len(z)
        out = np.empty(n)

        x_est = z[0] if self.x0 is None else self.x0
        p = self.p0

        for t in range(n):
            x_pred = x_est
            p_pred = p + self.q

            k = p_pred / (p_pred + self.r)
            x_est = x_pred + k * (z[t] - x_pred)
            p = (1 - k) * p_pred

            out[t] = x_est
        return out


def kalman_filter(z: np.ndarray, q: float = 1e-3, r: float = 1.0) -> np.ndarray:
    return KalmanFilter1D(q=q, r=r).filter(z)


def _savgol_coeffs(window: int, poly_order: int) -> np.ndarray:
    if window % 2 == 0:
        raise ValueError("window должен быть нечётным")
    if poly_order >= window:
        raise ValueError("poly_order должен быть меньше window")

    half = window // 2
    positions = np.arange(-half, half + 1)
    A = np.vander(positions, poly_order + 1, increasing=True)
    pinv = np.linalg.pinv(A)
    return pinv[0]


def savitzky_golay(x: np.ndarray, window: int = 7, poly_order: int = 2) -> np.ndarray:
    x = np.asarray(x, float)
    coeffs = _savgol_coeffs(window, poly_order)
    half = window // 2
    padded = np.pad(x, (half, half), mode="reflect")
    out = np.convolve(padded, coeffs[::-1], mode="valid")
    return out[: len(x)]


def lms_filter(x: np.ndarray, n_taps: int = 16, mu: float = 0.1,
               delay: int = 1, normalized: bool = True,
               remove_mean: bool = True):
    x = np.asarray(x, float)
    mean = x.mean() if remove_mean else 0.0
    xc = x - mean

    n = len(xc)
    w = np.zeros(n_taps)
    y = np.zeros(n)
    e = np.zeros(n)
    w_history = np.zeros((n, n_taps))
    eps = 1e-8

    for i in range(n):
        u = np.zeros(n_taps)
        for k in range(n_taps):
            idx = i - delay - k
            if idx >= 0:
                u[k] = xc[idx]

        d = xc[i]
        y[i] = np.dot(w, u)
        e[i] = d - y[i]

        if normalized:
            step = mu / (eps + np.dot(u, u))
        else:
            step = mu
        w = w + step * e[i] * u
        w_history[i] = w

    return y + mean, e, w_history
