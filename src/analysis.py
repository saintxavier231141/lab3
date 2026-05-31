import numpy as np
import pandas as pd


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    t = np.arange(len(x))
    slope, intercept = np.polyfit(t, x, 1)
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "max": float(x.max()),
        "trend_slope": float(slope),
    }


def autocorrelation(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    x = np.asarray(x, float)
    n = len(x)
    if max_lag is None:
        max_lag = n // 2
    x = x - x.mean()
    denom = np.dot(x, x)
    acf = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        acf[lag] = np.dot(x[: n - lag], x[lag:]) / denom
    return acf


def dominant_period(x: np.ndarray, min_period: int = 2):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    spectrum = np.fft.rfft(x)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)

    valid = freqs > 0
    f_valid = freqs[valid]
    p_valid = power[valid]

    periods = 1.0 / f_valid
    mask = periods >= min_period
    if not np.any(mask):
        return None, freqs, power

    best_idx = np.argmax(p_valid[mask])
    best_period = float(periods[mask][best_idx])
    return best_period, freqs, power


def acf_period(acf: np.ndarray, min_lag: int = 2):
    if len(acf) <= min_lag + 1:
        return None
    inner = acf[min_lag:-1]
    if len(inner) < 1:
        return None
    peaks = []
    for i in range(1, len(inner) - 1):
        if inner[i] > inner[i - 1] and inner[i] >= inner[i + 1]:
            peaks.append((i + min_lag, inner[i]))
    if not peaks:
        return None
    best_lag = max(peaks, key=lambda p: p[1])[0]
    return int(best_lag)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(method="pearson")


def cross_correlation(a: np.ndarray, b: np.ndarray, max_lag: int = 10) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = len(a)
    lags = range(-max_lag, max_lag + 1)
    out = []
    for lag in lags:
        if lag < 0:
            r = np.mean(a[-lag:] * b[: n + lag])
        elif lag > 0:
            r = np.mean(a[: n - lag] * b[lag:])
        else:
            r = np.mean(a * b)
        out.append(r)
    return np.array(out)
