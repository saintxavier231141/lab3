import random

import numpy as np
import pandas as pd

PRODUCTS = ["мыло", "порошок", "средство", "краска", "пена", "прибыль"]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def variant_from_isu(isu: int) -> int:
    digits_sum = sum(int(d) for d in str(abs(isu)))
    v = digits_sum % 60
    return 60 if v == 0 else v


def load_sell(path: str = "sell.csv") -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", encoding="cp1251", header=None,
                      skiprows=3, decimal=",")
    days = raw.iloc[:, 0].astype(int)
    data = raw.iloc[:, 1:].astype(float)

    columns = []
    for v in range(1, 61):
        for product in PRODUCTS:
            columns.append(f"v{v:02d}_{product}")
    data.columns = columns
    data.index = days
    data.index.name = "day"
    return data


def get_variant_data(df: pd.DataFrame, variant: int) -> pd.DataFrame:
    cols = [f"v{variant:02d}_{p}" for p in PRODUCTS]
    sub = df[cols].copy()
    sub.columns = PRODUCTS
    return sub


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.abs(a - b)))


def snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    clean, noisy = np.asarray(clean, float), np.asarray(noisy, float)
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((noisy - clean) ** 2)
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def roughness(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.mean(np.diff(x, n=2) ** 2))


def residual_std(original: np.ndarray, filtered: np.ndarray) -> float:
    return float(np.std(np.asarray(original, float) - np.asarray(filtered, float)))
