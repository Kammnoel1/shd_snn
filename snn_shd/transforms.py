import numpy as np


def merge_channels(n_merge: int):
    def _transform(times: int, units: int):
        return times, units // n_merge

    return _transform


def channel_jitter(sigma: float, num_neurons: int):
    def _transform(times, units):
        noise = np.random.normal(0, sigma, size=units.shape)
        jittered = np.round(units + noise)
        jittered = np.clip(jittered, 0, num_neurons - 1).astype(int)
        return times, jittered

    return _transform
