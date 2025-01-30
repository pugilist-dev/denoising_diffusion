import torch


def normalize_to_neg1_1(x):
    return x * 2 - 1

def unnormalize_to_zero_one(x):
    return (x + 1) / 2

def identity(x):
    return x

def default(val, def_val):
    return def_val if val is None else val

def cycle(dl):
    while True:
        for data in dl:
            yield data

def extract(
    constants, timestamps, shape
):
    batch_size = timestamps.shape[0]
    out = constants.gather(-1, timestamps)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(timestamps.device)