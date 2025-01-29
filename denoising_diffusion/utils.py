import torch


def normalize_to_neg1_1(x):
    return x * 2 - 1

def unnormalize_to_zero_one(x):
    return (x + 1) / 2

def identity(x):
    return x

def default(val, def_val):
    return def_val if val is None else val