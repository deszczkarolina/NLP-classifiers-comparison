import numpy as np


def split_dataset(df, split_percent):
    total_samples = len(df)
    cutoff = int(np.floor(split_percent * total_samples))
    first_part = df[0:cutoff]
    second_part = df[cutoff + 1: total_samples]
    return first_part, second_part
