import numpy as np


def nan_split_2D(data, axis=0):
    if axis == 0:
        nan_list = data[:, 0]
    elif axis == 1:
        nan_list = data[0, :]
    else:
        raise ValueError("axis must be 0 or 1")
    nan_list.ravel
    nan_bool = np.isnan(nan_list)
    nan_int = [-1 if x == False else 1 for x in nan_bool]
    nan_index = list(np.where([(x-y) != 0 for x, y in zip(nan_int, nan_int[1:])])[0]+1)
    data_split = np.split(data, nan_index, axis=axis)
    return data_split