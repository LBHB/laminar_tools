import numpy as np
import pandas as pd


def column_split(data, axis=0):
    data = np.array(data)
    chan_map = pd.read_csv('/auto/users/wingertj/code/csd_project/src/probe_specific/channel_map_64D_reverse.csv',
                           delimiter=',', header=None, names=["channel number", "x", "y"])

    # take columns with same x offset
    left_ch_nums = chan_map.index[chan_map['x'] == -20].tolist()
    right_ch_nums = chan_map.index[chan_map['x'] == 20].tolist()
    center_ch_nums = chan_map.index[chan_map['x'] == 0].tolist()

    # slice data for each column and return
    left_data = np.take(data, indices=left_ch_nums, axis=axis)
    right_data = np.take(data, indices=right_ch_nums, axis=axis)
    center_data = np.take(data, indices=center_ch_nums, axis=axis)

    return left_data, center_data, right_data
