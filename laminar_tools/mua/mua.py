from scipy.signal import coherence
from nems_lbhb.baphy_experiment import BAPHYExperiment
import numpy as np
from scipy import fft
from scipy.signal import welch
from scipy.fftpack import fftfreq
import pandas as pd
from laminar_tools.probe_specific.probe64d import column_split
from laminar_tools.csd.csd import csd1d
from joblib import Memory
import os
import getpass
linux_user = getpass.getuser()


cachedir = f"/auto/users/{linux_user}/data/cache"
memory = Memory(cachedir, verbose=0)


def site_mua(siteid, batch, stim_names, prepoststim, fs=100, muabp=(500, 5000)):
    """
    Bandpass filters mua and smooths with a 10 bin boxcar. Slices out prepoststim windows around events.

    :param batch:
    :param lp: lowpass frequency
    :param hp: highpass frequency
    :param fs: sampling frequency desired in hz
    :param siteid: siteid
    :param stim_names: list of stimulus names to take epochs from
    :param prepoststim: time in seconds that should be taken before and after stimulus
    :return: stim_ids, mua events around stimulus, mua stimulus epochs, raw mua
    """
    import numpy as np
    from nems_lbhb.baphy_experiment import BAPHYExperiment
    from nems.utils import smooth

    batch = 322

    ex = BAPHYExperiment(batch=batch, cellid=siteid)
    rec = ex.get_recording(mua = True, raw=False, pupil=False, resp=False, stim=False, recache=False, rawchans=None,
                           rasterfs=fs, muabp=bp)

    # high and low pass filter to lfp and mua
    mua_ = rec['mua']._data.copy()
    #mua_ = smooth(mua_, 10)

    # determine length of prestim silence, convert to sample bins,  and assign to stim_time
    ep = rec['mua'].epochs
    epoch = ep.loc[ep['name'].str.startswith("PreStimSilence"), 'name'].values[0]
    stim_list = ep.loc[ep['name'].str.startswith("STIM"), 'name'].values[:]
    epochs = rec['mua'].get_epoch_bounds(epoch)
    fs = rec['mua'].fs

    stim_time = int(np.round(np.array(epochs)[0, 1] - np.array(epochs)[0, 0], decimals=2) * fs)

    # extract epochs
    mua_epochs = rec['mua'].extract_epoch('TRIAL')
    # #  extract epochs around stims
    #
    # stim_epochs = []
    # stim_ids = []
    # for stim in stim_names:
    #     stim_epoch = rec['mua'].get_epoch_bounds(ep.loc[ep['name'].str.startswith(stim), 'name'].values[0])
    #     stimulus = str(stim)
    #     for j in range(len(stim_epoch)):
    #         stim_ids.append(stimulus)
    #     stim_epochs.append(stim_epoch)
    # stim_epochs = np.array(stim_epochs)
    # stim_epochs = stim_epochs.reshape(int(len(stim_epochs[:, 0, 0]) * len(stim_epochs[0, :, 0])), 2)
    #
    # # loop through stimuli and extract epochs from mua
    #
    # mua_epochs = np.zeros((len(stim_epochs), len(mua_[:, 0]), int(np.round((stim_epochs[0, 1] - stim_epochs[0, 0]),
    #                                                                        decimals=2) * fs)))
    # stim_epochs = np.round((stim_epochs * fs), decimals=2)
    # for i in range(len(stim_epochs)):
    #     mua_epochs[i, :, :] = mua_[:, int(stim_epochs[i, 0]):int(stim_epochs[i, 1])]

    # convert prepoststim time to sample bins
    sample_window = int(np.round(prepoststim * fs))

    # slice out prepoststimulus window from epochs
    mua_events = mua_epochs[:, :, int(stim_time - sample_window):int(stim_time + sample_window)]

    return stim_list, mua_events, mua_epochs, mua_

@memory.cache
def parmfile_mua_FTC(parmfile):

    # same settings as template
    fs = 200
    bp = (500, 5000)
    ex = BAPHYExperiment(parmfile=parmfile)

    # load data
    print("loading data...")
    rec = ex.get_recording(mua = True, raw=False, pupil=False, resp=False, stim=False, recache=False, rawchans=None,
                           rasterfs=fs, muabp=bp)

    mua_ = rec['raw']._data.copy()

    try:
        channel_xy = rec['raw'].meta['channel_xy']
    except:
        channel_xy = [None]

    probes = rec['raw'].meta['probe']
    probe_type = rec['raw'].meta['probe_type']

    for prb_ind in range(len(probes)):
        if probe_type == 'NPX':
            column_xy = {k:v for (k,v) in channel_xy[prb_ind].items() if v[0] == '11'}
            probe_letter = probes[prb_ind][-1:]
            if len(probes) >= 2:
                physical_channel_num = [prb_ch for prb_ch in rec['raw'].chans if f"{probe_letter}-" in prb_ch]
            else:
                physical_channel_num = rec['raw'].chans
            # sort xy col dict based on xy position
            column_xy_sorted = sorted(column_xy, key=lambda k: int(channel_xy[prb_ind][k][1]))
            col_sort_dict = {k: column_xy[k] for k in column_xy_sorted}
            # check to see if newer version of baphy_experiment is using letters in channel naming scheme
            chan_name_letters = any([channel_char.isalpha() for channel  in physical_channel_num for channel_char in channel])
            if chan_name_letters:
                physical_channel_int = [int(ch[2:]) for ch in physical_channel_num]
                sorted_physical_index = [rec['raw'].chans.index(f"{probe_letter}-"+ch) for ch in column_xy_sorted]
            else:
                physical_channel_int = [int(ch[2:]) for ch in physical_channel_num]
                sorted_physical_index = [physical_channel_num.index(ch) for ch in column_xy_sorted]
            column_nums_sorted = [int(ch) for ch in column_xy_sorted]
            left_mua = np.take(mua_, sorted_physical_index, axis=0)

            # deal with discontinuity...fill missing channels with nans
            #find indexes where channels are discontinous
            try:
                # diffs = [[bool((y - x) == 1) for x, y in zip(physical_channel_int, physical_channel_int[1:])].index(False)]
                column_diffs = [i for i,x in enumerate([(y-x) == 4 for x,y in zip(column_nums_sorted, column_nums_sorted[1:])]) if x == False]

            except:
                print("all channels are contiguous with one another")
                column_diffs = False
            if column_diffs:
                for dif in column_diffs:
                    # how many channels are missing between indexes
                    gap_len = (column_nums_sorted[dif + 1] - (column_nums_sorted[dif] + 4))/4
                    gap_chans = np.arange(column_nums_sorted[dif]+4, column_nums_sorted[dif + 1], 4)
                    # fill lfp with NaNs of the same shape as data and number of missing channels
                    gap_chans_y = np.arange(int(column_xy[str(column_nums_sorted[dif])][1])+40, int(column_xy[str(column_nums_sorted[dif+1])][1]), 40)
                    channel_xy_gap_filled = channel_xy.copy()
                    for i in range(len(gap_chans)):
                        column_xy[str(gap_chans[i])] = ['11', gap_chans_y[i]]
                        channel_xy_gap_filled[str(gap_chans[i])] = ['11', gap_chans_y[i]]
                    column_xy_sorted = sorted(column_xy, key=lambda k: int(column_xy[k][1]))
                    left_mua_gap = np.empty((len(gap_chans), len(left_mua[0, :])))
                    left_mua_gap[:] = np.nan
                    left_mua = np.concatenate((left_mua[:dif+1], left_mua_gap, left_mua[dif+1:]), axis=0)

        elif probe_type == 'UCLA':
            column_xy = {k:v for (k,v) in channel_xy.items() if v[0] == '-20'}
            column_xy_sorted = sorted(column_xy, key=lambda k: int(channel_xy[k][1]))
            column_nums_sorted = [int(ch)-1 for ch in column_xy_sorted]
            left_mua = np.take(mua_, column_nums_sorted, axis=0)
        else:
            raise ValueError("Unsupported probe type")

    return left_mua
