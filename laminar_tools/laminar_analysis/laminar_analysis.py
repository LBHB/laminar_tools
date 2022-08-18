import numpy as np
import pandas as pd
from src.csd.csd import detect_peaks, csd1d
from src.lfp.lfp import coherence_matrix, welch_relative_power
from src.probe_specific.probe64d import column_split
from src.probe_specific.NHP_neuropixelPhase3A import column_split_npx
import scipy.io as io


def column_data_64D(column, siteid, mua_, lfp_, lfp_events, stim_window, rasterfs, neighborhood, threshold):
    """

    :param threshold:
    :param neighborhood:
    :param siteid:
    :param column:
    :param mua_:
    :param lfp_:
    :param lfp_events:
    :param stim_window:
    :param rasterfs:
    :return:
    """
    # channel spacing in microns
    channel_spacing = 50
    # load channel maps
    chan_map = pd.read_csv('/auto/users/wingertj/code/csd_project/src/probe_specific/channel_map_64D_reverse.csv',
                           delimiter=',', header=None, names=["channel number", "x", "y"])
    if column == 'left':
        chan_nums = chan_map.index[chan_map['x'] == -20].tolist()
        column_index = 0
    elif column == 'right':
        chan_nums = chan_map.index[chan_map['x'] == 20].tolist()
        column_index = 2
    elif column == 'center':
        chan_nums = chan_map.index[chan_map['x'] == 0].tolist()
        column_index = 1
    else:
        raise ValueError("column expected: left, right, center")
    # separate columns of data
    column_lfp = column_split(lfp_, axis=0)[column_index]
    column_mua = column_split(mua_, axis=0)[column_index]
    column_lfp_events = column_split(lfp_events, axis=1)[column_index]

    # calculate csd for each trial
    column_csd = np.zeros_like(column_lfp_events[:, :-2, :])
    for i in range(len(column_lfp_events[:, 0, 0])):
        column_csd[i, :, :] = csd1d(column_lfp_events[i, :, :], 7)

    # save csd to DataFrame
    column_csd_df = pd.DataFrame(column_csd.mean(axis=0))
    column_csd_df["channel index"] = chan_nums[1:-1]
    column_csd_df["column"] = column
    column_csd_df["siteid"] = siteid

    # save erps to csv
    column_erp_df = pd.DataFrame(column_lfp_events.mean(axis=0))
    column_erp_df["channel index"] = chan_nums
    column_erp_df["column"] = column
    column_erp_df["siteid"] = siteid

    # channel by channel coherence analysis

    # common average reference
    column_lfp_cavg = column_lfp - column_lfp.mean(axis=0)

    print('calculating coherence matrix...')

    # calculate channel by channel coherence for common average referenced lfp
    f, column_cohmat = coherence_matrix(column_lfp, fs=rasterfs, nperseg=256)

    # find avg channel by channel coherence in the gamma range
    idx30 = np.where(f > 30)[0][0]
    idx100 = np.where(f < 100)[0][-1]
    column_gamma_cohmat = np.squeeze(column_cohmat[idx30:idx100, :, :].mean(axis=0))

    print("saving coherence data")
    # save coherence data to csv
    column_gamma_coherence = pd.DataFrame(column_gamma_cohmat, columns=chan_nums)
    column_gamma_coherence["channel index"] = chan_nums
    column_gamma_coherence['siteid'] = siteid
    column_gamma_coherence['column'] = column

    # calculate channel by channel coherence non-common average referenced lfp
    print("calculating coherence matrix after common average reference")
    f, column_cohmat_cavg = coherence_matrix(column_lfp_cavg, fs=rasterfs, nperseg=512)

    # find avg channel by channel coherence in the gamma range
    idx30 = np.where(f > 30)[0][0]
    idx100 = np.where(f < 100)[0][-1]
    column_gamma_cohmat_cavg = np.squeeze(column_cohmat_cavg[idx30:idx100, :, :].mean(axis=0))

    print("saving common average coherence data")
    # save coherence data to csv
    column_gamma_coherence_cavg = pd.DataFrame(column_gamma_cohmat_cavg, columns=chan_nums)
    column_gamma_coherence_cavg["channel index"] = chan_nums
    column_gamma_coherence_cavg['siteid'] = siteid
    column_gamma_coherence_cavg['column'] = column

    # calculate power spectrum across channels
    print('calculating relative power...')

    # freqs, power, resampled_freqs, resampled_power = channel_power(column_lfp, rasterfs)

    freqs, power, relative_power = welch_relative_power(column_lfp, rasterfs, nperseg=1024)
    # relative power spectrum by average power in that frequency

    # relative_power = relative_power_matrix(resampled_power)

    # slice out power spectrum below nyquist (non-negative frequencies)

    # relative_power = relative_power[:, 0:int(np.where(freqs == freqs.max())[0][0])]
    print('saving power df')
    power_df = pd.DataFrame(power, columns=freqs)
    power_df['siteid'] = siteid
    power_df['channel index'] = chan_nums
    power_df['column'] = column

    print("saving relative power..")
    relative_power_df = pd.DataFrame(relative_power, columns=freqs)
    relative_power_df['siteid'] = siteid
    relative_power_df['channel index'] = chan_nums
    relative_power_df['column'] = column

    # save average coherence across channels
    gamma_coherence_distribution_cavg = \
        column_gamma_cohmat_cavg.mean(axis=1) / column_gamma_cohmat_cavg.mean(axis=1).max()
    channel_features = pd.DataFrame()
    channel_features["gamma coherence cavg"] = gamma_coherence_distribution_cavg

    # save average coherence across channels
    gamma_coherence_distribution = column_gamma_cohmat.mean(axis=1) / column_gamma_cohmat.mean(axis=1).max()
    channel_features["gamma coherence"] = gamma_coherence_distribution

    # channel with the least gamma coherence

    loco = gamma_coherence_distribution_cavg.min()
    loco = np.where(gamma_coherence_distribution_cavg == loco)

    # find peak in mua around stimulus
    # find power of mua over time
    mua_power = column_mua.sum(axis=1)

    # save mua power distribution
    channel_features["mua power"] = mua_power
    # find peak over channels which should correspond to mid-layer 5 - Buzsaki visual cortex paper

    layer5amp = mua_power.max()

    # find channel corresponding to mua peak
    layer5 = np.where(mua_power == layer5amp)

    # normalize mua power distribution to max
    mua_power_norm = mua_power / layer5amp
    channel_features["mua power normalized"] = mua_power_norm

    # save raw channel data
    channel_features['siteid'] = siteid
    channel_features['channel index'] = chan_nums
    channel_features['column'] = column

    # create new dataframe for csd features
    csd_features = pd.DataFrame()

    # find CSD peaks
    large_peaks, all_peaks = detect_peaks(column_csd.mean(axis=0), neighbors=neighborhood, threshold=threshold)

    # save peak data
    peak_list = [peak for (peak, loc) in large_peaks]
    csd_features['peak amplitude'] = peak_list
    csd_features['siteid'] = siteid
    peak_channel = []
    for i in range(len(large_peaks)):
        peak_channel.append(large_peaks[i][1][0])
    csd_features['column channel index'] = peak_channel
    csd_features['probe channel index'] = [chan_nums[int(channel)] for channel in peak_channel]
    # find distance of CSD peaks from peak mua
    mua_csd_peak_distance = []
    for i in range(len(large_peaks)):
        mua_csd_peak_distance.append(int(large_peaks[i][1][0] - layer5))

    # save coordinates of csd peaks referenced to mua peak
    csd_features['mua to peak'] = mua_csd_peak_distance

    # find distance of CSD peaks from channel with lowest coherence
    loco_csd_peak_distance = []
    for i in range(len(large_peaks)):
        loco_csd_peak_distance.append(int(large_peaks[i][1][0] - loco))

    # save coordinates of csd peaks referenced to mua peak
    csd_features['loco to peak'] = loco_csd_peak_distance

    # find distance of CSD peaks from stimulus onset
    stim_csd_peak_distance = []
    for i in range(len(large_peaks)):
        stim_csd_peak_distance.append((large_peaks[i][1][1] - stim_window) / rasterfs)

    # save coordinates of csd peaks referenced to stimulus onset
    csd_features['stim to peak'] = stim_csd_peak_distance

    return column_csd_df, column_erp_df, column_gamma_coherence, column_gamma_coherence_cavg, power_df, relative_power_df, channel_features, csd_features


def column_data_npx(column, siteid, mua_, lfp_, lfp_events, stim_window, rasterfs, neighborhood, threshold):
    """

    :param threshold:
    :param neighborhood:
    :param siteid:
    :param column:
    :param mua_:
    :param lfp_:
    :param lfp_events:
    :param stim_window:
    :param rasterfs:
    :return:
    """
    # channel spacing in microns
    channel_spacing = 50
    # load channel maps
    npx_dict = io.loadmat('/auto/data/code/KiloSort/chanMap_neuropixPhase3A.mat')
    chan_map = pd.DataFrame(npx_dict['chanMap'], columns=['chanMap'])
    chan_map['xcoords'] = npx_dict['xcoords']
    chan_map['ycoords'] = npx_dict['ycoords']
    x_vals = np.sort(chan_map.xcoords.unique())

    if column == 'col0':
        chan_nums = chan_map.index[chan_map['xcoords'] == x_vals[0]].tolist()
        column_index = 0
    elif column == 'col1':
        chan_nums = chan_map.index[chan_map['xcoords'] == x_vals[1]].tolist()
        column_index = 1
    elif column == 'col2':
        chan_nums = chan_map.index[chan_map['xcoords'] == x_vals[2]].tolist()
        column_index = 2
    elif column == 'col3':
        chan_nums = chan_map.index[chan_map['xcoords'] == x_vals[3]].tolist()
        column_index = 3
    else:
        raise ValueError("column expected: col0, col1, col2, col3")

    # separate columns of data
    column_lfp = column_split_npx(lfp_, axis=0)[column_index]
    column_mua = column_split_npx(mua_, axis=0)[column_index]
    column_lfp_events = column_split_npx(lfp_events, axis=1)[column_index]

    # calculate csd for each trial
    column_csd = np.zeros_like(column_lfp_events[:, :-2, :])
    for i in range(len(column_lfp_events[:, 0, 0])):
        column_csd[i, :, :] = csd1d(column_lfp_events[i, :, :], 25)

    # save csd to DataFrame
    column_csd_df = pd.DataFrame(column_csd.mean(axis=0))
    column_csd_df["channel index"] = chan_nums[1:-1]
    column_csd_df["column"] = column
    column_csd_df["siteid"] = siteid

    # save erps to csv
    column_erp_df = pd.DataFrame(column_lfp_events.mean(axis=0))
    column_erp_df["channel index"] = chan_nums
    column_erp_df["column"] = column
    column_erp_df["siteid"] = siteid

    # channel by channel coherence analysis

    # common average reference
    column_lfp_cavg = column_lfp - column_lfp.mean(axis=0)

    print('calculating coherence matrix...')

    # calculate channel by channel coherence for common average referenced lfp
    f, column_cohmat = coherence_matrix(column_lfp, fs=rasterfs, nperseg=256)

    # find avg channel by channel coherence in the gamma range
    idx30 = np.where(f > 30)[0][0]
    idx100 = np.where(f < 100)[0][-1]
    column_gamma_cohmat = np.squeeze(column_cohmat[idx30:idx100, :, :].mean(axis=0))

    print("saving coherence data")
    # save coherence data to csv
    column_gamma_coherence = pd.DataFrame(column_gamma_cohmat, columns=chan_nums)
    column_gamma_coherence["channel index"] = chan_nums
    column_gamma_coherence['siteid'] = siteid
    column_gamma_coherence['column'] = column

    # calculate channel by channel coherence non-common average referenced lfp
    print("calculating coherence matrix after common average reference")
    f, column_cohmat_cavg = coherence_matrix(column_lfp_cavg, fs=rasterfs, nperseg=512)

    # find avg channel by channel coherence in the gamma range
    idx30 = np.where(f > 30)[0][0]
    idx100 = np.where(f < 100)[0][-1]
    column_gamma_cohmat_cavg = np.squeeze(column_cohmat_cavg[idx30:idx100, :, :].mean(axis=0))

    print("saving common average coherence data")
    # save coherence data to csv
    column_gamma_coherence_cavg = pd.DataFrame(column_gamma_cohmat_cavg, columns=chan_nums)
    column_gamma_coherence_cavg["channel index"] = chan_nums
    column_gamma_coherence_cavg['siteid'] = siteid
    column_gamma_coherence_cavg['column'] = column

    # calculate power spectrum across channels
    print('calculating relative power...')
    freqs, power, relative_power = welch_relative_power(column_lfp, rasterfs, 1024)

    # freqs, power, resampled_freqs, resampled_power = channel_power(column_lfp, rasterfs)

    # relative power spectrum by average power in that frequency

    # relative_power = relative_power_matrix(resampled_power)

    # slice out power spectrum below nyquist (non-negative frequencies)

    # relative_power = relative_power[:, 0:int(np.where(freqs == freqs.max())[0][0])]

    print('saving power df')
    power_df = pd.DataFrame(power, columns=freqs)
    power_df['siteid'] = siteid
    power_df['channel index'] = chan_nums
    power_df['column'] = column

    print("saving relative power..")
    relative_power_df = pd.DataFrame(relative_power, columns=freqs)
    relative_power_df['siteid'] = siteid
    relative_power_df['channel index'] = chan_nums
    relative_power_df['column'] = column

    # save average coherence across channels
    gamma_coherence_distribution_cavg = \
        column_gamma_cohmat_cavg.mean(axis=1) / column_gamma_cohmat_cavg.mean(axis=1).max()
    channel_features = pd.DataFrame()
    channel_features["gamma coherence cavg"] = gamma_coherence_distribution_cavg

    # save average coherence across channels
    gamma_coherence_distribution = column_gamma_cohmat.mean(axis=1) / column_gamma_cohmat.mean(axis=1).max()
    channel_features["gamma coherence"] = gamma_coherence_distribution

    # channel with the least gamma coherence

    loco = gamma_coherence_distribution_cavg.min()
    loco = np.where(gamma_coherence_distribution_cavg == loco)

    # find peak in mua around stimulus
    # find power of mua over time
    mua_power = column_mua.sum(axis=1)

    # save mua power distribution
    channel_features["mua power"] = mua_power
    # find peak over channels which should correspond to mid-layer 5 - Buzsaki visual cortex paper

    layer5amp = mua_power.max()

    # find channel corresponding to mua peak
    layer5 = np.where(mua_power == layer5amp)

    # normalize mua power distribution to max
    mua_power_norm = mua_power / layer5amp
    channel_features["mua power normalized"] = mua_power_norm

    # save raw channel data
    channel_features['siteid'] = siteid
    channel_features['channel index'] = chan_nums
    channel_features['column'] = column

    # create new dataframe for csd features
    csd_features = pd.DataFrame()

    # find CSD peaks
    large_peaks, all_peaks = detect_peaks(column_csd.mean(axis=0), neighbors=neighborhood, threshold=threshold)

    # save peak data
    peak_list = [peak for (peak, loc) in large_peaks]
    csd_features['peak amplitude'] = peak_list
    csd_features['siteid'] = siteid
    peak_channel = []
    for i in range(len(large_peaks)):
        peak_channel.append(large_peaks[i][1][0])
    csd_features['column channel index'] = peak_channel
    csd_features['probe channel index'] = [chan_nums[int(channel)] for channel in peak_channel]
    # find distance of CSD peaks from peak mua
    mua_csd_peak_distance = []
    for i in range(len(large_peaks)):
        mua_csd_peak_distance.append(int(large_peaks[i][1][0] - layer5))

    # save coordinates of csd peaks referenced to mua peak
    csd_features['mua to peak'] = mua_csd_peak_distance

    # find distance of CSD peaks from channel with lowest coherence
    loco_csd_peak_distance = []
    for i in range(len(large_peaks)):
        loco_csd_peak_distance.append(int(large_peaks[i][1][0] - loco))

    # save coordinates of csd peaks referenced to mua peak
    csd_features['loco to peak'] = loco_csd_peak_distance

    # find distance of CSD peaks from stimulus onset
    stim_csd_peak_distance = []
    for i in range(len(large_peaks)):
        stim_csd_peak_distance.append((large_peaks[i][1][1] - stim_window) / rasterfs)

    # save coordinates of csd peaks referenced to stimulus onset
    csd_features['stim to peak'] = stim_csd_peak_distance

    return column_csd_df, column_erp_df, column_gamma_coherence, column_gamma_coherence_cavg, power_df, relative_power_df, channel_features, csd_features