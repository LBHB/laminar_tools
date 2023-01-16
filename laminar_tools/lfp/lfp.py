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

cachedir = "/auto/users/wingertj/data/cache"
memory = Memory(cachedir, verbose=0)


@memory.cache
def site_event_lfp(siteid, batch, stim_names, prepoststim, rawlp=100, rawhp=None, fs=300):
    """
    :param fs: sampling rate
    :param prepoststim: window before and after stimulus (output is centered around stim)
    :param siteid: siteid for recording of interest
    :param batch: stimulus set of interest 322 = natural sounds
    :param stim_names: list of stimulus names, set to 'REFERENCE' to obtain all stimuli
    :param rawlp: lowpass filter settings
    :param raw: highpass filter setting if desired
    :return: stim_ids, left lfp sliced around stimulus window, left lfp stimulus epochs, raw lfp signal
    """

    # load lfp data
    ex = BAPHYExperiment(batch=batch, cellid=siteid)
    rec = ex.get_recording(raw=True, pupil=False, resp=False, stim=False, recache=False, rawchans=None, rasterfs=fs,
                           rawlp=rawlp, rawhp=rawhp)
    lfp_ = rec['raw']._data.copy()

    # determine length of prestim silence, convert to sample bins, and assign to stim_time
    ep = rec['raw'].epochs
    epoch = ep.loc[ep['name'].str.startswith("PreStimSilence"), 'name'].values[0]
    stim_list = ep.loc[ep['name'].str.startswith("STIM"), 'name'].values[:]
    epochs = rec['raw'].get_epoch_bounds(epoch)
    fs = rec['raw'].fs

    stim_time = int(np.round(np.array(epochs)[0, 1] - np.array(epochs)[0, 0], decimals=2) * fs)

    #  extract epochs around stims
    stim_epochs = []
    stim_ids = []
    for stim in stim_names:
        stim_epoch = rec['raw'].get_epoch_bounds(ep.loc[ep['name'].str.startswith(stim), 'name'].values[0])
        stimulus = str(stim)
        for i in range(len(stim_epoch)):
            stim_ids.append(stimulus)
        stim_epochs.append(stim_epoch)
    stim_epochs = np.array(stim_epochs)
    stim_epochs = stim_epochs.reshape(int(len(stim_epochs[:, 0, 0]) * len(stim_epochs[0, :, 0])), 2)

    # loop through stimuli and extract epochs from LFP

    lfp_epochs = np.zeros(
        (len(stim_epochs), len(lfp_[:, 0]),
         int(np.round((stim_epochs[0, 1] - stim_epochs[0, 0]), decimals=2) * fs)))

    stim_epochs = np.round((stim_epochs * fs), decimals=2)
    for i in range(len(stim_epochs)):
        lfp_epochs[i, :, :] = lfp_[:, int(stim_epochs[i, 0]):int(stim_epochs[i, 1])]

    # convert prepoststim time to sample bins
    prepoststim = int(np.round(prepoststim * fs))

    # slice out prepoststimulus window from epochs

    lfp_events = lfp_epochs[:, :, int(stim_time - prepoststim):int(stim_time + prepoststim)]

    return stim_list, lfp_events, lfp_epochs, lfp_

@memory.cache
def parmfile_event_lfp(parmfile):

    # same settings as template
    rasterfs = 500
    rawlp = 250
    rawhp = 1
    stim_window = 0.2 * rasterfs
    ex = BAPHYExperiment(parmfile=parmfile)

    # load data
    print("loading data...")
    rec = ex.get_recording(raw=True, resp=False, pupil=False, recache=False, rawchans=None, stim=False,
                           rasterfs=rasterfs, rawlp=rawlp, rawhp=rawhp)
    lfp_ = rec['raw']._data.copy()

    # remove channel offset - seems to be an issue with neuropixels shouldn't impact anything else
    offset = lfp_.mean(axis=1)
    lfp_ = lfp_ - offset[:, np.newaxis]

    # find stimulus onset
    ep = rec['raw'].epochs
    epoch = ep.loc[ep['name'].str.startswith("PreStimSilence"), 'name'].values[0]
    epochs = rec['raw'].get_epoch_bounds(epoch)
    stim_time = int(np.round(np.array(epochs)[0, 1] - np.array(epochs)[0, 0], decimals=2) * rasterfs)

    # check that there is a long enough prestimulus window to use same settings as template for CSD else shorten
    if stim_time < stim_window:
        stim_window = stim_time

    # extract event epochs
    lfp_epochs = rec['raw'].extract_epoch('TRIAL')

    # extract window around stimulus
    lfp_events = lfp_epochs[:, :, int(stim_time - stim_window):int(stim_time + stim_window)]

    # extract data for each column of electrodes
    # check if channel map is returned - if it is returned and channels > 64 assume neuropixel
    try:
        channel_xy = rec['raw'].meta['channel_xy']
    except:
        channel_xy = [None]

    if len(channel_xy) > 64:
        column_xy = {k:v for (k,v) in channel_xy.items() if v[0] == '11'}
        column_xy_sorted = sorted(column_xy, key=lambda k: int(channel_xy[k][1]))
        colunn_nums_sorted = [int(ch)-1 for ch in column_xy_sorted]
        left_lfp = np.take(lfp_, colunn_nums_sorted, axis=0)
        left_lfp_events = np.take(lfp_events, colunn_nums_sorted, axis=1)

        # calculate csd for each trial
        left_csd = np.zeros_like(left_lfp_events[:, :-2, :])
        for i in range(len(left_lfp_events[:, 0, 0])):
            left_csd[i, :, :] = csd1d(left_lfp_events[i, :, :], 11)

    else:
        column_xy = {k:v for (k,v) in channel_xy.items() if v[0] == '-20'}
        column_xy_sorted = sorted(column_xy, key=lambda k: int(channel_xy[k][1]))
        column_nums_sorted = [int(ch)-1 for ch in column_xy_sorted]
        left_lfp = np.take(lfp_, column_nums_sorted, axis=0)
        left_lfp_events = np.take(lfp_events, column_nums_sorted, axis=1)

        # calculate csd for each trial
        left_csd = np.zeros_like(left_lfp_events[:, :-2, :])
        for i in range(len(left_lfp_events[:, 0, 0])):
            left_csd[i, :, :] = csd1d(left_lfp_events[i, :, :], 11)
        # left_lfp_events, center_lfp_events, right_lfp_events = column_split(lfp_events, axis=1)
        # left_lfp, center_lfp, right_lfp = column_split(lfp_, axis=0)
        # column_xy_sorted = None
        #
        # # calculate csd for each trial
        # left_csd = np.zeros_like(left_lfp_events[:, :-2, :])
        # for i in range(len(left_lfp_events[:, 0, 0])):
        #     left_csd[i, :, :] = csd1d(left_lfp_events[i, :, :], 7)

    left_csd = left_csd.mean(axis=0)

    # nan pad csd
    csd_pad = np.empty((1, len(left_csd[0, :])))
    csd_pad[:] = np.nan
    left_csd_padded = np.vstack((csd_pad, left_csd, csd_pad))

    # relative power spectrum
    freqs, power, relative_power = welch_relative_power(left_lfp, rasterfs, nperseg=1024)

    # channel by channel coherence matrix
    fx, coh_mat = coherence_matrix(left_lfp, rasterfs, nperseg=1024)

    windowtime = stim_window / rasterfs

    erp = left_lfp_events.mean(axis=0)

    return left_csd_padded, power, freqs, windowtime, rasterfs, column_xy_sorted, channel_xy, coh_mat, erp


def coherence_matrix(signal_mat, fs, nperseg):
    """
    :param signal_mat: a channels by time array/matrix
    :param fs: sampling frequency of signal
    :param nperseg: number of sample bins in windows to calculate spectra from using welch's method with hanning
    window and 50% overlap between segments.
    :return: array of frequencies (related to segment length nyquist limit) and the channel by channel coherence matrix
    at all frequencies in a [freq, chan, chan] format.
    """
    channels, time = signal_mat.shape
    frequencies = int(nperseg / 2) + 1
    cohmat = np.zeros([int(frequencies), int(channels), int(channels)])
    for i in range(int(channels)):
        x = signal_mat[i, :]
        for j in range(int(channels)):
            y = signal_mat[j, :]
            f, Cxy = coherence(x, y, fs=fs, nperseg=nperseg)
            cohmat[:, i, j] = Cxy
    return f, cohmat


def channel_power(lfp, fs, bin_size=0.5):
    """
    Returns the scaled power spectrum for each electrode with the associated frequencies.
    :param bin_size: data will be binned into frequency bins of this width be
    :param lfp: data in channels by time
    :param fs: sampling rate of data
    :return: list of frequencies and the scaled power at those frequencies as well as a resampled version of the power
    """
    # use discrete fast fourier transform to get fourier coefficients. Take absolute value to get amplitude.
    # multiply by 2 given symmetric nature of fourier transform to capture all power within frequency.
    powspec = 2 * abs(fft(lfp)) / len(lfp[0, :])
    powspec = powspec[:, :int(len(powspec[0, :]) / 2)]
    # Find frequencies associated with coefficients in hz. Max frequency should be Nyquist limit.
    freqs = fftfreq(len(lfp[0, :]), 1 / fs)
    freqs = freqs[:int(len(freqs) / 2)]
    resampled_powspec = np.zeros((len(powspec[:, 0]), int((fs / 2) / bin_size)))
    for i in range(int((fs / 2) / bin_size)):
        for ch in range(len(powspec[:, 0])):
            # find upper index
            dif = freqs - (i * bin_size + bin_size)
            dif[dif > 0] = np.NINF
            upper_index = np.where(dif == dif.max())
            upper_index = upper_index[0] + 1
            dif = freqs - (i * bin_size)
            dif[dif < 0] = np.inf
            lower_index = np.where(dif == dif.min())
            lower_index = lower_index[0]
            resampled_powspec[ch, i] = np.mean(powspec[ch, int(lower_index):int(upper_index)])
            resampled_freqs = np.linspace(0, int(fs / 2), int((fs / 2) / bin_size))

    return freqs, powspec, resampled_freqs, resampled_powspec

def welch_relative_power(lfp, fs, nperseg):
    """
    Calcuates the power spectrum as an average of nperseg sliding windowed, hanning tapered, half-overlapped segments.
    For each frequency bin, finds the relative maximum and normalizes each frequency bin to its own maximum. This
    generates the relative maximum power spectrum over channels.

    :param lfp: Channel x time matrix of lfp
    :param fs: sampling rate of lfp
    :param nperseg: number of samples to use for sliding window
    :return: frequencies, power spectrum, relative power spectrum
    """
    # calcuate the power spectrum as an average of nperseg sliding windowed, hanning tapered, half-overlapped segments.
    f, Pxx = welch(lfp, fs=fs, window='hann', nperseg=nperseg, scaling='spectrum')

    # for each frequency bin calculate the relative maximum
    Pxx_relative_maximums = Pxx.max(axis=0)

    # normalize each frequency bin to the relative_maximums
    relative_Pxx = Pxx / Pxx_relative_maximums

    return f, Pxx, relative_Pxx
