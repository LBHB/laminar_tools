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
