import numpy as np
import pandas as pd
import scipy.io as io
from laminar_tools.csd.csd import detect_peaks, csd1d
from laminar_tools.lfp.lfp import coherence_matrix, welch_relative_power, parmfile_event_lfp
from laminar_tools.probe_specific.probe64d import column_split
from laminar_tools.probe_specific.NHP_neuropixelPhase3A import column_split_npx
from skimage.metrics import structural_similarity as ssim


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


def aligned_csd_psd(parmfile):

    psd_template = np.load("/auto/users/wingertj/code/csd_project/data/laminar_features/template/final_psd_template.npy")
    csd, psd = parmfile_event_lfp(parmfile)
    averaged_template, ssim_index, ssim = maximal_laminar_similarity(template=psd_template, image=psd, overlap=10,
                                                                     ssim_window=5, expansion=True)

    nan_pad_psd, nan_pad_csd = nan_pad_images(psd, csd, ssim_index, already_padded=False)

    return nan_pad_csd, nan_pad_csd, ssim


def maximal_laminar_similarity(template, image, overlap=10, ssim_window=5, expansion=True):
    """
    One dimensional (row) alignment of an image to a template image based on structural similarity score. Uses a
    convolution style sliding window where image is slid along template one step at a time. Can specifiy degree of overlap
    desired between template and image as well as window size used for structural similarity calculation. Expansion
    will add the overhanging bits of both the template and the image if the sliding step with the highest score does not
    have a perfect match for length of template and image.

    :param template: starting image that other image will be referenced against
    :param image: image that will be slid along the template and aligned with respect to that template
    :param overlap: number of rows that must overlap between image and template.
    :param ssim_window: window size parameter for structural similarity
    :param expansion: True/False - elongates output template if the step where image and template are maximally similar
                        are not perfectly overlapping
    :return: new_template after image is aligned, the number of shifts/steps the image moved along the template, ssim score
    """

    # store shifted images and corresponding template pieces
    spatial_shifts, shifted_templates, shifted_images = spatial_shifting(template, image)

    # slice off sides of shifted matrices to ensure a certain degree of overlap

    shifted_images_short = shifted_images[:, :, overlap:-overlap]
    shifted_templates_short = shifted_templates[:, :, overlap:-overlap]

    # compute image similarity between each of the shifts
    shift_similarity = []
    for i in range(len(shifted_templates_short[0, 0, :])):
        shifted_image = shifted_images_short[:, :, i]
        shifted_image = shifted_image[~np.isnan(shifted_image).any(axis=1)]
        shifted_template = shifted_templates_short[:, :, i]
        shifted_template = shifted_template[~np.isnan(shifted_template).any(axis=1)]
        shift_similarity.append(ssim(shifted_image, shifted_template, win_size=ssim_window))

    # generate new template by averaging template and image together at spatial shift with max similarity

    max_similarity_index = shift_similarity.index(max(shift_similarity))
    shifted_image = shifted_images_short[:, :, max_similarity_index]
    shifted_image = shifted_image[~np.isnan(shifted_image).any(axis=1)]
    shifted_template = shifted_templates_short[:, :, max_similarity_index]
    shifted_template = shifted_template[~np.isnan(shifted_template).any(axis=1)]
    template_overlap = (shifted_template + shifted_image) / 2

    shift_index = max_similarity_index + overlap

    if expansion:

        # using the maximum shift index, use the inverse of the slice operation used to shift the images to find non-overlapping
        # segments to append to the template to extend its bounds
        new_template = expand(template, image, spatial_shifts, template_overlap, shift_index)
    else:
        new_template = template_overlap

    return new_template, shift_index, max(shift_similarity)


def post_ssim_avg_csd(csd_steps, step_index):

    template = csd_steps[0]
    images = csd_steps[1:]

    for i in range(len(images)):
        # create all spatial shifts of image and template as in laminar similarity so step index returned from laminar
        # similarity can be used
        image = images[i]
        spatial_shifts, shifted_templates, shifted_images = spatial_shifting(template, images[i])

        # using the step index from laminar similarity analysis, average the shifted overlapping CSD segments
        shifted_image = shifted_images[:, :, step_index[i]]
        shifted_image = shifted_image[~np.isnan(shifted_image).any(axis=1)]
        shifted_template = shifted_templates[:, :, step_index[i]]
        shifted_template = shifted_template[~np.isnan(shifted_template).any(axis=1)]
        template_overlap = (shifted_template + shifted_image) / 2

        # expand based on overhang of template or image
        shift_index = step_index[i]
        new_template = expand(template, image, spatial_shifts, template_overlap, shift_index)
        template = new_template

    return template

# def old_spatial_shifting(template, image):
#     """
#     Spatially shift an image relative to a template creating a matrix of all possible degrees of overlap.
#
#     :param template:
#     :param image:
#     :return: shifted_images, shifted_templates
#     """
#
#     temrow, temcol = template.shape
#     imrow, imcol = image.shape
#
#     # total number of shifts should be same as convolution M+N-1?
#     spatial_shifts = range(temrow + imrow - 1)
#
#     # initialize matrices to store shifted images and corresponding template pieces
#     shifted_images = np.empty((imrow, imcol, len(spatial_shifts)))
#     shifted_images[:] = np.nan
#     shifted_templates = np.empty((imrow, imcol, len(spatial_shifts)))
#     shifted_templates[:] = np.nan
#
#     # for each spatial shift slice the image and the template and store shifts
#     for i in spatial_shifts:
#         if i < len(image[:, 0]):
#             shift_image = image[(len(image[:, 0]) - 1) - i:, :]
#             shift_template = template[:1 + i, :]
#         elif len(template[:, 0]) > i >= len(image[:, 0]):
#             shift_image = image
#             shift_template = template[i - len(image[:, 0]) + 1:i + 1, :]
#         elif i >= len(template):
#             shift_image = image[:len(spatial_shifts) - i, :]
#             shift_template = template[i - len(image[:, 0]) + 1:i + 1, :]
#         shifted_images[:len(shift_image[:, 0]), :, i] = shift_image
#         shifted_templates[:len(shift_template[:, 0]), :, i] = shift_template
#
#     return spatial_shifts, shifted_templates, shifted_images


def spatial_shifting(template, image):
    """
    Spatially shift an image relative to a template creating a matrix of all possible degrees of overlap.

    :param template:
    :param image:
    :return: shifted_images, shifted_templates
    """

    ### define number of shifts ###

    temrow, temcol = template.shape
    imrow, imcol = image.shape

    # total number of shifts should be same as convolution M+N-1
    spatial_shifts = range(temrow + imrow - 1)

    # max kernel size = smaller of temrow or imrow
    if temrow <= imrow:
        max_kernel_size = temrow
    else:
        max_kernel_size = imrow

    ### create empty matrices to store values ###
    shifted_images = np.empty((max_kernel_size, imcol, len(spatial_shifts)))
    shifted_images[:] = np.nan
    shifted_templates = np.empty((max_kernel_size, temcol, len(spatial_shifts)))
    shifted_templates[:] = np.nan

    ### loop through spatial shifts and slice out samples for both image and template ###
    # 4 conditions:
    # 1: the spatial step is less than the image and the template where both samples grow in size
    # 2: If the spatial step is larger the image and smaller than the template the image stays the same size while
        # the template window stays the same size as the image but continues shifting
    # 3: If the spatial step is smaller than the image but larger than the template the template stays the same size
        # while the image window stays the same size as the template but continues shifting
    # 4: The spatial step is larger than both the length of the image and the template. Both the image and the
        # template sample get smaller
    for i in spatial_shifts:
        if i < imrow and i < temrow:
            shift_image = image[(imrow - 1) - i:, :]
            shift_template = template[:i + 1, :]
        elif temrow > i >=  imrow:
            shift_image = image
            shift_template = template[i - imrow + 1:i + 1, :]
        elif imrow > i >= temrow:
            shift_image = image[(imrow - 1) - i:(imrow - 1) - i + temrow]
            shift_template = template
        elif i >= imrow and i >= temrow:
            shift_image = image[:len(spatial_shifts) - i, :]
            shift_template = template[i - imrow + 1:i + 1, :]
        shifted_images[:len(shift_image[:, 0]), :, i] = shift_image
        shifted_templates[:len(shift_template[:, 0]), :, i] = shift_template

    return spatial_shifts, shifted_templates, shifted_images


# def old_expand(template, image, spatial_shifts, overlap, shift_index):
#
#     if shift_index < len(image[:, 0]):
#         image_overhang = image[:(len(image[:, 0]) - 1) - shift_index, :]
#         template_overhang = template[1 + shift_index:, :]
#         new_template = np.vstack((image_overhang, overlap, template_overhang))
#     elif len(template[:, 0]) > shift_index >= len(image[:, 0]):
#         template_overhang_above = template[:shift_index - len(image[:, 0]) + 1, :]
#         template_overhang_below = template[shift_index + 1:, :]
#         new_template = np.vstack((template_overhang_above, overlap, template_overhang_below))
#     elif shift_index >= len(template):
#         image_overhang = image[len(spatial_shifts) - shift_index:, :]
#         template_overhang_above = template[:shift_index - len(image[:, 0]) + 1, :]
#         new_template = np.vstack((template_overhang_above, overlap, image_overhang))
#
#     return new_template

def expand(template, image, spatial_shifts, overlap, shift_index):

    """
    following SSIM analysis, this will stack the overhang of the image and or template if there is not perfect overlap.
    :param template:
    :param image:
    :param spatial_shifts:
    :param overlap:
    :param shift_index:
    :return:
    """
    temrow, temcol = template.shape
    imrow, imcol = image.shape


    if shift_index < imrow and shift_index < temrow:
        image_overhang = image[:(imrow - 1) - shift_index, :]
        template_overhang = template[1 + shift_index:, :]
        new_template = np.vstack((image_overhang, overlap, template_overhang))
    elif temrow > shift_index >= imrow:
        template_overhang_above = template[:shift_index - imrow + 1, :]
        template_overhang_below = template[shift_index + 1:, :]
        new_template = np.vstack((template_overhang_above, overlap, template_overhang_below))
    elif imrow > shift_index >= temrow:
        image_overhang_above = image[:(imrow - 1) - shift_index]
        image_overhang_below = image[(imrow - 1) - shift_index + temrow:]
        new_template = np.vstack((image_overhang_above, overlap, image_overhang_below))
    elif shift_index >= temrow and shift_index >= imrow:
        image_overhang = image[len(spatial_shifts) - shift_index:, :]
        template_overhang_above = template[:shift_index - len(image[:, 0]) + 1, :]
        new_template = np.vstack((template_overhang_above, overlap, image_overhang))

    return new_template


def nan_pad_images(psds, csds, shifts, already_padded=False):
    """For plotting purposes, images are padded with nans to match length of template in a way that captures spatial
    shifts
    """

    if not already_padded:
        # given that CSDs are shorter by 2 channels, pad the CSDs with a row of nans on either end.
        csd_pad = np.empty((1, len(csds[0][0, :])))
        csd_pad[:] = np.nan
        for i in range(len(csds)):
            csds[i] = np.vstack((csd_pad, csds[i], csd_pad))
    # create matrix equal to final length of template and fill with nans
    # length
    imrow, imcol = psds[0].shape
    csdrow, csdcol = csds[0].shape
    nan_padded_psds = np.empty((imrow, imcol, len(psds)))
    nan_padded_psds[:] = np.nan
    nan_padded_csds = np.empty((imrow, len(csds[0][0, :]), len(psds)))
    nan_padded_csds[:] = np.nan

    nan_padded_psds[:, :, 0] = psds[0]
    nan_padded_csds[:, :, 0] = csds[0]
    for i in range(len(shifts)):
        shift = shifts[i]
        psd = psds[i + 1]
        csd = csds[i + 1]
        row, col, pen = nan_padded_psds.shape
        if shift < row and shift < len(psd[:, 0]):
            pad = np.empty(((len(psd[:, 0]) - (shift + 1)), imcol, len(psds)))
            pad[:] = np.nan
            csd_pad = np.empty(((len(psd[:, 0]) - (shift + 1)), csdcol, len(psds)))
            csd_pad[:] = np.nan
            nan_padded_psds = np.vstack((pad, nan_padded_psds))
            nan_padded_csds = np.vstack((csd_pad, nan_padded_csds))
            nan_padded_psds[:len(psd[:, 0]), :, i + 1] = psd
            nan_padded_csds[:len(psd[:, 0]), :, i + 1] = csd
        elif shift < row and shift >= len(psd[:, 0]):
            nan_padded_psds[(shift + 1)-len(psd[:, 0]):shift+1, :, i+1] = psd
            nan_padded_csds[(shift + 1)-len(psd[:, 0]):shift+1, :, i+1] = csd
        elif shift >= row and shift < len(psd[:, 0]):
            upper_pad = np.empty(((shift + 1 - imrow), imcol, len(psds)))
            upper_pad[:] = np.nan
            lower_pad = np.empty((len(psd[:, 0]) - (shift + 1), imcol, len(psds)))
            lower_pad[:] = np.nan
            csd_upper_pad = np.empty(((shift + 1 - imrow), csdcol, len(psds)))
            csd_upper_pad[:] = np.nan
            csd_lower_pad = np.empty((len(psd[:, 0]) - (shift + 1), csdcol, len(psds)))
            csd_lower_pad[:] = np.nan
            nan_padded_psds = np.vstack((lower_pad, nan_padded_psds, upper_pad))
            nan_padded_csds = np.vstack((csd_lower_pad, nan_padded_csds, csd_upper_pad))
            nan_padded_psds[:, :, i + 1] = psd
            nan_padded_csds[:, :, i + 1] = csd
        elif shift > row and shift > len(psd[:, 0]):
            pad = np.empty((shift - imrow, imcol, len(psds)))
            pad[:] = np.nan
            csd_pad = np.empty((shift - imrow, csdcol, len(psds)))
            csd_pad[:] = np.nan
            nan_padded_psds = np.vstack((nan_padded_psds, pad))
            nan_padded_csds = np.vstack((nan_padded_csds, csd_pad))

    return nan_padded_psds, nan_padded_csds


def pad_to_template(template, psds, csds, shifts, already_padded=False):
    """
    function to pad a single PSD, CSD based on the template and the shift index
    """
    if not already_padded:
        # given that CSDs are shorter by 2 channels, pad the CSDs with a row of nans on either end.
        csd_pad = np.empty((1, len(csds[0][0, :])))
        csd_pad[:] = np.nan
        for i in range(len(csds)):
            csds[i] = np.vstack((csd_pad, csds[i], csd_pad))
    # create matrix equal to final length of template and fill with nans
    # length
    imrow, imcol = template.shape
    csdrow, csdcol = csds[0].shape
    nan_padded_psds = np.empty((imrow, imcol, len(psds)))
    nan_padded_psds[:] = np.nan
    nan_padded_csds = np.empty((imrow, len(csds[0][0, :]), len(psds)))
    nan_padded_csds[:] = np.nan

    for i in range(len(shifts)):
        shift = shifts[i]
        psd = psds[i]
        csd = csds[i]
        row, col, pen = nan_padded_psds.shape
        if shift < row and shift < len(psd[:, 0]):
            upper_pad = np.empty(((len(psd[:, 0]) - (shift + 1)), imcol, len(psds)))
            upper_pad[:] = np.nan
            lower_pad = False
            csd_pad = np.empty(((len(psd[:, 0]) - (shift + 1)), csdcol, len(psds)))
            csd_pad[:] = np.nan
            nan_padded_psds = np.vstack((upper_pad, nan_padded_psds))
            nan_padded_csds = np.vstack((csd_pad, nan_padded_csds))
            nan_padded_psds[:len(psd[:, 0]), :, i] = psd
            nan_padded_csds[:len(psd[:, 0]), :, i] = csd
        elif shift < row and shift >= len(psd[:, 0]):
            nan_padded_psds[(shift + 1)-len(psd[:, 0]):shift+1, :, i] = psd
            nan_padded_csds[(shift + 1)-len(psd[:, 0]):shift+1, :, i] = csd
            upper_pad = row - (shift + 1)
            lower_pad = ((shift + 1)-len(psd[:, 0]))
        elif shift >= row and shift < len(psd[:, 0]):
            upper_pad = np.empty(((shift + 1 - imrow), imcol, len(psds)))
            upper_pad[:] = np.nan
            lower_pad = np.empty((len(psd[:, 0]) - (shift + 1), imcol, len(psds)))
            lower_pad[:] = np.nan
            csd_upper_pad = np.empty(((shift + 1 - imrow), csdcol, len(psds)))
            csd_upper_pad[:] = np.nan
            csd_lower_pad = np.empty((len(psd[:, 0]) - (shift + 1), csdcol, len(psds)))
            csd_lower_pad[:] = np.nan
            nan_padded_psds = np.vstack((lower_pad, nan_padded_psds, upper_pad))
            nan_padded_csds = np.vstack((csd_lower_pad, nan_padded_csds, csd_upper_pad))
            nan_padded_psds[:, :, i] = psd
            nan_padded_csds[:, :, i] = csd
        elif shift > row and shift > len(psd[:, 0]):
            lower_pad = np.empty((shift - imrow, imcol, len(psds)))
            lower_pad[:] = np.nan
            upper_pad = False
            csd_pad = np.empty((shift - imrow, csdcol, len(psds)))
            csd_pad[:] = np.nan
            nan_padded_psds = np.vstack((nan_padded_psds, lower_pad))
            nan_padded_csds = np.vstack((nan_padded_csds, csd_pad))
            nan_padded_psds[:, :, i] = psd
            nan_padded_csds[:, :, i] = csd
        padding = [lower_pad, upper_pad]
    return nan_padded_psds, nan_padded_csds, padding