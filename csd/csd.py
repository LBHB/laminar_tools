from scipy.ndimage import maximum_filter, binary_erosion
import numpy as np
from scipy.signal import convolve2d

# def csd1d(lfp, ch_spacing, filter_width):
#     """
#     Simple csd calculation in one dimension.
#     :param lfp: channels x time
#     :param ch_spacing: contact spacing
#     :return: csd
#     """
#
#     # spatial fiter data to remove electrode to electrode jitter
#     spatial_filter = np.hamming(filter_width)[:, np.newaxis]
#     spatial_filter = spatial_filter / spatial_filter.sum()
#     spatialfilt_lfp = convolve2d(lfp, spatial_filter, mode='same', boundary='symm')
#
#     csd = np.zeros_like(spatialfilt_lfp[:-2, :])
#     for ch in range(0, int(len(spatialfilt_lfp[:-2, :]))):
#         v0 = spatialfilt_lfp[ch + 1, :]
#         va = spatialfilt_lfp[ch, :]
#         vb = spatialfilt_lfp[ch + 2, :]
#         csd[ch, :] = (vb + va - 2 * v0) / (2*ch_spacing) ** 2
#     return csd

def csd1d(lfp, filter_width):
    # spatial fiter data to remove electrode to electrode jitter
    spatial_filter = np.hamming(filter_width)[:, np.newaxis]
    spatial_filter = spatial_filter / spatial_filter.sum()
    spatialfilt_lfp = convolve2d(lfp, spatial_filter, mode='same', boundary='symm')

    csd = np.diff(spatialfilt_lfp, n=2, axis=0)
    return csd

def detect_peaks(image, neighbors=None, threshold=0.5):
    """
    Detects local peaks and troughs in image above a threshold using
    ndimage.maximum_filter and returns values and locations

    :param image: A 2D matrix/image
    :param neighbors: shape of surrounding pixel array to use as the local filter (n*n array)
    :param threshold: (0-1) percentage of maximum peak value to use as threshold
    :return: A list of peak/trough values and their coordinates as well as a mask of all the peaks/troughs in the image
    """

    if neighbors is None:
        neighbors = [3, 3]

    # Take abs of image so peaks and troughs are detected
    image_abs = abs(image)

    # define an n-connected  neighborhood
    neighborhood = np.ones((neighbors[0], neighbors[1]))

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image_abs, footprint=neighborhood) == image_abs
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image_abs == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    # peak coordinates
    peak_loc = np.array(np.where(detected_peaks == 1)).transpose()

    # peak values
    peak_values = []
    for peak in peak_loc:
        peak_value = image[peak[0], peak[1]]
        peak_values.append(peak_value)

    # find max value and set threshold to identify relatively large peaks and create list with coordinates
    max_peak = max(abs(np.array(peak_values)))

    largest_peaks = [(peak_value, peak_loc) for peak_value, peak_loc in zip(peak_values, peak_loc) if
                     abs(peak_value) >= threshold * max_peak]

    return largest_peaks, detected_peaks


def path_area(vs):
    """
    returns the area contained within a path of vertices - returned by contour function
    :param vs: vertices of the path
    :return: area
    """
    a = 0
    x0, y0 = vs[0]
    for [x1, y1] in vs[1:]:
        dx = x1 - x0
        dy = y1 - y0
        a += 0.5 * (y0 * dx - x0 * dy)
        x0 = x1
        y0 = y1
    return a


def contour_areas(path_obj):
    """
    Finds area of all individual paths from a path object.
    :param path_obj: returned by contour function
    :return: list of areas for each path
    """
    areas = []
    for i in range(len(path_obj.collections)):
        vs = path_obj.collections[i].get_paths()
        for island in vs:
            ai = path_area(island.vertices)
            ai = np.abs(ai)
            areas.append(ai)
    return areas

# def contours_larger_than(): remove areas for contour lines that have extremely small diameters