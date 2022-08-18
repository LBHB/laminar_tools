import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

parent_dir = "/auto/users/wingertj/code/csd_project/data/laminar_features/"


# pull in dataframe containing probe relative power
relative_power = pd.read_csv((parent_dir + 'relative_power.csv'))

# get list of all unique penetrations
sites = relative_power.siteid.unique()

# drop unused columns
left_df = relative_power[relative_power['column'] == 'left'].drop(['Unnamed: 0', 'Unnamed: 0.1', 'channel index',
                                                                   'column'], axis=1)
# treat the image shifting problem like convolution?
# define template for each site
for temp_site in sites:
    template = left_df[left_df['siteid'] == temp_site].loc[:, ~left_df.columns.isin(['siteid'])].to_numpy()
    other_sites = np.delete(sites, np.where(sites == temp_site), axis=0)
    for site in other_sites[1:]:
        print(site)
        image = left_df[left_df['siteid'] == site].loc[:, ~left_df.columns.isin(['siteid'])].to_numpy()
        imrow, imcol = image.shape

        # total number of shifts should be same as convolution M+N-1
        spatial_shifts = range(len(template[:, 0]) + len(image[:, 0]) - 1)

        # initialize matrices to store shifted images
        shifted_images = np.empty((imrow, imcol, len(spatial_shifts)))
        shifted_images[:] = np.nan
        shifted_templates = np.empty((imrow, imcol, len(spatial_shifts)))
        shifted_templates[:] = np.nan
        # for each spatial shift slice the image and the template and store shifts
        for i in spatial_shifts:
            if i < len(image[:, 0]):
                shift_image = image[(len(image[:, 0]) - 1) - i:, :]
                shift_template = template[:1+i, :]
            elif len(template[:, 0]) > i >= len(image[:, 0]):
                shift_image = image
                shift_template = template[i-len(image[:, 0])+1:i+1, :]
            elif i >= len(template):
                shift_image = image[:len(spatial_shifts) - i, :]
                shift_template = template[i-len(image[:, 0])+1:i+1, :]
            shifted_images[:len(shift_image[:, 0]), :, i] = shift_image
            shifted_templates[:len(shift_template[:, 0]), :, i] = shift_template

        # slice off sides of shifted matrices where the image is less than half overlapped - switched to 12 channels of overlap

        shifted_images_short = shifted_images[:, :, 10:-10]
        shifted_templates_short = shifted_templates[:, :, 10:-10]

        # compute image similarity between each of the shifts
        shift_similarity = []
        for i in range(len(shifted_templates_short[0, 0, :])):
            shifted_image = shifted_images_short[:, :, i]
            shifted_image = shifted_image[~np.isnan(shifted_image).any(axis=1)]
            shifted_template = shifted_templates_short[:, :, i]
            shifted_template = shifted_template[~np.isnan(shifted_template).any(axis=1)]
            shift_similarity.append(ssim(shifted_image, shifted_template, win_size=5))
        plt.show()

        # generate new template by averaging template and image together at spatial shift with max similarity
        if max(shift_similarity) > 0.75:
            max_similarity_index = shift_similarity.index(max(shift_similarity))
            shifted_image = shifted_images_short[:, :, max_similarity_index]
            shifted_image = shifted_image[~np.isnan(shifted_image).any(axis=1)]
            shifted_template = shifted_templates_short[:, :, max_similarity_index]
            shifted_template = shifted_template[~np.isnan(shifted_template).any(axis=1)]
            template_overlap = (shifted_template + shifted_image) / 2

            # using the maximum shift index, use the inverse of the slice operation used to shift the images to find non-overlapping
            # segments to append to the template to extend its bounds

            shift_index = max_similarity_index + int(len(image[:, 0])/2)
            if shift_index < len(image[:, 0]):
                image_overhang = image[:(len(image[:, 0]) - 1) - shift_index, :]
                template_overhang = template[1+shift_index:, :]
                template = np.vstack((image_overhang, template_overlap, template_overhang))
            elif len(template[:, 0]) > shift_index >= len(image[:, 0]):
                template_overhang_above = template[:shift_index-len(image[:, 0])+1, :]
                template_overhang_below = template[shift_index+1:, :]
                template = np.vstack((template_overhang_above, template_overlap, template_overhang_below))
            elif shift_index >= len(template):
                image_overhang = image[len(spatial_shifts) - shift_index:, :]
                template_overhang_above = template[:shift_index-len(image[:, 0])+1, :]
                template = np.vstack((template_overhang_above, template_overlap, image_overhang))

            # fig, ax = plt.subplots(3, 2)
            # ax[0, 0].imshow(template, origin='lower', aspect='auto', clim=[0, 1])
            # ax[0, 0].set_title("template")
            # ax[0, 1].imshow(image, origin='lower', aspect='auto', clim=[0, 1])
            # ax[0, 1].set_title("site image")
            # ax[1, 0].imshow(shifted_template, origin='lower', aspect='auto', clim=[0, 1])
            # ax[1, 0].set_title("template max similarity")
            # ax[1, 1].imshow(shifted_image, origin='lower', aspect='auto', clim=[0, 1])
            # ax[1, 1].set_title("site image max similarity")
            # ax[2, 0].imshow(template, origin='lower', aspect='auto', clim=[0, 1])
            # ax[2, 0].set_title("new template")
            # fig.suptitle(temp_site)
            # plt.show()
        else:
            continue
    fig, ax = plt.subplots()
    ax.imshow(template, origin='lower', aspect='auto', clim=[0, 1])
    ax.set_title(temp_site)
    plt.show()