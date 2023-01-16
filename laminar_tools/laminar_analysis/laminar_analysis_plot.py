import matplotlib.pyplot as plt
import numpy as np


def ssim_step_plot(penetration_sites, template_steps, csd_steps, step_index, ssim_scores, fs):
    avg_template = np.nanmean(template_steps, axis=2)
    avg_csd = np.nanmean(csd_steps, axis=2)
    max_power = avg_template.max(axis=0)

    # make psd relative
    relative_power_template = avg_template/max_power

    fig, axs = plt.subplots(2, len(template_steps[0, 0, :]) + 1, sharey=True, figsize =(20, 8), constrained_layout=True)
    axs[0, 0].imshow(template_steps[:, :, 0]/max_power, origin='lower', aspect='auto', extent=[0, 250, 0, len(relative_power_template[:, 0])])
    axs[1, 0].imshow(csd_steps[:, :, 0], origin='lower', aspect='auto', extent=[-(len(csd_steps[0, :, 0])/2)/fs, (len(csd_steps[0, :, 0])/2)/fs, 0, len(avg_csd[:, 0])])
    for i in range(len(step_index)):
        axs[0, i+1].imshow(template_steps[:, :, i+1]/max_power, origin='lower', aspect='auto', extent=[0, 250, 0, len(relative_power_template[:, 0])])
        axs[1, i+1].imshow(csd_steps[:, :, i+1], origin='lower', aspect='auto', extent=[-(len(csd_steps[0, :, 0])/2)/fs, (len(csd_steps[0, :, 0])/2)/fs, 0, len(avg_csd[:, 0])])
        axs[0, i+1].set_title(penetration_sites[i + 1] + "\n score: " + str(round(ssim_scores[i], 2)))
        axs[0, i+1].set_xlabel('frequency')
    axs[0, -1].imshow(relative_power_template, origin='lower', aspect='auto', extent=[0, 250, 0, len(relative_power_template[:, 0])])
    axs[1, -1].imshow(avg_csd, origin='lower',aspect='auto', extent=[-(len(csd_steps[0, :, 0])/2)/fs, (len(csd_steps[0, :, 0])/2)/fs, 0, len(avg_csd[:, 0])])
    axs[0, 0].set_title(penetration_sites[0])
    axs[0, 0].set_xlabel('frequency')
    axs[0, 0].set_ylabel('depth in channels')
    axs[1, 0].set_xlabel('time (s)')
    axs[1, 0].set_ylabel('depth in channels')
    axs[0, -1].set_title('Overlapped \n relative \n power')
    axs[0, -1].set_xlabel('frequency')
    axs[1, -1].set_title('Overlapped \n CSD')
    axs[1, -1].set_xlabel('time (s)')
    return fig

def old_ssim_step_plot(penetration_sites, template_steps, step_index, ssim_scores, csd_steps, fs):

    # create relative max power base on max values of final template
    max_power = template_steps[-1].max(axis=0)

    # generate figure for plotting with variable length

    fig, axs = plt.subplots(2, len(template_steps), sharey=True, figsize =(20, 8), constrained_layout=True)

    # try to waterfall plot for each power_spectrum and csd
    psd_lower_extent = 0
    psd_upper_extent = len(template_steps[0][:, 0])
    csd_lower_extent = 0
    csd_upper_extent = len(csd_steps[0][:, 0])

    axs[0, 0].imshow(template_steps[0]/max_power, origin='lower', aspect='auto', extent=[0, 250, psd_lower_extent, psd_upper_extent])
    axs[0, 0].set_title(penetration_sites[0])
    axs[0, 0].set_xlabel('frequency')
    axs[0, 0].set_ylabel('depth in channels')
    axs[1, 0].imshow(csd_steps[0], origin='lower', aspect='auto', extent=[-(len(csd_steps[0][0, :])/2)/fs, (len(csd_steps[0][0, :])/2)/fs, csd_lower_extent, csd_upper_extent])
    axs[1, 0].set_xlabel('time (s)')
    axs[1, 0].set_ylabel('depth in channels')
    for i in range(len(step_index)):
        psd_upper_extent = psd_lower_extent + step_index[i]
        psd_lower_extent = psd_lower_extent-(len(template_steps[i + 1][:, 0]) - step_index[i])
        csd_upper_extent = csd_lower_extent + step_index[i]
        csd_lower_extent = csd_lower_extent - (len(csd_steps[i + 1][:, 0]) - step_index[i])
        axs[0, i + 1].imshow(template_steps[i + 1]/max_power, origin='lower', aspect='auto',
                         extent=[0, 250, psd_lower_extent, psd_upper_extent])
        axs[0, i + 1].set_title(penetration_sites[i + 1] + "\n score: " + str(round(ssim_scores[i], 2)))
        axs[0, i + 1].set_xlabel('frequency')
        axs[1, i + 1].imshow(csd_steps[i + 1], origin='lower', aspect='auto',
                         extent=[-(len(csd_steps[0][0, :])/2)/fs, (len(csd_steps[0][0, :])/2)/fs, csd_lower_extent, csd_upper_extent])
        axs[1, i + 1].set_title(penetration_sites[i + 1] + "\n CSD")
        axs[1, i + 1].set_xlabel('time (s)')
    axs[0, -1].imshow(template_steps[-1]/max_power, origin='lower', aspect='auto',
                      extent=[0, 250, psd_lower_extent, len(template_steps[0][:, 0])])
    axs[0, -1].set_title('Overlapped \n relative \n power')
    axs[0, -1].set_xlabel('frequency')
    axs[1, -1].imshow(csd_steps[-1], origin='lower', aspect='auto',
                      extent=[-(len(csd_steps[0][0, :])/2)/fs, (len(csd_steps[0][0, :])/2)/fs, csd_lower_extent, len(csd_steps[0][:, 0])])
    axs[1, -1].set_title('Overlapped \n CSD')
    axs[1, -1].set_xlabel('time (s)')
    axs[0, 0].set_ylim(psd_lower_extent, len(template_steps[0][:, 0]))
    axs[1, 0].set_ylim(psd_lower_extent, len(template_steps[0][:, 0]))

    return fig