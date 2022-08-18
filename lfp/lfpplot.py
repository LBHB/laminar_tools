import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def laminar_plot(label, csd, relative_power, gamma_cohmat, freqs, fs, beta_range=[10, 30], gamma_range=[50, 250],
                 guess=True):

    num_channels, num_freqs = relative_power.shape
    # L2/3 to 4 transition point may be identified by gamma to beta power ratio
    relative_gamma = relative_power[:, (freqs > gamma_range[0]) & (freqs < gamma_range[1])].mean(axis=1)
    relative_beta = relative_power[:, (freqs > beta_range[0]) & (freqs < beta_range[1])].mean(axis=1)
    gamma_beta_ratio = relative_gamma / relative_beta
    gamma_over_beta = gamma_beta_ratio > 1
    below_gamma_peak = gamma_over_beta[:np.where(relative_gamma == relative_gamma.max())[0][0]]
    try:
        first_crossing_below_gamma_peak = np.argwhere(below_gamma_peak)[0]
    except:
        print('no beta power over gamma peak detected...not deep enough?...setting to lowest channel')
        first_crossing_below_gamma_peak = 0

    # for gamma coherence matrix the second eigenvector seems to capture the maximal disimilarity
    # take 0 crossing point?
    w, v = np.linalg.eig(gamma_cohmat)
    transition = np.where(np.diff(np.signbit(v[:, 1])))[0]

    cmax = abs(csd).max()
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=[12, 3], gridspec_kw={'width_ratios': [2, 3, 1, 3, 1]},
                                                  sharey=True)
    fig.subplots_adjust(wspace=0.4)
    im1 = ax1.imshow(csd, origin='lower', aspect='auto', extent=[-(len(csd[0, :])/2)/fs, (len(csd[0, :])/2)/fs, 1,
                                                                 (num_channels-2)], clim=[-cmax, cmax])
    ax1_divider = make_axes_locatable(ax1)
    cbar_ax1 = ax1_divider.append_axes("right", size="7%", pad="30%")
    cbar_ax1.axis('off')
    cbar = fig.colorbar(im1, ax=cbar_ax1, ticks=[-cmax, cmax])
    cbar.ax.set_yticklabels(['sink', 'source'])
    ax1.set_ylabel('channel')
    ax1.set_xlabel('time (s)')
    ax1.set_yticks(np.arange(0, num_channels, 5))
    ax1.set_title('CSD')
    im2 = ax2.imshow(relative_power, aspect='auto', origin='lower', extent=[0, int(max(freqs)), 0, (num_channels-1)])
    ax2_divider = make_axes_locatable(ax2)
    cbar_ax2 = ax2_divider.append_axes("right", size="7%", pad="30%")
    cbar_ax2.axis('off')
    cbar1 = fig.colorbar(im2, ax=cbar_ax2)
    if guess:
        ax2.axhline(y=first_crossing_below_gamma_peak, color='red', label='gamma/beta')
        ax2.legend(fontsize=10)
    ax2.set_xlabel("frequency")
    ax2.set_title('relative lfp power')
    ax3.plot(relative_gamma, np.arange(0, len(relative_gamma)), color='red', label='gamma')
    ax3.plot(relative_beta, np.arange(0, len(relative_beta)), color='blue', label='beta')
    ax3.set_title('gamma/beta ratio')
    ax3.set_xlabel('avg relative power')
    ax3.legend(fontsize=10)
    ax4.imshow(gamma_cohmat, origin='lower', aspect='auto')
    if guess:
        ax4.axhline(y=transition[0], color='red', label='lowest coherence')
    ax4.set_xlabel('channel')
    # ax4.set_xticks(np.arange(num_channels))
    ax4.set_title('gamma coherence matrix')
    ax4.set_aspect('equal', adjustable='box')
    ax5.plot(v[:, 1], np.arange(0, len(v[:, 1])), color='red')
    ax5.set_xlabel('PC2')
    ax5.set_title('PC2 coherence matrix')
    fig.suptitle(label)
    plt.show()
