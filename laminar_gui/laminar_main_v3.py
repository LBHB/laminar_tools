import sys
import warnings
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.lines as lines
from PyQt5.QtWidgets import QApplication, QWidget, QTreeWidgetItem
from nems0 import db
from nems_lbhb.baphy_experiment import BAPHYExperiment
from laminar_tools.lfp.lfp import parmfile_event_lfp
from laminar_tools.laminar_analysis.laminar_analysis import maximal_laminar_similarity, pad_to_template
from laminar_gui_v3 import Ui_mainWidget
from functools import partial
from pathlib import Path
import json
from scipy.interpolate import interp1d

class LaminarUi(QWidget):
    def __init__(self, *args, **kwargs):
        super(LaminarUi, self).__init__(*args, **kwargs)
        self.ui = Ui_mainWidget()
        self.ui.setupUi(self)
        self.ui.siteList.setSelectionMode(self.ui.siteList.MultiSelection)
        self.show()

class LaminarModel():
    def __init__(self, view):
        self._view = view
        self.load_template()
        self.template_landmarks = list(self._view.ui.layerBorders.keys())
        self.template_landmarkPosition = {'BS/1': 28, '3/4': 18, '4/5': 12, '6/WM': 5, 'WM/HC': 0}
        self.template_lines = {}
        self.landmarks = self.template_landmarks
        self.landmarkPosition = {border: self.template_landmarkPosition[border] for border in self.landmarks}
        self.landmarkBoolean = {border: False for border in self.landmarks}
        self.lines = list()
        self.linedict = {}
        self.template_landmarkBoolean = True
        self.template_lines = {}
        self.depth_mapped = {}

    def animals(self, active):
        # get animals:
        species = 'ferret'
        require_active = active
        sql = f"SELECT * FROM gAnimal WHERE lab='lbhb' and species='{species}'"
        if require_active:
            sql += " AND onschedule<2"
        dAnimal = db.pd_query(sql)
        self.animallist = dAnimal['animal'].to_list()

    def sites(self, animal):
        runclass = None
        if runclass is None:
            sql = "SELECT DISTINCT gCellMaster.* FROM gCellMaster INNER JOIN gDataRaw ON gCellMaster.id=gDataRaw.masterid" + \
                  f" WHERE animal='{animal}' AND not(gDataRaw.bad) AND gCellMaster.training=0 ORDER BY gCellMaster.siteid"
        else:
            sql = "SELECT DISTINCT gCellMaster.* FROM gCellMaster INNER JOIN gDataRaw ON gCellMaster.id=gDataRaw.masterid" + \
                  f" WHERE animal='{animal}' AND not(gDataRaw.bad) AND gDataRaw.runclass='{runclass}' AND gCellMaster.training=0 ORDER BY gCellMaster.siteid"

        dSites = db.pd_query(sql)
        #site_list = dSites['siteid'].to_list()
        site_list = dSites['penname'].to_list()
        site_list = list(set(site_list))
        site_list.sort()
        self.sitelist = site_list

    def parmfiles(self, siteid):
        sql = f"SELECT gDataRaw.* FROM gDataRaw WHERE cellid like '{siteid}%%' and bad=0 and training = 0"
        dRawFiles = db.pd_query(sql)

        # get parmfiles for siteid
        parmfiles = dRawFiles['parmfile'].to_list()
        self.parmfilelist = parmfiles

        # get rawids for siteid
        self.rawids = dRawFiles['id'].to_list()

        # check if channel mapping in database
        self.dbcheck = ['Yes' if i is not None else 'No' for i in dRawFiles['depthinfo']]

    def load_template(self):
        template_psd = np.load("/auto/users/wingertj/code/csd_project/data/laminar_features/template/final_psd_template_v2.npy")
        template_csd = np.load("/auto/users/wingertj/code/csd_project/data/laminar_features/template/final_csd_template_v2.npy")
        max_power = template_psd.max(axis=0)
        self.temp_max_power = max_power
        self.template_psd = template_psd
        self.template_csd = template_csd

    def site_csd_psd(self, parmfile, align=True):
        csd, psd, freqs, stim_window, rasterfs, column_xy, channel_xy, coh_mat, erp = parmfile_event_lfp(parmfile)
        max_power = psd.max(axis=0)
        if align:
            averaged_template, ssim_index, ssim = maximal_laminar_similarity(template=self.template_psd, image=psd, overlap=10,
                                                                             ssim_window=5, expansion=True)
            csd = [csd, ]
            psd = [psd, ]
            ssim_index_list = [ssim_index, ]
            self.ssim = ssim
            self.ssim_index = ssim_index
            nan_pad_psd, nan_pad_csd, padding = pad_to_template(self.template_psd, psd, csd, ssim_index_list, already_padded=True)
            self.padding = padding
            upper_pad = np.empty((len(coh_mat[:, 0, 0]), padding[1], len(coh_mat[0,:])))
            upper_pad[:] = np.nan
            lower_pad = np.empty((len(coh_mat[:, 0, 0]), padding[0], len(coh_mat[0,:])))
            lower_pad[:] = np.nan
            coh_mat = np.concatenate((lower_pad, coh_mat, upper_pad), axis=1)
            psd = np.squeeze(nan_pad_psd)
            csd = np.squeeze(nan_pad_csd)

        self.erp = erp
        self.coh = coh_mat
        self.freqs = freqs
        self.rasterfs = rasterfs
        self.window = stim_window
        self.window_samples = stim_window*rasterfs
        self.psd = psd
        self.site_max_power = max_power
        self.csd = csd
        self.column_xy = column_xy
        self.channel_xy = channel_xy

    def no_normalization(self):
        self.psd_norm = self.psd
        self.template_psd_norm = self.template_psd
        self.unpadded_psd_norm = self.psd_norm[~np.isnan(self.psd_norm)]
        self.cmax = max([self.template_psd_norm.max(), self.unpadded_psd_norm.max()])

    def site_normalization(self):
        self.psd_norm = self.psd/self.site_max_power
        self.template_psd_norm = self.template_psd/self.site_max_power
        self.unpadded_psd_norm = self.psd_norm[~np.isnan(self.psd_norm)]
        self.cmax = max([self.template_psd_norm.max(), self.unpadded_psd_norm.max()])

    def local_normalization(self):
        self.psd_norm = self.psd/self.site_max_power
        self.template_psd_norm = self.template_psd/self.temp_max_power
        self.unpadded_psd_norm = self.psd_norm[~np.isnan(self.psd_norm)]

    def temp_normalization(self):
        self.psd_norm = self.psd/self.temp_max_power
        self.template_psd_norm = self.template_psd/self.temp_max_power
        self.unpadded_psd_norm = self.psd_norm[~np.isnan(self.psd_norm)]
        self.cmax = max([self.template_psd_norm.max(), self.unpadded_psd_norm.max()])

    def erase_lines(self):
        print('errasing lines...')
        while len(self.linedict.keys()) != 0:
            keys = list(self.linedict.keys())
            artists = self.linedict.pop(keys[0])
            for artist in artists:
                artist.remove()
        print('done')

    def reset_line(self, canvas, line):
        self.landmarkPosition[line] = self.template_landmarkPosition[line]
        self.draw_lines(canvas.ax)

    def reset_lines(self, canvas):
        self.landmarkPosition = {border: self.template_landmarkPosition[border] for border in self.landmarks}
        self.draw_lines(canvas.ax)

    def draw_lines(self, ax):
        self.erase_lines()
        print('drawing lines...')
        for sName in self.landmarks:
            try:
                sBool = self.landmarkBoolean[sName]
                sPos = self.landmarkPosition[sName]
                if sBool:
                    top, bottom = sName.split('/')
                    self.linedict[sName] = [ax.axhline(sPos, color='red', linewidth=2, picker=5),
                                            ax.text(0, sPos + 0.5, top, color='orange', fontsize=10),
                                            ax.text(0, sPos - 2, bottom, color='orange', fontsize=10)]
            except:
                continue
        print('done')

    def erase_template_lines(self):
        while len(self.template_lines.keys()) != 0:
            keys = list(self.template_lines.keys())
            artists = self.template_lines.pop(keys[0])
            for artist in artists:
                artist.remove()

    def template_draw_lines(self, ax):
        self.erase_template_lines()
        for sName in self.template_landmarks:
            if self.template_landmarkBoolean:
                top, bottom = sName.split('/')
                sPos = self.template_landmarkPosition[sName]
                self.template_lines[sName] = [ax.axhline(sPos, color='red', linewidth=2),
                                            ax.text(0, sPos + 0.5, top, color='orange', fontsize=10),
                                            ax.text(0, sPos - 2, bottom, color='orange', fontsize=10)]

    def template_plot(self, canvas, temp_psd):
        """
        plots the laminar data
        """
        print('plotting laminar data...')
        self.clear_canvas(canvas)
        if self._view.ui.localnormradioButton.isChecked():
            self.cmax = self.template_psd_norm.max()
        if temp_psd:
            im = canvas.ax.imshow(self.template_psd_norm, origin='lower', aspect='auto', clim=[0, self.cmax])
            canvas.ax.set_xlim(self.freqs[0], self.freqs[-1])
            canvas.ax.set_xlabel("frequency")
            canvas.fig.colorbar(im, cax=canvas.cax)
        else:
            self.template_csd_sitematched = self.template_csd[:, int(len(self.template_csd[0,:])/2 - self.window_samples):int(len(self.template_csd[0,:])/2 + self.window_samples)]
            canvas.ax.imshow(self.template_csd_sitematched, origin='lower', aspect='auto')
            x_ticks = np.linspace(0, len(self.template_csd_sitematched[0, :]), 5)
            x_ticklabels = np.round(np.linspace(-self.window, self.window, 5), decimals=2)
            canvas.ax.set_xticks(x_ticks)
            canvas.ax.set_xticklabels(x_ticklabels)
            canvas.ax.set_xlabel("time (s)")
        self.template_draw_lines(canvas.ax)

    def site_plot(self, canvas, site_psd, site_csd, site_coh, site_erp):
        """
        plots the laminar data
        """
        print('plotting laminar data...')
        self.clear_canvas(canvas)
        if self._view.ui.localnormradioButton.isChecked():
            self.cmax = self.unpadded_psd_norm.max()
        if site_psd:
            im = canvas.ax.imshow(self.psd_norm, origin='lower', aspect='auto', clim=[0, self.cmax])
            canvas.ax.set_xlim(self.freqs[0], self.freqs[-1])
            canvas.ax.set_xlabel("frequency")
            canvas.fig.colorbar(im, cax=canvas.cax)
            y_ticks = np.arange(0, len(self.psd_norm[:, 0]), 8)
            canvas.ax.set_yticks(y_ticks)
            if len(self.channel_xy) > 64:
                y_tick_channels = np.take(self.column_xy, y_ticks, axis=0)
                canvas.ax.set_yticklabels(["ch" + str(i) + '\n' + str(self.channel_xy[i][1]) + 'um' for i in y_tick_channels], fontsize=6)

        elif site_csd:
            canvas.ax.imshow(self.csd, origin='lower', aspect='auto')
            x_ticks = np.linspace(0, len(self.csd[0, :]), 5)
            x_ticklabels = np.round(np.linspace(-self.window, self.window, 5), decimals=2)
            canvas.ax.set_xticks(x_ticks)
            canvas.ax.set_xticklabels(x_ticklabels)
            canvas.ax.set_xlabel("time (s)")
            y_ticks = np.arange(0, len(self.psd_norm[:, 0]), 8)
            canvas.ax.set_yticks(y_ticks)
            if len(self.channel_xy) > 64:
                y_tick_channels = np.take(self.column_xy, y_ticks, axis=0)
                canvas.ax.set_yticklabels(["ch" + str(i) + '\n' + str(self.channel_xy[i][1]) + 'um' for i in y_tick_channels], fontsize=6)

        elif site_coh:
            # idx30 = np.where(self.freqs > 30)[0][0]
            # idx150 = np.where(self.freqs < 150)[0][-1]
            idx1 = np.where(self.freqs > 1)[0][0]
            idx15 = np.where(self.freqs < 15)[0][-1]
            gamma_cohmat = np.squeeze(self.coh.mean(axis=0))
            canvas.ax.imshow(gamma_cohmat, origin='lower', aspect='auto')
            y_ticks = np.arange(0, len(self.psd_norm[:, 0]), 8)
            canvas.ax.set_yticks(y_ticks)
            if len(self.channel_xy) > 64:
                y_tick_channels = np.take(self.column_xy, y_ticks, axis=0)
                canvas.ax.set_yticklabels(["ch" + str(i) + '\n' + str(self.channel_xy[i][1]) + 'um' for i in y_tick_channels], fontsize=6)
        elif site_erp:
            for i in range(len(self.erp[:, 0])):
                canvas.ax.plot((self.erp[i, :] + 500*i))
        canvas.ax.set_ylabel("channels")
        canvas.ax.set_title(str(self.parmfile))
        self.draw_lines(canvas.ax)

    def clear_canvas(self, canvas):
        canvas.ax.clear()
        canvas.cax.clear()

    # def depth_mapping(self):
    #     print('mapping to nominal depths')
    #     # take padding into consideration if matched to template
    #
    #     if self.padding:
    #         lower_pad_size = self.padding[0]
    #     else:
    #         lower_pad_size = 0
    #
    #     # hard coded depths based on average cortical thickness of 1.5mm and use in prior literature
    #     BS1_depth = -800
    #     L34_depth = 0
    #     L45_depth = 200
    #     L6WM_depth = 800
    #
    #     temp_landmark_dict = {k:v-lower_pad_size for (k, v) in self.landmarkPosition.items()}
    #
    #     # Marker memory
    #     position_memory = {'landmarkBoolean':self.landmarkBoolean, 'landmarkPosition':temp_landmark_dict}
    #
    #     # Markers
    #     BS1_position = int(self.channel_xy[self.column_xy[(int(round(self.landmarkPosition['BS/1']) + 1))]][1])
    #     L34_position = int(self.channel_xy[self.column_xy[(int(round(self.landmarkPosition['3/4']) + 1))]][1])
    #     L45_position = int(self.channel_xy[self.column_xy[(int(round(self.landmarkPosition['4/5']) + 1))]][1])
    #     L6WM_position = int(self.channel_xy[self.column_xy[(int(round(self.landmarkPosition['6/WM']) + 1))]][1])
    #
    #     # Above BS remapping - extrapolate from BS/L4
    #     y_in = np.array([BS1_position, L34_position])
    #     y_out = np.array([BS1_depth, L34_depth])
    #     f = interp1d(y_in, y_out, fill_value='extrapolate')
    #     # channel depths above BS
    #     ABS_channels = {k:v for (k,v) in self.channel_xy.items() if (int(v[1]) >= BS1_position)}
    #     ABS_keys = list(ABS_channels.keys())
    #     ABS_positions = [int(v[1]) for (k,v) in ABS_channels.items()]
    #     ABS_depths = f(ABS_positions)
    #     ABS_region = ['BS' for i in ABS_depths]
    #     ABS_regvalues = [[reg, depth] for (reg, depth) in zip(ABS_region, ABS_depths)]
    #     ABS_dict = {k:v for (k,v) in zip(ABS_keys, ABS_regvalues)}
    #
    #     # BS/L4 remapping
    #     y_in = np.array([BS1_position, L34_position])
    #     y_out = np.array([BS1_depth, L34_depth])
    #     f = interp1d(y_in, y_out)
    #     # channel depths inbetween BS/L4
    #     BSL4_channels = {k:v for (k,v) in self.channel_xy.items() if (int(v[1]) >= L34_position) and (int(v[1]) <= BS1_position)}
    #     BSL4_keys = list(BSL4_channels.keys())
    #     BSL4_positions = [int(v[1]) for (k,v) in BSL4_channels.items()]
    #     BSL4_depths = f(BSL4_positions)
    #     BSL4_region = ['L1/L3' for i in BSL4_depths]
    #     BSL4_regvalues = [[reg, depth] for (reg, depth) in zip(BSL4_region, BSL4_depths)]
    #     BSL4_dict = {k:v for (k,v) in zip(BSL4_keys, BSL4_regvalues)}
    #
    #     # L34/L56 remapping
    #     y_in = np.array([L34_position, L45_position])
    #     y_out = np.array([L34_depth, L45_depth])
    #     f = interp1d(y_in, y_out)
    #     # channel depths inbetween L3/L5
    #     L3456_channels = {k:v for (k,v) in self.channel_xy.items() if (int(v[1]) >= L45_position) and (int(v[1]) <= L34_position)}
    #     L3456_keys = list(L3456_channels.keys())
    #     L3456_positions = [int(v[1]) for (k,v) in L3456_channels.items()]
    #     L3456_depths = f(L3456_positions)
    #     L3456_region = ['L4' for i in L3456_depths]
    #     L3456_regvalues = [[reg, depth] for (reg, depth) in zip(L3456_region, L3456_depths)]
    #     L3456_dict = {k:v for (k,v) in zip(L3456_keys, L3456_regvalues)}
    #
    #     # L56/L6WM remapping
    #     y_in = np.array([L45_position, L6WM_position])
    #     y_out = np.array([L45_depth, L6WM_depth])
    #     f = interp1d(y_in, y_out)
    #     # channel depths inbetween L4/L5
    #     L456WM_channels = {k:v for (k,v) in self.channel_xy.items() if (int(v[1]) >= L6WM_position) and (int(v[1]) <= L45_position)}
    #     L456WM_keys = list(L456WM_channels.keys())
    #     L456WM_positions = [int(v[1]) for (k,v) in L456WM_channels.items()]
    #     L456WM_depths = f(L456WM_positions)
    #     L456WM_region = ['L5/L6' for i in L456WM_depths]
    #     L456WM_regvalues = [[reg, depth] for (reg, depth) in zip(L456WM_region, L456WM_depths)]
    #     L456WM_dict = {k:v for (k,v) in zip(L456WM_keys, L456WM_regvalues)}
    #
    #     # Below WM remapping
    #     y_in = np.array([L45_position, L6WM_position])
    #     y_out = np.array([L45_depth, L6WM_depth])
    #     f = interp1d(y_in, y_out, fill_value='extrapolate')
    #     # channel depths inbetween L4/L5
    #     BWM_channels = {k:v for (k,v) in self.channel_xy.items() if (int(v[1]) < L6WM_position)}
    #     BWM_keys = list(BWM_channels.keys())
    #     BWM_positions = [int(v[1]) for (k,v) in BWM_channels.items()]
    #     BWM_depths = f(BWM_positions)
    #     BWM_region = ['WM' for i in BWM_depths]
    #     BWM_regvalues = [[reg, depth] for (reg, depth) in zip(BWM_region, BWM_depths)]
    #     BWM_dict = {k:v for (k,v) in zip(BWM_keys, BWM_regvalues)}
    #
    #     self.depth_mapped = {**ABS_dict, **BSL4_dict, **L3456_dict, **L456WM_dict, **BWM_dict, **position_memory}

    def depth_mapping_new(self):
        print('mapping to nominal depths')
        # take padding into consideration if matched to template

        try:
            lower_pad_size = self.padding[0]
        except:
            lower_pad_size = 0

        temp_landmark_dict = {k: v - lower_pad_size for (k, v) in self.landmarkPosition.items()}

        # Marker memory
        position_memory = {'landmarkBoolean': self.landmarkBoolean, 'landmarkPosition': temp_landmark_dict}

        # hard coded depths based on average cortical thickness of 1.5mm and use in prior literature
        # BS1_depth = -800
        # L34_depth = 0
        # L45_depth = 200
        # L6WM_depth = 800

        site_area = self._view.ui.areatext.toPlainText()
        if site_area == '':
            raise ValueError('Please input a site: A1, PEG')
        active_landmarks = [k for (k, v) in self.landmarkBoolean.items() if v == True]
        active_positions = [int(self.channel_xy[self.column_xy[int(round(temp_landmark_dict[lm]))]][1]) for lm in active_landmarks]
        active_assignments = [self._view.ui.layerBorders[lm] for lm in active_landmarks]

        if len(active_landmarks) < 2:
            raise ValueError("Need more than 1 landmark...")
        # sort all channels based on depth - for each channel find closest landmark. If landmark is higher up, then check and see if there is landmark one index lower.
        # If there is a landmark one index lower, create an interp1d mapping between the two landmarks and their assignments. Remap the channel with
        #  with the name being a split of the upper/lower landmark. If there is not a lower index. take the landmark index one higher than closest landmark.
        #  Create an interp1d mapping between the two. Remap the channel with the name being the lower label of the landmark above. In the case of the closest landmark
        # being below the channel. Check and see if there is landmark above the nearest landmark. If there is, interp1d, remap, and name as a split of the two landmarks.
        # if there is not a landmark above the closest landmark. Find the landmark below the closest landmark to the channel. Interp1d, remap, and name as the upper
        # label of the closest landmark.

        # ToDo prevent an abundance of names in the DB, require the channel name to be an existing split of all nearest landmarks.
        # do not allow a landmark to be skipped. BS/1 and 3/4 must be in order. Can not jump directly from BS/1 to 4/5
        # which would suggest no active 3/4 boundary. Forces user to guess if boundary is not apparent.

        channel_dict = {}
        for ch in list(self.channel_xy.keys()):
            # find closest active channel
            channel_position = int(self.channel_xy[ch][1])
            min_index = abs(np.array(active_positions) - channel_position).argmin()
            if channel_position <= active_positions[min_index]:
                try:
                    lower_landmark = active_landmarks[min_index+1]
                    upper_landmark = active_landmarks[min_index]
                    lower_position = active_positions[min_index+1]
                    upper_position = active_positions[min_index]
                    lower_assignment = active_assignments[min_index+1]
                    upper_assignment = active_assignments[min_index]
                    lower_top, lower_bottom = lower_landmark.split('/')
                    upper_top, upper_bottom = upper_landmark.split('/')
                    location_label = ''.join([upper_bottom, lower_top])
                except:
                    lower_landmark = active_landmarks[min_index]
                    upper_landmark = active_landmarks[min_index-1]
                    lower_position = active_positions[min_index]
                    upper_position = active_positions[min_index-1]
                    lower_assignment = active_assignments[min_index]
                    upper_assignment = active_assignments[min_index-1]
                    lower_top, lower_bottom = lower_landmark.split('/')
                    upper_top, upper_bottom = upper_landmark.split('/')
                    location_label = lower_bottom

            elif channel_position > active_positions[min_index]:
                if min_index > 0:
                    lower_landmark = active_landmarks[min_index]
                    upper_landmark = active_landmarks[min_index-1]
                    lower_position = active_positions[min_index-1]
                    upper_position = active_positions[min_index]
                    lower_assignment = active_assignments[min_index-1]
                    upper_assignment = active_assignments[min_index]
                    lower_top, lower_bottom = lower_landmark.split('/')
                    upper_top, upper_bottom = upper_landmark.split('/')
                    location_label = ''.join([upper_bottom, lower_top])
                elif min_index == 0:
                    lower_landmark = active_landmarks[min_index+1]
                    upper_landmark = active_landmarks[min_index]
                    lower_position = active_positions[min_index+1]
                    upper_position = active_positions[min_index]
                    lower_assignment = active_assignments[min_index+1]
                    upper_assignment = active_assignments[min_index]
                    lower_top, lower_bottom = lower_landmark.split('/')
                    upper_top, upper_bottom = upper_landmark.split('/')
                    location_label = upper_top

            y_in = np.array([upper_position, lower_position])
            y_out = np.array([upper_assignment, lower_assignment])
            f = interp1d(y_in, y_out, fill_value='extrapolate')
            channel_dict[ch] = [location_label, int(f(channel_position))]
        complete_dict = {}
        complete_dict['channel info'] = channel_dict
        complete_dict['parmfile'] = self.parmfile
        complete_dict['site area'] = site_area
        self.depth_mapped = {**complete_dict, **position_memory}

    def load_depth_from_db(self):
        # load from database
        sql = f"SELECT * FROM gDataRaw WHERE id={int(self.rawid)}"
        draw = db.pd_query(sql)
        loadedds = json.loads(draw.loc[0,'depthinfo'])
        if self._view.ui.sitealigncheckBox.isChecked():
            lower_pad_size = self.padding[0]
            for landmark in list(loadedds['landmarkPosition'].keys()):
                self.landmarkPosition[landmark] = loadedds['landmarkPosition'][landmark]+lower_pad_size
        else:
            for landmark in list(loadedds['landmarkPosition'].keys()):
                self.landmarkPosition[landmark] = loadedds['landmarkPosition'][landmark]
        self.landmarkBoolean = loadedds['landmarkBoolean']

    def load_area_from_db(self):
        sql = f"SELECT * from gCellMaster WHERE cellid='{self.siteid}'"
        d = db.pd_query(sql)
        try:
            self.area = list(d['area'][0].split(','))[0]
        except:
            self.area = ''
        self._view.ui.areatext.setText(self.area)

        # after spike sorting area info gets propogated ot single cell file info ..
        # sql = "SELECT * from sCellFile WHERE cellid='CLT007a-002-1'"
        # sql = "SELECT * from gSingleCell WHERE cellid='CLT007a-002-1'"

class LaminarCtrl():
    def __init__(self, model, view):
        self._view = view
        self._model = model
        self.updateanimalcomboBox(active=True)
        self._connectSignals()

    def assign_database(self):
        self._model.depth_mapping_new()
        self._model.depth_mapped
        site_info = self._view.ui.siteList.selectedItems()
        if site_info:
            for baseNode in site_info:
                # baseNode = site_info[0]
                getChildNode = baseNode.text(1)
                rawid = int(getChildNode)
                depthstring = json.dumps(self._model.depth_mapped)
                print("updating database...")
                sql = f"UPDATE gDataRaw set depthinfo='{depthstring}' WHERE id={rawid}"
                sql
                db.sql_command(sql)
        self.update_siteList(self._view.ui.sitecomboBox.currentIndex())

    def updateanimalcomboBox(self, active):
        self._view.ui.animalcomboBox.clear()
        self._model.animals(active)
        self._view.ui.animalcomboBox.addItems(
            self._model.animallist)

    def update_siteComboBox(self, index):
        self._view.ui.sitecomboBox.clear()
        animal = self._view.ui.animalcomboBox.itemText(index)
        if animal:
            self._model.sites(animal)
            self._view.ui.sitecomboBox.addItems(self._model.sitelist)

    def update_siteList(self, index):
        self._view.ui.siteList.blockSignals(True)
        self._view.ui.siteList.setSortingEnabled(False)
        self._view.ui.siteList.clear()
        tree_items = list()

        site = self._view.ui.sitecomboBox.itemText(index)
        if site:
            self._model.parmfiles(site)
            files = self._model.parmfilelist
            rawids = self._model.rawids
            dbcheck = self._model.dbcheck
            for i in range(len(files)):
                item = QTreeWidgetItem(None)
                item.setText(0, files[i])
                item.setText(1, str(rawids[i]))
                item.setText(2, dbcheck[i])
                tree_items.append(item)

        self._view.ui.siteList.insertTopLevelItems(0, tree_items)
        self._view.ui.siteList.setSortingEnabled(True)
        self._view.ui.siteList.blockSignals(False)

    def normalization(self):
        if self._view.ui.tempmaxnormradioButton.isChecked():
            self._model.temp_normalization()
            self._model.template_plot(self._view.ui.templateCanvas.canvas, self._view.ui.tempPSDradioButton.isChecked())
            self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked(),
                                  self._view.ui.siteCSDradioButton.isChecked(),
                                  self._view.ui.siteCOHradioButton.isChecked(),
                                  self._view.ui.siteERPradioButton.isChecked())
            self._view.ui.templateCanvas.canvas.draw()
            self._view.ui.siteCanvas.canvas.draw()
        elif self._view.ui.sitemaxnormradioButton.isChecked():
            self._model.site_normalization()
            self._model.template_plot(self._view.ui.templateCanvas.canvas, self._view.ui.tempPSDradioButton.isChecked())
            self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked(),
                                  self._view.ui.siteCSDradioButton.isChecked(),
                                  self._view.ui.siteCOHradioButton.isChecked(),
                                  self._view.ui.siteERPradioButton.isChecked())
            self._view.ui.templateCanvas.canvas.draw()
            self._view.ui.siteCanvas.canvas.draw()
        elif self._view.ui.localnormradioButton.isChecked():
            self._model.local_normalization()
            self._model.template_plot(self._view.ui.templateCanvas.canvas, self._view.ui.tempPSDradioButton.isChecked())
            self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked(),
                                  self._view.ui.siteCSDradioButton.isChecked(),
                                  self._view.ui.siteCOHradioButton.isChecked(),
                                  self._view.ui.siteERPradioButton.isChecked())
            self._view.ui.templateCanvas.canvas.draw()
            self._view.ui.siteCanvas.canvas.draw()

        elif self._view.ui.nonormradioButton.isChecked():
            self._model.no_normalization()
            self._model.template_plot(self._view.ui.templateCanvas.canvas, self._view.ui.tempPSDradioButton.isChecked())
            self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked(),
                                  self._view.ui.siteCSDradioButton.isChecked(),
                                  self._view.ui.siteCOHradioButton.isChecked(),
                                  self._view.ui.siteERPradioButton.isChecked())
            self._view.ui.templateCanvas.canvas.draw()
            self._view.ui.siteCanvas.canvas.draw()

    def load_site_csd_psd(self):
        align = self._view.ui.sitealigncheckBox.isChecked()
        getSelected = self._view.ui.siteList.selectedItems()
        if getSelected:
            baseNode = getSelected[0]
            getChildNode = baseNode.text(0)
            self._model.parmfile = str(getChildNode)
            rawid = baseNode.text(1)
            db_available = baseNode.text(2)
            self._model.rawid = rawid
        self._model.siteid = self._view.ui.sitecomboBox.currentText()
        self._model.load_area_from_db()
        animalid = self._view.ui.animalcomboBox.currentText()
        rawpath = Path('/auto/data/daq')
        sql = f"SELECT gDataRaw.* FROM gDataRaw WHERE cellid like '{self._model.siteid}%%' and bad=0 and training = 0"
        dRawFiles = db.pd_query(sql)
        resppath = dRawFiles['resppath'][0]
        parmfilepath = [rawpath/animalid/self._model.siteid/self._model.parmfile]
        self._model.site_csd_psd(parmfilepath, align=align)
        if db_available != 'No':
            self._model.load_depth_from_db()
        self.normalization()
        self._model.template_plot(self._view.ui.templateCanvas.canvas, self._view.ui.tempPSDradioButton.isChecked())
        self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked(),
                              self._view.ui.siteCSDradioButton.isChecked(), self._view.ui.siteCOHradioButton.isChecked(),
                              self._view.ui.siteERPradioButton.isChecked())
        self._view.ui.templateCanvas.canvas.draw()
        self._view.ui.siteCanvas.canvas.draw()

    def update_plots(self):
        template_psd_requested = self._view.ui.tempPSDradioButton.isChecked()
        site_psd_requested = self._view.ui.sitePSDradioButton.isChecked()
        site_csd_requested = self._view.ui.siteCSDradioButton.isChecked()
        site_coh_requested = self._view.ui.siteCOHradioButton.isChecked()
        site_erp_requested = self._view.ui.siteERPradioButton.isChecked()
        self._model.template_plot(self._view.ui.templateCanvas.canvas, template_psd_requested)
        self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked(),
                              self._view.ui.siteCSDradioButton.isChecked(),
                              self._view.ui.siteCOHradioButton.isChecked(),
                              self._view.ui.siteERPradioButton.isChecked())
        self._view.ui.templateCanvas.canvas.draw()
        self._view.ui.siteCanvas.canvas.draw()

    def updatelandmarkcomboBox(self, sepName, checkbox):
        print(f'updating dropdown with {sepName}...')
        self._model.landmarkBoolean[sepName] = checkbox.isChecked()
        self._view.ui.landmarkcomboBox.clear()
        for sepName, sBool in self._model.landmarkBoolean.items():
            if sBool:
                self._view.ui.landmarkcomboBox.addItem(sepName)
        # erase lines and text if box unchecked
        # self._model.draw_lines(self._view.ui.siteCanvas.canvas.ax)
        # self._view.ui.siteCanvas.canvas.draw()
        print('done')

    def template_lines(self):
        self._model.template_landmarkBoolean = self._view.ui.templatelandmarkcheckBox.isChecked()
        template_psd_requested = self._view.ui.tempPSDradioButton.isChecked()
        self._model.template_plot(self._view.ui.templateCanvas.canvas, template_psd_requested)
        self._view.ui.templateCanvas.canvas.draw()

    def site_lines(self):
        self._model.landmarkBoolean = self._model.landmarkBoolean

    def linereset(self):
        line = self._view.ui.landmarkcomboBox.currentText()
        if self.currentLine in self._model.linedict:
            self._model.reset_line(self._view.ui.siteCanvas.canvas, line)
            self._view.ui.siteCanvas.canvas.draw()
            self.lineconnect()
        else:
            print("line not found in dictionary...")

    def linesreset(self):
            self._model.reset_lines(self._view.ui.siteCanvas.canvas)
            self._view.ui.siteCanvas.canvas.draw()
            self.lineconnect()

    def lineconnect(self):
        self._model.draw_lines(self._view.ui.siteCanvas.canvas.ax)
        self.currentLine = self._view.ui.landmarkcomboBox.currentText()
        if self.currentLine in self._model.linedict:
            self.line = self._model.linedict[self.currentLine][0]
            self.toptxt = self._model.linedict[self.currentLine][1]
            self.bottomtxt = self._model.linedict[self.currentLine][2]
        else:
            self.line = []
        self._view.ui.siteCanvas.canvas.draw_idle()
        self.sid = self._view.ui.siteCanvas.canvas.mpl_connect('pick_event', self.clickonline)

    def clickonline(self, event):
        if event.artist == self.line:
            print("line selected ", event.artist)
            self.follower = self._view.ui.siteCanvas.canvas.mpl_connect("motion_notify_event", self.followmouse)
            self.releaser = self._view.ui.siteCanvas.canvas.mpl_connect("button_press_event", self.releaseonclick)

    def followmouse(self, event):
        self.line.set_ydata([event.ydata, event.ydata])
        self.toptxt.set_position([0, event.ydata + 0.5])
        self.bottomtxt.set_position([0, event.ydata - 2])
        self._view.ui.siteCanvas.canvas.draw_idle()

    def releaseonclick(self, event):
        self._view.ui.siteCanvas.canvas.mpl_disconnect(self.releaser)
        self._view.ui.siteCanvas.canvas.mpl_disconnect(self.follower)
        self._model.landmarkPosition[self.currentLine] = self.line.get_ydata()[0]
        self._model.draw_lines(self._view.ui.siteCanvas.canvas.ax)
        self.lineconnect()

    def _connectSignals(self):
        self._view.ui.siteactivecheckBox.stateChanged.connect(self.updateanimalcomboBox)
        self._view.ui.animalcomboBox.currentIndexChanged.connect(self.update_siteComboBox)
        self._view.ui.sitecomboBox.currentIndexChanged.connect(self.update_siteList)
        self._view.ui.siteplotpushButton.clicked.connect(self.load_site_csd_psd)
        self._view.ui.tempPSDradioButton.toggled.connect(self.update_plots)
        self._view.ui.sitePSDradioButton.toggled.connect(self.update_plots)
        self._view.ui.siteCOHradioButton.toggled.connect(self.update_plots)
        self._view.ui.siteERPradioButton.toggled.connect(self.update_plots)
        # set the subset of separators to consider
        for boxName, checkBox in self._view.ui.layerCheckBoxes.items():
            checkBox.stateChanged.connect(partial(self.updatelandmarkcomboBox,
                                                  boxName, checkBox))
        self._view.ui.landmarkcomboBox.currentIndexChanged.connect(self.lineconnect)
        self._view.ui.nonormradioButton.toggled.connect(self.normalization)
        self._view.ui.tempmaxnormradioButton.toggled.connect(self.normalization)
        self._view.ui.sitemaxnormradioButton.toggled.connect(self.normalization)
        self._view.ui.localnormradioButton.toggled.connect(self.normalization)
        self._view.ui.lineresetButton.clicked.connect(self.linereset)
        self._view.ui.resetlandmarkspushButton.clicked.connect(self.linesreset)
        self._view.ui.templatelandmarkcheckBox.toggled.connect(self.template_lines)
        self._view.ui.assignButton.clicked.connect(self.assign_database)

def main():
    """Main function."""
    # Create an instance of QApplication
    laminar = QApplication(sys.argv)
    # Show the calculator's GUI
    view = LaminarUi()
    # get model functions
    model = LaminarModel(view=view)
    # create instance of the controller
    controller = LaminarCtrl(model=model, view=view)
    # Execute the calculator's main loop
    sys.exit(laminar.exec_())

if __name__ == '__main__':
    main()