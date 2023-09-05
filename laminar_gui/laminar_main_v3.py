import sys
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.lines as lines
from PyQt5.QtWidgets import QApplication, QWidget, QTreeWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from nems0 import db
from nems_lbhb.baphy_io import probe_finder, npx_channel_map_finder
from laminar_tools.lfp.lfp import parmfile_event_lfp
from laminar_tools.mua.mua import parmfile_mua_FTC
from laminar_tools.laminar_analysis.laminar_analysis import maximal_laminar_similarity, pad_to_template
from laminar_gui_v3 import Ui_mainWidget
from functools import partial
from pathlib import Path
import json
from scipy.interpolate import interp1d
from nems_lbhb import baphy_io as io
import datetime as dt
import  re

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
        self.template_landmarkPosition = {'BS/1': 28, '3/4': 18, '4/5': 12,
                                          '6/6d': 5, '5d/4d': 4, '4d/3d': 2, '1d/Bd': 0,
                                          '6/WM': 5, 'WM/HC': 0}
        self.template_lines = {}
        self.landmarks = self.template_landmarks
        self.landmarkPosition = {border: self.template_landmarkPosition[border] for border in self.landmarks}
        self.landmarkBoolean = {border: False for border in self.landmarks}
        self.lines = list()
        self.linedict = {}
        self.template_landmarkBoolean = True
        self.template_lines = {}
        self.depth_mapped = {}
        self.figpathroot = "/auto/users/wingertj"
        self.raw_data_path = Path('/auto/data/daq')
        self.loadedds = {}

    def update_default_landmarkPositions(self):
        self.landmarkPosition = {}
        self.landmarkBoolean = {border: False for border in self.landmarks}
        for probe in self.probe:
            self.landmarkPosition[probe] = {border: self.template_landmarkPosition[border] for border in self.landmarks}

    def animals(self, active):
        # get animals:
        species = 'ferret'
        require_active = active
        sql = f"SELECT * FROM gAnimal WHERE lab='lbhb' and species='{species}'"
        if require_active:
            sql += " AND onschedule<2"
        dAnimal = db.pd_query(sql)
        animallist = dAnimal['animal'].to_list()
        animallist.sort()
        self.animallist = animallist

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
        self.siteids = dRawFiles['cellid'].to_list()
        # get rawids for siteid
        self.rawids = dRawFiles['id'].to_list()

        # check if channel mapping in database
        # self.dbcheck = ['Yes' if i is not None else 'No' for i in dRawFiles['depthinfo']]
        substring = 'Probe'
        pattern = re.compile(f'{re.escape(substring)}([A-Za-z])')
        try:
            self.dbcheck = [', '.join(pattern.findall(di)) if di is not None else 'No' for di in dRawFiles['depthinfo']]
        except:
            self.dbcheck = ['Yes' if i is not None else 'No' for i in dRawFiles['depthinfo']]

    def site_probe_check(self):
        # find open-ephys folder for each parmfile
        site_probe_list = []
        site_probe_type = []
        for parmfile in self.parmfilelist:
            try:
                animalid = self._view.ui.animalcomboBox.currentText()
                siteid = self._view.ui.sitecomboBox.currentText()
                try:
                    data_path = self.raw_data_path/animalid/siteid/'raw'/parmfile[:9]
                    OE_experiments = [x for x in data_path.iterdir() if x.is_dir()]
                except:
                    data_path = self.raw_data_path/animalid/siteid/parmfile/'raw'
                    OE_experiments = [x for x in data_path.iterdir() if x.is_dir()]
                experiment_probe_list = []
                experiment_probe_type = []
                for OE_folder in OE_experiments:
                    probes, probe_type = probe_finder(OE_folder)
                    experiment_probe_list.append(probes)
                    experiment_probe_type.append(probe_type)
                # Quick check to make sure all probes match across an experiment - they should. If not raise an error.
                if len(set([len(prb_list) for prb_list in experiment_probe_list]))== 1:
                    prb_name_check = [all([experiment_probe_list[0][i] in experiment_probe_list[j] for j in range(len(experiment_probe_list))]) for i in range(len(experiment_probe_list[0]))]
                    if all(prb_name_check):
                        parmfile_probes = [prb_letter[-1:] for prb_letter in experiment_probe_list[0]]
                    else:
                        raise ValueError("Probes in each experiment do not match. Which experiment should be used?")
                else:
                    raise ValueError("Probes in each experiment do not match. Which experiment should be used?")

                site_probe_list.append(parmfile_probes)
                site_probe_type.append(experiment_probe_type[0])
            except:
                print(f"Unable to find raw data path for {parmfile}...unexpected data path?")

        return site_probe_list, site_probe_type

    def BNB_FTC_channel_match(self, FTC_parmfile, BNB_parmfile):

        # FTC channel map
        try:
            data_path = FTC_parmfile
            OE_experiments = [x for x in data_path.iterdir() if x.is_dir()]
        except:
            data_path = FTC_parmfile
            OE_experiments = [x for x in data_path.iterdir() if x.is_dir()]
        FTC_channel_list = []
        for OE_folder in OE_experiments:
            FTC_channel_list.append(npx_channel_map_finder(OE_folder))

        # BNB channel map
        try:
            data_path = BNB_parmfile
            OE_experiments = [x for x in data_path.iterdir() if x.is_dir()]
        except:
            data_path = BNB_parmfile
            OE_experiments = [x for x in data_path.iterdir() if x.is_dir()]
        BNB_channel_list = []
        for OE_folder in OE_experiments:
            BNB_channel_list.append(npx_channel_map_finder(OE_folder))

        # use the first experiment in the lists because channel maps shouldn't change between experiments within same recording
        FTC_channels = FTC_channel_list[0]
        BNB_channels = BNB_channel_list[0]

        return FTC_channels == BNB_channels


    def load_template(self):
        template_psd = np.load("/auto/users/wingertj/code/csd_project/data/laminar_features/template/final_psd_template_v2.npy")
        template_csd = np.load("/auto/users/wingertj/code/csd_project/data/laminar_features/template/final_csd_template_v2.npy")
        max_power = template_psd.max(axis=0)
        self.temp_max_power = max_power
        self.template_psd = template_psd
        self.template_csd = template_csd

    def site_csd_psd(self, parmfile, align=True):
        self.load_template()
        csd, psd, freqs, stim_window, rasterfs, column_xy_sorted, column_xy, channel_xy, coh_mat, probe, probe_type = parmfile_event_lfp(parmfile)
        max_power = [np.nanmax(psd[i], axis=0) for i in range(len(psd))]
        if align:
            averaged_template, ssim_index, ssim = maximal_laminar_similarity(template=self.template_psd, image=psd, overlap=10,
                                                                             ssim_window=5, expansion=True)
            csd = [csd, ]
            psd = [psd, ]
            ssim_index_list = [ssim_index, ]
            self.ssim = ssim
            self.ssim_index = ssim_index
            nan_pad_psd, nan_pad_csd, padding, template, template_csd = pad_to_template(self.template_psd, self.template_csd, psd, csd, ssim_index_list, already_padded=True)
            self.template_psd = template
            self.template_csd = template_csd
            self.padding = padding
            upper_pad = np.empty((len(coh_mat[:, 0, 0]), int(padding[1]), len(coh_mat[0,:])))
            upper_pad[:] = np.nan
            lower_pad = np.empty((len(coh_mat[:, 0, 0]), int(padding[0]), len(coh_mat[0,:])))
            lower_pad[:] = np.nan
            coh_mat = np.concatenate((lower_pad, coh_mat, upper_pad), axis=1)
            psd = np.squeeze(nan_pad_psd)
            csd = np.squeeze(nan_pad_csd)

        # self.erp = erp
        self.coh = coh_mat
        self.freqs = freqs
        self.rasterfs = rasterfs
        self.window = stim_window
        self.window_samples = stim_window*rasterfs
        self.psd = psd
        self.site_max_power = max_power
        self.csd = csd
        self.column_xy = column_xy
        self.column_keys = column_xy_sorted
        self.channel_xy = channel_xy
        self.probe = probe
        self.current_probe_index = [index for index, probe_id in enumerate(self.probe) if self.current_probe == probe_id[-1:]][0]
        self.probe_type = probe_type

    def FTC_mua_heatmap(self, parmfile):
        pass


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
        self.psd_norm = [self.psd[i]/self.site_max_power[i] for i in range(len(self.psd))]
        self.template_psd_norm = self.template_psd/self.temp_max_power
        self.unpadded_psd_norm = [self.psd_norm[i][~np.isnan(self.psd_norm[i])] for i in range(len(self.psd_norm))]

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
        if self.probe[self.current_probe_index] in list(self.loadedds.keys()):
            self.landmarkPosition[line] = self.loadedds[self.probe[self.current_probe_index]]['landmarkPosition'][line]
        else:
            self.landmarkPosition[line] = self.template_landmarkPosition[line]
        self.draw_lines(canvas.ax)

    def reset_lines(self, canvas):
        if self.probe[self.current_probe_index] in list(self.loadedds.keys()):
            self.load_depth_from_db()
        else:
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
            self.cmax = np.nanmax(self.template_psd_norm)
        if temp_psd:
            im = canvas.ax.imshow(self.template_psd_norm, origin='lower', aspect='auto', clim=[0, self.cmax])
            canvas.ax.set_xlim(self.freqs[0][0], self.freqs[0][-1])
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

    def site_plot(self, canvas, site_psd, site_csd, site_coh):
        """
        plots the laminar data
        """
        print('plotting laminar data...')
        self.clear_canvas(canvas)
        if self._view.ui.localnormradioButton.isChecked():
            self.cmax = self.unpadded_psd_norm[self.current_probe_index].max()
        if site_psd:
            self._view.ui.figsavelineEdit.setText(f"{self.figpathroot}/{self.parmfile[:-8]}_PSD.pdf")
            im = canvas.ax.imshow(self.psd_norm[self.current_probe_index], origin='lower', aspect='auto', clim=[0, self.cmax])
            canvas.ax.set_xlim(self.freqs[self.current_probe_index][0], self.freqs[self.current_probe_index][-1])
            canvas.ax.set_xlabel("frequency")
            canvas.fig.colorbar(im, cax=canvas.cax)
            y_ticks = np.arange(0, len(self.psd_norm[self.current_probe_index][:, 0]), 8)
            canvas.ax.set_yticks(y_ticks)
            if self.probe_type == 'NPX':
                y_tick_channels = np.take(self.column_keys[self.current_probe_index], y_ticks, axis=0)
                canvas.ax.set_yticklabels(["ch" + str(i) + '\n' + str(self.column_xy[self.current_probe_index][i][1]) + 'um' for i in y_tick_channels], fontsize=6)

        elif site_csd:
            self._view.ui.figsavelineEdit.setText(f"{self.figpathroot}/{self.parmfile[:-8]}_CSD.pdf")
            im = canvas.ax.imshow(self.csd[self.current_probe_index], origin='lower', aspect='auto')
            x_ticks = np.linspace(0, len(self.csd[self.current_probe_index][0, :]), 5)
            x_ticklabels = np.round(np.linspace(-self.window, self.window, 5), decimals=2)
            canvas.ax.set_xticks(x_ticks)
            canvas.ax.set_xticklabels(x_ticklabels)
            canvas.ax.set_xlabel("time (s)")
            cbar = canvas.fig.colorbar(im, cax=canvas.cax, ticks=[self.csd[self.current_probe_index][1:-1, :].max(), self.csd[self.current_probe_index][1:-1, :].min()])
            cbar.ax.set_yticklabels(['source', 'sink'])
            y_ticks = np.arange(0, len(self.psd_norm[self.current_probe_index][:, 0]), 8)
            canvas.ax.set_yticks(y_ticks)
            if self.probe_type == 'NPX':
                y_tick_channels = np.take(self.column_keys[self.current_probe_index], y_ticks, axis=0)
                canvas.ax.set_yticklabels(["ch" + str(i) + '\n' + str(self.column_xy[self.current_probe_index][i][1]) + 'um' for i in y_tick_channels], fontsize=6)

        elif site_coh:
            self._view.ui.figsavelineEdit.setText(f"{self.figpathroot}/{self.parmfile[:-8]}_COH.pdf")
            # idx30 = np.where(self.freqs > 30)[0][0]
            # idx150 = np.where(self.freqs < 150)[0][-1]
            idx1 = np.where(self.freqs[self.current_probe_index] > 1)[0][0]
            idx15 = np.where(self.freqs[self.current_probe_index] < 15)[0][-1]
            gamma_cohmat = np.squeeze(self.coh[self.current_probe_index].mean(axis=0))
            im = canvas.ax.imshow(gamma_cohmat, origin='lower', aspect='auto')
            cbar = canvas.fig.colorbar(im, cax=canvas.cax)
            y_ticks = np.arange(0, len(self.psd_norm[self.current_probe_index][:, 0]), 8)
            canvas.ax.set_yticks(y_ticks)
            canvas.ax.set_xticks(y_ticks)
            if self.probe_type == 'NPX':
                y_tick_channels = np.take(self.column_keys[self.current_probe_index], y_ticks, axis=0)
                canvas.ax.set_yticklabels(["ch" + str(i) + '\n' + str(self.column_xy[self.current_probe_index][i][1]) + 'um' for i in y_tick_channels], fontsize=6)
                canvas.ax.set_xticklabels(["ch" + str(i) + '\n' + str(self.column_xy[self.current_probe_index][i][1]) + 'um' for i in y_tick_channels], fontsize=6)
        # elif site_erp:
        #     self._view.ui.figsavelineEdit.setText(f"{self.figpathroot}/{self.parmfile[:-8]}_ERP.pdf")
        #     for i in range(len(self.erp[:, 0])):
        #         canvas.ax.plot((self.erp[i, :] + 500*i))
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
        site_area_deep = self._view.ui.areatextdeep.toPlainText()
        active_landmarks = [k for (k, v) in self.landmarkBoolean.items() if v == True]
        active_positions = [int(self.column_xy[self.column_keys[int(round(temp_landmark_dict[lm]))]][1]) for lm in active_landmarks]
        active_assignments = [self._view.ui.layerBorders[lm] for lm in active_landmarks]

        if (len(active_landmarks) < 2) and (self._view.ui.badsitecheckBox.isChecked() == False):
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
            # This isn't a good solution to the problem (-Jereme and Greg 2023_04_13)
            try:
                min_index = abs(np.array(active_positions) - channel_position).argmin()
            except:
                pass
            if self._view.ui.badsitecheckBox.isChecked():
                electrode_depth = 'NA'
                location_label = electrode_depth

            elif channel_position <= active_positions[min_index]:
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
            try:
                y_in = np.array([upper_position, lower_position])
                y_out = np.array([upper_assignment, lower_assignment])
                f = interp1d(y_in, y_out, fill_value='extrapolate')
                channel_dict[ch] = [location_label, int(f(channel_position)), channel_position]
            except:
                channel_dict[ch] = [location_label, location_label, channel_position]
        complete_dict = {}
        complete_dict['channel info'] = channel_dict
        complete_dict['parmfile'] = self.parmfile
        complete_dict['site area'] = site_area
        complete_dict['site area deep'] = site_area_deep
        self.depth_mapped = {**complete_dict, **position_memory}

    def depth_mapping_from_pixel_value(self):
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
        site_area_deep = self._view.ui.areatextdeep.toPlainText()
        active_landmarks = [k for (k, v) in self.landmarkBoolean.items() if v == True]
        # convert channels into plot pixels - divide actual depth by column channel spacing
        column_distances = [int(v[1]) for (k,v) in self.column_xy[self.current_probe_index].items()]
        column_diffs = [column_distances[i+1]-column_distances[i] for i in range(len(column_distances)-1)]
        # take most common difference - might run into issues if the electrode pattern is really odd
        column_spacing = max(set(column_diffs), key=column_diffs.count)
        active_positions_pixels = [temp_landmark_dict[lm] for lm in active_landmarks]
        active_positions = [int(self.column_xy[self.current_probe_index][self.column_keys[self.current_probe_index][int(round(temp_landmark_dict[lm]))]][1]) for lm in active_landmarks]
        active_assignments = [self._view.ui.layerBorders[lm] for lm in active_landmarks]

        if (len(active_landmarks) < 2) and (self._view.ui.badsitecheckBox.isChecked() == False):
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
        for ch in list(self.channel_xy[self.current_probe_index].keys()):
            # get channel position in pixels
            channel_position = int(self.channel_xy[self.current_probe_index][ch][1])/column_spacing
            # find
            try:
                min_index = abs(np.array(active_positions_pixels) - channel_position).argmin()
                upper_values = sorted([val for val in np.array(active_positions_pixels) if val-channel_position >= 0])
                lower_values = sorted([val for val in np.array(active_positions_pixels) if val-channel_position < 0])
            except:
                pass
            if self._view.ui.badsitecheckBox.isChecked():
                electrode_depth = 'NA'
                location_label = electrode_depth

            elif lower_values and upper_values:
                lower_landmark = active_landmarks[np.where(active_positions_pixels == lower_values[-1])[0][0]]
                upper_landmark = active_landmarks[np.where(active_positions_pixels == upper_values[0])[0][0]]
                lower_position = active_positions[np.where(active_positions_pixels == lower_values[-1])[0][0]]
                upper_position = active_positions[np.where(active_positions_pixels == upper_values[0])[0][0]]
                lower_assignment = active_assignments[np.where(active_positions_pixels == lower_values[-1])[0][0]]
                upper_assignment = active_assignments[np.where(active_positions_pixels == upper_values[0])[0][0]]
                lower_top, lower_bottom = lower_landmark.split('/')
                upper_top, upper_bottom = upper_landmark.split('/')
                location_label = ''.join([upper_bottom, lower_top])

            elif lower_values and not upper_values:
                lower_landmark = active_landmarks[np.where(active_positions_pixels == lower_values[-2])[0][0]]
                upper_landmark = active_landmarks[np.where(active_positions_pixels == lower_values[-1])[0][0]]
                lower_position = active_positions[np.where(active_positions_pixels == lower_values[-2])[0][0]]
                upper_position = active_positions[np.where(active_positions_pixels == lower_values[-1])[0][0]]
                lower_assignment = active_assignments[np.where(active_positions_pixels == lower_values[-2])[0][0]]
                upper_assignment = active_assignments[np.where(active_positions_pixels == lower_values[-1])[0][0]]
                lower_top, lower_bottom = lower_landmark.split('/')
                upper_top, upper_bottom = upper_landmark.split('/')
                location_label = upper_top

            elif upper_values and not lower_values:
                lower_landmark = active_landmarks[np.where(active_positions_pixels == upper_values[0])[0][0]]
                upper_landmark = active_landmarks[np.where(active_positions_pixels == upper_values[1])[0][0]]
                lower_position = active_positions[np.where(active_positions_pixels == upper_values[0])[0][0]]
                upper_position = active_positions[np.where(active_positions_pixels == upper_values[1])[0][0]]
                lower_assignment = active_assignments[np.where(active_positions_pixels == upper_values[0])[0][0]]
                upper_assignment = active_assignments[np.where(active_positions_pixels == upper_values[1])[0][0]]
                lower_top, lower_bottom = lower_landmark.split('/')
                upper_top, upper_bottom = upper_landmark.split('/')
                location_label = lower_bottom

            try:
                y_in = np.array([upper_position, lower_position])
                y_out = np.array([upper_assignment, lower_assignment])
                f = interp1d(y_in, y_out, fill_value='extrapolate')
                channel_dict[ch] = [location_label, int(f(channel_position*column_spacing)), channel_position]
            except:
                channel_dict[ch] = [location_label, location_label, channel_position]
        complete_dict = {}
        complete_dict['channel info'] = channel_dict
        complete_dict['parmfile'] = self.parmfile
        complete_dict['site area'] = site_area
        complete_dict['site area deep'] = site_area_deep

        # assign new depth mapping to depth dictionary currently in celldb if it exists
        self.depth_mapped = self.loadedds
        self.depth_mapped[self.probe[self.current_probe_index]] = {**complete_dict, **position_memory}
        # update database loaded depths
        self.loadedds = self.depth_mapped


    def load_depth_from_db(self):
        # load from database
        sql = f"SELECT * FROM gDataRaw WHERE id={int(self.rawid)}"
        draw = db.pd_query(sql)
        loadedds = json.loads(draw.loc[0,'depthinfo'])
        self.loadedds = {}
        self.loadedds = loadedds
        if self._view.ui.sitealigncheckBox.isChecked():
            lower_pad_size = self.padding[0]
            for landmark in list(loadedds[self.probe[self.current_probe_index]]['landmarkPosition'].keys()):
                self.landmarkPosition[landmark] = loadedds[self.probe[self.current_probe_index]]['landmarkPosition'][landmark]+lower_pad_size
        else:
            for probe in self.probe:
                try:
                    for landmark in list(loadedds[probe]['landmarkPosition'].keys()):
                        self.landmarkPosition[landmark] = loadedds[self.probe[self.current_probe_index]]['landmarkPosition'][landmark]
                except:
                    continue
        try:
            self.landmarkBoolean = loadedds[self.probe[self.current_probe_index]]['landmarkBoolean']
            self.area = loadedds[self.probe[self.current_probe_index]]['site area']
            self.area_deep = loadedds[self.probe[self.current_probe_index]]['site area deep']
        except:
            self.area = ''
            self.area_deep = ''

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

    def celldb_save_plots(self):
        sql = f"SELECT pendate from gPenetration where penname='{self.siteid}'"
        d = db.pd_query(sql)
        year = d['pendate'][0][:4]
        animalid = self._view.ui.animalcomboBox.currentText().lower()
        parmfile = self.parmfile[:-2]
        fig_loc = f"/auto/data/web/celldb/analysis/{animalid}/{year}/{parmfile}.lfp_depth_markers.jpg"
        f, ax = plt.subplots(1,3, figsize=(15, 5), layout='tight')
        im = ax[0].imshow(self.psd_norm[self.current_probe_index], origin='lower', aspect='auto', clim=[0, self.cmax])
        ax[0].set_xlim(self.freqs[self.current_probe_index][0], self.freqs[self.current_probe_index][-1])
        ax[0].set_xlabel("frequency")
        # f.colorbar(im, cax=ax[0].cax)
        y_ticks = np.arange(0, len(self.psd_norm[self.current_probe_index][:, 0]), 8)
        ax[0].set_yticks(y_ticks)
        if self.probe == 'NPX':
            y_tick_channels = np.take(self.column_keys[self.current_probe_index], y_ticks, axis=0)
            ax[0].set_yticklabels(["ch" + str(i) + '\n' + str(self.column_xy[self.current_probe_index][i][1]) + 'um' for i in y_tick_channels], fontsize=6)

        im2 = ax[1].imshow(self.csd[self.current_probe_index], origin='lower', aspect='auto')
        x_ticks = np.linspace(0, len(self.csd[self.current_probe_index][0, :]), 5)
        x_ticklabels = np.round(np.linspace(-self.window, self.window, 5), decimals=2)
        ax[1].set_xticks(x_ticks)
        ax[1].set_xticklabels(x_ticklabels)
        ax[1].set_xlabel("time (s)")
        # cbar = canvas.fig.colorbar(im, cax=canvas.cax, ticks=[self.csd[1:-1, :].max(), self.csd[1:-1, :].min()])
        # cbar.ax.set_yticklabels(['source', 'sink'])
        y_ticks = np.arange(0, len(self.psd_norm[self.current_probe_index][:, 0]), 8)
        ax[1].set_yticks(y_ticks)
        if self.probe == 'NPX':
            y_tick_channels = np.take(self.column_keys[self.current_probe_index], y_ticks, axis=0)
            ax[1].set_yticklabels(["ch" + str(i) + '\n' + str(self.column_xy[self.current_probe_index][i][1]) + 'um' for i in y_tick_channels], fontsize=6)
        idx1 = np.where(self.freqs[self.current_probe_index] > 1)[0][0]
        idx15 = np.where(self.freqs[self.current_probe_index] < 15)[0][-1]
        gamma_cohmat = np.squeeze(self.coh[self.current_probe_index].mean(axis=0))
        im3 = ax[2].imshow(gamma_cohmat, origin='lower', aspect='auto')
        # cbar = canvas.fig.colorbar(im, cax=canvas.cax)
        y_ticks = np.arange(0, len(self.psd_norm[self.current_probe_index][:, 0]), 8)
        ax[2].set_yticks(y_ticks)
        ax[2].set_xticks(y_ticks)
        if self.probe == 'NPX':
            y_tick_channels = np.take(self.column_keys[self.current_probe_index], y_ticks, axis=0)
            ax[2].set_yticklabels(
                ["ch" + str(i) + '\n' + str(self.column_xy[self.current_probe_index][i][1]) + 'um' for i in y_tick_channels], fontsize=6)
            ax[2].set_xticklabels(
                ["ch" + str(i) + '\n' + str(self.column_xy[self.current_probe_index][i][1]) + 'um' for i in y_tick_channels], fontsize=6)

        # draw some lines
        for sName in self.landmarks:
            try:
                sBool = self.landmarkBoolean[sName]
                sPos = self.landmarkPosition[sName]
                if sBool:
                    top, bottom = sName.split('/')
                    self.linedict[sName] = [ax[0].axhline(sPos, color='red', linewidth=2, picker=5),
                                            ax[0].text(0, sPos + 0.5, top, color='orange', fontsize=10),
                                            ax[0].text(0, sPos - 2, bottom, color='orange', fontsize=10), ax[1].axhline(sPos, color='red', linewidth=2, picker=5),
                                            ax[1].text(0, sPos + 0.5, top, color='orange', fontsize=10),
                                            ax[1].text(0, sPos - 2, bottom, color='orange', fontsize=10), ax[2].axhline(sPos, color='red', linewidth=2, picker=5),
                                            ax[2].text(0, sPos + 0.5, top, color='orange', fontsize=10),
                                            ax[2].text(0, sPos - 2, bottom, color='orange', fontsize=10)]
            except:
                continue
        f.savefig(fig_loc)

class LaminarCtrl():
    def __init__(self, model, view):
        self._view = view
        self._model = model
        self.updateanimalcomboBox(active=True)
        self.update_siteComboBox(index=0)
        self.update_siteList(index=0)
        self._connectSignals()
        self.default_palette = self._view.palette()

    def savecurrentfig(self):
        if self._view.ui.figsavecheckBox.isChecked():
            try:
                import matplotlib.pyplot as plt
                params = {'axes.spines.right': False,
                          'axes.spines.top': False,
                          'pdf.fonttype': 42,
                          'ps.fonttype': 42}
                plt.rcParams.update(params)
                figpath = self._view.ui.figsavelineEdit.text()
                self._view.ui.siteCanvas.canvas.fig.savefig(figpath)
            except:
                print("could not save fig...not valid path?")
        else:
            pass

    def assign_database(self):
        self._model.depth_mapping_from_pixel_value()
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

        uniqueids = list(set(self._model.siteids))
        for siteid in uniqueids:
            try:
                io.get_spike_info(siteid=siteid, save_to_db=True)
            except:
                print("Spike info not found. Still needs to be sorted?")
        self.update_siteList(self._view.ui.sitecomboBox.currentIndex())
        self._model.celldb_save_plots()

    def changeTheme(self):
        if self._view.ui.themecheckBox.isChecked():
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.WindowText, Qt.white)
            dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.ToolTipText, Qt.white)
            dark_palette.setColor(QPalette.Text, Qt.white)
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, Qt.white)
            dark_palette.setColor(QPalette.BrightText, Qt.red)
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
            dark_palette.setColor(QPalette.Active, QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, Qt.darkGray)
            dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, Qt.darkGray)
            dark_palette.setColor(QPalette.Disabled, QPalette.Text, Qt.darkGray)
            dark_palette.setColor(QPalette.Disabled, QPalette.Light, QColor(53, 53, 53))
            self._view.setPalette(dark_palette)
        else:
            self._view.setPalette(self.default_palette)

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

    def update_probecomboBox(self):
        getSelected = self._view.ui.siteList.selectedItems()
        if getSelected:
            baseNode = getSelected[0]
            probes = baseNode.text(3)
            self._model.site_probes = probes.split(", ")
            self._view.ui.probecomboBox.clear()
            self._view.ui.probecomboBox.addItems(self._model.site_probes)
            self._model.current_probe = self._view.ui.probecomboBox.currentText()
        else:
            pass


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
            probe_check, probe_type = self._model.site_probe_check()
            self._model.parmfile_probes = probe_check
            self._model.parmfile_probe_type = {pf: prb_type for (pf, prb_type) in zip(files, probe_type)}
            for i in range(len(files)):
                item = QTreeWidgetItem(None)
                item.setText(0, files[i])
                item.setText(1, str(rawids[i]))
                item.setText(2, dbcheck[i])
                try:
                    item.setText(3, ', '.join(probe_check[i]))
                except:
                    pass
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
            # self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked(),
            #                       self._view.ui.siteCSDradioButton.isChecked(),
            #                       self._view.ui.siteCOHradioButton.isChecked(),
            #                       self._view.ui.siteERPradioButton.isChecked())
            self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked(),
                                  self._view.ui.siteCSDradioButton.isChecked(),
                                  self._view.ui.siteCOHradioButton.isChecked())
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
        self.update_probecomboBox()
        align = self._view.ui.sitealigncheckBox.isChecked()
        getSelected = self._view.ui.siteList.selectedItems()
        if getSelected:
            baseNode = getSelected[0]
            getChildNode = baseNode.text(0)
            self._model.parmfile = str(getChildNode)
            rawid = baseNode.text(1)
            self._model.db_available = baseNode.text(2)
            self._model.rawid = rawid
        self._model.siteid = self._view.ui.sitecomboBox.currentText()
        self._model.load_area_from_db()
        animalid = self._view.ui.animalcomboBox.currentText()
        rawpath = self._model.raw_data_path
        sql = f"SELECT gDataRaw.* FROM gDataRaw WHERE cellid like '{self._model.siteid}%%' and bad=0 and training = 0"
        dRawFiles = db.pd_query(sql)
        resppath = dRawFiles['resppath'][0]
        parmfilepath = [rawpath/animalid/self._model.siteid/self._model.parmfile]
        self._model.bnb_parmfile_path = parmfilepath
        self._model.bnb_raw_path = rawpath / animalid / self._model.siteid / 'raw' / self._model.parmfile[:9]
        # get nearest FTC info if it exists.
        FTC_parmfile_ints = [int(parmfile[7:9]) for parmfile in self._model.parmfilelist if 'FTC' in parmfile]
        FTC_parmfiles = [parmfile for parmfile in self._model.parmfilelist if 'FTC' in parmfile]
        BNB_int = int(self._model.parmfile[7:9])
        if FTC_parmfile_ints:
            # find parmfile closest to selected BNB
            FTC_BNB_dist = [abs(FTC_int - BNB_int) for FTC_int in FTC_parmfile_ints]
            parmfile_dists = list(zip(FTC_BNB_dist, FTC_parmfiles))
            parmfile_dists.sort()
            channel_match = False
            for pf_ind, parmfile in parmfile_dists:
                if channel_match:
                    break
                self._model.ftc_parmfile = parmfile
                self._model.ftc_parmfile_path = [rawpath/animalid/self._model.siteid/self._model.ftc_parmfile]
                self._model.ftc_raw_path = rawpath / animalid / self._model.siteid / 'raw' / self._model.ftc_parmfile[:9]
                try:
                    if self._model.parmfile_probe_type[parmfile] == 'NPX':
                        self._model.BNB_FTC_channel_match(self._model.ftc_raw_path, self._model.bnb_raw_path)
                        self._model.FTC_mua_heatmap(self._model.ftc_parmfile_path)
                        channel_match = True
                    elif self._model.parmfile_probe_type[parmfile] == 'UCLA':
                        # assume the recordings use the same channel map
                        self._model.FTC_mua_heatmap(self._model.ftc_parmfile_path)
                        channel_match = True
                    else:
                        channel_match = False
                        print("Channel maps between BNB and FTC might not match because probe type is unknown. Skipping")
                except:
                    continue

        self._model.site_csd_psd(parmfilepath, align=align)
        # update default model for probes in current site
        self._model.update_default_landmarkPositions
        if self._model.db_available != 'No':
            # load saved depth data
            self._model.load_depth_from_db()
            # reassign gui settings - checkboxes and text
            for landmark, landbool in list(self._model.landmarkBoolean.items()):
                if landbool:
                    try:
                        self._view.ui.layerCheckBoxes[landmark].setChecked(landbool)
                    except:
                        print("change in gui landmarks from what is in database - leaving blank")
                        del self._model.landmarkBoolean[landmark]
            self._view.ui.areatext.setText(self._model.area)
            self._view.ui.areatextdeep.setText(self._model.area_deep)
        else:
            self._model.loadedds = {}
        self.normalization()
        self._model.template_plot(self._view.ui.templateCanvas.canvas, self._view.ui.tempPSDradioButton.isChecked())
        # self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked(),
        #                       self._view.ui.siteCSDradioButton.isChecked(), self._view.ui.siteCOHradioButton.isChecked(),
        #                       self._view.ui.siteERPradioButton.isChecked())
        self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked(),
                              self._view.ui.siteCSDradioButton.isChecked(), self._view.ui.siteCOHradioButton.isChecked())
        self._view.ui.templateCanvas.canvas.draw()
        self._view.ui.siteCanvas.canvas.draw()
        self.lineconnect()

    def update_plots(self):
        try:
            template_psd_requested = self._view.ui.tempPSDradioButton.isChecked()
            site_psd_requested = self._view.ui.sitePSDradioButton.isChecked()
            site_csd_requested = self._view.ui.siteCSDradioButton.isChecked()
            site_coh_requested = self._view.ui.siteCOHradioButton.isChecked()
            # site_erp_requested = self._view.ui.siteERPradioButton.isChecked()
            self._model.current_probe = self._view.ui.probecomboBox.currentText()
            self.update_current_probe_index()
            self._model.template_plot(self._view.ui.templateCanvas.canvas, template_psd_requested)
            # self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked(),
            #                       self._view.ui.siteCSDradioButton.isChecked(),
            #                       self._view.ui.siteCOHradioButton.isChecked(),
            #                       self._view.ui.siteERPradioButton.isChecked())
            self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked(),
                                  self._view.ui.siteCSDradioButton.isChecked(),
                                  self._view.ui.siteCOHradioButton.isChecked())
            self._view.ui.templateCanvas.canvas.draw()
            self._view.ui.siteCanvas.canvas.draw()
            self.lineconnect()
        except:
            print("Can't update plot. Site not loaded?")

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

    def update_current_probe_index(self):
        self._model.current_probe_index = [index for index, probe_id in enumerate(self._model.probe) if
                                    self._model.current_probe == probe_id[-1:]][0]

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

    def landmarkreset(self):
        self._model.landmarkBoolean = {key:False for key in self._model.landmarkBoolean.keys()}
        for boxName, checkBox in self._view.ui.layerCheckBoxes.items():
            checkBox.setChecked(False)
        self.update_plots()

    def _connectSignals(self):
        self._view.ui.siteactivecheckBox.stateChanged.connect(self.updateanimalcomboBox)
        self._view.ui.animalcomboBox.currentIndexChanged.connect(self.update_siteComboBox)
        self._view.ui.sitecomboBox.currentIndexChanged.connect(self.update_siteList)
        self._view.ui.siteplotpushButton.clicked.connect(self.load_site_csd_psd)
        self._view.ui.tempPSDradioButton.toggled.connect(self.update_plots)
        self._view.ui.sitePSDradioButton.toggled.connect(self.update_plots)
        self._view.ui.siteCOHradioButton.toggled.connect(self.update_plots)
        self._view.ui.siteFTCradioButton.toggled.connect(self.update_plots)
        # self._view.ui.siteERPradioButton.toggled.connect(self.update_plots)
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
        self._view.ui.figsavepushButton.clicked.connect(self.savecurrentfig)
        self._view.ui.themecheckBox.toggled.connect(self.changeTheme)
        self._view.ui.badsitecheckBox.toggled.connect(self.landmarkreset)
        self._view.ui.probecomboBox.currentIndexChanged.connect(self.update_plots)


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
