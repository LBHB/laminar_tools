import os
import sys
import warnings
from PyQt5.QtWidgets import QApplication, QWidget, QTreeWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.lines as lines
from laminar_tools.lfp.lfp import parmfile_event_lfp
from laminar_tools.mua.mua import parmfile_mua_FTC
from laminar_tools.laminar_analysis.laminar_analysis import maximal_laminar_similarity, pad_to_template
from laminar_gui_v3 import Ui_mainWidget
from functools import partial
from pathlib import Path
import json
from scipy.interpolate import interp1d
import nems.tools.signal
from nems_lbhb import baphy_io as io
import datetime as dt
import  re
from nems_lbhb.plots import ftc_heatmap
from nems_lbhb import db, baphy_experiment
from nems_lbhb.baphy_io import probe_finder, npx_channel_map_finder

from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        self.ftc_response_cache = {}
        self.ftc_candidate_for_bnb = None

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
                parmfile_probes = ['?']
                experiment_probe_type = ['?']
                site_probe_list.append(parmfile_probes)
                site_probe_type.append(experiment_probe_type[0])
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

    @staticmethod
    def _parmfile_run_number(parmfile):
        """Return the session number after the site letter (for example a07)."""
        match = re.search(r'a(\d+)(?=_)', Path(str(parmfile)).name,
                          flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f'Cannot determine session number from {parmfile}')
        return int(match.group(1))

    def _clear_ftc_candidate(self):
        self.ftc_parmfile = None
        self.ftc_parmfile_path = None
        self.ftc_raw_path = None
        self.ftc_candidate_for_bnb = None

    def find_compatible_ftc_parmfile(self, bnb_parmfile):
        """Select the nearest same-site FTC run with a compatible channel map."""
        try:
            bnb_run = self._parmfile_run_number(bnb_parmfile)
        except ValueError as error:
            self._clear_ftc_candidate()
            print(error)
            return None

        candidates = []
        for parmfile in self.parmfilelist:
            if 'FTC' not in str(parmfile).upper():
                continue
            try:
                candidates.append((self._parmfile_run_number(parmfile), parmfile))
            except ValueError:
                print(f'Skipping FTC with an unrecognized name: {parmfile}')

        for _, candidate in sorted(
                candidates, key=lambda item: (abs(item[0] - bnb_run), item[1])):
            raw_path = self.parmfile_raw_path(candidate)
            probe_type = self.parmfile_probe_type.get(candidate, '?')
            try:
                if probe_type == 'NPX':
                    channel_match = self.BNB_FTC_channel_match(
                        raw_path, self.bnb_raw_path
                    )
                elif probe_type == 'UCLA':
                    channel_match = True
                else:
                    print(f'Skipping {candidate}: unknown probe type {probe_type}')
                    continue
            except Exception as error:
                print(f'Skipping {candidate}: channel-map check failed ({error})')
                continue

            if channel_match:
                self.ftc_parmfile = candidate
                self.ftc_parmfile_path = [
                    self.raw_data_path / self._view.ui.animalcomboBox.currentText()
                    / self.siteid / candidate
                ]
                self.ftc_raw_path = raw_path
                self.ftc_candidate_for_bnb = (self.siteid, bnb_parmfile)
                print(f'Using FTC candidate {candidate} for {bnb_parmfile}')
                return candidate

            print(f'Skipping {candidate}: channel map does not match {bnb_parmfile}')

        self._clear_ftc_candidate()
        print(f'No compatible FTC candidate found for {bnb_parmfile}')
        return None


    def load_template(self):
        template_psd = np.load("/auto/users/wingertj/code/csd_project/data/laminar_features/template/final_psd_template_v2.npy")
        template_csd = np.load("/auto/users/wingertj/code/csd_project/data/laminar_features/template/final_csd_template_v2.npy")
        max_power = template_psd.max(axis=0)
        self.temp_max_power = max_power
        self.template_psd = template_psd
        self.template_csd = template_csd

    def site_csd_psd(self, parmfile, align=True):
        self.load_template()
        self.padding = None
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

    def remove_ax(self, canvas):
        # delete axes from canvas
        axes = canvas.ax if isinstance(canvas.ax, (list, tuple, np.ndarray)) else [canvas.ax]
        for axis in axes:
            if axis in canvas.fig.axes:
                canvas.fig.delaxes(axis)
        try:
            if canvas.cax in canvas.fig.axes:
                canvas.fig.delaxes(canvas.cax)
        except AttributeError:
            pass

    @staticmethod
    def _prepare_ftc_response(resp):
        """Collapse PSI's frequency:channel FTC names to frequency epochs."""
        resp = resp.copy()
        resp.epochs = resp.epochs.copy()
        resp.epochs['name'] = resp.epochs['name'].str.replace(
            r'^(STIM_[^:]+):[^,]+$', r'\1', regex=True
        )
        return resp

    @staticmethod
    def _threshold_channel_info(cellids, probe):
        probe = str(probe)
        selected = []
        depths = []
        for cellid in cellids:
            match = re.search(r'-(\d+)([A-Za-z])$', cellid)
            if match is None or match.group(2) != probe:
                continue
            selected.append(cellid)
            depths.append(int(match.group(1)))
        return selected, np.asarray(depths)

    def _lfp_channel_grid(self):
        keys = [str(key) for key in self.column_keys[self.current_probe_index]]
        positions = self.column_xy[self.current_probe_index]
        row_count = self.psd_norm[self.current_probe_index].shape[0]

        if row_count > len(keys) and self.padding is not None:
            lower = int(self.padding[0])
            upper = row_count - len(keys) - lower
            keys = ([None] * lower) + keys + ([None] * max(upper, 0))

        depths = [None if key is None else int(positions[key][1]) for key in keys]
        return keys, depths

    @staticmethod
    def _raw_to_physical_channels(raw_path, probe):
        raw_path = Path(raw_path)
        oe_folders = sorted(path for path in raw_path.iterdir() if path.is_dir())
        if not oe_folders:
            oe_folders = [raw_path]

        for oe_folder in oe_folders:
            for probe_geometry in npx_channel_map_finder(oe_folder):
                for probe_name, channel_geometry in probe_geometry.items():
                    if probe_name.endswith(str(probe)):
                        return {
                            raw_channel: int(physical_channel)
                            for raw_channel, physical_channel in enumerate(
                                channel_geometry.keys(), start=1
                            )
                        }
        raise ValueError(f'No Open Ephys channel map found for probe {probe}')

    def _align_threshold_to_lfp(self, resp, probe):
        raw_to_physical = self._raw_to_physical_channels(
            self.ftc_raw_path, probe
        )
        physical_events = {}
        for cellid in resp.chans:
            match = re.search(r'-(\d+)([A-Za-z])$', cellid)
            if match is None or match.group(2) != str(probe):
                continue
            physical_channel = raw_to_physical.get(int(match.group(1)))
            if physical_channel is not None:
                physical_events[physical_channel] = np.asarray(resp._data[cellid])

        grid, depths = self._lfp_channel_grid()
        present = np.asarray([
            key is not None and int(key) in physical_events for key in grid
        ])
        if not present.any():
            raise ValueError(
                f'Threshold channels do not overlap the probe {probe} LFP column'
            )

        data = {}
        for row, key in enumerate(grid):
            physical_channel = None if key is None else int(key)
            channel_name = (
                f'{self.siteid}-{physical_channel:03d}{probe}'
                if physical_channel is not None
                else f'{self.siteid}-padding-{row:03d}{probe}'
            )
            data[channel_name] = physical_events.get(
                physical_channel, np.asarray([], dtype=float)
            )

        aligned = nems.tools.signal.PointProcess(
            fs=resp.fs, data=data, name='resp', recording=self.siteid,
            chans=list(data), epochs=resp.epochs.copy(),
        )
        depth_labels = np.asarray([
            '' if depth is None else depth for depth in depths
        ], dtype=object)
        return aligned, depth_labels, present, grid, depths

    @staticmethod
    def _set_channel_ticks(ax, grid, depths):
        ticks = np.arange(0, len(grid), 8)
        labels = []
        for tick in ticks:
            key = grid[tick]
            depth = depths[tick]
            labels.append('' if key is None else f'ch{key}\n{depth}um')
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=6)

    @staticmethod
    def _interpolate_ftc_gaps(image, present):
        """Fill only interior detector-channel gaps for display readability."""
        image = np.asarray(image, dtype=float).copy()
        if image.ndim != 2 or image.shape[0] != len(present):
            return image
        valid = np.flatnonzero(present)
        if valid.size < 2:
            return image
        image[:valid[0], :] = np.nan
        image[valid[-1] + 1:, :] = np.nan
        for row in range(valid[0] + 1, valid[-1]):
            if not present[row]:
                image[row, :] = np.nan
        for column in range(image.shape[1]):
            image[valid[0]:valid[-1] + 1, column] = np.interp(
                np.arange(valid[0], valid[-1] + 1), valid,
                image[valid, column]
            )
        return image

    @staticmethod
    def _evoked_spike_metric(resp, response_len=0.1):
        """Return mean post-tone minus prestimulus activity per channel."""
        raster = resp.rasterize()
        epochs = raster.epochs
        pre = epochs.loc[epochs['name'] == 'PreStimSilence']
        if pre.empty:
            raise ValueError('No PreStimSilence epochs available')
        pre_bins = int(np.nanmean(pre['end'] - pre['start']) * raster.fs)
        response_bins = max(int(response_len * raster.fs), 1)
        stim_names = sorted(
            name for name in epochs['name'].unique()
            if str(name).startswith('STIM_')
        )
        if not stim_names:
            raise ValueError('No stimulus epochs available')

        changes = []
        for stim_name in stim_names:
            extracted = raster.extract_epoch(stim_name)
            if extracted.shape[0] == 0:
                continue
            end_bin = min(pre_bins + response_bins, extracted.shape[2])
            if end_bin <= pre_bins or pre_bins == 0:
                continue
            baseline = extracted[:, :, :pre_bins].mean(axis=(0, 2))
            evoked = extracted[:, :, pre_bins:end_bin].mean(axis=(0, 2))
            changes.append(evoked - baseline)
        if not changes:
            raise ValueError('Stimulus epochs contain no usable samples')
        return np.nanmean(np.stack(changes), axis=0)

    @staticmethod
    def _remove_ftc_axes(canvas):
        for axis in getattr(canvas, '_ftc_axes', []):
            try:
                canvas.figure.delaxes(axis)
            except (KeyError, ValueError):
                pass
        canvas._ftc_axes = []

    @staticmethod
    def _sorted_channel_info(cellids, probe, probe_count):
        selected = [cellid for cellid in cellids if f'-{probe}-' in cellid]
        if not selected and probe_count == 1:
            selected = list(cellids)
        depths = np.asarray([int(cellid.split('-')[-2]) for cellid in selected])
        return selected, depths

    @staticmethod
    def _load_binary_threshold_times(openephys_folder, trial_start, siteid):
        """Read OpenEphys threshold times without loading spike waveforms."""
        trial_ttls = io.load_trial_starts_openephys_master(openephys_folder)
        if len(trial_ttls) == 0:
            raise ValueError('No OpenEphys trial TTLs found')
        adjustment = float(trial_start) - float(trial_ttls[0])

        spike_dict = {}
        structure_files = Path(openephys_folder).glob(
            'Record Node */experiment*/recording*/structure.oebin'
        )
        for structure_file in structure_files:
            with structure_file.open() as handle:
                structure = json.load(handle)
            gui_version = float('.'.join(structure['GUI version'].split('.')[:2]))
            for spike_info in structure.get('spikes', []):
                folder_key = 'folder' if gui_version >= 0.6 else 'folder_name'
                sample_name = 'sample_numbers.npy' if gui_version >= 0.6 else 'spike_times.npy'
                sample_file = structure_file.parent / 'spikes' / spike_info[folder_key] / sample_name
                if not sample_file.exists():
                    continue

                electrode = spike_info['name'].split()[-1].rjust(4, '0')
                cellid = f'{siteid}-{electrode}'
                eventtimes = np.asarray(np.load(sample_file, mmap_mode='r')).reshape(-1)
                eventtimes = eventtimes / float(spike_info['sample_rate']) + adjustment
                eventtimes = eventtimes[eventtimes > 0]
                if cellid in spike_dict:
                    spike_dict[cellid] = np.concatenate((spike_dict[cellid], eventtimes))
                else:
                    spike_dict[cellid] = eventtimes

        if not spike_dict:
            raise ValueError('No binary OpenEphys threshold events found')
        return spike_dict

    def _load_threshold_response(self, ex, fs):
        aligned_events = ex.get_baphy_events(
            correction_method='openephys', rasterfs=fs
        )
        exptparams = ex.get_baphy_exptparams()
        globalparams = ex.get_baphy_globalparams()
        epochs = [
            baphy_experiment.baphy_events_to_epochs(
                events, params, globals_, file_index, rasterfs=fs
            )
            for file_index, (events, params, globals_) in enumerate(
                zip(aligned_events, exptparams, globalparams)
            )
        ]
        threshold_dicts = []
        for raw_path, events in zip(ex.openephys_folder, aligned_events):
            trial_starts = events.loc[
                events['name'].astype(str).str.startswith('TRIALSTART'), 'start'
            ]
            if trial_starts.empty:
                raise ValueError('No TRIALSTART events available for FTC alignment')
            threshold_dicts.append(
                self._load_binary_threshold_times(
                    raw_path, trial_starts.iloc[0], ex.siteid
                )
            )

        signals = [
            nems.tools.signal.PointProcess(
                fs=fs, data=spikes, name='resp', recording=ex.siteid,
                chans=list(spikes), epochs=events,
            )
            for spikes, events in zip(threshold_dicts, epochs)
        ]
        response = signals[0]
        for signal in signals[1:]:
            response = response.append_time(signal)
        return response

    def FTC_heatmap_plot(self, canvas, parmfile, source='Threshold spikes'):
        # remove current artists
        self.clear_canvas(canvas)
        self._remove_ftc_axes(canvas)
        self.remove_ax(canvas)
        axes = canvas.figure.subplots(
            1, 2, sharey=True, gridspec_kw={'width_ratios': [5, 1]}
        )
        canvas._ftc_axes = list(axes)
        canvas.ax = axes[0]
        metric_ax = axes[1]
        canvas.figure.subplots_adjust(wspace=0.04)

        try:
            siteid = self.parmfile[:7]
            print(f"Loading {parmfile}")
            ex = baphy_experiment.BAPHYExperiment(parmfile=parmfile)
            fs = 100
            probe = self.probe[self.current_probe_index][-1:]
            present = None
            channel_grid = None
            channel_depths = None

            if source == 'Threshold spikes':
                cache_key = tuple(str(path) for path in ex.parmfile)
                if cache_key not in self.ftc_response_cache:
                    self.ftc_response_cache[cache_key] = (
                        self._load_threshold_response(ex, fs)
                    )
                resp = self.ftc_response_cache[cache_key].copy()
                resp, depths, present, channel_grid, channel_depths = (
                    self._align_threshold_to_lfp(resp, probe)
                )
                cellids = resp.chans
            elif source == 'Raw MUA':
                rec = ex.get_recording(
                    mua=True, raw=False, pupil=False, resp=False, stim=False,
                    recache=False, rawchans=None, rasterfs=fs, muabp=(500, 5000)
                )
                resp = rec['mua'].copy()
                cellids = [c for c in resp.chans if f'{probe}-' in c]
                resp = resp.extract_channels(cellids)
                depths = np.asarray([int(c.split('-')[-1]) for c in cellids])
            elif source == 'Sorted units':
                rec = ex.get_recording(loadkey=f'psth.fs{fs}')
                resp = rec['resp'].copy()
                cellids, depths = self._sorted_channel_info(
                    resp.chans, probe, len(self.probe)
                )
                resp = resp.extract_channels(cellids)
            else:
                raise ValueError(f"Unknown FTC source: {source}")

            if not cellids:
                raise ValueError(f"No {source} channels found for probe {probe}")

            resp = self._prepare_ftc_response(resp)
            evoked_metric = self._evoked_spike_metric(resp)
            ftc_heatmap(siteid=siteid, resp=resp, depths=depths, probe=probe,
                        smooth_win=3, snr_norm=(source == 'Raw MUA'),
                        ax=canvas.ax)
            canvas.ax.set_aspect('auto')
            if present is not None:
                heatmap = np.asarray(canvas.ax.images[-1].get_array()).copy()
                interpolate = True
                try:
                    interpolate = self._view.ui.FTCinterpolateCheckBox.isChecked()
                except AttributeError:
                    pass
                if interpolate:
                    heatmap = self._interpolate_ftc_gaps(heatmap, present)
                else:
                    heatmap[~present, :] = np.nan
                cmap = canvas.ax.images[-1].get_cmap().copy()
                cmap.set_bad(canvas.ax.get_facecolor())
                canvas.ax.images[-1].set_cmap(cmap)
                canvas.ax.images[-1].set_data(np.ma.masked_invalid(heatmap))
                self._set_channel_ticks(
                    canvas.ax, channel_grid, channel_depths
                )
                evoked_metric[~present] = np.nan
            y = np.arange(len(evoked_metric))
            metric_ax.axvline(0, color='0.3', linewidth=1.0)
            metric_ax.plot(evoked_metric, y, color='#8b0000', linewidth=1.8)
            metric_ax.fill_betweenx(y, 0, evoked_metric, color='#c62828', alpha=0.55)
            metric_ax.set_xlabel('Evoked Δ\nspikes/100 ms', fontsize=7)
            metric_ax.tick_params(axis='x', labelsize=6)
            metric_ax.tick_params(axis='y', left=False, labelleft=False)
            metric_ax.set_ylim(canvas.ax.get_ylim())
            canvas.ax.set_title(f'{siteid} Probe {probe}: {source}')
        except Exception as error:
            print(f"Can't load {source} FTC: {error}")
            canvas.ax.set_axis_off()
            canvas.ax.text(0.5, 0.5, f'{source} FTC unavailable',
                           ha='center', va='center', transform=canvas.ax.transAxes)
            metric_ax.set_axis_off()

        # draw to canvas
        self._view.ui.templateCanvas.canvas.draw()

    def parmfile_raw_path(self, parmfile):
        animalid = self._view.ui.animalcomboBox.currentText()
        site_root = self.raw_data_path / animalid / self.siteid
        legacy_path = site_root / 'raw' / parmfile[:9]
        psi_path = site_root / parmfile / 'raw'
        return legacy_path if legacy_path.exists() else psi_path

    def update_FTC_parmfile(self):
        candidate_key = (self.siteid, self.parmfile)
        if (self.ftc_candidate_for_bnb == candidate_key
                and self.ftc_parmfile_path is not None):
            return True
        return self.find_compatible_ftc_parmfile(self.parmfile) is not None

    def FTC_unavailable_plot(self, canvas, message):
        self.clear_canvas(canvas)
        self._remove_ftc_axes(canvas)
        self.remove_ax(canvas)
        canvas.ax = canvas.fig.add_subplot(111)
        canvas.ax.set_axis_off()
        canvas.ax.text(0.5, 0.5, message, ha='center', va='center',
                       transform=canvas.ax.transAxes)
        canvas.draw()


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
        self._remove_ftc_axes(canvas)
        # reset template canvas
        self.remove_ax(self._view.ui.templateCanvas.canvas)
        self._view.ui.templateCanvas.canvas.ax = self._view.ui.templateCanvas.canvas.fig.add_subplot(111)
        self._view.ui.templateCanvas.canvas.divider = make_axes_locatable(self._view.ui.templateCanvas.canvas.ax)
        self._view.ui.templateCanvas.canvas.cax = self._view.ui.templateCanvas.canvas.divider.append_axes("right",
                                                                                                          size="5%",
                                                                                                          pad=0.05)
        self._view.ui.templateCanvas.canvas.xax = self._view.ui.templateCanvas.canvas.cax.get_xaxis()
        self._view.ui.templateCanvas.canvas.xax.set_visible(False)
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
            if self.probe_type == 'NPX':
                channel_grid, channel_depths = self._lfp_channel_grid()
                self._set_channel_ticks(canvas.ax, channel_grid, channel_depths)

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
            if self.probe_type == 'NPX':
                channel_grid, channel_depths = self._lfp_channel_grid()
                self._set_channel_ticks(canvas.ax, channel_grid, channel_depths)

        elif site_coh:
            self._view.ui.figsavelineEdit.setText(f"{self.figpathroot}/{self.parmfile[:-8]}_COH.pdf")
            # idx30 = np.where(self.freqs > 30)[0][0]
            # idx150 = np.where(self.freqs < 150)[0][-1]
            idx1 = np.where(self.freqs[self.current_probe_index] > 1)[0][0]
            idx15 = np.where(self.freqs[self.current_probe_index] < 15)[0][-1]
            gamma_cohmat = np.squeeze(self.coh[self.current_probe_index].mean(axis=0))
            im = canvas.ax.imshow(gamma_cohmat, origin='lower', aspect='auto')
            cbar = canvas.fig.colorbar(im, cax=canvas.cax)
            if self.probe_type == 'NPX':
                channel_grid, channel_depths = self._lfp_channel_grid()
                self._set_channel_ticks(canvas.ax, channel_grid, channel_depths)
                ticks = np.arange(0, len(channel_grid), 8)
                labels = [
                    '' if channel_grid[tick] is None
                    else f'ch{channel_grid[tick]}\n{channel_depths[tick]}um'
                    for tick in ticks
                ]
                canvas.ax.set_xticks(ticks)
                canvas.ax.set_xticklabels(labels, fontsize=6)
        # elif site_erp:
        #     self._view.ui.figsavelineEdit.setText(f"{self.figpathroot}/{self.parmfile[:-8]}_ERP.pdf")
        #     for i in range(len(self.erp[:, 0])):
        #         canvas.ax.plot((self.erp[i, :] + 500*i))
        canvas.ax.set_ylabel("channels")
        canvas.ax.set_title(str(self.parmfile))
        self.draw_lines(canvas.ax)


    def clear_canvas(self, canvas):
        try:
            canvas.ax.clear()
        except:
            print("canvas.ax object does not exist")
        try:
            canvas.cax.clear()
        except:
            print("canvas cax object does not exist")

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
        print('Mapping to nominal depths')
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
                channel_dict[ch] = [location_label, float(f(channel_position)), channel_position]
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
        minchanposition = np.min([int(self.channel_xy[self.current_probe_index][ch][1]) for ch in list(self.channel_xy[self.current_probe_index].keys())])
        for ch in list(self.channel_xy[self.current_probe_index].keys()):
            # get channel position in pixels
            channel_position = (int(self.channel_xy[self.current_probe_index][ch][1])-minchanposition)/column_spacing
            #channel_position = (int(self.channel_xy[self.current_probe_index][ch][1]))/column_spacing
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
                channel_dict[ch] = [location_label, int(f(channel_position*column_spacing+minchanposition)), channel_position]
            except:
                channel_dict[ch] = [location_label, location_label, channel_position]
        complete_dict = {}
        complete_dict['channel info'] = channel_dict
        complete_dict['parmfile'] = self.parmfile
        complete_dict['site area'] = site_area
        complete_dict['site area deep'] = site_area_deep

        # assign new depth mapping to depth dictionary currently in celldb if it exists
        try:
            self.load_depth_from_db()
        except:
            pass
        self.depth_mapped = self.loadedds
        self.depth_mapped[self.probe[self.current_probe_index]] = {**complete_dict, **position_memory}
        # update database loaded depths
        # self.loadedds = self.depth_mapped


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
        parmfile = self.parmfile.replace('.m','')
        fig_path = f"/auto/data/web/celldb/analysis/{animalid}/{year}"
        fig_loc = f"{fig_path}/{parmfile}.lfp_depth_markers.jpg"
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

        os.makedirs(fig_path, exist_ok=True)
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
        # load new depths to update model
        self._model.load_depth_from_db()

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
        self._view.ui.probecomboBox.blockSignals(True)
        if getSelected:
            baseNode = getSelected[0]
            probes = baseNode.text(3)
            self._model.site_probes = probes.split(", ")
            self._view.ui.probecomboBox.clear()
            self._view.ui.probecomboBox.addItems(self._model.site_probes)
            self._view.ui.probecomboBox.blockSignals(False)
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
                    item.setText(3, 'A')
                    pass
                tree_items.append(item)

        self._view.ui.siteList.insertTopLevelItems(0, tree_items)
        self._view.ui.siteList.setSortingEnabled(True)
        self._view.ui.siteList.blockSignals(False)

    def normalization(self, *args, redraw=True):
        if self._view.ui.tempmaxnormradioButton.isChecked():
            self._model.temp_normalization()
        elif self._view.ui.sitemaxnormradioButton.isChecked():
            self._model.site_normalization()
        elif self._view.ui.localnormradioButton.isChecked():
            self._model.local_normalization()
        elif self._view.ui.nonormradioButton.isChecked():
            self._model.no_normalization()
        if redraw:
            self.update_plots()

    def plot_spike_panel(self):
        canvas = self._view.ui.templateCanvas.canvas
        if self._view.ui.templateViewCheckBox.isChecked():
            self._model.template_plot(
                canvas, self._view.ui.tempPSDradioButton.isChecked()
            )
        elif self._view.ui.siteFTCcheckBox.isChecked():
            if self._model.update_FTC_parmfile():
                self._model.FTC_heatmap_plot(
                    canvas,
                    self._model.ftc_parmfile_path,
                    source=self._view.ui.FTCsourceComboBox.currentText(),
                )
            else:
                self._model.FTC_unavailable_plot(
                    canvas, 'No compatible FTC run found for this BNB session'
                )
        else:
            self._model.clear_canvas(canvas)
            canvas.ax.set_axis_off()
            canvas.draw()

    def update_spike_mode(self):
        template_visible = self._view.ui.templateViewCheckBox.isChecked()
        self._view.ui.tempPSDradioButton.setEnabled(template_visible)
        self._view.ui.tempCSDradioButton.setEnabled(template_visible)
        self._view.ui.tempPSDradioButton.setVisible(template_visible)
        self._view.ui.tempCSDradioButton.setVisible(template_visible)
        self._view.ui.siteFTCcheckBox.setVisible(not template_visible)
        self._view.ui.FTCsourceComboBox.setVisible(not template_visible)
        self._view.ui.FTCinterpolateCheckBox.setVisible(not template_visible)
        self.update_plots()

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
        self._model.bnb_raw_path = self._model.parmfile_raw_path(self._model.parmfile)
        self._model.find_compatible_ftc_parmfile(self._model.parmfile)

        self._model.site_csd_psd(parmfilepath, align=align)
        self._model.current_probe = self._view.ui.probecomboBox.currentText()
        self.update_current_probe_index()
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
        self.normalization(redraw=False)
        self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked(),
                              self._view.ui.siteCSDradioButton.isChecked(), self._view.ui.siteCOHradioButton.isChecked())
        self.plot_spike_panel()
        self._view.ui.siteCanvas.canvas.draw()
        self.lineconnect()

    def update_plots(self):
        try:
            self._model.current_probe = self._view.ui.probecomboBox.currentText()
            self.update_current_probe_index()
            self.plot_spike_panel()
            self.update_lfp_plot(update_probe=False)
        except Exception as error:
            print(f"Can't update plot. Site not loaded? {error}")

    def update_lfp_plot(self, *args, update_probe=True):
        """Redraw only the cached LFP view without rebuilding the FTC panel."""
        if args and isinstance(args[0], bool) and not args[0]:
            return
        try:
            if update_probe:
                self._model.current_probe = self._view.ui.probecomboBox.currentText()
                self.update_current_probe_index()
            self._model.site_plot(
                self._view.ui.siteCanvas.canvas,
                self._view.ui.sitePSDradioButton.isChecked(),
                self._view.ui.siteCSDradioButton.isChecked(),
                self._view.ui.siteCOHradioButton.isChecked(),
            )
            self._view.ui.siteCanvas.canvas.draw()
            self.lineconnect()
        except Exception as error:
            print(f"Can't update LFP plot. Site not loaded? {error}")

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

    def update_probe_plots(self):
        # update depth info for new probe
        # change selected probe
        self._model.current_probe = self._view.ui.probecomboBox.currentText()
        self.update_current_probe_index()
        # load depths for new probe
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
        self.update_plots()

    def template_lines(self):
        self._model.template_landmarkBoolean = self._view.ui.templatelandmarkcheckBox.isChecked()
        if self._view.ui.templateViewCheckBox.isChecked():
            self.plot_spike_panel()

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
        self._view.ui.tempCSDradioButton.toggled.connect(self.update_plots)
        self._view.ui.sitePSDradioButton.toggled.connect(self.update_lfp_plot)
        self._view.ui.siteCSDradioButton.toggled.connect(self.update_lfp_plot)
        self._view.ui.siteCOHradioButton.toggled.connect(self.update_lfp_plot)
        self._view.ui.siteFTCcheckBox.toggled.connect(self.update_plots)
        self._view.ui.FTCsourceComboBox.currentIndexChanged.connect(self.update_plots)
        self._view.ui.FTCinterpolateCheckBox.toggled.connect(self.update_plots)
        self._view.ui.templateViewCheckBox.toggled.connect(self.update_spike_mode)
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
        self._view.ui.probecomboBox.currentIndexChanged.connect(self.update_probe_plots)


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
