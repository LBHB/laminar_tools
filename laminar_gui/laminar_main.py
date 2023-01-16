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
from laminar_gui_v2 import Ui_mainWidget
from functools import partial
from pathlib import Path

class LaminarUi(QWidget):
    def __init__(self, *args, **kwargs):
        super(LaminarUi, self).__init__(*args, **kwargs)
        self.ui = Ui_mainWidget()
        self.ui.setupUi(self)
        self.show()

class LaminarModel():
    def __init__(self):
        self.all_recordings()
        self.load_template()
        self.landmarks = ['BS/1', '1/2', '2/3', '3/4', '4/5', '5/6']
        self.landmarkPosition = {border: 0 for border in self.landmarks}
        self.landmarkBoolean = {border: False for border in self.landmarks}
        self.lines = list()
        self.linedict = {}

    def all_recordings(self):
        recordings = {}
        # get animals:
        species = 'ferret'
        require_active = False
        sql = f"SELECT * FROM gAnimal WHERE lab='lbhb' and species='{species}'"
        if require_active:
            sql += " AND onschedule<2"
        dAnimal = db.pd_query(sql)
        animals = dAnimal['animal'].to_list()

        for animal in animals:
            runclass = None
            if runclass is None:
                sql = "SELECT DISTINCT gCellMaster.* FROM gCellMaster INNER JOIN gDataRaw ON gCellMaster.id=gDataRaw.masterid" + \
                      f" WHERE animal='{animal}' AND not(gDataRaw.bad) AND gCellMaster.training=0 ORDER BY gCellMaster.siteid"
            else:
                sql = "SELECT DISTINCT gCellMaster.* FROM gCellMaster INNER JOIN gDataRaw ON gCellMaster.id=gDataRaw.masterid" + \
                      f" WHERE animal='{animal}' AND not(gDataRaw.bad) AND gDataRaw.runclass='{runclass}' AND gCellMaster.training=0 ORDER BY gCellMaster.siteid"

            dSites = db.pd_query(sql)
            site_list = dSites['siteid'].to_list()

            site_dict = {}
            for siteid in site_list:
                sql = f"SELECT gDataRaw.* FROM gDataRaw WHERE cellid='{siteid}'"
                dRawFiles = db.pd_query(sql)
                parmfiles = dRawFiles['parmfile'].to_list()
                site_dict[siteid] = parmfiles

            recordings[animal] = site_dict

        self.recordings = recordings

    def load_template(self):
        template_psd = np.load("/auto/users/wingertj/code/csd_project/data/laminar_features/template/final_psd_template.npy")
        template_csd = np.load("/auto/users/wingertj/code/csd_project/data/laminar_features/template/final_csd_template.npy")
        max_power = template_psd.max(axis=0)
        self.max_power = max_power
        self.template_psd = template_psd/max_power
        self.template_csd = template_csd

    def site_csd_psd(self, parmfile):
        csd, psd = parmfile_event_lfp(parmfile)
        averaged_template, ssim_index, ssim = maximal_laminar_similarity(template=self.template_psd, image=psd, overlap=10,
                                                                         ssim_window=5, expansion=True)
        csd = [csd, ]
        psd = [psd, ]
        ssim_index = [ssim_index, ]
        nan_pad_psd, nan_pad_csd = pad_to_template(self.template_psd, psd, csd, ssim_index, already_padded=False)

        psd = np.squeeze(nan_pad_psd)
        csd = np.squeeze(nan_pad_csd)
        self.psd = psd/self.max_power
        self.csd = csd
        self.ssim = ssim

    def erase_lines(self):
        print('errasing lines...')
        while len(self.linedict.keys()) != 0:
            keys = list(self.linedict.keys())
            artists = self.linedict.pop(keys[0])
            for artist in artists:
                artist.remove()
        print('done')

    def draw_lines(self, ax):
        self.erase_lines()
        print('drawing lines...')
        for sName in self.landmarks:
            sBool = self.landmarkBoolean[sName]
            sPos = self.landmarkPosition[sName]
            if sBool:
                top, bottom = sName.split('/')
                self.linedict[sName] = [ax.axhline(sPos, color='red', linewidth=2, picker=5),
                                        ax.text(0, sPos + 0.5, top, color='orange', fontsize=10),
                                        ax.text(0, sPos - 2, bottom, color='orange', fontsize=10)]

        print('done')

    def template_plot(self, canvas, temp_psd):
        """
        plots the laminar data
        """
        print('plotting laminar data...')
        self.clear_canvas(canvas.ax)
        if temp_psd:
            canvas.ax.imshow(self.template_psd, origin='lower', aspect='auto')
        else:
            canvas.ax.imshow(self.template_csd, origin='lower', aspect='auto')

    def site_plot(self, canvas, site_psd):
        """
        plots the laminar data
        """
        print('plotting laminar data...')
        self.clear_canvas(canvas.ax)
        if site_psd:
            canvas.ax.imshow(self.psd, origin='lower', aspect='auto')
        else:
            canvas.ax.imshow(self.csd, origin='lower', aspect='auto')
        self.draw_lines(canvas.ax)

    def clear_canvas(self, axes):
        axes.clear()

class LaminarCtrl():
    def __init__(self, model, view):
        self._view = view
        self._model = model
        self.populateComboBoxes()
        self._connectSignals()

    def populateComboBoxes(self):
        self._view.ui.animalcomboBox.addItems(
            self._model.recordings)

    def update_siteComboBox(self, index):
        self._view.ui.sitecomboBox.clear()
        animal = self._view.ui.animalcomboBox.itemText(index)
        if animal:
            self._view.ui.sitecomboBox.addItems(self._model.recordings[animal])

    def update_siteList(self, index):
        self._view.ui.siteList.blockSignals(True)
        self._view.ui.siteList.setSortingEnabled(False)
        self._view.ui.siteList.clear()
        tree_items = list()

        site = self._view.ui.sitecomboBox.itemText(index)
        if site:
            files = self._model.recordings[self._view.ui.animalcomboBox.currentText()][site]
            for file in files:
                item = QTreeWidgetItem(None)
                item.setText(0, file)
                tree_items.append(item)

        self._view.ui.siteList.insertTopLevelItems(0, tree_items)
        self._view.ui.siteList.setSortingEnabled(True)
        self._view.ui.siteList.blockSignals(False)

    def load_site_csd_psd(self):
        getSelected = self._view.ui.siteList.selectedItems()
        if getSelected:
            baseNode = getSelected[0]
            getChildNode = baseNode.text(0)
            parmfile = str(getChildNode)
        siteid = self._view.ui.sitecomboBox.currentText()
        animalid = self._view.ui.animalcomboBox.currentText()
        rawpath = Path('/auto/data/daq')
        sql = f"SELECT gDataRaw.* FROM gDataRaw WHERE cellid='{siteid}'"
        dRawFiles = db.pd_query(sql)
        resppath = dRawFiles['resppath'][0]
        parmfilepath = [rawpath/animalid/siteid[:-1]/parmfile]
        self._model.site_csd_psd(parmfilepath)
        self._model.template_plot(self._view.ui.templateCanvas.canvas, self._view.ui.tempPSDradioButton.isChecked())
        self._model.site_plot(self._view.ui.siteCanvas.canvas, self._view.ui.sitePSDradioButton.isChecked())
        self._view.ui.templateCanvas.canvas.draw()
        self._view.ui.siteCanvas.canvas.draw()

    def update_site_csd_psd(self):
        template_psd_requested = self._view.ui.tempPSDradioButton.isChecked()
        site_psd_requested = self._view.ui.sitePSDradioButton.isChecked()
        self._model.template_plot(self._view.ui.templateCanvas.canvas, template_psd_requested)
        self._model.site_plot(self._view.ui.siteCanvas.canvas, site_psd_requested)
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
        self._view.ui.animalcomboBox.currentIndexChanged.connect(self.update_siteComboBox)
        self._view.ui.sitecomboBox.currentIndexChanged.connect(self.update_siteList)
        self._view.ui.sitepushButton.clicked.connect(self.load_site_csd_psd)
        self._view.ui.tempPSDradioButton.toggled.connect(self.update_site_csd_psd)
        self._view.ui.sitePSDradioButton.toggled.connect(self.update_site_csd_psd)
        # set the subset of separators to consider
        for boxName, checkBox in self._view.ui.layerCheckBoxes.items():
            checkBox.stateChanged.connect(partial(self.updatelandmarkcomboBox,
                                                  boxName, checkBox))
        self._view.ui.landmarkcomboBox.currentIndexChanged.connect(self.lineconnect)

def main():
    """Main function."""
    # Create an instance of QApplication
    laminar = QApplication(sys.argv)
    # Show the calculator's GUI
    view = LaminarUi()
    # get model functions
    model = LaminarModel()
    # create instance of the controller
    controller = LaminarCtrl(model=model, view=view)
    # Execute the calculator's main loop
    sys.exit(laminar.exec_())

if __name__ == '__main__':
    main()
