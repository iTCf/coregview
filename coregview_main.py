import sys
import os
import glob
import mne
import cortex
import datetime
import numpy as np
import pandas as pd
import os.path as op
import pymatreader as pym
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QTabWidget, QLineEdit, \
    QPushButton, QGridLayout, QComboBox, QListWidget, QSlider, QHBoxLayout, QLabel, QSpinBox, QDialog, QSizePolicy,\
    QSplitter
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont, QPixmap, QResizeEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from nilearn.plotting import plot_connectome
from coregview_info import ch185, ix185, dir_data, dir_base, dir_resources, dir_analysis
from coregview_fx import make_bip_coords, read_run_json, load_pickle, find_closest_vert


class App(QMainWindow):

    def __init__(self):
        super().__init__()

        # Geometry
        self.left = 10
        self.top = 10
        self.title = 'CoregView'
        self.width = 1650
        self.height = 1000

        sshFile = op.join(dir_resources, 'coregview.stylesheet')
        with open(sshFile, "r") as fh:
            self.setStyleSheet(fh.read())

        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Config
        self.dir_data = dir_data
        #self.dir_comments = os.path.join(dir_analysis, 'Comments')
        self.subjs = []
        self.subj = ''
        self.subj_sessions = []
        self.sess = ''
        self.hdeeg_chans = ch185
        lay = mne.channels.read_layout('EGI256')
        self.egi_pos = lay.pos[ix185, :2]
        info_epo = mne.read_epochs(os.path.join(dir_resources, 'info-epo.fif'))
        # todo: load digi from each subject instead of template
        self.mne_info = info_epo
        self.seeg_chans = []
        self.seeg_coords = []
        self.data = []
        self.seeg_lines = []
        self.current_hdeeg = 0
        self.current_seeg = 0
        self.current_time = 0
        self.evo_hdeeg = []
        self.times = []
        self.seeg_ch_info = []
        self.seeg_show = 'Amplitude'
        self.vmin_seeg = -500
        self.vmax_seeg = 500
        self.vmin_hdeeg = -20
        self.vmax_hdeeg = 20
        self.pial = {}
        self.white = {}
        self.fidu = {}
        self.bf_ready = False
        self.seeg_topo_ready = False

        for h in ['rh', 'lh']:
            self.pial[h] = mne.read_surface(os.path.join(dir_resources, '%s_norm.pial' % h))

        self.main_layout = QVBoxLayout()
        self.menu_layout = QHBoxLayout()
        self.view_layout = QGridLayout()
        self.info_layout = QHBoxLayout()
        self.edit_layout = QHBoxLayout()

        # Subjects
        self.subjs = self.fx_get_subj_names()
        self.subjBox = QComboBox(self)
        for s in ['SUBJECTS'] + self.subjs:
            self.subjBox.addItem(s)
        self.subjBox.activated[str].connect(self.on_subj_select)

        # Sessions
        self.sessBox = QComboBox(self)
        self.sessBox.addItem('SESSIONS')

        # Open Power
        # todo: open power
        self.open_power = QPushButton('Power', self)
        self.open_power.clicked.connect(self.on_open_power)

        # Open PCI
        # todo: open pci
        self.open_pci = QPushButton('PCI', self)
        self.open_pci.clicked.connect(self.on_open_pci)

        # Open STC
        # todo: open stc
        self.open_stc = QPushButton('STC', self)
        self.open_stc.clicked.connect(self.on_open_stc)

        # Open surface
        self.open_surf = QPushButton('Surface', self)
        self.open_surf.clicked.connect(self.on_open_surf)

        # Open comments
        self.open_comm = QPushButton('Comments', self)
        self.open_comm.clicked.connect(self.on_open_comm)

        # Good/bad lights
        self.hdeeg_good = QLabel()
        self.hdeeg_good.setPixmap(QPixmap(os.path.join(dir_resources, 'hdeeg_empty.png')))
        self.hdeeg_good.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.seeg_good = QLabel()
        self.seeg_good.setPixmap(QPixmap(os.path.join(dir_resources, 'seeg_empty.png')))
        self.seeg_good.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Area box
        self.area_label = QLabel()
        self.area_label.setText('Area:')
        self.area_value = QLabel()
        self.area_value.setStyleSheet('background-color: #dcdfe5;')
        self.area_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.area_value.setAlignment(Qt.AlignCenter)

        # GMPI box
        self.gmpi_label = QLabel()
        self.gmpi_label.setText('GMPI:')
        self.gmpi_value = QLabel()
        self.gmpi_value.setStyleSheet('background-color: #dcdfe5;')

        self.gmpi_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.gmpi_value.setAlignment(Qt.AlignCenter)

        # todo: add stimulation site info
        # Stim area box
        self.st_area_label = QLabel()
        self.st_area_label.setText('Stim Area:')
        self.st_area_value = QLabel()
        self.st_area_value.setStyleSheet('background-color: #dcdfe5;')
        self.st_area_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.st_area_value.setAlignment(Qt.AlignCenter)

        # Stim GMPI box
        self.st_gmpi_label = QLabel()
        self.st_gmpi_label.setText('Stim GMPI:')
        self.st_gmpi_value = QLabel()
        self.st_gmpi_value.setStyleSheet('background-color: #dcdfe5;')

        self.st_gmpi_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.st_gmpi_value.setAlignment(Qt.AlignCenter)

        # SEEG show
        self.seeg_show_but = QComboBox()
        for s in ['Amplitude', 'Z-Score', 'Absolute']:
            self.seeg_show_but.addItem(s)
        self.seeg_show_but.activated[str].connect(self.on_change_seeg_show)

        # LIMS
        lims_hdeeg_label = QLabel()
        lims_hdeeg_label.setText('HDEEG:')
        lims_hdeeg_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        lims_seeg_label = QLabel()
        lims_seeg_label.setText('SEEG:')
        lims_seeg_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        lims_splitter = QSplitter(Qt.Vertical)

        # HDEEG Topo clims
        vmin_hdeeg_topo_label = QLabel()
        vmin_hdeeg_topo_label.setText('topo min:')
        vmin_hdeeg_topo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        vmax_hdeeg_topo_label = QLabel()
        vmax_hdeeg_topo_label.setText('topo max:')
        vmax_hdeeg_topo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.vmin_hdeeg_topo_box = QSpinBox()
        self.vmin_hdeeg_topo_box.setRange(-200, 0)
        self.vmin_hdeeg_topo_box.setSingleStep(10)
        self.vmin_hdeeg_topo_box.setValue(self.vmin_hdeeg)
        self.vmin_hdeeg_topo_box.valueChanged.connect(self.on_change_lims_topo_hdeeg)

        self.vmax_hdeeg_topo_box = QSpinBox()
        self.vmax_hdeeg_topo_box.setRange(0, 200)
        self.vmax_hdeeg_topo_box.setSingleStep(10)
        self.vmax_hdeeg_topo_box.setValue(self.vmax_hdeeg)
        self.vmax_hdeeg_topo_box.valueChanged.connect(self.on_change_lims_topo_hdeeg)

        # SEEG Topo clims
        vmin_seeg_topo_label = QLabel()
        vmin_seeg_topo_label.setText('topo min:')
        vmin_seeg_topo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        vmax_seeg_topo_label = QLabel()
        vmax_seeg_topo_label.setText('topo max:')
        vmax_seeg_topo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.vmin_seeg_topo_box = QSpinBox()
        self.vmin_seeg_topo_box.setRange(-5000, 0)
        self.vmin_seeg_topo_box.setValue(self.vmin_seeg)
        self.vmin_seeg_topo_box.setSingleStep(100)
        self.vmin_seeg_topo_box.valueChanged.connect(self.on_change_lims_topo_seeg)

        self.vmax_seeg_topo_box = QSpinBox()
        self.vmax_seeg_topo_box.setRange(0, 5000)
        self.vmax_seeg_topo_box.setValue(self.vmax_seeg)
        self.vmax_seeg_topo_box.setSingleStep(100)
        self.vmax_seeg_topo_box.valueChanged.connect(self.on_change_lims_topo_seeg)

        # HDEEG BF clims
        vmin_hdeeg_label = QLabel()
        vmin_hdeeg_label.setText('min:')
        vmin_hdeeg_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        vmax_hdeeg_label = QLabel()
        vmax_hdeeg_label.setText('max:')
        vmax_hdeeg_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.vmin_hdeeg_box = QSpinBox()
        self.vmin_hdeeg_box.setRange(-200, 0)
        self.vmin_hdeeg_box.setSingleStep(10)
        self.vmin_hdeeg_box.setValue(self.vmin_hdeeg)
        self.vmin_hdeeg_box.valueChanged.connect(self.on_change_lims_hdeeg)

        self.vmax_hdeeg_box = QSpinBox()
        self.vmax_hdeeg_box.setRange(0, 200)
        self.vmax_hdeeg_box.setSingleStep(10)
        self.vmax_hdeeg_box.setValue(self.vmax_hdeeg)
        self.vmax_hdeeg_box.valueChanged.connect(self.on_change_lims_hdeeg)

        # SEEG BF clims
        vmin_seeg_label = QLabel()
        vmin_seeg_label.setText('min:')
        vmin_seeg_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        vmax_seeg_label = QLabel()
        vmax_seeg_label.setText('max:')
        vmax_seeg_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.vmin_seeg_box = QSpinBox()
        self.vmin_seeg_box.setRange(-10000, 0)
        self.vmin_seeg_box.setValue(self.vmin_seeg)
        self.vmin_seeg_box.setSingleStep(100)
        self.vmin_seeg_box.valueChanged.connect(self.on_change_lims_seeg)

        self.vmax_seeg_box = QSpinBox()
        self.vmax_seeg_box.setRange(0, 10000)
        self.vmax_seeg_box.setValue(self.vmax_seeg)
        self.vmax_seeg_box.setSingleStep(100)
        self.vmax_seeg_box.valueChanged.connect(self.on_change_lims_seeg)

        # Layouts
        self.menu_layout.addWidget(self.subjBox)
        self.menu_layout.addWidget(self.sessBox)
        self.menu_layout.addWidget(self.open_power)
        self.menu_layout.addWidget(self.open_pci)
        self.menu_layout.addWidget(self.open_stc)
        self.menu_layout.addWidget(self.open_surf)
        self.menu_layout.addWidget(self.open_comm)
        self.menu_layout.addWidget(self.hdeeg_good)
        self.menu_layout.addWidget(self.seeg_good)

        self.info_layout.addWidget(self.area_label)
        self.info_layout.addWidget(self.area_value)
        self.info_layout.addWidget(self.gmpi_label)
        self.info_layout.addWidget(self.gmpi_value)
        self.info_layout.addWidget(self.st_area_label)
        self.info_layout.addWidget(self.st_area_value)
        self.info_layout.addWidget(self.st_gmpi_label)
        self.info_layout.addWidget(self.st_gmpi_value)
        self.info_layout.addWidget(self.seeg_show_but)

        self.edit_layout.addWidget(lims_hdeeg_label)
        self.edit_layout.addWidget(vmin_hdeeg_label)
        self.edit_layout.addWidget(self.vmin_hdeeg_box)
        self.edit_layout.addWidget(vmax_hdeeg_label)
        self.edit_layout.addWidget(self.vmax_hdeeg_box)

        self.edit_layout.addWidget(vmin_hdeeg_topo_label)
        self.edit_layout.addWidget(self.vmin_hdeeg_topo_box)
        self.edit_layout.addWidget(vmax_hdeeg_topo_label)
        self.edit_layout.addWidget(self.vmax_hdeeg_topo_box)

        self.edit_layout.addWidget(lims_splitter)

        self.edit_layout.addWidget(lims_seeg_label)
        self.edit_layout.addWidget(vmin_seeg_label)
        self.edit_layout.addWidget(self.vmin_seeg_box)
        self.edit_layout.addWidget(vmax_seeg_label)
        self.edit_layout.addWidget(self.vmax_seeg_box)

        self.edit_layout.addWidget(vmin_seeg_topo_label)
        self.edit_layout.addWidget(self.vmin_seeg_topo_box)
        self.edit_layout.addWidget(vmax_seeg_topo_label)
        self.edit_layout.addWidget(self.vmax_seeg_topo_box)

        # HDEEG selector
        self.hdeeg_select = QListWidget(self)
        for s in ['HDEEG'] + self.hdeeg_chans:
            self.hdeeg_select.addItem(s)
        self.hdeeg_select.currentItemChanged.connect(self.on_hdeeg_ch_change)

        # SEEG selector
        self.seeg_select = QListWidget(self)
        self.seeg_select.addItem('SEEG')
        self.seeg_select.currentItemChanged.connect(self.on_seeg_ch_change)

        # Butterfly HDEEG
        self.bf_hdeeg = WidgetPlot(title='HDEEG')

        # Topo HDEEG
        self.topo_hdeeg = TopoPlot(data=None, pos=self.mne_info.info, title='%s ms' % self.current_time)

        # Butterfly SEEG
        self.bf_seeg = WidgetPlot(title='SEEG')

        # Topo SEEG
        # self.seeg_locs = SeegLocs()

        # Scroll time
        self.slider_widget = SliderWidget()
        self.slider_widget.slider_time.valueChanged.connect(self.on_slider_change)
        # todo: correct slider size to match butterfly plots

        # View layout
        self.view_layout.addWidget(self.hdeeg_select, 0, 0, 1, 1)
        self.view_layout.addWidget(self.seeg_select, 2, 0, 1, 1)
        self.view_layout.addWidget(self.bf_hdeeg, 0, 1, 1, 1)
        self.view_layout.addWidget(self.slider_widget, 1, 1, 1, 1)
        self.view_layout.addWidget(self.bf_seeg, 2, 1, 1, 1)
        self.view_layout.addWidget(self.topo_hdeeg, 0, 2, 1, 1)
        # self.view_layout.addWidget(self.seeg_locs, 14, 12, 10, 10)

        self.view_layout.setColumnStretch(0, 1)
        self.view_layout.setColumnStretch(1, 6)
        self.view_layout.setColumnStretch(2, 4)
        # self.view_layout.setColumnStretch(2, 4)
        # self.hdeeg_select.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        # self.seeg_select.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.main_layout.addLayout(self.menu_layout)
        self.main_layout.addLayout(self.view_layout)
        self.main_layout.addLayout(self.info_layout)
        self.main_layout.addLayout(self.edit_layout)
        wid.setLayout(self.main_layout)

        print(self.subjs)

        # Initialize
        self.init_ui()

    def init_ui(self):
        print('Initializing UI')
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    def on_subj_select(self, subj):
        if subj != 'SUBJECTS':
            subj = str(subj)
            self.subj = subj

            files = glob.glob(os.path.join(self.dir_data, subj, '%s_*_ieeg-epo.fif' % subj))
            fnames = [os.path.split(f)[-1] for f in files]
            sessions = [f.replace('%s_' % subj, '') for f in fnames]
            sessions = [s.replace('_ieeg-epo.fif', '') for s in sessions]

            self.subj_sessions = sessions
            self.subj_sessions.sort()

            fname_seeg_coords = op.join(dir_base, self.subj, 'ieeg', f'{self.subj}_task-ccepcoreg_space-MNI152NLin2009aSym_electrodes.tsv')
            self.seeg_coords = pd.read_csv(fname_seeg_coords, sep='\t')

            self.sessBox.clear()
            self.sessBox.addItem('SESSIONS')

            for s in self.subj_sessions:
                self.sessBox.addItem(s)
            self.sessBox.activated[str].connect(self.on_sess_select)

    def on_sess_select(self, sess):
        if sess != 'SESSIONS':

            fname_ses_info = op.join(dir_base, self.subj, 'ieeg', f'{self.subj}_{sess}_raw.json')
            self.ses_info = read_run_json(fname_ses_info)
            self.stim_ch = self.ses_info['Description'].split(' ')[2]
            self.stim_cond = self.ses_info['Description'].split(' ')[3]
            self.bf_ready = False

            # Load Data
            sess = str(sess)
            self.sess = sess
            self.fx_load_data(sess)
            self.seeg_chans = list(self.data['seeg'].ch_names)
            # self.bad_chans_ix = self.data['seeg'].info['bads']
            self.bad_chans = self.data['seeg'].info['bads']

            # self.seeg_chans = [c for c, b in zip(self.seeg_chans, self.bad_chans_ix) if b == 0]

            ch_info_bip = make_bip_coords(self.seeg_coords)
            # stim_ch = self.sess.split('_')[0]

            # stim info
            # st_area = ch_info_bip.loc[ch_info_bip.name == self.stim_ch].area.values[0]
            # self.st_area_value.setText(st_area)
            #
            # st_gmpi = ch_info_bip.loc[ch_info_bip.name == self.stim_ch].gmpi.values[0]
            # self.st_gmpi_value.setText(str(st_gmpi))

            self.stim_coords_mri_norm = ch_info_bip.loc[ch_info_bip.label == self.stim_ch][['x', 'y', 'z']].values
            # self.stim_coords_surf_norm = ch_info_bip.loc[ch_info_bip.label == self.stim_ch][['x', 'y', 'z']].values[0] #todo
            # self.stim_coords_surf = ch_info_bip.loc[ch_info_bip.name == stim_ch][['x_surf', 'y_surf', 'z_surf']].values[0]

            # clean channels and sort by oder in data
            ch_info_bip = ch_info_bip.loc[ch_info_bip.label.isin(self.seeg_chans)]
            ch_info_bip['label'] = pd.Categorical(ch_info_bip['label'], self.seeg_chans)
            ch_info_bip = ch_info_bip.sort_values('label')

            self.seeg_ch_info = ch_info_bip

            # Add channels
            self.seeg_select.clear()
            for s in ['SEEG'] + self.seeg_chans:
                self.seeg_select.addItem(s)

            self.times = (self.data['seeg'].times*1e3).astype(int)

            # Plot HDEEG Butterfly
            self.evo_hdeeg, _, _ = self.make_bf(kind='HDEEG')
            self.bf_hdeeg.canvas.axes.clear()

            self.hdeeg_lines = self.bf_hdeeg.canvas.plot(self.times, self.evo_hdeeg, names=self.hdeeg_chans, title='HDEEG')

            # Cache for blit when slider changes
            self.bf_hdeeg_back = self.bf_hdeeg.canvas.copy_from_bbox(self.bf_hdeeg.canvas.axes.bbox)

            ylim_hdeeg = self.bf_hdeeg.canvas.axes.get_ylim()
            self.vline_hdeeg = self.bf_hdeeg.canvas.axes.plot([0, 0],
                                                              [ylim_hdeeg[0],
                                                               ylim_hdeeg[1]],
                                                              'k--')
            self.bf_hdeeg.canvas.axes.set_title('HDEEG')
            self.bf_hdeeg.canvas.axes.set_ylabel(r'Amplitude ($\mu$V)')
            self.bf_hdeeg.canvas.axes.set_xlabel(r'Time (ms)')
            self.bf_hdeeg.canvas.draw()

            self.vmax_hdeeg_box.setValue(int(self.vmax_hdeeg))
            self.vmin_hdeeg_box.setValue(int(self.vmin_hdeeg))

            # Plot SEEG Butterfly
            self.evo_seeg, self.evo_z_seeg, self.evo_abs_seeg = self.make_bf(kind='SEEG_bipolar', bad_trials=None)

            self.bf_seeg.canvas.axes.clear()

            if self.seeg_show == 'Amplitude':
                self.seeg_lines = self.bf_seeg.canvas.plot(self.times, self.evo_seeg, names=self.seeg_chans,
                                                           title='SEEG', seeg_show=self.seeg_show)
            elif self.seeg_show == 'Z-Score':
                self.seeg_lines = self.bf_seeg.canvas.plot(self.times, self.evo_z_seeg, names=self.seeg_chans,
                                                           title='SEEG', seeg_show=self.seeg_show)
            elif self.seeg_show == 'Absolute':
                self.seeg_lines = self.bf_seeg.canvas.plot(self.times, self.evo_abs_seeg, names=self.seeg_chans,
                                                           title='SEEG', seeg_show=self.seeg_show)

            # Cache for blit when slider changes
            self.bf_seeg_back = self.bf_seeg.canvas.copy_from_bbox(self.bf_seeg.canvas.axes.bbox)

            ylim_seeg = self.bf_seeg.canvas.axes.get_ylim()
            self.vline_seeg = self.bf_seeg.canvas.axes.plot([0, 0],
                                                            [ylim_seeg[0],
                                                             ylim_seeg[1]],
                                                            'k--')

            self.bf_seeg.canvas.draw()

            self.vmax_seeg_box.setValue(int(self.vmax_seeg))
            self.vmin_seeg_box.setValue(int(self.vmin_seeg))

            self.bf_ready = True  # plot is ready

            # Plot SEEG Locs
            self.seeg_coords = self.seeg_ch_info[['x', 'y', 'z']].values
            self.seeg_locs = SeegLocs(coords=self.seeg_coords, stim_coords=self.stim_coords_mri_norm,
                                      seeg_show=self.seeg_show)
            self.view_layout.addWidget(self.seeg_locs, 2, 2, 1, 1)
            self.seeg_topo_ready = True
            # Set slider range
            self.slider_widget.set_lims(self.times.min(), self.times.max())

            # if self.data['METADATA']['hdeeg_good_bad'] == 'Good': #todo
            #     self.hdeeg_good.setPixmap(QPixmap(os.path.join(dir_local,
            #                                                    'resources', 'hdeeg_good.png')))
            # else:
            #     self.hdeeg_good.setPixmap(QPixmap(os.path.join(dir_local,
            #                                                    'resources', 'hdeeg_bad.png')))
            #
            # if self.data['METADATA']['seeg_good_bad'] == 'Good':
            #     self.seeg_good.setPixmap(QPixmap(os.path.join(dir_local,
            #                                                   'resources', 'seeg_good.png')))
            # else:
            #     self.seeg_good.setPixmap(QPixmap(os.path.join(dir_local,
            #                                                   'resources', 'seeg_bad.png')))
            self.bf_ready = True

    def on_hdeeg_ch_change(self, new_ch):
        new_ch = new_ch.text()
        if new_ch == 'HDEEG':
            self.hdeeg_lines[self.current_hdeeg].set_c('tab:orange')
            self.hdeeg_lines[self.current_hdeeg].set_linewidth(1)
            self.hdeeg_lines[self.current_hdeeg].set_alpha(0.5)
            self.update_blit(kind='HDEEG')

            self.bf_hdeeg.canvas.draw()
            return
        ix = self.hdeeg_chans.index(new_ch)
        print(new_ch, ix)

        self.hdeeg_lines[self.current_hdeeg].set_c('tab:orange')
        self.hdeeg_lines[self.current_hdeeg].set_linewidth(1)
        self.hdeeg_lines[self.current_hdeeg].set_alpha(0.5)
        self.hdeeg_lines[self.current_hdeeg].set_zorder(self.current_hdeeg)

        self.current_hdeeg = ix
        self.hdeeg_lines[ix].set_c('tab:green')
        self.hdeeg_lines[ix].set_linewidth(3)
        self.hdeeg_lines[ix].set_alpha(1)
        self.hdeeg_lines[ix].set_zorder(300)

        self.vline_hdeeg[0].set_alpha(0)
        self.bf_hdeeg.canvas.draw()
        self.bf_hdeeg_back = self.bf_hdeeg.canvas.copy_from_bbox(self.bf_hdeeg.canvas.axes.bbox)
        self.vline_hdeeg[0].set_alpha(1)

        self.bf_hdeeg.canvas.draw()
        # self.topo_hdeeg.plot_sel_chan(ix)

    def on_seeg_ch_change(self, new_ch):
        try:
            new_ch = new_ch.text()
        except AttributeError:
            return
        if new_ch == 'SEEG':
            self.seeg_lines[self.current_seeg].set_c('tab:orange')
            self.seeg_lines[self.current_seeg].set_linewidth(1)
            self.seeg_lines[self.current_seeg].set_alpha(0.5)

            self.update_blit(kind='SEEG')

            self.bf_seeg.canvas.draw()

            sizes = np.repeat(15, len(self.seeg_coords))
            self.seeg_locs.conn.axes['x'].ax.collections[0].set_sizes(sizes)
            self.seeg_locs.conn.axes['y'].ax.collections[0].set_sizes(sizes)
            self.seeg_locs.conn.axes['z'].ax.collections[0].set_sizes(sizes)
            self.seeg_locs.draw()

            self.area_value.setText('')
            self.gmpi_value.setText('')

            return

        ix = self.seeg_chans.index(new_ch)
        print(new_ch, ix)

        self.seeg_lines[self.current_seeg].set_c('tab:orange')
        self.seeg_lines[self.current_seeg].set_linewidth(1)
        self.seeg_lines[self.current_seeg].set_alpha(0.5)
        self.seeg_lines[self.current_seeg].set_zorder(ix)

        self.current_seeg = ix
        self.seeg_lines[ix].set_c('tab:green')
        self.seeg_lines[ix].set_linewidth(3)
        self.seeg_lines[ix].set_alpha(1)
        self.seeg_lines[ix].set_zorder(300)

        self.vline_seeg[0].set_alpha(0)
        self.bf_seeg.canvas.draw()
        self.bf_seeg_back = self.bf_seeg.canvas.copy_from_bbox(self.bf_seeg.canvas.axes.bbox)
        self.vline_seeg[0].set_alpha(1)

        self.bf_seeg.canvas.draw()

        sizes = np.repeat(15, len(self.seeg_coords))
        sizes[ix] = 100
        self.seeg_locs.conn.axes['x'].ax.collections[0].set_sizes(sizes)
        self.seeg_locs.conn.axes['y'].ax.collections[0].set_sizes(sizes)
        self.seeg_locs.conn.axes['z'].ax.collections[0].set_sizes(sizes)
        self.seeg_locs.draw()

        # labels
        # ch_area = self.seeg_ch_info.loc[self.seeg_ch_info.label == new_ch].area.values[0] # todo
        # self.area_value.setText(ch_area)
        #
        # ch_gmpi = self.seeg_ch_info.loc[self.seeg_ch_info.name == new_ch].gmpi.values[0]
        # self.gmpi_value.setText(str(ch_gmpi))

    def on_slider_change(self, val):
        self.current_time = val
        ix, = np.where(self.times == val)
        data = self.evo_hdeeg[ix, :].squeeze()
        self.topo_hdeeg.plot(data, '%s ms' % self.current_time)

        self.vline_hdeeg[0].set_xdata([val, val])
        self.bf_hdeeg.canvas.restore_region(self.bf_hdeeg_back)
        self.bf_hdeeg.canvas.axes.draw_artist(self.vline_hdeeg[0])
        self.bf_hdeeg.canvas.blit(self.bf_hdeeg.canvas.axes.bbox)

        self.vline_seeg[0].set_xdata([val, val])
        self.bf_seeg.canvas.restore_region(self.bf_seeg_back)
        self.bf_seeg.canvas.axes.draw_artist(self.vline_seeg[0])
        self.bf_seeg.canvas.blit(self.bf_seeg.canvas.axes.bbox)

        c_data = self.get_color_data()

        self.seeg_locs.plot(data=c_data, coords=self.seeg_coords, sel=None, stim_coords=None)
        # self.seeg_locs.update_cbar(self.vmin_topo_seeg, self.vmax_topo_seeg, title=self.seeg_show)

    def on_open_surf(self):
        ix, = np.where(self.times == self.current_time)
        if self.seeg_show == 'Amplitude':
            data = self.evo_seeg[ix, :].squeeze()
        elif self.seeg_show == 'Z-Score':
            data = self.evo_z_seeg[ix, :].squeeze()
        elif self.seeg_show == 'Absolute':
            data = self.evo_abs_seeg[ix, :].squeeze()

        cmap = 'viridis' if self.seeg_show == 'Absolute' else 'bwr'

        self.surfaces = Surfaces(
            parent=self,
            coords=self.seeg_coords,
            stim_coords=self.stim_coords_mri_norm,  # optional, if available
            vals=data,
            lims=[self.vmin_seeg_topo_box.value(), self.vmax_seeg_topo_box.value()],
            cmap=cmap
        )
        self.surfaces.show()

    def on_open_power(self):
        fname_power = os.path.join(dir_analysis, 'Power_Phase', '%s_%s_power.mat' % (self.subj, self.sess))
        if not os.path.isfile(fname_power):
            print('Power files not found')
            return
        self.power = pym.read_mat(fname_power)['PP']
        self.power_widget = Power(self, data=self.power)

    def on_open_pci(self):
        fname_pci_st = os.path.join(dir_analysis, 'PCI', '%s_%s-pci-st.pkl' % (self.subj, self.sess))
        fname_pci_lz = os.path.join(dir_analysis, 'PCI', '%s_%s-pci-lz.pkl' % (self.subj, self.sess))
        if (not os.path.isfile(fname_pci_st)) or (not os.path.isfile(fname_pci_lz)):
            print('PCI files not found')
            return
        self.pci_st = load_pickle(fname_pci_st)
        self.pci_lz = load_pickle(fname_pci_lz)
        self.pci_widget = PCI(self, pci_st=self.pci_st, pci_lz=self.pci_lz, times=self.times)

    def on_open_stc(self):
        fname_stc = os.path.join(dir_analysis, 'STC', 'orig', '%s_%s%s' % (self.subj, self.sess, '-stc-rh.stc'))

        if not os.path.isfile(fname_stc):
            print('STC not found')
            return
        stc = mne.read_source_estimate(fname_stc)
        stc.resample(256)
        stc.data = np.abs(stc.data)

        # subj_name = subj_name_from_lut(self.subj, fname_subj_lut) #todo
        #self.stc_plot = Surfaces(self, stim_coords=self.stim_coords_surf, stc=stc, subj_name=subj_name)
        # mne.viz.set_3d_backend('pyvista')
        # stc.plot(subject=subj_name, subjects_dir=dir_fs_subjects, hemi='both')


    def on_change_seeg_show(self, show):
        self.seeg_show = str(show)
        self.bf_seeg.canvas.axes.clear()

        if show == 'Z-Score':
            self.seeg_lines = self.bf_seeg.canvas.plot(self.times, self.evo_z_seeg, names=self.seeg_chans, title='SEEG',
                                                       seeg_show='Z-Score')
            self.bf_seeg.canvas.axes.set_ylabel('Z-Score')
            self.seeg_locs.cb1.set_label('Z-Score')

        elif show == 'Amplitude':
            self.seeg_lines = self.bf_seeg.canvas.plot(self.times, self.evo_seeg, names=self.seeg_chans, title='SEEG')
            self.bf_seeg.canvas.axes.set_ylabel(r'Amplitude ($\mu$V)')
            self.seeg_locs.cb1.set_label(r'Amplitude ($\mu$V)')

        elif show == 'Absolute':
            self.seeg_lines = self.bf_seeg.canvas.plot(self.times, self.evo_abs_seeg, names=self.seeg_chans, title='SEEG',
                                                       seeg_show='Absolute')
            self.bf_seeg.canvas.axes.set_ylabel(r'Absolute Amplitude ($\mu$V)')
            self.seeg_locs.cb1.set_label(r'Absolute Amplitude ($\mu$V)')

        ylim_seeg = self.bf_seeg.canvas.axes.get_ylim()

        self.vmin_seeg = ylim_seeg[0]
        self.vmax_seeg = ylim_seeg[1]

        self.bf_ready = False
        self.vmax_seeg_box.setValue(int(ylim_seeg[1]))
        self.vmin_seeg_box.setValue(int(ylim_seeg[0]))
        self.bf_ready = True

        self.vline_seeg = self.bf_seeg.canvas.axes.plot([self.current_time, self.current_time],
                                                        [ylim_seeg[0],
                                                         ylim_seeg[1]],
                                                        'k--')
        self.bf_seeg.canvas.axes.set_title('SEEG')
        self.bf_seeg.canvas.axes.set_xlabel(r'Time (ms)')

        self.update_blit(kind='SEEG')

        self.bf_seeg.canvas.draw()

        if self.seeg_show == 'Absolute':
            vmin_seeg_topo = ylim_seeg[0]
            vmax_seeg_topo = ylim_seeg[1]
        else:
            vmax_seeg_topo = np.max(np.abs((ylim_seeg[0], ylim_seeg[1])))
            vmin_seeg_topo = -vmax_seeg_topo

        self.seeg_topo_ready = False
        self.vmin_seeg_topo_box.setValue(int(vmin_seeg_topo))
        self.vmax_seeg_topo_box.setValue(int(vmax_seeg_topo))
        self.seeg_topo_ready = True
        c_data = self.get_color_data()
        self.seeg_locs.plot(data=c_data, coords=self.seeg_coords, sel=None, stim_coords=None, seeg_show=self.seeg_show)
        self.seeg_locs.update_cbar(self.vmin_seeg_topo_box.value(), self.vmax_seeg_topo_box.value(), title=self.seeg_show)
        self.seeg_locs.draw()

        print(self.vmin_seeg, self.vmax_seeg)

    def on_change_lims_topo_hdeeg(self):
        self.vmin_topo_hdeeg = self.vmin_hdeeg_topo_box.value()
        self.vmax_topo_hdeeg = self.vmax_hdeeg_topo_box.value()

        ix, = np.where(self.times == self.current_time)
        data = self.evo_hdeeg[ix, :].squeeze()

        self.topo_hdeeg.vmin = self.vmin_topo_hdeeg
        self.topo_hdeeg.vmax = self.vmax_topo_hdeeg
        self.topo_hdeeg.plot(data, title='%s ms' % self.current_time)
        self.topo_hdeeg.update_cbar(self.vmin_topo_hdeeg, self.vmax_topo_hdeeg)

    def on_change_lims_hdeeg(self):
        if self.bf_ready:
            self.vmin_hdeeg = self.vmin_hdeeg_box.value()
            self.vmax_hdeeg = self.vmax_hdeeg_box.value()

            self.bf_hdeeg.canvas.axes.clear()
            self.hdeeg_lines = self.bf_hdeeg.canvas.plot(self.times, self.evo_hdeeg, names=self.hdeeg_chans, title='HDEEG',
                                                         ylim=[self.vmin_hdeeg, self.vmax_hdeeg])
            self.update_blit(kind='HDEEG')
            ylim_hdeeg = self.bf_hdeeg.canvas.axes.get_ylim()
            self.vline_hdeeg = self.bf_hdeeg.canvas.axes.plot([self.current_time, self.current_time],
                                                              [ylim_hdeeg[0],
                                                               ylim_hdeeg[1]],
                                                              'k--')
            self.bf_hdeeg.canvas.draw()

    def on_change_lims_topo_seeg(self):
        if self.seeg_topo_ready:
            c_data = self.get_color_data()

            self.seeg_locs.plot(data=c_data, coords=self.seeg_coords, sel=None, stim_coords=None, seeg_show=self.seeg_show)
            self.seeg_locs.update_cbar(self.vmin_seeg_topo_box.value(), self.vmax_seeg_topo_box.value(), title=self.seeg_show)

    def on_change_lims_seeg(self):
        if self.bf_ready:
            self.vmin_seeg = self.vmin_seeg_box.value()
            self.vmax_seeg = self.vmax_seeg_box.value()

            self.bf_seeg.canvas.axes.clear()
            if self.seeg_show == 'Amplitude':
                self.seeg_lines = self.bf_seeg.canvas.plot(self.times, self.evo_seeg, names=self.seeg_chans, title='SEEG',
                                                           ylim=[self.vmin_seeg, self.vmax_seeg])
            else:
                self.seeg_lines = self.bf_seeg.canvas.plot(self.times, self.evo_z_seeg, names=self.seeg_chans, title='SEEG',
                                                           seeg_show='Z-Score', ylim=[self.vmin_seeg, self.vmax_seeg])

            self.update_blit(kind='SEEG')

            ylim_seeg = self.bf_seeg.canvas.axes.get_ylim()
            self.vline_seeg = self.bf_seeg.canvas.axes.plot([self.current_time, self.current_time],
                                                            [ylim_seeg[0],
                                                            ylim_seeg[1]],
                                                            'k--')

            self.bf_seeg.canvas.draw()

    def on_open_comm(self):
        self.comments = Comments(self, dir_comments=self.dir_comments, session=self.sess, subject=self.subj)

    def fx_get_subj_names(self):
        files = os.listdir(self.dir_data)
        subjs = []
        for f in files:
            if f.startswith('sub'):
                subjs.append(f.split('_')[0])
        subjs = list(set(subjs))
        subjs.sort()
        return subjs

    def fx_get_sess_info(self, fname):
        split = fname.split('_')
        sess_info = {}
        sess_info['subj'] = split[0]
        sess_info['ch'] = split[1]
        sess_info['cond'] = split[2]
        sess_info['intens'] = split[3]
        return sess_info

    def fx_load_data(self, sess):
        fnames = glob.glob(os.path.join(self.dir_data, self.subj,'%s_%s*' % (self.subj, sess)))
        data = {'eeg': mne.read_epochs(fnames[0], verbose=False, preload=True),
                'seeg': mne.read_epochs(fnames[1], verbose=False, preload=True)}
        data['seeg'].drop_channels(data['seeg'].info['bads'])
        data['eeg']._data *= 1e6
        data['seeg']._data *= 1e6
        self.data = {'eeg': data['eeg'], 'seeg': data['seeg']}
        del data

    def get_color_data(self):
        ix, = np.where(self.times == self.current_time)
        if self.seeg_show == 'Amplitude':
            data = self.evo_seeg[ix, :].squeeze()
        elif self.seeg_show == 'Z-Score':
            data = self.evo_z_seeg[ix, :].squeeze()
        elif self.seeg_show == 'Absolute':
            data = np.abs(self.evo_seeg[ix, :].squeeze())
        # sp = cm.get_cmap('viridis', 256) if self.seeg_show == 'Absolute' else cm.get_cmap('bwr', 256)
        sp = mpl.colormaps['viridis'] if self.seeg_show == 'Absolute' else mpl.colormaps['bwr']
        norm = mpl.colors.Normalize(vmin=self.vmin_seeg_topo_box.value(), vmax=self.vmax_seeg_topo_box.value())
        c_data = sp(norm(data))
        return c_data

    def make_bf(self, kind, bad_trials=None):
        if kind == 'SEEG_bipolar':
            epo = self.data['seeg'].get_data()
            evo_z = None
        else:
            epo = self.data['eeg'].get_data()
            evo_z = None
        evo = np.nanmean(epo, 0).T

        if kind == 'SEEG_bipolar':
            evo_z = evo.copy()

        for ix_ch, ch in enumerate(evo.T):
            baseline = ch[self.times < -0.05]
            bl_mean = np.mean(baseline)

            # z-score
            if kind == 'SEEG_bipolar':
                bl_std = np.std(baseline)
                evo_z[:, ix_ch] = (evo[:, ix_ch] - bl_mean) / bl_std

            evo[:, ix_ch] = evo[:, ix_ch] - bl_mean

        evo_abs = np.abs(evo)
        # evo_z[np.isnan(evo_z)] = 0
        return evo, evo_z, evo_abs

    def update_blit(self, kind=None):
        if kind == 'SEEG':
            self.vline_seeg[0].set_alpha(0)
            self.bf_seeg.canvas.draw()
            self.bf_seeg_back = self.bf_seeg.canvas.copy_from_bbox(self.bf_seeg.canvas.axes.bbox)
            self.vline_seeg[0].set_alpha(1)

        elif kind == 'HDEEG':
            self.vline_hdeeg[0].set_alpha(0)
            self.bf_hdeeg.canvas.draw()
            self.bf_hdeeg_back = self.bf_hdeeg.canvas.copy_from_bbox(self.bf_hdeeg.canvas.axes.bbox)
            self.vline_hdeeg[0].set_alpha(1)

    # todo: fix blit when resizing window
    # def resizeEvent(self, event):
    #     print("resize")
    #     QMainWindow.resizeEvent(self, event)
    #     if hasattr(App, 'vline_seeg'):
    #         self.update_blit('SEEG')
    #         self.update_blit('HDEEG')


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=App, width=5, height=4, dpi=100, data=None,
                 times=None, names=None, title='title'):

        self.data = data
        self.names = names
        self.times = times

        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.mpl_connect(self, s='button_press_event', func=self.on_click)
        FigureCanvas.mpl_connect(self, s='resize_event', func=self.on_resize)
        # self.axes.callbacks.connect('ylim_changed', self.on_ylim_changed)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot(times, data, names, title)

    def plot(self, times, data, names, title, seeg_show='Amplitude', ylim=None):
        # ax = self.figure.add_subplot(111)
        # ax = self.axes
        lines = None

        if data is not None:
            if title == 'HDEEG':
                lines = self.axes.plot(times, data, 'tab:orange', alpha=0.5, linewidth=0.5)
            elif title == 'SEEG':
                lines = self.axes.plot(times, data, 'tab:orange', alpha=0.6, linewidth=0.9)
            elif 'PCI' in title:
                lines = self.axes.plot(times, data, 'tab:grey', alpha=0.8, linewidth=1.5)

            if 'LZ' in title:
                self.axes.plot(times[data.argmax()], data.max(), '.', markersize=20)

            # self.axes.set_ylim([np.min(data) + np.min(data)*0.1, np.max(data) + np.max(data)*0.05])
            if ylim is None:
                ylim = [np.min(data) + np.min(data)*0.1, np.max(data) + np.max(data)*0.1]

            if title in ['HDEEG', 'SEEG']:
                pass
                #ylim[0] = (ylim[0] // 10) * 10 - 10
                #ylim[1] = (ylim[1] // 10) * 10 + 10

            if (title == 'SEEG') and (seeg_show == 'Absolute'):
                ylim[0] = 0
                ylim[1] = np.max(np.abs(data)) * 1.1

            if title == 'HDEEG':
                self.parent().parent().parent().vmin_hdeeg = ylim[0]
                self.parent().parent().parent().vmax_hdeeg = ylim[1]

            elif title == 'SEEG':
                self.parent().parent().parent().vmin_seeg = ylim[0]
                self.parent().parent().parent().vmax_seeg = ylim[1]
                # self.parent().parent().parent().vmin_seeg_box.setValue(ylim[0])
                # self.parent().parent().parent().vmax_seeg_box.setValue(ylim[0])

            self.axes.set_ylim(ylim)
            self.axes.set_xlim([np.min(times), np.max(times)])

            self.data = data
            self.names = names
            self.times = times
            self.title = title

        self.axes.set_title(title)
        if 'PCI' not in title:
            self.axes.set_ylabel(seeg_show)

        self.axes.set_xlabel(r'Time (ms)')
        self.draw()
        return lines

    def on_click(self, event):
        if self.parent().canvas.widgetlock.locked():
            return
        x = event.xdata
        y = event.ydata
        ix_t, = np.where(self.times == int(x))
        indmin = (np.abs(self.data[ix_t, :] - y)).argmin()
        # print(self.names[indmin])
        if self.title == 'SEEG':
            self.parent().parent().parent().seeg_select.setCurrentRow(indmin+1)
        elif self.title == 'HDEEG':
            self.parent().parent().parent().hdeeg_select.setCurrentRow(indmin+1)
        elif 'HDEEG - PCI' in self.title:
            self.parent().parent().pcist_hdeeg_selector.setCurrentIndex(indmin)
        elif 'SEEG - PCI' in self.title:
            self.parent().parent().pcist_seeg_selector.setCurrentIndex(indmin)

    def on_resize(self, axes):
        if self.parent().parent() is not None:
            if self.parent().parent().parent().bf_ready:
                print('resize')
                self.parent().parent().parent().update_blit(kind='HDEEG')
                self.parent().parent().parent().update_blit(kind='SEEG')


class WidgetPlot(QWidget):
    def __init__(self, data=None, times=None, names=None, title='title', *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = PlotCanvas(self, width=10, height=8, data=data,
                                 times=times, names=names, title=title)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)


class TopoPlot(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100, data=None,
                 pos=None, title='%s ms', c_title='Amplitude\n ($\mu$V)', vmin=-20, vmax=20,
                 measure='Amp', cmap='bwr'):
        fig = Figure(figsize=(width, height), dpi=dpi)
        # self.axes = fig.add_subplot(111)
        cmap = plt.cm.bwr
        self.data = data
        self.pos = pos
        self.title = title
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = cmap
        self.c_title = c_title

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        if measure == 'PCI':
            self.vmax = np.abs(data).max()
            self.vmin = -self.vmax
            self.title = 'HDEEG - PCI ST'
            self.c_title = 'Component \nweight'

        self.ax = self.figure.add_subplot(111)
        self.cax = self.figure.add_axes([0.8, 0.2, 0.025, 0.5])
        norm = mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        self.cb1 = mpl.colorbar.ColorbarBase(self.cax, cmap=self.cmap,
                                             norm=norm)
        self.cax.set_title(self.c_title)

        if data is None:
            self.data = [0] * 185

        self.plot(self.data, self.title)
        # todo: use self.title, etc?

    def plot(self, data, title):
        self.ax.cla()
        _ = mne.viz.plot_topomap(data, self.pos, vlim=(self.vmin, self.vmax),
                                 axes=self.ax,
                                 show=False, cmap=self.cmap)

        # todo: FIX topomap adjust_subplot calls gcf which creates a new figure
        self.ax.set_title(title)
        self.title = title
        self.draw()

    def update_cbar(self, vmin, vmax):
        self.cax.clear()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        self.cb1 = mpl.colorbar.ColorbarBase(self.cax, cmap='bwr',
                                             norm=norm)
        self.cax.set_title(self.c_title)
        self.draw()

    # todo: mark selected channel on topomap
    # def plot_sel_chan(self, ix):
    #     pos = self.pos[ix]
    #     self.ax.plot(pos[1], pos[0], '*')
    #     self.draw()
    #     print(pos)


class SliderWidget(QWidget):
    def __init__(self, start_val=0, min_val=-300, max_val=700, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QHBoxLayout())
        self.min_label = QLabel()
        self.min_label.setText(str(min_val))
        self.max_label = QLabel()
        self.max_label.setText(str(max_val))

        self.slider_time = QSlider(Qt.Horizontal)
        self.slider_time.setMinimum(min_val)
        self.slider_time.setMaximum(max_val)
        self.slider_time.setValue(start_val)
        self.slider_time.setTickInterval(50)
        self.slider_time.setSingleStep(10)
        self.slider_time.autoFillBackground()

        self.layout().addWidget(self.min_label)
        self.layout().addWidget(self.slider_time)
        self.layout().addWidget(self.max_label)

    def set_lims(self, min_v, max_v):
        self.min_label.setText(str(min_v))
        self.max_label.setText(str(max_v))
        self.slider_time.setMinimum(min_v)
        self.slider_time.setMaximum(max_v)


class SeegLocs(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100, data=None,
                 coords=None, sel=None, stim_coords=None, seeg_show='Amplitude'):
        fig = Figure(figsize=(width, height), dpi=dpi)
        # self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot(data, coords, sel, stim_coords, seeg_show)

    def plot(self, data, coords, sel, stim_coords, seeg_show='Amplitude'):
        self.cmap = 'bwr' if seeg_show in ['Amplitude', 'Z-Score'] else 'viridis'
        self.cmap = plt.cm.bwr
        adj_mat = np.zeros((len(coords), len(coords)))
        if data is None:
            self.ax = self.figure.add_subplot(111)
            self.conn = plot_connectome(adj_mat, node_coords=coords, node_color='tab:gray',
                                        display_mode='ortho', node_size=15, axes=self.ax, black_bg=False)
            self.conn.add_markers(stim_coords, marker_size=50, marker_color='g', marker='*')

            self.cax = self.figure.add_axes([0.27, 0.1, 0.5, 0.025])
            norm = mpl.colors.Normalize(vmin=-500, vmax=500)
            self.cb1 = mpl.colorbar.ColorbarBase(self.cax, cmap=self.cmap,
                                            norm=norm, orientation='horizontal')
            self.cb1.set_label(seeg_show)

        else:
            self.conn.axes['x'].ax.collections[0].set_color(data)
            self.conn.axes['y'].ax.collections[0].set_color(data)
            self.conn.axes['z'].ax.collections[0].set_color(data)

            if sel is not None:
                sizes = np.repeat(30, len(coords))
                sizes[sel] = 200
                self.conn.axes['x'].ax.collections[0].set_sizes(sizes)
                self.conn.axes['y'].ax.collections[0].set_sizes(sizes)
                self.conn.axes['z'].ax.collections[0].set_sizes(sizes)

            self.draw()
        # todo: add plot navigation (zoom, save, etc) to seeg locs

    def update_cbar(self, vmin, vmax, title):
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        self.cax.clear()
        self.cb1 = mpl.colorbar.ColorbarBase(self.cax, cmap=self.cmap,
                                             norm=norm, orientation='horizontal')
        self.cb1.set_label(title)
        self.draw()


class Surfaces(QDialog):
    def __init__(self, parent=None, coords=None, stim_coords=None, vals=None, lims=None, pials=None, subj_name=None, cmap='bwr'):
        super(Surfaces, self).__init__(parent)
        self.setGeometry(500, 50, 800, 800)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # PyVista Qt interactor
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        load_hemis = ['lh', 'rh']
        pial_surfs = {}
        for h in load_hemis:
            ndim_vect = np.repeat(3, len(parent.pial[h][1])).reshape(-1, 1)
            pial_surf = pv.PolyData(parent.pial[h][0], np.hstack([ndim_vect, parent.pial[h][1]]))
            pial_surfs[h] = pial_surf

        for h in load_hemis:
            self.plotter.add_mesh(pial_surfs[h], color='grey', opacity=0.4, show_edges=False)
            pts = self.plotter.add_points(coords,scalars=vals, rgb=False, render_points_as_spheres=True,
                                          point_size=25, color='k', cmap=cmap)
            pts.mapper.SetScalarRange(lims[0], lims[1])
            self.plotter.add_scalar_bar(mapper=pts.mapper)
        self.plotter.show()
        pass

        # # Plot pial surface (must be a PyVista PolyData)
        #     if isinstance(pial, pv.PolyData):
        #         self.plotter.add_mesh(pial, color='white', opacity=0.3, show_edges=False)
        #
        #     # Add electrode coordinates
        #     if coords is not None:
        #         points = pv.PolyData(coords)
        #         scalars = vals if vals is not None else None
        #         self.plotter.add_mesh(points, render_points_as_spheres=True,
        #                               point_size=10, scalars=scalars,
        #                               cmap=cmap, clim=lims)
        #
        #     if stim_coords is not None:
        #         stim_points = pv.PolyData(stim_coords)
        #         self.plotter.add_mesh(stim_points, color='red',
        #                               render_points_as_spheres=True, point_size=12)
        #
        # else:
        #     print("STC-based visualization not yet implemented with PyVista")
        #
        # self.plotter.show()



# class SurfViz(HasTraits): #todo (pyvista)
#     def __init__(self, pial, coords, stim_coords, vals, lims, cmap='bwr'):
#         super(SurfViz, self).__init__()
#         coords['vals'] = vals
#         self.h_coords = {'lh': coords.loc[coords['x_norm_surf'] < 0].copy(),
#                          'rh': coords.loc[coords['x_norm_surf'] > 0].copy()}
#         self.stim_coords = stim_coords
#         self.pial = pial
#         self.vals = vals
#         self.lims = lims
#         self.cmap = cmap
#     scene = Instance(MlabSceneModel, ())
#
#     @on_trait_change('scene.activated')
#     def update_plot(self):
#         sides = 'lh' if (len(self.h_coords['lh']) > 0) and (len(self.h_coords['rh']) == 0) else 'rh' \
#             if (len(self.h_coords['lh']) == 0) and (len(self.h_coords['rh']) > 0) else ['lh', 'rh']
#
#         for s in ['lh', 'rh']:
#             if s in sides:
#                 self.scene.mlab.triangular_mesh(self.pial[s][0][:, 0], self.pial[s][0][:, 1],
#                                                 self.pial[s][0][:, 2],
#                                                 self.pial[s][1], opacity=0.37, color=(0.7, 0.7, 0.7))
#
#                 self.scene.mlab.points3d(self.h_coords[s].x_norm_surf.values, self.h_coords[s].y_norm_surf.values,
#                                          self.h_coords[s].z_norm_surf.values, self.h_coords[s].vals.values, scale_mode='vector', scale_factor=3,
#                                          vmin=self.lims[0], vmax=self.lims[1], colormap=self.cmap)
#                 self.scene.mlab.points3d(self.stim_coords[0], self.stim_coords[1],
#                                          self.stim_coords[2], scale_mode='none', mode='cube',
#                                          scale_factor=3, color=(0, 0.4, 0.13))
#         if 'lh' in sides:
#             self.scene.mlab.view(135, 90)
#         else:
#             self.scene.mlab.view(45, 90)
#
#     view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
#                      height=250, width=300, show_label=False),
#                 resizable=True  # We need this to resize with the parent widget
#                 )
#
#
# class Surfaces(QDialog):
#     def __init__(self, parent=App, coords=None, stim_coords=None, vals=None, lims=None, stc=None, subj_name=None, cmap='bwr'):
#         super(Surfaces, self).__init__(parent)
#         self.setGeometry(500, 50, 800, 800)
#         layout = QVBoxLayout(self)
#         layout.setContentsMargins(0, 0, 0, 0)
#         layout.setSpacing(0)
#
#         if coords is None:
#             coords = self.parent().seeg_ch_info
#             pial = self.parent().pial
#
#             # print coords
#             if stc is None:
#                 self.visualization = SurfViz(pial, coords, stim_coords, vals, lims, cmap)
#
#             else:
#                 self.visualization = StcViz(stc, stim_coords, subj_name)
#
#             # The edit_traits call will generate the widget to embed.
#             self.ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
#             layout.addWidget(self.ui)
#             self.ui.setParent(self)


class Comments(QDialog):
    def __init__(self, parent=App, dir_comments=None, session=None, subject=None):
        super(Comments, self).__init__(parent)

        self.setGeometry(450, 50, 800, 900)
        self.setWindowTitle('Comments')

        # Comment files
        self.prepr_comm_file = os.path.join(dir_comments, 'preprocessing', '%s_%s_prepr.txt' % (subject, session))
        self.analy_comm_file = os.path.join(dir_comments, 'analysis', '%s_%s_analy.txt' % (subject, session))

        tabs = QTabWidget(self)
        tab_prepro = QWidget()
        tab_analy = QWidget()

        tabs.addTab(tab_analy, "Analysis")
        tabs.addTab(tab_prepro, "Preprocessing")

        # Preprocessing
        prepro_layout = QVBoxLayout()
        self.text_prepr = QTextEdit()
        self.text_prepr.setReadOnly(True)
        self.text_prepr.setStyleSheet('background-color: #d4d7db;')

        prepro_layout.addWidget(self.text_prepr)
        tab_prepro.setLayout(prepro_layout)

        if os.path.isfile(self.prepr_comm_file):
            self.text_prepr.setText(self.load_file(self.prepr_comm_file))

        # Analysis
        analy_prev_label = QLabel()
        analy_prev_label.setText('Previous Comments')

        analy_new_label = QLabel()
        analy_new_label.setText('New Comments')

        self.text_analy_prev = QTextEdit()
        self.text_analy_prev.setReadOnly(True)
        self.text_analy_prev.setStyleSheet('background-color: #d4d7db;')

        self.text_analy_new = QTextEdit()
        analy_menu_layout = QHBoxLayout()

        user_label = QLabel()
        user_label.setText('User:')
        self.user_value = QLineEdit()

        save_but = QPushButton('Save')
        save_but.clicked.connect(self.save_file)

        analy_menu_layout.addWidget(user_label)
        analy_menu_layout.addWidget(self.user_value)
        analy_menu_layout.addWidget(save_but)

        analy_layout = QVBoxLayout()
        analy_layout.addLayout(analy_menu_layout)
        analy_layout.addWidget(analy_new_label)
        analy_layout.addWidget(self.text_analy_new)
        analy_layout.addWidget(analy_prev_label)
        analy_layout.addWidget(self.text_analy_prev)
        tab_analy.setLayout(analy_layout)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(tabs)

        self.setLayout(main_layout)
        if os.path.isfile(self.analy_comm_file):
            self.text_analy_prev.setText(self.load_file(self.analy_comm_file))
        self.show()

    @staticmethod
    def load_file(filename):
        f = open(filename, 'r')
        text = f.read()
        f.close()
        return text

    def save_file(self):
        user = 'Anonymous' if self.user_value.text() == '' else self.user_value.text()
        date = datetime.datetime.now().strftime('%d/%m/%y - %H:%M')

        filename = self.analy_comm_file
        f = open(filename, 'a')
        filedata = self.text_analy_new.toPlainText()
        filedata = '#%s (%s):\n' % (user, date) + filedata
        filedata = str(filedata) + "\n\n"
        f.write(filedata)
        f.close()

        self.text_analy_prev.setText(self.load_file(filename))
        self.text_analy_new.clear()


class PCI(QDialog):
    def __init__(self, parent=App, pci_lz=None, pci_st=None, times=None):
        super(PCI, self).__init__(parent)

        self.setGeometry(0, 0, 1850, 1000)
        self.setWindowTitle('PCI')

        self.current_hdeeg_comp = 0
        self.current_seeg_comp = 0

        self.pci_st = pci_st

        # Layouts
        self.wid_layout = QVBoxLayout()
        self.vals_layout = QHBoxLayout()
        self.plots_layout = QGridLayout()

        # Vals
        self.pci_lz_label = QLabel()
        self.pci_lz_label.setText('HDEEG - PCI LZ:')
        self.pci_lz_value = QLabel()
        self.pci_lz_value.setText('%0.2f' % pci_lz['PCI']['bl']['PCI']['unifSTD'])
        self.pci_lz_value.setStyleSheet('background-color: #dcdfe5;')
        self.pci_lz_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.pci_lz_value.setAlignment(Qt.AlignCenter)

        self.pci_st_hdeeg_label = QLabel()
        self.pci_st_hdeeg_label.setText('HDEEG - PCI ST:')
        self.pci_st_hdeeg_value = QLabel()
        self.pci_st_hdeeg_value.setText('%0.2f' % pci_st['HDEEG']['PCI'])

        self.pci_st_hdeeg_value.setStyleSheet('background-color: #dcdfe5;')
        self.pci_st_hdeeg_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.pci_st_hdeeg_value.setAlignment(Qt.AlignCenter)

        self.pci_st_seeg_label = QLabel()
        self.pci_st_seeg_label.setText('SEEG - PCI ST:')
        self.pci_st_seeg_value = QLabel()
        self.pci_st_seeg_value.setText('%0.2f' % pci_st['SEEG_bipolar']['PCI'])

        self.pci_st_seeg_value.setStyleSheet('background-color: #dcdfe5;')
        self.pci_st_seeg_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.pci_st_seeg_value.setAlignment(Qt.AlignCenter)

        self.vals_layout.addWidget(self.pci_lz_label)
        self.vals_layout.addWidget(self.pci_lz_value)
        self.vals_layout.addWidget(self.pci_st_hdeeg_label)
        self.vals_layout.addWidget(self.pci_st_hdeeg_value)
        self.vals_layout.addWidget(self.pci_st_seeg_label)
        self.vals_layout.addWidget(self.pci_st_seeg_value)

        # Plots
        # LZ
        pci = pci_lz['PCI']['bl']['complexity_time']['STD'] / pci_lz['PCI']['bl']['C0unif']
        times_lz = np.arange(0, len(pci))
        self.hdeeg_lz_tc = WidgetPlot(title='HDEEG - PCI LZ', data=pci, times=times_lz)

        # Sess label
        self.sess_label = QLabel()
        self.sess_label.setText('%s - %s' % (self.parent().subj, self.parent().sess))
        self.sess_label.setStyleSheet('background-color: #dcdfe5;')
        self.sess_label.setAlignment(Qt.AlignCenter)

        # Eigenvals
        self.eigenvals = PciInfo(pci_st=pci_st)
        self.eigenvals.axes[0].patches[0].set_facecolor('tab:olive')
        self.eigenvals.axes[1].patches[0].set_facecolor('tab:olive')

        # ST HDEEG
        all_pos = self.parent().mne_info
        bad_ch = self.parent().data['HDEEG']['interpolated_channels']
        good_ch = bad_ch == 0
        #good_pos = all_pos[good_ch]
        good_pos = all_pos.copy().pick_channels([all_pos.ch_names[ix] for ix in range(len(all_pos.ch_names)) if good_ch[ix] == True])
        self.pcist_hdeeg_topo = TopoPlot(data=pci_st['HDEEG']['vk'][0, :], pos=good_pos.info, measure='PCI')

        self.pcist_hdeeg_selector = QComboBox()
        for i in range(pci_st['HDEEG']['vk'].shape[0]):
            self.pcist_hdeeg_selector.addItem('Component %s' % (i+1))
        self.pcist_hdeeg_selector.currentIndexChanged.connect(self.on_hdeeg_select)

        self.pcist_hdeeg_tc = WidgetPlot(title='HDEEG - PCI ST Components', data=pci_st['HDEEG']['signal_svd'].T, times=times)
        self.pcist_hdeeg_tc.canvas.axes.lines[self.current_hdeeg_comp].set_c('tab:olive')
        self.pcist_hdeeg_tc.canvas.axes.lines[self.current_hdeeg_comp].set_linewidth(3)

        # ST SEEG
        self.vmax_seeg = np.abs(pci_st['SEEG_bipolar']['vk']).max()
        self.vmin_seeg = -self.vmax_seeg
        self.pcist_seeg_topo = SeegLocs(coords=self.parent().seeg_coords, stim_coords=self.parent().stim_coords_mri_norm)

        sp = cm.get_cmap('bwr', 256)
        norm = mpl.colors.Normalize(vmin=self.vmin_seeg, vmax=self.vmax_seeg)
        c_data = sp(norm(pci_st['SEEG_bipolar']['vk'][0, :]))
        self.pcist_seeg_topo.plot(data=c_data, coords=self.parent().seeg_coords, sel=None, stim_coords=None)
        self.pcist_seeg_topo.update_cbar(vmin=self.vmin_seeg, vmax=self.vmax_seeg,
                                         title='Component weight')

        self.pcist_seeg_selector = QComboBox()
        for i in range(pci_st['SEEG_bipolar']['vk'].shape[0]):
            self.pcist_seeg_selector.addItem('Component %s' % (i+1))
        self.pcist_seeg_selector.currentIndexChanged.connect(self.on_seeg_select)

        self.pcist_seeg_tc = WidgetPlot(title='SEEG - PCI ST Components', data=pci_st['SEEG_bipolar']['signal_svd'].T, times=times)
        self.pcist_seeg_tc.canvas.axes.lines[self.current_seeg_comp].set_c('tab:olive')
        self.pcist_seeg_tc.canvas.axes.lines[self.current_seeg_comp].set_linewidth(3)

        self.plots_layout.addWidget(self.hdeeg_lz_tc, 0, 0, 1, 1)
        self.plots_layout.addWidget(self.sess_label, 1, 0, 1, 1)
        self.plots_layout.addWidget(self.eigenvals, 2, 0, 1, 1)
        self.plots_layout.addWidget(self.pcist_hdeeg_tc, 0, 1, 1, 1)
        self.plots_layout.addWidget(self.pcist_hdeeg_selector, 1, 1, 1, 1)
        self.plots_layout.addWidget(self.pcist_hdeeg_topo, 2, 1, 1, 1)
        self.plots_layout.addWidget(self.pcist_seeg_tc, 0, 2, 1, 1)
        self.plots_layout.addWidget(self.pcist_seeg_selector, 1, 2, 1, 1)
        self.plots_layout.addWidget(self.pcist_seeg_topo, 2, 2, 1, 1)

        self.wid_layout.addLayout(self.vals_layout)
        self.wid_layout.addLayout(self.plots_layout)
        self.setLayout(self.wid_layout)

        self.show()

    def on_hdeeg_select(self, ix):
        self.pcist_hdeeg_tc.canvas.axes.lines[self.current_hdeeg_comp].set_c('tab:gray')
        self.pcist_hdeeg_tc.canvas.axes.lines[self.current_hdeeg_comp].set_linewidth(0.9)
        self.eigenvals.axes[0].patches[self.current_hdeeg_comp].set_facecolor('tab:gray')

        self.current_hdeeg_comp=ix
        self.pcist_hdeeg_tc.canvas.axes.lines[self.current_hdeeg_comp].set_c('tab:olive')
        self.pcist_hdeeg_tc.canvas.axes.lines[self.current_hdeeg_comp].set_linewidth(3)
        self.pcist_hdeeg_tc.canvas.draw()

        self.pcist_hdeeg_topo.plot(data=self.pci_st['HDEEG']['vk'][self.current_hdeeg_comp, :], title='HDEEG - PCI ST')
        self.eigenvals.axes[0].patches[self.current_hdeeg_comp].set_facecolor('tab:olive')
        self.eigenvals.draw()

    def on_seeg_select(self, ix):
        self.pcist_seeg_tc.canvas.axes.lines[self.current_seeg_comp].set_c('tab:gray')
        self.pcist_seeg_tc.canvas.axes.lines[self.current_seeg_comp].set_linewidth(0.9)
        self.eigenvals.axes[1].patches[self.current_seeg_comp].set_facecolor('tab:gray')


        self.current_seeg_comp = ix
        self.pcist_seeg_tc.canvas.axes.lines[self.current_seeg_comp].set_c('tab:olive')
        self.pcist_seeg_tc.canvas.axes.lines[self.current_seeg_comp].set_linewidth(3)
        self.pcist_seeg_tc.canvas.draw()

        self.eigenvals.axes[1].patches[self.current_seeg_comp].set_facecolor('tab:olive')
        self.eigenvals.draw()

        sp = cm.get_cmap('bwr', 256)
        norm = mpl.colors.Normalize(vmin=self.vmin_seeg, vmax=self.vmax_seeg)
        c_data = sp(norm(self.pci_st['SEEG_bipolar']['vk'][self.current_seeg_comp, :]))
        self.pcist_seeg_topo.plot(data=c_data, coords=self.parent().seeg_coords,
                                  sel=None, stim_coords=None)

    # todo: click select component


class PciInfo(FigureCanvas):

    def __init__(self, parent=App, width=5, height=8, dpi=100, pci_st=None):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.subplots(2, 1)

        FigureCanvas.__init__(self, fig)
        # self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot(pci_st)

    def plot(self, pci_st):

        n_comp_hdeeg = pci_st['HDEEG']['n_dims']
        n_comp_seeg = pci_st['SEEG_bipolar']['n_dims']

        tot_var_hdeeg = np.sum(pci_st['HDEEG']['eigenvalues'])
        tot_var_seeg = np.sum(pci_st['SEEG_bipolar']['eigenvalues'])

        comp_eig_hdeeg = pci_st['HDEEG']['eigenvalues'][:n_comp_hdeeg]
        comp_eig_seeg = pci_st['SEEG_bipolar']['eigenvalues'][:n_comp_seeg]

        self.axes[0].barh(np.arange(n_comp_hdeeg)+1,
                          (comp_eig_hdeeg / tot_var_hdeeg) * 100,
                          color='tab:gray')
        self.axes[0].set_ylabel('HDEEG - PCI ST\nComponent')
        self.axes[0].set_yticks(np.arange(n_comp_hdeeg)+1)
        self.axes[0].set_yticklabels(np.arange(n_comp_hdeeg)+1)
        self.axes[0].invert_yaxis()

        self.axes[1].barh(np.arange(n_comp_seeg)+1,
                          (comp_eig_seeg / tot_var_seeg) * 100,
                          color='tab:gray')
        self.axes[1].set_ylabel('SEEG - PCI ST\nComponent')
        self.axes[1].set_yticks(np.arange(n_comp_seeg)+1)
        self.axes[1].set_yticklabels(np.arange(n_comp_seeg)+1)
        self.axes[1].invert_yaxis()
        self.axes[1].set_xlabel('% variance explained')
        self.draw()


# class StcViz(HasTraits):
#     def __init__(self, stc, stim_coords, subj_name):
#         super(StcViz, self).__init__()
#
#         self.stc = stc
#         self.stim_coords = stim_coords
#         self.subj_name = subj_name
#         self.hemi = 'lh' if stim_coords[0] < 0 else 'rh'
#     scene = Instance(MlabSceneModel, ())
#
#     @on_trait_change('scene.activated')
#     def update_plot(self):
#         stc_plot = self.stc.plot(hemi='split', subjects_dir=dir_fs_subjects, initial_time=0,
#                  time_viewer=True, subject=self.subj_name, size=(1200, 600),
#                  colormap='viridis', surface='inflated')
#         stc_plot.add_foci(coords=self.stim_coords, hemi=self.hemi, color='r',
#                           map_surface='pial')
#         # todo: fix event loop already running when adding foci
#
#     view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
#                      height=250, width=300, show_label=False),
#                 resizable=True  # We need this to resize with the parent widget
#                 )


class Power(QDialog):
    def __init__(self, parent=App, data=None):
        super(Power, self).__init__(parent)

        self.setGeometry(0, 0, 1850, 1000)
        self.setWindowTitle('Power')

        self.data = data
        self.sorter = 'hf'

        # create layouts
        self.main_layout = QHBoxLayout()
        self.hdeeg_layout = QVBoxLayout()
        self.bip_layout = QVBoxLayout()
        self.mono_layout = QVBoxLayout()

        self.menu_hdeeg_layout = QGridLayout()
        self.menu_bip_layout = QGridLayout()
        self.menu_mono_layout = QGridLayout()

        # fill menus
        # hdeeg
        hdeeg_label = QLabel()
        hdeeg_label.setText('HDEEG')
        hdeeg_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        hdeeg_label_sort = QLabel()
        hdeeg_label_sort.setText('Sort by:')
        hdeeg_label_sort.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        hdeeg_sortby = QComboBox()
        for s in ['High-frequency', 'Delta', 'PLF']:
            hdeeg_sortby.addItem(s)
        hdeeg_sortby.activated[str].connect(self.on_sort_select_hdeeg)

        self.hdeeg_ch_sel = QComboBox()
        for s in ch185:
            self.hdeeg_ch_sel.addItem(s)
        self.hdeeg_ch_sel.activated[str].connect(self.on_hdeeg_ch_sel)

        hdeeg_tf_but = QPushButton('TF', self)
        hdeeg_tf_but.clicked.connect(self.on_open_tf_hdeeg)

        # bipolar
        bip_label = QLabel()
        bip_label.setText('Bipolar')
        bip_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        bip_label_sort = QLabel()
        bip_label_sort.setText('Sort by:')
        bip_label_sort.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        bip_sortby = QComboBox()
        for s in ['High-frequency', 'Delta', 'PLF']:
            bip_sortby.addItem(s)
        bip_sortby.activated[str].connect(self.on_sort_select_bip)

        self.bip_ch_sel = QComboBox()
        for s in data['ersp_labels_bipo']:
            self.bip_ch_sel.addItem(s)
        self.bip_ch_sel.activated[str].connect(self.on_bip_ch_sel)

        bip_tf_but = QPushButton('TF', self)
        bip_tf_but.clicked.connect(self.on_open_tf_bipo)

        # monopolar
        mono_label = QLabel()
        mono_label.setText('Monopolar')
        mono_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        mono_label_sort = QLabel()
        mono_label_sort.setText('Sort by:')
        mono_label_sort.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        mono_sortby = QComboBox()
        for s in ['High-frequency', 'Delta', 'PLF']:
            mono_sortby.addItem(s)
        mono_sortby.activated[str].connect(self.on_sort_select_mono)

        self.mono_ch_sel = QComboBox()
        for s in data['ersp_labels_mono']:
            self.mono_ch_sel.addItem(s)
        self.mono_ch_sel.activated[str].connect(self.on_mono_ch_sel)

        mono_tf_but = QPushButton('TF', self)
        mono_tf_but.clicked.connect(self.on_open_tf_mono)

        # plots
        self.plots_hdeeg = PowerPlots(parent=self, data=data, kind='hdeeg')
        self.plots_bip = PowerPlots(parent=self, data=data, kind='bipo')
        self.plots_mono = PowerPlots(parent=self, data=data, kind='mono')

        # fill layouts
        self.menu_hdeeg_layout.addWidget(hdeeg_label, 0, 0, 1, 4)
        self.menu_hdeeg_layout.addWidget(hdeeg_label_sort, 1, 1, 1, 1)
        self.menu_hdeeg_layout.addWidget(hdeeg_sortby, 1, 2, 1, 1)
        self.menu_hdeeg_layout.addWidget(self.hdeeg_ch_sel, 2, 1, 1, 1)
        self.menu_hdeeg_layout.addWidget(hdeeg_tf_but, 2, 2, 1, 1)

        self.menu_bip_layout.addWidget(bip_label, 0, 0, 1, 4)
        self.menu_bip_layout.addWidget(bip_label_sort, 1, 1, 1, 1)
        self.menu_bip_layout.addWidget(bip_sortby, 1, 2, 1, 1)
        self.menu_bip_layout.addWidget(self.bip_ch_sel, 2, 1, 1, 1)
        self.menu_bip_layout.addWidget(bip_tf_but, 2, 2, 1, 1)

        self.menu_mono_layout.addWidget(mono_label, 0, 0, 1, 4)
        self.menu_mono_layout.addWidget(mono_label_sort, 1, 1, 1, 1)
        self.menu_mono_layout.addWidget(mono_sortby, 1, 2, 1, 1)
        self.menu_mono_layout.addWidget(self.mono_ch_sel, 2, 1, 1, 1)
        self.menu_mono_layout.addWidget(mono_tf_but, 2, 2, 1, 1)

        self.hdeeg_layout.addLayout(self.menu_hdeeg_layout)
        self.hdeeg_layout.addWidget(self.plots_hdeeg)
        self.bip_layout.addLayout(self.menu_bip_layout)
        self.bip_layout.addWidget(self.plots_bip)
        self.mono_layout.addLayout(self.menu_mono_layout)
        self.mono_layout.addWidget(self.plots_mono)

        self.main_layout.addLayout(self.hdeeg_layout)
        self.main_layout.addLayout(self.bip_layout)
        self.main_layout.addLayout(self.mono_layout)

        self.setLayout(self.main_layout)

        self.show()

    def on_sort_select_hdeeg(self, sorter):
        sorter = str(sorter)
        self.sorter = 'hf' if sorter == 'High-frequency' else 'delta' if \
            sorter == 'Delta' else 'plf' if sorter == 'PLF' else None
        self.plots_hdeeg.plot_raster(sorter=self.sorter)

    def on_sort_select_bip(self, sorter):
        sorter = str(sorter)
        self.sorter = 'hf' if sorter == 'High-frequency' else 'delta' if \
            sorter == 'Delta' else 'plf' if sorter == 'PLF' else None
        self.plots_bip.plot_raster(sorter=self.sorter)

    def on_sort_select_mono(self, sorter):
        sorter = str(sorter)
        self.sorter = 'hf' if sorter == 'High-frequency' else 'delta' if \
            sorter == 'Delta' else 'plf' if sorter == 'PLF' else None
        self.plots_mono.plot_raster(sorter=self.sorter)

    def on_hdeeg_ch_sel(self, chan):
        ch = str(chan)
        ix = ch185.index(ch)
        print(ix, chan)
        self.plots_hdeeg.plot_raster(sorter=self.sorter, curr_ix=ix)

    def on_bip_ch_sel(self, chan):
        ch = str(chan)
        ix = list(self.data['ersp_labels_bipo']).index(ch)
        self.plots_bip.plot_raster(sorter=self.sorter, curr_ix=ix)

    def on_mono_ch_sel(self, chan):
        ch = str(chan)
        ix = list(self.data['ersp_labels_mono']).index(ch)
        self.plots_mono.plot_raster(sorter=self.sorter, curr_ix=ix)

    def on_open_tf_hdeeg(self):
        self.plots_hdeeg.plot_tf()

    def on_open_tf_bipo(self):
        self.plots_bip.plot_tf()

    def on_open_tf_mono(self):
        self.plots_mono.plot_tf()


class PowerPlots(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, data=None,
                 kind=None):

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = []

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        FigureCanvas.mpl_connect(self, s='button_press_event', func=self.on_click)

        self.measures = ['hf', 'delta', 'plf']
        self.kind = kind
        self.labels = ch185 if kind == 'hdeeg' else list(data['ersp_labels_%s' % kind])
        self.curr_ix = 0

        self.data = {}
        self.data['ersp'] = data['ersp_%s' % kind][:, :, :]
        self.data['hf'] = data['ersp_%s' % kind][:, data['ersp_freqs_%s' % kind] > 20, :].mean(1)
        self.data['delta'] = data['delta_%s' % kind].mean(2)
        self.data['plf'] = data['plf_stat_%s' % kind]
        self.stats = data['erspboot_%s' % kind]

        self.times_ersp = data['ersp_times_%s' % kind]
        self.times = data['times']

        self.sorters = {}
        self.abs_max = {}
        self.freqs = data['ersp_freqs_%s' % kind]

        hf_ch = self.data['hf'][:, (self.times_ersp > 0) & (self.times_ersp < 300)].mean(1)
        self.sorters['hf'] = hf_ch.argsort()

        delta_ch = self.data['delta'][:, (self.times > 0) & (self.times < 300)].mean(1)
        self.sorters['delta'] = delta_ch.argsort()

        plf_ch = self.data['plf'][:, (self.times > 0) & (self.times < 300)].mean(1)
        self.sorters['plf'] = plf_ch.argsort()

        self.abs_max['hf'] = np.max(np.abs(self.data['hf']))
        self.abs_max['delta'] = np.max(np.abs(self.data['delta']))

        self.nchans = {m: self.data[m].shape[0] for m in self.measures}

        self.sorted_data = []

        if kind in ['hdeeg', 'bipo', 'mono']:
            self.plot_raster()

    def plot_raster(self, sorter='hf', curr_ix=None):
        self.fig.clear()
        self.axes = self.fig.subplots(3, 2)
        self.sorter = sorter

        sorter_all = self.sorters['hf'] if sorter == 'hf' else self.sorters['delta'] if \
            sorter == 'delta' else self.sorters['plf']

        self.sorted_labels = [self.labels[i] for i in sorter_all]

        self.sorted_data = {k: self.data[k][sorter_all] for k in self.measures}

        for ix, m in enumerate(self.measures):
            cmap = 'plasma' if m == 'plf' else 'coolwarm' if m == 'hf' else 'viridis'
            vmin = 0 if m == 'plf' else -self.abs_max[m]
            vmax = 1 if m == 'plf' else self.abs_max[m]

            tmin = self.times_ersp[0] if m == 'hf' else self.times[0]
            tmax = self.times_ersp[-1] if m == 'hf' else self.times[-1]

            im = self.axes[ix, 0].imshow(self.sorted_data[m], extent=(tmin, tmax, self.nchans[m], 1),
                                         aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)


            cb = self.fig.colorbar(im, ax=self.axes[ix, 0])

            cb.ax.set_title(m)

            if curr_ix is None:
                self.curr_ix = sorter_all.argsort().argmax()
                self.arrow_ix = len(self.labels)
            else:
                self.curr_ix = curr_ix
                self.arrow_ix = sorter_all.argsort()[curr_ix]

            if self.kind == 'hdeeg':
                self.parent().hdeeg_ch_sel.setCurrentIndex(self.curr_ix)
            elif self.kind == 'bipo':
                self.parent().bip_ch_sel.setCurrentIndex(self.curr_ix)
            else:
                self.parent().mono_ch_sel.setCurrentIndex(self.curr_ix)

            times = self.times if m in ['delta', 'plf'] else self.times_ersp

            self.axes[ix, 1].plot(times, self.data[m][self.curr_ix, :])
            self.axes[ix, 0].set_ylabel('sorted channels')
            self.axes[ix, 1].set_ylabel(m)

            # todo: match indexes

            self.axes[ix, 0].annotate('', xy=(tmax, self.arrow_ix),
                                      xytext=(tmax+10, self.arrow_ix),
                                      arrowprops=dict(arrowstyle="->", color='g'))

        for ax in self.axes.flatten():
            ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                      linestyles='--', alpha=0.4)

        for i in range(2):
            self.axes[2, i].set_xlabel('times (ms)')

        self.fig.tight_layout()
        self.draw()

    def plot_tf(self):
        # todo: put in widget to avoid loop running warnings
        fig, ax = plt.subplots(1, 1)
        vmax = np.max(np.abs(self.data['ersp'][self.curr_ix, :, :]))
        vmin = -vmax

        ch_stats = self.stats[self.curr_ix, :, :]
        data = self.data['ersp'][self.curr_ix, :, :]

        low_th = np.repeat(ch_stats[:, 0].reshape(ch_stats.shape[0], -1), data.shape[1], axis=1)
        high_th = np.repeat(ch_stats[:, 1].reshape(ch_stats.shape[0], -1), data.shape[1], axis=1)
        mask = np.ma.masked_where((data > low_th) & (data < high_th), data)

        ax.imshow(data, aspect='auto', cmap='Greys',
                  extent=(self.times_ersp[0], self.times_ersp[-1],
                          self.freqs[0], self.freqs[-1]), vmin=vmin, vmax=vmax,
                  origin='lower', interpolation='lanczos')

        im = ax.imshow(mask, aspect='auto', cmap='coolwarm',
                  extent=(self.times_ersp[0], self.times_ersp[-1],
                          self.freqs[0], self.freqs[-1]), vmin=vmin, vmax=vmax,
                  origin='lower', interpolation='nearest')

        cb = self.fig.colorbar(im, ax=ax)

        cb.ax.set_title('ersp')

        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (ms)')
        ax.vlines(x=0, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                  linestyles='--', alpha=0.4)
        ax.set_title(self.labels[self.curr_ix])
        plt.show()

    def on_click(self, event):
        y = event.ydata
        selected_ch = self.sorted_labels[int(y)-1]
        curr_ix = self.labels.index(selected_ch)
        self.plot_raster(sorter=self.sorter, curr_ix=curr_ix)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    icon = os.path.join(dir_resources, 'coregview_icon.png')
    app.setWindowIcon(QIcon(icon))
    ex = App()
    sys.exit(app.exec_())

