import sys
from PyQt5 import QtGui
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from numpy.lib.function_base import sinc
from numpy.lib.shape_base import tile
from numpy.ma import dot
from Samplind_And_Recovery import Ui_MainWindow
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyqtgraph import PlotWidget, PlotItem
import pyqtgraph as pg
from PyQt5.QtWidgets import QBoxLayout, QLineEdit, QSpinBox
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QDoubleValidator, QValidator
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from scipy.interpolate import interp1d
import numpy.fft as fft
from scipy.interpolate import make_interp_spline
from numpy import savetxt, newaxis


class MainWindow(qtw.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.graphChannels = [
            self.ui.graphicsView_sample, self.ui.graphicsView_recover]

        self.setChannelChecked = [self.ui.checkBox]

        self.channelComponents = [self.ui.channel_sample,
                                  self.ui.channel_recover]

        self.channel_sample = 0
        self.channel_recover = 1

        self.ui.checkBox.setChecked(True)
        self.ui.channel_sample.show()
        self.ui.channel_recover.show()
        self.pen1 = pg.mkPen(color=(255, 0, 0))
        self.pen2 = pg.mkPen(color=(0, 139, 139))
        self.pen3 = pg.mkPen(color=(255, 255, 0))
        self.pen4 = pg.mkPen(color=(0, 255, 0))
        self.pen5 = pg.mkPen(color=(138, 43, 226))
        self.pen6 = pg.mkPen(color=(255, 99, 71))
        self.signal_ID = self.ui.comboBox_choosesignal.currentIndex()
        self.sum = 0
        self.currentindex = []
        self.AmplitudeList = []
        self.TimeList = []
        self.signals = []
        self.signals_names = []
        self.ui.lineEdit.textChanged.connect(self.signal_name)
        self.ui.spinBox_frequency.valueChanged.connect(self.draw_signal)
        self.ui.spinBoxMagnitude.valueChanged.connect(self.draw_signal)
        self.ui.spinBox_Phase.valueChanged.connect(self.draw_signal)
        self.ui.pushButton_ADD.clicked.connect(lambda: self.add_signal())
        self.ui.pushButton_Delete.clicked.connect(lambda: self.removesignals())
        self.ui.pushButton_confirmSum.clicked.connect(lambda: self.confirm())
        self.ui.actionchannel_1.triggered.connect(lambda: self.open_file())
        self.ui.slider.setMinimum(1)
        self.ui.slider.setMaximum(3)
        

        self.ui.slider.valueChanged.connect(
            lambda: self.getFsample(self.time, self.amplitude))
        self.ui.recover_upper_graph.clicked.connect(
            lambda: self.recover_upper_graph(self.sampled_time, self.sampling_signal, self.time, self.time_to_sample))
        self.ui.recover_down_graph.clicked.connect(
            lambda: self.recover_down_graph(self.sampled_time, self.sampling_signal, self.time, self.time_to_sample))
        self.ui.checkBox.stateChanged.connect(
            lambda: self.toggle(self.channel_sample))
        self.show()
        self.frequency = 0
        self.magnitude = 0
        self.phase = 0


    def toggle(self, channel_sample: int, ) -> None:
        if (self.channelComponents[channel_sample].isVisible()):
            self.channelComponents[channel_sample].hide()
            self.setChannelChecked[channel_sample].setChecked(False)
        else:
            self.setChannelChecked[channel_sample].setChecked(True)
            self.channelComponents[channel_sample].show()

    def signal_name(self, text):
        self.signalname = text

    def get_freq(self):
        self.frequency = self.ui.spinBox_frequency.value()
        return (self.frequency)

    def get_magnitude(self):
        self.magnitude = self.ui.spinBoxMagnitude.value()
        return (self.magnitude)

    def get_phase(self):
        self.phase = self.ui.spinBox_Phase.value()
        return (self.phase)

    def draw_signal(self):
        self.ui.GraphViewSignal.clear()
        self.sample_rate = 100
        self.start_time = 0
        self.end_time = 2
        self.time = np.arange(float(self.start_time), float(
            self.end_time), float(1 / self.sample_rate))

        self.sinewave = self.get_magnitude() * np.sin(2.0 * np.pi * self.get_freq()
                                                      * self.time + self.get_phase())
        self.ui.GraphViewSignal.plot(self.sinewave, pen=self.pen4)

    def add_signal(self):
        self.signals.append(self.sinewave)
        self.signals_names.append(self.signalname)
        self.ui.comboBox_choosesignal.addItem(
            self.signals_names[self.signal_ID])
        self.draw_summition()

    def removesignal(self, xs):
        self.signals_names.remove(self.signals_names[xs])
        self.ui.comboBox_choosesignal.clear()
        for i in range(0, len(self.signals_names)):
            self.ui.comboBox_choosesignal.addItem(self.signals_names[i])

    def removesignals(self):
        for i in range(0, len(self.signals_names)):
            if self.ui.comboBox_choosesignal.currentIndex() == i:
                self.removesignal(i)
                del self.signals[i]

    def draw_summition(self):
        self.ui.GraphViewComposer.clear()
        self.sample_rate = 100
        self.start_time = 0
        self.end_time = 2
        self.time = np.arange(
            self.start_time, self.end_time, 1 / self.sample_rate)
        self.sum = 0
        for i in range(0, len(self.signals)):
            self.sum += self.signals[i]
        self.ui.GraphViewComposer.plot(self.sum, pen=self.pen4)

    def confirm(self):
        self.ui.graphicsView_sample.clear()
        self.amplitude = self.sum
        self.time = self.time
        self.plot(self.time, self.amplitude)

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_url, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;csv Files (*.csv)", options=options)
        if file_url:
            print(file_url)
            filename = Path(file_url).stem
            print(filename)
            self.read_file(file_url)


    def read_file(self, file_path):
        self.ui.graphicsView_sample.clear()
        path = file_path
        read_data = pd.read_csv(path)
        self.amplitude = read_data.values[:, 1]
        self.AmplitudeList.append(self.amplitude)
        self.time = read_data.values[:, 0]
        self.TimeList.append(self.time)
        for i in range(len(self.AmplitudeList)):
            if i <= 6:
                self.plot(self.TimeList[i], self.AmplitudeList[i])


    def getFsample(self, time, amplitude):
        self.ui.graphicsView_sample.clear()
        self.ui.graphicsView_recover.clear()
        self.ui.graphicsView_sample.plot(time, amplitude, pen=self.pen4)
        spec = fft.fft(amplitude)
        freqs = fft.fftfreq(len(spec))
        threshold = 0.5 * max(abs(spec))
        mask = abs(spec) > threshold
        peaks1 = freqs[mask]
        peaks = abs(peaks1)
        F_max = max(peaks * 100)
        self.fmax = int(F_max)
        self.fs = self.ui.slider.value() * self.fmax
        self.time_to_sample = 1/(self.fs)
        self.sampled_time = (time)[::int(100 / self.fs)]
        self.sampling_signal = (amplitude)[::int(100 / (self.fs))]
        self.ui.graphicsView_sample.plot(
            self.sampled_time, self.sampling_signal, pen=None, symbol='o')
        

    def recover_upper_graph(self, timesampled, datasampled, originaltime, TS):
        self.ui.graphicsView_recover.clear()
        y_interpolated = 0
        for index in range(0, len(timesampled)):
            y_interpolated += datasampled[index] * \
                np.sinc((np.array(originaltime)-TS*index)/TS)
        self.ui.graphicsView_recover.plot(
            originaltime, y_interpolated, pen=self.pen4)

        

    def recover_down_graph(self, timesampled, datasampled, originaltime, TS):
        y_interpolated = 0
        for index in range(0, len(timesampled)):
            y_interpolated += datasampled[index] * \
                np.sinc((np.array(originaltime)-TS*index)/TS)
        self.ui.graphicsView_sample.plot(
            originaltime, y_interpolated, pen=self.pen4)
      

    def plot(self, x, y):
        self.ui.graphicsView_sample.plot(x, y, pen=self.pen4)


if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
