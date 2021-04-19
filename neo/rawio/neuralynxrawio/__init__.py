"""
RawIO for reading data from Neuralynx files.
This IO supports NCS, NEV, NSE and NTT file formats.


NCS contains the sampled signal for one channel
NEV contains events
NSE contains spikes and waveforms for mono electrodes
NTT contains spikes and waveforms for tetrodes
"""

from neo.rawio.neuralynxrawio.neuralynxrawio import NeuralynxRawIO
