"""
Class for reading  spikegadgets file.
Only signals ability at the moment.

https://spikegadgets.com/spike-products/


The file ".rec" have :
  * a fist part in text with xml informations
  * a second part for the binary buffer

Author: Samuel Garcia
"""
from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np

from xml.etree import ElementTree

class SpikeGadgetsRawIO(BaseRawIO):
    extensions = ['rec']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        
        # parse file until "</Configuration>"
        header_size = None
        with open(self.filename, mode='rb') as f:
            while True:
                line = f.readline()
                if b"</Configuration>" in line:
                    header_size = f.tell()
                    break
            if header_size is None:
                ValueError("SpikeGadgets : the xml header do not contain </Configuration>")
            print(header_size)
            f.seek(0)
            header_txt = f.read(header_size).decode('utf8')
        
        # explore xml header
        root = ElementTree.fromstring(header_txt)

        gconf = sr = root.find('GlobalConfiguration')
        hconf = root.find('HardwareConfiguration')
        self._sampling_rate = float(hconf.attrib['samplingRate'])
        
        signal_channels = []
        for chan_ind, child in enumerate(hconf.find('Device')):
            name = child.attrib['id']
            chan_id = chan_ind  #TODO change this to str with new rawio refactoring
            dtype = 'int16' # TODO check this
            units = 'uV' # TODO check where is the info
            gain = 1. # TODO check where is the info
            offset = 0. # TODO check where is the info
            group_id = 0
            signal_channels.append((name, chan_id, self._sampling_rate, 'int16',
                                 units, gain, offset, group_id))            
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        self._raw_memmap = np.memmap(self.filename, mode='r', offset=header_size, dtype='int16')
        self._raw_memmap = self._raw_memmap.reshape(-1, signal_channels.size)

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        unit_channels = []
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)

        # fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_channels'] = signal_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        size = self._raw_memmap.shape[0]
        t_stop = size / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        size = self._raw_memmap.shape[0]
        return size

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        raw_signals = self._raw_memmap[slice(i_start, i_stop), :][:, channel_indexes]
        return raw_signals

