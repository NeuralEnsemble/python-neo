"""
Class for reading  spikegadgets file.
Only signals ability at the moment.

https://spikegadgets.com/spike-products/

Some doc here: https://bitbucket.org/mkarlsso/trodes/wiki/Configuration

The file ".rec" have :
  * a fist part in text with xml informations
  * a second part for the binary buffer

Author: Samuel Garcia
"""
from .baserawio import (BaseRawIO, _signal_channel_dtype,  _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype)

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
            
            f.seek(0)
            header_txt = f.read(header_size).decode('utf8')
        
        # explore xml header
        root = ElementTree.fromstring(header_txt)

        gconf = sr = root.find('GlobalConfiguration')
        hconf = root.find('HardwareConfiguration')
        self._sampling_rate = float(hconf.attrib['samplingRate'])
        
        # explore sub stream
        # the raw part is a complex vector of struct that depend on channel maps.
        # the "main_dtype" represent it
        main_dtype = []
        for device in hconf:
            bytes = int(device.attrib['numBytes'])
            name = device.attrib['name']
            sub_dtype = (name, 'u1', (bytes, ))
            main_dtype.append(sub_dtype)
        self._main_dtype = np.dtype(main_dtype)
        
        self._raw_memmap = np.memmap(self.filename, mode='r', offset=header_size, dtype=self._main_dtype)
        print(self._raw_memmap[:3])
        
        # wlak channels and keep only "analog" one
        stream_ids = []
        signal_streams = []
        signal_channels = []
        for device in hconf:
            stream_id = device.attrib['name']
            for channel in device:
                #~ print(channel, channel.attrib)
                
                if channel.attrib['dataType'] == 'analog':
                    
                    if stream_id not in stream_ids:
                        stream_ids.append(stream_id)
                        stream_name = stream_id
                        signal_streams.append((stream_name, stream_id))
                    
                    name = channel.attrib['id']
                    chan_id = channel.attrib['id']
                    dtype = 'uint8' # TODO check this
                    units = 'uV' # TODO check where is the info
                    gain = 1. # TODO check where is the info
                    offset = 0. # TODO check where is the info
                    signal_channels.append((name, chan_id, self._sampling_rate, 'int16',
                                         units, gain, offset, stream_id))            

        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        print(signal_channels)
        print(signal_streams)
        

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        size = self._raw_memmap.shape[0]
        t_stop = size / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        size = self._raw_memmap.shape[0]
        return size

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
        stream_id = self.header['signal_streams'][stream_index]['id']
        print(stream_id)
        
        raw = self._raw_memmap[i_start:i_stop]
        print(raw.dtype)
        
        print(raw[stream_id])
        
        
        
        #~ if channel_indexes is None:
            #~ channel_indexes = slice(None)
        #~ raw_signals = [:, channel_indexes]
        #~ return raw_signals

