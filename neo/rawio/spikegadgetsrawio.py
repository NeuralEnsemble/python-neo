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
            print(header_txt[-10:])
            f.seek(header_size)
            print(f.read(10))
        
        #~ exit()
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
        #~ print(self._main_dtype)
        
        #~ print(self._main_dtype.itemsize)
        
        self._raw_memmap = np.memmap(self.filename, mode='r', offset=header_size, dtype=self._main_dtype)
        
        # wlak channels and keep only "analog" one
        stream_ids = []
        signal_streams = []
        signal_channels = []
        self._bytes_in_streams  = {}
        for device in hconf:
            stream_id = device.attrib['name']
            for channel in device:
                #~ print(channel, channel.attrib)
                
                if channel.attrib['dataType'] == 'analog':
                    
                    if stream_id not in stream_ids:
                        stream_ids.append(stream_id)
                        stream_name = stream_id
                        signal_streams.append((stream_name, stream_id))
                        self._bytes_in_streams[stream_id] = []
                    
                    name = channel.attrib['id']
                    chan_id = channel.attrib['id']
                    dtype = 'uint16' # TODO check this
                    units = 'uV' # TODO check where is the info
                    gain = 1. # TODO check where is the info
                    offset = 0. # TODO check where is the info
                    signal_channels.append((name, chan_id, self._sampling_rate, 'int16',
                                         units, gain, offset, stream_id))
                    
                    self._bytes_in_streams[stream_id].append(int(channel.attrib['startByte']))

        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        #~ print(signal_channels)
        #~ print(signal_streams)
        print(self._bytes_in_streams)
        

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
        # info from GlobalConfiguration in xml are copied to block and seg annotations
        bl_ann = self.raw_annotations['blocks'][0]
        seg_ann =  self.raw_annotations['blocks'][0]['segments'][0]
        for ann in (bl_ann, seg_ann):
            ann.update(gconf.attrib)

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
        
        raw_unit8 = self._raw_memmap[stream_id][i_start:i_stop]
        print('raw_unit8', raw_unit8.shape, raw_unit8.dtype)
        
        if channel_indexes is None:
            channel_indexes = slice(channel_indexes)
            
        nb = len(self._bytes_in_streams[stream_id])
        chan_inds = np.arange(nb)[channel_indexes]
        print('chan_inds', chan_inds)
        
        byte_mask = np.zeros(raw_unit8.shape[1], dtype='bool')
        for chan_ind in chan_inds:
            bytes = self._bytes_in_streams[stream_id][chan_ind]
            # int16
            byte_mask[bytes] = True
            byte_mask[bytes+1] = True
        
        print(byte_mask)
        
        raw_unit8_mask = raw_unit8[:, byte_mask]
        print('raw_unit8_mask', raw_unit8_mask.shape, raw_unit8_mask.strides)
        
        shape = raw_unit8_mask.shape
        shape = (shape[0], shape[1] // 2)
        raw_unit16 = raw_unit8_mask.flatten().view('uint16').reshape(shape)
        print(raw_unit16.shape,raw_unit16.strides)
        
        return raw_unit16

