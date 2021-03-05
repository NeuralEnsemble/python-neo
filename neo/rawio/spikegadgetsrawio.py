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
            
            #~ print(header_txt[-10:])
            #~ f.seek(header_size)
            #~ print(f.read(10))
        
        #~ exit()
        # explore xml header
        root = ElementTree.fromstring(header_txt)
        gconf = sr = root.find('GlobalConfiguration')
        hconf = root.find('HardwareConfiguration')
        sconf = root.find('SpikeConfiguration')
        
        self._sampling_rate = float(hconf.attrib['samplingRate'])
        num_ephy_channels = int(hconf .attrib['numChannels'])
        
        # explore sub stream and count packet size
        # first bytes is 0x55
        packet_size = 1
        
        #~ main_dtype = []
        stream_bytes = {}
        for device in hconf:
            stream_id = device.attrib['name']
            num_bytes = int(device.attrib['numBytes'])
            stream_bytes[stream_id] = packet_size
            packet_size += num_bytes
        #~ print('packet_size', packet_size)
        #~ print(stream_bytes)
        #~ exit()
        
        # timesteamps 4 uint32
        self._timestamp_byte = packet_size
        packet_size += 4
        
        packet_size += 2 * num_ephy_channels
        
        print('packet_size', packet_size)
        
        #~ num_ephy_channels = num_ephy_channels * 4
        #~ num_ephy_channels = 0
        #~ chan_ids = []
        #~ for chan_ind, trode in enumerate(sconf):
            #~ for  spikechan in trode:
                #~ print(spikechan, spikechan.attrib)
                #~ num_ephy_channels += 1
                #~ chan_ids.append(int(spikechan.attrib['hwChan']))
        #~ print('num_ephy_channels', num_ephy_channels)
        #~ print(np.sort(chan_ids))
        
        
        
        
        
        raw_memmap = np.memmap(self.filename, mode='r', offset=header_size, dtype='<u1')
        
        inds, = np.nonzero(raw_memmap == 0x55)
        #~ print(inds)
        #~ print(np.diff(inds))
        #~ exit()
        
        
        num_packet = raw_memmap.size // packet_size
        #~ print(num_packet, num_packet*packet_size, raw_memmap.size)
        raw_memmap = raw_memmap[:num_packet*packet_size]
        self._raw_memmap = raw_memmap.reshape(-1, packet_size)
        

        stream_ids = []
        signal_streams = []
        signal_channels = []
        
        # walk deveice and keep only "analog" one
        self._mask_channels_bytes  = {}
        for device in hconf:
            stream_id = device.attrib['name']
            for channel in device:
                #~ print(channel, channel.attrib)
                
                if 'interleavedDataIDByte' in channel.attrib:
                    # TODO deal with "headstageSensor" wich have interleaved
                    continue
                
                if channel.attrib['dataType'] == 'analog':
                    
                    if stream_id not in stream_ids:
                        stream_ids.append(stream_id)
                        stream_name = stream_id
                        signal_streams.append((stream_name, stream_id))
                        self._mask_channels_bytes[stream_id] = []
                    
                    name = channel.attrib['id']
                    chan_id = channel.attrib['id']
                    dtype = 'int16' # TODO check this
                    units = 'uV' # TODO check where is the info
                    gain = 1. # TODO check where is the info
                    offset = 0. # TODO check where is the info
                    signal_channels.append((name, chan_id, self._sampling_rate, 'int16',
                                         units, gain, offset, stream_id))
                    
                    #~ self._bytes_in_streams[stream_id].append()
                    num_bytes = stream_bytes[stream_id] + int(channel.attrib['startByte'])
                    chan_mask = np.zeros(packet_size, dtype='bool')
                    # int6: 2 bytes
                    chan_mask[num_bytes] = True
                    chan_mask[num_bytes+1] = True
                    self._mask_channels_bytes[stream_id].append(chan_mask)
        
        if num_ephy_channels > 0:
            stream_id = 'trodes'
            stream_name = stream_id
            signal_streams.append((stream_name, stream_id))
            self._mask_channels_bytes[stream_id] = []
            
            chan_ind = 0
            for trode in sconf:
                for  schan in trode:
                    name = 'trode' + trode.attrib['id'] + 'chan' + schan.attrib['hwChan']
                    chan_id = schan.attrib['hwChan']
                    units = 'uV' # TODO check where is the info
                    gain = 1. # TODO check where is the info
                    offset = 0. # TODO check where is the info
                    signal_channels.append((name, chan_id, self._sampling_rate, 'int16',
                                         units, gain, offset, stream_id))
                    
                    chan_mask = np.zeros(packet_size, dtype='bool')
                    
                    num_bytes = packet_size - 2 * num_ephy_channels + 2 * chan_ind
                    chan_mask[num_bytes] = True
                    chan_mask[num_bytes+1] = True
                    self._mask_channels_bytes[stream_id].append(chan_mask)
                    
                    chan_ind += 1
        
        # make mask as array
        self._mask_streams = {}
        for stream_id, l in self._mask_channels_bytes.items():
            mask = np.array(l)
            self._mask_channels_bytes[stream_id] = mask
            self._mask_streams[stream_id] = np.any(mask, axis=0)



        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        

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

        raw_unit8 = self._raw_memmap[i_start:i_stop]
        #~ print('raw_unit8', raw_unit8.shape, raw_unit8.dtype)
        
        num_chan = len(self._mask_channels_bytes[stream_id])
        if channel_indexes is None:
            # no loop
            stream_mask = self._mask_streams[stream_id]
        else:
            #~ print('channel_indexes', channel_indexes)
            #~ print('chan_inds', chan_inds)

            if  instance(channel_indexes, slice):
                chan_inds = np.arange(num_chan)[channel_indexes]
            else:
                chan_inds = channel_indexes
            stream_mask = np.zeros(raw_unit8.shape[1], dtype='bool')
            for chan_ind in chan_inds:
                chan_mask = self._mask_channels_bytes[stream_id][chan_ind]
                stream_mask |= chan_mask

        
        
        #~ print(stream_mask)
        
        # thisi fo a copy
        raw_unit8_mask = raw_unit8[:, stream_mask]
        shape = raw_unit8_mask.shape
        shape = (shape[0], shape[1] // 2)
        raw_unit16 = raw_unit8_mask.flatten().view('int16').reshape(shape)
        #~ print(raw_unit16.shape,raw_unit16.strides)
        
        return raw_unit16


