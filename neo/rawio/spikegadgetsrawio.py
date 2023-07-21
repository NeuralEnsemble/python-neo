"""
Class for reading spikegadgets files.
Only continuous signals are supported at the moment.

https://spikegadgets.com/spike-products/

Documentation of the format:
https://bitbucket.org/mkarlsso/trodes/wiki/Configuration

Note :
  * this file format have multiple version. news version include the gain for scaling.
     The actual implementation do not contain this feature because we don't have
     files to test this. So now the gain is "hardcoded" to 1. and so units
     is not handled correctly.

The ".rec" file format contains:
  * a first  text part with information in an XML structure
  * a second part for the binary buffer

Author: Samuel Garcia
"""
from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
    _spike_channel_dtype, _event_channel_dtype)

import numpy as np

from xml.etree import ElementTree


class SpikeGadgetsRawIO(BaseRawIO):
    extensions = ['rec']
    rawmode = 'one-file'

    def __init__(self, filename='', selected_streams=None):
        """
        Class for reading spikegadgets files.
        Only continuous signals are supported at the moment.

        Initialize a SpikeGadgetsRawIO for a single ".rec" file.

        Args:
            filename: str
                The filename
            selected_streams: None, list, str
                sublist of streams to load/expose to API
                useful for spikeextractor when one stream only is needed.
                For instance streams = ['ECU', 'trodes']
                'trodes' is name for ephy channel (ntrodes)
        """
        BaseRawIO.__init__(self)
        self.filename = filename
        self.selected_streams = selected_streams

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
                ValueError("SpikeGadgets: the xml header does not contain '</Configuration>'")

            f.seek(0)
            header_txt = f.read(header_size).decode('utf8')

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
        stream_bytes = {}
        for device in hconf:
            stream_id = device.attrib['name']
            num_bytes = int(device.attrib['numBytes'])
            stream_bytes[stream_id] = packet_size
            packet_size += num_bytes

        # timestamps 4 uint32
        self._timestamp_byte = packet_size
        packet_size += 4

        packet_size += 2 * num_ephy_channels

        # read the binary part lazily
        raw_memmap = np.memmap(self.filename, mode='r', offset=header_size, dtype='<u1')

        num_packet = raw_memmap.size // packet_size
        raw_memmap = raw_memmap[:num_packet * packet_size]
        self._raw_memmap = raw_memmap.reshape(-1, packet_size)

        # create signal channels
        stream_ids = []
        signal_streams = []
        signal_channels = []

        # walk in xml device and keep only "analog" one
        self._mask_channels_bytes = {}
        for device in hconf:
            stream_id = device.attrib['name']
            for channel in device:

                if 'interleavedDataIDByte' in channel.attrib:
                    # TODO LATER: deal with "headstageSensor" which have interleaved
                    continue

                if channel.attrib['dataType'] == 'analog':

                    if stream_id not in stream_ids:
                        stream_ids.append(stream_id)
                        stream_name = stream_id
                        signal_streams.append((stream_name, stream_id))
                        self._mask_channels_bytes[stream_id] = []

                    name = channel.attrib['id']
                    chan_id = channel.attrib['id']
                    dtype = 'int16'
                    # TODO LATER : handle gain correctly according the file version
                    units = ''
                    gain = 1.
                    offset = 0.
                    signal_channels.append((name, chan_id, self._sampling_rate, 'int16',
                                         units, gain, offset, stream_id))

                    num_bytes = stream_bytes[stream_id] + int(channel.attrib['startByte'])
                    chan_mask = np.zeros(packet_size, dtype='bool')
                    chan_mask[num_bytes] = True
                    chan_mask[num_bytes + 1] = True
                    self._mask_channels_bytes[stream_id].append(chan_mask)

        if num_ephy_channels > 0:
            stream_id = 'trodes'
            stream_name = stream_id
            signal_streams.append((stream_name, stream_id))
            self._mask_channels_bytes[stream_id] = []

            chan_ind = 0
            for trode in sconf:
                for schan in trode:
                    name = 'trode' + trode.attrib['id'] + 'chan' + schan.attrib['hwChan']
                    chan_id = schan.attrib['hwChan']
                    # TODO LATER : handle gain correctly according the file version
                    units = ''
                    gain = 1.
                    offset = 0.
                    signal_channels.append((name, chan_id, self._sampling_rate, 'int16',
                                         units, gain, offset, stream_id))

                    chan_mask = np.zeros(packet_size, dtype='bool')
                    num_bytes = packet_size - 2 * num_ephy_channels + 2 * chan_ind
                    chan_mask[num_bytes] = True
                    chan_mask[num_bytes + 1] = True
                    self._mask_channels_bytes[stream_id].append(chan_mask)

                    chan_ind += 1

        # make mask as array (used in _get_analogsignal_chunk(...))
        self._mask_streams = {}
        for stream_id, l in self._mask_channels_bytes.items():
            mask = np.array(l)
            self._mask_channels_bytes[stream_id] = mask
            self._mask_streams[stream_id] = np.any(mask, axis=0)

        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # remove some stream if no wanted
        if self.selected_streams is not None:
            if isinstance(self.selected_streams, str):
                self.selected_streams = [self.selected_streams]
            assert isinstance(self.selected_streams, list)

            keep = np.in1d(signal_streams['id'], self.selected_streams)
            signal_streams = signal_streams[keep]

            keep = np.in1d(signal_channels['stream_id'], self.selected_streams)
            signal_channels = signal_channels[keep]

        # No events channels
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes  channels
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
        seg_ann = self.raw_annotations['blocks'][0]['segments'][0]
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

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index,
                                channel_indexes):
        stream_id = self.header['signal_streams'][stream_index]['id']

        raw_unit8 = self._raw_memmap[i_start:i_stop]

        num_chan = len(self._mask_channels_bytes[stream_id])
        re_order = None
        if channel_indexes is None:
            # no loop : entire stream mask
            stream_mask = self._mask_streams[stream_id]
        else:
            # accumulate mask
            if isinstance(channel_indexes, slice):
                chan_inds = np.arange(num_chan)[channel_indexes]
            else:
                chan_inds = channel_indexes

                if np.any(np.diff(channel_indexes) < 0):
                    # handle channel are not ordered
                    sorted_channel_indexes = np.sort(channel_indexes)
                    re_order = np.array([list(sorted_channel_indexes).index(ch)
                                         for ch in channel_indexes])

            stream_mask = np.zeros(raw_unit8.shape[1], dtype='bool')
            for chan_ind in chan_inds:
                chan_mask = self._mask_channels_bytes[stream_id][chan_ind]
                stream_mask |= chan_mask

        # this copies the data from the memmap into memory
        raw_unit8_mask = raw_unit8[:, stream_mask]
        shape = raw_unit8_mask.shape
        shape = (shape[0], shape[1] // 2)
        # reshape the and retype by view
        raw_unit16 = raw_unit8_mask.flatten().view('int16').reshape(shape)

        if re_order is not None:
            raw_unit16 = raw_unit16[:, re_order]

        return raw_unit16
