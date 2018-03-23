# -*- coding: utf-8 -*-
"""
Reading from neuroscope format files.
Ref: http://neuroscope.sourceforge.net/

It is an old format from Buzsaki's lab

Some old open datasets from spike sorting
are still using this format.

This only the signals.
This should be done (but maybe never will):
  * SpikeTrain file   '.clu'  '.res'
  * Event  '.ext.evt'  or '.evt.ext'

Author: Samuel Garcia

"""
from __future__ import unicode_literals, print_function, division, absolute_import

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np
from xml.etree import ElementTree


class NeuroScopeRawIO(BaseRawIO):
    extensions = ['xml', 'dat']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename.replace('.xml', '').replace('.dat', '')

    def _parse_header(self):
        filename = self.filename.replace('.xml', '').replace('.dat', '')

        tree = ElementTree.parse(filename + '.xml')
        root = tree.getroot()
        acq = root.find('acquisitionSystem')
        nbits = int(acq.find('nBits').text)
        nb_channel = int(acq.find('nChannels').text)
        self._sampling_rate = float(acq.find('samplingRate').text)
        voltage_range = float(acq.find('voltageRange').text)
        # offset = int(acq.find('offset').text)
        amplification = float(acq.find('amplification').text)

        # find groups for channels
        channel_group = {}
        for grp_index, xml_chx in enumerate(
                root.find('anatomicalDescription').find('channelGroups').findall('group')):
            for xml_rc in xml_chx:
                channel_group[int(xml_rc.text)] = grp_index

        if nbits == 16:
            sig_dtype = 'int16'
            gain = voltage_range / (2 ** 16) / amplification / 1000.
            # ~ elif nbits==32:
            # Not sure if it is int or float
            # ~ dt = 'int32'
            # ~ gain  = voltage_range/2**32/amplification
        else:
            raise (NotImplementedError)

        self._raw_signals = np.memmap(filename + '.dat', dtype=sig_dtype,
                                      mode='r', offset=0).reshape(-1, nb_channel)

        # signals
        sig_channels = []
        for c in range(nb_channel):
            name = 'ch{}grp{}'.format(c, channel_group[c])
            chan_id = c
            units = 'mV'
            offset = 0.
            group_id = 0
            sig_channels.append((name, chan_id, self._sampling_rate,
                                 sig_dtype, units, gain, offset, group_id))
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

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
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._raw_signals.shape[0] / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        return self._raw_signals.shape[0]

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        raw_signals = self._raw_signals[slice(i_start, i_stop), channel_indexes]
        return raw_signals
