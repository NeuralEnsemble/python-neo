# -*- coding: utf-8 -*-

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.blackrockrawio import BlackrockRawIO


def _move_channel_indexes_and_analogsignals(from_block, to_block):
    if len(from_block.segments) != len(to_block.segments):
        raise ValueError('Can not assign segments between block 1 and 2. Different number of '
                         'segments present.')

    for seg_id in range(len(from_block.segments)):
        for ana in from_block.segments[seg_id].analogsignals:
            # redirect links from data object to container objects
            ana.segment = to_block.segments[seg_id]
            ana.channel_index.block = to_block

            # add links from container objects to analogsignal
            ana.segment.analogsignals.append(ana)
            # channel index was already relinked for another segment
            if ana.channel_index not in to_block.channel_indexes:
                to_block.channel_indexes.append(ana.channel_index)

            # remove (now) duplicated units from channel_index, remove irregular signals
            ana.channel_index.units = []
            ana.channel_index.irregularlysampledsignals = []


class BlackrockIO_single_nsx(BlackrockRawIO, BaseFromRaw):
    """
    Supplementary class for reading BlackRock data using only a single nsx file.
    """
    name = 'Blackrock IO for single nsx'
    description = "This IO reads a pair of corresponding nev and nsX files of the Blackrock " \
                  "" + "(Cerebus) recording system."

    _prefered_signal_group_mode = 'split-all'

    def __init__(self, filename, nsx_to_load=None, **kargs):
        BlackrockRawIO.__init__(self, filename=filename, nsx_to_load=nsx_to_load, **kargs)
        BaseFromRaw.__init__(self, filename)


class BlackrockIO(BlackrockIO_single_nsx):
    name = 'Blackrock IO'
    description = "This IO reads .nev/.nsX files of the Blackrock (Cerebus) recording system."

    def __init__(self, filename, nsx_to_load='all', **kargs):
        BlackrockIO_single_nsx.__init__(self, filename)
        if nsx_to_load == 'all':
            self._selected_nsx = self._avail_nsx
        else:
            self._selected_nsx = [nsx_to_load]
        self._nsx_ios = []
        for nsx in self._selected_nsx:
            self._nsx_ios.append(BlackrockIO_single_nsx(filename, nsx_to_load=nsx, **kargs))

    def read_block(self, **kargs):
        bl = self._nsx_ios[0].read_block(**kargs)
        for nsx_ios in self._nsx_ios[1:]:
            nsx_block = nsx_ios.read_block(**kargs)
            _move_channel_indexes_and_analogsignals(nsx_block, bl)
            del nsx_block
        return bl

    def read_segment(self, **kargs):
        seg = self._nsx_ios[0].read_segment(**kargs)
        for nsx_ios in self._nsx_ios[1:]:
            nsx_seg = nsx_ios.read_segment(**kargs)
            seg.analogsignals.extend(nsx_seg.analogsignals)
            for ana in nsx_seg.analogsignals:
                ana.segment = seg
                ana.channel_index = None
            del nsx_seg
        return seg
