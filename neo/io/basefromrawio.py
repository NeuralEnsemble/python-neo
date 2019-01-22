# -*- coding: utf-8 -*-
"""
BaseFromRaw
======

BaseFromRaw implement a bridge between the new neo.rawio API
and the neo.io legacy that give neo.core object.
The neo.rawio API is more restricted and limited and do not cover tricky
cases with asymetrical tree of neo object.
But if a format is done in neo.rawio the neo.io is done for free
by inheritance of this class.
Furthermore, IOs that inherits this BaseFromRaw also have the ability
of the lazy load with proxy objects.


"""
# needed for python 3 compatibility
from __future__ import print_function, division, absolute_import
# from __future__ import unicode_literals is not compatible with numpy.dtype both py2 py3

import warnings
import collections
import logging
import numpy as np

from neo import logging_handler
from neo.core import (AnalogSignal, Block,
                      Epoch, Event,
                      IrregularlySampledSignal,
                      ChannelIndex,
                      Segment, SpikeTrain, Unit)
from neo.io.baseio import BaseIO

from neo.io.proxyobjects import (AnalogSignalProxy,
                SpikeTrainProxy, EventProxy, EpochProxy,
                ensure_signal_units, check_annotations,
                ensure_second, proxyobjectlist)


import quantities as pq


class BaseFromRaw(BaseIO):
    """
    This implement generic reader on top of RawIO reader.

    Arguments depend on `mode` (dir or file)

    File case::

        reader = BlackRockIO(filename='FileSpec2.3001.nev')

    Dir case::

        reader = NeuralynxIO(dirname='Cheetah_v5.7.4/original_data')

    Other arguments are IO specific.

    """
    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, AnalogSignal,
                         SpikeTrain, Unit, ChannelIndex, Event, Epoch]
    readable_objects = [Block, Segment]
    writeable_objects = []

    support_lazy = True

    name = 'BaseIO'
    description = ''
    extentions = []

    mode = 'file'

    _prefered_signal_group_mode = 'split-all'  # 'group-by-same-units'
    _prefered_units_group_mode = 'split-all'  # 'all-in-one'

    def __init__(self, *args, **kargs):
        BaseIO.__init__(self, *args, **kargs)
        self.parse_header()

    def read_block(self, block_index=0, lazy=False, signal_group_mode=None,
                   units_group_mode=None, load_waveforms=False):
        """


        :param block_index: int default 0. In case of several block block_index can be specified.

        :param lazy: False by default.

        :param signal_group_mode: 'split-all' or 'group-by-same-units' (default depend IO):
        This control behavior for grouping channels in AnalogSignal.
            * 'split-all': each channel will give an AnalogSignal
            * 'group-by-same-units' all channel sharing the same quantity units ar grouped in
            a 2D AnalogSignal

        :param units_group_mode: 'split-all' or 'all-in-one'(default depend IO)
        This control behavior for grouping Unit in ChannelIndex:
            * 'split-all': each neo.Unit is assigned to a new neo.ChannelIndex
            * 'all-in-one': all neo.Unit are grouped in the same neo.ChannelIndex
              (global spike sorting for instance)

        :param load_waveforms: False by default. Control SpikeTrains.waveforms is None or not.

        """

        if signal_group_mode is None:
            signal_group_mode = self._prefered_signal_group_mode

        if units_group_mode is None:
            units_group_mode = self._prefered_units_group_mode

        # annotations
        bl_annotations = dict(self.raw_annotations['blocks'][block_index])
        bl_annotations.pop('segments')
        bl_annotations = check_annotations(bl_annotations)

        bl = Block(**bl_annotations)

        # ChannelIndex are plit in 2 parts:
        #  * some for AnalogSignals
        #  * some for Units

        # ChannelIndex for AnalogSignals
        all_channels = self.header['signal_channels']
        channel_indexes_list = self.get_group_channel_indexes()
        for channel_index in channel_indexes_list:
            for i, (ind_within, ind_abs) in self._make_signal_channel_subgroups(
                    channel_index, signal_group_mode=signal_group_mode).items():
                if signal_group_mode == "split-all":
                    chidx_annotations = self.raw_annotations['signal_channels'][i]
                elif signal_group_mode == "group-by-same-units":
                    # this should be done with array_annotation soon:
                    keys = list(self.raw_annotations['signal_channels'][ind_abs[0]].keys())
                    # take key from first channel of the group
                    chidx_annotations = {key: [] for key in keys}
                    for j in ind_abs:
                        for key in keys:
                            v = self.raw_annotations['signal_channels'][j].get(key, None)
                            chidx_annotations[key].append(v)
                if 'name' in list(chidx_annotations.keys()):
                    chidx_annotations.pop('name')
                chidx_annotations = check_annotations(chidx_annotations)
                # this should be done with array_annotation soon:
                ch_names = all_channels[ind_abs]['name'].astype('S')
                neo_channel_index = ChannelIndex(index=ind_within,
                                                 channel_names=ch_names,
                                                 channel_ids=all_channels[ind_abs]['id'],
                                                 name='Channel group {}'.format(i),
                                                 **chidx_annotations)

                bl.channel_indexes.append(neo_channel_index)

        # ChannelIndex and Unit
        # 2 case are possible in neo defifferent IO have choosen one or other:
        #  * All units are grouped in the same ChannelIndex and indexes are all channels:
        #    'all-in-one'
        #  * Each units is assigned to one ChannelIndex: 'split-all'
        # This is kept for compatibility
        unit_channels = self.header['unit_channels']
        if units_group_mode == 'all-in-one':
            if unit_channels.size > 0:
                channel_index = ChannelIndex(index=np.array([], dtype='i'),
                                             name='ChannelIndex for all Unit')
                bl.channel_indexes.append(channel_index)
            for c in range(unit_channels.size):
                unit_annotations = self.raw_annotations['unit_channels'][c]
                unit_annotations = check_annotations(unit_annotations)
                unit = Unit(**unit_annotations)
                channel_index.units.append(unit)

        elif units_group_mode == 'split-all':
            for c in range(len(unit_channels)):
                unit_annotations = self.raw_annotations['unit_channels'][c]
                unit_annotations = check_annotations(unit_annotations)
                unit = Unit(**unit_annotations)
                channel_index = ChannelIndex(index=np.array([], dtype='i'),
                                             name='ChannelIndex for Unit')
                channel_index.units.append(unit)
                bl.channel_indexes.append(channel_index)

        # Read all segments
        for seg_index in range(self.segment_count(block_index)):
            seg = self.read_segment(block_index=block_index, seg_index=seg_index,
                                    lazy=lazy, signal_group_mode=signal_group_mode,
                                    load_waveforms=load_waveforms)
            bl.segments.append(seg)

        # create link to other containers ChannelIndex and Units
        for seg in bl.segments:
            for c, anasig in enumerate(seg.analogsignals):
                bl.channel_indexes[c].analogsignals.append(anasig)

            nsig = len(seg.analogsignals)
            for c, sptr in enumerate(seg.spiketrains):
                if units_group_mode == 'all-in-one':
                    bl.channel_indexes[nsig].units[c].spiketrains.append(sptr)
                elif units_group_mode == 'split-all':
                    bl.channel_indexes[nsig + c].units[0].spiketrains.append(sptr)

        bl.create_many_to_one_relationship()

        return bl

    def read_segment(self, block_index=0, seg_index=0, lazy=False,
                     signal_group_mode=None, load_waveforms=False, time_slice=None,
                     strict_slicing=True):
        """
        :param block_index: int default 0. In case of several block block_index can be specified.

        :param seg_index: int default 0. Index of segment.

        :param lazy: False by default.

        :param signal_group_mode: 'split-all' or 'group-by-same-units' (default depend IO):
        This control behavior for grouping channels in AnalogSignal.
            * 'split-all': each channel will give an AnalogSignal
            * 'group-by-same-units' all channel sharing the same quantity units ar grouped in
            a 2D AnalogSignal

        :param load_waveforms: False by default. Control SpikeTrains.waveforms is None or not.

        :param time_slice: None by default means no limit.
            A time slice is (t_start, t_stop) both are quantities.
            All object AnalogSignal, SpikeTrain, Event, Epoch will load only in the slice.

        :param strict_slicing: True by default.
             Control if an error is raise or not when one of  time_slice member (t_start or t_stop)
             is outside the real time range of the segment.
        """

        if lazy:
            assert time_slice is None, 'For lazy=true you must specify time_slice when loading'

        if signal_group_mode is None:
            signal_group_mode = self._prefered_signal_group_mode

        # annotations
        seg_annotations = dict(self.raw_annotations['blocks'][block_index]['segments'][seg_index])
        for k in ('signals', 'units', 'events'):
            seg_annotations.pop(k)
        seg_annotations = check_annotations(seg_annotations)

        seg = Segment(index=seg_index, **seg_annotations)

        # AnalogSignal
        signal_channels = self.header['signal_channels']
        if signal_channels.size > 0:
            channel_indexes_list = self.get_group_channel_indexes()
            for channel_indexes in channel_indexes_list:
                for i, (ind_within, ind_abs) in self._make_signal_channel_subgroups(
                        channel_indexes,
                        signal_group_mode=signal_group_mode).items():
                    # make a proxy...
                    anasig = AnalogSignalProxy(rawio=self, global_channel_indexes=ind_abs,
                                    block_index=block_index, seg_index=seg_index)

                    if not lazy:
                        # ... and get the real AnalogSIgnal if not lazy
                        anasig = anasig.load(time_slice=time_slice, strict_slicing=strict_slicing)
                        # TODO magnitude_mode='rescaled'/'raw'

                    anasig.segment = seg
                    seg.analogsignals.append(anasig)

        # SpikeTrain and waveforms (optional)
        unit_channels = self.header['unit_channels']
        for unit_index in range(len(unit_channels)):
            # make a proxy...
            sptr = SpikeTrainProxy(rawio=self, unit_index=unit_index,
                                                block_index=block_index, seg_index=seg_index)

            if not lazy:
                # ... and get the real SpikeTrain if not lazy
                sptr = sptr.load(time_slice=time_slice, strict_slicing=strict_slicing,
                                        load_waveforms=load_waveforms)
                # TODO magnitude_mode='rescaled'/'raw'

            sptr.segment = seg
            seg.spiketrains.append(sptr)

        # Events/Epoch
        event_channels = self.header['event_channels']
        for chan_ind in range(len(event_channels)):
            if event_channels['type'][chan_ind] == b'event':
                e = EventProxy(rawio=self, event_channel_index=chan_ind,
                                        block_index=block_index, seg_index=seg_index)
                if not lazy:
                    e = e.load(time_slice=time_slice, strict_slicing=strict_slicing)
                e.segment = seg
                seg.events.append(e)
            elif event_channels['type'][chan_ind] == b'epoch':
                e = EpochProxy(rawio=self, event_channel_index=chan_ind,
                                        block_index=block_index, seg_index=seg_index)
                if not lazy:
                    e = e.load(time_slice=time_slice, strict_slicing=strict_slicing)
                e.segment = seg
                seg.epochs.append(e)

        seg.create_many_to_one_relationship()
        return seg

    def _make_signal_channel_subgroups(self, channel_indexes,
                                       signal_group_mode='group-by-same-units'):
        """
        For some RawIO channel are already splitted in groups.
        But in any cases, channel need to be splitted again in sub groups
        because they do not have the same units.

        They can also be splitted one by one to match previous behavior for
        some IOs in older version of neo (<=0.5).

        This method aggregate signal channels with same units or split them all.
        """
        all_channels = self.header['signal_channels']
        if channel_indexes is None:
            channel_indexes = np.arange(all_channels.size, dtype=int)
        channels = all_channels[channel_indexes]

        groups = collections.OrderedDict()
        if signal_group_mode == 'group-by-same-units':
            all_units = np.unique(channels['units'])

            for i, unit in enumerate(all_units):
                ind_within, = np.nonzero(channels['units'] == unit)
                ind_abs = channel_indexes[ind_within]
                groups[i] = (ind_within, ind_abs)

        elif signal_group_mode == 'split-all':
            for i, chan_index in enumerate(channel_indexes):
                ind_within = [i]
                ind_abs = channel_indexes[ind_within]
                groups[i] = (ind_within, ind_abs)
        else:
            raise (NotImplementedError)
        return groups
