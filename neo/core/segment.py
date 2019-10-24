# -*- coding: utf-8 -*-
'''
This module defines :class:`Segment`, a container for data sharing a common
time basis.

:class:`Segment` derives from :class:`Container`,
from :module:`neo.core.container`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

from datetime import datetime

import numpy as np

from copy import deepcopy

from neo.core.container import Container


class Segment(Container):
    '''
    A container for data sharing a common time basis.

    A :class:`Segment` is a heterogeneous container for discrete or continous
    data sharing a common clock (time basis) but not necessary the same
    sampling rate, start or end time.

    *Usage*::
        >>> from neo.core import Segment, SpikeTrain, AnalogSignal
        >>> from quantities import Hz, s
        >>>
        >>> seg = Segment(index=5)
        >>>
        >>> train0 = SpikeTrain(times=[.01, 3.3, 9.3], units='sec', t_stop=10)
        >>> seg.spiketrains.append(train0)
        >>>
        >>> train1 = SpikeTrain(times=[100.01, 103.3, 109.3], units='sec',
        ...                     t_stop=110)
        >>> seg.spiketrains.append(train1)
        >>>
        >>> sig0 = AnalogSignal(signal=[.01, 3.3, 9.3], units='uV',
        ...                     sampling_rate=1*Hz)
        >>> seg.analogsignals.append(sig0)
        >>>
        >>> sig1 = AnalogSignal(signal=[100.01, 103.3, 109.3], units='nA',
        ...                     sampling_period=.1*s)
        >>> seg.analogsignals.append(sig1)

    *Required attributes/properties*:
        None

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :file_datetime: (datetime) The creation date and time of the original
            data file.
        :rec_datetime: (datetime) The date and time of the original recording
        :index: (int) You can use this to define a temporal ordering of
            your Segment. For instance you could use this for trial numbers.

    Note: Any other additional arguments are assumed to be user-specific
    metadata and stored in :attr:`annotations`.

    *Properties available on this object*:
        :all_data: (list) A list of all child objects in the :class:`Segment`.

    *Container of*:
        :class:`Epoch`
        :class:`Event`
        :class:`AnalogSignal`
        :class:`IrregularlySampledSignal`
        :class:`SpikeTrain`

    '''

    _data_child_objects = ('AnalogSignal',
                           'Epoch', 'Event',
                           'IrregularlySampledSignal', 'SpikeTrain', 'ImageSequence')
    _single_parent_objects = ('Block',)
    _recommended_attrs = ((('file_datetime', datetime),
                           ('rec_datetime', datetime),
                           ('index', int)) +
                          Container._recommended_attrs)
    _repr_pretty_containers = ('analogsignals',)

    def __init__(self, name=None, description=None, file_origin=None,
                 file_datetime=None, rec_datetime=None, index=None,
                 **annotations):
        '''
        Initialize a new :class:`Segment` instance.
        '''
        super(Segment, self).__init__(name=name, description=description,
                                      file_origin=file_origin, **annotations)

        self.file_datetime = file_datetime
        self.rec_datetime = rec_datetime
        self.index = index

    # t_start attribute is handled as a property so type checking can be done
    @property
    def t_start(self):
        '''
        Time when first signal begins.
        '''
        t_starts = [sig.t_start for sig in self.analogsignals +
                    self.spiketrains + self.irregularlysampledsignals]
        t_starts += [e.times[0] for e in self.epochs + self.events if len(e.times) > 0]

        # t_start is not defined if no children are present
        if len(t_starts) == 0:
            return None

        t_start = min(t_starts)
        return t_start

    # t_stop attribute is handled as a property so type checking can be done
    @property
    def t_stop(self):
        '''
        Time when last signal ends.
        '''
        t_stops = [sig.t_stop for sig in self.analogsignals +
                   self.spiketrains + self.irregularlysampledsignals]
        t_stops += [e.times[-1] for e in self.epochs + self.events if len(e.times) > 0]

        # t_stop is not defined if no children are present
        if len(t_stops) == 0:
            return None

        t_stop = max(t_stops)
        return t_stop

    def take_spiketrains_by_unit(self, unit_list=None):
        '''
        Return :class:`SpikeTrains` in the :class:`Segment` that are also in a
        :class:`Unit` in the :attr:`unit_list` provided.
        '''
        if unit_list is None:
            return []
        spiketrain_list = []
        for spiketrain in self.spiketrains:
            if spiketrain.unit in unit_list:
                spiketrain_list.append(spiketrain)
        return spiketrain_list

    # def take_analogsignal_by_unit(self, unit_list=None):
    #     '''
    #     Return :class:`AnalogSignal` objects in the :class:`Segment` that are
    #     have the same :attr:`channel_index` as any of the :class:`Unit: objects
    #     in the :attr:`unit_list` provided.
    #     '''
    #     if unit_list is None:
    #         return []
    #     channel_indexes = []
    #     for unit in unit_list:
    #         if unit.channel_indexes is not None:
    #             channel_indexes.extend(unit.channel_indexes)
    #     return self.take_analogsignal_by_channelindex(channel_indexes)
    #
    # def take_analogsignal_by_channelindex(self, channel_indexes=None):
    #     '''
    #     Return :class:`AnalogSignal` objects in the :class:`Segment` that have
    #     a :attr:`channel_index` that is in the :attr:`channel_indexes`
    #     provided.
    #     '''
    #     if channel_indexes is None:
    #         return []
    #     anasig_list = []
    #     for anasig in self.analogsignals:
    #         if anasig.channel_index in channel_indexes:
    #             anasig_list.append(anasig)
    #     return anasig_list

    def take_slice_of_analogsignalarray_by_unit(self, unit_list=None):
        '''
        Return slices of the :class:`AnalogSignal` objects in the
        :class:`Segment` that correspond to a :attr:`channel_index`  of any of
        the :class:`Unit` objects in the :attr:`unit_list` provided.
        '''
        if unit_list is None:
            return []
        indexes = []
        for unit in unit_list:
            if unit.get_channel_indexes() is not None:
                indexes.extend(unit.get_channel_indexes())

        return self.take_slice_of_analogsignalarray_by_channelindex(indexes)

    def take_slice_of_analogsignalarray_by_channelindex(self,
                                                        channel_indexes=None):
        '''
        Return slices of the :class:`AnalogSignalArrays` in the
        :class:`Segment` that correspond to the :attr:`channel_indexes`
        provided.
        '''
        if channel_indexes is None:
            return []

        sliced_sigarrays = []
        for sigarr in self.analogsignals:
            if sigarr.get_channel_index() is not None:
                ind = np.in1d(sigarr.get_channel_index(), channel_indexes)
                sliced_sigarrays.append(sigarr[:, ind])

        return sliced_sigarrays

    def construct_subsegment_by_unit(self, unit_list=None):
        '''
        Return a new :class:`Segment that contains the :class:`AnalogSignal`,
        :class:`AnalogSignal`, and :class:`SpikeTrain`
        objects common to both the current :class:`Segment` and any
        :class:`Unit` in the :attr:`unit_list` provided.

        *Example*::

            >>> from neo.core import (Segment, Block, Unit, SpikeTrain,
            ...                       ChannelIndex)
            >>>
            >>> blk = Block()
            >>> chx = ChannelIndex(name='group0')
            >>> blk.channel_indexes = [chx]
            >>>
            >>> for ind in range(5):
            ...         unit = Unit(name='Unit #%s' % ind, channel_index=ind)
            ...         chx.units.append(unit)
            ...
            >>>
            >>> for ind in range(3):
            ...     seg = Segment(name='Simulation #%s' % ind)
            ...     blk.segments.append(seg)
            ...     for unit in chx.units:
            ...         train = SpikeTrain([1, 2, 3], units='ms', t_start=0.,
            ...                            t_stop=10)
            ...         train.unit = unit
            ...         unit.spiketrains.append(train)
            ...         seg.spiketrains.append(train)
            ...
            >>>
            >>> seg0 = blk.segments[-1]
            >>> seg1 = seg0.construct_subsegment_by_unit(chx.units[:2])
            >>> len(seg0.spiketrains)
            5
            >>> len(seg1.spiketrains)
            2

        '''
        seg = Segment()
        seg.spiketrains = self.take_spiketrains_by_unit(unit_list)
        seg.analogsignals = \
            self.take_slice_of_analogsignalarray_by_unit(unit_list)
        # TODO copy others attributes
        return seg

    def time_slice(self, t_start=None, t_stop=None, reset_time=False, **kwargs):
        """
        Creates a time slice of a Segment containing slices of all child
        objects.

        Parameters:
        -----------
        t_start: Quantity
            Starting time of the sliced time window.
        t_stop: Quantity
            Stop time of the sliced time window.
        reset_time: bool
            If True the time stamps of all sliced objects are set to fall
            in the range from t_start to t_stop.
            If False, original time stamps are retained.
            Default is False.

        Keyword Arguments:
        ------------------
            Additional keyword arguments used for initialization of the sliced
            Segment object.

        Returns:
        --------
        subseg: Segment
            Temporal slice of the original Segment from t_start to t_stop.
        """
        subseg = Segment(**kwargs)

        for attr in ['file_datetime', 'rec_datetime', 'index',
                     'name', 'description', 'file_origin']:
            setattr(subseg, attr, getattr(self, attr))

        subseg.annotations = deepcopy(self.annotations)

        t_shift = - t_start

        # cut analogsignals and analogsignalarrays
        for ana_id in range(len(self.analogsignals)):
            if hasattr(self.analogsignals[ana_id], '_rawio'):
                ana_time_slice = self.analogsignals[ana_id].load(time_slice=(t_start, t_stop))
            else:
                ana_time_slice = self.analogsignals[ana_id].time_slice(t_start, t_stop)
            if reset_time:
                ana_time_slice = ana_time_slice.time_shift(t_shift)
            subseg.analogsignals.append(ana_time_slice)

        # cut irregularly sampled signals
        for irr_id in range(len(self.irregularlysampledsignals)):
            if hasattr(self.irregularlysampledsignals[irr_id], '_rawio'):
                ana_time_slice = self.irregularlysampledsignals[irr_id].load(
                    time_slice=(t_start, t_stop))
            else:
                ana_time_slice = self.irregularlysampledsignals[irr_id].time_slice(t_start, t_stop)
            if reset_time:
                ana_time_slice = ana_time_slice.time_shift(t_shift)
            subseg.irregularlysampledsignals.append(ana_time_slice)

        # cut spiketrains
        for st_id in range(len(self.spiketrains)):
            if hasattr(self.spiketrains[st_id], '_rawio'):
                st_time_slice = self.spiketrains[st_id].load(time_slice=(t_start, t_stop))
            else:
                st_time_slice = self.spiketrains[st_id].time_slice(t_start, t_stop)
            if reset_time:
                st_time_slice = st_time_slice.time_shift(t_shift)
            subseg.spiketrains.append(st_time_slice)

        # cut events
        for ev_id in range(len(self.events)):
            if hasattr(self.events[ev_id], '_rawio'):
                ev_time_slice = self.events[ev_id].load(time_slice=(t_start, t_stop))
            else:
                ev_time_slice = self.events[ev_id].time_slice(t_start, t_stop)
            if reset_time:
                ev_time_slice = ev_time_slice.time_shift(t_shift)
            # appending only non-empty events
            if len(ev_time_slice):
                subseg.events.append(ev_time_slice)

        # cut epochs
        for ep_id in range(len(self.epochs)):
            if hasattr(self.epochs[ep_id], '_rawio'):
                ep_time_slice = self.epochs[ep_id].load(time_slice=(t_start, t_stop))
            else:
                ep_time_slice = self.epochs[ep_id].time_slice(t_start, t_stop)
            if reset_time:
                ep_time_slice = ep_time_slice.time_shift(t_shift)
            # appending only non-empty epochs
            if len(ep_time_slice):
                subseg.epochs.append(ep_time_slice)

        subseg.create_relationship()

        return subseg
