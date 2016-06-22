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
        :class:`EpochArray`
        :class:`Event`
        :class:`EventArray`
        :class:`AnalogSignal`
        :class:`AnalogSignalArray`
        :class:`IrregularlySampledSignal`
        :class:`Spike`
        :class:`SpikeTrain`

    '''

    _data_child_objects = ('AnalogSignal', 'AnalogSignalArray',
                           'Epoch', 'EpochArray',
                           'Event', 'EventArray',
                           'IrregularlySampledSignal',
                           'Spike', 'SpikeTrain')
    _single_parent_objects = ('Block',)
    _recommended_attrs = ((('file_datetime', datetime),
                           ('rec_datetime', datetime),
                           ('index', int)) +
                          Container._recommended_attrs)
    _repr_pretty_containers = ('analogsignals', 'analogsignalarrays')

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
        t_starts = [sig.t_start for sig in self.analogsignalarrays + self.analogsignals +  self.spiketrains]
        t_starts += [e.times[0] for e in self.epocharrays + self.eventarrays + self.irregularlysampledsignals if len(e.times)>0]
        t_starts += [e.time for e in self.events + self.epochs + self.spikes]

        # t_start is not defined if no children are present
        if len(t_starts)==0:
            return None

        t_start = min(t_starts)
        return t_start

    # t_stop attribute is handled as a property so type checking can be done
    @property
    def t_stop(self):
        '''
        Time when last signal ends.
        '''
        t_stops = [sig.t_stop for sig in self.analogsignalarrays + self.analogsignals +  self.spiketrains]
        t_stops += [e.times[-1] for e in self.epocharrays + self.eventarrays + self.irregularlysampledsignals if len(e.times)>0]
        t_stops += [e.time for e in self.events + self.epochs + self.spikes]

        # t_stop is not defined if no children are present
        if len(t_stops)==0:
            return None

        t_stop = max(t_stops)
        return t_stop

    def take_spikes_by_unit(self, unit_list=None):
        '''
        Return :class:`Spike` objects in the :class:`Segment` that are also in
        a :class:`Unit` in the :attr:`unit_list` provided.
        '''
        if unit_list is None:
            return []
        spike_list = []
        for spike in self.spikes:
            if spike.unit in unit_list:
                spike_list.append(spike)
        return spike_list

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

    def take_analogsignal_by_unit(self, unit_list=None):
        '''
        Return :class:`AnalogSignal` objects in the :class:`Segment` that are
        have the same :attr:`channel_index` as any of the :class:`Unit: objects
        in the :attr:`unit_list` provided.
        '''
        if unit_list is None:
            return []
        channel_indexes = []
        for unit in unit_list:
            if unit.channel_indexes is not None:
                channel_indexes.extend(unit.channel_indexes)
        return self.take_analogsignal_by_channelindex(channel_indexes)

    def take_analogsignal_by_channelindex(self, channel_indexes=None):
        '''
        Return :class:`AnalogSignal` objects in the :class:`Segment` that have
        a :attr:`channel_index` that is in the :attr:`channel_indexes`
        provided.
        '''
        if channel_indexes is None:
            return []
        anasig_list = []
        for anasig in self.analogsignals:
            if anasig.channel_index in channel_indexes:
                anasig_list.append(anasig)
        return anasig_list

    def take_slice_of_analogsignalarray_by_unit(self, unit_list=None):
        '''
        Return slices of the :class:`AnalogSignalArray` objects in the
        :class:`Segment` that correspond to a :attr:`channel_index`  of any of
        the :class:`Unit` objects in the :attr:`unit_list` provided.
        '''
        if unit_list is None:
            return []
        indexes = []
        for unit in unit_list:
            if unit.channel_indexes is not None:
                indexes.extend(unit.channel_indexes)

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
        for sigarr in self.analogsignalarrays:
            if sigarr.channel_indexes is not None:
                ind = np.in1d(sigarr.channel_indexes, channel_indexes)
                sliced_sigarrays.append(sigarr[:, ind])

        return sliced_sigarrays

    def construct_subsegment_by_unit(self, unit_list=None):
        '''
        Return a new :class:`Segment that contains the :class:`AnalogSignal`,
        :class:`AnalogSignalArray`, :class:`Spike`:, and :class:`SpikeTrain`
        objects common to both the current :class:`Segment` and any
        :class:`Unit` in the :attr:`unit_list` provided.

        *Example*::

            >>> from neo.core import (Segment, Block, Unit, SpikeTrain,
            ...                       RecordingChannelGroup)
            >>>
            >>> blk = Block()
            >>> rcg = RecordingChannelGroup(name='group0')
            >>> blk.recordingchannelgroups = [rcg]
            >>>
            >>> for ind in range(5):
            ...         unit = Unit(name='Unit #%s' % ind, channel_index=ind)
            ...         rcg.units.append(unit)
            ...
            >>>
            >>> for ind in range(3):
            ...     seg = Segment(name='Simulation #%s' % ind)
            ...     blk.segments.append(seg)
            ...     for unit in rcg.units:
            ...         train = SpikeTrain([1, 2, 3], units='ms', t_start=0.,
            ...                            t_stop=10)
            ...         train.unit = unit
            ...         unit.spiketrains.append(train)
            ...         seg.spiketrains.append(train)
            ...
            >>>
            >>> seg0 = blk.segments[-1]
            >>> seg1 = seg0.construct_subsegment_by_unit(rcg.units[:2])
            >>> len(seg0.spiketrains)
            5
            >>> len(seg1.spiketrains)
            2

        '''
        seg = Segment()
        seg.analogsignals = self.take_analogsignal_by_unit(unit_list)
        seg.spikes = self.take_spikes_by_unit(unit_list)
        seg.spiketrains = self.take_spiketrains_by_unit(unit_list)
        seg.analogsignalarrays = \
            self.take_slice_of_analogsignalarray_by_unit(unit_list)
        #TODO copy others attributes
        return seg
