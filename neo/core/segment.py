# -*- coding: utf-8 -*-
'''
This module defines :class:`Segment`, a container for data sharing a common
time basis.

:class:`Segment` derives from :class:`BaseNeo`,
from :module:`neo.core.baseneo`.
'''

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np

from neo.core.baseneo import BaseNeo


class Segment(BaseNeo):
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

    def __init__(self, name=None, description=None, file_origin=None,
                 file_datetime=None, rec_datetime=None, index=None,
                 **annotations):
        '''
        Initialize a new :class:`Segment` instance.
        '''
        BaseNeo.__init__(self, name=name, file_origin=file_origin,
                         description=description, **annotations)
        self.file_datetime = file_datetime
        self.rec_datetime = rec_datetime
        self.index = index

        self.epochs = []
        self.epocharrays = []
        self.events = []
        self.eventarrays = []
        self.analogsignals = []
        self.analogsignalarrays = []
        self.irregularlysampledsignals = []
        self.spikes = []
        self.spiketrains = []

        self.block = None

    @property
    def all_data(self):
        '''
        Returns a list of all child objects in the :class:`Segment`.
        '''
        return sum((self.epochs, self.epocharrays, self.events,
                    self.eventarrays, self.analogsignals,
                    self.analogsignalarrays, self.irregularlysampledsignals,
                    self.spikes, self.spiketrains), [])

    def filter(self, **kwargs):
        '''
        Return a list of child objects matching *any* of the search terms
        in either their attributes or annotations.

        Examples::

            >>> segment.filter(name="Vm")
        '''
        results = []
        for key, value in kwargs.items():
            for obj in self.all_data:
                if hasattr(obj, key) and getattr(obj, key) == value:
                    results.append(obj)
                elif key in obj.annotations and obj.annotations[key] == value:
                    results.append(obj)
        return results

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

    def merge(self, other):
        '''
        Merge the contents of another :class:`Segment` into this one.

        For each array-type object in the other :class:`Segment`, if its name
        matches that of an object of the same type in this :class:`Segment`,
        the two arrays will be joined by concatenation. Non-array objects will
        just be added to this segment.
        '''
        for container in ("epochs",  "events",  "analogsignals",
                          "irregularlysampledsignals", "spikes",
                          "spiketrains"):
            getattr(self, container).extend(getattr(other, container))
        for container in ("epocharrays", "eventarrays", "analogsignalarrays"):
            objs = getattr(self, container)
            lookup = dict((obj.name, i) for i, obj in enumerate(objs))
            for obj in getattr(other, container):
                if obj.name in lookup:
                    ind = lookup[obj.name]
                    try:
                        newobj = getattr(self, container)[ind].merge(obj)
                    except AttributeError as e:
                        raise AttributeError("%s. container=%s, obj.name=%s, \
                                              shape=%s" % (e, container,
                                                           obj.name,
                                                           obj.shape))
                    getattr(self, container)[ind] = newobj
                else:
                    lookup[obj.name] = obj
                    getattr(self, container).append(obj)
        # TODO: merge annotations

    def size(self):
        '''
        Get dictionary containing the names of child containers in the current
        :class:`Segment` as keys and the number of children of that type
        as values.
        '''
        return dict((name, len(getattr(self, name)))
                    for name in ("epochs",  "events",  "analogsignals",
                                 "irregularlysampledsignals", "spikes",
                                 "spiketrains", "epocharrays", "eventarrays",
                                 "analogsignalarrays"))

    def _repr_pretty_(self, pp, cycle):
        '''
        Handle pretty-printing the :class:`Segment`.
        '''
        pp.text(self.__class__.__name__)
        pp.text(" with ")
        first = True
        for (value, readable) in [
                (self.analogsignals, "analogs"),
                (self.analogsignalarrays, "analog arrays"),
                (self.events, "events"),
                (self.eventarrays, "event arrays"),
                (self.epochs, "epochs"),
                (self.epocharrays, "epoch arrays"),
                (self.irregularlysampledsignals, "epoch arrays"),
                (self.spikes, "spikes"),
                (self.spiketrains, "spike trains"),
                ]:
            if value:
                if first:
                    first = False
                else:
                    pp.text(", ")
                pp.text("{0} {1}".format(len(value), readable))
        if self._has_repr_pretty_attrs_():
            pp.breakable()
            self._repr_pretty_attrs_(pp, cycle)

        if self.analogsignals:
            pp.breakable()
            pp.text("# Analog signals (N={0})".format(len(self.analogsignals)))
            for (i, asig) in enumerate(self.analogsignals):
                pp.breakable()
                pp.text("{0}: ".format(i))
                with pp.indent(3):
                    pp.pretty(asig)

        if self.analogsignalarrays:
            pp.breakable()
            pp.text("# Analog signal arrays (N={0})"
                    .format(len(self.analogsignalarrays)))
            for i, asarr in enumerate(self.analogsignalarrays):
                pp.breakable()
                pp.text("{0}: ".format(i))
                with pp.indent(3):
                    pp.pretty(asarr)
