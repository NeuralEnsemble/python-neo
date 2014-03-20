# -*- coding: utf-8 -*-
"""
Tools for IO coder:
  * Creating RecordingChannel and making links with AnalogSignals and
    SPikeTrains
"""

import collections

import numpy as np

from neo.core import (AnalogSignal, AnalogSignalArray, Block,
                      Epoch, EpochArray, Event, EventArray,
                      IrregularlySampledSignal,
                      RecordingChannel, RecordingChannelGroup,
                      Segment, Spike, SpikeTrain, Unit)


#def finalize_block(block):
#    populate_RecordingChannel(block)
#    block.create_many_to_one_relationship()

    # Special case this tricky many-to-many relationship
    # we still need links from recordingchannel to analogsignal
#    for rcg in block.recordingchannelgroups:
#        for rc in rcg.recordingchannels:
#            rc.create_many_to_one_relationship()


def populate_RecordingChannel(bl, remove_from_annotation=True):
    """
    When a Block is
    Block>Segment>AnalogSIgnal
    this function auto create all RecordingChannel following these rules:
      * when 'channel_index ' is in AnalogSIgnal the corresponding
        RecordingChannel is created.
      * 'channel_index ' is then set to None if remove_from_annotation
      * only one RecordingChannelGroup is created

    It is a utility at the end of creating a Block for IO.

    Usage:
    >>> populate_RecordingChannel(a_block)
    """
    recordingchannels = {}
    for seg in bl.segments:
        for anasig in seg.analogsignals:
            if getattr(anasig, 'channel_index', None) is not None:
                ind = int(anasig.channel_index)
                if ind not in recordingchannels:
                    recordingchannels[ind] = RecordingChannel(index=ind)
                    if 'channel_name' in anasig.annotations:
                        channel_name = anasig.annotations['channel_name']
                        recordingchannels[ind].name = channel_name
                        if remove_from_annotation:
                            anasig.annotations.pop('channel_name')
                recordingchannels[ind].analogsignals.append(anasig)
                anasig.recordingchannel = recordingchannels[ind]
                if remove_from_annotation:
                    anasig.channel_index = None

    indexes = np.sort(list(recordingchannels.keys())).astype('i')
    names = np.array([recordingchannels[idx].name for idx in indexes],
                     dtype='S')
    rcg = RecordingChannelGroup(name='all channels',
                                channel_indexes=indexes,
                                channel_names=names)
    bl.recordingchannelgroups.append(rcg)
    for ind in indexes:
        # many to many relationship
        rcg.recordingchannels.append(recordingchannels[ind])
        recordingchannels[ind].recordingchannelgroups.append(rcg)


def iteritems(D):
    try:
        return D.iteritems()  # Python 2
    except AttributeError:
        return D.items()  # Python 3


class LazyList(collections.MutableSequence):
    """ An enhanced list that can load its members on demand. Behaves exactly
    like a regular list for members that are Neo objects. Each item should
    contain the information that ``load_lazy_cascade`` needs to load the
    respective object.
    """
    _container_objects = set(
        [Block, Segment, RecordingChannelGroup, RecordingChannel, Unit])
    _neo_objects = _container_objects.union(
        [AnalogSignal, AnalogSignalArray, Epoch, EpochArray, Event, EventArray,
         IrregularlySampledSignal, Spike, SpikeTrain])

    def __init__(self, io, lazy, items=None):
        """
        :param io: IO instance that can load items.
        :param lazy: Lazy parameter with which the container object
            using the list was loaded.
        :param items: Optional, initial list of items.
        """
        if items is None:
            self._data = []
        else:
            self._data = items
        self._lazy = lazy
        self._io = io

    def __getitem__(self, index):
        item = self._data.__getitem__(index)
        if isinstance(index, slice):
            return LazyList(self._io, item)

        if type(item) in self._neo_objects:
            return item

        loaded = self._io.load_lazy_cascade(item, self._lazy)
        self._data[index] = loaded
        return loaded

    def __delitem__(self, index):
        self._data.__delitem__(index)

    def __len__(self):
        return self._data.__len__()

    def __setitem__(self, index, value):
        self._data.__setitem__(index, value)

    def insert(self, index, value):
        self._data.insert(index, value)

    def append(self, value):
        self._data.append(value)

    def reverse(self):
        self._data.reverse()

    def extend(self, values):
        self._data.extend(values)

    def remove(self, value):
        self._data.remove(value)

    def __str__(self):
        return '<' + self.__class__.__name__ + '>' + self._data.__str__()

    def __repr__(self):
        return '<' + self.__class__.__name__ + '>' + self._data.__repr__()
