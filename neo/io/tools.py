# -*- coding: utf-8 -*-
"""
Tools for IO coder:
  * For creating parent (many_to_one_relationship)
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
from neo.description import one_to_many_relationship


#def finalize_block(block):
#    populate_RecordingChannel(block)
#    create_many_to_one_relationship(block)

    # Special case this tricky many-to-many relationship
    # we still need links from recordingchannel to analogsignal
#    for rcg in block.recordingchannelgroups:
#        for rc in rcg.recordingchannels:
#            create_many_to_one_relationship(rc)


def create_many_to_one_relationship(ob, force=False):
    """
    Create many_to_one relationship when one_to_many relationships exist.
    Ex: For each Segment in block.segments it sets segment.block to the
    parent Block. It is a utility at the end of creating a Block for IO.

    Note:
        This is recursive.
        It works on Block but also work on others neo objects.

    Usage:
    >>> create_many_to_one_relationship(a_block)
    >>> create_many_to_one_relationship(a_block, force=True)

    You want to run populate_RecordingChannel first, because this will create
    new objects that this method will link up.

    If force is True overwrite any existing relationships

    """
    # Determine what class was passed, and whether it has children
    classname = ob.__class__.__name__
    if classname not in one_to_many_relationship:
        # No children
        return

    # Iterate through children and build backward links
    for childname in one_to_many_relationship[classname]:
        # Doesn't have links to children
        if not hasattr(ob, childname.lower()+'s'):
            continue

        # get a list of children of type childname and iterate through
        sub = getattr(ob, childname.lower()+'s')
        for child in sub:
            # set a link to parent `ob`, of class `classname`
            if getattr(child, classname.lower()) is None or force:
                setattr(child, classname.lower(), ob)
            # recursively:
            create_many_to_one_relationship(child, force=force)


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
