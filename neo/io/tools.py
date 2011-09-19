# encoding: utf-8
"""
Tools for IO coder:
  * For creating parent (many_to_one_relationship)
  * Creating RecordingChannel and making links with AnalogSignals and SPikeTrains
"""
import numpy as np

from ..core import *
from ..description import one_to_many_reslationship, many_to_one_reslationship



def create_many_to_one_relationship(ob):
    """
    When many_to_one relationship when one_to_many exists.
    Ex: when block.segments it create segment.block
    It is a utility at the end of creating a Block for IO.
    
    Note:
        this is recursive. It work on Block but also work on others neo objects.
    
    Usage:
    >>> create_many_to_one_relationship(a_block)
    
    """
    classname =ob.__class__.__name__
    if classname not in  one_to_many_reslationship: 
        return
    
    for childname in one_to_many_reslationship[classname]:
        if not hasattr(ob, childname.lower()+'s'): continue
        sub = getattr(ob, childname.lower()+'s')
        for child in sub:
            if not hasattr(child, classname.lower()):
                setattr(child, classname.lower(), ob)
            # recursively:
            create_many_to_one_relationship(child)


def populate_RecordingChannel(bl, remove_from_annotation = True):
    """
    When a Block is
    Block>Segment>AnalogSIgnal or/and Block>Segment>SpikeTrain
    this function auto create all RecordingChannel
    following these rules:
      * when 'channel_index ' is in _AnalogSIgnal._annotations the corresponding RecordingChannel is created.
      * 'channel_index ' is then removed from _annotations dict if remove_from_annotation
      * only one RecordingChannelGroup is created
    
    It is a utility at the end of creating a Block for IO.
    
    Usage:
    >>> populate_RecordingChannel(a_block)
    """
    recordingchannels = { }
    for seg in bl.segments:
        
        for sub in ['analogsignals', 'spiketrains' ]:
            if not hasattr(seg, sub) : continue
            for child in getattr(seg, sub):
                # child is AnaologSIgnal or SpikeTrain
                if 'channel_index' in child._annotations:
                    ind = child._annotations['channel_index']
                    if  ind not in recordingchannels:
                        recordingchannels[ind] = RecordingChannel(index = ind)
                        if 'channel_name' in child._annotations:
                            recordingchannels[ind].name = child._annotations['channel_name']
                            if remove_from_annotation:
                                child._annotations.pop('channel_name')
                    recordingchannels[ind].analogsignals.append(child)
                    if remove_from_annotation:
                        child._annotations.pop('channel_index')
    indexes = np.sort(recordingchannels.keys())
    rcg = RecordingChannelGroup(name = 'all channels', channel_indexes = indexes)
    bl.recordingchannelgroups.append(rcg)
    for ind in indexes:
        rcg.recordingchannels.append(recordingchannels[ind])


