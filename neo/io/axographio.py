"""
AxographIO
==========

IO class for reading AxoGraph files (.axgd, .axgx)
"""

from neo.io.basefromrawio import BaseFromRaw
from neo.rawio.axographrawio import AxographRawIO


class AxographIO(AxographRawIO, BaseFromRaw):
    """
    IO class for reading AxoGraph files (.axgd, .axgx)

    Args:
        filename (string):
            File name of the AxoGraph file to read.
        force_single_segment (bool):
            Episodic files are normally read as multi-Segment Neo objects. This
            parameter can force AxographIO to put all signals into a single
            Segment. Default: False.

    Example:
        >>> import neo
        >>> r = neo.io.AxographIO(filename=filename)
        >>> blk = r.read_block(signal_group_mode='split-all')
        >>> display(blk)

        >>> # get signals
        >>> seg_index = 0  # episode number
        >>> sigs = [sig for sig in blk.segments[seg_index].analogsignals
        ...         if sig.name in channel_names]
        >>> display(sigs)

        >>> # get event markers (same for all segments/episodes)
        >>> ev = blk.segments[0].events[0]
        >>> print([ev for ev in zip(ev.times, ev.labels)])

        >>> # get interval bars (same for all segments/episodes)
        >>> ep = blk.segments[0].epochs[0]
        >>> print([ep for ep in zip(ep.times, ep.durations, ep.labels)])

        >>> # get notes
        >>> print(blk.annotations['notes'])
    """

    name = 'AxographIO'
    description = 'This IO reads .axgd/.axgx files created with AxoGraph'

    _prefered_signal_group_mode = 'group-by-same-units'

    def __init__(self, filename='', force_single_segment=False):
        AxographRawIO.__init__(self, filename, force_single_segment)
        BaseFromRaw.__init__(self, filename)
