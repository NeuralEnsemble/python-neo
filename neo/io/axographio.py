"""
README
===============================================================================
This is an adapter to represent axographio objects as neo objects.

axographio is a file i/o Python module that can read in axograph ".axgx" files.
It is available under a BSD-3-Clause license and can be installed from pip.
The following file types are supported:

 - AXGX/AXGD (Axograph X file format)

Based on stimfitio.pyfrom neo.io

11 JUL 2018, W. Hart, Swinburne University, Australia
"""

# needed for python 3 compatibility
from __future__ import absolute_import

from datetime import datetime
import os
import sys

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal

try:
    import axographio
except ImportError as err:
    HAS_AXOGRAPHIO = False
    AXOGRAPHIO_ERR = err
else:
    HAS_AXOGRAPHIO = True
    AXOGRAPHIO_ERR = None


class AxographIO(BaseIO):
    """
    Class for converting an Axographio object to a Neo object.
    Provides a standardized representation of the data as defined by the neo
    project; this is useful to explore the data with an increasing number of
    electrophysiology software tools that rely on the Neo standard.

    axographio is a file i/o Python module that can read in axograph ".axgx" files.
    It is available under a BSD-3-Clause license and can be installed from pip.
    The following file types are supported:

    - AXGX/AXGD (Axograph X file format)

    Example usage:
        >>> import neo
        >>> neo_obj = neo.io.AxographIO("file.axgx")
        or
        >>> import axographio
        >>> axo_obj = axographio.read("file.axgx")
        >>> neo_obj = neo.io.AxographIO(axo_obj)
    """

    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, AnalogSignal]
    readable_objects = [Block]
    writeable_objects = []

    has_header = False
    is_streameable = False

    read_params = {Block: []}
    write_params = None

    name = 'AXOGRAPH'
    extensions = ['axgx', 'axgd']

    mode = 'file'

    def __init__(self, filename=None):
        """
        Arguments:
            filename : Either a filename or an axographio object
        """
        if not HAS_AXOGRAPHIO:
            raise AXOGRAPHIO_ERR

        BaseIO.__init__(self)

        if hasattr(filename, 'lower'):
            self.filename = filename
            self.axo_obj = None
        else:
            self.axo_obj = filename
            self.filename = None

    def read_block(self, **kargs):
        if self.filename is not None:
            self.axo_obj = axographio.read(self.filename)

        # Build up the block
        blk = Block()

        blk.rec_datetime = None
        if self.filename is not None:
            # modified time is not ideal but less prone to
            # cross-platform issues than created time (ctime)
            blk.file_datetime = datetime.fromtimestamp(os.path.getmtime(self.filename))

            # store the filename if it is available
            blk.file_origin = self.filename

        # determine the channel names and counts
        _, channel_ordering = np.unique(self.axo_obj.names[1:], return_index=True)
        channel_names = np.array(self.axo_obj.names[1:])[np.sort(channel_ordering)]
        channel_count = len(channel_names)

        # determine the time signal and sample period
        sample_period = self.axo_obj.data[0].step * pq.s
        start_time = self.axo_obj.data[0].start * pq.s

        # Attempt to read units from the channel names
        channel_unit_names = [x.split()[-1].strip('()') for x in channel_names]
        channel_units = []

        for unit in channel_unit_names:
            try:
                channel_units.append(pq.Quantity(1, unit))
            except LookupError:
                channel_units.append(None)

        # Strip units from channel names
        channel_names = [' '.join(x.split()[:-1]) for x in channel_names]

        # build up segments by grouping axograph columns
        for seg_idx in range(1, len(self.axo_obj.data), channel_count):
            seg = Segment(index=seg_idx)

            # add in the channels
            for chan_idx in range(0, channel_count):
                signal = pq.Quantity(
                    self.axo_obj.data[seg_idx + chan_idx], channel_units[chan_idx])
                analog = AnalogSignal(signal,
                                      sampling_period=sample_period, t_start=start_time,
                                      name=channel_names[chan_idx], channel_index=chan_idx)
                seg.analogsignals.append(analog)

            blk.segments.append(seg)

        blk.create_many_to_one_relationship()

        return blk
