"""
README
===============================================================================
This is an adapter to represent stfio objects as neo objects.

stfio is a standalone file i/o Python module that ships with the Stimfit
program (http://www.stimfit.org). It's a Python wrapper around Stimfit's file
i/o library (libstfio) that natively provides support for the following file
types:

 - ABF (Axon binary file format; pClamp 6--9)
 - ABF2 (Axon binary file format 2; pClamp 10+)
 - ATF (Axon text file format)
 - AXGX/AXGD (Axograph X file format)
 - CFS (Cambridge electronic devices filing system)
 - HEKA (HEKA binary file format)
 - HDF5 (Hierarchical data format 5; only hdf5 files written by Stimfit or
   stfio are supported)

In addition, libstfio can use the biosig file i/o library as an additional file
handling backend (http://biosig.sourceforge.net/), extending support to more
than 30 additional file formats (http://pub.ist.ac.at/~schloegl/biosig/TESTED).

Based on exampleio.py and axonio.py from neo.io

08 Feb 2014, C. Schmidt-Hieber, University College London
"""

# needed for python 3 compatibility
from __future__ import absolute_import

import sys

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal

try:
    import stfio
except ImportError as err:
    HAS_STFIO = False
    STFIO_ERR = err
else:
    HAS_STFIO = True
    STFIO_ERR = None


class StimfitIO(BaseIO):
    """
    Class for converting a stfio Recording to a neo object.
    Provides a standardized representation of the data as defined by the neo
    project; this is useful to explore the data with an increasing number of
    electrophysiology software tools that rely on the neo standard.

    Example usage:
        >>> import neo
        >>> neo_obj = neo.io.StimfitIO("file.abf")
        or
        >>> import stfio
        >>> stfio_obj = stfio.read("file.abf")
        >>> neo_obj = neo.io.StimfitIO(stfio_obj)
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

    name = 'Stimfit'
    extensions = ['abf', 'dat', 'axgx', 'axgd', 'cfs']

    mode = 'file'

    def __init__(self, filename=None):
        """
        Arguments:
            filename : Either a filename or a stfio Recording object
        """
        if not HAS_STFIO:
            raise STFIO_ERR

        BaseIO.__init__(self)

        if hasattr(filename, 'lower'):
            self.filename = filename
            self.stfio_rec = None
        else:
            self.stfio_rec = filename
            self.filename = None

    def read_block(self, lazy=False, cascade=True):

        if self.filename is not None:
            self.stfio_rec = stfio.read(self.filename)

        bl = Block()
        bl.description = self.stfio_rec.file_description
        bl.annotate(comment=self.stfio_rec.comment)
        try:
            bl.rec_datetime = self.stfio_rec.datetime
        except:
            bl.rec_datetime = None

        if not cascade:
            return bl

        dt = np.round(self.stfio_rec.dt * 1e-3, 9) * pq.s  # ms to s
        sampling_rate = 1.0/dt
        t_start = 0 * pq.s

        # iterate over sections first:
        for j, recseg in enumerate(self.stfio_rec[0]):
            seg = Segment(index=j)
            length = len(recseg)

            # iterate over channels:
            for i, recsig in enumerate(self.stfio_rec):
                name = recsig.name
                unit = recsig.yunits
                try:
                    pq.Quantity(1, unit)
                except:
                    unit = ''

                if lazy:
                    signal = pq.Quantity([], unit)
                else:
                    signal = pq.Quantity(recsig[j], unit)
                anaSig = AnalogSignal(signal, sampling_rate=sampling_rate,
                                      t_start=t_start, name=str(name),
                                      channel_index=i)
                if lazy:
                    anaSig.lazy_shape = length
                seg.analogsignals.append(anaSig)

            bl.segments.append(seg)
            t_start = t_start + length * dt

        bl.create_many_to_one_relationship()

        return bl
