# -*- coding: utf-8 -*-
"""
Class for reading data from a .kwik dataset

Depends on: scipy
            h5py >= 2.5.0 at least > 2.2.1

Supported: Read

Author: Mikkel E. Lepperød @CINPLA

"""
# TODO: units
# TODO: enable reading of several files e.g. downsampled or filtered
# TODO: enable writing to file
# TODO: stimulus and tracing data

# needed for python 3 compatibility
from __future__ import absolute_import

# note neo.core needs only numpy and quantities
import numpy as np
import quantities as pq
import h5py
import os.path as op

# but my specific IO can depend on many other packages
try:
    from scipy import stats
except ImportError as err:
    HAVE_SCIPY = False
    SCIPY_ERR = err
else:
    HAVE_SCIPY = True
    SCIPY_ERR = None

# I need to subclass BaseIO
from neo.io.baseio import BaseIO

# to import from core
from neo.core import (Segment, SpikeTrain, Unit, EpochArray, AnalogSignal,
                      RecordingChannel, RecordingChannelGroup, Block)
import neo.io.tools

class KwikIO(BaseIO):
    """
    Class for "reading" experimental data from a .kwik file.

    Generates a :class:`Segment` with a :class:`AnalogSignal`

    """

    is_readable = True # This class can only read data
    is_writable = False # write is not supported

    # This class is able to directly or indirectly handle the following objects
    # You can notice that this greatly simplifies the full Neo object hierarchy
    supported_objects    = [ Block, Segment, AnalogSignal,
                          RecordingChannel, RecordingChannelGroup]

    # This class can return either a Block or a Segment
    # The first one is the default ( self.read )
    # These lists should go from highest object to lowest object because
    # common_io_test assumes it.
    readable_objects  = [ Block ]

    # This class is not able to write objects
    writeable_objects   = [ ]

    has_header         = False
    is_streameable     = False

    name               = 'Kwik'
    description        = 'This IO reads experimental data from a .kwik dataset'
    extensions         = []#[ 'kwik', 'kwd', 'kwx' ] #TODO: change when OpenElectrophy works

    # mode can be 'file' or 'dir' or 'fake' or 'database'
    # the main case is 'file' but some reader are base on a directory or a database
    # this info is for GUI stuff also
    mode = 'file'

    def __init__(self, filename) :
        """
        Arguments:
            filename : the filename

        """
        BaseIO.__init__(self)
        self._filename = filename
        basename, ext = op.splitext(filename)
        self._basename = basename
        self._kwik = h5py.File(filename, 'r')
        self._kwd = h5py.File(basename + '.raw.kwd', 'r') #TODO read filename from kwik file - depends on openephys issue

    def read_block(self,
                     lazy=False,
                     cascade=True,
                     dataset=0,
                     channel_index=None,
                     sampling_rate=None
                    ):
        """
        Arguments:
            Channel_index: can be int, iterable or None to select one, many or all channel(s)
            dataset: points to a specific dataset in the .kwik and .raw.kwd file,
                     however this will be an issue to change in e.g. OpenElectrophy or Spykeviewer
            sampling_rate: None or float/int if not None raw data will be downsampled
        """

        attrs = {}
        attrs['kwik'] = self._kwik['recordings'][str(dataset)].attrs
        attrs['kwd'] = self._kwd['recordings'][str(dataset)].attrs
        attrs['shape'] = self._kwd['recordings'][str(dataset)]['data'].shape
        try:
            attrs['app_data'] = self._kwd['recordings'][str(dataset)]['application_data'].attrs # TODO: find bitvolt conversion in phy generated data
        except:
            attrs['app_data'] = False

        blk = Block()
        if cascade:
            seg = Segment( file_origin=self._filename ,name='test' )
            blk.segments += [ seg ]
            if channel_index:
                if type(channel_index) is int: channel_index = [channel_index]
            else:
                channel_index = range(0,attrs['shape'][1])

            for idx in channel_index:
                # read nested analosignal
                ana = self._read_traces(attrs=attrs,
                                        channel_index=idx,
                                        lazy=lazy,
                                        cascade=cascade,
                                        dataset=dataset,
                                        sampling_rate=sampling_rate,
                                         )
                seg.analogsignals += [ ana ]

            seg.duration = (attrs['shape'][0]
                          / attrs['kwik']['sample_rate']) * pq.s

            neo.tools.populate_RecordingChannel(blk)
        blk.create_many_to_one_relationship()
        return blk

    def _read_traces(self,
                      attrs,
                      channel_index,
                      lazy=False,
                      cascade=True,
                      dataset=0,
                      sampling_rate=None,
                      ):
        """
        read raw traces with given sampling_rate, if sampling_rate is None
        default from acquisition system is given. channel_index can be int or
        iterable, if None all channels are read
        """

        if sampling_rate:
            sliceskip = int(attrs['kwik']['sample_rate']/sampling_rate)
        else:
            sliceskip = 1
            sampling_rate = attrs['kwik']['sample_rate']

        if attrs['app_data']:
            bit_volts = attrs['app_data']['channel_bit_volts']
            sig_unit = 'uV'
        else:
            bit_volts = np.ones((attrs['shape'][1])) # TODO: find conversion in phy generated files
            sig_unit =  'bit'
        if lazy:
            anasig = AnalogSignal([],
                                  units=sig_unit,
                                  sampling_rate=sampling_rate*pq.Hz,
                                  t_start=attrs['kwik']['start_time']*pq.s,
                                  channel_index=channel_index,
                                  )
            # we add the attribute lazy_shape with the size if loaded
            anasig.lazy_shape = attrs['shape'] #TODO: wrong if downsampled
        else:
            data = self._kwd['recordings'][str(dataset)]['data'].value[:,channel_index]
            data = data[0:-1:sliceskip] * bit_volts[channel_index]
            anasig = AnalogSignal(data,
                                       units=sig_unit,
                                       sampling_rate=sampling_rate*pq.Hz,
                                       t_start=attrs['kwik']['start_time']*pq.s,
                                       channel_index=channel_index,
                                       )
            data = [] # delete from memory
        # for attributes out of neo you can annotate
        anasig.annotate(info='raw traces')
        return anasig
