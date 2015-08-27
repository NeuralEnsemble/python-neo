# -*- coding: utf-8 -*-
"""
Class for reading data from a .kwik dataset

Depends on: scipy
            h5py >= 2.5.0

Supported: Read

Author: Mikkel E. Lepperød @CINPLA

"""
# TODO: units
# TODO: enable reading of all otputfiles
# TODO: enable writing to file
# TODO: stimulus and tracking data
# TODO: enable read all datasets

# needed for python 3 compatibility
from __future__ import absolute_import
from __future__ import division

import numpy as np
import quantities as pq
import os
#version checking
from distutils import version
# check h5py
try:
    import h5py
except ImportError as err:
    HAVE_H5PY = False
    H5PY_ERR = err
else:
    if version.LooseVersion(h5py.__version__) < '2.5.0':
        HAVE_H5PY = False
        H5PY_ERR = ImportError("your h5py version is too old to " +
                                 "support KwikIO, you need at least 2.5.0 " +
                                 "You have %s" % h5py.__version__)
    else:
        HAVE_H5PY = True
        H5PY_ERR = None
# check scipy
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
    extensions         = [ 'kwik' ]
    mode = 'file'

    def __init__(self, filename, dataset=0) :
        """
        Arguments:
            filename : the filename
            dataset: points to a specific dataset in the .kwik and .raw.kwd file,
                     however this can be an issue to change in e.g. OpenElectrophy or Spykeviewer
        """
        BaseIO.__init__(self)
        self._filename = filename
        self._path, file = os.path.split(filename)
        self._kwik = h5py.File(filename, 'r')
        self._dataset = dataset
        try:
            rawfile = self._kwik['recordings'][str(self._dataset)]['raw'].attrs['hdf5_path'] # klustakwik/phy and newest version of open ephys
            rawfile = rawfile.split('/')[0]
        except:
            rawfile = file.split('.')[0] + '_100.raw.kwd' # first version of open ephys files
        self._kwd = h5py.File(self._path + os.sep + rawfile, 'r')
        self._attrs = {}
        self._attrs['kwik'] = self._kwik['recordings'][str(self._dataset)].attrs
        self._attrs['kwd'] = self._kwd['recordings'][str(self._dataset)].attrs
        self._attrs['shape'] = self._kwd['recordings'][str(self._dataset)]['data'].shape
        try:
            self._attrs['app_data'] = self._kwd['recordings'][str(self._dataset)]['application_data'].attrs # TODO: find bitvolt conversion in phy generated data
        except:
            self._attrs['app_data'] = False

    def read_block(self,
                     lazy=False,
                     cascade=True,
                     channel_index=None
                    ):
        """
        Arguments:
            Channel_index: can be int, iterable or None to select one, many or all channel(s)

        """

        blk = Block()
        if cascade:
            seg = Segment( file_origin=self._filename )
            blk.segments += [ seg ]



            if channel_index:
                if type(channel_index) is int: channel_index = [ channel_index ]
                if type(channel_index) is list: channel_index = np.array( channel_index )
            else:
                channel_index = np.arange(0,self._attrs['shape'][1])

            rcg = RecordingChannelGroup(name='all channels',
                                 channel_indexes=channel_index)
            blk.recordingchannelgroups.append(rcg)

            for idx in channel_index:
                # read nested analosignal
                ana = self.read_analogsignal(channel_index=idx,
                                        lazy=lazy,
                                        cascade=cascade,
                                         )
                chan = RecordingChannel(index=int(idx))
                seg.analogsignals += [ ana ]
                chan.analogsignals += [ ana ]
                rcg.recordingchannels.append(chan)
            seg.duration = (self._attrs['shape'][0]
                          / self._attrs['kwik']['sample_rate']) * pq.s

            # neo.tools.populate_RecordingChannel(blk)
        blk.create_many_to_one_relationship()
        return blk

    def read_analogsignal(self,
                      channel_index=None,
                      lazy=False,
                      cascade=True,
                      ):
        """
        Read raw traces
        Arguments:
            channel_index: must be integer
        """
        try:
            channel_index = int(channel_index)
        except TypeError:
            print('channel_index must be int, not %s' %type(channel_index))

        if self._attrs['app_data']:
            bit_volts = self._attrs['app_data']['channel_bit_volts']
            sig_unit = 'uV'
        else:
            bit_volts = np.ones((self._attrs['shape'][1])) # TODO: find conversion in phy generated files
            sig_unit =  'bit'
        if lazy:
            anasig = AnalogSignal([],
                                  units=sig_unit,
                                  sampling_rate=self._attrs['kwik']['sample_rate']*pq.Hz,
                                  t_start=self._attrs['kwik']['start_time']*pq.s,
                                  channel_index=channel_index,
                                  )
            # we add the attribute lazy_shape with the size if loaded
            anasig.lazy_shape = self._attrs['shape'][0]
        else:
            data = self._kwd['recordings'][str(self._dataset)]['data'].value[:,channel_index]
            data = data * bit_volts[channel_index]
            anasig = AnalogSignal(data,
                                       units=sig_unit,
                                       sampling_rate=self._attrs['kwik']['sample_rate']*pq.Hz,
                                       t_start=self._attrs['kwik']['start_time']*pq.s,
                                       channel_index=channel_index,
                                       )
            data = [] # delete from memory
        # for attributes out of neo you can annotate
        anasig.annotate(info='raw traces')
        return anasig
