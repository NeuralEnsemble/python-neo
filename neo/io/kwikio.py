# -*- coding: utf-8 -*-
"""
Class for reading data from a .kwik dataset

Depends on: scipy
            h5py >= 2.5.0 at least > 2.2.1

Supported: Read

Author: Mikkel E. Lepperød @CINPLA

"""
# TODO: units
# TODO: channelindex for traces - count from 0 or 1?

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
from neo.core import (Segment, SpikeTrain, Unit, EpochArray,
                      RecordingChannel, RecordingChannelGroup, AnalogSignal,
                      AnalogSignalArray)



# I need to subclass BaseIO
class KwikIO(BaseIO):
    """
    Class for "reading" experimental data from a .kwik file.

    Generates a :class:`Segment` with a :class:`AnalogSignal`,
    a :class:`SpikeTrain` and a :class:`EpochArray`.

    """

    is_readable = True # This class can only read data
    is_writable = False # write is not supported

    # This class is able to directly or indirectly handle the following objects
    # You can notice that this greatly simplifies the full Neo object hierarchy
    supported_objects  = [ Segment, SpikeTrain, Unit, EpochArray,
                          RecordingChannel, RecordingChannelGroup, AnalogSignal,
                          AnalogSignalArray ]

    # This class can return either a Block or a Segment
    # The first one is the default ( self.read )
    # These lists should go from highest object to lowest object because
    # common_io_test assumes it.
    readable_objects    = [ Segment, SpikeTrain, Unit, EpochArray,
                          RecordingChannel, RecordingChannelGroup, AnalogSignal,
                          AnalogSignalArray ]
    # This class is not able to write objects
    writeable_objects   = [ ]

    has_header         = False
    is_streameable     = False

    name               = 'Kwik'
    description        = 'This IO reads experimental data from a .kwik dataset'
    extensions         = []#[ 'kwd', 'kwx', 'kwik' ]

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
        self._kwd = h5py.File(basename + '.raw.kwd', 'r') #TODO read filename from kwik file

    def read_segment(self,
                     lazy=False,
                     cascade=True,
                     dataset=0,
                     channel_index=None,
                     sampling_rate=None
                    ):
        """
        Channel_index can be int, iterable or None to select one, many or all channel(s)
        """

        attrs = {}
        attrs['kwik'] = self._kwik['recordings'][str(dataset)].attrs
        attrs['kwd'] = self._kwd['recordings'][str(dataset)].attrs
        attrs['shape'] = self._kwd['recordings'][str(dataset)]['data'].shape
        try:
            attrs['app_data'] = self._kwd['recordings'][str(dataset)]['application_data'].attrs
        except:
            attrs['app_data'] = False

        if attrs['kwik']['start_time'] == 'N.': #TODO bad fix
            start_time = 0
        else:
            start_time = attrs['kwik']['start_time']

        # create an empty segment
        seg = Segment( name=self._basename ) #TODO: fetch a meaningfull name of the segment

        if cascade:
            # read nested analosignal
            ana = self._read_traces(attrs=attrs,
                                    start_time=start_time,
                                    lazy=lazy,
                                    cascade=cascade,
                                    dataset=dataset,
                                    channel_index=channel_index,
                                    sampling_rate=sampling_rate,
                                         )
            seg.analogsignals += [ ana ]
            # # read nested spiketrain
            # num_spiketrain_by_channel = 3
            # for i in range(attrs['shape'][1]):
            #     for _ in range(num_spiketrain_by_channel):
            #         sptr = self._read_spiketrain(lazy=lazy,
            #                                     channel_index=i,
            #                                     )
            #         seg.spiketrains += [ sptr ]
            #
            # epo = self._read_stimulus(lazy=lazy)
            # seg.epocharrays += [ epo ]

            seg.duration = (attrs['shape'][0] #TODO: this duration is not necessarily correct after downsample
                          / attrs['kwik']['sample_rate']
                          + start_time) * pq.s

        seg.create_many_to_one_relationship()
        return seg

    def _read_traces(self,
                      attrs,
                      start_time,
                      lazy=False,
                      cascade=True,
                      dataset=0,
                      channel_index=None,
                      sampling_rate=None
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

        if channel_index is not None:
            if type(channel_index) is int: channel_index = [channel_index]
        else:
            channel_index = range(0,attrs['shape'][1])
        if attrs['app_data']:
            bit_volts = attrs['app_data']['channel_bit_volts']
            sig_unit = 'uV'
        else:
            bit_volts = np.ones((attrs['shape'][1]))
            bit_depth = attrs['kwik']['bit_depth']
            sig_unit =  'bit'#str(bit_depth)
        if lazy:
            anasig = AnalogSignal([],
                                  units=sig_unit,
                                  sampling_rate=sampling_rate*pq.Hz,
                                  t_start=start_time*pq.s,
                                  channel_index=np.array(channel_index),
                                  )
            # we add the attribute lazy_shape with the size if loaded
            anasig.lazy_shape = attrs['shape'] #TODO: wrong if downsampled
        else:
            data_array = []
            for idx in channel_index:
                data = self._kwd['recordings'][str(dataset)]['data'].value[:,idx]
                data_array.append(data[0:-1:sliceskip] * bit_volts[idx])
            data = np.array(data_array).T
            data_array = [] #delete from memory
            anasig = AnalogSignalArray(data,
                                       units=sig_unit,
                                       sampling_rate=sampling_rate*pq.Hz,
                                       t_start=start_time*pq.s,
                                       channel_index=np.array(channel_index),
                                       )

        # for attributes out of neo you can annotate
        anasig.annotate(info='raw traces')
        return anasig

    # def _read_stimulus(self, lazy):
    #     epo = EpochArray()
    #     if lazy:
    #         # in lazy case no data are read
    #         pass
    #     else:
    #         n = 96
    #         epo.times = np.linspace(1,100,n)* pq.s
    #         # all duration are the same
    #         epo.durations = np.ones(n)*500*pq.ms
    #         # label
    #         l = [ ]
    #         for i in range(n):
    #             if np.mod(i,2)==0: l.append( 'Evoked' )
    #         else : l.append( 'Spontaneous' )
    #         epo.labels = np.array( l )
    #
    #     return epo
    #
    # def _read_spiketrain(self,
    #                     lazy = False,
    #                     cascade = True,
    #                     segment_duration = 15.,
    #                     t_start = -1,
    #                     channel_index = 0,
    #                     ):
    #     """
    #     With this IO SpikeTrain can be accessed directly with its channel number
    #     """
    #     # There are 2 possibles behaviour for a SpikeTrain
    #     # holding many Spike instance or directly holding spike times
    #     # we choose here the first :
    #     if not HAVE_SCIPY:
    #         raise SCIPY_ERR
    #
    #     num_spike_by_spiketrain = 40
    #     sr = 10000.
    #
    #     if lazy:
    #         times = [ ]
    #     else:
    #         times = (np.random.rand(num_spike_by_spiketrain)*segment_duration +
    #                  t_start)
    #
    #     # create a spiketrain
    #     spiketr = SpikeTrain(times, t_start = t_start*pq.s, t_stop = (t_start+segment_duration)*pq.s ,
    #                                         units = pq.s,
    #                                         name = 'it is a spiketrain from exampleio',
    #                                         )
    #
    #     if lazy:
    #         # we add the attribute lazy_shape with the size if loaded
    #         spiketr.lazy_shape = (num_spike_by_spiketrain,)
    #
    #     # ours spiketrains also hold the waveforms:
    #
    #     # 1 generate a fake spike shape (2d array if trodness >1)
    #     w1 = -stats.nct.pdf(np.arange(11,60,4), 5,20)[::-1]/3.
    #     w2 = stats.nct.pdf(np.arange(11,60,2), 5,20)
    #     w = np.r_[ w1 , w2 ]
    #     w = -w/max(w)
    #
    #     if not lazy:
    #         # in the neo API the waveforms attr is 3 D in case tetrode
    #         # in our case it is mono electrode so dim 1 is size 1
    #         waveforms  = np.tile( w[np.newaxis,np.newaxis,:], ( num_spike_by_spiketrain ,1, 1) )
    #         waveforms *=  np.random.randn(*waveforms.shape)/6+1
    #         spiketr.waveforms = waveforms*pq.mV
    #         spiketr.sampling_rate = sr * pq.Hz
    #         spiketr.left_sweep = 1.5* pq.s
    #
    #     # for attributes out of neo you can annotate
    #     spiketr.annotate(channel_index = channel_index)
    #
    #     return spiketr
