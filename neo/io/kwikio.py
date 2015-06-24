# -*- coding: utf-8 -*-
"""
Class for reading data from a .kwik dataset

For the user, it generates a :class:`Segment` or a :class:`Block` with a
sinusoidal :class:`AnalogSignal`, a :class:`SpikeTrain` and an
:class:`EventArray`.

Depends on: scipy

Supported: Read

Author: Mikkel E. Lepperød @CINPLA

"""
# TODO: units
# TODO: channelindex for LFP

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
    Class for "reading" fake data from an imaginary file.

    For the user, it generates a :class:`Segment` or a :class:`Block` with a
    sinusoidal :class:`AnalogSignal`, a :class:`SpikeTrain` and an
    :class:`EventArray`.

    For a developer, it is just an example showing guidelines for someone who wants
    to develop a new IO module.

    Two rules for developers:
      * Respect the Neo IO API (:ref:`neo_io_API`)
      * Follow :ref:`io_guiline`

    Usage:
        >>> from neo import io
        >>> r = io.ExampleIO(filename='itisafake.nof')
        >>> seg = r.read_segment(lazy=False, cascade=True)
        >>> print(seg.analogsignals)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<AnalogSignal(array([ 0.19151945,  0.62399373,  0.44149764, ...,  0.96678374,
        ...
        >>> print(seg.spiketrains)    # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
         [<SpikeTrain(array([ -0.83799524,   6.24017951,   7.76366686,   4.45573701,
            12.60644415,  10.68328994,   8.07765735,   4.89967804,
        ...
        >>> print(seg.eventarrays)    # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<EventArray: TriggerB@9.6976 s, TriggerA@10.2612 s, TriggerB@2.2777 s, TriggerA@6.8607 s, ...
        >>> anasig = r.read_analogsignal(lazy=True, cascade=False)
        >>> print(anasig._data_description)
        {'shape': (150000,)}
        >>> anasig = r.read_analogsignal(lazy=False, cascade=False)

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

    name               = 'kwik'

    extensions         = [ 'kwd', 'kwx', 'kwik' ]

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
        self.filename = filename
        basename, ext = op.splitext(filename)
        self.kwik = h5py.File(filename, 'r')
        self.kwd = h5py.File(basename + '.raw.kwd', 'r')

    def read_segment(self,
                     lazy=False,
                     cascade=True,
                     dataset=0,
                     lfpchannel=12
                     tracechannel=None
                    ):
        """
        lfpchannel can be int  - how to select all?
        """

        raw_shape = self.kwd['recordings'][str(dataset)]['data'].shape # Do we need this?
        attrs = self.kwd['recordings'][str(dataset)].attrs
        # create an empty segment
        seg = Segment( name='session something' )

        if cascade:
            if LFPchannel is not None
            # read nested analosignal
                ana = self.read_lfpdata(lazy=lazy,
                                        kwd=kwd,
                                        channel_index=lfpchannel,
                                        dataset=dataset,
                                             )
                seg.analogsignals += [ ana ]
            if tracechannel is not None
                # read nested analosignal
                ana = self.read_rawdata(lazy=lazy,
                                        kwd=kwd,
                                        channel_index=tracechannel,
                                        dataset=dataset,
                                             )
                seg.analogsignals += [ ana ]
            # read nested spiketrain
            num_spiketrain_by_channel = 3
            for i in range(raw_shape[1]):
                for _ in range(num_spiketrain_by_channel):
                    sptr = self.read_spiketrain(lazy=lazy,
                                                channel_index=i,
                                                )
                    seg.spiketrains += [ sptr ]

            epo = self.read_stimulus(lazy=lazy)
            seg.epocharrays += [ epo ]

            seg.duration = (raw_shape[0]
                          / attrs['sample_rate']
                          + attrs['start_time'])

        seg.create_many_to_one_relationship()
        return seg

    def read_lfpdata(self,
                          lazy=False,
                          cascade=True
                          channel_index
                          dataset=0,
                          ):
        """
        Downsample and filter raw data into LFP
        """

        attrs = self.kwd['recordings'][str(dataset)].attrs

        if lazy:
            anasig = AnalogSignal([],
                                  units='V',
                                  sampling_rate=attrs['sample_rate']*pq.Hz,
                                  t_start=attrs['start_time']*pq.s,
                                  channel_index=channel_index,
                                  )
            # we add the attribute lazy_shape with the size if loaded
            anasig.lazy_shape = self.kwd['recordings'][str(dataset)]['data'].shape[0]
        else:
            data   = self.kwd['recordings'][str(dataset)]['data']
            for i in arange(0,)
            anasig = AnalogSignalArray(data,
                                  units='V',
                                  sampling_rate=attrs['sample_rate']*pq.Hz,
                                  t_start=attrs['start_time']*pq.s,
                                  channel_index=channel_index,
                                  )

        # for attributes out of neo you can annotate
        anasig.annotate(info = 'low pass filtered')

        return anasig

    def read_rawdata(self,
                          lazy=False,
                          cascade=True
                          channel_index=0,
                          dataset=0,
                          ):
        """

        """

        attrs = self.kwd['recordings'][str(dataset)].attrs

        if lazy:
            anasig = AnalogSignal([],
                                  units='V',
                                  sampling_rate=attrs['sample_rate']*pq.Hz,
                                  t_start=attrs['start_time']*pq.s,
                                  channel_index=channel_index,
                                  )
            # we add the attribute lazy_shape with the size if loaded
            anasig.lazy_shape = self.kwd['recordings'][str(dataset)]['data'].shape[0]
        else:
            data   = np.array(self.kwd['recordings'][str(dataset)]['data'][:,:])
            anasig = AnalogSignalArray(data,
                                  units='V',
                                  sampling_rate=attrs['sample_rate']*pq.Hz,
                                  t_start=attrs['start_time']*pq.s,
                                  channel_index=channel_index,
                                  )

        # for attributes out of neo you can annotate
        anasig.annotate(info = 'rawdata')

        return anasig

    def read_stimulus(self, lazy):
        epo = EpochArray()
        if lazy:
            # in lazy case no data are read
            pass
        else:
            n = 96
            epo.times = np.linspace(1,100,n)* pq.s
            # all duration are the same
            epo.durations = np.ones(n)*500*pq.ms
            # label
            l = [ ]
            for i in range(n):
                if np.mod(i,2)==0: l.append( 'Evoked' )
            else : l.append( 'Spontaneous' )
            epo.labels = np.array( l )

        return epo

    def read_spiketrain(self,
                        lazy = False,
                        cascade = True,
                        segment_duration = 15.,
                        t_start = -1,
                        channel_index = 0,
                        ):
        """
        With this IO SpikeTrain can e acces directly with its channel number
        """
        # There are 2 possibles behaviour for a SpikeTrain
        # holding many Spike instance or directly holding spike times
        # we choose here the first :
        if not HAVE_SCIPY:
            raise SCIPY_ERR

        num_spike_by_spiketrain = 40
        sr = 10000.

        if lazy:
            times = [ ]
        else:
            times = (np.random.rand(num_spike_by_spiketrain)*segment_duration +
                     t_start)

        # create a spiketrain
        spiketr = SpikeTrain(times, t_start = t_start*pq.s, t_stop = (t_start+segment_duration)*pq.s ,
                                            units = pq.s,
                                            name = 'it is a spiketrain from exampleio',
                                            )

        if lazy:
            # we add the attribute lazy_shape with the size if loaded
            spiketr.lazy_shape = (num_spike_by_spiketrain,)

        # ours spiketrains also hold the waveforms:

        # 1 generate a fake spike shape (2d array if trodness >1)
        w1 = -stats.nct.pdf(np.arange(11,60,4), 5,20)[::-1]/3.
        w2 = stats.nct.pdf(np.arange(11,60,2), 5,20)
        w = np.r_[ w1 , w2 ]
        w = -w/max(w)

        if not lazy:
            # in the neo API the waveforms attr is 3 D in case tetrode
            # in our case it is mono electrode so dim 1 is size 1
            waveforms  = np.tile( w[np.newaxis,np.newaxis,:], ( num_spike_by_spiketrain ,1, 1) )
            waveforms *=  np.random.randn(*waveforms.shape)/6+1
            spiketr.waveforms = waveforms*pq.mV
            spiketr.sampling_rate = sr * pq.Hz
            spiketr.left_sweep = 1.5* pq.s

        # for attributes out of neo you can annotate
        spiketr.annotate(channel_index = channel_index)

        return spiketr
