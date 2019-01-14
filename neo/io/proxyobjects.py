# -*- coding: utf-8 -*-
"""
Here a list of proxy object that can be used when lazy=True at neo.io level.

This idea is to be able to postpone that real in memory loading
for objects that contains big data (AnalogSIgnal, SpikeTrain, Event, Epoch).

The implementation rely on neo.rawio, so it will available only for neo.io that
ineherits neo.rawio.

"""

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo


from neo.core import (AnalogSignal,
                      Epoch, Event, SpikeTrain)
from neo.core.dataobject import ArrayDict


class BaseProxy(BaseNeo):
    def __init__(self, array_annotations=None, **annotations):
        # this for py27 str vs py3 str in neo attributes ompatibility
        annotations = check_annotations(annotations)
        if 'file_origin' not in annotations:
            # the str is to make compatible with neo_py27 where attribute
            # used to be str so raw bytes
            annotations['file_origin'] = str(self._rawio.source_name())

        # this mock the array annotaions to avoid inherits DataObject
        self.array_annotations = ArrayDict(self.shape[-1])
        if array_annotations is not None:
            self.array_annotations.update(array_annotations)

        BaseNeo.__init__(self, **annotations)


class AnalogSignalProxy(BaseProxy):
    '''
    This object mimic AnalogSignal except that it does not
    have the signals array itself. All attributes and annotations are here.

    The goal is to postpone the loading of data into memory
    when reading a file with the new lazy load system based
    on neo.rawio.

    This object must not be constructed directly but is given
    neo.io when lazy=True instead of a true AnalogSignal.

    The AnalogSignalProxy is able to load:
      * only a slice of time
      * only a subset of channels
      * have an internal raw magnitude identic to the file (int16) with
        a pq.CompoundUnit().

    Usage:
    >>> proxy_anasig = AnalogSignalProxy(rawio=self.reader,
                                                                global_channel_indexes=None,
                                                                block_index=0,
                                                                seg_index=0)
    >>> anasig = proxy_anasig.load()
    >>> slice_of_anasig = proxy_anasig.load(time_slice=(1.*pq.s, 2.*pq.s))
    >>> some_channel_of_anasig = proxy_anasig.load(channel_indexes=[0,5,10])

    '''
    _single_parent_objects = ('Segment', 'ChannelIndex')
    _necessary_attrs = (('sampling_rate', pq.Quantity, 0),
                                    ('t_start', pq.Quantity, 0))
    _recommended_attrs = BaseNeo._recommended_attrs

    def __init__(self, rawio=None, global_channel_indexes=None, block_index=0, seg_index=0):
        self._rawio = rawio
        self._block_index = block_index
        self._seg_index = seg_index
        if global_channel_indexes is None:
            global_channel_indexes = slice(None)
        total_nb_chan = self._rawio.header['signal_channels'].size
        self._global_channel_indexes = np.arange(total_nb_chan)[global_channel_indexes]
        self._nb_chan = self._global_channel_indexes.size

        sig_chans = self._rawio.header['signal_channels'][self._global_channel_indexes]

        assert np.unique(sig_chans['units']).size == 1, 'Channel do not have same units'
        assert np.unique(sig_chans['dtype']).size == 1, 'Channel do not have same dtype'
        assert np.unique(sig_chans['sampling_rate']).size == 1, \
                    'Channel do not have same sampling_rate'

        self.units = ensure_signal_units(sig_chans['units'][0])
        self.dtype = sig_chans['dtype'][0]
        self.sampling_rate = sig_chans['sampling_rate'][0] * pq.Hz
        self.sampling_period = 1. / self.sampling_rate
        sigs_size = self._rawio.get_signal_size(block_index=block_index, seg_index=seg_index,
                                        channel_indexes=self._global_channel_indexes)
        self.shape = (sigs_size, self._nb_chan)
        self.t_start = self._rawio.get_signal_t_start(block_index, seg_index,
                                    self._global_channel_indexes) * pq.s

        # magnitude_mode='raw' is supported only if all offset=0
        # and all gain are the same
        support_raw_magnitude = np.all(sig_chans['gain'] == sig_chans['gain'][0]) and \
                                                    np.all(sig_chans['offset'] == 0.)

        if support_raw_magnitude:
            str_units = ensure_signal_units(sig_chans['units'][0]).units.dimensionality.string
            self._raw_units = pq.CompoundUnit('{}*{}'.format(sig_chans['gain'][0], str_units))
        else:
            self._raw_units = None

        # both necessary attr and annotations
        annotations = {}
        annotations['name'] = self._make_name(None)
        if len(sig_chans) == 1:
            # when only one channel raw_annotations are set to standart annotations
            d = self._rawio.raw_annotations['blocks'][block_index]['segments'][seg_index][
                'signals'][self._global_channel_indexes[0]]
            annotations.update(d)

        array_annotations = {
            'channel_names': np.array(sig_chans['name'], copy=True),
            'channel_ids': np.array(sig_chans['id'], copy=True),
        }

        BaseProxy.__init__(self, array_annotations=array_annotations, **annotations)

    def _make_name(self, channel_indexes):
        sig_chans = self._rawio.header['signal_channels'][self._global_channel_indexes]
        if channel_indexes is not None:
            sig_chans = sig_chans[channel_indexes]
        if len(sig_chans) == 1:
            name = sig_chans['name'][0]
        else:
            name = 'Channel bundle ({}) '.format(','.join(sig_chans['name']))
        return name

    @property
    def duration(self):
        '''Signal duration'''
        return self.shape[0] / self.sampling_rate

    @property
    def t_stop(self):
        '''Time when signal ends'''
        return self.t_start + self.duration

    def load(self, time_slice=None, strict_slicing=True,
                    channel_indexes=None, magnitude_mode='rescaled'):
        '''
        *Args*:
            :time_slice: None or tuple of the time slice expressed with quantities.
                            None is the entire signal.
            :channel_indexes: None or list. Channels to load. None is all channels
                    Be carefull that channel_indexes represent the local channel index inside
                    the AnalogSignal and not the global_channel_indexes like in rawio.
            :magnitude_mode: 'rescaled' or 'raw'.
                For instance if the internal dtype is int16:
                    * **rescaled** give [1.,2.,3.]*pq.uV and the dtype is float32
                    * **raw** give [10, 20, 30]*pq.CompoundUnit('0.1*uV')
                The CompoundUnit with magnitude_mode='raw' is usefull to
                postpone the scaling when needed and having an internal dtype=int16
                but it less intuitive when you don't know so well quantities.
            :strict_slicing: True by default.
                Control if an error is raise or not when one of  time_slice member
                (t_start or t_stop) is outside the real time range of the segment.
        '''

        if channel_indexes is None:
            channel_indexes = slice(None)

        sr = self.sampling_rate

        if time_slice is None:
            i_start, i_stop = None, None
            sig_t_start = self.t_start
        else:
            t_start, t_stop = time_slice
            if t_start is None:
                i_start = None
                sig_t_start = self.t_start
            else:
                t_start = ensure_second(t_start)
                if strict_slicing:
                    assert self.t_start <= t_start <= self.t_stop, 't_start is outside'
                else:
                    t_start = max(t_start, self.t_start)
                # the i_start is ncessary ceil
                i_start = int(np.ceil((t_start - self.t_start).magnitude * sr.magnitude))
                # this needed to get the real t_start of the first sample
                # because do not necessary match what is demanded
                sig_t_start = self.t_start + i_start / sr

            if t_stop is None:
                i_stop = None
            else:
                t_stop = ensure_second(t_stop)
                if strict_slicing:
                    assert self.t_start <= t_stop <= self.t_stop, 't_stop is outside'
                else:
                    t_stop = min(t_stop, self.t_stop)
                i_stop = int((t_stop - self.t_start).magnitude * sr.magnitude)

        raw_signal = self._rawio.get_analogsignal_chunk(block_index=self._block_index,
                    seg_index=self._seg_index, i_start=i_start, i_stop=i_stop,
                    channel_indexes=self._global_channel_indexes[channel_indexes])

        # if slice in channel : change name and array_annotations
        if raw_signal.shape[1] != self._nb_chan:
            name = self._make_name(channel_indexes)
            array_annotations = {k: v[channel_indexes] for k, v in self.array_annotations.items()}
        else:
            name = self.name
            array_annotations = self.array_annotations

        if magnitude_mode == 'raw':
            assert self._raw_units is not None,\
                    'raw magnitude is not support gain are not the same for all channel or offset is not 0'
            sig = raw_signal
            units = self._raw_units
        elif magnitude_mode == 'rescaled':
            # dtype is float32 when internally it is float32 or int16
            if self.dtype == 'float64':
                dtype = 'float64'
            else:
                dtype = 'float32'
            sig = self._rawio.rescale_signal_raw_to_float(raw_signal, dtype=dtype,
                                    channel_indexes=self._global_channel_indexes[channel_indexes])
            units = self.units

        anasig = AnalogSignal(sig, units=units, copy=False, t_start=sig_t_start,
                    sampling_rate=self.sampling_rate, name=name,
                    file_origin=self.file_origin, description=self.description,
                    array_annotations=array_annotations, **self.annotations)

        return anasig


class SpikeTrainProxy(BaseProxy):
    '''
    This object mimic SpikeTrain except that it does not
    have the spike time nor waveforms.
    All attributes and annotations are here.

    The goal is to postpone the loading of data into memory
    when reading a file with the new lazy load system based
    on neo.rawio.

    This object must not be constructed directly but is given
    neo.io when lazy=True instead of a true SpikeTrain.

    The SpikeTrainProxy is able to load:
      * only a slice of time
      * load wveforms or not.
      * have an internal raw magnitude identic to the file (generally the ticks
        of clock in int64) or the rescale to seconds.

    Usage:
    >>> proxy_sptr = SpikeTrainProxy(rawio=self.reader, unit_channel=0,
                        block_index=0, seg_index=0,)
    >>> sptr = proxy_sptr.load()
    >>> slice_of_sptr = proxy_sptr.load(time_slice=(1.*pq.s, 2.*pq.s))

    '''

    _single_parent_objects = ('Segment', 'Unit')
    _quantity_attr = 'times'
    _necessary_attrs = (('t_start', pq.Quantity, 0),
                                    ('t_stop', pq.Quantity, 0))
    _recommended_attrs = ()

    def __init__(self, rawio=None, unit_index=None, block_index=0, seg_index=0):

        self._rawio = rawio
        self._block_index = block_index
        self._seg_index = seg_index
        self._unit_index = unit_index

        nb_spike = self._rawio.spike_count(block_index=block_index, seg_index=seg_index,
                                        unit_index=unit_index)
        self.shape = (nb_spike, )

        self.t_start = self._rawio.segment_t_start(block_index, seg_index) * pq.s
        self.t_stop = self._rawio.segment_t_stop(block_index, seg_index) * pq.s

        # both necessary attr and annotations
        annotations = {}
        for k in ('name', 'id'):
            annotations[k] = self._rawio.header['unit_channels'][unit_index][k]
        ann = self._rawio.raw_annotations['blocks'][block_index]['segments'][seg_index]['units'][unit_index]
        annotations.update(ann)

        h = self._rawio.header['unit_channels'][unit_index]
        wf_sampling_rate = h['wf_sampling_rate']
        if not np.isnan(wf_sampling_rate) and wf_sampling_rate > 0:
            self.sampling_rate = wf_sampling_rate * pq.Hz
            self.left_sweep = (h['wf_left_sweep'] / self.sampling_rate).rescale('s')
            self._wf_units = ensure_signal_units(h['wf_units'])
        else:
            self.sampling_rate = None
            self.left_sweep = None

        BaseProxy.__init__(self, **annotations)

    def load(self, time_slice=None, strict_slicing=True,
                    magnitude_mode='rescaled', load_waveforms=False):
        '''
        *Args*:
            :time_slice: None or tuple of the time slice expressed with quantities.
                            None is the entire signal.
            :strict_slicing: True by default.
                 Control if an error is raise or not when one of  time_slice
                 member (t_start or t_stop) is outside the real time range of the segment.
            :magnitude_mode: 'rescaled' or 'raw'.
            :load_waveforms: bool load waveforms or not.
        '''

        t_start, t_stop = consolidate_time_slice(time_slice, self.t_start,
                                                                    self.t_stop, strict_slicing)
        _t_start, _t_stop = prepare_time_slice(time_slice)

        spike_timestamps = self._rawio.get_spike_timestamps(block_index=self._block_index,
                        seg_index=self._seg_index, unit_index=self._unit_index, t_start=_t_start,
                        t_stop=_t_stop)

        if magnitude_mode == 'raw':
            # we must modify a bit the neo.rawio interface to also read the spike_timestamps
            # underlying clock wich is not always same as sigs
            raise(NotImplementedError)
        elif magnitude_mode == 'rescaled':
            dtype = 'float64'
            spike_times = self._rawio.rescale_spike_timestamp(spike_timestamps, dtype=dtype)
            units = 's'

        if load_waveforms:
            assert self.sampling_rate is not None, 'Do not have waveforms'

            raw_wfs = self._rawio.get_spike_raw_waveforms(block_index=self._block_index,
                seg_index=self._seg_index, unit_index=self._unit_index,
                            t_start=_t_start, t_stop=_t_stop)
            if magnitude_mode == 'rescaled':
                float_wfs = self._rawio.rescale_waveforms_to_float(raw_wfs,
                                dtype='float32', unit_index=self._unit_index)
                waveforms = pq.Quantity(float_wfs, units=self._wf_units,
                            dtype='float32', copy=False)
            elif magnitude_mode == 'raw':
                # could code also CompundUnit here but it is over killed
                # so we used dimentionless
                waveforms = pq.Quantity(raw_wfs, units='',
                            dtype=raw_wfs.dtype, copy=False)
        else:
            waveforms = None

        sptr = SpikeTrain(spike_times, t_stop, units=units, dtype=dtype,
                t_start=t_start, copy=False, sampling_rate=self.sampling_rate,
                waveforms=waveforms, left_sweep=self.left_sweep, name=self.name,
                file_origin=self.file_origin, description=self.description, **self.annotations)

        return sptr


class _EventOrEpoch(BaseProxy):
    _single_parent_objects = ('Segment',)
    _quantity_attr = 'times'

    def __init__(self, rawio=None, event_channel_index=None, block_index=0, seg_index=0):

        self._rawio = rawio
        self._block_index = block_index
        self._seg_index = seg_index
        self._event_channel_index = event_channel_index

        nb_event = self._rawio.event_count(block_index=block_index, seg_index=seg_index,
                                        event_channel_index=event_channel_index)
        self.shape = (nb_event, )

        self.t_start = self._rawio.segment_t_start(block_index, seg_index) * pq.s
        self.t_stop = self._rawio.segment_t_stop(block_index, seg_index) * pq.s

        # both necessary attr and annotations
        annotations = {}
        for k in ('name', 'id'):
            annotations[k] = self._rawio.header['event_channels'][event_channel_index][k]
        ann = self._rawio.raw_annotations['blocks'][block_index]['segments'][seg_index]['events'][event_channel_index]
        annotations.update(ann)

        BaseProxy.__init__(self, **annotations)

    def load(self, time_slice=None, strict_slicing=True):
        '''
        *Args*:
            :time_slice: None or tuple of the time slice expressed with quantities.
                            None is the entire signal.
            :strict_slicing: True by default.
                 Control if an error is raise or not when one of  time_slice member (t_start or t_stop)
                 is outside the real time range of the segment.
        '''

        t_start, t_stop = consolidate_time_slice(time_slice, self.t_start,
                                                                    self.t_stop, strict_slicing)
        _t_start, _t_stop = prepare_time_slice(time_slice)

        timestamp, durations, labels = self._rawio.get_event_timestamps(block_index=self._block_index,
                        seg_index=self._seg_index, event_channel_index=self._event_channel_index,
                        t_start=_t_start, t_stop=_t_stop)

        dtype = 'float64'
        times = self._rawio.rescale_event_timestamp(timestamp, dtype=dtype)
        units = 's'

        if durations is not None:
            durations = self._rawio.rescale_epoch_duration(durations, dtype=dtype) * pq.s

        # this should be remove when labesl will be unicode
        labels = labels.astype('S')

        h = self._rawio.header['event_channels'][self._event_channel_index]
        if h['type'] == b'event':
            ret = Event(times=times, labels=labels, units='s',
                name=self.name, file_origin=self.file_origin,
                description=self.description, **self.annotations)
        elif h['type'] == b'epoch':
            ret = Epoch(times=times, durations=durations, labels=labels,
                units='s',
                name=self.name, file_origin=self.file_origin,
                description=self.description, **self.annotations)

        return ret


class EventProxy(_EventOrEpoch):
    '''
    This object mimic Event except that it does not
    have the times nor labels.
    All other attributes and annotations are here.

    The goal is to postpone the loading of data into memory
    when reading a file with the new lazy load system based
    on neo.rawio.

    This object must not be constructed directly but is given
    neo.io when lazy=True instead of a true Event.

    The EventProxy is able to load:
      * only a slice of time

    Usage:
    >>> proxy_event = EventProxy(rawio=self.reader, event_channel_index=0,
                        block_index=0, seg_index=0,)
    >>> event = proxy_event.load()
    >>> slice_of_event = proxy_event.load(time_slice=(1.*pq.s, 2.*pq.s))

    '''
    _necessary_attrs = (('times', pq.Quantity, 1),
                        ('labels', np.ndarray, 1, np.dtype('S')))


class EpochProxy(_EventOrEpoch):
    '''
    This object mimic Epoch except that it does not
    have the times nor labels nor durations.
    All other attributes and annotations are here.

    The goal is to postpone the loading of data into memory
    when reading a file with the new lazy load system based
    on neo.rawio.

    This object must not be constructed directly but is given
    neo.io when lazy=True instead of a true Epoch.

    The EpochProxy is able to load:
      * only a slice of time

    Usage:
    >>> proxy_epoch = EpochProxy(rawio=self.reader, event_channel_index=0,
                        block_index=0, seg_index=0,)
    >>> epoch = proxy_epoch.load()
    >>> slice_of_epoch = proxy_epoch.load(time_slice=(1.*pq.s, 2.*pq.s))

    '''
    _necessary_attrs = (('times', pq.Quantity, 1),
                        ('durations', pq.Quantity, 1),
                        ('labels', np.ndarray, 1, np.dtype('S')))


proxyobjectlist = [AnalogSignalProxy, SpikeTrainProxy, EventProxy,
                            EpochProxy]


unit_convert = {'Volts': 'V', 'volts': 'V', 'Volt': 'V',
                'volt': 'V', ' Volt': 'V', 'microV': 'V'}


def ensure_signal_units(units):
    # test units
    units = units.replace(' ', '')
    if units in unit_convert:
        units = unit_convert[units]
    try:
        units = pq.Quantity(1, units)
    except:
        logging.warning('Units "{}" can not be converted to a quantity. Using dimensionless '
                        'instead'.format(units))
        units = ''
    return units


def check_annotations(annotations):
    # force type to str for some keys
    # imposed for tests
    for k in ('name', 'description', 'file_origin'):
        if k in annotations:
            annotations[k] = str(annotations[k])

    if 'coordinates' in annotations:
        # some rawio expose some coordinates in annotations but is not standardized
        # (x, y, z) or polar, at the moment it is more resonable to remove them
        annotations.pop('coordinates')

    return annotations


def ensure_second(v):
    if isinstance(v, float):
        return v * pq.s
    elif isinstance(v, pq.Quantity):
        return v.rescale('s')
    elif isinstance(v, int):
        return float(v) * pq.s


def prepare_time_slice(time_slice):
    """
    This give clean time slice but keep None
    for calling rawio slice
    """
    if time_slice is None:
        t_start, t_stop = None, None
    else:
        t_start, t_stop = time_slice

    if t_start is not None:
        t_start = ensure_second(t_start).rescale('s').magnitude

    if t_stop is not None:
        t_stop = ensure_second(t_stop).rescale('s').magnitude

    return (t_start, t_stop)


def consolidate_time_slice(time_slice, seg_t_start, seg_t_stop, strict_slicing):
    """
    This give clean time slice in quantity for t_start/t_stop of object
    None is replace by seg limit.
    """
    if time_slice is None:
        t_start, t_stop = None, None
    else:
        t_start, t_stop = time_slice

    if t_start is None:
        t_start = seg_t_start
    else:
        if strict_slicing:
            assert seg_t_start <= t_start <= seg_t_stop, 't_start is outside'
        else:
            t_start = max(t_start, seg_t_start)
    t_start = ensure_second(t_start)

    if t_stop is None:
        t_stop = seg_t_stop
    else:
        if strict_slicing:
            assert seg_t_start <= t_stop <= seg_t_stop, 't_stop is outside'
        else:
            t_stop = min(t_stop, seg_t_stop)
    t_stop = ensure_second(t_stop)

    return (t_start, t_stop)
