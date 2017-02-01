# -*- coding: utf-8 -*-
"""
Class for reading data from NeuroExplorer (.nex)

Documentation for dev :
http://www.neuroexplorer.com/code.html

Depend on:

Supported : Read

Author: sgarcia,luc estebanez

"""

import os
import struct

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Segment, AnalogSignal, SpikeTrain, Epoch, Event


class NeuroExplorerIO(BaseIO):
    """
    Class for reading nex files.

    Usage:
        >>> from neo import io
        >>> r = io.NeuroExplorerIO(filename='File_neuroexplorer_1.nex')
        >>> seg = r.read_segment(lazy=False, cascade=True)
        >>> print seg.analogsignals # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<AnalogSignal(array([ 39.0625    ,   0.        ,   0.        , ...,
        >>> print seg.spiketrains   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<SpikeTrain(array([  2.29499992e-02,   6.79249987e-02, ...
        >>> print seg.events   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<Event: @21.1967754364 s, @21.2993755341 s, @21.350725174 s, ...
        >>> print seg.epochs   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        [<neo.core.epoch.Epoch object at 0x10561ba90>,
         <neo.core.epoch.Epoch object at 0x10561bad0>]
    """

    is_readable = True
    is_writable = False

    supported_objects = [Segment, AnalogSignal, SpikeTrain, Event, Epoch]
    readable_objects = [Segment]
    writeable_objects = []

    has_header = False
    is_streameable = False

    # This is for GUI stuff: a definition for parameters when reading.
    read_params = {Segment: []}
    write_params = None

    name = 'NeuroExplorer'
    extensions = ['nex']

    mode = 'file'

    def __init__(self, filename=None):
        """
        This class read a nex file.

        Arguments:
            filename: the filename to read
        """
        BaseIO.__init__(self)
        self.filename = filename

    def read_segment(self, lazy=False, cascade=True):
        fid = open(self.filename, 'rb')
        global_header = HeaderReader(fid, GlobalHeader).read_f(offset=0)
        # ~ print globalHeader
        #~ print 'version' , globalHeader['version']
        seg = Segment()
        seg.file_origin = os.path.basename(self.filename)
        seg.annotate(neuroexplorer_version=global_header['version'])
        seg.annotate(comment=global_header['comment'])

        if not cascade:
            return seg

        offset = 544
        for i in range(global_header['nvar']):
            entity_header = HeaderReader(fid, EntityHeader).read_f(
                offset=offset + i * 208)
            entity_header['name'] = entity_header['name'].replace('\x00', '')

            #print 'i',i, entityHeader['type']

            if entity_header['type'] == 0:
                # neuron
                if lazy:
                    spike_times = [] * pq.s
                else:
                    spike_times = np.memmap(self.filename, np.dtype('i4'), 'r',
                                            shape=(entity_header['n']),
                                            offset=entity_header['offset'])
                    spike_times = spike_times.astype('f8') / global_header[
                        'freq'] * pq.s
                sptr = SpikeTrain(
                    times=spike_times,
                    t_start=global_header['tbeg'] /
                    global_header['freq'] * pq.s,
                    t_stop=global_header['tend'] /
                    global_header['freq'] * pq.s,
                    name=entity_header['name'])
                if lazy:
                    sptr.lazy_shape = entity_header['n']
                sptr.annotate(channel_index=entity_header['WireNumber'])
                seg.spiketrains.append(sptr)

            if entity_header['type'] == 1:
                # event
                if lazy:
                    event_times = [] * pq.s
                else:
                    event_times = np.memmap(self.filename, np.dtype('i4'), 'r',
                                            shape=(entity_header['n']),
                                            offset=entity_header['offset'])
                    event_times = event_times.astype('f8') / global_header[
                        'freq'] * pq.s
                labels = np.array([''] * event_times.size, dtype='S')
                evar = Event(times=event_times, labels=labels,
                             channel_name=entity_header['name'])
                if lazy:
                    evar.lazy_shape = entity_header['n']
                seg.events.append(evar)

            if entity_header['type'] == 2:
                # interval
                if lazy:
                    start_times = [] * pq.s
                    stop_times = [] * pq.s
                else:
                    start_times = np.memmap(self.filename, np.dtype('i4'), 'r',
                                            shape=(entity_header['n']),
                                            offset=entity_header['offset'])
                    start_times = start_times.astype('f8') / global_header[
                        'freq'] * pq.s
                    stop_times = np.memmap(self.filename, np.dtype('i4'), 'r',
                                           shape=(entity_header['n']),
                                           offset=entity_header['offset'] +
                                           entity_header['n'] * 4)
                    stop_times = stop_times.astype('f') / global_header[
                        'freq'] * pq.s
                epar = Epoch(times=start_times,
                             durations=stop_times - start_times,
                             labels=np.array([''] * start_times.size,
                                             dtype='S'),
                             channel_name=entity_header['name'])
                if lazy:
                    epar.lazy_shape = entity_header['n']
                seg.epochs.append(epar)

            if entity_header['type'] == 3:
                # spiketrain and wavefoms
                if lazy:
                    spike_times = [] * pq.s
                    waveforms = None
                else:

                    spike_times = np.memmap(self.filename, np.dtype('i4'), 'r',
                                            shape=(entity_header['n']),
                                            offset=entity_header['offset'])
                    spike_times = spike_times.astype('f8') / global_header[
                        'freq'] * pq.s

                    waveforms = np.memmap(self.filename, np.dtype('i2'), 'r',
                                          shape=(entity_header['n'], 1,
                                                 entity_header['NPointsWave']),
                                          offset=entity_header['offset'] +
                                          entity_header['n'] * 4)
                    waveforms = (waveforms.astype('f') *
                                 entity_header['ADtoMV'] +
                                 entity_header['MVOffset']) * pq.mV
                t_stop = global_header['tend'] / global_header['freq'] * pq.s
                if spike_times.size > 0:
                    t_stop = max(t_stop, max(spike_times))
                sptr = SpikeTrain(
                    times=spike_times,
                    t_start=global_header['tbeg'] /
                    global_header['freq'] * pq.s,
                    #~ t_stop = max(globalHeader['tend']/
                    #~ globalHeader['freq']*pq.s,max(spike_times)),
                    t_stop=t_stop, name=entity_header['name'],
                    waveforms=waveforms,
                    sampling_rate=entity_header['WFrequency'] * pq.Hz,
                    left_sweep=0 * pq.ms)
                if lazy:
                    sptr.lazy_shape = entity_header['n']
                sptr.annotate(channel_index=entity_header['WireNumber'])
                seg.spiketrains.append(sptr)

            if entity_header['type'] == 4:
                # popvectors
                pass

            if entity_header['type'] == 5:
                # analog
                timestamps = np.memmap(self.filename, np.dtype('i4'), 'r',
                                       shape=(entity_header['n']),
                                       offset=entity_header['offset'])
                timestamps = timestamps.astype('f8') / global_header['freq']
                fragment_starts = np.memmap(self.filename, np.dtype('i4'), 'r',
                                            shape=(entity_header['n']),
                                            offset=entity_header['offset'])
                fragment_starts = fragment_starts.astype('f8') / global_header[
                    'freq']
                t_start = timestamps[0] - fragment_starts[0] / float(
                    entity_header['WFrequency'])
                del timestamps, fragment_starts

                if lazy:
                    signal = [] * pq.mV
                else:
                    signal = np.memmap(self.filename, np.dtype('i2'), 'r',
                                       shape=(entity_header['NPointsWave']),
                                       offset=entity_header['offset'])
                    signal = signal.astype('f')
                    signal *= entity_header['ADtoMV']
                    signal += entity_header['MVOffset']
                    signal = signal * pq.mV

                ana_sig = AnalogSignal(
                    signal=signal, t_start=t_start * pq.s,
                    sampling_rate=entity_header['WFrequency'] * pq.Hz,
                    name=entity_header['name'],
                    channel_index=entity_header['WireNumber'])
                if lazy:
                    ana_sig.lazy_shape = entity_header['NPointsWave']
                seg.analogsignals.append(ana_sig)

            if entity_header['type'] == 6:
                # markers  : TO TEST
                if lazy:
                    times = [] * pq.s
                    labels = np.array([], dtype='S')
                    markertype = None
                else:
                    times = np.memmap(self.filename, np.dtype('i4'), 'r',
                                      shape=(entity_header['n']),
                                      offset=entity_header['offset'])
                    times = times.astype('f8') / global_header['freq'] * pq.s
                    fid.seek(entity_header['offset'] + entity_header['n'] * 4)
                    markertype = fid.read(64).replace('\x00', '')
                    labels = np.memmap(
                        self.filename, np.dtype(
                            'S' + str(entity_header['MarkerLength'])),
                        'r', shape=(entity_header['n']),
                        offset=entity_header['offset'] +
                        entity_header['n'] * 4 + 64)
                ea = Event(times=times,
                           labels=labels.view(np.ndarray),
                           name=entity_header['name'],
                           channel_index=entity_header['WireNumber'],
                           marker_type=markertype)
                if lazy:
                    ea.lazy_shape = entity_header['n']
                seg.events.append(ea)

        seg.create_many_to_one_relationship()
        return seg


GlobalHeader = [
    ('signature', '4s'),
    ('version', 'i'),
    ('comment', '256s'),
    ('freq', 'd'),
    ('tbeg', 'i'),
    ('tend', 'i'),
    ('nvar', 'i'),
]

EntityHeader = [
    ('type', 'i'),
    ('varVersion', 'i'),
    ('name', '64s'),
    ('offset', 'i'),
    ('n', 'i'),
    ('WireNumber', 'i'),
    ('UnitNumber', 'i'),
    ('Gain', 'i'),
    ('Filter', 'i'),
    ('XPos', 'd'),
    ('YPos', 'd'),
    ('WFrequency', 'd'),
    ('ADtoMV', 'd'),
    ('NPointsWave', 'i'),
    ('NMarkers', 'i'),
    ('MarkerLength', 'i'),
    ('MVOffset', 'd'),
    ('dummy', '60s'),
]

MarkerHeader = [
    ('type', 'i'),
    ('varVersion', 'i'),
    ('name', '64s'),
    ('offset', 'i'),
    ('n', 'i'),
    ('WireNumber', 'i'),
    ('UnitNumber', 'i'),
    ('Gain', 'i'),
    ('Filter', 'i'),
]


class HeaderReader():
    def __init__(self, fid, description):
        self.fid = fid
        self.description = description

    def read_f(self, offset=0):
        self.fid.seek(offset)
        d = {}
        for key, fmt in self.description:
            val = struct.unpack(fmt, self.fid.read(struct.calcsize(fmt)))
            if len(val) == 1:
                val = val[0]
            else:
                val = list(val)
            d[key] = val
        return d
