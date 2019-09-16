# -*- coding: utf-8 -*-
"""
NestSionRawIO is a class for reading output files from NEST simulations
( http://www.nest-simulator.org/ ) written with the library SIONlib.
SIONlib ( http://www.fz-juelich.de/jsc/sionlib ) is a scalable I/O library for
parallel access to task-local files.



Author: Johanna Senk
"""
from __future__ import unicode_literals, print_function, division, absolute_import

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np

import nestio # TODO from https://github.com/apeyser/nestio-tools (bring to setup.py, maybe change of name?)


class Nest3RawIO(BaseRawIO):
    """
    Class for "reading" fake data from an imaginary file.

    For the user, it give acces to raw data (signals, event, spikes) as they
    are in the (fake) file int16 and int64.

    For a developer, it is just an example showing guidelines for someone who wants
    to develop a new IO module.

    Two rules for developers:
      * Respect the Neo RawIO API (:ref:`_neo_rawio_API`)
      * Follow :ref:`_io_guiline`

    This fake IO:
        * have 2 blocks
        * blocks have 2 and 3 segments
        * have 16 signal_channel sample_rate = 10000
        * have 3 unit_channel
        * have 2 event channel: one have *type=event*, the other have
          *type=epoch*


    Usage:
        >>> import neo.rawio
        >>> r = neo.rawio.ExampleRawIO(filename='output.sion')
        >>> r.parse_header()
        >>> print(r)
        >>> raw_chunk = r.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=0, i_stop=1024,  channel_names=channel_names)
        >>> float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, dtype='float64',
                            channel_indexes=[0, 3, 6])
        >>> spike_timestamp = reader.spike_timestamps(unit_index=0, t_start=None, t_stop=None)
        >>> spike_times = reader.rescale_spike_timestamp(spike_timestamp, 'float64')
        >>> ev_timestamps, _, ev_labels = reader.event_timestamps(event_channel_index=0)

    """
    name = 'Nest3RawIO'
    description = ''
    extensions = ['sion']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):

        self.description = '' # TODO
        self.header = {}

        # access the .sion file with the reader from nestio-tools
        self.reader = nestio.NestReader(self.filename)

        # one block with one segment per .sion file
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]

        # block annotations: global information on simulation
        block_ann = { 'nest_version': self.reader.nest_version,
                      'sionlib_rec_backend_version' : self.reader.sionlib_rec_backend_version,
                      'sim_resolution': self.reader.resolution * 1e-3}  # in s

        # segment annotations: global information on simulation
        seg_ann = {}

        # loop through data
        unit_channels = []
        sig_channels = []
        for rec_dev in self.reader:
            neuron_ids = np.unique(np.array(self.reader[rec_dev.gid])['f0'])
            for nid in neuron_ids:

                # spike data as units
                if rec_dev.name == b'spike_detector':
                    unit_name = 'sd{}unit{}'.format(rec_dev.gid, nid)
                    unit_id = '{}#{}'.format(rec_dev.gid, nid)
                    wf_units = ''
                    wf_gain = 0.0
                    wf_offset = 0.0
                    wf_left_sweep = 0
                    wf_sampling_rate = 0.0
                    unit_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                          wf_offset, wf_left_sweep, wf_sampling_rate))

                # analog data as signals
                elif rec_dev.name == b'multimeter':
                    for obs in rec_dev.double_observables + rec_dev.long_observables:
                        ch_name = 'sd{}unit{}'.format(rec_dev.gid, nid)
                        chan_id = '{}#{}'.format(rec_dev.gid, nid)
                        sr = 0
                        dtype = float if obs in rec_dev.double_observables else int # or rec_dev.dtype
                        units = 0
                        gain = 0
                        offset = 0
                        group_id = 0
                        sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, group_id))

            #i.gid, i.name, i.label, i.double_n_val, i.double_observables, i.long_n_val, i.long_observables, i.origin,
            # i.rows, i.dtype, i.t_start, i.t_stop

        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        self.header['unit_channels'] = unit_channels

        # # signals and units are not global but specific to to the recording devices
        self.header['signal_channels'] = np.array([], dtype=_signal_channel_dtype)
        # self.header['unit_channels'] = np.array([], dtype=_unit_channel_dtype)

        # # no events or epochs
        self.header['event_channels'] = np.array([], dtype=_event_channel_dtype)


        # minimal annotations from BaseRawIO
        self._generate_minimal_annotations()

        self.raw_annotations['blocks'][0].update(block_ann)

        # # global information on simulation
        # self.nest_version =  self.reader.nest_version
        # self.sionlib_rec_backend_version =  self.reader.sionlib_rec_backend_version
        # self.sim_resolution = self.reader.resolution * 1e-3 # in s
        # self.sim_t_start = self.reader.t_start * 1e-3 # in s
        # self.sim_t_stop = self.reader.t_end * 1e-3 # in s

        # # set number of blocks and segments
        # self.header['nb_block'] = 0
        # self.header['nb_segment'] = []
        # for rec_dev in self.reader:
        #     # one block per recording device
        #     self.header['nb_block'] += 1
        #     # one segment per observable (at least one per block)
        #     self.header['nb_segment'].append(
        #         np.max([1, rec_dev.double_n_val + rec_dev.long_n_val]))



        # minimal annotations from BaseRawIO
        # self._generate_minimal_annotations()

        # bl_ann = self.raw_annotations['blocks'][block_index]

        # # annotate blocks with information on the recording device
        # for block_index,rec_dev in enumerate(self.reader):
        #     ba = self.raw_annotations['blocks'][block_index]

        #     ba['gid'] = rec_dev.gid
        #     ba['rec_dev'] = rec_dev.name
        #     ba['label'] = rec_dev.label

        #     # annotate segments: specify data columns
        #     seg_index = 0
        #     double_index = 0
        #     long_index = 0
        #     while seg_index < self.header['nb_segment'][block_index]:
        #         sa = ba['segments'][seg_index]
        #         sa['data'] = {}
        #         sa['data'].update({'gids': ['f0'],
        #                            'times': ['f1']})

        #         if double_index < rec_dev.double_n_val:
        #             sa['data'].update({rec_dev.double_observables[double_index]: ['f3', double_index]})
        #             double_index += 1

        #         if long_index < rec_dev.long_n_val and double_index >= rec_dev.double_n_val:
        #             sa['data'].update({rec_dev.long_observables[long_index]: ['f4', long_index]})
        #             long_index += 1
        #         seg_index += 1


    def _segment_t_start(self, block_index, seg_index):
        # DONE
        # INDEPENDENT OF SEG_INDEX

        # this must return an float scale in second
        # this t_start will be shared by all object in the segment
        # except AnalogSignal
        # gid_rec_dev = self.raw_annotations['blocks'][block_index]['gid']
        # t_start_rec_dev = self.reader[gid_rec_dev].t_start * self.sim_resolution
        # t_start = np.max([self.sim_t_start, t_start_rec_dev])
        # return t_start
        return  self.reader.t_start * 1e-3


    def _segment_t_stop(self, block_index, seg_index):
        # DONE
        # INDEPENDENT OF SEG_INDEX

        # this must return an float scale in second
        # gid_rec_dev = self.raw_annotations['blocks'][block_index]['gid']
        # t_stop_rec_dev = self.reader[gid_rec_dev].t_stop * self.sim_resolution
        # t_stop = np.min([self.sim_t_stop, t_stop_rec_dev])
        # return t_stop

        return  self.reader.t_end * 1e-3


    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        # TODO
        #
        # we are lucky: signals in all segment have the same shape!! (10.0 seconds)
        # it is not always the case
        # this must return an int = the number of sample

        # Note that channel_indexes can be ignored for most cases
        # except for several sampling rate.
        return rows


    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        # DONE
        # same as _segment_t_start
        #
        # This give the t_start of signals.
        # Very often this equal to _segment_t_start but not
        # always.
        # this must return an float scale in second

        # Note that channel_indexes can be ignored for most cases
        # except for several sampling rate.

        # Here this is the same.
        # this is not always the case
        return 0
        # return self._segment_t_start(block_index, seg_index)


    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        # TODO
        # this must return a signal chunk limited with
        # i_start/i_stop (can be None)
        # channel_indexes can be None (=all channel) or a list or numpy.array
        # This must return a numpy array 2D (even with one channel).
        # This must return the original dtype. No conversion here.
        # This must as fast as possible.
        # Everything that can be done in _parse_header() must not be here.

        # Here we are lucky:  our signals is always zeros!!
        # it is not always the case
        # internally signals are int16
        # convertion to real units is done with self.header['signal_channels']

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = 100000

        assert i_start >= 0, "I don't like your jokes"
        assert i_stop <= 100000, "I don't like your jokes"

        if channel_indexes is None:
            nb_chan = 16
        else:
            nb_chan = len(channel_indexes)
        raw_signals = np.zeros((i_stop - i_start, nb_chan), dtype='int16')
        return raw_signals


    def _spike_count(self, block_index, seg_index, unit_index):
        # Must return the nb of spike for given (block_index, seg_index, unit_index)
        sd_id, nid = self.header['unit_channels'][unit_index][1].split('#')
        sd_id, nid = int(sd_id), int(nid)

        assert self.reader[sd_id].name == b'spike_detector', \
            'This unit was not recorded by a spike_detector!'

        data = np.asarray(self.reader[sd_id])
        return np.sum(data['f0'] == nid)



    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        sim_resolution = self.raw_annotations['blocks'][block_index]['sim_resolution']
        # extract spike detector id and n....
        sd_id, nid = self.header['unit_channels'][unit_index][1].split('#')
        sd_id, nid = int(sd_id), int(nid)

        assert self.reader[sd_id].name == b'spike_detector', \
            'This unit was not recorded by a spike_detector!'

        data = np.asarray(self.reader[sd_id])
        idx = np.argwhere(data['f0'] == nid)

        # TODO: Check if first and last possible spike is within the limits below
        sd_t_start = self.reader[sd_id].t_start * sim_resolution
        sd_t_stop = self.reader[sd_id].t_stop * sim_resolution

        spike_start = max(sd_t_start, self.segment_t_start(block_index, seg_index))
        spike_stop = min(sd_t_stop, self.segment_t_stop(block_index, seg_index))

        if t_start is None:
            t_start = spike_start
        if t_stop is None:
            t_stop = spike_stop

        assert sd_t_start <= t_start, 't_start ({}) must be larger than or equal to beginning of spike recording ({}).' \
                                      ''.format(t_start,spike_start)
        assert sd_t_stop >= t_stop, 't_stop ({}) must be smaller than or equal to end of spike recording ({}).' \
                                    ''.format(t_stop, spike_stop)

        # # TODO: IS THIS CORRECT?
        # # step * resolution + offset, result in s
        all_spike_timestamps = data['f1'][idx] * sim_resolution + data['f2'][idx] * 1e-3
        #
        mask = (all_spike_timestamps >= t_start) & (all_spike_timestamps <= t_stop)
        spike_timestamps = all_spike_timestamps[mask]

        return spike_timestamps


    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        # SPIKE_TIMESTAMPS ARE ALREADY IN S BECAUSE OF STEP AND OFFSET, THIS CHANGES ONLY DTYPE
        spike_times = spike_timestamps.astype(dtype)
        return spike_times


    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        return None


    def _event_count(self, block_index, seg_index, event_channel_index):
        return 0


    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        return None


    def _rescale_event_timestamp(self, event_timestamps, dtype):
        # must rescale to second a particular event_timestamps
        # with a fixed dtype so the user can choose the precisino he want.

        # really easy here because in our case it is already seconds
        event_times = event_timestamps.astype(dtype)
        return event_times


    def _rescale_epoch_duration(self, raw_duration, dtype):
        # really easy here because in our case it is already seconds
        durations = raw_duration.astype(dtype)
        return durations
