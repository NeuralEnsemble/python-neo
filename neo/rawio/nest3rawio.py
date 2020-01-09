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


try:
    import nestio  # TODO from https://github.com/apeyser/nestio-tools (bring to setup.py, maybe change of name?)
    HAVE_NESTIO = True
except ImportError:
    HAVE_NESTIO = False
    nestio = None



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

    _obs_colid_mapping = {}  # {gid: column_id]
    signal_recording_devices = [b'multimeter']
    spike_recording_devices = [b'spike_detector']

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

        # TODO: Create one Channel_Index per recording device. Add 'label' annotation (rec_dev.label)
        # TODO: Pre-sort data now already for faster time range selection later
        # loop through data
        unit_channels = []
        sig_channels = []

        # usefull to get local channel index in nsX from the global channel index
        local_sig_indexes = []
        self._nids_per_rec_dev = {}

        # sorting data for faster access later
        self._sorted_data = {}
        for rec_dev in self.reader:
            gid = rec_dev.gid
            original_data = np.asarray(self.reader[gid])
            sorting_order = np.lexsort((original_data['f0'],original_data['f1']))
            self._sorted_data[gid] = original_data[sorting_order,...] # primary sorting time, secondary sorting channels

        for rec_dev in self.reader:
            gid = rec_dev.gid
            data = self._sorted_data[gid]


            # spike data as units
            if rec_dev.name in self.spike_recording_devices:
                neuron_ids = np.unique(data['f0'])
                for nid in neuron_ids:
                    unit_name = 'rd{}unit{}'.format(gid, nid)
                    unit_id = '{}#{}'.format(gid, nid)
                    wf_units = ''  # There are no waveforms recorded in this format.
                    wf_gain = 0.0
                    wf_offset = 0.0
                    wf_left_sweep = 0
                    wf_sampling_rate = 0.0
                    unit_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                          wf_offset, wf_left_sweep, wf_sampling_rate))

            # analog data as signals
            elif rec_dev.name in self.signal_recording_devices:
                samples_per_timestep = np.searchsorted(data['f1'], data['f1'][0], side='right', sorter=None)
                neuron_ids = data['f0'][:samples_per_timestep]
                self._nids_per_rec_dev[gid] = neuron_ids

                for nid in neuron_ids:
                    self._obs_colid_mapping[gid] = {}
                    sampling_rate = self._get_sampling_rate(rec_dev) # in s
                    for cols, observables in zip(['f3', 'f4'], [rec_dev.double_observables, rec_dev.long_observables]):
                        for col_id, obs in enumerate(observables):
                            ch_name = 'rd{}unit{}signal{}'.format(gid, nid, obs.decode())
                            chan_id = nid  #'{}#{}#{}'.format(rec_dev.gid, nid, obs.decode())
                            sr = sampling_rate
                            dtype = data[cols].dtype  # float if obs in rec_dev.double_observables else int # or
                            units = self._get_signal_unit(obs)
                            gain = 1.
                            offset = 0.
                            group_id = gid
                            sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, group_id))

                            # self._obs_colid_mapping[gid][obs.decode()] = (cols, col_id)

                            local_sig_indexes.append((cols, col_id))


        self._local_sig_indexes = np.array(local_sig_indexes)

            #i.gid, i.name, i.label, i.double_n_val, i.double_observables, i.long_n_val, i.long_observables, i.origin,
            # i.rows, i.dtype, i.t_start, i.t_stop

        # finalize header
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        event_channels = np.array([], dtype=_event_channel_dtype)
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels


        # minimal annotations from BaseRawIO
        self._generate_minimal_annotations()

        self.raw_annotations['blocks'][0].update(block_ann)

        seg_ann = self.raw_annotations['blocks'][0]['segments'][0]
        seg_ann['name'] = 'Seg #{} Block #{}'.format(0, 0)
        seg_ann['seg_extra_info'] = 'This is the seg {} of block {}'.format(0, 0)

        # for rec_dev in self.reader:
        #     neuron_ids = np.unique(np.array(self.reader[rec_dev.gid])['f0'])
        #     for nid in neuron_ids:
        #
        #
        # for c in range(len(sig_channels)):
        #     anasig_an = seg_ann['signals'][c]
        #     anasig_an['info'] = 'This is a good signals'
        # for c in range(3):
        #     spiketrain_an = seg_ann['units'][c]
        #     spiketrain_an['quality'] = 'Good!!'
        # for c in range(2):
        #     event_an = seg_ann['events'][c]
        #     if c == 0:
        #         event_an['nickname'] = 'Miss Event 0'
        #     elif c == 1:
        #         event_an['nickname'] = 'MrEpoch 1'


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
        # the channel_indexes belong to the same recording device (checked by baserawio)
        # number of samples available in requested channels
        rd_id, _ = self._get_gid_and_local_indexes(channel_indexes)

        # all channels for this recording device have the same number of samples
        # the samples per channel are therefore the total samples / number of channels (nids)
        sig_size = len(self._sorted_data[rd_id]) / len(self._nids_per_rec_dev[rd_id])
        assert sig_size == int(sig_size), 'Error in signal size extraction'
        return int(sig_size)


    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):

        rd_id, local_ids = self._get_gid_and_local_indexes(channel_indexes)

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
        # channel_indexes is checked by BaseRawIO to belong to a single group_id and characteristics

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

        # TODO: This needs performance optimization



        rd_id, local_ids = self._get_gid_and_local_indexes(channel_indexes)
        # local_ids = col, col_id

        # checking for consistent col, as this defines the dtype of the signal
        assert all(local_ids[:,0]==local_ids[0,0]), 'Attempting to load signals with different data types into single AnalogSignal'
        datacolumn_id = local_ids[0,0]

        # all signals have the same number of samples for a signal recording device
        samples_per_nid = self._get_signal_size(block_index, seg_index, channel_indexes=[0])
        nids = self.header['signal_channels'][channel_indexes]['id']

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = samples_per_nid

        # extracting all rows containing requested times and nids
        # mask_per_time_step = np.in1d(self._nids_per_rec_dev[rd_id], nids)
        # nid_mask = np.repeat(mask_per_time_step, samples_per_nid)
        nid_mask = True
        time_mask = np.zeros(self._sorted_data[rd_id].shape[0], dtype=bool)
        time_mask[i_start*samples_per_nid: i_stop*samples_per_nid] = True
        mask = np.logical_and(nid_mask, time_mask)

        # Extract relevant data packets
        data = self._sorted_data[rd_id][mask]#.reshape((samples_per_nid,len(nids))) # (t, nid)


        # unfolding 2d data structure using advanced indexing: nid->channel_index
        m =np.searchsorted(self._nids_per_rec_dev, nids) # this has len(channel_indexes)
        l = len(data) / self._nids_per_rec_dev[rd_id]
        ma = np.array([m+i*len(self._nids_per_rec_dev[rd_id]) for i in range(l)])
        data_signals = data[datacolumn_id][ma]

        return data_signals.reshape((samples_per_nid, len(channel_indexes)))


        def get_local_signal(data, local_id):
            col, col_id = local_id
            return data[col][col_id]

        # np.empty(shape=(le))

        vget_local = np.vectorize(get_local_signal)

        data = vget_local(data, local_ids)
        # Problem: Indexing into array with named dtype is not compatible with advanced indexing.
        # New approach: used flattened representation of original data and use advanced indexing there

        # Update:
        # signals will be separated for int and float dtype (as defined by baserawio common characteristics check)







        # Extract relevant data columns from packets
        #  note: nids can have more than one signal recorded per row. see col and col_id
        # this function should be vectorized
        # for col, col_id in local_ids:
        #     data[col][col_id]


        return data

        # data = np.asarray(self.reader[rd_id]) # sorted by time, nid
        #
        # # Extract signal values of all selected nids
        #
        #
        # for channel_index in channel_indexes:
        #     rd_id, nid, signal = self.header['signal_channels'][channel_index][1].split('#')
        #     rd_id, nid = int(rd_id), int(nid)
        #
        #
        #
        #
        #     idx = np.argwhere(data['f0'] == nid)

            # if i_start is None:
            #     i_start = 0
            # if i_stop is None:
            #     i_stop = sum(idx)

        #     res.append(np.sort(data[idx])[i_start: i_stop][col][col_id])
        #
        # return np.asarray(res)


        # assert i_start >= 0, "I don't like your jokes"
        # assert i_stop <= 100000, "I don't like your jokes"

        # if channel_indexes is None:
        #     nb_chan = 16
        # else:
        #     nb_chan = len(channel_indexes)
        # raw_signals = np.zeros((i_stop - i_start, nb_chan), dtype='int16')
        # return raw_signals


    def _spike_count(self, block_index, seg_index, unit_index):
        # Must return the nb of spike for given (block_index, seg_index, unit_index)
        sd_id, nid = self.header['unit_channels'][unit_index][1].split('#')
        sd_id, nid = int(sd_id), int(nid)

        assert self.reader[sd_id].name in self.spike_recording_devices, \
            'This unit was not recorded by a spike detector!'

        data = np.asarray(self.reader[sd_id])
        return np.sum(data['f0'] == nid)



    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        sim_resolution = self.raw_annotations['blocks'][block_index]['sim_resolution']
        # extract spike detector id and n....
        sd_id, nid = self.header['unit_channels'][unit_index][1].split('#')
        sd_id, nid = int(sd_id), int(nid)

        assert self.reader[sd_id].name in self.spike_recording_devices, \
            'This unit was not recorded by a spike detector!'

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

    def _get_sampling_rate(self, rec_dev):

        assert rec_dev.name in self.signal_recording_devices, 'Recording device {} does not have a sampling rate {}'.format(rec_dev.name)

        gid = rec_dev.gid
        data = self._sorted_data[gid]

        samples_per_nid = len(data['f1']) / len(self._nids_per_rec_dev[gid])
        sampling_rate = (data['f1'][-1] - data['f1'][0]) / (samples_per_nid - 1) * self.reader.resolution * 1e-3
        # TODO: Does the offset (data['f2']) play a role here?

        return sampling_rate # in s

    def _get_signal_unit(self, obs):
        if obs==b'V_m':
            return 'mV'
        if obs in [b'I_syn_ex', b'I_syn_in']:
            return 'pA'
        raise ValueError('Unit can not be extracted from recordable name {}'.format(obs))


    ### templates from other RawIOs

    def _get_gid_and_local_indexes(self, channel_indexes):
        # internal helper to get rd gid and local channel indexes from global channel indexes
        # when this is called channel_indexes are always in the same group_id this is checked at BaseRaw level
        if channel_indexes is None:
            channel_indexes = slice(None)
        gid = self.header['signal_channels'][channel_indexes]['group_id'][0]
        if channel_indexes is None:
            local_indexes = slice(None)
        else:
            local_indexes = self._local_sig_indexes[channel_indexes]
        return gid, local_indexes