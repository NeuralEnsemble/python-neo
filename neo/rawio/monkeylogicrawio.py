"""
RawIO Class for MonkeyLogic files

The RawIO assumes all segments and all blocks have the same structure.
It supports all kinds of NEO objects.
This IO does not support lazy loading.
Reading of bhv2 files based on https://monkeylogic.nimh.nih.gov/docs_BHV2BinaryStructure.html

Author: Julia Sprenger
"""

import numpy as np
import struct

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype)


class MLBLock(dict):
    n_byte_dtype = {'logical': (1, '?'),
                    'char': (1, 'c'),
                    'integers': (8, 'Q'),
                    'uint8': (1, 'B'),
                    'single': (4, 'f'),
                    'double': (8, 'd')}

    @staticmethod
    def generate_block(f):
        """
        Generate a new ML block object
        :param f: file handle to read to create new block
        :return:
        """
        LN = f.read(8)
        # No MLBlock, e.g. due to EOF
        if not LN:
            return None
        LN = struct.unpack('Q', LN)[0]
        # print(f'\nLN: {LN}')
        var_name = f.read(LN)
        # print(var_name)

        LT = f.read(8)
        LT = struct.unpack('Q', LT)[0]
        # print(f'LT: {LT}')
        var_type = f.read(LT)
        # print(var_type)

        DV = f.read(8)
        DV = struct.unpack('Q', DV)[0]
        # print(f'DV: {DV}')
        var_size = f.read(DV * 8)
        var_size = struct.unpack(f'{DV}Q', var_size)
        # print(var_size)

        return MLBLock(LN, var_name, LT, var_type, DV, var_size)

    def __bool__(self):
        if any((self.LN, self.LT)):
            return True
        else:
            return False

    def __init__(self, LN, var_name, LT, var_type, DV, var_size):
        self.LN = LN
        self.var_name = var_name.decode()
        self.LT = LT
        self.var_type = var_type.decode()
        self.DV = DV
        self.var_size = var_size

        self.children = self
        self.data = None

    def __repr__(self):
        if self.data is None:
            shape = 0
            dt = ''
        else:
            shape = getattr(self.data, 'shape', len(self.data))
            dt = f' dtype: {self.var_type}'

        return f'MLBLock [{shape}|{len(self)}] "{self.var_name}"{dt}'

    def read_data(self, f, recursive=False):
        """
        Read data based on the file handle f

        Parameters
        ----------
        f file handle
        recursive

        Returns
        -------

        """

        # reading basic data types
        if self.var_type in self.n_byte_dtype:
            n_byte, format = self.n_byte_dtype[self.var_type]

            data = np.empty(shape=np.prod(self.var_size), dtype=format)

            for i in range(np.prod(self.var_size)):
                d = f.read(n_byte)
                d = struct.unpack(format, d)[0]
                data[i] = d

            data = data.reshape(self.var_size)

            # decoding characters
            if self.var_type == 'char':
                data = np.char.decode(data)

                # handling convert array to string when only single dimension
                if np.prod(self.var_size) == np.max(self.var_size):
                    data = ''.join(c for c in data.flatten())

            # print(f'data: {data}')

            self.data = data

        # reading potentially nested data types
        elif self.var_type == 'struct':
            n_fields = f.read(8)
            n_fields = struct.unpack('Q', n_fields)[0]

            for field in range(n_fields * np.prod(self.var_size)):
                bl = MLBLock.generate_block(f)
                if recursive:
                    self[bl.var_name] = bl
                bl.read_data(f, recursive=recursive)

        elif self.var_type == 'cell':
            for field in range(np.prod(self.var_size)):
                bl = MLBLock.generate_block(f)
                if recursive:
                    self[bl.var_name] = bl
                bl.read_data(f, recursive=recursive)

        else:
            raise ValueError(f'unknown variable type {self.var_type}')

        self.flatten()

    def flatten(self):
        '''
        Reassigning data objects to be children of parent dict
        block1.block2.data -> block1.data as block2 anyway does not contain keys
        '''
        for k, v in self.items():
            # Sanity check: Blocks can either have children or contain data
            if v.data is not None and len(v.keys()):
                raise ValueError(f'Block {k} has {len(k)} children and data: {v.data}')

            if v.data is not None:
                self[k] = v.data


class MonkeyLogicRawIO(BaseRawIO):
    extensions = ['bhv2']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = str(filename)

    def _source_name(self):
        return self.filename

    def _data_sanity_checks(self):
        for trial_id in self.trial_ids:
            events = self.ml_blocks[f'Trial{trial_id}']['BehavioralCodes']

            # sanity check: last event == trial end
            first_event_code = events['CodeNumbers'][0]
            last_event_code = events['CodeNumbers'][-1]
            assert first_event_code == 9  # 9 denotes sending of trial start event
            assert last_event_code == 18  # 18 denotes sending of trial end event

    def _parse_header(self):
        self.ml_blocks = {}

        with open(self.filename, 'rb') as f:
            while bl := MLBLock.generate_block(f):
                bl.read_data(f, recursive=True)
                self.ml_blocks[bl.var_name] = bl

        trial_rec = self.ml_blocks['TrialRecord']
        self.trial_ids = np.arange(1, int(trial_rec['CurrentTrialNumber']))

        self._data_sanity_checks()

        exclude_signals = ['SampleInterval']

        # rawio configuration
        signal_streams = []
        signal_channels = []

        if 'Trial1' in self.ml_blocks:
            chan_id = 0
            stream_id = 0
            chan_names = []

            ana_block = self.ml_blocks['Trial1']['AnalogData']

            def _register_signal(sig_block, name):
                nonlocal stream_id
                nonlocal chan_id
                if not isinstance(sig_data, dict) and any(sig_data.shape):
                    signal_streams.append((name, stream_id))

                    sr = 1  # TODO: Where to get the sampling rate info?
                    dtype = type(sig_data)
                    units = ''  # TODO: Where to find the unit info?
                    gain = 1  # TODO: Where to find the gain info?
                    offset = 0  # TODO: Can signals have an offset in ML?
                    stream_id = 0  # all analog data belong to same stream

                    if sig_block.shape[1] == 1:
                        signal_channels.append((name, chan_id, sr, dtype, units, gain, offset,
                                                stream_id))
                        chan_id += 1
                    else:
                        for sub_chan_id in range(sig_block.shape[1]):
                            signal_channels.append(
                                (name, chan_id, sr, dtype, units, gain, offset,
                                 stream_id))
                            chan_id += 1

            for sig_name, sig_data in ana_block.items():
                if sig_name in exclude_signals:
                    continue

                # 1st level signals ('Trial1'/'AnalogData'/<signal>')
                if not isinstance(sig_data, dict) and any(sig_data.shape):
                    _register_signal(sig_data, name=sig_name)

                # 2nd level signals ('Trial1'/'AnalogData'/<signal_group>/<signal>')
                elif isinstance(sig_data, dict):
                    for sig_sub_name, sig_sub_data in sig_data.items():
                        if not isinstance(sig_sub_data, dict):
                            name = f'{sig_name}/{sig_sub_name}'
                            chan_names.append(name)
                            _register_signal(sig_sub_data, name=name)

        # ML does not record spike information
        spike_channels = []

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        event_channels = []
        event_channels.append(('ML Trials', 0, 'event'))
        # event_channels.append(('ML Trials', 1, 'epoch')) # no epochs supported yet
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [len(self.trial_ids)]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

        # adding custom annotations and array annotations

        ignore_annotations = [
            # data blocks
            'AnalogData', 'AbsoluteTrialStartTime', 'BehavioralCodes', 'CodeNumbers',
            # ML temporary variables
            'ConditionsThisBlock',
            'CurrentBlock', 'CurrentBlockCount', 'CurrentBlockCondition',
            'CurrentBlockInfo', 'CurrentBlockStimulusInfo', 'CurrentTrialNumber',
            'CurrentTrialWithinBlock', 'LastTrialAnalogData', 'LastTrialCodes',
            'NextBlock', 'NextCondition']

        def _filter_keys(full_dict, ignore_keys, simplify=True):
            res = {}
            for k, v in full_dict.items():
                if k in ignore_keys:
                    continue

                if isinstance(v, dict):
                    res[k] = _filter_keys(v, ignore_keys)
                else:
                    if simplify and isinstance(v, np.ndarray) and np.prod(v.shape) == 1:
                        v = v.flat[0]
                    res[k] = v
            return res

        ml_ann = {k: v for k, v in self.ml_blocks.items() if k in ['MLConfig', 'TrialRecord']}
        ml_ann = _filter_keys(ml_ann, ignore_annotations)
        bl_ann = self.raw_annotations['blocks'][0]
        bl_ann.update(ml_ann)

        for trial_id in self.trial_ids:
            ml_trial = self.ml_blocks[f'Trial{trial_id}']
            assert ml_trial['Trial'] == trial_id

            seg_ann = self.raw_annotations['blocks'][0]['segments'][trial_id-1]
            seg_ann.update(_filter_keys(ml_trial, ignore_annotations))

        event_ann = seg_ann['events'][0]  # 0 is event
        # epoch_ann = seg_ann['events'][1]  # 1 is epoch

    def _segment_t_start(self, block_index, seg_index):
        assert block_index == 0

        t_start = self.ml_blocks[f'Trial{seg_index + 1}']['AbsoluteTrialStartTime'][0][0]
        return t_start

    def _segment_t_stop(self, block_index, seg_index):
        t_start = self._segment_t_start(block_index, seg_index)
        # using stream 0 as all analogsignal stream should have the same duration
        duration = self._get_signal_size(block_index, seg_index, 0)

        return t_start + duration

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_name, stream_id = self.header['signal_streams'][stream_index]

        block = self.ml_blocks[f'Trial{seg_index + 1}']['AnalogData']
        for sn in stream_name.split('/'):  # dealing with 1st and 2nd level signals
            block = block[sn]

        size = block.shape[0]
        return size  # size is per signal, not the sum of all channel_indexes

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        sig_t_start = self._segment_t_start(block_index, seg_index)
        return sig_t_start

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index,
                                channel_indexes):
        stream_name, stream_id = self.header['signal_streams'][stream_index]

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self.get_signal_size(block_index, seg_index, stream_index)

        raw_signals_list = []
        block = self.ml_blocks[f'Trial{seg_index + 1}']['AnalogData']
        for sn in stream_name.split('/'):
            block = block[sn]

        if channel_indexes is None:
            raw_signals = block
        else:
            raw_signals = block[channel_indexes]

        raw_signals = raw_signals[i_start:i_stop]
        return raw_signals

    def _spike_count(self, block_index, seg_index, unit_index):
        return None

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        return None

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        return None

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        return None

    def _event_count(self, block_index, seg_index, event_channel_index):
        assert event_channel_index == 0
        times = self.ml_blocks[f'Trial{seg_index + 1}']['BehavioralCodes']['CodeTimes']

        return len(times)

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):

        durations = None
        assert block_index == 0
        assert event_channel_index == 0

        timestamp = self.ml_blocks[f'Trial{seg_index + 1}']['BehavioralCodes']['CodeTimes']
        timestamp = timestamp.flatten()
        labels = self.ml_blocks[f'Trial{seg_index + 1}']['BehavioralCodes']['CodeNumbers']
        labels = labels.flatten()

        if t_start is not None:
            keep = timestamp >= t_start
            timestamp, labels = timestamp[keep], labels[keep]

        if t_stop is not None:
            keep = timestamp <= t_stop
            timestamp, labels = timestamp[keep], labels[keep]

        return timestamp, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        # times are stored in millisecond, see
        # shttps://monkeylogic.nimh.nih.gov/docs_GettingStarted.html#FormatsSupported
        event_timestamps /= 1000
        return event_timestamps.astype(dtype)  # return in seconds

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        # times are stored in millisecond, see
        # shttps://monkeylogic.nimh.nih.gov/docs_GettingStarted.html#FormatsSupported
        raw_duration /= 1000
        return raw_duration.astype(dtype)  # return in seconds
