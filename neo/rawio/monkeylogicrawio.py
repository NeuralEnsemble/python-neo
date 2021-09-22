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

            for field in range(n_fields*np.prod(self.var_size)):
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

        # Sanity check: Blocks can only have children or contain data
        if self.data is not None and len(self.keys()):
            raise ValueError(f'Block {self.var_name} has {len(self)} children and data: {self.data}')





class MonkeyLogicRawIO(BaseRawIO):

    extensions = ['bhv2']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = str(filename)

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        self.ml_blocks = {}

        with open(self.filename, 'rb') as f:
            while bl := MLBLock.generate_block(f):
                bl.read_data(f, recursive=True)
                self.ml_blocks[bl.var_name] = bl

        trial_rec = self.ml_blocks['TrialRecord']
        self.trial_ids = np.arange(1, int(trial_rec['CurrentTrialNumber'].data))

        exclude_signals = ['SampleInterval']

        # rawio configuration
        signal_streams = []
        signal_channels = []

        if 'Trial1' in self.ml_blocks:
            chan_id = 0
            stream_id = 0
            chan_names = []

            ana_block = self.ml_blocks['Trial1']['AnalogData']

            def _register_signal(sig_block, prefix=''):
                nonlocal stream_id
                nonlocal chan_id
                if sig_data.data is not None and any(sig_data.data.shape):
                    signal_streams.append((prefix + sig_data.var_name, stream_id))

                    ch_name = sig_data.var_name
                    sr = 1  # TODO: Where to get the sampling rate info?
                    dtype = type(sig_data.data)
                    units = ''  # TODO: Where to find the unit info?
                    gain = 1  # TODO: Where to find the gain info?
                    offset = 0  # TODO: Can signals have an offset in ML?
                    stream_id = 0  # all analog data belong to same stream

                    if sig_block.data.shape[1] == 1:
                        signal_channels.append((prefix + ch_name, chan_id, sr, dtype, units, gain, offset,
                                            stream_id))
                        chan_id += 1
                    else:
                        for sub_chan_id in range(sig_block.data.shape[1]):
                            signal_channels.append(
                                (prefix + ch_name, chan_id, sr, dtype, units, gain, offset,
                                 stream_id))
                            chan_id += 1



            # 1st level signals ('Trial1'/'AnalogData'/<signal>')
            for sig_name, sig_data in ana_block.items():
                if sig_name in exclude_signals:
                    continue

                # 1st level signals
                if sig_data.data is not None and any(sig_data.data.shape):
                    _register_signal(sig_data)

                # 2nd level signals
                elif sig_data.keys():
                    for sig_sub_name, sig_sub_data in sig_data.items():
                        if sig_sub_data.data is not None:
                            chan_names.append(f'{sig_name}/{sig_sub_name}')
                            _register_signal(sig_sub_data, prefix=f'{sig_name}/')


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
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

        # adding custom annotations and array annotations

        ignore_annotations = ['AnalogData', 'AbsoluteTrialStartTime']
        array_annotation_keys = []

        ml_anno = {k: v for k, v in sorted(self.ml_blocks.items()) if not k.startswith('Trial')}
        bl_ann = self.raw_annotations['block'][0]
        bl_ann.update(ml_anno)

        # TODO annotate segments according to trial properties
        seg_ann = self.raw_annotations['blocks'][0]['segments'][0]
        seg_ann.update(ml_anno)

        event_ann = seg_ann['events'][0]  # 0 is event
        # epoch_ann = seg_ann['events'][1]  # 1 is epoch

        # TODO: add annotations for AnalogSignals
        # TODO: add array_annotations for AnalogSignals

        # ml_anno = {k: v for k, v in sorted(self.ml_blocks.items()) if k.startswith('Trial')}
        #
        # raise NotImplementedError()
        #
        # # extract array annotations
        # event_ann.update(self._filter_properties(props, 'ep'))
        # ev_idx += 1
        #
        # # adding array annotations to analogsignals
        # annotated_anasigs = []
        # sig_ann = seg_ann['signals']
        # # this implementation relies on analogsignals always being
        # # stored in the same stream order across segments
        # stream_id = 0
        # for da_idx, da in enumerate(group.data_arrays):
        #     if da.type != "neo.analogsignal":
        #         continue
        #     anasig_id = da.name.split('.')[-2]
        #     # skip already annotated signals as each channel already
        #     # contains the complete set of annotations and
        #     # array_annotations
        #     if anasig_id in annotated_anasigs:
        #         continue
        #     annotated_anasigs.append(anasig_id)
        #
        #     # collect annotation properties
        #     props = [p for p in da.metadata.props
        #              if p.type != 'ARRAYANNOTATION']
        #     props_dict = self._filter_properties(props, "analogsignal")
        #     sig_ann[stream_id].update(props_dict)
        #
        #     # collect array annotation properties
        #     props = [p for p in da.metadata.props
        #              if p.type == 'ARRAYANNOTATION']
        #     props_dict = self._filter_properties(props, "analogsignal")
        #     sig_ann[stream_id]['__array_annotations__'].update(
        #         props_dict)
        #
        #     stream_id += 1
        #
        # return

    def _segment_t_start(self, block_index, seg_index):
        if 'Trial1' in self.ml_blocks:
            t_start = self.ml_blocks['Trial1']['AbsoluteTrialStartTime'].data[0][0]
        else:
            t_start = 0
        return t_start

    def _segment_t_stop(self, block_index, seg_index):
        last_trial = self.ml_blocks[f'Trial{self.trial_ids[-1]}']

        t_start = last_trial['AbsoluteTrialStartTime'].data[0][0]
        t_stop = t_start + 10  # TODO: Find sampling rates to determine trial end
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_name, stream_id = self.header['signal_streams'][stream_index]

        block = self.ml_blocks[f'Trial{seg_index + 1}']['AnalogData']
        for sn in stream_name.split('/'):  # dealing with 1st and 2nd level signals
            block = block[sn]

        size = block.data.shape[0]
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
        block = self.ml_blocks[f'Trial{seg_index+1}']['AnalogData']
        for sn in stream_name.split('/'):
            block = block[sn]

        if channel_indexes is None:
            raw_signals = block.data
        else:
            raw_signals = block.data[channel_indexes]

        raw_signals = raw_signals[i_start:i_stop]
        return raw_signals

    def _spike_count(self, block_index, seg_index, unit_index):
        count = 0
        head_id = self.header['spike_channels'][unit_index][1]
        for mt in self.file.blocks[block_index].groups[seg_index].multi_tags:
            for src in mt.sources:
                if mt.type == 'neo.spiketrain' and [src.type == "neo.unit"]:
                    if head_id == src.id:
                        return len(mt.positions)
        return count

    def _get_spike_timestamps(self, block_index, seg_index, unit_index,
                              t_start, t_stop):
        block = self.unit_list['blocks'][block_index]
        segment = block['segments'][seg_index]
        spike_dict = segment['spiketrains']
        spike_timestamps = spike_dict[unit_index]
        spike_timestamps = np.transpose(spike_timestamps)

        if t_start is not None or t_stop is not None:
            lim0 = t_start
            lim1 = t_stop
            mask = (spike_timestamps >= lim0) & (spike_timestamps <= lim1)
            spike_timestamps = spike_timestamps[mask]
        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index,
                                 t_start, t_stop):
        # this must return a 3D numpy array (nb_spike, nb_channel, nb_sample)
        seg = self.unit_list['blocks'][block_index]['segments'][seg_index]
        waveforms = seg['spiketrains_unit'][unit_index]['waveforms']
        if not waveforms:
            return None
        raw_waveforms = np.array(waveforms)

        if t_start is not None:
            lim0 = t_start
            mask = (raw_waveforms >= lim0)
            # use nan to keep the shape
            raw_waveforms = np.where(mask, raw_waveforms, np.nan)
        if t_stop is not None:
            lim1 = t_stop
            mask = (raw_waveforms <= lim1)
            raw_waveforms = np.where(mask, raw_waveforms, np.nan)
        return raw_waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        assert event_channel_index == 0
        times = self.ml_blocks[f'Trial{seg_index+1}']['BehavioralCodes']['CodeTimes'].data

        return len(times)

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):

        durations = None
        assert block_index == 0
        assert event_channel_index == 0

        timestamp = self.ml_blocks[f'Trial{seg_index+1}']['BehavioralCodes']['CodeTimes'].data
        timestamp = timestamp.flatten()
        labels = self.ml_blocks[f'Trial{seg_index+1}']['BehavioralCodes']['CodeNumbers'].data
        labels = labels.flatten()

        if t_start is not None:
            keep = timestamp >= t_start
            timestamp, labels = timestamp[keep], labels[keep]

        if t_stop is not None:
            keep = timestamp <= t_stop
            timestamp, labels = timestamp[keep], labels[keep]

        return timestamp, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        # TODO: Figure out unit and scaling of event timestamps
        event_timestamps /= 1000  # assume this is in milliseconds
        return event_timestamps.astype(dtype)  # return in seconds

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        # TODO: Figure out unit and scaling of event timestamps
        raw_duration /= 1000  # assume this is in milliseconds
        return raw_duration.astype(dtype)  # return in seconds

