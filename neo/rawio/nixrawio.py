"""
RawIO Class for NIX files

The RawIO assumes all segments and all blocks have the same structure.
It supports all kinds of NEO objects.

Author: Chek Yin Choi, Julia Sprenger
"""

from distutils.version import LooseVersion as Version
import numpy as np

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype)
from ..io.nixio import check_nix_version

try:
    import nixio as nix

    HAVE_NIX = True
except ImportError:
    HAVE_NIX = False
    nix = None


# When reading metadata properties, the following keys are ignored since they
# are used to store Neo object properties.
# This dictionary is used in the _filter_properties() method.
neo_attributes = {
    "segment": ["index"],
    "analogsignal": ["units", "copy", "sampling_rate", "t_start"],
    "spiketrain": ["units", "copy", "sampling_rate", "t_start", "t_stop",
                   "waveforms", "left_sweep"],
    "event": ["times", "labels", "units", "durations", "copy"]
}


class NIXRawIO(BaseRawIO):

    extensions = ['nix']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        check_nix_version()
        BaseRawIO.__init__(self)
        self.filename = str(filename)

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        self.file = nix.File.open(self.filename, nix.FileMode.ReadOnly)
        if 'version' in self.file.sections['neo']:
            self._file_version = Version(self.file.sections['neo']['version'])
        else:
            self._file_version = Version('0.5.2')  # default if unknown

        signal_channels = []
        self.neo_struct = {'blocks': []}
        bl_idx = 0
        for bl in self.file.blocks:
            seg_dict = {'segments': []}
            self.neo_struct['blocks'].append(seg_dict)
            seg_idx = 0
            for seg in bl.groups:
                if seg.type != 'neo.segment':
                    continue
                signal_dict = {'signals': [],
                               'signal_types': [],
                               'signal_ids': []}
                self.neo_struct['blocks'][bl_idx]['segments'].append(signal_dict)

                # assume consistent stream / signal order across segments
                for da_idx, da in enumerate(seg.data_arrays):
                    # todo: This should also cover irreg & imagseq signals
                    if da.type in ["neo.analogsignal"]:
                        if self._file_version < Version('0.11.0dev0'):
                            anasig_id = da.name.split('.')[-2]
                        else:
                            anasig_id = da.name

                        # start a new signal if analogsignal id is new or changed
                        # This can be simplified when dropping support for old mapping
                        # no object exists yet -> create new object
                        if len(signal_dict['signals']) == 0:
                            signal_idx = 0
                            signal_dict['signals'].append({'data': [da]})
                            signal_dict['signal_types'].append(da.type)
                            signal_dict['signal_ids'].append(anasig_id)
                        # object is different -> create new object
                        elif anasig_id != signal_dict['signal_ids'][signal_idx]:
                            signal_idx += 1
                            signal_dict['signals'].append({'data': [da]})
                            signal_dict['signal_types'].append(da.type)
                            signal_dict['signal_ids'].append(anasig_id)
                        # object already exists (old nix mapping version)
                        else:
                            assert signal_dict['signal_ids'][signal_idx] == anasig_id
                            assert signal_dict['signal_types'][signal_idx] == da.type
                            signal_dict['signals'][signal_idx]['data'].append(da)
                seg_idx += 1
            bl_idx += 1

        # extract metadata from collected streams (t_start, t_stop, units, dtype, sampling_rate)
        for bl_idx, bl in enumerate(self.neo_struct['blocks']):
            for seg_idx, seg in enumerate(bl['segments']):
                for signal_idx, signal in enumerate(seg['signals']):
                    signal['units'] = []
                    signal['channel_names'] = []
                    t_start, t_stop = np.inf, -np.inf
                    chan_count, sample_count = 0, None
                    units, dtype, sampling_rate = None, None, None
                    for da in signal['data']:
                        time_dim = da.dimensions[0]  # in neo convention time is always dim 0
                        t_start = min(t_start, time_dim.offset)
                        duration = time_dim.sampling_interval * da.shape[0]
                        t_stop = max(t_start, time_dim.offset + duration)

                        n_chan = da.shape[-1] if len(da.shape) > 1 else 1
                        chan_count += n_chan
                        sample_count = da.shape[0] if sample_count is None else sample_count
                        assert sample_count == da.shape[0]
                        dtype = da.dtype if dtype is None else dtype
                        assert dtype == da.dtype
                        if sampling_rate is None:
                            sampling_rate = 1 / da.dimensions[0].sampling_interval
                        assert sampling_rate == 1 / da.dimensions[0].sampling_interval
                        # only channel_names and units are not shared by channels
                        signal['channel_names'].extend([da.metadata['neo_name']] * n_chan)
                        signal['units'].extend([da.unit] * n_chan)
                    signal['t_start'] = t_start
                    signal['t_stop'] = t_stop
                    signal['channel_count'] = chan_count
                    signal['sample_count'] = sample_count
                    signal['dtype'] = dtype
                    signal['sampling_rate'] = sampling_rate

                # calculate t_start and t_stop on segment level
                t_start, t_stop = np.inf, -np.inf
                for signal_idx, signal in enumerate(seg['signals']):
                    t_start = min(t_start, signal['t_start'])
                    t_stop = max(t_stop, signal['t_stop'])
                seg['t_start'] = t_start
                seg['t_stop'] = t_stop

        # extract streams from collected data objects
        seg0 = self.neo_struct['blocks'][0]['segments'][0]
        self.streams = {'signals': [], 'stream_ids': []}

        # consistency checks of data array across blocks and segments
        for bl_idx in range(1, len(self.neo_struct['blocks'])):
            bl_dict = self.neo_struct['blocks'][bl_idx]
            for seg_idx in range(1, len(bl_dict['segments'])):
                seg = bl_dict['segments'][seg_idx]
                assert len(seg0['signals']) == len(seg['signals'])
                for do_idx in range(len(seg0['signals'])):
                    assert seg0['signals'][do_idx]['channel_count'] == \
                        seg['signals'][do_idx]['channel_count']

        for signal_idx, signal in enumerate(seg0['signals']):
            # using the signal id in block 0 seg 0 to identify the whole stream across blocks
            stream_id = seg0['signal_ids'][signal_idx]
            self.streams['stream_ids'].append(stream_id)
            self.streams['signals'].append([])
            for bl_idx in range(len(self.neo_struct['blocks'])):
                bl = self.neo_struct['blocks'][bl_idx]
                for seg_idx in range(len(bl['segments'])):
                    seg = bl['segments'][seg_idx]
                    do = seg['signals'][signal_idx]
                    do['stream_id'] = stream_id

                    self.streams['signals'][signal_idx].append(do)

        # generate global signal channels for rawio
        chan_id = 0
        for signals_dict in seg0['signals']:
            stream_id = signals_dict['stream_id']
            dtype = signals_dict['dtype']
            sr = signals_dict['sampling_rate']
            gain = 1
            offset = 0.
            for inner_ch_idx in range(signals_dict['channel_count']):
                ch_name = signals_dict['channel_names'][inner_ch_idx]
                units = signals_dict['units'][inner_ch_idx]
                signal_channels.append((ch_name, chan_id, sr, dtype,
                                        units, gain, offset, stream_id))
                chan_id += 1

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        signal_streams = np.zeros(len(self.streams['stream_ids']), dtype=_signal_stream_dtype)
        signal_streams['id'] = self.streams['stream_ids']
        signal_streams['name'] = ''

        # SPIKETRAINS
        self.spiketrain_list = {'blocks': []}
        for block_index, blk in enumerate(self.file.blocks):
            seg_groups = [g for g in blk.groups if g.type == "neo.segment"]
            d = {'segments': []}
            self.spiketrain_list['blocks'].append(d)
            for seg_index, seg in enumerate(seg_groups):
                d = {'spiketrains': []}
                self.spiketrain_list['blocks'][block_index]['segments'].append(d)
                st_idx = 0
                for st in seg.multi_tags:
                    block = self.spiketrain_list['blocks'][block_index]
                    segment = block['segments'][seg_index]
                    if st.type == 'neo.spiketrain':
                        d = {'waveforms': None,
                             'spiketrain_id': st.id,
                             'unit_id': None,
                             'data': st,
                             'spike_count': len(st.positions),
                             't_start': None,
                             't_stop': None
                             }
                        segment['spiketrains'].append(d)
                        wftypestr = "neo.waveforms"
                        if 't_start' in st.metadata and 't_stop' in st.metadata:
                            t_start = st.metadata['t_start']
                            t_stop = st.metadata['t_stop']
                        else:
                            t_start = st.positions.dimensions[0].offset
                            t_stop = st.positions.dimensions[0].stop  # TODO: fix this
                        d['t_start'] = t_start
                        d['t_stop'] = t_stop
                        if (st.features and st.features[0].data.type == wftypestr):
                            waveforms = st.features[0].data
                            if waveforms:
                                d['waveforms'] = waveforms
                            # assume one spiketrain has one waveform

                        # spiketrains of first segment are used for unit ids across segment
                        if (block_index, seg_index) == (0, 0):
                            d['unit_id'] = d['spiketrain_id']
                        else:
                            seg0 = self.spiketrain_list['blocks'][0]['segments'][0]
                            d['unit_id'] = seg0['spiketrains'][st_idx]['unit_id']

                        st_idx += 1
                segment['t_start'] = min([s['t_start'] for s in segment['spiketrains']])
                segment['t_stop'] = max([s['t_stop'] for s in segment['spiketrains']])

        # check for consistent spiketrain channels across blocks and segments
        # For now assume that the order of spiketrain channels across segments is consistent

        seg0 = self.spiketrain_list['blocks'][0]['segments'][0]
        # use spiketrain id in first segment as unit id across segments
        for bl_idx, bl in enumerate(self.spiketrain_list['blocks']):
            for seg_idx, seg in enumerate(bl['segments']):
                assert len(seg['spiketrains']) == len(seg0['spiketrains'])
                for st_idx, st in enumerate(seg['spiketrains']):
                    assert st['unit_id'] == seg0['spiketrains'][st_idx]['unit_id']

        # create neo.rawio spike_channels
        spike_channels = []
        seg0 = self.spiketrain_list['blocks'][0]['segments'][0]
        for st in seg0['spiketrains']:
            unit_name = st['data'].metadata['neo_name']
            unit_id = st['unit_id']
            wf_left_sweep = 0
            wf_units = None
            wf_sampling_rate = 0
            if st['data'].features:
                wf = st['data'].features[0].data
                wf_units = wf.unit
                dim = wf.dimensions[-1]  # last wf dimension is time
                interval = dim.sampling_interval
                wf_sampling_rate = 1 / interval
                if wf.metadata:
                    wf_left_sweep = wf.metadata["left_sweep"]
            wf_gain = 1
            wf_offset = 0.
            spike_channels.append(
                (unit_name, unit_id, wf_units, wf_gain,
                 wf_offset, wf_left_sweep, wf_sampling_rate)
            )
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)


        event_channels = []
        event_count = 0
        epoch_count = 0
        for bl in self.file.blocks:
            seg_groups = [g for g in bl.groups if g.type == "neo.segment"]
            for seg in seg_groups:
                for mt in seg.multi_tags:
                    if mt.type == "neo.event":
                        ev_name = mt.metadata['neo_name']
                        ev_id = event_count
                        event_count += 1
                        ev_type = "event"
                        event_channels.append((ev_name, ev_id, ev_type))
                    if mt.type == "neo.epoch":
                        ep_name = mt.metadata['neo_name']
                        ep_id = epoch_count
                        epoch_count += 1
                        ep_type = "epoch"
                        event_channels.append((ep_name, ep_id, ep_type))
                break
            break
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.header = {}
        self.header['nb_block'] = len(self.file.blocks)
        self.header['nb_segment'] = [
            len(seg_groups)
            for bl in self.file.blocks
        ]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()
        for blk_idx, blk in enumerate(self.file.blocks):
            seg_groups = [g for g in blk.groups if g.type == "neo.segment"]
            bl_ann = self.raw_annotations['blocks'][blk_idx]
            props = blk.metadata.inherited_properties()
            bl_ann.update(self._filter_properties(props, "block"))
            for grp_idx, group in enumerate(seg_groups):
                seg_ann = bl_ann['segments'][grp_idx]
                props = group.metadata.inherited_properties()
                seg_ann.update(self._filter_properties(props, "segment"))

                sp_idx = 0
                ev_idx = 0
                for mt in group.multi_tags:
                    if mt.type == 'neo.spiketrain' and seg_ann['spikes']:
                        st_ann = seg_ann['spikes'][sp_idx]
                        props = mt.metadata.inherited_properties()
                        st_ann.update(self._filter_properties(props, 'spiketrain'))
                        sp_idx += 1
                    # if order is preserving, the annotations
                    # should go to the right place, need test
                    if mt.type == "neo.event" or mt.type == "neo.epoch":
                        neo_type = mt.type.replace('neo.', '')

                        # only add annotations when events exist
                        if seg_ann['events'] != []:
                            event_ann = seg_ann['events'][ev_idx]

                            # adding regular annotations
                            props = [p for p in mt.metadata.props
                                     if p.type != 'ARRAYANNOTATION']
                            props_dict = self._filter_properties(props, neo_type)
                            event_ann.update(props_dict)

                            # adding array_annotations
                            props = [p for p in mt.metadata.props
                                     if p.type == 'ARRAYANNOTATION']
                            props_dict = self._filter_properties(props, neo_type)
                            event_ann['__array_annotations__'].update(props_dict)

                            ev_idx += 1

                # adding array annotations to analogsignals
                annotated_anasigs = []
                sig_ann = seg_ann['signals']
                # this implementation relies on analogsignals always being
                # stored in the same stream order across segments
                stream_id = 0
                for da_idx, da in enumerate(group.data_arrays):
                    if da.type != "neo.analogsignal":
                        continue

                    if self._file_version < Version('0.11.0dev0'):
                        anasig_id = da.name.split('.')[-2]
                        # skip already annotated signals as each channel already
                        # contains the complete set of annotations and
                        # array_annotations
                        if anasig_id in annotated_anasigs:
                            continue
                    else:
                        anasig_id = da.name

                    annotated_anasigs.append(anasig_id)

                    # collect annotation properties
                    props = [p for p in da.metadata.props
                             if p.type != 'ARRAYANNOTATION']
                    props_dict = self._filter_properties(props, "analogsignal")
                    sig_ann[stream_id].update(props_dict)

                    # collect array annotation properties
                    props = [p for p in da.metadata.props
                             if p.type == 'ARRAYANNOTATION']
                    props_dict = self._filter_properties(props, "analogsignal")
                    sig_ann[stream_id]['__array_annotations__'].update(
                        props_dict)

                    stream_id += 1

    def _segment_t_start(self, block_index, seg_index):
        return min(self.neo_struct['blocks'][block_index]['segments'][seg_index]['t_start'],
                   self.spiketrain_list['blocks'][block_index]['segments'][seg_index]['t_start'])

    def _segment_t_stop(self, block_index, seg_index):
        return max(self.neo_struct['blocks'][block_index]['segments'][seg_index]['t_stop'],
                   self.spiketrain_list['blocks'][block_index]['segments'][seg_index]['t_stop'])

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_id = self.streams['stream_ids'][stream_index]
        for do in self.neo_struct['blocks'][block_index]['segments'][seg_index]['signals']:
            if do['stream_id'] == stream_id:
                return do['sample_count']

        raise ValueError(f'Could not find data object for block {block_index}, segment '
                         f'{seg_index} and stream {stream_id}.')

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        seg = self.neo_struct['blocks'][block_index]['segments'][seg_index]
        sig_t_start = seg['signals'][stream_index]['t_start']
        return sig_t_start  # assume same group_id always same t_start

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self.get_signal_size(block_index, seg_index, stream_index)

        segment = self.neo_struct['blocks'][block_index]['segments'][seg_index]
        if self._file_version < Version('0.11.0dev0'):
            das = segment['signals'][stream_index]['data']
            da = np.asarray(das).transpose()
        else:
            da = segment['signals'][stream_index]['data'][0]

        if channel_indexes is not None:
            mask = channel_indexes
        else:
            mask = slice(None, None)
        raw_signals = da[..., mask][i_start: i_stop]
        return raw_signals

    def _spike_count(self, block_index, seg_index, unit_index):
        # unit index == unit id
        seg = self.spiketrain_list['blocks'][block_index]['segments'][seg_index]
        st = seg['spiketrains'][unit_index]
        assert st['unit_id'] == self.header['spike_channels'][unit_index][1]
        return st['spike_count']

    def _get_spike_timestamps(self, block_index, seg_index, unit_index,
                              t_start, t_stop):
        block = self.spiketrain_list['blocks'][block_index]
        segment = block['segments'][seg_index]
        spike_dict = segment['spiketrains']
        spike_timestamps = np.array(spike_dict[unit_index]['data'].positions)  # dtype = float
        # spike_timestamps = np.transpose(spike_timestamps)

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
        seg = self.spiketrain_list['blocks'][block_index]['segments'][seg_index]
        waveforms = seg['spiketrains'][unit_index]['waveforms']
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
        event_count = 0
        segment = self.file.blocks[block_index].groups[seg_index]
        for event in segment.multi_tags:
            if event.type == 'neo.event' or event.type == 'neo.epoch':
                if event_count == event_channel_index:
                    return len(event.positions)
                else:
                    event_count += 1
        return event_count

    def _get_event_timestamps(self, block_index, seg_index,
                              event_channel_index, t_start, t_stop):
        timestamp = []
        labels = []
        durations = None
        if event_channel_index is None:
            raise IndexError
        for mt in self.file.blocks[block_index].groups[seg_index].multi_tags:
            if mt.type == "neo.event" or mt.type == "neo.epoch":
                labels.append(mt.positions.dimensions[0].labels)
                po = mt.positions
                if (po.type == "neo.event.times" or po.type == "neo.epoch.times"):
                    timestamp.append(po)
                channel = self.header['event_channels'][event_channel_index]
                if channel['type'] == b'epoch' and mt.extents:
                    if mt.extents.type == 'neo.epoch.durations':
                        durations = np.array(mt.extents)
                        break
        timestamp = timestamp[event_channel_index][:]
        timestamp = np.array(timestamp, dtype="float")
        labels = labels[event_channel_index][:]
        labels = np.array(labels, dtype='U')
        if t_start is not None:
            keep = timestamp >= t_start
            timestamp, labels = timestamp[keep], labels[keep]

        if t_stop is not None:
            keep = timestamp <= t_stop
            timestamp, labels = timestamp[keep], labels[keep]
        return timestamp, durations, labels  # only the first fits in rescale

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        ev_unit = ''
        for mt in self.file.blocks[0].groups[0].multi_tags:
            if mt.type == "neo.event":
                ev_unit = mt.positions.unit
                break
        if ev_unit == 'ms':
            event_timestamps /= 1000
        event_times = event_timestamps.astype(dtype)
        # supposing unit is second, other possibilities maybe mS microS...
        return event_times  # return in seconds

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        ep_unit = ''
        for mt in self.file.blocks[0].groups[0].multi_tags:
            if mt.type == "neo.epoch":
                ep_unit = mt.positions.unit
                break
        if ep_unit == 'ms':
            raw_duration /= 1000
        durations = raw_duration.astype(dtype)
        # supposing unit is second, other possibilities maybe mS microS...
        return durations  # return in seconds

    def _filter_properties(self, properties, neo_type):
        """
        Takes a collection of NIX metadata properties and the name of a Neo
        type and returns a dictionary representing the Neo object annotations.
        Properties that represent the attributes of the Neo object type are
        filtered, based on the global 'neo_attributes' dictionary.
        """
        annotations = dict()
        attrs = neo_attributes.get(neo_type, list())
        for prop in properties:
            # filter neo_name explicitly
            if not (prop.name in attrs or prop.name == "neo_name"):
                values = prop.values
                if len(values) == 1:
                    values = values[0]
                annotations[str(prop.name)] = values
        return annotations
