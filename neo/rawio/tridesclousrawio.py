"""


"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np

from pathlib import Path


class TridesclousRawIO(BaseRawIO):
    """

    """
    extensions = []
    rawmode = 'one-dir'

    def __init__(self, dirname='', chan_grp=None):
        BaseRawIO.__init__(self)
        self.dirname = dirname
        self.chan_grp = chan_grp

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        try:
            import tridesclous as tdc
        except ImportError:
            print('tridesclous is not installed')

        tdc_folder = Path(self.dirname)

        tdc_dataio = tdc.DataIO(str(self.dirname))
        chan_grp = self.chan_grp
        if chan_grp is None:
            # if chan_grp is not provided, take the first one if unique
            chan_grps = list(tdc_dataio.channel_groups.keys())
            assert len(chan_grps) == 1, 'There are several groups in the folder, specify chan_grp=...'
            chan_grp = chan_grps[0]

        self._sampling_rate = float(tdc_dataio.sample_rate)
        catalogue = tdc_dataio.load_catalogue(name='initial', chan_grp=chan_grp)

        labels = catalogue['clusters']['cluster_label']
        labels = labels[labels >= 0]
        self.unit_labels = labels

        nb_segment = tdc_dataio.nb_segment

        self._all_spikes = []
        for seg_index in range(nb_segment):
            self._all_spikes.append(tdc_dataio.get_spikes(seg_num=seg_index,
                        chan_grp=chan_grp, i_start=None, i_stop=None).copy())

        self._sampling_rate = tdc_dataio.sample_rate
        sr = self._sampling_rate

        self._t_starts = [0.] * nb_segment
        self._t_stops = [tdc_dataio.segment_shapes[s][0]/sr
                                                for s in range(nb_segment)]

        sig_channels = []
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        unit_channels = []
        for unit_index, unit_label in enumerate(labels):
            unit_name = f'unit{unit_index} #{unit_label}'
            unit_id = f'{unit_label}'
            wf_units = ''
            wf_gain = 0
            wf_offset = 0.
            wf_left_sweep = 0
            wf_sampling_rate = 0
            unit_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                  wf_offset, wf_left_sweep, wf_sampling_rate))
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)

        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # fille into header dict
        # This is mandatory!!!!!
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [nb_segment]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        assert block_index == 0
        return self._t_starts[seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        assert block_index == 0
        return self._t_stops[seg_index]

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        return None

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return None

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        return None

    def _spike_count(self, block_index, seg_index, unit_index):
        assert block_index == 0
        spikes = self._all_spikes[seg_index]
        unit_label = self.unit_labels[unit_index]
        mask = spikes['cluster_label'] == unit_label
        nb_spikes = np.sum(mask)
        return nb_spikes

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        assert block_index == 0
        unit_label = self.unit_labels[unit_index]
        spikes = self._all_spikes[seg_index]
        mask = spikes['cluster_label'] == unit_label
        spike_timestamps = spikes['index'][mask].copy()

        if t_start is not None:
            start_frame = int(t_start * self._sampling_rate)
            spike_timestamps = spike_timestamps[spike_timestamps >= start_frame]
        if t_stop is not None:
            end_frame = int(t_stop * self._sampling_rate)
            spike_timestamps = spike_timestamps[spike_timestamps < end_frame]

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self._sampling_rate
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        return None

    def _event_count(self, block_index, seg_index, event_channel_index):
        return None

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        return None

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        return None

    def _rescale_epoch_duration(self, raw_duration, dtype):
        return None
