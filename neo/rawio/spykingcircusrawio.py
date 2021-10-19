"""


"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _spike_channel_dtype,
                        _event_channel_dtype)

import numpy as np

from pathlib import Path

try:
    import h5py
    HAVE_HDF5 = True
except ImportError:
    HAVE_HDF5 = False


def _load_sample_rate(params_file):
    sample_rate = None
    with params_file.open('r') as f:
        for r in f.readlines():
            if 'sampling_rate' in r:
                sample_rate = r.split('=')[-1]
                if '#' in sample_rate:
                    sample_rate = sample_rate[:sample_rate.find('#')]
                sample_rate = float(sample_rate)
    return sample_rate


class SpykingCircusRawIO(BaseRawIO):
    """
    RawIO reader to load results that have been obtained via SpyKING CIRCUS
    http://spyking-circus.rtfd.org

    You simply need to specify the output folder created by SpyKING CIRCUS where
    the results have been stored.
    """
    extensions = []
    rawmode = 'one-folder'

    def __init__(self, dirname=''):
        BaseRawIO.__init__(self)
        self.dirname = dirname

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        spykingcircus_folder = Path(self.dirname)
        listfiles = spykingcircus_folder.iterdir()
        results = None
        sample_rate = None

        parent_folder = None
        result_folder = None
        for f in listfiles:
            if f.is_dir():
                if any([f_.suffix == '.hdf5' for f_ in f.iterdir()]):
                    parent_folder = spykingcircus_folder
                    result_folder = f

        if parent_folder is None:
            parent_folder = spykingcircus_folder.parent
            for f in parent_folder.iterdir():
                if f.is_dir():
                    if any([f_.suffix == '.hdf5' for f_ in f.iterdir()]):
                        result_folder = spykingcircus_folder

        assert isinstance(parent_folder, Path) and \
            isinstance(result_folder, Path), "Not a valid spyking circus folder"

        # load files
        for f in result_folder.iterdir():
            if 'result.hdf5' in str(f):
                results = f
            if 'result-merged.hdf5' in str(f):
                results = f
                break

        # load params
        for f in parent_folder.iterdir():
            if f.suffix == '.params':
                sample_rate = _load_sample_rate(f)
            else:
                raise Exception('Can not find the .params file')

        if sample_rate is not None:
            self._sampling_frequency = sample_rate

        if results is None:
            raise Exception(spykingcircus_folder, " is not a spyking circus folder")
        f_results = h5py.File(results, 'r')

        self._all_spikes = []
        for temp in f_results['spiketimes'].keys():
            self._all_spikes.append(np.array(f_results['spiketimes'][temp]).astype('int64'))

        self._kwargs = {'folder_path': str(Path(spykingcircus_folder).absolute())}

        sig_channels = []
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        unit_channels = []
        for unit_index in range(len(self._all_spikes)):
            unit_name = f'unit{unit_index} #{unit_index}'
            unit_id = f'{unit_index}'
            wf_units = ''
            wf_gain = 0
            wf_offset = 0.
            wf_left_sweep = 0
            wf_sampling_rate = 0
            unit_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                  wf_offset, wf_left_sweep, wf_sampling_rate))
        unit_channels = np.array(unit_channels, dtype=_spike_channel_dtype)

        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        self._duration = f_results['info']['duration'][0]
        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        return self._duration

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        return None

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return None

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        return None

    def _spike_count(self, block_index, seg_index, unit_index):
        nb_spikes = len(self._all_spikes[unit_index])
        return nb_spikes

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        assert block_index == 0
        assert seg_index == 0

        spike_timestamps = self._all_spikes[unit_index].copy()

        if t_start is not None:
            start_frame = int(t_start * self._sampling_rate)
            spike_timestamps = spike_timestamps[spike_timestamps >= start_frame]
        if t_stop is not None:
            end_frame = int(t_stop * self._sampling_rate)
            spike_timestamps = spike_timestamps[spike_timestamps < end_frame]

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        # must rescale to second a particular spike_timestamps
        # with a fixed dtype so the user can choose the precisino he want.
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self._sampling_frequency  # because 10kHz
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
