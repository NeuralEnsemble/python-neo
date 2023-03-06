"""
Class for reading data from maxwell biosystem device:
  * MaxOne
  * MaxTwo

https://www.mxwbio.com/resources/mea/

The implementation is a mix between:
  * the implementation in spikeextractors
     https://github.com/SpikeInterface/spikeextractors/blob/master/spikeextractors/extractors/maxwellextractors/maxwellextractors.py
 * the implementation in spyking-circus
    https://github.com/spyking-circus/spyking-circus/blob/master/circus/files/maxwell.py

The implementation do not handle spike at the moment.

For maxtwo device, each well will be a different signal stream.

Author : Samuel Garcia, Alessio Buccino, Pierre Yger
"""
import os
from pathlib import Path
import platform
from urllib.request import urlopen

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype)

import numpy as np


class MaxwellRawIO(BaseRawIO):
    """
    Class for reading MaxOne or MaxTwo files.
    """
    extensions = ['h5']
    rawmode = 'one-file'

    def __init__(self, filename='', rec_name=None):
        BaseRawIO.__init__(self)
        self.filename = filename
        self.rec_name = rec_name

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        import h5py

        h5 = h5py.File(self.filename, mode='r')
        self.h5_file = h5
        version = h5['version'][0].decode()

        # create signal stream
        # one stream per well
        signal_streams = []
        if int(version) == 20160704:
            self._old_format = True
            signal_streams.append(('well000', 'well000'))
        elif int(version) > 20160704:
            # multi stream stream (one well is one stream)
            self._old_format = False
            stream_ids = list(h5['wells'].keys())
            for stream_id in stream_ids:
                rec_names = list(h5['wells'][stream_id].keys())
                if len(rec_names) > 1:
                    if self.rec_name is None:
                        raise ValueError("Detected multiple recordings. Please select a "
                                         "single recording using the `rec_name` parameter. "
                                         f"Possible rec_name {rec_names}")
                else:
                    self.rec_name = rec_names[0]
                signal_streams.append((stream_id, stream_id))
        else:
            raise NotImplementedError(
                f'This version {version} is not supported')
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        # create signal channels
        max_sig_length = 0
        self._signals = {}
        sig_channels = []
        for stream_id in signal_streams['id']:
            if int(version) == 20160704:
                sr = 20000.
                settings = h5["settings"]
                if 'lsb' in settings:
                    gain_uV = settings['lsb'][0] * 1e6
                else:
                    if "gain" not in settings:
                        print("'gain' amd 'lsb' not found in settings. "
                              "Setting gain to 512 (default)")
                        gain = 512
                    else:
                        gain = settings['gain'][0]
                    gain_uV = 3.3 / (1024 * gain) * 1e6
                sigs = h5['sig']
                mapping = h5["mapping"]
                ids = np.array(mapping['channel'])
                ids = ids[ids >= 0]
                self._channel_slice = ids
            elif int(version) > 20160704:
                settings = h5['wells'][stream_id][self.rec_name]['settings']
                sr = settings['sampling'][0]
                if 'lsb' in settings:
                    gain_uV = settings['lsb'][0] * 1e6
                else:
                    if "gain" not in settings:
                        print("'gain' amd 'lsb' not found in settings. "
                              "Setting gain to 512 (default)")
                        gain = 512
                    else:
                        gain = settings['gain'][0]
                    gain_uV = 3.3 / (1024 * gain) * 1e6
                mapping = settings['mapping']
                sigs = h5['wells'][stream_id][self.rec_name]['groups']['routed']['raw']

            channel_ids = np.array(mapping['channel'])
            electrode_ids = np.array(mapping['electrode'])
            mask = channel_ids >= 0
            channel_ids = channel_ids[mask]
            electrode_ids = electrode_ids[mask]

            for i, chan_id in enumerate(channel_ids):
                elec_id = electrode_ids[i]
                ch_name = f'ch{chan_id} elec{elec_id}'
                offset_uV = 0
                sig_channels.append((ch_name, str(chan_id), sr, 'uint16', 'uV',
                                     gain_uV, offset_uV, stream_id))

            self._signals[stream_id] = sigs
            max_sig_length = max(max_sig_length, sigs.shape[1])

        self._t_stop = max_sig_length / sr

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = sig_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()
        bl_ann = self.raw_annotations['blocks'][0]
        bl_ann['maxwell_version'] = version

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        return self._t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_id = self.header['signal_streams'][stream_index]['id']
        sigs = self._signals[stream_id]
        return sigs.shape[1]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        stream_id = self.header['signal_streams'][stream_index]['id']
        sigs = self._signals[stream_id]

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = sigs.shape[1]

        resorted_indexes = None
        if channel_indexes is None:
            channel_indexes = slice(None)
        else:
            if np.array(channel_indexes).size > 1 and np.any(np.diff(channel_indexes) < 0):
                # get around h5py constraint that it does not allow datasets
                # to be indexed out of order
                sorted_channel_indexes = np.sort(channel_indexes)
                resorted_indexes = np.array(
                    [list(channel_indexes).index(ch) for ch in sorted_channel_indexes])

        try:
            if resorted_indexes is None:
                if self._old_format:
                    sigs = sigs[self._channel_slice, i_start:i_stop]
                    sigs = sigs[channel_indexes]
                else:
                    sigs = sigs[channel_indexes, i_start:i_stop]
            else:
                if self._old_format:
                    sigs = sigs[self._channel_slice, i_start:i_stop]
                    sigs = sigs[sorted_channel_indexes]
                else:
                    sigs = sigs[sorted_channel_indexes, i_start:i_stop]
                sigs = sigs[resorted_indexes]
        except OSError as e:
            print('*' * 10)
            print(_hdf_maxwell_error)
            print('*' * 10)
            raise(e)
        sigs = sigs.T

        return sigs


_hdf_maxwell_error = """Maxwell file format is based on HDF5.
The internal compression requires a custom plugin!!!
This is a big pain for the end user.
You, as a end user, should ask Maxwell company to change this.
Please visit this page and install the missing decompression libraries:
https://share.mxwbio.com/d/4742248b2e674a85be97/
Then, link the decompression library by setting the `HDF5_PLUGIN_PATH` to your
installation location, e.g. via
os.environ['HDF5_PLUGIN_PATH'] = '/path/to/cutum/hdf5/plugin/'

Alternatively, you can use the auto_install_maxwell_hdf5_compression_plugin() below
function that do it automagically.
"""


def auto_install_maxwell_hdf5_compression_plugin(hdf5_plugin_path=None, force_download=True):
    if hdf5_plugin_path is None:
        hdf5_plugin_path = os.getenv('HDF5_PLUGIN_PATH', None)
        if hdf5_plugin_path is None:
            hdf5_plugin_path = Path.home() / 'hdf5_plugin_path_maxwell'
            os.environ['HDF5_PLUGIN_PATH'] = str(hdf5_plugin_path)
    hdf5_plugin_path = Path(hdf5_plugin_path)
    hdf5_plugin_path.mkdir(exist_ok=True)

    if platform.system() == 'Linux':
        remote_lib = 'https://share.mxwbio.com/d/4742248b2e674a85be97/files/?p=%2FLinux%2Flibcompression.so&dl=1'
        local_lib = hdf5_plugin_path / 'libcompression.so'
    elif platform.system() == 'Darwin':
        remote_lib = 'https://share.mxwbio.com/d/4742248b2e674a85be97/files/?p=%2FMacOS%2Flibcompression.dylib&dl=1'
        local_lib = hdf5_plugin_path / 'libcompression.dylib'
    elif platform.system() == 'Windows':
        remote_lib = 'https://share.mxwbio.com/d/4742248b2e674a85be97/files/?p=%2FWindows%2Fcompression.dll&dl=1'
        local_lib = hdf5_plugin_path / 'compression.dll'

    if not force_download and local_lib.is_file():
        print(f'lib h5 compression for maxwell already already in {local_lib}')
        return

    dist = urlopen(remote_lib)
    with open(local_lib, 'wb') as f:
        f.write(dist.read())
