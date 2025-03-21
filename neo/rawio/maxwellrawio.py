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

This implementation does not handle spikes at the moment.

For maxtwo device, each well will be a different signal stream.

Author : Samuel Garcia, Alessio Buccino, Pierre Yger
"""

import os
from pathlib import Path
import platform
import warnings
from urllib.request import urlopen

import numpy as np

from .baserawio import (
    BaseRawWithBufferApiIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)

from neo.core import NeoReadWriteError


class MaxwellRawIO(BaseRawWithBufferApiIO):
    """
    Class for reading MaxOne or MaxTwo files.

    Parameters
    ----------

    filename: str, default: ''
        The .h5 file to be loaded
    rec_name: str | None, default: None
        If the file has multiple recordings, specify the one to read.
        For 24-well plates, the rec_name needs to be specified since different well
        rows generate different recording ids.
        E.g., rec0001, rec0002, etc.

    """

    extensions = ["h5"]
    rawmode = "one-file"

    def __init__(self, filename="", rec_name=None):
        BaseRawWithBufferApiIO.__init__(self)
        self.filename = filename
        self.rec_name = rec_name

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        import h5py

        h5file = h5py.File(self.filename, mode="r")
        self.h5_file = h5file
        version = h5file["version"][0].decode()

        # create signal stream
        # one stream per well
        signal_streams = []
        if int(version) == 20160704:
            self._old_format = True
            signal_streams.append(("well000", "well000", "well000"))
        elif int(version) > 20160704:
            # multi stream stream (one well is one stream)
            self._old_format = False
            well_ids = list(h5file["wells"].keys())
            unique_rec_names = []
            for well_name in well_ids:
                rec_names = list(h5file["wells"][well_name].keys())
                for rec_name in rec_names:
                    unique_rec_names.append(rec_name)
            # check consistency of rec_names
            unique_rec_names = np.unique(unique_rec_names)
            if len(unique_rec_names) > 1:
                if self.rec_name is None:
                    raise ValueError(
                        f"Detected multiple recording IDs across wells. "
                        f"Please select a single recording using the `rec_name` parameter. "
                        f"Possible rec_names: {unique_rec_names}"
                    )
                else:
                    if self.rec_name not in unique_rec_names:
                        raise NeoReadWriteError(f"rec_name {self.rec_name} not found")
            else:
                self.rec_name = unique_rec_names[0]
            # add streams that contain the selected rec_name
            for well_name in well_ids:
                rec_names = list(h5file["wells"][well_name].keys())
                if self.rec_name in rec_names:
                    signal_streams.append((well_name, well_name, well_name))
        else:
            raise NotImplementedError(f"This version {version} is not supported")

        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        # one stream per buffer
        signal_buffers = np.zeros(signal_streams.size, dtype=_signal_buffer_dtype)
        signal_buffers["id"] = signal_streams["id"]
        signal_buffers["name"] = signal_streams["name"]

        # create signal channels
        max_sig_length = 0
        self._buffer_descriptions = {0: {0: {}}}
        self._stream_buffer_slice = {}
        sig_channels = []
        well_indices_to_remove = []
        for stream_index, stream_id in enumerate(signal_streams["id"]):

            if int(version) == 20160704:
                sr = 20000.0
                settings = h5file["settings"]
                if "lsb" in settings:
                    gain_uV = settings["lsb"][0] * 1e6
                else:
                    if "gain" not in settings:
                        print("'gain' amd 'lsb' not found in settings. " "Setting gain to 512 (default)")
                        gain = 512
                    else:
                        gain = settings["gain"][0]
                    gain_uV = 3.3 / (1024 * gain) * 1e6
                hdf5_path = "sig"
                mapping = h5file["mapping"]
                ids = np.array(mapping["channel"])
                ids = ids[ids >= 0]
                self._stream_buffer_slice[stream_id] = ids
            elif int(version) > 20160704:
                settings = h5file["wells"][stream_id][self.rec_name]["settings"]
                sr = settings["sampling"][0]
                if "lsb" in settings:
                    gain_uV = settings["lsb"][0] * 1e6
                else:
                    if "gain" not in settings:
                        print("'gain' amd 'lsb' not found in settings. " "Setting gain to 512 (default)")
                        gain = 512
                    else:
                        gain = settings["gain"][0]
                    gain_uV = 3.3 / (1024 * gain) * 1e6
                mapping = settings["mapping"]
                if "routed" in h5file["wells"][stream_id][self.rec_name]["groups"]:
                    hdf5_path = f"/wells/{stream_id}/{self.rec_name}/groups/routed/raw"
                else:
                    warnings.warn(f"No 'routed' group found for well {stream_id}")
                    well_indices_to_remove.append(stream_index)
                    continue

                self._stream_buffer_slice[stream_id] = None

            buffer_id = stream_id
            shape = h5file[hdf5_path].shape
            self._buffer_descriptions[0][0][buffer_id] = {
                "type": "hdf5",
                "file_path": str(self.filename),
                "hdf5_path": hdf5_path,
                "shape": shape,
                "time_axis": 1,
            }
            self._stream_buffer_slice[stream_id] = slice(None)

            channel_ids = np.array(mapping["channel"])
            electrode_ids = np.array(mapping["electrode"])
            mask = channel_ids >= 0
            channel_ids = channel_ids[mask]
            electrode_ids = electrode_ids[mask]

            for i, chan_id in enumerate(channel_ids):
                elec_id = electrode_ids[i]
                ch_name = f"ch{chan_id} elec{elec_id}"
                offset_uV = 0
                buffer_id = stream_id
                sig_channels.append(
                    (ch_name, str(chan_id), sr, "uint16", "uV", gain_uV, offset_uV, stream_id, buffer_id)
                )

            max_sig_length = max(max_sig_length, shape[1])

        self._t_stop = max_sig_length / sr

        if len(well_indices_to_remove) > 0:
            signal_streams = np.delete(signal_streams, np.array(well_indices_to_remove))

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [1]
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = sig_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        self._generate_minimal_annotations()
        bl_ann = self.raw_annotations["blocks"][0]
        bl_ann["maxwell_version"] = version

    def _segment_t_start(self, block_index, seg_index):
        return 0.0

    def _segment_t_stop(self, block_index, seg_index):
        return self._t_stop

    def _get_analogsignal_buffer_description(self, block_index, seg_index, buffer_id):
        return self._buffer_descriptions[block_index][seg_index][buffer_id]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.0

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
        try:
            return super()._get_analogsignal_chunk(
                block_index, seg_index, i_start, i_stop, stream_index, channel_indexes
            )
        except OSError as e:
            print("*" * 10)
            print(_hdf_maxwell_error)
            print("*" * 10)
            raise (e)


_hdf_maxwell_error = """Maxwell file format is based on HDF5.
The internal compression requires a custom plugin!!!
This is a big pain for the end user.
You, as a end user, should ask Maxwell company to change this.
Please visit this page and install the missing decompression libraries:
https://share.mxwbio.com/d/4742248b2e674a85be97/
Then, link the decompression library by setting the `HDF5_PLUGIN_PATH` to your
installation location, e.g. via
os.environ['HDF5_PLUGIN_PATH'] = '/path/to/custom/hdf5/plugin/'

Alternatively, you can use the auto_install_maxwell_hdf5_compression_plugin() below
function that do it automagically.
"""


def auto_install_maxwell_hdf5_compression_plugin(hdf5_plugin_path=None, force_download=True):
    if hdf5_plugin_path is None:
        hdf5_plugin_path = os.getenv("HDF5_PLUGIN_PATH", None)
        if hdf5_plugin_path is None:
            hdf5_plugin_path = Path.home() / "hdf5_plugin_path_maxwell"
            os.environ["HDF5_PLUGIN_PATH"] = str(hdf5_plugin_path)
    hdf5_plugin_path = Path(hdf5_plugin_path)
    hdf5_plugin_path.mkdir(exist_ok=True)

    if platform.system() == "Linux":
        remote_lib = "https://share.mxwbio.com/d/4742248b2e674a85be97/files/?p=%2FLinux%2Flibcompression.so&dl=1"
        local_lib = hdf5_plugin_path / "libcompression.so"
    elif platform.system() == "Darwin":
        remote_lib = "https://share.mxwbio.com/d/4742248b2e674a85be97/files/?p=%2FMacOS%2Flibcompression.dylib&dl=1"
        local_lib = hdf5_plugin_path / "libcompression.dylib"
    elif platform.system() == "Windows":
        remote_lib = "https://share.mxwbio.com/d/4742248b2e674a85be97/files/?p=%2FWindows%2Fcompression.dll&dl=1"
        local_lib = hdf5_plugin_path / "compression.dll"

    if not force_download and local_lib.is_file():
        print(f"The h5 compression library for Maxwell is already located in {local_lib}!")
        return

    dist = urlopen(remote_lib)
    with open(local_lib, "wb") as f:
        f.write(dist.read())
