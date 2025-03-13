"""
NeuroNexus has their own file format based on their Allego Recording System
https://www.neuronexus.com/webinar/allego-software-updates/

The format involves 3 files:
  * The *.xdat.json metadata file
  * The *_data.xdat binary file of all raw data
  * The *_timestamps.xdat binary file of the timestamp data

Based on sample data is appears that the binary file is always a float32 format
Other information can be found within the metadata json file


The metadata file has a pretty complicated structure as far as I can tell
a lot of which is dedicated to probe information, which won't be handle at the
the Neo level.

It appears that the metadata['status'] provides most of the information necessary
for generating the initial memory map (global sampling frequency), n_channels,
n_samples.

metadata['sapiens_base']['biointerface_map'] provides all the channel specific information
like channel_names, channel_ids, channel_types.

An additional note on channels. It appears that analog channels are called `pri` or
`ai0` within the metadata whereas digital channels are called `din0` or `dout0`.
In this first implementation it is up to the user to do the appropriate channel slice
to only get the data they want. This is a buffer-based approach that Sam likes.
Eventually we will try to divide these channels into streams (analog vs digital) or
we can come up with a work around if users open an issue requesting this.

Zach McKenzie

"""

from __future__ import annotations
from pathlib import Path
import json
import datetime
import sys
import re

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


class NeuroNexusRawIO(BaseRawWithBufferApiIO):

    extensions = ["xdat", "json"]
    rawmode = "one-file"

    def __init__(self, filename: str | Path = ""):
        """
        The Allego NeuroNexus reader for the `xdat` file format

        Parameters
        ----------
        filename: str | Path, default: ''
            The filename of the metadata file should end in .xdat.json

        Notes
        -----
        * The format involves 3 files:
            * The *.xdat.json metadata file
            * The *_data.xdat binary file of all raw data
            * The *_timestamps.xdat binary file of the timestamp data

        From the metadata the other two files are located within the same directory
        and loaded.

        * The metadata is stored as the metadata attribute for individuals hoping
        to extract probe information, but Neo itself does not load any of the probe
        information

        Examples
        --------
        >>> from neo.rawio import NeuronexusRawIO
        >>> reader = NeuronexusRawIO(filename='abc.xdat.json')
        >>> reader.parse_header()
        >>> raw_chunk = reader.get_analogsignal_chunk(block_index=0
                seg_index=0,
                stream_index=0)
        # this isn't necessary for this reader since data is stored as float uV, but
        # this is included in case there is a future change to the format
        >>> float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, stream_index=0)

        """

        BaseRawWithBufferApiIO.__init__(self)

        if not Path(filename).is_file():
            raise FileNotFoundError(f"The metadata file {filename} was not found")
        if Path(filename).suffix != ".json":
            raise NeoReadWriteError(
                f"The json metadata file should be given, filename entered is {Path(filename).stem}"
            )
        meta_filename = Path(filename)
        binary_file = meta_filename.parent / (meta_filename.stem.split(".")[0] + "_data.xdat")

        if not binary_file.exists() and not binary_file.is_file():
            raise FileNotFoundError(f"The data.xdat file {binary_file} was not found. Is it in the same directory?")
        timestamp_file = meta_filename.parent / (meta_filename.stem.split(".")[0] + "_timestamp.xdat")
        if not timestamp_file.exists() and not timestamp_file.is_file():
            raise FileNotFoundError(
                f"The timestamps.xdat file {timestamp_file} was not found. Is it in the same directory?"
            )

        self.filename = filename
        self.binary_file = binary_file
        self.timestamp_file = timestamp_file

    def _source_name(self):
        # return the metadata filename only
        return self.filename

    def _parse_header(self):

        # read metadata
        self.metadata = self.read_metadata(self.filename)

        # Collect information necessary for memory map
        self._sampling_frequency = self.metadata["status"]["samp_freq"]
        self._n_samples, self._n_channels = self.metadata["status"]["shape"]
        # Stored as a simple float32 binary file
        BINARY_DTYPE = "float32"
        binary_file = self.binary_file
        timestamp_file = self.timestamp_file

        # the will cretae a memory map with teh generic mechanism
        buffer_id = "0"
        self._buffer_descriptions = {0: {0: {}}}
        self._buffer_descriptions[0][0][buffer_id] = {
            "type": "raw",
            "file_path": str(binary_file),
            "dtype": BINARY_DTYPE,
            "order": "C",
            "file_offset": 0,
            "shape": (self._n_samples, self._n_channels),
        }
        # Make the memory map for timestamp
        self._timestamps = np.memmap(
            timestamp_file,
            dtype=np.int64,  # this is from the allego sample reader timestamps are np.int64
            mode="r",
            offset=0,  # headerless binary file
        )

        # We can do a quick timestamp check to make sure it is the correct timestamp data for the
        # given metadata
        if self._timestamps[0] != self.metadata["status"]["timestamp_range"][0]:
            metadata_start = self.metadata["status"]["timestamp_range"][0]
            data_start = self._teimstamps[0]
            raise NeoReadWriteError(
                f"The metadata indicates a different starting timestamp {metadata_start} than the data starting timestamp {data_start}"
            )

        # organize the channels
        signal_channels = []
        channel_info = self.metadata["sapiens_base"]["biointerface_map"]

        # as per dicussion with the Neo/SpikeInterface teams stream_id will become buffer_id
        # and because all data is stored in the same buffer stream for the moment all channels
        # will be in stream_id = 0. In the future this will be split into sub_streams based on
        # type but for now it will be the end-users responsability for this.
        stream_id = "0"  # hard-coded see note above
        buffer_id = "0"
        for channel_index, channel_name in enumerate(channel_info["chan_name"]):
            channel_id = channel_info["ntv_chan_name"][channel_index]
            # 'ai0' indicates analog data which is stored as microvolts
            if channel_info["chan_type"][channel_index] == "ai0":
                units = "uV"
            # 'd' means digital. Per discussion with neuroconv users the use of
            # 'a.u.' makes the units clearer
            elif channel_info["chan_type"][channel_index][0] == "d":
                units = "a.u."
            # aux channel
            else:
                units = "V"

            signal_channels.append(
                (
                    channel_name,
                    channel_id,
                    self._sampling_frequency,
                    BINARY_DTYPE,
                    units,
                    1,  # no gain
                    0,  # no offset
                    stream_id,
                    buffer_id,
                )
            )

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        buffer_id = "0"
        signal_buffers = np.array([("", buffer_id)], dtype=_signal_buffer_dtype)

        stream_ids = np.unique(signal_channels["stream_id"])
        signal_streams = np.zeros(stream_ids.size, dtype=_signal_stream_dtype)
        signal_streams["id"] = [str(stream_id) for stream_id in stream_ids]
        # One unique buffer
        signal_streams["buffer_id"] = buffer_id
        self._stream_buffer_slice = {}
        for stream_index, stream_id in enumerate(stream_ids):
            name = stream_id_to_stream_name.get(int(stream_id), "")
            signal_streams["name"][stream_index] = name
            chan_inds = np.flatnonzero(signal_channels["stream_id"] == stream_id)
            self._stream_buffer_slice[stream_id] = chan_inds

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # Put all the necessary info in the header
        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [1]
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        # Add the minimum annotations
        self._generate_minimal_annotations()

        # date comes out as: '2024-07-01T13:04:49.4972245-04:00' so in ISO format
        datetime_string = self.metadata["status"]["start_time"]

        # Python 3.10 and older expect iso format to only have 3 or 6 decimal places
        if sys.version_info.minor < 11:
            datetime_string = re.sub(r"(\.\d{6})\d+", r"\1", datetime_string)

        rec_datetime = datetime.datetime.fromisoformat(datetime_string)

        bl_annotations = self.raw_annotations["blocks"][0]
        seg_annotations = bl_annotations["segments"][0]
        for d in (bl_annotations, seg_annotations):
            d["rec_datetime"] = rec_datetime

    def _segment_t_stop(self, block_index, seg_index):

        t_stop = self.metadata["status"]["t_range"][1]
        return t_stop

    def _segment_t_start(self, block_index, seg_index):

        t_start = self.metadata["status"]["t_range"][0]
        return t_start

    def _get_signal_t_start(self, block_index, seg_index, stream_index):

        t_start = self.metadata["status"]["t_range"][0]
        return t_start

    #######################################
    # Helper Functions

    def read_metadata(self, fname_metadata):
        """
        Metadata is just a heavily nested json file

        Parameters
        ----------
        fname_metada: str | Path
            The *.xdat.json file for the current recording

        Returns
        -------
        metadata: dict
            Returns the metadata as a dictionary"""

        fname_metadata = Path(fname_metadata)
        with open(fname_metadata, "rb") as read_file:
            metadata = json.load(read_file)

        return metadata

    def _get_analogsignal_buffer_description(self, block_index, seg_index, buffer_id):
        return self._buffer_descriptions[block_index][seg_index][buffer_id]


# this is pretty useless right now, but I think after a
# refactor with sub streams we could adapt this for the sub-streams
# so let's leave this here for now :)
stream_id_to_stream_name = {"0": "Neuronexus Allego Data"}
