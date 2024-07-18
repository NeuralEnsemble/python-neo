from pathlib import Path
from packaging.version import Version
import warnings
import json

import numpy as np

from neo.core import NeoReadWriteError

from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)

class NeuronexusRawIO(BaseRawIO):

    extensions = ['xdat']
    rawmode = 'one-file'

    def __init__(self, filename: str | Path = ""):

        if not Path(filename).is_file():
            raise FileNotFoundError(f"The metadata file {filename} was not found")
        if not Path(filename).suffix != ".json":
            raise NeoReadWriteError("The json metadata should be given")
        
        self.filename = filename

    def _source_name(self):

        return self.filename
    
    def _parse_header(self):

        self.metadata = self._read_metadata(self.filename)
        self.sampling_frequency = self.metadata["status"]['samp_freq']

        n_samples, n_channels = self.metadata["status"]["shape"]
        # Stored as a simple float32 binary file
        dtype = "float32"

        filename = Path(self.filename)
        binary_file = filename.parent / filename.stem.split('.')[0] + '_data.xdat'
        timestamp_file = filename.parent / filename.stem.split('.')[0] + '_timestamp.xdat'

        self._raw_data = np.memmap(binary_file, dtype=dtype, mode='r', shape = (n_channels, n_samples), offset = 0).T

        self._timestamps = np.memmep(timestamp_file, dtype-dtype, mode='r', )




        signal_channels = []
        channel_info = self.metdata['sapiens_base']['biointerface_map']

        for channel_index, channel_name in channel_info['chan_name']:
            channel_id = channel_info['ntv_chan_name'][channel_index]
            sampling_rate = channel_info['samp_freq'][channel_index]
            if channel_info['chan_type'][channel_index] == "ai0":
                stream_id = 0
                units = 'uV'
            elif channel_info['chan_type'][channel_index][0] == 'd':
                units = 'a.u.'
                if channel_info['chan_type'][channel_index] == 'din0':
                    stream_id = 1
                else: 
                    stream_id = 2
            else:
                # aux channel
                units = 'V'
                stream_id = 3
            
            signal_channels.append(

                    channel_name,
                    channel_id,
                    sampling_rate,
                    "float32",
                    units,
                    1, #need to confirm
                    0, # need to confirm
                    stream_id,
            )
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        stream_ids = np.unique(signal_channels["stream_id"])
        signal_streams = np.zeros(stream_ids.size, dtype=_signal_stream_dtype)

        signal_streams["id"] = [str(stream_id) for stream_id in stream_ids]
        for stream_index, stream_id in enumerate(stream_ids):
            name = stream_id_to_stream_name.get(int(stream_id), "")
            signal_streams["name"][stream_index] = name

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [1]
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels
        

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
        pass


    def _read_metadata(self, fname_metadata):

        fname_metadata = Path(fname_metadata)

        with open(fname_metadata, 'rb') as read_file:
            metadata = json.load(read_file)
        
        return metadata



stream_id_to_stream_name = {0: "Amplifier Data",
                     1: "Digital-In",
                     2: "Digital-Out",
                     3: "Auxiliary"}