"""
In development 16/10/2021

Class for reading data from WaveSurfer, a software written by
Boaz Mohar and Adam Taylor https://wavesurfer.janelia.org/

Requires the PyWaveSurfer module written by Boaz Mohar and Adam Taylor.

To Discuss:
- Wavesurfer also has analog output, and digital input / output channels, but here only supported analog input. Is this okay?
- I believe the signal streams field is configured correctly here, used AxonRawIO as a guide.
- each segment (sweep) has it's own timestamp, so I beleive no events_signals is correct (similar to winwcprawio not axonrawio)

1) Upload test files (kindly provided by Boaz Mohar and Adam Taylor) to g-node portal
2) write RawIO and IO tests

2. Step 2: RawIO test:
* create a file in neo/rawio/tests with the same name with "test_" prefix
* copy paste neo/rawio/tests/test_examplerawio.py and do the same

4.Step 4 : IO test
* create a file in neo/test/iotest with the same previous name with "test_" prefix
* copy/paste from neo/test/iotest/test_exampleio.py
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype)
import numpy as np

try:
    from pywavesurfer import ws
except ImportError as err:
    HAS_PYWAVESURFER = False
    PYWAVESURFER_ERR = err
else:
    HAS_PYWAVESURFER = True
    PYWAVESURFER_ERR = None

class WaveSurferRawIO(BaseRawIO):

    extensions = ['fake']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

        if not HAS_PYWAVESURFER:
            raise PYWAVESURFER_ERR

    def _source_name(self):
        return self.filename

    def _parse_header(self):

        pyws_data = ws.loadDataFile(self.filename, format_string="double")
        header = pyws_data["header"]

        # Raw Data
        self._raw_signals = {}
        self._t_starts = {}

        for seg_index in range(int(header["NSweepsPerRun"])):

            sweep_id = "sweep_{0:04d}".format(seg_index + 1)                     # e.g. "sweep_0050"
            self._raw_signals[seg_index] = pyws_data[sweep_id]["analogScans"].T  # reshape to data x channel for Neo standard
            self._t_starts[seg_index] = np.float64(pyws_data[sweep_id]["timestamp"])

        # Signal Channels
        signal_channels = []
        ai_channel_names = header["AIChannelNames"].astype(str)
        ai_channel_units = header["AIChannelUnits"].astype(str)
        self._sampling_rate = np.float64(pyws_data["header"]["AcquisitionSampleRate"])

        for ch_idx, (ch_name, ch_units) in enumerate(zip(ai_channel_names,
                                                         ai_channel_units)):
            ch_id = ch_idx + 1
            dtype = "float64"  # as loaded with "double" argument from PyWaveSurfer
            gain = 1
            offset = 0
            stream_id = "0"  # chan_id
            signal_channels.append((ch_name, ch_id, self._sampling_rate, dtype, ch_units, gain, offset, stream_id))

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # Spike Channels (no spikes)
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # Event Channels (no events)
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # Signal Streams
        signal_streams = np.array([('Signals', '0')], dtype=_signal_stream_dtype)

        # Header Dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [int(header["NSweepsPerRun"])]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()  # TODO: return to this and add annotations

    def _segment_t_start(self, block_index, seg_index):
        return self._t_starts[seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._t_starts[seg_index] + \
                 self._raw_signals[seg_index].shape[0] / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        shape = self._raw_signals[seg_index].shape
        return shape[0]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return self._t_starts[seg_index]

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        raw_signals = self._raw_signals[seg_index][slice(i_start, i_stop), channel_indexes]
        return raw_signals
