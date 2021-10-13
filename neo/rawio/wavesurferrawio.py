"""

  2. Step 2: RawIO test:
    * create a file in neo/rawio/tests with the same name with "test_" prefix ####################### TODO (email people)
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
    """
    Two rules for developers:
      * Respect the :ref:`neo_rawio_API`
      * Follow the :ref:`io_guiline`
    """
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
        """
        talk about scaling

        WAVESURFER

            1) ask about the files at do not work in the test
            2) ask if it is okay to upload the test files (e.g. test2) to the website
            3) ask if single sampling rate only possible
            4) # TODO: are channel units ever entered by the user or always in standard form?
            5 TODO: check these timestamps are definately time that starts # ASK
            6) # TODO: find out why native this is double-nested list for a scalar (e.g. [[time]] (dont ask)

        NEO
        1) document this well and IO
        2) ask about and upload to tests, push to repo
        3) ask if required to handle AI, DI and DO
        4) sampling streams: # TODO: dont understand this, for now treat all channels as the same. I think different units is fine, just not samplign rate. # TODO: maybe these are split at a later level?? Do not understand this, copied from AxonIO         # Sampling rate is always unique. But units are different across channels. Presume this is okay based on axonrawio.
        5) double check events channel (No events TODO: I am not sure about this. Timestamps are in each segment (?))

        """
        import sys
        sys.path.append(r"C:\fMRIData\git-repo\PyWaveSurfer")

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

        self._generate_minimal_annotations()  # TODO: return to this, # TODO: ADD ANNOTATIONS

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
