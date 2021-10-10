"""
ExampleRawIO is a class of a  fake example.
This is to be used when coding a new RawIO.


Rules for creating a new class:
  1. Step 1: Create the main class
    * Create a file in **neo/rawio/** that endith with "rawio.py"
    * Create the class that inherits from BaseRawIO
    * copy/paste all methods that need to be implemented.
    * code hard! The main difficulty is `_parse_header()`.
      In short you have a create a mandatory dict than
      contains channel informations::

            self.header = {}
            self.header['nb_block'] = 2
            self.header['nb_segment'] = [2, 3]
            self.header['signal_streams'] = signal_streams
            self.header['signal_channels'] = signal_channels
            self.header['spike_channels'] = spike_channels
            self.header['event_channels'] = event_channels

  2. Step 2: RawIO test:
    * create a file in neo/rawio/tests with the same name with "test_" prefix
    * copy paste neo/rawio/tests/test_examplerawio.py and do the same

  3. Step 3 : Create the neo.io class with the wrapper
    * Create a file in neo/io/ that ends with "io.py"
    * Create a class that inherits both your RawIO class and BaseFromRaw class
    * copy/paste from neo/io/exampleio.py

  4.Step 4 : IO test
    * create a file in neo/test/iotest with the same previous name with "test_" prefix
    * copy/paste from neo/test/iotest/test_exampleio.py
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype)

import numpy as np


class WaveSurferIO(BaseRawIO):
    """
    Class for "reading" fake data from an imaginary file.

    For the user, it gives access to raw data (signals, event, spikes) as they
    are in the (fake) file int16 and int64.

    For a developer, it is just an example showing guidelines for someone who wants
    to develop a new IO module.

    Two rules for developers:
      * Respect the :ref:`neo_rawio_API`
      * Follow the :ref:`io_guiline`

    This fake IO:
        * has 2 blocks
        * blocks have 2 and 3 segments
        * has  2 signals streams  of 8 channel each (sample_rate = 10000) so 16 channels in total
        * has 3 spike_channels
        * has 2 event channels: one has *type=event*, the other has
          *type=epoch*


    Usage:
        >>> import neo.rawio
        >>> r = neo.rawio.ExampleRawIO(filename='itisafake.nof')
        >>> r.parse_header()
        >>> print(r)
        >>> raw_chunk = r.get_analogsignal_chunk(block_index=0, seg_index=0,
                            i_start=0, i_stop=1024,  channel_names=channel_names)
        >>> float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, dtype='float64',
                            channel_indexes=[0, 3, 6])
        >>> spike_timestamp = reader.spike_timestamps(spike_channel_index=0,
                            t_start=None, t_stop=None)
        >>> spike_times = reader.rescale_spike_timestamp(spike_timestamp, 'float64')
        >>> ev_timestamps, _, ev_labels = reader.event_timestamps(event_channel_index=0)

    """
    extensions = ['fake']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        # note that this filename is ued in self._source_name
        self.filename = filename

    def _source_name(self):
        # this function is used by __repr__
        # for general cases self.filename is good
        # But for URL you could mask some part of the URL to keep
        # the main part.
        return self.filename

    def _parse_header(self):
        """
        talk about scaling
        """
        # TODO: add wavesurfer dependency check
        import sys
        sys.path.append(r"C:\fMRIData\git-repo\PyWaveSurfer")
        from pywavesurfer import ws

        pyws_data = ws.loadDataFile(self.filename, format_string="double")
        header = pyws_data["header"]

# Raw Data -------------------------------------------------------------------------------------------------------------------------------------------
        # TODO: find out if its worth importthing AO and digital channels
        self._raw_signals = {}
        self._t_starts = {}

        for seg_index in range(int(header["NSweepsPerRun"])):

            sweep_id = "sweep_{0:04d}".format(seg_index + 1)  # e.g. "sweep_0050"
            self._raw_signals[seg_index] = pyws_data[sweep_id]["analogScans"].T  # reshape to data x channel for Neo standard
            self._t_starts[seg_index] = np.float64(pyws_data[sweep_id]["timestamp"])  # TODO: find out why native this is double-nested list for a scalar (e.g. [[time]]

# Header ---------------------------------------------------------------------------------------------------------------------------------------------

        # Signal Channels
        # For now just grab the used AI channels
        signal_channels = []
        ai_channel_names = header["AIChannelNames"].astype(str)  # TODO: are channel units ever entered by the user or always in standard form?
        ai_channel_units = header["AIChannelUnits"].astype(str)
        self._sampling_rate = np.float64(pyws_data["header"]["AcquisitionSampleRate"])  # TODO: find out why native this is double-nested list for a scalar (e.g. [[SR]]

        for ch_idx, (ch_name, ch_units) in enumerate(zip(ai_channel_names,
                                                         ai_channel_units)):
            ch_id = ch_idx + 1
            dtype = "float64"  # as loaded with "double" argument from PyWaveSurfer
            gain = 1
            offset = 0
            stream_id = "0"  # chan_id  # TODO: dont understand this, for now treat all channels as the same. I think different units is fine, just not samplign rate

            signal_channels.append((ch_name, ch_id, self._sampling_rate, dtype, ch_units, gain, offset, stream_id))
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # No events TODO: I am not sure about this. Timestamps are in each segment (?)
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # FIND OUT: what is U64, dtype and ID.
        # Sampling rate is always unique. But units are different across channels. Presume this is okay based on axonrawio.
        signal_streams = np.array([('Signals', '0')], dtype=_signal_stream_dtype)  # TODO: maybe these are split at a later level?? Do not understand this, copied from AxonIO

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [int(header["NSweepsPerRun"])]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()  # TODO: return to this, # TODO: ADD ANNOTATIONS

    def _segment_t_start(self, block_index, seg_index):  # TODO: check these timestamps are definately time that starts # ASK
        return self._t_starts[seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._t_starts[seg_index] + \
                 self._raw_signals[seg_index].shape[0] / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        shape = self._raw_signals[seg_index].shape
        return shape[0]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):  # TODO: check several samplign rates are not supported in WaveSurfer
        return self._t_starts[seg_index]

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        raw_signals = self._raw_signals[seg_index][slice(i_start, i_stop), channel_indexes]
        return raw_signals
