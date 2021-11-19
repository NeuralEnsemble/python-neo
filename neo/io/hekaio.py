"""
Class for reading data from WaveSurfer, a software written by
Boaz Mohar and Adam Taylor https://wavesurfer.janelia.org/

This is a wrapper around the PyWaveSurfer module written by Boaz Mohar and Adam Taylor,
using the "double" argument to load the data as 64-bit double.
"""
import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal
from baserawio import _signal_channel_dtype, _signal_stream_dtype, _spike_channel_dtype, _event_channel_dtype  # TODO: not sure about this  # from ..rawio.
# ..rawio.
try:
    from load_heka import LoadHeka  # TOOD: what is package called?
except ImportError as err:
    HAS_LOADHEKA = False
    LOADHEKA_ERR = err
else:
    HAS_LOADHEKA = True
    LOADHEKA_ERR = None


class HekaIO(BaseIO):
    """
    """
    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, AnalogSignal]
    readable_objects = [Block]
    writeable_objects = []

    has_header = True
    is_streameable = False

    read_params = {Block: []}
    write_params = None

    name = 'Heka'
    extensions = ['.dat']

    mode = 'file'

    def __init__(self, filename, group_idx, series_idx, use_recreated_stim_protocol):
        """
        Arguments:
            filename : a filename
        """
        if not HAS_LOADHEKA:
            raise LOADHEKA_ERR

        BaseIO.__init__(self)

        self.filename = filename
        self.heka = None
        self.header = {}
        self.group_idx = group_idx
        self.series_idx = series_idx
        self.num_sweeps = None
        self.use_recreated_stim_protocol = use_recreated_stim_protocol
        self.read_block()

    def read_block(self, lazy=False):
        assert not lazy, 'Do not support lazy'

        self.heka = LoadHeka(self.filename, only_load_header=True)
        self.num_sweeps = self.heka.get_num_sweeps_in_series(self.group_idx, self.series_idx)
        channels = self.heka.get_series_channels(self.group_idx, self.series_idx)
        self.fill_header(channels)

        # unit used as index because HEKA is always A or V, and will heka_reader will error is not TODO THIS IS VERY WEAK
        # TODO: this is very weak to load even if doesnt exist
        series_data = {"V": self.heka.get_series_data("Vm", self.group_idx, self.series_idx, include_stim_protocol=self.use_recreated_stim_protocol),  # TODO: no need to a
                       "A": self.heka.get_series_data("Im", self.group_idx, self.series_idx, include_stim_protocol=self.use_recreated_stim_protocol)}
        bl = Block()

        # iterate over sections first:
        for seg_index in range(self.num_sweeps):  # TODO: make sampling rate overall the same and set as attribute, do a final check here (and on units!? )

            seg = Segment(index=seg_index)

            # iterate over channels:
            for chan_idx, recsig in enumerate(channels):

                unit = recsig["unit"]
                name = recsig["name"]
                sampling_rate = 1 / recsig["sampling_step"] * 1 / pq.s  # todo: CALCULATED TWICE AS WELL IN HEADER TAKE THIS AND OTHER PARAMS FROM HEADER!?
                t_start = series_data[unit]["time"][seg_index, 0] * pq.s # TODO: add generate stimulus

                # TOOD: TEST WITH STIM
                if series_data[unit]["stim"] and series_data[unit]["stim"]["unit"] == series_data["data"]["unit"]:  # TODO: this will probably break when no chanel here!! dummy!!
                    recdata = series_data[unit]["stim"][seg_index, :]
                else:
                    recdata = series_data[unit]["data"][seg_index, :]
                signal = pq.Quantity(recdata, unit).T

                anaSig = AnalogSignal(signal, sampling_rate=sampling_rate,
                                      t_start=t_start, name=name,
                                      channel_index=chan_idx)
                seg.analogsignals.append(anaSig)
                bl.segments.append(seg)

        bl.create_many_to_one_relationship()

        return bl

    def fill_header(self, channels):

        signal_channels = []

        for ch_idx, chan in enumerate(channels):
            ch_id = ch_idx + 1
            ch_name = chan["name"]
            ch_units = chan["unit"]
            dtype = chan["dtype"]  # MOVE TO BOTTOM
            sampling_rate = 1 / chan["sampling_step"] * 1 / pq.s
            gain = 1
            offset = 0
            stream_id = "0"
            signal_channels.append((ch_name, ch_id, sampling_rate, dtype, ch_units, gain, offset, stream_id))

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
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [self.num_sweeps]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels
