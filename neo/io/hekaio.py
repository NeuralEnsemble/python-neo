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
    from load_heka.load_heka import LoadHeka  # TOOD: what is package called?
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

    def __init__(self, filename, group_idx, series_idx, load_recreated_stim_protocol=True):
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
        self.load_recreated_stim_protocol = load_recreated_stim_protocol
        self.read_block()

    def read_block(self, lazy=False):
        assert not lazy, 'Do not support lazy'

        self.heka = LoadHeka(self.filename, only_load_header=True)
        self.num_sweeps = self.heka.get_num_sweeps_in_series(self.group_idx, self.series_idx)
        channels = self.heka.get_series_channels(self.group_idx, self.series_idx)
        self.fill_header(channels)

        # TODO: this is very lazy to try and load both channels without knowing if they exist. Read off header to decide what to load
        series_data = {"V": self.heka.get_series_data("Vm", self.group_idx, self.series_idx, include_stim_protocol=self.load_recreated_stim_protocol),
                       "A": self.heka.get_series_data("Im", self.group_idx, self.series_idx, include_stim_protocol=self.load_recreated_stim_protocol)}
        bl = Block()

        # iterate over sections first:
        for seg_index in range(self.num_sweeps):

            seg = Segment(index=seg_index)

            # iterate over channels:
            for chan_idx, recsig in enumerate(channels):

                unit = self.header["signal_channels"]["units"][chan_idx]
                name = self.header["signal_channels"]["name"][chan_idx]
                sampling_rate = self.header["signal_channels"]["sampling_rate"][chan_idx] * 1 / pq.s
                t_start = series_data[unit]["time"][seg_index, 0] * pq.s

                recdata, name = self.get_chan_data_or_stim_data_if_does_not_exist(name, seg_index,
                                                                                  series_data, unit)

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
        self.check_channel_sampling_rate(channels)

        for ch_idx, chan in enumerate(channels):
            ch_id = ch_idx + 1
            ch_name = chan["name"]
            ch_units = chan["unit"]
            dtype = chan["dtype"]
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

    @staticmethod
    def get_chan_data_or_stim_data_if_does_not_exist(name, seg_index, series_data, unit):
        """
        Get stim data if second channel does not exist f4 is a good test of this! 
        """
        if series_data[unit]["data"] is None and \
                series_data[unit]["stim"]["data"] \
                and series_data[unit]["stim"]["units"] == unit:
            recdata = series_data[unit]["stim"]["data"][seg_index, :]
            name = "stim_" + name
        else:
            recdata = series_data[unit]["data"][seg_index, :]

        return recdata, name

    @staticmethod
    def check_channel_sampling_rate(channels):
        # this is already checked in load-heka-python but sanity checked here
        sampling_rate = []
        for chan in channels:
            sampling_rate.append(chan["sampling_step"])

        assert len(np.unique(sampling_rate)), "HEAK record sampling are not the same "
