"""
Quick way to load Hekafiles into Neo format using load-heka-python module.

TODO: implement at rawio level for inclusion in Neo
"""
import numpy as np
import quantities as pq
import copy

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal
from ..rawio.baserawio import _signal_channel_dtype, _signal_stream_dtype, _spike_channel_dtype, _event_channel_dtype

try:
    from load_heka.load_heka import LoadHeka
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

    def __init__(self, filename, group_idx, series_idx):
        """
        Assumes HEKA file units is A and V for data and stimulus. This is enforced in LoadHeka level.
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
        self.series_data = None

    def read_block(self, lazy=False, force_order_to_recording_mode=False):
        assert not lazy, 'Do not support lazy'

        self.heka = LoadHeka(self.filename, only_load_header=True)

        self.series_data = {"V": self.heka.get_series_data("Vm", self.group_idx, self.series_idx, include_stim_protocol=True),  # try and load both, even if doesn't exist
                            "A": self.heka.get_series_data("Im", self.group_idx, self.series_idx, include_stim_protocol=True)}
        bl = Block()

        self.channels = self.heka.get_series_channels(self.group_idx, self.series_idx)
        self.recording_mode = self.channels[0]["recording_mode"]

        self.num_sweeps = self.heka.get_num_sweeps_in_series(self.group_idx, self.series_idx)
        self.fill_header()

        if force_order_to_recording_mode:
            self.make_header_order_match_recording_mode()

        # iterate over sections first:
        for seg_index in range(self.num_sweeps):

            seg = Segment(index=seg_index)

            # iterate over channels:
            for chan_idx, recsig in enumerate(self.header["signal_channels"]):  # TODO fix channels!

                unit = self.header["signal_channels"]["units"][chan_idx]
                name = self.header["signal_channels"]["name"][chan_idx]
                sampling_rate = self.header["signal_channels"]["sampling_rate"][chan_idx] * 1 / pq.s
                t_start = self.series_data[unit]["time"][seg_index, 0] * pq.s

                recdata = self.get_chan_data_or_stim_data_if_does_not_exist(seg_index, unit)
                signal = pq.Quantity(recdata, unit).T

                anaSig = AnalogSignal(signal, sampling_rate=sampling_rate,
                                      t_start=t_start, name=name,
                                      channel_index=chan_idx)
                seg.analogsignals.append(anaSig)
                bl.segments.append(seg)

        bl.create_many_to_one_relationship()

        return bl

    def fill_header(self):

        signal_channels = []

        for ch_idx, chan in enumerate(self.channels):

            ch_id = ch_idx + 1
            ch_name = chan["name"]
            ch_units = chan["unit"]
            dtype = chan["dtype"]
            sampling_rate = 1 / chan["sampling_step"] * 1 / pq.s
            gain = 1
            offset = 0
            stream_id = "0"
            signal_channels.append((ch_name, ch_id, sampling_rate, dtype, ch_units, gain, offset, stream_id))  # turned into numpy array after stim channel added

        self.append_stimulus_if_analogisignal_does_not_exist(signal_channels)

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
        self.header['signal_channels'] =  np.array(signal_channels, dtype=_signal_channel_dtype)
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self.check_channel_sampling_rate()

    def append_stimulus_if_analogisignal_does_not_exist(self, signal_channels):
        for unit in ["A", "V"]:
            if not np.any(self.series_data[unit]["data"]) and np.any(self.series_data[unit]["stim"]):
                signal_channels.append(("stimulation",
                                        2,
                                        1 / self.series_data[unit]["stim"]["ts"] * 1 / pq.s,
                                        "float64",
                                        self.series_data["A"]["stim"]["units"], 1, 0, "0"))

    def check_channel_sampling_rate(self):
        """
        this is already checked in load-heka-python but sanity checked here
        """
        sampling_rate = []
        for chan in self.header["signal_channels"]:
            sampling_rate.append(chan["sampling_rate"])

        assert len(np.unique(sampling_rate)), "HEAK record sampling are not the same "

    def make_header_order_match_recording_mode(self):

        first_channel_units = self.header["signal_channels"][0]["units"]

        if self.recording_mode == "CClamp" and first_channel_units != "V" or \
                      self.recording_mode == "VClamp" and first_channel_units != "A":

            channels = self.header["signal_channels"]

            first_chan = copy.deepcopy(channels[0])
            channels[0] = channels[1]
            channels[1] = first_chan

    def get_chan_data_or_stim_data_if_does_not_exist(self, seg_index, unit):
        """
        Get stim data if second channel does not exist f4 is a good test of this!
        """
        if not np.any(self.series_data[unit]["data"]) and np.any(self.series_data[unit]["stim"]):
            recdata = self.series_data[unit]["stim"]["data"][seg_index, :]
        else:
            recdata = self.series_data[unit]["data"][seg_index, :]

        return recdata
