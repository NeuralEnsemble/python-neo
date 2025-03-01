"""
Quick way to load Hekafiles into Neo format using load-heka-python module.
Joseph John Ziminski 2021
TODO: implement at rawio level for inclusion in Neo
"""
import numpy as np
import quantities as pq
import copy

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal
from ..rawio.baserawio import _signal_channel_dtype, _signal_stream_dtype, _spike_channel_dtype, _event_channel_dtype

try:
    from load_heka_python.load_heka import LoadHeka
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

    def __init__(self, filename, group_idx, series_idx, add_zero_offset, stimulus_experimental_mode):
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
        self.add_zero_offset = add_zero_offset
        self.stimulus_experimental_mode = stimulus_experimental_mode
        self.num_sweeps = None
        self.series_data = None

    def read_block(self, lazy=False):
        assert not lazy, 'Do not support lazy'

        self.heka = LoadHeka(self.filename)

        bl = Block()

        self.orig_num_channels = len(
            self.heka.get_series_channels(self.group_idx, self.series_idx)
        )

        # First get the raw channels from the data. This does not include
        # stimulation channels which are only loaded if there is only 1 channels.

        # Next, fill the header formatted for neo. This will also add
        # the stimulus channel to `self.header` if there is 1 channel
        # and the stimulus exists.
        self.num_sweeps = self.heka.get_num_sweeps_in_series(self.group_idx, self.series_idx)

        self.series_data = self.get_series_data()
        self.recording_mode = self.series_data[0]["recording_mode"]

        if self.orig_num_channels == 1:
            self.update_series_data_with_stim_trace_if_exists()

        self.reorganise_series_data_to_cc_or_vc_mode()

        self.make_header()

        if (true_sweep_num := self.series_data[0]["time"].shape[0]) != self.num_sweeps:
            # sometimes the heka header can be wrong if the protocol stopped before end.
            self.num_sweeps = true_sweep_num
            self.header['nb_segment'] = [self.num_sweeps]

        # iterate over sections first:
        for seg_index in range(self.num_sweeps):

            seg = Segment(index=seg_index)

            # iterate over channels:
            for channel_index, recsig in enumerate(self.header["signal_channels"]):
                unit = self.header["signal_channels"]["units"][channel_index]
                name = self.header["signal_channels"]["name"][channel_index]
                sampling_rate = self.header["signal_channels"]["sampling_rate"][channel_index] * (1 / pq.s)

                t_start = self.series_data[channel_index]["time"][seg_index, 0] * pq.s
                recdata = self.series_data[channel_index]["data"][seg_index, :]

                if unit in ["pV", "fA"]:  # Quantity does not support
                    signal = pq.Quantity(recdata).T
                else:
                    signal = pq.Quantity(recdata, unit).T

                anaSig = AnalogSignal(signal, sampling_rate=sampling_rate,
                                      t_start=t_start, name=name,
                                      channel_index=channel_index)
                seg.analogsignals.append(anaSig)
            bl.segments.append(seg)

        return bl

    def reorganise_series_data_to_cc_or_vc_mode(self):
        """
        """
        is_leak = [ele["data_kinds"][0]["IsLeak"] if "data_kinds" in ele else False for ele in self.series_data]
        units = [ele["units"] for ele in self.series_data]

        assert np.sum(is_leak) <= 1, "More than 1 leak channel not currently supported."
        non_leak = [unit for unit, leak in zip(units, is_leak) if not leak]

        assert len(non_leak) <= 2, "Only 2 non-leak channels are currently supported"

        if any(is_leak):
            leak_idx = is_leak.index(True)
            if leak_idx != len(is_leak) - 1:
                self.series_data.append(self.series_data.pop(leak_idx))

        if len(self.series_data) == 2:
            first_channel_units = units[0]

            if self.recording_mode == "CClamp" and first_channel_units != "V" or \
                    self.recording_mode == "VClamp" and first_channel_units != "A":
                self.series_data[0], self.series_data[1] = self.series_data[1], self.series_data[0]

    def get_series_data(self):
        """
        """
        series_data = []
        for ch_idx in range(self.orig_num_channels):
            series_data.append(
                self.heka.get_series_data(
                    self.group_idx,
                    self.series_idx,
                    ch_idx,
                    include_stim_protocol="experimental" if self.stimulus_experimental_mode else True,
                    add_zero_offset=self.add_zero_offset,
                    fill_with_mean=True,
                    stim_channel_idx=None,
                )
            )
        return series_data

    def update_series_data_with_stim_trace_if_exists(self):
        """
        """
        if np.any(self.series_data[0]["stim"]):
            stim_data = self.series_data[0]["stim"]
            stim_data["time"] = self.series_data[0]["time"]
            self.series_data.append(stim_data)

    def make_header(self):

        signal_channels = []
        heka_metadata = {
            "add_zero_offset": self.add_zero_offset,
            "zero_offsets": [],
        }
        for ch_idx, chan_data in enumerate(self.series_data):
            ch_id = ch_idx + 1
            ch_name = chan_data["name"]
            ch_units = chan_data["units"]
            dtype = chan_data["dtype"]
            sampling_rate = (1 / chan_data["sampling_step"]) * 1 / pq.s
            gain = 1
            offset = 0
            stream_id = "0"
            signal_channels.append((ch_name, ch_id, sampling_rate, dtype, ch_units, gain, offset, stream_id))  # turned into numpy array after stim channel added

            # zero offsets will not exist for stim data
            heka_metadata["zero_offsets"].append(
                chan_data["zero_offsets"] if "zero_offsets" in chan_data else None
            )

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
        self.header['signal_channels'] = np.array(signal_channels, dtype=_signal_channel_dtype)
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels
        self.header["heka_metadata"] = heka_metadata

        self.check_channel_sampling_rate_and_channel_order()

    def check_channel_sampling_rate_and_channel_order(self):
        """
        this is already checked in load-heka-python but sanity checked here
        """
        sampling_rate = []
        for chan in self.header["signal_channels"]:
            sampling_rate.append(chan["sampling_rate"])

        assert len(np.unique(sampling_rate)), "HEKA record sampling are not the same "

        if self.recording_mode == "CClamp":
            assert self.header["signal_channels"]["units"][0] == "V", "bad vc"
        elif self.recording_mode == "VClamp":
            assert self.header["signal_channels"]["units"][0] == "A", "bad cc"

    def make_header_order_match_recording_mode(self):
        """
        """
        first_channel_units = self.header["signal_channels"][0]["units"]

        if self.recording_mode == "CClamp" and first_channel_units != "V" or \
                self.recording_mode == "VClamp" and first_channel_units != "A":

            channels = self.header["signal_channels"]

            first_chan = copy.deepcopy(channels[0])
            channels[0] = channels[1]
            channels[1] = first_chan

