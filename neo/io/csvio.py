"""
Quick way to load Hekafiles into Neo format using load-heka-python module.
Joseph John Ziminski 2021
TODO: implement at rawio level for inclusion in Neo
"""
from __future__ import annotations

import numpy as np
import quantities as pq
import copy
from typing import Optional, Literal, TYPE_CHECKING
from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal
from ..rawio.baserawio import _signal_channel_dtype, _signal_stream_dtype, _spike_channel_dtype, _event_channel_dtype
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

class CSVIO(BaseIO):
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

    name = 'CSV'
    extensions = ['.csv']

    mode = 'file'

    def __init__(self, filename, ordered_datatype, ordered_units, interleaved_or_consecutive, row_or_column, has_header, has_index, time_from_sampling_rate):
        """
        Assumes HEKA file units is A and V for data and stimulus. This is enforced in LoadHeka level.
        """
        BaseIO.__init__(self)

        self.filename = filename
        self.dataset, self.channels, self.num_sweeps = load_csv_file(
            filename, ordered_datatype, ordered_units, interleaved_or_consecutive, row_or_column, has_header, has_index, time_from_sampling_rate
        )
        self.header = {}

    def read_block(self, lazy=False):
        assert not lazy, 'Do not support lazy'

        bl = Block()
        self.fill_header()

        # iterate over sections first:
        for seg_index in range(self.num_sweeps):

            seg = Segment(index=seg_index)

            # iterate over channels:
            for chan_idx, recsig in enumerate(self.header["signal_channels"]):

                unit = self.header["signal_channels"]["units"][chan_idx]  # revisit if we can loose the indexing
                name = self.header["signal_channels"]["name"][chan_idx]
                sampling_rate = self.header["signal_channels"]["sampling_rate"][chan_idx] * 1 / pq.s

                t_start = self.dataset["time"][seg_index] * pq.s
                recdata = self.dataset[name][:, seg_index]

                if unit in ["pV", "fA"]:  # Quantity does not support
                    signal = pq.Quantity(recdata).T
                else:
                    signal = pq.Quantity(recdata, unit).T

                anaSig = AnalogSignal(signal, sampling_rate=sampling_rate,
                                      t_start=t_start, name=name,
                                      channel_index=chan_idx)
                seg.analogsignals.append(anaSig)
            bl.segments.append(seg)

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


def load_csv_file(
    filepath: Path,
    ordered_datatypes: List[Literal["time", "vm", "im"]],
    ordered_units: List[str],
    interleaved_or_consecutive: Literal["interleaved", "concecutive"],
    row_or_column: Literal["row", "column"],
    has_header: bool,
    has_index: bool,
    time_from_fs: Optional[Union[float, int]],
) -> Tuple[Dict, List, int]:
    """
    """
    num_datatypes = len(ordered_datatypes)

    # Check Inputs
    if not all ([ele in ["time", "im", "vm"] for ele in ordered_datatypes]):
        raise ValueError("`ordered_datatypes` must be one of ['time', 'vm', 'im']")

    if "time" not in ordered_datatypes and not time_from_fs["on"]:
        raise ValueError("Either 'time' must be in the .csv or the sampling"
                         "rate provided in `time_from_fs`.")

    if len(ordered_units) != num_datatypes:
        raise ValueError("`ordered_units` must contain one unit "
                         "for each datatype in `ordered_datatypes`.")

    # Load CSV
    if row_or_column == "row":
        has_header, has_index = has_index, has_header

    data = pd.read_csv(
        filepath,
        header=0 if has_header else None,
        index_col=0 if has_index else None
    )

    if row_or_column == "row":
        data = data.T

    try:
        data = data.to_numpy(dtype=np.float64)
    except ValueError as e:
        raise ValueError(f"{e} Check that `has_header` and `has_index` are set correctly.")

    num_cols = data.shape[1]
    num_samples = data.shape[0]
    num_recs, remainder = np.divmod(num_cols, num_datatypes)

    if remainder != 0:
        raise ValueError(
            f"The number of columns ({num_cols}) and datatypes ({num_datatypes}) "
            f"does not divide evently into records. "
            f"Check that each datatype has an entry for all records and "
            f"that column or row-wise and header / index options are set correctly."
        )

    # Index out data from the CSV
    if interleaved_or_consecutive == "interleaved":

        array_1 = data[:, 0:num_cols:num_datatypes]

        if num_datatypes > 1:
            array_2 = data[:, 1:num_cols:num_datatypes]

        if num_datatypes > 2:
            array_3 = data[:, 2:num_cols:num_datatypes]

    elif interleaved_or_consecutive == "consecutive":

        array_1 = data[:, :num_recs]

        if num_datatypes > 1:
            array_2 = data[:, num_recs:num_recs * 2]

        if num_datatypes > 2:
            array_3 = data[:, num_recs * 2 : num_recs * 3]

    else:
        raise ValueError("`interleaved_or_consecutive` not recognised.")

    # Save the data, generating time array if required.
    results = {"time": None, "im": None, "vm": None}

    results[ordered_datatypes[0]] = array_1

    if num_datatypes > 1:
        results[ordered_datatypes[1]] = array_2

    if num_datatypes > 2:
        results[ordered_datatypes[2]] = array_3

    if time_from_fs["on"]:
        sampling_step = 1 / time_from_fs["value"]
        first_rec_samples = np.arange(num_recs) * num_samples * sampling_step
        results["time"] = first_rec_samples
    else:
        sampling_step = results["time"][1][0] - results["time"][0][0]
        results["time"] = results["time"][0, :]

    # Save channel information for Neo
    channels = []
    for name, unit in zip(ordered_datatypes, ordered_units):
        if name != "time":
            channels.append({"name": name, "unit": unit, "dtype": np.dtype("f8"), "sampling_step": sampling_step})

    return results, channels, num_recs
