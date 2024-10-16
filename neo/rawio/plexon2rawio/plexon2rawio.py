"""
Plexon2RawIO is a class to read Plexon PL2 files.

This IO is based on the pypl2lib module, see https://github.com/NeuralEnsemble/pypl2

On non-windows systems this IO requires Wine, see https://www.winehq.org/

Note on IO performance and memory consumption:
This IO is only partially lazy: To load a data chunk of an individual
signal channels, the complete data of that channel will be loaded into memory
and cached. This occurs as a limitation of the pypl2 package and might be improved
in future versions by exploiting the block-wise loading of signals.

To clear the currently cached signal data use the `clear_analogsignal_cache()` method.

Note on Neo header and spike channels:
The IO only considers enabled channels and will not list disabled channels in its header.

There is no 1-1 correspondence of PL2 spike channels and neo spike channels as each unit of
a PL2 spike channel will be represented as an individual neo spike channel.

Author: Julia Sprenger
"""

import pathlib
import warnings
import platform
import re

from urllib.request import urlopen
from datetime import datetime

import numpy as np


from ..baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)


class Plexon2RawIO(BaseRawIO):
    """
    Class for "reading" data from a PL2 file

    Parameters
    ----------
    filename: str | Path
        The *.pl2 file to be loaded
    pl2_dll_file_path: str | Path | None, default: None
        The path to the necessary dll for loading pl2 files
        If None will find correct dll for architecture and if it does not exist will download it
    reading_attempts: int, default: 25
        Number of attempts to read the file before raising an error
        This opening process is somewhat unreliable and might fail occasionally. Adjust this higher
        if you encounter problems in opening the file.

    Notes
    -----
    * This IO is only partially lazy
    * The IO only considers enabled channels and will not list disabled channels in its header.
    * There is no 1-1 correspondence of PL2 spike channels and neo spike channels as each unit of
      a PL2 spike channel will be represented as an individual neo spike channel.

    Examples
    --------
    >>> import neo.rawio
    >>> r = neo.rawio.Plexon2RawIO(filename='my_data.pl2')
    >>> r.parse_header()
    >>> print(r)
    >>> raw_chunk = r.get_analogsignal_chunk(block_index=0,
                                             seg_index=0,
                                             i_start=0,
                                             i_stop=1024,
                                             stream_index=0,
                                             channel_indexes=range(10))
    >>> float_chunk = r.rescale_signal_raw_to_float(raw_chunk,
                                                    dtype='float64',
                                                    stream_index=0,
                                                    channel_indexes=[0, 3, 6])
    >>> spike_timestamp = r.get_spike_timestamps(spike_channel_index=0,
                                                 t_start=None,
                                                 t_stop=None)
    >>> spike_times = r.rescale_spike_timestamp(spike_timestamp, dtype='float64')
    >>> ev_timestamps, _, ev_labels = r.get_event_timestamps(event_channel_index=0)

    """

    extensions = ["pl2"]
    rawmode = "one-file"

    def __init__(self, filename, pl2_dll_file_path=None, reading_attempts=25):

        # signals, event and spiking data will be cached
        # cached signal data can be cleared using `clear_analogsignal_cache()()`
        self._analogsignal_cache = {}
        self._event_channel_cache = {}
        self._spike_channel_cache = {}

        # note that this filename is used in self._source_name
        self.filename = pathlib.Path(filename)

        if (not self.filename.exists()) or (not self.filename.is_file()):
            raise ValueError(f"{self.filename} is not a file.")

        BaseRawIO.__init__(self)

        # download default PL2 dll once if not yet available
        if pl2_dll_file_path is None:
            architecture = platform.architecture()[0]
            if architecture == "64bit" and platform.system() in ["Windows", "Darwin"]:
                file_name = "PL2FileReader64.dll"
            else:  # Apparently wine uses the 32 bit version in linux
                file_name = "PL2FileReader.dll"
            pl2_dll_folder = pathlib.Path.home() / ".plexon_dlls_for_neo"
            pl2_dll_folder.mkdir(exist_ok=True)
            pl2_dll_file_path = pl2_dll_folder / file_name

            if not pl2_dll_file_path.exists():
                url = f"https://raw.githubusercontent.com/Neuralensemble/pypl2/master/bin/{file_name}"
                dist = urlopen(url=url)

                with open(pl2_dll_file_path, "wb") as f:
                    warnings.warn(f"dll file does not exist, downloading plexon dll to {pl2_dll_file_path}")
                    f.write(dist.read())

        # Instantiate wrapper for Windows DLL
        from neo.rawio.plexon2rawio.pypl2.pypl2lib import PyPL2FileReader

        self.pl2reader = PyPL2FileReader(pl2_dll_file_path=pl2_dll_file_path)

        for attempt in range(reading_attempts):
            self.pl2reader.pl2_open_file(self.filename)

            # Verify the file handle is valid.
            if self.pl2reader._file_handle.value != 0:
                # File handle is valid, exit the loop early
                break
            else:
                if attempt == reading_attempts - 1:
                    self.pl2reader._print_error()
                    raise IOError(f"Opening {self.filename} failed after {reading_attempts} attempts.")

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        """
        Collect information about the file, construct neo header and provide annotations
        """

        # Scanning sources and populating signal channels at the same time. Sources have to have
        # same sampling rate and number of samples to belong to one stream.
        signal_channels = []
        channel_num_samples = []

        # We will build the stream ids based on the channel prefixes
        # The channel prefixes are the first characters of the channel names which have the following format:
        # WB{number}, FPX{number}, SPKCX{number}, AI{number}, etc
        # We will extract the prefix and use it as stream id
        regex_prefix_pattern = r"^\D+"  # Match any non-digit character at the beginning of the string

        for channel_index in range(self.pl2reader.pl2_file_info.m_TotalNumberOfAnalogChannels):
            achannel_info = self.pl2reader.pl2_get_analog_channel_info(channel_index)
            # only consider active channels
            if not (achannel_info.m_ChannelEnabled and achannel_info.m_ChannelRecordingEnabled):
                continue

            # assign to matching stream or create new stream based on signal characteristics
            rate = achannel_info.m_SamplesPerSecond
            num_samples = achannel_info.m_NumberOfValues
            channel_num_samples.append(num_samples)

            ch_name = achannel_info.m_Name.decode()
            chan_id = f"source{achannel_info.m_Source}.{achannel_info.m_Channel}"
            dtype = "int16"
            units = achannel_info.m_Units.decode()
            gain = achannel_info.m_CoeffToConvertToUnits
            offset = 0.0  # PL2 files don't contain information on signal offset

            channel_prefix = re.match(regex_prefix_pattern, ch_name).group(0)
            stream_id = channel_prefix
            buffer_id = ""
            signal_channels.append((ch_name, chan_id, rate, dtype, units, gain, offset, stream_id, buffer_id))

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        channel_num_samples = np.array(channel_num_samples)

        # We are using channel prefixes as stream_ids
        # The meaning of the channel prefixes was provided by a Plexon Engineer, see here:
        # https://github.com/NeuralEnsemble/python-neo/pull/1495#issuecomment-2184256894
        stream_id_to_stream_name = {
            "WB": "WB-Wideband",
            "FP": "FPl-Low Pass Filtered",
            "SP": "SPKC-High Pass Filtered",
            "AI": "AI-Auxiliary Input",
            "AIF": "AIF-Auxiliary Input Filtered",
        }

        unique_stream_ids = np.unique(signal_channels["stream_id"])
        signal_streams = []
        for stream_id in unique_stream_ids:
            # We are using the channel prefixes as ids
            # The users of plexon can modify the prefix of the channel names (e.g. `my_prefix` instead of `WB`).
            # In that case we use the channel prefix both as stream id and name
            stream_name = stream_id_to_stream_name.get(stream_id, stream_id)
            buffer_id = ""
            signal_streams.append((stream_name, stream_id, buffer_id))
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        # In plexon buffer is unknown
        signal_buffers = np.array([], dtype=_signal_buffer_dtype)

        self._stream_id_samples = {}
        self._stream_index_to_stream_id = {}
        for stream_index, stream_id in enumerate(signal_streams["id"]):
            # Keep a mapping from stream_index to stream_id
            self._stream_index_to_stream_id[stream_index] = stream_id

            # We extract the number of samples for each stream
            mask = signal_channels["stream_id"] == stream_id
            signal_num_samples = np.unique(channel_num_samples[mask])
            assert signal_num_samples.size == 1, "All channels in a stream must have the same number of samples"
            self._stream_id_samples[stream_id] = signal_num_samples[0]

        # pre-loading spike channel_data for later usage
        self._spike_channel_cache = {}
        spike_channels = []
        for c in range(self.pl2reader.pl2_file_info.m_TotalNumberOfSpikeChannels):
            schannel_info = self.pl2reader.pl2_get_spike_channel_info(c)

            # only consider active channels
            if not (schannel_info.m_ChannelEnabled and schannel_info.m_ChannelRecordingEnabled):
                continue

            # In a PL2 spike channel header, the field "m_NumberOfUnits" denotes the number
            # of units to which spikes detected on that channel have been assigned. It does
            # not account for unsorted spikes, i.e., spikes that have not been assigned to
            # a unit. It is Plexon's convention to assign unsorted spikes to channel_unit_id=0,
            # and sorted spikes to channel_unit_id's 1, 2, 3...etc. Therefore, for a given value of
            # m_NumberOfUnits, there are m_NumberOfUnits+1 channel_unit_ids to consider - 1
            # unsorted channel_unit_id (0) + the m_NumberOfUnits sorted channel_unit_ids.
            for channel_unit_id in range(schannel_info.m_NumberOfUnits + 1):
                unit_name = f"{schannel_info.m_Name.decode()}.{channel_unit_id}"
                unit_id = f"unit{schannel_info.m_Channel}.{channel_unit_id}"
                wf_units = schannel_info.m_Units
                wf_gain = schannel_info.m_CoeffToConvertToUnits
                wf_offset = 0.0  # A waveform offset is not provided in PL2 files
                wf_left_sweep = schannel_info.m_PreThresholdSamples
                wf_sampling_rate = schannel_info.m_SamplesPerSecond
                spike_channels.append(
                    (unit_name, unit_id, wf_units, wf_gain, wf_offset, wf_left_sweep, wf_sampling_rate)
                )

        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # creating event/epoch channel
        self._event_channel_cache = {}
        event_channels = []
        for i in range(self.pl2reader.pl2_file_info.m_NumberOfDigitalChannels):
            echannel_info = self.pl2reader.pl2_get_digital_channel_info(i)

            # only consider active channels
            if not (echannel_info.m_ChannelEnabled and echannel_info.m_ChannelRecordingEnabled):
                continue

            # event channels are characterized by (name, id, type), with type in ['event', 'epoch']
            channel_name = echannel_info.m_Name.decode()
            event_channels.append((channel_name, echannel_info.m_Channel, "event"))

        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # fill into header dict
        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [1]  # It seems pl2 can only contain a single segment
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        self._generate_minimal_annotations()

        # this pprint lines really help for understand the nested (and complicated sometimes) dict
        # from pprint import pprint
        # pprint(self.raw_annotations)

        # Note: pl2_file_info.m_ReprocessorDateTime seems to be always empty.
        # To be checked against alternative pl2 reader.

        # Provide additional, recommended annotations for the final neo objects.
        block_index = 0
        bl_ann = self.raw_annotations["blocks"][block_index]
        bl_ann["name"] = f"Block containing PL2 data#{block_index}"
        bl_ann["file_origin"] = self.filename
        file_info = self.pl2reader.pl2_file_info
        block_info = {attr: getattr(file_info, attr) for attr, _ in file_info._fields_}

        # convert ctypes datetime objects to datetime.datetime objects for annotations
        from .pypl2.pypl2lib import tm

        for anno_key, anno_value in block_info.items():
            if isinstance(anno_value, tm):
                tmo = anno_value
                # invalid datetime information if year is <1
                if tmo.tm_year != 0:
                    microseconds = block_info["m_CreatorDateTimeMilliseconds"] * 1000
                    # tm_mon range is 0..11 https://cplusplus.com/reference/ctime/tm/
                    # python is 1..12 https://docs.python.org/3/library/datetime.html#datetime.datetime
                    # so month needs to be tm_mon+1; also tm_sec could cause problems in the case of leap
                    # seconds, but this is harder to defend against.
                    year = tmo.tm_year  # This has base 1900 in the c++ struct specification so we need to add 1900
                    year += 1900
                    dt = datetime(
                        year=year,
                        month=tmo.tm_mon + 1,
                        day=tmo.tm_mday,
                        hour=tmo.tm_hour,
                        minute=tmo.tm_min,
                        second=tmo.tm_sec,
                        microsecond=microseconds,
                    )
                    # ignoring daylight saving time information for now as timezone is unknown

                else:
                    dt = None

                block_info[anno_key] = dt

        bl_ann.update(block_info)
        for seg_index in range(1):
            seg_ann = bl_ann["segments"][seg_index]

            # some attributes don't apply to neo spike channels as these cover only a subpart of
            # the data of a PL2 spike channel
            spike_annotation_keys = [
                "m_SortEnabled",
                "m_SortRangeStart",
                "m_SortRangeEnd",
                "m_SourceTrodality",
                "m_OneBasedTrode",
                "m_OneBasedChannelInTrode",
                "m_Source",
                "m_Channel",
            ]
            for spike_channel_idx, spike_header in enumerate(self.header["spike_channels"]):
                schannel_name = spike_header["name"].split(".")[0]
                schannel_info = self.pl2reader.pl2_get_spike_channel_info_by_name(schannel_name)

                spiketrain_an = seg_ann["spikes"][spike_channel_idx]
                for key in spike_annotation_keys:
                    spiketrain_an[key] = getattr(schannel_info, key)

            event_annotation_keys = ["m_Source", "m_Channel", "m_Name"]
            for event_channel_idx, event_header in enumerate(self.header["event_channels"]):
                dchannel_name = event_header["name"]
                dchannel_info = self.pl2reader.pl2_get_digital_channel_info_by_name(dchannel_name)

                event_an = seg_ann["events"][event_channel_idx]
                for key in event_annotation_keys:
                    event_an[key] = getattr(dchannel_info, key)

            signal_array_annotation_keys = [
                "m_SourceTrodality",
                "m_OneBasedTrode",
                "m_OneBasedChannelInTrode",
                "m_Source",
                "m_Channel",
                "m_MaximumNumberOfFragments",
            ]

            for stream_idx, stream_header in enumerate(self.header["signal_streams"]):
                signal_array_annotations = {key: [] for key in signal_array_annotation_keys}
                stream_id = stream_header["id"]
                # extract values of individual signals

                stream_channel_mask = self.header["signal_channels"]["stream_id"] == stream_id
                for signal_idx, signal_header in enumerate(self.header["signal_channels"][stream_channel_mask]):
                    achannel_name = signal_header["name"]
                    achannel_info = self.pl2reader.pl2_get_analog_channel_info_by_name(achannel_name)
                    for key in signal_array_annotation_keys:
                        signal_array_annotations[key].append(getattr(achannel_info, key))

                seg_ann["signals"][stream_idx]["__array_annotations__"] = signal_array_annotations

    def _segment_t_start(self, block_index, seg_index):
        # this must return a float values in seconds
        return self.pl2reader.pl2_file_info.m_StartRecordingTime / self.pl2reader.pl2_file_info.m_TimestampFrequency

    def _segment_t_stop(self, block_index, seg_index):
        # this must return a float value in seconds
        end_time = (
            self.pl2reader.pl2_file_info.m_StartRecordingTime + self.pl2reader.pl2_file_info.m_DurationOfRecording
        )
        return float(end_time / self.pl2reader.pl2_file_info.m_TimestampFrequency)

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_id = self._stream_index_to_stream_id[stream_index]
        num_samples = int(self._stream_id_samples[stream_id])
        return num_samples

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        # This returns the t_start of signals as a float value in seconds

        # TODO: Does the fragment_timestamp[0] need to be added here for digital signals?
        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
        # this must return a signal chunk in a signal stream
        # limited with i_start/i_stop (can be None)
        # channel_indexes can be None (=all channel in the stream) or a list or numpy.array
        # This must return a numpy array 2D (even with one channel).
        # This must return the original dtype. No conversion here.
        # This must as fast as possible.
        # To speed up this call all preparatory calculations should be implemented
        # in _parse_header().

        stream_id = self.header["signal_streams"][stream_index]["id"]
        mask = self.header["signal_channels"]["stream_id"] == stream_id
        stream_channels = self.header["signal_channels"][mask]

        n_channels = len(stream_channels)
        n_samples = self.get_signal_size(block_index, seg_index, stream_index)

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = n_samples

        if i_start < 0 or i_stop > n_samples:
            raise IndexError(f"Indexes ({i_start}, {i_stop}) out of range for signal with {n_samples} samples")

        # converting channel_indexes to array representation
        if channel_indexes is None:
            channel_indexes = np.arange(len(stream_channels), dtype="int")
        elif isinstance(channel_indexes, slice):
            channel_indexes = np.arange(len(stream_channels), dtype="int")[channel_indexes]
        else:
            channel_indexes = np.asarray(channel_indexes)

        # channel index sanity check
        if any(channel_indexes < 0) or any(channel_indexes >= n_channels):
            raise IndexError(f"Channel index out of range {channel_indexes} for stream with {n_channels} channels")

        nb_chan = len(channel_indexes)

        raw_signals = np.empty((i_stop - i_start, nb_chan), dtype="int16")
        for i, channel_idx in enumerate(channel_indexes):
            channel_name = stream_channels["name"][channel_idx]

            # use previously loaded channel data if possible
            if channel_name in self._analogsignal_cache:
                values = self._analogsignal_cache[channel_name]
            else:
                res = self.pl2reader.pl2_get_analog_channel_data_by_name(channel_name)
                fragment_timestamps, fragment_counts, values = res
                self._analogsignal_cache[channel_name] = values

            raw_signals[:, i] = values[i_start:i_stop]

        return raw_signals

    def clear_analogsignal_cache(self):
        for channel_name, values in self._analogsignal_cache.items():
            del values
        self._analogsignal_cache = {}

    def _spike_count(self, block_index, seg_index, spike_channel_index):
        channel_header = self.header["spike_channels"][spike_channel_index]
        channel_name, channel_unit_id = channel_header["name"].split(".")
        channel_unit_id = int(channel_unit_id)

        # loading spike channel data on demand when not already cached
        if channel_name not in self._spike_channel_cache:
            self._spike_channel_cache[channel_name] = self.pl2reader.pl2_get_spike_channel_data_by_name(channel_name)

        spike_timestamps, unit_ids, waveforms = self._spike_channel_cache[channel_name]
        nb_spikes = np.count_nonzero(unit_ids == channel_unit_id)

        return nb_spikes

    def _get_spike_timestamps(self, block_index, seg_index, spike_channel_index, t_start, t_stop):
        channel_header = self.header["spike_channels"][spike_channel_index]
        channel_name, channel_unit_id = channel_header["name"].split(".")
        channel_unit_id = int(channel_unit_id)

        # loading spike channel data on demand when not already cached
        if channel_name not in self._spike_channel_cache:
            self._spike_channel_cache[channel_name] = self.pl2reader.pl2_get_spike_channel_data_by_name(channel_name)

        spike_timestamps, unit_ids, waveforms = self._spike_channel_cache[channel_name]

        time_mask = self._get_timestamp_time_mask(t_start, t_stop, spike_timestamps)

        unit_mask = unit_ids[time_mask] == channel_unit_id
        spike_timestamps = spike_timestamps[time_mask][unit_mask]

        # spike timestamps are counted from the session start recording time
        spike_timestamps += self.pl2reader.pl2_file_info.m_StartRecordingTime

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        # must rescale to second a particular spike_timestamps
        # with a fixed dtype so the user can choose the precision they want
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self.pl2reader.pl2_file_info.m_TimestampFrequency
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, spike_channel_index, t_start, t_stop):
        # this must return a 3D numpy array (nb_spike, nb_channel, nb_sample)
        # in the original dtype
        # this must be as fast as possible.
        # the same clip t_start/t_start must be used in _spike_timestamps()

        channel_header = self.header["spike_channels"][spike_channel_index]
        channel_name, channel_unit_id = channel_header["name"].split(".")

        # loading spike channel data on demand when not already cached
        if channel_name not in self._spike_channel_cache:
            self._spike_channel_cache[channel_name] = self.pl2reader.pl2_get_spike_channel_data_by_name(channel_name)

        spike_timestamps, unit_ids, waveforms = self._spike_channel_cache[channel_name]

        time_mask = self._get_timestamp_time_mask(t_start, t_stop, spike_timestamps)

        unit_mask = unit_ids[time_mask] == int(channel_unit_id)
        waveforms = waveforms[time_mask][unit_mask]

        # add tetrode dimension
        waveforms = np.expand_dims(waveforms, axis=1)
        return waveforms

    def _get_timestamp_time_mask(self, t_start, t_stop, timestamps):

        if t_start is not None or t_stop is not None:
            # restrict spikes to given limits (in seconds)
            timestamp_frequency = self.pl2reader.pl2_file_info.m_TimestampFrequency
            lim0 = int(t_start * timestamp_frequency)
            lim1 = int(t_stop * self.pl2reader.pl2_file_info.m_TimestampFrequency)

            # limits are with respect to segment t_start and not to time 0
            lim0 -= self.pl2reader.pl2_file_info.m_StartRecordingTime
            lim1 -= self.pl2reader.pl2_file_info.m_StartRecordingTime

            time_mask = (timestamps >= lim0) & (timestamps <= lim1)

        else:
            time_mask = slice(None, None)

        return time_mask

    def _event_count(self, block_index, seg_index, event_channel_index):

        channel_header = self.header["event_channels"][event_channel_index]
        channel_name = channel_header["name"]

        # loading event channel data on demand when not already cached
        if channel_name not in self._event_channel_cache:
            self._event_channel_cache[channel_name] = self.pl2reader.pl2_get_digital_channel_data_by_name(channel_name)

        event_times, values = self._event_channel_cache[channel_name]

        return len(event_times)

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        # the main difference between spike channel and event channel
        # is that for here we have 3 numpy array timestamp, durations, labels
        # durations must be None for 'event'
        # label must a dtype ='U'

        channel_header = self.header["event_channels"][event_channel_index]
        channel_name = channel_header["name"]

        # loading event channel data on demand when not already cached
        if channel_name not in self._event_channel_cache:
            self._event_channel_cache[channel_name] = self.pl2reader.pl2_get_digital_channel_data_by_name(channel_name)

        event_times, labels = self._event_channel_cache[channel_name]
        labels = np.asarray(labels, dtype="U")

        time_mask = self._get_timestamp_time_mask(t_start, t_stop, event_times)

        # events don't have a duration. Epochs are not supported
        durations = None

        # event timestamps are counted from the session start recording time
        return_times = event_times[time_mask] + self.pl2reader.pl2_file_info.m_StartRecordingTime

        return return_times, durations, labels[time_mask]

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        # must rescale to second a particular event_timestamps
        # with a fixed dtype so the user can choose the precision he want.

        event_times = event_timestamps.astype(dtype)
        event_times /= self.pl2reader.pl2_file_info.m_TimestampFrequency
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        durations = raw_duration.astype(dtype)
        durations /= self.pl2reader.pl2_file_info.m_TimestampFrequency
        return durations

    def close(self):
        self.pl2reader.pl2_close_all_files()
