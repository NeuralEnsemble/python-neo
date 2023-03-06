"""
This module implements file reader for AlphaOmega MPX file format version 4.

This module expect default channel names from the AlphaOmega record system (RAW
###, SPK ###, LFP ###, AI ###,…).

This module reads all \*.mpx files in a directory (not recursively) by default.
If you provide a list of \*.lsx files only the \*.mpx files referenced by those
\*.lsx files will be loaded.

The specifications are mostly extracted from the "AlphaRS User Manual V1.0.1.pdf"
manual provided with the AlphaRS hardware. The specifications are described in
the chapter 6: ALPHARS FILE FORMAT. See at the end of this file for file format
blocks description.
Some informations missing from the file specifications were kindly provided by
AlphaOmega engineers.

.. note::
    Not a lot of memory optimization effort was put into this module. You should
    expect a big memory usage when loading data with this module

Author: Thomas Perret <thomas.perret@isc.cnrs.fr>
"""

import io
import logging
import mmap
import os
import struct

from collections import defaultdict
from datetime import datetime
from itertools import chain
from pathlib import Path, PureWindowsPath

import numpy as np

from .baserawio import (
    BaseRawIO,
    _signal_stream_dtype,
    _signal_channel_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)


class AlphaOmegaRawIO(BaseRawIO):
    """
    AlphaOmega MPX file format 4 reader. Handles several segments.

    A segment is a continuous record (when record starts/stops).

    Only files in current `dirname` are loaded, subfolders are not explored.

    :param dirname: folder from where to load the data
    :type dirname: str or Path-like
    :param lsx_files: list of lsx files in `dirname` referencing mpx files to
        load (optional). If None (default), read all mpx files in `dirname`
    :type lsx_files: list of strings or None
    :param prune_channels: if True removes the empty channels, defaults to True
    :type prune_channels: bool

    .. warning::
        Because channels must be gathered into coherent streams, channels names
        **must** be the default channel names in AlphaRS or Alpha LAB SNR
        software.
    """

    extensions = ["lsx", "mpx"]
    rawmode = "one-dir"

    STREAM_CHANNELS = (
        ("Accelerometer", "ACC", "ACC"),
        ("Spike Train", "SPK", "SPK"),
        ("Local Field Potential", "LFP", "LFP"),
        ("Raw Analog", "RAW", "RAW"),
        ("Analog Input", "AI", "AI"),
        ("User Defined", "UD", "UD"),
    )

    EVENT_CHANNELS = (
        ("TTL", "TTL", "TTL"),
        ("Digital Output", "DOUT", "DOUT"),
        ("User Event", "UD", "UD"),
        ("Stim Marker", "StimMarker", "StimMarker"),
        ("Digital Port", "InPort", "InPort"),
        ("Internal Detection", "Internal Detection", "Internal Detection"),
    )

    def __init__(self, dirname="", lsx_files=None, prune_channels=True):
        super().__init__(dirname=dirname)
        self.dirname = Path(dirname)

        self._lsx_files = lsx_files
        self._mpx_files = None
        self._prune_channels = prune_channels
        self._opened_files = {}
        self._ignore_unknown_datablocks = True  # internal debug property

        if self.dirname.is_dir():
            self._explore_folder()
        else:
            raise ValueError(f"{self.dirname} is not a folder")

    def _explore_folder(self):
        """
        If class was instantiated with lsx_files (list of .lsx files), load only
        the files referenced in these lsx files otherwise, load all *.mpx files
        in `dirname`.
        It does not explores the subfolders.
        """
        filenames = []
        if self._lsx_files is not None:
            for index_file in self._lsx_files:
                index_file = self.dirname / index_file
                with open(index_file, "r") as f:
                    for line in f:
                        # a line is a Microsoft Windows path. As we cannot
                        # instantiate a WindowsPath on other OS than MS
                        # Windows, we use the PureWindowsPath class
                        filename = PureWindowsPath(line.strip())
                        filename = self.dirname / filename.name
                        if not filename.is_file():
                            self.logger.warning(f"File {filename} does not exist")
                        else:
                            filenames.append(filename)
        else:
            # Load all mpx. Filter in only files in case there's a folder with
            # .mpx extension
            filenames = list(filter(lambda x: x.is_file(), self.dirname.glob("*.mpx")))
        if not filenames:
            self.logger.error(f"Found no AlphaOmega *.mpx files in {self.dirname}")
        else:
            # Sorting lexicographically should be enough to load the files in
            # correct order. This could improve slightly loading performances
            filenames.sort()
            self._mpx_files = filenames

    def _source_name(self):
        return str(self.dirname)

    def _read_file_datablocks(self, filename, prune_channels=True):
        """Read datablocks from AlphaOmega MPX file version 4.

        :param filename: the MPX filename to read datablocks from
        :type filename: Path-like object or str
        :param prune_channels: Remove references to channels and ports which
            doesn't contain any data recorded. Be careful when using this option
            with multiple-file data since it could theoretically leads to
            exception raised when data recorded in further files are merged into
            the first file pruned from these channels.
        :type prune_channels: bool
        """
        continuous_analog_channels = {}
        segmented_analog_channels = {}
        digital_channels = {}
        channel_type = {}
        stream_data_channels = {}
        ports = {}
        events = []
        unknown_datablocks = []
        with open(filename, "rb") as f:
            length, block_type = HeaderType.unpack(f.read(HeaderType.size))
            # First block is always type h and size 60
            if block_type != b"h":
                try:
                    bt = block_type.decode()
                    self.logger.error("First block must be of type h not type {bt}")
                except UnicodeDecodeError:
                    self.logger.error(
                        (
                            f"First block must be of type h not type "
                            f"{int.from_bytes(bt, 'little')} (int format)"
                        )
                    )
                raise Exception(
                    "First block of AlphaOmega MPX file format must be of type h"
                )
            if not length == 60:
                self.logger.error("First block must be of size 60 (got size {length})")
                raise Exception(
                    "First block of AlphaOmega MPX file format must be of size 60"
                )
            (
                next_block,
                version,
                hour,
                minute,
                second,
                hsecond,
                day,
                month,
                year,
                dayofweek,
                minimum_time,
                maximum_time,
                erase_count,
                map_version,
                application_name,
                resource_version,
                reserved,
            ) = SDataHeader.unpack(f.read(SDataHeader.size))
            if map_version != 4:
                self.logger.error("Only map version 4 is supported")
                raise Exception("Only AlphaOmega MPX file format 4 is supported")
            resource_version = "".join([str(b) for b in resource_version])
            try:
                resource_version = int(resource_version)
            except ValueError:
                self.logger.error(
                    f"m_ResourceVersion should be an integer (got: {resource_version}"
                )
            metadata = {
                "application_version": version,
                "application_name": decode_string(application_name),
                "record_date": datetime(
                    year, month, day, hour, minute, second, 10000 * hsecond
                ),
                "start_time": minimum_time,
                "stop_time": maximum_time,
                "erase_count": erase_count,
                "map_version": map_version,
                "resource_version": resource_version,
                "max_sample_rate": 0,
            }

            pos = 0
            while True:
                pos += length
                f.seek(pos)
                header_bytes = f.read(HeaderType.size)

                length, block_type = HeaderType.unpack(header_bytes)
                if length == 65535:
                    # The stop condition for reading MPX datablocks is base on
                    # the # length of the block: if the block has length 65535
                    # (or -1 in signed integer value) we know we have reached
                    # the end of the file
                    # We could also check that we are at the end of the file
                    # after this block
                    break

                if block_type == b"h":
                    self.logger.error(
                        "Type h block must exist only at the beginning of file"
                    )
                    raise Exception(
                        "AlphaOmega MPX file format must not have type h block after first block"
                    )

                if block_type == b"2":
                    (
                        next_block,
                        is_analog,
                        is_input,
                        channel_number,
                        *spike_color,
                    ) = Type2DataBlock.unpack(f.read(Type2DataBlock.size))
                    if is_analog and is_input:
                        (
                            mode,
                            amplitude,
                            sample_rate,
                            spike_count,
                            mode_spike,
                        ) = SDefAnalog.unpack(f.read(SDefAnalog.size))
                        mode_spike_mode = (mode_spike & 0b1111000000000000) >> 12
                        mode_spike_master = mode_spike_mode == 1
                        mode_spike_slave = mode_spike_mode == 2
                        mode_spike_linked_channel = mode_spike & 0b0000111111111111
                        mode_spike = {
                            "mode_spike_orig": mode_spike,
                            "mode_spike_master": mode_spike_master,
                            "mode_spike_slave": mode_spike_slave,
                            "mode_spike_linked_channel": mode_spike_linked_channel,
                        }
                        metadata["max_sample_rate"] = max(
                            metadata["max_sample_rate"], sample_rate * 1000
                        )
                        if amplitude <= 5:
                            # This is true for any logging software for map
                            # version >4 (specs say only for ALab SNR but AO
                            # engineer says it's true for any software)
                            amplitude = 1250000 / 2 ** 15
                        if mode == 0:
                            # continuous analog channel definition block
                            assert channel_number not in channel_type
                            channel_type[channel_number] = "continuous_analog"
                            duration, total_gain_100 = SDefContinAnalog.unpack(
                                f.read(SDefContinAnalog.size)
                            )
                            name_length = length - 38
                            name = get_name(f, name_length)
                            assert channel_number not in continuous_analog_channels
                            continuous_analog_channels[channel_number] = {
                                "spike_color": spike_color,
                                "bit_resolution": amplitude,
                                "sample_rate": sample_rate * 1000,
                                "spike_count": spike_count,
                                "mode_spike": mode_spike,
                                "duration": duration,
                                "gain": total_gain_100 / 100,
                                "name": name,
                                "positions": defaultdict(list),
                            }
                        elif mode == 1:
                            # segmented analog channel definition block
                            assert channel_number not in channel_type
                            channel_type[channel_number] = "segmented_analog"
                            (
                                pre_trig_ms,
                                post_trig_ms,
                                level_value,
                                trg_mode,
                                yes_rms,
                                total_gain_100,
                            ) = SDefLevelAnalog.unpack(f.read(SDefLevelAnalog.size))
                            name_length = length - 48
                            name = get_name(f, name_length)
                            assert channel_number not in segmented_analog_channels
                            segmented_analog_channels[channel_number] = {
                                "spike_color": spike_color,
                                "bit_resolution": amplitude,
                                "sample_rate": sample_rate * 1000,
                                "spike_count": spike_count,
                                "mode_spike": mode_spike,
                                "pre_trig_duration": pre_trig_ms / 1000,
                                "post_trig_duration": post_trig_ms / 1000,
                                "level_value": level_value,
                                "trg_mode": trg_mode,
                                "automatic_level_base_rms": yes_rms,
                                "gain": total_gain_100 / 100,
                                "name": name,
                                "positions": defaultdict(list),
                            }
                        else:
                            self.logger.error(
                                f"Unknown type 2 analog block mode: {mode}"
                            )
                            continue
                    elif is_analog == 0 and is_input == 1:
                        # digital input channel definition
                        assert channel_number not in channel_type
                        channel_type[channel_number] = "digital"
                        (
                            sample_rate,
                            save_trigger,
                            duration,
                            prev_status,
                        ) = SDefDigitalInput.unpack(f.read(SDefDigitalInput.size))
                        metadata["max_sample_rate"] = max(
                            metadata["max_sample_rate"], sample_rate * 1000
                        )
                        assert channel_number not in digital_channels
                        name_length = length - 30
                        name = get_name(f, name_length)
                        digital_channels[channel_number] = {
                            "spike_color": spike_color,
                            "sample_rate": sample_rate * 1000,
                            "save_trigger": save_trigger,
                            "duration": duration,
                            "prev_status": prev_status,
                            "name": name,
                            "samples": [],
                        }
                    else:
                        self.logger.error(
                            f"Unknown type 2 block: analog={is_analog}, input={is_input}"
                        )
                        continue
                elif block_type == b"S":
                    # stream data definition block
                    next_block, channel_number, sample_rate = SDefStream.unpack(
                        f.read(SDefStream.size)
                    )
                    metadata["max_sample_rate"] = max(
                        metadata["max_sample_rate"], sample_rate * 1000
                    )
                    assert channel_number not in channel_type
                    channel_type[channel_number] = "stream_data"
                    name_length = length - 18
                    name = get_name(f, name_length)
                    stream_data_channels[channel_number] = {
                        "sample_rate": sample_rate * 1000,
                        "name": name,
                    }
                elif block_type == b"b":
                    # digital input/output port definition block
                    board_number, port, sample_rate, prev_value = SDefPortX.unpack(
                        f.read(SDefPortX.size)
                    )
                    metadata["max_sample_rate"] = max(
                        metadata["max_sample_rate"], sample_rate * 1000
                    )
                    assert port not in channel_type
                    channel_type[port] = "port"
                    name_length = length - 18
                    name = get_name(f, name_length)
                    ports[port] = {
                        "board_number": board_number,
                        "sample_rate": sample_rate * 1000,
                        "prev_value": prev_value,
                        "name": name,
                        "samples": [],
                    }
                elif block_type == b"5":
                    # channel data block
                    unit_number, channel_number = SDataChannel.unpack(
                        f.read(SDataChannel.size)
                    )
                    assert channel_number in channel_type
                    unit_number = int.from_bytes(unit_number, "little")
                    if "analog" in channel_type[channel_number]:
                        data_length = (length - 10) / 2
                        assert int(data_length) == data_length
                        data_length = int(data_length)
                        data_start = f.tell()
                        f.seek(2 * data_length, io.SEEK_CUR)
                        if channel_type[channel_number].startswith("continuous"):
                            assert channel_number in continuous_analog_channels
                            continuous_analog_channels[channel_number]["positions"][
                                filename
                            ].append(
                                (
                                    SDataChannel_sample_id.unpack(
                                        f.read(SDataChannel_sample_id.size)
                                    )[0],
                                    data_start,
                                    data_length,
                                )
                            )
                        elif channel_type[channel_number].startswith("segmented"):
                            assert channel_number in segmented_analog_channels
                            if unit_number > 0 and unit_number <= 4:
                                segmented_analog_channels[channel_number]["positions"][
                                    filename
                                ].append(
                                    (
                                        SDataChannel_sample_id.unpack(
                                            f.read(SDataChannel_sample_id.size)
                                        )[0],
                                        data_start,
                                        data_length,
                                    )
                                )
                            elif unit_number == 0:
                                segmented_analog_channels[channel_number]["positions"][
                                    filename
                                ].append(
                                    (
                                        SDataChannel_sample_id.unpack(
                                            f.read(SDataChannel_sample_id.size)
                                        )[0],
                                        data_start,
                                        data_length,
                                    )
                                )
                            else:
                                self.logger.error(
                                    f"Unknown unit_number={unit_number} in channel data block"
                                )
                                continue
                    elif channel_type[channel_number] == "digital":
                        assert channel_number in digital_channels
                        sample_number, value = SDataChannelDigital.unpack(
                            f.read(SDataChannelDigital.size)
                        )
                        digital_channels[channel_number]["samples"].append(
                            (
                                sample_number,
                                value,
                            )
                        )
                    elif channel_type[channel_number] == "port":
                        assert channel_number in ports
                        # specifications says that for ports it should be "<Lh"
                        # but the data shows clearly "<HL"
                        value, sample_number = SDataChannelPort.unpack(
                            f.read(SDataChannelPort.size)
                        )
                        ports[channel_number]["samples"].append(
                            (
                                sample_number,
                                value,
                            )
                        )
                    else:
                        self.logger.error(
                            f"Unknown channel_type={channel_type[channel_number]} for block type 5"
                        )
                elif block_type == b"E":
                    type_event, timestamp = SAOEvent.unpack(f.read(SAOEvent.size))
                    stream_data_length = length - 8
                    events.append(
                        {
                            "timestamp": timestamp,
                            "stream_data": struct.unpack(
                                f"<{stream_data_length}s", f.read(stream_data_length)
                            )[0],
                        }
                    )
                else:
                    if not self._ignore_unknown_datablocks:
                        try:
                            bt = block_type.decode()
                            self.logger.debug(
                                f"Unknown block type: block length: {length}, block_type: {bt}"
                            )
                        except UnicodeDecodeError:
                            self.logger.debug(
                                (
                                    f"Unknown block type: block length: {length}, "
                                    f"block_type: {int.from_bytes(block_type, 'little')} "
                                    "(int format)"
                                )
                            )
                        unknown_datablocks.append(
                            {
                                "length": length,
                                "block_type": block_type,
                                "data": f.read(length),
                            }
                        )
        if prune_channels:
            to_remove = []
            for channel_id, channel in continuous_analog_channels.items():
                if not channel["positions"]:
                    to_remove.append(channel_id)
            for channel_id in to_remove:
                del continuous_analog_channels[channel_id]
            to_remove = []
            for channel_id, channel in segmented_analog_channels.items():
                if not channel["positions"]:
                    to_remove.append(channel_id)
            for channel_id in to_remove:
                del segmented_analog_channels[channel_id]
            to_remove = []
            for channel_id, channel in digital_channels.items():
                if not channel["samples"]:
                    to_remove.append(channel_id)
            for channel_id in to_remove:
                del digital_channels[channel_id]
            to_remove = []
            for port_id, port in ports.items():
                if not port["samples"]:
                    to_remove.append(port_id)
            for port_id in to_remove:
                del ports[port_id]
            del to_remove
        return (
            metadata,
            continuous_analog_channels,
            segmented_analog_channels,
            digital_channels,
            channel_type,
            stream_data_channels,
            ports,
            events,
            unknown_datablocks,
        )

    def _merge_segments(self, factor_period=1.5):
        """This method merge segments that are the same segment but comes from
        different files. The AlphaOmega data recording system split files with
        a configurable maximum time and size but with a global non-configurable
        maximum size of 1GB.
        Two segment are merged if they validate the following conditions:
            - start of next segment is less than `factor_period` (default 1.5)
            the system's sampling period + end of previous segment. We can't
            check with exactly 1 sampling period because timings are floats
            which are not exact period values
            - The two segment must also have the same record date
            (YEAR-MONTH-DAY). This could potentially lead to errors if
            recordings are longer than a day or run over the night
            - The next segment must have a datetime greater or equal than the
            previous segment (datetime : YEAR-MONTH-DAY-HOUR-MINUTE-SECOND)

        :param factor_period: how many sample period to consider the next segment
            is part of the previous one. This should be 1 <= `factor_period` < 2,
            defaults to 1.5
        :type factor_period: float
        """
        segments_to_merge = []
        for segment in self._segments:
            possible_same_segments = [
                s
                for s in self._segments
                if s["metadata"]["record_date"].date()
                == segment["metadata"]["record_date"].date()
                and s["metadata"]["record_date"]
                >= segment["metadata"]["record_date"]
                and 0
                <= (s["metadata"]["start_time"] - segment["metadata"]["stop_time"])
                <= factor_period / segment["metadata"]["max_sample_rate"]
                and s is not segment
            ]
            if len(possible_same_segments) not in (0, 1):
                self.logger.error(f"Cannot merge segments. Found {len(possible_same_segments)} segments following segment: {segment['metadata']}")
                continue
            if possible_same_segments:
                existing_merges = [s for segs in segments_to_merge for s in segs]
                if existing_merges:
                    existing_merges.sort(key=lambda x: x["metadata"]["start_time"])
                    segment = existing_merges[0]
                segments_to_merge.append((segment, possible_same_segments[0]))
        for segment, segment_to_merge in segments_to_merge:
            sample_rate = segment["metadata"]["max_sample_rate"]
            sample_rate_merge = segment_to_merge["metadata"]["max_sample_rate"]
            if sample_rate != sample_rate_merge:
                self.logger.error(f"Segment to merge has sample_rate={sample_rate_merge}, expected {sample_rate}. Continuing anyway.")
            segment["metadata"]["stop_time"] = segment_to_merge["metadata"][
                "stop_time"
            ]
            segment["metadata"]["filenames"].extend(
                segment_to_merge["metadata"]["filenames"]
            )
            for stream in segment_to_merge["streams"]:
                for channel_id in segment_to_merge["streams"][stream]:
                    try:
                        channel = segment["streams"][stream][channel_id]
                    except KeyError:
                        # there can potentially have segment without stream for this
                        # channel but the segment to merge has stream for this channel
                        segment["streams"][stream][channel_id] = segment_to_merge["streams"][stream][channel_id]
                    else:
                        for f in segment_to_merge["streams"][stream][channel_id]["positions"]:
                            channel["positions"][f].extend(
                                segment_to_merge["streams"][stream][channel_id]["positions"][f]
                            )
            for channel_id in segment_to_merge["events"]:
                try:
                    channel = segment["events"][channel_id]
                except KeyError:
                    # it's possible  that the first segment doesn't record any
                    # port channel data and the port could have been pruned
                    segment["events"][channel_id] = segment_to_merge["events"][channel_id]
                else:
                    channel["samples"].extend(
                        segment_to_merge["events"][channel_id]["samples"]
                    )
            for channel_id in segment_to_merge["spikes"]:
                try:
                    channel = segment["spikes"][channel_id]
                except KeyError:
                    # there can potentially have segment without spike for this
                    # channel but the segment to merge has spikes for this channel
                    segment["spikes"][channel_id] = segment_to_merge["spikes"][channel_id]
                else:
                    for f in segment_to_merge["spikes"][channel_id]["positions"]:
                        channel["positions"][f].extend(
                            segment_to_merge["spikes"][channel_id]["positions"][f]
                        )
            segment["ao_events"].extend(segment_to_merge["ao_events"])
            for channel_id in segment_to_merge["stream_data"]:
                # To be honest, I have no idea what is a
                # stream_data_channels so let's just overwrite it here
                segment["stream_data"][channel_id] = segment_to_merge[
                    "stream_data"
                ][channel_id]
            self._segments.remove(segment_to_merge)

    def _parse_header(self):
        segments = []
        for i, filename in enumerate(self._mpx_files):
            metadata, cac, sac, dc, ct, sd, p, e, ub = self._read_file_datablocks(
                filename, self._prune_channels
            )
            metadata["filenames"] = [filename]
            streams = {}
            for stream_name, channel_name_start, stream_id in self.STREAM_CHANNELS:
                channels = {
                    i: c
                    for i, c in cac.items()
                    if c["name"].startswith(channel_name_start)
                }
                streams[stream_id] = channels
            events = dc.copy()
            events.update(p)
            segment = {
                "metadata": metadata,
                "streams": streams,
                "events": events,
                "spikes": sac,
                "ao_events": e,
                "stream_data": sd,
            }
            segments.append(segment)
        self._segments = segments
        # We merge segments after having loading all the files because they
        # could be loaded in any order
        self._merge_segments()

        signal_streams = set(
            (stream_name, stream_id)
            for segment in self._segments
            for stream in segment["streams"]
            for stream_name, _, stream_id in self.STREAM_CHANNELS
            if stream_id == stream and segment["streams"][stream]
        )
        signal_streams = list(signal_streams)
        signal_streams.sort(key=lambda x: x[1])
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        signal_channels = set(
            (
                channel["name"],
                channel_id,
                channel["sample_rate"],
                np.dtype(np.short).name,
                "uV",
                channel["gain"] / channel["bit_resolution"],
                0,
                stream,
            )
            for segment in self._segments
            for stream in segment["streams"]
            for channel_id, channel in segment["streams"][stream].items()
        )
        signal_channels = list(signal_channels)
        signal_channels.sort(key=lambda x: (x[7], x[0]))
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        spike_channels = set(
            (
                c["name"],
                i,
                "uV",
                c["gain"] / c["bit_resolution"],
                0,
                round(c["pre_trig_duration"] * c["sample_rate"]),
                c["sample_rate"],
            )
            for segment in self._segments
            for i, c in segment["spikes"].items()
        )
        spike_channels = list(spike_channels)
        spike_channels.sort(key=lambda x: x[0])
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        event_channels = set(
            (event["name"], i, "event")
            for segment in self._segments
            for i, event in segment["events"].items()
        )
        event_channels = list(event_channels)
        event_channels.sort(key=lambda x: x[1])
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [len(self._segments)]
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        self._generate_minimal_annotations()

        bl_ann = self.raw_annotations["blocks"][0]
        bl_ann["name"] = "Block #{}{}".format(
            0, " from lsx file(s) {self._lsx_files}" if self._lsx_files else ""
        )
        bl_ann["file_origin"] = (
            "\n".join(str(self.dirname / f) for f in self._lsx_files)
            if self._lsx_files else bl_ann["file_origin"]
        )
        bl_ann["rec_datetime"] = self._segments[0]["metadata"][
            "record_date"
        ]
        for seg_index, segment in enumerate(self._segments):
            seg_ann = bl_ann["segments"][seg_index]
            seg_ann["name"] = "Seg #{} Block #0".format(seg_index)
            seg_ann["file_origin"] = "\n".join(
                str(f)
                for f in self._segments[seg_index]["metadata"][
                    "filenames"
                ]
            )
            seg_ann["rec_datetime"] = self._segments[seg_index][
                "metadata"
            ]["record_date"]
            for c_index, c in enumerate(seg_ann["signals"]):
                c = c.copy()
                c["file_origin"] = "\n".join(
                    set(
                        str(f)
                        for channels in self._segments[seg_index][
                            "streams"
                        ][c["stream_id"]].values()
                        for f in channels["positions"]
                    )
                )
                seg_ann["signals"][c_index] = c
            for e_index, e in enumerate(seg_ann["events"]):
                e = e.copy()
                e["file_origin"] = seg_ann["file_origin"]
                seg_ann["events"][e_index] = e

        # We open files and create mmap objects
        for filename in self._mpx_files:
            if filename not in self._opened_files:
                self._opened_files[filename] = {}
                self._opened_files[filename]["file"] = filename.open(mode="rb")
                self._opened_files[filename]["mmap"] = mmap.mmap(
                    self._opened_files[filename]["file"].fileno(),
                    0,
                    access=mmap.ACCESS_READ,
                )

    def __del__(self):
        # To be sure we close the file when object is deleted. Be aware that the
        # __del__ method is not necessarily called when interpreter exits so we
        # could still leave file opened. This is probably bad…
        for filename in self._opened_files:
            self._opened_files[filename]["mmap"].close()
            self._opened_files[filename]["file"].close()
        if hasattr(super(), "__del__"):
            super().__del__()

    def _segment_t_start(self, block_index, seg_index):
        return self._segments[seg_index]["metadata"]["start_time"]

    def _segment_t_stop(self, block_index, seg_index):
        return self._segments[seg_index]["metadata"]["stop_time"]

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_id = self.header["signal_streams"][stream_index]["id"]
        sizes = [
            sum(
                sample[2]
                for sample_by_file in channel["positions"].values()
                for sample in sample_by_file
            )
            for channel in self._segments[seg_index]["streams"][
                stream_id
            ].values()
        ]
        assert all(s == sizes[0] for s in sizes)
        return sizes[0]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(
        self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes
    ):
        if i_start is None:
            i_start = 0
        signal_size = self._get_signal_size(block_index, seg_index, stream_index)
        if i_stop is None or i_stop > signal_size:
            i_stop = signal_size
        stream_id = self.header["signal_streams"][stream_index]["id"]
        mask = self.header["signal_channels"]["stream_id"] == stream_id
        channel_ids = self.header["signal_channels"][mask]["id"][
            channel_indexes
        ].flatten()

        # the data refers to timestamp (see docstrings) which does not start at
        # 0 but at time_start (see type H docstring)
        first_pos = []
        for channel_id in channel_ids:
            channel_id = int(channel_id)
            first_pos.append(
                min(
                    p[0]
                    for f in self._segments[seg_index]["streams"][stream_id][
                        channel_id
                    ]["positions"].values()
                    for p in f
                )
            )

        # now we need to get all file data block indexes for the signal and
        # timestamp we want
        file_chunks = defaultdict(list)
        for i, channel_id in enumerate(channel_ids):
            channel_id = int(channel_id)
            effective_start = i_start + first_pos[i]
            effective_stop = i_stop + first_pos[i]
            for filename, positions in self._segments[seg_index]["streams"][
                stream_id
            ][channel_id]["positions"].items():
                file_chunks[filename].extend(
                    [
                        (i, *p)
                        for p in positions
                        if p[0] + p[2] > effective_start and p[0] < effective_stop
                    ]
                )

        # we almost surely loaded more than asked (because the blocks are not
        # contiguous) so we need to know where to cut the results
        slices_channels = []
        for channel_index, channel_id in enumerate(channel_ids):
            channel_id = int(channel_id)
            min_pos = (
                min(
                    p[0]
                    for f in file_chunks.values()
                    for i, *p in f
                    if i == channel_index
                )
                - first_pos[channel_index]
            )
            max_pos = (
                max(
                    p[0] + p[2]
                    for f in file_chunks.values()
                    for i, *p in f
                    if i == channel_index
                )
                - first_pos[channel_index]
            )
            slices_channels.append((min_pos, max_pos))
        min_size = min(s[0] for s in slices_channels)
        max_size = max(s[1] for s in slices_channels)
        sigs = np.ndarray((max_size - min_size, len(channel_ids)), dtype=np.short)

        for filename in file_chunks:
            # we sort by chunk position in the file because we want to optimize
            # IO access and possibly read in sequential access. This is mainly
            # true for hard drives but shouldn't hurt flash memory
            file_chunks[filename].sort(key=lambda x: x[2])
        for filename in file_chunks:
            for channel_index, chunk_index, file_position, chunk_size in file_chunks[
                filename
            ]:
                sig_offset = chunk_index - first_pos[channel_index] - min_size
                sigs[
                    sig_offset : sig_offset + chunk_size, channel_index
                ] = np.frombuffer(
                    self._opened_files[filename]["mmap"],
                    dtype=np.short,
                    count=chunk_size,
                    offset=file_position,
                )
        return sigs[i_start - min_size : i_stop - min_size, :]

    def _spike_count(self, block_index, seg_index, spike_channel_index):
        spike_id = int(self.header["spike_channels"]["id"][spike_channel_index])
        nb_spikes = sum(
            len(f) for f in self._segments[seg_index]["spikes"][spike_id]["positions"].values()
        )
        return nb_spikes

    def _get_spike_timestamps(
        self, block_index, seg_index, spike_channel_index, t_start, t_stop
    ):
        if self._spike_count(block_index, seg_index, spike_channel_index):
            spike_id = int(self.header["spike_channels"]["id"][spike_channel_index])
            spikes = self._segments[seg_index]["spikes"][spike_id]
            if t_start is None:
                t_start = self._segment_t_start(block_index, seg_index)
            if t_stop is None:
                t_stop = self._segment_t_stop(block_index, seg_index)
            effective_start = t_start * spikes["sample_rate"]
            effective_stop = t_stop * spikes["sample_rate"]
            timestamps = np.array([p[0] for f in spikes["positions"].values() for p in f if effective_start <= p[0] <= effective_stop])
        else:
            timestamps = np.array([], dtype=np.uint32)
        return timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        # let's hope every spike channels have the same sampling rate
        sample_rate = int(self.header["spike_channels"]["wf_sampling_rate"][0])
        spike_timestamps = spike_timestamps.astype(dtype) / sample_rate
        return spike_timestamps

    def _get_spike_raw_waveforms(
        self, block_index, seg_index, spike_channel_index, t_start, t_stop
    ):
        spike_id = int(self.header["spike_channels"]["id"][spike_channel_index])
        #  nb_spikes = self._spike_count(block_index, seg_index, spike_channel_index)
        nb_spikes = self._get_spike_timestamps(block_index, seg_index, spike_channel_index, t_start, t_stop).size
        spikes = self._segments[seg_index]["spikes"][spike_id]
        spike_length = {p[2] for f in spikes["positions"].values() for p in f}
        assert len(spike_length) == 1
        spike_length = spike_length.pop()
        waveforms = np.ndarray((nb_spikes, spike_length), dtype=np.short)
        if t_start is None:
            t_start = self._segment_t_start(block_index, seg_index)
        if t_stop is None:
            t_stop = self._segment_t_stop(block_index, seg_index)
        effective_start = t_start * spikes["sample_rate"]
        effective_stop = t_stop * spikes["sample_rate"]
        i = 0
        for filename in spikes["positions"]:
            for timestamp, file_position, length in spikes["positions"][filename]:
                if effective_start <= timestamp <= effective_stop:
                    waveforms[i, :length] = np.frombuffer(
                        self._opened_files[filename]["mmap"],
                        dtype=np.short,
                        count=length,
                        offset=file_position,
                    )
                    i += 1
        waveforms.shape = nb_spikes, 1, spike_length
        return waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        event_id = int(self.header["event_channels"]["id"][event_channel_index])
        try:
            nb_events = len(self._segments[seg_index]["events"][event_id]["samples"])
        except KeyError:
            # No event in this segment
            nb_events = 0
        return nb_events

    def _get_event_timestamps(
        self, block_index, seg_index, event_channel_index, t_start, t_stop
    ):
        if self._event_count(block_index, seg_index, event_channel_index):
            event_id = int(self.header["event_channels"]["id"][event_channel_index])
            event = self._segments[seg_index]["events"][event_id]

            timestamps = np.array([s[0] for s in event["samples"]], dtype=np.uint32)
            if t_start is None:
                t_start = self._segment_t_start(block_index, seg_index)
            if t_stop is None:
                t_stop = self._segment_t_stop(block_index, seg_index)
            effective_start = t_start * event["sample_rate"]
            effective_stop = t_stop * event["sample_rate"]
            mask = (effective_start <= timestamps) & (timestamps <= effective_stop)
            timestamps = timestamps[mask]
            labels = np.array([s[1] for s in event["samples"]], dtype="U")
            labels = labels[mask]
        else:
            timestamps = np.array([], dtype=np.uint32)
            labels = np.array([], dtype="U")
        return timestamps, None, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        event_id = int(self.header["event_channels"]["id"][event_channel_index])
        for segment in self._segments:
            if event_id in segment["events"]:
                event = segment["events"][event_id]
                break
        event_times = event_timestamps.astype(dtype) / event["sample_rate"]
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        pass


def decode_string(encoded_string):
    """According to AlphaOmega engineers, all strings are NULL terminated and
    ASCII encoded"""
    return encoded_string[: encoded_string.find(b"\x00")].decode("ascii")


def get_name(f, name_length):
    """Helper function to read a string from opened binary file"""
    return decode_string(struct.unpack(f"<{name_length}s", f.read(name_length))[0])


HeaderType = struct.Struct("<Hc")
"""All datablocks start with the same common structure:
    - length (ushort): the size (in bytes) of the datablock
    - datablock_type (char): the type of datablock (described after)

There are two main datablock types:
    1. definition datablocks (types H, 2, S, B): these datablock describe metadata of
       channels and ports
    2. data datablocks (types 5, E): these datablocks contains records data of the
       previously defined channels and ports

Other datablocks exist in the data but are ignored in this implementation as per the
specification: "Any block type other than the ones described below should be
ignored."

All data is little-endian (hence the '<' in Struct calls).
"""

SDataHeader = struct.Struct("<xlhBBBBBBHBxddlB10s4sxl")
"""Type H datablock is unique and the first datablock. It specifies file metadatas:
    - alignment byte: ignore
    - next_datablock (long): offset of the next datablock from beginning of file
    - version (short): program version number
    - hour (unsigned char): start hour of data saving
    - minute (unsigned char): start minute of the data saving
    - second (unsigned char): start second of the data saving
    - hsecond (unsigned char): start 100th of seconds of the data saving
    - day (unsigned char): start day of the data saving
    - month (unsigned char): start month of the data saving
    - year (unsigned short): start year of the data saving
    - dayofweek (unsigned char): start day of week of the data saving
    - minimum_time (double): minimal acquisition time in seconds
    - maximum_time (double): maximal acquisition time in seconds
    - erase_count (long): number of erase messages in the file
    - map_version (unsigned char): MPX file format version (only 4 if supported
      by this implementation)
    - application_name (10-char string): name of the recording application.
      Should be "ARS" for AlphaRS hardware or "ALab SNR" for AlphaLab SnR hardware
    - resource_version (4-char string): C++ version used to compile recording
      software: each byte is read separately as a char and concatenated and cast
      into an int to create the original value (e.g.: b"\x00\x01\x00\x00" -> 100)
    - alignment byte: ignore
    - reserved (long): not used
"""

Type2DataBlock = struct.Struct("<xlhhhxBBB")
"""There are two (or three, depending on your interpretation) types of Type 2
datablocks:
    1. Analog channels definition
      1.a. Continuous (RAW, LFP, SPK)
      1.b. Segmented (SEG)
    2. Digital channels definition (mainly TTL)

all type 2 datablocks starts with the same structure:
    - alignment byte: ignore
    - next_datablock (long)
    - is_analog (short): 0=Digital, 1=Analog
    - is_input (short): 0=Output, 1=Input
    - channel_number: the (unique) channel number identifier from the recording software
    - alignment byte: ignore, this and the following bytes are COLORREF
      convention from Windef.h: hex value: 0x00bbggrr
    - spike_color_blue (unsigned char)
    - spike_color_green (unsigned char)
    - spike_color_red (unsigned char)
"""
SDefAnalog = struct.Struct("<hffhh")
"""
Then if is_analog and is_input:
    - mode (short): 0=Continuous, 1=(Level or Segmented)
    - amplitude (float): bit resolution. For MAP file version 4 if amplitude < 5
                         amplitude = 1_250_000/2**15
    - sample_rate (float): in kHz (or more precisely in kilosample per seconds)
    - spike_count (short): size of each data datablock (short) + timestamp (unsigned long)
    - mode_spike (2 bytes): read as hex data 0xMCCC:
        - M: 1=Master, 2=Slave
        - CCC: linked channel
        Be careful here, the first byte cover MC and the second byte the last
        part of the linked channel CC
"""
SDefContinAnalog = struct.Struct("<fh")
"""
    Then if mode is Continuous:
        - duration (float): unknown
        - total_gain_100 (short): 100 x total_gain applied to this channel
        - name (n-char string): channel name; n=length-38
"""
SDefLevelAnalog = struct.Struct("<ffhhhh")
"""
    Then if mode is Level of Segmented:
        - pre_trig_msec (float): number of milliseconds before segment trigger
        - post_trig_msec (float): number of milliseconds after segment trigger
        - level_value (short): unknown (should be the level that trigger a
          spike detection)
        - trg_mode (short): unknown (level or template mode?)
        - yes_rms (short): 1 if automatic level calculation base on RMS
        - total_gain_100 (short): see above
        - name (n-char string): channel name; n=length-48
"""
SDefDigitalInput = struct.Struct("<fhfh")
"""
Then if not is_analog and is_input
    - sample_rate (float): see above
    - save_trigger (short): unknown
    - duration (float): unknown
    - prev_status (short): not used (I guess the previously recorded value,
      could be useful for segments merged from several files)
    - name (n-char string): channel name; n=length-30
"""
"""All other combinations of is_analog and is_input are unknown (not described
in the specification and therefore not supported by this implementation)"""

SDefStream = struct.Struct("<xlhf")
"""Type S datablock: Stream data definition:
    - alignment byte: ignore
    - next_datablock (long): see above
    - channel_number (short): see above
    - sample_rate (float): see above
    - name (n-char string): channel name; n=length-18

"""

SDefPortX = struct.Struct("<xiifH")
"""Type b datablock: Digital Input/Output port definition:
    - board_number (int): not sure… maybe in case of multiple connected
      AlphaOmega setups?
    - port (int): unique port number identifier
    - sample_rate (float): in kHz (see above)
    - prev_value (ushort): not used (see above)
    - name (n-char string): port name; n=length-18
"""

SDataChannel = struct.Struct("<ch")
"""Type 5 datablock: channel data:
    - unit_number (char): for analog segmented channels: unit number; 0=Level,
      1=Unit1, 2=Unit2, 3=Unit3, 4=Unit4
    - channel_number: the previously defined channel_number (in one of the
      definition datablocks)
"""
SDataChannel_sample_id = struct.Struct("<L")
"""
Then for analog channels:
    - sample_value (n-short): array of data values
    - first_sample_number (ulong): for continuous channels: first sample number
      in the channel records. This is the timestamp (see :attr:`SAOEvent`) of
      the first sample in the data datablocks
"""
SDataChannelDigital = struct.Struct("<Lh")
"""
Then for digital channels:

    - sample_number (ulong): the sample number of the event
    - value (short): value of the event
"""
SDataChannelPort = struct.Struct("<HL")
"""
Then for digital ports:
    - value (ushort)
    - sample_number (ulong)

.. warning::
    The specification says that for port channels it should be the same as for
    digital channels but the data (and AO engineers) says otherwise. According
    to AlphaOmega engineer, this could change in a future logging software
    release and could break this reader.
"""

SAOEvent = struct.Struct("<cL")
"""Type E: stream data datablock:
    - type_event (char): event type only b"S" for now
    - timestamp (ulong): a counter initialized at 0 at hardware boot and that
      advances at the sampling rate of the system
    - stream_data (n-char): Stream Data - spec says: "Refer to SreamFormat.h or
                            use dll to decipher this stream of data"; n=length-8
"""
