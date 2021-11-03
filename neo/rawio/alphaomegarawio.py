import io
import logging
import os
import struct

from collections import defaultdict
from datetime import datetime
from pathlib import Path

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
    Handles several blocks and segments.

    A block is a recording define in a *.lsx file. AlphaRS software creates a
    new file each time the software is opened.
    A segment is a continuous record (when record starts/stops).

    Because channels must be gathered into coherent streams, channels names must
    be the default channel names in AlphaRS software (or Alpha LAB SNR).
    """
    extensions = ["lsx", "mpx"]
    rawmode = "one-dir"

    STREAM_CHANNELS = [
        ("Accelerometer", "ACC", 1),
        ("Spike Train", "SPK", 2),
        #  ("Segmented Analog", "SEG", 3),
        ("Local Field Potential", "LFP", 4),
        ("Raw Analog", "RAW", 5),
        ("Analog Input", "AI", 6),
    ]

    def __init__(self, dirname):
        #  super().__init__(self)
        BaseRawIO.__init__(self)
        self.dirname = Path(dirname)
        if self.dirname.is_dir():
            self._explore_folder()
        else:
            self.logger.error(f"{self.dirname} is not a folder")

    def _explore_folder(self):
        """
        Explores an AlphaOmega folder and tries to load the index files (*.lsx) to
        get the data files (*.mpx). If the folder contains only *.mpx files, it will
        load them without splitting them into blocks.
        It does not explores the folder recursively.
        """
        self.filenames = defaultdict(list)

        # first check if there is any *.mpx files
        filenames = list(self.dirname.glob("*.mpx"))
        filenames.sort()
        if not filenames:
            self.logger.error(f"Found no AlphaOmega *.mpx files in {self.dirname}")
        else:
            index_files = list(self.dirname.glob("*.lsx"))
            # sort index files by creation order
            index_files.sort()
            if not index_files:
                self.logger.warning("No *.lsx files found. Will try to load all *.mpx files")
                self.filenames[""].extend(filenames)
            else:
                for index_file in index_files:
                    with open(index_file, "r") as f:
                        for line in f:
                            filename = line.strip().split("\\")[-1]  # line is a Windows path
                            filename = self.dirname / filename
                            if not filename.is_file():
                                self.logger.warning(f"File {filename} does not exist")
                            else:
                                self.filenames[index_file.name].append(filename)
                                filenames.remove(filename)
                if filenames:
                    self.logger.info("Some *.mpx files not referenced. Will try to load them.")
                    self.filenames[""].extend(filenames)

    def _source_name(self):
        return str(self.dirname)

    def _read_file_blocks(self, filename):
        continuous_analog_channels = {}
        segmented_analog_channels = {}
        digital_channels = {}
        channel_type = {}
        stream_data = {}
        ports = {}
        events = []
        unknown_blocks = []
        with open(filename, "rb") as f:
            length, block_type = HeaderType.unpack(f.read(HeaderType.size))
            # First block is always type h and size 60
            if block_type != b"h":
                try:
                    bt = block_type.decode()
                    self.logger.error("First block must be of type h not type {bt}")
                except UnicodeDecodeError:
                    self.logger.error("First block must be of type h not type {int.from_bytes(bt, 'little')} (int format)")
                raise Exception("First block of AlphaOmega MPX file format must be of type h")
            if not length == 60:
                self.logger.error("First block must be of size 60 (got size {length})")
                raise Exception("First block of AlphaOmega MPX file format must be of size 60")
            next_block, version, hour, minute, second, hsecond, day, month, year, dayofweek, minimum_time, maximum_time, erase_count, map_version, application_name, resource_version, reserved = SDataHeader.unpack(f.read(SDataHeader.size))
            if map_version != 4:
                self.logger.error("Only map version 4 is supported")
                raise Exception("Only AlphaOmega MPX file format 4 is supported")
            metadata = {
                "application_version": version,
                "application_name": decode_string(application_name),
                "record_date": datetime(year, month, day, hour, minute, second, 10000 * hsecond),
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
                if len(header_bytes) < HeaderType.size:
                    break

                length, block_type = HeaderType.unpack(header_bytes)

                if block_type == b"h":
                    self.logger.error("Type h block must exist only at the beginning of file")
                    raise Exception("AlphaOmega MPX file format must not have type h block after first block")

                if block_type == b"2":
                    next_block, is_analog, is_input, channel_number, spike_color = Type2Block.unpack(f.read(Type2Block.size))
                    if is_analog and is_input:
                        mode, amplitude, sample_rate, spike_count, mode_spike = SDefAnalog.unpack(f.read(SDefAnalog.size))
                        metadata["max_sample_rate"] = max(metadata["max_sample_rate"], sample_rate * 1000)
                        if amplitude <= 5:
                            self.logger.warning("Should do something about that")
                        if metadata["application_name"] == "ALab SNR":
                            self.logger.warning("Should do something about that also")
                        if mode == 0:
                            # continuous analog channel definition block
                            assert channel_number not in channel_type
                            channel_type[channel_number] = "continuous_analog"
                            duration, total_gain_100 = SDefContinAnalog.unpack(f.read(SDefContinAnalog.size))
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
                                "positions": [],
                            }
                        elif mode == 1:
                            # segmented analog channel definition block
                            assert channel_number not in channel_type
                            channel_type[channel_number] = "segmented_analog"
                            pre_trigm_sec, post_trigm_sec, level_value, trg_mode, yes_rms, total_gain_100 = SDefLevelAnalog.unpack(f.read(SDefLevelAnalog.size))
                            name_length = length - 48
                            name = get_name(f, name_length)
                            assert channel_number not in segmented_analog_channels
                            segmented_analog_channels[channel_number] = {
                                "spike_color": spike_color,
                                "bit_resolution": amplitude,
                                "sample_rate": sample_rate * 1000,
                                "spike_count": spike_count,
                                "mode_spike": mode_spike,
                                "pre_trigm_sec": pre_trigm_sec,
                                "post_trigm_sec": post_trigm_sec,
                                "level_value": level_value,
                                "trg_mode": trg_mode,
                                "automatic_level_base_rms": yes_rms,
                                "gain": total_gain_100 / 100,
                                "name": name,
                                "positions": [],
                            }
                        else:
                            self.logger.error(f"Unknown type 2 analog block mode: {mode}")
                            continue
                    elif is_analog == 0 and is_input == 1:
                        # digital input channel definition
                        assert channel_number not in channel_type
                        channel_type[channel_number] = "digital"
                        sample_rate, save_trigger, duration, prev_status = SDefDigitalInput.unpack(f.read(SDefDigitalInput.size))
                        metadata["max_sample_rate"] = max(metadata["max_sample_rate"], sample_rate * 1000)
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
                        self.logger.error(f"Unknown type 2 block: analog={is_analog}, input={is_input}")
                        continue
                elif block_type == b"S":
                    # stream data definition block
                    next_block, channel_number, sample_rate = SDefStream.unpack(f.read(SDefStream.size))
                    metadata["max_sample_rate"] = max(metadata["max_sample_rate"], sample_rate * 1000)
                    assert channel_number not in channel_type
                    channel_type[channel_number] = "stream_data"
                    name_length = length - 18
                    name = get_name(f, name_length)
                    stream_data[channel_number] = {
                        "sample_rate": sample_rate * 1000,
                        "name": name,
                    }
                elif block_type == b"b":
                    # digital input/output port definition block
                    board_number, port, sample_rate, prev_value = SDefPortX.unpack(f.read(SDefPortX.size))
                    metadata["max_sample_rate"] = max(metadata["max_sample_rate"], sample_rate * 1000)
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
                    unit_number, channel_number = SDataChannel.unpack(f.read(SDataChannel.size))
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
                            continuous_analog_channels[channel_number]["positions"].append({
                                "filename": filename,
                                "first_sample_number": SDataChannel_sample_id.unpack(f.read(SDataChannel_sample_id.size))[0],
                                "data_start": data_start,
                                "data_length": data_length,
                            })
                        elif channel_type[channel_number].startswith("segmented"):
                            assert channel_number in segmented_analog_channels
                            if unit_number > 0 and unit_number <= 4:
                                segmented_analog_channels[channel_number]["positions"].append({
                                    "filename": filename,
                                    "first_template_point": SDataChannel_sample_id.unpack(f.read(SDataChannel_sample_id.size))[0],
                                    "data_start": data_start,
                                    "data_length": data_length,
                                })
                            elif unit_number == 0:
                                segmented_analog_channels[channel_number]["positions"].append({
                                    "filename": filename,
                                    "level_crossing_point": SDataChannel_sample_id.unpack(f.read(SDataChannel_sample_id.size))[0],
                                    "data_start": data_start,
                                    "data_length": data_length,
                                })
                            else:
                                self.logger.error(f"Unknown unit_number={unit_number} in channel data block")
                                continue
                    elif channel_type[channel_number] == "digital":
                        assert channel_number in digital_channels
                        sample_number, value = SDataChannel_sample_value.unpack(f.read(SDataChannel_sample_value.size))
                        digital_channels[channel_number]["samples"].append({
                            "sample_number": sample_number,
                            "value": value,
                        })
                    elif channel_type[channel_number] == "port":
                        assert channel_number in ports
                        # specifications says that for ports it should be "<Lh"
                        # but the data shows clearly "<HL"
                        value, sample_number = SDataChannel_value_sample.unpack(f.read(SDataChannel_value_sample.size))
                        ports[channel_number]["samples"].append({
                            "sample_number": sample_number,
                            "value": value,
                        })
                    else:
                        self.logger.error(f"Unknown channel_type={channel_type[channel_number]} for block type 5")
                elif block_type == b"E":
                    type_event, timestamp = SAOEvent.unpack(f.read(SAOEvent.size))
                    stream_data_length = length - 8
                    events.append({
                        "timestamp": timestamp,
                        "stream_data": struct.unpack(f"<{stream_data_length}s", f.read(stream_data_length))[0],
                    })
                else:
                    try:
                        bt = block_type.decode()
                        self.logger.debug(f"Unknown block type: block length: {length}, block_type: {bt}")
                    except UnicodeDecodeError:
                        self.logger.debug(f"Unknown block type: block length: {length}, block_type: {int.from_bytes(block_type, 'little')} (int format)")
                    unknown_blocks.append({
                        "length": length,
                        "block_type": block_type,
                        "data": f.read(length),
                    })
        return (
            metadata,
            continuous_analog_channels,
            segmented_analog_channels,
            digital_channels,
            channel_type,
            ports,
            events,
            unknown_blocks
        )

    def _parse_header(self):
        blocks = []
        for index_file, filenames in self.filenames.items():
            segments = []
            continuous_analog_channels = {}
            for i, filename in enumerate(filenames):
                metadata, cac, sac, dc, ct, p, e, ub = self._read_file_blocks(filename)
                if i > 0:
                    assert prev_metadata["max_sample_rate"] == metadata["max_sample_rate"]
                    if (prev_metadata["stop_time"] + 1 / prev_metadata["max_sample_rate"]) >= metadata["start_time"] and prev_metadata["stop_time"] < metadata["start_time"]:  # we're in the same segment
                        segment = segments[-1]
                        segment["metadata"]["stop_time"] = metadata["stop_time"]
                        segment["metadata"]["filenames"].append(filename)
                        assert segment["channel_type"] == ct
                        for channel_id in cac:
                            assert channel_id in segment["continuous_analog_channels"]
                            segment["continuous_analog_channels"][channel_id]["positions"].extend(cac[channel_id]["positions"])
                        for channel_id in sac:
                            assert channel_id in segment["segmented_analog_channels"]
                            segment["segmented_analog_channels"][channel_id]["positions"].extend(sac[channel_id]["positions"])
                        for channel_id in dc:
                            assert channel_id in segment["digital_channels"]
                            segment["digital_channels"][channel_id]["samples"].extend(dc[channel_id]["samples"])
                        for port in p:
                            assert port in segment["ports"]
                            segment["ports"][port]["samples"].extend(p[port]["samples"])
                        segment["events"].extend(e)
                        segment["unknown_blocks"].extend(ub)
                    else:
                        metadata["filenames"] = [filename]
                        segments.append({
                            "metadata": metadata,
                            "continuous_analog_channels": cac,
                            "segmented_analog_channels": sac,
                            "digital_channels": dc,
                            "channel_type": ct,
                            "ports": p,
                            "events": e,
                            "unknown_blocks": ub,
                        })
                else:
                    metadata["filenames"] = [filename]
                    segments.append({
                        "metadata": metadata,
                        "continuous_analog_channels": cac,
                        "segmented_analog_channels": sac,
                        "digital_channels": dc,
                        "channel_type": ct,
                        "ports": p,
                        "events": e,
                        "unknown_blocks": ub,
                    })
                prev_metadata = metadata
            blocks.append(segments)
        self.blocks = blocks

        # this expect default channel names
        #  signal_streams_names = set(
            #  channel["name"][:3].strip()
            #  for block in self.blocks
            #  for segment in block
            #  for channel_type in ("continuous_analog_channels", "segmented_analog_channels")
            #  for channel in segment[channel_type].values()
            #  if channel["positions"]
        #  )
        #  signal_streams = []
        #  for stream in signal_streams_names:
            #  stream_id = stream
            #  signal_streams.append((stream_id, stream_id))
        #  self.signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        self.STREAM_CHANNELS.sort(key=lambda x: x[2])  # just in case
        signal_streams = []
        for stream_name, channel_name_start, stream_id in self.STREAM_CHANNELS:
            non_empty_channels = [
                channel
                for block in self.blocks
                for segment in block
                for channel_type in ("continuous_analog_channels", "segmented_analog_channels")
                for channel in segment[channel_type].values()
                if channel["positions"] and channel["name"].startswith(channel_name_start)
            ]
            if non_empty_channels:
                signal_streams.append((stream_name, stream_id))
        signal_streams.sort(key=lambda x: x[1])
        self.signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        signal_channels_ids = set(
            channel_id
            for block in self.blocks
            for segment in block
            for channel_type in ("continuous_analog_channels", "segmented_analog_channels")
            for channel_id in segment[channel_type]
            if segment[channel_type][channel_id]["positions"]
        )
        signal_channels = []
        for block in self.blocks:
            for segment in block:
                #  for channel_type in ("continuous_analog_channels", "segmented_analog_channels"):
                for channel_id, channel in segment["continuous_analog_channels"].items():
                    if channel_id in signal_channels_ids:
                        stream_id = [
                            s_id for s_name, c_name, s_id in self.STREAM_CHANNELS
                            if channel["name"].startswith(c_name)
                        ]
                        assert len(stream_id) == 1
                        stream_id = stream_id[0]
                        signal_channels.append((
                            channel["name"],
                            channel_id,
                            channel["sample_rate"],
                            np.dtype(np.short).name,
                            "uV",
                            channel["gain"] / channel["bit_resolution"],
                            0.,
                            stream_id,
                        ))
                        signal_channels_ids.difference_update((channel_id,))
        self.signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        self.spike_channels = np.array([], dtype=_spike_channel_dtype)
        self.event_channels = np.array([], dtype=_event_channel_dtype)

        self.header = {}
        self.header["nb_block"] = len(self.filenames)
        self.header["nb_segment"] = [len(segment) for segment in self.blocks]
        self.header["signal_streams"] = self.signal_streams
        self.header["signal_channels"] = self.signal_channels
        self.header["spike_channels"] = self.spike_channels
        self.header["event_channels"] = self.event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return self.blocks[block_index][seg_index]["metadata"]["start_time"]

    def _segment_t_stop(self, block_index, seg_index):
        return self.blocks[block_index][seg_index]["metadata"]["stop_time"]

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream = self.header["signal_streams"][stream_index]["id"]
        mask = self.header["signal_channels"]["stream_id"] == stream
        names = self.header["signal_channels"]["name"][mask]
        channels = [
            channel for channel in self.blocks[block_index][seg_index]["continuous_analog_channels"].values()
            if channel["name"] in names
        ]
        sizes = [sum(sample["data_length"] for sample in channel["positions"]) for channel in channels]
        assert all(s == sizes[0] for s in sizes)
        return sizes[0]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return self._segment_t_start(block_index, seg_index)

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._get_signal_size(block_index, seg_index, stream_index)
        stream = self.header["signal_streams"][stream_index]["id"]
        mask = self.header["signal_channels"]["stream_id"] == stream


    def _spike_count(self, block_index, seg_index, spike_channel_index):
        pass

    def _get_spike_timestamps(self, block_index, seg_index, spike_channel_index,
                              t_start, t_stop):
        pass

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        pass

    def _get_spike_raw_waveforms(self, block_index, seg_index, spike_channel_index,
                                 t_start, t_stop):
        pass

    def _event_count(self, block_index, seg_index, event_channel_index):
        pass

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index,
                              t_start, t_stop):
        pass

    def _rescale_event_timestamp(self, event_timestamps, dtype,
                                 event_channel_index):
        pass

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        pass


def decode_string(encoded_string):
    return encoded_string[:encoded_string.find(b"\x00")].decode()


def get_name(f, name_length):
    return decode_string(struct.unpack(f"<{name_length}s", f.read(name_length))[0])



HeaderType = struct.Struct("<Hc")

# type h
SDataHeader = struct.Struct("<xlhBBBBBBHBxddlB10s4sxl")

# type2
Type2Block = struct.Struct("<xlhhhl")
SDefAnalog = struct.Struct("<hffhh")
SDefContinAnalog = struct.Struct("<fh")
SDefLevelAnalog =  struct.Struct("<ffhhhh")
SDefDigitalInput = struct.Struct("<fhfh")

# type S
SDefStream = struct.Struct("<xlhf")

# type b
SDefPortX = struct.Struct("<xiifH")

# type 5
SDataChannel = struct.Struct("<ch")
SDataChannel_sample_id = struct.Struct("<L")
SDataChannel_sample_value = struct.Struct("<Lh")
SDataChannel_value_sample = struct.Struct("<HL")

# type E
SAOEvent = struct.Struct("<cL")
