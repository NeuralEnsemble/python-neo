import io
import logging
import struct

from datetime import datetime

from .baserawio import BaseRawIO


logger = logging.getLogger("neo")


class AlphaOmegaRawIO(BaseRawIO):
    extensions = ["mpx"]
    rawmode = "one-file"

    def __init__(self, filename=""):
        #  super().__init__(self)
        BaseRawIO.__init__(self)
        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        _continuous_analog_channels = {}
        _segmented_analog_channels = {}
        _digital_channels = {}
        _channel_type = {}
        stream_data = {}
        _ports = {}
        _events = []
        _unknown_blocks = []
        with open(self.filename, "rb") as f:
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
            self.metadata = {
                "application_version": version,
                "application_name": decode_string(application_name),
                "record_date": datetime(year, month, day, hour, minute, second, 10000 * hsecond),
                "start_time": minimum_time,
                "stop_time": maximum_time,
                "erase_count": erase_count,
                "map_version": map_version,
                "resource_version": resource_version,
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
                        if amplitude <= 5:
                            self.logger.warning("Should do something about that")
                        if self.metadata["application_name"] == "ALab SNR":
                            self.logger.warning("Should do something about that also")
                        if mode == 0:
                            # continuous analog channel definition block
                            assert channel_number not in _channel_type
                            _channel_type[channel_number] = "continuous_analog"
                            duration, total_gain_100 = SDefContinAnalog.unpack(f.read(SDefContinAnalog.size))
                            name_length = length - 38
                            name = get_name(f, name_length)
                            assert channel_number not in _continuous_analog_channels
                            _continuous_analog_channels[channel_number] = {
                                "spike_color": spike_color,
                                "bit_resolution": amplitude,
                                "sample_rate": sample_rate,
                                "spike_count": spike_count,
                                "mode_spike": mode_spike,
                                "duration": duration,
                                "gain": total_gain_100 / 100,
                                "name": name,
                                "positions": [],
                            }
                        elif mode == 1:
                            # segmented analog channel definition block
                            assert channel_number not in _channel_type
                            _channel_type[channel_number] = "segmented_analog"
                            pre_trigm_sec, post_trigm_sec, level_value, trg_mode, yes_rms, total_gain_100 = SDefLevelAnalog.unpack(f.read(SDefLevelAnalog.size))
                            name_length = length - 48
                            name = get_name(f, name_length)
                            assert channel_number not in _segmented_analog_channels
                            _segmented_analog_channels[channel_number] = {
                                "spike_color": spike_color,
                                "bit_resolution": amplitude,
                                "sample_rate": sample_rate,
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
                        assert channel_number not in _channel_type
                        _channel_type[channel_number] = "digital"
                        sample_rate, save_trigger, duration, prev_status = SDefDigitalInput.unpack(f.read(SDefDigitalInput.size))
                        assert channel_number not in _digital_channels
                        name_length = length - 30
                        name = get_name(f, name_length)
                        _digital_channels[channel_number] = {
                            "spike_color": spike_color,
                            "sample_rate": sample_rate,
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
                    assert channel_number not in _channel_type
                    _channel_type[channel_number] = "stream_data"
                    name_length = length - 18
                    name = get_name(f, name_length)
                    stream_data[channel_number] = {
                        "sample_rate": sample_rate,
                        "name": name,
                    }
                elif block_type == b"b":
                    # digital input/output port definition block
                    board_number, port, sampling_rate, prev_value = SDefPortX.unpack(f.read(SDefPortX.size))
                    assert port not in _channel_type
                    _channel_type[port] = "port"
                    name_length = length - 18
                    name = get_name(f, name_length)
                    _ports[port] = {
                        "board_number": board_number,
                        "sample_rate": sample_rate,
                        "prev_value": prev_value,
                        "name": name,
                        "samples": [],
                    }
                elif block_type == b"5":
                    # channel data block
                    unit_number, channel_number = SDataChannel_part1.unpack(f.read(SDataChannel_part1.size))
                    assert channel_number in _channel_type
                    unit_number = int.from_bytes(unit_number, "little")
                    if "analog" in _channel_type[channel_number]:
                        data_length = (length - 10) / 2
                        assert int(data_length) == data_length
                        data_length = int(data_length)
                        data_start = f.tell()
                        f.seek(2 * data_length, io.SEEK_CUR)
                        if _channel_type[channel_number].startswith("continuous"):
                            assert channel_number in _continuous_analog_channels
                            _continuous_analog_channels[channel_number]["positions"].append({
                                "first_sample_number": struct.unpack("<L", f.read(4))[0],
                                "data_start": data_start,
                                "data_length": data_length,
                            })
                        elif _channel_type[channel_number].startswith("segmented"):
                            assert channel_number in _segmented_analog_channels
                            if unit_number > 0 and unit_number <= 4:
                                _segmented_analog_channels[channel_number]["positions"].append({
                                    "first_template_point": struct.unpack("<L", f.read(4))[0],
                                    "data_start": data_start,
                                    "data_length": data_length,
                                })
                            elif unit_number == 0:
                                _segmented_analog_channels[channel_number]["positions"].append({
                                    "level_crossing_point": struct.unpack("<L", f.read(4))[0],
                                    "data_start": data_start,
                                    "data_length": data_length,
                                })
                            else:
                                self.logger.error(f"Unknown unit_number={unit_number} in channel data block")
                                continue
                    elif _channel_type[channel_number] == "digital":
                        assert channel_number in _digital_channels
                        sample_number, value = struct.unpack("<Lh", f.read(6))
                        _digital_channels[channel_number]["samples"].append({
                            "sample_number": sample_number,
                            "value": value,
                        })
                    elif _channel_type[channel_number] == "port":
                        assert channel_number in _ports
                        sample_number, value = struct.unpack("<Lh", f.read(6))
                        _ports[channel_number]["samples"].append({
                            "sample_number": sample_number,
                            "value": value,
                        })
                    else:
                        self.logger.error(f"Unknown _channel_type={_channel_type[channel_number]} for block type 5")
                elif block_type == b"E":
                    type_event, timestamp = SAOEvent.unpack(f.read(SAOEvent.size))
                    stream_data_length = length - 8
                    _events.append({
                        "timestamp": timestamp,
                        "stream_data": struct.unpack(f"<{stream_data_length}s", f.read(stream_data_length)),
                    })
                else:
                    try:
                        bt = block_type.decode()
                        logger.debug(f"Unknown block type: block length: {length}, block_type: {bt}")
                    except UnicodeDecodeError:
                        logger.debug(f"Unknown block type: block length: {length}, block_type: {int.from_bytes(block_type, 'little')} (int format)")
                    _unknown_blocks.append({
                        "length": length,
                        "block_type": block_type,
                        "data": f.read(length),
                    })
        self._continuous_analog_channels = _continuous_analog_channels
        self._segmented_analog_channels = _segmented_analog_channels
        self._digital_channels = _digital_channels
        self._channel_type = _channel_type
        self._ports = _ports
        self._events = _events
        self._unknown_blocks = _unknown_blocks

        self.header = {}

    def _segment_t_start(self, block_index, seg_index):
        pass

    def _segment_t_stop(self, block_index, seg_index):
        pass

    def _get_signal_size(self, block_index, seg_index, stream_index):
        pass

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        pass

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        pass

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


def read_blocks(filename):
    blocks = []
    with open(filename, "rb") as f:
        length, block_type = HeaderType.unpack(f.read(HeaderType.size))
        try:
            bt = block_type.decode()
            logger.debug(f"block length: {length}, block_type: {bt}")
        except UnicodeDecodeError:
            logger.debug("block length: {length}, block_type: {} (int format)".format(int.from_bytes(block_type, "little")))
        if not block_type == b"h":
            try:
                t = block_type.decode()
                raise Exception(f"First block must be type h, got type {t}")
            except UnicodeDecodeError:
                raise Exception("First block must be of type h, got type {} (int value)".format(int.from_bytes(block_type, "little")))
        if not length == 60:
            raise Exception("First block must be of length 60")
        next_block, version, hour, minute, second, hsecond, day, month, year, dayofweek, minimum_time, maximum_time, erase_count, map_version, application_name, resource_version, reserved = SDataHeader.unpack(f.read(length - 3))
        blocks.append({
            "length": length,
            "block_type": block_type,
            "next_block": next_block,
            "version": version,
            "hour": hour,
            "minute": minute,
            "second": second,
            "hsecond": hsecond,
            "day": day,
            "month": month,
            "year": year,
            "dayofweek": dayofweek,
            "minimum_time": minimum_time,
            "maximum_time": maximum_time,
            "erase_count": erase_count,
            "map_version": map_version,
            "application_name": application_name,
            "resource_version": resource_version,
            "reserved": reserved,
        })

        pos = 0
        while True:
            #  if f.tell() > next_block:
                #  # the previous block has no next_block
                #  f.seek(length, 1)
            #  else:
                #  f.seek(next_block)
            pos += length
            f.seek(pos)
            header_data = f.read(HeaderType.size)
            if len(header_data) < 3:
                break
            length, block_type = HeaderType.unpack(header_data)
            try:
                bt = block_type.decode()
                logger.debug(f"block length: {length}, block_type: {bt}")
            except UnicodeDecodeError:
                logger.debug("block length: {length}, block_type: {} (int format)".format(int.from_bytes(block_type, "little")))
            blocks.append({
                "length": length,
                "block_type": block_type,
            })
            assert block_type != b"h"
            if block_type == b"2":
                next_block, is_analog, is_input, channel_number, spike_color = Type2Block.unpack(f.read(Type2Block.size))
                blocks[-1].update({
                    "next_block": next_block,
                    "is_analog": is_analog,
                    "is_input": is_input,
                    "channel_number": channel_number,
                    "spike_color": spike_color,
                })
                if is_analog and is_input:
                    mode, amplitude, sample_rate, spike_count, mode_spike = SDefAnalog.unpack(f.read(SDefAnalog.size))
                    if amplitude <= 5:
                        self.logger.warning("Should do something about that")
                    if blocks[0]["application_name"].startswith(b"ALab SNR"):
                        self.logger.warning("Should do something about that also")
                    blocks[-1].update({
                        "mode": mode,
                        "amplitude": amplitude,
                        "sample_rate": sample_rate,
                        "spike_count": spike_count,
                        "mode_spike": mode_spike,
                    })
                    if mode == 0:
                        duration, total_gain_100 = SDefContinAnalog.unpack(f.read(SDefContinAnalog.size))
                        blocks[-1].update({
                            "duration": duration,
                            "total_gain_100": total_gain_100,
                        })
                        name_length = length - 38
                        blocks[-1]["name"] = struct.unpack(f"<{name_length}s", f.read(name_length))
                    elif mode == 1:
                        pre_trigm_sec, post_trigm_sec, level_value, trg_mode, yes_rms, total_gain_100 = SDefLevelAnalog.unpack(f.read(SDefLevelAnalog.size))
                        blocks[-1].update({
                            "pre_trigm_sec": pre_trigm_sec,
                            "post_trigm_sec": post_trigm_sec,
                            "level_value": level_value,
                            "trg_mode": trg_mode,
                            "yes_rms": yes_rms,
                            "total_gain_100": total_gain_100,
                        })
                        name_length = length - 48
                        blocks[-1]["name"] = struct.unpack(f"<{name_length}s", f.read(name_length))
                    else:
                        logger.warning(f"Unknown mode: {mode}")
                        del blocks[-1]
                        continue
                elif is_analog == 0 and is_input == 1:
                    sample_rate, save_trigger, duration, prev_status = SDefDigitalInput.unpack(f.read(SDefDigitalInput.size))
                    blocks[-1].update({
                        "sample_rate": sample_rate,
                        "save_trigger": save_trigger,
                        "duration": duration,
                        "prev_status": prev_status,
                    })
                    name_length = length - 30
                    blocks[-1]["name"] = struct.unpack(f"<{name_length}s", f.read(name_length))
                else:
                    logger.warning(f"Unknown type2 block: analog: {is_analog}, input: {is_input}")
                    del blocks[-1]
                    continue
            elif block_type == b"S":
                next_block, number, sample_rate = SDefStream.unpack(f.read(SDefStream.size))
                blocks[-1].update({
                    "next_block": next_block,
                    "number": number,
                    "sample_rate": sample_rate,
                })
                name_length = length - 18
                blocks[-1]["name"] = struct.unpack(f"<{name_length}s", f.read(name_length))
            elif block_type == b"b":
                board_number, port, sampling_rate, prev_value = SDefPortX.unpack(f.read(SDefPortX.size))
                blocks[-1].update({
                    "board_number": board_number,
                    "port": port,
                    "sampling_rate": sampling_rate,
                    "prev_value": prev_value,
                })
                name_length = length - 18
                blocks[-1]["name"] = struct.unpack(f"<{name_length}s", f.read(name_length))
            elif block_type == b"5":
                unit_number, number = SDataChannel_part1.unpack(f.read(SDataChannel_part1.size))
                blocks[-1].update({
                    "unit_number": unit_number,
                    "number": number,
                })
                possible_blocks = [b for b in blocks if b["block_type"] == b"2" and b["channel_number"] == number]
                possible_ports = [b for b in blocks if b["block_type"] == b"b" and b["port"] == number]
                assert (len(possible_blocks) == 1 and len(possible_ports) == 0) or (len(possible_ports) == 1 and len(possible_blocks) == 0)
                if possible_blocks:
                    b = possible_blocks[0]
                    if b["is_analog"]:
                        data_length = (length - 10) / 2
                        assert int(data_length) == data_length
                        data_length = int(data_length)
                        prod = True
                        if prod:
                            data = struct.unpack(f"<{data_length}h", f.read(2 * data_length))
                            blocks[-1]["data"] = data
                        else:
                            logger.info(f"data length: {data_length}")
                            f.seek(2 * data_length, 1)
                        first_sample_number = struct.unpack("<L", f.read(4))
                        blocks[-1]["first_sample_number"] = first_sample_number
                    else:
                        sample_number, value = struct.unpack("<Lh", f.read(6))
                        blocks[-1].update({
                            "sample_number": sample_number,
                            "value": value,
                        })
                elif possible_ports:
                    b = possible_ports[0]
                    sample_number, value = struct.unpack("<Lh", f.read(6))
                    blocks[-1].update({
                        "sample_number": sample_number,
                        "value": value,
                    })
            elif block_type == b"E":
                type_event, timestamp = SAOEvent.unpack(f.read(SAOEvent.size))
                blocks[-1].update({
                    "type_event": type_event,
                    "timestamp": timestamp,
                })
                stream_data_length = length - 8
                blocks[-1]["stream_data"] = struct.unpack(f"<{stream_data_length}s", f.read(stream_data_length))
            else:
                del blocks[-1]
                try:
                    bt = block_type.decode()
                    logger.warning(f"Unknown block_type: {bt} (size: {length})")
                except UnicodeDecodeError:
                    logger.warning("Unknown block type: {} (int format) (size: {})".format(int.from_bytes(block_type, "little"), length))
    return blocks


HeaderType = struct.Struct("<Hc")

# type h
#  SDataHeader = struct.Struct("<HcxlhBBBBBBHBxddlB10s4sxl")
SDataHeader = struct.Struct("<xlhBBBBBBHBxddlB10s4sxl")

# type2
Type2Block = struct.Struct("<xlhhhl")
SDefAnalog = struct.Struct("<hffhh")
SDefContinAnalog = struct.Struct("<fh")
# SDefContinAnalog_m_Name =
SDefLevelAnalog =  struct.Struct("<ffhhhh")
# SDefLevelAnalog_m_Name
SDefDigitalInput = struct.Struct("<fhfh")
# SDefDigitalInput_n_Name

# type S
SDefStream = struct.Struct("<xlhf")
# SDefStream_m_Name

# type b
SDefPortX = struct.Struct("<xiifH")
# SDefPortX_m_Name

# type 5
SDataChannel_part1 = struct.Struct("<ch")
SDataChannel_data = struct.Struct("<h")
SDataChannel_part2 = struct.Struct("<LLh")

# type E
SAOEvent = struct.Struct("<cL")

# SOAEvent_m_StreamData
