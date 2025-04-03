"""
Class for reading data from a SpikeGLX system  (NI-DAQ for neuropixel probe)
See https://billkarsh.github.io/SpikeGLX/


Here an adaptation of the spikeglx tools into the neo rawio API.

Note that each pair of ".bin"/."meta" files is represented as a stream of channels
that share the same sampling rate.
It will be one AnalogSignal multi channel at neo.io level.

Contrary to other implementations this IO reads the entire folder and subfolder and:
  * deals with severals segment based on the `_gt0`, `_gt1`, `_gt2`, etc postfixes
  * deals with all signals "imec0", "imec1" for neuropixel probes and also
     external signal like"nidq". This is the "device"
  * For imec device both "ap" and "lf" are extracted so one device have several "streams"

Note:
  * there are several versions depending the neuropixel probe generation (`1.x`/`2.x`/`3.x`)
    Here, we assume that the `meta` file has the same structure across all generations.
    This need so be checked.
    This IO is developed based on neuropixel generation 2.0, single shank recordings.


# Not implemented yet in this reader:
  * contact SpkeGLX developer to see how to deal with absolute t_start when several segment
  * contact SpkeGLX developer to understand the last channel SY0 function
  * better handling of annotations at object level by sub group of device (after rawio change)
  * better handling of channel location


See:
https://billkarsh.github.io/SpikeGLX/
https://billkarsh.github.io/SpikeGLX/#offline-analysis-tools
https://billkarsh.github.io/SpikeGLX/#metadata-guides
https://github.com/SpikeInterface/spikeextractors/blob/master/spikeextractors/extractors/spikeglxrecordingextractor/spikeglxrecordingextractor.py

This reader handle:

imDatPrb_type=1 (NP 1.0)
imDatPrb_type=21 (NP 2.0, single multiplexed shank)
imDatPrb_type=24 (NP 2.0, 4-shank)
imDatPrb_type=1030 (NP 1.0-NHP 45mm SOI90 - NHP long 90um wide, staggered contacts)
imDatPrb_type=1031 (NP 1.0-NHP 45mm SOI125 - NHP long 125um wide, staggered contacts)
imDatPrb_type=1032 (NP 1.0-NHP 45mm SOI115 / 125 linear - NHP long 125um wide, linear contacts)
imDatPrb_type=1022 (NP 1.0-NHP 25mm - NHP medium)
imDatPrb_type=1015 (NP 1.0-NHP 10mm - NHP short)

Author : Samuel Garcia
Some functions are copied from Graham Findlay
"""

from pathlib import Path
import os
import re

import numpy as np

from .baserawio import (
    BaseRawWithBufferApiIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)
from .utils import get_memmap_shape


class SpikeGLXRawIO(BaseRawWithBufferApiIO):
    """
    Class for reading data from a SpikeGLX system

    Parameters
    ----------
    dirname: str, default: ''
        The spikeglx folder containing meta/bin files
    load_sync_channel: bool, default: False
        The last channel (SY0) of each stream is a fake channel used for synchronisation
    load_channel_location: bool, default: False
        If True probeinterface is used to load the channel locations from the directory

    Notes
    -----
    * This IO reads the entire folder and subfolders locating the `.bin` and `.meta` files
    * Handles gates and triggers as segments (based on the `_gt0`, `_gt1`, `_t0` , `_t1` in filenames)
    * Handles all signals coming from different acquisition cards ("imec0", "imec1", etc) in a typical
        PXIe chassis setup and also external signal like "nidq".
    * For imec devices both "ap" and "lf" are extracted so even a one device setup will have several "streams"

    Examples
    --------
    >>> import neo.rawio
    >>> reader = neo.rawio.SpikeGLXRawIO(dirname='path/to/data/')
    >>> reader.parse_header()
    >>> print(reader)
    >>> raw_chunk = reader.get_analogsignal_chunk(block_index=0,
                                                  seg_index=0,
                                                  i_start=None,
                                                  i_stop=None,
                                                  stream_index=0)
    """

    # file formats used by spikeglxio
    extensions = ["meta", "bin"]
    rawmode = "one-dir"

    def __init__(self, dirname="", load_sync_channel=False, load_channel_location=False):
        BaseRawWithBufferApiIO.__init__(self)
        self.dirname = dirname
        self.load_sync_channel = load_sync_channel
        self.load_channel_location = load_channel_location

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        self.signals_info_list = scan_files(self.dirname)

        # sort stream_name by higher sampling rate first
        srates = {info["stream_name"]: info["sampling_rate"] for info in self.signals_info_list}
        stream_names = sorted(list(srates.keys()), key=lambda e: srates[e])[::-1]
        nb_segment = np.unique([info["seg_index"] for info in self.signals_info_list]).size

        self.signals_info_dict = {}
        # one unique block
        self._buffer_descriptions = {0: {}}
        self._stream_buffer_slice = {}
        for info in self.signals_info_list:
            seg_index, stream_name = info["seg_index"], info["stream_name"]
            key = (seg_index, stream_name)
            if key in self.signals_info_dict:
                raise KeyError(f"key {key} is already in the signals_info_dict")
            self.signals_info_dict[key] = info

            buffer_id = stream_name
            block_index = 0

            if seg_index not in self._buffer_descriptions[0]:
                self._buffer_descriptions[block_index][seg_index] = {}

            self._buffer_descriptions[block_index][seg_index][buffer_id] = {
                "type": "raw",
                "file_path": info["bin_file"],
                "dtype": "int16",
                "order": "C",
                "file_offset": 0,
                "shape": get_memmap_shape(info["bin_file"], "int16", num_channels=info["num_chan"], offset=0),
            }

        # create channel header
        signal_buffers = []
        signal_streams = []
        signal_channels = []
        for stream_name in stream_names:
            # take first segment
            info = self.signals_info_dict[0, stream_name]

            buffer_id = stream_name
            buffer_name = stream_name
            signal_buffers.append((buffer_name, buffer_id))

            stream_id = stream_name

            signal_streams.append((stream_name, stream_id, buffer_id))

            # add channels to global list
            for local_chan in range(info["num_chan"]):
                chan_name = info["channel_names"][local_chan]
                chan_id = f"{stream_name}#{chan_name}"
                signal_channels.append(
                    (
                        chan_name,
                        chan_id,
                        info["sampling_rate"],
                        "int16",
                        info["units"],
                        info["channel_gains"][local_chan],
                        info["channel_offsets"][local_chan],
                        stream_id,
                        buffer_id,
                    )
                )

            # all channel by dafult unless load_sync_channel=False
            self._stream_buffer_slice[stream_id] = None
            # check sync channel validity
            if "nidq" not in stream_name:
                if not self.load_sync_channel and info["has_sync_trace"]:
                    # the last channel is remove from the stream but not from the buffer
                    last_chan = signal_channels[-1]
                    last_chan = last_chan[:-2] + ("", buffer_id)
                    signal_channels = signal_channels[:-1] + [last_chan]
                    self._stream_buffer_slice[stream_id] = slice(0, -1)
                if self.load_sync_channel and not info["has_sync_trace"]:
                    raise ValueError("SYNC channel is not present in the recording. " "Set load_sync_channel to False")

        signal_buffers = np.array(signal_buffers, dtype=_signal_buffer_dtype)
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # No events
        event_channels = []
        # This is true only in case of 'nidq' stream
        for stream_name in stream_names:
            if "nidq" in stream_name:
                info = self.signals_info_dict[0, stream_name]
                if len(info["digital_channels"]) > 0:
                    # add event channels
                    for local_chan in info["digital_channels"]:
                        chan_name = local_chan
                        chan_id = f"{stream_name}#{chan_name}"
                        event_channels.append((chan_name, chan_id, "event"))
                    # add events_memmap
                    data = np.memmap(info["bin_file"], dtype="int16", mode="r", offset=0, order="C")
                    data = data.reshape(-1, info["num_chan"])
                    # The digital word is usually the last channel, after all the individual analog channels
                    extracted_word = data[:, len(info["analog_channels"])]
                    self._events_memmap = np.unpackbits(extracted_word.astype(np.uint8)[:, None], axis=1)
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # deal with nb_segment and t_start/t_stop per segment

        self._t_starts = {stream_name: {} for stream_name in stream_names}
        self._t_stops = {seg_index: 0.0 for seg_index in range(nb_segment)}

        for stream_name in stream_names:
            for seg_index in range(nb_segment):
                info = self.signals_info_dict[seg_index, stream_name]

                frame_start = float(info["meta"]["firstSample"])
                sampling_frequency = info["sampling_rate"]
                t_start = frame_start / sampling_frequency

                self._t_starts[stream_name][seg_index] = t_start
                t_stop = info["sample_length"] / info["sampling_rate"]
                self._t_stops[seg_index] = max(self._t_stops[seg_index], t_stop)

        # fille into header dict
        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [nb_segment]
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        # insert some annotation at some place
        self._generate_minimal_annotations()
        self._generate_minimal_annotations()

        for seg_index in range(nb_segment):
            seg_ann = self.raw_annotations["blocks"][0]["segments"][seg_index]
            seg_ann["name"] = f"Segment {seg_index}"

            for c, signal_stream in enumerate(signal_streams):
                stream_name = signal_stream["name"]
                sig_ann = self.raw_annotations["blocks"][0]["segments"][seg_index]["signals"][c]

                if self.load_channel_location:
                    # need probeinterface to be installed
                    import probeinterface

                    info = self.signals_info_dict[seg_index, stream_name]
                    if "imroTbl" in info["meta"] and info["stream_kind"] == "ap":
                        # only for ap channel
                        probe = probeinterface.read_spikeglx(info["meta_file"])
                        loc = probe.contact_positions
                        if self.load_sync_channel:
                            # one fake channel  for "sys0"
                            loc = np.concatenate((loc, [[0.0, 0.0]]), axis=0)
                        for ndim in range(loc.shape[1]):
                            sig_ann["__array_annotations__"][f"channel_location_{ndim}"] = loc[:, ndim]

    def _segment_t_start(self, block_index, seg_index):
        return 0.0

    def _segment_t_stop(self, block_index, seg_index):
        return self._t_stops[seg_index]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        stream_name = self.header["signal_streams"][stream_index]["name"]
        return self._t_starts[stream_name][seg_index]

    def _event_count(self, event_channel_idx, block_index=None, seg_index=None):
        timestamps, _, _ = self._get_event_timestamps(block_index, seg_index, event_channel_idx, None, None)
        return timestamps.size

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start=None, t_stop=None):
        timestamps, durations, labels = [], None, []
        info = self.signals_info_dict[0, "nidq"]  # There are no events that are not in the nidq stream
        dig_ch = info["digital_channels"]
        if len(dig_ch) > 0:
            event_data = self._events_memmap
            channel = dig_ch[event_channel_index]
            ch_idx = 7 - int(channel[2:])  # They are in the reverse order
            this_stream = event_data[:, ch_idx]
            this_rising = np.where(np.diff(this_stream) == 1)[0] + 1
            this_falling = (
                np.where(np.diff(this_stream) == 255)[0] + 1
            )  # because the data is in unsigned 8 bit, -1 = 255!
            if len(this_rising) > 0:
                timestamps.extend(this_rising)
                labels.extend([f"{channel} ON"] * len(this_rising))
            if len(this_falling) > 0:
                timestamps.extend(this_falling)
                labels.extend([f"{channel} OFF"] * len(this_falling))
        timestamps = np.asarray(timestamps)
        if len(labels) == 0:
            labels = np.asarray(labels, dtype="U1")
        else:
            labels = np.asarray(labels)
        return timestamps, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        info = self.signals_info_dict[0, "nidq"]  # There are no events that are not in the nidq stream
        event_times = event_timestamps.astype(dtype) / float(info["sampling_rate"])
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        return None

    def _get_analogsignal_buffer_description(self, block_index, seg_index, buffer_id):
        return self._buffer_descriptions[block_index][seg_index][buffer_id]


def scan_files(dirname):
    """
    Scan for pairs of `.bin` and `.meta` files and return information about it.

    After exploring the folder, the segment index (`seg_index`) is construct as follow:
      * if only one `gate_num=0` then `trigger_num` = `seg_index`
      * if only one `trigger_num=0` then `gate_num` = `seg_index`
      * if both are increasing then seg_index increased by gate_num, trigger_num order.
    """
    info_list = []

    for root, dirs, files in os.walk(dirname):
        for file in files:
            if not file.endswith(".meta"):
                continue
            meta_filename = Path(root) / file
            bin_filename = meta_filename.with_suffix(".bin")
            if meta_filename.exists() and bin_filename.exists():
                meta = read_meta_file(meta_filename)
                info = extract_stream_info(meta_filename, meta)

                info["meta_file"] = str(meta_filename)
                info["bin_file"] = str(bin_filename)
                info_list.append(info)

    if len(info_list) == 0:
        raise FileNotFoundError(f"No appropriate combination of .meta and .bin files were detected in {dirname}")

    # This sets non-integers values before integers
    normalize = lambda x: x if isinstance(x, int) else -1

    # Segment index is determined by the gate_num and trigger_num in that order
    def get_segment_tuple(info):
        # Create a key from the normalized gate_num and trigger_num
        gate_num = normalize(info.get("gate_num"))
        trigger_num = normalize(info.get("trigger_num"))
        return (gate_num, trigger_num)

    unique_segment_tuples = {get_segment_tuple(info) for info in info_list}
    sorted_keys = sorted(unique_segment_tuples)

    # Map each unique key to a corresponding index
    segment_tuple_to_segment_index = {key: idx for idx, key in enumerate(sorted_keys)}

    for info in info_list:
        info["seg_index"] = segment_tuple_to_segment_index[get_segment_tuple(info)]

    for info in info_list:
        # device_kind is imec, nidq
        if info.get("device_kind") == "imec":
            info["device_index"] = info["device"].split("imec")[-1]
        else:
            info["device_index"] = ""  # TODO: Handle multi nidq case, maybe use meta["typeNiEnabled"]

    # Define stream base on device_kind [imec|nidq], device_index and stream_kind [ap|lf] for imec
    # Stream format is "{device_kind}{device_index}.{stream_kind}"
    for info in info_list:
        device_kind = info["device_kind"]
        device_index = info["device_index"]
        stream_kind = f".{info['stream_kind']}" if info["stream_kind"] else ""
        stream_name = f"{device_kind}{device_index}{stream_kind}"

        info["stream_name"] = stream_name

    return info_list


def parse_spikeglx_fname(fname):
    """
    Parse recording identifiers from a SpikeGLX style filename.

    spikeglx naming follow this rules:
    https://github.com/billkarsh/SpikeGLX/blob/15ec8898e17829f9f08c226bf04f46281f106e5f/Markdown/UserManual.md#gates-and-triggers

    Example file name structure:
    Consider the filenames: `Noise4Sam_g0_t0.nidq.bin` or `Noise4Sam_g0_t0.imec0.lf.bin`
    The filenames consist of 3 or 4 parts separated by `.`
    1. "Noise4Sam_g0_t0" will be the `name` variable. This choosen by the user at recording time.
    2. "g0" is the "gate_num"
    3. "t0" is the "trigger_num"
    4. "nidq" or "imec0" will give the `device`
    5. "lf" or "ap" will be the `stream_kind`
       `stream_name` variable is the concatenation of `device.stream_kind`

    If CatGT is used, then the trigger numbers in the file names ("t0"/"t1"/etc.)
    will be renamed to "tcat". In this case, the parsed "trigger_num" will be set to "cat".

    This function is copied/modified from Graham Findlay.

    Notes:
        * Sometimes the original file name is modified by the user and "_g0_" or "_t0_"
          are manually removed. In that case gate_name and trigger_num will be None.

    Parameters
    ---------
    fname: str
        The filename to parse without the extension, e.g. "my-run-name_g0_t1.imec2.lf"

    Returns
    -------
    run_name: str
        The run name, e.g. "my-run-name".
    gate_num: int or None
        The gate identifier, e.g. 0.
    trigger_num: int | str or None
        The trigger identifier, e.g. 1. If CatGT is used, then the trigger_num will be set to "cat".
    device: str
        The probe identifier, e.g. "imec2"
    stream_kind: str or None
        The data type identifier, "lf" or "ap" or None
    """
    re_standard = re.findall(r"(\S*)_g(\d*)_t(\d*)\.(\S*).(ap|lf)", fname)
    re_tcat = re.findall(r"(\S*)_g(\d*)_tcat.(\S*).(ap|lf)", fname)
    re_nidq = re.findall(r"(\S*)_g(\d*)_t(\d*)\.(\S*)", fname)
    if len(re_standard) == 1:
        # standard case with probe
        run_name, gate_num, trigger_num, device, stream_kind = re_standard[0]
    elif len(re_tcat) == 1:
        # tcat case
        run_name, gate_num, device, stream_kind = re_tcat[0]
        trigger_num = "cat"
    elif len(re_nidq) == 1:
        # case for nidaq
        run_name, gate_num, trigger_num, device = re_nidq[0]
        stream_kind = None
    else:
        # the naming do not correspond lets try something more easy
        # example: sglx_xxx.imec0.ap
        re_else = re.findall(r"(\S*)\.(\S*).(ap|lf)", fname)
        re_else_nidq = re.findall(r"(\S*)\.(\S*)", fname)
        if len(re_else) == 1:
            run_name, device, stream_kind = re_else[0]
            gate_num, trigger_num = None, None
        elif len(re_else_nidq) == 1:
            # easy case for nidaq, example: sglx_xxx.nidq
            run_name, device = re_else_nidq[0]
            gate_num, trigger_num, stream_kind = None, None, None
        else:
            raise ValueError(f"Cannot parse filename {fname}")

    if gate_num is not None:
        gate_num = int(gate_num)
    if trigger_num is not None and trigger_num != "cat":
        trigger_num = int(trigger_num)

    return (run_name, gate_num, trigger_num, device, stream_kind)


def read_meta_file(meta_file):
    """parse the meta file"""
    with open(meta_file, mode="r") as f:
        lines = f.read().splitlines()

    meta = {}
    # Fix taken from: https://github.com/SpikeInterface/probeinterface/blob/
    # 19d6518fbc67daca71aba5e99d8aa0d445b75eb7/probeinterface/io.py#L649-L662
    for line in lines:
        split_lines = line.split("=")
        if len(split_lines) != 2:
            continue
        k, v = split_lines
        if k.startswith("~"):
            # replace by the list
            k = k[1:]
            v = v[1:-1].split(")(")[1:]
        meta[k] = v

    return meta


def extract_stream_info(meta_file, meta):
    """Extract info from the meta dict"""

    num_chan = int(meta["nSavedChans"])
    if "snsApLfSy" in meta:
        # AP and LF meta have this field
        ap, lf, sy = [int(s) for s in meta["snsApLfSy"].split(",")]
        has_sync_trace = sy == 1
    else:
        # NIDQ case
        has_sync_trace = False

    # This is the original name that the file had. It might not match the current name if the user changed it
    bin_file_path = meta["fileName"]
    fname = Path(bin_file_path).stem

    run_name, gate_num, trigger_num, device, stream_kind = parse_spikeglx_fname(fname)

    if "imec" in fname.split(".")[-2]:
        device = fname.split(".")[-2]
        stream_kind = fname.split(".")[-1]
        units = "uV"
        # please note the 1e6 in gain for this uV

        # metad['imroTbl'] contain two gain per channel  AP and LF
        # except for the last fake channel
        per_channel_gain = np.ones(num_chan, dtype="float64")
        if (
            "imDatPrb_type" not in meta
            or meta["imDatPrb_type"] == "0"
            or meta["imDatPrb_type"] in ("1015", "1016", "1022", "1030", "1031", "1032", "1100", "1121", "1300")
        ):
            # This work with NP 1.0 case with different metadata versions
            # https://github.com/billkarsh/SpikeGLX/blob/15ec8898e17829f9f08c226bf04f46281f106e5f/Markdown/Metadata_30.md
            if stream_kind == "ap":
                index_imroTbl = 3
            elif stream_kind == "lf":
                index_imroTbl = 4
            for c in range(num_chan - 1):
                v = meta["imroTbl"][c].split(" ")[index_imroTbl]
                per_channel_gain[c] = 1.0 / float(v)
            gain_factor = float(meta["imAiRangeMax"]) / 512
            channel_gains = gain_factor * per_channel_gain * 1e6
        elif meta["imDatPrb_type"] in ("21", "24", "2003", "2004", "2013", "2014"):
            # This work with NP 2.0 case with different metadata versions
            # https://github.com/billkarsh/SpikeGLX/blob/15ec8898e17829f9f08c226bf04f46281f106e5f/Markdown/Metadata_30.md#imec
            # We allow also LF streams for NP2.0 because CatGT can produce them
            # See: https://github.com/SpikeInterface/spikeinterface/issues/1949
            if "imChan0apGain" in meta:
                per_channel_gain[:-1] = 1 / float(meta["imChan0apGain"])
            else:
                per_channel_gain[:-1] = 1 / 80.0
            max_int = int(meta["imMaxInt"]) if "imMaxInt" in meta else 8192
            gain_factor = float(meta["imAiRangeMax"]) / max_int
            channel_gains = gain_factor * per_channel_gain * 1e6
        else:
            raise NotImplementedError("This meta file version of spikeglx" " is not implemented")
    else:
        device = fname.split(".")[-1]
        stream_kind = ""
        units = "V"
        channel_gains = np.ones(num_chan)

        # there are differents kinds of channels with different gain values
        mn, ma, xa, dw = [int(e) for e in meta["snsMnMaXaDw"].split(sep=",")]
        per_channel_gain = np.ones(num_chan, dtype="float64")
        per_channel_gain[0:mn] = 1.0 / float(meta["niMNGain"])
        per_channel_gain[mn : mn + ma] = 1.0 / float(meta["niMAGain"])
        # this scaling come from the code in this zip
        # https://billkarsh.github.io/SpikeGLX/Support/SpikeGLX_Datafile_Tools.zip
        # in file readSGLX.py line76
        # this is equivalent of 2**15
        gain_factor = float(meta["niAiRangeMax"]) / 32768
        channel_gains = per_channel_gain * gain_factor

    probe_slot = meta.get("imDatPrb_slot", None)
    probe_port = meta.get("imDatPrb_port", None)
    probe_dock = meta.get("imDatPrb_dock", None)

    info = {}
    info["fname"] = fname
    info["meta"] = meta
    for k in ("niSampRate", "imSampRate"):
        if k in meta:
            info["sampling_rate"] = float(meta[k])
    info["num_chan"] = num_chan

    info["sample_length"] = int(meta["fileSizeBytes"]) // 2 // num_chan
    info["gate_num"] = gate_num
    info["trigger_num"] = trigger_num
    info["device"] = device
    info["stream_kind"] = stream_kind
    # All non-production probes (phase 3B onwards) have "typeThis", otherwise revert to file parsing
    info["device_kind"] = meta.get("typeThis", device.split(".")[0])
    info["units"] = units
    info["channel_names"] = [txt.split(";")[0] for txt in meta["snsChanMap"]]
    info["channel_gains"] = channel_gains
    info["channel_offsets"] = np.zeros(info["num_chan"])
    info["has_sync_trace"] = has_sync_trace
    info["probe_slot"] = int(probe_slot) if probe_slot else None
    info["probe_port"] = int(probe_port) if probe_port else None
    info["probe_dock"] = int(probe_dock) if probe_dock else None

    if "nidq" in device:
        info["digital_channels"] = []
        info["analog_channels"] = [channel for channel in info["channel_names"] if not channel.startswith("XD")]
        # Digital/event channels are encoded within the digital word, so that will need more handling
        if meta.get("niXDChans1", "") != "":
            nixd_chans1_items = meta["niXDChans1"].split(",")
            for item in nixd_chans1_items:
                if ":" in item:
                    start, end = map(int, item.split(":"))
                    info["digital_channels"].extend([f"XD{i}" for i in range(start, end + 1)])
                else:
                    info["digital_channels"].append(f"XD{int(item)}")
    return info
