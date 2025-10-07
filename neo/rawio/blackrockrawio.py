"""
Module for reading data from files in the Blackrock in raw format.

This work is based on:
  * Chris Rodgers - first version
  * Michael Denker, Lyuba Zehl - second version
  * Samuel Garcia - third version
  * Lyuba Zehl, Michael Denker - fourth version
  * Samuel Garcia, Julia Srenger - fifth version
  * Chadwick Boulay - FileSpec 3.0 and 3.0-PTP

This IO supports reading only.
This IO is able to read:
  * the nev file which contains spikes
  * ns1, ns2, .., ns6 files that contain signals at different sampling rates

This IO can handle the following Blackrock file specifications:
  * 2.1
  * 2.2
  * 2.3
  * 3.0
  * 3.0 with PTP timestamps (Gemini systems)

The neural data channels are 1 - 128.
The analog inputs are 129 - 144. (129 - 137 AC coupled, 138 - 144 DC coupled)

spike- and event-data; 30000 Hz in NEV file.
"ns1": "analog data: 500 Hz",
"ns2": "analog data: 1000 Hz",
"ns3": "analog data: 2000 Hz",
"ns4": "analog data: 10000 Hz",
"ns5": "analog data: 30000 Hz",
"ns6": "analog data: 30000 Hz (no digital filter)"


The possible file extensions of the Cerebus system and their content:
    ns1: contains analog data; sampled at 500 Hz (+ digital filters)
    ns2: contains analog data; sampled at 1000 Hz (+ digital filters)
    ns3: contains analog data; sampled at 2000 Hz (+ digital filters)
    ns4: contains analog data; sampled at 10000 Hz (+ digital filters)
    ns5: contains analog data; sampled at 30000 Hz (+ digital filters)
    ns6: contains analog data; sampled at 30000 Hz (no digital filters)
    nev: contains spike- and event-data; sampled at 30000 Hz
    sif: contains institution and patient info (XML)
    ccf: contains Cerebus configurations

TODO:
  * videosync events (file spec 2.3)
  * tracking events (file spec 2.3)
  * buttontrigger events (file spec 2.3)
  * config events (file spec 2.3)
  * check left sweep settings of Blackrock
  * check nsx offsets (file spec 2.1)
  * add info of nev ext header (NSASEXEX) to non-neural events
    (file spec 2.1 and 2.2)
  * read sif file information
  * read ccf file information
  * fix reading of periodic sampling events (non-neural event type)
    (file spec 2.1 and 2.2)
"""

import datetime
import os
import re
import warnings
import math

import numpy as np
import quantities as pq

from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)

from neo.core import NeoReadWriteError



class BlackrockRawIO(BaseRawIO):
    """
    Class for reading data in from a file set recorded by the Blackrock (Cerebus) recording system.
    Upon initialization, the class is linked to the available set of Blackrock files.

    Parameters
    ----------
    filename: str, default: ''
        File name (without extension) of the set of Blackrock files to associate with.
        Any .nsX or .nev, .sif, or .ccf extensions are ignored when parsing this parameter.
    nsx_override: str | None, default: None
        File name of the .nsX files (without extension). If None, filename is used.
    nev_override: str | None, default: None
        File name of the .nev file (without extension). If None, filename is used.
    nsx_to_load: int | list | 'max' | 'all' | None, default: None
        IDs of nsX file from which to load data, e.g., if set to 5 only data from the ns5 file are loaded.
        If 'all', then all nsX will be loaded. Contrary to previous version of the IO  (<0.7), nsx_to_load
        must be set at the init before parse_header().
    load_nev: bool, default: True
        Load (or not) events/spikes by ignoring or not the nev file.

    Notes
    -----
    * Note: This routine will handle files according to specification 2.1, 2.2,
    2.3, 3.0 and 3.0-ptp. Recording pauses that may occur in file specifications
    2.2 and 2.3 are automatically extracted and the data set is split into different
    segments.

    * The Blackrock data format consists not of a single file, but a set of
    different files. This constructor associates itself with a set of files
    that constitute a common data set. By default, all files belonging to
    the file set have the same base name, but different extensions.
    However, by using the override parameters, individual filenames can
    be set.

    Examples
    --------
    >>> import neo.rawio
    >>> # Inspect a set of file consisting of files FileSpec2.3001.ns5 and FileSpec2.3001.nev
    >>> reader = neo.rawio.BlackrockRawIO(filename='FileSpec2.3001', nsx_to_load=5)
    >>> reader.parse_header()
    >>> print(reader)


    """

    extensions = ["ns" + str(_) for _ in range(1, 7)]
    extensions.extend(["nev", "sif", "ccf"])  # 'sif', 'ccf' not yet supported
    rawmode = "multi-file"

    # We need to document the origin of this value
    main_sampling_rate = 30000.0

    def __init__(
        self, filename=None, nsx_override=None, nev_override=None, nsx_to_load=None, load_nev=True, verbose=False
    ):
        """
        Initialize the BlackrockIO class.
        """
        BaseRawIO.__init__(self)

        self.filename = str(filename)

        # remove extension from base _filenames
        for ext in self.extensions:
            if self.filename.endswith(os.path.extsep + ext):
                self.filename = self.filename.replace(os.path.extsep + ext, "")

        self.nsx_to_load = nsx_to_load

        # remove extensions from overrides
        self._filenames = {}
        if nsx_override:
            self._filenames["nsx"] = re.sub(os.path.extsep + "ns[1,2,3,4,5,6]$", "", nsx_override)
        else:
            self._filenames["nsx"] = self.filename
        if nev_override:
            self._filenames["nev"] = re.sub(os.path.extsep + "nev$", "", nev_override)
        else:
            self._filenames["nev"] = self.filename

        self._filenames["sif"] = self.filename
        self._filenames["ccf"] = self.filename

        # check which files are available
        self._avail_files = dict.fromkeys(self.extensions, False)
        self._avail_nsx = []
        for ext in self.extensions:
            if ext.startswith("ns"):
                file2check = "".join([self._filenames["nsx"], os.path.extsep, ext])
            else:
                file2check = "".join([self._filenames[ext], os.path.extsep, ext])

            if os.path.exists(file2check):
                self._avail_files[ext] = True
                if ext.startswith("ns"):
                    self._avail_nsx.append(int(ext[-1]))

        if not load_nev:
            self._avail_files["nev"] = False

        if not self._avail_files["nev"] and not self._avail_nsx:
            raise IOError("No Blackrock files found in specified path")

        # These dictionaries are used internally to map the file specification
        # revision of the nsx and nev files to one of the reading routines
        # NEV
        self._waveform_size = {
            "2.1": self._get_waveform_size_spec_v21,
            "2.2": self._get_waveform_size_spec_v21,
            "2.3": self._get_waveform_size_spec_v22_30,
            "3.0": self._get_waveform_size_spec_v22_30,
        }
        self._channel_labels = {
            "2.1": self._get_channel_labels_spec_v21,
            "2.2": self._get_channel_labels_spec_v22_30,
            "2.3": self._get_channel_labels_spec_v22_30,
            "3.0": self._get_channel_labels_spec_v22_30,
        }
        self._nonneural_evdicts = {
            "2.1": self._get_nonneural_evdicts_spec_v21_22,
            "2.2": self._get_nonneural_evdicts_spec_v21_22,
            "2.3": self._get_nonneural_evdicts_spec_v23,
            "3.0": self._get_nonneural_evdicts_spec_v23,
        }
        self._comment_evdict = {
            "2.1": self._get_comment_evdict_spec_v21_22,
            "2.2": self._get_comment_evdict_spec_v21_22,
            "2.3": self._get_comment_evdict_spec_v21_22,
            "3.0": self._get_comment_evdict_spec_v21_22,
        }

    def _parse_header(self):

        event_channels = []
        spike_channels = []
        signal_buffers = []
        signal_streams = []
        signal_channels = []

        # Step1 NEV file
        if self._avail_files["nev"]:
            # Load file spec and headers of available

            # read nev file specification
            self._nev_spec = self._extract_nev_file_spec()

            # read nev headers
            nev_filename = f"{self._filenames['nev']}.nev"
            self._nev_basic_header, self._nev_ext_header = self._read_nev_header(self._nev_spec, nev_filename)

            self.nev_data = self._read_nev_data(self._nev_spec, nev_filename)
            spikes, spike_segment_ids = self.nev_data["Spikes"]

            # scan all channel to get number of Unit
            spike_channels = []
            self.internal_unit_ids = []  # pair of chan['packet_id'], spikes['unit_class_nb']
            for i in range(len(self._nev_ext_header[b"NEUEVWAV"])):

                # electrode_id values are stored at uint16 which can overflow when
                # multiplying by 1000 below. We convert to a regular python int which
                # won't overflow
                channel_id = int(self._nev_ext_header[b"NEUEVWAV"]["electrode_id"][i])

                chan_mask = spikes["packet_id"] == channel_id
                chan_spikes = spikes[chan_mask]

                # all `unit_class_nb` is uint8. Also will have issues with overflow
                # cast this to python int
                all_unit_id = np.unique(chan_spikes["unit_class_nb"])
                for u, unit_id in enumerate(all_unit_id):
                    unit_id = int(unit_id)
                    self.internal_unit_ids.append((channel_id, unit_id))
                    name = f"ch{channel_id}#{unit_id}"
                    _id = f"Unit {1000 * channel_id + unit_id}"
                    wf_gain = self._nev_params("digitization_factor")[channel_id] / 1000.0
                    wf_offset = 0.0
                    wf_units = "uV"
                    # TODO: Double check if this is the correct assumption (10 samples)
                    # default value: threshold crossing after 10 samples of waveform
                    wf_left_sweep = 10
                    wf_sampling_rate = self.main_sampling_rate
                    spike_channels.append((name, _id, wf_units, wf_gain, wf_offset, wf_left_sweep, wf_sampling_rate))

            # scan events
            # NonNeural: serial and digital input
            events_data, event_segment_ids = self.nev_data["NonNeural"]
            ev_dict = self._nonneural_evdicts[self._nev_spec](events_data)
            if "Comments" in self.nev_data:
                comments_data, comments_segment_ids = self.nev_data["Comments"]
                ev_dict.update(self._comment_evdict[self._nev_spec](comments_data))
            for ev_name in ev_dict:
                event_channels.append((ev_name, "", "event"))
            # TODO: TrackingEvents
            # TODO: ButtonTrigger
            # TODO: VideoSync

        # Step2 NSX file
        # Load file spec and headers of available nsx files
        self._nsx_spec = {}
        self._nsx_basic_header = {}
        self._nsx_ext_header = {}
        self._nsx_data_header = {}
        self._nsx_sampling_frequency = {}

        # Read headers
        for nsx_nb in self._avail_nsx:
            spec_version = self._nsx_spec[nsx_nb] = self._extract_nsx_file_spec(nsx_nb)
            # read nsx headers
            self._nsx_basic_header[nsx_nb], self._nsx_ext_header[nsx_nb] = self._read_nsx_header(spec_version, nsx_nb)

            # The Blackrock defines period as the number of  1/30_000 seconds between data points
            # E.g. it is 1 for 30_000, 3 for 10_000, etc
            nsx_period = self._nsx_basic_header[nsx_nb]["period"]
            sampling_rate = 30_000.0 / nsx_period
            self._nsx_sampling_frequency[nsx_nb] = float(sampling_rate)

        # Parase data packages
        for nsx_nb in self._avail_nsx:

            # The only way to know if it is the Precision Time Protocol of file spec 3.0
            # is to check for nanosecond timestamp resolution.
            is_ptp_variant = (
                "timestamp_resolution" in self._nsx_basic_header[nsx_nb].dtype.names
                and self._nsx_basic_header[nsx_nb]["timestamp_resolution"] == 1_000_000_000
            )
            if is_ptp_variant:
                data_header_spec = "3.0-ptp"
            else:
                data_header_spec = spec_version

        # nsx_to_load can be either int, list, 'max', 'all' (aka None)
        # here make a list only
        if self.nsx_to_load is None or self.nsx_to_load == "all":
            self.nsx_to_load = list(self._avail_nsx)
        elif self.nsx_to_load == "max":
            if len(self._avail_nsx):
                self.nsx_to_load = [max(self._avail_nsx)]
            else:
                self.nsx_to_load = []
        elif isinstance(self.nsx_to_load, int):
            self.nsx_to_load = [self.nsx_to_load]
        elif isinstance(self.nsx_to_load, list):
            pass
        else:
            raise (ValueError("nsx_to_load is wrong"))

        missing_nsx_files = [nsx_nb for nsx_nb in self.nsx_to_load if nsx_nb not in self._avail_nsx]
        if missing_nsx_files:
            missing_list = ", ".join(f"ns{nsx_nb}" for nsx_nb in missing_nsx_files)
            raise FileNotFoundError(
                f"Requested NSX file(s) not found: {missing_list}. Available NSX files: {self._avail_nsx}"
            )

        # check that all files come from the same specification
        all_spec = [self._nsx_spec[nsx_nb] for nsx_nb in self.nsx_to_load]
        if self._avail_files["nev"]:
            all_spec.append(self._nev_spec)
        if not all(all_spec[0] == spec for spec in all_spec):
            raise NeoReadWriteError("Files don't have the same internal version")

        if len(self.nsx_to_load) > 0 and self._nsx_spec[self.nsx_to_load[0]] == "2.1" and not self._avail_files["nev"]:
            pass
            # Because rescaling to volts requires information from nev file (dig_factor)
            # Remove if raw loading becomes possible
            # raise IOError("For loading Blackrock file version 2.1 .nev files are required!")

        self.nsx_datas = {}
        # Keep public attribute for backward compatibility but let's use the private one and maybe deprecate this at some point
        self.sig_sampling_rates = {
            nsx_number: self._nsx_sampling_frequency[nsx_number] for nsx_number in self.nsx_to_load
        }
        if len(self.nsx_to_load) > 0:
            for nsx_nb in self.nsx_to_load:
                basic_header = self._nsx_basic_header[nsx_nb]
                spec_version = self._nsx_spec[nsx_nb]
                # The only way to know if it is the Precision Time Protocol of file spec 3.0
                # is to check for nanosecond timestamp resolution.
                is_ptp_variant = (
                    "timestamp_resolution" in basic_header.dtype.names
                    and basic_header["timestamp_resolution"] == 1_000_000_000
                )
                if is_ptp_variant:
                    data_spec = "3.0-ptp"
                else:
                    data_spec = spec_version

                # Parse data blocks (creates memmap, extracts data+timestamps)
                data_blocks = self._parse_nsx_data(data_spec, nsx_nb)

                # Segment the data (analyzes gaps, reports issues)
                segments = self._segment_nsx_data(data_blocks, nsx_nb)

                # Store in existing structures for backward compatibility
                self._nsx_data_header[nsx_nb] = {
                    seg_idx: {k: v for k, v in seg.items() if k != 'data'}
                    for seg_idx, seg in segments.items()
                }
                self.nsx_datas[nsx_nb] = {
                    seg_idx: seg['data']
                    for seg_idx, seg in segments.items()
                }

                # Match NSX and NEV segments for v2.3
                if self._avail_files["nev"]:
                    self._match_nsx_and_nev_segment_ids(nsx_nb)

                sr = self._nsx_sampling_frequency[nsx_nb]

                if spec_version in ["2.2", "2.3", "3.0"]:
                    ext_header = self._nsx_ext_header[nsx_nb]
                elif spec_version == "2.1":
                    # v2.1 has no extended headers - construct from NEV digitization factors
                    ext_header = self._build_nsx_v21_ext_header(nsx_nb)

                if len(ext_header) > 0:
                    # in blackrock : one stream per buffer so same id
                    buffer_id = stream_id = str(nsx_nb)
                    stream_name = f"nsx{nsx_nb}"
                    signal_buffers.append((stream_name, buffer_id))
                    signal_streams.append((stream_name, stream_id, buffer_id))
                for i, chan in enumerate(ext_header):
                    if spec_version in ["2.2", "2.3", "3.0"]:
                        ch_name = chan["electrode_label"].decode()
                        ch_id = str(chan["electrode_id"])
                        units = chan["units"].decode()
                    elif spec_version == "2.1":
                        ch_name = chan["labels"]
                        ch_id = str(self._nsx_ext_header[nsx_nb][i]["electrode_id"])
                        units = chan["units"]
                    sig_dtype = "int16"
                    # max_analog_val/min_analog_val/max_digital_val/min_analog_val are int16!!!!!
                    # dangerous situation so cast to float everyone
                    if np.isnan(float(chan["min_analog_val"])):
                        gain = 1
                        offset = 0
                    else:
                        gain = (float(chan["max_analog_val"]) - float(chan["min_analog_val"])) / (
                            float(chan["max_digital_val"]) - float(chan["min_digital_val"])
                        )
                        offset = -float(chan["min_digital_val"]) * gain + float(chan["min_analog_val"])
                    buffer_id = stream_id = str(nsx_nb)
                    signal_channels.append((ch_name, ch_id, sr, sig_dtype, units, gain, offset, stream_id, buffer_id))

            # check nb segment per nsx
            nb_segments_for_nsx = [len(self.nsx_datas[nsx_nb]) for nsx_nb in self.nsx_to_load]
            if not all(nb == nb_segments_for_nsx[0] for nb in nb_segments_for_nsx):
                raise NeoReadWriteError("Segment nb not consistent across nsX files")
            self._nb_segment = nb_segments_for_nsx[0]

            self._delete_empty_segments()

            # t_start/t_stop for segment are given by nsx limits or nev limits
            self._sigs_t_starts = {nsx_nb: [] for nsx_nb in self.nsx_to_load}
            self._seg_t_starts, self._seg_t_stops = [], []
            for data_bl in range(self._nb_segment):
                t_stop = 0.0
                for nsx_nb in self.nsx_to_load:
                    spec = self._nsx_spec[nsx_nb]
                    if "timestamp_resolution" in self._nsx_basic_header[nsx_nb].dtype.names:
                        ts_res = self._nsx_basic_header[nsx_nb]["timestamp_resolution"]
                    elif spec == "2.1":
                        ts_res = 30_000  # v2.1 always uses 30kHz timestamp resolution
                    else:
                        ts_res = 30_000
                    period = self._nsx_basic_header[nsx_nb]["period"]
                    sec_per_samp = period / 30_000  # Maybe 30_000 should be ['sample_resolution']
                    length = self.nsx_datas[nsx_nb][data_bl].shape[0]
                    timestamps = self._nsx_data_header[nsx_nb][data_bl]["timestamp"]
                    if timestamps is None:
                        # V2.1 format has no timestamps
                        t_start = 0.0
                        t_stop = max(t_stop, length / self._nsx_sampling_frequency[nsx_nb])
                    elif hasattr(timestamps, "size") and timestamps.size == length:
                        # FileSpec 3.0 with PTP -- use the per-sample timestamps
                        t_start = timestamps[0] / ts_res
                        t_stop = max(t_stop, timestamps[-1] / ts_res + sec_per_samp)
                    else:
                        # Standard format with scalar timestamp
                        t_start = timestamps / ts_res
                        t_stop = max(t_stop, t_start + length / self._nsx_sampling_frequency[nsx_nb])
                    self._sigs_t_starts[nsx_nb].append(t_start)

                if self._avail_files["nev"]:
                    max_nev_time = 0
                    for k, (data, ev_ids) in self.nev_data.items():
                        segment_mask = ev_ids == data_bl
                        if data[segment_mask].size > 0:
                            t = data[segment_mask][-1]["timestamp"] / self._nev_basic_header["timestamp_resolution"]

                            max_nev_time = max(max_nev_time, t)
                    if max_nev_time > t_stop:
                        t_stop = max_nev_time
                    min_nev_time = max_nev_time
                    for k, (data, ev_ids) in self.nev_data.items():
                        segment_mask = ev_ids == data_bl
                        if data[segment_mask].size > 0:
                            t = data[segment_mask][0]["timestamp"] / self._nev_basic_header["timestamp_resolution"]
                            min_nev_time = min(min_nev_time, t)
                    if min_nev_time < t_start:
                        t_start = min_nev_time
                self._seg_t_starts.append(t_start)
                self._seg_t_stops.append(float(t_stop))

        else:
            # When only nev is available, only segments that are documented in nev can be detected

            max_nev_times = {}
            min_nev_times = {}

            # Find maximal and minimal time for each nev segment
            for k, (data, ev_ids) in self.nev_data.items():
                for i in np.unique(ev_ids):
                    curr_data = data[ev_ids == i]
                    if curr_data.size > 0:
                        if max(curr_data["timestamp"]) >= max_nev_times.get(i, 0):
                            max_nev_times[i] = max(curr_data["timestamp"])
                        if min(curr_data["timestamp"]) <= min_nev_times.get(i, max_nev_times[i]):
                            min_nev_times[i] = min(curr_data["timestamp"])

            # Calculate t_start and t_stop for each segment in seconds
            resolution = self._nev_basic_header["timestamp_resolution"]
            self._seg_t_starts = [v / float(resolution) for k, v in sorted(min_nev_times.items())]
            self._seg_t_stops = [v / float(resolution) for k, v in sorted(max_nev_times.items())]
            self._nb_segment = len(self._seg_t_starts)
            self._sigs_t_starts = [None] * self._nb_segment

        # finalize header
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_buffers = np.array(signal_buffers, dtype=_signal_buffer_dtype)

        self.header = {}
        self.header["nb_block"] = 1
        self.header["nb_segment"] = [self._nb_segment]
        self.header["signal_buffers"] = signal_buffers
        self.header["signal_streams"] = signal_streams
        self.header["signal_channels"] = signal_channels
        self.header["spike_channels"] = spike_channels
        self.header["event_channels"] = event_channels

        rec_datetime = self._nev_params("rec_datetime") if self._avail_files["nev"] else None

        # Put annotations at some places for compatibility
        # with previous BlackrockIO version
        self._generate_minimal_annotations()

        block_ann = self.raw_annotations["blocks"][0]
        block_ann["description"] = "Block of data from Blackrock file set."
        block_ann["file_origin"] = self.filename
        block_ann["name"] = "Blackrock Data Block"
        block_ann["rec_datetime"] = rec_datetime
        block_ann["avail_file_set"] = [k for k, v in self._avail_files.items() if v]
        block_ann["avail_nsx"] = self._avail_nsx
        block_ann["avail_nev"] = self._avail_files["nev"]
        #  'sif' and 'ccf' files not yet supported
        # block_ann['avail_sif'] = self._avail_files['sif']
        # block_ann['avail_ccf'] = self._avail_files['ccf']
        block_ann["rec_pauses"] = False

        for seg_index in range(self._nb_segment):
            seg_ann = block_ann["segments"][seg_index]
            seg_ann["file_origin"] = self.filename
            seg_ann["name"] = f"Segment {seg_index}"
            seg_ann["description"] = "Segment containing data from t_start to t_stop"
            if seg_index == 0:
                # if more than 1 segment means pause
                # so datetime is valid only for seg_index=0
                seg_ann["rec_datetime"] = rec_datetime

            for c in range(signal_streams.size):
                sig_ann = seg_ann["signals"][c]
                stream_id = signal_streams["id"][c]
                nsx_nb = int(stream_id)
                sig_ann["description"] = f"AnalogSignal from  nsx{nsx_nb}"
                sig_ann["file_origin"] = self._filenames["nsx"] + ".ns" + str(nsx_nb)
                sig_ann["nsx"] = nsx_nb
                # handle signal array annotations from nsx header
                if self._nsx_spec[nsx_nb] in ["2.2", "2.3"] and nsx_nb in self._nsx_ext_header:
                    mask = signal_channels["stream_id"] == stream_id
                    channels = signal_channels[mask]
                    nsx_header = self._nsx_ext_header[nsx_nb]
                    for key in (
                        "physical_connector",
                        "connector_pin",
                        "hi_freq_corner",
                        "lo_freq_corner",
                        "hi_freq_order",
                        "lo_freq_order",
                        "hi_freq_type",
                        "lo_freq_type",
                    ):
                        values = []
                        for chan_id in channels["id"]:
                            chan_id = int(chan_id)
                            idx = list(nsx_header["electrode_id"]).index(chan_id)
                            values.append(nsx_header[key][idx])
                        values = np.array(values)
                        sig_ann["__array_annotations__"][key] = values

            for c in range(spike_channels.size):
                st_ann = seg_ann["spikes"][c]
                channel_id, unit_id = self.internal_unit_ids[c]
                st_ann["channel_id"] = channel_id
                st_ann["unit_id"] = unit_id
                if unit_id == 0:
                    st_ann["unit_classification"] = "unclassified"
                elif 1 <= unit_id <= 16:
                    st_ann["unit_classification"] = "sorted"
                elif unit_id == 255:
                    st_ann["unit_classification"] = "noise"
                else:  # 17-254 are reserved
                    st_ann["unit_classification"] = "reserved"
                st_ann["unit_tag"] = st_ann["unit_classification"]
                st_ann["description"] = f"SpikeTrain channel_id: {channel_id}, unit_id: {unit_id}"
                st_ann["file_origin"] = self._filenames["nev"] + ".nev"

            if self._avail_files["nev"]:
                ev_dict = self._nonneural_evdicts[self._nev_spec](events_data)
                if "Comments" in self.nev_data:
                    ev_dict.update(self._comment_evdict[self._nev_spec](comments_data))
                    color_codes = ["#{:08X}".format(code) for code in comments_data["color"]]
                    color_codes = np.array(color_codes, dtype="S9")
                for c in range(event_channels.size):
                    # Next line makes ev_ann a reference to seg_ann['events'][c]
                    ev_ann = seg_ann["events"][c]
                    name = event_channels["name"][c]
                    ev_ann["description"] = ev_dict[name]["desc"]
                    ev_ann["file_origin"] = self._filenames["nev"] + ".nev"
                    if name == "comments":
                        ev_ann["color_codes"] = color_codes

    def _source_name(self):
        return self.filename

    def _segment_t_start(self, block_index, seg_index):
        return self._seg_t_starts[seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        return self._seg_t_stops[seg_index]

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_id = self.header["signal_streams"][stream_index]["id"]
        nsx_nb = int(stream_id)
        memmap_data = self.nsx_datas[nsx_nb][seg_index]
        return memmap_data.shape[0]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        stream_id = self.header["signal_streams"][stream_index]["id"]
        nsx_nb = int(stream_id)
        return self._sigs_t_starts[nsx_nb][seg_index]

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
        stream_id = self.header["signal_streams"][stream_index]["id"]
        nsx_nb = int(stream_id)
        memmap_data = self.nsx_datas[nsx_nb][seg_index]
        if channel_indexes is None:
            channel_indexes = slice(None)
        sig_chunk = memmap_data[i_start:i_stop, channel_indexes]
        return sig_chunk

    def _spike_count(self, block_index, seg_index, unit_index):
        channel_id, unit_id = self.internal_unit_ids[unit_index]

        all_spikes = self.nev_data["Spikes"][0]
        mask = (all_spikes["packet_id"] == channel_id) & (all_spikes["unit_class_nb"] == unit_id)
        if self._nb_segment == 1:
            # very fast
            nb = int(np.sum(mask))
        else:
            # must clip in time time range
            timestamp = all_spikes[mask]["timestamp"]
            sl = self._get_timestamp_slice(timestamp, seg_index, None, None)
            timestamp = timestamp[sl]
            nb = timestamp.size
        return nb

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        channel_id, unit_id = self.internal_unit_ids[unit_index]

        all_spikes, event_segment_ids = self.nev_data["Spikes"]

        # select by channel_id and unit_id
        mask = (
            (all_spikes["packet_id"] == channel_id)
            & (all_spikes["unit_class_nb"] == unit_id)
            & (event_segment_ids == seg_index)
        )
        unit_spikes = all_spikes[mask]

        timestamp = unit_spikes["timestamp"]
        sl = self._get_timestamp_slice(timestamp, seg_index, t_start, t_stop)
        timestamp = timestamp[sl]

        return timestamp

    def _get_timestamp_slice(self, timestamp, seg_index, t_start, t_stop):
        if self._nb_segment > 1:
            # we must clip event in seg time limits
            if t_start is None:
                t_start = self._seg_t_starts[seg_index]
            if t_stop is None:
                t_stop = self._seg_t_stops[seg_index] + 1 / float(self._nev_basic_header["timestamp_resolution"])

        if t_start is None:
            ind_start = None
        else:
            ts = math.ceil(t_start * self._nev_basic_header["timestamp_resolution"])
            ind_start = np.searchsorted(timestamp, ts)

        if t_stop is None:
            ind_stop = None
        else:
            ts = int(t_stop * self._nev_basic_header["timestamp_resolution"])
            ind_stop = np.searchsorted(timestamp, ts)  # +1

        sl = slice(ind_start, ind_stop)
        return sl

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self._nev_basic_header["timestamp_resolution"]
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        channel_id, unit_id = self.internal_unit_ids[unit_index]
        all_spikes, event_segment_ids = self.nev_data["Spikes"]

        mask = (
            (all_spikes["packet_id"] == channel_id)
            & (all_spikes["unit_class_nb"] == unit_id)
            & (event_segment_ids == seg_index)
        )
        unit_spikes = all_spikes[mask]

        wf_dtype = self._nev_params("waveform_dtypes")[channel_id]
        wf_size = self._nev_params("waveform_size")[channel_id]
        wf_byte_size = np.dtype(wf_dtype).itemsize * wf_size

        dt1 = [
            ("extra", "S{}".format(unit_spikes["waveform"].dtype.itemsize - wf_byte_size)),
            ("ch_waveform", "S{}".format(wf_byte_size)),
        ]

        waveforms = unit_spikes["waveform"].view(dt1)["ch_waveform"].flatten().view(wf_dtype)

        waveforms = waveforms.reshape(int(unit_spikes.size), 1, int(wf_size))

        timestamp = unit_spikes["timestamp"]
        sl = self._get_timestamp_slice(timestamp, seg_index, t_start, t_stop)
        waveforms = waveforms[sl]

        return waveforms

    def _event_count(self, block_index, seg_index, event_channel_index):
        name = self.header["event_channels"]["name"][event_channel_index]
        if name == "comments":
            events_data, event_segment_ids = self.nev_data["Comments"]
            ev_dict = self._comment_evdict[self._nev_spec](events_data)[name]
        else:
            events_data, event_segment_ids = self.nev_data["NonNeural"]
            ev_dict = self._nonneural_evdicts[self._nev_spec](events_data)[name]
        mask = ev_dict["mask"] & (event_segment_ids == seg_index)
        if self._nb_segment == 1:
            # very fast
            nb = int(np.sum(mask))
        else:
            # must clip in time time range
            timestamp = events_data[ev_dict["mask"]]["timestamp"]
            sl = self._get_timestamp_slice(timestamp, seg_index, None, None)
            timestamp = timestamp[sl]
            nb = timestamp.size
        return nb

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        name = self.header["event_channels"]["name"][event_channel_index]
        if name == "comments":
            events_data, event_segment_ids = self.nev_data["Comments"]
            ev_dict = self._comment_evdict[self._nev_spec](events_data)[name]
            # If immediate decoding is desired:
            encoding = {0: "latin_1", 1: "utf_16", 255: "latin_1"}
            labels = [data[ev_dict["field"]].decode(encoding[data["char_set"]]) for data in events_data]
            labels = np.array(labels, dtype="U")
        else:
            events_data, event_segment_ids = self.nev_data["NonNeural"]
            ev_dict = self._nonneural_evdicts[self._nev_spec](events_data)[name]
            labels = events_data[ev_dict["field"]].astype("U")

        mask = ev_dict["mask"] & (event_segment_ids == seg_index)
        timestamp = events_data[mask]["timestamp"]
        labels = labels[mask]

        # time clip
        sl = self._get_timestamp_slice(timestamp, seg_index, t_start, t_stop)
        timestamp = timestamp[sl]
        labels = labels[sl]
        durations = None

        return timestamp, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        ev_times = event_timestamps.astype(dtype)
        ev_times /= self._nev_basic_header["timestamp_resolution"]
        return ev_times

    ###################################################
    ###################################################

    # Above here code from Lyuba Zehl, Michael Denker
    # coming from previous BlackrockIO

    def _extract_nsx_file_spec(self, nsx_nb):
        """
        Extract file specification from an .nsx file.
        """
        filename = f"{self._filenames['nsx']}.ns{nsx_nb}"

        # Header structure of files specification 2.2 and higher. For files 2.1
        # and lower, the entries ver_major and ver_minor are not supported.
        dt0 = [("file_id", "S8"), ("ver_major", "uint8"), ("ver_minor", "uint8")]

        nsx_file_id = np.fromfile(filename, count=1, dtype=dt0)[0]
        if nsx_file_id["file_id"].decode() == "NEURALSG":
            spec = "2.1"
        elif nsx_file_id["file_id"].decode() in ["NEURALCD", "BRSMPGRP"]:
            spec = f"{nsx_file_id['ver_major']}.{nsx_file_id['ver_minor']}"
        else:
            raise IOError("Unsupported NSX file type.")

        return spec

    def _extract_nev_file_spec(self):
        """
        Extract file specification from an .nev file
        """
        filename = f"{self._filenames['nev']}.nev"
        # Header structure of files specification 2.2 and higher. For files 2.1
        # and lower, the entries ver_major and ver_minor are not supported.
        dt0 = [("file_id", "S8"), ("ver_major", "uint8"), ("ver_minor", "uint8")]

        nev_file_id = np.fromfile(filename, count=1, dtype=dt0)[0]
        if nev_file_id["file_id"].decode() in ["NEURALEV", "BREVENTS"]:
            spec = f"{nev_file_id['ver_major']}.{nev_file_id['ver_minor']}"
        else:
            raise IOError(f"NEV file type {nev_file_id['file_id'].decode()} is not supported")

        return spec

    def _read_nsx_header(self, spec, nsx_nb):
        """
        Extract nsx header information for any specification version.
        
        Parameters
        ----------
        spec : str
            The specification version (e.g., "2.1", "2.2", "2.3", "3.0")
        nsx_nb : int
            The NSX file number (e.g., 5 for ns5)
        
        Returns
        -------
        nsx_basic_header : numpy structured array
            Basic header information
        nsx_ext_header : numpy memmap
            Extended header information
        """
        # Construct filename
        filename = f"{self._filenames['nsx']}.ns{nsx_nb}"
        
        # Get basic header structure for this spec
        basic_header_dtype = NSX_BASIC_HEADER_TYPES[spec]
        nsx_basic_header = np.fromfile(filename, count=1, dtype=basic_header_dtype)[0]
        
        # Get extended header structure for this spec  
        ext_header_dtype = NSX_EXT_HEADER_TYPES[spec]
        offset_dt0 = np.dtype(basic_header_dtype).itemsize
        channel_count = int(nsx_basic_header["channel_count"])
        nsx_ext_header = np.memmap(filename, shape=channel_count, offset=offset_dt0, dtype=ext_header_dtype, mode="r")
        
        return nsx_basic_header, nsx_ext_header

    def _read_nsx_dataheader(self, spec, nsx_nb, offset):
        """
        Reads data header following the given offset of an nsx file.
        
        Parameters
        ----------
        spec : str
            The specification version (e.g., "2.2", "2.3", "3.0")
        nsx_nb : int
            The NSX file number
        offset : int
            Offset position in the file
        """
        filename = f"{self._filenames['nsx']}.ns{nsx_nb}"

        # Get data header structure for this spec
        data_header_dtype = NSX_DATA_HEADER_TYPES[spec]
        if data_header_dtype is None:
            return None  # v2.1 has no data headers
            
        packet_header = np.memmap(filename, dtype=data_header_dtype, shape=1, offset=offset, mode="r")[0]

        return packet_header

    def _parse_nsx_data(self, spec, nsx_nb):
        """
        Parse NSX data blocks from file and extract data with timestamps.

        This is the main router function for NSX data parsing. It creates a memory-mapped
        view of the file internally and extracts data blocks with their associated timestamps.

        The function handles three different NSX file format variants, each with different
        internal structure for storing data and timestamps.

        NSX FILE FORMAT VARIANTS
        ========================

        STANDARD FORMAT (v2.2, v2.3, v3.0 non-PTP)
        ------------------------------------------
        File structure:
        ┌─────────────────────────────────────────────┐
        │ BASIC HEADER (fixed size)                   │  ← File metadata
        │ - file_id, version, period, etc.            │
        ├─────────────────────────────────────────────┤
        │ EXTENDED HEADER (per channel)               │  ← Channel info
        │ - electrode_id, label, units, etc.          │
        ├─────────────────────────────────────────────┤
        │ DATA BLOCK 1 HEADER                         │  ← Block metadata
        │ - header_flag=1                             │
        │ - timestamp (scalar, e.g., 0)               │
        │ - nb_data_points (e.g., 1000)               │
        ├─────────────────────────────────────────────┤
        │ DATA BLOCK 1 DATA                           │  ← Actual samples
        │ - 1000 samples × N channels                 │
        │ - int16 values                              │
        ├─────────────────────────────────────────────┤
        │ DATA BLOCK 2 HEADER                         │  ← Next block
        │ - header_flag=1                             │
        │ - timestamp (scalar, e.g., 30000)           │
        │ - nb_data_points (e.g., 1000)               │
        ├─────────────────────────────────────────────┤
        │ DATA BLOCK 2 DATA                           │
        │ - 1000 samples × N channels                 │
        └─────────────────────────────────────────────┘

        Key characteristics:
        - Headers are EXPLICIT and SPARSE (only at block boundaries)
        - Each block has ONE scalar timestamp for ALL samples in that block
        - Reader LOOPS through file, finding headers
        - Multiple blocks exist when recording was paused/resumed
        - Timestamp indicates when each block started recording

        PTP FORMAT (v3.0 with Precision Time Protocol)
        -----------------------------------------------
        File structure:
        ┌─────────────────────────────────────────────┐
        │ BASIC HEADER (fixed size)                   │
        │ - timestamp_resolution = 1,000,000,000      │  ← Nanosecond precision!
        ├─────────────────────────────────────────────┤
        │ EXTENDED HEADER (per channel)               │
        ├─────────────────────────────────────────────┤
        │ PACKET 0:                                   │  ← Each sample = packet
        │ - reserved (1 byte)                         │
        │ - timestamp (8 bytes, e.g., 1000)           │
        │ - num_data_points (always 1)                │
        │ - samples (N channels × int16)              │
        ├─────────────────────────────────────────────┤
        │ PACKET 1:                                   │
        │ - reserved                                  │
        │ - timestamp (e.g., 1033)                    │
        │ - num_data_points (1)                       │
        │ - samples (N channels × int16)              │
        ├─────────────────────────────────────────────┤
        │ PACKET 2:                                   │
        │ - reserved                                  │
        │ - timestamp (e.g., 1066)                    │
        │ - num_data_points (1)                       │
        │ - samples (N channels × int16)              │
        ├─────────────────────────────────────────────┤
        │ ...thousands more packets...                │
        │ PACKET 500:                                 │
        │ - timestamp (e.g., 50000)                   │  ← BIG GAP!
        │ ...                                         │
        └─────────────────────────────────────────────┘

        Key characteristics:
        - NO separate headers and data - they're INTERLEAVED
        - EVERY sample has its own timestamp (per-sample nanosecond precision)
        - File is ONE CONTINUOUS ARRAY of uniform packets
        - Segments must be INFERRED by detecting timestamp gaps
        - Timestamp gap > 2× sampling period indicates segment boundary

        V2.1 FORMAT
        -----------
        Simplest format:
        - Single continuous data block
        - No timestamps in data section
        - No multiple blocks (no pause/resume support)

        Parameters
        ----------
        spec : str
            File specification version ("2.1", "2.2", "2.3", "3.0", "3.0-ptp")
        nsx_nb : int
            NSX file number (e.g., 5 for ns5 file)

        Returns
        -------
        dict
            Dictionary mapping block index to block information:
            {
                block_idx: {
                    "data": np.ndarray,
                        View into memory-mapped file with shape (samples, channels)
                    "timestamps": scalar, np.ndarray, or None,
                        - Standard format: scalar (one timestamp per block)
                        - PTP format: array (one timestamp per sample)
                        - v2.1 format: None (no timestamps)
                    # Additional metadata as needed
                },
                ...
            }

        Notes
        -----
        - This function creates the file memmap internally
        - Data views are created using np.ndarray with buffer parameter (memory efficient)
        - Returned data is NOT YET SEGMENTED (segmentation happens in a separate step)
        - For standard format, each block from the file is one dict entry
        - For PTP format, all data is in a single block (block_idx=0)
        """
        if spec == "2.1":
            return self._parse_nsx_data_v21(nsx_nb)
        elif spec == "3.0-ptp":
            return self._parse_nsx_data_v30_ptp(nsx_nb)
        else:  # 2.2, 2.3, 3.0 standard
            return self._parse_nsx_data_v22_v30(spec, nsx_nb)

    def _parse_nsx_data_v21(self, nsx_nb):
        """
        Parse v2.1 NSX data blocks.

        V2.1 format is the simplest:
        - Single continuous data block
        - No timestamps
        - No pause/resume support

        Returns
        -------
        dict
            {0: {"data": np.ndarray, "timestamps": None}}
        """
        filename = f"{self._filenames['nsx']}.ns{nsx_nb}"

        # Create file memmap
        file_memmap = np.memmap(filename, dtype='uint8', mode='r')

        # Calculate header size and data points for v2.1
        channels = int(self._nsx_basic_header[nsx_nb]["channel_count"])
        bytes_in_headers = (
            self._nsx_basic_header[nsx_nb].dtype.itemsize
            + self._nsx_ext_header[nsx_nb].dtype.itemsize * channels
        )
        filesize = self._get_file_size(filename)
        num_samples = int((filesize - bytes_in_headers) / (2 * channels) - 1)
        offset = bytes_in_headers
        # Create data view into memmap
        data = np.ndarray(
            shape=(num_samples, channels),
            dtype='int16',
            buffer=file_memmap,
            offset=offset
        )

        return {
            0: {
                "data": data,
                "timestamps": None,
            }
        }

    def _parse_nsx_data_v22_v30(self, spec, nsx_nb):
        """
        Parse standard format NSX data blocks (v2.2, 2.3, 3.0).

        Standard format has:
        - Explicit block headers in file
        - Each block has scalar timestamp
        - Multiple blocks when recording paused/resumed

        Returns
        -------
        dict
            {block_idx: {"data": np.ndarray, "timestamps": scalar}, ...}
        """
        filename = f"{self._filenames['nsx']}.ns{nsx_nb}"

        # Create file memmap
        file_memmap = np.memmap(filename, dtype='uint8', mode='r')

        # Get file parameters
        filesize = self._get_file_size(filename)
        channels = int(self._nsx_basic_header[nsx_nb]["channel_count"])
        current_offset = int(self._nsx_basic_header[nsx_nb]["bytes_in_headers"])

        data_blocks = {}
        block_idx = 0

        # Loop through file, reading block headers
        while current_offset < filesize:
            # Read header at current position
            header = self._read_nsx_dataheader(spec, nsx_nb, current_offset)

            if header["header_flag"] != 1:
                raise ValueError(
                    f"Invalid NSX data block header at offset {current_offset:#x} "
                    f"in ns{nsx_nb} file. Expected header_flag=1, got {header['header_flag']}."
                )

            num_samples = int(header["nb_data_points"])
            data_offset = current_offset + header.dtype.itemsize
            timestamp = header["timestamp"]

            # Create data view into memmap for this block
            data = np.ndarray(
                shape=(num_samples, channels),
                dtype='int16',
                buffer=file_memmap,
                offset=data_offset
            )

            data_blocks[block_idx] = {
                "data": data,
                "timestamps": timestamp,
            }

            # Jump to next block
            data_size_bytes = num_samples * channels * 2  # int16 = 2 bytes
            current_offset = data_offset + data_size_bytes
            block_idx += 1

        return data_blocks

    def _parse_nsx_data_v30_ptp(self, nsx_nb):
        """
        Parse PTP format NSX data (v3.0 with Precision Time Protocol).

        PTP format has:
        - Interleaved structure (timestamp + sample per packet)
        - Array of timestamps (one per sample)
        - Continuous data (segmentation inferred from timestamp gaps)

        Returns
        -------
        dict
            {0: {"data": np.ndarray, "timestamps": np.ndarray}}
        """
        filename = f"{self._filenames['nsx']}.ns{nsx_nb}"

        # Get file parameters
        filesize = self._get_file_size(filename)
        header_size = int(self._nsx_basic_header[nsx_nb]["bytes_in_headers"])
        channel_count = int(self._nsx_basic_header[nsx_nb]["channel_count"])

        # Create structured memmap (timestamp + samples per packet)
        ptp_dt = NSX_DATA_HEADER_TYPES["3.0-ptp"](channel_count)
        npackets = int((filesize - header_size) / np.dtype(ptp_dt).itemsize)
        file_memmap = np.memmap(
            filename,
            dtype=ptp_dt,
            shape=npackets,
            offset=header_size,
            mode='r'
        )

        # Verify this is truly PTP (all packets should have 1 sample)
        if not np.all(file_memmap["num_data_points"] == 1):
            # Not actually PTP! Fall back to standard format
            return self._parse_nsx_data_v22_v30("3.0", nsx_nb)

        # Extract data and timestamps from structured array
        data = file_memmap["samples"]
        timestamps = file_memmap["timestamps"]

        return {
            0: {
                "data": data,
                "timestamps": timestamps,
            }
        }

    def _report_nsx_timestamp_gaps(self, gap_indices, timestamps_in_seconds, time_differences, threshold, nsx_nb):
        """
        Report detected timestamp gaps with detailed table.

        This function generates a warning message with a detailed table showing
        where gaps were detected in the timestamp sequence.

        Parameters
        ----------
        gap_indices : np.ndarray
            Indices where gaps were detected
        timestamps_in_seconds : np.ndarray
            All timestamps converted to seconds
        time_differences : np.ndarray
            Time differences between consecutive timestamps
        threshold : float
            Gap threshold in seconds
        nsx_nb : int
            NSX file number for the report
        """
        import warnings

        threshold_ms = threshold * 1000
        num_segments = len(gap_indices) + 1

        # Calculate gap details
        gap_durations_seconds = time_differences[gap_indices]
        gap_durations_ms = gap_durations_seconds * 1000
        gap_positions_seconds = timestamps_in_seconds[gap_indices] - timestamps_in_seconds[0]

        # Build gap detail table
        gap_detail_lines = [
            f"| {index:>15,} | {pos:>21.6f} | {dur:>21.3f} |\n"
            for index, pos, dur in zip(gap_indices, gap_positions_seconds, gap_durations_ms)
        ]

        segmentation_report_message = (
            f"\nFound {len(gap_indices)} gaps for nsx {nsx_nb} where samples are farther apart than {threshold_ms:.3f} ms.\n"
            f"Data will be segmented at these locations to create {num_segments} segments.\n\n"
            "Gap Details:\n"
            "+-----------------+-----------------------+-----------------------+\n"
            "| Sample Index    | Sample at (Seconds)   | Gap Jump (ms)         |\n"
            "+-----------------+-----------------------+-----------------------+\n"
            + "".join(gap_detail_lines)
            + "+-----------------+-----------------------+-----------------------+\n"
        )
        warnings.warn(segmentation_report_message)

    def _segment_nsx_data(self, data_blocks_dict, nsx_nb):
        """
        Segment NSX data based on timestamp gaps.

        Takes the data blocks returned by _parse_nsx_data() and creates segments.
        Segmentation logic depends on the file format:

        - Standard format (multiple blocks): Each block IS a segment
        - PTP format (single block with timestamp array): Detect gaps in timestamps
        - V2.1 format (no timestamps): Single segment

        Parameters
        ----------
        data_blocks_dict : dict
            Dictionary from _parse_nsx_data():
            {block_idx: {"data": np.ndarray, "timestamps": scalar/array/None}}
        nsx_nb : int
            NSX file number

        Returns
        -------
        dict
            {
                seg_idx: {
                    "data": np.ndarray,
                    "timestamps": scalar, array, or None,
                    "nb_data_points": int,
                    "header": int or None,
                    "offset_to_data_block": None (deprecated but kept for compatibility)
                },
                ...
            }
        """
        segments = {}

        # Case 1: Multiple blocks (Standard format) - each block is a segment
        if len(data_blocks_dict) > 1:
            for block_idx, block_info in data_blocks_dict.items():
                segments[block_idx] = {
                    "data": block_info["data"],
                    "timestamp": block_info["timestamps"],  # Use singular for backward compatibility
                    "nb_data_points": block_info["data"].shape[0],
                    "header": 1,  # Standard format has headers
                    "offset_to_data_block": None,  # Not needed (have data directly)
                }

        # Case 2: Single block - check if PTP (array timestamps) or simple (no timestamps)
        elif len(data_blocks_dict) == 1:
            block_info = data_blocks_dict[0]
            data = block_info["data"]
            timestamps = block_info["timestamps"]

            # PTP format: array of timestamps - need to detect gaps
            if isinstance(timestamps, np.ndarray):
                # Analyze timestamp gaps
                sampling_rate = self._nsx_sampling_frequency[nsx_nb]
                segmentation_threshold = 2.0 / sampling_rate

                timestamps_sampling_rate = self._nsx_basic_header[nsx_nb]["timestamp_resolution"]
                timestamps_in_seconds = timestamps / timestamps_sampling_rate

                time_differences = np.diff(timestamps_in_seconds)
                gap_indices = np.argwhere(time_differences > segmentation_threshold).flatten()

                # Report gaps if found
                if len(gap_indices) > 0:
                    self._report_nsx_timestamp_gaps(
                        gap_indices, timestamps_in_seconds, time_differences,
                        segmentation_threshold, nsx_nb
                    )

                # Create segments based on gaps
                segment_starts = np.hstack((0, gap_indices + 1))
                segment_boundaries = list(segment_starts) + [len(data)]

                for seg_idx, start in enumerate(segment_starts):
                    end = segment_boundaries[seg_idx + 1]

                    segments[seg_idx] = {
                        "data": data[start:end],
                        "timestamp": timestamps[start:end],  # Use singular for backward compatibility
                        "nb_data_points": end - start,
                        "header": None,  # PTP has no headers
                        "offset_to_data_block": None,
                    }

            # V2.1 or single block standard format: no segmentation needed
            else:
                segments[0] = {
                    "data": data,
                    "timestamp": timestamps,  # Use singular for backward compatibility
                    "nb_data_points": data.shape[0],
                    "header": None,
                    "offset_to_data_block": None,
                }

        return segments

    def _build_nsx_v21_ext_header(self, nsx_nb):
        """
        Build extended header structure for v2.1 NSX files.

        v2.1 NSX files don't have extended headers with analog/digital ranges.
        We estimate these from the digitization factor in the NEV file.
        dig_factor = max_analog_val / max_digital_val
        We set max_digital_val = 1000, so max_analog_val = dig_factor
        dig_factor is in nV, so units are uV.

        Information from Kian Torab, Blackrock Microsystems.
        """
        ext_header = []

        for i, elid in enumerate(self._nsx_ext_header[nsx_nb]["electrode_id"]):
            # Get digitization factor from NEV
            if self._avail_files["nev"]:
                # Workaround for DigitalFactor overflow in buggy Cerebus systems
                # Fix from NPMK toolbox (openNEV, line 464, git rev d0a25eac)
                dig_factor = self._nev_params("digitization_factor")[elid]
                if dig_factor == 21516:
                    dig_factor = 152592.547
                units = "uV"
            else:
                dig_factor = float("nan")
                units = ""
                if i == 0:  # Only warn once
                    warnings.warn("Cannot rescale to voltage, raw data will be returned.", UserWarning)

            # Generate label
            if elid < 129:
                label = f"chan{elid}"
            else:
                label = f"ainp{(elid - 129 + 1)}"

            ext_header.append({
                "labels": label,
                "units": units,
                "min_analog_val": -float(dig_factor),
                "max_analog_val": float(dig_factor),
                "min_digital_val": -1000,
                "max_digital_val": 1000,
            })

        return ext_header

    def _read_nev_header(self, spec, filename):
        """
        Extract nev header information for any specification version.
        
        Parameters
        ----------
        spec : str
            The specification version (e.g., "2.1", "2.2", "2.3", "3.0")
        filename : str
            The NEV filename to read from
        
        Returns
        -------
        nev_basic_header : np.ndarray
            Basic header information
        nev_ext_header : dict
            Extended header information by packet ID
        """
        # Note: This function only uses the passed parameters, not self attributes
        # This makes it easy to convert to @staticmethod later
        
        # basic header (same for all versions)
        dt0 = [
            # Set to "NEURALEV"
            ("file_type_id", "S8"),
            ("ver_major", "uint8"),
            ("ver_minor", "uint8"),
            # Flags
            ("additionnal_flags", "uint16"),
            # File index of first data sample
            ("bytes_in_headers", "uint32"),
            # Number of bytes per data packet (sample)
            ("bytes_in_data_packets", "uint32"),
            # Time resolution of time stamps in Hz
            ("timestamp_resolution", "uint32"),
            # Sampling frequency of waveforms in Hz
            ("sample_resolution", "uint32"),
            ("year", "uint16"),
            ("month", "uint16"),
            ("weekday", "uint16"),
            ("day", "uint16"),
            ("hour", "uint16"),
            ("minute", "uint16"),
            ("second", "uint16"),
            ("millisecond", "uint16"),
            ("application_to_create_file", "S32"),
            ("comment_field", "S256"),
            # Number of extended headers
            ("nb_ext_headers", "uint32"),
        ]

        nev_basic_header = np.fromfile(filename, count=1, dtype=dt0)[0]

        shape = nev_basic_header["nb_ext_headers"]
        offset_dt0 = np.dtype(dt0).itemsize

        # This is the common structure of the beginning of extended headers
        dt1 = [("packet_id", "S8"), ("info_field", "S24")]

        raw_ext_header = np.memmap(filename, offset=offset_dt0, dtype=dt1, shape=shape, mode="r")


        # Get extended header types for this spec
        header_types = NEV_EXT_HEADER_TYPES_BY_SPEC[spec]
        
        # Parse extended headers by packet type
        # Strategy: view() entire array first, then mask for efficiency
        # Since all NEV extended header packets are fixed-width (32 bytes), temporarily
        # interpreting a "NEUEVWAV" packet as "ARRAYNME" structure is safe - the raw bytes
        # are just reinterpreted without copying. We immediately filter out mismatched packets
        # with the mask, keeping only those that actually belong to the current packet type.
        nev_ext_header = {}
        for packet_id, dtype_def in header_types.items():
            mask = raw_ext_header["packet_id"] == packet_id
            nev_ext_header[packet_id] = raw_ext_header.view(dtype_def)[mask]

        return nev_basic_header, nev_ext_header
    def _read_nev_data(self, spec, filename):
        """
        Extract nev data for any specification version.
        
        Parameters
        ----------
        spec : str
            The specification version (e.g., "2.1", "2.2", "2.3", "3.0")
        filename : str
            The NEV filename to read from
        """
        packet_size_bytes = self._nev_basic_header["bytes_in_data_packets"]
        header_size = self._nev_basic_header["bytes_in_headers"]

        if self._nev_basic_header["ver_major"] >= 3:
            ts_format = "uint64"
            header_skip = 10
        else:
            ts_format = "uint32"
            header_skip = 6

        # read all raw data packets and markers
        dt0 = [("timestamp", ts_format), ("packet_id", "uint16"), ("value", f"S{packet_size_bytes - header_skip}")]

        # expected number of data packets. We are not sure why, but it seems we can get partial data packets
        # based on blackrock's own code this is okay so applying an int to round down is necessary to obtain the
        # memory map of full packets and toss the partial packet.
        # See reference: https://github.com/BlackrockNeurotech/Python-Utilities/blob/fa75aa671680306788e10d3d8dd625f9da4ea4f6/brpylib/brpylib.py#L580-L587
        data_packages_in_bytes = self._get_file_size(filename) - header_size
        n_packets = int(data_packages_in_bytes / packet_size_bytes)

        raw_data = np.memmap(
            filename,
            offset=header_size,
            dtype=dt0,
            shape=(n_packets,),
            mode="r",
        )

        # Get packet identifiers and types directly from spec-based dictionaries
        packet_identifiers = NEV_PACKET_IDENTIFIERS_BY_SPEC[spec]
        data_types = NEV_PACKET_DATA_TYPES_BY_SPEC[spec]

        # Apply masks and create type definitions
        masks = {}
        types = {}
        for data_type, packet_id_spec in packet_identifiers.items():
            if isinstance(packet_id_spec, tuple):
                # Range check (min, max)
                min_val, max_val = packet_id_spec
                masks[data_type] = (min_val <= raw_data["packet_id"]) & (raw_data["packet_id"] <= max_val)
            else:
                # Equality check
                masks[data_type] = (raw_data["packet_id"] == packet_id_spec)
            
            types[data_type] = data_types[data_type](packet_size_bytes)

        event_segment_ids = self._get_event_segment_ids(raw_data, masks, spec)

        # Extract data for each packet type using view-then-mask pattern
        # Strategy: reinterpret entire raw_data array with each packet type's structure, then filter
        # All NEV data packets are fixed-width, so temporarily viewing "Spikes" data as "Comments"
        # structure is safe - we immediately filter to keep only packets that actually match.
        # This avoids creating copies of large data arrays during the parsing process.
        data = {}
        for data_type in packet_identifiers:
            mask = masks[data_type]
            data[data_type] = (raw_data.view(types[data_type])[mask], event_segment_ids[mask])

        return data


    def _get_reset_event_mask(self, raw_event_data, masks, spec):
        """
        Extract mask for reset comment events in 2.3 .nev file
        """
        if "Comments" not in masks:
            return np.zeros(len(raw_event_data), dtype=bool)
            
        restart_mask = np.logical_and(
            masks["Comments"],
            raw_event_data["value"] == b"\x00\x00\x00\x00\x00\x00critical load restart",
        )
        # TODO: Fix hardcoded number of bytes
        return restart_mask

    def _get_event_segment_ids(self, raw_event_data, masks, spec):
        """
        Construct array of corresponding segment ids for each event for nev version 2.3
        """

        if spec in ["2.1", "2.2"]:
            # No pause or reset mechanism present for file version 2.1 and 2.2
            return np.zeros(len(raw_event_data), dtype=int)

        elif spec in ["2.3", "3.0"]:
            reset_ev_mask = self._get_reset_event_mask(raw_event_data, masks, spec)
            reset_ev_ids = np.where(reset_ev_mask)[0]

            # consistency check for monotone increasing time stamps
            # - Use logical comparator (instead of np.diff) to avoid unsigned dtype issues.
            # - Only consider handled/known event types.
            mask_handled = np.any([mask for mask in masks.values()], axis=0)
            jump_ids_handled = (
                np.where(
                    raw_event_data["timestamp"][mask_handled][1:] < raw_event_data["timestamp"][mask_handled][:-1]
                )[0]
                + 1
            )
            jump_ids = np.where(mask_handled)[0][jump_ids_handled]  # jump ids in full set of events (incl. unhandled)
            overlap = np.isin(jump_ids, reset_ev_ids)
            if not all(overlap):
                # additional resets occurred without a reset event being stored
                additional_ids = jump_ids[np.invert(overlap)]
                warnings.warn(
                    f"Detected {len(additional_ids)} undocumented segments within "
                    f"nev data after timestamps {additional_ids}."
                    ""
                )
                reset_ev_ids = sorted(np.unique(np.concatenate((reset_ev_ids, jump_ids))))

            event_segment_ids = np.zeros(len(raw_event_data), dtype=int)
            for reset_event_id in reset_ev_ids:
                event_segment_ids[reset_event_id:] += 1

            self._nb_segment_nev = len(reset_ev_ids) + 1
            return event_segment_ids

        else:
            raise ValueError(f"Unknown File Spec {spec}")

    def _match_nsx_and_nev_segment_ids(self, nsx_nb):
        """
        Ensure matching ids of segments detected in nsx and nev file for version 2.3
        """

        # NSX required for matching, if not available, warn the user
        if not self._avail_nsx:
            warnings.warn(
                "No nsX available so it cannot be checked whether "
                "the segments in nev are all correct. Most importantly, "
                "recording pauses will not be detected",
                UserWarning,
            )
            return

        # Only needs to be done for nev version 2.3
        if self._nev_spec == "2.3":
            nsx_offset = self._nsx_data_header[nsx_nb][0]["timestamp"]
            # Multiples of 1/30.000s that pass between two nsX samples
            nsx_period = self._nsx_basic_header[nsx_nb]["period"]
            # NSX segments needed as dict and list
            nonempty_nsx_segments = {}
            list_nonempty_nsx_segments = []
            # Counts how many segments CAN be created from nev
            nb_possible_nev_segments = self._nb_segment_nev

            # Nonempty segments are those containing at least 2 samples
            # These have to be able to be mapped to nev
            for k, v in sorted(self._nsx_data_header[nsx_nb].items()):
                if v["nb_data_points"] > 1:
                    nonempty_nsx_segments[k] = v
                    list_nonempty_nsx_segments.append(v)

            # Account for paused segments
            # This increases nev event segment ids if from the nsx an additional segment is found
            # If one new segment, i.e. that could not be determined from the nev was found,
            # all following ids need to be increased to account for the additional segment before
            for k, (data, ev_ids) in self.nev_data.items():

                # Check all nonempty nsX segments
                for i, seg in enumerate(list_nonempty_nsx_segments[:]):

                    # Last timestamp in this nsX segment
                    # Not subtracting nsX offset from end because spike extraction might continue
                    end_of_current_nsx_seg = (
                        seg["timestamp"] + seg["nb_data_points"] * self._nsx_basic_header[nsx_nb]["period"]
                    )

                    mask_after_seg = (ev_ids == i) & (data["timestamp"] > end_of_current_nsx_seg + nsx_period)

                    # Show warning if spikes do not fit any segment (+- 1 sampling 'tick')
                    # Spike should belong to segment before
                    mask_outside = (ev_ids == i) & (
                        data["timestamp"] < int(seg["timestamp"]) - int(nsx_offset) - int(nsx_period)
                    )

                    if len(data[mask_outside]) > 0:
                        warnings.warn(f"Spikes outside any segment. Detected on segment #{i}")
                        ev_ids[mask_outside] -= 1

                    # If some nev data are outside of this nsX segment, increase their segment ids
                    # and the ids of all following segments. They are checked for the next nsX
                    # segment then. If they do not fit any of them,
                    # a warning will be shown, indicating how far outside the segment spikes are
                    # If they fit the next segment, more segments are possible in nev,
                    # because a new one has been discovered
                    if len(data[mask_after_seg]) > 0:
                        # Warning if spikes are after last segment
                        if i == len(list_nonempty_nsx_segments) - 1:
                            # Get timestamp resolution from header (available for v2.2+)
                            timestamp_resolution = self._nsx_basic_header[nsx_nb]["timestamp_resolution"]
                            time_after_seg = (
                                data[mask_after_seg]["timestamp"][-1] - end_of_current_nsx_seg
                            ) / timestamp_resolution
                            warnings.warn(f"Spikes {time_after_seg}s after last segment.")
                            # Break out of loop because it's the last iteration
                            # and the spikes should stay connected to last segment
                            break

                        # If reset and no segment detected in nev, then these segments cannot be
                        # distinguished in nev, which is a big problem
                        # XXX 96 is an arbitrary number based on observations in available files
                        elif list_nonempty_nsx_segments[i + 1]["timestamp"] - nsx_offset <= 96:
                            # If not all definitely belong to the next segment,
                            # then it cannot be distinguished where some belong
                            if len(data[ev_ids == i]) != len(data[mask_after_seg]):
                                raise ValueError("Some segments in nsX cannot be detected in nev")

                        # Actual processing if no problem has occurred
                        nb_possible_nev_segments += 1
                        ev_ids[ev_ids > i] += 1
                        ev_ids[mask_after_seg] += 1

            # consistency check: same number of segments for nsx and nev data
            if nb_possible_nev_segments != len(nonempty_nsx_segments):
                raise NeoReadWriteError(
                    f"Inconsistent ns{nsx_nb} and nev file. {nb_possible_nev_segments} "
                    f"segments present in .nev file, but {len(nonempty_nsx_segments)} in "
                    f"ns{nsx_nb} file."
                )

            new_nev_segment_id_mapping = dict(zip(range(nb_possible_nev_segments), sorted(list(nonempty_nsx_segments))))

            # replacing event ids by matched event ids in place
            for k, (data, ev_ids) in self.nev_data.items():
                if len(ev_ids):
                    ev_ids[:] = np.vectorize(new_nev_segment_id_mapping.__getitem__)(ev_ids)



    def _nev_params(self, param_name):
        """
        Returns wanted nev parameter.
        """
        nev_parameters = {
            "bytes_in_data_packets": self._nev_basic_header["bytes_in_data_packets"],
            "rec_datetime": datetime.datetime(
                year=self._nev_basic_header["year"],
                month=self._nev_basic_header["month"],
                day=self._nev_basic_header["day"],
                hour=self._nev_basic_header["hour"],
                minute=self._nev_basic_header["minute"],
                second=self._nev_basic_header["second"],
                microsecond=int(self._nev_basic_header["millisecond"]) * 1000,
            ),
            "max_res": self._nev_basic_header["timestamp_resolution"],
            "channel_ids": self._nev_ext_header[b"NEUEVWAV"]["electrode_id"],
            "channel_labels": self._channel_labels[self._nev_spec](),
            "event_unit": pq.CompoundUnit(f"1.0/{self._nev_basic_header['timestamp_resolution']} * s"),
            "nb_units": dict(
                zip(
                    self._nev_ext_header[b"NEUEVWAV"]["electrode_id"],
                    self._nev_ext_header[b"NEUEVWAV"]["nb_sorted_units"],
                )
            ),
            "digitization_factor": dict(
                zip(
                    self._nev_ext_header[b"NEUEVWAV"]["electrode_id"],
                    self._nev_ext_header[b"NEUEVWAV"]["digitization_factor"],
                )
            ),
            "data_size": self._nev_basic_header["bytes_in_data_packets"],
            "waveform_size": self._waveform_size[self._nev_spec](),
            "waveform_dtypes": self._get_waveforms_dtype(),
            "waveform_sampling_rate": self._nev_basic_header["sample_resolution"] * pq.Hz,
            "waveform_time_unit": pq.CompoundUnit(f"1.0/{self._nev_basic_header['sample_resolution']} * s"),
            "waveform_unit": pq.uV,
        }

        return nev_parameters[param_name]

    def _get_file_size(self, filename):
        """
        Returns the file size in bytes for the given file.
        """
        filebuf = open(filename, "rb")
        filebuf.seek(0, os.SEEK_END)
        file_size = int(filebuf.tell())
        filebuf.close()

        return file_size

    def _get_min_time(self):
        """
        Returns the smallest time that can be determined from the recording for
        use as the lower bound n in an interval [n,m).
        """
        tp = []
        if self._avail_files["nev"]:
            tp.extend(self._get_nev_rec_times()[0])
        for nsx_i in self._avail_nsx:
            tp.extend(self._nsx_rec_times[self._nsx_spec[nsx_i]](nsx_i)[0])

        return min(tp)

    def _get_max_time(self):
        """
        Returns the largest time that can be determined from the recording for
        use as the upper bound m in an interval [n,m).
        """
        tp = []
        if self._avail_files["nev"]:
            tp.extend(self._get_nev_rec_times()[1])
        for nsx_i in self._avail_nsx:
            tp.extend(self._nsx_rec_times[self._nsx_spec[nsx_i]](nsx_i)[1])

        return max(tp)

    def _get_nev_rec_times(self):
        """
        Extracts minimum and maximum time points from a nev file.
        """
        filename = ".".join([self._filenames["nev"], "nev"])

        dt = [("timestamp", "uint32")]
        offset = self._get_file_size(filename) - self._nev_params("bytes_in_data_packets")
        last_data_packet = np.memmap(filename, offset=offset, dtype=dt, mode="r")[0]

        n_starts = [0 * self._nev_params("event_unit")]
        n_stops = [last_data_packet["timestamp"] * self._nev_params("event_unit")]

        return n_starts, n_stops

    def _get_waveforms_dtype(self):
        """
        Extracts the actual waveform dtype set for each channel.
        """
        # Blackrock code giving the appropriate dtype
        conv = {0: "int8", 1: "int8", 2: "int16", 4: "int32"}

        # get all electrode ids from nev ext header
        all_el_ids = self._nev_ext_header[b"NEUEVWAV"]["electrode_id"]

        # get the dtype of waveform (this is stupidly complicated)
        if self._is_set(np.array(self._nev_basic_header["additionnal_flags"]), 0):
            dtype_waveforms = {k: "int16" for k in all_el_ids}
        else:
            # extract bytes per waveform
            waveform_bytes = self._nev_ext_header[b"NEUEVWAV"]["bytes_per_waveform"]
            # extract dtype for waveforms fro each electrode
            dtype_waveforms = dict(zip(all_el_ids, conv[waveform_bytes]))

        return dtype_waveforms

    def _get_channel_labels_spec_v21(self):
        """
        Returns labels for all channels for file spec 2.1
        """
        elids = self._nev_ext_header[b"NEUEVWAV"]["electrode_id"]
        labels = []

        for elid in elids:
            if elid < 129:
                labels.append(f"chan{elid}")
            else:
                labels.append(f"ainp{(elid - 12 +1 )}")

        return dict(zip(elids, labels))

    def _get_channel_labels_spec_v22_30(self):
        """
        Returns labels for all channels for file spec 2.2 and 2.3
        """
        elids = self._nev_ext_header[b"NEUEVWAV"]["electrode_id"]
        labels = self._nev_ext_header[b"NEUEVLBL"]["label"]

        return dict(zip(elids, labels)) if len(labels) > 0 else None

    def _get_waveform_size_spec_v21(self):
        """
        Returns waveform sizes for all channels for file spec 2.1 and 2.2
        """
        wf_dtypes = self._get_waveforms_dtype()
        nb_bytes_wf = self._nev_basic_header["bytes_in_data_packets"] - 8

        wf_sizes = {ch: int(nb_bytes_wf / np.dtype(dt).itemsize) for ch, dt in wf_dtypes.items()}

        return wf_sizes

    def _get_waveform_size_spec_v22_30(self):
        """
        Returns waveform sizes for all channels for file spec 2.3
        """
        elids = self._nev_ext_header[b"NEUEVWAV"]["electrode_id"]
        spike_widths = self._nev_ext_header[b"NEUEVWAV"]["spike_width"]

        return dict(zip(elids, spike_widths))

    def _get_left_sweep_waveforms(self):
        """
        Returns left sweep of waveforms for each channel. Left sweep is defined
        as the time from the beginning of the waveform to the trigger time of
        the corresponding spike.
        """
        # TODO: Double check if this is the actual setting for Blackrock
        wf_t_unit = self._nev_params("waveform_time_unit")
        all_ch = self._nev_params("channel_ids")

        # TODO: Double check if this is the correct assumption (10 samples)
        # default value: threshold crossing after 10 samples of waveform
        wf_left_sweep = {ch: 10 * wf_t_unit for ch in all_ch}

        # non-default: threshold crossing at center of waveform
        # wf_size = self._nev_params('waveform_size')
        # wf_left_sweep = dict(
        #     [(ch, (wf_size[ch] / 2) * wf_t_unit) for ch in all_ch])

        return wf_left_sweep

    def _get_nonneural_evdicts_spec_v21_22(self, data):
        """
        Defines event types and the necessary parameters to extract them from
        a 2.1 and 2.2 nev file.
        """
        # TODO: add annotations of nev ext header (NSASEXEX) to event types

        # digital events
        event_types = {
            "digital_input_port": {
                "name": "digital_input_port",
                "field": "digital_input",
                "mask": data["packet_insertion_reason"] == 1,
                "desc": "Events of the digital input port",
            },
            "serial_input_port": {
                "name": "serial_input_port",
                "field": "digital_input",
                "mask": data["packet_insertion_reason"] == 129,
                "desc": "Events of the serial input port",
            },
        }

        # analog input events via threshold crossings
        for ch in range(5):
            event_types.update(
                {
                    f"analog_input_channel_{ch + 1}": {
                        "name": f"analog_input_channel_{ch + 1}",
                        "field": f"analog_input_channel_{ch + 1}",
                        "mask": self._is_set(data["packet_insertion_reason"], ch + 1),
                        "desc": f"Values of analog input channel {ch + 1} in mV " "(+/- 5000)",
                    }
                }
            )

        # TODO: define field and desc
        event_types.update(
            {
                "periodic_sampling_events": {
                    "name": "periodic_sampling_events",
                    "field": "digital_input",
                    "mask": self._is_set(data["packet_insertion_reason"], 6),
                    "desc": "Periodic sampling event of a certain frequency",
                }
            }
        )

        return event_types

    def _delete_empty_segments(self):
        """
        If there are empty segments (e.g. due to a reset or clock synchronization across
        two systems), these can be discarded.
        If this is done, all the data and data_headers need to be remapped to become a range
        starting from 0 again. Nev data are mapped accordingly to stay with their corresponding
        segment in the nsX data.
        """

        # Discard empty segments
        removed_seg = []
        for data_bl in range(self._nb_segment):
            keep_seg = True
            for nsx_nb in self.nsx_to_load:
                length = self.nsx_datas[nsx_nb][data_bl].shape[0]
                keep_seg = keep_seg and (length >= 2)

            if not keep_seg:
                removed_seg.append(data_bl)
                for nsx_nb in self.nsx_to_load:
                    self.nsx_datas[nsx_nb].pop(data_bl)
                    self._nsx_data_header[nsx_nb].pop(data_bl)

        # Keys need to be increasing from 0 to maximum in steps of 1
        # To ensure this after removing empty segments, some keys need to be re mapped
        for i in removed_seg[::-1]:
            for j in range(i + 1, self._nb_segment):
                # remap nsx seg index
                for nsx_nb in self.nsx_to_load:
                    data = self.nsx_datas[nsx_nb].pop(j)
                    self.nsx_datas[nsx_nb][j - 1] = data

                    data_header = self._nsx_data_header[nsx_nb].pop(j)
                    self._nsx_data_header[nsx_nb][j - 1] = data_header

                # Also remap nev data, ev_ids are the equivalent to keys above
                if self._avail_files["nev"]:
                    for k, (data, ev_ids) in self.nev_data.items():
                        ev_ids[ev_ids == j] -= 1

            self._nb_segment -= 1

    def _get_nonneural_evdicts_spec_v23(self, data):
        """
        Defines event types and the necessary parameters to extract them from
        a 2.3 nev file.
        """
        # digital events
        if not np.all(np.isin(data["packet_insertion_reason"], [1, 129])):
            # Blackrock spec gives reason==64 means PERIODIC, but never seen this live
            warnings.warn("Unknown event codes found", RuntimeWarning)
        event_types = {
            "digital_input_port": {
                "name": "digital_input_port",
                "field": "digital_input",
                "mask": self._is_set(data["packet_insertion_reason"], 0)
                & ~self._is_set(data["packet_insertion_reason"], 7),
                "desc": "Events of the digital input port",
            },
            "serial_input_port": {
                "name": "serial_input_port",
                "field": "digital_input",
                "mask": self._is_set(data["packet_insertion_reason"], 0)
                & self._is_set(data["packet_insertion_reason"], 7),
                "desc": "Events of the serial input port",
            },
        }

        return event_types

    def _get_comment_evdict_spec_v21_22(self, data):
        return {
            "comments": {"name": "comments", "field": "comment", "mask": data["packet_id"] == 65535, "desc": "Comments"}
        }

    def _is_set(self, flag, pos):
        """
        Checks if bit is set at the given position for flag. If flag is an
        array, an array will be returned.
        """
        return flag & (1 << pos) > 0


# Extended header types for different NEV file specifications
# Structure: {spec: {packet_id: data_type_definition}}
NEV_EXT_HEADER_TYPES_BY_SPEC = {
    "2.1": {
        b"NEUEVWAV": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("physical_connector", "uint8"),
            ("connector_pin", "uint8"),
            ("digitization_factor", "uint16"),
            ("energy_threshold", "uint16"),
            ("hi_threshold", "int16"),
            ("lo_threshold", "int16"),
            ("nb_sorted_units", "uint8"),
            ("bytes_per_waveform", "uint8"),
            ("unused", "S10"),
        ],
        b"ARRAYNME": [("packet_id", "S8"), ("electrode_array_name", "S24")],
        b"ECOMMENT": [("packet_id", "S8"), ("extra_comment", "S24")],
        b"CCOMMENT": [("packet_id", "S8"), ("continued_comment", "S24")],
        b"MAPFILE": [("packet_id", "S8"), ("mapFile", "S24")],
        b"NSASEXEV": [
            ("packet_id", "S8"),
            ("frequency", "uint16"),
            ("digital_input_config", "uint8"),
            ("analog_channel_1_config", "uint8"),
            ("analog_channel_1_edge_detec_val", "uint16"),
            ("analog_channel_2_config", "uint8"),
            ("analog_channel_2_edge_detec_val", "uint16"),
            ("analog_channel_3_config", "uint8"),
            ("analog_channel_3_edge_detec_val", "uint16"),
            ("analog_channel_4_config", "uint8"),
            ("analog_channel_4_edge_detec_val", "uint16"),
            ("analog_channel_5_config", "uint8"),
            ("analog_channel_5_edge_detec_val", "uint16"),
            ("unused", "S6"),
        ],
    },
    "2.2": {
        b"NEUEVWAV": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("physical_connector", "uint8"),
            ("connector_pin", "uint8"),
            ("digitization_factor", "uint16"),
            ("energy_threshold", "uint16"),
            ("hi_threshold", "int16"),
            ("lo_threshold", "int16"),
            ("nb_sorted_units", "uint8"),
            ("bytes_per_waveform", "uint8"),
            ("spike_width", "uint16"),
            ("unused", "S8"),
        ],
        b"ARRAYNME": [("packet_id", "S8"), ("electrode_array_name", "S24")],
        b"ECOMMENT": [("packet_id", "S8"), ("extra_comment", "S24")],
        b"CCOMMENT": [("packet_id", "S8"), ("continued_comment", "S24")],
        b"MAPFILE": [("packet_id", "S8"), ("mapFile", "S24")],
        b"NEUEVLBL": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("label", "S16"),
            ("unused", "S6"),
        ],
        b"NEUEVFLT": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("hi_freq_corner", "uint32"),
            ("hi_freq_order", "uint32"),
            ("hi_freq_type", "uint16"),
            ("lo_freq_corner", "uint32"),
            ("lo_freq_order", "uint32"),
            ("lo_freq_type", "uint16"),
            ("unused", "S2"),
        ],
        b"DIGLABEL": [
            ("packet_id", "S8"),
            ("label", "S16"),
            ("mode", "uint8"),
            ("unused", "S7"),
        ],
        b"NSASEXEV": [
            ("packet_id", "S8"),
            ("frequency", "uint16"),
            ("digital_input_config", "uint8"),
            ("analog_channel_1_config", "uint8"),
            ("analog_channel_1_edge_detec_val", "uint16"),
            ("analog_channel_2_config", "uint8"),
            ("analog_channel_2_edge_detec_val", "uint16"),
            ("analog_channel_3_config", "uint8"),
            ("analog_channel_3_edge_detec_val", "uint16"),
            ("analog_channel_4_config", "uint8"),
            ("analog_channel_4_edge_detec_val", "uint16"),
            ("analog_channel_5_config", "uint8"),
            ("analog_channel_5_edge_detec_val", "uint16"),
            ("unused", "S6"),
        ],
    },
    "2.3": {
        b"NEUEVWAV": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("physical_connector", "uint8"),
            ("connector_pin", "uint8"),
            ("digitization_factor", "uint16"),
            ("energy_threshold", "uint16"),
            ("hi_threshold", "int16"),
            ("lo_threshold", "int16"),
            ("nb_sorted_units", "uint8"),
            ("bytes_per_waveform", "uint8"),
            ("spike_width", "uint16"),
            ("unused", "S8"),
        ],
        b"ARRAYNME": [("packet_id", "S8"), ("electrode_array_name", "S24")],
        b"ECOMMENT": [("packet_id", "S8"), ("extra_comment", "S24")],
        b"CCOMMENT": [("packet_id", "S8"), ("continued_comment", "S24")],
        b"MAPFILE": [("packet_id", "S8"), ("mapFile", "S24")],
        b"NEUEVLBL": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("label", "S16"),
            ("unused", "S6"),
        ],
        b"NEUEVFLT": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("hi_freq_corner", "uint32"),
            ("hi_freq_order", "uint32"),
            ("hi_freq_type", "uint16"),
            ("lo_freq_corner", "uint32"),
            ("lo_freq_order", "uint32"),
            ("lo_freq_type", "uint16"),
            ("unused", "S2"),
        ],
        b"DIGLABEL": [
            ("packet_id", "S8"),
            ("label", "S16"),
            ("mode", "uint8"),
            ("unused", "S7"),
        ],
        b"VIDEOSYN": [
            ("packet_id", "S8"),
            ("video_source_id", "uint16"),
            ("video_source", "S16"),
            ("frame_rate", "float32"),
            ("unused", "S2"),
        ],
        b"TRACKOBJ": [
            ("packet_id", "S8"),
            ("trackable_type", "uint16"),
            ("trackable_id", "uint16"),
            ("point_count", "uint16"),
            ("video_source", "S16"),
            ("unused", "S2"),
        ],
        b"NSASEXEV": [
            ("packet_id", "S8"),
            ("frequency", "uint16"),
            ("digital_input_config", "uint8"),
            ("analog_channel_1_config", "uint8"),
            ("analog_channel_1_edge_detec_val", "uint16"),
            ("analog_channel_2_config", "uint8"),
            ("analog_channel_2_edge_detec_val", "uint16"),
            ("analog_channel_3_config", "uint8"),
            ("analog_channel_3_edge_detec_val", "uint16"),
            ("analog_channel_4_config", "uint8"),
            ("analog_channel_4_edge_detec_val", "uint16"),
            ("analog_channel_5_config", "uint8"),
            ("analog_channel_5_edge_detec_val", "uint16"),
            ("unused", "S6"),
        ],
    },
    "3.0": {
        # Version 3.0 uses the same structure as 2.3
        b"NEUEVWAV": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("physical_connector", "uint8"),
            ("connector_pin", "uint8"),
            ("digitization_factor", "uint16"),
            ("energy_threshold", "uint16"),
            ("hi_threshold", "int16"),
            ("lo_threshold", "int16"),
            ("nb_sorted_units", "uint8"),
            ("bytes_per_waveform", "uint8"),
            ("spike_width", "uint16"),
            ("unused", "S8"),
        ],
        b"ARRAYNME": [("packet_id", "S8"), ("electrode_array_name", "S24")],
        b"ECOMMENT": [("packet_id", "S8"), ("extra_comment", "S24")],
        b"CCOMMENT": [("packet_id", "S8"), ("continued_comment", "S24")],
        b"MAPFILE": [("packet_id", "S8"), ("mapFile", "S24")],
        b"NEUEVLBL": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("label", "S16"),
            ("unused", "S6"),
        ],
        b"NEUEVFLT": [
            ("packet_id", "S8"),
            ("electrode_id", "uint16"),
            ("hi_freq_corner", "uint32"),
            ("hi_freq_order", "uint32"),
            ("hi_freq_type", "uint16"),
            ("lo_freq_corner", "uint32"),
            ("lo_freq_order", "uint32"),
            ("lo_freq_type", "uint16"),
            ("unused", "S2"),
        ],
        b"DIGLABEL": [
            ("packet_id", "S8"),
            ("label", "S16"),
            ("mode", "uint8"),
            ("unused", "S7"),
        ],
        b"VIDEOSYN": [
            ("packet_id", "S8"),
            ("video_source_id", "uint16"),
            ("video_source", "S16"),
            ("frame_rate", "float32"),
            ("unused", "S2"),
        ],
        b"TRACKOBJ": [
            ("packet_id", "S8"),
            ("trackable_type", "uint16"),
            ("trackable_id", "uint16"),
            ("point_count", "uint16"),
            ("video_source", "S16"),
            ("unused", "S2"),
        ],
        b"NSASEXEV": [
            ("packet_id", "S8"),
            ("frequency", "uint16"),
            ("digital_input_config", "uint8"),
            ("analog_channel_1_config", "uint8"),
            ("analog_channel_1_edge_detec_val", "uint16"),
            ("analog_channel_2_config", "uint8"),
            ("analog_channel_2_edge_detec_val", "uint16"),
            ("analog_channel_3_config", "uint8"),
            ("analog_channel_3_edge_detec_val", "uint16"),
            ("analog_channel_4_config", "uint8"),
            ("analog_channel_4_edge_detec_val", "uint16"),
            ("analog_channel_5_config", "uint8"),
            ("analog_channel_5_edge_detec_val", "uint16"),
            ("unused", "S6"),
        ],
    },
}


# Packet identifiers for different NEV file specifications
# Used to create masks that filter raw data packets by their packet ID field.
# Single values indicate equality check, tuples (min, max) indicate range check.
# According to NEV spec: packet IDs < 32768 identify channels, IDs >= 32768 are system events.
NEV_PACKET_IDENTIFIERS_BY_SPEC = {
    "2.1": {
        "NonNeural": 0,
        "Spikes": (1, 255),  # Packet IDs in this range identify spike events on electrodes
    },
    "2.2": {
        "NonNeural": 0,
        "Spikes": (1, 255),  # Packet IDs in this range identify spike events on electrodes
    },
    "2.3": {
        "NonNeural": 0,
        "Spikes": (1, 2048),  # Packet IDs in this range identify spike events on electrodes
        "Comments": 0xFFFF,
        "VideoSync": 0xFFFE,
        "TrackingEvents": 0xFFFD,
        "ButtonTrigger": 0xFFFC,
        "ConfigEvent": 0xFFFB,
    },
    "3.0": {
        "NonNeural": 0,
        "Spikes": (1, 2048),  # Packet IDs in this range identify spike events on electrodes
        "Comments": 0xFFFF,
        "VideoSync": 0xFFFE,
        "TrackingEvents": 0xFFFD,
        "ButtonTrigger": 0xFFFC,
        "ConfigEvent": 0xFFFB,
    },
}


# Data types for different NEV file specifications
# Structure: {spec: {data_type: lambda function that returns dtype definition}}
NEV_PACKET_DATA_TYPES_BY_SPEC = {
    "2.1": {
        "NonNeural": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("packet_insertion_reason", "uint8"),
            ("reserved", "uint8"),
            ("digital_input", "uint16"),
            ("analog_input_channel_1", "int16"),
            ("analog_input_channel_2", "int16"),
            ("analog_input_channel_3", "int16"),
            ("analog_input_channel_4", "int16"),
            ("analog_input_channel_5", "int16"),
            ("unused", f"S{packet_size_bytes - 20}"),
        ],
        "Spikes": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("unit_class_nb", "uint8"),
            ("reserved", "uint8"),
            ("waveform", f"S{packet_size_bytes - 8}"),
        ],
    },
    "2.2": {
        "NonNeural": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("packet_insertion_reason", "uint8"),
            ("reserved", "uint8"),
            ("digital_input", "uint16"),
            ("analog_input_channel_1", "int16"),
            ("analog_input_channel_2", "int16"),
            ("analog_input_channel_3", "int16"),
            ("analog_input_channel_4", "int16"),
            ("analog_input_channel_5", "int16"),
            ("unused", f"S{packet_size_bytes - 20}"),
        ],
        "Spikes": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("unit_class_nb", "uint8"),
            ("reserved", "uint8"),
            ("waveform", f"S{packet_size_bytes - 8}"),
        ],
    },
    "2.3": {
        "NonNeural": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("packet_insertion_reason", "uint8"),
            ("reserved", "uint8"),
            ("digital_input", "uint16"),
            ("unused", f"S{packet_size_bytes - 10}"),
        ],
        "Spikes": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("unit_class_nb", "uint8"),
            ("reserved", "uint8"),
            ("waveform", f"S{packet_size_bytes - 8}"),
        ],
        "Comments": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("char_set", "uint8"),
            ("flag", "uint8"),
            ("color", "uint32"),
            ("comment", f"S{packet_size_bytes - 12}"),
        ],
        "VideoSync": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("video_file_nb", "uint16"),
            ("video_frame_nb", "uint32"),
            ("video_elapsed_time", "uint32"),
            ("video_source_id", "uint32"),
            ("unused", "int8", (packet_size_bytes - 20,)),
        ],
        "TrackingEvents": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("parent_id", "uint16"),
            ("node_id", "uint16"),
            ("node_count", "uint16"),
            ("point_count", "uint16"),
            ("tracking_points", "uint16", ((packet_size_bytes - 14) // 2,)),
        ],
        "ButtonTrigger": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("trigger_type", "uint16"),
            ("unused", "int8", (packet_size_bytes - 8,)),
        ],
        "ConfigEvent": lambda packet_size_bytes: [
            ("timestamp", "uint32"),
            ("packet_id", "uint16"),
            ("config_change_type", "uint16"),
            ("config_changed", f"S{packet_size_bytes - 8}"),
        ],
    },
    "3.0": {
        "NonNeural": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("packet_insertion_reason", "uint8"),
            ("dlen", "uint8"),
            ("digital_input", "uint16"),
            ("unused", f"S{packet_size_bytes - 14}"),
        ],
        "Spikes": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("unit_class_nb", "uint8"),
            ("dlen", "uint8"),
            ("waveform", f"S{packet_size_bytes - 12}"),
        ],
        "Comments": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("char_set", "uint8"),
            ("flag", "uint8"),
            ("color", "uint32"),
            ("comment", f"S{packet_size_bytes - 16}"),
        ],
        "VideoSync": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("video_file_nb", "uint16"),
            ("video_frame_nb", "uint32"),
            ("video_elapsed_time", "uint32"),
            ("video_source_id", "uint32"),
            ("unused", "int8", (packet_size_bytes - 24,)),
        ],
        "TrackingEvents": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("parent_id", "uint16"),
            ("node_id", "uint16"),
            ("node_count", "uint16"),
            ("point_count", "uint16"),
            ("tracking_points", "uint16", ((packet_size_bytes - 18) // 2,)),
        ],
        "ButtonTrigger": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("trigger_type", "uint16"),
            ("unused", "int8", (packet_size_bytes - 12,)),
        ],
        "ConfigEvent": lambda packet_size_bytes: [
            ("timestamp", "uint64"),
            ("packet_id", "uint16"),
            ("config_change_type", "uint16"),
            ("config_changed", f"S{packet_size_bytes - 12}"),
        ],
    },
}


# Basic header types for different NSX file specifications
NSX_BASIC_HEADER_TYPES = {
    "2.1": [
        ("file_id", "S8"),
        ("label", "S16"),
        ("period", "uint32"),
        ("channel_count", "uint32"),
    ],
    "2.2": [
        ("file_id", "S8"),
        ("ver_major", "uint8"),
        ("ver_minor", "uint8"),
        ("bytes_in_headers", "uint32"),
        ("label", "S16"),
        ("comment", "S256"),
        ("period", "uint32"),
        ("timestamp_resolution", "uint32"),
        ("year", "uint16"),
        ("month", "uint16"),
        ("weekday", "uint16"),
        ("day", "uint16"),
        ("hour", "uint16"),
        ("minute", "uint16"),
        ("second", "uint16"),
        ("millisecond", "uint16"),
        ("channel_count", "uint32"),
    ],
    "2.3": [
        ("file_id", "S8"),
        ("ver_major", "uint8"),
        ("ver_minor", "uint8"),
        ("bytes_in_headers", "uint32"),
        ("label", "S16"),
        ("comment", "S256"),
        ("period", "uint32"),
        ("timestamp_resolution", "uint32"),
        ("year", "uint16"),
        ("month", "uint16"),
        ("weekday", "uint16"),
        ("day", "uint16"),
        ("hour", "uint16"),
        ("minute", "uint16"),
        ("second", "uint16"),
        ("millisecond", "uint16"),
        ("channel_count", "uint32"),
    ],
    "3.0": [
        ("file_id", "S8"),
        ("ver_major", "uint8"),
        ("ver_minor", "uint8"),
        ("bytes_in_headers", "uint32"),
        ("label", "S16"),
        ("comment", "S256"),
        ("period", "uint32"),
        ("timestamp_resolution", "uint32"),
        ("year", "uint16"),
        ("month", "uint16"),
        ("weekday", "uint16"),
        ("day", "uint16"),
        ("hour", "uint16"),
        ("minute", "uint16"),
        ("second", "uint16"),
        ("millisecond", "uint16"),
        ("channel_count", "uint32"),
    ],
}


# Extended header types for different NSX file specifications
NSX_EXT_HEADER_TYPES = {
    "2.1": [
        ("electrode_id", "uint32"),
    ],
    "2.2": [
        ("type", "S2"),
        ("electrode_id", "uint16"),
        ("electrode_label", "S16"),
        ("physical_connector", "uint8"),
        ("connector_pin", "uint8"),
        ("min_digital_val", "int16"),
        ("max_digital_val", "int16"),
        ("min_analog_val", "int16"),
        ("max_analog_val", "int16"),
        ("units", "S16"),
        ("hi_freq_corner", "uint32"),
        ("hi_freq_order", "uint32"),
        ("hi_freq_type", "uint16"),
        ("lo_freq_corner", "uint32"),
        ("lo_freq_order", "uint32"),
        ("lo_freq_type", "uint16"),
    ],
    "2.3": [
        ("type", "S2"),
        ("electrode_id", "uint16"),
        ("electrode_label", "S16"),
        ("physical_connector", "uint8"),
        ("connector_pin", "uint8"),
        ("min_digital_val", "int16"),
        ("max_digital_val", "int16"),
        ("min_analog_val", "int16"),
        ("max_analog_val", "int16"),
        ("units", "S16"),
        ("hi_freq_corner", "uint32"),
        ("hi_freq_order", "uint32"),
        ("hi_freq_type", "uint16"),
        ("lo_freq_corner", "uint32"),
        ("lo_freq_order", "uint32"),
        ("lo_freq_type", "uint16"),
    ],
    "3.0": [
        ("type", "S2"),
        ("electrode_id", "uint16"),
        ("electrode_label", "S16"),
        ("physical_connector", "uint8"),
        ("connector_pin", "uint8"),
        ("min_digital_val", "int16"),
        ("max_digital_val", "int16"),
        ("min_analog_val", "int16"),
        ("max_analog_val", "int16"),
        ("units", "S16"),
        ("hi_freq_corner", "uint32"),
        ("hi_freq_order", "uint32"),
        ("hi_freq_type", "uint16"),
        ("lo_freq_corner", "uint32"),
        ("lo_freq_order", "uint32"),
        ("lo_freq_type", "uint16"),
    ],
}

# NSX Data Header Types by specification version
# These define the structure of data block headers within NSX files
NSX_DATA_HEADER_TYPES = {
    # Version 2.1 has no data headers - data is stored continuously after the main header
    "2.1": None,
    # Versions 2.2+ use data block headers with timestamp size based on major version
    "2.2": [("header_flag", "uint8"), ("timestamp", "uint32"), ("nb_data_points", "uint32")],
    "2.3": [("header_flag", "uint8"), ("timestamp", "uint32"), ("nb_data_points", "uint32")],
    "3.0": [("header_flag", "uint8"), ("timestamp", "uint64"), ("nb_data_points", "uint32")],
    # PTP variant has a completely different structure with samples embedded
    "3.0-ptp": lambda channel_count: [
        ("reserved", "uint8"),
        ("timestamps", "uint64"),
        ("num_data_points", "uint32"),
        ("samples", "int16", (channel_count,))
    ]
}
