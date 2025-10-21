"""
Class for reading data from pCLAMP and AxoScope
files (.abf version 1 and 2), developed by Molecular device/Axon technologies.

- abf = Axon binary file
- atf is a text file based format from axon that could be
  read by AsciiIO (but this file is less efficient.)


This code is a port of abfload and abf2load
written in Matlab (BSD-2-Clause licence) by :
 - Copyright (c) 2009, Forrest Collman, fcollman@princeton.edu
 - Copyright (c) 2004, Harald Hentschke
and available here:
http://www.mathworks.com/matlabcentral/fileexchange/22114-abf2load


The StringsSection parsing (parse_axon_soup) now relies on an idea
presented in pyABF MIT License Copyright (c) 2018 Scott W Harden
written by Scott Harden. His unofficial documentation for the formats
is here:
https://swharden.com/pyabf/abf2-file-format/
strings section:
[uModifierNameIndex, uCreatorNameIndex, uProtocolPathIndex, lFileComment, lADCCChannelNames, lADCUnitsIndex
lDACChannelNameIndex, lDACUnitIndex, lDACFilePath, nLeakSubtractADC]
['', 'Clampex', '', 'C:/path/protocol.pro', 'some comment', 'IN 0', 'mV', 'IN 1', 'mV', 'Cmd 0', 'pA',
'Cmd 1', 'pA', 'Cmd 2', 'mV', 'Cmd 3', 'mV']

Information on abf 1 and 2 formats is available here:
http://www.moleculardevices.com/pages/software/developer_info.html

This file supports the old (ABF1) and new (ABF2) format.
ABF1 (clampfit <=9) and ABF2 (clampfit >10)

All possible mode are possible :
    - event-driven variable-length mode 1 -> return several Segments per Block
    - event-driven fixed-length mode 2 or 5 -> return several Segments
    - gap free mode -> return one (or several) Segment in the Block

Supported : Read

Author: Samuel Garcia, JS Nowacki

Note: j.s.nowacki@gmail.com has a C++ library with SWIG bindings which also
reads abf files - would be good to cross-check

"""

import struct
import datetime
from io import open, BufferedReader

import numpy as np

from .baserawio import (
    BaseRawWithBufferApiIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)
from neo.core import NeoReadWriteError


class AxonRawIO(BaseRawWithBufferApiIO):
    """
    Class for Class for reading data from pCLAMP and AxoScope files (.abf version 1 and 2)

    Parameters
    ----------
    filename: str, default: ''
        The *.abf file to be read

    Notes
    -----
    This code is a port of abfload and abf2load written in Matlab (BSD-2-Clause licence) by
    Copyright (c) 2009, Forrest Collman, fcollman@princeton.edu
    Copyright (c) 2004, Harald Hentschke

    Examples
    --------

    >>> import neo.rawio
    >>> reader = neo.rawio.AxonRawIO(filename='mydata.abf')
    >>> reader.parse_header()
    >>> print(reader)

    """

    extensions = ["abf"]
    rawmode = "one-file"

    def __init__(self, filename=""):
        BaseRawWithBufferApiIO.__init__(self)
        self.filename = filename

    def _parse_header(self):
        info = self._axon_info = parse_axon_soup(self.filename)

        version = info["fFileVersionNumber"]

        # file format
        if info["nDataFormat"] == 0:
            sig_dtype = np.dtype("i2")
        elif info["nDataFormat"] == 1:
            sig_dtype = np.dtype("f4")

        if version < 2.0:
            nbchannel = info["nADCNumChannels"]
            head_offset = info["lDataSectionPtr"] * BLOCKSIZE + info["nNumPointsIgnored"] * sig_dtype.itemsize
            totalsize = info["lActualAcqLength"]
        elif version >= 2.0:
            nbchannel = info["sections"]["ADCSection"]["llNumEntries"]
            head_offset = info["sections"]["DataSection"]["uBlockIndex"] * BLOCKSIZE
            totalsize = info["sections"]["DataSection"]["llNumEntries"]

        # 3 possible modes
        if version < 2.0:
            mode = info["nOperationMode"]
        elif version >= 2.0:
            mode = info["protocol"]["nOperationMode"]

        if mode not in [1, 2, 3, 5]:
            raise NeoReadWriteError(f"Mode {mode} is not currently supported in Neo")
        # event-driven variable-length mode (mode 1)
        # event-driven fixed-length mode (mode 2 or 5)
        # gap free mode (mode 3) can be in several episodes

        # read sweep pos
        if version < 2.0:
            nbepisod = info["lSynchArraySize"]
            offset_episode = info["lSynchArrayPtr"] * BLOCKSIZE
        elif version >= 2.0:
            nbepisod = info["sections"]["SynchArraySection"]["llNumEntries"]
            offset_episode = info["sections"]["SynchArraySection"]["uBlockIndex"] * BLOCKSIZE
        if nbepisod > 0:
            episode_array = np.memmap(
                self.filename, [("offset", "i4"), ("len", "i4")], "r", shape=nbepisod, offset=offset_episode
            )
        else:
            episode_array = np.empty(1, [("offset", "i4"), ("len", "i4")])
            episode_array[0]["len"] = totalsize
            episode_array[0]["offset"] = 0

        # sampling_rate
        if version < 2.0:
            self._sampling_rate = 1.0 / (info["fADCSampleInterval"] * nbchannel * 1.0e-6)
        elif version >= 2.0:
            self._sampling_rate = 1.0e6 / info["protocol"]["fADCSequenceInterval"]

        # one sweep = one segment
        nb_segment = episode_array.size

        stream_id = "0"
        buffer_id = "0"

        # Get raw data by segment
        # self._raw_signals = {}
        self._t_starts = {}
        self._buffer_descriptions = {0: {}}
        self._stream_buffer_slice = {stream_id: None}
        pos = 0
        for seg_index in range(nb_segment):
            length = episode_array[seg_index]["len"]

            if version < 2.0:
                fSynchTimeUnit = info["fSynchTimeUnit"]
            elif version >= 2.0:
                fSynchTimeUnit = info["protocol"]["fSynchTimeUnit"]

            if (fSynchTimeUnit != 0) and (mode == 1):
                length /= fSynchTimeUnit

            self._buffer_descriptions[0][seg_index] = {}
            self._buffer_descriptions[0][seg_index][buffer_id] = {
                "type": "raw",
                "file_path": str(self.filename),
                "dtype": str(sig_dtype),
                "order": "C",
                "file_offset": head_offset + pos * sig_dtype.itemsize,
                "shape": (int(length // nbchannel), int(nbchannel)),
            }
            pos += length

            t_start = float(episode_array[seg_index]["offset"])
            if fSynchTimeUnit == 0:
                t_start = t_start / self._sampling_rate
            else:
                t_start = t_start * fSynchTimeUnit * 1e-6
            self._t_starts[seg_index] = t_start

        # Create channel header
        if version < 2.0:
            channel_ids = [chan_num for chan_num in info["nADCSamplingSeq"] if chan_num >= 0]
        else:
            channel_ids = list(range(nbchannel))

        signal_channels = []
        adc_nums = []
        for chan_index, chan_id in enumerate(channel_ids):
            if version < 2.0:
                name = info["sADCChannelName"][chan_id].replace(b" ", b"")
                units = safe_decode_units(info["sADCUnits"][chan_id])
                adc_num = info["nADCPtoLChannelMap"][chan_id]
            elif version >= 2.0:
                ADCInfo = info["listADCInfo"][chan_id]
                name = ADCInfo["ADCChNames"].replace(b" ", b"")
                units = safe_decode_units(ADCInfo["ADCChUnits"])
                adc_num = ADCInfo["nADCNum"]
            adc_nums.append(adc_num)

            if info["nDataFormat"] == 0:
                # int16 gain/offset
                if version < 2.0:
                    gain = info["fADCRange"]
                    gain /= info["fInstrumentScaleFactor"][chan_id]
                    gain /= info["fSignalGain"][chan_id]
                    gain /= info["fADCProgrammableGain"][chan_id]
                    gain /= info["lADCResolution"]
                    if info["nTelegraphEnable"][chan_id] == 0:
                        pass
                    elif info["nTelegraphEnable"][chan_id] == 1:
                        gain /= info["fTelegraphAdditGain"][chan_id]
                    else:
                        self.logger.warning("ignoring buggy nTelegraphEnable")
                    offset = info["fInstrumentOffset"][chan_id]
                    offset -= info["fSignalOffset"][chan_id]
                elif version >= 2.0:
                    gain = info["protocol"]["fADCRange"]
                    gain /= info["listADCInfo"][chan_id]["fInstrumentScaleFactor"]
                    gain /= info["listADCInfo"][chan_id]["fSignalGain"]
                    gain /= info["listADCInfo"][chan_id]["fADCProgrammableGain"]
                    gain /= info["protocol"]["lADCResolution"]
                    if info["listADCInfo"][chan_id]["nTelegraphEnable"]:
                        gain /= info["listADCInfo"][chan_id]["fTelegraphAdditGain"]
                    offset = info["listADCInfo"][chan_id]["fInstrumentOffset"]
                    offset -= info["listADCInfo"][chan_id]["fSignalOffset"]
            else:
                gain, offset = 1.0, 0.0

            signal_channels.append(
                (name, str(chan_id), self._sampling_rate, sig_dtype, units, gain, offset, stream_id, buffer_id)
            )

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # one unique signal stream and buffer
        signal_buffers = np.array([("Signals", buffer_id)], dtype=_signal_buffer_dtype)
        signal_streams = np.array([("Signals", stream_id, buffer_id)], dtype=_signal_stream_dtype)

        # only one events channel : tag
        # In ABF timstamps are not attached too any particular segment
        # so each segment acess all event
        timestamps = []
        labels = []
        comments = []
        for i, tag in enumerate(info["listTag"]):
            timestamps.append(tag["lTagTime"])
            labels.append(str(tag["nTagType"]))
            comments.append(str(clean_string(tag["sComment"])))
        self._raw_ev_timestamps = np.array(timestamps)
        self._ev_labels = np.array(labels, dtype="U")
        self._ev_comments = np.array(comments, dtype="U")
        event_channels = [("Tag", "", "event")]
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

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
        bl_annotations = self.raw_annotations["blocks"][0]

        bl_annotations["rec_datetime"] = info["rec_datetime"]
        bl_annotations["abf_version"] = version

        for seg_index in range(nb_segment):
            seg_annotations = bl_annotations["segments"][seg_index]
            seg_annotations["abf_version"] = version

            signal_an = self.raw_annotations["blocks"][0]["segments"][seg_index]["signals"][0]
            nADCNum = np.array([adc_nums[c] for c in range(signal_channels.size)])
            signal_an["__array_annotations__"]["nADCNum"] = nADCNum

            for c in range(event_channels.size):
                ev_ann = seg_annotations["events"][c]
                ev_ann["comments"] = self._ev_comments

    def _source_name(self):
        return self.filename

    def _segment_t_start(self, block_index, seg_index):
        return self._t_starts[seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        sig_size = self.get_signal_size(block_index, seg_index, 0)
        t_stop = self._t_starts[seg_index] + sig_size / self._sampling_rate
        return t_stop

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return self._t_starts[seg_index]

    def _get_analogsignal_buffer_description(self, block_index, seg_index, buffer_id):
        return self._buffer_descriptions[block_index][seg_index][buffer_id]

    def _event_count(self, block_index, seg_index, event_channel_index):
        return self._raw_ev_timestamps.size

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        # In ABF timestamps are not attached too any particular segment
        # so each segment accesses all events
        timestamp = self._raw_ev_timestamps
        labels = self._ev_labels
        durations = None

        if t_start is not None:
            keep = timestamp >= int(t_start * self._sampling_rate)
            timestamp = timestamp[keep]
            labels = labels[keep]

        if t_stop is not None:
            keep = timestamp <= int(t_stop * self._sampling_rate)
            timestamp = timestamp[keep]
            labels = labels[keep]

        return timestamp, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        event_times = event_timestamps.astype(dtype) / self._sampling_rate
        return event_times

    def read_raw_protocol(self):
        """
        Read stimulus waveforms (DAC output protocols) from ABF2 files.

        Purpose
        -------
        In electrophysiology experiments, the DAC (Digital-to-Analog Converter)
        outputs generate stimulus waveforms (voltage steps in voltage-clamp,
        current injections in current-clamp). This function reconstructs these
        commanded output waveforms from the protocol definition stored in the
        file, NOT from recorded data.

        Why This Matters
        ----------------
        - Recorded data shows what was MEASURED (ADC inputs: membrane voltage,
          current, etc.)
        - Protocol waveforms show what was COMMANDED (DAC outputs: stimulus
          patterns)
        - Having both allows you to analyze the relationship between stimulus
          and response

        Limitations
        -----------
        - ABF2 files only: ABF1 files don't store protocol sections, though
          protocols can theoretically be reconstructed from ABF1 headers
        - Waveforms are RECONSTRUCTED from epoch definitions, not recorded data
        - Some complex protocols may not reconstruct perfectly (see TODO in code)

        Returns
        -------
        sigs_by_segments : list of list of numpy.ndarray
            Outer list: one entry per episode/sweep (length = nEpi)
            Inner list: one entry per DAC channel (length = nDAC)
            Arrays: stimulus waveform samples (length = nSam samples per episode)

            Structure: sigs_by_segments[episode_idx][DAC_idx] = 1D array

            Example for 3 episodes, 2 DAC channels:
              sigs_by_segments[0][0] -> Episode 0, DAC 0 waveform (nSam samples)
              sigs_by_segments[0][1] -> Episode 0, DAC 1 waveform (nSam samples)
              sigs_by_segments[1][0] -> Episode 1, DAC 0 waveform (nSam samples)
              ...

        sig_names : list of str
            DAC channel names (length = nDAC)
            Example: ['Cmd 0', 'Cmd 1']

        sig_units : list of str
            DAC channel units (length = nDAC)
            Example: ['mV', 'pA']

        Example Usage
        -------------
        >>> reader = AxonRawIO(filename='mydata.abf')
        >>> reader.parse_header()
        >>> waveforms, names, units = reader.read_raw_protocol()
        >>> # Get DAC 0 waveform for episode 2:
        >>> dac0_ep2 = waveforms[2][0]
        >>> print(f"{names[0]} waveform in {units[0]}: {dac0_ep2}")

        Notes
        -----
        Waveforms are constructed from:
        - Holding levels (fDACHoldingLevel)
        - Epoch definitions (EpochPerDACSection) describing step changes
        - Episode number (waveforms can change across episodes via increment
          parameters)

        Author: JS Nowacki
        """
        info = self._axon_info

        if info["fFileVersionNumber"] < 2.0:
            raise IOError("Protocol section is only present in ABF2 files.")

        nADC = info["sections"]["ADCSection"]["llNumEntries"]  # Number of ADC channels
        nDAC = info["sections"]["DACSection"]["llNumEntries"]  # Number of DAC channels
        nSam = int(info["protocol"]["lNumSamplesPerEpisode"] / nADC)  # Number of samples per episode
        nEpi = info["lActualEpisodes"]  # Actual number of episodes

        # Make a list of segments with analog signals with just holding levels
        # List of segments relates to number of episodes, as for recorded data
        sigs_by_segments = []
        for epiNum in range(nEpi):
            # One analog signal for each DAC in segment (episode)
            signals = []
            for DACNum in range(nDAC):
                sig = np.ones(nSam) * info["listDACInfo"][DACNum]["fDACHoldingLevel"]
                # If there are epoch infos for this DAC
                if DACNum in info["dictEpochInfoPerDAC"]:
                    # Save last sample index
                    i_last = int(nSam * 15625 / 10**6)
                    # TODO guess for first holding
                    # Go over EpochInfoPerDAC and change the analog signal
                    # according to the epochs
                    epochInfo = info["dictEpochInfoPerDAC"][DACNum]
                    for epochNum, epoch in epochInfo.items():
                        i_begin = i_last
                        i_end = i_last + epoch["lEpochInitDuration"] + epoch["lEpochDurationInc"] * epiNum
                        dif = i_end - i_begin
                        sig[i_begin:i_end] = np.ones(dif) * (
                            epoch["fEpochInitLevel"] + epoch["fEpochLevelInc"] * epiNum
                        )
                        i_last += epoch["lEpochInitDuration"] + epoch["lEpochDurationInc"] * epiNum
                signals.append(sig)
            sigs_by_segments.append(signals)

        sig_names = []
        sig_units = []
        for DACNum in range(nDAC):
            name = info["listDACInfo"][DACNum]["DACChNames"].decode("utf-8")
            units = safe_decode_units(info["listDACInfo"][DACNum]["DACChUnits"])
            sig_names.append(name)
            sig_units.append(units)

        return sigs_by_segments, sig_names, sig_units


def parse_axon_soup(filename):
    """
    Parse ABF file header and return metadata dict.

    The strategy here differs from the original script under Matlab.
    In the original script for ABF2, it completes the header with
    information that is located in other structures.

    Returns info with sub dicts depending on version:
        ABF1: listTag
        ABF2: sections, protocol, listTags, listADCInfo, listDACInfo, dictEpochInfoPerDAC

    Parameters
    ----------
    filename : str
        Path to the ABF file

    Returns
    -------
    dict or None
        Header dictionary with file metadata, or None if file signature is invalid
    """
    with open(filename, "rb") as fid:
        f = StructFile(fid)
        signature = f.read(4)

        if signature == b"ABF ":
            return _parse_abf_v1(f, headerDescriptionV1)
        elif signature == b"ABF2":
            return _parse_abf_v2(f, headerDescriptionV2)
        else:
            return None


def _parse_abf_v1(f, header_description):
    """
    Parse ABF 1.x files (pCLAMP 6-9).

    Overview
    --------
    ABF 1.x uses a single, large, fixed-length header (~5KB) that contains
    EVERYTHING you need to parse the file. Think of it as a "road map" where
    every piece of metadata and every pointer is at a known, fixed location.

    Key Design Principle: DIRECT ACCESS
    ------------------------------------
    Unlike ABF 2.x which uses indirection (Section Index -> Sections), ABF 1.x
    puts all metadata directly in the header at fixed byte offsets.

    Example: To get the number of channels:
      Simply read 2 bytes at offset 120 -> nADCNumChannels

    Example: To find the data section:
      Read lDataSectionPtr at offset 40 -> multiply by 512 -> data location

    Limitations:
    - Maximum 16 ADC channels (arrays are fixed size in header)
    - Strings are fixed-length (384 bytes for protocol path, 10 bytes per
      channel name, etc.)
    - Cannot add new features without breaking file format

    Benefits:
    - Simple to parse (read header, done!)
    - Fast (direct memory mapping, no indirection)
    - Everything in one place

    Historical Context: ABF 1.x merged the older CLAMPEX and FETCHEX file
    formats into a single, more versatile format for electrophysiology data.

    Parsing Order (as implemented in this function):
    ------------------------------------------------
    1. Read entire fixed header (bytes 0-5119) -> all metadata available
    2. Parse Tag section (using lTagSectionPtr from header)
    3. Adjust protocol path formatting
    4. Compute datetime from header fields

    Format: Fixed header with static field offsets
    Block Size: 512 bytes
    Header Fields: See headerDescriptionV1 for complete field list

    File Layout:
    +----------------------------------------------------------------------+
    | HEADER SECTION (Blocks 0-9, ~5KB)                                   |
    +----------------------------------------------------------------------+
    | Contains all metadata for the entire file, including acquisition    |
    | parameters and pointers to all other data sections:                 |
    |   - File signature, version, operation mode                         |
    |   - Pointers to data/tag/synch sections (in block numbers)          |
    |   - Channel configuration (up to 16 ADC channels)                   |
    |   - Sampling parameters, gains, offsets                             |
    |   - Protocol path and timing information                            |
    | See headerDescriptionV1 for all fields (offset, format, name)       |
    +----------------------------------------------------------------------+
    | TAG SECTION (at lTagSectionPtr x 512)                                |
    +----------------------------------------------------------------------+
    | User-defined markers/comments during acquisition                    |
    | Each tag: 64 bytes (time, comment, type, voice tag)                 |
    | See TagInfoDescription for structure                                |
    +----------------------------------------------------------------------+
    | SYNCH ARRAY SECTION (Physical disk layout)                          |
    +----------------------------------------------------------------------+
    | Purpose: Index of episodes/sweeps within the continuous data stream |
    | Critical for episodic acquisition modes (mode 2, 5)                 |
    |                                                                       |
    | Electrophysiology Context:                                           |
    |   - Episode/Sweep = Single experimental trial or stimulation cycle  |
    |   - In voltage-clamp: one voltage step protocol execution           |
    |   - In current-clamp: one current injection sequence                |
    |   - Data contains multiple episodes concatenated in data section    |
    |                                                                       |
    | Neo Mapping:                                                         |
    |   - Each episode becomes one neo.Segment                             |
    |   - offset = starting sample index in the continuous data array      |
    |   - len = number of samples in this episode (all channels combined) |
    |                                                                       |
    | Start Position:                                                      |
    |   Byte = lSynchArrayPtr x 512                                        |
    |   Total entries = lSynchArraySize (number of episodes/segments)     |
    |                                                                       |
    | Each Entry Structure (8 bytes):                                      |
    |   Offset  Size  Type   Field        Description                     |
    |   0       4     int32  offset       Episode start position (samples)|
    |   4       4     int32  len          Episode length (samples)        |
    |                                                                       |
    | Example: 3 episodes, 3 channels, 5000 samples per channel per ep:   |
    |                                                                       |
    |   Disk Offset  | Content                                             |
    |   -------------|--------------------------------------------------   |
    |   +0  bytes    | Episode 0 offset   (4 bytes, int32) = 0            |
    |   +4  bytes    | Episode 0 length   (4 bytes, int32) = 15000        |
    |   +8  bytes    | Episode 1 offset   (4 bytes, int32) = 15000        |
    |   +12 bytes    | Episode 1 length   (4 bytes, int32) = 15000        |
    |   +16 bytes    | Episode 2 offset   (4 bytes, int32) = 30000        |
    |   +20 bytes    | Episode 2 length   (4 bytes, int32) = 15000        |
    |                                                                       |
    | Note: len = samples_per_channel x nADCNumChannels (interleaved)     |
    +----------------------------------------------------------------------+
    | DATA SECTION (Physical disk layout)                                 |
    +----------------------------------------------------------------------+
    | Start Position:                                                      |
    |   Byte = (lDataSectionPtr x 512) + (nNumPointsIgnored x item_size)  |
    |   item_size = 2 bytes if nDataFormat=0 (int16)                      |
    |   item_size = 4 bytes if nDataFormat=1 (float32)                    |
    |                                                                       |
    | Example: 3 channels (Ch0, Ch1, Ch2), int16 format:                  |
    |                                                                       |
    |   Disk Offset  | Content                                             |
    |   -------------|--------------------------------------------------   |
    |   +0  bytes    | Sample 0, Channel 0  (2 bytes, int16)              |
    |   +2  bytes    | Sample 0, Channel 1  (2 bytes, int16)              |
    |   +4  bytes    | Sample 0, Channel 2  (2 bytes, int16)              |
    |   +6  bytes    | Sample 1, Channel 0  (2 bytes, int16)              |
    |   +8  bytes    | Sample 1, Channel 1  (2 bytes, int16)              |
    |   +10 bytes    | Sample 1, Channel 2  (2 bytes, int16)              |
    |   +12 bytes    | Sample 2, Channel 0  (2 bytes, int16)              |
    |   +14 bytes    | Sample 2, Channel 1  (2 bytes, int16)              |
    |   +16 bytes    | Sample 2, Channel 2  (2 bytes, int16)              |
    |   ... continues for lActualAcqLength total samples                  |
    |                                                                       |
    | Pattern: Channels are interleaved within each time point            |
    |   [Ch0,Ch1,Ch2] [Ch0,Ch1,Ch2] [Ch0,Ch1,Ch2] ...                     |
    +----------------------------------------------------------------------+

    Parameters
    ----------
    f : StructFile
        File object positioned after reading the signature
    header_description : list
        List of (key, offset, fmt) tuples describing header structure

    Returns
    -------
    dict
        Header dictionary with file metadata
    """
    # construct dict
    header = {}
    for key, offset, fmt in header_description:
        val = f.read_f(fmt, offset=offset)
        if len(val) == 1:
            header[key] = val[0]
        else:
            header[key] = np.array(val)

    # correction of version number and starttime
    header["lFileStartTime"] += header["nFileStartMillisecs"] * 0.001

    # tags
    listTag = []
    for i in range(header["lNumTagEntries"]):
        f.seek(header["lTagSectionPtr"] + i * 64)
        tag = {}
        for key, fmt in TagInfoDescription:
            val = f.read_f(fmt)
            if len(val) == 1:
                tag[key] = val[0]
            else:
                tag[key] = np.array(val)
        listTag.append(tag)
    header["listTag"] = listTag

    # protocol name formatting
    header["sProtocolPath"] = clean_string(header["sProtocolPath"])
    header["sProtocolPath"] = header["sProtocolPath"].replace(b"\\", b"/")

    # date and time
    YY = 1900
    MM = 1
    DD = 1
    hh = int(header["lFileStartTime"] / 3600.0)
    mm = int((header["lFileStartTime"] - hh * 3600) / 60)
    ss = header["lFileStartTime"] - hh * 3600 - mm * 60
    ms = int(np.mod(ss, 1) * 1e6)
    ss = int(ss)
    header["rec_datetime"] = datetime.datetime(YY, MM, DD, hh, mm, ss, ms)

    return header


def _parse_abf_v2(f, header_description):
    """
    Parse ABF 2.x files (pCLAMP 10+, 2006-present).

    Overview
    --------
    ABF 2.x uses a "table of contents" architecture, similar to a book or
    database. Think of it as three layers:

    1. MAIN HEADER (bytes 0-75): Cover page with basic file info
    2. SECTION INDEX (bytes 76-827): Table of contents - WHERE everything is
    3. SECTIONS (variable locations): The actual data chapters

    Key Design Principle: INDIRECTION
    ----------------------------------
    Unlike ABF 1.x where everything is at fixed offsets, ABF 2.x uses a
    two-step lookup process:

    Step 1: Read Section Index to find WHERE a section lives
    Step 2: Jump to that location and read the section data

    Example: To read ADC channel info:
      1. Read sections['ADCSection'] -> {uBlockIndex: 50, uBytes: 128, llNumEntries: 3}
      2. Jump to byte (50 x 512) and read 3 entries of 128 bytes each

    Critical Feature: CENTRAL STRING POOL
    --------------------------------------
    All strings (channel names, units, paths) are stored ONCE in StringsSection.
    Other sections store only INTEGER INDICES pointing to strings, not the
    strings themselves. This is why StringsSection must be parsed before
    ADC/DAC sections.

    Example:
      StringsSection contains: ['Clampex', 'IN 0', 'mV', 'IN 1', 'pA', ...]
      ADCInfo[0] has lADCChannelNameIndex = 1  -> name is strings[1] = 'IN 0'
      ADCInfo[0] has lADCUnitsIndex = 2        -> units is strings[2] = 'mV'

    Benefits of This Architecture:
    - Variable channel/epoch counts (no hard limits like ABF 1.x's 16 channels)
    - Efficient storage (strings not duplicated)
    - Extensible (new section types don't break old parsers)

    Parsing Order (as implemented in this function):
    ------------------------------------------------
    1. Read Main Header (bytes 0-75) -> basic file metadata
    2. Read Section Index (bytes 76-827) -> build 'sections' dict
    3. Parse StringsSection -> build 'strings' array (MUST be before step 4!)
    4. Parse ADCSection -> use string indices to get channel names/units
    5. Parse ProtocolSection -> acquisition parameters
    6. Parse TagSection -> user markers
    7. Parse DACSection -> use string indices to get output channel info
    8. Parse EpochPerDACSection -> waveform definitions
    9. Parse EpochSection -> epoch metadata
    10. Compute datetime from header fields

    Parsing Note: This is a reverse-engineering effort. Official method uses
    Axon's ABFFIO.DLL. Some parts (e.g., Strings parsing) are brittle.

    Format: Section-based with table of contents
    Block Size: 512 bytes
    Header Fields: See headerDescriptionV2 for main header fields

    File Layout:
    +----------------------------------------------------------------------+
    | MAIN HEADER (Bytes 0-75)                                             |
    +----------------------------------------------------------------------+
    | Minimal header with file metadata and version info                  |
    | See headerDescriptionV2 for complete field list                     |
    +----------------------------------------------------------------------+
    | SECTION INDEX (Bytes 76-827) - THE KEY TO PARSING                    |
    +----------------------------------------------------------------------+
    | Table of contents: 18 sections x 16 bytes per entry                 |
    | Each entry: uBlockIndex, uBytes, llNumEntries                        |
    | Section names defined in sectionNames:                               |
    |   ProtocolSection, ADCSection, DACSection, EpochSection,             |
    |   StringsSection, DataSection, TagSection, SynchArraySection, etc.   |
    +----------------------------------------------------------------------+
    | PROTOCOL SECTION (at sections['ProtocolSection'] location)          |
    +----------------------------------------------------------------------+
    | Core acquisition settings (512 bytes): operation mode, sample rate, |
    | resolution, etc. See protocolInfoDescription for fields              |
    +----------------------------------------------------------------------+
    | STRINGS SECTION (at sections['StringsSection'] location)            |
    +----------------------------------------------------------------------+
    | CENTRAL STRING REPOSITORY: All strings (channel names, units,       |
    | protocol paths) stored here. Other sections store indices, not       |
    | strings directly. This is fundamental to ABF 2.x design.             |
    | Format: \x00\x00[str1]\x00[str2]\x00...                              |
    | Accessed via indices like lADCChannelNameIndex                       |
    +----------------------------------------------------------------------+
    | ADC SECTION (at sections['ADCSection'] location)                    |
    +----------------------------------------------------------------------+
    | Per-channel input configs: 128 bytes per entry                       |
    | Contains: gains, offsets, telegraph settings, string indices         |
    | See ADCInfoDescription for fields                                    |
    +----------------------------------------------------------------------+
    | DAC SECTION (at sections['DACSection'] location)                    |
    +----------------------------------------------------------------------+
    | Per-channel output configs: 256 bytes per entry                      |
    | Contains: holding levels, scale factors, waveform settings,          |
    | string indices. See DACInfoDescription for fields                    |
    +----------------------------------------------------------------------+
    | EPOCH PER DAC SECTION (at sections['EpochPerDACSection'] location)  |
    +----------------------------------------------------------------------+
    | Waveform epoch definitions: 48 bytes per entry                       |
    | Defines stimulus waveforms per DAC channel                           |
    | See EpochPerDACDescription for fields                                |
    +----------------------------------------------------------------------+
    | TAG SECTION (at sections['TagSection'] location)                    |
    +----------------------------------------------------------------------+
    | User-defined markers/comments during acquisition                     |
    | Each tag: 64 bytes. See TagInfoDescription for structure             |
    +----------------------------------------------------------------------+
    | SYNCH ARRAY SECTION (Physical disk layout)                          |
    +----------------------------------------------------------------------+
    | Purpose: Index of episodes/sweeps within the continuous data stream |
    | Critical for episodic acquisition modes (mode 2, 5)                 |
    |                                                                       |
    | Electrophysiology Context:                                           |
    |   - Episode/Sweep = Single experimental trial or stimulation cycle  |
    |   - In voltage-clamp: one voltage step protocol execution           |
    |   - In current-clamp: one current injection sequence                |
    |   - Data contains multiple episodes concatenated in data section    |
    |                                                                       |
    | Neo Mapping:                                                         |
    |   - Each episode becomes one neo.Segment                             |
    |   - offset = starting sample index in the continuous data array      |
    |   - len = number of samples in this episode (all channels combined) |
    |                                                                       |
    | Start Position:                                                      |
    |   Byte = sections['SynchArraySection']['uBlockIndex'] x 512         |
    |   Total entries = sections['SynchArraySection']['llNumEntries']     |
    |                  (number of episodes/segments)                       |
    |                                                                       |
    | Each Entry Structure (8 bytes):                                      |
    |   Offset  Size  Type   Field        Description                     |
    |   0       4     int32  offset       Episode start position (samples)|
    |   4       4     int32  len          Episode length (samples)        |
    |                                                                       |
    | Example: 3 episodes, 3 channels, 5000 samples per channel per ep:   |
    |                                                                       |
    |   Disk Offset  | Content                                             |
    |   -------------|--------------------------------------------------   |
    |   +0  bytes    | Episode 0 offset   (4 bytes, int32) = 0            |
    |   +4  bytes    | Episode 0 length   (4 bytes, int32) = 15000        |
    |   +8  bytes    | Episode 1 offset   (4 bytes, int32) = 15000        |
    |   +12 bytes    | Episode 1 length   (4 bytes, int32) = 15000        |
    |   +16 bytes    | Episode 2 offset   (4 bytes, int32) = 30000        |
    |   +20 bytes    | Episode 2 length   (4 bytes, int32) = 15000        |
    |                                                                       |
    | Note: len = samples_per_channel x num_channels (interleaved)        |
    |       num_channels from ADCSection llNumEntries                      |
    +----------------------------------------------------------------------+
    | DATA SECTION (Physical disk layout)                                 |
    +----------------------------------------------------------------------+
    | Start Position:                                                      |
    |   Byte = sections['DataSection']['uBlockIndex'] x 512               |
    |   Total samples = sections['DataSection']['llNumEntries']           |
    |   item_size = 2 bytes if nDataFormat=0 (int16)                      |
    |   item_size = 4 bytes if nDataFormat=1 (float32)                    |
    |                                                                       |
    | Example: 3 channels (Ch0, Ch1, Ch2), int16 format:                  |
    |                                                                       |
    |   Disk Offset  | Content                                             |
    |   -------------|--------------------------------------------------   |
    |   +0  bytes    | Sample 0, Channel 0  (2 bytes, int16)              |
    |   +2  bytes    | Sample 0, Channel 1  (2 bytes, int16)              |
    |   +4  bytes    | Sample 0, Channel 2  (2 bytes, int16)              |
    |   +6  bytes    | Sample 1, Channel 0  (2 bytes, int16)              |
    |   +8  bytes    | Sample 1, Channel 1  (2 bytes, int16)              |
    |   +10 bytes    | Sample 1, Channel 2  (2 bytes, int16)              |
    |   +12 bytes    | Sample 2, Channel 0  (2 bytes, int16)              |
    |   +14 bytes    | Sample 2, Channel 1  (2 bytes, int16)              |
    |   +16 bytes    | Sample 2, Channel 2  (2 bytes, int16)              |
    |   ... continues for llNumEntries total samples                      |
    |                                                                       |
    | Pattern: Channels are interleaved within each time point            |
    |   [Ch0,Ch1,Ch2] [Ch0,Ch1,Ch2] [Ch0,Ch1,Ch2] ...                     |
    +----------------------------------------------------------------------+

    Note: All section locations calculated as: uBlockIndex x 512

    Parameters
    ----------
    f : StructFile
        File object positioned after reading the signature
    header_description : list
        List of (key, offset, fmt) tuples describing header structure

    Returns
    -------
    dict
        Header dictionary with file metadata
    """
    # construct dict
    header = {}
    for key, offset, fmt in header_description:
        val = f.read_f(fmt, offset=offset)
        if len(val) == 1:
            header[key] = val[0]
        else:
            header[key] = np.array(val)

    # correction of version number and starttime
    n = header["fFileVersionNumber"]
    header["fFileVersionNumber"] = n[3] + 0.1 * n[2] + 0.01 * n[1] + 0.001 * n[0]
    header["lFileStartTime"] = header["uFileStartTimeMS"] * 0.001

    # sections
    sections = {}
    for s, sectionName in enumerate(sectionNames):
        uBlockIndex, uBytes, llNumEntries = f.read_f("IIl", offset=76 + s * 16)
        sections[sectionName] = {}
        sections[sectionName]["uBlockIndex"] = uBlockIndex
        sections[sectionName]["uBytes"] = uBytes
        sections[sectionName]["llNumEntries"] = llNumEntries
    header["sections"] = sections

    # strings sections
    # hack for reading channels names and units
    # this section is not very detailed and so the code
    # not very robust.
    f.seek(sections["StringsSection"]["uBlockIndex"] * BLOCKSIZE)
    big_string = f.read(sections["StringsSection"]["uBytes"])
    # this idea comes from pyABF https://github.com/swharden/pyABF
    # previously we searched for clampex, Clampex etc, but this was
    # brittle. pyABF believes that looking for the \x00\x00 is more
    # robust. We find these values, replace mu->u, then split into
    # a set of strings
    indexed_string = big_string[big_string.rfind(b"\x00\x00") :]
    # replace mu -> u for easy display
    indexed_string = indexed_string.replace(b"\xb5", b"\x75")
    # we need to remove one of the \x00 to have the indices be
    # the correct order
    indexed_string = indexed_string.split(b"\x00")[1:]
    strings = indexed_string

    # ADC sections
    header["listADCInfo"] = []
    for i in range(sections["ADCSection"]["llNumEntries"]):
        # read ADCInfo
        f.seek(sections["ADCSection"]["uBlockIndex"] * BLOCKSIZE + sections["ADCSection"]["uBytes"] * i)
        ADCInfo = {}
        for key, fmt in ADCInfoDescription:
            val = f.read_f(fmt)
            if len(val) == 1:
                ADCInfo[key] = val[0]
            else:
                ADCInfo[key] = np.array(val)
        ADCInfo["ADCChNames"] = strings[ADCInfo["lADCChannelNameIndex"]]
        ADCInfo["ADCChUnits"] = strings[ADCInfo["lADCUnitsIndex"]]
        header["listADCInfo"].append(ADCInfo)

    # protocol sections
    protocol = {}
    f.seek(sections["ProtocolSection"]["uBlockIndex"] * BLOCKSIZE)
    for key, fmt in protocolInfoDescription:
        val = f.read_f(fmt)
        if len(val) == 1:
            protocol[key] = val[0]
        else:
            protocol[key] = np.array(val)
    header["protocol"] = protocol
    header["sProtocolPath"] = strings[header["uProtocolPathIndex"]]

    # tags
    listTag = []
    for i in range(sections["TagSection"]["llNumEntries"]):
        f.seek(sections["TagSection"]["uBlockIndex"] * BLOCKSIZE + sections["TagSection"]["uBytes"] * i)
        tag = {}
        for key, fmt in TagInfoDescription:
            val = f.read_f(fmt)
            if len(val) == 1:
                tag[key] = val[0]
            else:
                tag[key] = np.array(val)
        listTag.append(tag)

    header["listTag"] = listTag

    # DAC sections
    header["listDACInfo"] = []
    for i in range(sections["DACSection"]["llNumEntries"]):
        # read DACInfo
        f.seek(sections["DACSection"]["uBlockIndex"] * BLOCKSIZE + sections["DACSection"]["uBytes"] * i)
        DACInfo = {}
        for key, fmt in DACInfoDescription:
            val = f.read_f(fmt)
            if len(val) == 1:
                DACInfo[key] = val[0]
            else:
                DACInfo[key] = np.array(val)
        DACInfo["DACChNames"] = strings[DACInfo["lDACChannelNameIndex"]]
        DACInfo["DACChUnits"] = strings[DACInfo["lDACChannelUnitsIndex"]]

        header["listDACInfo"].append(DACInfo)

    # EpochPerDAC  sections
    # header['dictEpochInfoPerDAC'] is dict of dicts:
    #  - the first index is the DAC number
    #  - the second index is the epoch number
    # It has to be done like that because data may not exist
    # and may not be in sorted order
    header["dictEpochInfoPerDAC"] = {}
    for i in range(sections["EpochPerDACSection"]["llNumEntries"]):
        #  read DACInfo
        f.seek(
            sections["EpochPerDACSection"]["uBlockIndex"] * BLOCKSIZE
            + sections["EpochPerDACSection"]["uBytes"] * i
        )
        EpochInfoPerDAC = {}
        for key, fmt in EpochInfoPerDACDescription:
            val = f.read_f(fmt)
            if len(val) == 1:
                EpochInfoPerDAC[key] = val[0]
            else:
                EpochInfoPerDAC[key] = np.array(val)

        DACNum = EpochInfoPerDAC["nDACNum"]
        EpochNum = EpochInfoPerDAC["nEpochNum"]
        # Checking if the key exists, if not, the value is empty
        # so we have to create empty dict to populate
        if DACNum not in header["dictEpochInfoPerDAC"]:
            header["dictEpochInfoPerDAC"][DACNum] = {}

        header["dictEpochInfoPerDAC"][DACNum][EpochNum] = EpochInfoPerDAC

    # Epoch sections
    header["EpochInfo"] = []
    for i in range(sections["EpochSection"]["llNumEntries"]):
        # read EpochInfo
        f.seek(sections["EpochSection"]["uBlockIndex"] * BLOCKSIZE + sections["EpochSection"]["uBytes"] * i)
        EpochInfo = {}
        for key, fmt in EpochInfoDescription:
            val = f.read_f(fmt)
            if len(val) == 1:
                EpochInfo[key] = val[0]
            else:
                EpochInfo[key] = np.array(val)
        header["EpochInfo"].append(EpochInfo)

    # date and time
    YY = int(header["uFileStartDate"] / 10000)
    MM = int((header["uFileStartDate"] - YY * 10000) / 100)
    DD = int(header["uFileStartDate"] - YY * 10000 - MM * 100)
    hh = int(header["uFileStartTimeMS"] / 1000.0 / 3600.0)
    mm = int((header["uFileStartTimeMS"] / 1000.0 - hh * 3600) / 60)
    ss = header["uFileStartTimeMS"] / 1000.0 - hh * 3600 - mm * 60
    ms = int(np.mod(ss, 1) * 1e6)
    ss = int(ss)
    header["rec_datetime"] = datetime.datetime(YY, MM, DD, hh, mm, ss, ms)

    return header


class StructFile(BufferedReader):
    def read_f(self, fmt, offset=None):
        if offset is not None:
            self.seek(offset)
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))


def clean_string(s):
    s = s.rstrip(b"\x00")
    s = s.rstrip(b" ")
    return s


def safe_decode_units(s):
    s = s.replace(b" ", b"")
    s = s.replace(b"\xb5", b"u")  # \xb5 is µ
    s = s.replace(b"\xb0", b"\xc2\xb0")  # \xb0 is °
    s = s.decode("utf-8")
    return s


BLOCKSIZE = 512

headerDescriptionV1 = [
    ("fFileSignature", 0, "4s"),
    ("fFileVersionNumber", 4, "f"),
    ("nOperationMode", 8, "h"),
    ("lActualAcqLength", 10, "i"),
    ("nNumPointsIgnored", 14, "h"),
    ("lActualEpisodes", 16, "i"),
    ("lFileStartTime", 24, "i"),
    ("lDataSectionPtr", 40, "i"),
    ("lTagSectionPtr", 44, "i"),
    ("lNumTagEntries", 48, "i"),
    ("lSynchArrayPtr", 92, "i"),
    ("lSynchArraySize", 96, "i"),
    ("nDataFormat", 100, "h"),
    ("nADCNumChannels", 120, "h"),
    ("fADCSampleInterval", 122, "f"),
    ("fSynchTimeUnit", 130, "f"),
    ("lNumSamplesPerEpisode", 138, "i"),
    ("lPreTriggerSamples", 142, "i"),
    ("lEpisodesPerRun", 146, "i"),
    ("fADCRange", 244, "f"),
    ("lADCResolution", 252, "i"),
    ("nFileStartMillisecs", 366, "h"),
    ("nADCPtoLChannelMap", 378, "16h"),
    ("nADCSamplingSeq", 410, "16h"),
    ("sADCChannelName", 442, "10s" * 16),
    ("sADCUnits", 602, "8s" * 16),
    ("fADCProgrammableGain", 730, "16f"),
    ("fInstrumentScaleFactor", 922, "16f"),
    ("fInstrumentOffset", 986, "16f"),
    ("fSignalGain", 1050, "16f"),
    ("fSignalOffset", 1114, "16f"),
    ("nDigitalEnable", 1436, "h"),
    ("nActiveDACChannel", 1440, "h"),
    ("nDigitalHolding", 1584, "h"),
    ("nDigitalInterEpisode", 1586, "h"),
    ("nDigitalValue", 2588, "10h"),
    ("lDACFilePtr", 2048, "2i"),
    ("lDACFileNumEpisodes", 2056, "2i"),
    ("fDACCalibrationFactor", 2074, "4f"),
    ("fDACCalibrationOffset", 2090, "4f"),
    ("nWaveformEnable", 2296, "2h"),
    ("nWaveformSource", 2300, "2h"),
    ("nInterEpisodeLevel", 2304, "2h"),
    ("nEpochType", 2308, "20h"),
    ("fEpochInitLevel", 2348, "20f"),
    ("fEpochLevelInc", 2428, "20f"),
    ("lEpochInitDuration", 2508, "20i"),
    ("lEpochDurationInc", 2588, "20i"),
    ("nTelegraphEnable", 4512, "16h"),
    ("fTelegraphAdditGain", 4576, "16f"),
    ("sProtocolPath", 4898, "384s"),
]

headerDescriptionV2 = [
    ("fFileSignature", 0, "4s"),
    ("fFileVersionNumber", 4, "4b"),
    ("uFileInfoSize", 8, "I"),
    ("lActualEpisodes", 12, "I"),
    ("uFileStartDate", 16, "I"),
    ("uFileStartTimeMS", 20, "I"),
    ("uStopwatchTime", 24, "I"),
    ("nFileType", 28, "H"),
    ("nDataFormat", 30, "H"),
    ("nSimultaneousScan", 32, "H"),
    ("nCRCEnable", 34, "H"),
    ("uFileCRC", 36, "I"),
    ("FileGUID", 40, "I"),
    ("uCreatorVersion", 56, "I"),
    ("uCreatorNameIndex", 60, "I"),
    ("uModifierVersion", 64, "I"),
    ("uModifierNameIndex", 68, "I"),
    ("uProtocolPathIndex", 72, "I"),
]

sectionNames = [
    "ProtocolSection",
    "ADCSection",
    "DACSection",
    "EpochSection",
    "ADCPerDACSection",
    "EpochPerDACSection",
    "UserListSection",
    "StatsRegionSection",
    "MathSection",
    "StringsSection",
    "DataSection",
    "TagSection",
    "ScopeSection",
    "DeltaSection",
    "VoiceTagSection",
    "SynchArraySection",
    "AnnotationSection",
    "StatsSection",
]

protocolInfoDescription = [
    ("nOperationMode", "h"),
    ("fADCSequenceInterval", "f"),
    ("bEnableFileCompression", "b"),
    ("sUnused1", "3s"),
    ("uFileCompressionRatio", "I"),
    ("fSynchTimeUnit", "f"),
    ("fSecondsPerRun", "f"),
    ("lNumSamplesPerEpisode", "i"),
    ("lPreTriggerSamples", "i"),
    ("lEpisodesPerRun", "i"),
    ("lRunsPerTrial", "i"),
    ("lNumberOfTrials", "i"),
    ("nAveragingMode", "h"),
    ("nUndoRunCount", "h"),
    ("nFirstEpisodeInRun", "h"),
    ("fTriggerThreshold", "f"),
    ("nTriggerSource", "h"),
    ("nTriggerAction", "h"),
    ("nTriggerPolarity", "h"),
    ("fScopeOutputInterval", "f"),
    ("fEpisodeStartToStart", "f"),
    ("fRunStartToStart", "f"),
    ("lAverageCount", "i"),
    ("fTrialStartToStart", "f"),
    ("nAutoTriggerStrategy", "h"),
    ("fFirstRunDelayS", "f"),
    ("nChannelStatsStrategy", "h"),
    ("lSamplesPerTrace", "i"),
    ("lStartDisplayNum", "i"),
    ("lFinishDisplayNum", "i"),
    ("nShowPNRawData", "h"),
    ("fStatisticsPeriod", "f"),
    ("lStatisticsMeasurements", "i"),
    ("nStatisticsSaveStrategy", "h"),
    ("fADCRange", "f"),
    ("fDACRange", "f"),
    ("lADCResolution", "i"),
    ("lDACResolution", "i"),
    ("nExperimentType", "h"),
    ("nManualInfoStrategy", "h"),
    ("nCommentsEnable", "h"),
    ("lFileCommentIndex", "i"),
    ("nAutoAnalyseEnable", "h"),
    ("nSignalType", "h"),
    ("nDigitalEnable", "h"),
    ("nActiveDACChannel", "h"),
    ("nDigitalHolding", "h"),
    ("nDigitalInterEpisode", "h"),
    ("nDigitalDACChannel", "h"),
    ("nDigitalTrainActiveLogic", "h"),
    ("nStatsEnable", "h"),
    ("nStatisticsClearStrategy", "h"),
    ("nLevelHysteresis", "h"),
    ("lTimeHysteresis", "i"),
    ("nAllowExternalTags", "h"),
    ("nAverageAlgorithm", "h"),
    ("fAverageWeighting", "f"),
    ("nUndoPromptStrategy", "h"),
    ("nTrialTriggerSource", "h"),
    ("nStatisticsDisplayStrategy", "h"),
    ("nExternalTagType", "h"),
    ("nScopeTriggerOut", "h"),
    ("nLTPType", "h"),
    ("nAlternateDACOutputState", "h"),
    ("nAlternateDigitalOutputState", "h"),
    ("fCellID", "3f"),
    ("nDigitizerADCs", "h"),
    ("nDigitizerDACs", "h"),
    ("nDigitizerTotalDigitalOuts", "h"),
    ("nDigitizerSynchDigitalOuts", "h"),
    ("nDigitizerType", "h"),
]

ADCInfoDescription = [
    ("nADCNum", "h"),
    ("nTelegraphEnable", "h"),
    ("nTelegraphInstrument", "h"),
    ("fTelegraphAdditGain", "f"),
    ("fTelegraphFilter", "f"),
    ("fTelegraphMembraneCap", "f"),
    ("nTelegraphMode", "h"),
    ("fTelegraphAccessResistance", "f"),
    ("nADCPtoLChannelMap", "h"),
    ("nADCSamplingSeq", "h"),
    ("fADCProgrammableGain", "f"),
    ("fADCDisplayAmplification", "f"),
    ("fADCDisplayOffset", "f"),
    ("fInstrumentScaleFactor", "f"),
    ("fInstrumentOffset", "f"),
    ("fSignalGain", "f"),
    ("fSignalOffset", "f"),
    ("fSignalLowpassFilter", "f"),
    ("fSignalHighpassFilter", "f"),
    ("nLowpassFilterType", "b"),
    ("nHighpassFilterType", "b"),
    ("fPostProcessLowpassFilter", "f"),
    ("nPostProcessLowpassFilterType", "c"),
    ("bEnabledDuringPN", "b"),
    ("nStatsChannelPolarity", "h"),
    ("lADCChannelNameIndex", "i"),
    ("lADCUnitsIndex", "i"),
]

TagInfoDescription = [
    ("lTagTime", "i"),
    ("sComment", "56s"),
    ("nTagType", "h"),
    ("nVoiceTagNumber_or_AnnotationIndex", "h"),
]

DACInfoDescription = [
    ("nDACNum", "h"),
    ("nTelegraphDACScaleFactorEnable", "h"),
    ("fInstrumentHoldingLevel", "f"),
    ("fDACScaleFactor", "f"),
    ("fDACHoldingLevel", "f"),
    ("fDACCalibrationFactor", "f"),
    ("fDACCalibrationOffset", "f"),
    ("lDACChannelNameIndex", "i"),
    ("lDACChannelUnitsIndex", "i"),
    ("lDACFilePtr", "i"),
    ("lDACFileNumEpisodes", "i"),
    ("nWaveformEnable", "h"),
    ("nWaveformSource", "h"),
    ("nInterEpisodeLevel", "h"),
    ("fDACFileScale", "f"),
    ("fDACFileOffset", "f"),
    ("lDACFileEpisodeNum", "i"),
    ("nDACFileADCNum", "h"),
    ("nConditEnable", "h"),
    ("lConditNumPulses", "i"),
    ("fBaselineDuration", "f"),
    ("fBaselineLevel", "f"),
    ("fStepDuration", "f"),
    ("fStepLevel", "f"),
    ("fPostTrainPeriod", "f"),
    ("fPostTrainLevel", "f"),
    ("nMembTestEnable", "h"),
    ("nLeakSubtractType", "h"),
    ("nPNPolarity", "h"),
    ("fPNHoldingLevel", "f"),
    ("nPNNumADCChannels", "h"),
    ("nPNPosition", "h"),
    ("nPNNumPulses", "h"),
    ("fPNSettlingTime", "f"),
    ("fPNInterpulse", "f"),
    ("nLTPUsageOfDAC", "h"),
    ("nLTPPresynapticPulses", "h"),
    ("lDACFilePathIndex", "i"),
    ("fMembTestPreSettlingTimeMS", "f"),
    ("fMembTestPostSettlingTimeMS", "f"),
    ("nLeakSubtractADCIndex", "h"),
    ("sUnused", "124s"),
]

EpochInfoPerDACDescription = [
    ("nEpochNum", "h"),
    ("nDACNum", "h"),
    ("nEpochType", "h"),
    ("fEpochInitLevel", "f"),
    ("fEpochLevelInc", "f"),
    ("lEpochInitDuration", "i"),
    ("lEpochDurationInc", "i"),
    ("lEpochPulsePeriod", "i"),
    ("lEpochPulseWidth", "i"),
    ("sUnused", "18s"),
]

EpochInfoDescription = [
    ("nEpochNum", "h"),
    ("nDigitalValue", "h"),
    ("nDigitalTrainValue", "h"),
    ("nAlternateDigitalValue", "h"),
    ("nAlternateDigitalTrainValue", "h"),
    ("bEpochCompression", "b"),
    ("sUnused", "21s"),
]
