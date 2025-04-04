"""
baserawio
======

Classes
-------

BaseRawIO
abstract class which should be overridden to write a RawIO.

RawIO is a low level API in neo that provides fast  access to the raw data.
When possible, all IOs should/implement this level following these guidelines:
  * internal use of memmap (or hdf5)
  * fast reading of the header (do not read the complete file)
  * neo tree object is symmetric and logical: same channel/units/event
    along all block and segments.

For this level, datasets of recordings are mapped as follows:

A channel refers to a physical channel of a recording in an experiment. It is identified by a
channel_id. Recordings from a channel consist of sections of samples which are recorded
contiguously in time; in other words, a section of a channel has a specific sampling_rate,
start_time, and length (and thus also stop_time, which is the time of the sample which would
lie one sampling interval beyond the last sample present in that section).

A stream consists of a set of channels which all have the same structure of their sections of
recording and the same data type of samples. Each stream has a unique stream_id and has a name,
which does not need to be unique. A stream thus has multiple channels which all have the same
sampling rate and are on the same clock, have the same sections with t_starts and lengths, and
the same data type for their samples. The samples in a stream can thus be retrieved as an Numpy
array, a chunk of samples.

Channels within a stream can be accessed be either their channel_id, which must be unique within
a stream, or by their channel_index, which is a 0 based index to all channels within the stream.
Note that a single channel of recording may be represented within multiple streams, and such is
the case for RawIOs which may have both unfiltered and filtered or downsampled versions of the
signals from a single recording channel. In such a case, a single channel and channel_id may be
represented by a different channel_index within different streams. Lists of channel_indexes are
often convenient to pass around a selection of channels within a stream.

At the neo.io level, one AnalogSignal with multiple channels can be created for each stream. Such
an AnalogSignal may have multiple Segments, with each segment containing the sections from each
channel with the same t_start and length. Such multiple Segments for a RawIO will have the
same sampling rate. It is thus possible to retrieve the t_start and length
of the sections of the channels for a Block and Segment of a stream.

So this handles **only** one simplified but very frequent case of dataset:
    * Only one channel set  for AnalogSignal stable along Segment
    * Only one channel set  for SpikeTrain stable along Segment
    * AnalogSignal have all the same sampling_rate across all Segment
    * t_start/t_stop are the same for many object (SpikeTrain, Event) inside a Segment

Signal channels are handled by group of "stream".
One stream will result at neo.io level in one AnalogSignal with multiple channels.

A helper class `neo.io.basefromrawio.BaseFromRaw` transforms a RawIO to
neo legacy IO. In short all "neo.rawio" classes are also "neo.io"
with lazy reading capability.

With this API the IO have an attributes `header` with necessary keys.
This  `header` attribute is done in `_parse_header(...)` method.
See ExampleRawIO as example.

BaseRawIO also implements a possible persistent cache system that can be used
by some RawIOs to avoid a very long parse_header() call. The idea is that some variable
or vector can be stored somewhere (near the file, /tmp, any path) for use across multiple
constructions of a RawIO for a given set of data.

"""

from __future__ import annotations

import logging
import numpy as np
import os
import sys

from neo import logging_handler

from .utils import get_memmap_chunk_from_opened_file


possible_raw_modes = [
    "one-file",
    "multi-file",
    "one-dir",
]  # 'multi-dir', 'url', 'other'

error_header = "Header is not read yet, do parse_header() first"

_signal_buffer_dtype = [
    ("name", "U64"),  # not necessarily unique
    ("id", "U64"),  # must be unique
]
# To be left an empty array if the concept of buffer is undefined for a reader.
_signal_stream_dtype = [
    ("name", "U64"),  # not necessarily unique
    ("id", "U64"),  # must be unique
    (
        "buffer_id",
        "U64",
    ),  # should be "" (empty string) when the stream is not nested under a buffer or the buffer is undefined for some reason.
]

_signal_channel_dtype = [
    ("name", "U64"),  # not necessarily unique
    ("id", "U64"),  # must be unique
    ("sampling_rate", "float64"),
    ("dtype", "U16"),
    ("units", "U64"),
    ("gain", "float64"),
    ("offset", "float64"),
    ("stream_id", "U64"),
    ("buffer_id", "U64"),
]

# TODO for later: add t_start and length in _signal_channel_dtype
# this would simplify all t_start/t_stop stuff for each RawIO class

_common_sig_characteristics = ["sampling_rate", "dtype", "stream_id"]

_spike_channel_dtype = [
    ("name", "U64"),
    ("id", "U64"),
    # for waveform
    ("wf_units", "U64"),
    ("wf_gain", "float64"),
    ("wf_offset", "float64"),
    ("wf_left_sweep", "int64"),
    ("wf_sampling_rate", "float64"),
]

# in rawio event and epoch are handled the same way
# except, that duration is `None` for events
_event_channel_dtype = [
    ("name", "U64"),
    ("id", "U64"),
    ("type", "S5"),  # epoch or event
]


class BaseRawIO:
    """
    Generic class to handle.

    """

    name = "BaseRawIO"
    description = ""
    extensions = []

    rawmode = None  # one key from possible_raw_modes

    #   TODO Why multi-file would have a single filename is confusing here - shouldn't
    #   the name of this argument be filenames_list or filenames_base or similar?
    #
    #   When rawmode=='one-file' kargs MUST contains 'filename' the filename
    #   When rawmode=='multi-file' kargs MUST contains 'filename' one of the filenames.
    #   When rawmode=='one-dir' kargs MUST contains 'dirname' the dirname.

    def __init__(self, use_cache: bool = False, cache_path: str = "same_as_resource", **kargs):
        """
        init docstring should be filled out at the rawio level so the user knows whether to
        input filename or dirname.

        """
        # create a logger for the IO class
        fullname = self.__class__.__module__ + "." + self.__class__.__name__
        self.logger = logging.getLogger(fullname)
        # Create a logger for 'neo' and add a handler to it if it doesn't have one already.
        # (it will also not add one if the root logger has a handler)
        corename = self.__class__.__module__.split(".")[0]
        corelogger = logging.getLogger(corename)
        rootlogger = logging.getLogger()
        if not corelogger.handlers and not rootlogger.handlers:
            corelogger.addHandler(logging_handler)

        self.use_cache = use_cache
        if use_cache:
            self.setup_cache(cache_path)
        else:
            self._cache = None

        self.header = None
        self.is_header_parsed = False

        self._has_buffer_description_api = False

    def has_buffer_description_api(self) -> bool:
        """
        Return if the reader handle the buffer API.
        If True then the reader support internally `get_analogsignal_buffer_description()`
        """
        return self._has_buffer_description_api

    def parse_header(self):
        """
        Parses the header of the file(s) to allow for faster computations
        for all other functions

        """
        # this must create
        # self.header['nb_block']
        # self.header['nb_segment']
        # self.header['signal_buffers']
        # self.header['signal_streams']
        # self.header['signal_channels']
        # self.header['spike_channels']
        # self.header['event_channels']

        self._parse_header()
        self._check_stream_signal_channel_characteristics()
        self.is_header_parsed = True

    def source_name(self):
        """Return fancy name of file source"""
        return self._source_name()

    def __repr__(self):
        txt = f"{self.__class__.__name__}: {self.source_name()}\n"
        if self.header is not None:
            nb_block = self.block_count()
            txt += f"nb_block: {nb_block}\n"
            nb_seg = [self.segment_count(i) for i in range(nb_block)]
            txt += f"nb_segment:  {nb_seg}\n"

            # signal streams
            v = [
                s["name"] + f" (chans: {self.signal_channels_count(i)})"
                for i, s in enumerate(self.header["signal_streams"])
            ]
            v = pprint_vector(v)
            txt += f"signal_streams: {v}\n"

            for k in ("signal_channels", "spike_channels", "event_channels"):
                ch = self.header[k]
                v = pprint_vector(self.header[k]["name"])
                txt += f"{k}: {v}\n"

        return txt

    def _repr_html_(self):
        """
        HTML representation for the raw recording base.

        Returns
        -------
        html : str
            The HTML representation as a string.
        """
        html = []
        html.append('<div style="font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto;">')

        # Header
        html.append(f'<h3 style="color: #2c3e50;">{self.__class__.__name__}: {self.source_name()}</h3>')

        if self.is_header_parsed:
            # Basic info
            nb_block = self.block_count()
            html.append(f"<p><strong>nb_block:</strong> {nb_block}</p>")
            nb_seg = [self.segment_count(i) for i in range(nb_block)]
            html.append(f"<p><strong>nb_segment:</strong> {nb_seg}</p>")

            # CSS for tables - using only black, white, and gray colors
            html.append(
                """
            <style>
                #{unique_id} table.neo-table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                    font-size: 14px;
                    color: inherit;
                    background-color: transparent;
                }}
                #{unique_id} table.neo-table th,
                #{unique_id} table.neo-table td {{
                    border: 1px solid #888;
                    padding: 8px;
                    text-align: left;
                }}
                #{unique_id} table.neo-table th {{
                    background-color: rgba(128,128,128,0.2);
                }}
                #{unique_id} table.neo-table tr:nth-child(even) {{
                    background-color: rgba(128,128,128,0.1);
                }}
                #{unique_id} details {{
                    margin-bottom: 15px;
                    border: 1px solid rgba(128,128,128,0.3);
                    border-radius: 4px;
                    overflow: hidden;
                    background-color: transparent;
                }}
                #{unique_id} summary {{
                    padding: 10px;
                    background-color: rgba(128,128,128,0.2);
                    cursor: pointer;
                    font-weight: bold;
                    color: inherit;
                }}
                #{unique_id} details[open] summary {{
                    border-bottom: 1px solid rgba(128,128,128,0.3);
                }}
                #{unique_id} .table-container {{
                    padding: 10px;
                    overflow-x: auto;
                    background-color: transparent;
                }}
            </style>
            """
            )

            # Signal Streams
            signal_streams = self.header["signal_streams"]
            if signal_streams.size > 0:
                html.append("<details>")
                html.append("<summary>Signal Streams</summary>")
                html.append('<div class="table-container">')
                html.append('<table class="neo-table">')
                html.append("<thead><tr><th>Name</th><th>ID</th><th>Buffer ID</th><th>Channel Count</th></tr></thead>")
                html.append("<tbody>")

                for i, stream in enumerate(signal_streams):
                    html.append("<tr>")
                    html.append(f'<td>{stream["name"]}</td>')
                    html.append(f'<td>{stream["id"]}</td>')
                    html.append(f'<td>{stream["buffer_id"]}</td>')
                    html.append(f"<td>{self.signal_channels_count(i)}</td>")
                    html.append("</tr>")

                html.append("</tbody></table>")
                html.append("</div>")
                html.append("</details>")

            # Signal Channels
            signal_channels = self.header["signal_channels"]
            if signal_channels.size > 0:
                html.append("<details>")
                html.append("<summary>Signal Channels</summary>")
                html.append('<div class="table-container">')
                html.append('<table class="neo-table">')
                html.append(
                    "<thead><tr><th>Name</th><th>ID</th><th>Sampling Rate</th><th>Data Type</th><th>Units</th><th>Gain</th><th>Offset</th><th>Stream ID</th><th>Buffer ID</th></tr></thead>"
                )
                html.append("<tbody>")

                for channel in signal_channels:
                    html.append("<tr>")
                    html.append(f'<td>{channel["name"]}</td>')
                    html.append(f'<td>{channel["id"]}</td>')
                    html.append(f'<td>{channel["sampling_rate"]}</td>')
                    html.append(f'<td>{channel["dtype"]}</td>')
                    html.append(f'<td>{channel["units"]}</td>')
                    html.append(f'<td>{channel["gain"]}</td>')
                    html.append(f'<td>{channel["offset"]}</td>')
                    html.append(f'<td>{channel["stream_id"]}</td>')
                    html.append(f'<td>{channel["buffer_id"]}</td>')
                    html.append("</tr>")

                html.append("</tbody></table>")
                html.append("</div>")
                html.append("</details>")

            # Spike Channels
            spike_channels = self.header["spike_channels"]
            if spike_channels.size > 0:
                html.append("<details>")
                html.append("<summary>Spike Channels</summary>")
                html.append('<div class="table-container">')
                html.append('<table class="neo-table">')
                html.append(
                    "<thead><tr><th>Name</th><th>ID</th><th>WF Units</th><th>WF Gain</th><th>WF Offset</th><th>WF Left Sweep</th><th>WF Sampling Rate</th></tr></thead>"
                )
                html.append("<tbody>")

                for channel in spike_channels:
                    html.append("<tr>")
                    html.append(f'<td>{channel["name"]}</td>')
                    html.append(f'<td>{channel["id"]}</td>')
                    html.append(f'<td>{channel["wf_units"]}</td>')
                    html.append(f'<td>{channel["wf_gain"]}</td>')
                    html.append(f'<td>{channel["wf_offset"]}</td>')
                    html.append(f'<td>{channel["wf_left_sweep"]}</td>')
                    html.append(f'<td>{channel["wf_sampling_rate"]}</td>')
                    html.append("</tr>")

                html.append("</tbody></table>")
                html.append("</div>")
                html.append("</details>")

            # Event Channels
            event_channels = self.header["event_channels"]
            if event_channels.size > 0:
                html.append("<details>")
                html.append("<summary>Event Channels</summary>")
                html.append('<div class="table-container">')
                html.append('<table class="neo-table">')
                html.append("<thead><tr><th>Name</th><th>ID</th><th>Type</th></tr></thead>")
                html.append("<tbody>")

                for channel in event_channels:
                    html.append("<tr>")
                    html.append(f'<td>{channel["name"]}</td>')
                    html.append(f'<td>{channel["id"]}</td>')
                    html.append(
                        f'<td>{channel["type"].decode("utf-8") if isinstance(channel["type"], bytes) else channel["type"]}</td>'
                    )
                    html.append("</tr>")

                html.append("</tbody></table>")
                html.append("</div>")
                html.append("</details>")
        else:
            html.append("<p><em>Call <code>parse_header()</code> to load the reader data.</p>")

        html.append("</div>")
        return "\n".join(html)

    def _generate_minimal_annotations(self):
        """
        Helper function that generates a nested dict for annotations.

        Must be called when these are Ok after self.header is done
        and thus when these functions return the correct values:
          * block_count()
          * segment_count()
          * signal_streams_count()
          * signal_channels_count()
          * spike_channels_count()
          * event_channels_count()

        There are several sources and kinds of annotations that will
        be forwarded to the neo.io level and used to enrich neo objects:
            * annotations of objects common across segments
                * signal_streams > neo.AnalogSignal annotations
                * signal_channels > neo.AnalogSignal array_annotations split by stream
                * spike_channels > neo.SpikeTrain
                * event_channels > neo.Event and neo.Epoch
            * annotations that depend of the block_id/segment_id of the object:
              * nested in raw_annotations['blocks'][block_index]['segments'][seg_index]['signals']

        Usage after a call to this function we can do this to populate more annotations:

        raw_annotations['blocks'][block_index][ 'nickname'] = 'super block'
        raw_annotations['blocks'][block_index]
                        ['segments']['important_key'] = 'important value'
        raw_annotations['blocks'][block_index]
                        ['segments'][seg_index]
                        ['signals']['nickname'] = 'super signals stream'
        raw_annotations['blocks'][block_index]
                        ['segments'][seg_index]
                        ['signals']['__array_annotations__']
                        ['channels_quality'] = ['bad', 'good', 'medium', 'good']
        raw_annotations['blocks'][block_index]
                        ['segments'][seg_index]
                        ['spikes'][spike_chan]['nickname'] =  'super neuron'
        raw_annotations['blocks'][block_index]
                        ['segments'][seg_index]
                        ['spikes'][spike_chan]
                        ['__array_annotations__']['spike_amplitudes'] = [-1.2, -10., ...]
        raw_annotations['blocks'][block_index]
                        ['segments'][seg_index]
                        ['events'][ev_chan]['nickname'] = 'super trigger'
        raw_annotations['blocks'][block_index]
                        ['segments'][seg_index]
                        ['events'][ev_chan]
                        Z['__array_annotations__']['additional_label'] = ['A', 'B', 'A', 'C', ...]


        Theses annotations will be used at the neo.io API directly in objects.

        Standard annotation like name/id/file_origin are already generated here.
        """
        signal_streams = self.header["signal_streams"]
        signal_channels = self.header["signal_channels"]
        spike_channels = self.header["spike_channels"]
        event_channels = self.header["event_channels"]

        # use for AnalogSignal.annotations and AnalogSignal.array_annotations
        signal_stream_annotations = []
        for c in range(signal_streams.size):
            stream_id = signal_streams[c]["id"]
            channels = signal_channels[signal_channels["stream_id"] == stream_id]
            d = {}
            d["name"] = signal_streams["name"][c]
            d["stream_id"] = stream_id
            d["file_origin"] = self._source_name()
            d["__array_annotations__"] = {}
            for key in ("name", "id"):
                values = np.array([channels[key][chan] for chan in range(channels.size)])
                d["__array_annotations__"]["channel_" + key + "s"] = values
            signal_stream_annotations.append(d)

        # used for SpikeTrain.annotations and SpikeTrain.array_annotations
        spike_annotations = []
        for c in range(spike_channels.size):
            # use for Unit.annotations
            d = {}
            d["name"] = spike_channels["name"][c]
            d["id"] = spike_channels["id"][c]
            d["file_origin"] = self._source_name()
            d["__array_annotations__"] = {}
            spike_annotations.append(d)

        # used for Event/Epoch.annotations and Event/Epoch.array_annotations
        event_annotations = []
        for c in range(event_channels.size):
            # not used in neo.io at the moment could useful one day
            d = {}
            d["name"] = event_channels["name"][c]
            d["id"] = event_channels["id"][c]
            d["file_origin"] = self._source_name()
            d["__array_annotations__"] = {}
            event_annotations.append(d)

        # duplicate this signal_stream_annotations/spike_annotations/event_annotations
        # across blocks and segments and create annotations
        ann = {}
        ann["blocks"] = []
        for block_index in range(self.block_count()):
            d = {}
            d["file_origin"] = self.source_name()
            d["segments"] = []
            ann["blocks"].append(d)

            for seg_index in range(self.segment_count(block_index)):
                d = {}
                d["file_origin"] = self.source_name()
                # copy nested
                d["signals"] = signal_stream_annotations.copy()
                d["spikes"] = spike_annotations.copy()
                d["events"] = event_annotations.copy()
                ann["blocks"][block_index]["segments"].append(d)

        self.raw_annotations = ann

    def _repr_annotations(self):
        txt = "Raw annotations\n"
        for block_index in range(self.block_count()):
            bl_a = self.raw_annotations["blocks"][block_index]
            txt += f"*Block {block_index}\n"
            for k, v in bl_a.items():
                if k in ("segments",):
                    continue
                txt += f"  -{k}: {v}\n"
            for seg_index in range(self.segment_count(block_index)):
                seg_a = bl_a["segments"][seg_index]
                txt += f"  *Segment {seg_index}\n"
                for k, v in seg_a.items():
                    if k in (
                        "signals",
                        "spikes",
                        "events",
                    ):
                        continue
                    txt += f"    -{k}: {v}\n"

                # annotations by channels for spikes/events/epochs
                for child in (
                    "signals",
                    "events",
                    "spikes",
                ):
                    if child == "signals":
                        n = self.header["signal_streams"].shape[0]
                    else:
                        n = self.header[child[:-1] + "_channels"].shape[0]
                    for c in range(n):
                        neo_name = {"signals": "AnalogSignal", "spikes": "SpikeTrain", "events": "Event/Epoch"}[child]
                        txt += f"    *{neo_name} {c}\n"
                        child_a = seg_a[child][c]
                        for k, v in child_a.items():
                            if k == "__array_annotations__":
                                continue
                            txt += f"      -{k}: {v}\n"
                        for k, values in child_a["__array_annotations__"].items():
                            values = ", ".join([str(v) for v in values[:4]])
                            values = "[ " + values + " ..."
                            txt += f"      -{k}: {values}\n"

        return txt

    def print_annotations(self):
        """Print formatted raw_annotations"""
        print(self._repr_annotations())

    def block_count(self):
        """Returns the number of blocks"""
        return self.header["nb_block"]

    def segment_count(self, block_index: int):
        """
        Returns count of segments for a given block

        Parameters
        ----------
        block_index: int
            The index of the block to do the segment count for
        Returns
        -------
        count: int
            The number of segments for a given block

        """
        return self.header["nb_segment"][block_index]

    def signal_streams_count(self):
        """Return the number of signal streams.
        Same for all Blocks and Segments.
        """
        return len(self.header["signal_streams"])

    def signal_channels_count(self, stream_index: int):
        """Returns the number of signal channels for a given stream.
        This number is the same for all Blocks and Segments.

        Parameters
        ----------
        stream_index: int
            the stream index in which to count the signal channels

        Returns
        -------
        count: int
            the number of signal channels of a given stream
        """
        stream_id = self.header["signal_streams"][stream_index]["id"]
        channels = self.header["signal_channels"]
        channels = channels[channels["stream_id"] == stream_id]
        return len(channels)

    def spike_channels_count(self):
        """Return the number of unit (aka spike) channels.
        Same for all Blocks and Segments.
        """
        return len(self.header["spike_channels"])

    def event_channels_count(self):
        """Return the number of event/epoch channels.
        Same for all Blocks and Segments.
        """
        return len(self.header["event_channels"])

    def segment_t_start(self, block_index: int, seg_index: int):
        """
        Global t_start of a Segment Shared by all objects except
        for AnalogSignal.

        Parameters
        ----------
        block_index: int
            The index of the block to find the segment t_start
        seg_index: int
            The index of the segment within the block_index in which to find the t_start

        Returns
        -------
        t_start: float
            the time of global t_start of a segment within a block

        """
        return self._segment_t_start(block_index, seg_index)

    def segment_t_stop(self, block_index, seg_index):
        """
        Global t_stop of a Segment in s. Shared by all objects except
        for AnalogSignal.

        Parameters
        ----------
        block_index: int
            The index of the block to find the segment t_start
        seg_index: int
            The index of the segment within the block_index in which to find the t_start

        Returns
        -------
        t_stop: float
            the time of global t_stop of a segment within a block

        """
        return self._segment_t_stop(block_index, seg_index)

    ###
    # signal and channel zone

    def _check_stream_signal_channel_characteristics(self):
        """
        Check that all channels that belonging to the same stream_id
        have the same stream id and _common_sig_characteristics. These
        presently includes:
          * sampling_rate
          * units
          * dtype
        """
        signal_streams = self.header["signal_streams"]
        signal_channels = self.header["signal_channels"]
        if signal_streams.size > 0:
            if signal_channels.size < 1:
                raise ValueError("Signal stream exists but there are no signal channels")

        for stream_index in range(signal_streams.size):
            stream_id = signal_streams[stream_index]["id"]
            mask = signal_channels["stream_id"] == stream_id
            characteristics = signal_channels[mask][_common_sig_characteristics]
            unique_characteristics = np.unique(characteristics)
            if unique_characteristics.size != 1:
                raise ValueError(
                    f"Some channels in stream_id {stream_id} "
                    f"do not have the same {_common_sig_characteristics} {unique_characteristics}"
                )

            # also check that channel_id is unique inside a stream
            channel_ids = signal_channels[mask]["id"]
            if np.unique(channel_ids).size != channel_ids.size:
                raise ValueError(f"signal_channels do not have unique ids for stream {stream_index}")

        self._several_channel_groups = signal_streams.size > 1

    def channel_name_to_index(self, stream_index: int, channel_names: list[str]):
        """
        Inside a stream, transform channel_names to channel_indexes.
        Based on self.header['signal_channels']
        channel_indexes are zero-based offsets within the stream

        Parameters
        ----------
        stream_index: int
            The stream in which to convert channel_names to their respective channel_indexes
        channel_names: list[str]
            The channel names to convert to channel_indexes

        Returns
        -------
        channel_indexes: np.array[int]
            the channel_indexes associated with the given channel_ids

        """
        stream_id = self.header["signal_streams"][stream_index]["id"]
        mask = self.header["signal_channels"]["stream_id"] == stream_id
        signal_channels = self.header["signal_channels"][mask]
        chan_names = list(signal_channels["name"])
        if signal_channels.size != np.unique(chan_names).size:
            raise ValueError("Channel names are not unique")
        channel_indexes = np.array([chan_names.index(name) for name in channel_names])
        return channel_indexes

    def channel_id_to_index(self, stream_index: int, channel_ids: list[str]):
        """
        Inside a stream, transform channel_ids to channel_indexes.
        Based on self.header['signal_channels']
        channel_indexes are zero-based offsets within the stream

        Parameters
        ----------
        stream_index: int
            the stream index in which to convert the channel_ids to channel_indexes
        channel_ids: list[str]
            the list of channel_ids to convert to channel_indexes

        Returns
        -------
        channel_indexes: np.array[int]
             the channel_indexes associated with the given channel_ids
        """
        # unique ids is already checked in _check_stream_signal_channel_characteristics
        stream_id = self.header["signal_streams"][stream_index]["id"]
        mask = self.header["signal_channels"]["stream_id"] == stream_id
        signal_channels = self.header["signal_channels"][mask]
        chan_ids = list(signal_channels["id"])
        channel_indexes = np.array([chan_ids.index(chan_id) for chan_id in channel_ids])
        return channel_indexes

    def _get_channel_indexes(
        self,
        stream_index: int,
        channel_indexes: list[int] | None,
        channel_names: list[str] | None,
        channel_ids: list[str] | None,
    ):
        """
        Select channel_indexes for a stream based on channel_indexes/channel_names/channel_ids
        depending on which one is not None.

        Parameters
        ----------
        stream_index: int,
            the stream index in which to get channel_indexes
        channel_indexes: list[int] | None
            the channel_indexes desired
        channel_names: list[str] | None
            the names of the channels to be converted to channel_indexes. Give this or channel_ids for conversion
        channel_ids: list[str] | None
            the ids of the channels to be converted to channel_indexes. Give this or channel_names for conversion

        Returns
        -------
        channel_indexes: np.array[int]
            The desired channel_indexes for functions requiring channel_indexes

        """
        if channel_indexes is None and channel_names is not None:
            channel_indexes = self.channel_name_to_index(stream_index, channel_names)
        elif channel_indexes is None and channel_ids is not None:
            channel_indexes = self.channel_id_to_index(stream_index, channel_ids)
        return channel_indexes

    def _get_stream_index_from_arg(self, stream_index_arg: int | None):
        """
        Verifies the desired stream_index exists

        Parameters
        ----------
        stream_index_arg: int | None, default: None
            The stream_index to verify
            If None checks if only one stream exists and then returns 0 if it is single stream

        Returns
        -------
        stream_index: int
            The stream_index to be used for function requiring a stream_index

        """
        if stream_index_arg is None:
            if self.header["signal_streams"].size != 1:
                raise ValueError("stream_index must be given for files with multiple streams")
            stream_index = 0
        else:
            if stream_index_arg < 0 or stream_index_arg >= self.header["signal_streams"].size:
                raise ValueError(f"stream_index must be between 0 and {self.header['signal_streams'].size}")
            stream_index = stream_index_arg
        return stream_index

    def get_signal_size(self, block_index: int, seg_index: int, stream_index: int | None = None):
        """
        Retrieves the length of a single section of the channels in a stream.

        Parameters
        ----------
        block_index: int
            The desired block in which to get a signal size
        seg_index: int
            The desired segment of the block in which to get the signal size
        stream_index: int | None, default: None
            The optional stream index in which to determine signal size
            This is required for data with multiple streams

        Returns
        -------
        signal_size: int
            The number of samples for a given signal within the desired block, segment, and stream

        """
        stream_index = self._get_stream_index_from_arg(stream_index)

        return self._get_signal_size(block_index, seg_index, stream_index)

    def get_signal_t_start(self, block_index: int, seg_index: int, stream_index: int | None = None):
        """
        Retrieves the t_start of a single section of the channels in a stream.

        Parameters
        ----------
        block_index: int
            The desired block in which to get a t_start
        seg_index: int
            The desired segment of the block in which to get the t_start
        stream_index: int | None, default: None
            The optional stream index in which to determine t_start
            This is required for data with multiple streams

        Returns
        -------
        signal_t_start: float
            The start time for a given signal within the desired block, segment, and stream

        """
        stream_index = self._get_stream_index_from_arg(stream_index)
        return self._get_signal_t_start(block_index, seg_index, stream_index)

    def get_signal_sampling_rate(self, stream_index: int | None = None):
        """
        Retrieves the sampling rate for a stream and all channels withinin that stream.

        Parameters
        ----------
        stream_index: int | None, default: None
            The desired stream index in which to get the sampling_rate
            This is required for data with multiple streams

        Returns
        -------
        sr: float
            The sampling rate of a given stream and all channels in that stream

        """
        stream_index = self._get_stream_index_from_arg(stream_index)
        stream_id = self.header["signal_streams"][stream_index]["id"]
        mask = self.header["signal_channels"]["stream_id"] == stream_id
        signal_channels = self.header["signal_channels"][mask]
        sr = signal_channels[0]["sampling_rate"]
        return float(sr)

    def get_analogsignal_chunk(
        self,
        block_index: int = 0,
        seg_index: int = 0,
        i_start: int | None = None,
        i_stop: int | None = None,
        stream_index: int | None = None,
        channel_indexes: list[int] | None = None,
        channel_names: list[str] | None = None,
        channel_ids: list[str] | None = None,
        prefer_slice: bool = False,
    ):
        """
        Returns a chunk of raw signal as a Numpy array.

        Parameters
        ----------
        block_index: int, default: 0
            The block with the desired analog signal
        seg_index: int, default: 0
            The segment containing the desired analog signal
        i_start: int | None, default: None
            The index of the first sample (not time) of the desired analog signal
        i_stop: int | None, default: None
            The index of one past the last sample (not time) of the desired analog signal
        stream_index: int | None, default: None
            The index of the stream containing the channels to assess for the analog signal
            This is required for data with multiple streams
        channel_indexes: list[int] | np.array[int]|  slice | None, default: None
            The list of indexes of channels to retrieve
            One of channel_indexes, channel_names, or channel_ids must be given
        channel_names: list[str] | None, default: None
            The list of channel names to retrieve
            One of channel_indexes, channel_names, or channel_ids must be given
        channel_ids: list[str] | None, default: None
            The list of channel_ids to retrieve
            One of channel_indexes, channel_names, or channel_ids must be given

        Returns
        -------
        raw_chunk: np.array (n_samples, n_channels)
            The array with the raw signal samples

        Notes
        -----
        Rows are the samples and columns are the channels
        The channels are chosen either by channel_names,
        if provided, otherwise by channel_ids, if provided, otherwise by channel_indexes, if
        provided, otherwise all channels are selected.

        Examples
        --------
        # tetrode with 1 sec recording at sampling_rate = 1000. Hz
        >>> rawio_reader.parse_header()
        >>> raw_sigs = rawio_reader.get_analogsignal_chunk(block_index=2, seg_index=0, stream_index=0)
        >>> raw_sigs.shape
        (1000,4) # 1000 samples by 4 channels
        >>> raw_sigs.dtype
        'int16' # returns the dtype from the recording itself

        # If we only want one electrode
        >>> raw_sigs_one_electrode = rawio_reader.get_analogsignal_chunk(block_index=2, seg_index=0, stream_index=0, channel_indexes=[0])
        >>> raw_sigs_one_electrode.shape
        (1000,1)

        """

        signal_streams = self.header["signal_streams"]
        signal_channels = self.header["signal_channels"]
        no_signal_streams = signal_streams.size == 0
        no_channels = signal_channels.size == 0
        if no_signal_streams or no_channels:
            error_message = (
                "get_analogsignal_chunk can't be called on a file with no signal streams or channels."
                "Double check that your file contains signal streams and channels."
            )
            raise AttributeError(error_message)

        stream_index = self._get_stream_index_from_arg(stream_index)
        channel_indexes = self._get_channel_indexes(stream_index, channel_indexes, channel_names, channel_ids)

        # some check on channel_indexes
        if isinstance(channel_indexes, list):
            channel_indexes = np.asarray(channel_indexes)

        if isinstance(channel_indexes, np.ndarray):
            if channel_indexes.dtype == "bool":
                if self.signal_channels_count(stream_index) != channel_indexes.size:
                    raise ValueError(
                        "If channel_indexes is a boolean it must have be the same length as the "
                        f"number of channels {self.signal_channels_count(stream_index)}"
                    )
                (channel_indexes,) = np.nonzero(channel_indexes)

        if prefer_slice and isinstance(channel_indexes, np.ndarray):
            # Check if channel_indexes are contiguous and transform to slice argument if possible.
            # This is useful for memmap or hdf5 where providing a slice causes a lazy read,
            # rather than a list of indexes that make a copy (like numpy.take()).
            if np.all(np.diff(channel_indexes) == 1):
                channel_indexes = slice(channel_indexes[0], channel_indexes[-1] + 1)

        raw_chunk = self._get_analogsignal_chunk(block_index, seg_index, i_start, i_stop, stream_index, channel_indexes)

        return raw_chunk

    def rescale_signal_raw_to_float(
        self,
        raw_signal: np.ndarray,
        dtype: np.dtype = "float32",
        stream_index: int | None = None,
        channel_indexes: list[int] | None = None,
        channel_names: list[str] | None = None,
        channel_ids: list[str] | None = None,
    ):
        """
        Rescales a chunk of raw signals which are provided as a Numpy array. These are normally
        returned by a call to get_analogsignal_chunk.

        Parameters
        ----------
        raw_signal: np.array (n_samples, n_channels)
            The numpy array of samples with columns being samples for a single channel
        dtype: np.dype, default: "float32"
            The datatype for returning scaled samples, must be acceptable by the numpy dtype constructor
        stream_index: int | None, default: None
            The index of the stream containing the channels to assess
        channel_indexes: list[int], np.array[int], slice | None, default: None
            The list of indexes of channels to retrieve
        channel_names: list[str] | None, default: None
            The list of channel names to retrieve
        channel_ids: list[str] | None, default: None
            list of channel_ids to retrieve

        Returns
        -------
        float_signal: np.array (n_samples, n_channels)
            The rescaled signal

        Notes
        -----
        The channels are specified either by channel_names, if provided, otherwise by channel_ids,
        if provided, otherwise by channel_indexes, if provided, otherwise all channels are selected.

        These are rawio dependent because rescaling of the NumPy array requires the offset and gain
        stored within the header of the rawio


        Examples
        --------
        # Once we have a `raw_sigs` using rawio.get_analogsignal_chunk() we can convert to voltages with a desired dtype
        # If we used `stream_index=0` with `get_analogsignal_chunk` we use `stream_index=0` here
        >>> float_sigs = rawio_reader.rescale_signal_raw_to_float(raw_signal=raw_sigs, dtype='float32', stream_index=0)
        >>> float_sigs.dtype
        'float32'
        >>> float_sigs.shape
        (1000,4)
        >>> float_sigs.shape == raw_sigs.shape
        True


        """
        stream_index = self._get_stream_index_from_arg(stream_index)
        channel_indexes = self._get_channel_indexes(stream_index, channel_indexes, channel_names, channel_ids)
        if channel_indexes is None:
            channel_indexes = slice(None)

        stream_id = self.header["signal_streams"][stream_index]["id"]
        mask = self.header["signal_channels"]["stream_id"] == stream_id
        channels = self.header["signal_channels"][mask]
        if channel_indexes is None:
            channel_indexes = slice(None)
        channels = channels[channel_indexes]

        float_signal = raw_signal.astype(dtype)

        if np.any(channels["gain"] != 1.0):
            float_signal *= channels["gain"]

        if np.any(channels["offset"] != 0.0):
            float_signal += channels["offset"]

        return float_signal

    # spiketrain and unit zone
    def spike_count(self, block_index: int = 0, seg_index: int = 0, spike_channel_index: int = 0):
        """
        Returns the spike count for a given block, segment, and spike_channel_index

        Parameters
        ----------
        block_index: int, default: 0
            The block with the desired segment to assess
        seg_index: int, default: 0
            The segment containing the desired section to assess
        spike_channel_index: int, default: 0
            The spike_channel_index for assessing spike_count

        Returns
        -------
        spike_count: int
            The number of spikes in the block and segment

        """
        return self._spike_count(block_index, seg_index, spike_channel_index)

    def get_spike_timestamps(
        self,
        block_index: int = 0,
        seg_index: int = 0,
        spike_channel_index: int = 0,
        t_start: float | None = None,
        t_stop: float | None = None,
    ):
        """
        Returns the spike_timestamps in samples (see note for dtype)

        Parameters
        ----------
        block_index: int, default: 0
            The block containing the section to get the spike timestamps
        seg_index: int, default: 0
            The segment containing the section to get the spike timestamps
        spike_channel_index: int, default: 0
            The channel in which to collect spike timestamps
        t_start: float | None, default: None
            The time in seconds for the start of the section to get spike timestamps
            None indicates to start at the beginning of the segment
        t_stop: float | None, default: None
            The time in seconds for the end of the section to get spike timestamps
            None indicates to end at the end of the segment

        Returns
        -------
        timestamp: np.array
            The spike timestamps

        Notes
        -----
        The timestamp datatype is as close to the format itself. Sometimes float/int32/int64.
        Sometimes it is the index on the signal but not always.
        The conversion to second or index_on_signal is done outside this method.


        Examples
        --------
        # to look at block 1, segment 0, and channel 3 on a tetrode from 10
        # seconds to 30 seconds we would do:
        >>> timestamps = rawio_reader.get_spike_timestamps(block_index=1,
                                                           seg_index=0,
                                                           spike_channel_index=3,
                                                           t_start=10,
                                                           t_stop=30)
        """
        timestamp = self._get_spike_timestamps(block_index, seg_index, spike_channel_index, t_start, t_stop)
        return timestamp

    def rescale_spike_timestamp(self, spike_timestamps: np.ndarray, dtype: np.dtype = "float64"):
        """
        Rescale spike timestamps from samples to seconds.

        Parameters
        ----------
        spike_timestamps: np.ndarray
            The array containing the spike_timestamps to convert
        dtype: np.dtype, default: "float64"
            The dtype in which to convert the spike time in seconds. Must be accepted by the numpy.dtype constructor

        Returns
        -------
        scaled_spike_timestamps: np.array
            The spiketimes in seconds

        Examples
        --------
        # After running `get_spike_timestamps` and returning timestamps we can do the following:
        >>> scaled_spike_timestamps = rawio_reader.rescale_spike_timestamps(spike_timestamps=timestamps,
                                                                            dtype='float64')

        """
        return self._rescale_spike_timestamp(spike_timestamps, dtype)

    # spiketrain waveform zone
    def get_spike_raw_waveforms(
        self,
        block_index: int = 0,
        seg_index: int = 0,
        spike_channel_index: int = 0,
        t_start: float | None = None,
        t_stop: float | None = None,
    ):
        """
        Gets the waveforms for one channel within one segment of one block

        Parameters
        ----------
        block_index: int, default: 0
            The block containing the desired set of waveform data
        seg_index: int, default: 0
            The segment containing the desired set of waveform data
        spike_channel_index: int, default: 0
            The channel index on which to get waveform data
        t_start: float | None, default: None
            The time in seconds for the start of the section to get waveforms
            None indicates to start at the beginning of the segment
        t_stop: float | None, default: None
            The time in seconds for the end of the section to waveforms
            None indicates to end at the end of the segment

        Returns
        -------
        wf: np.ndarray (nb_spike, nb_channel, nb_sample))
            A NumPy array of spikes, channels and samples
        """
        wf = self._get_spike_raw_waveforms(block_index, seg_index, spike_channel_index, t_start, t_stop)
        return wf

    def rescale_waveforms_to_float(
        self, raw_waveforms: np.ndarray, dtype: np.dtype = "float32", spike_channel_index: int = 0
    ):
        """
        Rescale waveforms to based on the rawio's waveform gain and waveform offset

        Parameters
        ----------
        raw_waveforms: np.ndarray
            The array containing the spike_timestamps to convert
        dtype: np.dtype, default: "float64"
            The dtype in which to convert the spike time to. Must be accepted by the numpy.dtype constructor
        spike_channel_index: int, default: 0
            The channel index of the desired channel to  rescale

        Returns
        -------
        float_waveforms: np.ndarray (nb_spikes, nb_channels, nb_samples)
            The scaled waveforms to the dtype specified by dtype
        """
        wf_gain = self.header["spike_channels"]["wf_gain"][spike_channel_index]
        wf_offset = self.header["spike_channels"]["wf_offset"][spike_channel_index]

        float_waveforms = raw_waveforms.astype(dtype)

        if wf_gain != 1.0:
            float_waveforms *= wf_gain
        if wf_offset != 0.0:
            float_waveforms += wf_offset

        return float_waveforms

    # event and epoch zone
    def event_count(self, block_index: int = 0, seg_index: int = 0, event_channel_index: int = 0):
        """
        Returns the count of events for a particular block, segment, and channel_index

        Parameters
        ----------
        block_index: int, default: 0
            The block in which to count the events
        seg_index: int, default: 0
            The segment within the block given by block_index in which to count events
        event_channel_index: int, default: 0
            The index of the channel in which to count events

        Returns
        -------
        n_events: int
            The number of events in the given block, segment, and event_channel_index
        """
        return self._event_count(block_index, seg_index, event_channel_index)

    def get_event_timestamps(
        self,
        block_index: int = 0,
        seg_index: int = 0,
        event_channel_index: int = 0,
        t_start: float | None = None,
        t_stop: float | None = None,
    ):
        """
        Returns the event timestamps along with their labels and durations

        Parameters
        ----------
        block_index: int, default: 0
            The block in which to count the events
        seg_index: int, default: 0
            The segment within the block given by block_index in which to count events
        event_channel_index: int, default: 0
            The index of the channel in which to count events
        t_start: float | None, default: None
            The time in seconds for the start of the section to get waveforms
            None indicates to start at the beginning of the segment
        t_stop: float | None, default: None
            The time in seconds for the end of the section to waveforms
            None indicates to end at the end of the segment

        Returns
        -------
        timestamp: np.array
            The timestamps of events (in samples)
        durations: np.array
            The durations of each event
        labels: np.array
            The labels of the events

        Notes
        -----
        The timestamp datatype is as close to the format itself. Sometimes float/int32/int64.
        Sometimes it is the index on the signal but not always.
        The conversion to second or index_on_signal is done outside this method.

        Examples
        --------
        # A given rawio reader that generates events data. For this example we will
        # look at Block 0, Segment 1, on Channel 1, with a start time at the beginning
        # of the segment and an end time of 5 minutes (300 s)
        >>> event_timestamps, durations, labels = rawio_reader.get_event_timestamps(block_index=0,
                                                                                    seg_index=1,
                                                                                    event_channel_index=1,
                                                                                    t_start=None,
                                                                                    t_stop=300)

        """
        timestamp, durations, labels = self._get_event_timestamps(
            block_index, seg_index, event_channel_index, t_start, t_stop
        )
        return timestamp, durations, labels

    def rescale_event_timestamp(
        self, event_timestamps: np.ndarray, dtype: np.dtype = "float64", event_channel_index: int = 0
    ):
        """
        Rescale event timestamps to seconds.

        Parameters
        ----------
        event_timestamps: np.ndarray
            The array containing the event timestamps to convert
        dtype: np.dtype, default: "float64"
            The dtype in which to convert the event time in seconds. Must be accepted by the numpy.dtype constructor
        event_channel_index: int, default: 0
            The channel index for scaling the events

        Returns
        -------
        scaled_event_timestamps: np.array
            The scaled event timestamps in seconds

        Examples
        --------
        # Using the event_timestamps from the `get_event_timestamps` function we can then scale from samples into
        # seconds using this `rescale_event_timestamp`. We use the same event_channel_index as used during the
        # `get_event_timestamps`
        >>> event_timestamps_seconds = rawio_reader.rescale_event_timestamp(event_timestamps=event_timestamps,
                                                                            dtype='float64',
                                                                            event_channel_index=1)

        """
        return self._rescale_event_timestamp(event_timestamps, dtype, event_channel_index)

    def rescale_epoch_duration(
        self, raw_duration: np.ndarray, dtype: np.dtype = "float64", event_channel_index: int = 0
    ):
        """
        Rescales the epoch duration from samples to seconds

        Parameters
        ----------
        raw_duration: np.ndarray
            The array containing the epoch times in samples
        dtype: np.dtype, default: "float64"
            The dtype in which to convert the spike time in seconds. Must be accepted by the numpy.dtype constructor
        event_channel_index: int, default: 0
            The channel on which to index for scaling epochs

        Returns
        -------
        scaled_epoch_durations: np.array
            The scaled epoch durations in seconds

        Examples
        --------
        # In this example we use the durations obtained from running `get_event_timestamps`
        >>> duration_seconds = rawio_reader.rescale_epoch_duration(raw_durations=durations,
                                                                   dtype='float64',
                                                                   event_channel_index=0)
        """
        return self._rescale_epoch_duration(raw_duration, dtype, event_channel_index)

    def setup_cache(self, cache_path: "home" | "same_as_resource", **init_kargs):
        try:
            import joblib
        except ImportError:
            raise ImportError("Using the RawIO cache needs joblib to be installed")

        if self.rawmode in ("one-file", "multi-file"):
            resource_name = self.filename
        elif self.rawmode == "one-dir":
            resource_name = self.dirname
        else:
            raise (NotImplementedError)

        if cache_path == "home":
            if sys.platform.startswith("win"):
                dirname = os.path.join(os.environ["APPDATA"], "neo_rawio_cache")
            elif sys.platform.startswith("darwin"):
                dirname = "~/Library/Application Support/neo_rawio_cache"
            else:
                dirname = os.path.expanduser("~/.config/neo_rawio_cache")
            dirname = os.path.join(dirname, self.__class__.__name__)

            if not os.path.exists(dirname):
                os.makedirs(dirname)
        elif cache_path == "same_as_resource":
            dirname = os.path.dirname(resource_name)
        else:
            if not os.path.exists(cache_path):
                raise ValueError("cache_path does not exist use 'home' or 'same_as_resource' to make this auto")

        # the hash of the resource (dir of file) is done with filename+datetime
        # TODO make something more sophisticated when rawmode='one-dir' that use all
        #  filename and datetime
        d = dict(ressource_name=resource_name, mtime=os.path.getmtime(resource_name))
        hash = joblib.hash(d, hash_name="md5")

        # name is constructed from the resource_name and the hash
        name = f"{os.path.basename(resource_name)}_{hash}"
        self.cache_filename = os.path.join(dirname, name)

        if os.path.exists(self.cache_filename):
            self.logger.warning(f"Use existing cache file {self.cache_filename}")
            self._cache = joblib.load(self.cache_filename)
        else:
            self.logger.warning(f"Create cache file {self.cache_filename}")
            self._cache = {}
            self.dump_cache()

    def add_in_cache(self, **kargs):
        if not self.use_cache:
            raise ValueError("Can not use add_in_cache if not using cache")
        self._cache.update(kargs)
        self.dump_cache()

    def dump_cache(self):
        if not self.use_cache:
            raise ValueError("Can not use dump_cache if not using cache")
        joblib.dump(self._cache, self.cache_filename)

    ##################

    # Functions to be implemented in IO below here

    def _parse_header(self):
        raise (NotImplementedError)
        # must call
        # self._generate_empty_annotations()

    def _source_name(self):
        raise (NotImplementedError)

    def _segment_t_start(self, block_index: int, seg_index: int):
        raise (NotImplementedError)

    def _segment_t_stop(self, block_index: int, seg_index: int):
        raise (NotImplementedError)

    ###
    # signal and channel zone
    def _get_signal_size(self, block_index: int, seg_index: int, stream_index: int):
        """
        Return the size of a set of AnalogSignals indexed by channel_indexes.

        All channels indexed must have the same size and t_start.
        """
        raise (NotImplementedError)

    def _get_signal_t_start(self, block_index: int, seg_index: int, stream_index: int):
        """
        Return the t_start of a set of AnalogSignals indexed by channel_indexes.

        All channels indexed must have the same size and t_start.
        """
        raise (NotImplementedError)

    def _get_analogsignal_chunk(
        self,
        block_index: int,
        seg_index: int,
        i_start: int | None,
        i_stop: int | None,
        stream_index: int,
        channel_indexes: list[int] | None,
    ):
        """
        Return the samples from a set of AnalogSignals indexed
        by stream_index and channel_indexes (local index inner stream).

        RETURNS
        -------
            array of samples, with each requested channel in a column
        """
        raise (NotImplementedError)

    ###
    # spiketrain and unit zone
    def _spike_count(self, block_index: int, seg_index: int, spike_channel_index: int):
        raise (NotImplementedError)

    def _get_spike_timestamps(
        self, block_index: int, seg_index: int, spike_channel_index: int, t_start: float | None, t_stop: float | None
    ):
        raise (NotImplementedError)

    def _rescale_spike_timestamp(self, spike_timestamps: np.ndarray, dtype: np.dtype):
        raise (NotImplementedError)

    ###
    # spike waveforms zone
    def _get_spike_raw_waveforms(
        self, block_index: int, seg_index: int, spike_channel_index: int, t_start: float | None, t_stop: float | None
    ):
        raise (NotImplementedError)

    ###
    # event and epoch zone
    def _event_count(self, block_index: int, seg_index: int, event_channel_index: int):
        raise (NotImplementedError)

    def _get_event_timestamps(
        self, block_index: int, seg_index: int, event_channel_index: int, t_start: float | None, t_stop: float | None
    ):
        raise (NotImplementedError)

    def _rescale_event_timestamp(self, event_timestamps: np.ndarray, dtype: np.dtype):
        raise (NotImplementedError)

    def _rescale_epoch_duration(self, raw_duration: np.ndarray, dtype: np.dtype):
        raise (NotImplementedError)

    ###
    # buffer api zone
    # must be implemented if has_buffer_description_api=True
    def get_analogsignal_buffer_description(self, block_index: int = 0, seg_index: int = 0, buffer_id: str = None):
        if not self.has_buffer_description_api:
            raise ValueError("This reader do not support buffer_description API")
        descr = self._get_analogsignal_buffer_description(block_index, seg_index, buffer_id)
        return descr

    def _get_analogsignal_buffer_description(self, block_index, seg_index, buffer_id):
        raise (NotImplementedError)


class BaseRawWithBufferApiIO(BaseRawIO):
    """
    Generic class for reader that support "buffer api".

    In short reader that are internally based on:

      * np.memmap
      * hdf5

    In theses cases _get_signal_size and _get_analogsignal_chunk are totaly generic and do not need to be implemented in the class.

    For this class sub classes must implements theses two dict:
       * self._buffer_descriptions[block_index][seg_index] = buffer_description
       * self._stream_buffer_slice[buffer_id] = None or slicer o indices

    """

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self._has_buffer_description_api = True

    def _get_signal_size(self, block_index, seg_index, stream_index):
        buffer_id = self.header["signal_streams"][stream_index]["buffer_id"]
        buffer_desc = self.get_analogsignal_buffer_description(block_index, seg_index, buffer_id)
        # some hdf5 revert teh buffer
        time_axis = buffer_desc.get("time_axis", 0)
        return buffer_desc["shape"][time_axis]

    def _get_analogsignal_chunk(
        self,
        block_index: int,
        seg_index: int,
        i_start: int | None,
        i_stop: int | None,
        stream_index: int,
        channel_indexes: list[int] | None,
    ):

        stream_id = self.header["signal_streams"][stream_index]["id"]
        buffer_id = self.header["signal_streams"][stream_index]["buffer_id"]

        buffer_slice = self._stream_buffer_slice[stream_id]

        buffer_desc = self.get_analogsignal_buffer_description(block_index, seg_index, buffer_id)

        i_start = i_start or 0
        i_stop = i_stop or buffer_desc["shape"][0]

        if buffer_desc["type"] == "raw":

            # open files on demand and keep reference to opened file
            if not hasattr(self, "_memmap_analogsignal_buffers"):
                self._memmap_analogsignal_buffers = {}
            if block_index not in self._memmap_analogsignal_buffers:
                self._memmap_analogsignal_buffers[block_index] = {}
            if seg_index not in self._memmap_analogsignal_buffers[block_index]:
                self._memmap_analogsignal_buffers[block_index][seg_index] = {}
            if buffer_id not in self._memmap_analogsignal_buffers[block_index][seg_index]:
                fid = open(buffer_desc["file_path"], mode="rb")
                self._memmap_analogsignal_buffers[block_index][seg_index][buffer_id] = fid
            else:
                fid = self._memmap_analogsignal_buffers[block_index][seg_index][buffer_id]

            num_channels = buffer_desc["shape"][1]

            raw_sigs = get_memmap_chunk_from_opened_file(
                fid,
                num_channels,
                i_start,
                i_stop,
                np.dtype(buffer_desc["dtype"]),
                file_offset=buffer_desc["file_offset"],
            )

        elif buffer_desc["type"] == "hdf5":

            # open files on demand and keep reference to opened file
            if not hasattr(self, "_hdf5_analogsignal_buffers"):
                self._hdf5_analogsignal_buffers = {}
            if block_index not in self._hdf5_analogsignal_buffers:
                self._hdf5_analogsignal_buffers[block_index] = {}
            if seg_index not in self._hdf5_analogsignal_buffers[block_index]:
                self._hdf5_analogsignal_buffers[block_index][seg_index] = {}
            if buffer_id not in self._hdf5_analogsignal_buffers[block_index][seg_index]:
                import h5py

                h5file = h5py.File(buffer_desc["file_path"], mode="r")
                self._hdf5_analogsignal_buffers[block_index][seg_index][buffer_id] = h5file
            else:
                h5file = self._hdf5_analogsignal_buffers[block_index][seg_index][buffer_id]

            hdf5_path = buffer_desc["hdf5_path"]
            full_raw_sigs = h5file[hdf5_path]

            time_axis = buffer_desc.get("time_axis", 0)
            if time_axis == 0:
                raw_sigs = full_raw_sigs[i_start:i_stop, :]
            elif time_axis == 1:
                raw_sigs = full_raw_sigs[:, i_start:i_stop].T
            else:
                raise RuntimeError("Should never happen")

            if buffer_slice is not None:
                raw_sigs = raw_sigs[:, buffer_slice]

        else:
            raise NotImplementedError()

        # this is a pre slicing when the stream do not contain all channels (for instance spikeglx when load_sync_channel=False)
        if buffer_slice is not None:
            raw_sigs = raw_sigs[:, buffer_slice]

        # channel slice requested
        if channel_indexes is not None:
            raw_sigs = raw_sigs[:, channel_indexes]

        return raw_sigs

    def __del__(self):
        if hasattr(self, "_memmap_analogsignal_buffers"):
            for block_index in self._memmap_analogsignal_buffers.keys():
                for seg_index in self._memmap_analogsignal_buffers[block_index].keys():
                    for buffer_id, fid in self._memmap_analogsignal_buffers[block_index][seg_index].items():
                        fid.close()
            del self._memmap_analogsignal_buffers

        if hasattr(self, "_hdf5_analogsignal_buffers"):
            for block_index in self._hdf5_analogsignal_buffers.keys():
                for seg_index in self._hdf5_analogsignal_buffers[block_index].keys():
                    for buffer_id, h5_file in self._hdf5_analogsignal_buffers[block_index][seg_index].items():
                        h5_file.close()
            del self._hdf5_analogsignal_buffers


def pprint_vector(vector, lim: int = 8):
    vector = np.asarray(vector)
    if vector.ndim != 1:
        raise ValueError(f"`vector` must have a dimension of 1 and not {vector.ndim}")
    if len(vector) > lim:
        part1 = ", ".join(e for e in vector[: lim // 2])
        part2 = " , ".join(e for e in vector[-lim // 2 :])
        txt = f"[{part1} ... {part2}]"
    else:
        part1 = ", ".join(e for e in vector)
        txt = f"[{part1}]"
    return txt
