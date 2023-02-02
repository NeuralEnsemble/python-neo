"""
AxographRawIO
=============

RawIO class for reading AxoGraph files (.axgd, .axgx)

Original author: Jeffrey Gill

Documentation of the AxoGraph file format provided by the developer is
incomplete and in some cases incorrect. The primary sources of official
documentation are found in two out-of-date documents:

    - AxoGraph X User Manual, provided with AxoGraph and also available online:
        https://axograph.com/documentation/AxoGraph%20User%20Manual.pdf

    - AxoGraph_ReadWrite.h, a header file that is part of a C++ program
      provided with AxoGraph, and which is also available here:
        https://github.com/CWRUChielLab/axographio/blob/master/
            axographio/include/axograph_readwrite/AxoGraph_ReadWrite.h

These were helpful starting points for building this RawIO, especially for
reading the beginnings of AxoGraph files, but much of the rest of the file
format was deciphered by reverse engineering and guess work. Some portions of
the file format remain undeciphered.

The AxoGraph file format is versatile in that it can represent both time series
data collected during data acquisition and non-time series data generated
through manual or automated analysis, such as power spectrum analysis. This
implementation of an AxoGraph file format reader makes no effort to decoding
non-time series data. For simplicity, it makes several assumptions that should
be valid for any file generated directly from episodic or continuous data
acquisition without significant post-acquisition modification by the user.

Detailed logging is provided during header parsing for debugging purposes:
    >>> import logging
    >>> r = AxographRawIO(filename)
    >>> r.logger.setLevel(logging.DEBUG)
    >>> r.parse_header()

Background and Terminology
--------------------------

Acquisition modes:
    AxoGraph can operate in two main data acquisition modes:
    - Episodic "protocol-driven" acquisition mode, in which the program records
      from specified signal channels for a fixed duration each time a trigger
      is detected. Each trigger-activated recording is called an "episode".
      From files acquired in this mode, AxographRawIO creates multiple Neo
      Segments, one for each episode, unless force_single_segment=True.
    - Continuous "chart recorder" acquisition mode, in which it creates a
      continuous recording that can be paused and continued by the user
      whenever they like. From files acquired in this mode, AxographRawIO
      creates a single Neo Segment.

"Episode": analogous to a Neo Segment
    See descriptions of acquisition modes above and of groups below.

"Column": analogous to a Quantity array
    A column is a 1-dimensional array of data, stored in any one of a number of
    data types (e.g., scaled ints or floats). In the oldest version of the
    AxoGraph file format, even time was stored as a 1-dimensional array. In
    newer versions, time is stored as a special type of "column" that is really
    just a starting time and a sampling period.

    Column data appears in series in the file, i.e., all of the first column's
    data appears before the second column's. As an aside, because of this
    design choice AxoGraph cannot write data to disk as it is collected but
    must store it all in memory until data acquisition ends. This also affected
    how file slicing was implemented for this RawIO: Instead of using a single
    memmap to address into a 2-dimensional block of data, AxographRawIO
    constructs multiple 1-dimensional memmaps, one for each column, each with
    its own offset.

    Each column's data array is preceded by a header containing the column
    title, which normally contains the units (e.g., "Current (nA)"). Data
    recorded in episodic acquisition mode will contain a repeating sequence of
    column names, where each repetition corresponds to an episode (e.g.,
    "Time", "Column A", "Column B", "Column A", "Column B", etc.).

    AxoGraph offers a spreadsheet view for viewing all column data.

"Trace": analogous to a single-channel Neo AnalogSignal
    A trace is a 2-dimensional series. Raw data is not stored in the part of
    the file concerned with traces. Instead, in the header for each trace are
    indexes pointing to two data columns, defined earlier in the file,
    corresponding to the trace's x and y data. These indexes can be changed in
    AxoGraph under the "Assign X and Y Columns" tool, though doing so may
    violate assumptions made by AxographRawIO.

    For time series data collected under the usual data acquisition modes that
    has not been modified after collection by the user, the x-index always
    points to the time column; one trace exists for each non-time column, with
    the y-index pointing to that column.

    Traces are analogous to AnalogSignals in Neo. However, for simplicity of
    implementation, AxographRawIO does not actually check the pairing of
    columns in the trace headers. Instead it assumes the default pairing
    described above when it creates signal channels while scanning through
    columns. Older versions of the AxoGraph file format lack trace headers
    entirely, so this is the most general solution.

    Trace headers contain additional information about the series, such as plot
    style, which is parsed by AxographRawIO and made available in
    self.info['trace_header_info_list'] but is otherwise unused.

"Group": analogous to a Neo ChannelIndex for matching channels across Segments
    A group is a collection of one or more traces. Like traces, raw data is not
    stored in the part of the file concerned with groups. Instead, each trace
    header contains an index pointing to the group it is assigned to. Group
    assignment of traces can be changed in AxoGraph under the "Group Traces"
    tool, or by using the "Merge Traces" or "Separate Traces" commands, though
    doing so may violate assumptions made by AxographRawIO.

    Files created in episodic acquisition mode contain multiple traces per
    group, one for each episode. In that mode, a group corresponds to a signal
    channel and is analogous to a ChannelIndex in Neo; the traces within the
    group represent the time series recorded for that channel across episodes
    and are analogous to AnalogSignals from multiple Segments in Neo.

    In contrast, files created in continuous acquisition mode contain one trace
    per group, each corresponding to a signal channel. In that mode, groups and
    traces are basically conceptually synonymous, though the former can still
    be thought of as analogous to ChannelIndexes in Neo for a single-Segment.

    Group headers are only consulted by AxographRawIO to determine if is safe
    to interpret a file as episodic and therefore translatable to multiple
    Segments in Neo. Certain criteria have to be met, such as all groups
    containing equal numbers of traces and each group having homogeneous signal
    parameters. If trace grouping was modified by the user after data
    acquisition, this may result in the file being interpreted as
    non-episodic. Older versions of the AxoGraph file format lack group headers
    entirely, so these files are never deemed safe to interpret as episodic,
    even if the column names follow a repeating sequence as described above.

"Tag" / "Event marker": analogous to a Neo Event
    In continuous acquisition mode, the user can press a hot key to tag a
    moment in time with a short label. Additionally, if the user stops or
    restarts data acquisition in this mode, a tag is created automatically with
    the label "Stop" or "Start", respectively. These are displayed by AxoGraph
    as event markers. AxographRawIO will organize all event markers into a
    single Neo Event channel with the name "AxoGraph Tags".

    In episodic acquisition mode, the tag hot key behaves differently. The
    current episode number is recorded in a user-editable notes section of the
    file, made available by AxographRawIO in self.info['notes']. Because these
    do not correspond to moments in time, they are not processed into Neo
    Events.

"Interval bar": analogous to a Neo Epoch
    After data acquisition, the user can annotate an AxoGraph file with
    horizontal, labeled bars called interval bars that span a specified period
    of time. These are not episode specific. AxographRawIO will organize all
    interval bars into a single Neo Epoch channel with the name "AxoGraph
    Intervals".
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype)

import os
from datetime import datetime
from io import open, BufferedReader
from struct import unpack, calcsize

import numpy as np


class AxographRawIO(BaseRawIO):
    """
    RawIO class for reading AxoGraph files (.axgd, .axgx)

    Args:
        filename (string):
            File name of the AxoGraph file to read.
        force_single_segment (bool):
            Episodic files are normally read as multi-Segment Neo objects. This
            parameter can force AxographRawIO to put all signals into a single
            Segment. Default: False.

    Example:
        >>> import neo
        >>> r = neo.rawio.AxographRawIO(filename=filename)
        >>> r.parse_header()
        >>> print(r)

        >>> # get signals
        >>> raw_chunk = r.get_analogsignal_chunk(
        ...     block_index=0, seg_index=0,
        ...     i_start=0, i_stop=1024,
        ...     channel_names=channel_names)
        >>> float_chunk = r.rescale_signal_raw_to_float(
        ...     raw_chunk,
        ...     dtype='float64',
        ...     channel_names=channel_names)
        >>> print(float_chunk)

        >>> # get event markers
        >>> ev_raw_times, _, ev_labels = r.get_event_timestamps(
        ...     event_channel_index=0)
        >>> ev_times = r.rescale_event_timestamp(
        ...     ev_raw_times, dtype='float64')
        >>> print([ev for ev in zip(ev_times, ev_labels)])

        >>> # get interval bars
        >>> ep_raw_times, ep_raw_durations, ep_labels = r.get_event_timestamps(
        ...     event_channel_index=1)
        >>> ep_times = r.rescale_event_timestamp(
        ...     ep_raw_times, dtype='float64')
        >>> ep_durations = r.rescale_epoch_duration(
        ...     ep_raw_durations, dtype='float64')
        >>> print([ep for ep in zip(ep_times, ep_durations, ep_labels)])

        >>> # get notes
        >>> print(r.info['notes'])

        >>> # get other miscellaneous info
        >>> print(r.info)
    """
    name = 'AxographRawIO'
    description = 'This IO reads .axgd/.axgx files created with AxoGraph'
    extensions = ['axgd', 'axgx', '']
    rawmode = 'one-file'

    def __init__(self, filename, force_single_segment=False):
        BaseRawIO.__init__(self)
        self.filename = filename
        self.force_single_segment = force_single_segment

    def _parse_header(self):

        self.header = {}

        self._scan_axograph_file()

        if not self.force_single_segment and self._safe_to_treat_as_episodic():
            self.logger.debug('Will treat as episodic')
            self._convert_to_multi_segment()
        else:
            self.logger.debug('Will not treat as episodic')
        self.logger.debug('')

        self._generate_minimal_annotations()
        blk_annotations = self.raw_annotations['blocks'][0]
        blk_annotations['format_ver'] = self.info['format_ver']
        blk_annotations['comment'] = self.info['comment'] if 'comment' in self.info else None
        blk_annotations['notes'] = self.info['notes'] if 'notes' in self.info else None
        blk_annotations['rec_datetime'] = self._get_rec_datetime()

        # modified time is not ideal but less prone to
        # cross-platform issues than created time (ctime)
        blk_annotations['file_datetime'] = datetime.fromtimestamp(
            os.path.getmtime(self.filename))

    def _source_name(self):
        return self.filename

    def _segment_t_start(self, block_index, seg_index):
        # same for all segments
        return self._t_start

    def _segment_t_stop(self, block_index, seg_index):
        # same for all signals in all segments
        t_stop = self._t_start + \
            len(self._raw_signals[seg_index][0]) * self._sampling_period
        return t_stop

    ###
    # signal and channel zone

    def _get_signal_size(self, block_index, seg_index, stream_index):
        # same for all signals in all segments
        return len(self._raw_signals[seg_index][0])

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        # same for all signals in all segments
        return self._t_start

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):

        if channel_indexes is None or \
           np.all(channel_indexes == slice(None, None, None)):
            channel_indexes = range(self.signal_channels_count(stream_index))

        raw_signals = [self._raw_signals
                       [seg_index]
                       [channel_index]
                       [slice(i_start, i_stop)]
                       for channel_index in channel_indexes]
        raw_signals = np.array(raw_signals).T  # loads data into memory

        return raw_signals

    ###
    # spiketrain and unit zone

    def _spike_count(self, block_index, seg_index, unit_index):
        # not supported
        return None

    def _get_spike_timestamps(self, block_index, seg_index, unit_index,
                              t_start, t_stop):
        # not supported
        return None

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        # not supported
        return None

    ###
    # spike waveforms zone

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index,
                                 t_start, t_stop):
        # not supported
        return None

    ###
    # event and epoch zone

    def _event_count(self, block_index, seg_index, event_channel_index):

        # Retrieve size of either event or epoch channel:
        #   event_channel_index: 0 AxoGraph Tags, 1 AxoGraph Intervals
        # AxoGraph tags can only be inserted in continuous data acquisition
        # mode. When the tag hot key is pressed in episodic acquisition mode,
        # the notes are updated with the current episode number instead of an
        # instantaneous event marker being created. This means that Neo-like
        # Events cannot be generated by AxoGraph for multi-Segment (episodic)
        # files. Furthermore, Neo-like Epochs (interval markers) are not
        # episode specific. For these reasons, this function ignores seg_index.

        return self._raw_event_epoch_timestamps[event_channel_index].size

    def _get_event_timestamps(self, block_index, seg_index,
                              event_channel_index, t_start, t_stop):

        # Retrieve either event or epoch data, unscaled:
        #   event_channel_index: 0 AxoGraph Tags, 1 AxoGraph Intervals
        # AxoGraph tags can only be inserted in continuous data acquisition
        # mode. When the tag hot key is pressed in episodic acquisition mode,
        # the notes are updated with the current episode number instead of an
        # instantaneous event marker being created. This means that Neo-like
        # Events cannot be generated by AxoGraph for multi-Segment (episodic)
        # files. Furthermore, Neo-like Epochs (interval markers) are not
        # episode specific. For these reasons, this function ignores seg_index.

        timestamps = self._raw_event_epoch_timestamps[event_channel_index]
        durations = self._raw_event_epoch_durations[event_channel_index]
        labels = self._event_epoch_labels[event_channel_index]

        if durations is None:
            # events
            if t_start is not None:
                # keep if event occurs after t_start ...
                keep = timestamps >= int(t_start / self._sampling_period)
                timestamps = timestamps[keep]
                labels = labels[keep]

            if t_stop is not None:
                # ... and before t_stop
                keep = timestamps <= int(t_stop / self._sampling_period)
                timestamps = timestamps[keep]
                labels = labels[keep]
        else:
            # epochs
            if t_start is not None:
                # keep if epoch ends after t_start ...
                keep = timestamps + durations >= \
                    int(t_start / self._sampling_period)
                timestamps = timestamps[keep]
                durations = durations[keep]
                labels = labels[keep]

            if t_stop is not None:
                # ... and starts before t_stop
                keep = timestamps <= int(t_stop / self._sampling_period)
                timestamps = timestamps[keep]
                durations = durations[keep]
                labels = labels[keep]

        return timestamps, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        # Scale either event or epoch start times from sample index to seconds
        # (t_start shouldn't be added)
        event_times = event_timestamps.astype(dtype) * self._sampling_period
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype, event_channel_index):
        # Scale epoch durations from samples to seconds
        epoch_durations = raw_duration.astype(dtype) * self._sampling_period
        return epoch_durations

    ###
    # multi-segment zone

    def _safe_to_treat_as_episodic(self):
        """
        The purpose of this function is to determine if the file contains any
        irregularities in its grouping of traces such that it cannot be treated
        as episodic. Even "continuous" recordings can be treated as
        single-episode recordings and could be identified as safe by this
        function. Recordings in which the user has changed groupings to create
        irregularities should be caught by this function.
        """

        # First check: Old AxoGraph file formats do not contain enough metadata
        # to know for certain that the file is episodic.
        if self.info['format_ver'] < 3:
            self.logger.debug('Cannot treat as episodic because old format '
                              'contains insufficient metadata')
            return False

        # Second check: If the file is episodic, it should report that it
        # contains more than 1 episode.
        if 'n_episodes' not in self.info:
            self.logger.debug('Cannot treat as episodic because episode '
                              'metadata is missing or could not be parsed')
            return False
        if self.info['n_episodes'] == 1:
            self.logger.debug('Cannot treat as episodic because file reports '
                              'one episode')
            return False

        # Third check: If the file is episodic, groups of traces should all
        # contain the same number of traces, one for each episode. This is
        # generally true of "continuous" (single-episode) recordings as well,
        # which normally have 1 trace per group.
        if 'group_header_info_list' not in self.info:
            self.logger.debug('Cannot treat as episodic because group '
                              'metadata is missing or could not be parsed')
            return False
        if 'trace_header_info_list' not in self.info:
            self.logger.debug('Cannot treat as episodic because trace '
                              'metadata is missing or could not be parsed')
            return False

        group_id_to_col_indexes = {}
        for group_id in self.info['group_header_info_list']:
            col_indexes = []
            for trace_header in self.info['trace_header_info_list'].values():
                if trace_header['group_id_for_this_trace'] == group_id:
                    col_indexes.append(trace_header['y_index'])
            group_id_to_col_indexes[group_id] = col_indexes
        n_traces_by_group = {k: len(v) for k, v in
                             group_id_to_col_indexes.items()}
        all_groups_have_same_number_of_traces = len(np.unique(list(
            n_traces_by_group.values()))) == 1

        if not all_groups_have_same_number_of_traces:
            self.logger.debug('Cannot treat as episodic because groups differ '
                              'in number of traces')
            return False

        # Fourth check: The number of traces in each group should equal
        # n_episodes.
        n_traces_per_group = np.unique(list(n_traces_by_group.values()))
        if n_traces_per_group != self.info['n_episodes']:
            self.logger.debug('Cannot treat as episodic because n_episodes '
                              'does not match number of traces per group')
            return False

        # Fifth check: If the file is episodic, all traces within a group
        # should have identical signal channel parameters (e.g., name, units)
        # except for their unique ids. This too is generally true of
        # "continuous" (single-episode) files, which normally have 1 trace per
        # group.
        signal_channels_with_ids_dropped = \
            self.header['signal_channels'][
                [n for n in self.header['signal_channels'].dtype.names
                 if n != 'id']]
        group_has_uniform_signal_parameters = {}
        for group_id, col_indexes in group_id_to_col_indexes.items():
            # subtract 1 from indexes in next statement because time is not
            # included in signal_channels
            signal_params_for_group = np.array(
                signal_channels_with_ids_dropped[np.array(col_indexes) - 1])
            group_has_uniform_signal_parameters[group_id] = \
                len(np.unique(signal_params_for_group)) == 1
        all_groups_have_uniform_signal_parameters = \
            np.all(list(group_has_uniform_signal_parameters.values()))

        if not all_groups_have_uniform_signal_parameters:
            self.logger.debug('Cannot treat as episodic because some groups '
                              'have heterogeneous signal parameters')
            return False

        # all checks passed
        self.logger.debug('Can treat as episodic')
        return True

    def _convert_to_multi_segment(self):
        """
        Reshape signal headers and signal data for an episodic file
        """

        self.header['nb_segment'] = [self.info['n_episodes']]

        # drop repeated signal headers
        self.header['signal_channels'] = \
            self.header['signal_channels'].reshape(
                self.info['n_episodes'], -1)[0]

        # reshape signal memmap list
        new_sig_memmaps = []
        n_channels = len(self.header['signal_channels'])
        sig_memmaps = self._raw_signals[0]
        for first_index in np.arange(0, len(sig_memmaps), n_channels):
            new_sig_memmaps.append(
                sig_memmaps[first_index:first_index + n_channels])
        self._raw_signals = new_sig_memmaps

        self.logger.debug('New number of segments: {}'.format(
            self.info['n_episodes']))

        return

    def _get_rec_datetime(self):
        """
        Determine the date and time at which the recording was started from
        automatically generated notes. How these notes should be parsed differs
        depending on whether the recording was obtained in episodic or
        continuous acquisition mode.
        """

        rec_datetime = None
        date_string = ''
        time_string = ''
        datetime_string = ''

        if 'notes' not in self.info:
            return None

        for note_line in self.info['notes'].split('\n'):

            # episodic acquisition mode
            if note_line.startswith('Created on '):
                date_string = note_line.strip('Created on ')
            if note_line.startswith('Start data acquisition at '):
                time_string = note_line.strip('Start data acquisition at ')

            # continuous acquisition mode
            if note_line.startswith('Created : '):
                datetime_string = note_line.strip('Created : ')

        if date_string and time_string:
            datetime_string = ' '.join([date_string, time_string])

        if datetime_string:
            try:
                rec_datetime = datetime.strptime(datetime_string,
                                                 '%a %b %d %Y %H:%M:%S')
            except ValueError:
                pass

        return rec_datetime

    def _scan_axograph_file(self):
        """
        This function traverses the entire AxoGraph file, constructing memmaps
        for signals and collecting channel information and other metadata
        """

        self.info = {}

        with open(self.filename, 'rb') as fid:
            f = StructFile(fid)

            self.logger.debug('filename: {}'.format(self.filename))
            self.logger.debug('')

            # the first 4 bytes are always a 4-character file type identifier
            # - for early versions of AxoGraph, this identifier was 'AxGr'
            # - starting with AxoGraph X, the identifier is 'axgx'
            header_id = f.read(4).decode('utf-8')
            self.info['header_id'] = header_id
            assert header_id in ['AxGr', 'axgx'], \
                'not an AxoGraph binary file! "{}"'.format(self.filename)

            self.logger.debug('header_id: {}'.format(header_id))

            # the next two numbers store the format version number and the
            # number of data columns to follow
            # - for 'AxGr' files, these numbers are 2-byte unsigned short ints
            # - for 'axgx' files, these numbers are 4-byte long ints
            # - the 4-character identifier changed from 'AxGr' to 'axgx' with
            #   format version 3
            if header_id == 'AxGr':
                format_ver, n_cols = f.read_f('HH')
                assert format_ver == 1 or format_ver == 2, \
                    'mismatch between header identifier "{}" and format ' \
                    'version "{}"!'.format(header_id, format_ver)
            elif header_id == 'axgx':
                format_ver, n_cols = f.read_f('ll')
                assert format_ver >= 3, \
                    'mismatch between header identifier "{}" and format ' \
                    'version "{}"!'.format(header_id, format_ver)
            else:
                raise NotImplementedError(
                    'unimplemented file header identifier "{}"!'.format(
                        header_id))
            self.info['format_ver'] = format_ver
            self.info['n_cols'] = n_cols

            self.logger.debug('format_ver: {}'.format(format_ver))
            self.logger.debug('n_cols: {}'.format(n_cols))
            self.logger.debug('')

            ##############################################
            # BEGIN COLUMNS

            sig_memmaps = []
            sig_channels = []
            for i in range(n_cols):

                self.logger.debug('== COLUMN INDEX {} =='.format(i))

                ##############################################
                # NUMBER OF DATA POINTS IN COLUMN

                n_points = f.read_f('l')

                self.logger.debug('n_points: {}'.format(n_points))

                ##############################################
                # COLUMN TYPE

                # depending on the format version, data columns may have a type
                # - prior to version 3, column types did not exist and data was
                #   stored in a fixed pattern
                # - beginning with version 3, several data types are available
                #   as documented in AxoGraph_ReadWrite.h
                if format_ver == 1 or format_ver == 2:
                    col_type = None
                elif format_ver >= 3:
                    col_type = f.read_f('l')
                else:
                    raise NotImplementedError(
                        'unimplemented file format version "{}"!'.format(
                            format_ver))

                self.logger.debug('col_type: {}'.format(col_type))

                ##############################################
                # COLUMN NAME AND UNITS

                # depending on the format version, column titles are stored
                # differently
                # - prior to version 3, column titles were stored as
                #   fixed-length 80-byte Pascal strings
                # - beginning with version 3, column titles are stored as
                #   variable-length strings (see StructFile.read_string for
                #   details)
                if format_ver == 1 or format_ver == 2:
                    title = f.read_f('80p').decode('utf-8')
                elif format_ver >= 3:
                    title = f.read_f('S')
                else:
                    raise NotImplementedError(
                        'unimplemented file format version "{}"!'.format(
                            format_ver))

                self.logger.debug('title: {}'.format(title))

                # units are given in parentheses at the end of a column title,
                # unless units are absent
                if len(title.split()) > 0 and title.split()[-1][0] == '(' and \
                   title.split()[-1][-1] == ')':
                    name = ' '.join(title.split()[:-1])
                    units = title.split()[-1].strip('()')
                else:
                    name = title
                    units = ''

                self.logger.debug('name: {}'.format(name))
                self.logger.debug('units: {}'.format(units))

                ##############################################
                # COLUMN DTYPE, SCALE, OFFSET

                if format_ver == 1:

                    # for format version 1, all columns are arrays of floats

                    dtype = 'f'
                    gain, offset = 1, 0  # data is neither scaled nor off-set

                elif format_ver == 2:

                    # for format version 2, the first column is a "series" of
                    # regularly spaced values specified merely by a first value
                    # and an increment, and all subsequent columns are arrays
                    # of shorts with a scaling factor

                    if i == 0:

                        # series
                        first_value, increment = f.read_f('ff')

                        self.logger.debug(
                            'interval: {}, freq: {}'.format(
                                increment, 1 / increment))
                        self.logger.debug(
                            'start: {}, end: {}'.format(
                                first_value,
                                first_value + increment * (n_points - 1)))

                        # assume this is the time column
                        t_start, sampling_period = first_value, increment
                        self.info['t_start'] = t_start
                        self.info['sampling_period'] = sampling_period

                        self.logger.debug('')

                        continue  # skip memmap, chan info for time col

                    else:

                        # scaled short
                        dtype = 'h'
                        gain, offset = \
                            f.read_f('f'), 0  # data is scaled without offset

                elif format_ver >= 3:

                    # for format versions 3 and later, the column type
                    # determines how the data should be read
                    # - column types 1, 2, 3, and 8 are not defined in
                    #   AxoGraph_ReadWrite.h
                    # - column type 9 is different from the others in that it
                    #   represents regularly spaced values
                    #   (such as times at a fixed frequency) specified by a
                    #   first value and an increment, without storing a large
                    #   data array

                    if col_type == 9:

                        # series
                        first_value, increment = f.read_f('dd')

                        self.logger.debug(
                            'interval: {}, freq: {}'.format(
                                increment, 1 / increment))
                        self.logger.debug(
                            'start: {}, end: {}'.format(
                                first_value,
                                first_value + increment * (n_points - 1)))

                        if i == 0:

                            # assume this is the time column
                            t_start, sampling_period = first_value, increment
                            self.info['t_start'] = t_start
                            self.info['sampling_period'] = sampling_period

                            self.logger.debug('')

                            continue  # skip memmap, chan info for time col

                        else:

                            raise NotImplementedError(
                                'series data are supported only for the first '
                                'data column (time)!')

                    elif col_type == 4:

                        # short
                        dtype = 'h'
                        gain, offset = 1, 0  # data neither scaled nor off-set

                    elif col_type == 5:

                        # long
                        dtype = 'l'
                        gain, offset = 1, 0  # data neither scaled nor off-set

                    elif col_type == 6:

                        # float
                        dtype = 'f'
                        gain, offset = 1, 0  # data neither scaled nor off-set

                    elif col_type == 7:

                        # double
                        dtype = 'd'
                        gain, offset = 1, 0  # data neither scaled nor off-set

                    elif col_type == 10:

                        # scaled short
                        dtype = 'h'
                        gain, offset = f.read_f('dd')  # data scaled w/ offset

                    else:

                        raise NotImplementedError(
                            'unimplemented column type "{}"!'.format(col_type))

                else:

                    raise NotImplementedError(
                        'unimplemented file format version "{}"!'.format(
                            format_ver))

                ##############################################
                # COLUMN MEMMAP AND CHANNEL INFO

                # create a memory map that allows accessing parts of the file
                # without loading it all into memory
                array = np.memmap(
                    self.filename,
                    mode='r',
                    dtype=f.byte_order + dtype,
                    offset=f.tell(),
                    shape=n_points)

                # advance the file position to after the data array
                f.seek(array.nbytes, 1)

                if i == 0:
                    # assume this is the time column containing n_points values

                    # verify times are spaced regularly
                    diffs = np.diff(array)
                    increment = np.median(diffs)
                    max_frac_step_deviation = np.max(np.abs(
                        diffs / increment - 1))
                    tolerance = 1e-3
                    if max_frac_step_deviation > tolerance:
                        self.logger.debug('largest proportional deviation '
                                          'from median step size in the first '
                                          'column exceeds the tolerance '
                                          'of ' + str(tolerance) + ':'
                                          ' ' + str(max_frac_step_deviation))
                        raise ValueError('first data column (assumed to be '
                                         'time) is not regularly spaced')

                    first_value = array[0]

                    self.logger.debug(
                        'interval: {}, freq: {}'.format(
                            increment, 1 / increment))
                    self.logger.debug(
                        'start: {}, end: {}'.format(
                            first_value,
                            first_value + increment * (n_points - 1)))

                    t_start, sampling_period = first_value, increment
                    self.info['t_start'] = t_start
                    self.info['sampling_period'] = sampling_period

                    self.logger.debug('')

                    continue  # skip saving memmap, chan info for time col

                else:
                    # not a time column

                    self.logger.debug('gain: {}, offset: {}'.format(gain, offset))
                    self.logger.debug('initial data: {}'.format(
                        array[:5] * gain + offset))

                    # channel_info will be cast to _signal_channel_dtype
                    channel_info = (
                        name, str(i), 1 / sampling_period, f.byte_order + dtype,
                        units, gain, offset, '0')

                    self.logger.debug('channel_info: {}'.format(channel_info))
                    self.logger.debug('')

                    sig_memmaps.append(array)
                    sig_channels.append(channel_info)

            # END COLUMNS
            ##############################################

            # initialize lists for events and epochs
            raw_event_timestamps = []
            raw_epoch_timestamps = []
            raw_epoch_durations = []
            event_labels = []
            epoch_labels = []

            # the remainder of the file may contain metadata, events and epochs
            try:

                ##############################################
                # COMMENT

                self.logger.debug('== COMMENT ==')

                comment = f.read_f('S')
                self.info['comment'] = comment

                self.logger.debug(comment if comment else 'no comment!')
                self.logger.debug('')

                ##############################################
                # NOTES

                self.logger.debug('== NOTES ==')

                notes = f.read_f('S')
                self.info['notes'] = notes

                self.logger.debug(notes if notes else 'no notes!')
                self.logger.debug('')

                ##############################################
                # TRACES

                self.logger.debug('== TRACES ==')

                n_traces = f.read_f('l')
                self.info['n_traces'] = n_traces

                self.logger.debug('n_traces: {}'.format(n_traces))
                self.logger.debug('')

                trace_header_info_list = {}
                group_ids = []
                for i in range(n_traces):

                    # AxoGraph traces are 1-indexed in GUI, so use i+1 below
                    self.logger.debug('== TRACE #{} =='.format(i + 1))

                    trace_header_info = {}

                    if format_ver < 6:
                        # before format version 6, there was only one version
                        # of the header, and version numbers were not provided
                        trace_header_info['trace_header_version'] = 1
                    else:
                        # for format versions 6 and later, the header version
                        # must be read
                        trace_header_info['trace_header_version'] = \
                            f.read_f('l')

                    if trace_header_info['trace_header_version'] == 1:
                        TraceHeaderDescription = TraceHeaderDescriptionV1
                    elif trace_header_info['trace_header_version'] == 2:
                        TraceHeaderDescription = TraceHeaderDescriptionV2
                    else:
                        raise NotImplementedError(
                            'unimplemented trace header version "{}"!'.format(
                                trace_header_info['trace_header_version']))

                    for key, fmt in TraceHeaderDescription:
                        trace_header_info[key] = f.read_f(fmt)
                    # AxoGraph traces are 1-indexed in GUI, so use i+1 below
                    trace_header_info_list[i + 1] = trace_header_info
                    group_ids.append(
                        trace_header_info['group_id_for_this_trace'])

                    self.logger.debug(trace_header_info)
                    self.logger.debug('')
                self.info['trace_header_info_list'] = trace_header_info_list

                ##############################################
                # GROUPS

                self.logger.debug('== GROUPS ==')

                n_groups = f.read_f('l')
                self.info['n_groups'] = n_groups
                group_ids = \
                    np.sort(list(set(group_ids)))  # remove duplicates and sort
                assert n_groups == len(group_ids), \
                    'expected group_ids to have length {}: {}'.format(
                        n_groups, group_ids)

                self.logger.debug('n_groups: {}'.format(n_groups))
                self.logger.debug('group_ids: {}'.format(group_ids))
                self.logger.debug('')

                group_header_info_list = {}
                for i in group_ids:

                    # AxoGraph groups are 0-indexed in GUI, so use i below
                    self.logger.debug('== GROUP #{} =='.format(i))

                    group_header_info = {}

                    if format_ver < 6:
                        # before format version 6, there was only one version
                        # of the header, and version numbers were not provided
                        group_header_info['group_header_version'] = 1
                    else:
                        # for format versions 6 and later, the header version
                        # must be read
                        group_header_info['group_header_version'] = \
                            f.read_f('l')

                    if group_header_info['group_header_version'] == 1:
                        GroupHeaderDescription = GroupHeaderDescriptionV1
                    else:
                        raise NotImplementedError(
                            'unimplemented group header version "{}"!'.format(
                                group_header_info['group_header_version']))

                    for key, fmt in GroupHeaderDescription:
                        group_header_info[key] = f.read_f(fmt)
                    # AxoGraph groups are 0-indexed in GUI, so use i below
                    group_header_info_list[i] = group_header_info

                    self.logger.debug(group_header_info)
                    self.logger.debug('')
                self.info['group_header_info_list'] = group_header_info_list

                ##############################################
                # UNKNOWN

                self.logger.debug('>> UNKNOWN 1 <<')

                # 36 bytes of undeciphered data (types here are guesses)
                unknowns = f.read_f('9l')

                self.logger.debug(unknowns)
                self.logger.debug('')

                ##############################################
                # EPISODES

                self.logger.debug('== EPISODES ==')

                # a subset of episodes can be selected for "review", or
                # episodes can be paged through one by one, and the indexes of
                # those currently in review appear in this list
                episodes_in_review = []
                n_episodes = f.read_f('l')
                self.info['n_episodes'] = n_episodes
                for i in range(n_episodes):
                    episode_bool = f.read_f('Z')
                    if episode_bool:
                        episodes_in_review.append(i + 1)
                self.info['episodes_in_review'] = episodes_in_review

                self.logger.debug('n_episodes: {}'.format(n_episodes))
                self.logger.debug('episodes_in_review: {}'.format(
                    episodes_in_review))

                if format_ver == 5:

                    # the test file for version 5 contains this extra list of
                    # episode indexes with unknown purpose
                    old_unknown_episode_list = []
                    n_episodes2 = f.read_f('l')
                    for i in range(n_episodes2):
                        episode_bool = f.read_f('Z')
                        if episode_bool:
                            old_unknown_episode_list.append(i + 1)

                    self.logger.debug('old_unknown_episode_list: {}'.format(
                        old_unknown_episode_list))
                    if n_episodes2 != n_episodes:
                        self.logger.debug(
                            'n_episodes2 ({}) and n_episodes ({}) '
                            'differ!'.format(n_episodes2, n_episodes))

                # another list of episode indexes with unknown purpose
                unknown_episode_list = []
                n_episodes3 = f.read_f('l')
                for i in range(n_episodes3):
                    episode_bool = f.read_f('Z')
                    if episode_bool:
                        unknown_episode_list.append(i + 1)

                self.logger.debug('unknown_episode_list: {}'.format(
                    unknown_episode_list))
                if n_episodes3 != n_episodes:
                    self.logger.debug(
                        'n_episodes3 ({}) and n_episodes ({}) '
                        'differ!'.format(n_episodes3, n_episodes))

                # episodes can be masked to be removed from the pool of
                # reviewable episodes completely until unmasked, and the
                # indexes of those currently masked appear in this list
                masked_episodes = []
                n_episodes4 = f.read_f('l')
                for i in range(n_episodes4):
                    episode_bool = f.read_f('Z')
                    if episode_bool:
                        masked_episodes.append(i + 1)
                self.info['masked_episodes'] = masked_episodes

                self.logger.debug('masked_episodes: {}'.format(
                    masked_episodes))
                if n_episodes4 != n_episodes:
                    self.logger.debug(
                        'n_episodes4 ({}) and n_episodes ({}) '
                        'differ!'.format(n_episodes4, n_episodes))
                self.logger.debug('')

                ##############################################
                # UNKNOWN

                self.logger.debug('>> UNKNOWN 2 <<')

                # 68 bytes of undeciphered data (types here are guesses)
                unknowns = f.read_f('d 9l d 4l')

                self.logger.debug(unknowns)
                self.logger.debug('')

                ##############################################
                # FONTS

                if format_ver >= 6:
                    font_categories = ['axis titles', 'axis labels (ticks)',
                                       'notes', 'graph title']
                else:
                    # would need an old version of AxoGraph to determine how it
                    # used these settings
                    font_categories = ['everything (?)']

                font_settings_info_list = {}
                for i in font_categories:

                    self.logger.debug('== FONT SETTINGS FOR {} =='.format(i))

                    font_settings_info = {}
                    for key, fmt in FontSettingsDescription:
                        font_settings_info[key] = f.read_f(fmt)

                    # I don't know why two arbitrary values were selected to
                    # represent this switch, but it seems they were
                    # - setting1 could contain other undeciphered data as a
                    #   bitmask, like setting2
                    assert font_settings_info['setting1'] in \
                        [FONT_BOLD, FONT_NOT_BOLD], \
                        'expected setting1 ({}) to have value FONT_BOLD ' \
                        '({}) or FONT_NOT_BOLD ({})'.format(
                            font_settings_info['setting1'],
                            FONT_BOLD,
                            FONT_NOT_BOLD)

                    # size is stored 10 times bigger than real value
                    font_settings_info['size'] = \
                        font_settings_info['size'] / 10.0
                    font_settings_info['bold'] = \
                        bool(font_settings_info['setting1'] == FONT_BOLD)
                    font_settings_info['italics'] = \
                        bool(font_settings_info['setting2'] & FONT_ITALICS)
                    font_settings_info['underline'] = \
                        bool(font_settings_info['setting2'] & FONT_UNDERLINE)
                    font_settings_info['strikeout'] = \
                        bool(font_settings_info['setting2'] & FONT_STRIKEOUT)
                    font_settings_info_list[i] = font_settings_info

                    self.logger.debug(font_settings_info)
                    self.logger.debug('')
                self.info['font_settings_info_list'] = font_settings_info_list

                ##############################################
                # X-AXIS SETTINGS

                self.logger.debug('== X-AXIS SETTINGS ==')

                x_axis_settings_info = {}
                for key, fmt in XAxisSettingsDescription:
                    x_axis_settings_info[key] = f.read_f(fmt)
                self.info['x_axis_settings_info'] = x_axis_settings_info

                self.logger.debug(x_axis_settings_info)
                self.logger.debug('')

                ##############################################
                # UNKNOWN

                self.logger.debug('>> UNKNOWN 3 <<')

                # 108 bytes of undeciphered data (types here are guesses)
                unknowns = f.read_f('8l 3d 13l')

                self.logger.debug(unknowns)
                self.logger.debug('')

                ##############################################
                # EVENTS / TAGS

                self.logger.debug('=== EVENTS / TAGS ===')

                n_events, n_events_again = f.read_f('ll')
                self.info['n_events'] = n_events

                self.logger.debug('n_events: {}'.format(n_events))

                # event / tag timing is stored as an index into time
                raw_event_timestamps = []
                event_labels = []
                for i in range(n_events_again):
                    event_index = f.read_f('l')
                    raw_event_timestamps.append(event_index)
                n_events_yet_again = f.read_f('l')
                for i in range(n_events_yet_again):
                    title = f.read_f('S')
                    event_labels.append(title)

                event_list = []
                for event_label, event_index in \
                        zip(event_labels, raw_event_timestamps):
                    # t_start shouldn't be added here
                    event_time = event_index * sampling_period
                    event_list.append({
                        'title': event_label,
                        'index': event_index,
                        'time': event_time})
                self.info['event_list'] = event_list
                for event in event_list:
                    self.logger.debug(event)
                self.logger.debug('')

                ##############################################
                # UNKNOWN

                self.logger.debug('>> UNKNOWN 4 <<')

                # 28 bytes of undeciphered data (types here are guesses)
                unknowns = f.read_f('7l')

                self.logger.debug(unknowns)
                self.logger.debug('')

                ##############################################
                # EPOCHS / INTERVAL BARS

                self.logger.debug('=== EPOCHS / INTERVAL BARS ===')

                n_epochs = f.read_f('l')
                self.info['n_epochs'] = n_epochs

                self.logger.debug('n_epochs: {}'.format(n_epochs))

                epoch_list = []
                for i in range(n_epochs):
                    epoch_info = {}
                    for key, fmt in EpochInfoDescription:
                        epoch_info[key] = f.read_f(fmt)
                    epoch_list.append(epoch_info)
                self.info['epoch_list'] = epoch_list

                # epoch / interval bar timing and duration are stored in
                # seconds, so here they are converted to (possibly non-integer)
                # indexes into time to fit into the procrustean beds of
                # _rescale_event_timestamp and _rescale_epoch_duration
                raw_epoch_timestamps = []
                raw_epoch_durations = []
                epoch_labels = []
                for epoch in epoch_list:
                    raw_epoch_timestamps.append(
                        epoch['t_start'] / sampling_period)
                    raw_epoch_durations.append(
                        (epoch['t_stop'] - epoch['t_start']) / sampling_period)
                    epoch_labels.append(epoch['title'])
                    self.logger.debug(epoch)
                self.logger.debug('')

                ##############################################
                # UNKNOWN

                self.logger.debug(
                    '>> UNKNOWN 5 (includes y-axis plot ranges) <<')

                # lots of undeciphered data
                rest_of_the_file = f.read()

                self.logger.debug(rest_of_the_file)
                self.logger.debug('')

                self.logger.debug('End of file reached (expected)')

            except EOFError as e:
                if format_ver == 1 or format_ver == 2:
                    # for format versions 1 and 2, metadata like graph display
                    # information was stored separately in the "resource fork"
                    # of the file, so reaching the end of the file before all
                    # metadata is parsed is expected
                    self.logger.debug('End of file reached (expected)')
                    pass
                else:
                    # for format versions 3 and later, there should be metadata
                    # stored at the end of the file, so warn that something may
                    # have gone wrong, but try to continue anyway
                    self.logger.warning('End of file reached unexpectedly '
                                        'while parsing metadata, will attempt '
                                        'to continue')
                    self.logger.debug(e, exc_info=True)
                    pass

            except UnicodeDecodeError as e:
                # warn that something went wrong with reading a string, but try
                # to continue anyway
                self.logger.warning('Problem decoding text while parsing '
                                    'metadata, will ignore any remaining '
                                    'metadata and attempt to continue')
                self.logger.debug(e, exc_info=True)
                pass

        self.logger.debug('')

        ##############################################
        # RAWIO HEADER

        # event_channels will be cast to _event_channel_dtype
        event_channels = []
        event_channels.append(('AxoGraph Tags', '', 'event'))
        event_channels.append(('AxoGraph Intervals', '', 'epoch'))

        if len(sig_channels) > 0:
            signal_streams = [('Signals', '0')]
        else:
            signal_streams = []

        # organize header
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = np.array(signal_streams, dtype=_signal_stream_dtype)
        self.header['signal_channels'] = np.array(sig_channels, dtype=_signal_channel_dtype)
        self.header['event_channels'] = np.array(event_channels, dtype=_event_channel_dtype)
        self.header['spike_channels'] = np.array([], dtype=_spike_channel_dtype)

        ##############################################
        # DATA OBJECTS

        # organize data
        self._sampling_period = sampling_period
        self._t_start = t_start
        self._raw_signals = [sig_memmaps]  # first index is seg_index
        self._raw_event_epoch_timestamps = [
            np.array(raw_event_timestamps),
            np.array(raw_epoch_timestamps)]
        self._raw_event_epoch_durations = [
            None,
            np.array(raw_epoch_durations)]
        self._event_epoch_labels = [
            np.array(event_labels, dtype='U'),
            np.array(epoch_labels, dtype='U')]


class StructFile(BufferedReader):
    """
    A container for the file buffer with some added convenience functions for
    reading AxoGraph files
    """

    def __init__(self, *args, **kwargs):
        # As far as I've seen, every AxoGraph file uses big-endian encoding,
        # regardless of the system architecture on which it was created, but
        # here I provide means for controlling byte ordering in case a counter
        # example is found.
        self.byte_order = kwargs.pop('byte_order', '>')
        if self.byte_order == '>':
            # big-endian
            self.utf_16_decoder = 'utf-16-be'
        elif self.byte_order == '<':
            # little-endian
            self.utf_16_decoder = 'utf-16-le'
        else:
            # unspecified
            self.utf_16_decoder = 'utf-16'
        super().__init__(*args, **kwargs)

    def read_and_unpack(self, fmt):
        """
        Calculate the number of bytes corresponding to the format string, read
        in that number of bytes, and unpack them according to the format string
        """
        try:
            return unpack(
                self.byte_order + fmt,
                self.read(calcsize(self.byte_order + fmt)))
        except Exception as e:
            if e.args[0].startswith('unpack requires a buffer of'):
                raise EOFError(e)
            else:
                raise

    def read_string(self):
        """
        The most common string format in AxoGraph files is a variable length
        string with UTF-16 encoding, preceded by a 4-byte integer (long)
        specifying the length of the string in bytes. Unlike a Pascal string
        ('p' format), these strings are not stored in a fixed number of bytes
        with padding at the end. This function reads in one of these variable
        length strings
        """

        # length may be -1, 0, or a positive integer
        length = self.read_and_unpack('l')[0]
        if length > 0:
            return self.read(length).decode(self.utf_16_decoder)
        else:
            return ''

    def read_bool(self):
        """
        AxoGraph files encode each boolean as 4-byte integer (long) with value
        1 = True, 0 = False. This function reads in one of these booleans.
        """
        return bool(self.read_and_unpack('l')[0])

    def read_f(self, fmt, offset=None):
        """
        This function is a wrapper for read_and_unpack that adds compatibility
        with two new format strings:
            'S': a variable length UTF-16 string, readable with read_string
            'Z': a boolean encoded as a 4-byte integer, readable with read_bool
        This method does not implement support for numbers before the new
        format strings, such as '3Z' to represent 3 bools (use 'ZZZ' instead).
        """

        if offset is not None:
            self.seek(offset)

        # place commas before and after each instance of S or Z
        for special in ['S', 'Z']:
            fmt = fmt.replace(special, ',' + special + ',')

        # split S and Z into isolated strings
        fmt = fmt.split(',')

        # construct a tuple of unpacked data
        data = ()
        for subfmt in fmt:
            if subfmt == 'S':
                data += (self.read_string(),)
            elif subfmt == 'Z':
                data += (self.read_bool(),)
            else:
                data += self.read_and_unpack(subfmt)

        if len(data) == 1:
            return data[0]
        else:
            return data


FONT_BOLD = 75      # mysterious arbitrary constant
FONT_NOT_BOLD = 50  # mysterious arbitrary constant
FONT_ITALICS = 1
FONT_UNDERLINE = 2
FONT_STRIKEOUT = 4

TraceHeaderDescriptionV1 = [
    # documented in AxoGraph_ReadWrite.h
    ('x_index', 'l'),
    ('y_index', 'l'),
    ('err_bar_index', 'l'),
    ('group_id_for_this_trace', 'l'),
    ('hidden', 'Z'),  # AxoGraph_ReadWrite.h incorrectly states "shown" instead
    ('min_x', 'd'),
    ('max_x', 'd'),
    ('min_positive_x', 'd'),
    ('x_is_regularly_spaced', 'Z'),
    ('x_increases_monotonically', 'Z'),
    ('x_interval_if_regularly_spaced', 'd'),
    ('min_y', 'd'),
    ('max_y', 'd'),
    ('min_positive_y', 'd'),
    ('trace_color', 'xBBB'),
    ('display_joined_line_plot', 'Z'),
    ('line_thickness', 'd'),
    ('pen_style', 'l'),
    ('display_symbol_plot', 'Z'),
    ('symbol_type', 'l'),
    ('symbol_size', 'l'),
    ('draw_every_data_point', 'Z'),
    ('skip_points_by_distance_instead_of_pixels', 'Z'),
    ('pixels_between_symbols', 'l'),
    ('display_histogram_plot', 'Z'),
    ('histogram_type', 'l'),
    ('histogram_bar_separation', 'l'),
    ('display_error_bars', 'Z'),
    ('display_pos_err_bar', 'Z'),
    ('display_neg_err_bar', 'Z'),
    ('err_bar_width', 'l'),
]

# documented in AxoGraph_ReadWrite.h
# - only one difference exists between versions 1 and 2
TraceHeaderDescriptionV2 = list(TraceHeaderDescriptionV1)  # make a copy
TraceHeaderDescriptionV2.insert(3, ('neg_err_bar_index', 'l'))

GroupHeaderDescriptionV1 = [
    # undocumented and reverse engineered
    ('title', 'S'),
    ('unknown1', 'h'),     # 2 bytes of undeciphered data (types are guesses)
    ('units', 'S'),
    ('unknown2', 'hll'),   # 10 bytes of undeciphered data (types are guesses)
]

FontSettingsDescription = [
    # undocumented and reverse engineered
    ('font', 'S'),
    ('size', 'h'),         # divide this 2-byte integer by 10 to get font size
    ('unknown1', '5b'),    # 5 bytes of undeciphered data (types are guesses)
    ('setting1', 'B'),     # includes bold setting
    ('setting2', 'B'),     # italics, underline, strikeout specified in bitmap
]

XAxisSettingsDescription = [
    # undocumented and reverse engineered
    ('unknown1', '3l2d'),  # 28 bytes of undeciphered data (types are guesses)
    ('plotted_x_range', 'dd'),
    ('unknown2', 'd'),     # 8 bytes of undeciphered data (types are guesses)
    ('auto_x_ticks', 'Z'),
    ('x_minor_ticks', 'd'),
    ('x_major_ticks', 'd'),
    ('x_axis_title', 'S'),
    ('unknown3', 'h'),     # 2 bytes of undeciphered data (types are guesses)
    ('units', 'S'),
    ('unknown4', 'h'),     # 2 bytes of undeciphered data (types are guesses)
]

EpochInfoDescription = [
    # undocumented and reverse engineered
    ('title', 'S'),
    ('t_start', 'd'),
    ('t_stop', 'd'),
    ('y_pos', 'd'),
]
