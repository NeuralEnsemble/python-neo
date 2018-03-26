# -*- coding: utf-8 -*-
"""
BCI2000RawIO is a class to read BCI2000 .dat files.
https://www.bci2000.org/mediawiki/index.php/Technical_Reference:BCI2000_File_Format
"""
from __future__ import print_function, division, absolute_import  # unicode_literals

from .baserawio import BaseRawIO, _signal_channel_dtype, _unit_channel_dtype, _event_channel_dtype

import numpy as np
import re

try:
    from urllib.parse import unquote
except ImportError:
    from urllib import url2pathname as unquote


class BCI2000RawIO(BaseRawIO):
    """
    Class for reading data from a BCI2000 .dat file, either version 1.0 or 1.1
    """
    extensions = ['dat']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename
        self._my_events = None

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        file_info, state_defs, param_defs = parse_bci2000_header(self.filename)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]

        sig_channels = []
        for chan_ix in range(file_info['SourceCh']):
            ch_name = param_defs['ChannelNames']['value'][chan_ix] \
                if 'ChannelNames' in param_defs else 'ch' + str(chan_ix)
            chan_id = chan_ix + 1
            sr = param_defs['SamplingRate']['value']  # Hz
            dtype = file_info['DataFormat']
            units = 'uV'
            gain = param_defs['SourceChGain']['value'][chan_ix]
            offset = param_defs['SourceChOffset']['value'][chan_ix]
            group_id = 0
            sig_channels.append((ch_name, chan_id, sr, dtype, units, gain, offset, group_id))
        self.header['signal_channels'] = np.array(sig_channels, dtype=_signal_channel_dtype)

        self.header['unit_channels'] = np.array([], dtype=_unit_channel_dtype)

        # creating event channel for each state variable
        event_channels = []
        for st_ix, st_tup in enumerate(state_defs):
            event_channels.append((st_tup[0], 'ev_' + str(st_ix), 'event'))
        self.header['event_channels'] = np.array(event_channels, dtype=_event_channel_dtype)

        # Add annotations.

        # Generates basic annotations in nested dict self.raw_annotations
        self._generate_minimal_annotations()

        self.raw_annotations['blocks'][0].update({
            'file_info': file_info,
            'param_defs': param_defs
        })
        for ev_ix, ev_dict in enumerate(self.raw_annotations['event_channels']):
            ev_dict.update({
                'length': state_defs[ev_ix][1],
                'startVal': state_defs[ev_ix][2],
                'bytePos': state_defs[ev_ix][3],
                'bitPos': state_defs[ev_ix][4]
            })

        import time
        time_formats = ['%a %b %d %H:%M:%S %Y', '%Y-%m-%dT%H:%M:%S']
        try:
            self._global_time = time.mktime(time.strptime(param_defs['StorageTime']['value'],
                                                          time_formats[0]))
        except:
            self._global_time = time.mktime(time.strptime(param_defs['StorageTime']['value'],
                                                          time_formats[1]))

        # Save variables to make it easier to load the binary data.
        self._read_info = {
            'header_len': file_info['HeaderLen'],
            'n_chans': file_info['SourceCh'],
            'sample_dtype': {
                'int16': np.int16,
                'int32': np.int32,
                'float32': np.float32}.get(file_info['DataFormat']),
            'state_vec_len': file_info['StatevectorLen'],
            'sampling_rate': param_defs['SamplingRate']['value']
        }
        # Calculate the dtype for a single timestamp of data. This contains the data + statevector
        self._read_info['line_dtype'] = [
            ('raw_vector', self._read_info['sample_dtype'], self._read_info['n_chans']),
            ('state_vector', np.uint8, self._read_info['state_vec_len'])]
        import os
        self._read_info['n_samps'] = int((os.stat(self.filename).st_size - file_info['HeaderLen'])
                                         / np.dtype(self._read_info['line_dtype']).itemsize)

        # memmap is fast so we can get the data ready for reading now.
        self._memmap = np.memmap(self.filename, dtype=self._read_info['line_dtype'],
                                 offset=self._read_info['header_len'], mode='r')

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        return self._read_info['n_samps'] / self._read_info['sampling_rate']

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        return self._read_info['n_samps']

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._read_info['n_samps']
        assert (0 <= i_start <= self._read_info['n_samps']), "i_start outside data range"
        assert (0 <= i_stop <= self._read_info['n_samps']), "i_stop outside data range"
        if channel_indexes is None:
            channel_indexes = np.arange(self.header['signal_channels'].size)
        return self._memmap['raw_vector'][i_start:i_stop, channel_indexes]

    def _spike_count(self, block_index, seg_index, unit_index):
        return 0

    def _get_spike_timestamps(self, block_index, seg_index, unit_index, t_start, t_stop):
        return None

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        return None

    def _get_spike_raw_waveforms(self, block_index, seg_index, unit_index, t_start, t_stop):
        return None

    def _event_count(self, block_index, seg_index, event_channel_index):
        return self._event_arrays_list[event_channel_index][0].shape[0]

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        # Return 3 numpy arrays: timestamp, durations, labels
        # durations must be None for 'event'
        # label must a dtype ='U'
        ts, dur, labels = self._event_arrays_list[event_channel_index]
        # seg_t_start = self._segment_t_start(block_index, seg_index)
        keep = np.ones(ts.shape, dtype=np.bool)
        if t_start is not None:
            keep = np.logical_and(keep, ts >= t_start)
        if t_stop is not None:
            keep = np.logical_and(keep, ts <= t_stop)
        return ts[keep], dur[keep], labels[keep]

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        event_times = (event_timestamps / float(self._read_info['sampling_rate'])).astype(dtype)
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype):
        durations = (raw_duration / float(self._read_info['sampling_rate'])).astype(dtype)
        return durations

    @property
    def _event_arrays_list(self):
        if self._my_events is None:
            self._my_events = []
            for s_ix, sd in enumerate(self.raw_annotations['event_channels']):
                ev_times = durs = vals = np.array([])
                # Skip these big but mostly useless (?) states.
                if sd['name'] not in ['SourceTime', 'StimulusTime']:
                    # Determine which bytes of self._memmap['state_vector'] are needed.
                    nbytes = int(np.ceil((sd['bitPos'] + sd['length']) / 8))
                    byte_slice = slice(sd['bytePos'], sd['bytePos'] + nbytes)
                    # Then determine how to mask those bytes to get only the needed bits.
                    bit_mask = np.array([255] * nbytes, dtype=np.uint8)
                    bit_mask[0] &= 255 & (255 << sd['bitPos'])  # Fix the mask for the first byte
                    extra_bits = 8 - (sd['bitPos'] + sd['length']) % 8
                    bit_mask[-1] &= 255 & (255 >> extra_bits)  # Fix the mask for the last byte
                    # When converting to an int, we need to know which integer type it will become
                    n_max_bytes = 1 << (nbytes - 1).bit_length()
                    view_type = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}.get(n_max_bytes)
                    # Slice and mask the data
                    masked_byte_array = self._memmap['state_vector'][:, byte_slice] & bit_mask
                    # Convert byte array to a vector of ints:
                    # pad to give even columns then view as larger int type
                    state_vec = np.pad(masked_byte_array,
                                       (0, n_max_bytes - nbytes),
                                       'constant').view(dtype=view_type)
                    state_vec = np.right_shift(state_vec, sd['bitPos'])[:, 0]

                    # In the state vector, find 'events' whenever the state changes
                    st_ch_ix = np.where(np.hstack((0, np.diff(state_vec))) != 0)[0]  # event inds
                    if len(st_ch_ix) > 0:
                        ev_times = st_ch_ix
                        durs = np.asarray([None] * len(st_ch_ix))
                        # np.hstack((np.diff(st_ch_ix), len(state_vec) - st_ch_ix[-1]))
                        vals = np.char.mod('%d', state_vec[st_ch_ix])  # event val, string'd

                self._my_events.append([ev_times, durs, vals])

        return self._my_events


def parse_bci2000_header(filename):
    # typically we want parameter values in Hz, seconds, or microvolts.
    scales_dict = {
        'hz': 1, 'khz': 1000, 'mhz': 1000000,
        'uv': 1, 'muv': 1, 'mv': 1000, 'v': 1000000,
        's': 1, 'us': 0.000001, 'mus': 0.000001, 'ms': 0.001, 'min': 60,
        'sec': 1, 'usec': 0.000001, 'musec': 0.000001, 'msec': 0.001
    }

    def rescale_value(param_val, data_type):
        unit_str = ''
        if param_val.lower().startswith('0x'):
            param_val = int(param_val, 16)
        elif data_type in ['int', 'float']:
            matches = re.match('(-*\d+)(\w*)', param_val)
            if matches is not None:  # Can be None for % in def, min, max vals
                param_val, unit_str = matches.group(1), matches.group(2)
                param_val = int(param_val) if data_type == 'int' else float(param_val)
                if len(unit_str) > 0:
                    param_val *= scales_dict.get(unit_str.lower(), 1)
        else:
            param_val = unquote(param_val)
        return param_val, unit_str

    def parse_dimensions(param_list):
        num_els = param_list.pop(0)
        # Sometimes the number of elements isn't given,
        # but the list of element labels is wrapped with {}
        if num_els == '{':
            num_els = param_list.index('}')
            el_labels = [unquote(param_list.pop(0)) for x in range(num_els)]
            param_list.pop(0)  # Remove the '}'
        else:
            num_els = int(num_els)
            el_labels = [str(ix) for ix in range(num_els)]
        return num_els, el_labels

    import io
    with io.open(filename, 'rb') as fid:

        # Parse the file header (plain text)

        # The first line contains basic information which we store in a dictionary.
        temp = fid.readline().decode('utf8').split()
        keys = [k.rstrip('=') for k in temp[::2]]
        vals = temp[1::2]
        # Insert default version and format
        file_info = {'BCI2000V': 1.0, 'DataFormat': 'int16'}
        file_info.update(**dict(zip(keys, vals)))
        # From string to float/int
        file_info['BCI2000V'] = float(file_info['BCI2000V'])
        for k in ['HeaderLen', 'SourceCh', 'StatevectorLen']:
            if k in file_info:
                file_info[k] = int(file_info[k])

        # The next lines contain state vector definitions.
        temp = fid.readline().decode('utf8').strip()
        assert temp == '[ State Vector Definition ]', \
            "State definitions not found in header %s" % filename
        state_defs = []
        state_def_dtype = [('name', 'a64'),
                           ('length', int),
                           ('startVal', int),
                           ('bytePos', int),
                           ('bitPos', int)]
        while True:
            temp = fid.readline().decode('utf8').strip()
            if len(temp) == 0 or temp[0] == '[':
                # Presence of '[' signifies new section.
                break
            temp = temp.split()
            state_defs.append((temp[0], int(temp[1]), int(temp[2]), int(temp[3]), int(temp[4])))
        state_defs = np.array(state_defs, dtype=state_def_dtype)

        # The next lines contain parameter definitions.
        # There are many, and their formatting can be complicated.
        assert temp == '[ Parameter Definition ]', \
            "Parameter definitions not found in header %s" % filename
        param_defs = {}
        while True:
            temp = fid.readline().decode('utf8')
            if fid.tell() >= file_info['HeaderLen']:
                # End of header.
                break
            if len(temp.strip()) == 0:
                continue  # Skip empty lines
            # Everything after the '//' is a comment.
            temp = temp.strip().split('//', 1)
            param_def = {'comment': temp[1].strip() if len(temp) > 1 else ''}
            # Parse the parameter definition. Generally it is sec:cat:name dtype name param_value+
            temp = temp[0].split()
            param_def.update(
                {'section_category_name': [unquote(x) for x in temp.pop(0).split(':')]})
            dtype = temp.pop(0)
            param_name = unquote(temp.pop(0).rstrip('='))
            # Parse the rest. Parse method depends on the dtype
            param_value, units = None, None
            if dtype in ('int', 'float'):
                param_value = temp.pop(0)
                if param_value == 'auto':
                    param_value = np.nan
                    units = ''
                else:
                    param_value, units = rescale_value(param_value, dtype)
            elif dtype in ('string', 'variant'):
                param_value = unquote(temp.pop(0))
            elif dtype.endswith('list'):  # e.g., intlist, stringlist, floatlist, list
                dtype = dtype[:-4]
                # The list parameter values will begin with either
                # an int to specify the number of elements
                # or a list of labels surrounded by { }.
                num_elements, element_labels = parse_dimensions(temp)  # This will pop off info.
                param_def.update({'element_labels': element_labels})
                pv_un = [rescale_value(pv, dtype) for pv in temp[:num_elements]]
                if len(pv_un) > 0:
                    param_value, units = zip(*pv_un)
                else:
                    param_value, units = np.nan, ''
                temp = temp[num_elements:]
                # Sometimes an element list will be a list of ints even though
                # the element_type is '' (str)...
                # This usually happens for known parameters, such as SourceChOffset,
                # that can be dealt with explicitly later.
            elif dtype.endswith('matrix'):
                dtype = dtype[:-6]
                # The parameter values will be preceded by two dimension descriptors,
                # first rows then columns. Each dimension might be described by an
                # int or a list of labels surrounded by {}
                n_rows, row_labels = parse_dimensions(temp)
                n_cols, col_labels = parse_dimensions(temp)
                param_def.update({'row_labels': row_labels, 'col_labels': col_labels})

                param_value = []
                units = []
                for row_ix in range(n_rows):
                    cols = []
                    for col_ix in range(n_cols):
                        col_val, _units = rescale_value(temp[row_ix * n_cols + col_ix], dtype)
                        cols.append(col_val)
                        units.append(_units)
                    param_value.append(cols)
                temp = temp[n_rows * n_cols:]

            param_def.update({
                'value': param_value,
                'units': units,
                'dtype': dtype
            })

            # At the end of the parameter definition, we might get
            # default, min, max values for the parameter.
            temp.reverse()
            if len(temp):
                param_def.update({'max_val': rescale_value(temp.pop(0), dtype)})
            if len(temp):
                param_def.update({'min_val': rescale_value(temp.pop(0), dtype)})
            if len(temp):
                param_def.update({'default_val': rescale_value(temp.pop(0), dtype)})

            param_defs.update({param_name: param_def})
            # End parameter block
    # Outdent to close file
    return file_info, state_defs, param_defs
