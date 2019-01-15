# -*- coding: utf-8 -*-
"""

Support for intan tech rhd  and rhs files.

This 2 formats are more or less the same but:
  * some variance in headers.
  * rhs amplifier is more complexe because the optional DC channel

RHS supported version 1.0
RHD supported version  1.0 1.1 1.2 1.3 2.0

See:
  * http://intantech.com/files/Intan_RHD2000_data_file_formats.pdf
  * http://intantech.com/files/Intan_RHS2000_data_file_formats.pdf

Author: Samuel Garcia

"""
from __future__ import print_function, division, absolute_import
# from __future__ import unicode_literals is not compatible with numpy.dtype both py2 py3

from .baserawio import (BaseRawIO, _signal_channel_dtype, _unit_channel_dtype,
                        _event_channel_dtype)

import numpy as np
from collections import OrderedDict
from distutils.version import LooseVersion as V


class IntanRawIO(BaseRawIO):
    """

    """
    extensions = ['rhd', 'rhs']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)

        self.filename = filename

    def _source_name(self):
        return self.filename

    def _parse_header(self):

        if self.filename.endswith('.rhs'):
            self._global_info, self._ordered_channels, data_dtype,\
                header_size, self._block_size = read_rhs(self.filename)
        elif self.filename.endswith('.rhd'):
            self._global_info, self._ordered_channels, data_dtype,\
                header_size, self._block_size = read_rhd(self.filename)

        # memmap raw data with the complicated structured dtype
        self._raw_data = np.memmap(self.filename, dtype=data_dtype, mode='r', offset=header_size)

        # check timestamp continuity
        timestamp = self._raw_data['timestamp'].flatten()
        assert np.all(np.diff(timestamp) == 1), 'timestamp have gaps'

        # signals
        sig_channels = []
        for c, chan_info in enumerate(self._ordered_channels):
            name = chan_info['native_channel_name']
            chan_id = c  # the chan_id have no meaning in intan
            if chan_info['signal_type'] == 20:
                # exception for temperature
                sig_dtype = 'int16'
            else:
                sig_dtype = 'uint16'
            group_id = 0
            sig_channels.append((name, chan_id, chan_info['sampling_rate'],
                                sig_dtype, chan_info['units'], chan_info['gain'],
                                chan_info['offset'], chan_info['signal_type']))
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        self._max_sampling_rate = np.max(sig_channels['sampling_rate'])
        self._max_sigs_length = self._raw_data.size * self._block_size

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        unit_channels = []
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)

        # fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._max_sigs_length / self._max_sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        assert channel_indexes is not None, 'channel_indexes cannot be None, several signal size'
        assert np.unique(self.header['signal_channels'][channel_indexes]['group_id']).size == 1
        channel_names = self.header['signal_channels'][channel_indexes]['name']
        chan_name = channel_names[0]
        size = self._raw_data[chan_name].size
        return size

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._get_signal_size(block_index, seg_index, channel_indexes)

        if channel_indexes is None:
            channel_indexes = slice(None)
        channel_names = self.header['signal_channels'][channel_indexes]['name']

        shape = self._raw_data[channel_names[0]].shape

        # some channel (temperature) have 1D field so shape 1D
        # because 1 sample per block
        if len(shape) == 2:
            # this is the general case with 2D
            block_size = shape[1]
            block_start = i_start // block_size
            block_stop = i_stop // block_size + 1

            sl0 = i_start % block_size
            sl1 = sl0 + (i_stop - i_start)

        sigs_chunk = np.zeros((i_stop - i_start, len(channel_names)), dtype='uint16')
        for i, chan_name in enumerate(channel_names):
            data_chan = self._raw_data[chan_name]
            if len(shape) == 1:
                sigs_chunk[:, i] = data_chan[i_start:i_stop]
            else:
                sigs_chunk[:, i] = data_chan[block_start:block_stop].flatten()[sl0:sl1]

        return sigs_chunk


def read_qstring(f):
    length = np.fromfile(f, dtype='uint32', count=1)[0]
    if length == 0xFFFFFFFF or length == 0:
        return ''
    txt = f.read(length).decode('utf-16')
    return txt


def read_variable_header(f, header):
    info = {}
    for field_name, field_type in header:
        if field_type == 'QString':
            field_value = read_qstring(f)
        else:
            field_value = np.fromfile(f, dtype=field_type, count=1)[0]
        info[field_name] = field_value
    return info


###############
# RHS ZONE

rhs_global_header = [
    ('magic_number', 'uint32'),  # 0xD69127AC

    ('major_version', 'int16'),
    ('minor_version', 'int16'),

    ('sampling_rate', 'float32'),

    ('dsp_enabled', 'int16'),

    ('actual_dsp_cutoff_frequency', 'float32'),
    ('actual_lower_bandwidth', 'float32'),
    ('actual_lower_settle_bandwidth', 'float32'),
    ('actual_upper_bandwidth', 'float32'),
    ('desired_dsp_cutoff_frequency', 'float32'),
    ('desired_lower_bandwidth', 'float32'),
    ('desired_lower_settle_bandwidth', 'float32'),
    ('desired_upper_bandwidth', 'float32'),

    ('notch_filter_mode', 'int16'),

    ('desired_impedance_test_frequency', 'float32'),
    ('actual_impedance_test_frequency', 'float32'),

    ('amp_settle_mode', 'int16'),
    ('charge_recovery_mode', 'int16'),

    ('stim_step_size', 'float32'),
    ('recovery_current_limit', 'float32'),
    ('recovery_target_voltage', 'float32'),

    ('note1', 'QString'),
    ('note2', 'QString'),
    ('note3', 'QString'),

    ('dc_amplifier_data_saved', 'int16'),

    ('board_mode', 'int16'),

    ('ref_channel_name', 'QString'),

    ('nb_signal_group', 'int16'),
]

rhs_signal_group_header = [
    ('signal_group_name', 'QString'),
    ('signal_group_prefix', 'QString'),
    ('signal_group_enabled', 'int16'),
    ('channel_num', 'int16'),
    ('amplified_channel_num', 'int16'),
]

rhs_signal_channel_header = [
    ('native_channel_name', 'QString'),
    ('custom_channel_name', 'QString'),
    ('native_order', 'int16'),
    ('custom_order', 'int16'),
    ('signal_type', 'int16'),
    ('channel_enabled', 'int16'),
    ('chip_channel_num', 'int16'),
    ('command_stream', 'int16'),
    ('board_stream_num', 'int16'),
    ('spike_scope_trigger_mode', 'int16'),
    ('spike_scope_voltage_thresh', 'int16'),
    ('spike_scope_digital_trigger_channel', 'int16'),
    ('spike_scope_digital_edge_polarity', 'int16'),
    ('electrode_impedance_magnitude', 'float32'),
    ('electrode_impedance_phase', 'float32'),
]


def read_rhs(filename):
    BLOCK_SIZE = 128  # sample per block

    with open(filename, mode='rb') as f:
        global_info = read_variable_header(f, rhs_global_header)

        channels_by_type = {k: [] for k in [0, 3, 4, 5, 6]}
        for g in range(global_info['nb_signal_group']):
            group_info = read_variable_header(f, rhs_signal_group_header)

            if bool(group_info['signal_group_enabled']):
                for c in range(group_info['channel_num']):
                    chan_info = read_variable_header(f, rhs_signal_channel_header)
                    assert chan_info['signal_type'] not in (1, 2)
                    if bool(chan_info['channel_enabled']):
                        channels_by_type[chan_info['signal_type']].append(chan_info)

        header_size = f.tell()

    sr = global_info['sampling_rate']

    # construct dtype by re-ordering channels by types
    ordered_channels = []
    data_dtype = [('timestamp', 'int32', BLOCK_SIZE)]

    # 0: RHS2000 amplifier channel.
    for chan_info in channels_by_type[0]:
        name = chan_info['native_channel_name']
        chan_info['sampling_rate'] = sr
        chan_info['units'] = 'uV'
        chan_info['gain'] = 0.195
        chan_info['offset'] = -32768 * 0.195
        ordered_channels.append(chan_info)
        data_dtype += [(name, 'uint16', BLOCK_SIZE)]

    if bool(global_info['dc_amplifier_data_saved']):
        for chan_info in channels_by_type[0]:
            name = chan_info['native_channel_name']
            chan_info_dc = dict(chan_info)
            chan_info_dc['native_channel_name'] = name + '_DC'
            chan_info_dc['sampling_rate'] = sr
            chan_info_dc['units'] = 'mV'
            chan_info_dc['gain'] = 19.23
            chan_info_dc['offset'] = -512 * 19.23
            chan_info_dc['signal_type'] = 10  # put it in another group
            ordered_channels.append(chan_info_dc)
            data_dtype += [(name + '_DC', 'uint16', BLOCK_SIZE)]

    for chan_info in channels_by_type[0]:
        name = chan_info['native_channel_name']
        chan_info_stim = dict(chan_info)
        chan_info_stim['native_channel_name'] = name + '_STIM'
        chan_info_stim['sampling_rate'] = sr
        # stim channel are coplicated because they are coded
        # with bits, they do not fit the gain/offset rawio strategy
        chan_info_stim['units'] = ''
        chan_info_stim['gain'] = 1.
        chan_info_stim['offset'] = 0.
        chan_info_stim['signal_type'] = 11  # put it in another group
        ordered_channels.append(chan_info_stim)
        data_dtype += [(name + '_STIM', 'uint16', BLOCK_SIZE)]

    # 3: Analog input channel.
    # 4: Analog output channel.
    for sig_type in [3, 4, ]:
        for chan_info in channels_by_type[sig_type]:
            name = chan_info['native_channel_name']
            chan_info['sampling_rate'] = sr
            chan_info['units'] = 'V'
            chan_info['gain'] = 0.0003125
            chan_info['offset'] = -32768 * 0.0003125
            ordered_channels.append(chan_info)
            data_dtype += [(name, 'uint16', BLOCK_SIZE)]

    # 5: Digital input channel.
    # 6: Digital output channel.
    for sig_type in [5, 6]:
        # at the moment theses channel are not in sig channel list
        # but they are in the raw memamp
        if len(channels_by_type[sig_type]) > 0:
            name = {5: 'DIGITAL-IN', 6: 'DIGITAL-OUT'}[sig_type]
            data_dtype += [(name, 'uint16', BLOCK_SIZE)]

    return global_info, ordered_channels, data_dtype, header_size, BLOCK_SIZE


###############
# RHD ZONE

rhd_global_header_base = [
    ('magic_number', 'uint32'),  # 0xC6912702
    ('major_version', 'int16'),
    ('minor_version', 'int16'),
]


rhd_global_header_part1 = [
    ('sampling_rate', 'float32'),

    ('dsp_enabled', 'int16'),

    ('actual_dsp_cutoff_frequency', 'float32'),
    ('actual_lower_bandwidth', 'float32'),
    ('actual_upper_bandwidth', 'float32'),
    ('desired_dsp_cutoff_frequency', 'float32'),
    ('desired_lower_bandwidth', 'float32'),
    ('desired_upper_bandwidth', 'float32'),

    ('notch_filter_mode', 'int16'),

    ('desired_impedance_test_frequency', 'float32'),
    ('actual_impedance_test_frequency', 'float32'),

    ('note1', 'QString'),
    ('note2', 'QString'),
    ('note3', 'QString'),

]

rhd_global_header_v11 = [
    ('num_temp_sensor_channels', 'int16'),
]

rhd_global_header_v13 = [
    ('eval_board_mode', 'int16'),
]

rhd_global_header_v20 = [
    ('reference_channel', 'QString'),
]

rhd_global_header_final = [
    ('nb_signal_group', 'int16'),
]

rhd_signal_group_header = [
    ('signal_group_name', 'QString'),
    ('signal_group_prefix', 'QString'),
    ('signal_group_enabled', 'int16'),
    ('channel_num', 'int16'),
    ('amplified_channel_num', 'int16'),
]

rhd_signal_channel_header = [
    ('native_channel_name', 'QString'),
    ('custom_channel_name', 'QString'),
    ('native_order', 'int16'),
    ('custom_order', 'int16'),
    ('signal_type', 'int16'),
    ('channel_enabled', 'int16'),
    ('chip_channel_num', 'int16'),
    ('board_stream_num', 'int16'),
    ('spike_scope_trigger_mode', 'int16'),
    ('spike_scope_voltage_thresh', 'int16'),
    ('spike_scope_digital_trigger_channel', 'int16'),
    ('spike_scope_digital_edge_polarity', 'int16'),
    ('electrode_impedance_magnitude', 'float32'),
    ('electrode_impedance_phase', 'float32'),
]


def read_rhd(filename):
    with open(filename, mode='rb') as f:

        global_info = read_variable_header(f, rhd_global_header_base)

        version = V('{major_version}.{minor_version}'.format(**global_info))

        # the header size depend on the version :-(
        header = list(rhd_global_header_part1)  # make a copy

        if version >= '1.1':
            header = header + rhd_global_header_v11
        else:
            global_info['num_temp_sensor_channels'] = 0

        if version >= '1.3':
            header = header + rhd_global_header_v13
        else:
            global_info['eval_board_mode'] = 0

        if version >= '2.0':
            header = header + rhd_global_header_v20
        else:
            global_info['reference_channel'] = ''

        header = header + rhd_global_header_final

        global_info.update(read_variable_header(f, header))

        # read channel group and channel header
        channels_by_type = {k: [] for k in [0, 1, 2, 3, 4, 5]}
        for g in range(global_info['nb_signal_group']):
            group_info = read_variable_header(f, rhd_signal_group_header)

            if bool(group_info['signal_group_enabled']):
                for c in range(group_info['channel_num']):
                    chan_info = read_variable_header(f, rhd_signal_channel_header)
                    if bool(chan_info['channel_enabled']):
                        channels_by_type[chan_info['signal_type']].append(chan_info)

        header_size = f.tell()

    sr = global_info['sampling_rate']

    # construct the data block dtype and reorder channels
    if version >= '2.0':
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 60  # 256 channels

    ordered_channels = []

    if version >= '1.2':
        data_dtype = [('timestamp', 'int32', BLOCK_SIZE)]
    else:
        data_dtype = [('timestamp', 'uint32', BLOCK_SIZE)]

    # 0: RHD2000 amplifier channel
    for chan_info in channels_by_type[0]:
        name = chan_info['native_channel_name']
        chan_info['sampling_rate'] = sr
        chan_info['units'] = 'uV'
        chan_info['gain'] = 0.195
        chan_info['offset'] = -32768 * 0.195
        ordered_channels.append(chan_info)
        data_dtype += [(name, 'uint16', BLOCK_SIZE)]

    # 1: RHD2000 auxiliary input channel
    for chan_info in channels_by_type[1]:
        name = chan_info['native_channel_name']
        chan_info['sampling_rate'] = sr / 4.
        chan_info['units'] = 'V'
        chan_info['gain'] = 0.0000374
        chan_info['offset'] = 0.
        ordered_channels.append(chan_info)
        data_dtype += [(name, 'uint16', BLOCK_SIZE // 4)]

    # 2: RHD2000 supply voltage channel
    for chan_info in channels_by_type[2]:
        name = chan_info['native_channel_name']
        chan_info['sampling_rate'] = sr / BLOCK_SIZE
        chan_info['units'] = 'V'
        chan_info['gain'] = 0.0000748
        chan_info['offset'] = 0.
        ordered_channels.append(chan_info)
        data_dtype += [(name, 'uint16')]

    # temperature is not an official channel in the header
    for i in range(global_info['num_temp_sensor_channels']):
        name = 'temperature_{}'.format(i)
        chan_info = {'native_channel_name': name, 'signal_type': 20}
        chan_info['sampling_rate'] = sr / BLOCK_SIZE
        chan_info['units'] = 'Celsius'
        chan_info['gain'] = 0.001
        chan_info['offset'] = 0.
        ordered_channels.append(chan_info)
        data_dtype += [(name, 'int16')]

    # 3: USB board ADC input channel
    for chan_info in channels_by_type[3]:
        name = chan_info['native_channel_name']
        chan_info['sampling_rate'] = sr
        chan_info['units'] = 'V'
        if global_info['eval_board_mode'] == 0:
            chan_info['gain'] = 0.000050354
            chan_info['offset'] = 0.
        elif global_info['eval_board_mode'] == 1:
            chan_info['gain'] = 0.00015259
            chan_info['offset'] = -32768 * 0.00015259
        elif global_info['eval_board_mode'] == 13:
            chan_info['gain'] = 0.0003125
            chan_info['offset'] = -32768 * 0.0003125
        ordered_channels.append(chan_info)
        data_dtype += [(name, 'uint16', BLOCK_SIZE)]

    # 4: USB board digital input channel
    # 5: USB board digital output channel
    for sig_type in [4, 5]:
        # at the moment theses channel are not in sig channel list
        # but they are in the raw memamp
        if len(channels_by_type[sig_type]) > 0:
            name = {4: 'DIGITAL-IN', 5: 'DIGITAL-OUT'}[sig_type]
            data_dtype += [(name, 'uint16', BLOCK_SIZE)]

    return global_info, ordered_channels, data_dtype, header_size, BLOCK_SIZE
