# -*- coding: utf-8 -*-
"""

Support for intan tech rhd  and rhs files.

This 2 formats are more or less the same but:
  * some variance in headers.
  * rhs amplifier is more complexe because the optional DC channel

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


BLOCK_SIZE = 128 #  sample per block


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
            self._global_info, self._channels_info, data_dtype, header_size = read_rhs(self.filename)
            #  self._dc_amplifier_data_saved = bool(self._global_info['dc_amplifier_data_saved'])
        elif self.filename.endswith('.rhd'):
            self._global_info, self._channels_info, data_dtype, header_size = read_rhd(self.filename)
            #  self._dc_amplifier_data_saved = False
        
        self._sampling_rate = self._global_info['sampling_rate']
        
        print(len(data_dtype))
        self._raw_data = np.memmap(self.filename, dtype=data_dtype, mode='r', offset=header_size)
        self._sigs_length = self._raw_data.size * BLOCK_SIZE
        
        # TODO check timestamp continuity
        #~ timestamp = self._raw_data['timestamp'].flatten()
        #~ assert np.all(np.diff(timestamp)==1)
        
        # signals
        sig_channels = []
        for c, chan_info in enumerate(self._channels_info):
            name = chan_info['native_channel_name']
            chan_id = c
            units = 'uV'
            offset = 0. # TODO
            gain = 1. # TODO
            sig_dtype = 'uint16'
            group_id = 0
            sig_channels.append((name, chan_id, self._sampling_rate,
                                sig_dtype, units, gain, offset, group_id))
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

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
        t_stop = self._sigs_length / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        return self._sigs_length

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._sigs_length

        block_start = i_start // BLOCK_SIZE
        block_stop = i_stop // BLOCK_SIZE + 1
        sl0 = i_start % BLOCK_SIZE
        sl1 = sl0 + (i_stop - i_start)

        if channel_indexes is None:
            channel_indexes = slice(None)
        channel_names = self.header['signal_channels'][channel_indexes]['name']

        sigs_chunk = np.zeros((i_stop - i_start, len(channel_names)), dtype='uint16')
        for i, chan_name in enumerate(channel_names):
            data = self._raw_data[chan_name]
            sigs_chunk[:, i] = data[block_start:block_stop].flatten()[sl0:sl1]

        return sigs_chunk





def read_qstring(f):
    length = np.fromfile(f, dtype='uint32', count=1)[0]
    if length == 0xFFFFFFFF or length == 0:
        return ''
    txt = f.read(length).decode('utf-16')
    return txt

rhs_global_header =[
    ('magic_number', 'uint32'),  # 0xD69127AC for rhs   0xC6912702 for rdh
    
    ('major_version', 'int16'),
    ('minor_version', 'int16'),
    
    ('sampling_rate', 'float32'),
    
    ('dsp_enabled', 'int16'),
    
    ('actual_dsp_cutoff_frequency', 'float32'),
    ('actual_lower_bandwidth', 'float32'),
    ('actual_lower_settle_bandwidth', 'float32'),   #######
    ('actual_upper_bandwidth', 'float32'),
    ('desired_dsp_cutoff_frequency', 'float32'),
    ('desired_lower_bandwidth', 'float32'),
    ('desired_lower_settle_bandwidth', 'float32'),   #####
    ('desired_upper_bandwidth', 'float32'),
    
    ('notch_filter_mode', 'int16'), # 0 :no filter 1: 50Hz 2 : 60Hz
    
    ('desired_impedance_test_frequency', 'float32'),
    ('actual_impedance_test_frequency', 'float32'),
    
    ('amp_settle_mode', 'int16'),  ####
    ('charge_recovery_mode', 'int16'), ####
    
    ('stim_step_size', 'float32'),  ####
    ('recovery_current_limit', 'float32'),  ####
    ('recovery_target_voltage', 'float32'), ####

    ('note1', 'QString'),
    ('note2', 'QString'),
    ('note3', 'QString'),
    
    ('dc_amplifier_data_saved', 'int16'),  ###### nb_temp_sensor
    
    
    
    ('board_mode', 'int16'),
    
    ('ref_channel_name', 'QString'),
    
    ('nb_signal_group', 'int16'),
    
]

signal_group_header = [
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
    ('command_stream', 'int16'),  #######
    ('board_stream_num', 'int16'),
    ('spike_scope_trigger_mode', 'int16'),
    ('spike_scope_voltage_thresh', 'int16'),
    ('spike_scope_digital_trigger_channel', 'int16'),
    ('spike_scope_digital_edge_polarity', 'int16'),
    ('electrode_impedance_magnitude', 'float32'),
    ('electrode_impedance_phase', 'float32'),
    
]


def read_variable_header(f, header):
    info = {}
    for field_name, field_type in header:
        
        if field_type == 'QString':
            field_value = read_qstring(f)
            #~ print(field_name, field_type,  len(field_value), field_value)
        else:
            field_value = np.fromfile(f, dtype=field_type, count=1)[0]
            #~ print(field_name, field_type,  field_value)
        
        #~ print(field_name, ':',  field_value)
        info[field_name] = field_value
    
    return info



def read_rhd(filename):
    return
    


# signal_type
# 0: RHS2000 amplifier channel.
# 3: Analog input channel.
# 4: Analog output channel.
# 5: Digital input channel.
# 6: Digital output channel.



def read_rhs(filename):
    with open(filename, mode='rb') as f:
        global_info = read_variable_header(f, rhs_global_header)
        
        print(global_info['dc_amplifier_data_saved'], bool(global_info['dc_amplifier_data_saved']))
        channels_info = []
        data_dtype = [('timestamp', 'int32', BLOCK_SIZE)]
        for g in range(global_info['nb_signal_group']):
            #~ print('goup', g)
            group_info = read_variable_header(f, signal_group_header)
            print(group_info)
            if bool(group_info['signal_group_enabled']):
                for c in range(group_info['channel_num']):
                    #~ print('  c', c)
                    chan_info = read_variable_header(f, rhs_signal_channel_header)
                    
                    if bool(chan_info['channel_enabled']):
                        channels_info.append(chan_info)
                        print('goup', g, 'channel', c, chan_info['native_channel_name'])
                        name = chan_info['native_channel_name']
                        data_dtype +=[(name, 'int32', BLOCK_SIZE)]
                        
                        if chan_info['signal_type'] == 0:
                            if bool(global_info['dc_amplifier_data_saved']):
                                chan_info_dc = dict(chan_info)
                                chan_info_dc['native_channel_name'] = name+'_DC'
                                channels_info.append(chan_info_dc)
                                data_dtype +=[(name+'_DC', 'int32', BLOCK_SIZE)]
                                
                            chan_info_stim = dict(chan_info)
                            chan_info_stim['native_channel_name'] = name+'_STIM'
                            channels_info.append(chan_info_stim)
                            data_dtype +=[(name+'_STIM', 'int32', BLOCK_SIZE)]
                        
        header_size = f.tell()
    
    return global_info, channels_info, data_dtype, header_size


