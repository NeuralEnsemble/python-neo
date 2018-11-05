# -*- coding: utf-8 -*-
"""

Support for intan tech rhd  and rhs files.

This 2 formats are more or less the same with some variance in headers.

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
            info = read_rhs(self.filename)
        elif self.filename.endswith('.rhd'):
            info = read_rh(self.filename)
        
        exit()
            
        
        
        # signals
        sig_channels = []
        for c in range(nb_channel):
            name = 'ch{}grp{}'.format(c, channel_group[c])
            chan_id = c
            units = 'mV'
            offset = 0.
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
        t_stop = self._raw_signals.shape[0] / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        return self._raw_signals.shape[0]

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        raw_signals = self._raw_signals[slice(i_start, i_stop), channel_indexes]
        return raw_signals




def read_qstring(f):
    a = ''
    length = np.fromfile(f, dtype='uint32', count=1)[0]
    
    print('length', length, hex(length))
    #~ exit()
    
    if length == 0xFFFFFFFF or length == '':
        return ''
    
    txt =''
    txt = f.read(length // 2)  #.decode('utf-16')
    #~ print(txt)
    #~ txt = txt.decode()
    
    #~ for ii in range(length):
        #~ print('ii', ii)
        #~ newchar = np.fromfile(f, 'u2', 1)[0]
        #~ print(newchar)
        #~ a += newchar.tostring().decode('utf-16')
    return txt


import struct
import os
import sys
def read_qstring(fid):
    """Read Qt style QString.  

    The first 32-bit unsigned number indicates the length of the string (in bytes).  
    If this number equals 0xFFFFFFFF, the string is null.

    Strings are stored as unicode.
    """

    length, = struct.unpack('<I', fid.read(4))
    print(length)
    if length == int('ffffffff', 16): return ""

    if length > (os.fstat(fid.fileno()).st_size - fid.tell() + 1) :
        print(length)
        raise Exception('Length too long.')

    # convert length from bytes to 16-bit Unicode words
    length = int(length / 2)

    data = []
    for i in range(0, length):
        c, = struct.unpack('<H', fid.read(2))
        data.append(c)

    if sys.version_info >= (3,0):
        a = ''.join([chr(c) for c in data])
    else:
        a = ''.join([unichr(c) for c in data])

    return a




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

signal_channel_header = [
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


def read_variable_header(f, header):
    info = {}
    for field_name, field_type in header:
        
        if field_type == 'QString':
            field_value = read_qstring(f)
            print(field_name, field_type,  len(field_value), field_value)
        else:
            field_value = np.fromfile(f, dtype=field_type, count=1)[0]
            print(field_name, field_type,  field_value)
            
        info[field_name] = field_value
    
    return info



def read_rhd(filename):
    return
    

def read_rhs(filename):
    with open(filename, mode='rb') as f:
        info = read_variable_header(f, rhs_global_header)
    
    
    
    
    
    
    
    