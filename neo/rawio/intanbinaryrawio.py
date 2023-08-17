"""
Support for Intan's binary file mode which is turned on by setting the save
option in Intan to 'One File Per Signal Type' or 'One File Per Channel'. 
The reader for this format is for rhd only since it uses the rhd header. 
Need to confirm if rhs can also use this save format.

RHD supported 1.x, 2.x, and 3.0, 3.1, 3.2
See:
  * http://intantech.com/files/Intan_RHD2000_data_file_formats.pdf
  * http://intantech.com/files/Intan_RHS2000_data_file_formats.pdf

Author: Zach McKenzie, based on Samuel Garcia's IntanRawIO
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype, _common_sig_characteristics)

from .intanrawio import (read_variable_header, rhd_global_header_base, rhd_global_header_part1, 
                         rhd_global_header_v11, rhd_global_header_v13, rhd_global_header_v20, rhd_global_header_final,
                         rhd_signal_group_header,  rhd_signal_channel_header, stream_type_to_name)
from pathlib import Path

import numpy as np
from collections import OrderedDict
from packaging.version import Version as V



class IntanBinaryRawIO(BaseRawIO):
    """
    Class for processing Intan Data when saved in binary format. Requires an
    `info.rhd` as well one file per signal stream or one file per channel

    Parameters
    ----------
    dirname: str, Path
        The root directory containing the info.rhd file

    """
    extensions = ['rhd', 'dat']
    rawmode = 'one-dir'

    def __init__(self, dirname=''):
        BaseRawIO.__init__(self)
        self.dirname = dirname

    def _source_name(self):
        return self.dirname

    def _parse_header(self):

        dir_path = Path(self.dirname)
        assert dir_path.is_dir(), 'IntanBinaryRawIO requires the root directory containing info.rhd'
        
        header_file = dir_path / 'info.rhd'

        for file in possible_raw_files:
            if (dir_path / file).is_file():
                stream_mode = True
                break
            else:
                stream_mode = False

        self.stream_mode = stream_mode

        if stream_mode:
            raw_file_dict = create_raw_file_stream(dir_path)
        else:
            raw_file_dict = create_raw_file_channel(dir_path)

        self._global_info, self._ordered_channels, data_dtype, self._block_size = read_rhd(header_file)

        self._raw_data ={}
        for stream_index, sub_datatype in data_dtype.items():
            if stream_mode:
                self._raw_data[stream_index] = np.memmap(raw_file_dict[stream_index], dtype=sub_datatype, mode='r')
            else:
                self._raw_data[stream_index] = []
                for channel_index, datatype in enumerate(sub_datatype):
                    self._raw_data[stream_index].append(np.memmap(raw_file_dict[stream_index][channel_index], dtype=[datatype], mode='r'))

        # check timestamp continuity
        if stream_mode:
            timestamp = self._raw_data[6]['timestamp'].flatten()
        else:
            timestamp = self._raw_data[6][0]['timestamp'].flatten()
        assert np.all(np.diff(timestamp) == 1), 'timestamp have gaps'

        # signals
        signal_channels = []
        for c, chan_info in enumerate(self._ordered_channels):
            name = chan_info['native_channel_name']
            chan_id = str(c)  # the chan_id have no meaning in intan
            if chan_info['signal_type'] == 20:
                # exception for temperature
                sig_dtype = 'int16'
            else:
                sig_dtype = 'uint16'
            stream_id = str(chan_info['signal_type'])
            signal_channels.append((name, chan_id, chan_info['sampling_rate'],
                                sig_dtype, chan_info['units'], chan_info['gain'],
                                chan_info['offset'], stream_id))
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        stream_ids = np.unique(signal_channels['stream_id'])
        signal_streams = np.zeros(stream_ids.size, dtype=_signal_stream_dtype)
        signal_streams['id'] = stream_ids
        for stream_index, stream_id in enumerate(stream_ids):
            signal_streams['name'][stream_index] = stream_type_to_name.get(int(stream_id), '')

        self._max_sampling_rate = np.max(signal_channels['sampling_rate'])

        if stream_mode:
            self._max_sigs_length = max([raw_data.size * self._block_size for raw_data in self._raw_data.values()])
        else:
            self._max_sigs_length = max([len(raw_data)* raw_data[0].size * self._block_size for raw_data in self._raw_data.values()])

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()


    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._max_sigs_length / self._max_sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_id = self.header['signal_streams'][stream_index]['id']
        mask = self.header['signal_channels']['stream_id'] == stream_id
        signal_channels = self.header['signal_channels'][mask]
        channel_names = signal_channels['name']
        chan_name0 = channel_names[0]
        if self.stream_mode:
            size = self._raw_data[stream_index][chan_name0].size
        else:
            size = self._raw_data[stream_index][0][chan_name0].size
        return size

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):

        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._get_signal_size(block_index, seg_index, stream_index)

        stream_id = self.header['signal_streams'][stream_index]['id']
        mask = self.header['signal_channels']['stream_id'] == stream_id
        signal_channels = self.header['signal_channels'][mask]
        if channel_indexes is None:
            channel_indexes = slice(None)
        channel_names = signal_channels['name'][channel_indexes]
        
        if self.stream_mode:
            shape = self._raw_data[stream_index][channel_names[0]].shape
        else:
            shape = self._raw_data[stream_index][0][channel_names[0]].shape

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
            if self.stream_mode:
                data_chan = self._raw_data[stream_index][chan_name]
            else:
                data_chan = self._raw_data[stream_index][i][chan_name]
            if len(shape) == 1:
                sigs_chunk[:, i] = data_chan[i_start:i_stop]
            else:
                sigs_chunk[:, i] = data_chan[block_start:block_stop].flatten()[sl0:sl1]

        return sigs_chunk

############
# RHD Zone for Binary Files

# For One File Per Signal
possible_raw_files = ['amplifier.dat', 
                      'auxiliary.dat', 
                      'supply.dat',
                      'analogin.dat', 
                      'digitalin.dat', 
                      'digitalout.dat',]

# For One File Per Channel
possible_raw_file_prefixes = ['amp', 'aux', 'vdd', 'board-ANALOG', 'board-DIGITAL-IN', 'board-DIGITAL-OUT']

def create_raw_file_stream(dirname):
    """Function for One File Per Signal Type"""
    raw_file_dict = {}
    for raw_index, raw_file in enumerate(possible_raw_files):
        if Path(dirname / raw_file).is_file():
            raw_file_dict[raw_index] = Path(dirname /raw_file)
    raw_file_dict[6] = Path(dirname / 'time.dat')

    return raw_file_dict


def create_raw_file_channel(dirname):
    """Utility function for One File Per Channel"""
    file_names = dirname.glob('**/*.dat')
    files = [file for file in file_names if file.is_file()]
    raw_file_dict = {}
    for raw_index, prefix in enumerate(possible_raw_file_prefixes):
        raw_file_dict[raw_index]= [file for file in files if prefix in file.name]
    raw_file_dict[6] = [Path(dirname / 'time.dat')]

    return raw_file_dict

    
def read_rhd(filename):
    with open(filename, mode='rb') as f:

        global_info = read_variable_header(f, rhd_global_header_base)
        version = V('{major_version}.{minor_version}'.format(**global_info))

        # the header size depends on the version :-(
        header = list(rhd_global_header_part1)  # make a copy

        if version >= V('1.1'):
            header = header + rhd_global_header_v11
        else:
            global_info['num_temp_sensor_channels'] = 0

        if version >= V('1.3'):
            header = header + rhd_global_header_v13
        else:
            global_info['eval_board_mode'] = 0

        if version >= V('2.0'):
            header = header + rhd_global_header_v20
        else:
            global_info['reference_channel'] = ''

        header = header + rhd_global_header_final

        global_info.update(read_variable_header(f, header))

        # read channel group and channel header
        channels_by_type = {k: [] for k in [0, 1, 2, 3, 4, 5]}
        data_dtype = {k: [] for k in range(7)} # 5 channels + 6 is for time stamps
        for g in range(global_info['nb_signal_group']):
            group_info = read_variable_header(f, rhd_signal_group_header)

            if bool(group_info['signal_group_enabled']):
                for c in range(group_info['channel_num']):
                    chan_info = read_variable_header(f, rhd_signal_channel_header)
                    if bool(chan_info['channel_enabled']):
                        channels_by_type[chan_info['signal_type']].append(chan_info)


    sr = global_info['sampling_rate']

    # construct the data block dtype and reorder channels
    if version >= V('2.0'):
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 60  # 256 channels

    ordered_channels = []
    
    # 6: Timestamp stored in time.dat
    if version >= V('1.2'):
        data_dtype[6] = [('timestamp', 'int32', BLOCK_SIZE)]
    else:
        data_dtype[6] = [('timestamp','uint32', BLOCK_SIZE)]

    # 0: RHD2000 amplifier channel stored in amplifier.dat/amp-*
    for chan_info in channels_by_type[0]:
        name = chan_info['native_channel_name']
        chan_info['sampling_rate'] = sr
        chan_info['units'] = 'uV'
        chan_info['gain'] = 0.195
        chan_info['offset'] = -32768 * 0.195
        ordered_channels.append(chan_info)
        data_dtype[0] += [(name, 'uint16', BLOCK_SIZE)]

    # 1: RHD2000 auxiliary input channel stored in auxiliary.dat/aux-*
    for chan_info in channels_by_type[1]:
        name = chan_info['native_channel_name']
        chan_info['sampling_rate'] = sr / 4.
        chan_info['units'] = 'V'
        chan_info['gain'] = 0.0000374
        chan_info['offset'] = 0.
        ordered_channels.append(chan_info)
        data_dtype[1] += [(name,'uint16', BLOCK_SIZE // 4)]

    # 2: RHD2000 supply voltage channel stored in supply.dat/vdd-*
    for chan_info in channels_by_type[2]:
        name = chan_info['native_channel_name']
        chan_info['sampling_rate'] = sr / BLOCK_SIZE
        chan_info['units'] = 'V'
        chan_info['gain'] = 0.0000748
        chan_info['offset'] = 0.
        ordered_channels.append(chan_info)
        data_dtype[2] += [(name, 'uint16',)]

    # temperature is not an official channel in the header
    #for i in range(global_info['num_temp_sensor_channels']):
    #    name = 'temperature_{}'.format(i)
    #    chan_info = {'native_channel_name': name, 'signal_type': 20}
    #    chan_info['sampling_rate'] = sr / BLOCK_SIZE
    #    chan_info['units'] = 'Celsius'
    #    chan_info['gain'] = 0.001
    #    chan_info['offset'] = 0.
    #    ordered_channels.append(chan_info)
    #    data_dtype += [(name,'int16',)]

    # 3: USB board ADC input channel stored in analogin.dat/board-ANALOG-*
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
        data_dtype[3]+= [(name,'uint16', BLOCK_SIZE)]

    # 4: USB board digital input channel stored in digitalin.dat/board-DIGITAL-IN-*
    # 5: USB board digital output channel stored in digitalout.dat/board-DIGITAL-OUT-*
    for sig_type in [4, 5]:
        # at the moment theses channel are not in sig channel list
        # but they are in the raw memamp
        if len(channels_by_type[sig_type]) > 0:
            name = {4: 'DIGITAL-IN', 5: 'DIGITAL-OUT'}[sig_type]
            data_dtype[sig_type] += [(name,'uint16', BLOCK_SIZE)]
    
    if bool(global_info['notch_filter_mode']) and version >= V('3.0'):
        global_info['notch_filter_applied'] = True
    else:
        global_info['notch_filter_applied'] = False
    
    return global_info, ordered_channels, data_dtype, BLOCK_SIZE
