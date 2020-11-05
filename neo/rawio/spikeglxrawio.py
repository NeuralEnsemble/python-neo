"""
Class for reading data from a SpikeGLX system  (NI-DAQ for neuropixel probe)
See https://billkarsh.github.io/SpikeGLX/


Here an adaptation of the spikeglx tools into the neo rawio API.

Note that each pair of ".bin"/."meta" represent a group of analog signal that share the same sampling rate.

Contrary to other implementations this read the entire folder and subfolder so:
  * It deal with severals segment taken from the naming "_gt0", "_gt1", "_gt2", ...
  * It deal with all signal "imec0", "imec1" for neuropixel probes and anlso external signal like"nidq"
    This the "device"
  * For imec device both "ap" and "lf" are extracted so one device have several "stream"


# TODO:
  * contact SpkeGLX developer to see how to deal with absolut t_start when several segment
  * contact SpkeGLX developer to understand the last channel SY0 function


See:
https://billkarsh.github.io/SpikeGLX/#offline-analysis-tools

And also:
https://github.com/SpikeInterface/spikeextractors/blob/master/spikeextractors/extractors/spikeglxrecordingextractor/spikeglxrecordingextractor.py


Author : Samuel Garcia
"""



from .baserawio import BaseRawIO, _signal_channel_dtype, _unit_channel_dtype, _event_channel_dtype

from pathlib import Path
import os
import re

import numpy as np


class SpikeGLXRawIO(BaseRawIO):
    """
    Class for reading data from a SpikeGLX system
    https://billkarsh.github.io/SpikeGLX/
    """
    extensions = []
    rawmode = 'one-dir'

    def __init__(self, dirname=''):
        BaseRawIO.__init__(self)
        self.dirname = dirname

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        
        self.signals_info_list = scan_files(self.dirname)
        print(len(self.signals_info_list))
        print([info['stream_name'] for info in self.signals_info_list])
        
        stream_names = list(np.unique([info['stream_name'] for info in self.signals_info_list]))
        # TODO order stream_names depending the sampling_rate higher sampling_rate should be first
        nb_segment = np.unique([info['seg_index'] for info in self.signals_info_list]).size
        
        self._memmaps = {}
        self.signals_info_dict = {}
        self._global_channel_to_stream = {}
        self._global_channel_to_local_channel = []
        sig_channels = []
        global_chan = 0
        for info in self.signals_info_list:
            # key is (seg_index, stream_name)
            key = (info['seg_index'], info['stream_name'])
            self.signals_info_dict[key] = info
            
            # create memmap
            data = np.memmap(info['bin_file'], dtype='int16', mode='r',
                        shape=(info['sample_length'], info['num_chan']), offset=0, order='C')
            self._memmaps[key] = data
            
            group_id = stream_names.index(info['stream_name'])
            
            # add channels to global list
            for local_chan in range(info['num_chan']):
                self._global_channel_to_stream[global_chan] = info['stream_name']
                self._global_channel_to_local_channel.append(local_chan)
                chan_name = info['channel_names'][local_chan]
                sig_channels.append((chan_name, global_chan, info['sampling_rate'], 'int16', info['units'], 
                                    info['channel_gains'][local_chan], info['channel_offsets'][local_chan], group_id))
                                    
                # the channel id is a global counter and so equivalent to channel_index
                # this is bad : this should be changed by an id base on a str
                global_chan += 1

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        self._global_channel_to_local_channel = np.array(self._global_channel_to_local_channel, dtype='int64')

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        unit_channels = []
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)
        
        # deal with nb_segment and t_start/t_stop per segment
        self._t_starts = { seg_index:0. for seg_index in range(nb_segment) }
        self._t_stops = { seg_index:0. for seg_index in range(nb_segment) }
        for seg_index in range(nb_segment):
            for stream_name in stream_names:
                info = self.signals_info_dict[seg_index, stream_name]
                t_stop = info['sample_length'] / info['sampling_rate']
                self._t_stops[seg_index] = max(self._t_stops[seg_index], t_stop)
        
        # fille into header dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [nb_segment]
        self.header['signal_channels'] = sig_channels
        self.header['unit_channels'] = unit_channels
        self.header['event_channels'] = event_channels

        # insert some annotation at some place
        self._generate_minimal_annotations()
        
        # TODO in annotations :
        #  *  channel location
        #  * name, stream_name, stream_kind, datatime

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        return self._t_stops[seg_index]

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        assert channel_indexes is not None
        stream_name = self._global_channel_to_stream[channel_indexes[0]]
        memmap = self._memmaps[seg_index, stream_name]
        return int(memmap.shape[0])

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, channel_indexes):
        assert channel_indexes is not None
        stream_name = self._global_channel_to_stream[channel_indexes[0]]
        memmap = self._memmaps[seg_index, stream_name]
        
        local_chans = self._global_channel_to_local_channel[channel_indexes]
        if np.all(np.diff(local_chans) == 1):
            # consecutive channel then slice this avoid a copy (because of ndarray.take(...)
            # and so keep the underlying memmap
            local_chans = slice(local_chans[0], local_chans[0]+len(local_chans))
        
        raw_signals = memmap[slice(i_start, i_stop), local_chans]
        
        return raw_signals


def scan_files(dirname):
    """
    Scan pair of bin/meta files and return information about it.
    """
    l = []
    
    for root, dirs, files in os.walk(dirname):

        for file in files:
            if not file.endswith('.meta'):
                continue
            meta_filename = Path(root) / file
            bin_filename = Path(root)  / file.replace('.meta', '.bin')
            meta = read_meta_file(meta_filename)

            num_chan = int(meta['nSavedChans'])

            # when file is Noise4Sam_g0_t0.nidq.bin or Noise4Sam_g0_t0.imec0.lf.bin
            # name is the first part "Noise4Sam_g0_t0"
            # gtX X is the seg_index here 0
            # nidq or imec0 is the device
            # lf or ap is "signal_kind"
            # stream_name = device + signal_kind
            name = file.split('.')[0]
            r = re.findall('_g(\d*)_t', name)
            seg_index =  int(r[0][0])
            device = file.split('.')[1]
            print(meta_filename)
            if 'imec' in device:
                signal_kind = file.split('.')[2]
                stream_name = device + '.' + signal_kind
                units = 'uV'
                # please note the 1e-6 in gain for this uV

                # metad['imroTbl'] contain two gain per channel  AP and LF
                # except for the last fake channel
                per_channel_gain = np.ones(num_chan, dtype='float64')
                if signal_kind == 'ap':
                    index_imroTbl = 3
                elif signal_kind == 'lf':
                    index_imroTbl = 4
                for c in range(num_chan-1):
                    # the last channel don't have gain
                    per_channel_gain[c] = 1. / float(meta['imroTbl'][c].split(' ')[index_imroTbl])
                gain_factor = float(meta['imAiRangeMax']) / 512
                channel_gains = per_channel_gain * gain_factor * 1e-6

            else:
                signal_kind = ''
                stream_name = device
                units = 'V'
                channel_gains = np.ones(num_chan)

                # there differents kind of channel with diffrents gain
                mn, ma, xa, dw = [int(e) for e in meta['snsMnMaXaDw'].split(sep=',')]
                per_channel_gain = np.ones(num_chan, dtype='float64')
                per_channel_gain[0:mn] = float(meta['niMNGain'])
                per_channel_gain[mn:mn+ma] = float(meta['niMAGain'])
                gain_factor = float(meta['niAiRangeMax']) / 32768
                channel_gains = per_channel_gain * gain_factor

            info = {}
            info['name'] = name
            info['meta'] = meta  # is it usefull?
            info['bin_file'] = str(bin_filename)
            for k in ('niSampRate', 'imSampRate'):
                if k in meta:
                    info['sampling_rate'] = float(meta[k])
            info['num_chan'] = num_chan
            
            info['sample_length'] = int(meta['fileSizeBytes']) // 2 // num_chan
            r = re.findall('_g(\d*)_t', name)
            info['seg_index'] = seg_index
            info['device'] = device
            info['signal_kind'] = signal_kind
            info['stream_name'] = stream_name
            info['units'] = units
            info['channel_names'] = [txt.split(';')[0] for txt in meta['snsChanMap']]
            #TODO
            info['channel_gains'] = channel_gains
            info['channel_offsets'] = np.zeros(info['num_chan'])

            l.append(info)

    return l


def read_meta_file(meta_file):
    """parse the meta file"""
    with open(meta_file, mode='r') as f:
        lines = f.read().splitlines()
    
    info = {}
    for line in lines:
        k, v = line.split('=')
        if k.startswith('~'):
            # replace by the list
            k = k[1:]
            v = v[1:-1].split(')(')[1:]
        info[k] = v

    return info
