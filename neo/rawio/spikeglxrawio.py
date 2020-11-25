"""
Class for reading data from a SpikeGLX system  (NI-DAQ for neuropixel probe)
See https://billkarsh.github.io/SpikeGLX/


Here an adaptation of the spikeglx tools into the neo rawio API.

Note that each pair of ".bin"/."meta" files is represented as a group of two AnalogSignals
that share the same sampling rate.

Contrary to other implementations this IO reads the entire folder and subfolder and:
  * deals with severals segment based on the `_gt0`, `_gt1`, `_gt2`, etc postfixes
  * deals with all signals "imec0", "imec1" for neuropixel probes and also
     external signal like"nidq". This is the "device"
  * For imec device both "ap" and "lf" are extracted so one device have several "streams"

Note:
  * there are several versions depending the neuropixel probe generation (`1.0`/`2.0`/`3.0`)
     Here, we assume that the `meta` file has the same structure across all generations.
     TODO: This need so be checked.
     This IO is developed based on neuropixel generation 2.0, single shank recordings.


# TODO:
  * contact SpkeGLX developer to see how to deal with absolut t_start when several segment
  * contact SpkeGLX developer to understand the last channel SY0 function
  * better handling of annotations at object level by sub group of device (need rawio change)
  * better handling of channel location


See:
https://billkarsh.github.io/SpikeGLX/#offline-analysis-tools
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

        # sort stream_name by hiher sampling rate first
        srates = {info['stream_name']: info['sampling_rate'] for info in self.signals_info_list}
        stream_names = sorted(list(srates.keys()), key=lambda e: srates[e])[::-1]

        nb_segment = np.unique([info['seg_index'] for info in self.signals_info_list]).size

        self._memmaps = {}
        self.signals_info_dict = {}
        for info in self.signals_info_list:
            # key is (seg_index, stream_name)
            key = (info['seg_index'], info['stream_name'])
            self.signals_info_dict[key] = info

            # create memmap
            data = np.memmap(info['bin_file'], dtype='int16', mode='r',
                        shape=(info['sample_length'], info['num_chan']), offset=0, order='C')
            self._memmaps[key] = data

        # create channel header
        self._global_channel_to_stream = {}
        self._global_channel_to_local_channel = []
        self._channel_location = {}
        sig_channels = []
        global_chan = 0
        signal_annotations = []
        for stream_name in stream_names:
            # take first segment
            info = self.signals_info_dict[0, stream_name]

            group_id = stream_names.index(info['stream_name'])

            # add channels to global list
            for local_chan in range(info['num_chan']):
                self._global_channel_to_stream[global_chan] = info['stream_name']
                self._global_channel_to_local_channel.append(local_chan)
                chan_name = info['channel_names'][local_chan]
                sig_channels.append((chan_name, global_chan, info['sampling_rate'], 'int16',
                                    info['units'], info['channel_gains'][local_chan],
                                    info['channel_offsets'][local_chan], group_id))

                # annotation
                ann = {}
                ann['stream'] = info['stream_name']
                signal_annotations.append(ann)

                # channel location
                if 'channel_location' in info:
                    self._channel_location[info['seg_index'], info['device']] = \
                                                                                                    info['channel_location']

                # the channel id is a global counter and so equivalent to channel_index
                # this is bad and should rather be changed to a string based id
                global_chan += 1

        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)
        self._global_channel_to_local_channel = np.array(
                                    self._global_channel_to_local_channel, dtype='int64')

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        unit_channels = []
        unit_channels = np.array(unit_channels, dtype=_unit_channel_dtype)

        # deal with nb_segment and t_start/t_stop per segment
        self._t_starts = {seg_index: 0. for seg_index in range(nb_segment)}
        self._t_stops = {seg_index: 0. for seg_index in range(nb_segment)}
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
        self._generate_minimal_annotations()
        block_ann = self.raw_annotations['blocks'][0]
        block_ann['file_origin'] = self.dirname

        for seg_index in range(nb_segment):
            seg_ann = block_ann['segments'][seg_index]
            seg_ann['file_origin'] = self.dirname
            seg_ann['name'] = "Segment {}".format(seg_index)

            for c in range(sig_channels.size):
                anasig_an = seg_ann['signals'][c]
                anasig_an.update(signal_annotations[c])

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
            local_chans = slice(local_chans[0], local_chans[0] + len(local_chans))

        raw_signals = memmap[slice(i_start, i_stop), local_chans]

        return raw_signals

    def get_channel_location(self, seg_index=0, device=None, x_pitch=21, y_pitch=20):
        if device is None:
            if len(self._channel_location) == 1:
                locations = list(self._channel_location.values())[0]
            else:
                raise ValueError('device must specified')
        else:
            locations = self._channel_location[seg_index, device]
        locations = locations * [[x_pitch, y_pitch]]
        return locations


def scan_files(dirname):
    """
    Scan for pairs of `.bin` and `.meta` files and return information about it.
    """
    info_list = []

    for root, dirs, files in os.walk(dirname):

        for file in files:
            if not file.endswith('.meta'):
                continue
            meta_filename = Path(root) / file
            bin_filename = Path(root) / file.replace('.meta', '.bin')
            meta = read_meta_file(meta_filename)

            num_chan = int(meta['nSavedChans'])

            # Example file name structure:
            # Consider the filenames: `Noise4Sam_g0_t0.nidq.bin` or `Noise4Sam_g0_t0.imec0.lf.bin`
            # The filenames consist of 3 or 4 parts separated by `.`
            # `name` corresponds to everything before first dot, here `Noise4Sam_g0_t0`
            # `seg_index` corresponds to X in `gtX`, here 0
            # `device` corresponds to the second part of the filename, here `nidq` or `imec0`
            # `signal_kind` corresponds to the optional third part of the filename, here `lf`. Valid values for `signal_kind` are `lf` and `ap`.
            # `stream_name` is the concatenation of `device.signal_kind`
            name = file.split('.')[0]
            r = re.findall(r'_g(\d*)_t', name)
            seg_index = int(r[0][0])
            device = file.split('.')[1]
            if 'imec' in device:
                signal_kind = file.split('.')[2]
                stream_name = device + '.' + signal_kind
                units = 'uV'
                # please note the 1e6 in gain for this uV

                # metad['imroTbl'] contain two gain per channel  AP and LF
                # except for the last fake channel
                per_channel_gain = np.ones(num_chan, dtype='float64')
                if signal_kind == 'ap':
                    index_imroTbl = 3
                elif signal_kind == 'lf':
                    index_imroTbl = 4
                # the last channel doesn't have a gain value
                for c in range(num_chan - 1):
                    per_channel_gain[c] = 1. / float(meta['imroTbl'][c].split(' ')[index_imroTbl])
                gain_factor = float(meta['imAiRangeMax']) / 512
                channel_gains = per_channel_gain * gain_factor * 1e6

            else:
                signal_kind = ''
                stream_name = device
                units = 'V'
                channel_gains = np.ones(num_chan)

                # there are differents kinds of channels with different gain values
                mn, ma, xa, dw = [int(e) for e in meta['snsMnMaXaDw'].split(sep=',')]
                per_channel_gain = np.ones(num_chan, dtype='float64')
                per_channel_gain[0:mn] = float(meta['niMNGain'])
                per_channel_gain[mn:mn + ma] = float(meta['niMAGain'])
                gain_factor = float(meta['niAiRangeMax']) / 32768
                channel_gains = per_channel_gain * gain_factor

            info = {}
            info['name'] = name
            info['meta'] = meta
            info['bin_file'] = str(bin_filename)
            for k in ('niSampRate', 'imSampRate'):
                if k in meta:
                    info['sampling_rate'] = float(meta[k])
            info['num_chan'] = num_chan

            info['sample_length'] = int(meta['fileSizeBytes']) // 2 // num_chan
            info['seg_index'] = seg_index
            info['device'] = device
            info['signal_kind'] = signal_kind
            info['stream_name'] = stream_name
            info['units'] = units
            info['channel_names'] = [txt.split(';')[0] for txt in meta['snsChanMap']]
            info['channel_gains'] = channel_gains
            info['channel_offsets'] = np.zeros(info['num_chan'])

            if signal_kind == 'ap':
                channel_location = []
                for e in meta['snsShankMap']:
                    x_pos = int(e.split(':')[1])
                    y_pos = int(e.split(':')[2])
                    channel_location.append([x_pos, y_pos])

                info['channel_location'] = np.array(channel_location)

            info_list.append(info)

    return info_list


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
