"""
Class for reading data from a SpikeGLX system  (NI-DAQ for neuropixel probe)
See https://billkarsh.github.io/SpikeGLX/


Here an adaptation of the spikeglx tools into the neo rawio API.

Note that each pair of ".bin"/."meta" files is represented as a stream of channels
that share the same sampling rate.
It will be one AnalogSignal multi channel at neo.io level.

Contrary to other implementations this IO reads the entire folder and subfolder and:
  * deals with severals segment based on the `_gt0`, `_gt1`, `_gt2`, etc postfixes
  * deals with all signals "imec0", "imec1" for neuropixel probes and also
     external signal like"nidq". This is the "device"
  * For imec device both "ap" and "lf" are extracted so one device have several "streams"

Note:
  * there are several versions depending the neuropixel probe generation (`1.x`/`2.x`/`3.x`)
    Here, we assume that the `meta` file has the same structure across all generations.
    This need so be checked.
    This IO is developed based on neuropixel generation 2.0, single shank recordings.


# Not implemented yet in this reader:
  * contact SpkeGLX developer to see how to deal with absolute t_start when several segment
  * contact SpkeGLX developer to understand the last channel SY0 function
  * better handling of annotations at object level by sub group of device (after rawio change)
  * better handling of channel location


See:
https://billkarsh.github.io/SpikeGLX/
https://billkarsh.github.io/SpikeGLX/#offline-analysis-tools
https://billkarsh.github.io/SpikeGLX/#metadata-guides
https://github.com/SpikeInterface/spikeextractors/blob/master/spikeextractors/extractors/spikeglxrecordingextractor/spikeglxrecordingextractor.py

This reader handle:

imDatPrb_type=1 (NP 1.0)
imDatPrb_type=21 (NP 2.0, single multiplexed shank)
imDatPrb_type=24 (NP 2.0, 4-shank)

Author : Samuel Garcia
Some functions are copied from Graham Findlay
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                _spike_channel_dtype, _event_channel_dtype)

from pathlib import Path
import os
import re

import numpy as np


class SpikeGLXRawIO(BaseRawIO):
    """
    Class for reading data from a SpikeGLX system

    dirname:
        The spikeglx folder containing meta/bin files
    load_sync_channel=False/True
        The last channel (SY0) of each stream is a fake channel used for synchronisation.
    """
    # file formats used by spikeglxio
    extensions = ['meta', 'bin']
    rawmode = 'one-dir'

    def __init__(self, dirname='', load_sync_channel=False, load_channel_location=False):
        BaseRawIO.__init__(self)
        self.dirname = dirname
        self.load_sync_channel = load_sync_channel
        self.load_channel_location = load_channel_location

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        self.signals_info_list = scan_files(self.dirname)

        # sort stream_name by higher sampling rate first
        srates = {info['stream_name']: info['sampling_rate'] for info in self.signals_info_list}
        stream_names = sorted(list(srates.keys()), key=lambda e: srates[e])[::-1]

        nb_segment = np.unique([info['seg_index'] for info in self.signals_info_list]).size

        self._memmaps = {}
        self.signals_info_dict = {}
        for info in self.signals_info_list:
            # key is (seg_index, stream_name)
            key = (info['seg_index'], info['stream_name'])
            assert key not in self.signals_info_dict
            self.signals_info_dict[key] = info

            # create memmap
            data = np.memmap(info['bin_file'], dtype='int16', mode='r', offset=0, order='C')
            # this should be (info['sample_length'], info['num_chan'])
            # be some file are shorten
            data = data.reshape(-1, info['num_chan'])
            self._memmaps[key] = data

        # create channel header
        signal_streams = []
        signal_channels = []
        for stream_name in stream_names:
            # take first segment
            info = self.signals_info_dict[0, stream_name]

            stream_id = stream_name
            stream_index = stream_names.index(info['stream_name'])
            signal_streams.append((stream_name, stream_id))

            # add channels to global list
            for local_chan in range(info['num_chan']):
                chan_name = info['channel_names'][local_chan]
                chan_id = f'{stream_name}#{chan_name}'
                signal_channels.append((chan_name, chan_id, info['sampling_rate'], 'int16',
                                    info['units'], info['channel_gains'][local_chan],
                                    info['channel_offsets'][local_chan], stream_id))
            if not self.load_sync_channel:
                signal_channels = signal_channels[:-1]

        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)
        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # No events
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # No spikes
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

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
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        # insert some annotation at some place
        self._generate_minimal_annotations()
        self._generate_minimal_annotations()
        block_ann = self.raw_annotations['blocks'][0]

        for seg_index in range(nb_segment):
            seg_ann = self.raw_annotations['blocks'][0]['segments'][seg_index]
            seg_ann['name'] = "Segment {}".format(seg_index)

            for c, signal_stream in enumerate(signal_streams):
                stream_name = signal_stream['name']
                sig_ann = self.raw_annotations['blocks'][0]['segments'][seg_index]['signals'][c]

                if self.load_channel_location:
                    # need probeinterface to be installed
                    import probeinterface
                    info = self.signals_info_dict[seg_index, stream_name]
                    if 'imroTbl' in info['meta'] and info['stream_kind'] == 'ap':
                        # only for ap channel
                        probe = probeinterface.read_spikeglx(info['meta_file'])
                        loc = probe.contact_positions
                        if self.load_sync_channel:
                            # one fake channel  for "sys0"
                            loc = np.concatenate((loc, [[0., 0.]]), axis=0)
                        for ndim in range(loc.shape[1]):
                            sig_ann['__array_annotations__'][f'channel_location_{ndim}'] = loc[:, ndim]

    def _segment_t_start(self, block_index, seg_index):
        return 0.

    def _segment_t_stop(self, block_index, seg_index):
        return self._t_stops[seg_index]

    def _get_signal_size(self, block_index, seg_index, stream_index):
        stream_id = self.header['signal_streams'][stream_index]['id']
        memmap = self._memmaps[seg_index, stream_id]
        return int(memmap.shape[0])

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        stream_id = self.header['signal_streams'][stream_index]['id']
        memmap = self._memmaps[seg_index, stream_id]
        if channel_indexes is None:
            if self.load_sync_channel:
                channel_selection = slice(None)
            else:
                channel_selection = slice(-1)
        elif isinstance(channel_indexes, slice):
            if self.load_sync_channel:
                # simple
                channel_selection = channel_indexes
            else:
                # more tricky because negative
                sl_start = channel_indexes.start
                sl_stop = channel_indexes.stop
                sl_step = channel_indexes.step
                if sl_stop is not None and sl_stop < 0:
                    sl_stop = sl_stop - 1
                elif sl_stop is None:
                    sl_stop = -1
                channel_selection = slice(sl_start, sl_stop, sl_step)
        elif not isinstance(channel_indexes, slice):
            if np.all(np.diff(channel_indexes) == 1):
                # consecutive channel then slice this avoid a copy (because of ndarray.take(...)
                # and so keep the underlying memmap
                channel_selection = slice(channel_indexes[0],
                                          channel_indexes[0] + len(channel_indexes))
            else:
                channel_selection = channel_indexes
        else:
            raise ValueError('get_analogsignal_chunk : channel_indexes'
                             'must be slice or list or array of int')

        raw_signals = memmap[slice(i_start, i_stop), channel_selection]

        return raw_signals


def scan_files(dirname):
    """
    Scan for pairs of `.bin` and `.meta` files and return information about it.

    After exploring the folder, the segment index (`seg_index`) is construct as follow:
      * if only one `gate_num=0` then `trigger_num` = `seg_index`
      * if only one `trigger_num=0` then `gate_num` = `seg_index`
      * if both are increasing then seg_index increased by gate_num, trigger_num order.
    """
    info_list = []

    for root, dirs, files in os.walk(dirname):
        for file in files:
            if not file.endswith('.meta'):
                continue
            meta_filename = Path(root) / file
            bin_filename = meta_filename.with_suffix('.bin')
            if meta_filename.exists() and bin_filename.exists():
                meta = read_meta_file(meta_filename)
                info = extract_stream_info(meta_filename, meta)

                info['meta_file'] = str(meta_filename)
                info['bin_file'] = str(bin_filename)
                info_list.append(info)

    # the segment index will depend on both 'gate_num' and 'trigger_num'
    # so we order by 'gate_num' then 'trigger_num'
    # None is before any int
    def make_key(info):
        k0 = info['gate_num']
        if k0 is None:
            k0 = -1
        k1 = info['trigger_num']
        if k1 is None:
            k1 = -1
        return (k0, k1)
    order_key = list({make_key(info) for info in info_list})
    order_key = sorted(order_key)
    for info in info_list:
        info['seg_index'] = order_key.index(make_key(info))

    return info_list


def parse_spikeglx_fname(fname):
    """
    Parse recording identifiers from a SpikeGLX style filename.

    spikeglx naming follow this rules:
    https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#gates-and-triggers

    Example file name structure:
    Consider the filenames: `Noise4Sam_g0_t0.nidq.bin` or `Noise4Sam_g0_t0.imec0.lf.bin`
    The filenames consist of 3 or 4 parts separated by `.`
    1. "Noise4Sam_g0_t0" will be the `name` variable. This choosen by the user at recording time.
    2. "_g0_" is the "gate_num"
    3. "_t0_" is the "trigger_num"
    4. "nidq" or "imec0" will give the `device`
    5. "lf" or "ap" will be the `stream_kind`
        `stream_name` variable is the concatenation of `device.stream_kind`

    This function is copied/modified from Graham Findlay.

    Notes:
       * Sometimes the original file name is modified by the user and "_gt0_" or "_t0_"
          are manually removed. In that case gate_name and trigger_num will be None.

    Parameters
    ---------
    fname: str
        The filename to parse without the extension, e.g. "my-run-name_g0_t1.imec2.lf"
    Returns
    -------
    run_name: str
        The run name, e.g. "my-run-name".
    gate_num: int or None
        The gate identifier, e.g. 0.
    trigger_num: int or None
        The trigger identifier, e.g. 1.
    device: str
        The probe identifier, e.g. "imec2"
    stream_kind: str or None
        The data type identifier, "lf" or "ap" or None
    """
    r = re.findall(r'(\S*)_g(\d*)_t(\d*)\.(\S*).(ap|lf)', fname)
    if len(r) == 1:
        # standard case with probe
        run_name, gate_num, trigger_num, device, stream_kind = r[0]
    else:
        r = re.findall(r'(\S*)_g(\d*)_t(\d*)\.(\S*)', fname)
        if len(r) == 1:
            # case for nidaq
            run_name, gate_num, trigger_num, device = r[0]
            stream_kind = None
        else:
            # the naming do not correspond lets try something more easy
            r = re.findall(r'(\S*)\.(\S*).(ap|lf)', fname)
            if len(r) == 1:
                run_name, device, stream_kind = r[0]
                gate_num, trigger_num = None, None

    if gate_num is not None:
        gate_num = int(gate_num)
    if trigger_num is not None:
        trigger_num = int(trigger_num)

    return (run_name, gate_num, trigger_num, device, stream_kind)


def read_meta_file(meta_file):
    """parse the meta file"""
    with open(meta_file, mode='r') as f:
        lines = f.read().splitlines()

    meta = {}
    # Fix taken from: https://github.com/SpikeInterface/probeinterface/blob/
    # 19d6518fbc67daca71aba5e99d8aa0d445b75eb7/probeinterface/io.py#L649-L662
    for line in lines:
        split_lines = line.split('=')
        if len(split_lines) != 2:
            continue
        k, v = split_lines
        if k.startswith('~'):
            # replace by the list
            k = k[1:]
            v = v[1:-1].split(')(')[1:]
        meta[k] = v

    return meta


def extract_stream_info(meta_file, meta):
    """Extract info from the meta dict"""

    num_chan = int(meta['nSavedChans'])
    fname = Path(meta_file).stem
    run_name, gate_num, trigger_num, device, stream_kind = parse_spikeglx_fname(fname)
    device = fname.split('.')[1]

    if 'imec' in device:
        stream_kind = fname.split('.')[2]
        stream_name = device + '.' + stream_kind
        units = 'uV'
        # please note the 1e6 in gain for this uV

        # metad['imroTbl'] contain two gain per channel  AP and LF
        # except for the last fake channel
        per_channel_gain = np.ones(num_chan, dtype='float64')
        if 'imDatPrb_type' not in meta or meta['imDatPrb_type'] == '0':
            # This work with NP 1.0 case with different metadata versions
            # https://github.com/billkarsh/SpikeGLX/blob/gh-pages/Support/Metadata_3A.md#imec
            # https://github.com/billkarsh/SpikeGLX/blob/gh-pages/Support/Metadata_3B1.md#imec
            # https://github.com/billkarsh/SpikeGLX/blob/gh-pages/Support/Metadata_3B2.md#imec
            if stream_kind == 'ap':
                index_imroTbl = 3
            elif stream_kind == 'lf':
                index_imroTbl = 4
            for c in range(num_chan - 1):
                v = meta['imroTbl'][c].split(' ')[index_imroTbl]
                per_channel_gain[c] = 1. / float(v)
            gain_factor = float(meta['imAiRangeMax']) / 512
            channel_gains = gain_factor * per_channel_gain * 1e6
        elif meta['imDatPrb_type'] in ('21', '24') and stream_kind == 'ap':
            # This work with NP 2.0 case with different metadata versions
            # https://github.com/billkarsh/SpikeGLX/blob/gh-pages/Support/Metadata_20.md#channel-entries-by-type
            # https://github.com/billkarsh/SpikeGLX/blob/gh-pages/Support/Metadata_20.md#imec
            # https://github.com/billkarsh/SpikeGLX/blob/gh-pages/Support/Metadata_30.md#imec
            per_channel_gain[:-1] = 1 / 80.
            gain_factor = float(meta['imAiRangeMax']) / 8192
            channel_gains = gain_factor * per_channel_gain * 1e6
        else:
            raise NotImplementedError('This meta file version of spikeglx'
                                      'is not implemented')
    else:
        stream_kind = ''
        stream_name = device
        units = 'V'
        channel_gains = np.ones(num_chan)

        # there are differents kinds of channels with different gain values
        mn, ma, xa, dw = [int(e) for e in meta['snsMnMaXaDw'].split(sep=',')]
        per_channel_gain = np.ones(num_chan, dtype='float64')
        per_channel_gain[0:mn] = 1. / float(meta['niMNGain'])
        per_channel_gain[mn:mn + ma] = 1. / float(meta['niMAGain'])
        # this scaling come from the code in this zip
        # https://billkarsh.github.io/SpikeGLX/Support/SpikeGLX_Datafile_Tools.zip
        # in file readSGLX.py line76
        # this is equivalent of 2**15
        gain_factor = float(meta['niAiRangeMax']) / 32768
        channel_gains = per_channel_gain * gain_factor

    info = {}
    info['fname'] = fname
    info['meta'] = meta
    for k in ('niSampRate', 'imSampRate'):
        if k in meta:
            info['sampling_rate'] = float(meta[k])
    info['num_chan'] = num_chan

    info['sample_length'] = int(meta['fileSizeBytes']) // 2 // num_chan
    info['gate_num'] = gate_num
    info['trigger_num'] = trigger_num
    info['device'] = device
    info['stream_kind'] = stream_kind
    info['stream_name'] = stream_name
    info['units'] = units
    info['channel_names'] = [txt.split(';')[0] for txt in meta['snsChanMap']]
    info['channel_gains'] = channel_gains
    info['channel_offsets'] = np.zeros(info['num_chan'])

    return info
