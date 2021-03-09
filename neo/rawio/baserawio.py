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
  * neo tree object is symetric and logical: same channel/units/event
    along all block and segments.


So this handles **only** one simplified but very frequent case of dataset:
    * Only one channel set  for AnalogSignal (aka ChannelIndex) stable along Segment
    * Only one channel set  for SpikeTrain (aka Unit) stable along Segment
    * AnalogSignal have all the same sampling_rate acroos all Segment
    * t_start/t_stop are the same for many object (SpikeTrain, Event) inside a Segment

signal channels  are handled by group of  "stream".
one stream will at neo.io level one AnalogSignal with multi-channel.


A helper class `neo.io.basefromrawio.BaseFromRaw` transform a RawIO to
neo legacy IO. In short all "neo.rawio" classes are also "neo.io"
with lazy reading capability.


With this API the IO have an attributes `header` with necessary keys.
This  `header` attribute is done in `_parse_header(...)` method.
See ExampleRawIO as example.


BaseRawIO implement a possible presistent cache system that can be used
by some IOs to avoid very long parse_header(). The idea is that some variable
or vector can be store somewhere (near the file, /tmp, any path)


"""

import logging
import numpy as np
import os
import sys

from neo import logging_handler

try:
    import joblib

    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False

possible_raw_modes = ['one-file', 'multi-file', 'one-dir', ]  # 'multi-dir', 'url', 'other'

error_header = 'Header is not read yet, do parse_header() first'

_signal_stream_dtype = [
    ('name', 'U64'),  # not necessary unique
    ('id', 'U64'),  # must be unique
]

_signal_channel_dtype = [
    ('name', 'U64'),  # not necessarily unique
    ('id', 'U64'),  # must be unique
    ('sampling_rate', 'float64'),
    ('dtype', 'U16'),
    ('units', 'U64'),
    ('gain', 'float64'),
    ('offset', 'float64'),
    ('stream_id', 'U64'),
]
# TODO for later: add t_start and length in _signal_channel_dtype
# this would simplify all t_start/t_stop stuff for each RawIO class

_common_sig_characteristics = ['sampling_rate', 'dtype', 'stream_id']

_spike_channel_dtype = [
    ('name', 'U64'),
    ('id', 'U64'),
    # for waveform
    ('wf_units', 'U64'),
    ('wf_gain', 'float64'),
    ('wf_offset', 'float64'),
    ('wf_left_sweep', 'int64'),
    ('wf_sampling_rate', 'float64'),
]

# in rawio event and epoch are handled the same way
# except, that duration is `None` for events
_event_channel_dtype = [
    ('name', 'U64'),
    ('id', 'U64'),
    ('type', 'S5'),  # epoch or event
]


class BaseRawIO:
    """
    Generic class to handle.

    """

    name = 'BaseIO'
    description = ''
    extensions = []

    rawmode = None  # one key in possible_raw_modes

    def __init__(self, use_cache=False, cache_path='same_as_resource', **kargs):
        """

        When rawmode=='one-file' kargs MUST contains 'filename' the filename
        When rawmode=='multi-file' kargs MUST contains 'filename' one of the filenames.
        When rawmode=='one-dir' kargs MUST contains 'dirname' the dirname.


        """
        # create a logger for the IO class
        fullname = self.__class__.__module__ + '.' + self.__class__.__name__
        self.logger = logging.getLogger(fullname)
        # create a logger for 'neo' and add a handler to it if it doesn't
        # have one already.
        # (it will also not add one if the root logger has a handler)
        corename = self.__class__.__module__.split('.')[0]
        corelogger = logging.getLogger(corename)
        rootlogger = logging.getLogger()
        if not corelogger.handlers and not rootlogger.handlers:
            corelogger.addHandler(logging_handler)

        self.use_cache = use_cache
        if use_cache:
            assert HAVE_JOBLIB, 'You need to install joblib for cache'
            self.setup_cache(cache_path)
        else:
            self._cache = None

        self.header = None

    def parse_header(self):
        """
        This must parse the file header to get all stuff for fast use later on.

        This must create
        self.header['nb_block']
        self.header['nb_segment']
        self.header['signal_streams']
        self.header['signal_channels']
        self.header['spike_channels']
        self.header['event_channels']



        """
        self._parse_header()
        self._check_stream_signal_channel_characteristics()

    def source_name(self):
        """Return fancy name of file source"""
        return self._source_name()

    def __repr__(self):
        txt = '{}: {}\n'.format(self.__class__.__name__, self.source_name())
        if self.header is not None:
            nb_block = self.block_count()
            txt += 'nb_block: {}\n'.format(nb_block)
            nb_seg = [self.segment_count(i) for i in range(nb_block)]
            txt += 'nb_segment:  {}\n'.format(nb_seg)

            # signal streams
            v = [s['name'] + f' (chans: {self.signal_channels_count(i)})'
                for i, s in enumerate(self.header['signal_streams'])]
            v = pprint_vector(v)
            txt += f'signal_streams: {v}\n'

            for k in ('signal_channels', 'spike_channels', 'event_channels'):
                ch = self.header[k]
                v = pprint_vector(self.header[k]['name'])
                txt += f'{k}: {v}\n'

        return txt

    def _generate_minimal_annotations(self):
        """
        Helper function that generate a nested dict for annotations.

        Must be called when these are Ok after self.header is done
        And so when theses function are ready:
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
        signal_streams = self.header['signal_streams']
        signal_channels = self.header['signal_channels']
        spike_channels = self.header['spike_channels']
        event_channels = self.header['event_channels']

        # use for AnalogSignal.annotations and AnalogSignal.array_annotations
        signal_stream_annotations = []
        for c in range(signal_streams.size):
            stream_id = signal_streams[c]['id']
            channels = signal_channels[signal_channels['stream_id'] == stream_id]
            d = {}
            d['name'] = signal_streams['name'][c]
            d['stream_id'] = stream_id
            d['file_origin'] = self._source_name()
            d['__array_annotations__'] = {}
            for key in ('name', 'id'):
                values = np.array([channels[key][chan] for chan in range(channels.size)])
                d['__array_annotations__']['channel_' + key + 's'] = values
            signal_stream_annotations.append(d)

        # used for SpikeTrain.annotations and SpikeTrain.array_annotations
        spike_annotations = []
        for c in range(spike_channels.size):
            # use for Unit.annotations
            d = {}
            d['name'] = spike_channels['name'][c]
            d['id'] = spike_channels['id'][c]
            d['file_origin'] = self._source_name()
            d['__array_annotations__'] = {}
            spike_annotations.append(d)

        # used for Event/Epoch.annotations and Event/Epoch.array_annotations
        event_annotations = []
        for c in range(event_channels.size):
            # not used in neo.io at the moment could usefull one day
            d = {}
            d['name'] = event_channels['name'][c]
            d['id'] = event_channels['id'][c]
            d['file_origin'] = self._source_name()
            d['__array_annotations__'] = {}
            event_annotations.append(d)

        # duplicate this signal_stream_annotations/spike_annotations/event_annotations
        # accros blocks and segments and create annotations
        ann = {}
        ann['blocks'] = []
        for block_index in range(self.block_count()):
            d = {}
            d['file_origin'] = self.source_name()
            d['segments'] = []
            ann['blocks'].append(d)

            for seg_index in range(self.segment_count(block_index)):
                d = {}
                d['file_origin'] = self.source_name()
                # copy nested
                d['signals'] = signal_stream_annotations.copy()
                d['spikes'] = spike_annotations.copy()
                d['events'] = event_annotations.copy()
                ann['blocks'][block_index]['segments'].append(d)

        self.raw_annotations = ann

    def _repr_annotations(self):
        txt = 'Raw annotations\n'
        for block_index in range(self.block_count()):
            bl_a = self.raw_annotations['blocks'][block_index]
            txt += '*Block {}\n'.format(block_index)
            for k, v in bl_a.items():
                if k in ('segments',):
                    continue
                txt += '  -{}: {}\n'.format(k, v)
            for seg_index in range(self.segment_count(block_index)):
                seg_a = bl_a['segments'][seg_index]
                txt += '  *Segment {}\n'.format(seg_index)
                for k, v in seg_a.items():
                    if k in ('signals', 'spikes', 'events',):
                        continue
                    txt += '    -{}: {}\n'.format(k, v)

                # annotations by channels for spikes/events/epochs
                for child in ('signals', 'events', 'spikes', ):
                    if child == 'signals':
                        n = self.header['signal_streams'].shape[0]
                    else:
                        n = self.header[child[:-1] + '_channels'].shape[0]
                    for c in range(n):
                        neo_name = {'signals': 'AnalogSignal',
                                    'spikes': 'SpikeTrain',
                                    'events': 'Event/Epoch'}[child]
                        txt += f'    *{neo_name} {c}\n'
                        child_a = seg_a[child][c]
                        for k, v in child_a.items():
                            if k == '__array_annotations__':
                                continue
                            txt += f'      -{k}: {v}\n'
                        for k, values in child_a['__array_annotations__'].items():
                            values = ', '.join([str(v) for v in values[:4]])
                            values = '[ ' + values + ' ...'
                            txt += f'      -{k}: {values}\n'

        return txt

    def print_annotations(self):
        """Print formated raw_annotations"""
        print(self._repr_annotations())

    def block_count(self):
        """return number of blocks"""
        return self.header['nb_block']

    def segment_count(self, block_index):
        """return number of segment for a given block"""
        return self.header['nb_segment'][block_index]

    def signal_streams_count(self):
        """Return the number of signal  stream channels.
        Same along all Blocks and Segments.
        """
        return len(self.header['signal_streams'])

    def signal_channels_count(self, stream_index):
        """Return the number of signal channels for a given stream
        This number is constant across Blocks and Segments.
        """
        stream_id = self.header['signal_streams'][stream_index]['id']
        channels = self.header['signal_channels']
        channels = channels[channels['stream_id'] == stream_id]
        return len(channels)

    def spike_channels_count(self):
        """Return the number of unit (aka spike) channels.
        Same along all Blocks and Segment.
        """
        return len(self.header['spike_channels'])

    def event_channels_count(self):
        """Return the number of event/epoch channels.
        Same allong all Blocks and Segments.
        """
        return len(self.header['event_channels'])

    def segment_t_start(self, block_index, seg_index):
        """Global t_start of a Segment in s. Shared by all objects except
        for AnalogSignal.
        """
        return self._segment_t_start(block_index, seg_index)

    def segment_t_stop(self, block_index, seg_index):
        """Global t_start of a Segment in s. Shared by all objects except
        for AnalogSignal.
        """
        return self._segment_t_stop(block_index, seg_index)

    ###
    # signal and channel zone

    def _check_stream_signal_channel_characteristics(self):
        """
        This check that all channel that belong to the same stream_id
        have common characteristics:
          * stream_id (explicite channel group)
          * sampling_rate (global along block and segment)
          * units
          * dtype
        """
        signal_streams = self.header['signal_streams']
        signal_channels = self.header['signal_channels']
        if signal_streams.size > 0:
            assert signal_channels.size > 0, 'Signal stream but no signal_channels!!!'

        for stream_index in range(signal_streams.size):
            stream_id = signal_streams[stream_index]['id']
            mask = signal_channels['stream_id'] == stream_id
            characteristics = signal_channels[mask][_common_sig_characteristics]
            unique_characteristics = np.unique(characteristics)
            assert unique_characteristics.size == 1, \
                f'Some channel in stream_id {stream_id} ' \
                f'do not have same {_common_sig_characteristics} {unique_characteristics}'

            # also check that id is unique inside a stream
            channel_ids = signal_channels[mask]['id']
            assert np.unique(channel_ids).size == channel_ids.size, \
                f'signal_channels dont have unique ids for stream {stream_index}'

        self._several_channel_groups = signal_streams.size > 1

    def channel_name_to_index(self, stream_index, channel_names):
        """
        Inside a stream, transform channel_names to channel_indexes.
        Based on self.header['signal_channels']
        channel_indexes is local to stream
        """
        stream_id = self.header['signal_streams'][stream_index]['id']
        mask = self.header['signal_channels']['stream_id'] == stream_id
        signal_channels = self.header['signal_channels'][mask]
        chan_names = list(signal_channels['name'])
        assert signal_channels.size == np.unique(chan_names).size, 'Channel names not unique'
        channel_indexes = np.array([chan_names.index(name) for name in channel_names])
        return channel_indexes

    def channel_id_to_index(self, stream_index, channel_ids):
        """
        Inside a stream,  transform channel_ids to channel_indexes.
        Based on self.header['signal_channels']
        channel_indexes is local to stream
        """
        # unique ids is already check in _check_stream_signal_channel_characteristics
        stream_id = self.header['signal_streams'][stream_index]['id']
        mask = self.header['signal_channels']['stream_id'] == stream_id
        signal_channels = self.header['signal_channels'][mask]
        chan_ids = list(signal_channels['id'])
        channel_indexes = np.array([chan_ids.index(chan_id) for chan_id in channel_ids])
        return channel_indexes

    def _get_channel_indexes(self, stream_index, channel_indexes, channel_names, channel_ids):
        """
        Select channel_indexes from channel_indexes/channel_names/channel_ids
        depending which is not None.
        """
        if channel_indexes is None and channel_names is not None:
            channel_indexes = self.channel_name_to_index(stream_index, channel_names)
        elif channel_indexes is None and channel_ids is not None:
            channel_indexes = self.channel_id_to_index(stream_index, channel_ids)
        return channel_indexes

    def _get_stream_index(self, stream_index):
        if stream_index is None:
            assert self.header['signal_streams'].size == 1
            stream_index = 0
        else:
            assert 0 <= stream_index < self.header['signal_streams'].size
        return stream_index

    def get_signal_size(self, block_index, seg_index, stream_index=None):
        stream_index = self._get_stream_index(stream_index)
        return self._get_signal_size(block_index, seg_index, stream_index)

    def get_signal_t_start(self, block_index, seg_index, stream_index=None):
        stream_index = self._get_stream_index(stream_index)
        return self._get_signal_t_start(block_index, seg_index, stream_index)

    def get_signal_sampling_rate(self, stream_index=None):
        stream_index = self._get_stream_index(stream_index)
        stream_id = self.header['signal_streams'][stream_index]['id']
        mask = self.header['signal_channels']['stream_id'] == stream_id
        signal_channels = self.header['signal_channels'][mask]
        sr = signal_channels[0]['sampling_rate']
        return float(sr)

    def get_analogsignal_chunk(self, block_index=0, seg_index=0, i_start=None, i_stop=None,
                               stream_index=None, channel_indexes=None, channel_names=None,
                               channel_ids=None, prefer_slice=False):
        """
        Return a chunk of raw signal.
        """
        stream_index = self._get_stream_index(stream_index)
        channel_indexes = self._get_channel_indexes(stream_index, channel_indexes,
                                                    channel_names, channel_ids)

        # some check on channel_indexes
        if isinstance(channel_indexes, list):
            channel_indexes = np.asarray(channel_indexes)

        if isinstance(channel_indexes, np.ndarray):
            if channel_indexes.dtype == 'bool':
                assert self.signal_channels_count(stream_index) == channel_indexes.size
                channel_indexes, = np.nonzero(channel_indexes)

        if prefer_slice and isinstance(channel_indexes, np.ndarray):
            # check if channel_indexes are coninuous and transform to slice
            # this is usefull for memmap or hdf5 where slice make read lazy
            # contrary to indexes that make a copy (like numpy.take())
            if np.all(np.diff(channel_indexes) == 1):
                channel_indexes = slice(channel_indexes[0], channel_indexes[-1] + 1)

        raw_chunk = self._get_analogsignal_chunk(
            block_index, seg_index, i_start, i_stop, stream_index, channel_indexes)

        return raw_chunk

    def rescale_signal_raw_to_float(self, raw_signal, dtype='float32', stream_index=None,
                                    channel_indexes=None, channel_names=None, channel_ids=None):
        stream_index = self._get_stream_index(stream_index)
        channel_indexes = self._get_channel_indexes(stream_index, channel_indexes,
                                                    channel_names, channel_ids)
        if channel_indexes is None:
            channel_indexes = slice(None)

        stream_id = self.header['signal_streams'][stream_index]['id']
        mask = self.header['signal_channels']['stream_id'] == stream_id
        channels = self.header['signal_channels'][mask]
        if channel_indexes is None:
            channel_indexes = slice(None)
        channels = channels[channel_indexes]

        float_signal = raw_signal.astype(dtype)

        if np.any(channels['gain'] != 1.):
            float_signal *= channels['gain']

        if np.any(channels['offset'] != 0.):
            float_signal += channels['offset']

        return float_signal

    # spiketrain and unit zone
    def spike_count(self, block_index=0, seg_index=0, spike_channel_index=0):
        return self._spike_count(block_index, seg_index, spike_channel_index)

    def get_spike_timestamps(self, block_index=0, seg_index=0, spike_channel_index=0,
                             t_start=None, t_stop=None):
        """
        The timestamp is as close to the format itself. Sometimes float/int32/int64.
        Sometimes it is the index on the signal but not always.
        The conversion to second or index_on_signal is done outside here.

        t_start/t_sop are limits in seconds.
        """
        timestamp = self._get_spike_timestamps(block_index, seg_index,
                                               spike_channel_index, t_start, t_stop)
        return timestamp

    def rescale_spike_timestamp(self, spike_timestamps, dtype='float64'):
        """
        Rescale spike timestamps to seconds.
        """
        return self._rescale_spike_timestamp(spike_timestamps, dtype)

    # spiketrain waveform zone
    def get_spike_raw_waveforms(self, block_index=0, seg_index=0, spike_channel_index=0,
                                t_start=None, t_stop=None):
        wf = self._get_spike_raw_waveforms(block_index, seg_index,
                                           spike_channel_index, t_start, t_stop)
        return wf

    def rescale_waveforms_to_float(self, raw_waveforms, dtype='float32',
                                   spike_channel_index=0):
        wf_gain = self.header['spike_channels']['wf_gain'][spike_channel_index]
        wf_offset = self.header['spike_channels']['wf_offset'][spike_channel_index]

        float_waveforms = raw_waveforms.astype(dtype)

        if wf_gain != 1.:
            float_waveforms *= wf_gain
        if wf_offset != 0.:
            float_waveforms += wf_offset

        return float_waveforms

    # event and epoch zone
    def event_count(self, block_index=0, seg_index=0, event_channel_index=0):
        return self._event_count(block_index, seg_index, event_channel_index)

    def get_event_timestamps(self, block_index=0, seg_index=0, event_channel_index=0,
                             t_start=None, t_stop=None):
        """
        The timestamp is as close to the format itself. Sometimes float/int32/int64.
        Sometimes it is the index on the signal but not always.
        The conversion to second or index_on_signal is done outside here.

        t_start/t_sop are limits in seconds.

        returns
            timestamp
            labels
            durations

        """
        timestamp, durations, labels = self._get_event_timestamps(
            block_index, seg_index, event_channel_index, t_start, t_stop)
        return timestamp, durations, labels

    def rescale_event_timestamp(self, event_timestamps, dtype='float64',
                    event_channel_index=0):
        """
        Rescale event timestamps to s
        """
        return self._rescale_event_timestamp(event_timestamps, dtype, event_channel_index)

    def rescale_epoch_duration(self, raw_duration, dtype='float64',
                    event_channel_index=0):
        """
        Rescale epoch raw duration to s
        """
        return self._rescale_epoch_duration(raw_duration, dtype, event_channel_index)

    def setup_cache(self, cache_path, **init_kargs):
        if self.rawmode in ('one-file', 'multi-file'):
            resource_name = self.filename
        elif self.rawmode == 'one-dir':
            resource_name = self.dirname
        else:
            raise (NotImplementedError)

        if cache_path == 'home':
            if sys.platform.startswith('win'):
                dirname = os.path.join(os.environ['APPDATA'], 'neo_rawio_cache')
            elif sys.platform.startswith('darwin'):
                dirname = '~/Library/Application Support/neo_rawio_cache'
            else:
                dirname = os.path.expanduser('~/.config/neo_rawio_cache')
            dirname = os.path.join(dirname, self.__class__.__name__)

            if not os.path.exists(dirname):
                os.makedirs(dirname)
        elif cache_path == 'same_as_resource':
            dirname = os.path.dirname(resource_name)
        else:
            assert os.path.exists(cache_path), \
                'cache_path do not exists use "home" or "same_as_resource" to make this auto'

        # the hash of the resource (dir of file) is done with filename+datetime
        # TODO make something more sophisticated when rawmode='one-dir' that use all
        #  filename and datetime
        d = dict(ressource_name=resource_name, mtime=os.path.getmtime(resource_name))
        hash = joblib.hash(d, hash_name='md5')

        # name is constructed from the real_n,ame and the hash
        name = '{}_{}'.format(os.path.basename(resource_name), hash)
        self.cache_filename = os.path.join(dirname, name)

        if os.path.exists(self.cache_filename):
            self.logger.warning('Use existing cache file {}'.format(self.cache_filename))
            self._cache = joblib.load(self.cache_filename)
        else:
            self.logger.warning('Create cache file {}'.format(self.cache_filename))
            self._cache = {}
            self.dump_cache()

    def add_in_cache(self, **kargs):
        assert self.use_cache
        self._cache.update(kargs)
        self.dump_cache()

    def dump_cache(self):
        assert self.use_cache
        joblib.dump(self._cache, self.cache_filename)

    ##################

    # Functions to be implemented in IO below here

    def _parse_header(self):
        raise (NotImplementedError)
        # must call
        # self._generate_empty_annotations()

    def _source_name(self):
        raise (NotImplementedError)

    def _segment_t_start(self, block_index, seg_index):
        raise (NotImplementedError)

    def _segment_t_stop(self, block_index, seg_index):
        raise (NotImplementedError)

    ###
    # signal and channel zone
    def _get_signal_size(self, block_index, seg_index, stream_index):
        """
        Return the size of a set of AnalogSignals indexed by channel_indexes.

        All channels indexed must have the same size and t_start.
        """
        raise (NotImplementedError)

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        """
        Return the t_start of a set of AnalogSignals indexed by channel_indexes.

        All channels indexed must have the same size and t_start.
        """
        raise (NotImplementedError)

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        """
        Return the samples from a set of AnalogSignals indexed
        by channel_indexes (local index inner stream).

        All channels indexed must have the same size and t_start.

        RETURNS
        -------
            array of samples, with each requested channel in a column
        """

        raise (NotImplementedError)

    ###
    # spiketrain and unit zone
    def _spike_count(self, block_index, seg_index, spike_channel_index):
        raise (NotImplementedError)

    def _get_spike_timestamps(self, block_index, seg_index,
                              spike_channel_index, t_start, t_stop):
        raise (NotImplementedError)

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        raise (NotImplementedError)

    ###
    # spike waveforms zone
    def _get_spike_raw_waveforms(self, block_index, seg_index,
                                 spike_channel_index, t_start, t_stop):
        raise (NotImplementedError)

    ###
    # event and epoch zone
    def _event_count(self, block_index, seg_index, event_channel_index):
        raise (NotImplementedError)

    def _get_event_timestamps(self, block_index, seg_index, event_channel_index, t_start, t_stop):
        raise (NotImplementedError)

    def _rescale_event_timestamp(self, event_timestamps, dtype):
        raise (NotImplementedError)

    def _rescale_epoch_duration(self, raw_duration, dtype):
        raise (NotImplementedError)


def pprint_vector(vector, lim=8):
    vector = np.asarray(vector)
    assert vector.ndim == 1
    if len(vector) > lim:
        part1 = ', '.join(e for e in vector[:lim // 2])
        part2 = ' , '.join(e for e in vector[-lim // 2:])
        txt = f"[{part1} ... {part2}]"
    else:
        part1 = ', '.join(e for e in vector[:lim // 2])
        txt = f"[{part1}]"
    return txt
