"""
Here a list for testing neo.rawio API compliance.
This is called automatically by `BaseTestRawIO`

All rules are listed as function so it should be easier to:
  * identify the rawio API
  * debug
  * discuss rules

"""
import time

if not hasattr(time, 'perf_counter'):
    time.perf_counter = time.time
import logging

import numpy as np

from neo.rawio.baserawio import (_signal_channel_dtype, _signal_stream_dtype,
    _spike_channel_dtype, _event_channel_dtype, _common_sig_characteristics)


def print_class(reader):
    return reader.__class__.__name__


def header_is_total(reader):
    """
    Test if hedaer contains:
      * 'nb_block'
      * 'nb_segment'
      * 'signal_streams'
      * 'signal_channels'
      * 'spike_channels'
      * 'event_channels'

    """
    h = reader.header

    assert 'nb_block' in h, "`nb_block`missing in header"
    assert 'nb_segment' in h, "`nb_segment`missing in header"
    assert len(h['nb_segment']) == h['nb_block']

    assert 'signal_streams' in h, 'signal_streams missing in header'
    if h['signal_streams'] is not None:
        dt = h['signal_streams'].dtype
        for k, _ in _signal_stream_dtype:
            assert k in dt.fields, f'{k} not in signal_streams.dtype'

    assert 'signal_channels' in h, 'signal_channels missing in header'
    if h['signal_channels'] is not None:
        dt = h['signal_channels'].dtype
        for k, _ in _signal_channel_dtype:
            assert k in dt.fields, f'{k} not in signal_channels.dtype'

    assert 'spike_channels' in h, 'spike_channels missing in header'
    if h['spike_channels'] is not None:
        dt = h['spike_channels'].dtype
        for k, _ in _spike_channel_dtype:
            assert k in dt.fields, f'{k} not in spike_channels.dtype'

    assert 'event_channels' in h, 'event_channels missing in header'
    if h['event_channels'] is not None:
        dt = h['event_channels'].dtype
        for k, _ in _event_channel_dtype:
            assert k in dt.fields, f'{k} not in event_channels.dtype'


def count_element(reader):
    """
    Count block/segment/signals/spike/events

    """

    nb_stream = reader.signal_streams_count()
    for stream_index in range(nb_stream):
        nb_chan = reader.signal_channels_count(stream_index)
    nb_unit = reader.spike_channels_count()
    nb_event_channel = reader.event_channels_count()

    nb_block = reader.block_count()
    assert nb_block > 0, '{} have {} block'.format(print_class(reader), nb_block)

    for block_index in range(nb_block):
        nb_seg = reader.segment_count(block_index)

        for seg_index in range(nb_seg):
            t_start = reader.segment_t_start(block_index=block_index, seg_index=seg_index)
            t_stop = reader.segment_t_stop(block_index=block_index, seg_index=seg_index)
            assert t_stop > t_start

            for stream_index in range(nb_stream):
                sig_size = reader.get_signal_size(block_index, seg_index,
                                                  stream_index=stream_index)

                for spike_channel_index in range(nb_unit):
                    nb_spike = reader.spike_count(block_index=block_index, seg_index=seg_index,
                                                  spike_channel_index=spike_channel_index)

                for event_channel_index in range(nb_event_channel):
                    nb_event = reader.event_count(block_index=block_index, seg_index=seg_index,
                                                  event_channel_index=event_channel_index)


def iter_over_sig_chunks(reader, stream_index, channel_indexes, chunksize=1024):
    nb_block = reader.block_count()

    # read all chunk in RAW data
    for block_index in range(nb_block):
        nb_seg = reader.segment_count(block_index)
        for seg_index in range(nb_seg):
            sig_size = reader.get_signal_size(block_index, seg_index, stream_index)

            nb = sig_size // chunksize + 1
            for i in range(nb):
                i_start = i * chunksize
                i_stop = min((i + 1) * chunksize, sig_size)
                raw_chunk = reader.get_analogsignal_chunk(block_index=block_index,
                                        seg_index=seg_index,
                                        i_start=i_start, i_stop=i_stop,
                                        stream_index=stream_index,
                                        channel_indexes=channel_indexes)
                yield raw_chunk


def read_analogsignals(reader):
    """
    Read and convert some signals chunks.

    Test special case when signal_channels do not have same sampling_rate.
    AKA _need_chan_index_check
    """
    nb_stream = reader.signal_streams_count()
    if nb_stream == 0:
        return

    for stream_index in range(nb_stream):
        sr = reader.get_signal_sampling_rate(stream_index=stream_index)
        assert type(sr) == float, 'Type of sampling is {} should float'.format(type(sr))

        # make other test on the first chunk of first block first block
        block_index = 0
        seg_index = 0

        sig_size = reader.get_signal_size(block_index, seg_index, stream_index)

        # read all chunk for all channel all block all segment
        channel_indexes = None
        for raw_chunk in iter_over_sig_chunks(reader, stream_index,
                                    channel_indexes, chunksize=1024):
            assert raw_chunk.ndim == 2

        i_start = 0
        sig_size = reader.get_signal_size(block_index, seg_index, stream_index)
        i_stop = min(1024, sig_size)

        nb_chan = reader.signal_channels_count(stream_index)
        channel_indexes = np.arange(nb_chan, dtype=int)

        signal_channels = reader.header['signal_channels']
        stream_id = reader.header['signal_streams'][stream_index]['id']
        mask = signal_channels['stream_id'] == stream_id
        channel_names = signal_channels['name'][mask]
        channel_ids = signal_channels['id'][mask]

        # acces by channel inde/ids/names should give the same chunk
        channel_indexes2 = channel_indexes[::2]
        channel_names2 = channel_names[::2]
        channel_ids2 = channel_ids[::2]

        # slice by index
        raw_chunk0 = reader.get_analogsignal_chunk(block_index=block_index, seg_index=seg_index,
                                        i_start=i_start, i_stop=i_stop,
                                        stream_index=stream_index,
                                        channel_indexes=channel_indexes2)
        assert raw_chunk0.ndim == 2
        assert raw_chunk0.shape[0] == i_stop
        assert raw_chunk0.shape[1] == len(channel_indexes2)

        # slice by ids
        raw_chunk2 = reader.get_analogsignal_chunk(block_index=block_index, seg_index=seg_index,
                                        i_start=i_start, i_stop=i_stop,
                                        stream_index=stream_index,
                                        channel_ids=channel_ids2)
        np.testing.assert_array_equal(raw_chunk0, raw_chunk2)

        # channel names are not always unique inside a stream
        unique_chan_name = (np.unique(channel_names).size == channel_names.size)
        if unique_chan_name:
            raw_chunk1 = reader.get_analogsignal_chunk(block_index=block_index, seg_index=seg_index,
                                        i_start=i_start, i_stop=i_stop,
                                        stream_index=stream_index,
                                        channel_names=channel_names2)
            np.testing.assert_array_equal(raw_chunk0, raw_chunk1)

        # test prefer_slice=True/False
        if nb_chan >= 3:
            for prefer_slice in (True, False):
                raw_chunk3 = reader.get_analogsignal_chunk(block_index=block_index,
                                        seg_index=seg_index,
                                        i_start=i_start, i_stop=i_stop,
                                        stream_index=stream_index,
                                        channel_indexes=[1, 2])

        # convert to float32/float64
        for dt in ('float32', 'float64'):
            float_chunk0 = reader.rescale_signal_raw_to_float(raw_chunk0, dtype=dt,
                                        stream_index=stream_index,
                                        channel_indexes=channel_indexes2)
            float_chunk2 = reader.rescale_signal_raw_to_float(raw_chunk2, dtype=dt,
                                        stream_index=stream_index,
                                        channel_ids=channel_ids2)
            if unique_chan_name:
                float_chunk1 = reader.rescale_signal_raw_to_float(raw_chunk1,
                                        dtype=dt,
                                        stream_index=stream_index,
                                        channel_names=channel_names2)

            assert float_chunk0.dtype == dt
            np.testing.assert_array_equal(float_chunk0, float_chunk2)
            if unique_chan_name:
                np.testing.assert_array_equal(float_chunk0, float_chunk1)

        # read 500ms with several chunksize
        sr = reader.get_signal_sampling_rate(stream_index=stream_index)
        lenght_to_read = int(.5 * sr)
        if lenght_to_read < sig_size:
            ref_raw_sigs = reader.get_analogsignal_chunk(block_index=block_index,
                                                    seg_index=seg_index, i_start=0,
                                                    i_stop=lenght_to_read,
                                                    stream_index=stream_index,
                                                    channel_indexes=channel_indexes)
            for chunksize in (511, 512, 513, 1023, 1024, 1025):
                i_start = 0
                chunks = []
                while i_start < lenght_to_read:
                    i_stop = min(i_start + chunksize, lenght_to_read)
                    raw_chunk = reader.get_analogsignal_chunk(block_index=block_index,
                                                            seg_index=seg_index, i_start=i_start,
                                                            i_stop=i_stop,
                                                            stream_index=stream_index,
                                                            channel_indexes=channel_indexes)
                    chunks.append(raw_chunk)
                    i_start += chunksize
                chunk_raw_sigs = np.concatenate(chunks, axis=0)
                np.testing.assert_array_equal(ref_raw_sigs, chunk_raw_sigs)


def benchmark_speed_read_signals(reader):
    """
    A very basic speed measurement that read all signal
    in a file.
    """

    nb_stream = reader.signal_streams_count()
    if nb_stream == 0:
        return

    for stream_index in range(nb_stream):

        nb_samples = 0
        channel_indexes = None
        nb_chan = reader.signal_channels_count(stream_index)

        t0 = time.perf_counter()
        for raw_chunk in iter_over_sig_chunks(reader, stream_index,
                                    channel_indexes, chunksize=1024):
            nb_samples += raw_chunk.shape[0]
        t1 = time.perf_counter()
        if t0 != t1:
            speed = (nb_samples * nb_chan) / (t1 - t0) / 1e6
        else:
            speed = np.inf
        logging.info(
            f'{print_class(reader)} read ({nb_chan}channels x {nb_samples}samples)'
            f'in {t1 - t0:0.3f} s so speed {speed:0.3f} MSPS from {reader.source_name()}')


def read_spike_times(reader):
    """
    Read and convert all spike times.
    """

    nb_block = reader.block_count()
    nb_unit = reader.spike_channels_count()

    for block_index in range(nb_block):
        nb_seg = reader.segment_count(block_index)
        for seg_index in range(nb_seg):
            for spike_channel_index in range(nb_unit):
                nb_spike = reader.spike_count(block_index=block_index,
                                              seg_index=seg_index,
                                              spike_channel_index=spike_channel_index)
                if nb_spike == 0:
                    continue

                spike_timestamp = reader.get_spike_timestamps(block_index=block_index,
                                                        seg_index=seg_index,
                                                        spike_channel_index=spike_channel_index,
                                                        t_start=None, t_stop=None)
                assert spike_timestamp.shape[0] == nb_spike, 'nb_spike {} != {}'.format(
                    spike_timestamp.shape[0], nb_spike)

                spike_times = reader.rescale_spike_timestamp(spike_timestamp, 'float64')
                assert spike_times.dtype == 'float64'

                if spike_times.size > 3:
                    # load only one spike by forcing limits
                    t_start = spike_times[1] - 0.001
                    t_stop = spike_times[1] + 0.001

                    spike_timestamp2 = reader.get_spike_timestamps(block_index=block_index,
                                                    seg_index=seg_index,
                                                    spike_channel_index=spike_channel_index,
                                                    t_start=t_start, t_stop=t_stop)
                    assert spike_timestamp2.shape[0] == 1

                    spike_times2 = reader.rescale_spike_timestamp(spike_timestamp2, 'float64')
                    assert spike_times2[0] == spike_times[1]


def read_spike_waveforms(reader):
    """
    Read and convert some all waveforms.
    """
    nb_block = reader.block_count()
    nb_unit = reader.spike_channels_count()

    for block_index in range(nb_block):
        nb_seg = reader.segment_count(block_index)
        for seg_index in range(nb_seg):
            for spike_channel_index in range(nb_unit):
                nb_spike = reader.spike_count(block_index=block_index,
                                              seg_index=seg_index,
                                              spike_channel_index=spike_channel_index)
                if nb_spike == 0:
                    continue

                raw_waveforms = reader.get_spike_raw_waveforms(block_index=block_index,
                                                        seg_index=seg_index,
                                                        spike_channel_index=spike_channel_index,
                                                        t_start=None, t_stop=None)
                if raw_waveforms is None:
                    continue
                assert raw_waveforms.shape[0] == nb_spike
                assert raw_waveforms.ndim == 3

                for dt in ('float32', 'float64'):
                    float_waveforms = reader.rescale_waveforms_to_float(
                        raw_waveforms, dtype=dt, spike_channel_index=spike_channel_index)
                    assert float_waveforms.dtype == dt
                    assert float_waveforms.shape == raw_waveforms.shape


def read_events(reader):
    """
    Read and convert some event or epoch.
    """
    nb_block = reader.block_count()
    nb_event_channel = reader.event_channels_count()

    for block_index in range(nb_block):
        nb_seg = reader.segment_count(block_index)
        for seg_index in range(nb_seg):
            for ev_chan in range(nb_event_channel):
                nb_event = reader.event_count(block_index=block_index, seg_index=seg_index,
                                              event_channel_index=ev_chan)
                if nb_event == 0:
                    continue

                ev_timestamps, ev_durations, ev_labels = reader.get_event_timestamps(
                    block_index=block_index, seg_index=seg_index,
                    event_channel_index=ev_chan)
                assert ev_timestamps.shape[0] == nb_event, 'Wrong shape {}, {}'.format(
                    ev_timestamps.shape[0], nb_event)
                if ev_durations is not None:
                    assert ev_durations.shape[0] == nb_event
                assert ev_labels.shape[0] == nb_event

                ev_times = reader.rescale_event_timestamp(ev_timestamps, dtype='float64')
                assert ev_times.dtype == 'float64'


def has_annotations(reader):
    assert hasattr(reader, 'raw_annotations'), 'raw_annotation are not set'
