"""
Tests proxyobject mechanisms with ExampleRawIO
"""


import unittest

import numpy as np
import quantities as pq
from neo.rawio.examplerawio import ExampleRawIO
from neo.io.proxyobjects import (AnalogSignalProxy, SpikeTrainProxy,
                EventProxy, EpochProxy)

from neo.core import (Segment, AnalogSignal,
                      Epoch, Event, SpikeTrain)


from neo.test.tools import (assert_arrays_almost_equal,
                            assert_neo_object_is_compliant,
                            assert_same_attributes)


class BaseProxyTest(unittest.TestCase):
    def setUp(self):
        self.reader = ExampleRawIO(filename='my_filename.fake')
        self.reader.parse_header()


class TestAnalogSignalProxy(BaseProxyTest):

    def test_AnalogSignalProxy(self):
        proxy_anasig = AnalogSignalProxy(rawio=self.reader,
                    stream_index=0, inner_stream_channels=None,
                    block_index=0, seg_index=0,)

        assert proxy_anasig.sampling_rate == 10 * pq.kHz
        assert proxy_anasig.t_start == 0 * pq.s
        assert proxy_anasig.t_stop == 10 * pq.s
        assert proxy_anasig.duration == 10 * pq.s
        assert proxy_anasig.file_origin == 'my_filename.fake'

        # full load
        full_anasig = proxy_anasig.load(time_slice=None)
        assert isinstance(full_anasig, AnalogSignal)
        assert_same_attributes(proxy_anasig, full_anasig)

        # slice time
        anasig = proxy_anasig.load(time_slice=(2. * pq.s, 5 * pq.s))
        assert anasig.t_start == 2. * pq.s
        assert anasig.duration == 3. * pq.s
        assert anasig.shape == (30000, 8)
        assert_same_attributes(proxy_anasig.time_slice(2. * pq.s, 5 * pq.s), anasig)

        # ceil next sample when slicing
        anasig = proxy_anasig.load(time_slice=(1.99999 * pq.s, 5.000001 * pq.s))
        assert anasig.t_start == 2. * pq.s
        assert anasig.duration == 3. * pq.s
        assert anasig.shape == (30000, 8)

        # buggy time slice
        with self.assertRaises(AssertionError):
            anasig = proxy_anasig.load(time_slice=(2. * pq.s, 15 * pq.s))
        anasig = proxy_anasig.load(time_slice=(2. * pq.s, 15 * pq.s), strict_slicing=False)
        assert proxy_anasig.t_stop == 10 * pq.s

        # select channels
        anasig = proxy_anasig.load(channel_indexes=[3, 4, 5])
        assert anasig.shape[1] == 3

        # select channels and slice times
        anasig = proxy_anasig.load(time_slice=(2. * pq.s, 5 * pq.s), channel_indexes=[3, 4, 5])
        assert anasig.shape == (30000, 3)

        # magnitude mode rescaled
        anasig_float = proxy_anasig.load(magnitude_mode='rescaled')
        assert anasig_float.dtype == 'float32'
        assert anasig_float.units == pq.uV
        assert anasig_float.units == proxy_anasig.units

        # magnitude mode raw
        anasig_int = proxy_anasig.load(magnitude_mode='raw')
        assert anasig_int.dtype == 'int16'
        assert anasig_int.units == pq.CompoundUnit('0.0152587890625*uV')

        assert_arrays_almost_equal(anasig_float, anasig_int.rescale('uV'), 1e-9)

        # test array_annotations
        assert '__array_annotations__' not in proxy_anasig.annotations
        assert 'impedance' in proxy_anasig.array_annotations
        assert proxy_anasig.array_annotations['impedance'].size == 8
        assert 'impedance' in anasig_float.array_annotations
        assert anasig_float.array_annotations['impedance'].size == 8

    def test_global_local_channel_indexes(self):
        proxy_anasig = AnalogSignalProxy(rawio=self.reader,
                    stream_index=0, inner_stream_channels=slice(0, 8, 2),
                    block_index=0, seg_index=0)

        assert proxy_anasig.shape == (100000, 4)
        assert np.array_equal(proxy_anasig.array_annotations['channel_names'],
                        ['ch0', 'ch2', 'ch4', 'ch6'])

        # should be channel ch0 and ch6
        anasig = proxy_anasig.load(channel_indexes=[0, 3])
        assert anasig.shape == (100000, 2)
        assert np.array_equal(anasig.array_annotations['channel_names'],
                        ['ch0', 'ch6'])


class TestSpikeTrainProxy(BaseProxyTest):
    def test_SpikeTrainProxy(self):
        proxy_sptr = SpikeTrainProxy(rawio=self.reader, spike_channel_index=0,
                        block_index=0, seg_index=0)

        assert proxy_sptr.name == 'unit0'
        assert proxy_sptr.t_start == 0 * pq.s
        assert proxy_sptr.t_stop == 10 * pq.s
        assert proxy_sptr.shape == (20,)
        assert proxy_sptr.left_sweep == 0.002 * pq.s
        assert proxy_sptr.sampling_rate == 10 * pq.kHz

        # full load
        full_sptr = proxy_sptr.load(time_slice=None)
        assert isinstance(full_sptr, SpikeTrain)
        assert_same_attributes(proxy_sptr, full_sptr)
        assert full_sptr.shape == proxy_sptr.shape

        # slice time
        sptr = proxy_sptr.load(time_slice=(250 * pq.ms, 500 * pq.ms))
        assert sptr.t_start == .25 * pq.s
        assert sptr.t_stop == .5 * pq.s
        assert sptr.shape == (6,)
        assert_same_attributes(proxy_sptr.time_slice(250 * pq.ms, 500 * pq.ms), sptr)

        # buggy time slice
        with self.assertRaises(AssertionError):
            sptr = proxy_sptr.load(time_slice=(2. * pq.s, 15 * pq.s))
        sptr = proxy_sptr.load(time_slice=(2. * pq.s, 15 * pq.s), strict_slicing=False)
        assert sptr.t_stop == 10 * pq.s

        # magnitude mode rescaled
        sptr_float = proxy_sptr.load(magnitude_mode='rescaled')
        assert sptr_float.dtype == 'float64'
        assert sptr_float.units == pq.s

        # magnitude mode raw
        # TODO when raw mode implemented
        # sptr_int = proxy_sptr.load(magnitude_mode='raw')
        # assert sptr_int.dtype=='int64'
        # assert sptr_int.units==pq.CompoundUnit('1/10000*s')

        # assert_arrays_almost_equal(sptr_float, sptr_int.rescale('s'), 1e-9)

        # Without waveforms
        sptr = proxy_sptr.load(load_waveforms=False)
        assert sptr.waveforms is None

        # With waveforms
        sptr = proxy_sptr.load(load_waveforms=True, magnitude_mode='rescaled')
        assert sptr.waveforms is not None
        assert sptr.waveforms.shape == (20, 1, 50)
        assert sptr.waveforms.units == 1 * pq.uV

        # slice waveforms
        sptr = proxy_sptr.load(load_waveforms=True, time_slice=(250 * pq.ms, 500 * pq.ms))
        assert sptr.waveforms.shape == (6, 1, 50)

        # test array_annotations
        assert '__array_annotations__' not in proxy_sptr.annotations
        assert 'amplitudes' in proxy_sptr.array_annotations


class TestEventProxy(BaseProxyTest):
    def test_EventProxy(self):
        proxy_event = EventProxy(rawio=self.reader, event_channel_index=0,
                        block_index=0, seg_index=0)

        assert proxy_event.name == 'Some events'
        assert proxy_event.shape == (6,)

        # full load
        full_event = proxy_event.load(time_slice=None)
        assert isinstance(full_event, Event)
        assert_same_attributes(proxy_event, full_event, exclude=('times', 'labels'))
        assert full_event.shape == proxy_event.shape

        # slice time
        event = proxy_event.load(time_slice=(1 * pq.s, 2 * pq.s))
        assert event.shape == (2,)
        assert event.labels.shape == (2,)
        assert_same_attributes(proxy_event.time_slice(1 * pq.s, 2 * pq.s), event)

        # buggy time slice
        with self.assertRaises(AssertionError):
            event = proxy_event.load(time_slice=(2 * pq.s, 15 * pq.s))
        event = proxy_event.load(time_slice=(2 * pq.s, 15 * pq.s), strict_slicing=False)

        # test annotations/array_annotations
        assert '__array_annotations__' not in proxy_event.annotations
        assert 'nickname' in proxy_event.annotations
        assert 'button' in proxy_event.array_annotations


class TestEpochProxy(BaseProxyTest):
    def test_EpochProxy(self):
        proxy_epoch = EpochProxy(rawio=self.reader, event_channel_index=1,
                        block_index=0, seg_index=0)

        assert proxy_epoch.name == 'Some epochs'
        assert proxy_epoch.shape == (10,)

        # full load
        full_epoch = proxy_epoch.load(time_slice=None)
        assert isinstance(full_epoch, Epoch)
        assert_same_attributes(proxy_epoch, full_epoch, exclude=('times', 'labels', 'durations'))
        assert full_epoch.shape == proxy_epoch.shape

        # slice time
        epoch = proxy_epoch.load(time_slice=(1 * pq.s, 4 * pq.s))
        assert epoch.shape == (3,)
        assert epoch.labels.shape == (3,)
        assert epoch.durations.shape == (3,)
        assert_same_attributes(proxy_epoch.time_slice(1 * pq.s, 4 * pq.s), epoch)

        # buggy time slice
        with self.assertRaises(AssertionError):
            epoch = proxy_epoch.load(time_slice=(2 * pq.s, 15 * pq.s))
        epoch = proxy_epoch.load(time_slice=(2 * pq.s, 15 * pq.s), strict_slicing=False)

        # test annotations/array_annotations
        assert '__array_annotations__' not in proxy_epoch.annotations
        assert 'nickname' in proxy_epoch.annotations


class TestSegmentWithProxy(BaseProxyTest):
    def test_segment_with_proxy(self):
        seg = Segment()

        proxy_anasig = AnalogSignalProxy(rawio=self.reader,
                        stream_index=0, inner_stream_channels=None,
                        block_index=0, seg_index=0,)
        seg.analogsignals.append(proxy_anasig)

        proxy_sptr = SpikeTrainProxy(rawio=self.reader, spike_channel_index=0,
                        block_index=0, seg_index=0)
        seg.spiketrains.append(proxy_sptr)

        proxy_event = EventProxy(rawio=self.reader, event_channel_index=0,
                        block_index=0, seg_index=0)
        seg.events.append(proxy_event)

        proxy_epoch = EpochProxy(rawio=self.reader, event_channel_index=1,
                        block_index=0, seg_index=0)
        seg.epochs.append(proxy_epoch)


if __name__ == "__main__":
    unittest.main()
