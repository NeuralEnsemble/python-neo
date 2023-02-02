#
"""
Tests of neo.io.nwbio
"""

import os
import unittest
from datetime import datetime

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
from neo.test.iotest.common_io_test import BaseTestIO
from neo.core import AnalogSignal, SpikeTrain, Event, Epoch, IrregularlySampledSignal, Segment, \
    Block

from neo.rawio.examplerawio import ExampleRawIO
from neo.io.proxyobjects import (AnalogSignalProxy, SpikeTrainProxy, EventProxy, EpochProxy)

try:
    import pynwb
    from neo.io.nwbio import NWBIO

    HAVE_PYNWB = True
except (ImportError, SyntaxError):
    NWBIO = None
    HAVE_PYNWB = False
import quantities as pq
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


@unittest.skipUnless(HAVE_PYNWB, "requires pynwb")
class TestNWBIO(BaseTestIO, unittest.TestCase):
    ioclass = NWBIO
    entities_to_download = ["nwb"]
    entities_to_test = [
        # Files from Allen Institute:
        "nwb/H19.29.141.11.21.01.nwb",  # 7 MB
    ]

    def test_roundtrip(self):

        annotations = {
            "session_start_time": datetime.now()
        }
        # Define Neo blocks
        bl0 = Block(name='First block', **annotations)
        bl1 = Block(name='Second block', **annotations)
        bl2 = Block(name='Third block', **annotations)
        original_blocks = [bl0, bl1, bl2]

        num_seg = 4  # number of segments
        num_chan = 3  # number of channels

        for blk in original_blocks:

            for ind in range(num_seg):  # number of Segments
                seg = Segment(index=ind)
                seg.block = blk
                blk.segments.append(seg)

            for seg in blk.segments:  # AnalogSignal objects

                # 3 Neo AnalogSignals
                a = AnalogSignal(name='Signal_a %s' % (seg.name),
                                 signal=np.random.randn(44, num_chan) * pq.nA,
                                 sampling_rate=10 * pq.kHz,
                                 t_start=50 * pq.ms)
                b = AnalogSignal(name='Signal_b %s' % (seg.name),
                                 signal=np.random.randn(64, num_chan) * pq.mV,
                                 sampling_rate=8 * pq.kHz,
                                 t_start=40 * pq.ms)
                c = AnalogSignal(name='Signal_c %s' % (seg.name),
                                 signal=np.random.randn(33, num_chan) * pq.uA,
                                 sampling_rate=10 * pq.kHz,
                                 t_start=120 * pq.ms)
                # 2 Neo IrregularlySampledSignals
                d = IrregularlySampledSignal(np.arange(7.0) * pq.ms,
                                             np.random.randn(7, num_chan) * pq.mV)

                # 2 Neo SpikeTrains
                train = SpikeTrain(times=[1, 2, 3] * pq.s, t_start=1.0, t_stop=10.0)
                train2 = SpikeTrain(times=[4, 5, 6] * pq.s, t_stop=10.0)
                # todo: add waveforms

                # 1 Neo Event
                evt = Event(name='Event',
                            times=np.arange(0, 30, 10) * pq.ms,
                            labels=np.array(['ev0', 'ev1', 'ev2']))

                # 2 Neo Epochs
                epc = Epoch(times=np.arange(0, 30, 10) * pq.s,
                            durations=[10, 5, 7] * pq.ms,
                            labels=np.array(['btn0', 'btn1', 'btn2']))

                epc2 = Epoch(times=np.arange(10, 40, 10) * pq.s,
                             durations=[9, 3, 8] * pq.ms,
                             labels=np.array(['btn3', 'btn4', 'btn5']))

                seg.spiketrains.append(train)
                seg.spiketrains.append(train2)

                seg.epochs.append(epc)
                seg.epochs.append(epc2)

                seg.analogsignals.append(a)
                seg.analogsignals.append(b)
                seg.analogsignals.append(c)
                seg.irregularlysampledsignals.append(d)
                seg.events.append(evt)
                a.segment = seg
                b.segment = seg
                c.segment = seg
                d.segment = seg
                evt.segment = seg
                train.segment = seg
                train2.segment = seg
                epc.segment = seg
                epc2.segment = seg

        # write to file
        test_file_name = "test_round_trip.nwb"
        iow = NWBIO(filename=test_file_name, mode='w')
        iow.write_all_blocks(original_blocks)

        ior = NWBIO(filename=test_file_name, mode='r')
        retrieved_blocks = ior.read_all_blocks()

        self.assertEqual(len(retrieved_blocks), 3)
        self.assertEqual(len(retrieved_blocks[2].segments), num_seg)

        original_signal_22b = original_blocks[2].segments[2].analogsignals[1]
        retrieved_signal_22b = retrieved_blocks[2].segments[2].analogsignals[1]
        for attr_name in ("name", "units", "sampling_rate", "t_start"):
            retrieved_attribute = getattr(retrieved_signal_22b, attr_name)
            original_attribute = getattr(original_signal_22b, attr_name)
            self.assertEqual(retrieved_attribute, original_attribute)
        assert_array_equal(retrieved_signal_22b.magnitude, original_signal_22b.magnitude)

        original_issignal_22d = original_blocks[2].segments[2].irregularlysampledsignals[0]
        retrieved_issignal_22d = retrieved_blocks[2].segments[2].irregularlysampledsignals[0]
        for attr_name in ("name", "units", "t_start"):
            retrieved_attribute = getattr(retrieved_issignal_22d, attr_name)
            original_attribute = getattr(original_issignal_22d, attr_name)
            self.assertEqual(retrieved_attribute, original_attribute)
        assert_array_equal(retrieved_issignal_22d.times.rescale('ms').magnitude,
                           original_issignal_22d.times.rescale('ms').magnitude)
        assert_array_equal(retrieved_issignal_22d.magnitude, original_issignal_22d.magnitude)

        original_event_11 = original_blocks[1].segments[1].events[0]
        retrieved_event_11 = retrieved_blocks[1].segments[1].events[0]
        for attr_name in ("name",):
            retrieved_attribute = getattr(retrieved_event_11, attr_name)
            original_attribute = getattr(original_event_11, attr_name)
            self.assertEqual(retrieved_attribute, original_attribute)
        assert_array_equal(retrieved_event_11.rescale('ms').magnitude,
                           original_event_11.rescale('ms').magnitude)
        assert_array_equal(retrieved_event_11.labels, original_event_11.labels)

        original_spiketrain_131 = original_blocks[1].segments[1].spiketrains[1]
        retrieved_spiketrain_131 = retrieved_blocks[1].segments[1].spiketrains[1]
        for attr_name in ("name", "t_start", "t_stop"):
            retrieved_attribute = getattr(retrieved_spiketrain_131, attr_name)
            original_attribute = getattr(original_spiketrain_131, attr_name)
            self.assertEqual(retrieved_attribute, original_attribute)
        assert_array_equal(retrieved_spiketrain_131.times.rescale('ms').magnitude,
                           original_spiketrain_131.times.rescale('ms').magnitude)

        original_epoch_11 = original_blocks[1].segments[1].epochs[0]
        retrieved_epoch_11 = retrieved_blocks[1].segments[1].epochs[0]
        for attr_name in ("name",):
            retrieved_attribute = getattr(retrieved_epoch_11, attr_name)
            original_attribute = getattr(original_epoch_11, attr_name)
            self.assertEqual(retrieved_attribute, original_attribute)
        assert_array_equal(retrieved_epoch_11.rescale('ms').magnitude,
                           original_epoch_11.rescale('ms').magnitude)
        assert_allclose(retrieved_epoch_11.durations.rescale('ms').magnitude,
                        original_epoch_11.durations.rescale('ms').magnitude)
        assert_array_equal(retrieved_epoch_11.labels, original_epoch_11.labels)
        os.remove(test_file_name)

    def test_roundtrip_with_annotations(self):
        # test with NWB-specific annotations

        original_block = Block(name="experiment", session_start_time=datetime.now())
        segment = Segment(name="session 1")
        original_block.segments.append(segment)
        segment.block = original_block

        electrode_annotations = {
            "name": "electrode #1",
            "description": "intracellular electrode",
            "device": {
                "name": "electrode #1"
            }
        }
        stimulus_annotations = {
            "nwb_group": "stimulus",
            "nwb_neurodata_type": ("pynwb.icephys", "CurrentClampStimulusSeries"),
            "nwb_electrode": electrode_annotations,
            "nwb:sweep_number": 1,
            "nwb:gain": 1.0
        }
        response_annotations = {
            "nwb_group": "acquisition",
            "nwb_neurodata_type": ("pynwb.icephys", "CurrentClampSeries"),
            "nwb_electrode": electrode_annotations,
            "nwb:sweep_number": 1,
            "nwb:gain": 1.0,
            "nwb:bias_current": 1e-12,
            "nwb:bridge_balance": 70e6,
            "nwb:capacitance_compensation": 1e-12
        }
        stimulus = AnalogSignal(np.random.randn(100, 1) * pq.nA,
                                sampling_rate=5 * pq.kHz,
                                t_start=50 * pq.ms,
                                name="stimulus",
                                **stimulus_annotations)
        response = AnalogSignal(np.random.randn(100, 1) * pq.mV,
                                sampling_rate=5 * pq.kHz,
                                t_start=50 * pq.ms,
                                name="response",
                                **response_annotations)
        segment.analogsignals = [stimulus, response]
        stimulus.segment = response.segment = segment

        test_file_name = "test_round_trip_with_annotations.nwb"
        iow = NWBIO(filename=test_file_name, mode='w')
        iow.write_all_blocks([original_block])

        nwbfile = pynwb.NWBHDF5IO(test_file_name, mode="r").read()

        self.assertIsInstance(nwbfile.acquisition[response.name], pynwb.icephys.CurrentClampSeries)
        self.assertIsInstance(nwbfile.stimulus[stimulus.name],
                              pynwb.icephys.CurrentClampStimulusSeries)
        self.assertEqual(nwbfile.acquisition[response.name].bridge_balance,
                         response_annotations["nwb:bridge_balance"])

        ior = NWBIO(filename=test_file_name, mode='r')
        retrieved_block = ior.read_all_blocks()[0]

        original_response = original_block.segments[0].filter(name=response.name)[0]
        retrieved_response = retrieved_block.segments[0].filter(name=response.name)[0]
        for attr_name in ("name", "units", "sampling_rate", "t_start"):
            retrieved_attribute = getattr(retrieved_response, attr_name)
            original_attribute = getattr(original_response, attr_name)
            self.assertEqual(retrieved_attribute, original_attribute)
        assert_array_equal(retrieved_response.magnitude, original_response.magnitude)

        os.remove(test_file_name)

    def test_write_proxy_objects(self):
        test_file_name = self.local_test_dir / "test_round_trip_with_annotations.nwb"

        # generate dummy IO as basis for ProxyObjects
        self.proxy_reader = ExampleRawIO(filename='my_filename.fake')
        self.proxy_reader.parse_header()

        # generate test structure with proxy objects
        original_block = Block(name='myblock', session_start_time=datetime.now().astimezone(),
                               session_description=str(test_file_name),
                               identifier=str(test_file_name))
        seg = Segment(name='mysegment')
        original_block.segments.append(seg)

        # create proxy objects
        proxy_anasig = AnalogSignalProxy(rawio=self.proxy_reader, stream_index=0,
                                         inner_stream_channels=None, block_index=0, seg_index=0,)
        proxy_anasig.segment = seg
        seg.analogsignals.append(proxy_anasig)

        proxy_sptr = SpikeTrainProxy(rawio=self.proxy_reader, spike_channel_index=0, block_index=0,
                                     seg_index=0)
        proxy_sptr.segment = seg
        seg.spiketrains.append(proxy_sptr)

        proxy_event = EventProxy(rawio=self.proxy_reader, event_channel_index=0, block_index=0,
                                 seg_index=0)
        proxy_event.segment = seg
        seg.events.append(proxy_event)

        proxy_epoch = EpochProxy(rawio=self.proxy_reader, event_channel_index=1, block_index=0,
                                 seg_index=0)
        proxy_epoch.segment = seg
        seg.epochs.append(proxy_epoch)

        original_block.create_relationship()

        iow = NWBIO(filename=test_file_name, mode='w')

        # writing data via proxyobjects
        iow.write_all_blocks([original_block])

        # checking written data
        ior = NWBIO(filename=test_file_name, mode='r')
        retrieved_block = ior.read_all_blocks()[0]

        for original_segment, retrieved_segment in zip(original_block.segments,
                                                       retrieved_block.segments):
            assert_array_equal(original_segment.analogsignals[0].load().magnitude,
                               retrieved_segment.analogsignals[0].magnitude)
            assert_array_equal(original_segment.spiketrains[0].load().magnitude,
                               retrieved_segment.spiketrains[0].magnitude)
            assert_array_equal(original_segment.events[0].load().magnitude,
                               retrieved_segment.events[0].magnitude)
            assert_array_equal(original_segment.epochs[0].load().magnitude,
                               retrieved_segment.epochs[0].magnitude)

if __name__ == "__main__":
    if HAVE_PYNWB:
        print("pynwb.__version__ = ", pynwb.__version__)
    unittest.main()
