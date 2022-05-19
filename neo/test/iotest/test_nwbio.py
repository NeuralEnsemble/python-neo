#
"""
Tests of neo.io.nwbio
"""

from __future__ import unicode_literals, print_function, division, absolute_import

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
import nwbinspector
from nwbinspector import inspect_nwb
from pynwb.file import Subject


@unittest.skipUnless(HAVE_PYNWB, "requires pynwb")
class TestNWBIO(BaseTestIO, unittest.TestCase):
#class TestNWBIO(unittest.TestCase):
    ioclass = NWBIO
    entities_to_download = ["nwb"]
    entities_to_test = [
        # Files from Allen Institute:
        "nwb/H19.29.141.11.21.01.nwb",  # 7 MB
    ]

    def test_roundtrip(self):

        subject_annotations = {"nwb:subject_id": "012",
                                "nwb:age": "P90D",
                                "nwb:description": "mouse 5",
                                "nwb:species": "Mus musculus",
                                "nwb:sex": "M",
                              }
        annotations = {
            "session_start_time": datetime.now(),
            "subject": subject_annotations,
        }
        # Define Neo blocks
        bl0 = Block(name='First block',
                    experimenter="Experimenter's name",
                    experiment_description="Experiment description",
                    institution="Institution",
                    **annotations)
        bl1 = Block(name='Second block',
                    experimenter="Experimenter's name",
                    experiment_description="Experiment description",
                    institution="Institution",
                    **annotations)
        bl2 = Block(name='Third block',
                    experimenter="Experimenter's name",
                    experiment_description="Experiment description",
                    institution="Institution",
                    **annotations)
        original_blocks = [bl0, bl1, bl2]

        num_seg = 4  # number of segments
        num_chan = 6  # number of channels

        for j, blk in enumerate(original_blocks):

            for ind in range(num_seg):  # number of Segments
                seg = Segment(index=ind)
                seg.block = blk
                blk.segments.append(seg)

            for i, seg in enumerate(blk.segments):# AnalogSignal objects

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
                # 1 Neo IrregularlySampledSignals
                d = IrregularlySampledSignal([0.01, 0.03, 0.12]*pq.s, 
                                            [[4, 5], [5, 4], [6, 3]]*pq.nA)
                # 2 Neo SpikeTrains
                train = SpikeTrain(times=[1, 2, 3] * pq.s, t_start=1.0, t_stop=10.0)
                train2 = SpikeTrain(times=[4, 5, 6] * pq.s, t_stop=10.0)
                # todo: add waveforms

                seg.spiketrains.append(train)
                seg.spiketrains.append(train2)
                seg.analogsignals.append(a)
                seg.analogsignals.append(b)
                seg.analogsignals.append(c)
                seg.irregularlysampledsignals.append(d)
                a.segment = seg
                b.segment = seg
                c.segment = seg
                d.segment = seg
                train.segment = seg
                train2.segment = seg

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

        original_spiketrain_131 = original_blocks[1].segments[1].spiketrains[1]
        retrieved_spiketrain_131 = retrieved_blocks[1].segments[1].spiketrains[1]
        for attr_name in ("name", "t_start", "t_stop"):
            retrieved_attribute = getattr(retrieved_spiketrain_131, attr_name)
            original_attribute = getattr(original_spiketrain_131, attr_name)
            self.assertEqual(retrieved_attribute, original_attribute)
        assert_array_equal(retrieved_spiketrain_131.times.rescale('ms').magnitude,
                           original_spiketrain_131.times.rescale('ms').magnitude)

        #NWBInspector : Inspect NWB files for compliance with NWB Best Practices.
        results_roundtrip = list(inspect_nwb(nwbfile_path=test_file_name))
        #print("results test_roundtrip NWBInspector = ", results_roundtrip)

        os.remove(test_file_name)

    def test_roundtrip_with_annotations(self):
        #Test with NWB-specific annotations

        subject_annotations = {"nwb:subject_id": "011",
                                "nwb:age": "P90D",
                                "nwb:description": "mouse 4",
                                "nwb:species": "Mus musculus",
                                "nwb:sex": "F",
                              }
        original_block = Block(name="experiment", session_start_time=datetime.now(),
                               experimenter="Experimenter's name",
                               experiment_description="Experiment description",
                               institution="Institution",
                               subject=subject_annotations)
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

        #NWBInspector : Inspect NWB files 
        #for compliance with NWB Best Practices.
        results_roundtrip_with_annotations = list(inspect_nwb(nwbfile_path=test_file_name))
        #print("results test_roundtrip_with_annotations NWBInspector = ", results_roundtrip_with_annotations)
        
        os.remove(test_file_name)

    def test_roundtrip_with_not_constant_sampling_rate(self):
        # To check NWB Inspector for Epoch and Event
        # NWB Epochs = Neo Segments
        # Should work for multiple segments, not for multiple blocks
        # The specific test for Time Series not having a constant sample rate
        # For epochs and events

        annotations = {
            "session_start_time": datetime.now(),
        }
        # Define Neo blocks
        bl0 = Block(name='First block',
                    experimenter="Experimenter's name",
                    experiment_description="Experiment description",
                    institution="Institution",
                    **annotations)
        original_blocks = [bl0]

        num_seg = 2  # number of segments
        num_chan = 3  # number of channels

        for j, blk in enumerate(original_blocks):

            for ind in range(num_seg): # number of Segments
                seg = Segment(index=ind)
                seg.block = blk
                blk.segments.append(seg)

            for i, seg in enumerate(blk.segments):# AnalogSignal objects
            
                a = AnalogSignal(name='Signal_a %s' % (seg.name),
                                 signal=np.random.randn(44, num_chan) * pq.nA,
                                 sampling_rate=10 * pq.kHz,
                                 t_start=50 * pq.ms)

                epc = Epoch(times [0 + i * ind, 10 + i * ind, 33 + i * ind]*pq.s,
                    durations=[10, 5, 7] * pq.s,
                    labels=np.array(['btn0', 'btn1', 'btn2'], dtype='U'))

                epc2 = Epoch(times=[0.1 + i * ind, 30 + i * ind, 61 + i * ind]*pq.s,
                     durations=[10, 5, 7] * pq.s,
                     labels=np.array(['btn4', 'btn5', 'btn6']))

                evt = Event(name='Event',
                            times=[0.01 + i * ind, 11 + i * ind, 33 + i * ind]*pq.s,
                            labels=np.array(['ev0', 'ev1', 'ev2']))

                seg.epochs.append(epc)
                seg.epochs.append(epc2)
                seg.events.append(evt)
                seg.analogsignals.append(a)

                a.segment = seg
                epc.segment = seg
                epc2.segment = seg
                evt.segment = seg

        #write to file
        test_file_name = "test_round_trip_with_not_constant_sampling_rate.nwb"
        iow = NWBIO(filename=test_file_name, mode='w')
        iow.write_all_blocks(original_blocks)

        ior = NWBIO(filename=test_file_name, mode='r')
        retrieved_blocks = ior.read_all_blocks()

        self.assertEqual(len(retrieved_blocks), 1)

        #NWBInspector : Inspect NWB files 
        #for compliance with NWB Best Practices.
        results_roundtrip_specific_for_epochs = list(inspect_nwb(nwbfile_path=test_file_name))
        #print("results test_roundtrip_specific_for_epochs NWBInspector = ", results_roundtrip_specific_for_epochs)

        os.remove(test_file_name)


if __name__ == "__main__":
    if HAVE_PYNWB:
        print("pynwb.__version__ = ", pynwb.__version__)
    unittest.main()
