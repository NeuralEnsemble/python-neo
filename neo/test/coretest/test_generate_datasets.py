'''
Test to  make sure generated datasets are sane.
'''

import unittest

from datetime import datetime

import numpy as np
import quantities as pq

from neo.core import (class_by_name, Block, Segment, AnalogSignal,
                      IrregularlySampledSignal, SpikeTrain, Event, Epoch)
from neo.core.dataobject import DataObject
from neo.test.generate_datasets import (generate_one_simple_block, generate_one_simple_segment,
                                        generate_from_supported_objects,
                                        TEST_ANNOTATIONS)
from neo.test.tools import assert_arrays_equal, assert_neo_object_is_compliant


class Test__generate_one_simple_segment(unittest.TestCase):
    def test_defaults(self):
        res = generate_one_simple_segment()

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.analogsignals), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 0)
        self.assertEqual(len(res.events), 0)
        self.assertEqual(len(res.epochs), 0)

    def test_all_supported(self):
        objects = [Block, Segment, AnalogSignal, IrregularlySampledSignal,
                   SpikeTrain, Event, Epoch]

        res = generate_one_simple_segment(supported_objects=objects)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.analogsignals), 4)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 6)
        self.assertEqual(len(res.events), 3)
        self.assertEqual(len(res.epochs), 2)

    def test_half_supported(self):
        objects = [Segment, IrregularlySampledSignal, SpikeTrain, Epoch]

        res = generate_one_simple_segment(supported_objects=objects)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.analogsignals), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 6)
        self.assertEqual(len(res.events), 0)
        self.assertEqual(len(res.epochs), 2)

    def test_all_without_block(self):
        objects = [Segment, AnalogSignal, IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch]

        res = generate_one_simple_segment(supported_objects=objects)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.analogsignals), 4)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 6)
        self.assertEqual(len(res.events), 3)
        self.assertEqual(len(res.epochs), 2)

    def test_all_without_segment_valueerror(self):
        objects = [Block, AnalogSignal, IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch]

        self.assertRaises(ValueError, generate_one_simple_segment, supported_objects=objects)


class Test__generate_one_simple_block(unittest.TestCase):
    def test_defaults(self):
        res = generate_one_simple_block()

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 0)

    def test_all_supported(self):
        objects = [Block, Segment, AnalogSignal, IrregularlySampledSignal,
                   SpikeTrain, Event, Epoch]

        res = generate_one_simple_block(supported_objects=objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 3)
        seg1, seg2, seg3 = res.segments

        self.assertEqual(len(seg1.analogsignals), 4)
        self.assertEqual(len(seg1.irregularlysampledsignals), 0)
        self.assertEqual(len(seg1.spiketrains), 6)
        self.assertEqual(len(seg1.events), 3)
        self.assertEqual(len(seg1.epochs), 2)

        self.assertEqual(len(seg2.analogsignals), 4)
        self.assertEqual(len(seg2.irregularlysampledsignals), 0)
        self.assertEqual(len(seg2.spiketrains), 6)
        self.assertEqual(len(seg2.events), 3)
        self.assertEqual(len(seg2.epochs), 2)

        self.assertEqual(len(seg3.analogsignals), 4)
        self.assertEqual(len(seg3.irregularlysampledsignals), 0)
        self.assertEqual(len(seg3.spiketrains), 6)
        self.assertEqual(len(seg3.events), 3)
        self.assertEqual(len(seg3.epochs), 2)

    def test_half_supported(self):
        objects = [Block, Segment, IrregularlySampledSignal, SpikeTrain, Epoch]

        res = generate_one_simple_block(supported_objects=objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 3)
        seg1, seg2, seg3 = res.segments

        self.assertEqual(len(seg1.analogsignals), 0)
        self.assertEqual(len(seg1.irregularlysampledsignals), 0)
        self.assertEqual(len(seg1.spiketrains), 6)
        self.assertEqual(len(seg1.events), 0)
        self.assertEqual(len(seg1.epochs), 2)

        self.assertEqual(len(seg2.analogsignals), 0)
        self.assertEqual(len(seg2.irregularlysampledsignals), 0)
        self.assertEqual(len(seg2.spiketrains), 6)
        self.assertEqual(len(seg2.events), 0)
        self.assertEqual(len(seg2.epochs), 2)

        self.assertEqual(len(seg3.analogsignals), 0)
        self.assertEqual(len(seg3.irregularlysampledsignals), 0)
        self.assertEqual(len(seg3.spiketrains), 6)
        self.assertEqual(len(seg3.events), 0)
        self.assertEqual(len(seg3.epochs), 2)

    def test_all_without_segment(self):
        objects = [Block, AnalogSignal, IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch]

        res = generate_one_simple_block(supported_objects=objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 0)

    def test_all_without_block_valueerror(self):
        objects = [Segment, AnalogSignal, IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch]

        self.assertRaises(ValueError, generate_one_simple_block, supported_objects=objects)


class Test__generate_from_supported_objects(unittest.TestCase):
    def test_no_object_valueerror(self):
        objects = []

        self.assertRaises(ValueError, generate_from_supported_objects, objects)

    def test_all(self):
        objects = [Block, Segment, AnalogSignal, IrregularlySampledSignal,
                   SpikeTrain, Event, Epoch]

        res = generate_from_supported_objects(objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 3)
        seg1, seg2, seg3 = res.segments

        self.assertEqual(len(seg1.analogsignals), 4)
        self.assertEqual(len(seg1.irregularlysampledsignals), 0)
        self.assertEqual(len(seg1.spiketrains), 6)
        self.assertEqual(len(seg1.events), 3)
        self.assertEqual(len(seg1.epochs), 2)

        self.assertEqual(len(seg2.analogsignals), 4)
        self.assertEqual(len(seg2.irregularlysampledsignals), 0)
        self.assertEqual(len(seg2.spiketrains), 6)
        self.assertEqual(len(seg2.events), 3)
        self.assertEqual(len(seg2.epochs), 2)

        self.assertEqual(len(seg3.analogsignals), 4)
        self.assertEqual(len(seg3.irregularlysampledsignals), 0)
        self.assertEqual(len(seg3.spiketrains), 6)
        self.assertEqual(len(seg3.events), 3)
        self.assertEqual(len(seg3.epochs), 2)

    def test_block(self):
        objects = [Block]

        res = generate_from_supported_objects(objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 0)

    def test_block_segment(self):
        objects = [Segment, Block]

        res = generate_from_supported_objects(objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 3)
        seg1, seg2, seg3 = res.segments

        self.assertEqual(len(seg1.analogsignals), 0)
        self.assertEqual(len(seg1.irregularlysampledsignals), 0)
        self.assertEqual(len(seg1.spiketrains), 0)
        self.assertEqual(len(seg1.events), 0)
        self.assertEqual(len(seg1.epochs), 0)

        self.assertEqual(len(seg2.analogsignals), 0)
        self.assertEqual(len(seg2.irregularlysampledsignals), 0)
        self.assertEqual(len(seg2.spiketrains), 0)
        self.assertEqual(len(seg2.events), 0)
        self.assertEqual(len(seg2.epochs), 0)

        self.assertEqual(len(seg3.analogsignals), 0)
        self.assertEqual(len(seg3.irregularlysampledsignals), 0)
        self.assertEqual(len(seg3.spiketrains), 0)
        self.assertEqual(len(seg3.events), 0)
        self.assertEqual(len(seg3.epochs), 0)

    def test_segment(self):
        objects = [Segment]

        res = generate_from_supported_objects(objects)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.analogsignals), 0)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 0)
        self.assertEqual(len(res.events), 0)
        self.assertEqual(len(res.epochs), 0)

    def test_all_without_block(self):
        objects = [Segment, AnalogSignal, IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch]

        res = generate_from_supported_objects(objects)

        self.assertTrue(isinstance(res, Segment))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.analogsignals), 4)
        self.assertEqual(len(res.irregularlysampledsignals), 0)
        self.assertEqual(len(res.spiketrains), 6)
        self.assertEqual(len(res.events), 3)
        self.assertEqual(len(res.epochs), 2)

    def test_all_without_segment(self):
        objects = [Block, AnalogSignal, IrregularlySampledSignal, SpikeTrain,
                   Event, Epoch]

        res = generate_from_supported_objects(supported_objects=objects)

        self.assertTrue(isinstance(res, Block))
        assert_neo_object_is_compliant(res)

        self.assertEqual(len(res.segments), 0)
