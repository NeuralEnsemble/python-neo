# -*- coding: utf-8 -*-
"""
Tests of the neo.utils module
"""

import unittest

import numpy as np
import quantities as pq
from neo.rawio.examplerawio import ExampleRawIO
from neo.io.proxyobjects import (AnalogSignalProxy, SpikeTrainProxy,
                EventProxy, EpochProxy)

from neo.core.dataobject import ArrayDict
from neo.core import (Segment, AnalogSignal,
                      Epoch, Event, SpikeTrain)

from neo.test.tools import (assert_arrays_almost_equal,
                            assert_arrays_equal,
                            assert_neo_object_is_compliant,
                            assert_same_attributes)

from neo.utils import (get_events)


class BaseProxyTest(unittest.TestCase):
    def setUp(self):
        self.reader = ExampleRawIO(filename='my_filename.fake')
        self.reader.parse_header()


class TestUtilsWithoutProxyObjects(unittest.TestCase):
    def test__get_events(self):
        event = Event(times=[0.5, 10.0, 25.2] * pq.s)
        event.annotate(event_type='trial start')
        event.array_annotate(trial_id=[1, 2, 3])
        seg = Segment()
        seg.events = [event]

        # test getting the whole event via annotation
        result = get_events(seg, properties={'event_type': 'trial start'})

        self.assertEqual(len(result), 1)

        result = result[0]

        assert_same_attributes(result, event)

        # test getting an empty list by searching for a non-existent property
        empty = get_events(seg, properties={'event_type': 'trial stop'})

        self.assertEqual(len(empty), 0)

        # test getting only one event time
        trial_2 = get_events(seg, properties={'trial_id': 2})

        self.assertEqual(len(trial_2), 1)

        trial_2 = trial_2[0]

        self.assertEqual(event.name, trial_2.name)
        self.assertEqual(event.description, trial_2.description)
        self.assertEqual(event.file_origin, trial_2.file_origin)
        self.assertEqual(event.annotations['event_type'], trial_2.annotations['event_type'])
        assert_arrays_equal(trial_2.array_annotations['trial_id'], np.array([2]))
        self.assertIsInstance(trial_2.array_annotations, ArrayDict)

        # test getting more than one event time
        trials_1_2 = get_events(seg, properties={'trial_id': [1, 2]})

        self.assertEqual(len(trials_1_2), 1)

        trials_1_2 = trials_1_2[0]

        self.assertEqual(event.name, trials_1_2.name)
        self.assertEqual(event.description, trials_1_2.description)
        self.assertEqual(event.file_origin, trials_1_2.file_origin)
        self.assertEqual(event.annotations['event_type'], trials_1_2.annotations['event_type'])
        assert_arrays_equal(trials_1_2.array_annotations['trial_id'], np.array([1, 2]))
        self.assertIsInstance(trials_1_2.array_annotations, ArrayDict)

        # TODO: block with multiple segments, multiple events
