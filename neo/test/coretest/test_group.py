"""
Tests of the neo.core.group.Group class and related functions
"""


import unittest

import numpy as np
import quantities as pq
from numpy.testing import assert_array_equal

from neo.core.analogsignal import AnalogSignal
from neo.core.irregularlysampledsignal import IrregularlySampledSignal
from neo.core.spiketrain import SpikeTrain
from neo.core.segment import Segment
from neo.core.view import ChannelView
from neo.core.group import Group


class TestGroup(unittest.TestCase):

    def setUp(self):
        test_data = np.random.rand(100, 8) * pq.mV
        channel_names = np.array(["a", "b", "c", "d", "e", "f", "g", "h"])
        self.test_signal = AnalogSignal(test_data,
                                        sampling_period=0.1 * pq.ms,
                                        name="test signal",
                                        description="this is a test signal",
                                        array_annotations={"channel_names": channel_names},
                                        attUQoLtUaE=42)
        self.test_view = ChannelView(self.test_signal, [1, 2, 5, 7],
                              name="view of test signal",
                              description="this is a view of a test signal",
                              array_annotations={"something": np.array(["A", "B", "C", "D"])},
                              sLaTfat="fish")
        self.test_spiketrains = [SpikeTrain(np.arange(100.0), units="ms", t_stop=200),
                                 SpikeTrain(np.arange(0.5, 100.5), units="ms", t_stop=200)]
        self.test_segment = Segment()
        self.test_segment.analogsignals.append(self.test_signal)
        self.test_segment.spiketrains.extend(self.test_spiketrains)

    def test_create_group(self):
        objects = [self.test_view, self.test_signal, self.test_segment]
        objects.extend(self.test_spiketrains)
        group = Group(objects)

        assert group.analogsignals[0] is self.test_signal
        assert group.spiketrains[0] is self.test_spiketrains[0]
        assert group.spiketrains[1] is self.test_spiketrains[1]
        assert group.channelviews[0] is self.test_view
        assert len(group.irregularlysampledsignals) == 0
        assert group.segments[0].analogsignals[0] is self.test_signal

    def test_create_empty_group(self):
        group = Group()

    def test_children(self):
        group = Group(self.test_spiketrains + [self.test_view]
                      + [self.test_signal] + [self.test_segment])

        # note: ordering is by class name for data children (AnalogSignal, SpikeTrain),
        #       then container children (Segment)
        assert group.children == (self.test_signal,
                                  *self.test_spiketrains,
                                  self.test_view,
                                  self.test_segment)

    def test_with_allowed_types(self):
        objects = [self.test_signal] + self.test_spiketrains
        group = Group(objects, allowed_types=(AnalogSignal, SpikeTrain))
        assert group.analogsignals[0] is self.test_signal
        assert group.spiketrains[0] is self.test_spiketrains[0]
        assert group.spiketrains[1] is self.test_spiketrains[1]
        self.assertRaises(TypeError, group.add, self.test_view)

    def test_walk(self):
        # set up nested groups
        parent = Group(name="0")
        children = (Group(name="00"), Group(name="01"), Group(name="02"))
        parent.add(*children)
        grandchildren = (
            (Group(name="000"), Group(name="001")),
            [],
            (Group(name="020"), Group(name="021"), Group(name="022"))
        )
        for child, gchildren in zip(children, grandchildren):
            child.add(*gchildren)

        flattened = list(parent.walk())
        target = [parent, children[0], *grandchildren[0]]
        target.extend([children[1], children[2], *grandchildren[2]])
        self.assertEqual(flattened,
                         target)
