"""
Tests of the neo.core.view.ChannelView class and related functions
"""


import unittest

import numpy as np
import quantities as pq
from numpy.testing import assert_array_equal

from neo.core.analogsignal import AnalogSignal
from neo.core.irregularlysampledsignal import IrregularlySampledSignal
from neo.core.view import ChannelView


class TestView(unittest.TestCase):

    def setUp(self):
        self.test_data = np.random.rand(100, 8) * pq.mV
        channel_names = np.array(["a", "b", "c", "d", "e", "f", "g", "h"])
        self.test_signal = AnalogSignal(self.test_data,
                                        sampling_period=0.1 * pq.ms,
                                        name="test signal",
                                        description="this is a test signal",
                                        array_annotations={"channel_names": channel_names},
                                        attUQoLtUaE=42)

    def test_create_integer_index(self):
        view = ChannelView(self.test_signal, [1, 2, 5, 7],
                    name="view of test signal",
                    description="this is a view of a test signal",
                    array_annotations={"something": np.array(["A", "B", "C", "D"])},
                    sLaTfat="fish")

        assert view.obj is self.test_signal
        assert_array_equal(view.index, np.array([1, 2, 5, 7]))
        self.assertEqual(view.shape, (100, 4))
        self.assertEqual(view.name, "view of test signal")
        self.assertEqual(view.annotations["sLaTfat"], "fish")

    def test_create_boolean_index(self):
        view1 = ChannelView(self.test_signal, [1, 2, 5, 7])
        view2 = ChannelView(self.test_signal, np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=bool))
        assert_array_equal(view1.index, view2.index)
        self.assertEqual(view1.shape, view2.shape)

    def test_resolve(self):
        view = ChannelView(self.test_signal, [1, 2, 5, 7],
                    name="view of test signal",
                    description="this is a view of a test signal",
                    array_annotations={"something": np.array(["A", "B", "C", "D"])},
                    sLaTfat="fish")
        signal2 = view.resolve()
        self.assertIsInstance(signal2, AnalogSignal)
        self.assertEqual(signal2.shape, (100, 4))
        for attr in ('name', 'description', 'sampling_period', 'units'):
            self.assertEqual(getattr(self.test_signal, attr), getattr(signal2, attr))
        assert_array_equal(signal2.array_annotations["channel_names"],
                           np.array(["b", "c", "f", "h"]))
        assert_array_equal(self.test_data[:, [1, 2, 5, 7]], signal2.magnitude)
