"""
Tests of neo.io.elphyo
"""

import struct
import unittest

import quantities as pq

from neo import AnalogSignal

from neo.io import ElphyIO
from neo.io.elphyio import _serialize_analog_signal_channels
from neo.test.iotest.common_io_test import BaseTestIO


class TestElphyIO(BaseTestIO, unittest.TestCase):
    ioclass = ElphyIO
    entities_to_download = ["elphy"]
    entities_to_test = [
        "elphy/DATA1.DAT",
        "elphy/ElphyExample.DAT",
        "elphy/ElphyExample_Mode1.dat",
        "elphy/ElphyExample_Mode2.dat",
        "elphy/ElphyExample_Mode3.dat",
    ]

    def test_read_data(self):
        for filename in self.entities_to_test:
            io = ElphyIO(self.get_local_path(filename))
            bl = io.read_block()

            self.assertTrue(len(bl.segments) > 0)
            # ensure that at least one data object is generated for each file
            self.assertTrue(any(list(bl.segments[0].size.values())))


class TestElphyIOSerialization(unittest.TestCase):
    def test_analog_signal_serialization_keeps_each_channel_unit(self):
        signals = [
            AnalogSignal([[1], [2]] * pq.mV, sampling_rate=1000 * pq.Hz),
            AnalogSignal([[1], [2]] * pq.V, sampling_rate=1000 * pq.Hz),
        ]

        nbchan, nbpt, serialized, units = _serialize_analog_signal_channels(signals)

        self.assertEqual((nbchan, nbpt), (2, 2))
        self.assertEqual(units, [pq.mV, pq.V])
        self.assertEqual(struct.unpack("<4f", serialized), (1.0, 1.0, 2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
