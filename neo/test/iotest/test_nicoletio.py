"""
Tests of neo.io.exampleio
"""

import pathlib
import unittest

from neo.io.nicoletio import NicoletIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.iotest.tools import get_test_file_full_path
from neo.io.proxyobjects import AnalogSignalProxy, SpikeTrainProxy, EventProxy, EpochProxy
from neo import AnalogSignal, SpikeTrain

import quantities as pq
import numpy as np


class TestNicoletIO(
    BaseTestIO,
    unittest.TestCase,
):
    ioclass = NicoletIO
    entities_to_download = ["nicolet"]
    entities_to_test = [
        "nicolet/e_files/test.e",
    ]

    def setUp(self):
        super().setUp()
        for entity in self.entities_to_test:
            full_path = get_test_file_full_path(self.ioclass, filename=entity, directory=self.local_test_dir)
            pathlib.Path(full_path).touch()

    def tearDown(self) -> None:
        super().tearDown()
        for entity in self.entities_to_test:
            full_path = get_test_file_full_path(self.ioclass, filename=entity, directory=self.local_test_dir)
            #pathlib.Path(full_path).unlink(missing_ok=True)

    def test_read_segment_lazy(self):
        for entity in self.entities_to_test:
            r = NicoletIO(filename=get_test_file_full_path(self.ioclass, filename=entity, directory=self.local_test_dir))
            seg = r.read_segment(lazy=True)
            for ana in seg.analogsignals:
                assert isinstance(ana, AnalogSignalProxy)
                ana = ana.load()
                assert isinstance(ana, AnalogSignal)

            seg = r.read_segment(lazy=False)
            for anasig in seg.analogsignals:
                assert isinstance(ana, AnalogSignal)
                self.assertNotEqual(anasig.size, 0)

            # annotations
            for anasig in seg.analogsignals:
                assert anasig.name is not None
            for ev in seg.events:
                assert ev.name is not None
            for ep in seg.epochs:
                assert ep.name is not None

    def test_read_block(self):
        for entity in self.entities_to_test:
            r = NicoletIO(filename=get_test_file_full_path(self.ioclass, filename=entity, directory=self.local_test_dir))
            bl = r.read_block(lazy=True)

    def test_read_segment_with_time_slice(self):
        for entity in self.entities_to_test:
            r = NicoletIO(filename=get_test_file_full_path(self.ioclass, filename=entity, directory=self.local_test_dir))
            seg = r.read_segment(time_slice=None)
            shape_full = seg.analogsignals[0].shape
            event_full = seg.events[0]

            t_start, t_stop = 260 * pq.ms, 1.854 * pq.s
            seg = r.read_segment(time_slice=(t_start, t_stop))
            shape_slice = seg.analogsignals[0].shape
            event_slice = seg.events[0]

            assert shape_full[0] > shape_slice[0]

            assert event_full.size > event_slice.size
            assert np.all(event_slice.times >= t_start)
            assert np.all(event_slice.times <= t_stop)


if __name__ == "__main__":
    unittest.main()
