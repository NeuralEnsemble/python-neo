"""
Tests of neo.io.axonio
"""

import unittest

from neo.io import AxonIO
from neo.test.iotest.common_io_test import BaseTestIO

from neo.test.rawiotest.test_axonrawio import TestAxonRawIO

class TestAxonIO(BaseTestIO, unittest.TestCase):
    entities_to_download = TestAxonRawIO.entities_to_download
    entities_to_test = TestAxonRawIO.entities_to_test
    ioclass = AxonIO

    def test_annotations(self):
        reader = AxonIO(filename=self.get_local_path('axon/File_axon_2.abf'))
        bl = reader.read_block()
        ev = bl.segments[0].events[0]
        assert 'comments' in ev.annotations

    def test_read_protocol(self):
        for f in self.entities_to_test:
            filename = self.get_local_path(f)
            reader = AxonIO(filename=filename)
            bl = reader.read_block(lazy=True)
            if bl.annotations['abf_version'] >= 2.:
                reader.read_protocol()


if __name__ == "__main__":
    unittest.main()
