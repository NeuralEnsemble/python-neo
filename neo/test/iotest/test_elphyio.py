"""
Tests of neo.io.elphyo
"""

import unittest

from neo.io import ElphyIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestElphyIO(BaseTestIO, unittest.TestCase):
    ioclass = ElphyIO
    entities_to_download = [
        'elphy'
    ]
    entities_to_test = ['elphy/DATA1.DAT',
                        'elphy/ElphyExample.DAT',
                        'elphy/ElphyExample_Mode1.dat',
                        'elphy/ElphyExample_Mode2.dat',
                        'elphy/ElphyExample_Mode3.dat']

    def test_read_data(self):
        for filename in self.entities_to_test:
            io = ElphyIO(self.get_local_path(filename))
            bl = io.read_block()

            self.assertTrue(len(bl.segments) > 0)
            # ensure that at least one data object is generated for each file
            self.assertTrue(any(list(bl.segments[0].size.values())))


if __name__ == "__main__":
    unittest.main()
