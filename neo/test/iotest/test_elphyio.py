"""
Tests of neo.io.elphyo
"""

import unittest

from neo.io import ElphyIO
from neo.test.iotest.common_io_test import BaseTestIO



class TestElanIO(BaseTestIO, unittest.TestCase):
    ioclass = ElphyIO
    files_to_test = ['DATA1.DAT', 'ElphyExample.DAT',
                         'ElphyExample_Mode1.dat', 'ElphyExample_Mode2.dat',
                         'ElphyExample_Mode3.dat']
    files_to_download = ['DATA1.DAT', 'ElphyExample.DAT',
                         'ElphyExample_Mode1.dat', 'ElphyExample_Mode2.dat',
                         'ElphyExample_Mode3.dat']

    def test_read_data(self):
        io = ElphyIO(self.get_filename_path('DATA1.DAT'))
        bl = io.read_block()

        print(bl)



if __name__ == "__main__":
    unittest.main()