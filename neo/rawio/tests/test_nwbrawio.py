# Test to add a support for the NWB format

"""
Tests of neo.rawio.nwbrawio
"""

from __future__ import unicode_literals, print_function, division, absolute_import

import unittest

from neo.rawio.nwbrawio import NWBRawIO #, NWBReader
###from neo.rawio.nwbrawio_only_1_signal import NWBRawIO

from neo.rawio.tests.common_rawio_test import BaseTestRawIO

class TestNWBRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = NWBRawIO
    files_to_download = [

              '/home/elodie/envNWB/NWB_files/my_example_2.nwb', # Very simple file nwb create by me only TimeSeries
#              '/home/elodie/envNWB/NWB_files/brain_observatory.nwb', # nwb file given by Matteo Cantarelli
#              '/home/elodie/envNWB/NWB_files/mynwb.h5', # nwb file given by Lungsi
#              '/home/elodie/envNWB/NWB_files/GreBlu9508M_Site1_Call1.nwb',

		     ] 
    entities_to_test = files_to_download

if __name__ == "__main__":
    unittest.main()
