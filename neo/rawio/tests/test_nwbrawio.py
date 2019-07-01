# Test to add a support for the NWB format

"""
Tests of neo.rawio.nwbrawio
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import unittest
from neo.rawio.nwbrawio import NWBRawIO
from neo.rawio.tests.common_rawio_test import BaseTestRawIO
import pynwb
from pynwb import *

class TestNWBRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = NWBRawIO
    files_to_download = [

##              '/home/elodie/NWB_Files/my_example_2.nwb', # Very simple file nwb create by me only TimeSeries
###              '/home/elodie/NWB_Files/my_NWB_File_pynwb_101_2.nwb', # File created with the latest version of pynwb=1.0.1
#              '/home/elodie/NWB_Files/brain_observatory.nwb', # nwb file given by Matteo Cantarelli
#              '/home/elodie/NWB_Files/mynwb.h5', # nwb file given by Lungsi
#              '/home/elodie/NWB_Files/GreBlu9508M_Site1_Call1.nwb',

######              '/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101.nwb', # File created with the latest version of pynwb=1.0.1 File on my github
              '/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb', # File created with the latest version of pynwb=1.0.1 only with ephys data File on my github

		     ] 
    entities_to_test = files_to_download
    
if __name__ == "__main__":
    print("pynwb.__version__ = ", pynwb.__version__)
    unittest.main()
