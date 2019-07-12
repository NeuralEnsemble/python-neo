
"""
Tests of neo.io.nwbio
"""

#from __future__ import division
#
#import sys
#import unittest
#try:
#    import unittest2 as unittest
#except ImportError:
#    import unittest
#try:
#    import pynwb
#    HAVE_NWB = True
#except ImportError:
#    HAVE_NWB = False
#from neo.io import NWBIO
#from neo.test.iotest.common_io_test import BaseTestIO

from __future__ import unicode_literals, print_function, division, absolute_import
import unittest
from neo.io.nwbio import NWBIO
from neo.test.iotest.common_io_test import BaseTestIO
import pynwb
from pynwb import *

#@unittest.skipUnless(HAVE_NWB, "requires nwb")
###class TestNWBIO(BaseTestIO, unittest.TestCase, ): ############################ To change to test !
class TestNWBIO(unittest.TestCase, ):
    ioclass = NWBIO

    files_to_download =  [

              '/home/elodie/NWB_Files/NWB_File_python_3_pynwb_101_ephys_data.nwb', # File created with the latest version of pynwb=1.0.1 only with ephys data File on my github
#              '/home/elodie/NWB_Files/NWB_org/H19.28.012.11.05-2.nwb' # NWB file downloaded from http://download.alleninstitute.org/informatics-archive/prerelease/H19.28.012.11.05-2.nwb

    ]

    entities_to_test = files_to_download

if __name__ == "__main__":
    print("pynwb.__version__ = ", pynwb.__version__)
    unittest.main()



