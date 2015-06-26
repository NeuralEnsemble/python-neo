# -*- coding: utf-8 -*-
"""
Tests of neo.io.neuroshareio
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division

import sys
import os
import tarfile
import zipfile
import tempfile
import platform

try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    from urllib import urlretrieve  # Py2
except ImportError:
    from urllib.request import urlretrieve  # Py3


from neo.io import NeuroshareIO
from neo.test.iotest.common_io_test import BaseTestIO



class TestNeuroshareIO(unittest.TestCase, BaseTestIO):
    ioclass = NeuroshareIO
    files_to_test = [ ]
    files_to_download = [ 'Multichannel_fil_1.mcd', ]
    
    def setUp(self):
        BaseTestIO.setUp(self)
        if sys.platform.startswith('win'):
            distantfile = 'http://download.multichannelsystems.com/download_data/software/neuroshare/nsMCDLibrary_3.7b.zip'
            localfile = os.path.join(tempfile.gettempdir(),'nsMCDLibrary_3.7b.zip')
            if not os.path.exists(localfile):
                urlretrieve(distantfile, localfile)
            if platform.architecture()[0].startswith('64'):
                self.dllname = os.path.join(tempfile.gettempdir(),'Matlab/Matlab-Import-Filter/Matlab_Interface/nsMCDLibrary64.dll')
                if not os.path.exists(self.dllname):
                    zip = zipfile.ZipFile(localfile)
                    zip.extract('Matlab/Matlab-Import-Filter/Matlab_Interface/nsMCDLibrary64.dll', path = tempfile.gettempdir())
            else:
                self.dllname = os.path.join(tempfile.gettempdir(),'Matlab/Matlab-Import-Filter/Matlab_Interface/nsMCDLibrary.dll')
                if not os.path.exists(self.dllname):
                    zip = zipfile.ZipFile(localfile)
                    zip.extract('Matlab/Matlab-Import-Filter/Matlab_Interface/nsMCDLibrary.dll', path = tempfile.gettempdir())

        elif sys.platform.startswith('linux'):
            if platform.architecture()[0].startswith('64'):
                distantfile = 'http://download.multichannelsystems.com/download_data/software/neuroshare/nsMCDLibrary_Linux64_3.7b.tar.gz'
                localfile = os.path.join(tempfile.gettempdir(),'nsMCDLibrary_Linux64_3.7b.tar.gz')
            else:
                distantfile = 'http://download.multichannelsystems.com/download_data/software/neuroshare/nsMCDLibrary_Linux32_3.7b.tar.gz'
                localfile = os.path.join(tempfile.gettempdir(),'nsMCDLibrary_Linux32_3.7b.tar.gz')
            if not os.path.exists(localfile):
                urlretrieve(distantfile, localfile)
            self.dllname = os.path.join(tempfile.gettempdir(),'nsMCDLibrary/nsMCDLibrary.so')
            if not os.path.exists(self.dllname):
                tar = tarfile.open(localfile)
                tar.extract('nsMCDLibrary/nsMCDLibrary.so', path = tempfile.gettempdir())
        else:
            raise unittest.SkipTest("Not currently supported on OS X")
        
    
    def test_with_multichannel(self):
        filename0 = self.get_filename_path(self.files_to_download[0])
        reader = NeuroshareIO(filename0, self.dllname)
        blocks = reader.read()
        n = len(blocks[0].segments[0].analogsignals)
        assert n == 2, \
                    'For {} , nb AnalogSignal: {} (should be 2)'.format(self.files_to_download[0], n)


if __name__ == "__main__":
    unittest.main()
