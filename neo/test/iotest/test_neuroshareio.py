"""
Tests of neo.io.neuroshareio
"""

import sys
import os
import pathlib
import shutil
import tarfile
import zipfile
import tempfile
import platform
import unittest
from urllib.request import urlretrieve

from neo.io import NeuroshareIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestNeuroshareIO(BaseTestIO, unittest.TestCase):
    ioclass = NeuroshareIO
    entities_to_download = ['neuroshare']
    entities_to_test = ['neuroshare/Multichannel_fil_1.mcd']

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super().setUpClass()
        if sys.platform.startswith('win'):
            distantfile = 'http://download.multichannelsystems.com/download_data/software/neuroshare/nsMCDLibrary_3.7b.zip'
            localfile = os.path.join(tempfile.gettempdir(), 'nsMCDLibrary_3.7b.zip')
            if not os.path.exists(localfile):
                urlretrieve(distantfile, localfile)
            if platform.architecture()[0].startswith('64'):
                cls.dllname = os.path.join(
                    tempfile.gettempdir(),
                    'Matlab/Matlab-Import-Filter/Matlab_Interface/nsMCDLibrary64.dll')
                if not os.path.exists(cls.dllname):
                    zip = zipfile.ZipFile(localfile)
                    zip.extract(
                        'Matlab/Matlab-Import-Filter/Matlab_Interface/nsMCDLibrary64.dll',
                        path=tempfile.gettempdir())
            else:
                cls.dllname = os.path.join(
                    tempfile.gettempdir(),
                    'Matlab/Matlab-Import-Filter/Matlab_Interface/nsMCDLibrary.dll')
                if not os.path.exists(cls.dllname):
                    zip = zipfile.ZipFile(localfile)
                    zip.extract(
                        'Matlab/Matlab-Import-Filter/Matlab_Interface/nsMCDLibrary.dll',
                        path=tempfile.gettempdir())

        elif sys.platform.startswith('linux'):
            if platform.architecture()[0].startswith('64'):
                distantfile = 'http://download.multichannelsystems.com/download_data/software/neuroshare/nsMCDLibrary_Linux64_3.7b.tar.gz'
                localfile = os.path.join(tempfile.gettempdir(), 'nsMCDLibrary_Linux64_3.7b.tar.gz')
            else:
                distantfile = 'http://download.multichannelsystems.com/download_data/software/neuroshare/nsMCDLibrary_Linux32_3.7b.tar.gz'
                localfile = os.path.join(tempfile.gettempdir(), 'nsMCDLibrary_Linux32_3.7b.tar.gz')
            if not os.path.exists(localfile):
                urlretrieve(distantfile, localfile)
            cls.dllname = os.path.join(tempfile.gettempdir(), 'nsMCDLibrary/nsMCDLibrary.so')
            if not os.path.exists(cls.dllname):
                tar = tarfile.open(localfile)
                tar.extract('nsMCDLibrary/nsMCDLibrary.so', path=tempfile.gettempdir())
        else:
            raise unittest.SkipTest("Not currently supported on OS X")

        # move dll to path discoverable by neuroshare package
        discoverable_path = pathlib.Path.home() / ".neuroshare" / cls.dllname.split('/')[-1]
        if not discoverable_path.exists():
            discoverable_path.parent.mkdir(exist_ok=True)
            shutil.copyfile(cls.dllname, discoverable_path)
            cls._cleanup_library = True
        else:
            cls._cleanup_library = False

        cls.default_keyword_arguments['dllname'] = cls.dllname

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._cleanup_library:
            discoverable_path = pathlib.Path.home() / ".neuroshare" / cls.dllname.split('/')[-1]
            discoverable_path.unlink(missing_ok=True)

    def test_with_multichannel(self):
        for filename in self.files_to_test:
            filename = self.get_local_path(filename)
            reader = NeuroshareIO(filename, self.dllname)
            blocks = reader.read()
            n = len(blocks[0].segments[0].analogsignals)
            assert n > 0, f'Expect signals in file {filename}'


if __name__ == "__main__":
    unittest.main()
