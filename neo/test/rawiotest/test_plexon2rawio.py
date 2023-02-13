"""
Tests of neo.rawio.mearecrawio

"""

import unittest

from neo.rawio.plexon2rawio import Plexon2RawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


try:
    from neo.rawio.plexon2rawio.pypl2 import pypl2lib
    HAVE_PYPL2 = True
except ImportError:
    HAVE_PYPL2 = False


@unittest.skipUnless(HAVE_PYPL2, "requires pypl package and all its dependencies")
class TestPlexon2RawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = Plexon2RawIO
    entities_to_download = [
        'plexon'
    ]
    entities_to_test = [
        'plexon/4chDemoPL2.pl2'
    ]


if __name__ == "__main__":
    unittest.main()
