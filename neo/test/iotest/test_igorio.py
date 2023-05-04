"""
Tests of neo.io.igorproio
"""

import unittest

try:
    import igor

    HAVE_IGOR = True
except ImportError:
    HAVE_IGOR = False
from neo.io.igorproio import IgorIO
from neo.test.iotest.common_io_test import BaseTestIO

import numpy as np
from packaging.version import Version


@unittest.skipUnless(HAVE_IGOR, "requires igor")
@unittest.skipIf(Version(np.__version__) > Version('1.22.0'), "igor is not compatible with numpy > 1.22.0")
class TestIgorIO(BaseTestIO, unittest.TestCase):
    ioclass = IgorIO
    entities_to_download = [
        'igor'
    ]
    entities_to_test = [
        'igor/mac-version2.ibw',
        'igor/win-version2.ibw'
    ]


if __name__ == "__main__":
    unittest.main()
