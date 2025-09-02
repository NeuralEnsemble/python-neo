"""
Tests of neo.io.kwikio
"""

import unittest
import importlib.util
import importlib.metadata
from packaging.version import Version, parse

kwik_spec = importlib.util.find_spec('klusta')
# kwik no longer works with recent versions of numpy
numpy_version = parse(importlib.metadata.version('numpy'))
numpy_okay = numpy_version < Version('2.3.0')
if kwik_spec is not None and numpy_okay:
    HAVE_KWIK = True
else:
    HAVE_KWIK = False

from neo.io import kwikio
from neo.test.iotest.common_io_test import BaseTestIO


@unittest.skipUnless(HAVE_KWIK, "requires klusta")
class TestKwikIO(BaseTestIO, unittest.TestCase):
    ioclass = kwikio.KwikIO
    entities_to_download = ["kwik"]
    entities_to_test = ["kwik/neo.kwik"]


if __name__ == "__main__":
    unittest.main()
