"""
Tests of neo.io.baseio
"""

import unittest

from neo.core import objectlist
from neo.io.baseio import BaseIO


class TestIOObjects(unittest.TestCase):
    def test__raise_error_when_not_readable_or_writable(self):
        reader = BaseIO()
        for ob in objectlist:
            if ob not in BaseIO.readable_objects:
                meth = getattr(reader, 'read_' + ob.__name__.lower())
                self.assertRaises(AssertionError, meth, )

            if ob not in BaseIO.writeable_objects:
                meth = getattr(reader, 'write_' + ob.__name__.lower())
                self.assertRaises(AssertionError, meth, ())


if __name__ == "__main__":
    unittest.main()
