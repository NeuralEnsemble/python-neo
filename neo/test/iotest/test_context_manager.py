"""
Tests that IO and RawIO classes can be used as context managers
"""

import unittest

from neo.io.exampleio import ExampleIO
from neo.rawio.examplerawio import ExampleRawIO


class TestContextManager(unittest.TestCase):
    def _check(self, cls):
        closed = []

        class Tracked(cls):
            def close(self):
                closed.append(True)

        with Tracked(filename="fake.fake") as io:
            self.assertIsInstance(io, Tracked)
        self.assertEqual(closed, [True])

        with self.assertRaises(ValueError), Tracked(filename="fake.fake"):
            raise ValueError
        self.assertEqual(closed, [True, True])

    def test_io(self):
        self._check(ExampleIO)

    def test_rawio(self):
        self._check(ExampleRawIO)

    def test_default_close_is_noop(self):
        with ExampleIO(filename="fake.fake") as io:
            pass
        io.close()


if __name__ == "__main__":
    unittest.main()
