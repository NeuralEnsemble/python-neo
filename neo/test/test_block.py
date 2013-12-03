"""
Tests of the Block class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest

from neo.core.block import Block


class TestBlock(unittest.TestCase):
    def test_init(self):
        blk = Block(name='a block')
        self.assertEqual(blk.name, 'a block')
        self.assertEqual(blk.file_origin, None)


if __name__ == "__main__":
    unittest.main()
