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
        b = Block(name='a block')
        self.assertEqual(b.name, 'a block')
        self.assertEqual(b.file_origin, None)


