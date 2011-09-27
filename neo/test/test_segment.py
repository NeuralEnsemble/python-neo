"""
Tests of the Segment class
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
    
from neo.core.segment import Segment

class TestSegment(unittest.TestCase):
    def test_init(self):
        seg = Segment(name='a segment', index=3)
        self.assertEqual(seg.name, 'a segment')
        self.assertEqual(seg.file_origin, None)
        self.assertEqual(seg.index, 3)

