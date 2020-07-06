"""
Tests of neo.rawio.axographrawio
"""

import unittest

from neo.rawio.axographrawio import AxographRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestAxographRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = AxographRawIO
    files_to_download = [
        'AxoGraph_Graph_File',      # version 1 file, provided with AxoGraph
        'AxoGraph_Digitized_File',  # version 2 file, provided with AxoGraph
        'AxoGraph_X_File.axgx',     # version 5 file, provided with AxoGraph
        'File_axograph.axgd',       # version 6 file
        'episodic.axgd',
        'events_and_epochs.axgx',
        'written-by-axographio-with-linearsequence.axgx',
        'written-by-axographio-without-linearsequence.axgx',
        'corrupt-comment.axgx',
    ]
    entities_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()
