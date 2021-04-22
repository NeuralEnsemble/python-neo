"""
Tests of neo.rawio.axographrawio
"""

import unittest

from neo.rawio.axographrawio import AxographRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestAxographRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = AxographRawIO
    entities_to_test = [
        'axograph/AxoGraph_Graph_File',      # version 1 file, provided with AxoGraph
        'axograph/AxoGraph_Digitized_File',  # version 2 file, provided with AxoGraph
        'axograph/AxoGraph_X_File.axgx',     # version 5 file, provided with AxoGraph
        'axograph/File_axograph.axgd',       # version 6 file
        'axograph/episodic.axgd',
        'axograph/events_and_epochs.axgx',
        'axograph/written-by-axographio-with-linearsequence.axgx',
        'axograph/written-by-axographio-without-linearsequence.axgx',
        'axograph/corrupt-comment.axgx',
    ]
    entities_to_download = ['axograph']


if __name__ == "__main__":
    unittest.main()
