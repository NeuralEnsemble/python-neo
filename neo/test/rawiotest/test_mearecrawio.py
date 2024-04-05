"""
Tests of neo.rawio.mearecrawio

"""

import unittest

from neo.rawio.mearecrawio import MEArecRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


try:
    import MEArec as mr

    HAVE_MEAREC = True
except ImportError:
    HAVE_MEAREC = False


@unittest.skipUnless(HAVE_MEAREC, "requires MEArec package")
class TestMEArecRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = MEArecRawIO
    entities_to_download = ["mearec"]
    entities_to_test = ["mearec/mearec_test_10s.h5"]

    def test_not_loading_recordings(self):

        filename = self.entities_to_test[0]
        filename = self.get_local_path(filename)
        rawio = self.rawioclass(filename=filename, load_analogsignal=False)
        rawio.parse_header()

        # Test that rawio does not have a _recordings attribute
        self.assertFalse(hasattr(rawio, "_recordings"))

        # Test that calling get_spike_timestamps works
        rawio.get_spike_timestamps()

        # Test that caling anlogsignal chunk raises the right error
        with self.assertRaises(AttributeError):
            rawio.get_analogsignal_chunk()

    def test_not_loading_spiketrain(self):

        filename = self.entities_to_test[0]
        filename = self.get_local_path(filename)
        rawio = self.rawioclass(filename=filename, load_spiketrains=False)
        rawio.parse_header()

        # Test that rawio does not have a _spiketrains attribute
        self.assertFalse(hasattr(rawio, "_spiketrains"))

        # Test that calling analogsignal chunk works
        rawio.get_analogsignal_chunk()

        # Test that calling get_spike_timestamps raises an the right error
        with self.assertRaises(AttributeError):
            rawio.get_spike_timestamps()


if __name__ == "__main__":
    unittest.main()
