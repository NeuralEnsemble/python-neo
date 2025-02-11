"""
Tests of neo.rawio.mearecrawio

"""

import unittest

from neo.rawio.plexon2rawio import Plexon2RawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

from numpy.testing import assert_equal

try:
    from neo.rawio.plexon2rawio.pypl2 import pypl2lib

    HAVE_PYPL2 = True
except (ImportError, TimeoutError):
    HAVE_PYPL2 = False


@unittest.skipUnless(HAVE_PYPL2, "requires pypl package and all its dependencies")
class TestPlexon2RawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = Plexon2RawIO
    entities_to_download = ["plexon"]
    entities_to_test = ["plexon/4chDemoPL2.pl2", "plexon/NC16FPSPKEVT_1m.pl2"]

    def test_check_enabled_flags(self):
        """
        This test loads a 1-minute PL2 file with 16 channels' each
        of field potential (FP), and spike (SPK) data. The channels
        cycle through 4 possible combinations of m_ChannelEnabled
        and m_ChannelRecordingEnabled - (True, True), (True, False),
        (False, True), and (False, False). With 16 channels for each
        source, each combination of flags occurs 4 times. Only the
        first combination (True, True) causes data to be recorded to
        disk. Therefore, we expect the following channels to be loaded by
        Neo: FP01, FP05, FP09, FP13, SPK01, SPK05, SPK09, and SPK13.

        Note: the file contains event (EVT) data as well. Although event
        channel headers do contain m_ChannelEnabled and m_ChannelRecording-
        Enabled flags, the UI for recording PL2 files does not expose any
        controls by which these flags can be changed from (True, True).
        Therefore, no test for event channels is necessary here.
        """

        # Load data from NC16FPSPKEVT_1m.pl2, a 1-minute PL2 recording containing
        # 16-channels' each of field potential (FP), spike (SPK), and event (EVT)
        # data.
        reader = Plexon2RawIO(filename=self.get_local_path("plexon/NC16FPSPKEVT_1m.pl2"))
        reader.parse_header()

        # Check that the names of the loaded signal channels match what we expect
        signal_channel_names = reader.header["signal_channels"]["name"].tolist()
        expected_signal_channel_names = ["FP01", "FP05", "FP09", "FP13"]
        assert_equal(signal_channel_names, expected_signal_channel_names)

        # Check that the names of the loaded spike channels match what we expect
        spike_channel_names = reader.header["spike_channels"]["name"].tolist()
        expected_spike_channel_names = ["SPK01.0", "SPK05.0", "SPK09.0", "SPK13.0"]
        assert_equal(spike_channel_names, expected_spike_channel_names)


if __name__ == "__main__":
    unittest.main()
