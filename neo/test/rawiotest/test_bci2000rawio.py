"""
Tests of neo.rawio.bci2000rawio
"""

import unittest

import numpy as np

from neo.rawio.bci2000rawio import BCI2000RawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestBCI2000RawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = BCI2000RawIO

    entities_to_download = [
        "bci2000",
    ]

    entities_to_test = ["bci2000/eeg1_1.dat", "bci2000/eeg1_2.dat", "bci2000/eeg1_3.dat"]

    def test_scaling(self):
        # The header has SourceChGain= 64 0.01617 0.01591 ... and SourceChOffset= 64 43 55 ...
        # BCI2000 defines uV = (raw - SourceChOffset) * SourceChGain
        reader = BCI2000RawIO(filename=self.get_local_path("bci2000/eeg1_1.dat"))
        reader.parse_header()
        signal_channels = reader.header["signal_channels"]
        np.testing.assert_allclose(signal_channels["gain"][:2], [0.01617, 0.01591], rtol=1e-6)

        raw = reader.get_analogsignal_chunk(0, 0, 0, 3, 0, channel_indexes=[0, 1])
        uV = reader.rescale_signal_raw_to_float(raw, dtype="float64", stream_index=0, channel_indexes=[0, 1])
        expected = (raw.astype("float64") - [43, 55]) * [0.01617, 0.01591]
        np.testing.assert_allclose(uV, expected, rtol=1e-5)
        # first samples as decoded by BCI2000's own FileReader.py (BCI2kReader)
        np.testing.assert_allclose(uV[:, 0], [-16.218512, 1.3744501, -9.23307], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
