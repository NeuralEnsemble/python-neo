"""
Tests of neo.rawio.edfrawio
"""

import unittest
import tempfile
from pathlib import Path

import numpy as np

from neo.rawio.edfrawio import EDFRawIO, HAS_PYEDF
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

if HAS_PYEDF:
    import pyedflib


@unittest.skipUnless(HAS_PYEDF, "requires pyedflib")
class TestEDFCalibration(unittest.TestCase):
    def test_digital_endpoints_map_to_physical_endpoints(self):
        headers = [
            dict(
                label="identity",
                dimension="uV",
                sample_frequency=8,
                digital_min=-1000,
                digital_max=1000,
                physical_min=-1000,
                physical_max=1000,
            ),
            dict(
                label="offset",
                dimension="uV",
                sample_frequency=8,
                digital_min=0,
                digital_max=100,
                physical_min=-10,
                physical_max=30,
            ),
            dict(
                label="full_range",
                dimension="uV",
                sample_frequency=8,
                digital_min=-32768,
                digital_max=32767,
                physical_min=-200,
                physical_max=200,
            ),
        ]
        digital = [np.array([h["digital_min"], h["digital_max"]] * 4, dtype=np.int32) for h in headers]
        expected = np.array([[h["physical_min"], h["physical_max"]] * 4 for h in headers]).T
        with tempfile.TemporaryDirectory() as folder:
            filename = str(Path(folder) / "calibration.edf")
            with pyedflib.EdfWriter(filename, len(headers), file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
                writer.setSignalHeaders(headers)
                writer.writeSamples(digital, digital=True)
            with EDFRawIO(filename) as reader:
                reader.parse_header()
                for channels in (None, [2, 0], [1]):
                    raw = reader.get_analogsignal_chunk(channel_indexes=channels)
                    selected = slice(None) if channels is None else channels
                    np.testing.assert_array_equal(raw, np.asarray(digital).T[:, selected])
                    for dtype in ("float32", "float64"):
                        with self.subTest(channels=channels, dtype=dtype):
                            physical = reader.rescale_signal_raw_to_float(raw, dtype=dtype, channel_indexes=channels)
                            np.testing.assert_allclose(physical, expected[:, selected], rtol=1e-6, atol=1e-5)


class TestExampleRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = EDFRawIO
    entities_to_download = ["edf"]
    entities_to_test = [
        "edf/edf+C.edf",
    ]

    def test_context_handler(self):
        filename = self.get_local_path("edf/edf+C.edf")
        with EDFRawIO(filename) as io:
            io.parse_header()

        # Check that file was closed properly and can be opened again
        with open(filename) as f:
            pass

    def test_close(self):
        filename = self.get_local_path("edf/edf+C.edf")

        # Open file and close it again
        io1 = EDFRawIO(filename)
        io1.parse_header()
        io1.close()

        # Check that file was closed properly and can be opened again
        io2 = EDFRawIO(filename)
        io2.parse_header()
        io2.close()


if __name__ == "__main__":
    unittest.main()
