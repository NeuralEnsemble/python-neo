"""
Tests of neo.rawio.edfrawio
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neo.rawio.edfrawio import EDFRawIO, HAS_PYEDF
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


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


@unittest.skipUnless(HAS_PYEDF, "requires pyedflib")
class TestEDFRawIOAnnotations(unittest.TestCase):
    def test_annotations_without_duration_are_events(self):
        import pyedflib

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = str(Path(tmp_dir) / "annotations.edf")
            writer = pyedflib.EdfWriter(filename, 1, file_type=pyedflib.FILETYPE_EDFPLUS)
            writer.setSignalHeaders(
                [
                    dict(
                        label="ch1",
                        dimension="uV",
                        sample_frequency=100,
                        physical_min=-100,
                        physical_max=100,
                        digital_min=-32768,
                        digital_max=32767,
                    )
                ]
            )
            writer.writeSamples([np.zeros(1000)])
            # a duration of -1 writes an EDF+ annotation with no duration field
            writer.writeAnnotation(0, -1, "start")
            writer.writeAnnotation(2.5, -1, "stim")
            writer.writeAnnotation(0, 1.5, "baseline")
            writer.close()

            with EDFRawIO(filename) as io:
                io.parse_header()
                event_index, epoch_index = 0, 1

                self.assertEqual(io.event_count(event_channel_index=event_index), 2)
                times, durations, labels = io.get_event_timestamps(event_channel_index=event_index)
                np.testing.assert_array_equal(times, [0.0, 2.5])
                self.assertIsNone(durations)
                np.testing.assert_array_equal(labels, ["start", "stim"])

                self.assertEqual(io.event_count(event_channel_index=epoch_index), 1)
                times, durations, labels = io.get_event_timestamps(event_channel_index=epoch_index)
                np.testing.assert_array_equal(times, [0.0])
                np.testing.assert_array_equal(durations, [1.5])
                np.testing.assert_array_equal(labels, ["baseline"])


if __name__ == "__main__":
    unittest.main()
