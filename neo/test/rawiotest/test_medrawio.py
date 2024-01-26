import unittest
import numpy as np

from neo.rawio.medrawio import MedRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

try:
    import dhn_med_py

    HAVE_DHN_MED = True
except ImportError:
    HAVE_DHN_MED = False


# This runs standard tests, this is mandatory for all IOs
@unittest.skipUnless(HAVE_DHN_MED, "requires dhn_med_py package and all its dependencies")
class TestMedRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = MedRawIO
    entities_to_download = ["med"]
    entities_to_test = ["med/sine_waves.medd", "med/test.medd"]

    def test_close(self):

        filename = self.get_local_path("med/sine_waves.medd")

        raw_io1 = MedRawIO(filename, password="L2_password")
        raw_io1.parse_header()
        raw_io1.close()

        raw_io2 = MedRawIO(filename, password="L2_password")
        raw_io2.parse_header()
        raw_io2.close()

    def test_scan_med_directory(self):

        filename = self.get_local_path("med/sine_waves.medd")

        rawio = MedRawIO(filename, password="L2_password")
        rawio.parse_header()

        # Test that correct metadata and boundaries are extracted
        # from the MED session.  We know the correct answers since
        # we generated the test files.
        self.assertEqual(rawio.signal_streams_count(), 1)
        self.assertEqual(rawio._segment_t_start(0, 0), 0)
        self.assertEqual(rawio._segment_t_stop(0, 0), 180)

        # Verify it found all 3 channels
        self.assertEqual(rawio.num_channels_in_session, 3)
        self.assertEqual(rawio.header["signal_channels"].size, 3)

        # Verify if found the names of the 3 channels
        self.assertEqual(rawio.header["signal_channels"][0][0], "CSC_0001")
        self.assertEqual(rawio.header["signal_channels"][1][0], "CSC_0002")
        self.assertEqual(rawio.header["signal_channels"][2][0], "CSC_0003")

        # Read first 3 seconds of data from all channels
        raw_chunk = rawio.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=96000, stream_index=0, channel_indexes=None
        )

        # Test the first sample value of all 3 channels, which are
        # known to be [-1, -4, -4]
        np.testing.assert_array_equal(raw_chunk[0][:3], [-1, -4, -4])

        # Read 1 second of data from the second channel
        raw_chunk = rawio.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=32000, stream_index=0, channel_indexes=[1]
        )

        # Test known first sample of second channel: [-4]
        self.assertEqual(raw_chunk[0][0], -4)

        rawio.close()

        # Test on second test dataset, test.medd.
        filename = self.get_local_path("med/test.medd")

        rawio = MedRawIO(filename, password="L2_password")
        rawio.parse_header()

        # Test that correct metadata and boundaries are extracted
        # from the MED session.  We know the correct answers since
        # we generated the test files.

        # For this dataset, there are 3 continuous data ranges, with
        # approximately 10 seconds between the ranges.
        # There are 3 channels, two with a frequency of 1000 Hz and one
        # with a frequency of 5000 Hz.
        self.assertEqual(rawio.signal_streams_count(), 2)

        # Segment 0
        self.assertEqual(rawio._segment_t_start(0, 0), 0)
        self.assertEqual(rawio._segment_t_stop(0, 0), 39.8898)
        # Segment 1
        self.assertEqual(rawio._segment_t_start(0, 1), 50.809826)
        self.assertEqual(rawio._segment_t_stop(0, 1), 87.337646)
        # Segment 2
        self.assertEqual(rawio._segment_t_start(0, 2), 97.242057)
        self.assertEqual(rawio._segment_t_stop(0, 2), 180.016702)

        # Verify it found all 3 channels.
        self.assertEqual(rawio.num_channels_in_session, 3)
        self.assertEqual(rawio.header["signal_channels"].size, 3)

        # Verity if found the names of the 3 channels
        self.assertEqual(rawio.header["signal_channels"][0][0], "5k_ch1")
        self.assertEqual(rawio.header["signal_channels"][1][0], "1k_ch1")
        self.assertEqual(rawio.header["signal_channels"][2][0], "1k_ch2")

        # Read first 3 seconds of data from the first channel (5k_ch1)
        raw_chunk = rawio.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=15000, stream_index=0, channel_indexes=None
        )

        # Test the first three sample values returned, which are
        # known to be [-80, -79, -78]
        self.assertEqual(raw_chunk[0][0], -80)
        self.assertEqual(raw_chunk[1][0], -79)
        self.assertEqual(raw_chunk[2][0], -78)

        # Read first 3 seconds of data from the second channel and third
        # channels (1k_ch1 and 1k_ch2)
        raw_chunk = rawio.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=3000, stream_index=1, channel_indexes=None
        )

        # Test first sample returned of both channels, which are known
        # to be [-79, -80]
        np.testing.assert_array_equal(raw_chunk[0][:2], [-79, -80])

        # Read first 3 seconds of data from the second segment of the first channel (5k_ch1)
        raw_chunk = rawio.get_analogsignal_chunk(
            block_index=0, seg_index=1, i_start=0, i_stop=15000, stream_index=0, channel_indexes=None
        )

        # Test the first three sample values returned, which are
        # known to be [22, 23, 24]
        self.assertEqual(raw_chunk[0][0], 22)
        self.assertEqual(raw_chunk[1][0], 23)
        self.assertEqual(raw_chunk[2][0], 24)

        self.assertEqual(len(rawio.header["event_channels"]), 2)

        # Verify that there are 5 events in the dataset.
        # They are 3 "Note" type events, and 2 "NlxP", or neuralynx, type events.
        # The first segment has one event, and the second and third segments
        # each have 2 events.
        self.assertEqual(rawio.event_count(0, 0, 0), 1)
        self.assertEqual(rawio.event_count(0, 1, 0), 2)
        self.assertEqual(rawio.event_count(0, 2, 0), 2)

        # Get array of all events in first segment of data
        events = rawio._get_event_timestamps(0, 0, 0, rawio._segment_t_start(0, 0), rawio._segment_t_stop(0, 0))
        # Make sure it read 1 event
        self.assertEqual(len(events[0]), 1)

        # Get array of all events in second segment of data
        events = rawio._get_event_timestamps(0, 1, 0, rawio._segment_t_start(0, 1), rawio._segment_t_stop(0, 1))
        # Make sure it read 2 events
        self.assertEqual(len(events[0]), 2)

        # Verify the first event of the second segment is a Neuralynx type event, with correct time
        self.assertEqual(events[2][0][:4], "NlxP")
        self.assertEqual(events[0][0], 51.703509)

        # Get array of all events in third segment of data
        events = rawio._get_event_timestamps(0, 2, 0, rawio._segment_t_start(0, 2), rawio._segment_t_stop(0, 2))
        # Make sure it read 2 events
        self.assertEqual(len(events[0]), 2)

        # Verify the second event of the second segment is a Neuralynx type event, with correct time
        self.assertEqual(events[2][1][:4], "NlxP")
        self.assertEqual(events[0][1], 161.607036)

        rawio.close()

        # Test on second test dataset, test.medd, with preserving original timestamps.
        # Timestamps here are in UTC (seconds since midnight, 1 Jan 1970)
        filename = self.get_local_path("med/test.medd")

        rawio = MedRawIO(filename, password="L2_password", keep_original_times=True)
        rawio.parse_header()

        # Segment 0
        self.assertEqual(rawio._segment_t_start(0, 0), 1678111774.012236)
        self.assertEqual(rawio._segment_t_stop(0, 0), 1678111813.902036)
        # Segment 1
        self.assertEqual(rawio._segment_t_start(0, 1), 1678111824.822062)
        self.assertEqual(rawio._segment_t_stop(0, 1), 1678111861.349882)
        # Segment 2
        self.assertEqual(rawio._segment_t_start(0, 2), 1678111871.254293)
        self.assertEqual(rawio._segment_t_stop(0, 2), 1678111954.028938)

        # Verify that there are 5 events in the dataset.
        # They are 3 "Note" type events, and 2 "NlxP", or neuralynx, type events.
        # The first segment has one event, and the second and third segments
        # each have 2 events.
        self.assertEqual(rawio.event_count(0, 0, 0), 1)
        self.assertEqual(rawio.event_count(0, 1, 0), 2)
        self.assertEqual(rawio.event_count(0, 2, 0), 2)

        # Get array of all events in first segment of data
        events = rawio._get_event_timestamps(0, 0, 0, rawio._segment_t_start(0, 0), rawio._segment_t_stop(0, 0))
        # Make sure it read 1 event
        self.assertEqual(len(events[0]), 1)

        # Get array of all events in second segment of data
        events = rawio._get_event_timestamps(0, 1, 0, rawio._segment_t_start(0, 1), rawio._segment_t_stop(0, 1))
        # Make sure it read 2 events
        self.assertEqual(len(events[0]), 2)

        # Verify the first event of the second segment is a Neuralynx type event, with correct time
        self.assertEqual(events[2][0][:4], "NlxP")
        self.assertEqual(events[0][0], 1678111825.715745)

        # Get array of all events in third segment of data
        events = rawio._get_event_timestamps(0, 2, 0, rawio._segment_t_start(0, 2), rawio._segment_t_stop(0, 2))
        # Make sure it read 2 events
        self.assertEqual(len(events[0]), 2)

        # Verify the second event of the second segment is a Neuralynx type event, with correct time
        self.assertEqual(events[2][1][:4], "NlxP")
        self.assertEqual(events[0][1], 1678111935.619272)

        rawio.close()


if __name__ == "__main__":
    unittest.main()
