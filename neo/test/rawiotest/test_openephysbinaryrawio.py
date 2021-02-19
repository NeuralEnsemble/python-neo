import unittest

from neo.rawio.openephysbinaryrawio import OpenEphysBinaryRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestOpenEphysBinaryRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = OpenEphysBinaryRawIO
    entities_to_test = [
        'test_multiple_2020-12-17_17-34-35',
    ]

    files_to_download = [
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording1/sync_messages.txt",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording1/structure.oebin",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording1/events/Message_Center-904.0/TEXT_group_1/text.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording1/events/Message_Center-904.0/TEXT_group_1/timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording1/events/Message_Center-904.0/TEXT_group_1/channels.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording1/continuous/File_Reader-100.0/synchronized_timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording1/continuous/File_Reader-100.0/continuous.dat",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording1/continuous/File_Reader-100.0/timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording3/sync_messages.txt",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording3/structure.oebin",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording3/events/Message_Center-904.0/TEXT_group_1/text.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording3/events/Message_Center-904.0/TEXT_group_1/timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording3/events/Message_Center-904.0/TEXT_group_1/channels.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording3/continuous/File_Reader-100.0/synchronized_timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording3/continuous/File_Reader-100.0/continuous.dat",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording3/continuous/File_Reader-100.0/timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording2/sync_messages.txt",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording2/structure.oebin",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording2/events/Message_Center-904.0/TEXT_group_1/text.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording2/events/Message_Center-904.0/TEXT_group_1/timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording2/events/Message_Center-904.0/TEXT_group_1/channels.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording2/continuous/File_Reader-100.0/synchronized_timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording2/continuous/File_Reader-100.0/continuous.dat",
        "test_multiple_2020-12-17_17-34-35/RecordNode103/experiment1/recording2/continuous/File_Reader-100.0/timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording1/sync_messages.txt",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording1/structure.oebin",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording1/events/Message_Center-904.0/TEXT_group_1/text.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording1/events/Message_Center-904.0/TEXT_group_1/timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording1/events/Message_Center-904.0/TEXT_group_1/channels.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording1/continuous/File_Reader-100.0/synchronized_timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording1/continuous/File_Reader-100.0/continuous.dat",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording1/continuous/File_Reader-100.0/timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording3/sync_messages.txt",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording3/structure.oebin",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording3/events/Message_Center-904.0/TEXT_group_1/text.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording3/events/Message_Center-904.0/TEXT_group_1/timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording3/events/Message_Center-904.0/TEXT_group_1/channels.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording3/continuous/File_Reader-100.0/synchronized_timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording3/continuous/File_Reader-100.0/continuous.dat",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording3/continuous/File_Reader-100.0/timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording2/sync_messages.txt",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording2/structure.oebin",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording2/events/Message_Center-904.0/TEXT_group_1/text.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording2/events/Message_Center-904.0/TEXT_group_1/timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording2/events/Message_Center-904.0/TEXT_group_1/channels.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording2/continuous/File_Reader-100.0/synchronized_timestamps.npy",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording2/continuous/File_Reader-100.0/continuous.dat",
        "test_multiple_2020-12-17_17-34-35/RecordNode105/experiment1/recording2/continuous/File_Reader-100.0/timestamps.npy",        
    ]


if __name__ == "__main__":
    unittest.main()
