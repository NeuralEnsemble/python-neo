import unittest

from neo.rawio.openephysbinaryrawio import OpenEphysBinaryRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestOpenEphysBinaryRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = OpenEphysBinaryRawIO
    entities_to_download = [
        'openephysbinary'
    ]
    entities_to_test = [
        'openephysbinary/v0.5.3_two_neuropixels_stream',
        'openephysbinary/v0.4.4.1_with_video_tracking',
        'openephysbinary/v0.5.x_two_nodes',
        'openephysbinary/v0.6.x_neuropixels_multiexp_multistream',
    ]


if __name__ == "__main__":
    unittest.main()
