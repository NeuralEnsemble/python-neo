import unittest

from neo.rawio.monkeylogicrawio import MonkeyLogicRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

import logging

logging.getLogger().setLevel(logging.INFO)


class TestMonkeyLogicRawIO(BaseTestRawIO, unittest.TestCase, ):
    rawioclass = MonkeyLogicRawIO
    entities_to_download = [
        'monkeylogic'
    ]
    entities_to_test = []

    def setUp(self):
        # TODO update this
        filename = '/home/sprengerj/projects/monkey_logic/210909_TSCM_5cj_5cl_Riesling.bhv2'
        filename = '/home/sprengerj/projects/monkey_logic/sabrina/210810__learndms_userloop.bhv2'
        # filename = '/home/sprengerj/projects/monkey_logic/sabrina/210916__learndms_userloop.bhv2'
        # filename = '/home/sprengerj/projects/monkey_logic/sabrina/210917__learndms_userloop.bhv2'


        self.rawio = MonkeyLogicRawIO(filename)

    def test_scan_ncs_files(self):

        # Test BML style of Ncs files, similar to PRE4 but with fractional frequency
        # in the header and fractional microsPerSamp, which is then rounded as appropriate
        # in each record.

        rawio = self.rawio
        self.rawio.parse_header()

        # test values here from direct inspection of .ncs files
        # self.assertEqual(rawio._nb_segment, 1)
        # self.assertListEqual(rawio._timestamp_limits, [(0, 192000)])
        # self.assertEqual(rawio._sigs_length[0], 4608)
        # self.assertEqual(rawio._sigs_t_start[0], 0)
        # self.assertEqual(rawio._sigs_t_stop[0], 0.192)
        # self.assertEqual(len(rawio._sigs_memmaps), 1)
