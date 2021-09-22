"""
Tests of neo.io.monkeylogicio
"""

import unittest

from neo.io import MonkeyLogicIO
from neo.test.iotest.common_io_test import BaseTestIO

# class TestMonkeyLogicIO(BaseTestIO, unittest.TestCase):
#     entities_to_download = [
#         'monkeylogic'
#     ]
#     entities_to_test = [
#         'monkeylogic/mearec_test_10s.h5'
#     ]
#     ioclass = MonkeyLogicIO


class TestMonkeyLogicIO(unittest.TestCase):
    # TODO: Adjust this once ML files are on GIN

    def test_read(self):
        filename = '/home/sprengerj/projects/monkey_logic/210909_TSCM_5cj_5cl_Riesling.bhv2'
        filename = '/home/sprengerj/projects/monkey_logic/sabrina/210810__learndms_userloop.bhv2'
        # filename = '/home/sprengerj/projects/monkey_logic/sabrina/210916__learndms_userloop.bhv2'
        # filename = '/home/sprengerj/projects/monkey_logic/sabrina/210917__learndms_userloop.bhv2'
        io = MonkeyLogicIO(filename)
        io.read_block()

if __name__ == "__main__":
    unittest.main()
