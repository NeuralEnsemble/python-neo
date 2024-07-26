"""
Tests of neo.io.intanio
"""

import unittest

from neo.io import IntanIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestIntanIO(
    BaseTestIO,
    unittest.TestCase,
):
    ioclass = IntanIO
    entities_to_download = ["intan"]
    entities_to_test = [
        "intan/intan_rhs_test_1.rhs",  # Format header-attached
        "intan/intan_rhd_test_1.rhd",  # Format header-attached
        "intan/rhs_fpc_multistim_240514_082243/rhs_fpc_multistim_240514_082243.rhs",  # Format header-attached newer version
        "intan/intan_fpc_test_231117_052630/info.rhd",  # Format one-file-per-channel
        "intan/intan_fps_test_231117_052500/info.rhd",  # Format one file per signal
        "intan/intan_fpc_rhs_test_240329_091637/info.rhs",  # Format one-file-per-channel
        "intan/intan_fps_rhs_test_240329_091536/info.rhs",  # Format one-file-per-signal
        "intan/rhd_fpc_multistim_240514_082044/info.rhd",  # Multiple digital channels one-file-per-channel rhd
    ]


if __name__ == "__main__":
    unittest.main()
