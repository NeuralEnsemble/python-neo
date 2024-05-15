import unittest

from neo.rawio.intanrawio import IntanRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestIntanRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = IntanRawIO
    entities_to_download = ["intan"]
    entities_to_test = [
        "intan/intan_rhs_test_1.rhs",  # Format header-attached 
        "intan/intan_rhd_test_1.rhd",  # Format header attach
        "intan/intan_fpc_test_231117_052630/info.rhd",  # Format one-file-per-channel
        "intan/intan_fps_test_231117_052500/info.rhd",  # Format one file per signal
        "intan/intan_fpc_rhs_test_240329_091637/info.rhs",  # Format one-file-per-channel
        "intan/intan_fps_rhs_test_240329_091536/info.rhs",   # Format one-file-per-signal
        "intan/rhd_fpc_multistim_240514_082044/info.rhd",  # Multiple digital channels one-file-per-channel rhd
        # "intan/rhs_fpc_multistim_240514_082243/rhs_fpc_multistim_240514_082243.rhs",  # Multiple digital channels one-file-per-channel rhs
    ]



if __name__ == "__main__":
    unittest.main()
