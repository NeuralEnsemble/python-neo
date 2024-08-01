"""
Tests of neo.io.neuronexusio
"""

import unittest

from neo.io import NeuronexusIO
from neo.test.iotest.common_io_test import BaseTestIO


class TestNeuronexusIO(
    BaseTestIO,
    unittest.TestCase,
):
    ioclass = NeuronexusIO
    entities_to_download = ["neuronexus"]
    entities_to_test = ["neuronexus/allego_1/allego_2__uid0701-13-04-49.xdat.json"]


if __name__ == "__main__":
    unittest.main()
