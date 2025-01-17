import unittest
import numpy as np

from neo.rawio.neuronexusrawio import NeuroNexusRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestNeuroNexusRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = NeuroNexusRawIO
    entities_to_download = ["neuronexus"]
    entities_to_test = ["neuronexus/allego_1/allego_2__uid0701-13-04-49.xdat.json"]


if __name__ == "__main__":
    unittest.main()
