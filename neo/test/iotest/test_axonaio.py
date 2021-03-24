"""
Tests of neo.io.axonaio
"""

import unittest

from neo.io.axonaio import AxonaIO  # , HAVE_SCIPY
from neo.test.iotest.common_io_test import BaseTestIO
from neo.io.proxyobjects import (AnalogSignalProxy,
                SpikeTrainProxy, EventProxy, EpochProxy)
from neo import (AnalogSignal, SpikeTrain)

import quantities as pq
import numpy as np


# This run standart tests, this is mandatory for all IO
class TestAxonaIO(BaseTestIO, unittest.TestCase, ):
    ioclass = AxonaIO

    files_to_download = [
        'axona_raw.set', 
        'axona_raw.bin'
    ]
    files_to_test = files_to_download


if __name__ == "__main__":
    unittest.main()

# eof

