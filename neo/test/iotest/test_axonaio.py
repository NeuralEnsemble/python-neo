"""
Tests of neo.io.axonaio
"""

import unittest

from neo.io.axonaio import AxonaIO
from neo.test.iotest.common_io_test import BaseTestIO
from neo.io.proxyobjects import (AnalogSignalProxy,
                SpikeTrainProxy, EventProxy, EpochProxy)
from neo import (AnalogSignal, SpikeTrain)

import quantities as pq
import numpy as np


class TestAxonaIO(BaseTestIO, unittest.TestCase, ):
    ioclass = AxonaIO

    files_to_download = [
        'axona_raw.set',
        'axona_raw.bin'
    ]
    files_to_test = [
        'axona_raw.set'
    ]


if __name__ == "__main__":
    unittest.main()
