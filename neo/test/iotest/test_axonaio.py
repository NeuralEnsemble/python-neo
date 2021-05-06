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
    entities_to_download = [
        'axona'
    ]
    entities_to_test = [
        'axona/axona_raw.set',
        'axona/dataset_unit_spikes/20140815-180secs.set'
    ]


if __name__ == "__main__":
    unittest.main()
