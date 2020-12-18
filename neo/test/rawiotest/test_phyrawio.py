"""
Tests of neo.rawio.examplerawio

Note for dev:
if you write a new RawIO class your need to put some file
to be tested at g-node portal, Ask neuralensemble list for that.
The file need to be small.

Then you have to copy/paste/renamed the TestExampleRawIO
class and a full test will be done to test if the new coded IO
is compliant with the RawIO API.

If you have problems, do not hesitate to ask help github (prefered)
of neuralensemble list.

Note that same mechanism is used a neo.io API so files are tested
several time with neo.rawio (numpy buffer) and neo.io (neo object tree).
See neo.test.iotest.*


Author: Samuel Garcia

"""

import unittest

from neo.rawio.phyrawio import PhyRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO


class TestPhyRawIO(BaseTestRawIO, unittest.TestCase):
    rawioclass = PhyRawIO
    files_to_download = [
        'phy_example_0/spike_times.npy',
        'phy_example_0/spike_templates.npy',
        'phy_example_0/spike_clusters.npy',
        'phy_example_0/params.py',
        'phy_example_0/cluster_KSLabel.tsv',
        'phy_example_0/cluster_ContamPct.tsv',
        'phy_example_0/cluster_Amplitude.tsv',
        'phy_example_0/cluster_group.tsv',
    ]
    entities_to_test = ['phy_example_0']


if __name__ == "__main__":
    unittest.main()
