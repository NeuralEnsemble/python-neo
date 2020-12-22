"""
Tests of neo.rawio.phyrawio

Author: Regimantas Jurkus

"""

import unittest

from neo.rawio.phyrawio import PhyRawIO

from neo.test.rawiotest.common_rawio_test import BaseTestRawIO

import csv
import tempfile
from pathlib import Path
from collections import OrderedDict
import sys


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

    def test_csv_tsv_parser_with_csv(self):
        csv_tempfile = Path(tempfile.gettempdir()).joinpath('test.csv')
        with open(csv_tempfile, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['Header 1', 'Header 2'])
            csv_writer.writerow(['Value 1', 'Value 2'])

        # the parser in PhyRawIO runs csv.DictReader to parse the file
        # csv.DictReader for python version 3.6+ returns list of OrderedDict
        if (3, 6) <= sys.version_info < (3, 8):
            target = [OrderedDict({'Header 1': 'Value 1',
                                   'Header 2': 'Value 2'})]

        # csv.DictReader for python version 3.8+ returns list of dict
        elif sys.version_info >= (3, 8):
            target = [{'Header 1': 'Value 1', 'Header 2': 'Value 2'}]

        list_of_dict = PhyRawIO._parse_tsv_or_csv_to_list_of_dict(csv_tempfile)

        self.assertEqual(target, list_of_dict)

    def test_csv_tsv_parser_with_tsv(self):
        tsv_tempfile = Path(tempfile.gettempdir()).joinpath('test.tsv')
        with open(tsv_tempfile, 'w') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            tsv_writer.writerow(['Header 1', 'Header 2'])
            tsv_writer.writerow(['Value 1', 'Value 2'])

        # the parser in PhyRawIO runs csv.DictReader to parse the file
        # csv.DictReader for python version 3.6+ returns list of OrderedDict
        if (3, 6) <= sys.version_info < (3, 8):
            target = [OrderedDict({'Header 1': 'Value 1',
                                   'Header 2': 'Value 2'})]

        # csv.DictReader for python version 3.8+ returns list of dict
        elif sys.version_info >= (3, 8):
            target = [{'Header 1': 'Value 1', 'Header 2': 'Value 2'}]

        list_of_dict = PhyRawIO._parse_tsv_or_csv_to_list_of_dict(tsv_tempfile)

        self.assertEqual(target, list_of_dict)

    def test_csv_tsv_parser_error_raising(self):
        txt_tempfile = Path(tempfile.gettempdir()).joinpath('test.txt')
        with open(txt_tempfile, 'w') as txt_file:
            txt_file.write('This is a test')

        self.assertRaises(ValueError,
                          PhyRawIO._parse_tsv_or_csv_to_list_of_dict,
                          txt_tempfile)


if __name__ == "__main__":
    unittest.main()
