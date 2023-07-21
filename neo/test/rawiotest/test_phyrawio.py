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
    entities_to_download = [
        'phy'
    ]
    entities_to_test = [
        'phy/phy_example_0'
    ]

    def test_csv_tsv_parser_with_csv(self):
        csv_tempfile = Path(tempfile.gettempdir()).joinpath('test.csv')
        with open(csv_tempfile, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['cluster_id', 'some_annotation', 'some_other_annotation'])
            csv_writer.writerow([1, 'Good', 'Bad'])
            csv_writer.writerow([2, 10, -2])
            csv_writer.writerow([3, 1.23, -0.38])

        # the parser in PhyRawIO runs csv.DictReader to parse the file
        # csv.DictReader for python version 3.6+ returns list of OrderedDict
        if (3, 6) <= sys.version_info < (3, 8):
            target = [OrderedDict({'cluster_id': 1,
                                   'some_annotation': 'Good',
                                   'some_other_annotation': 'Bad'}),
                      OrderedDict({'cluster_id': 2,
                                   'some_annotation': 10,
                                   'some_other_annotation': -2}),
                      OrderedDict({'cluster_id': 3,
                                   'some_annotation': 1.23,
                                   'some_other_annotation': -0.38})]

        # csv.DictReader for python version 3.8+ returns list of dict
        elif sys.version_info >= (3, 8):
            target = [{'cluster_id': 1,
                       'some_annotation': 'Good',
                       'some_other_annotation': 'Bad'},
                      {'cluster_id': 2,
                       'some_annotation': 10,
                       'some_other_annotation': -2},
                      {'cluster_id': 3,
                       'some_annotation': 1.23,
                       'some_other_annotation': -0.38}]

        list_of_dict = PhyRawIO._parse_tsv_or_csv_to_list_of_dict(csv_tempfile)

        self.assertEqual(target, list_of_dict)

    def test_csv_tsv_parser_with_tsv(self):
        tsv_tempfile = Path(tempfile.gettempdir()).joinpath('test.tsv')
        with open(tsv_tempfile, 'w') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            tsv_writer.writerow(['cluster_id', 'some_annotation'])
            tsv_writer.writerow([1, 'Good'])
            tsv_writer.writerow([2, 10])
            tsv_writer.writerow([3, 1.23])

        # the parser in PhyRawIO runs csv.DictReader to parse the file
        # csv.DictReader for python version 3.6+ returns list of OrderedDict
        if (3, 6) <= sys.version_info < (3, 8):
            target = [OrderedDict({'cluster_id': 1,
                                   'some_annotation': 'Good'}),
                      OrderedDict({'cluster_id': 2,
                                   'some_annotation': 10}),
                      OrderedDict({'cluster_id': 3,
                                   'some_annotation': 1.23})]

        # csv.DictReader for python version 3.8+ returns list of dict
        elif sys.version_info >= (3, 8):
            target = [{'cluster_id': 1, 'some_annotation': 'Good'},
                      {'cluster_id': 2, 'some_annotation': 10},
                      {'cluster_id': 3, 'some_annotation': 1.23}]

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
