import unittest
from pathlib import Path

from neo.rawio.spikegadgetsrawio import SpikeGadgetsRawIO
from neo.test.rawiotest.common_rawio_test import BaseTestRawIO
from numpy.testing import assert_array_equal


class TestSpikeGadgetsRawIO(
    BaseTestRawIO,
    unittest.TestCase,
):
    rawioclass = SpikeGadgetsRawIO
    entities_to_download = ["spikegadgets"]
    entities_to_test = [
        "spikegadgets/20210225_em8_minirec2_ac.rec",
        "spikegadgets/W122_06_09_2019_1_fromSD.rec",
        "spikegadgets/SpikeGadgets_test_data_2xNpix1.0_20240318_173658.rec",
    ]

    def test_parse_header_missing_channels(self):

        file_path = Path(self.get_local_path("spikegadgets/SL18_D19_S01_F01_BOX_SLP_20230503_112642_stubbed.rec"))
        reader = SpikeGadgetsRawIO(filename=file_path)
        reader.parse_header()

        assert_array_equal(
            reader.header["signal_channels"]["id"],
            # fmt: off
            [
                'ECU_Ain1', 'ECU_Ain2', 'ECU_Ain3', 'ECU_Ain4', 'ECU_Ain5', 'ECU_Ain6',
                'ECU_Ain7', 'ECU_Ain8', 'ECU_Aout1', 'ECU_Aout2', 'ECU_Aout3', 'ECU_Aout4', '0',
                '32', '96', '160', '192', '224', '1', '33', '65', '97', '161', '193', '225', '2', '34',
                '98', '162', '194', '226', '3', '35', '67', '99', '163', '195', '227', '4', '36',
                '100', '164', '196', '228', '5', '37', '69', '101', '165', '197', '229', '6', '38',
                '102', '166', '198', '230', '7', '39', '71', '103', '167', '199', '231', '8', '40',
                '72', '104', '136', '168', '200', '232', '9', '41', '73', '105', '137', '169', '201',
                '233', '10', '42', '74', '106', '138', '170', '202', '234', '11', '43', '75', '107',
                '139', '171', '203', '235', '12', '44', '76', '108', '140', '172', '204', '236', '13',
                '45', '77', '109', '141', '173', '205', '237', '14', '46', '78', '110', '142', '174',
                '206', '238', '15', '47', '79', '111', '143', '175', '207', '239', '80', '144', '176',
                '208', '240', '17', '49', '81', '145', '177', '209', '241', '82', '146', '178', '210',
                '242', '19', '51', '83', '147', '179', '211', '243', '84', '148', '180', '212', '244',
                '21', '53', '85', '149', '181', '213', '245', '86', '150', '182', '214', '246', '23',
                '55', '87', '151', '183', '215', '247', '24', '56', '88', '152', '184', '216', '248',
                '25', '57', '89', '121', '153', '185', '217', '249', '26', '58', '90', '154', '186',
                '218', '250', '27', '59', '91', '123', '155', '187', '219', '251', '28', '60', '92',
                '156', '188', '220', '252', '29', '61', '93', '125', '157', '189', '221', '253', '30',
                '62', '94', '158', '190', '222', '254', '31', '63', '95', '127', '159', '191', '223',
                '255'
            ]
            # fmt: on
        )
