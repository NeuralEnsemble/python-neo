import unittest
from pathlib import Path
import tempfile

import numpy as np

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
        "spikegadgets/neuropixels2_4shank/20260122_134412_merged_cropped_1min_NP2.rec",
        "spikegadgets/msh_1024ch/msh_1024ch.rec",
        "spikegadgets/SpikeGadgets_test_data_Pixie384_20261104_cropped.rec",
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

    def test_neuropixels_uses_hwchan_ids(self):
        # Regression test for Neuropixels channel id and name semantics.
        # ids are f"hwChan{i}" and names are f"probe{spikeSortingGroup}_chan{i}",
        # where i is the channel index in the trodes stream (which equals the hwChan
        # the firmware writes at that byte position, since the SpikeGadgets MCU emits
        # Neuropixels samples in hwChan ascending order).
        file_path = Path(self.get_local_path("spikegadgets/SpikeGadgets_test_data_2xNpix1.0_20240318_173658.rec"))
        reader = SpikeGadgetsRawIO(filename=file_path)
        reader.parse_header()

        trodes_mask = reader.header["signal_channels"]["stream_id"] == "trodes"
        trodes_ids = list(reader.header["signal_channels"]["id"][trodes_mask])
        trodes_names = list(reader.header["signal_channels"]["name"][trodes_mask])

        self.assertEqual(trodes_ids[:4], ["hwChan0", "hwChan1", "hwChan2", "hwChan3"])
        self.assertEqual(trodes_ids[-4:], ["hwChan764", "hwChan765", "hwChan766", "hwChan767"])
        self.assertEqual(len(trodes_ids), 768)

        # Names embed the SpikeChannel spikeSortingGroup attribute. The two-probe NP1
        # workspace sets spikeSortingGroup=0 for one probe and =1 for the other; the
        # boundary is interleaved across channel indices (not at i=384) because the
        # two probes' hwChans interleave in chip blocks.
        self.assertEqual(trodes_names[0], "probe0_chan0")
        self.assertEqual(trodes_names[767], "probe1_chan767")
        groups = {n.split("_")[0] for n in trodes_names}
        self.assertEqual(groups, {"probe0", "probe1"})

    def test_nric_uses_hwchan_ids(self):
        # NRIC (Pixie384 headstage, device="neuropixelsNRIC") recordings share the
        # Neuropixels hwChan-ascending layout and therefore the same id/name
        # semantics: ids are f"hwChan{i}" and names are f"probe{spikeSortingGroup}_chan{i}".
        file_path = Path(self.get_local_path("spikegadgets/SpikeGadgets_test_data_Pixie384_20261104_cropped.rec"))
        reader = SpikeGadgetsRawIO(filename=file_path)
        reader.parse_header()

        trodes_mask = reader.header["signal_channels"]["stream_id"] == "trodes"
        trodes_ids = list(reader.header["signal_channels"]["id"][trodes_mask])
        trodes_names = list(reader.header["signal_channels"]["name"][trodes_mask])

        self.assertEqual(len(trodes_ids), 384)
        self.assertEqual(trodes_ids[0], "hwChan0")
        self.assertEqual(trodes_ids[-1], "hwChan383")
        # all channels of this recording belong to a single spikeSortingGroup "0"
        self.assertEqual(trodes_names[0], "probe0_chan0")
        self.assertEqual(trodes_names[-1], "probe0_chan383")

        # gain comes from the SpikeNTrode spikeScalingToUv attribute
        trodes_gains = reader.header["signal_channels"]["gain"][trodes_mask]
        trodes_units = reader.header["signal_channels"]["units"][trodes_mask]
        self.assertEqual(set(trodes_units), {"uV"})
        self.assertTrue(np.allclose(trodes_gains, 0.018311105685598315))

    def test_nric_traces_match_trodes_export(self):
        # Cross-check the decoded ephys region against Trodes' own export of the same
        # recording (raw_group0.dat, Fields: <voltage 384*int16>): this pins the
        # hwChan-ascending layout of the NRIC device to an independent decoder.
        file_path = Path(self.get_local_path("spikegadgets/SpikeGadgets_test_data_Pixie384_20261104_cropped.rec"))
        reader = SpikeGadgetsRawIO(filename=file_path)
        reader.parse_header()

        stream_index = list(reader.header["signal_streams"]["id"]).index("trodes")
        n_samples = 1000
        traces = reader.get_analogsignal_chunk(
            block_index=0, seg_index=0, i_start=0, i_stop=n_samples, stream_index=stream_index
        )
        self.assertEqual(traces.dtype, np.dtype("int16"))
        self.assertEqual(traces.shape, (n_samples, 384))

        dat_path = file_path.with_name("SpikeGadgets_test_data_Pixie384_20261104_trodes_raw_cropped.dat")
        with open(dat_path, "rb") as f:
            head = f.read(65536)
            end = head.find(b"<End settings>")
            offset = head.find(b"\n", end) + 1
        reference = np.memmap(dat_path, dtype="<i2", mode="r", offset=offset, shape=(n_samples, 384))

        assert_array_equal(traces, reference)

    def test_opening_gibberish_file(self):
        """Test that parsing a file without </Configuration> raises ValueError instead of infinite loop."""
        # Create a temporary file with gibberish content that doesn't have the required tag
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".rec") as temp_file:
            # Write simple gibberish content without the required </Configuration> tag
            temp_file.write(b"gibberish\n")
            temp_file.flush()

            reader = SpikeGadgetsRawIO(filename=temp_file.name)
            with self.assertRaises(ValueError) as cm:
                reader.parse_header()

            self.assertIn("xml header does not contain '</Configuration>'", str(cm.exception))
