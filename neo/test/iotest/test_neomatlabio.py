"""
Tests of neo.io.neomatlabio
"""

import os
import unittest
import pytest
from numpy.testing import assert_array_equal
import quantities as pq

from neo.core.analogsignal import AnalogSignal
from neo.core.irregularlysampledsignal import IrregularlySampledSignal
from neo import Block, Segment, SpikeTrain, ImageSequence, Group
from neo.test.iotest.common_io_test import BaseTestIO
from neo.test.generate_datasets import random_block

from neo.io.neomatlabio import NeoMatlabIO

try:
    import scipy.io

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


@unittest.skipUnless(HAVE_SCIPY, "requires scipy")
class TestNeoMatlabIO(BaseTestIO, unittest.TestCase):
    ioclass = NeoMatlabIO
    files_to_test = []
    files_to_download = []

    def test_write_read_single_spike(self):
        block1 = Block(name="test_neomatlabio")
        seg = Segment("segment1")
        spiketrain1 = SpikeTrain([1] * pq.s, t_stop=10 * pq.s, sampling_rate=1 * pq.Hz)
        spiketrain1.annotate(yep="yop", yip=None, yop=[3, 4, 5] * pq.ms)
        sig1 = AnalogSignal([4, 5, 6] * pq.A, sampling_period=1 * pq.ms)
        irrsig1 = IrregularlySampledSignal([0, 1, 2] * pq.ms, [4, 5, 6] * pq.A)
        img_sequence_array = [[[column for column in range(2)] for _ in range(2)] for _ in range(2)]
        image_sequence = ImageSequence(
            img_sequence_array, units="dimensionless", sampling_rate=1 * pq.Hz, spatial_scale=1 * pq.micrometer
        )
        block1.segments.append(seg)
        seg.spiketrains.append(spiketrain1)
        seg.analogsignals.append(sig1)
        seg.irregularlysampledsignals.append(irrsig1)
        seg.imagesequences.append(image_sequence)

        group1 = Group([spiketrain1, sig1])
        block1.groups.append(group1)

        # write block
        filename = self.get_local_path("matlabiotestfile.mat")
        io1 = self.ioclass(filename)
        io1.write_block(block1)

        # read block
        io2 = self.ioclass(filename)
        block2 = io2.read_block()

        self.assertEqual(block1.segments[0].spiketrains[0], block2.segments[0].spiketrains[0])

        assert_array_equal(block1.segments[0].analogsignals[0], block2.segments[0].analogsignals[0])

        assert_array_equal(
            block1.segments[0].irregularlysampledsignals[0].magnitude,
            block2.segments[0].irregularlysampledsignals[0].magnitude,
        )
        assert_array_equal(
            block1.segments[0].irregularlysampledsignals[0].times, block2.segments[0].irregularlysampledsignals[0].times
        )

        assert_array_equal(block1.segments[0].imagesequences[0], block2.segments[0].imagesequences[0])

        # test annotations
        spiketrain2 = block2.segments[0].spiketrains[0]
        assert spiketrain2.annotations["yep"] == "yop"
        assert spiketrain2.annotations["yip"] is None
        # a quantity annotation keeps its units and does not leak its companion field,
        # see https://github.com/NeuralEnsemble/python-neo/issues/852
        assert "yop_units" not in spiketrain2.annotations
        assert spiketrain2.annotations["yop"].dimensionality == pq.ms.dimensionality
        assert_array_equal(spiketrain2.annotations["yop"].magnitude, [3, 4, 5])

        # test group retrieval
        group2 = block2.groups[0]
        assert_array_equal(group1.analogsignals[0], group2.analogsignals[0])

    def test_write_read_random_blocks(self):
        for i in range(10):
            # generate random block
            block1 = random_block()

            # write block to file
            filename_orig = self.get_local_path(f"matlabio_randomtest_orig_{i}.mat")
            io1 = self.ioclass(filename_orig)
            io1.write_block(block1)

            # read block
            io2 = self.ioclass(filename_orig)
            block2 = io2.read_block()

            filename_roundtripped = self.get_local_path(f"matlabio_randomtest_roundtrip_{i}.mat")
            io3 = self.ioclass(filename_roundtripped)
            io3.write_block(block2)

            # the actual contents will differ since we're using Python object id as identifiers
            # but at least the file size should be the same
            assert os.stat(filename_orig).st_size == os.stat(filename_roundtripped).st_size


@unittest.skipUnless(HAVE_SCIPY, "requires scipy")
@pytest.mark.parametrize(
    "annotation",
    [
        "a string",
        42,
        None,
        [3, 4, 5],
        [3, 4, 5] * pq.ms,
        3 * pq.ms,
        {"a": 1, "b": None},
        {"a": [1, 2] * pq.mV, "b": {"deep": 5 * pq.Hz}},
    ],
    ids=["str", "int", "none", "array", "quantity_array", "quantity_scalar", "dict", "nested_dict"],
)
def test_write_read_annotation_roundtrip(tmp_path, annotation):
    """An annotation must survive a write then read unchanged.

    A test case from Issue #852 (https://github.com/NeuralEnsemble/python-neo/issues/852).
    Quantity annotations used to come back as bare magnitudes plus a stray ``<key>_units``
    key, nested mappings came back as ``scipy.io`` structs, and any array-valued
    annotation raised ``ValueError: The truth value of an array ... is ambiguous`` while
    being compared against the sentinel that stands in for `None`.

    This runs off a temporary file rather than the downloaded test data, so it also
    covers installations without datalad.
    """
    block = Block(name="test_annotations")
    segment = Segment(name="segment1")
    block.segments.append(segment)
    spiketrain = SpikeTrain([1, 2, 3] * pq.s, t_stop=10 * pq.s, yop=annotation)
    segment.spiketrains.append(spiketrain)
    segment.check_relationships()

    filename = tmp_path / "annotations.mat"
    NeoMatlabIO(filename=filename).write_block(block)
    read_back = NeoMatlabIO(filename=filename).read_block().segments[0].spiketrains[0].annotations

    assert list(read_back) == ["yop"], "no companion field should leak into the annotations"
    _assert_annotation_equal(read_back["yop"], annotation)


def _assert_annotation_equal(actual, expected):
    if expected is None:
        assert actual is None
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert sorted(actual) == sorted(expected)
        for key in expected:
            _assert_annotation_equal(actual[key], expected[key])
    elif isinstance(expected, pq.Quantity):
        assert isinstance(actual, pq.Quantity)
        assert actual.dimensionality == expected.dimensionality
        assert_array_equal(actual.magnitude, expected.magnitude)
    elif isinstance(expected, list):
        assert_array_equal(actual, expected)
    else:
        assert actual == expected


if __name__ == "__main__":
    unittest.main()
