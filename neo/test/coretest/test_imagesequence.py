import unittest
from neo.core import ImageSequence
from neo.core.regionofinterest import CircularRegionOfInterest, RectangularRegionOfInterest, PolygonRegionOfInterest
import quantities as pq
import numpy as np
from neo.core import Block, Segment


class TestImageSequence(unittest.TestCase):

    def setUp(self):
        self.data = []
        for frame in range(20):
            self.data.append([])
            for y in range(50):
                self.data[frame].append([])
                for x in range(50):
                    self.data[frame][y].append(x)

    def test_sampling_rate(self):
        # test if error is raise when not giving sample_rate
        with self.assertRaises(ValueError):
            ImageSequence(self.data, units='V', spatial_scale='m')

        with self.assertRaises(TypeError):
            ImageSequence(self.data, sampling_period=500, units='V', spatial_scale='m')
        # test if error is raise when not giving frequency at sampling rate
        with self.assertRaises(TypeError):
            ImageSequence(self.data, units='V', sampling_rate=500, spatial_scale='m')

    def test_error_spatial_scale(self):
        # test if error is raise when not giving spatial scale
        with self.assertRaises(ValueError):
            ImageSequence(self.data, units='V', sampling_rate=500 * pq.Hz)

    def test_units(self):
        with self.assertRaises(TypeError):
            ImageSequence(self.data, sampling_rate=500 * pq.Hz, spatial_scale='m')

    def test_wrong_dimensions(self):
        l = ImageSequence(self.data, sampling_rate=500 * pq.Hz, units='V',
                          spatial_scale='m')

        self.assertEqual(l.sampling_rate, 500 * pq.Hz)
        self.assertEqual(l.spatial_scale, 'm')
        # giving wrong dimension test if it give an error
        with self.assertRaises(ValueError):
            ImageSequence([[0, 1, 2, 4, 2], [0, 1, 2, 4, 5]],
                          sampling_rate=500 * pq.Hz, units='V', spatial_scale='m')


class TestMethodImageSequence(unittest.TestCase):

    def fake_region_of_interest(self):
        self.rect_ROI = RectangularRegionOfInterest(2, 2, 2, 2)
        self.data = []
        for frame in range(25):
            self.data.append([])
            for y in range(5):
                self.data[frame].append([])
                for x in range(5):
                    self.data[frame][y].append(x)

    def test_signal_from_region(self):
        self.fake_region_of_interest()
        l = ImageSequence(self.data, units='V', sampling_rate=500 * pq.Hz, spatial_scale='m').signal_from_region(
            self.rect_ROI)
        self.assertIsInstance(l, list)
        for i in range(len(l)):
            self.assertIsInstance(l[i], object)
        with self.assertRaises(ValueError):  # no pixels in region
            ImageSequence(self.data, units='V', sampling_rate=500 * pq.Hz,
                          spatial_scale='m').signal_from_region(RectangularRegionOfInterest(1, 1, 0, 0))
        with self.assertRaises(ValueError):
            ImageSequence(self.data, units='V', sampling_rate=500 *
                          pq.Hz, spatial_scale='m').signal_from_region()


if __name__ == "__main__":
    unittest.main()
