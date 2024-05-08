import quantities as pq
from neo.core.regionofinterest import RectangularRegionOfInterest, CircularRegionOfInterest, PolygonRegionOfInterest
from neo.core.imagesequence import ImageSequence
import unittest


class Test_CircularRegionOfInterest(unittest.TestCase):

    def test_result(self):
        seq = ImageSequence([[[]]], spatial_scale=1, frame_duration=20 * pq.ms)
        self.assertEqual((CircularRegionOfInterest(seq, 6, 6, 1).pixels_in_region()), [[6, 5], [5, 6], [6, 6]])
        self.assertEqual(
            (CircularRegionOfInterest(seq, 6, 6, 1.01).pixels_in_region()), [[6, 5], [5, 6], [6, 6], [7, 6], [6, 7]]
        )


class Test_RectangularRegionOfInterest(unittest.TestCase):

    def test_result(self):
        seq = ImageSequence([[[]]], spatial_scale=1, frame_duration=20 * pq.ms)
        self.assertEqual(
            RectangularRegionOfInterest(seq, 5, 5, 2, 2).pixels_in_region(), [[4, 4], [5, 4], [4, 5], [5, 5]]
        )


class Test_PolygonRegionOfInterest(unittest.TestCase):

    def test_result(self):
        seq = ImageSequence([[[]]], spatial_scale=1, frame_duration=20 * pq.ms)
        self.assertEqual(
            PolygonRegionOfInterest(seq, (3, 3), (2, 5), (5, 5), (5, 1), (1, 1)).pixels_in_region(),
            [(1, 1), (2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2), (3, 3), (4, 3), (3, 4), (4, 4)],
        )


if __name__ == "__main__":
    unittest.main()
