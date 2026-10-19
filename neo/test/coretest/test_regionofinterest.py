import quantities as pq
from neo.core.regionofinterest import RectangularRegionOfInterest, CircularRegionOfInterest, PolygonRegionOfInterest
from neo.core.imagesequence import ImageSequence
import unittest


class Test_CircularRegionOfInterest(unittest.TestCase):

    def test_result(self):
        seq = ImageSequence([[[]]], spatial_scale=1, frame_duration=20 * pq.ms)
        # `is_inside` tests a closed disc, so radius 1 and radius 1.01 enclose the same
        # pixel centres. The radius 1 expectation used to be [[6, 5], [5, 6], [6, 6]],
        # which contradicted both the line below it and `is_inside` itself.
        self.assertEqual(
            (CircularRegionOfInterest(seq, 6, 6, 1).pixels_in_region()), [[6, 5], [5, 6], [6, 6], [7, 6], [6, 7]]
        )
        self.assertEqual(
            (CircularRegionOfInterest(seq, 6, 6, 1.01).pixels_in_region()), [[6, 5], [5, 6], [6, 6], [7, 6], [6, 7]]
        )

    def test_pixels_in_region_matches_is_inside(self):
        """Every pixel `is_inside` accepts must be enumerated by `pixels_in_region`."""
        seq = ImageSequence([[[]]], spatial_scale=1, frame_duration=20 * pq.ms)
        for x, y, radius in ((6, 6, 1), (10, 10, 5), (6, 6, 2.5), (7, 4, 3), (5, 5, 1.01)):
            roi = CircularRegionOfInterest(seq, x, y, radius)
            enumerated = {tuple(pixel) for pixel in roi.pixels_in_region()}
            margin = int(radius) + 2
            accepted = {
                (px, py)
                for py in range(y - margin, y + margin + 1)
                for px in range(x - margin, x + margin + 1)
                if roi.is_inside(px, py)
            }
            self.assertEqual(enumerated, accepted, f"disagreement for x={x}, y={y}, radius={radius}")

    def test_pixels_in_region_is_symmetric_about_centre(self):
        """The disc must reach as far right and up as it does left and down."""
        seq = ImageSequence([[[]]], spatial_scale=1, frame_duration=20 * pq.ms)
        roi = CircularRegionOfInterest(seq, 10, 10, 5)
        pixels = roi.pixels_in_region()

        # (15, 10) sits exactly `radius` from the centre, so the closed disc includes it.
        self.assertTrue(roi.is_inside(15, 10))
        self.assertIn([15, 10], pixels)
        self.assertIn([10, 15], pixels)

        xs = [pixel[0] for pixel in pixels]
        ys = [pixel[1] for pixel in pixels]
        self.assertEqual((min(xs), max(xs)), (5, 15))
        self.assertEqual((min(ys), max(ys)), (5, 15))
        self.assertEqual(len(pixels), 81)


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
