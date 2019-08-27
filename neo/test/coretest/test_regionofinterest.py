from neo.core.regionofinterest import RectangularRegionOfInterest, \
                                      CircularRegionOfInterest,\
                                      PolygonRegionOfInterest
import unittest


class Test_CircularRegionOfInterest(unittest.TestCase):




    def test_result(self):
        self.x = 5
        self.y = 5
        self.r = 5

        print(CircularRegionOfInterest(self.y,self.x,self.r).return_list_pixel())

        self.assertEqual(CircularRegionOfInterest(self.y,self.x,self.r).return_list_pixel(),
                         [[10, 5], [7, 6], [8, 6], [9, 6], [10, 6], [11, 6], [12, 6], [13, 6],
                          [6, 7], [7, 7], [8, 7], [9, 7], [10, 7], [11, 7], [12, 7], [13, 7],
                          [14, 7], [6, 8], [7, 8], [8, 8], [9, 8], [10, 8], [11, 8], [12, 8],
                          [13, 8], [14, 8], [6, 9], [7, 9], [8, 9], [9, 9], [10, 9], [11, 9],
                          [12, 9], [13, 9], [14, 9], [5, 10], [6, 10], [7, 10], [8, 10], [9, 10],
                          [10, 10], [11, 10], [12, 10], [13, 10], [14, 10], [15, 10], [6, 11],
                          [7, 11], [8, 11], [9, 11], [10, 11], [11, 11], [12, 11], [13, 11], [14, 11],
                          [6, 12], [7, 12], [8, 12], [9, 12], [10, 12], [11, 12], [12, 12], [13, 12],
                          [14, 12], [6, 13], [7, 13], [8, 13], [9, 13], [10, 13], [11, 13], [12, 13],
                          [13, 13], [14, 13], [7, 14], [8, 14], [9, 14], [10, 14], [11, 14], [12, 14],
                          [13, 14], [10, 15]])







class Test_RectangularRegionOfInterest(unittest.TestCase):
    pass


class Test_PolygonRegionOfInterest(unittest.TestCase):
    pass