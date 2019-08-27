from neo.core.regionofinterest import RectangularRegionOfInterest, \
                                      CircularRegionOfInterest,\
                                      PolygonRegionOfInterest
import unittest


class Test_CircularRegionOfInterest(unittest.TestCase):

    def test_result(self):
        
        self.assertEqual((CircularRegionOfInterest(6,6,1).return_list_pixel()),
                         [[6, 5], [5, 6], [6, 6], [7, 6], [6, 7]])

class Test_RectangularRegionOfInterest(unittest.TestCase):
    
    def test_result(self):
        self.assertEqual(RectangularRegionOfInterest(5,5,2,2).return_list_pixel(),
                         [[4, 4], [5, 4], [4, 5], [5, 5]])
        


class Test_PolygonRegionOfInterest(unittest.TestCase):
    
    def test_result(self):
        self.assertEqual(PolygonRegionOfInterest((1,1),(1,4),(2,1),(4,1)).return_list_pixel(),
                         [(1, 1), (1, 2), (1, 3)])
        
        


if __name__ == "__main__":
    unittest.main()
