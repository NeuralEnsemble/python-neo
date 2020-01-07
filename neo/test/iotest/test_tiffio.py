import unittest
import os
from PIL import Image
import numpy as np
import shutil
from neo.io.tiffio import TiffIO
import quantities as pq


class TestTiffIO(unittest.TestCase):

    def test_read_group_of_tiff_grayscale(self):
        directory = 'test_tiff'
        if not os.path.exists(directory):
            os.makedirs(directory)
        # directory is live
        img = []
        for picture in range(10):
            img.append([])
            for y in range(50):
                img[picture].append([])
                for x in range(50):
                    img[picture][y].append(x)
        img = np.array(img, dtype=float)
        for image in range(10):
            Image.fromarray(img[image]).save(directory+'/tiff_exemple'+str(image)+".tif")

        ioclass = TiffIO(directory_path=directory, units='V', sampling_rate=1.0*pq.Hz,
                         spatial_scale=1.0*pq.micrometer)
        blck = ioclass.read_block()
        self.assertEqual(len(blck.segments), 1)
        self.assertEqual(len(blck.segments[0].imagesequences), 1)
        self.assertEqual(blck.segments[0].imagesequences[0].any(), img.any())
        self.assertEqual(blck.segments[0].imagesequences[0].sampling_rate, 1.0*pq.Hz)
        self.assertEqual(blck.segments[0].imagesequences[0].spatial_scale, 1.0 * pq.micrometer)

        # end of directory
        shutil.rmtree(directory)


if __name__ == "__main__":
    unittest.main()
