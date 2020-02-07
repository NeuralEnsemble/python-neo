"""
Test of neo.io.asciiimageio
"""
import os
import unittest
import quantities as pq
from neo.io import AsciiImageIO
import numpy as np


class TestAsciiImageIO(unittest.TestCase):

    def test_read_txt(self):
        img = ''
        img_list = []
        for frame in range(20):
            img_list.append([])
            for y in range(50):
                img_list[frame].append([])
                for x in range(50):
                    img += str(x)
                    img += '\t'
                    img_list[frame][y].append(x)
        img_list = np.array(img_list)
        file_name = "txt_test_file.txt"
        file = open(file_name, mode="w")
        file.write(str(img))
        file.close()

        object = AsciiImageIO(file_name='txt_test_file.txt',
                              nb_frame=20, nb_row=50, nb_column=50, units='V',
                              sampling_rate=1 * pq.Hz, spatial_scale=1 * pq.micrometer)
        block = object.read_block()
        self.assertEqual(len(block.segments), 1)
        self.assertEqual(len(block.segments[0].imagesequences), 1)
        self.assertEqual(block.segments[0].imagesequences[0].shape, (20, 50, 50))
        self.assertEqual(block.segments[0].imagesequences[0].any(), img_list.any())

        file.close()
        os.remove(file_name)


if __name__ == "__main__":
    unittest.main()
