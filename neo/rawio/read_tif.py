from neo.core import ImageSequence
from PIL import Image
import numpy as np
import os


def read_sequence(path, units, sampling_rate, spatial_scale):
    
    # research all the image in the given directory 
    file_name_list = os.listdir(path)
    liste_data_image = []
    for i in file_name_list:
        liste_data_image.append(np.array(Image.open(path + "/" + i)))
    liste_data_image = np.array(liste_data_image, dtype=np.float)
    image_sequence = ImageSequence(liste_data_image, units=units,
                                   sampling_rate=sampling_rate, spatial_scale=spatial_scale)
    return image_sequence

