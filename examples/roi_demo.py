"""
Example of working with RegionOfInterest objects
"""

import matplotlib.pyplot as plt
import numpy as np
from neo.core import CircularRegionOfInterest, RectangularRegionOfInterest, PolygonRegionOfInterest
from numpy.random import rand


def plot_roi(roi, shape):
    img = rand(120, 100)
    pir = np.array(roi.pixels_in_region()).T
    img[pir[1], pir[0]] = 5

    plt.imshow(img, cmap='gray_r')
    plt.clim(0, 5)

    ax = plt.gca()
    ax.add_artist(shape)


roi = CircularRegionOfInterest(x=50.3, y=50.8, radius=30.2)
shape = plt.Circle(roi.centre, roi.radius, color='r', fill=False)
plt.subplot(1, 3, 1)
plot_roi(roi, shape)

roi = RectangularRegionOfInterest(x=50.3, y=40.2, width=40.1, height=50.3)
shape = plt.Rectangle((roi.x - roi.width/2.0, roi.y - roi.height/2.0),
                      roi.width, roi.height, color='r', fill=False)
plt.subplot(1, 3, 2)
plot_roi(roi, shape)

roi = PolygonRegionOfInterest(
    (20.3, 30.2), (80.7, 30.1), (55.2, 59.4)
)
shape = plt.Polygon(np.array(roi.vertices), closed=True, color='r', fill=False)
plt.subplot(1, 3, 3)
plot_roi(roi, shape)

plt.show()
