from math import floor, ceil


class RegionOfInterest:
    """Abstract base class"""
    pass


class CircularRegionOfInterest(RegionOfInterest):
    """Representation of a circular ROI

    *Usage:*

    >>> roi = CircularRegionOfInterest(20.0, 20.0, radius=5.0)
    >>> signal = image_sequence.signal_from_region(roi)

    *Required attributes/properties*:
        :x, y: (integers or floats)
            Pixel coordinates of the centre of the ROI
        :radius: (integer or float)
            Radius of the ROI in pixels
    """

    def __init__(self, x, y, radius):

        self.y = y
        self.x = x
        self.radius = radius

    @property
    def centre(self):
        return (self.x, self.y)

    @property
    def center(self):
        return self.centre

    def is_inside(self, x, y):
        if ((x - self.x) * (x - self.x) +
                (y - self.y) * (y - self.y) <= self.radius * self.radius):
            return True
        else:
            return False

    def pixels_in_region(self):
        """Returns a list of pixels whose *centres* are within the circle"""
        pixel_in_list = []
        for y in range(int(floor(self.y - self.radius)), int(ceil(self.y + self.radius))):
            for x in range(int(floor(self.x - self.radius)), int(ceil(self.x + self.radius))):
                if self.is_inside(x, y):
                    pixel_in_list.append([x, y])

        return pixel_in_list


class RectangularRegionOfInterest(RegionOfInterest):
    """Representation of a rectangular ROI

    *Usage:*

    >>> roi = RectangularRegionOfInterest(20.0, 20.0, width=5.0, height=5.0)
    >>> signal = image_sequence.signal_from_region(roi)

    *Required attributes/properties*:
        :x, y: (integers or floats)
            Pixel coordinates of the centre of the ROI
        :width: (integer or float)
            Width (x-direction) of the ROI in pixels
        :height: (integer or float)
            Height (y-direction) of the ROI in pixels
    """

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def is_inside(self, x, y):
        if (self.x - self.width/2.0 <= x < self.x + self.width/2.0
                and self.y - self.height/2.0 <= y < self.y + self.height/2.0):
            return True
        else:
            return False

    def pixels_in_region(self):
        """Returns a list of pixels whose *centres* are within the rectangle"""
        pixel_list = []
        h = self.height
        w = self.width
        for y in range(int(floor(self.y - h / 2.0)), int(ceil(self.y + h / 2.0))):
            for x in range(int(floor(self.x - w / 2.0)), int(ceil(self.x + w / 2.0))):
                if self.is_inside(x, y):
                    pixel_list.append([x, y])
        return pixel_list


class PolygonRegionOfInterest(RegionOfInterest):
    """Representation of a polygonal ROI

    *Usage:*

    >>> roi = PolygonRegionOfInterest(
    ...     (20.0, 20.0),
    ...     (30.0, 20.0),
    ...     (25.0, 25.0)
    ... )
    >>> signal = image_sequence.signal_from_region(roi)

    *Required attributes/properties*:
        :vertices:
            tuples containing the (x, y) coordinates, as integers or floats,
            of the vertices of the polygon
    """

    def __init__(self, *vertices):
        self.vertices = vertices

    def polygon_ray_casting(self, bounding_points, bounding_box_positions):

        # from https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
        # user  Noresourses

        # Arrays containing the x- and y-coordinates of the polygon's vertices.
        vertx = [point[0] for point in bounding_points]
        verty = [point[1] for point in bounding_points]
        # Number of vertices in the polygon
        nvert = len(bounding_points)
        # Points that are inside
        points_inside = []

        # For every candidate position within the bounding box
        for idx, pos in enumerate(bounding_box_positions):
            testx, testy = (pos[0], pos[1])
            c = 0
            for i in range(0, nvert):
                j = i - 1 if i != 0 else nvert - 1
                if (((verty[i]*1.0 > testy*1.0) != (verty[j]*1.0 > testy*1.0)) and
                        (testx*1.0 < (vertx[j]*1.0 - vertx[i]*1.0) * (testy*1.0 - verty[i]*1.0) /
                         (verty[j]*1.0 - verty[i]*1.0) + vertx[i]*1.0)):
                    c += 1
            # If odd, that means that we are inside the polygon
            if c % 2 == 1:
                points_inside.append(pos)

        return points_inside

    def pixels_in_region(self):

        min_x, max_x, min_y, max_y = (self.vertices[0][0], self.vertices[0][0],
                                      self.vertices[0][1], self.vertices[0][1])

        for i in self.vertices:
            if i[0] < min_x:
                min_x = i[0]
            if i[0] > max_x:
                max_x = i[0]
            if i[1] < min_y:
                min_y = i[1]
            if i[1] > max_y:
                max_y = i[1]

        list_coord = []
        for y in range(int(floor(min_y)), int(ceil(max_y))):
            for x in range(int(floor(min_x)), int(ceil(max_x))):
                list_coord.append((x, y))

        pixel_list = self.polygon_ray_casting(self.vertices, list_coord)

        return pixel_list
