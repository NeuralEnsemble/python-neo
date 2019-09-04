class RegionOfInterest:
    pass


class CircularRegionOfInterest(RegionOfInterest):

    def __init__(self, x, y, radius):

        self.y = y
        self.x = x
        self.radius = radius

    def is_inside(self, circle_x, circle_y, rad, x, y):

        if ((x - circle_x) * (x - circle_x) +
                (y - circle_y) * (y - circle_y) <= rad * rad):
            return True
        else:
            return False

    def return_list_pixel(self):

        pixel_in_list = []
        for y in range(self.y - self.radius, self.y + self.radius + 1):
            for x in range(self.x - self.radius, self.x + self.radius + 1):

                if self.is_inside(self.x, self.y, self.radius, x, y):
                    pixel_in_list.append([x, y])

        return pixel_in_list


class RectangularRegionOfInterest(RegionOfInterest):

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def return_list_pixel(self):

        pixel_list = []
        for y in range(self.y - (int(self.h / 2)), self.y + (int(self.h / 2))):
            for x in range(self.x - (int(self.w / 2)), self.x + (int(self.w / 2))):
                pixel_list.append([x, y])

        return pixel_list


class PolygonRegionOfInterest(RegionOfInterest):

    def __init__(self, *nodes):
        self.nodes = nodes

    def polygon_ray_casting(self, bounding_points, bounding_box_positions):

        ## from https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
        ## user  Noresourses

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
                if (((verty[i] > testy) != (verty[j] > testy)) and
                        (testx < (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i])):
                    c += 1
            # If odd, that means that we are inside the polygon
            if c % 2 == 1:
                points_inside.append(pos)

        return points_inside

    def return_list_pixel(self):

        min_x, max_x, min_y, max_y = self.nodes[0][0], self.nodes[0][0], self.nodes[0][1], self.nodes[0][1]

        for i in self.nodes:
            if i[0] < min_x:
                min_x = i[0]
            if i[0] > max_x:
                max_x = i[0]
            if i[1] < min_y:
                min_y = i[1]
            if i[1] > max_y:
                max_y = i[1]
        list_coord = []
        for y in range(min_y, max_y):
            for x in range(min_x, max_y):
                list_coord.append((x, y))

        pixel_list = self.polygon_ray_casting(self.nodes, list_coord)

        return pixel_list
