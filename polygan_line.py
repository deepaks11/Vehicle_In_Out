from shapely.geometry import Point, LineString, Polygon


class LineIntersectionTest:
    def __init__(self, xyxy, line_zones):
        self.xyxy = xyxy
        self.line = LineString([line_zones[0], line_zones[1]])

    def point_line_intersection_test(self):
        try:
            count = 0
            any_intersecting_or_touching_or_standing = False  # To track if any object is intersecting, touching, or standing on the line

            p_x1, p_y1, p_x2, p_y2 = self.xyxy.astype(int)
            person_coord = [(p_x1, p_y1), (p_x1, p_y2), (p_x2, p_y1), (p_x2, p_y2)]

            # Create a rectangle polygon for the person's bounding box
            rect_polygon = Polygon(person_coord)

            # Check if the rectangle intersects with the line
            intersects = self.line.crosses(rect_polygon)

            if intersects:
                count += 1
                any_intersecting_or_touching_or_standing = True

            return any_intersecting_or_touching_or_standing

        except Exception as ex:
            print(ex)
            return False  # Return False and 0 if an exception occurs
