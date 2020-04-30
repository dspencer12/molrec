import math
import unittest

import numpy as np

from . import utils


class TestBlankImageCreation(unittest.TestCase):
    def test_image(self):
        image = utils.create_blank_image(10, 10)
        self.assertEqual((10, 10, 3), image.shape)
        np.testing.assert_array_equal(image[0, 0], np.array([255, 255, 255]))

    def test_image_coloured(self):
        image = utils.create_blank_image(100, 50, rgb_colour=(127, 199, 14))
        self.assertEqual((100, 50, 3), image.shape)
        np.testing.assert_array_equal(image[0, 0], np.array([14, 199, 127]))


class TestCoordinateAverage(unittest.TestCase):
    def test_line(self):
        """The average of straight line end points should be the midpoint."""
        average = utils.get_average_point(np.array([
            [10, 10],
            [20, 20]
        ]))
        np.testing.assert_array_equal(np.array([15, 15]), average)

    def test_square(self):
        """The average of square vertices should be the centre."""
        average = utils.get_average_point(np.array([
            [10, 10],
            [20, 10],
            [20, 20],
            [10, 20]
        ]))
        np.testing.assert_array_equal(np.array([15, 15]), average)

    def test_square_rounded(self):
        """The average of square vertices should be the centre, rounded to
        integer."""
        average = utils.get_average_point(np.array([
            [1, 1],
            [2, 1],
            [2, 2],
            [1, 2]
        ]))
        np.testing.assert_array_equal(np.array([2, 2]), average)

    def test_regular_hexagon(self):
        """The average of hexagonal vertices should be the centre."""
        average = utils.get_average_point(np.array([
            [150, 93],
            [50, 93],
            [0, 180],
            [50, 267],
            [150, 267],
            [200, 180]
        ]))
        np.testing.assert_array_equal(np.array([100, 180]), average)


class TestRotateCoordinates(unittest.TestCase):
    def test_line_origin_no_rotation(self):
        """Tests coordinate rotation of the endpoints of a line 0 degrees about
        the origin."""
        coords = np.array([
            [1, 1],
            [2, 2]
        ])
        transformed = utils.rotate_coordinates(coords, 0)
        np.testing.assert_array_equal(coords, transformed)

    def test_line_origin_90(self):
        """Tests coordinate rotation of the endpoints of a line 90 degrees
        clockwise about the origin."""
        coords = utils.rotate_coordinates(
            np.array([
                [1, 1],
                [2, 2]
            ]),
            -math.pi / 2
        )
        np.testing.assert_array_equal(
            np.array([
                [-1, 1],
                [-2, 2]
            ]),
            coords
        )

    def test_line_origin_180(self):
        """Tests coordinate rotation of the endpoints of a line 180 degrees
        clockwise about the origin."""
        coords = utils.rotate_coordinates(
            np.array([
                [1, 1],
                [2, 2]
            ]),
            -math.pi
        )
        np.testing.assert_array_equal(
            np.array([
                [-1, -1],
                [-2, -2]
            ]),
            coords
        )

    def test_line_origin_270(self):
        """Tests coordinate rotation of the endpoints of a line 270 degrees
        clockwise about the origin."""
        coords = utils.rotate_coordinates(
            np.array([
                [1, 1],
                [2, 2]
            ]),
            - (3 * math.pi) / 2
        )
        np.testing.assert_array_equal(
            np.array([
                [1, -1],
                [2, -2]
            ]),
            coords
        )

    def test_line_midpoint_no_rotation(self):
        """Tests the coordinate rotation of the endpoints of a line 0 degrees
        about its midpoint."""
        coords = np.array([
            [1, 1],
            [2, 2]
        ])
        transformed = utils.rotate_coordinates(
            coords,
            0,
            point=utils.get_average_point(coords)
        )
        np.testing.assert_array_equal(coords, transformed)

    def test_line_midpoint_45(self):
        """Tests the coordinate rotation of the endpoints of a line 45 degrees
        about its midpoint."""
        coords = np.array([
            [1, 1],
            [3, 3]
        ])
        transformed = utils.rotate_coordinates(
            coords,
            -math.pi / 4,
            point=utils.get_average_point(coords)
        )
        np.testing.assert_array_equal(
            np.array([
                [2, 1],
                [2, 3]
            ]),
            transformed
        )

    def test_line_midpoint_90(self):
        """Tests the coordinate rotation of the endpoints of a line 90 degrees
        clockwise about its midpoint."""
        coords = np.array([
            [1, 1],
            [3, 3]
        ])
        transformed = utils.rotate_coordinates(
            coords,
            -math.pi / 2,
            point=utils.get_average_point(coords)
        )
        np.testing.assert_array_equal(
            np.array([
                [3, 1],
                [1, 3]
            ]),
            transformed
        )

    def test_line_midpoint_180_rounded(self):
        """Tests the coordinate rotation of the endpoints of a line 180 degrees
        clockwise about its midpoint."""
        coords = np.array([
            [1, 1],
            [2, 2]
        ])
        transformed = utils.rotate_coordinates(
            coords,
            -math.pi,
            point=utils.get_average_point(coords)
        )
        # Note that because of working with integer coordinates, the midpoint
        # of the line is rounded to (2, 2) and rotation is thus with respect
        # to this point.
        np.testing.assert_array_equal(
            np.array([
                [3, 3],
                [2, 2]
            ]),
            transformed
        )

    def test_line_midpoint_180(self):
        """Tests the coordinate rotation of the endpoints of a line 180 degrees
        clockwise about its midpoint."""
        coords = np.array([
            [1, 1],
            [3, 3]
        ])
        transformed = utils.rotate_coordinates(
            coords,
            -math.pi,
            point=utils.get_average_point(coords)
        )
        np.testing.assert_array_equal(
            np.array([
                [3, 3],
                [1, 1]
            ]),
            transformed
        )

    def test_line_midpoint_270(self):
        """Tests the coordinate rotation of the endpoints of a line 270 degrees
        clockwise about its midpoint."""
        coords = np.array([
            [1, 1],
            [3, 3]
        ])
        transformed = utils.rotate_coordinates(
            coords,
            - (3 * math.pi) / 2,
            point=utils.get_average_point(coords)
        )
        np.testing.assert_array_equal(
            np.array([
                [1, 3],
                [3, 1]
            ]),
            transformed
        )


if __name__ == '__main__':
    unittest.main()
