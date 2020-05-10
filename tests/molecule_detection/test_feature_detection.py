import math
from typing import Callable, List, Optional, Tuple
import unittest

import cv2
import numpy as np

from molrec.molecule_detection.feature_detection import (
    detect_edges,
    get_vertices_from_edges,
    remove_parallel_edges
)
from tests.drawing import ShapeImage

from .utils import assert_allclose_unsorted, assert_lines_allclose_unsorted


def to_grey(image: ShapeImage) -> ShapeImage:
    """Converts the given `image` to greyscale."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.float32(grey)


class TestParallelLineRemoval(unittest.TestCase):
    def test_adjacent_parallel_lines_identical(self):
        """
        Tests that one line is removed when there are two identical lines.
        """
        lines = np.array([
            # x0, y0, x1, y1
            [[1, 1, 6, 6]],
            [[1, 1, 6, 6]]
        ])
        np.testing.assert_array_equal(
            np.array([
                [[1, 1, 6, 6]]
            ]),
            remove_parallel_edges(lines)
        )

    def test_adjacent_parallel_lines_same_gradient(self):
        """
        Tests that one line is removed when the lines share a gradient with
        a distance offset within the tolerance.
        """
        lines = np.array([
            [[1, 1, 6, 6]],
            [[2, 1, 7, 6]]
        ])
        np.testing.assert_array_equal(
            np.array([
                [[1, 1, 6, 6]]
            ]),
            remove_parallel_edges(lines, max_line_dist=5)
        )

    def test_adjacent_parallel_lines_gradient_within_tolerance(self):
        """
        Tests that one line is removed when the line gradients are within
        the gradient tolerance.
        """
        lines = np.array([
            [[1, 1, 6, 6]],
            [[2, 1, 8, 6]]
        ])
        np.testing.assert_array_equal(
            np.array([
                # This should be kept as it is the longer of the two segments
                [[2, 1, 8, 6]]
            ]),
            remove_parallel_edges(lines, gradient_tolerance=1)
        )

    def test_adjacent_lines_gradient_outside_tolerance(self):
        """
        Tests that both lines are retained when they do not have a common
        gradient within the tolerance level.
        """
        lines = np.array([
            [[1, 1, 6, 6]],
            [[2, 1, 6, 100]]
        ])
        np.testing.assert_array_equal(
            lines,
            remove_parallel_edges(lines, gradient_tolerance=1)
        )

    def test_nonadjacent_parallel_lines(self):
        """
        Tests that both lines are retained when their distance is greater than
        the maximum allowed.
        """
        lines = np.array([
            [[1, 1, 6, 6]],
            [[5, 1, 11, 6]]
        ])
        np.testing.assert_array_equal(
            lines,
            remove_parallel_edges(lines, max_line_dist=1)
        )

    def test_many_lines(self):
        """
        Tests that the correct lines are removed when multiple sets of parallel
        lines are present.
        """
        lines = np.array([
            [[1, 1, 7, 7]],
            [[4, 4, 9, 9]],
            [[3, 3, 11, 11]],
            [[-12, 1, 5, -3]],
            [[-10, 1, 3, -3]]
        ])
        np.testing.assert_array_equal(
            np.array([
                [[3, 3, 11, 11]],
                [[-12, 1, 5, -3]]
            ]),
            remove_parallel_edges(lines)
        )


class _BaseShapeTest(unittest.TestCase):
    bg_colour = (255, 255, 255)
    line_colour = (0, 0, 0)

    def _test_shape(
            self,
            image_size: Tuple[int, int],
            expected_corners: np.ndarray,
            drawer: Callable[[ShapeImage], None],
            expected_edges: Optional[np.ndarray] = None,
            vertex_atol: int = 10
    ):
        """
        Tests whether the expected vertex coordinates and edges are detected for
        the image.

        Note that if `expected_edges` is not provided, this method assumes
        connectivity between the coordinates in `expected_corners`.

        Args:
            image_size: The dimensions of the image as a two-tuple of integers.
            expected_corners: A numpy array of coordinates of the corners to
                              detect.
            drawer: A function to be executed to add shape(s) to the given
                    image. This function should only take the ShapeImage as an
                    argument.

        """
        image = ShapeImage.new(
            *image_size,
            background_colour=self.bg_colour,
            default_colour=self.line_colour
        )
        drawer(image)
        grey = to_grey(image)

        edges = detect_edges(grey)
        if expected_edges is None:
            # Identify expected edges by assuming connectivity between adjacent
            # vertices
            expected_edges = []
            if len(expected_corners) == 2:
                expected_edges.append(
                    ((*expected_corners[0][0], *expected_corners[1][0]),)
                )
            elif len(expected_corners) > 2:
                for ii, coord in enumerate(expected_corners):
                    next_idx = ii + 1 if ii + 1 < len(expected_corners) else 0
                    next_coord = expected_corners[next_idx]
                    expected_edges.append(((*coord[0], *next_coord[0]),))
            expected_edges = np.array(expected_edges)

        self.assertEqual(expected_edges.shape[0], edges.shape[0])
        assert_lines_allclose_unsorted(
            expected_edges,
            edges,
            atol=20
        )

        corners = get_vertices_from_edges(edges, image_size)
        self.assertEqual(expected_corners.shape[0], corners.shape[0])
        assert_allclose_unsorted(
            expected_corners,
            corners,
            atol=vertex_atol
        )

    def _test_line(
            self,
            image_size: Tuple[int, int],
            start: Tuple[int, int],
            end: Tuple[int, int]
    ):
        """
        Tests whether the expected coordinates are detected for a line in an
        image.

        Args:
            image_size: The dimensions of the image as a two-tuple of integers.
            start: The start point of the line.
            end: The end point of the line.

        """
        self._test_shape(
            image_size,
            np.array([[start], [end]]),
            lambda image: image.add_line(start, end)
        )

    def _test_multi_lines(
            self,
            image_size: Tuple[int, int],
            line_coords: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ):
        """
        """

        def draw(image: ShapeImage):
            for s, e in line_coords:
                image.add_line(s, e)

        unique_coords = set()
        for start, end in line_coords:
            unique_coords.add(start)
            unique_coords.add(end)

        self._test_shape(
            image_size=image_size,
            expected_corners=np.array([
                [coord] for coord in unique_coords
            ]),
            drawer=draw,
            expected_edges=np.array([
                [[*start, *end]] for start, end in line_coords
            ])
        )


class TestShapeCornerAndEdgeDetection(_BaseShapeTest):
    def test_line_unit_gradient(self):
        """2 'corners' should be detected for a line with unit gradient."""
        self._test_line((1000, 1000), (100, 100), (900, 900))

    def test_line_vertical(self):
        """2 'corners' should be detected for a vertical line."""
        self._test_line((1000, 1000), (100, 100), (100, 900))

    def test_line_horizontal(self):
        """2 'corners' should be detected for a horizontal line."""
        self._test_line((1000, 1000), (100, 100), (800, 100))

    def test_line_non_unit_gradients(self):
        """2 'corners' should be detected for lines with non-unit gradients."""
        coords = [
            ((100, 100), (200, 300)),
            ((100, 100), (900, 200)),
            ((900, 100), (400, 200))
        ]
        for start, end in coords:
            with self.subTest(start=start, end=end):
                self._test_line((1000, 1000), start, end)

    def test_lines_vshape(self):
        """
        3 corners and two lines should be detected for two lines in a
        V shape.
        """
        self._test_multi_lines(
            (1000, 1000),
            [
                ((400, 400), (350, 313)),
                ((400, 400), (450, 313))
            ]
        )

    def test_lines_propeller(self):
        """
        4 corners and three lines should be detected for three lines in a
        propeller arrangement.
        """
        self._test_multi_lines(
            (1000, 1000),
            [
                ((500, 500), (500, 400)),
                ((500, 500), (450, 587)),
                ((500, 500), (550, 587))
            ]
        )

    def test_square(self):
        """
        4 corners should be detected for a square."""
        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[100, 100]],
                [[200, 100]],
                [[200, 200]],
                [[100, 200]]
            ]),
            drawer=lambda image: image.add_square(
                100, start_coord=(100, 100)
            )
        )

    def test_square_rotated_45(self):
        """4 corners should be detected for a 45 degree rotated square."""
        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[150, 79]],
                [[221, 150]],
                [[150, 221]],
                [[79, 150]]
            ]),
            drawer=lambda image: image.add_square(
                100,
                start_coord=(100, 100),
                rotation_angle=-math.pi / 4
            )
        )

    def test_rectangle(self):
        """
        4 corners should be detected for a rectangle.

        """
        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[100, 100]],
                [[200, 100]],
                [[200, 400]],
                [[100, 400]]
            ]),
            drawer=lambda image: image.add_rectangle(
                100, 300, start_coord=(100, 100)
            )
        )

    def test_rectangle_rotated_45(self):
        """
        4 corners should be detected for a 45 degree rotated rectangle.

        """
        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[221, 109]],
                [[291, 179]],
                [[79, 391]],
                [[9, 321]]
            ]),
            drawer=lambda image: image.add_rectangle(
                100,
                300,
                start_coord=(100, 100),
                rotation_angle=-math.pi / 4
            )
        )

    def test_regular_hexagon(self):
        """
        6 corners should be detected for a hexagon.

        """
        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[100, 100]],
                [[187, 50]],
                [[274, 100]],
                [[274, 200]],
                [[187, 250]],
                [[100, 200]]
            ]),
            drawer=lambda image: image.add_regular_hexagon(
                100, start_coord=(100, 100)
            )
        )

    def test_regular_hexagon_60(self):
        """
        6 corners should be detected for a hexagon rotated 60 degrees clockwise.

        """
        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[87, 150]],
                [[136, 63]],
                [[235, 64]],
                [[285, 150]],
                [[236, 237]],
                [[137, 236]]
            ]),
            drawer=lambda image: image.add_regular_hexagon(
                100,
                start_coord=(100, 100),
                rotation_angle=math.pi / 6
            )
        )


# Colour permutations of the corner and edge detection tests
class TestShapeCornerAndEdgeDetectionRedOnWhite(
    TestShapeCornerAndEdgeDetection
):
    line_colour = (255, 0, 0)


class TestShapeCornerAndEdgeDetectionGreenOnWhite(
    TestShapeCornerAndEdgeDetection
):
    line_colour = (0, 255, 0)


class TestShapeCornerAndEdgeDetectionBlueOnWhite(
    TestShapeCornerAndEdgeDetection
):
    line_colour = (0, 0, 255)


class TestShapeCornerAndEdgeDetectionWhiteOnBlack(
    TestShapeCornerAndEdgeDetection
):
    bg_colour = (0, 0, 0)
    line_colour = (255, 255, 255)


class TestShapeCornerAndEdgeDetectionRedOnBlack(
    TestShapeCornerAndEdgeDetection
):
    bg_colour = (255, 0, 0)
    line_colour = (255, 255, 255)


class TestShapeCornerAndEdgeDetectionGreenOnBlack(
    TestShapeCornerAndEdgeDetection
):
    bg_colour = (0, 255, 0)
    line_colour = (255, 255, 255)


class TestShapeCornerAndEdgeDetectionBlueOnBlack(
    TestShapeCornerAndEdgeDetection
):
    bg_colour = (0, 0, 255)
    line_colour = (255, 255, 255)


if __name__ == '__main__':
    unittest.main()
