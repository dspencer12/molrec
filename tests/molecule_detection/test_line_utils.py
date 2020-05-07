import math
import unittest

from molrec.molecule_detection.line_utils import (
    calculate_gradient,
    calculate_intercept,
    calculate_segment_length,
    calculate_midpoint,
    calculate_parallel_distance,
    calculate_point_segment_distance,
)


class TestGradient(unittest.TestCase):
    def test_nonzero(self):
        self.assertEqual(1., calculate_gradient(1., 2., 3., 4.))

    def test_horizontal(self):
        self.assertEqual(0., calculate_gradient(3., 1., 10., 1.))

    def test_vertical(self):
        self.assertTrue(math.isnan(calculate_gradient(3., 3., 3., 10.)))


class TestIntercept(unittest.TestCase):
    def test_positive_gradient(self):
        self.assertEqual(-1., calculate_intercept(1., 1., 2.))

    def test_negative_gradient(self):
        self.assertEqual(4., calculate_intercept(2., 2., -1.))

    def test_zero_gradient(self):
        y = 4.
        self.assertEqual(y, calculate_intercept(1., y, 0.))

    def test_nan_gradient(self):
        self.assertTrue(math.isnan(calculate_intercept(1., 2., math.nan)))


class TestSegmentLength(unittest.TestCase):
    segments = [
        ((1., 1., 4., 5.), 5.),
        ((1., 1., 1., 1.), 0.),
    ]

    def test_segments(self):
        for coords, length in self.segments:
            with self.subTest():
                self.assertEqual(length, calculate_segment_length(*coords))


class TestSegmentMidpoint(unittest.TestCase):
    def test_positive_gradient(self):
        self.assertEqual((1.5, 1.5), calculate_midpoint(1, 1, 2, 2))

    def test_negative_gradient(self):
        self.assertEqual((0., 0.), calculate_midpoint(1, 1, -1, -1))

    def test_vertical(self):
        self.assertEqual((1, 2), calculate_midpoint(1, 1, 1, 3))

    def test_horizontal(self):
        self.assertEqual((2, 1), calculate_midpoint(1, 1, 3, 1))


class TestParallelDistance(unittest.TestCase):
    def test_sloped_lines(self):
        self.assertAlmostEqual(
            # 1 / SQRT(2)
            0.7071068,
            calculate_parallel_distance(1., 2., 1.)
        )

    def test_same_line(self):
        self.assertEqual(0., calculate_parallel_distance(1., 1., 5.))

    def test_vertical_line(self):
        self.assertTrue(
            math.isnan(calculate_parallel_distance(1., 2., math.nan))
        )


class TestPointSegmentDistance(unittest.TestCase):
    def test_point_on_segment(self):
        self.assertEqual(
            0.,
            calculate_point_segment_distance((1., 1.), (2., 2.), (1.5, 1.5))
        )

    def test_segment_is_point(self):
        self.assertEqual(
            1.,
            calculate_point_segment_distance((1., 1.), (1., 1.), (2., 1.))
        )

    def test_point_on_line_outside_segment(self):
        self.assertEqual(
            2 ** 0.5,
            calculate_point_segment_distance((1., 1.), (4., 4.), (5., 5.))
        )


if __name__ == '__main__':
    unittest.main()
