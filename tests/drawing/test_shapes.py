import unittest

from .shapes import ShapeImage


class TestShapeImage(unittest.TestCase):
    def setUp(self):
        self.image = ShapeImage.new(1000, 1000)

    def test_line(self):
        image = self.image.add_line((0, 0), (1000, 1000))
        self.assertIsInstance(image, ShapeImage)

    def test_square(self):
        image = self.image.add_square(100)
        self.assertIsInstance(image, ShapeImage)

    def test_rectangle(self):
        image = self.image.add_rectangle(100, 200)
        self.assertIsInstance(image, ShapeImage)

    def test_regular_hexagon(self):
        image = self.image.add_regular_hexagon(100)
        self.assertIsInstance(image, ShapeImage)

    def test_out_of_bounds_left(self):
        with self.assertRaises(ValueError):
            self.image.add_line((-1, 0), (1000, 1000))

    def test_out_of_bounds_right(self):
        with self.assertRaises(ValueError):
            self.image.add_line((10, 0), (1001, 1000))

    def test_out_of_bounds_top(self):
        with self.assertRaises(ValueError):
            self.image.add_line((0, -1), (1000, 1000))

    def test_out_of_bounds_bottom(self):
        with self.assertRaises(ValueError):
            self.image.add_line((0, 0), (1000, 1001))


if __name__ == '__main__':
    unittest.main()
