import unittest

import numpy as np

from molrec.molecule_detection.text_detection import east_detection
from tests.drawing import ShapeImage

from .utils import assert_allclose_unsorted


class TestEASTTextDetection(unittest.TestCase):
    def test_empty_image(self):
        """
        Tests that no bounding boxes are detected when the image is empty.

        """
        image = ShapeImage.new(512, 512)
        boxes = east_detection(image, apply_suppression=False)
        np.testing.assert_equal(boxes, np.array([]))

    def test_image_shape_no_text(self):
        """
        Tests that no bounding boxes are detected when the image contains lines
        but no text.

        """
        image = ShapeImage.new(512, 512)
        image.add_regular_hexagon(50, start_coord=(300, 300))
        boxes = east_detection(image, apply_suppression=False)
        np.testing.assert_equal(boxes, np.array([]))

    def test_one_word(self):
        """
        Tests that one bounding box is detected when the image contains a
        single word.

        """
        image = ShapeImage.new(512, 512)
        image.add_text('TEST', (300, 300), bottomLeftOrigin=True)
        boxes = east_detection(image)
        np.testing.assert_allclose(
            np.array([[300, 300, 370, 320]]), boxes, atol=40
        )

    def test_two_words_far(self):
        """
        Tests that two bounding boxes is detected when the image contains two
        words far apart.

        """
        image = ShapeImage.new(512, 512)
        image.add_text('TEST', (150, 150), bottomLeftOrigin=True)
        image.add_text('TEST2', (300, 400), bottomLeftOrigin=True)
        boxes = east_detection(image)
        assert_allclose_unsorted(
            np.array([[150, 150, 220, 170], [300, 400, 370, 420]]),
            boxes,
            atol=40
        )

    def test_shape_one_word_separate(self):
        """
        Tests that one bounding box is detected when the image contains a word
        and a shape.

        """
        image = ShapeImage.new(512, 512)
        image.add_regular_hexagon(50, start_coord=(300, 300))
        image.add_text('TEST', (150, 150))
        boxes = east_detection(image)
        np.testing.assert_allclose(
            np.array([[150, 150, 220, 170]]), boxes, atol=40
        )

    def test_shape_one_word_overlapping(self):
        """
        Tests that one bounding box is detected when the image contains a word
        and a shape overlapping.

        """
        image = ShapeImage.new(512, 512)
        image.add_regular_hexagon(50, start_coord=(300, 300))
        image.add_text('TEST', (300, 300), bottomLeftOrigin=True)
        boxes = east_detection(image)
        np.testing.assert_allclose(
            np.array([[300, 300, 370, 320]]), boxes, atol=40
        )

    def test_many_words(self):
        """
        Tests that bounding boxes are correctly detected when the image contains
        many (separated) words.

        """
        image = ShapeImage.new(512, 512)
        image.add_text('TEST', (150, 150), bottomLeftOrigin=True)
        image.add_text('TEST2', (300, 400), bottomLeftOrigin=True)
        image.add_text('TEST3', (80, 80), bottomLeftOrigin=True)
        image.add_text('TEST4', (200, 200), bottomLeftOrigin=True)
        image.add_text('TEST5', (420, 450), bottomLeftOrigin=True)
        boxes = east_detection(image)
        assert_allclose_unsorted(
            np.array([
                [150, 150, 220, 170],
                [300, 400, 370, 420],
                [80, 80, 150, 100],
                [200, 200, 270, 220],
                [420, 450, 490, 470]
            ]),
            boxes,
            atol=40
        )


if __name__ == '__main__':
    unittest.main()
