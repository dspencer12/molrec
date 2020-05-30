import unittest

import cv2
import numpy as np

from molrec.molecule_detection.masking import apply_mask, create_mask

from tests.drawing import ShapeImage


class TestCreateMask(unittest.TestCase):
    def test_no_masked_rectangles(self):
        """
        Tests that an "empty" mask is created if no masked rectangles are
        specified.

        """
        mask = create_mask((512, 512, 3), np.array([]))
        expected = np.full((512, 512, 3), 255)
        np.testing.assert_equal(mask, expected)

    def test_mask_one_rectangle(self):
        """
        Tests that a rectangular region is marked in the mask.

        """
        mask = create_mask(
            (512, 512, 3),
            np.array([[200, 200, 300, 300]])
        )
        expected = np.full((512, 512, 3), 255)
        expected[200:301, 200:301] = (0, 0, 0)
        np.testing.assert_equal(mask, expected)

    def test_mask_two_rectangles(self):
        """
        Tests that two rectangular regions are marked in the mask.

        """
        mask = create_mask(
            (1024, 1024, 3),
            np.array([[100, 150, 200, 200], [340, 300, 400, 410]])
        )
        expected = np.full((1024, 1024, 3), 255)
        expected[150:201, 100:201] = (0, 0, 0)
        expected[300:411, 340:401] = (0, 0, 0)
        np.testing.assert_equal(mask, expected)

    def test_mask_2d(self):
        """
        Tests that the expected mask is constructed for a 2-dimensional (gray)
        image.

        """
        mask = create_mask(
            (1024, 1024),
            np.array([[100, 150, 200, 200]])
        )
        expected = np.full((1024, 1024), 255)
        expected[150:201, 100:201] = 0
        np.testing.assert_equal(mask, expected)


class TestApplyMask(unittest.TestCase):
    def test_blank_mask(self):
        """
        Tests that the image is unchanged with a blank mask.

        """
        image = ShapeImage.new(512, 512)
        mask = create_mask(image.shape, np.array([]))
        masked = apply_mask(image, mask, (255, 255, 255))
        np.testing.assert_equal(masked, image)

    def test_mask_cover_one_text(self):
        """
        Tests that one box of text is successfully masked.

        """
        image = ShapeImage.new(1024, 1024)
        text_width, text_height = image.add_text('Testing', (250, 250))
        mask = create_mask(
            image.shape,
            # Add padding to height
            np.array([[250, 260, 250 + text_width, 250 - text_height]])
        )
        masked = apply_mask(image, mask, (255, 255, 255))
        np.testing.assert_equal(masked, np.full(image.shape, 255))

    def test_mask_cover_two_texts(self):
        """
        Tests that two boxes of text are successfully masked.

        """
        image = ShapeImage.new(1024, 1024)
        text_width1, text_height1 = image.add_text('Testing1', (250, 250))
        text_width2, text_height2 = image.add_text('more testing', (600, 600))
        mask = create_mask(
            image.shape,
            # Add padding to height
            np.array([
                [250, 260, 250 + text_width1, 250 - text_height1],
                [600, 610, 600 + text_width2, 600 - text_height2]
            ])
        )
        masked = apply_mask(image, mask, (255, 255, 255))
        np.testing.assert_equal(masked, np.full(image.shape, 255))

    def test_mask_cover_one_text_2d(self):
        """
        Tests that one box of text is successfully masked in a 2-dimensional
        (gray) image.

        """
        image = ShapeImage.new(1024, 1024)
        text_width, text_height = image.add_text('Testing', (250, 250))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = create_mask(
            image.shape,
            # Add padding to height
            np.array([[250, 260, 250 + text_width, 250 - text_height]])
        )
        masked = apply_mask(image, mask, 255)
        np.testing.assert_equal(masked, np.full(image.shape, 255))


if __name__ == '__main__':
    unittest.main()
