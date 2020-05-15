import unittest

import numpy as np

from molrec.molecule_detection.text_recognition import extract_text
from tests.drawing import ShapeImage

from .utils import assert_allclose_unsorted


class TestTextRecognition(unittest.TestCase):
    """
    These tests do not rely on east_detection to detect the bounding boxes.
    End-to-end tests of text detection and recognition are in the class
    TestTextDetectionAndRecognition.

    """
    def test_empty_image(self):
        """
        Tests that no text is detected and no error is raised with no boxes.

        """
        image = ShapeImage.new(1024, 1024)
        texts = extract_text(image, [])
        self.assertEqual([], texts)

    def test_one_word(self):
        """
        Tests that one word can be recognized correctly.

        """
        image = ShapeImage.new(512, 512)
        text = 'Testing'
        image.add_text(text, (300, 300))
        boxes = [(300, 270, 410, 310)]
        texts = extract_text(image, boxes)
        self.assertEqual(1, len(texts))
        np.testing.assert_allclose(
            np.array(boxes), np.array([texts[0][0]]), atol=50
        )
        self.assertEqual(text, texts[0][1])

    def test_two_words(self):
        image = ShapeImage.new(512, 512)
        text1 = 'Testing'
        image.add_text(text1, (300, 300))
        text2 = 'Hydroxybenzene'
        image.add_text(text2, (200, 200))
        boxes = [(300, 270, 410, 310), (200, 170, 400, 210)]
        texts = extract_text(image, boxes)
        self.assertEqual(2, len(texts))
        assert_allclose_unsorted(
            np.array(boxes), np.array([t[0] for t in texts]), atol=100
        )
        text_strings = [t[1] for t in texts]
        self.assertIn(text1, text_strings)
        self.assertIn(text2, text_strings)


if __name__ == '__main__':
    unittest.main()
