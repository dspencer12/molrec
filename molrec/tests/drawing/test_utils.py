import unittest

from . import utils


class TestBlankImageCreation(unittest.TestCase):
    def test_image(self):
        image = utils.create_blank_image(10, 10)
        self.assertEqual((10, 10, 3), image.shape)
        # TODO: test colour


if __name__ == '__main__':
    unittest.main()
