import unittest

import numpy as np

from tests.drawing import ShapeImage

from .test_feature_detection import _BaseShapeTest


class TestGeneratedMoleculeCornerAndEdgeDetection(_BaseShapeTest):
    def test_cyclohexane(self):
        """Tests that 6 corners and 6 edges are detected for cyclohexane."""
        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[400, 400]],
                [[487, 350]],
                [[574, 400]],
                [[574, 500]],
                [[487, 550]],
                [[400, 500]]
            ]),
            drawer=lambda image: image.add_regular_hexagon(
                100, start_coord=(400, 400)
            )
        )

    def test_methylcyclohexane(self):
        """
        Tests that 7 corners and 7 edges are detected for methylcyclohexane.
        """
        def draw(image: ShapeImage):
            image.add_regular_hexagon(
                100, start_coord=(400, 400)
            )
            image.add_line((487, 350), (487, 250))

        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[400, 400]],
                [[487, 350]],
                [[574, 400]],
                [[574, 500]],
                [[487, 550]],
                [[400, 500]],
                # Methyl group
                [[487, 250]]
            ]),
            drawer=draw,
            expected_edges=np.array([
                [[400, 400, 487, 350]],
                [[487, 350, 574, 400]],
                [[574, 400, 574, 500]],
                [[574, 500, 487, 550]],
                [[487, 550, 400, 500]],
                [[400, 500, 400, 400]],
                # To methyl group
                [[487, 350, 487, 250]]
            ])
        )

    def test_1_2_dimethylcyclohexane(self):
        """
        Tests that 8 corners and 8 edges are detected for
        1,2-dimethylcyclohexane.

        """
        def draw(image: ShapeImage):
            image.add_regular_hexagon(
                100, start_coord=(400, 400)
            )
            image.add_line((487, 350), (487, 250))
            image.add_line((574, 400), (661, 350))

        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[400, 400]],
                [[487, 350]],
                [[574, 400]],
                [[574, 500]],
                [[487, 550]],
                [[400, 500]],
                # Methyl groups
                [[487, 250]],
                [[661, 350]]
            ]),
            drawer=draw,
            expected_edges=np.array([
                [[400, 400, 487, 350]],
                [[487, 350, 574, 400]],
                [[574, 400, 574, 500]],
                [[574, 500, 487, 550]],
                [[487, 550, 400, 500]],
                [[400, 500, 400, 400]],
                # To methyl groups
                [[487, 350, 487, 250]],
                [[574, 400, 661, 350]]
            ])
        )

    def test_hexamethylcyclohexane(self):
        """
        Tests that 12 corners and 12 edges are detected for
        1,2,3,4,5,6-hexamethylcyclohexane.

        """
        def draw(image: ShapeImage):
            image.add_regular_hexagon(
                100, start_coord=(400, 400)
            )
            image.add_line((487, 350), (487, 250))
            image.add_line((574, 400), (661, 350))
            image.add_line((574, 500), (661, 550))
            image.add_line((487, 550), (487, 650))
            image.add_line((400, 500), (313, 550))
            image.add_line((400, 400), (313, 350))

        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[400, 400]],
                [[487, 350]],
                [[574, 400]],
                [[574, 500]],
                [[487, 550]],
                [[400, 500]],
                # Methyl groups
                [[487, 250]],
                [[661, 350]],
                [[661, 550]],
                [[487, 650]],
                [[313, 550]],
                [[313, 350]]
            ]),
            drawer=draw,
            expected_edges=np.array([
                [[400, 400, 487, 350]],
                [[487, 350, 574, 400]],
                [[574, 400, 574, 500]],
                [[574, 500, 487, 550]],
                [[487, 550, 400, 500]],
                [[400, 500, 400, 400]],
                # To methyl groups
                [[487, 350, 487, 250]],
                [[574, 400, 661, 350]],
                [[574, 500, 661, 550]],
                [[487, 550, 487, 650]],
                [[400, 500, 313, 550]],
                [[400, 400, 313, 350]]
            ])
        )

    def test_ethene(self):
        """
        Tests that 2 vertices and 2 lines are detected for ethene.

        """
        def draw(image: ShapeImage):
            image.add_line((400, 400), (500, 400))
            image.add_line((400, 410), (500, 410))

        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[400, 400]],
                [[500, 400]]
            ]),
            drawer=draw,
            expected_edges=np.array([
                [[400, 400, 500, 400]],
                [[400, 410, 500, 410]]
            ])
        )

    def test_ethyne(self):
        """
        Tests that 2 vertices and 3 lines are detected for ethyne.

        """
        def draw(image: ShapeImage):
            image.add_line((400, 400), (500, 400))
            image.add_line((400, 410), (500, 410))
            image.add_line((400, 420), (500, 420))

        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[400, 400]],
                [[500, 400]]
            ]),
            drawer=draw,
            expected_edges=np.array([
                [[400, 400, 500, 400]],
                [[400, 410, 500, 410]],
                [[400, 420, 500, 420]]
            ])
        )

    def test_propene(self):
        """
        Tests that 3 vertices and 3 lines are detected for propene.

        """
        def draw(image: ShapeImage):
            image.add_line((400, 400), (500, 400))
            image.add_line((400, 410), (500, 410))
            image.add_line((500, 400), (587, 350))

        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[400, 400]],
                [[500, 400]],
                [[587, 350]]
            ]),
            drawer=draw,
            expected_edges=np.array([
                [[400, 400, 500, 400]],
                [[400, 410, 500, 410]],
                [[500, 400, 587, 350]]
            ])
        )

    def test_isobutene(self):
        """
        Tests that 4 vertices and 4 lines are detected for isobutene.

        """
        def draw(image: ShapeImage):
            image.add_line((400, 400), (500, 400))
            image.add_line((400, 410), (500, 410))
            image.add_line((500, 400), (587, 350))
            image.add_line((500, 400), (587, 450))

        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[400, 400]],
                [[500, 400]],
                [[587, 350]],
                [[587, 450]]
            ]),
            drawer=draw,
            expected_edges=np.array([
                [[400, 400, 500, 400]],
                [[400, 410, 500, 410]],
                [[500, 400, 587, 350]],
                [[500, 400, 587, 450]]
            ])
        )

    def test_benzene(self):
        """
        Tests that 6 vertices and 9 edges are detected for benzene.

        """
        def draw(image: ShapeImage):
            image.add_regular_hexagon(100, start_coord=(400, 400))
            image.add_line((415, 405), (487, 364))
            image.add_line((415, 495), (487, 536))
            image.add_line((559, 409), (559, 491))

        self._test_shape(
            image_size=(1000, 1000),
            expected_corners=np.array([
                [[400, 400]],
                [[487, 350]],
                [[574, 400]],
                [[574, 500]],
                [[487, 550]],
                [[400, 500]]
            ]),
            drawer=draw,
            expected_edges=np.array([
                [[400, 400, 487, 350]],
                [[487, 350, 574, 400]],
                [[574, 400, 574, 500]],
                [[574, 500, 487, 550]],
                [[487, 550, 400, 500]],
                [[400, 500, 400, 400]],
                # "Double" bonds
                [[415, 405, 487, 364]],
                [[415, 495, 487, 536]],
                [[559, 409, 559, 491]]
            ]),
            vertex_atol=15
        )


if __name__ == '__main__':
    unittest.main()
