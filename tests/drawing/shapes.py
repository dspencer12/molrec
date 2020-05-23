"""

"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import utils


class ShapeImage(np.ndarray):
    """
    A subclass of np.ndarray to represent an image, including public methods
    to draw lines/shapes on the image.

    This method subclasses np.ndarray in order to behave as an ndarray. The
    subclass method is specific for this class (see
    https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html for
    full justification of the class structure).

    If a new image is to be created, the `new` method should be used to
    simultaneously initialize a new, blank image array.

    """
    def __new__(
            cls,
            input_array: np.ndarray,
            default_colour: Optional[utils.RGBColour] = None
    ) -> ShapeImage:
        obj: ShapeImage = np.asarray(input_array).view(cls)
        obj.default_colour = default_colour or obj.default_colour
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        default_colour = getattr(obj, 'default_colour', None)
        self.default_colour = default_colour or (0, 0, 0)

    @classmethod
    def new(
            cls,
            width: int,
            height: int,
            background_colour: Optional[utils.RGBColour] = None,
            default_colour: Optional[utils.RGBColour] = None
    ) -> ShapeImage:
        """
        Constructs a new `ShapeImage` of the given `width` and `height`.

        Args:
            width: The width of the new image in pixels.
            height: The height of the new image in pixels.
            background_colour: The background colour of the image.
            default_colour: The default colour for new lines drawn on the image.

        Returns:
            A new instance of ShapeImage.

        """
        return ShapeImage(
            utils.create_blank_image(
                width,
                height,
                rgb_colour=background_colour
            ),
            default_colour=default_colour
        )

    def add_shape(
            self,
            coords: List[utils.PixelCoord],
            colour: Optional[utils.RGBColour] = None,
            rotation_angle: float = 0.,
            **kwargs
    ) -> ShapeImage:
        """
        Draws an arbitrary shape on the image by joining the `coords` in the
        given order.

        Args:
            coords: List of (x, y) vertex coordinates.
            colour: Line colour in RGB format.
                    Defaults to ShapeImage.default_colour.
            rotation_angle: Anticlockwise coordinate rotation angle (around
                            shape center) in radians. Defaults to zero.
            kwargs: Additional keyword arguments for cv2.line.

        Returns:
            ShapeImage.

        """
        transformed = np.array(coords, dtype='int64')
        average = utils.get_average_point(transformed)
        transformed = utils.rotate_coordinates(
            transformed,
            rotation_angle,
            point=average
        )

        for x, y in transformed:
            if x < 0 or x > self.shape[0] or y < 0 or y > self.shape[1]:
                raise ValueError(
                    f'Transformed image coordinate {(x, y)} out of bounds for '
                    f'image of size {self.shape}'
                )

        for ii, coord in enumerate(transformed):
            next_index = ii + 1 if ii + 1 < len(transformed) else 0
            cv2.line(
                self,
                tuple(coord),
                tuple(transformed[next_index]),
                tuple(reversed(colour or self.default_colour)),
                **kwargs
            )

        return self

    def add_text(
            self,
            text: str,
            pos: utils.PixelCoord,
            font: int = cv2.FONT_HERSHEY_SIMPLEX,
            font_scale: float = 1.,
            font_colour: utils.RGBColour = (0, 0, 0),
            thickness: int = 1,
            **kwargs
    ) -> Tuple[int, int]:
        """
        Adds `text` to the image.

        Args:
            text: The text string to insert in the image.
            pos: The coordinate of the top left text corner.
            font:
            font_scale:
            font_colour:
            thickness: Text line thickness.
            kwargs: Extra arguments for cv2.putText.

        Returns:
            Tuple of text width and height.

        """
        cv2.putText(
            self, text, pos, font, font_scale, tuple(reversed(font_colour)),
            thickness, **kwargs
        )
        return cv2.getTextSize(text, font, font_scale, thickness)[0]

    def add_line(
            self,
            start: utils.PixelCoord,
            end: utils.PixelCoord,
            colour: Optional[utils.RGBColour] = None,
            rotation_angle: float = 0.,
            **kwargs
    ) -> ShapeImage:
        """
        Adds a line from `start` to `end`.

        Args:
            start: Start point of the line.
            end: Endpoint of the line.
            colour: Line colour in RGB format.
                    Defaults to ShapeImage.default_colour.
            rotation_angle: Anticlockwise coordinate rotation angle (around
                            shape center) in radians. Defaults to zero.
            kwargs: Additional keyword arguments for cv2.line.

        Returns:
            ShapeImage.

        """
        return self.add_shape(
            [start, end],
            colour=colour,
            rotation_angle=rotation_angle,
            **kwargs
        )

    def add_rectangle(
            self,
            width: int,
            height: int,
            start_coord: utils.PixelCoord = (0, 0),
            colour: Optional[utils.RGBColour] = None,
            rotation_angle: float = 0.,
            **kwargs
    ) -> ShapeImage:
        """
        Adds a rectangle with sides of length `width` and `height`.

        Args:
            width: Width of the rectangle in pixels.
            height: Height of the rectangle in pixels.
            start_coord: The coordinate of the top-left vertex.
                         Defaults to (0, 0).
            colour: Line colour in RGB format.
                    Defaults to ShapeImage.default_colour.
            rotation_angle: Anticlockwise coordinate rotation angle (around
                            shape center) in radians. Defaults to zero.

        Returns:
            ShapeImage.

        """
        return self.add_shape(
            [
                start_coord,
                (start_coord[0] + width, start_coord[1]),
                (start_coord[0] + width, start_coord[1] + height),
                (start_coord[0], start_coord[1] + height)
            ],
            colour=colour,
            rotation_angle=rotation_angle,
            **kwargs
        )

    def add_square(
            self,
            length: int,
            start_coord: utils.PixelCoord = (0, 0),
            colour: Optional[utils.RGBColour] = None,
            rotation_angle: float = 0.,
            **kwargs
    ) -> ShapeImage:
        """
        Adds a square with sides of length `length`.

        Args:
            length: Length of each side of the square in pixels.
            start_coord: The coordinate of the top-left vertex.
                         Defaults to (0, 0).
            colour: Line colour in RGB format.
                    Defaults to ShapeImage.default_colour.
            rotation_angle: Anticlockwise coordinate rotation angle (around
                            shape center) in radians. Defaults to zero.

        Returns:
            ShapeImage.

        """
        return self.add_rectangle(
            length,
            length,
            start_coord=start_coord,
            colour=colour,
            rotation_angle=rotation_angle,
            **kwargs
        )

    def add_regular_pentagon(self):
        raise NotImplementedError()

    def add_regular_hexagon(
            self,
            length: int,
            start_coord: Optional[utils.PixelCoord] = None,
            colour: Optional[utils.RGBColour] = None,
            rotation_angle: float = 0.,
            **kwargs
    ) -> ShapeImage:
        """
        Adds a regular hexagon with sides of length `length`.

        Args:
            length: Length of each side of the square in pixels.
            start_coord: The coordinate of the top-left vertex.
                         Defaults to (0, length / 2).
            colour: Line colour in RGB format.
                    Defaults to ShapeImage.default_colour.
            rotation_angle: Anticlockwise coordinate rotation angle (around
                            shape center) in radians. Defaults to zero.

        Returns:
            ShapeImage.

        """
        a = int(length / 2)
        start_coord = start_coord or (0, a)
        a_root_3 = int((3 ** 0.5) * a)
        return self.add_shape(
            [
                start_coord,
                (start_coord[0] + a_root_3, start_coord[1] - length // 2),
                (start_coord[0] + 2 * a_root_3, start_coord[1]),
                (start_coord[0] + 2 * a_root_3, start_coord[1] + length),
                (start_coord[0] + a_root_3,
                 start_coord[1] + int(1.5 * length)),
                (start_coord[0], start_coord[1] + length)
            ],
            colour=colour,
            rotation_angle=rotation_angle,
            **kwargs
        )
