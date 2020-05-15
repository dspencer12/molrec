from typing import List, Tuple

import numpy as np
import pytesseract as tesseract


def extract_text(
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        padding: float = 0.2
) -> List[Tuple[Tuple[int, int, int, int], str]]:
    """
    """
    texts = []
    for (start_x, start_y, end_x, end_y) in boxes:
        # Compute x and y deltas for padding
        d_x = int((end_x - start_x) * padding)
        d_y = int((end_y - start_y) * padding)

        # Apply padding to each side of the bounding box
        start_x = max(0, start_x - d_x)
        start_y = max(0, start_y - d_y)
        end_x = min(image.shape[0], end_x + (d_x * 2))
        end_y = min(image.shape[1], end_y + (d_y * 2))

        # Extract the actual, padded ROI
        roi = image[start_y:end_y, start_x:end_x]

        # --psm 7 treats the region of interest as a single line of text
        text = tesseract.image_to_string(roi, config='-l eng --oem 1 --psm 7')

        texts.append(((start_x, start_y, end_x, end_y), text))

    return texts
