import numpy as np


def assert_allclose_unsorted(array1: np.ndarray, array2: np.ndarray, **kwargs):
    """
    Asserts that two ndarrays are approximately equal, independent of the order
    on the first axis.

    Args:
        array1: Actual array.
        array2: Expected array.
        kwargs: Other arguments for np.testing.assert_allclose.

    """
    for coord1 in array1:
        for coord2 in array2:
            try:
                np.testing.assert_allclose(coord1, coord2, **kwargs)
            except AssertionError:
                pass
            else:
                break
        else:
            raise AssertionError(f'Vertex {coord1} not found in {array2}')


def assert_lines_allclose_unsorted(
        array1: np.ndarray,
        array2: np.ndarray,
        **kwargs
):
    """

    """
    for line1 in array1:
        for line2 in array2:
            try:
                np.testing.assert_allclose(line1, line2, **kwargs)
            except AssertionError:
                pass
            else:
                break
            try:
                # Test against the reversed coordinates in line2
                np.testing.assert_allclose(
                    line1,
                    np.array(
                        [[line2[0][2], line2[0][3], line2[0][0], line2[0][1]]]
                    ),
                    **kwargs
                )
            except AssertionError:
                pass
            else:
                break
        else:
            raise AssertionError(f'Segment {line1} not found in {array2}')
