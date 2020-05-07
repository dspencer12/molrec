import numpy as np


def sort_coords(a: np.ndarray):
    """
    Sorts an array based on the first element of the second axis.

    Sorts lexically based on the first, then second, columns of each coordinate.

    Args:
        a: Array to sort.

    Returns:
        Sorted array.

    """
    return a[np.lexsort((a[:, 0, 0], a[:, 0, 1]))]


def assert_allclose_unsorted(array1: np.ndarray, array2: np.ndarray, **kwargs):
    """
    Asserts that two ndarrays are approximately equal, independent of the order
    on the first axis.

    Args:
        array1: Actual array.
        array2: Expected array.
        kwargs: Other arguments for np.testing.assert_allclose.

    """
    np.testing.assert_allclose(
        sort_coords(array1),
        sort_coords(array2),
        **kwargs
    )


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
