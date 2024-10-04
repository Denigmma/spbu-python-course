import pytest
import math

from project.vector_operations import (
    coordinates_vectors,
    vector_length,
    angle_between_vectors,
)


def test_coordinates_vectors():
    assert coordinates_vectors([1, 2, 3], [4, 5, 6]) == 32

    # Test on zero vectors
    assert coordinates_vectors([0, 0, 0], [0, 0, 0]) == 0

    # Orthogonal vector test
    assert coordinates_vectors([1, 0], [0, 1]) == 0

    # Test for the case of negative values
    assert coordinates_vectors([-1, -2, -3], [4, 5, 6]) == -32

    # Test for different vector sizes (error expected)
    with pytest.raises(ValueError):
        coordinates_vectors([1, 2, 3], [4, 5])


def test_vector_length():
    assert vector_length([3, 4]) == 5

    # Test on zero vectors
    assert vector_length([0, 0, 0]) == 0

    # Test for a vector with negative values
    assert vector_length([-3, -4]) == 5

    # One-dimensional vector test
    assert vector_length([7]) == 7

    # Vector length (magnitude) for a vector with floating-point numbers
    v = [1.5, 2.5, 3.5]
    expected_result = math.sqrt(1.5**2 + 2.5**2 + 3.5**2)
    assert vector_length(v) == pytest.approx(expected_result, 0.001)


def test_angle_between_vectors():
    assert (
        pytest.approx(angle_between_vectors([1, 0], [0, 1]), 0.0001) == 1.5708
    )  # π/2 radians

    # Test for the angle between the same vectors (must be 0)
    assert pytest.approx(angle_between_vectors([1, 0], [1, 0]), 0.0001) == 0

    # Angle test between opposite vectors (π radians)
    assert pytest.approx(angle_between_vectors([1, 0], [-1, 0]), 0.0001) == 3.1416

    # Test for vectors with different sizes
    assert pytest.approx(angle_between_vectors([1, 2, 3], [4, 5, 6]), 0.001) == 0.2257

    # Zero vector test (error expected)
    with pytest.raises(ValueError):
        angle_between_vectors([0, 0], [1, 2])
