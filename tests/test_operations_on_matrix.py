import pytest

from project.matrix_operations import (
    matrix_addition,
    matrix_multiplication,
    transpose_matrix,
)


def test_matrix_addition():
    m1 = [[1, 2], [3, 4]]
    m2 = [[5, 6], [7, 8]]
    expected_result = [[6, 8], [10, 12]]
    assert matrix_addition(m1, m2) == expected_result

    # Edge case: adding zero matrix
    m1 = [[1, 2], [3, 4]]
    m2 = [[0, 0], [0, 0]]
    expected_result = [[1, 2], [3, 4]]
    assert matrix_addition(m1, m2) == expected_result

    # Edge case: matrices with negative numbers
    m1 = [[-1, -2], [-3, -4]]
    m2 = [[5, 6], [7, 8]]
    expected_result = [[4, 4], [4, 4]]
    assert matrix_addition(m1, m2) == expected_result

    # Matrix addition with floating-point numbers
    m1 = [[1.5, 2.5], [3.5, 4.5]]
    m2 = [[0.5, 1.5], [2.5, 3.5]]
    expected_result = [[2.0, 4.0], [6.0, 8.0]]
    assert matrix_addition(m1, m2) == expected_result

    # Edge case: matrices with a mix of integers and floats
    m1 = [[1, 2.5], [3, 4.5]]
    m2 = [[5.5, 6], [7.5, 8]]
    expected_result = [[6.5, 8.5], [10.5, 12.5]]
    assert matrix_addition(m1, m2) == expected_result


def test_matrix_multiplication():
    m1 = [[1, 2], [3, 4]]
    m2 = [[5, 6], [7, 8]]
    expected_result = [[19, 22], [43, 50]]
    assert matrix_multiplication(m1, m2) == expected_result

    # Non-square matrix multiplication test (for example 2x3 * 3x2)
    m1 = [[1, 2, 3], [4, 5, 6]]
    m2 = [[7, 8], [9, 10], [11, 12]]
    expected_result = [[58, 64], [139, 154]]
    assert matrix_multiplication(m1, m2) == expected_result

    # Edge case: multiplying by identity matrix
    m1 = [[1, 2], [3, 4]]
    identity_matrix = [[1, 0], [0, 1]]
    expected_result = [[1, 2], [3, 4]]
    assert matrix_multiplication(m1, identity_matrix) == expected_result

    # Edge case: multiplying by a zero matrix
    zero_matrix = [[0, 0], [0, 0]]
    expected_result = [[0, 0], [0, 0]]
    assert matrix_multiplication(m1, zero_matrix) == expected_result

    # Matrix multiplication with floating-point numbers
    m1 = [[1.5, 2.5], [3.5, 4.5]]
    m2 = [[5.5, 6.5], [7.5, 8.5]]
    expected_result = [[27, 31], [53, 61]]
    assert matrix_multiplication(m1, m2) == expected_result

    # Matrix multiplication with floating-point numbers #2
    m1 = [[1.5, 2], [3.5, 4]]
    m2 = [[5, 6], [7, 8.7]]
    expected_result = [[21.5, 26.4], [45.5, 55.8]]
    assert matrix_multiplication(m1, m2) == expected_result


def test_transpose_matrix():
    m = [[1, 2], [3, 4]]
    expected_result = [[1, 3], [2, 4]]
    assert transpose_matrix(m) == expected_result

    # Transpose of a non-square matrix (2x3)
    m = [[1, 2, 3], [4, 5, 6]]
    expected_result = [[1, 4], [2, 5], [3, 6]]
    assert transpose_matrix(m) == expected_result

    # Edge case: transpose of a 1xN matrix (row vector)
    m = [[1, 2, 3]]
    expected_result = [[1], [2], [3]]
    assert transpose_matrix(m) == expected_result

    # Edge case: transpose of an Nx1 matrix (column vector)
    m = [[1], [2], [3]]
    expected_result = [[1, 2, 3]]
    assert transpose_matrix(m) == expected_result

    # Transpose of a matrix with floating-point numbers
    m = [[1.1, 2.2], [3.3, 4.4]]
    expected_result = [[1.1, 3.3], [2.2, 4.4]]
    assert transpose_matrix(m) == expected_result

    # Transpose of a non-square matrix with floats
    m = [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]
    expected_result = [[1.1, 4.4], [2.2, 5.5], [3.3, 6.6]]
    assert transpose_matrix(m) == expected_result
