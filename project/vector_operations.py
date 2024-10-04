import math


def coordinates_vectors(v1: list[float], v2: list[float]) -> float:
    """
    Calculates the dot product of two vectors.

    Args:
        v1 (list[float]): The first vector.
        v2 (list[float]): The second vector.

    Returns:
        float: The dot product of the two vectors.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    result = 0.0
    for x, y in zip(v1, v2):
        result += x * y
    return int(result) if result.is_integer() else result


def vector_length(v: list[float]) -> float:
    """
    Calculates the length (magnitude) of a vector.

    Args:
        v (list[float]): The vector.

    Returns:
        float: The length of the vector.
    """
    result = 0.0
    for x in v:
        result += x**2
    length = math.sqrt(result)
    return int(length) if length.is_integer() else length


def angle_between_vectors(v1: list[float], v2: list[float]) -> float:
    """
    Calculates the angle between two vectors in radians.

    Args:
        v1 (list[float]): The first vector.
        v2 (list[float]]): The second vector.

    Returns:
        float: The angle between the two vectors in radians.
    """
    coordinates = coordinates_vectors(v1, v2)
    len_v1 = vector_length(v1)
    len_v2 = vector_length(v2)
    if len_v1 * len_v2 == 0:
        raise ValueError("One of the vectors has zero length")
    result = math.acos(coordinates / (len_v1 * len_v2))
    return int(result) if result.is_integer() else result
