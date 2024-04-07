"""
Modules:

1. sys: Provides access to objects used or maintained by the interpreter
and functions that interact with the interpreter.

2. re: Provides support for regular expressions, powerful tools for matching patterns in text.

3. numpy (imported as np): Fundamental package for scientific computing with Python,
offering numerical arrays and mathematical functions.

4. numpy.typing (imported as npt): Typing module for NumPy, providing type hints for
static type checkers.

5. matplotlib.pyplot (imported as plt): Provides a MATLAB-like plotting framework for
creating static, animated, and interactive visualizations.
"""
import sys
import re
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def generate_centroids(min_x: int, max_x: int, min_y: int, max_y: int,
                       n_clusters: int) -> npt.NDArray[np.float32]:
    """
    Generate centroids for K-means clustering.

    :param min_x: Minimum x-coordinate.
    :param max_x: Maximum x-coordinate.
    :param min_y: Minimum y-coordinate.
    :param max_y: Maximum y-coordinate.
    :param n_clusters: Number of clusters.

    :return: Array of centroids.
    :rtype: np.ndarray[np.float32]
    """
    x_centroids = np.random.uniform(min_x, max_x, n_clusters)
    y_centroids = np.random.uniform(min_y, max_y, n_clusters)
    centroids = np.array([[x_centroids[i], y_centroids[i]] for i in range(n_clusters)])
    return centroids


def generate_colours(quantity: int) -> npt.NDArray[np.float64]:
    """
    Generate colors for visualization.

    :param quantity: Number of colors to generate.
    :type quantity: int
    :return: Array of generated colors.
    :rtype: np.ndarray[np.float64]
    """

    colours = np.random.rand(quantity, 3)
    return colours


def compute_distances(centroids: npt.NDArray[np.float32],
                      points: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    """
    Compute distances between centroids and points.

    :param centroids: Array of centroid coordinates.
    :type centroids: np.ndarray[np.float32]
    :param points: Array of point coordinates.
    :type points: np.ndarray[np.int64]

    :return: Array of distances.
    :rtype: np.ndarray[np.float32]
    """

    distances = []
    for point in points:
        distance = np.sqrt((point[0] - centroids[:, 0]) ** 2 + (point[1] - centroids[:, 1]) ** 2)
        distances.append(distance)
    return np.array(distances)


def make_scatter_plot(points: npt.NDArray[np.int64], centroids: npt.NDArray[np.float32],
                      colours: npt.NDArray[np.float64],
                      points_colour: npt.NDArray[np.float64]) -> None:
    """
    Create a scatter plot of points and centroids.

    :param points: Array of point coordinates.
    :type points: np.ndarray[np.int64]
    :param centroids: Array of centroid coordinates.
    :type centroids: np.ndarray[np.float32]
    :param colours: Array of colors for centroids.
    :type colours: np.ndarray[np.float64]
    :param points_colour: Array of colors for points.
    :type points_colour: np.ndarray[np.float64]

    :return: None
    """

    plt.scatter(points[:, 0], points[:, 1], c=points_colour, alpha=0.4)
    plt.scatter(centroids[:, 0], centroids[:, 1], c=colours)
    plt.show()


def make_assignments(distances: npt.NDArray[np.float32]) -> npt.NDArray[np.int16]:
    """
    Assign points to their nearest centroids.

    :param distances: Array of distances between points and centroids.
    :type distances: np.ndarray[np.float32]

    :return: Array of assignments.
    :rtype: np.ndarray[np.int16]
    """

    assignments: npt.NDArray[np.int16] = np.argmin(distances, axis=1)
    return assignments.reshape(-1, 1)


def get_points_from_file(input_file_path: str) -> npt.NDArray[np.int64]:
    """
    Read points from a file and return as an array.

    :param input_file_path: Path to the input file.
    :type input_file_path: str

    :return: Array of points.
    :rtype: np.ndarray[np.int64]
    """

    with open(input_file_path, 'r', encoding='utf-8') as f:
        points = []
        points_from_file = f.readlines()
        for point in points_from_file:
            point_ = re.split(r'\s+', point.strip())
            points.append(point_)
        return np.array(points, dtype=np.int64)


def assign_colours_to_points(assignments: npt.NDArray[np.int16],
                             colours: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Assign colors to points based on assignments.

    :param assignments: Array of assignments.
    :type assignments: np.ndarray[np.int16]
    :param colours: Array of colors.
    :type colours: np.ndarray[np.float64]

    :return: Array of colors assigned to points.
    :rtype: np.ndarray[np.float64]
    """

    points_colour: npt.NDArray[np.float64] = colours[assignments]
    return points_colour.reshape(-1, 3)


def recalculate_centroids(points: npt.NDArray[np.int64], assignments: npt.NDArray[np.int16],
                          centroids: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Recalculate centroids based on current assignments.

    :param points: Array of point coordinates.
    :type points: np.ndarray[np.int64]
    :param assignments: Array of assignments.
    :type assignments: np.ndarray[np.int16]
    :param centroids: Array of centroid coordinates.
    :type centroids: np.ndarray[np.float32]

    :return: Array of recalculated centroids.
    :rtype: np.ndarray[np.float32]
    """

    new_centroids = []
    for i, centroid in enumerate(centroids):
        assignment = (assignments == i).flatten()
        cluster = points[assignment]
        if len(cluster) == 0:
            new_centroids.append(centroid)
        else:
            new_centroids.append(np.mean(cluster, axis=0))
    return np.array(new_centroids)


def save_results(output_file_path: str, n_clusters: str,
                 centroids: npt.NDArray[np.float32]) -> None:
    """
    Save clustering results to a file.

    :param output_file_path: Path to the output file.
    :type output_file_path: str
    :param n_clusters: Number of clusters.
    :type n_clusters: str
    :param centroids: Array of centroid coordinates.
    :type centroids: np.ndarray[np.float32]

    :return: None
    """

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(n_clusters + '\n')
        for i, centroid in enumerate(centroids):
            file.write(str(i) + ' ' + str(centroid) + '\n')


def kmeans() -> None:
    """
    Perform K-means clustering algorithm.

    Reads input file, performs clustering, visualizes results, and saves them.

    :return: None
    """
    input_file_path, output_file_path, n_clusters = sys.argv[1:]
    points = get_points_from_file(input_file_path)
    centroids = generate_centroids(points[:, 0].min(), points[:, 0].max(),
                                   points[:, 1].min(), points[:, 1].max(),
                                   int(n_clusters))
    colours = generate_colours(int(n_clusters))
    for _ in range(10):
        distances = compute_distances(centroids, points)
        assignments = make_assignments(distances)
        points_colour = assign_colours_to_points(assignments, colours)
        make_scatter_plot(points, centroids, colours, points_colour)
        centroids = recalculate_centroids(points, assignments, centroids)

    save_results(output_file_path, n_clusters, centroids)


if __name__ == '__main__':
    kmeans()
