"""Oriented-bounding-box alignment (method 'approximate' of the paper).

Hybrid genetic algorithm + Nelder-Mead over rotations of the convex hull, with an
explicit random seed for reproducibility.
"""

import numpy as np
import scipy.optimize
import scipy.spatial
from scipy.spatial.transform import Rotation as R


def _obb_volume(points, rotation_matrix):
    rotated = points @ rotation_matrix.T
    return float(np.prod(rotated.max(axis=0) - rotated.min(axis=0)))


def _objective(rot_vector, points):
    return _obb_volume(points, R.from_rotvec(rot_vector).as_matrix())


def align_obb(points, population_size=30, generations=200, nelder_mead_iters=10, seed=0):
    """Rotate points into (approximately) their minimum-volume OBB frame.

    Returns (aligned_points, obb_volume).
    """
    rng = np.random.default_rng(seed)
    hull = scipy.spatial.ConvexHull(points)
    hull_points = points[hull.vertices]

    population = rng.normal(size=(population_size, 3))
    population[0] = 0.0  # identity rotation as a seed candidate
    best_solution = None
    best_volume = np.inf

    for _ in range(generations):
        volumes = np.array([_objective(ind, hull_points) for ind in population])
        order = np.argsort(volumes)
        population = population[order[: population_size // 2]]
        if volumes[order[0]] < best_volume:
            best_volume = volumes[order[0]]
            best_solution = population[0].copy()

        children = []
        while len(children) < population_size:
            i, j = rng.choice(len(population), 2, replace=False)
            child = (population[i] + population[j]) / 2 + rng.normal(size=3) * 0.1
            children.append(child)
        population = np.array(children)

    result = scipy.optimize.minimize(
        _objective, best_solution, args=(hull_points,),
        method="Nelder-Mead", options={"maxiter": nelder_mead_iters},
    )
    rotation = R.from_rotvec(result.x).as_matrix()
    volume = _obb_volume(hull_points, rotation)
    if volume > best_volume:  # keep the GA solution if NM made things worse
        rotation = R.from_rotvec(best_solution).as_matrix()
        volume = best_volume

    return points @ rotation.T, volume


def align(points, method="approximate", seed=0):
    """Entry point: method 'approximate' (GA+NM) or 'none'."""
    if method == "none":
        return points, None
    if method == "approximate":
        return align_obb(points, seed=seed)
    raise ValueError(f"unknown alignment method: {method}")
