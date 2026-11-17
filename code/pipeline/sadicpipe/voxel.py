"""Voxelisation, morphological closing, cavity filling, EDT and depth indices.

Discrete conventions used for all the experiments of the paper:
- discretisation: a voxel belongs to an atom if its centre lies within r + 0.5*resolution
  of the atom centre;
- closing: ball structuring element of radius ceil((probe + 0.5*res)/res) voxels;
- cavity fill: background connected components (6-connectivity) with 1-voxel padding;
- reference radius: max EDT (sampling=res) over atom-centre voxels;
- depth indices: reference radius + 0.5*res, translated template sphere (TranSph)
  or per-atom reconstructed sphere (ReconSph).
"""

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, label


def compute_extremes(atoms, radii, resolution):
    """Axis-aligned bounding box of the union of spheres, extended to a whole number of voxels."""
    extremes = np.empty((3, 2), dtype=np.float64)
    extremes[:, 0] = (atoms - radii[:, None]).min(axis=0)
    extremes[:, 1] = (atoms + radii[:, None]).max(axis=0)
    span = extremes[:, 1] - extremes[:, 0]
    extremes[:, 1] += resolution * (1 - np.modf(span / resolution)[0])
    return extremes


def cartesian_to_grid(coords, extremes, resolution):
    return np.floor((coords - extremes[:, 0]) / resolution).astype(np.int32)


def grid_to_cartesian(indices, extremes, resolution):
    return indices * resolution + extremes[:, 0] + resolution / 2


def discretize(atoms, radii, extremes, resolution, half_voxel_offset=True):
    """Binary voxel grid of the union of atom spheres."""
    eff_radii = radii + (0.5 * resolution if half_voxel_offset else 0.0)
    dims = np.ceil((extremes[:, 1] - extremes[:, 0]) / resolution).astype(np.int32)
    solid = np.zeros(dims, dtype=bool)

    for center, radius in zip(atoms, eff_radii):
        lo = np.maximum(np.floor((center - radius - extremes[:, 0]) / resolution).astype(np.int32), 0)
        hi = np.minimum(np.ceil((center + radius - extremes[:, 0]) / resolution).astype(np.int32), dims)
        if np.any(hi <= lo):
            continue
        xs = np.arange(lo[0], hi[0])
        ys = np.arange(lo[1], hi[1])
        zs = np.arange(lo[2], hi[2])
        px = xs * resolution + extremes[0, 0] + resolution / 2
        py = ys * resolution + extremes[1, 0] + resolution / 2
        pz = zs * resolution + extremes[2, 0] + resolution / 2
        d2 = (
            ((px - center[0]) ** 2)[:, None, None]
            + ((py - center[1]) ** 2)[None, :, None]
            + ((pz - center[2]) ** 2)[None, None, :]
        )
        region = solid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        np.logical_or(region, d2 <= radius * radius, out=region)

    return solid


def ball_structuring_element(radius_voxels):
    r = int(radius_voxels)
    ax = np.arange(-r, r + 1)
    d2 = ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2
    return d2 <= r * r


def morphological_closing(solid, resolution, probe_radius, half_voxel_offset=True):
    """Binary closing with a discrete ball structuring element (MorphPC)."""
    eff_probe = probe_radius + (0.5 * resolution if half_voxel_offset else 0.0)
    int_radius = int(np.ceil(eff_probe / resolution))
    se = ball_structuring_element(int_radius)
    pad = 2 * int_radius
    expanded = np.pad(solid, pad, mode="constant", constant_values=False)
    closed = binary_closing(expanded, structure=se)
    return closed[pad:-pad, pad:-pad, pad:-pad]


_SIX_CONN = np.array(
    [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
     [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=np.int32)


def fill_cavities(solid):
    """Fill background components not connected to the outside (6-connectivity).

    Returns (filled_solid, n_background_components, n_filled_voxels).
    """
    padded = np.pad(solid, 1, mode="constant", constant_values=False)
    components, n_components = label(~padded, structure=_SIX_CONN)
    outside_label = components[0, 0, 0]
    filled = components != outside_label
    filled = filled[1:-1, 1:-1, 1:-1]
    n_filled = int(filled.sum()) - int(solid.sum())
    return filled, int(n_components), n_filled


def reference_radius(solid, atoms, extremes, resolution):
    """Largest EDT value over atom-centre voxels.

    Returns (radius in Angstrom, index of the atom where it is attained).
    """
    centers = cartesian_to_grid(atoms, extremes, resolution)
    if (centers < 0).any() or (centers >= np.array(solid.shape)).any():
        raise ValueError("atom centre outside the voxel grid")
    edt = distance_transform_edt(solid, sampling=resolution)
    values = edt[centers[:, 0], centers[:, 1], centers[:, 2]]
    idx = int(np.argmax(values))
    return float(values[idx]), idx


def _template_sphere(reference_radius_eff, extremes, resolution):
    """Voxelised template sphere used by the translated-sphere method."""
    grid_radius = int((2 * reference_radius_eff / resolution) // 2)
    diameter = grid_radius * 2 + 1
    origin = np.array([diameter // 2] * 3, dtype=np.int32)
    cartesian_origin = grid_to_cartesian(origin, extremes, resolution)
    ax = np.arange(diameter)
    px = grid_to_cartesian(np.stack([ax, np.zeros_like(ax), np.zeros_like(ax)], axis=1),
                           extremes, resolution)[:, 0]
    py = grid_to_cartesian(np.stack([np.zeros_like(ax), ax, np.zeros_like(ax)], axis=1),
                           extremes, resolution)[:, 1]
    pz = grid_to_cartesian(np.stack([np.zeros_like(ax), np.zeros_like(ax), ax], axis=1),
                           extremes, resolution)[:, 2]
    d2 = (
        ((px - cartesian_origin[0]) ** 2)[:, None, None]
        + ((py - cartesian_origin[1]) ** 2)[None, :, None]
        + ((pz - cartesian_origin[2]) ** 2)[None, None, :]
    )
    sphere = d2 <= reference_radius_eff ** 2
    return sphere, grid_radius


def depth_indices_translated(solid, atoms, ref_radius, extremes, resolution,
                             half_voxel_offset=True):
    """TranSph: one template sphere, translated on each atom centre."""
    ref_eff = ref_radius + (0.5 * resolution if half_voxel_offset else 0.0)
    sphere, grid_radius = _template_sphere(ref_eff, extremes, resolution)
    sphere_volume = int(sphere.sum())
    dims = np.array(solid.shape, dtype=np.int32)
    diameter = grid_radius * 2 + 1

    centers = cartesian_to_grid(atoms, extremes, resolution)
    depth = np.empty(len(atoms), dtype=np.float32)
    for i, c in enumerate(centers):
        s_lo = np.maximum(c - grid_radius, 0)
        s_hi = np.minimum(c + grid_radius + 1, dims)
        t_lo = s_lo - (c - grid_radius)
        t_hi = diameter - ((c + grid_radius + 1) - s_hi)
        inter = np.logical_and(
            solid[s_lo[0]:s_hi[0], s_lo[1]:s_hi[1], s_lo[2]:s_hi[2]],
            sphere[t_lo[0]:t_hi[0], t_lo[1]:t_hi[1], t_lo[2]:t_hi[2]],
        )
        depth[i] = 2.0 * (1.0 - np.count_nonzero(inter) / sphere_volume)
    return depth


def depth_indices_reconstructed(solid, atoms, ref_radius, extremes, resolution,
                                half_voxel_offset=True):
    """ReconSph: the reference sphere is voxelised again around each atom centre."""
    ref_eff = ref_radius + (0.5 * resolution if half_voxel_offset else 0.0)
    sq_ref = ref_eff ** 2
    dims = np.array(solid.shape, dtype=np.int32)
    depth = np.empty(len(atoms), dtype=np.float32)

    for i, center in enumerate(atoms):
        lo = np.floor((center - ref_eff - extremes[:, 0]) / resolution).astype(np.int32)
        hi = np.ceil((center + ref_eff - extremes[:, 0]) / resolution).astype(np.int32)
        xs = np.arange(lo[0], hi[0])
        ys = np.arange(lo[1], hi[1])
        zs = np.arange(lo[2], hi[2])
        px = xs * resolution + extremes[0, 0] + resolution / 2
        py = ys * resolution + extremes[1, 0] + resolution / 2
        pz = zs * resolution + extremes[2, 0] + resolution / 2
        d2 = (
            ((px - center[0]) ** 2)[:, None, None]
            + ((py - center[1]) ** 2)[None, :, None]
            + ((pz - center[2]) ** 2)[None, None, :]
        )
        sphere_box = d2 <= sq_ref
        sphere_volume = int(sphere_box.sum())

        s_lo = np.maximum(lo, 0)
        s_hi = np.minimum(hi, dims)
        b_lo = s_lo - lo
        b_hi = np.array(sphere_box.shape, dtype=np.int32) - (hi - s_hi)
        inter = np.logical_and(
            solid[s_lo[0]:s_hi[0], s_lo[1]:s_hi[1], s_lo[2]:s_hi[2]],
            sphere_box[b_lo[0]:b_hi[0], b_lo[1]:b_hi[1], b_lo[2]:b_hi[2]],
        )
        depth[i] = 2.0 * (1.0 - np.count_nonzero(inter) / sphere_volume)
    return depth
