"""Plot translucent Gaussian ellipsoid projections and export column counts without Mitsuba.

Usage: python examples/visualize_result.py results/checkpoints/my_result
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import EllipseCollection, LineCollection
import numpy as np
from plyfile import PlyData
from tqdm import tqdm


def find_result_ply(result_path, level=0):
    # accept a training directory, asset directory, data directory, or explicit ply file.
    path = Path(result_path)
    filename = f'root.primitives_pyr{level}.ply'
    candidates = [path] if path.is_file() else [path / f'optimized_asset_pyr{level}' / 'data' / filename,
                                               path / 'data' / filename, path / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Cannot find {filename} under {path}')


def gaussian_covariances(scales, quaternions):
    # normalize xyzw rotations and build world-space covariance matrices.
    quaternions = np.asarray(quaternions, dtype=np.float64)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    x, y, z, w = quaternions.T
    rotation = np.empty((len(scales), 3, 3), dtype=np.float64)
    rotation[:, 0, :] = np.column_stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)))
    rotation[:, 1, :] = np.column_stack((2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)))
    rotation[:, 2, :] = np.column_stack((2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)))
    return np.einsum('nik,nk,njk->nij', rotation, np.square(scales), rotation)


def load_result_geometry(filename):
    # read compact columns without starting a rendering backend.
    ply = PlyData.read(str(filename), mmap=False)
    elements = {element.name for element in ply.elements}
    if {'metadata', 'column', 'primitive'} <= elements:
        columns, primitives = ply['column'].data, ply['primitive'].data
        metadata = ply['metadata'].data[0]
        version = int(metadata['format_version'])
        if version not in (1, 2):
            raise ValueError(f'Unsupported column PLY version: {version}')
        counts = columns['primitive_count'].astype(np.int64) if version == 2 else np.full(len(columns), int(metadata['primitives_per_column']), dtype=np.int64)
        if np.any(counts <= 0) or counts.sum() != len(primitives):
            raise ValueError('Column counts must be positive and sum to the number of primitives')
        parents = np.repeat(np.arange(len(columns)), counts)
        centers = np.column_stack((columns['center_x'][parents], primitives['center_y'], columns['center_z'][parents]))
        scales = np.column_stack((columns['scale_x'][parents], primitives['scale_y'], columns['scale_z'][parents])).astype(np.float64)

        # reconstruct each column's rotation about y before forming projected ellipses.
        angles = columns['rotation_angle'][parents].astype(np.float64)
        quaternions = np.column_stack((np.zeros(len(parents)), np.sin(angles / 2), np.zeros(len(parents)), np.cos(angles / 2)))
        return {'centers': centers, 'weights': primitives['opacity'].astype(np.float64), 'counts': counts,
                'covariances': gaussian_covariances(scales, quaternions), 'extent': float(metadata['extent'])}

    # decode physical scales and xyzw rotations from independent gaussian records.
    vertices = ply['vertex'].data
    centers = np.column_stack((vertices['x'], vertices['y'], vertices['z']))
    scales = np.exp(np.column_stack([vertices[f'scale_{axis}'] for axis in range(3)]).astype(np.float64))
    quaternions = np.column_stack([vertices[f'rot_{axis}'] for axis in (1, 2, 3, 0)])
    weight_name = next((name for name in ('opacities_0', 'sigma_t_0', 'sigmat_0') if name in vertices.dtype.names), None)
    weights = vertices[weight_name].astype(np.float64) if weight_name else None
    extent = vertices['extent_0'].astype(np.float64) if 'extent_0' in vertices.dtype.names else 3.0
    return {'centers': centers, 'weights': weights, 'counts': None,
            'covariances': gaussian_covariances(scales, quaternions), 'extent': extent}


def projected_ellipses(covariances, axes, extent):
    # project the full ellipsoid rather than drawing a central plane cross-section.
    covariance = covariances[:, axes, :][:, :, axes]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    radii = np.sqrt(np.maximum(eigenvalues, 0)) * np.asarray(extent).reshape(-1, 1)
    angles = np.degrees(np.arctan2(eigenvectors[:, 1, 1], eigenvectors[:, 0, 1]))
    return 2 * radii[:, 1], 2 * radii[:, 0], angles


def axis_limits(values):
    # leave visible margins even when all centers share a single coordinate.
    lower, upper = float(values.min()), float(values.max())
    margin = max((upper - lower) * 0.005, 1e-4)
    return lower - margin, upper + margin


def projection_limits(centers, covariances, axes, extent):
    # include the complete projected ellipsoid supports in shared world-space bounds.
    radius = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2)) * np.asarray(extent).reshape(-1, 1)
    return np.array([axis_limits(np.concatenate((centers[:, axis] - radius[:, axis], centers[:, axis] + radius[:, axis]))) for axis in axes])


def oblique_camera(elevation=25.0, azimuth=-60.0):
    # use an orthonormal camera frame with y as world up.
    elevation, azimuth = np.radians([elevation, azimuth])
    se, ce, sa, ca = np.sin(elevation), np.cos(elevation), np.sin(azimuth), np.cos(azimuth)
    return np.array([[-sa, 0.0, ca], [-se * ca, ce, -se * sa], [ce * ca, se, ce * sa]])


def oblique_geometry(centers, covariances, elevation=25.0, azimuth=-60.0):
    # rotate ellipsoids and centers into the same camera frame.
    camera = oblique_camera(elevation, azimuth)
    projected_centers = centers @ camera.T
    projected_covariances = np.einsum('ai,nij,bj->nab', camera, covariances, camera)
    return projected_centers, projected_covariances


def column_geometry_segments(centers, covariances, extent, counts, selected):
    """Draw each visible column's own axial centers and elliptical bounding outline."""
    # sort selected centers within their original columns without adding cross-column edges.
    empty = np.empty((0, 2, 3), dtype=np.float64)
    if not len(selected):
        return empty, empty
    parents = np.repeat(np.arange(len(counts)), counts)
    ordered = selected[np.lexsort((centers[selected, 1], parents[selected]))]
    same_column = parents[ordered[:-1]] == parents[ordered[1:]]
    pairs = np.column_stack((ordered[:-1][same_column], ordered[1:][same_column]))
    axial = centers[pairs]

    # bound the selected primitives along y while retaining their shared xz footprint.
    _, starts = np.unique(parents[ordered], return_index=True)
    representatives = ordered[starts]
    extents = np.broadcast_to(extent, (len(centers),))
    y_radius = np.sqrt(covariances[ordered, 1, 1]) * extents[ordered]
    lower = np.minimum.reduceat(centers[ordered, 1] - y_radius, starts)
    upper = np.maximum.reduceat(centers[ordered, 1] + y_radius, starts)
    footprint_extent = np.maximum.reduceat(extents[ordered], starts)

    # generate rotated xz ellipse rings for each column independently.
    covariance_xz = covariances[representatives][:, (0, 2), :][:, :, (0, 2)]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_xz)
    transform = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[:, None, :] * footprint_extent[:, None, None]
    angle = np.linspace(0, 2 * np.pi, 33)
    circle = np.stack((np.cos(angle), np.sin(angle)))
    ring_xz = np.einsum('nij,jp->npi', transform, circle) + centers[representatives][:, None, (0, 2)]

    # join top and bottom rings to show the column envelope rather than a neighbor graph.
    bottom = np.empty((len(starts), len(angle), 3), dtype=np.float64)
    bottom[:, :, 0], bottom[:, :, 2] = ring_xz[:, :, 0], ring_xz[:, :, 1]
    bottom[:, :, 1] = lower[:, None]
    top = bottom.copy()
    top[:, :, 1] = upper[:, None]
    rings = np.concatenate((bottom, top))
    edges = np.stack((rings[:, :-1], rings[:, 1:]), axis=2).reshape(-1, 2, 3)
    sides = np.stack((bottom[:, (0, 8, 16, 24)], top[:, (0, 8, 16, 24)]), axis=2).reshape(-1, 2, 3)
    return axial, np.concatenate((edges, sides))


def primitive_slices(centers, axis, width):
    """Partition centers into fixed-width slabs; the last interval includes its upper boundary."""
    # assign every center to exactly one slab while retaining empty intervals.
    coordinates = centers[:, axis].astype(np.float64)
    lower = np.floor(coordinates.min() / width) * width
    count = max(1, int(np.ceil((coordinates.max() - lower) / width)))
    edges = lower + np.arange(count + 1) * width
    groups = np.clip(np.searchsorted(edges, coordinates, side='right') - 1, 0, count - 1)
    for index in range(count):
        yield edges[index], edges[index + 1], np.flatnonzero(groups == index)


def save_projection(centers, covariances, extent, colors, norm, axes, filename, dpi, alpha, center_size=2.0, limits=None, segments=None):
    # fill the entire image with equal-scale projected content and no plot decorations.
    first, second = axes
    if limits is None:
        limits = projection_limits(centers, covariances, axes, extent)
    spans = limits[:, 1] - limits[:, 0]
    side_view = 1 in axes
    width = 12 if side_view else 8
    height = width * spans[1] / spans[0]
    if height > 12:
        width, height = width * 12 / height, 12
    fig = plt.figure(figsize=(max(width, 2 / dpi), max(height, 2 / dpi)))
    ax = fig.add_axes([0, 0, 1, 1])
    if len(centers):
        # batch translucent ellipses in data units to preserve their physical sizes.
        widths, heights, angles = projected_ellipses(covariances, axes, extent)
        rgba = plt.get_cmap('magma')(norm(colors)) if colors is not None else np.tile([0.22, 0.49, 0.67, 1.0], (len(centers), 1))
        rgba[:, 3] = alpha
        outlines = rgba.copy()
        outlines[:, 3] = min(2 * alpha, 1.0)
        ellipses = EllipseCollection(widths, heights, angles, units='xy', offsets=centers[:, axes], transOffset=ax.transData,
                                    facecolors=rgba, edgecolors=outlines, linewidths=0.12, rasterized=True)
        ax.add_collection(ellipses)

        # show each column envelope and its own axial centers below center markers.
        if segments is not None:
            for lines, color, opacity, linewidth in zip(segments, ('#245e72', '#407b92'), (0.7, 0.18), (0.45, 0.22)):
                ax.add_collection(LineCollection(lines[:, :, axes], colors=color, alpha=opacity, linewidths=linewidth, zorder=2))

        # emphasize physical primitive centers above the translucent projected supports.
        if center_size > 0:
            ax.scatter(centers[:, first], centers[:, second], s=center_size, c='#17324d', edgecolors='white',
                       linewidths=0.15, alpha=0.9, zorder=3, rasterized=True)
    ax.set(xlim=limits[0], ylim=limits[1], aspect='equal')
    ax.set_axis_off()
    fig.savefig(filename, dpi=dpi, pad_inches=0)
    plt.close(fig)


def visualize_result(result_path, output_path=None, level=0, dpi=300, alpha=0.01, extent=None,
                     slice_axis='x', slice_width=None, center_size=2.0, slice_3d=False,
                     view_elevation=25.0, view_azimuth=-60.0, column_lines=True):
    # resolve the saved model and prepare a standalone output directory.
    ply_path = find_result_ply(result_path, level)
    base = Path(result_path) if Path(result_path).is_dir() else ply_path.parent
    output = Path(output_path) if output_path else base / 'visualizations' / f'pyr{level}'
    output.mkdir(parents=True, exist_ok=True)

    # remove only obsolete k-group images produced by earlier script versions.
    for filename in output.glob('columns_k*.png'):
        if re.fullmatch(r'columns_k\d+(?:_front_xy|_side_zy)?', filename.stem):
            filename.unlink()

    geometry = load_result_geometry(ply_path)
    centers, weights, counts = geometry['centers'], geometry['weights'], geometry['counts']
    covariances = geometry['covariances']
    extent = geometry['extent'] if extent is None else extent
    if len(centers) == 0 or not np.isfinite(centers).all() or not np.isfinite(covariances).all():
        raise ValueError('Visualization requires nonempty finite primitive geometry')
    if not 0 < alpha <= 1 or not np.all(np.isfinite(extent) & (np.asarray(extent) > 0)):
        raise ValueError('Use alpha in (0, 1] and positive finite extent')
    colors = np.log10(np.maximum(np.abs(weights), 1e-12)) if weights is not None else None
    norm = Normalize(float(colors.min()), max(float(colors.max()), float(colors.min()) + 1e-6)) if colors is not None else None

    files = []
    if slice_width is None:
        # write three full projections with visible centers when slicing is disabled.
        views = [('front_xy.png', (0, 1)), ('side_zy.png', (2, 1)), ('top_xz.png', (0, 2))]
        segments = column_geometry_segments(centers, covariances, extent, counts, np.arange(len(centers))) if counts is not None and column_lines else None
        for name, axes in tqdm(views, desc='Export projections', unit='view'):
            filename = output / name
            save_projection(centers, covariances, extent, colors, norm, axes, filename, dpi, alpha, center_size, segments=segments)
            files.append(filename)
    else:
        # project each center-selected slab along its normal using consistent world bounds.
        if not np.isfinite(slice_width) or slice_width <= 0:
            raise ValueError('slice_width must be positive and finite in world-space units')
        axis = 'xyz'.index(slice_axis)
        axes = {0: (2, 1), 1: (0, 2), 2: (0, 1)}[axis]
        folder = output / f'slices_{slice_axis}_width_{slice_width:g}'
        folder.mkdir(parents=True, exist_ok=True)
        for filename in folder.glob('slice_*.png'):
            if re.fullmatch(r'slice_\d+(?:_3d)?', filename.stem):
                filename.unlink()
        limits = projection_limits(centers, covariances, axes, extent)
        extents = np.broadcast_to(extent, (len(centers),))
        rows = ['index\taxis\tstart\tend\tprimitive_count\tfilename' + ('\tfilename_3d' if slice_3d else '')]
        if slice_3d:
            view_centers, view_covariances = oblique_geometry(centers, covariances, view_elevation, view_azimuth)
            view_limits = projection_limits(view_centers, view_covariances, (0, 1), extent)
            camera = oblique_camera(view_elevation, view_azimuth)

        # export empty slices as blank images so no interval silently disappears.
        slices = list(primitive_slices(centers, axis, slice_width))
        for index, (lower, upper, selected) in enumerate(tqdm(slices, desc=f'Export {slice_axis.upper()} slices', unit='slice')):
            segments = column_geometry_segments(centers, covariances, extent, counts, selected) if counts is not None and column_lines else None
            filename = folder / f'slice_{index:04d}.png'
            save_projection(centers[selected], covariances[selected], extents[selected],
                            colors[selected] if colors is not None else None, norm, axes, filename, dpi, alpha, center_size, limits, segments)
            row = f'{index}\t{slice_axis}\t{lower:.12g}\t{upper:.12g}\t{len(selected)}\t{filename.name}'
            files.append(filename)

            # reuse projected ellipsoids for an exact orthographic oblique view without surface meshes.
            if slice_3d:
                filename_3d = folder / f'slice_{index:04d}_3d.png'
                ordered = selected[np.argsort(view_centers[selected, 2], kind='stable')]
                view_segments = tuple(lines @ camera.T for lines in segments) if segments is not None else None
                save_projection(view_centers[ordered], view_covariances[ordered], extents[ordered],
                                colors[ordered] if colors is not None else None, norm, (0, 1), filename_3d,
                                dpi, alpha, center_size, view_limits, view_segments)
                row += f'\t{filename_3d.name}'
                files.append(filename_3d)
            rows.append(row)
        summary = folder / 'slice_distribution.txt'
        summary.write_bytes(('\r\n'.join(rows) + '\r\n').encode('utf-8'))
        files.append(summary)

    # export a simple tab-separated distribution for later mermaid conversion.
    distribution = output / 'column_k_distribution.txt'
    if counts is not None:
        unique_k, frequencies = np.unique(counts, return_counts=True)
        rows = ['k\tcolumn_count\tprimitive_count']
        rows.extend(f'{int(k)}\t{int(frequency)}\t{int(k * frequency)}' for k, frequency in zip(unique_k, frequencies))
        distribution.write_bytes(('\r\n'.join(rows) + '\r\n').encode('utf-8'))
    else:
        distribution.write_bytes(b'# Independent Gaussian PLY: no column grouping is stored.\r\n')
    return files + [distribution]


def main():
    # expose a minimal command line for saved training results.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path, help='Training result directory, optimized asset directory, or PLY file')
    parser.add_argument('--output', type=Path, default=None, help='Output directory; defaults to RESULT/visualizations/pyrLEVEL')
    parser.add_argument('--level', type=int, default=0, help='Pyramid level to visualize')
    parser.add_argument('--dpi', type=int, default=300, help='PNG resolution in dots per inch')
    parser.add_argument('--alpha', type=float, default=0.01, help='Ellipse fill opacity; default 0.01')
    parser.add_argument('--extent', type=float, default=None, help='Ellipsoid radius in sigma units; defaults to the saved extent or 3')
    parser.add_argument('--slice_axis', choices=('x', 'y', 'z'), default='x', help='Slab normal axis; default x produces YZ projections')
    parser.add_argument('--slice_width', type=float, default=None, help='Slab width in world-space units; select primitives by center')
    parser.add_argument('--center_size', type=float, default=2.0, help='Area of emphasized center markers in square points; 0 hides them')
    parser.add_argument('--slice_3d', action='store_true', help='Also save an oblique orthographic ellipsoid view for every slice')
    parser.add_argument('--view_elevation', type=float, default=25.0, help='Elevation of oblique views in degrees')
    parser.add_argument('--view_azimuth', type=float, default=-60.0, help='Azimuth of oblique views in degrees')
    parser.add_argument('--no_column_lines', action='store_true', help='Hide each column bounding outline and its internal y center connections')
    args = parser.parse_args()
    for filename in visualize_result(args.result, args.output, args.level, args.dpi, args.alpha, args.extent,
                                     args.slice_axis, args.slice_width, args.center_size, args.slice_3d,
                                     args.view_elevation, args.view_azimuth, not args.no_column_lines):
        print(filename)


if __name__ == '__main__':
    main()
