from typing import List, Tuple
import shapely.geometry as sg
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from channest.polygonize import Cube
from channest.skeletonize import SkeletonizedLayer
from channest.widths import Widths, LayerWidths


def prepare_distance_cube(cube: Cube, zz: Cube) -> np.ndarray:
    assert cube.shape == zz.shape
    nz, nx, ny = cube.shape
    # Calculate 'steps', which indicate when a channel is entered (+1) or exited (-1) down each trace
    steps = np.diff(cube, axis=0, prepend=0)

    # Determine the index for when channels are exited and entered ('starts' and 'stops')
    indexes = np.ones((1, nx, ny)) * np.arange(nz).reshape((-1, 1, 1))
    starts = np.where(steps > 0, indexes, 0)
    stops = np.where(steps < 0, indexes, nz)

    # For each cell in the cube, find the closest cell, ahead of the current one, that marks the start of a channel. The
    # result is stored in i0. Similar for i1, except that it marks the end of a channel.
    #
    # This means that if cell (i, j, k) is within a channel, the value of i0[k, i, j] is the cell index that marks the
    # start of that channel. Similarly, the value of i1[k, i, j] is the cell index that marks the end of the same
    # channel.
    #
    # If cell (i, j, k) is NOT within a channel, i0[k, i, j] is the cell index of the start of the channel ahead of cell
    # (i, j, k), and i1[k, i, j] is the cell index of the end of the channel after (i, j, k). Note, in particular, that
    # these are two different channels. However, these values are not intended to be used in what follows.
    i0 = np.maximum.accumulate(starts, axis=0).astype(np.int)
    i1 = np.minimum.accumulate(stops[::-1, :, :], axis=0)[::-1, :, :].astype(np.int)

    # Calculate the thicknesses of the channel at each point where the channel is defined. If the channel is not defined
    # (cube == 0), leave the thickness at 0.0.
    dz = np.zeros_like(zz, dtype=np.float64)
    for i in range(nz):
        cix = cube[i, :, :].astype(np.bool)
        if not cix.any():
            continue
        zz_bot = zz[i1[i, cix] - 1, cix]
        zz_top = zz[i0[i, cix], cix]
        dz[i, cix] = zz_bot - zz_top
    return dz


class LayerHeights:
    def __init__(self, widths: LayerWidths, distance_map: np.ndarray):
        self._heights = [Heights(w, distance_map) for w in widths.widths]

    @property
    def heights(self) -> List['Heights']:
        return self._heights

    def flat_values(self) -> np.ndarray:
        try:
            return np.hstack([h.height_values for h in self._heights])
        except ValueError:
            return np.empty((0,), dtype=np.float)

    def flat_values_kde(self, h_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import scipy.stats as ss
        n_points = 200  # Hard-coded for simplicity
        x = np.linspace(0, h_max, n_points)
        v = self.flat_values()
        if v.size > 0:
            # Filter low values. Values that are practically 0 should not contribute to the density estimate, as these
            # are unintended artifacts of the height calculation algorithm
            v = v[v > 1e-12]

        if np.unique(v).size <= 1:
            # Require at least 2 unique values to create a KDE
            y = np.full_like(x, fill_value=np.nan)
        else:
            # Calculate KDE, but make sure to disable warnings from underflow as these are not relevant in this context.
            # Warnings are likely coming from a long interval (high h_max) compared to the distribution of the values
            old_warnings = np.seterr(under='ignore')
            y = ss.gaussian_kde(v)(x)
            np.seterr(**old_warnings)
        return x, y, v

    def flat_values_kde_modes(self, h_max: float) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.flat_values_kde(h_max)[:2]
        if np.isnan(y).any():
            return np.empty((0,)), np.empty((0,))
        gt_after = y[2:] < y[1:-1]
        gt_before = y[:-2] < y[1:-1]
        is_mode = np.zeros_like(y, dtype=np.bool)
        is_mode[1:-1] = gt_after & gt_before
        return x[is_mode], y[is_mode]

    def flat_values_max_mode(self, h_max: float) -> np.float:
        thickness, density = self.flat_values_kde_modes(h_max)
        if thickness.size == 0:
            return np.nan
        else:
            return thickness[np.argmax(density)]

    def mean_heights(self) -> np.ndarray:
        return np.array([
            np.mean(h.height_values)
            for h in self._heights
            if h.height_values.size > 0
        ])

    @staticmethod
    def calculate_max_height(layers: List['LayerHeights']):
        return np.max([np.max(h.flat_values())
                       for h in layers
                       if h.flat_values().size > 0])


class Heights:
    def __init__(self, w: Widths, distance_map: np.ndarray) -> None:
        self._distance_map = distance_map

        max_heights, max_height_locations = Heights._calculate_max_heights(w, distance_map)
        # Flatten values for simplified data extraction
        self._height_values = np.array([h for mh in max_heights for h in mh])

        # Construct height maps for QC
        self._height_map = np.zeros_like(distance_map, dtype=np.float)
        mh_locs = np.array([h for mhl in max_height_locations for h in mhl])
        if mh_locs.size > 0:
            # Map to indices
            mh_locs_i = np.round(mh_locs).astype(int)
            # Restrict to values within the area (potential side effect of np.round)
            nx, ny = self._distance_map.shape
            include = (mh_locs_i >= 0).all(axis=1)
            include &= mh_locs_i[:, 0] < nx
            include &= mh_locs_i[:, 1] < ny
            self._height_map[tuple(mh_locs_i[include].T)] = self._height_values[include]

    @property
    def vertical_distance_map(self) -> np.ndarray:
        return self._distance_map

    @property
    def height_map(self) -> np.ndarray:
        return self._height_map

    @property
    def height_values(self) -> np.ndarray:
        return self._height_values

    @staticmethod
    def _calculate_max_heights(w: Widths, distance_map: np.ndarray) -> Tuple[List[List[np.float]],
                                                                             List[List[np.ndarray]]]:
        # Find all sample points from segments
        interp_points: List[List[List[sg.Point]]] = []
        for channel in w.segments:
            interp_points.append([])
            for segment in channel:
                n = max(2, int(segment.length * 4))  # Corresponds to finest resolution 0.25
                ps = np.linspace(0, segment.length, n)
                interp_points[-1].append([segment.interpolate(_p) for _p in ps])

        # Define interpolator and interpolate heights
        interpolator = RegularGridInterpolator((np.arange(distance_map.shape[0]), np.arange(distance_map.shape[1])),
                                               distance_map,
                                               method='nearest')
        max_heights: List[List[np.float]] = []
        max_height_locations: List[List[np.ndarray]] = []
        for channel in interp_points:
            max_heights.append([])
            max_height_locations.append([])
            for segment in channel:
                segment_ps = np.array([np.array(s) for s in segment])
                values = interpolator(segment_ps)
                a_max = np.argmax(values)
                max_height_locations[-1].append(segment_ps[a_max])
                max_heights[-1].append(values[a_max])

        return max_heights, max_height_locations
