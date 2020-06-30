from typing import List, Optional

import shapely.geometry as sg
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from channest.skeletonize import SkeletonizedPolygon, SkeletonizedLayer


class LayerWidths:
    def __init__(self, sl: SkeletonizedLayer, map2d: np.ndarray, mean_map_threshold: float, turn_off_filters: bool):
        self._widths = [Widths(p, map2d, mean_map_threshold, turn_off_filters) for p in sl.skeletonized_polygons]

        # If any of the arrays are empty, set to to a length-1 zero array. This is to ensure all statistics have a
        # numeric value for each layer
        if len(self._widths) == 0:
            self._stacked_fw = np.zeros(1)
        else:
            self._stacked_fw = np.hstack([_w.full_widths for _w in self._widths])
            if self._stacked_fw.size == 0:
                self._stacked_fw = np.zeros(1)

    @property
    def widths(self):
        return self._widths

    def stat(self, func):
        return func(self._stacked_fw)


class Widths:
    def __init__(self, sp: SkeletonizedPolygon, map2d: np.ndarray, mean_map_threshold: float, turn_off_filters: bool):
        self._perp_segments = []
        for p in sp.pieces:
            w = _find_piece_widths(sp.polygon, p, dens=1)  # TODO: dens != 1?
            self._perp_segments.append(w)

        # If filters are turned off, nothing more to do
        if turn_off_filters is True:
            return

        # Filter width segments that crosses uncertain parts of the map
        if mean_map_threshold > 0.0:
            self._perp_segments = _filter_uncertain_segments(self._perp_segments, map2d, mean_map_threshold)

        # Filter width segments that cross the skeleton
        self._perp_segments = _filter_skeleton_crossing_segments(self._perp_segments, sp)

        # Filter segments touching the bounding box
        self._perp_segments = _filter_boundary_segments(self._perp_segments, map2d)

    @property
    def segments(self) -> List[List[sg.LineString]]:
        return self._perp_segments

    @property
    def full_widths(self) -> np.ndarray:
        return np.array([p.length for per_piece in self.segments for p in per_piece])


def _find_piece_widths(poly, piece, dens) -> List[sg.LineString]:
    widths = []
    coords = piece.coords
    for i in range(len(coords)-1):
        w = _find_line_widths(poly, coords[i], coords[i+1], dens)
        widths = widths + w
    return widths


def _find_line_widths(poly, c1, c2, dens) -> List[sg.LineString]:
    dx = c2[0]-c1[0]
    dy = c2[1]-c1[1]
    length = np.sqrt(dx*dx+dy*dy)
    widths = []
    if length > 2*dens:
        dx = dens*dx/length
        dy = dens*dy/length
        ndx = -1000*dy
        ndy = 1000*dx
        x = c1[0]+dx
        y = c1[1]+dy
        n_step = int(np.floor(length/dens))
        for i in range(n_step):
            p0 = sg.Point((x,y))
            p1 = (x+ndx,y+ndy)
            p2 = (x-ndx,y-ndy)
            line = sg.LineString([p1,p2])
            cuts = line.intersection(poly)
            if type(cuts) is sg.multilinestring.MultiLineString:
                d = np.array([l.hausdorff_distance(p0) for l in cuts])
                e_ind = np.argmin(d)
                edges = cuts[e_ind]
            else:
                # TODO: insert elif statement with proper type
                edges = cuts
            if (type(edges) is sg.linestring.LineString) and (len(list(edges.coords)) == 2):
                # TODO: does multi line string actually contribute at all?
                widths = widths + [edges]
            x = x+dx
            y = y+dy
    return widths


def _filter_uncertain_segments(segments: List[List[sg.LineString]],
                               map2d: np.ndarray,
                               mean_map_threshold: float) -> List[List[sg.LineString]]:
    interpolator = RegularGridInterpolator((np.arange(map2d.shape[0]), np.arange(map2d.shape[1])),
                                           map2d,
                                           method='nearest')
    filtered_segments = []
    for piece in segments:
        mn = []
        for segment in piece:
            # Interpolation points. Minimum 2 points, finest resolution 1.0.
            ip = np.linspace(0, segment.length, max(2, int(segment.length)))
            pts = np.array([np.array(segment.interpolate(i)) for i in ip])
            interp = interpolator(pts)
            mn.append(np.mean(interp))
        filtered_segments.append([_s for _s, _m in zip(piece, mn) if _m > mean_map_threshold])
    return filtered_segments


def _filter_skeleton_crossing_segments(segments: List[List[sg.LineString]],
                                       sp: SkeletonizedPolygon) -> List[List[sg.LineString]]:
    filtered_segments = []
    for perps in segments:
        filtered_segments.append([])
        for perp in perps:
            intersections = [_sp.intersects(perp) for _sp in sp.pieces]
            if sum(intersections) > 1:
                # Ignore this perp
                continue
            # Simplification. Should check if the single intersection is not with the piece segment this perp
            # originated from.
            filtered_segments[-1].append(perp)
    return filtered_segments


def _filter_boundary_segments(segments: List[List[sg.LineString]],
                              map2d: np.ndarray):
    filtered_segments = []
    for perps in segments:
        filtered_segments.append([perp
                                  for perp in perps
                                  if 0 < perp.bounds[0] < map2d.shape[0] - 1
                                  and 0 < perp.bounds[1] < map2d.shape[1] - 1
                                  and 0 < perp.bounds[2] < map2d.shape[0] - 1
                                  and 0 < perp.bounds[3] < map2d.shape[1] - 1])
    return filtered_segments
