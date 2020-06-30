from dataclasses import dataclass
from typing import Union, Optional, List, Tuple

import alphashape
import numpy as np
import scipy.signal as ss
import scipy.spatial.qhull as sp
import shapely.geometry as sg

from channest._utils import progress
from nrresqml.resqml import ResQml


Cube = np.ndarray


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class GridParameters:
    box: Box
    dx: float
    dy: float


def smoothen_cube(c: Cube, width: int) -> Cube:
    if width != 1:
        filt = np.ones(width, dtype=np.float).reshape((-1, 1, 1))
        filt /= np.abs(filt).sum()
        final = ss.fftconvolve(c, filt, mode='same')
    else:
        final = c
    return final


def create_cube(r: ResQml, box: Optional[Box]) -> Tuple[Cube, Cube, GridParameters]:
    from nrresqml.derivatives import dataextraction
    archel = dataextraction.extract_property(r, None, 'archel', True)
    channel = (np.array(archel) == 1).astype(np.float)
    ijk, xx, yy, pillars = dataextraction.extract_geometry(r, True, 'kij')
    x0, y0 = xx[0, 0], yy[0, 0]
    dx = xx[1, 0] - xx[0, 0]
    dy = yy[0, 1] - yy[0, 0]
    if box is not None:
        channel = dataextraction.crop_array(channel, x0, y0, dx, dy, box.x0, box.y0, box.x1, box.y1)
        pillars = dataextraction.crop_array(pillars, x0, y0, dx, dy, box.x0, box.y0, box.x1, box.y1)
    else:
        box = Box(np.min(xx), np.min(yy), np.max(xx), np.max(yy))
        pillars = np.array(pillars)
    return channel, pillars, GridParameters(box, dx, dy)


class HulledPolygons:
    def __init__(self, map2d: np.ndarray, threshold: float, alpha: float, minimum_polygon_area: float):
        assert map2d.ndim == 2
        self._map2d = map2d
        self._hulls = _poly_hull(map2d, threshold, alpha)
        self._large_hulls = _remove_small_polygons(self._hulls, minimum_polygon_area)
        # self._large_hulls_with_holes = [_punch_holes(map2d, p, threshold, alpha) for p in self._large_hulls]

    @property
    def polygons(self) -> List[sg.Polygon]:
        return self._large_hulls

    @property
    def map2d(self):
        return self._map2d


def calculate_polygons(c: Cube, z_indexes: Union[np.ndarray, slice, None], threshold: float, alpha: float,
                       minimum_polygon_area: float) -> List[HulledPolygons]:
    sub_c = c[z_indexes, :, :]
    hulled_polys = [
        HulledPolygons(_c, threshold, alpha, minimum_polygon_area) for _c in progress(sub_c, 'Calculating polygons')
    ]
    return hulled_polys


def _fast_alpha_hull(points, alpha) -> List[sg.Polygon]:
    """
    Alternative implementation of the alphashape-method from the alphashape package. Not currently in use, since the
    results may differ slightly from the alphashape-method. However, if speed is an issue, this method should be tested
    and developed further. An initial test indicate that this method is 13x faster than the alpha-shape method.

    The alphashape method is based on a Delaunay triangulation, followed by a filter applied to the triangles. However,
    alphashape (as of May 11, 2020) does not utilize the neighbor information in the Delaunay object when constructing
    the polygon boundary. Instead, it creates the polygon via shapely.ops functions polygonize and cascading_union. This
    neglects useful information that was calculated by Delaunay. _fast_alpha_hull is an attempt to remedy that and
    creates the polygon(s) based on the neighbor information in the Delaunay object.

    The results are almost similar, but not exactly. In particular triangles whose neighbors have all been filtered can
    be different. Illustration:

    o----o----o
    |  / |  / |
    | /  | /  |
    o----o----o
    |  / | x/ |    x - Removed (filtered) triangle
    | /  | /r |    r - Triangle (potentially) handled differently by _fast_alpha_hull and alphashape
    o----o----o

    Signature is the same as alphashape and this is intended to be able to replace alphashape.alphashape without need
    for additional changes.
    """
    if len(points) < 4:
        points = sg.MultiPoint(list(points))
        return [sg.Polygon(points.convex_hull)]
    from scipy.spatial import Delaunay
    tri = Delaunay(points)
    # Filter points (similar approach as alphashape, except that it is done in batch by utilizing numpy instead of a
    # for-loop
    __pts = tri.points[tri.simplices]
    __a = np.linalg.norm(__pts[:, 0, :] - __pts[:, 1, :], axis=1)
    __b = np.linalg.norm(__pts[:, 1, :] - __pts[:, 2, :], axis=1)
    __c = np.linalg.norm(__pts[:, 2, :] - __pts[:, 0, :], axis=1)
    __s = (__a + __b + __c) * 0.5
    __area = __s * (__s - __a) * (__s - __b) * (__s - __c)

    # Define a boolean array with shape (#Number of Delaunay triangles,). If __inc[i] is True, triangle i is kept and
    # will be used to generate the alpha polygon.
    __inc = np.where(__area > 0, (__a * __b * __c) / (4.0 * np.sqrt(__area)) < 1.0 / alpha, False)

    # Copy and adjust the neighbor array from Delaunay so that the filtered triangles are excluded.
    __nb = tri.neighbors.copy()
    __nb[~__inc] = -1  # Set non-included triangles to have no neighbors
    __nb[~np.isin(__nb, np.where(__inc))] = -1  # Deactivate non-included triangles as neighbors

    # Determine the connected components based on the neighbor information
    __nb_edges = np.argwhere(__nb != -1)
    __nb_edges[:, 1] = __nb[__nb_edges[:, 0], __nb_edges[:, 1]]
    import scipy.sparse as ss
    __nb_graph = ss.coo_matrix((
        np.ones(__nb_edges.shape[0], dtype=np.int),
        (__nb_edges[:, 0], __nb_edges[:, 1])),
        shape=(__nb.shape[0], __nb.shape[0]),
    )
    # __cc.size will equal the number of triangles in the Delaunay triangulation. Further, __cc[i] corresponds to the
    #  connected component index triangle i belongs to. Note that if triangle j has been removed (i.e. __inc[j] is
    #  False), __cc[j] will be a unique index. In other words, triangle j will be an isolated connected component.
    __cc = ss.csgraph.connected_components(__nb_graph)[1]

    # Identify which triangles that neighbors the boundary of the filtered triangulation. __single is a boolean array
    # such that __single[i] is True if triangle i has exactly one side that neighbors the boundary. Similar for
    # __double, but two sides must adhere to the same condition.
    __single = np.sum(__nb == -1, axis=1) == 1
    __double = np.sum(__nb == -1, axis=1) == 2

    # Iterate all the connected components and create one polygon per component
    __polys = []
    for __comp in range(np.max(__cc) + 1):
        if np.all(~__inc[__cc == __comp]):
            # None of the triangles in this component are included
            continue

        # Extract all vertex indices along the boundary from the triangles with a single side towards the boundary. The
        # resulting __e0 and __e1 are defined such that the pair __e0[i], __e1[i] indicate that there is a boundary edge
        # from vertex __e0[i] to vertex __e1[i].
        __comp_nn = __nb[__single & (__cc == __comp)]
        if __comp_nn.size == 0:
            # There are no 'single edge triangles' for this component
            __e0 = __e1 = np.empty((0,), dtype=np.int)
        else:
            __c0 = np.where(__comp_nn == -1)[1]
            __comp_v = tri.simplices[__single & (__cc == __comp)]
            __e0 = __comp_v[range(__comp_v.shape[0]), __c0 - 2]
            __e1 = __comp_v[range(__comp_v.shape[0]), __c0 - 1]

        # Similar as above for __single, except that there will be two edges for each triangle instead of one. The pairs
        # from __f0, __f1 then indicate the vertices of the first boundary edge, and similarly __g0, __g1 the second
        # boundary edge.
        __comp_dd = __nb[__double & (__cc == __comp)]
        if __comp_dd.size == 0:
            __f0 = __f1 = __g0 = __g1 = np.empty((0,), dtype=np.int)
        else:
            __comp_vv = tri.simplices[__double & (__cc == __comp)]
            __c0 = np.where(__comp_dd == -1)[1]
            __f0 = __comp_vv[range(__comp_vv.shape[0]), __c0[::2] - 2]
            __f1 = __comp_vv[range(__comp_vv.shape[0]), __c0[::2] - 1]
            __g0 = __comp_vv[range(__comp_vv.shape[0]), __c0[1::2] - 2]
            __g1 = __comp_vv[range(__comp_vv.shape[0]), __c0[1::2] - 1]

        # Merge all the boundary edges from above
        __h0 = np.hstack((__e0, __f0, __g0))
        __h1 = np.hstack((__e1, __f1, __g1))
        if __h0.size == 0:
            # No boundary edges. In other words, this component is an isolated triangle that has not been filtered. We
            # can probably do a check for this earlier, but it is not expected to have significant run-time impact.
            __path = tri.simplices[__cc == __comp].flatten()
            __polys.append(sg.Polygon(tri.points[__path]))
            continue

        # Reconstruct the boundary. __h0, __h1 contain all the information about which edges (vertex indices) that make
        # up the boundary (pair-wise such as described above). However, there is no particular order in the __h0, __h1
        # values. To determine the proper order of the edges, we do a shortest path search to get the outer-most border
        # (in case of self-intersecting boundaries).

        # Define the boundary graph. Note that this is a directed graph. Moreover, the way the edges in __h0 and __h1
        # was extracted ensures that all edges follow triangles in a counter-clockwise fashion. Hence, a path between
        # two nodes in the graph will always follow the boundary counter-clockwise.
        __g = ss.coo_matrix((np.ones_like(__h0), (__h0, __h1)))

        # Determine the 'right-most' vertex. We cannot start from an arbitrary point on the boundary. We must ensure
        # that the starting point for the shortest path search is on the true outer boundary.
        __coords = tri.points[__h1]
        __rm = np.argmax(__coords[:, 0])

        # Calculate the shortest path from the vertex __rm to all other edges on the boundary. Shortest in the sense of
        # number of edges, not Euclidean distance
        __predec = ss.csgraph.dijkstra(__g, unweighted=True, indices=__h1[__rm], return_predecessors=True)[1]
        # The 'right-most' vertex picked above should have one, and only one, edge to it, namely the edge going from
        # __h0[__rm] to __h1[__rm]. By reconstructing the shortest path from __h1[__rm] to __h0[__rm], we should
        # retrieve the boundary of the filtered triangulation, without any self-intersections, as self-intersections
        # causes loops in the traversal path. Note that 8-like boundaries are not an issue since the 'upper' and 'lower'
        # part of the 8-shape are separate components.
        __path = [__h0[__rm]]
        while __path[-1] != __h1[__rm] and len(__path) < __h0.size:
            __path.append(__predec[__path[-1]])
        # Finally, create the a polygon from the reconstructed boundary
        __poly_a = sg.Polygon(tri.points[__path])
        __polys.append(__poly_a)
        # QC:
        # import matplotlib.pyplot as plt
        # plt.tripcolor(tri.points[:, 0], tri.points[:, 1], tri.simplices.copy(), __inc, edgecolors='g')
        # for __0, __1 in zip(__h0, __h1):
        #     plt.plot([tri.points[__0, 0], tri.points[__1, 0]],
        #              [tri.points[__0, 1], tri.points[__1, 1]],
        #              'bo-')
        # plt.plot(tri.points[__path, 0], tri.points[__path, 1], 'ro-')
    out = __polys
    return out


def _poly_hull(map_2d: np.ndarray, threshold: float, alpha: float) -> List[sg.Polygon]:
    points = np.argwhere(map_2d > threshold)
    try:
        polys = alphashape.alphashape(points, alpha)
    except sp.QhullError:
        return []
    if isinstance(polys, sg.MultiPolygon):
        return [p for p in polys]
    elif isinstance(polys, sg.Polygon):
        return [polys]
    else:
        return []


def _remove_small_polygons(polys, min_area):
    return [p for p in polys if p.area > min_area]


def _punch_holes(map2d: np.ndarray, poly: sg.Polygon, threshold: float, alpha: float) -> sg.Polygon:
    """
    Currently not in use since it introduces another parameter: threshold for hole area (0.1 below). It is not clear
    that it provides value beyond adjusting the existing input parameters, and thus only introduces more uncertainty.

    Also, this return the original polygon if there were any issues. We need to address that properly.

    Delete this function later if it is never used (see git log for age)
    """
    mp = sg.MultiPoint([(i, j) for i in range(map2d.shape[0]) for j in range(map2d.shape[1])])
    inner_map = np.zeros_like(map2d)
    idx = np.array(poly.intersection(mp), dtype=np.int)
    inner_map[idx[:, 0], idx[:, 1]] = 1.0 - map2d[idx[:, 0], idx[:, 1]]
    inner_holes = _poly_hull(inner_map, 1.0 - threshold, alpha)
    large_inner_holes = [ih for ih in inner_holes if ih.area > 0.1 * poly.area]
    # QC:
    # import matplotlib.pyplot as plt
    # xy = np.array(poly.exterior)
    # plt.plot(xy[:, 0], xy[:, 1])
    #
    # for ih in inner_holes:
    #     xy = np.array(ih.exterior)
    #     plt.plot(xy[:, 0], xy[:, 1])
    # plt.show()
    if len(large_inner_holes) == 0:
        return poly
    else:
        new_poly = sg.Polygon(np.array(poly.exterior), [p.exterior for p in large_inner_holes])
        if not isinstance(new_poly, sg.Polygon):
            print(
                'Polygon type changed after removing holes. This is not intended, using the original polygon instead.'
            )
            return poly
        if new_poly.is_valid is False:
            print('Polygon not valid after removing holes. Using the original polygon instead')
            return poly
        print(f'Removed {len(large_inner_holes)} holes from polyogon. Area diff: {poly.area} - '
              f'{sum([ih.area for ih in large_inner_holes])}')
        return new_poly
