from typing import List

import numpy as np
import shapely.geometry
import skimage.morphology
import shapely.ops


def _find_image_skeleton(poly):
    n0 = int(poly.bounds[2])
    n1 = int(poly.bounds[3])
    full_set = shapely.geometry.MultiPoint([(i, j) for i in range(n0) for j in range(n1)])
    poly_set = full_set.intersection(poly)
    poly_map = np.zeros((n0, n1))
    idx = np.array(poly_set).astype(np.int)
    poly_map[idx[:, 0], idx[:, 1]] = 1
    skel_map = skimage.morphology.skeletonize(poly_map)
    return skel_map


def _find_neighbors(skel_map, b_ind):
    i = b_ind[0]
    j = b_ind[1]
    nb = []
    dx = [-1,-1,-1,0,0,1,1,1]
    dy = [-1,0,1,-1,1,-1,0,1]
    for k in range(8):
        nei_x = i+dx[k]
        nei_y = j+dy[k]
        if not 0 <= nei_x < skel_map.shape[0]:
            continue
        if not 0 <= nei_y < skel_map.shape[1]:
            continue
        if skel_map[nei_x, nei_y] == 1:
            nb.append((nei_x, nei_y))
    return nb


def _find_restricted_neighbours(skel_map, b_ind, r_ind):
    i = b_ind[0]
    j = b_ind[1]
    nb = []
    dx = [-1,-1,-1,0,0,1,1,1]
    dy = [-1,0,1,-1,1,-1,0,1]
    for k in range(8):
        ic = i+dx[k]
        jc = j+dy[k]
        if not 0 <= ic < skel_map.shape[0]:
            continue
        if not 0 <= jc < skel_map.shape[1]:
            continue
        if abs(ic-r_ind[0]) > 1 or abs(jc-r_ind[1]) > 1:
            if skel_map[i+dx[k],j+dy[k]] == 1:
                nb = nb +[(i+dx[k],j+dy[k])]
    return nb


def _process_map_piece(skel_map, branch, cur):
    p = [branch, cur]
    skel_map[cur[0], cur[1]] = 0
    fw = _find_restricted_neighbours(skel_map, cur, branch)
    while len(fw) == 1:
        cur = fw[0]
        p = p + [cur]
        skel_map[cur[0], cur[1]] = 0
        fw = _find_neighbors(skel_map, cur)
    next_branch = [cur]
    pl = shapely.geometry.linestring.LineString(p)
    piece = pl.simplify(0.8, False)
    return piece, next_branch


def _process_skeleton_map(skel_map):
    ind1, ind2 = np.argwhere(skel_map)[0, :]
    branches = [(ind1, ind2)]
    b_index = 0
    links = [[]]
    pieces = []
    skel_map_cp = skel_map.copy()  # Copy skel_map to not edit in-place
    skel_map_cp[ind1, ind2] = 0
    while b_index < len(branches):
        link = []
        nbs = _find_neighbors(skel_map_cp, branches[b_index])
        for nb in nbs:
            piece, b_inds = _process_map_piece(skel_map_cp, branches[b_index], nb)
            branches = branches + b_inds
            pieces = pieces + [piece]
            links = links + [[]]
            link = link + [len(pieces) - 1]
        links[b_index] = link
        b_index = b_index + 1
    return pieces, links


def _calculate_skeleton(poly):
    skel_map = _find_image_skeleton(poly)
    pieces, links = _process_skeleton_map(skel_map)
    return pieces, links


class SkeletonizedPolygon:
    def __init__(self, poly) -> None:
        self._polygon = poly
        self._skel_map = _find_image_skeleton(poly)
        self._pieces, self._links = _process_skeleton_map(self._skel_map)

    @property
    def polygon(self):
        return self._polygon

    @property
    def pieces(self):
        return self._pieces

    @property
    def links(self):
        return self._links

    @property
    def skeleton_map(self) -> np.ndarray:
        return self._skel_map

    def main_channel(self) -> shapely.geometry.LineString:
        import scipy.sparse.csgraph as sc
        import scipy.sparse as ss
        # Create graph
        # Find longest shortest path
        # Convert to LineString
        # Return
        if len(self._pieces) == 1:
            return self._pieces[0]
        # Convert links to graph edges
        assert len(self._links) == len(self._pieces) + 1
        edges = np.array([
            (j, li + 1)
            for j, link in enumerate(self._links)
            for li in link
        ])
        rows = edges[:, 0]
        cols = edges[:, 1]
        # Extract graph edges lengths
        lengths = np.array([p.length for p in self._pieces])
        data = lengths[cols - 1]
        # Define graph as csc matrix and calculate shortest path from node 0. Node 0 is assumed to be the 'most
        # upstream' node of the channel. This is also why we perform a directed search.
        g = ss.csc_matrix((data, (rows, cols)), shape=(len(self._links), len(self._links)))
        path_lengths, predecessors = sc.dijkstra(g, directed=True, return_predecessors=True, indices=0)
        path_lengths[np.isinf(path_lengths)] = 0.0  # In case there are inconsistencies
        # Reconstruct longest path
        im, jm = np.unravel_index(np.argmax(path_lengths), shape=g.shape)
        path = [jm]
        while path[-1] != -9999 and path[-1] != im:
            path.append(predecessors[path[-1]])
        path = path[::-1]
        # Construct concatenated coordinates. Node i corresponds to piece i - 1, and we can therefore use the correct
        # piece without mapping pieces to graph nodes first
        longest_line = shapely.ops.linemerge([self._pieces[p - 1] for p in path[1:]])
        return longest_line


class SkeletonizedLayer:
    def __init__(self, polys) -> None:
        self._skeletonized_polys = [SkeletonizedPolygon(p) for p in polys]

    @property
    def skeletonized_polygons(self) -> List[SkeletonizedPolygon]:
        return self._skeletonized_polys
