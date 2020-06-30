from dataclasses import dataclass
import json
import numpy as np
import os
import pathlib
import pickle

from channest._utils import progress
from channest.qc.layerheightplot import LayerHeightPlot
from channest.qc.layerwidthplot import LayerWidthPlot
from channest.qc.scatterwidthheight import create_width_height_scatter
from channest.summarize import create_summary
from nrresqml.resqml import ResQml
from typing import Any, Dict, Optional, Union
from channest import polygonize, skeletonize, widths, heights
from channest.qc import fences, sliceplot, stacking


@dataclass
class _Setup:
    merge_layers: int
    alpha_hull: float
    element_threshold: Optional[float]
    mean_map_threshold: float
    turn_off_filters: bool

    def name(self):
        return f'{self.merge_layers:_>2d}_{self.turn_off_filters}'


def calculate_channel_parameters(settings: Union[Dict[str, Any], str], output_directory: str):
    if isinstance(settings, str):
        # Read settings from settings file
        settings = json.load(open(settings))
    data_file = settings['data_file']
    merge_layers = settings.get('merge_layers', [5])
    alpha_hull = settings.get('alpha_hull', [0.6])
    element_threshold = settings.get('element_threshold', [None])
    mean_map_threshold = settings.get('mean_map_threshold', [0.9])
    minimum_polygon_area = settings.get('minimum_polygon_area', 100)
    turn_off_filters = settings.get('turn_off_filters', [False])
    step_z = settings.get('step_z', 1)
    z0 = settings.get('z0', 0)
    cropbox = settings.get('cropbox', None)
    generate_plots = settings.get('generate_plots', False)
    generate_fences = settings.get('generate_fences', False)
    pickle_data = settings.get('pickle_data', False)
    scatter_max_width = settings.get('scatter_max_width', 500.0)
    scatter_max_height = settings.get('scatter_max_height', 14.0)

    data_path = pathlib.Path(data_file)
    rq = ResQml.read_zipped(data_path)
    if cropbox is None:
        box = None
    else:
        box = polygonize.Box(cropbox['x_0'], cropbox['y_0'], cropbox['x_1'], cropbox['y_1'])

    # Prepare setups and output
    os.makedirs(output_directory, exist_ok=True)
    if not isinstance(merge_layers, list):
        merge_layers = [merge_layers]
    if not isinstance(alpha_hull, list):
        alpha_hull = [alpha_hull]
    if not isinstance(element_threshold, list):
        element_threshold = [element_threshold]
    setups = [_Setup(merge_layers=m, alpha_hull=a, element_threshold=e, mean_map_threshold=n, turn_off_filters=f)
              for m in merge_layers
              for e in element_threshold
              for a in alpha_hull
              for n in mean_map_threshold
              for f in turn_off_filters]

    # Core calculations
    results = []
    _cache = {}
    raw_cube, pillars, grid_params = polygonize.create_cube(rq, box)
    assert np.isclose(grid_params.dx, grid_params.dy, atol=0.1)  # Algorithms are calibrated for dx == dy (aprx.)
    dxy = grid_params.dx
    slicer = slice(z0, raw_cube.shape[0], step_z)
    for s in setups:
        et = 0.5 / s.merge_layers if s.element_threshold is None else s.element_threshold
        # Check cache if results have already been calculated
        _cache_state = (s.merge_layers, s.alpha_hull, s.element_threshold)
        if _cache_state in _cache:
            polys, poly_skels = _cache[_cache_state]
        else:
            cube = polygonize.smoothen_cube(raw_cube, s.merge_layers)
            polys = polygonize.calculate_polygons(cube,
                                                  slicer,
                                                  et,
                                                  s.alpha_hull,
                                                  minimum_polygon_area)
            poly_skels = [skeletonize.SkeletonizedLayer(h.polygons) for h in progress(polys, 'Skeletonizing layers')]
            _cache[_cache_state] = polys, poly_skels
        # Done calculating values OR fetching from cache
        # Widths
        poly_widths = [
            widths.LayerWidths(_sp,
                               (_p.map2d > et).astype(np.float),
                               s.mean_map_threshold,
                               s.turn_off_filters)
            for _p, _sp in progress(zip(polys, poly_skels),
                                    desc='Calculating widths', total=len(polys))
        ]
        flat_widths = [_w for w in poly_widths for _w in w.widths]

        # Heights
        vdist_cube = heights.prepare_distance_cube(raw_cube, pillars)
        poly_heights = [
            heights.LayerHeights(_pw, _dc)
            for _pw, _dc in progress(zip(poly_widths, vdist_cube[slicer]),
                                     desc='Calculating thicknesses', total=len(poly_widths))
        ]
        flat_heights = np.hstack([ph.flat_values() for ph in poly_heights])

        # Append to results
        results.append((s, polys, poly_skels, poly_widths, flat_widths, poly_heights, flat_heights))

    # Dump pickle file
    if pickle_data is True:
        pickle.dump(results, open(f'{output_directory}/data.pkl', 'wb'))

    # Write results
    lwp = LayerWidthPlot()
    lhp = LayerHeightPlot()
    for key, polys, poly_skels, poly_widths, flat_widths, poly_heights, flat_heights in progress(
            results, desc='Post-processing results'):
        post_fix = f'_{key.name()}' if len(results) > 1 else ''
        res = create_summary(dxy, polys, poly_skels, poly_widths, flat_widths, poly_heights, flat_heights)
        res['settings'] = settings
        o_sn = os.path.join(output_directory, f'summary{post_fix}.json')
        json.dump(res, open(o_sn, 'w'), indent='    ')
        create_width_height_scatter(f'{output_directory}/tw_scatter', poly_widths, poly_heights, dxy,
                                    scatter_max_height, scatter_max_width)
        if generate_plots is True:
            lwp.add_widths(poly_widths, [p.map2d for p in polys], 0.5 / key.merge_layers, post_fix)
            lhp.add_heights(poly_heights, post_fix)
            sliceplot.create_slice_plot(f'{output_directory}/slices{post_fix}',
                                        list(zip(polys, poly_skels, poly_widths)))
            sliceplot.create_heights_slice_plot(f'{output_directory}/heights{post_fix}', poly_heights)
            sliceplot.create_kde_slice_plot(f'{output_directory}/kde{post_fix}', poly_heights)
            stacking.create_stacking_plot(f'{output_directory}/stacking{post_fix}', poly_heights)
        if generate_fences is True:
            fence_dir = f'{output_directory}/fences'
            os.makedirs(fence_dir, exist_ok=True)
            fences.generate_fences(fence_dir, poly_skels, grid_params.box.x0, grid_params.box.y0, dxy)

    if generate_plots is True:
        lwp.write(f'{output_directory}/width-means')
        lhp.write(f'{output_directory}/thickness-means')
