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
    """
    Estimate channel parameters based on the provided parameters

    ### settings
    File path to a json file or a dictionary containing estimation settings. All settings are optional except
    **data_file**. In addition to these settings, advanced settings are described below. There are several available
    advanced settings. However, the default values have been determined experimentally and should work well for most
    Delft3D models. The advanced settings are documented below for completeness.

    - **data_file** File path to a RESQML model (.epc file)

    - **crop_box** Dictionary describing the extent of the model to use for estimation. Specified by providing keys x_0,
    x_1, y_0 and y_1 with float values. Delft3D models are typically starting at x=0, y=0.

    ### output_directory
    Directory to which output is written.  The following files are written (relative to the provided directory):

    - **tw_scatter.png** Scatter plot showing the channel thickness/width distribution per layer. Requires plotly-orca,
    otherwise, this is skipped.

    - **tw_scatter.html** Scatter plot showing the channel thickness/width distribution per layer. Same as
    tw_scatter.png, except as html (based on plotly) which adds zoom and pan functions.

    - **summary.json** JSON file containing the main results as well as the settings used to generate the results.
    Values under "channel-width" and "channel-height" are averaged over layers, with each layer having equal weight.
    Values under "segment-width" and "segment-height" are averaged over width segments, with each segment having equal
    weight.

    ### Advanced settings
    The advanced settings can be split in two: method-related and output-related. Some settings under method-related
    must be specified as lists of single values. All combinations of such values are then executed in a
    multi-configuration fashion, similar to vargrest. These settings are indicated by having a default values surrounded
    by [brackets].

    #### Method-related parameters:
    - **merge_layers** Number of layers to merge when calculating segments. Default is [5].

    - **alpha_hull** Parameter to the alpha hull algorithm. 0.0 yields the convex hull. Default is [0.6]

    - **element_threshold** Floating point threshold in number of layers for which points to include as channels in the
    merge layers. A value of None yields a default of including all points with a channel in at least one layer. Default
    is [None]

    - **mean_map_threshold** Threshold between 0.0 and 1.0 used when filtering segments that cross areas not labeled as
    channel. A value of 1.0 removes all segments touching an area not labeled as channel. A value of 0.0 will only
    remove segments that does not touch areas labeled as channel at all. Default is [0.9]

    - **minimum_polygon_area** Minimum area of the alpha polygon shape for it to be included in the estimation. Default
    is 100.

    - **turn_off_filters** Disables all segment filters when set to True. Default is [False].

    - **step_z** Sampling rate in z-direction in number of layers. Default is 1, which means all layers are sampled.

    - **z0** Starting layer for sampling in z-direction. Default is 0.

    #### Output-related parameters:

    - **generate_plots** Generate additional quality assessment plots. Default is False.

    - **generate_fences** Generate poly lines as text files along the longest channel in each layer. These lines can be
    important and used as “fences” in RMS for. Default is False.

    - **pickle_data** Store preliminary results in a Python pickle file. Main purpose is debugging or alternative
    post-processing. Default is False.

    - **scatter_max_width** Length of the x-axis of the TW scatter plot, representing channel width. Default is 500.

    - **scatter_max_height** Height of the y-axis of the TW scatter plot, representing channel thickness. Default is 14.
    """
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
        # Replace 'NaN' with 'null' to comply with JSON specification. This is not a bullet-proof replacement, but
        # should not be a problem since we have full control of the content of the output file
        new_jf = open(o_sn, 'r').read().replace('NaN', 'null')
        open(o_sn, 'w').write(new_jf)
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
