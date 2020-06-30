import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Callable

from channest.heights import LayerHeights
from channest.widths import LayerWidths
from channest.polygonize import HulledPolygons
from channest.skeletonize import SkeletonizedLayer


def create_kde_slice_plot(fn: str, data: List[LayerHeights]):
    f = make_subplots(specs=[[{"secondary_y": True}]])
    cumulative = []
    h_max = LayerHeights.calculate_max_height(data)
    # Ignore underflow warning from gaussian_kde
    max_v = 0
    for d in data:
        x, y, v = d.flat_values_kde(h_max)
        if np.isnan(y).any():
            cumulative.append(len(f.data))
            continue
        max_v = max(max_v, np.max(v))
        visible = len(f.data) == 0
        f.add_scatter(x=x, y=y, visible=visible, name='Density Estimate', fill='tozeroy')
        f.add_histogram(
            x=v,
            histnorm='',
            xbins=dict(start=-1e-6, end=h_max, size=h_max / x.size),
            name=f'Histogram (Total count: {v.size})',
            secondary_y=True,
            visible=visible,
            marker=dict(color='#0c6')
        )
        f.add_scatter(
            x=v,
            y=np.zeros_like(v),
            visible=visible,
            mode='markers',
            marker=dict(
                symbol='line-ns', line=dict(width=1, color='#c00'), size=24
            ),
            name='Data Points',
            secondary_y=True,
        )
        cumulative.append(len(f.data))
    sliders = _create_slider(cumulative)
    f.layout.yaxis.title = 'Density'
    f.layout.xaxis.title = 'Thickness [m]'
    f.update_yaxes(title_text="Density", secondary_y=False, rangemode='nonnegative')
    f.update_yaxes(title_text="Histogram (Count)", secondary_y=True, rangemode='nonnegative')
    f.update_layout(sliders=sliders, legend=dict(x=0.7, y=0.9, bgcolor="#fff"))
    f.write_html(fn + '.html', include_plotlyjs='cdn')


def create_slice_plot(fn: str, data: List[Tuple[HulledPolygons, SkeletonizedLayer, LayerWidths]]):
    _create_slider_based_plot(fn, _add_slice, data)


def create_heights_slice_plot(fn: str, data: List[LayerHeights]):
    import plotly.colors
    orig_cs = plotly.colors.PLOTLY_SCALES['Reds']
    # Adjust the colorscale so that the lowest values are fully transparent. NB: Current implementation creates poor
    # contrast for the lowest values since the first color is stretched over a longer interval than default (see
    # division by 100). However,
    cs = [
        [0, 'rgba(0,0,0,0.0)']
    ] + [
        [orig_cs[1][0] / 100, orig_cs[1][1][:-1].replace('rgb', 'rgba') + ',1.0)']
    ] + [
        [v, c[:-1].replace('rgb', 'rgba') + ',1.0)']
        for v, c in orig_cs[1:]
    ]

    def _add_h_slice(f: go.Figure, layer: LayerHeights):
        if len(layer.heights) == 0:
            return
        h_map = np.maximum.reduce([h.height_map for h in layer.heights])
        v_map = np.maximum.reduce([h.vertical_distance_map for h in layer.heights])
        f.add_heatmap(
            z=v_map.T,
            colorscale='Greys',
            showscale=False,
            zmin=0.0,
            zmax=5.0,
            visible=len(f.data) == 0,
        )
        f.add_heatmap(
            z=h_map.T,
            colorscale=cs,
            zmin=0.0,
            zmax=5.0,
            visible=len(f.data) == 0,
            colorbar=dict(
                len=0.5,
                thicknessmode='fraction',
                thickness=0.05,
                x=0.975,
                xpad=0,
            )
        )
    data = [(d,) for d in data]  # Transformation to adhere to _create_slider_based_plot
    _create_slider_based_plot(fn, _add_h_slice, data)


def _create_slider_based_plot(fn: str, slice_adder: Callable, data: List[Tuple]):
    f = go.Figure()
    cumulative_data = []
    for d in data:
        slice_adder(f, *d)
        cumulative_data.append(len(f.data))
    sliders = _create_slider(cumulative_data)
    f.layout.yaxis.scaleanchor = 'x'
    f.layout.plot_bgcolor = '#fff'
    f.update_layout(legend_orientation='h', sliders=sliders)
    f.write_html(fn + '.html', include_plotlyjs='cdn')


def _add_slice(fig, hp: HulledPolygons, ps: SkeletonizedLayer, ws: LayerWidths):
    assert len(ps.skeletonized_polygons) == len(ws.widths)
    visible = len(fig.data) == 0
    # Background
    fig.add_heatmap(
        z=hp.map2d.T,
        colorscale='Greys',
        zmin=0.0,
        zmax=1.3,  # Reduces 'darkness' in the image
        visible=visible,
    )
    # Polygons
    showlegend = True
    for poly in hp.polygons:
        crd = np.array(poly.exterior)
        fig.add_scatter(
            x=crd[:, 0],
            y=crd[:, 1],
            mode='lines',
            marker=dict(
                color='#f00',
            ),
            legendgroup='Polygons',
            name='Polygons',
            showlegend=showlegend,
            visible=visible,
        )
        showlegend = False
    # Skeletons
    showlegend = True
    for skel in ps.skeletonized_polygons:
        for p in skel.pieces:
            crd = np.array(p)
            fig.add_scatter(
                x=crd[:, 0],
                y=crd[:, 1],
                mode='lines',
                marker=dict(
                    color='#00f'
                ),
                legendgroup='Skeletons',
                name='Skeletons',
                showlegend=showlegend,
                visible=visible,
            )
            showlegend = False

    # Width segments
    showlegend = True
    for width in ws.widths:
        if len(width.segments) == 0:
            continue
        # Merge all pieces into one array. For more fine-grained control, add another for-loop (or see previous commits)
        # This will approximately increase the run time of create_slice_plot by 50%. The plot will still be quite
        # responsive. Adding two for-loops for even more fine-grained control will reduce the responsiveness of the plot
        # and also increase the run time of create_slice_plot.
        #   In base_model/0, run time goes from 13s (one for-loop) to 20s (double for-loop) to 60s (triple for-loop)
        try:
            crd = np.vstack([
                np.vstack((np.array(segment), np.full((1, 2), np.nan)))
                for per_piece in width.segments
                for segment in per_piece
                if len(per_piece) > 0
            ])
        except ValueError:
            # Can happen if the above list comprehension is empty
            crd = np.empty((0, 2))
        fig.add_scatter(
            x=crd[:, 0],
            y=crd[:, 1],
            mode='lines',
            marker=dict(
                color='#0b0',

            ),
            legendgroup='Widths',
            name='Widths',
            showlegend=showlegend,
            visible=visible,
        )
        showlegend = False


def _create_slider(cumulative_data):
    steps = []
    for i in range(len(cumulative_data)):
        s = dict(
            method='restyle',
            args=['visible', [False] * cumulative_data[-1]]
        )
        c_start = 0 if i == 0 else cumulative_data[i - 1]
        c_end = cumulative_data[i]
        for c in range(c_start, c_end):
            s['args'][1][c] = True
        steps.append(s)
    sliders = [dict(
        active=0,
        steps=steps
    )]
    return sliders
