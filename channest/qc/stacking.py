import numpy as np
from typing import List
from plotly.subplots import make_subplots
from plotly import graph_objs as go
from channest.heights import LayerHeights


def create_stacking_plot(fn: str, heights: List[LayerHeights]):
    _generate_stacking_plot(fn + '_skeleton', _generate_stacking_data(heights, 'height_map'))
    _generate_stacking_plot(fn + '_full', _generate_stacking_data(heights, 'vertical_distance_map'))


def _generate_stacking_data(heights: List[LayerHeights], property_name: str):
    nx, ny = [h.heights[0] for h in heights if len(h.heights) > 0][0].height_map.shape
    is_data = []
    hh_data = []
    for layer in heights:
        if len(layer.heights) == 0:
            merged_hm = np.zeros((nx, ny))
        else:
            merged_hm = np.sum(np.dstack([getattr(h, property_name) for h in layer.heights]), axis=-1)
        is_data.append(merged_hm > 0)
        hh_data.append(merged_hm)
    is_data = np.array(is_data)
    hh_data = np.array(hh_data)
    hh_data[~is_data] = np.nan
    return hh_data


def _distance_to_mouth(data: np.ndarray):
    # Highly specialized and only applicable to "full box" cases
    nx, ny = data.shape[1:]
    ci, cj = 0, ny / 2
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    d_center = np.sqrt((ii - ci) ** 2 + (jj - cj) ** 2)
    all_d = []
    all_t = []
    for d in data:
        if np.isnan(d).all():
            continue
        has_data = ~np.isnan(d)
        all_d.append(d_center[has_data])
        all_t.append(d[has_data])
    return np.hstack(all_d), np.hstack(all_t)


def _generate_stacking_plot(fn: str, data: np.ndarray):
    mean = np.nanmean(data, axis=0)
    _max = np.nanmax(data, axis=0)
    count = np.sum(~np.isnan(data), axis=0)

    fig = make_subplots(1, 4, subplot_titles=['Mean', 'Count', 'Max', 'T/D'])
    fig.add_heatmap(z=mean.T, row=1, col=1, showscale=False)
    fig.add_heatmap(z=count.T, row=1, col=2, showscale=False)
    fig.add_heatmap(z=_max.T, row=1, col=3, showscale=False)
    _add_distance_to_mouth_data(fig, data, row=1, col=4)
    fig.layout.yaxis.scaleanchor = 'x'
    fig.layout.yaxis2.scaleanchor = 'x2'
    fig.layout.yaxis3.scaleanchor = 'x3'
    fig.layout.yaxis4.title = 'Thickness'
    fig.layout.xaxis4.title = 'Distance to river mouth'
    fig.write_html(fn + '.html')


def _add_distance_to_mouth_data(fig: go.Figure, data: np.ndarray, **kwargs):
    dists, thicknesses = _distance_to_mouth(data)
    bins = np.linspace(0, np.max(dists), 30)
    ib = np.digitize(dists, bins)
    q10 = []
    q50 = []
    q90 = []
    mean = []
    for i in range(bins.size):
        t = thicknesses[ib == i + 1]
        if t.size == 0:
            q10.append(0)
            q50.append(0)
            q90.append(0)
            mean.append(0)
        else:
            q10.append(np.percentile(t, 10))
            q50.append(np.percentile(t, 50))
            q90.append(np.percentile(t, 90))
            mean.append(np.mean(t))

    db = bins[1] - bins[0]
    fig.add_scatter(x=dists, y=thicknesses, mode='markers', marker=dict(size=2), name='Raw Data', **kwargs)
    common = dict(x=db / 2 + bins[:-1], width=db, offsetgroup='whatever', **kwargs)
    fig.add_bar(y=q90, name='p<sub>90</sub>', **common)
    fig.add_bar(y=mean, name='Mean', **common)
    fig.add_bar(y=q50, name='p<sub>50</sub>', **common)
    fig.add_bar(y=q10, name='p<sub>10</sub>', **common)
