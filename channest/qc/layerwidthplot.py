import pathlib
import numpy as np
from typing import List, Union

from channest.widths import Widths, LayerWidths


class LayerWidthPlot:
    def __init__(self):
        import plotly.graph_objs as go
        self._f = go.Figure()
        self._f.layout.xaxis.range = (-0.5, 70.0)
        self._f.layout.xaxis.title = 'Width (#cells)'
        self._f.layout.yaxis.title = 'Sampled Layer #'
        self._f.layout.title = 'Channel Width Estimates'
        self._f.layout.hovermode = 'y'
        self._f.layout.legend.font.family = 'monospace'

    def add_widths(self,
                   widths: List[LayerWidths],
                   maps: List[np.ndarray],
                   mean_threshold: float,
                   suffix: str):
        # Calculate statistics
        means = np.array([_w.stat(np.mean) for _w in widths])
        medians = np.array([_w.stat(np.median) for _w in widths])
        stds = np.array([_w.stat(np.std) for _w in widths])
        maxs = np.array([_w.stat(np.max) for _w in widths])
        ys = np.arange(means.size)
        self._f.add_scatter(
            x=np.hstack((means + stds, means[::-1] - stds[::-1])),
            y=np.hstack((ys, ys[::-1])),
            mode='none',
            opacity=0.5,
            fill='toself',
            fillcolor='#aaa',
            name='Mean &plusmn; SD ' + suffix
        )
        self._f.add_scatter(
            x=means,
            y=ys,
            mode='lines+markers',
            name='Mean      ' + suffix
        )
        self._f.add_scatter(
            x=medians,
            y=ys,
            mode='lines+markers',
            name='Median    ' + suffix
        )
        self._f.add_scatter(
            x=maxs,
            y=ys,
            mode='markers+lines',
            marker=dict(
                color='#000',
            ),
            name='Max.      ' + suffix
        )

    def write(self, fn):
        self._f.write_html(fn + '.html', include_plotlyjs='cdn')
