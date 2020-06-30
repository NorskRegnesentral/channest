import numpy as np
from typing import List

from channest.heights import LayerHeights


class LayerHeightPlot:
    def __init__(self):
        import plotly.graph_objs as go
        self._f = go.Figure()
        self._f.layout.xaxis.range = (-0.5, 15.0)
        self._f.layout.xaxis.title = 'Thickness (m)'
        self._f.layout.yaxis.title = 'Sampled Layer #'
        self._f.layout.title = 'Channel Thickness Estimates'
        self._f.layout.hovermode = 'y'
        self._f.layout.legend.font.family = 'monospace'

    def add_heights(self,
                    heights: List[LayerHeights],
                    suffix: str):
        # Calculate statistics: mean, median, max, modes
        values = [h.flat_values() for h in heights]
        values = [v if v.size > 0 else np.zeros(1) for v in values]  # Transform to make stats calculations easier
        means = np.array([np.mean(v) for v in values])
        medians = np.array([np.median(v) for v in values])
        maxs = np.array([np.max(v) for v in values])
        ys = np.arange(means.size)
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
        h_max = LayerHeights.calculate_max_height(heights)
        main_modes = np.array([h.flat_values_max_mode(h_max) for h in heights])
        self._f.add_scatter(
            x=np.where(np.isnan(main_modes), 0, main_modes),
            y=ys,
            mode='markers+lines',
            marker=dict(
                color='#f90'
            ),
            name='Main mode ' + suffix
        )
        th = 0.01
        max_circle = max(400 / len(heights), 15)
        for i_layer, layer in enumerate(heights):
            kde_modes = layer.flat_values_kde_modes(h_max)
            x_points = np.insert(kde_modes[0], [0, kde_modes[0].size], [0, maxs[i_layer]])
            y_points = np.insert(kde_modes[1], [0, kde_modes[1].size], [0, 0])
            self._f.add_scatter(
                x=x_points,
                y=[i_layer] * x_points.size,
                mode='lines+markers',
                marker=dict(
                    color='#0c0',
                    # Hide small entries by setting size to 0:
                    size=np.where(y_points > th, 5 + y_points * (max_circle - 5), 0),
                    line=dict(width=2, color='#000')
                ),
                text=[f'Layer #: {i_layer}<br>Peak: {p:2.2f}'
                      for t, p in zip(x_points, y_points)],
                name='Modes     ' + suffix,
                legendgroup='Modes',
                showlegend=i_layer == 0,
            )

    def write(self, fn):
        self._f.write_html(fn + '.html', include_plotlyjs='cdn')
