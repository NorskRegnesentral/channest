import os
import plotly.graph_objs as go
import numpy as np
from typing import List

from channest.heights import LayerHeights
from channest.widths import LayerWidths


def create_width_height_scatter(fn: str,
                                poly_widths: List[LayerWidths],
                                poly_heights: List[LayerHeights],
                                dxy: float,
                                max_height: float,
                                max_width: float,
                                ):
    flat_widths = [w.stat(np.mean) * dxy for w in poly_widths]
    flat_heights = [h.flat_values_max_mode(max_height) for h in poly_heights]
    f = go.Figure()
    f.add_scatter(
        x=flat_widths,
        y=flat_heights,
        mode='markers',
        marker=dict(
            color=np.arange(len(flat_widths)),
            showscale=True,
            colorbar=dict(
                title='Layer [#]'
            )
        ),
    )
    f.layout.xaxis.title = 'Width [m]'
    f.layout.yaxis.title = 'Thickness [m]'
    f.layout.title = f'Thickness primary modes and width means'
    f.layout.xaxis.range = (0, max_width)
    f.layout.yaxis.range = (0, max_height)

    f.write_html(fn + '.html')
    try:
        f.write_image(fn + '.png', scale=2.0)
    except ValueError as e:
        # Can occur if orca is not installed or configured properly
        print(str(e))
        print('*' * 7, end='')
        print(f' Orca was not found. Execution continues without exporting {os.path.basename(fn)}.png ', end='')
        print('*' * 7)
