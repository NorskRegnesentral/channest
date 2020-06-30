import numpy as np
from typing import List, Dict, Any

from channest.heights import LayerHeights
from channest.widths import LayerWidths, Widths
from channest.polygonize import HulledPolygons
from channest.skeletonize import SkeletonizedLayer
from nrresqml.resqml import ResQml


def create_summary(dxy: float,
                   polys: List[HulledPolygons],
                   poly_skels: List[SkeletonizedLayer],
                   poly_widths: List[LayerWidths],
                   flat_widths: List[Widths],
                   poly_heights: List[LayerHeights],
                   flat_heights: np.ndarray) -> Dict[str, Any]:
    return _calculate_results(dxy, polys, poly_skels, poly_widths, flat_widths, poly_heights, flat_heights)


def _calculate_results(dxy: float,
                       polys: List[HulledPolygons],
                       poly_skels: List[SkeletonizedLayer],
                       poly_widths: List[LayerWidths],
                       flat_widths: List[Widths],
                       poly_heights: List[LayerHeights],
                       flat_heights: np.ndarray) -> Dict[str, Any]:
    results = {}
    # Segment widths
    sw = 'segment-width'
    if len(flat_widths) == 0:
        results[sw] = {
            'mean': np.nan,
            'sd': np.nan,
            'count': 0
        }
    else:
        per_segment = np.hstack([np.hstack(w.full_widths) for w in flat_widths if w.full_widths.size > 0])
        results[sw] = {
            'mean': np.mean(per_segment) * dxy,
            'sd': np.std(per_segment) * dxy,
            'count': per_segment.size,
        }

    # Channel widths
    mean_per_channel = [np.mean(w.full_widths) for w in flat_widths if w.full_widths.size > 0]
    results['channel-width'] = {
        'mean': np.mean(mean_per_channel) * dxy,
        'sd': np.std(mean_per_channel) * dxy,
        'count': len(mean_per_channel)
    }

    # Segment heights
    results['segment-height'] = {
        'mean': np.mean(flat_heights) if flat_heights.size > 0 else np.nan,
        'sd': np.std(flat_heights) if flat_heights.size > 0 else np.nan,
        'count': flat_heights.size
    }

    # Channel heights
    height_per_channel = np.hstack([h.mean_heights() for h in poly_heights])
    h_max = LayerHeights.calculate_max_height(poly_heights)
    mode_per_channel = np.array([h.flat_values_max_mode(h_max) for h in poly_heights])
    results['channel-height'] = {
        'mean': np.mean(height_per_channel) if height_per_channel.size > 0 else np.nan,
        'sd': np.std(height_per_channel) if height_per_channel.size > 0 else np.nan,
        'count': height_per_channel.size,
        'mode-mean': np.nanmean(mode_per_channel),
        'mode-sd': np.nanstd(mode_per_channel),
    }

    return results
