from typing import List

from channest.skeletonize import SkeletonizedLayer


def generate_fences(output_directory: str, sk_layers: List[SkeletonizedLayer], x0: float, y0: float, dxy: float):
    for i, s in enumerate(sk_layers):
        channels = [p.main_channel() for p in s.skeletonized_polygons]
        if len(channels) == 0:
            continue
        longest = max(channels, key=lambda x: x.length)
        coords = [f'{x0 + x * dxy} {y0 + y * dxy} 0.0\n' for x, y in longest.coords]
        open(f'{output_directory}/{i}.xyz', 'w').writelines(coords)
