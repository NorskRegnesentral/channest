import os
from channest.api import calculate_channel_parameters


__version__ = open(os.path.join(os.path.dirname(__file__), 'VERSION.txt')).read()
__all__ = ['calculate_channel_parameters', '__version__']
