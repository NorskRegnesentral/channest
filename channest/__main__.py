import argparse
import json

import channest
from time import perf_counter


parser = argparse.ArgumentParser(prog='python -m channest', description='Estimate channel parameters')
parser.add_argument(
    'input_file', metavar='<input-file>', help='Input file containing data to be analyzed'
)
parser.add_argument(
    'output_directory', metavar='<output-directory>', help='Destination directory for variogram estimation output'
)

args = parser.parse_args()
t0 = perf_counter()
settings = json.load(open(args.input_file))
channest.calculate_channel_parameters(settings, args.output_directory)
t1 = perf_counter()
print(f'Estimation completed in {t1 - t0} s')
