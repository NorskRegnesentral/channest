This package estimates channel parameters from Delft3D-based RESQML models. The repository is tightly linked with https://github.com/NorskRegnesentral/nrresqml 

The main function is
<pre>
channest.calculate_channel_parameters(settings, output_directory)
</pre>

## <tt>calculate_channel_parameters</tt>

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
important and used as "fences" in RMS for. Default is False.

- **pickle_data** Store preliminary results in a Python pickle file. Main purpose is debugging or alternative
post-processing. Default is False.

- **scatter_max_width** Length of the x-axis of the TW scatter plot, representing channel width. Default is 500.

- **scatter_max_height** Height of the y-axis of the TW scatter plot, representing channel thickness. Default is 14.