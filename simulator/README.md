# Localization Simulator

This Python-based simulator aims at providing an interface to evaluate
the performance of different localization algorithms with varying
parameters.

The simulator is stratified in several modules related to:
* Environment generation: anchor positions (in a local 3D coordinate system)
* Measurement technology & noise model (ToF, AoA...)
* Anchor selection procedure (see details below) for power consuption issues
* Geolocalization algorithm / Minimization cost function
* Visualization and statistics

Of course, the cost function as well as the noise model will depend on
the choice of the technology.

The simulator's goal is not to provide real time display of the localization
results, but to give information (statistics and diagrams) making it easier
to compare the performance of various localization procedures and
technologies in a given environment.

The organization in modules enables the user to customize it and to easily
change only some blocks of the whole procedure.

The only requirement is to observe the same API between consecutive blocks:

1. **BLOCK 1:**  Environment generation \\
This block must generate a dictionary of anchors containing three fields `'x'`,
`'y'` and '`z`' with unique IDs. It can be generated from an array of coordinates,
a text file, manual input, etc.

The `anchors` Python dictionary is given to the second block as an input.

This block also provides a grid of 3D locations (_i.e._ tag positions) where the
localization performance will be evaluated, as well as the number of tries for
each of these positions.


2. **BLOCK 2:**  Grid and measurement generation \\


