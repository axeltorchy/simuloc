# Wizzilab Localization Simulator

## Introduction

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


The user might want to compare (simultaneously or not) the influence of several
factors. Of course, it is not always realistic to compare the performance of two
completely different settings. Instead, one might prefer observing the influence
of a change in individual parameters to understand their impact on the result.

Thus, the chosen solution to be able to observe simultaneously the results is
to run several simulations and save the results in the same file. 

When reading the log file, the visualization module will detect all the
different configurations and superimpose the results on the same graphs.

Improvements could be adopted in future versions to deal with this issue. For
example, the user might want to analyse the performance of two different cost
functions, with 4 different models of noise for each cost function. The best
visualisation would then probably be four colors for the four models of noise,
plain lines for the first cost function and dashed lines for the second one.


## General description of the API / blocks

The only requirement is to observe the same API between consecutive blocks:

#### BLOCK 1:  Environment generation  
  This block must generate a dictionary of anchors containing three fields `'x'`,
`'y'` and '`z`' with unique IDs. It can be generated from an array of coordinates,
a text file, manual input, etc.  
  The `anchors` Python dictionary is given to the second block as an input.  
  This block also provides a grid of 3D locations (_i.e._ tag positions) where the
localization performance will be evaluated, as well as the number of tries for
each of these positions.  
  From now on, the anchor positions are fixed. However, the measurements will
have to include some randomness and thus will be re-generated for each try at
each of the above mentioned positions of the grid.


#### BLOCK 2:  Measurement generation & geolocalization
  This block is called at each try and each position of the above mentioned grid.  
  Depending on the technology, it will generate different kinds of measurements.  
  Thus, the localization algorithm to be called afterwards will depend on the
measurement type.

  For now, **only the UWB-TWR distance measurements** are supported.
A multilateration algorithm will then be called to compute the location of the
device.

A noise model and parameters have to be chosen. Indeed, one might want to
evaluate the robustness of the localization algorithm under varying noise
conditions. For example:
* Gaussian noise with expectancy $\mu$ and standard deviation $\sigma$
* Gaussian noise with parameters depending on the tag-anchor distance
* Fading/shadowing which can modelize temporary obstruction of the physical
channel

The measurements are then generated depending on an _a priori_ anchor
selection procedure which takes into account:
* the ranging distance
* the number of anchors to be selected
* the geometry of the anchors' disposition

Finally, the geolocalization algorithm is executed using the set of selected
anchor positions and measurements.

This block saves the results of all the localizations in a file as a JSON
file, and includes all the relevant information for the statistics for each
recording:
- initial position
- real tag position
- inferred tag position
- anchor selection method
- localization method: minimization cost function + precision / LS / NLS
- noise model
- selected anchors and ranging measurements

The file can get pretty big (several tens of megabytes) but saving it will
save a lot of time if visualizing the results or comparing the results of
several simulations is needed later on.


#### BLOCK 3: Visualization and statistics

This block simply reads the file generated by the second block and produces
readable textual statistics and plots graphs to analyse the performance of the
different localization methods.
* Summary of the simulation parameters
* Mean squared error / Mean absolute error
* Median squared error / Median absolute error
* Cumulative Distribution Function
* Maximum and minimum error
* Mean tag-anchor distances
* Histogram of z positions (useful to observe some phenomenons w.r.t. 
 


## Usage

### Description of the data structures

The simulator requires Python 3.5 or above as it incorporates type hinting,
_i.e._ the arguments and return types of functions are explicitly specified. 

In addition to that, all functions are documented so that their action,
arguments and return objects are fully described and leave no ambiguity.

It was also decided to use dictionaries as much as possible. Dictionaries
(Python type: `dict`) are associative arrays, _i.e._ key-indexed unordered
collections. All anchor sets are stored as dictionaries. These latter are
passed as function arguments by reference, which means that they can be
modified by the function itself. It is useful as it means that we can
keep the same dict of anchors and include the distances in the original
dictionary and replace a single value every time a new ranging is done without
creating a new dictionary.

An anchor set is a `dict` or `dict`. Here is an example with `n` anchors:
```
anchors = {
  ID_A1: { 'x': x_A1,
           'y': y_A1,
           'z': z_A1 },
  ID_A2: { 'x': x_A2,
           'y': y_A2,
           'z': z_A2 },
  ...
  ID_An: { 'x': x_An,
           'y': y_An,
           'z': z_An }
}
```



Hence, when anchors is passed as an argument and another dictionary is returned
by the anchor selection procedure function, the "inner" dictionaries are


It also means that anchor selection procedures should return the **IDs**
for the anchors to be selected, not a new dictionary of anchors, in order
to save memory.

### Imports and usage

The three modules `genenv`, `genmeas` and `genstats` must be imported
in the main script.

There are several ways to create an localization environment i.e. a set of
anchors to be fed as an input to the next blocks. It can be done manually by
creating a dictionary following the above described structure. However, the
`genenv` module provides three ways of doing so. 


An example script is provided in `example1.py` to show the order in which
to call the functions from the different modules.
 