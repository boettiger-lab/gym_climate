# gym_climate
WIP :construction:

## Installation
Run `pip install -e .` from the home directory to have `gym_climate` in your path. And run `pip install -r requirements.txt` to get the necessary :snake: packages.

## Environments

Currently, Nordhaus' DICE model has been implemented in `dice-v0`. 

## References

My current implementation is based off [pyDICE](https://github.com/hazem2410/PyDICE/blob/master/DICE2016.py), which follows the [original GAMS implementation](http://www.econ.yale.edu/~nordhaus/homepage/homepage/DICE2016R-091916ap.gms). Within pyDICE, there is a [concise equation guide](https://github.com/hazem2410/PyDICE/blob/master/PyDICE2016.pdf) to the DICE model. Nordhaus has a [longer, more detailed user guide](http://www.econ.yale.edu/~nordhaus/homepage/homepage/documents/DICE_Manual_100413r1.pdf). 
