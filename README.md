# Echelle

## Installation
`pip install echelle`
or download the git repository and 
`python setup.py install`

## Usage
To plot the echelle diagram,
```
from echelle import plot_echelle
plot_echelle(frequency, power, dnu)
```

For a dynamic Bokeh interface:
```
from echelle import interact_echelle
interact_echelle(frequency, power, dnu)
```
