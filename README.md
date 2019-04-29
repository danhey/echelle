# Echelle

## Installation
`pip install echelle`
or download the git repository and 
`python setup.py install`

## Usage

For a dynamic interface:
```
from echelle import interact_echelle
interact_echelle(frequency, power, dnu_min, dnu_max)
```

If you're using in a Jupyter notebook, I suggest:
```
interact_echelle(frequency, power, dnu_min, dnu_max, notebook=True)
```

To plot a non-interactive echelle diagram,
```
from echelle import plot_echelle
plot_echelle(frequency, power, dnu)
```

