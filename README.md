# Echelle

<!-- ![](docs/echelle_plot.png) -->
Echelle is a Python package for plotting and interacting with echelle diagrams.
In an echelle diagram, the amplitude spectrum of a star is stacked in equal
slices of delta nu, the large separation. 

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

If you're using echelle in a Jupyter notebook, I suggest calling `%matplotlib notebook` first.

To plot a non-interactive echelle diagram,
```
from echelle import plot_echelle
plot_echelle(frequency, power, dnu)
```

See the example.ipynb for common usage!

## Citing

If you make use of echelle in your work, please consider citing the Zenodo listing
```
@misc{daniel_2019_3403407,
  author       = {Daniel Hey},
  title        = {danhey/echelle: Initial release},
  month        = sep,
  year         = 2019,
  doi          = {10.5281/zenodo.3403407},
  url          = {https://doi.org/10.5281/zenodo.3403407}
}
```
