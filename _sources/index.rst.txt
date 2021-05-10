======================
echelle
======================

*echelle* is a simple Python package for creating and manipulating
echelle diagrams. Echelle diagrams are used mainly in asteroseismology where
they function as a diagnostic tool for estimating :math:`\Delta\nu`, the separation
between modes of the same degree :math:`\ell`.

In an echelle diagram, the power spectrum of the star is cut into equal chunks
of the large separation, :math:`\Delta\nu`, and stacked on top of eachother. For a correct value of 
:math:`\Delta\nu`, modes of the same degree will appear as a line.

.. figure:: example_echelle.gif
   :align: center


*echelle* provides the ability to dynamically change :math:`\Delta\nu`. This allows
for the rapid identification of the correct value. *echelle* features:

* Performance optimized dynamic echelle diagrams.
* Multiple backends for supporting Jupyter or terminal usage.

*echelle* is being actively developed in `a public repository on GitHub
<https://github.com/danhey/echelle>`_. Any contribution is welcome! If you 
have any trouble, `open an issue <https://github.com/danhey/echelle/issues>`_ .


Quickstart
==========

1. Install this package::

    pip install echelle

2. Calculate the power spectrum of your star::

    import echelle
    freq, power = echelle.power_spectrum(time, flux)

3. Make an interactive echelle diagram in your Jupyter notebook!::

    echelle.interact_echelle(freq, power, dnu_min, dnu_max)

User guide
==========

.. toctree::
   :maxdepth: 2

   user/install
   user/citation

Tutorials
==========

.. toctree::
   :maxdepth: 2

   notebooks/quickstart.ipynb
   notebooks/backend.ipynb

API
==========

.. toctree::
   :maxdepth: 2

   user/api
    

License & attribution
---------------------

Copyright 2019, 2020, 2021 Daniel Hey

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite this package and its dependencies. You
can find more information about how and what to cite in the `citation
<user/citation>`_ documentation.

These docs were made using `Sphinx <https://www.sphinx-doc.org>`_ and the
`Typlog theme <https://github.com/typlog/sphinx-typlog-theme>`_. They are built
and hosted on `Read the Docs <https://readthedocs.org>`_.

Changelog
---------

.. include:: ../CHANGES.rst