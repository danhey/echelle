---
title: 'Echelle: A Python package for dynamic echelle diagrams'
tags:
  - Python
  - astronomy
  - asteroseismology
authors:
  - name: Daniel R. Hey
    orcid: 
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Timothy R. Bedding
    affiliation: "1, 2"
  - name: Yaguang Li
    affiliation: "1, 2"
affiliations:
 - name: Institution Name
   index: 1
 - name: Institution Name
   index: 2
date: 20 May 2021
bibliography: paper.bib

---

# Summary

Asteroseismology is the study of oscillations in stars. One of the most important diagnostics is the frequency separation between modes of equal spherical degree ($\ell$). This separation, written as $\Delta\nu$, is approximately proportional to the mean density of the star (Ulrich 1986; https://ui.adsabs.harvard.edu/abs/1986ApJ...306L..37U/abstract).

A common method of diagnosing $\Delta\nu$ involves creating an echelle diagram, where the power spectrum of the star is cut into equal slices of $\Delta\nu$ and 
stacked atop eachother. Typically, $\Delta\nu$ is then finely adjusted until the ridges align.

The echelle diagram was introduced by Grec et al. (1983; https://ui.adsabs.harvard.edu/abs/1983SoPh...82...55G/abstract) for displaying the oscillation frequencies in the Sun, and has become an indispensable tool for helio- and asteroseismology. Its usefulness stems from the fact that p-mode oscillation modes are approximately equally spaced in frequency, with a characteristic spacing of $\Delta\nu$. By dividing the power spectrum into segments of equal length and stacking them vertically, these regularities become more visible. Modes with the same angular degree ($\ell$) align on vertical ridges, making it is easy to identify modes and to follow any deviations from regularity.  To allow fine adjustments, we have developed code that allows the echelle spacing to be changed interactively. This makes it straightforward to choose the best value of $\Delta\nu$ by adjusting the spacing until the ridges are as close to vertical as possible (e.g., Bedding et al 2020; https://ui.adsabs.harvard.edu/abs/2020Natur.581..147B/abstract).

Echelle diagrams can also be used to display gravity modes, which are approximately evenly spaced in period. An example for a red giant star displaying mixed dipolar modes is show in figure 2b of Murphy et al. (2021, https://ui.adsabs.harvard.edu/abs/2021MNRAS.505.2336M/abstract).

Tim?

# Statement of need

Tim?

# The _echelle_ code

For Dan to do.

# Documentation & case studies

For Dan to do.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References