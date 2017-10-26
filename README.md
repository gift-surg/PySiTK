# PySiTK 

PySiTK is a Python-based open-source toolkit that provides a collection of helper functions to facilitate IO, printing, plotting, ... and the interaction between [SimpleITK][simpleitk] and [ITK_NiftyMIC][itkniftymic].
Please not that currently **only Python 2** is supported.

The algorithm and software were developed by [Michael Ebner][mebner] at the [Translational Imaging Group][tig] in the [Centre for Medical Image Computing][cmic] at [University College London (UCL)][ucl].

If you have any questions or comments (or find bugs), please drop me an email to `michael.ebner.14@ucl.ac.uk`.

## Installation

Installation of the external dependencies:
* [ITK_NiftyMIC][itkniftymic]

Clone the PySiTK repository by
* `git clone git@cmiclab.cs.ucl.ac.uk:GIFT-Surg/SimpleReg.git` 

Install all Python-dependencies by 
* `pip install -r requirements.txt`

Install PySiTK by running
* `pip install -e .`


## Licensing and Copyright
Copyright (c) 2017, [University College London][ucl].
This framework is made available as free open-source software under the [BSD-3-Clause License][bsd]. Other licenses may apply for dependencies.

[citation]: https://www.sciencedirect.com/science/article/pii/S1053811917308042
[mebner]: http://cmictig.cs.ucl.ac.uk/people/phd-students/michael-ebner
[tig]: http://cmictig.cs.ucl.ac.uk
[bsd]: https://opensource.org/licenses/BSD-3-Clause
[giftsurg]: http://www.gift-surg.ac.uk
[cmic]: http://cmic.cs.ucl.ac.uk
[guarantors]: https://guarantorsofbrain.org/
[ucl]: http://www.ucl.ac.uk
[simpleitk]: http://www.simpleitk.org/
[wrapitk]: https://itk.org/Wiki/ITK/WrapITK_Status
[itkniftymic]: https://cmiclab.cs.ucl.ac.uk/GIFT-Surg/ITK_NiftyMIC/wikis/home