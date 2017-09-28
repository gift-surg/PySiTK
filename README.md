# PythonHelper 

This software package contains a collection of helper functions to facilitate IO, printing, plotting, ... and the interaction with [SimpleITK](http://www.simpleitk.org/) and [ITK](https://itk.org/).

It was developed in support of various research-focused toolkits within the [GIFT-Surg](http://www.gift-surg.ac.uk/) project.

If you have any questions or comments (or find bugs), please drop me an email to `michael.ebner.14@ucl.ac.uk`.

## Installation

Required dependencies can be installed using `pip` by running
* `pip install -r requirements.txt`
* `pip install -e .`

In addition, you will need to install `itk` for Python. In case you want to make use of the [Volumetric MRI Reconstruction from Motion Corrupted 2D Slices](https://cmiclab.cs.ucl.ac.uk/mebner/VolumetricReconstruction) tool or any of its dependencies, please install the ITK version as described there. Otherwise, simply run
* `pip install itk`


## License
This framework is licensed under the [MIT license ![MIT](https://raw.githubusercontent.com/legacy-icons/license-icons/master/dist/32x32/mit.png)](http://opensource.org/licenses/MIT)
