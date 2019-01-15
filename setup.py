###
# \file setup.py
#
# Install with symlink: 'pip install -e .'
# Changes to the source file will be immediately available to other users
# of the package
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='PySiTK',
      version='0.2.2',
      description='Python SimpleITK/WrapITK helper scripts',
      long_description=long_description,
      url='https://github.com/gift-surg/PySiTK',
      author='Michael Ebner',
      author_email='michael.ebner.14@ucl.ac.uk',
      license='BSD-3-Clause',
      packages=['pysitk'],
      install_requires=[
          "pip>=9.0.1",
          "setuptools>=36.6.0",
          "nibabel>=2.0.2",
          "matplotlib>=1.4.3",
          "numpy>=1.13.1",
          "cmake>=0.8.0",
          "ninja>=1.7.2",
          "SimpleITK>=1.1.0",
          "scikit_image>=0.13.1",
          "nose>=1.3.7",
          "pandas>=0.22",
          "seaborn>=0.8.1",
      ],
      zip_safe=False,
      keywords='development ITK SimpleITK',
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Healthcare Industry',
          'Intended Audience :: Science/Research',

          'License :: OSI Approved :: BSD License',

          'Topic :: Software Development :: Build Tools',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',

          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
      ],
      )
