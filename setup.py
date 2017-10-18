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
# \see https://python-packaging.readthedocs.io/en/latest/minimal.html
# \see https://python-packaging-user-guide.readthedocs.io/tutorials/distributing-packages/


from setuptools import setup

long_description = "This package contains scripts to facilitate IO, " \
    "printing, plotting, ... and the interaction between SimpleITK and ITK"

setup(name='PySiTK',
      version='0.1.dev1',
      description='Python helper scripts',
      long_description=long_description,
      url='https://cmiclab.cs.ucl.ac.uk/gift-surg/PySiTK',
      author='Michael Ebner',
      author_email='michael.ebner.14@ucl.ac.uk',
      license='MIT',
      packages=['pysitk'],
      install_requires=[
          "SimpleITK",
          "numpy",
          "nibabel",
          "matplotlib",
          "Pillow",
          "itk",
      ],
      zip_safe=False,
      keywords='development ITK SimpleITK',
      classifiers=[
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Developers',
          'Intended Audience :: Healthcare Industry',
          'Intended Audience :: Science/Research',

          'License :: OSI Approved :: MIT License',

          'Topic :: Software Development :: Build Tools',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',

          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
      ],
      )
