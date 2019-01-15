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

with open("./README.md", "r") as fh:
    long_description = fh.read()

def install_requires(fname="./requirements.txt"):
  with open(fname) as f:
      content = f.readlines()
  content = [x.strip() for x in content]
  return content

setup(name='PySiTK',
      version='0.2.6',
      description='Python SimpleITK/WrapITK helper scripts',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/gift-surg/PySiTK',
      author='Michael Ebner',
      author_email='michael.ebner.14@ucl.ac.uk',
      license='BSD-3-Clause',
      packages=['pysitk'],
      install_requires=install_requires(),
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
