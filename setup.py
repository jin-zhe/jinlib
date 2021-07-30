from setuptools import setup, find_packages

setup(
  name='jinlib',
  version='0.0.1',
  description='Personal convenience library',
  url='git@github.com:jin-zhe/jinlib.git',
  author='JIN Zhe',
  author_email='thejinzhe@gmail.com',
  classifiers=[
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
  ],
  license='unlicense',
  packages=find_packages(),
  zip_safe=False
)
