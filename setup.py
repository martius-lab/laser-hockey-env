from setuptools import setup

setup(name='laserhockey',
      version='2.0',
      description='Simple Hockey Environments',
      url='https://github.com/martius-lab',
      author='Georg Martius, MPI-IS Tuebingen, Autonomous Learning',
      author_email='georg.martius@tuebingen.mpg.de',
      license='MIT',
      packages=['laserhockey'],
      python_requires='>=3.6',
      install_requires=['gymnasium', 'numpy', 'box2d-py','pygame'],
      zip_safe=False)
