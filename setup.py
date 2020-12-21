from setuptools import setup

setup(name='laserhockeyenv',
      version='1.0',
      description='Simple Hockey Environments',
      url='https://github.com/martius-lab',
      author='Georg Martius, MPI-IS Tuebingen, Autonomous Learning',
      author_email='georg.martius@tuebingen.mpg.de',
      license='MIT',
      packages=['laserhockeyenv'],
      python_requires='>=3.6',
      install_requires=['gym', 'numpy', 'box2d', 'box2d-kengz'],
      zip_safe=False)
