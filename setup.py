from setuptools import setup
from setuptools import find_packages
import pip
try:
    from pip import main as pipmain
except:
    from pip._internal import main as pipmain

setup(name='numpy_neuro_network',
      version='0.0.1',
      description='Numpy Neural Network',
      author='Liang Yuan Cheng',
      author_email='woodlica@gmail.com',
      url='https://github.com/loui0620',
      license='Apache',
      packages=find_packages())

	
pipmain(['install', 'numpy'])
pipmain(['install', 'matplotlib'])
pipmain(['install', 'pandas'])