from setuptools import find_packages, setup

# read the contents of README file
from os import path
from io import open  # for Python 2 and 3 compatibility

this_directory = path.abspath(path.dirname(__file__))


# read the contents of README.md
def readme():
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        return f.read()
    
    
# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()
    
VERSION = "0.0.2"

setup(  name='PerturbNet',
      version=VERSION,
      license='GPL-3.0',
      description='PerturbNet',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/welch-lab/PerturbNet',
      author='Hengshi Yu, Weizhou Qian, Yuxuan Song, Joshua Welch',
      packages=find_packages(exclude=['test']),
      zip_safe=False,
      include_package_data=True,
      install_requires=requirements,
      python_requires=">=3.7,<3.8"
      #setup_requires=['setuptools>=38.6.0']
      )