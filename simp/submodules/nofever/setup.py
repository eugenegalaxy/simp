# To install the package run 'pip3 install -e .'
from setuptools import setup

setup(
    name="nofever",
    version="0.0.2",
    packages=['detection', 'linak', ],
    package_dir={'': 'nofever'},
    author="Really A Robot",
    author_email="jga@reallyarobot.com",
    description="A temperature measurement device to fight COVID-19",
    licence='',
    url="https://github.com/eugenegalaxy/NoFever",
    python_requires='>=3.6',
)
