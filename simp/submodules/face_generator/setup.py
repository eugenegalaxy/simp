# To install the package run 'pip3 install -e .'
from setuptools import setup

setup(
    name="face_generator",
    version="0.0.1",
    packages=['models', 'configs', ],
    package_dir={'': 'face_generator', 'models': 'face_generator/models', 'configs': 'face_generator/configs'},
    author="Really A Robot",
    author_email="jga@reallyarobot.com",
    description="Image inpainter rebuilt for NoFever purposes",
    licence='',
    # url="https://github.com/eugenegalaxy/NoFever",
    python_requires='>=3.6',
)
