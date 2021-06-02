# To install the package run 'pip3 install -e .'
from setuptools import setup

setup(
    name="simp",
    version="0.0.1",
    packages=['nofever', 'face_generator', 'simp', 'tests'],
    package_dir={'nofever': 'simp/submodules/nofever/nofever', 'face_generator':'simp/submodules/face_generator/face_generator', 'simp':'simp', 'tests':'simp/tests'},
    author="Really A Robot",
    author_email="jga@reallyarobot.com",
    description="Master Thesis project built by Really A Robot",
    licence='',
    # url="https://github.com/eugenegalaxy/NoFever",
    python_requires='>=3.6',
)
