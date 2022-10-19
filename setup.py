"""
This script is used to install the package and all its dependencies. Run

    python -m pip install .

to install the package.
"""

from setuptools import setup
from ffflows.version import VERSION

setup(
    name='ffflows',
    version=VERSION,
    description="Flows for flows in PyTorch",
    author="Sam Klein, John Raine",
    license="MIT",
    install_requires=[
        "numpy",
        "torch",
        "nflows"
    ],
    dependency_links=[],
)
