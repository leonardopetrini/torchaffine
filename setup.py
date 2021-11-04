"""
Installation script
"""
from setuptools import setup, find_packages

setup(
    name='torchaffine',
    packages=find_packages(exclude=["build"])
)