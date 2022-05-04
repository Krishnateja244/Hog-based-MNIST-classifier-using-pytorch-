#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='Hog-based-MNIST-classifier-using-pytorch',
    version='0.0.0',
    description='This Project classifies MNIST dataset consisting \
    of handwritten digits between 0-9 using Histogram of Oriented Gradients(HOG) features. Pytorch is used for building this classifier. ',
    author='Krishna Teja Nallanukala',
    author_email='krishnatejanallanukala@gmail.com',
    url='https://github.com/Krishnateja244/Hog-based-MNIST-classifier-using-pytorch-.git',
    install_requires=['torch'],
    packages=find_packages(),
)

