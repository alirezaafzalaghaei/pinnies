from setuptools import setup, find_packages

setup(
    name="pinnies",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'tqdm',
        'scipy',
        'numpy'
    ],
    description="A Physics-Informed Neural Network Framework for Solving Integral Equations",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Alireza Afzal Aghaei",
    author_email="alirezaafzalaghaei@gmail.com",
    url="https://github.com/alirezaafzalaghaei/pinnies",
    license="BSD",
)
