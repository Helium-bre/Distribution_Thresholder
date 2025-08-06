from setuptools import setup, find_packages

setup(
    name='Distribution_Thresholder',
    version='1.0',
    author='Hugo HE',
    description='Distribution-based thresholding approach for classification of anomaly score',
    packages=find_packages(),
    install_requires=['numpy>=1.26.4','scikit-learn>=1.6.1','scipy>=1.15.3']
)