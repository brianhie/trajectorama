from setuptools import setup, find_packages

setup(
    name='trajectorama',
    version='0.1',
    description='Single-cell trajectory integration',
    url='https://github.com/brianhie/trajectorama',
    download_url='https://github.com/brianhie/geosketch/archive/v0.1-beta.tar.gz',
    packages=find_packages(exclude=['bin', 'conf', 'data', 'target',]),
    install_requires=[
        'anndata>=0.6.22',
        'joblib>=0.13.2',
        'geosketch>=1.0',
        'numpy>=1.12.0',
        'scanpy>=1.4.4',
        'scikit-learn>=0.20rc1',
        'scipy>=1.0.0',
    ],
    author='Brian Hie',
    author_email='brianhie@mit.edu',
    license='MIT'
)
