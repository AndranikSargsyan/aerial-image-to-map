from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='qartezator',
    version='0.0.1',
    description='Qartezator is a translator between aerial images and maps.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='aerial-image-to-map',
    url='https://github.com/AndranikSargsyan/qartezator'
)