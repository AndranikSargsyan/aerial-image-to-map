import os
from setuptools import setup

package_dir = os.path.dirname(os.path.join(__file__))

requires = []
with open(os.path.join(package_dir, 'requirements.txt')) as f:
    for item in f.read().splitlines():
        if not item.strip().lower().startswith('--'):
            requires.append(item)

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
    url='https://github.com/AndranikSargsyan/qartezator',
    packages=['qartezator'],
    install_requires=requires
)
