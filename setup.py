from setuptools import find_packages, setup

NAME = 'laminar_tools'

VERSION = '0.0.1a'
GENERAL_REQUIRES = ['numpy', 'scipy', 'matplotlib']

EXTRAS_REQUIRES = {
}

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=True,
    author='LBHB',
    author_email='lbhb.ohsu@gmail.com',
    description='Cortical laminar analysis tools',
    url='http://hearingbrain.org',
    install_requires=GENERAL_REQUIRES,
    extras_require=EXTRAS_REQUIRES,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
)
