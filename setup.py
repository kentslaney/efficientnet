from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "virtualenv>=20.0",
    "tensorflow>=2.3",
    "tensorflow-addons>=0.11",
    "tensorflow-datasets>=3.2.1",
    "tensorflow-probability>=0.11.0",
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='train models'
)
