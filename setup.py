from setuptools import setup, find_packages

setup(
    name='channest',
    version=open('channest/VERSION.txt').read(),
    description='Package',
    author='Norwegian Computing Center',
    packages=find_packages(),
    include_package_data=True,
)
