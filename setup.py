from setuptools import find_packages, setup


setup(
    name='positron',
    packages=find_packages(include=['positron']),
    version='0.1.1',
    description='Blazingly fast deep learning library for Python',
    author='Martin Kondor (https://martinkondor.github.io)',
    license='MIT',
    install_requires=['numpy', 'pandas'],
    setup_requires=[],
    tests_require=[],
    test_suite='tests'
)
