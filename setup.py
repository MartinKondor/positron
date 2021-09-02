from setuptools import find_packages, setup


setup(
    name='positron',
    packages=find_packages(include=['positron']),
    version='0.1.0',
    description='A blazingly fast ML library for Python',
    author='Martin Kondor (https://martinkondor.github.io)',
    license='MIT',
    install_requires=['numpy'],
    setup_requires=[],
    tests_require=[],
    test_suite='tests'
)
