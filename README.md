# ⚡ positron

[![Project Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/MartinKondor/positron/)
[![version](https://img.shields.io/badge/version-v0.9-red.svg)](https://github.com/MartinKondor/positron)
[![GitHub Issues](https://img.shields.io/github/issues/MartinKondor/positron.svg)](https://github.com/MartinKondor/positron/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)
[![License](https://img.shields.io/badge/license-BSD-brightgreen.svg)](https://opensource.org/licenses/BSD)

Blazingly fast _functional_ deep learning library for Python.

The main focus of Positron is to implement functions that are common (or rare) in Linear Algebra, but are not in Numpy by default.

_There is a [Medium Article](https://martinkondor.medium.com/positron-linear-algebra-library-for-python-8a3c5c3e1c00) written about this library, make sure to check it out!_

## Features

| File      | Description |
| --------- | ----------- |
| example.py      | A simple neural network adaptation working with this library.       |
| deep.py      | Deep Learning learning related functions: feedforward, backprop etc.       |
| activ.py      | Common Deep Learning activation functions.       |
| maths.py      | Matrix operations: inverse, determinant, adjungate etc.       |
| prep.py      | Data preprocessing: time stamp to date, date to time stamp etc.        |
| score.py     | Scoring functions and their derivatives: residual sum of squares, mean absolute error etc.        |

## Tests

To run tests on a file, run the file directly, for example:

```$ python positron/math.py```

## Authors

* **[Martin Kondor](https://github.com/MartinKondor)**

## License 

Copyright &copy; Martin Kondor 2022

See [LICENSE](./LICENSE) file for more information.
