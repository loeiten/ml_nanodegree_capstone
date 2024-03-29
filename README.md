# Stock price predictor

[![Build Status](https://github.com/loeiten/ml_nanodegree_capstone/workflows/Test/badge.svg?branch=main)](https://github.com/loeiten/ml_nanodegree_capstone/actions?query=workflow%3A%22Test%22)
[![codecov](https://codecov.io/gh/loeiten/ml_nanodegree_capstone/branch/master/graph/badge.svg)](https://codecov.io/gh/loeiten/ml_nanodegree_capstone)
[![PEP8](https://img.shields.io/badge/code%20style-PEP8-brightgreen.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](https://github.com/loeiten/ml_nanodegree_capstone/blob/master/LICENSE)

Capstone project for Udacity's Machine Learning Nanodegree.

The assignment text can be found in 
[investment_and_trading_capstone_project.md](proposal/investment_and_trading_capstone_project.md),
the proposal in 
[capstone_proposal.pdf](capstone_proposal.pdf)
and the final report in
[capstone_report.pdf](capstone_report.pdf) (note this is best rendered in the
 markdown format [here](report/capstone_report.md))

![alt text](images/optimal_knn.png "Optimal prediction of the kNN estimator")

* [`data/`](data) - `.csv` files of stocks used in this project together with 
scripts used to obtain the files
* [`data_preparation/`](data_preparation) - Routines for reading the data
* [`estimators/`](estimators) - Custom made estimators
* [`images/`](images) - Saved plots
* [`notebooks/`](notebooks) - Notebooks where the analysis is performed
* [`proposal/`](proposal) - Files related to the capstone proposal
* [`report/`](report) - Files for the report
* [`tests/`](tests) - Unittests
* [`utils/`](utils) - Utilities for transformations, scoring and visualization