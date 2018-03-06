# Udacity
## Assignment
https://docs.google.com/document/d/1ycGeb1QYKATG6jvz74SAMqxrlek9Ed4RYrzWNhWS-0Q/pub
## Proposal rubric
https://review.udacity.com/#!/rubrics/410/view
## Templates and examples
https://github.com/udacity/machine-learning/tree/master/projects/capstone




# Resources
## Machine learning mastery
### Time series forecasting (more theory)
https://machinelearningmastery.com/time-series-forecasting-supervised-learning/
### Using pandas to do the job
https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
### Check for stationary
https://machinelearningmastery.com/time-series-data-stationary-python/


## Medium (not the best example of financial trading)
### NLP, CNN and RNN
https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-part-one-simple-time-series-forecasting-f992daa1045a
### Tools against overfitting
https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-1-2-correct-time-series-forecasting-backtesting-9776bfd9e589
### Multivariate
https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-2-1-multivariate-time-series-ab016ce70f57
### In general
https://medium.com/machine-learning-world/tagged/finance


## Slides
### ML strategies for Time Series prediction
http://www.ulb.ac.be/di/map/gbonte/ftp/time_ser.pdf
#### Notes
##### Stationarity
* Predicting a time series is possible if and only if the dependence between values existing in the past is preserved also in the future.
* In other terms, though measures change, the stochastic rule underlying their realization does not. This aspect is formalized by the notion of stationarity.
##### Random walk
* Random walk: As the mean and variance change with t the process is non-stationary
* The first differences of a random walk form a purely random process, which is stationary
* Examples of time series which behave like random walks are stock prices on successive days
##### NAR
* AR (like Markov process) models assume that the relation between past and future is linear



## Papers
### Machine data for sequential data: A review
http://web.engr.oregonstate.edu/~tgd/publications/mlsd-ssspr.pdf




# Stationarity
https://quant.stackexchange.com/questions/9192/how-to-normalize-stock-data
https://en.wikipedia.org/wiki/Stationary_process
https://quant.stackexchange.com/questions/2372/how-to-check-if-a-timeseries-is-stationary/2373
https://quant.stackexchange.com/questions/8875/why-non-stationary-data-cannot-be-analyzed



# Notes
## Trading course notes
* Pandas inner frame is numpy
* Numpy functions building on c
* Index must be the same when joining, can join inner
