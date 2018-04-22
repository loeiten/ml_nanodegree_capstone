# Machine Learning Engineer Nanodegree
## Capstone Project
April 22nd, 2018

Michael Løiten


## I. Definition

### Project Overview

In this project we try to predict the closing value (adjusted for 
[stock split](https://classroom.udacity.com/courses/ud501/lessons/4442578629/concepts/45792868240923)
and
[dividends](https://classroom.udacity.com/courses/ud501/lessons/4442578629/concepts/44273516340923))
based on adjusted closing values in the past.

> **Disclaimer:** In this project we do not seek to make the best possible 
prediction, rather, it is an exploratory exercise in how machine learning can
 be used for predicting the future based on information about the past. 

Predicting the closing value of stocks is interesting for several reasons:

1. The time series are sequential, so techniques in machine learning like 
randomization of the data and cross-validation can not be used in a straight 
forward manner.
2. Stocks are notoriously [non](https://www.investopedia.com/articles/trading/07/stationary.asp)
[-stationary](https://en.wikipedia.org/wiki/Stationary_process).
This means that both the measures changes with time, and the stochastic rule 
underlying their realization. 
The consequence is that traditional statistical techniques must be used with
care when analysing the data.
This also means that we need to take care when scaling the data, as 
several techniques uses the mean and standard deviation to scale the numbers.
3. Although we are predicting the stock market here, the *techniques* for 
forecasting can be applied to any forecasting problem, like forecasting 
weather, sales, populations etc. 
Note though, that the *findings* about the models and hyper parameters may not 
be directly transferable to a different forecasting problem.
4. The [efficient-market hypothesis](https://en.wikipedia.org/wiki/Efficient-market_hypothesis), 
or simply [EMH](https://www.investopedia.com/terms/e/efficientmarkethypothesis.asp),
states that stocks are already properly priced and reflects all available 
information.
Effectively, this means that "you can't beat the market".
More specifically it states that you cannot make any profit from any trading 
strategies.
Although there are several forms of this statement, even the weak form states 
that "Future prices cannot be predicted by analyzing prices from the past", 
and that
"[Technical analysis](https://en.wikipedia.org/wiki/Technical_analysis) 
techniques will not be able to consistently produce 
excess returns, though some forms of 
[fundamental analysis](https://en.wikipedia.org/wiki/Fundamental_analysis)
may still provide excess returns."
Stated differently, if we name our random variable $\phi$, the 
EMH states that $p(\phi_{t+k} | \phi_{t}) = p(\phi_{t+k}) \quad \forall k$ and
that $\operatorname{Cov}(\phi_{t+k} | \phi_{t})=0$, i.e. that all samples in 
the time series are independent (see also [these slides](http://www.ulb.ac.be/di/map/gbonte/ftp/time_ser.pdf)
for a nice introduction to the topic).
A weaker statement would be that stock prices act like a 
[random-walk](https://machinelearningmastery.com/gentle-introduction-random-walk-times-series-forecasting-python/).
In that case, the next time step is dependent on the previous time step which
gives the times series some degree of predictability.

The idea of using machine learning for trading is far from new, and both 

have been studying the topic.
There are several sources out there which have used similar approaches to 
predict the stock prices using machine learning.
There are papers by
[academics](https://www.sciencedirect.com/science/article/pii/S0957417405003015) 
and
[students](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.83.5299&rep=rep1&type=pdf);
[various](https://lilianweng.github.io/lil-log/2017/07/08/predict-stock-prices-using-RNN-part-1.html)
[blogs](https://medium.com/machine-learning-world/tagged/finance);
[kaggle notebooks](https://www.kaggle.com/pablocastilla/predict-stock-prices-with-lstm);
[previous](https://github.com/jessicayung/machine-learning-nd/tree/master/p5-capstone)
ML nanodegree
[graduates](https://github.com/Rajat-dhyani/Stock-Price-Predictor);
even vlogs describing how to use
[support vector regression](https://www.youtube.com/watch?v=SSu00IRRraY),
[lstm](https://www.youtube.com/watch?v=ftMq5ps503w&feature=youtu.be)
and
[sentiment analysis](https://www.youtube.com/watch?v=JuLCL3wCEAk)
for stock prediction.

Daily stock data has until recently been readily and freely available.
At the time of writing, the situation is somewhat changed, after Yahoo! Finance 
deprecated their APIs in late 2017, several of the distributors which 
provided daily stock data for free has deprecated their APIs. Latest Quandl 
stated

> As of April 11, 2018 this data feed is no longer actively supported by the 
Quandl community. We will continue to host this data feed on Quandl, but we do 
not recommend using it for investment or analysis.

Therefore, the data used in this project has been stored in the `data/` 
directory of the repository. More information about the origin of the 
downloaded data can be found under `proposal/capstone_proposal` under the 
section *Datasets and Inputs*.

In this project we have used the daily stock data containing information 
about opening price, highest traded price, lowest traded price, closing price, 
volume sold and adjusted closing value (ohlcva) for the 50 stocks with the 
highest weights in the S&P500 portfolio together with ^GSPC itself as of 6th
of March 2018.


### Problem Statement

In this project we would like to build estimators based on the k-nearest 
neighbors (kNN) and long short term memory (LSTM) algorithms and see if they 
can outperform the simple predictions made by latest day, random gaussian and
linear regression algorithms in the task of predicting the adjusted closing 
price of the stocks.
We will assume that what matters the most is the estimators ability to 
estimate the closing value 7, 14 and 28 days ahead, and that each prediction 
is of equal importance.
The prediction will be done in an rolling matter for all models except the 
LSTM models. That is, we

1. Fit the data up until day $x$
2. Do the predictions of day $y_1$, $y_2$ and $y_3$, at day $x$
3. Add the true data of day $x+1$, refit the models and predict for day 
$y_1+1$, $y_2+1$ and $y_3+1$
4. Add the true data of day $x+2$, refit the models and predict for day 
$y_1+2$, $y_2+2$ and $y_3+2$, and so on

We will also assume that the training time of the models is crucial, which is
why we are content with a normal prediction for the LSTM models. I.e., we

1. Fit the data on the training set
2. Make predictions of $y_1$, $y_2$ and $y_3$ on the test set.

> **Note:** We will fit one model (with the same hyper parameters) for each 
stock we are predicting for. I.e. we will not use model fitted on stock A to 
make predictions on stock B.

We will use the ochlva data Standard and Poor's 500 (S&P500) portfolio as of 
2018-03-06.

The number of ways to investigate and solve this problem is enormous, so in 
order to limit the scope we:

* Only look at four stocks: The `^GSPC`, `AAPL` (which had the highest weight
in the S&P 500 portfolio), `CMCSA` (which was the 25th highest weighted stock
in the S&P 500 portfolio) and `GILD` (which was the 50th highest weighted stock
in the S&P 500 portfolio)
* Only use the data for one stock at the time
* Only use the data from the adjusted close

That being said, it would be very interesting to include more of the ochlv 
data and even the information about other stocks in the prediction after 
doing a proper feature analysis (like looking at correlations, doing a PCA 
analysis etc.).

The strategy to solve the problem can be outlined as follows 
(for a more detailed description, see section
[III. Methodolody](#iii.-methodology)):

1. Investigate the predictive capability of the "simple models" 
(`latest_day`, `random_gaussian` and `linear_regression`). 
The notebooks for investigation can be found in `notebooks/1.*.ipynb`.
2. Investigate the predictive capability of the "advanced models" 
(`knn` and `lstm`). 
The notebooks for investigation can be found in `notebooks/2.*.ipynb`.
3. Tune the features and hyper parameters for the "advanced models" one by one.
Note that this could be done by performing a 
[grid search](http://scikit-learn.org/stable/modules/grid_search.html).
However, as we are interested in the trends, and since we are using a 
non-standard metric (see the [metrics](#metrics) section), we will search the 
parameters one by one, well aware of the fact that there may be combinations 
of the hyper parameters that potentially could give better predictions.
The notebooks for investigation can be found in `notebooks/3.*.ipynb`.

For each analysis we will for each stock investigated:

1. Read the data
2. Clean the data
3. Extract the adjusted close feature
4. Create the targets from the adjusted close, by shifting them by $t$ days 
towards the future 
(note that the $t$ latest observation would be without a target value)
5. (Optional) Make more features from the adjusted close by shifting them $u$
days towards the past
(note that the $s$ first observations would be without a target value)
6. (Optional) Scale the data
7. Split the data into a training set and a test set, or a training set, 
validation set and test set if we are tuning the hyper parameters
8. Perform a rolling or normal prediction.
9. (Optional) Rescale the data

### Metrics

As stated in the [problem statement](#problem-statement) we assume that the 
important factor when for example making a trading decision is the predicted 
value 7, 14 and 28 days ahead, and that each of these prediction is of equal
importance.
We must find a proper metric to address this problem.
We could have said that a prediction is good if it on average mispredicts the
closing value by less than $5 \%$.
However, if we look at the stocks we try to predict, we see that the stocks 
varies far less than $5 \%$ on the course of 28 days.
Instead, we could have used the mean squared error (MSE) of the test set to 
give an indication of how good the prediction is.
Using the positive square root of the MSE (RMSE) can be beneficial in order to 
make the order of the error easier comparable to the price by cancelling the 
effect of squaring.
However, as the absolute value of the stocks are quite different (especially 
when comparing `^GSPC` with the other stocks), the RMSE is a bad metric when 
comparing across different stocks,
Therefore, we will be using a form of normalized mean squared error 
([NMRSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalized_root-mean-square_deviation))
defined by
 
$$
\frac{\sqrt{\frac{\sum _{i=1}^{n}({\hat {y}}_{i}-y_{i})^{2}}{n}}}}}
{y_{\max} - y_{\min}}
$$

to assess the error.

## II. Analysis

### Data Exploration
In section, we will present the findings from `notebooks/0-data_analysis.ipynb`. 

As noted above, we will focus on the four stocks 

* `^GSPC` - Standard and Poor 500 portfolio
* `AAPL` - Apple Inc.
* `CMCSA` - Comcast corporation
* `GILD` - Gilead Sciences, Inc.

The adjusted closing value of the stocks (which we will take up the main focus 
in this project) can be summarized in the following table:


| Stock | Samples | Start date | End date   | NaNs | Mean        | Max         | Min         | Std        | Q1          | Q2          | Q3          |
|-------|---------|------------|------------|------|-------------|-------------|-------------|------------|-------------|-------------|-------------|
| ^GSPC | 1260    | 2013-03-07 | 2018-03-07 | 0    | 2079.750175 | 2872.870117 | 1541.609985 | 286.663626 | 1884.517517 | 2065.594971 | 2206.597473 |
| AAPL  | 1254    | 2013-03-06 | 2018-02-28 | 0    | 106.716004  | 179.260000  | 50.928800   | 32.229902  | 85.194266   | 106.182039  | 122.349196  |
| CMCSA | 1258    | 2013-03-06 | 2018-03-06 | 0    | 29.283641   | 42.990000   | 18.031155   | 6.269263   | 24.920172   | 28.271251   | 33.938108   |
| GILD  | 1259    | 2013-03-06 | 2018-03-06 | 0    | 79.645139   | 115.929959  | 41.946136   | 16.337412  | 69.417169   | 77.871612   | 94.513256   |

From the table, we can observe that:

1. The number of samples are almost the same, indicating that some trade days 
are missing for some of the stocks
2. There are no NaNs or 0s present
3. ^GSPC has values roughly one order of magnitude larger than the rest of the 
stocks

Note that the mean and standard deviation are the sample mean and standard 
deviation.
I.e. it does not represent the true mean and standard deviation, at least not 
if the process is non-stationary.

Also note that there are $1826$ days between 2013-03-07 and 2018-03-07.
The 1260 days reflects the fact that there are no trading during week-ends and
bank holidays. 
Also, if we were looking at smaller stocks, they could have "missing" data 
simply from the fact that no one was trading those stocks on the given day.

Visual inspection (see [Exploratory Visualization](#exploratory-visualization))
shows no sign of outliers in terms of erroneous data in the adjusted close 
values;
one example of such error could be a single day which has 10 times higher
closing value due to a decimal error arising from manual typing of the data.

Further we can clearly observe that the full time series contain clear 
growing and decaying trend in the range we are observing. 
By performing a stationarity test (similar to the one performed 
[here](https://machinelearningmastery.com/time-series-data-stationary-python/))
we will investigate if the test set of `CMCSA`, which appears to have the least 
linear trend, is indeed non-stationary.

 We will use the
[Augmented Dickey-Fuller](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)
test to test the probability that the time series contain a unit root and 
thereby is [trend-stationary](https://en.wikipedia.org/wiki/Trend_stationary),
i.e. if the trend is removed, then the resulting data is stationary.
If the time series is trend-stationary it is at least not stationary.
The hypotheses are:

* Null hypothesis: The series are non-stationary (i.e. there exist an time 
dependent trend)
* Alternative hypothesis: The series is stationary (i.e. no trend exist)

We've set the rejection hypothesis threshold of the p-value to  55 , meaning 
that we reject the null hypothesis (the series are stationary) if the p-value
is less than or equal to $0.05$.

We found that since $p > 0.05$ there was no evidence to reject the null 
hypothesis.
In other words, the time series is probably non-stationary.

### Exploratory Visualization

In section, we will present more findings from
`notebooks/0-data_analysis.ipynb`. 

In order to justify our choice of `Adj. Close` as the sole variable we will 
create the data set from we will perform some visual inspections on the 
features contained in the `.csv` files in the `data/` directory.

If we for example look at the ochl features of the `AAPL` stock (the rest can
 be found in `notebooks/0-data_analysis.ipynb`) shown below

![alt text](../images/AAPL_ohlc.png "AAPL open, high, low and close value")
 
We observe that overall, the ochl values follow each other to a high degree.
It  is important to note that they are not exactly the same, and in fact, how
the open, high and low aligns with the closing value can give important hints 
about the closing value the following day. 
The sudden drop is due to a stock split (which is accounted for in the 
adjusted close price).

These features can be compared with the plot of the volume data given below

![alt text](../images/volume.png "AAPL open, high, low and close value")

We observe that the volume does not follow the same trend as the ochl data, and 
has far less structure. 
Note that the ^GSPC volume has been divided by 100 to make it easier to visually
compare the results.

As the ochl features are quite similar, and as the volume data contain less 
structure we choose (in order to limit the scope) to only look at the 
adjusted close values (shown below), as these contain strong structures and are 
likely to contain data needed for accurate prediction. 

![alt text](../images/^GSPC_tvt.png "^GSPC training, validation and training 
set")
![alt text](../images/AAPL_tvt.png "AAPL training, validation and training 
set")
![alt text](../images/CMCSA_tvt.png "CMCSA training, validation and training 
set")
![alt text](../images/GILD_tvt.png "GILD training, validation and training 
set")


### Algorithms and Techniques

In this project we will use some quite different techniques in order to try 
to estimate the closing value, and we will here present each one briefly.

In order to make the benchmarking tests we will use the following techniques:

* *Last day prediction (custom made)*: 
This estimator simply uses the current value to make the predictions.
For example, assume that at day $d$ the closing value is $y$, then the 
estimator would predict a closing value of $y$ for the days $d+7$, $d+14$ and
$d+28$.
The estimator is very simple, and has the potential to yield great results (at
least in the case where the closing value changes minimally).
The fitting routine only looks at the last value of the data, and stores this
or these values in a variable.
In the prediction phase, the stored value(s) will be set to the predicted value.
This estimator has no hyper parameters to tune.

* *Random gaussian (custom made)*: 
It's often stated random guessing, or even using
[monkeys](https://www.forbes.com/sites/rickferri/2012/12/20/any-monkey-can-beat-the-market/#5f10485a630a)
for stock trading can outperform professional humans.
Having no monkeys at hand, it easier to implement a random number generator 
to do the stock prediction.
This predictor will store the mean and the standard deviation of the training 
data in the fitting step, and use a gaussian random number generator with 
using the means and standard deviations from the fitting step in order to 
make predictions.
Note that as the processes are non-stationary, the mean and standard 
deviation will change as time evolves.
Neither this estimator has any hyper parameters to tune.

* *Linear regression (from the `sklearn` package)*:
This is considered to be the simplest of the linear regressors.
It is using the
[ordinary least squares](http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) 
during fitting.
This will give the coefficients $a$ and $b$ in the linear equation $y=ax+b$ 
which will be used when predicting $y$ based on the input $x$.
Although the class have some 
[input parameters](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
for the constructor, we would not usually define these as tunable hyper 
parameters for the model

For trying to predict the closing value we will use the following techniques:

* *k-nearest neighbor regression (from the `sklearn` package)*:
Also this is a quite simple model, but often yields good results despite its 
simplicity.
Even sophisticated software like
[QuantDesk](https://classroom.udacity.com/courses/ud501/lessons/4684695874/concepts/46403887880923)
are using kNN.
The fitting phase just uploads the available data to a storage location (like
the local memory or a database).
When predicting, the algorithm will find the k nearest neighbors (according 
to a distance metric) and predict the new value based on the mean of these.
The algorithm has at least
[some](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor)
hyper parameters to play around with.
We can choose whether the mean should be weighted by for example distance or 
not, what metric to use for the calculation of distance and last but not 
least, the number of neighbors to take into account.
 
* *A Long Short Time Memory Recurring Neural Network (from the `keras` 
package, modified for our needs)*: 
The Long Short Time Memory (LSTM) neural network is a type of recurring 
neural network (RNN), which means that output of one of the 
[recurrences](https://www.youtube.com/watch?v=UNmqTiOnRfg) in 
the neural network serves as the input for the next one, like explained
[here](https://classroom.udacity.com/courses/ud730/lessons/6378983156/concepts/65523553300923).
A nice introduction to LSTM can be found in this
[blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).
Although figuring out how the input to this machinery can be
[mind boggling](https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/)
at first, all we need to do is to reshape our data to a 3-dimensional array\ 
where the dimensions represents
`[samples, time steps, features]`, where the `time step` dimension tells the 
LSTM how many
[times](https://github.com/keras-team/keras/issues/2045) 
the 
[recurrence should occur](https://stats.stackexchange.com/questions/288404/how-does-keras-generate-an-lstm-layer-whats-the-dimensionality).
The LSTM overcomes the vanishing gradient problem found in RNNs by having 
gates in each cell (equivalent to nodes in traditional neural 
network) which determines what should be kept in memory, and what should be 
forgotten, and is therefore very well suited for time series forecasting.
The LSTM learns the weights of the gates and the cells in the fitting phase 
and uses these in the prediction phase.
The hyper parameter tuning can be quite extensive, so we have in this project
limited the hyper parameter tuning to epochs, batch size, drop out rates, 
number of cells in the first layer, number of cells in the second layer and 
the number of time steps.

### Benchmark

The calculations of the benchmarks can be found in `notebooks/1.*.ipynb`.
We use the results of the simple algorithms above to get a feeling with how 
well we can perform with the simplest tools in the toolbox.
If one of the more advanced methods is only slightly better than these 
results, one should consider if it is worth the cost to use a more complex 
model.

By using the rolling prediction technique described in the
[Problem Statement](#problem-statement)
and the metric described in [Metrics](#metrics), we end up with the following
scores (the complete table with all the results can be found in 
[IV. Results](#iv.-results))

| Stock | Last day | Random Gaussian | Linear regression |
|-------|----------|-----------------|-------------------|
| ^GSPC | 7.01     | 577.33          | 6.15              |
| AAPL  | 1.50     | 96.01           | 1.34              |
| CMCSA | 0.51     | 18.11           | 0.48              |
| GILD  | 0.94     | 15.55           | 0.76              |
| Sum   | 9.95     | 706.99          | 8.72              |


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

Input to the LSTM is a 3-dimensional array where the dimensions represents
`[samples, time steps, features]`.
After reading the data the dimensions are `[samples, features]`.
The `time step` dimension tells the LSTM how many
[times](https://github.com/keras-team/keras/issues/2045) 
the 
[recurrence should occur](https://stats.stackexchange.com/questions/288404/how-does-keras-generate-an-lstm-layer-whats-the-dimensionality),
and we need a routine which transform the data to the desired format.
A function which does this job is `prepare_input`, found in 
`estimators/lstm.py`.

Datapoints (prices of different stocks) are not independent of each other -> Naive Bayes is not appropriate
Add features and do some PCA?

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

> **Note:** All classes and functions has been extensively documented in the 
source code.

> **Note:** The dictionaries `data_preparation`, `estimators` and `utils` 
contains functionality to perform the analysis, whereas the analysis itself 
is performed in the `notebooks` dictionary.

| Stock | knn (unoptimized) | lstm (unoptimized, unscaled) | lstm (unoptimized, scaled) |
|-------|-------------------|------------------------------|----------------------------|
| ^GSPC | 2.38              | 11827.14                     | 148.83                     |
| AAPL  | 0.99              | 503.91                       | 7.59                       |
| CMCSA | 0.33              | 23.31                        | 0.59                       |
| GILD  | 0.88              | 42.67                        | 0.75                       |
| Sum   | 4.58              | 12397.03                     | 157.77                     |



### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


| Stock | knn (optimized) | lstm (optimized) | knn (normal prediction) |
|-------|-----------------|------------------|-------------------------|
| ^GSPC | 0.54            | 6.70             | 93.65                   |
| AAPL  | 0.10            | 1.59             | 55.64                   |
| CMCSA | 0.04            | 0.58             | 0.97                    |
| GILD  | 0.05            | 0.76             | 2.42                    |
| Sum   | 0.73            | 9.63             | 152.68                  |


| Stock | knn (normal prediction) |
|-------|-------------------------|
| ^GSPC | 93.65                   |
| AAPL  | 55.64                   |
| CMCSA | 0.97                    |
| GILD  | 2.42                    |
| Sum   | 152.68                  |

## IV. Results
_(approx. 2-3 pages)_

Summary table of the results

| Stock | Last day | Random Gaussian | Linear regression | knn (unoptimized) | lstm (unoptimized, unscaled) | lstm (unoptimized, scaled) | knn (optimized) | lstm (optimized) | knn (normal prediction) |
|-------|----------|-----------------|-------------------|-------------------|------------------------------|----------------------------|-----------------|------------------|-------------------------|
| ^GSPC | 7.01     | 577.33          | 6.15              | 2.38              | 11827.14                     | 148.83                     | 0.54            | 6.70             | 93.65                   |
| AAPL  | 1.50     | 96.01           | 1.34              | 0.99              | 503.91                       | 7.59                       | 0.10            | 1.59             | 55.64                   |
| CMCSA | 0.51     | 18.11           | 0.48              | 0.33              | 23.31                        | 0.59                       | 0.04            | 0.58             | 0.97                    |
| GILD  | 0.94     | 15.55           | 0.76              | 0.88              | 42.67                        | 0.75                       | 0.05            | 0.76             | 2.42                    |
| Sum   | 9.95     | 706.99          | 8.72              | 4.58              | 12397.03                     | 157.77                     | 0.73            | 9.63             | 152.68                  |


### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?


## VI. Additional resources (not mentioned in the text)

### Tutorials from Machine learning mastery
https://machinelearningmastery.com/prepare-univariate-time-series-data-long-short-term-memory-networks/ 
https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

### Machine learning for trading (Udacitiy course)
https://classroom.udacity.com/courses/ud501