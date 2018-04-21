# Machine Learning Engineer Nanodegree
## Capstone Project
April 21st, 2018

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
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


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

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

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