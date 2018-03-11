# Machine Learning Engineer Nanodegree
## Capstone Proposal: Build a Stock Price Indicator  
March 10th, 2018
Michael Løiten

## Proposal
_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_

* (Checked) brief details on the background information of the domain from 
which the 
project is proposed
* (Checked) Historical information
* (Checked) How or why a problem in the domain can or should be solved
* (Checked) Related academic research should be appropriately cited in this 
section
* (checked?) a discussion of your personal motivation for investigating a 
particular problem in the domain is encouraged but not required.

Investment firms, hedge funds and even individuals have been using financial
models to better understand market behavior and make profitable investments and
trades. A wealth of information is available in the form of historical stock
prices and company performance data, suitable for machine learning algorithms 
to process.

Stock market trading has existed since [1602](https://en.wikipedia.org/wiki/Euronext_Amsterdam).
From its very start the trading of stocks has not only been a place for 
passive trading between buyer and seller, but it has also influenced national 
and global economies.

Although stock trading traditionally has been made manually by man, increased
computational power have paved way for
[algorithmic trading](https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp)
which typically builds on mathematical models or statistical techniques, and 
has been used extensively since the 
[1970s](https://www.quantinsti.com/blog/history-algorithmic-trading-hft/).

Traditionally statistical techniques such as looking at 
[Bollinger bands](https://en.wikipedia.org/wiki/Bollinger_Bands) have been 
used for making systematic trading decisions.
However, in today's marked, more information of the system is usually needed in 
order to reach good investment strategies.

One way to add new information is to use regression, through machine learning,
to try to predict the future value of a stock.
In addition, techniques such as those found in 
[reinforced](https://en.wikipedia.org/wiki/Reinforcement_learning) 
[learning](https://github.com/aikorea/awesome-rl) can help 
traders (human and machines) make better decisions based on more data than 
humans do (which often leads to short term decisions making the market 
volatile).

The idea of using machine learning for trading is far from new, and both 
[academics](https://www.sciencedirect.com/science/article/pii/S0957417405003015) 
and
[students](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.83.5299&rep=rep1&type=pdf)
have been studying the topic.

Apart from finance being interesting to studying in its own, in the domain of
machine learning it presents an interesting problem due to the nature of 
data, being sequential.
This means that traditional training techniques such as randomization of the 
training data and cross validation makes no sense.
The training must also be considered perishable as the stock market is rapidly
changing. 

 In addition, financial data is know for beeing 
[non-stationary](https://en.wikipedia.org/wiki/Stationary_process)
[stochastic](https://en.wikipedia.org/wiki/Stochastic)
(the stochastical nature is in some cases be modelled as a [Brownian motion 
Markovian process](http://www.ulb.ac.be/di/map/gbonte/ftp/time_ser.pdf)).
There are several
[tests](https://quant.stackexchange.com/questions/2372/how-to-check-if-a-timeseries-is-stationary/2373)
which can show that the stocks are non-stationarity, which has a consequence 
that the statistical moments are dependent on time, and that
[clever](https://quant.stackexchange.com/questions/9192/how-to-normalize-stock-data) 
[ways](https://quant.stackexchange.com/questions/14205/why-are-we-obsessed-over-normalizing-financial-data?noredirect=1&lq=1)
of 
[scaling](https://quant.stackexchange.com/questions/14205/why-are-we-obsessed-over-normalizing-financial-data?noredirect=1&lq=1)
the data is needed.

The techniques used to deal with stock data is also transferable to other 
sequential data such as weather forecasting, distribution on the 
electrical grid, maintenance prediction other fields where sensoric 
data is of importance.

### Problem Statement
_(approx. 1 paragraph)_

* (Checked) clearly describe the problem that is to be solved
* (Checked) The problem described should be well defined 
* (Checked) should have at least one relevant potential solution
* (Checked) describe the problem thoroughly such that it is clear that the problem is
 quantifiable
* (Checked) can be measurable
* (Checked) replicable

In this project, we will build a stock price predictor for the 50
stocks with the highest weights in the Standard and Poor's 500 (S&P500) 
portfolio as of 2018-03-06.
The predictor will be trained on historical daily data including the
opening price (Open), highest price the stock traded at (High),
how many stocks were traded (Volume) and closing price adjusted for 
[stock split](https://classroom.udacity.com/courses/ud501/lessons/4442578629/concepts/45792868240923)
and
[dividends](https://classroom.udacity.com/courses/ud501/lessons/4442578629/concepts/44273516340923)
(Adjusted Close (see for example "02-07 Dealing with data" in Udacity's 
course
[Machine learning for Trading](https://classroom.udacity.com/courses/ud501)
for more info)).
The predictor should be able to predict the Adjusted Close up until 28 days 
into the future. And the predicted stock value 7 days from the date of 
prediction should be within 5% of actual value, on average.

Although interesting, trading decisions based on machine learning will be 
outside of scope in this project.
Also automatic updating of the stock data will be outside the scope of this 
project. 


### Datasets and Inputs
_(approx. 2-3 paragraphs)_

* (Checked) the dataset(s) and/or input(s) thoroughly described
* (Implicitly checked) such as how they relate to the problem and why they 
should be used. 
* (Checked) Information such as how the dataset or input is (was) obtained
* (Checked) characteristics of the dataset or input
* (Checked) relevant references and citations
* (Checked) It should be clear how the dataset(s) or input(s) will be used in
 the project
* (Checked) whether their use is appropriate given the context of the problem.

For the scope of this project, we will use daily stock data containing open, 
high, low, close, volume (ohlcv) together with the adjusted close for the 50
stocks with the highest weights in the S&P500 portfolio together with ^GSPC 
itself.

The weights for S&P500 will has been collected from 
[SlickCharts](https://www.slickcharts.com/sp500) the 6th of March 2018.
 
The daily stock data have traditionally been readily available through free 
APIs such as Yahoo! Finance, Google finance and EDGAR.
However, late 2017, several APIs got deprecated.
Reading from 
[pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/index.html)
(a continuation of the now deprecated `pandas.io`)

> Yahoo! Finance has been immediately deprecated. Yahoo! substantially altered
 their API in late 2017 and the csv endpoint was retired.
 
> Google’a API has become less reliable during 2017. While the google
 datareader often works as expected, it is not uncommon to experience a range 
 of errors when attempting to read data, especially in bulk. 

As good free alternatives,
[Alpha Vantage](https://www.alphavantage.co/) and 
[Quandl](https://www.quandl.com/)
remains.

All stock data is downloaded through Quandl, with exception of the Standard and 
Poor 500 index, which is manually downloaded from Yahoo! Finance.
(Although Yahoo! Finance has 
[discontinued](https://stackoverflow.com/a/44092983/2786884) 
their free API, workarounds like 
[`fix_yahoo_finance`](https://github.com/ranaroussi/fix-yahoo-finance) which 
uses a smart way to retrieve the needed
[breadcrumbs](https://stackoverflow.com/questions/44030983/yahoo-finance-url-not-working).
Although the download could be through a
[script](http://quantlabs.net/blog/2017/08/fix-now-for-yahoo-finance-with-python-historical-datareader/),
the solution can render obsolete in the near future.)

The scope is set to look at a fixed dataset. However, an possible extension 
would be to update and train the dataset daily.
The most stable way to retrieve the data would probably be to buy the data from a professional vendor, 
through for example Quandl. 

### Solution Statement
_(approx. 1 paragraph)_

* (Checked) clearly describe a solution to the problem
* (Checked) The solution should be applicable to the project domain and appropriate for
 the dataset(s) or input(s) given
* (Checked) Additionally, describe the solution thoroughly such that it is clear that 
the solution is measurable
* (Checked) replicable 

A training model will be build.
This inputs a data range from a start date, up until a end date 
(not earlier than 28 days prior to the prediction), together with a list of 
ticker symbols within the 50 symbols of the S&P500 set.
The output will be the model used for prediction.

A data cleaning module based on the 
[01-05 Incomplete data](https://classroom.udacity.com/courses/ud501/lessons/3909458794/concepts/39567585520923)
of Udacity's course Machine learning for Trading will be made.
The data will pass through this module before the final prediction.

The predictor module will be based on the model, taking one or more of the
available ticker symbols from S&P500 (like GOOG, AAPL) as an input, yielding
the predicted adjusted close prognosis.

The modules will be made callable through a driver script.

### Benchmark Model
_(approximately 1-2 paragraphs)_

* (Checked) provide the details for a benchmark model or result that relates to 
the domain, problem statement, and intended solution
* (Checked) Ideally, the benchmark model or result contextualizes existing 
methods or known information in the domain and problem given, which could 
then be objectively compared to the solution.
* (Checked) Describe how the benchmark model or result is measurable (can be 
measured by some metric and clearly observed) with thorough detail.

Although other models are freely available online (see for example the blogs 
in the [Additional resources](additional-resources) section), the model will 
be benchmarked against simple models of the adjusted close like 

* Prediction equal the latest day
* [Ordinary least square](http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares) 
* [Polynomial regression](http://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)
of order 5
* Random guessing using a 
[Gaussian distribution](https://docs.python.org/3/library/random.html#random.gauss)
with the mean equaling the last day before prediction, and the standard 
deviation being based on the historical standard deviation.


### Evaluation Metrics
_(approx. 1-2 paragraphs)_

* (Checked) at least one evaluation metric that can be used to quantify the 
performance of both the benchmark model and the solution model.
* (Checked)  The evaluation metric(s) you propose should be appropriate given
 the context of the data, the problem statement, and the intended solution. 
* (Checked)  Describe how the evaluation metric(s) are derived and provide an
 example of their mathematical representations (if applicable). 
* (Checked)  Complex evaluation metrics should be clearly defined and 
quantifiable (can be expressed in mathematical or logical terms).

The relative error of the adjusted close will denote the success 
factor of the model.
The relative error is defined as

`|True value - Approximated vale|/True value`


### Project Design
_(approx. 1 page)_

* summarize a theoretical workflow for approaching a solution given the 
problem. 
* Provide thorough discussion for what strategies you may consider employing, 
* what analysis of the data might be required before being used, or which 
algorithms will be considered for your implementation
* The workflow and discussion that you provide should align with the 
qualities of the previous sections
* Additionally, you are encouraged to include small visualizations, 
pseudocode, or diagrams to aid in describing the project design
* The discussion should clearly outline your intended workflow of the 
capstone project.

In order to successfully build a model, one need to get a feeling of the data
set.
First the quality of the data will be checked.
Are there a lot of missing data?
Can all the missing data be mended, or should some ticker symbols be left out?
A visualization of the different data attributes will be done to get a 
feeling of general trends and patterns.
Feature engineering will also bee needed as we are working with categorical 
data.
The features will be created by shifting the data as explained 
[here](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/).

Following, the simple benchmark tests will be performed in order to see if 
fancy models obtained through machine learning is needed at all.
If it turns out that some simple methods performs really well, it could be 
interesting to see if machine learning has something to add to the problem at
at all.

The next step in the project would be to experiment with different techniques.
This can potentially take a lot of effort as there are many techniques to 
explore.
However the following techniques can be interesting to have a look at
* KNN, as this is a simple model, and has proved successful at for example 
[QuantDesk](https://classroom.udacity.com/courses/ud501/lessons/4684695874/concepts/46403887880923)
* [Generalized Linear Models](http://scikit-learn.org/stable/modules/linear_model.html) 
like Bayesian Regression
* Neural nets, in particular convolutional neural networks as these have a 
tendency to perform well when the "relative positions" of the features is 
important.

When checking evaluating the models we will look at
* How the performance of the models scales with the amount of data points
* The 
[learning curve](https://classroom.udacity.com/nanodegrees/nd009/parts/4a8bfaa1-d0d3-4cd9-b8c0-58149310e12f/modules/89692644-156f-4447-8601-ce46b5f1c572/lessons/bc61c575-ae7c-4243-bfc2-bff377e7216a/concepts/ddc42022-25b1-41e7-9daa-a9e9a0614e9f)
for variance and bias.
* The 
[model complexity](https://classroom.udacity.com/nanodegrees/nd009/parts/4a8bfaa1-d0d3-4cd9-b8c0-58149310e12f/modules/89692644-156f-4447-8601-ce46b5f1c572/lessons/bc61c575-ae7c-4243-bfc2-bff377e7216a/concepts/d9bb8167-05da-4fff-a7d1-cc6b4a88ae6d)
* The amount of features, i.e. how far back would we need to look in order to 
obtain reasonable results. 

Finally the full pipeline of training, cleaning and predicting will be made.



### Additional resources
* Udacity's course 
[Machine learning for Trading](https://classroom.udacity.com/courses/ud501)
* [Machine Learning Strategies for Time Series Forecasting](https://link.springer.com/chapter/10.1007/978-3-642-36318-4_3)
* [Machine data for sequential data: A review](http://web.engr.oregonstate.edu/~tgd/publications/mlsd-ssspr.pdf)
* The
[Machine](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
[learning](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
[mastery](https://machinelearningmastery.com/time-series-data-stationary-python/)
time series tutorial
* The 
[Medium](https://medium.com/machine-learning-world/tagged/finance)
blog (note that this blog does a 
[good](https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-part-one-simple-time-series-forecasting-f992daa1045a)
[job](https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-1-2-correct-time-series-forecasting-backtesting-9776bfd9e589)
[explaining](https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-2-1-multivariate-time-series-ab016ce70f57), 
although the results 
are somewhat poor).

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar
to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**)
written in a clear, concise and specific fashion? Are there any ambiguous
 terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your
 proposal?
- Have you properly proofread your proposal to assure there are minimal 
grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
