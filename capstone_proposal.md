# Machine Learning Engineer Nanodegree
## Capstone Proposal: Build a Stock Price Indicator  
March 10th, 2018

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

* clearly describe the problem that is to be solved
* The problem described should be well defined 
* should have at least one relevant potential solution
* describe the problem thoroughly such that it is clear that the problem is quantifiable
* can be measurable
* replicable



For this project, your task is to build a stock price predictor that takes 
daily trading data over a certain date range as input, and outputs projected 
estimates for given query dates. Note that the inputs will contain multiple 
metrics, such as opening price (Open), highest price the stock traded at 
(High), how many stocks were traded (Volume) and closing price adjusted for 
stock splits and dividends (Adjusted Close); your system only needs to predict 
the Adjusted Close price.

You are free to choose what form your project takes (a simple script, a web 
app/service, Android/iOS app, etc.), and any additions/modifications you want 
to make to the project (e.g. suggesting what trades to make). Make sure you 
document your intended features in your report.


02-07 Dealing with data
Adjusted
[stock split](https://classroom.udacity.com/courses/ud501/lessons/4442578629/concepts/45792868240923)
and
[dividends](https://classroom.udacity.com/courses/ud501/lessons/4442578629/concepts/44273516340923)




For your core stock predictor, implement:

A training interface that accepts a data range (start_date, end_date) and a list of ticker symbols (e.g. GOOG, AAPL), and builds a model of stock behavior. Your code should read the desired historical prices from the data source of your choice.
A query interface that accepts a list of dates and a list of ticker symbols, and outputs the predicted stock prices for each of those stocks on the given dates. Note that the query dates passed in must be after the training date range, and ticker symbols must be a subset of the ones trained on.


### Datasets and Inputs
_(approx. 2-3 paragraphs)_

* the dataset(s) and/or input(s) thoroughly described
* such as how they relate to the problem and why they should be used. 
* Information such as how the dataset or input is (was) obtained
* characteristics of the dataset or input
* relevant references and citations
* It should be clear how the dataset(s) or input(s) will be used in the project
* whether their use is appropriate given the context of the problem.

For the scope of this project, we will use daily stock data containing open, 
high, low, close, volume (ohlcv) together with the [adjusted close]() 
* Daily stock data have traditionally been readily available through free 
APIs such as Yahoo! Finance, Google finance and EDGAR.
* However, late 2017, several APIs got deprecated. Reading from 
[pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/index.html)
(a continuation of the now deprecated `pandas.io`)

> Yahoo! Finance has been immediately deprecated. Yahoo! substantially altered
 their API in late 2017 and the csv endpoint was retired.
 
> Google’a API has become less reliable during 2017. While the google
 datareader often works as expected, it is not uncommon to experience a range 
 of errors when attempting to read data, especially in bulk. 

[discontinuation](https://stackoverflow.com/a/44092983/2786884) of their free API

Although there exists workarounds like 
[`fix_yahoo_finance`](https://github.com/ranaroussi/fix-yahoo-finance) which 
uses a smart way to retrieve
[breadcrumbs](https://stackoverflow.com/questions/44030983/yahoo-finance-url-not-working),

[Alpha Vantage](https://www.alphavantage.co/) and [Quandl](https://www.quandl.com/)

All stock data is downloaded through Quandl, with exception of the Standard and 
Poor 500 index, which is manually downloaded from Yahoo! Finance. Although 
this could be
[automated](http://quantlabs.net/blog/2017/08/fix-now-for-yahoo-finance-with-python-historical-datareader/),
the solution can render obsolete in the near future.

The most stable way to retrieve the data would probably be to buy the data 
from a vendor, for example through Quandl. 

The scope is set to look at a fixed dataset. However, an possible extension 
would be to update and train the dataset daily.



### Solution Statement
_(approx. 1 paragraph)_

* clearly describe a solution to the problem
* The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given
* Additionally, describe the solution thoroughly such that it is clear that 
the solution is measurable
* replicable 

For your core stock predictor, implement:

A training interface that accepts a data range (start_date, end_date) and a list of ticker symbols (e.g. GOOG, AAPL), and builds a model of stock behavior. Your code should read the desired historical prices from the data source of your choice.
A query interface that accepts a list of dates and a list of ticker symbols, and outputs the predicted stock prices for each of those stocks on the given dates. Note that the query dates passed in must be after the training date range, and ticker symbols must be a subset of the ones trained on.


### Benchmark Model
_(approximately 1-2 paragraphs)_

* provide the details for a benchmark model or result that relates to the 
domain, problem statement, and intended solution
* Ideally, the benchmark model or result contextualizes existing methods or 
known information in the domain and problem given, which could then be objectively compared to the solution.
* Describe how the benchmark model or result is measurable (can be measured 
by some metric and clearly observed) with thorough detail.

Machine learning mastery
https://machinelearningmastery.com/time-series-forecasting-supervised-learning/
https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
https://machinelearningmastery.com/time-series-data-stationary-python/

Medium does a good job explaining, although the results are somewhat poor
https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-part-one-simple-time-series-forecasting-f992daa1045a
https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-1-2-correct-time-series-forecasting-backtesting-9776bfd9e589
https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-2-1-multivariate-time-series-ab016ce70f57
https://medium.com/machine-learning-world/tagged/finance

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

* at least one evaluation metric that can be used to quantify the performance
 of both the benchmark model and the solution model.
* The evaluation metric(s) you propose should be appropriate given the 
 context of the data, the problem statement, and the intended solution. 
* Describe how the evaluation metric(s) are derived and provide an example of
 their mathematical representations (if applicable). 
* Complex evaluation metrics should be clearly defined and quantifiable (can 
be expressed in mathematical or logical terms).


A basic run of the core system would involve one call to the training interface, and one or more calls to the query interface. Implement a train-test cycle to measure the performance of your model. Use it to test prediction accuracy for query dates at different intervals after the training end date, e.g. the day immediately after training end date, 7 days later, 14 days, 28 days, etc.

(Note: Pick the training period accordingly so that you have ground truth data for that many days in the future.)


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



### Additional resources
Udacity's course [Machine learning for Trading](https://classroom.udacity.com/courses/ud501)
[Machine Learning Strategies for Time Series Forecasting](https://link.springer.com/chapter/10.1007/978-3-642-36318-4_3)
[Machine data for sequential data: A review](http://web.engr.oregonstate.edu/~tgd/publications/mlsd-ssspr.pdf)
-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
