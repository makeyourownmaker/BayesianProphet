# BayesianProphet

![Lifecycle
](https://img.shields.io/badge/lifecycle-experimental-orange.svg?style=flat)
![Python
](https://img.shields.io/badge/Python-blue.svg?style=flat)

Bayesian optimisation of prophet temperature model with daily and yearly seasonalities plus extra regressors

If you like BayesianProphet, give it a star, or fork it and contribute!


## Installation/Usage

Required:
 * Recent version of [python](https://www.python.org/)
 * [Prophet](https://github.com/facebook/prophet) package
 * [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) package
 * [pandas](https://pandas.pydata.org/) package
 * [Jupyter](https://jupyter.org/)

To install the python packages:
```sh
pip install -r requirements.txt
```

After the above dependencies have been installed either,
 * clone the repository and open the notebook(s) in a local installation of Jupyter, or
 * try notebook(s) remotely
   * run on [Colab](https://colab.research.google.com/github/makeyourownmaker/BayesianProphet/blob/main/notebooks/BayesOptProphetHyperparameters.ipynb)
   * run on [MyBinder](https://mybinder.org/v2/gh/makeyourownmaker/BayesianProphet/main?filepath=notebooks%2FBayesOptProphetHyperparameters.ipynb)
   * view on [NBViewer](https://nbviewer.jupyter.org/github/makeyourownmaker/BayesianProphet/blob/main/notebooks/BayesOptProphetHyperparameters.ipynb)
   * view on [GitHub](https://github.com/makeyourownmaker/BayesianProphet/blob/main/notebooks/BayesOptProphetHyperparameters.ipynb)


## Details

See my
[time series and other models for Cambridge UK temperature forecasts in R repository](https://github.com/makeyourownmaker/CambridgeTemperatureModel)
for a detailed explanation of the data (including cleaning), baseline models, 
daily and yearly seasonality descriptions plus R prophet model.  Assumptions
and limitations are covered in the above repository and will not be repeated
here.  Additional exploratory data analysis is available in my
[Cambridge University Computer Laboratory Weather Station R Shiny repository](https://github.com/makeyourownmaker/ComLabWeatherShiny).

My primary interest is in "now-casting" or forecasts within the 
next 1 to 2 hours.  This is because I live close to the data source and 
the [UK met office](https://www.metoffice.gov.uk/) only update their public 
facing forecasts every 2 hours.

The python prophet implementation has a few advantages over the R 
implementation.  I'm interested in
[forcing trend growth to be flat](https://facebook.github.io/prophet/docs/additional_topics.html#flat-trend-and-custom-trends).
The already mentioned R prophet model has strong seasonality and
using zero changepoints usually results in either slightly 
increasing or decreasing trend.


### Bayesian Optimisation of Prophet Hyperparameters

I use over 10 years of training data (mid-2008 to 2018 inclusive)
and select test and validation data from 2019.  Compute time is
acceptable when using zero changepoints, so I use simple grid search 
for discrete parameters such as seasonality modes and separately 
regressor modes.

The parameter space of each prior scale runs over at least 4 orders
of magnitude.  Grid search is no longer an option.  Especially with
4 (2 seasonalities and 2 regressors) or more prior scales to tune.

Bayesian optimisation of prior scale parameters should give similar 
results to random search in less time.


## Roadmap

 * Expand documentation:
   * Include highlights from notebook(s)
   * Summarise the best prophet model including parameters and cross-validation results


## Contributing

Pull requests are welcome.  For major changes, please open an issue first to 
discuss what you would like to change.


## Alternatives

Terrence Neumann has written
[Bayesian Hyperparameter Optimization for Time Series](https://rpubs.com/tdneumann/351073)
which covers Bayesian optimisation of prophet models in R.


## See Also

[Prophet - Tool for producing high quality forecasts for time series data that has multiple seasonality with linear or non-linear growth](https://github.com/facebook/prophet)

[A Python implementation of global optimization with gaussian processes](https://github.com/fmfn/BayesianOptimization)

[Time series and other models for Cambridge UK temperature forecasts in R](https://github.com/makeyourownmaker/CambridgeTemperatureModel)

[Cambridge University Computer Laboratory Weather Station R Shiny Web App](https://github.com/makeyourownmaker/ComLabWeatherShiny)


## License

[GPL-2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
