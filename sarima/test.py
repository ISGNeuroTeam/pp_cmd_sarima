from datetime import datetime, timedelta
from time import time

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf

from sarima import sarima
import pandas as pd


def parser(s):
    return datetime.strptime(s, "%Y-%m-%d")


def main():
    catfish_sales = pd.read_csv('../catfish1.csv', parse_dates=[0], index_col=0,
                                date_parser=parser)
    catfish_sales = catfish_sales.asfreq(pd.infer_freq(catfish_sales.index))

    start_date = datetime(1996, 1, 1)
    end_date = datetime(2000, 1, 1)
    lim_catfish_sales = catfish_sales[start_date:end_date]

    catfish_sales_test = pd.read_csv('../catfish1.csv')
    catfish_sales_df = pd.DataFrame(catfish_sales_test)  # here we create a test dataframe

    train_end = datetime(1999, 7, 1)
    test_end = datetime(2000, 1, 1)

    train_data = lim_catfish_sales[:train_end]
    test_data = lim_catfish_sales[train_end + timedelta(days=1):test_end]

    my_order = (0, 1, 0)
    my_seasonal_order = (1, 0, 1, 12)
    # define model
    model = SARIMAX(catfish_sales, order=my_order, seasonal_order=my_seasonal_order)

    # fit the model
    start = time()
    model_fit = model.fit()
    end = time()
    print('Model Fitting Time:', end - start)

    # get the predictions and residuals
    predictions = model_fit.get_forecast(20)
    print(predictions.summary_frame())
    # residuals = test_data - predictions

    print(f'\n=======================')
    df = sarima(catfish_sales_df, 0, 1, 0, 1, 0, 1, 12, 20)  # here we use test dataframe
    print(df)


if __name__ == "__main__":
    main()
