from datetime import datetime

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def sarima(df: pd.DataFrame, p: int, d: int, q: int, sp: int, sd: int, sq: int, m, steps: int) -> pd.DataFrame:
    order = (p, d, q)
    seasonal_order = (sp, sd, sq, m)
    df = df.set_index('Date')
    model = SARIMAX(df, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    predictions = model_fit.get_forecast(steps=steps)
    predictions = predictions.predicted_mean.to_frame('sarima_prediction').rename_axis('time').reset_index()
    predictions.set_index('time')

    return predictions
