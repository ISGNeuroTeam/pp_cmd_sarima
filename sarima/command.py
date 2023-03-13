from datetime import datetime

import pandas as pd
from otlang.sdk.syntax import Keyword, Positional, OTLType
from pp_exec_env.base_command import BaseCommand, Syntax
from .sarima import sarima


class SarimaCommand(BaseCommand):
    # define syntax of your command here
    syntax = Syntax(
        [
            Keyword("AR", required=True, otl_type=OTLType.INTEGER),
            Keyword("I", required=True, otl_type=OTLType.INTEGER),
            Keyword("MA", required=True, otl_type=OTLType.INTEGER),
            Keyword("sAR", required=True, otl_type=OTLType.INTEGER),
            Keyword("sI", required=True, otl_type=OTLType.INTEGER),
            Keyword("sMA", required=True, otl_type=OTLType.INTEGER),
            Keyword("s", required=True, otl_type=OTLType.INTEGER),
            Keyword("forecast_steps", required=True, otl_type=OTLType.INTEGER),
        ],
    )
    use_timewindow = False  # Does not require time window arguments
    idempotent = True  # Does not invalidate cache

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_progress('Start Sarima model forecasting command')
        # that is how you get arguments
        ar = self.get_arg("AR").value
        i = self.get_arg("I").value
        ma = self.get_arg("MA").value
        sar = self.get_arg("sAR").value
        si = self.get_arg("sI").value
        sma = self.get_arg("sMA").value
        s = self.get_arg("s").value
        forecast_steps = self.get_arg("forecast_steps").value

        # Make your logic here
        result = sarima(df, ar, i, ma, sar, si, sma, s, forecast_steps)

        return result
