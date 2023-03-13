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
            Keyword("forecast_date", required=True, otl_type=OTLType.TEXT),
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
        forecast_date_tuple = (int(x) for x in self.get_arg("s").value.split(',') if len(x) > 0)
        forecast_date = datetime(*forecast_date_tuple)

        # Make your logic here
        result = sarima(ar, i, ma, sar, si, sma, s, forecast_date)

        # Add description of what going on for log progress
        self.log_progress('First part is complete.', stage=1, total_stages=2)
        #
        self.log_progress('Last transformation is complete', stage=2, total_stages=2)

        # Use ordinary logger if you need

        self.logger.debug(f'Command sarima get first positional argument = {first_positional_string_argument}')
        self.logger.debug(f'Command sarima get keyword argument = {kwarg_int_argument}')

        return result
