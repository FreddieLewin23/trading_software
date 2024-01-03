import os
import qi_client
import warnings
from datetime import datetime, timedelta
import pandas as pd
from algo_long import find_models_long, check_current_trades_long, filter_csv_by_date_bfo
from algo_short import find_models_short, check_current_trades_short

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['QI_API_KEY'] = ''
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = ''
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))

models_USD = [x.name
              for x in api_instance.get_models(tags='USD, Stock')
              if x.model_parameter == 'long term' and '_' not in x.name
              ][:3400]


def find_account_value_and_update_account_value_long_short(date):
    value = 100000
    df_completed_trades_long = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/backtest_long_short/completed_backtest_long.csv')
    df_completed_trades_short = pd.read_csv("/Users/FreddieLewin/PycharmProjects/new_dl_token/backtest_long_short/completed_backtest_short.csv")
    for index, row in df_completed_trades_long.iterrows():
        buy = row['trade_entry_price']
        sell = row['trade_exit_price']
        order_size = row['order_size']
        percent_return = ((sell - buy) / buy) * 100
        value_added = order_size * (percent_return / 100)
        value += value_added
    for index, row in df_completed_trades_short.iterrows():
        buy = row['trade_entry_price']
        sell = row['trade_exit_price']
        order_size = row['order_size']
        percent_return = ((buy - sell) / buy) * 100
        value_added = order_size * (percent_return / 100)
        value += value_added
    current_account_value = value
    df_account_tracker = pd.read_csv("account_value_short_long.csv")
    new_row = pd.DataFrame({'date': [date], 'account_value': [current_account_value]})
    df_account_tracker = pd.concat([df_account_tracker, new_row], ignore_index=True)
    df_account_tracker.to_csv("account_value_short_long.csv", index=False)
    return current_account_value


def weekdays_between_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    if start_date > end_date:
        return "Error: Start date should be before end date."
    weekdays = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:
            weekdays.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return weekdays


if __name__ == '__main__':
    dates_for_backtest = weekdays_between_dates('2016-11-11', '2023-12-30')
    for date in dates_for_backtest:
        find_models_long(date)
        find_models_short(date)
        check_current_trades_long(date)
        check_current_trades_short(date)
        find_account_value_and_update_account_value_long_short(date)


