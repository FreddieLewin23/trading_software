import pandas as pd
import numpy as np
import os
import Qi_wrapper
import qi_client
from datetime import datetime, timedelta
from trading_algo_backtest import find_models_to_buy_long_new_backtest, check_current_trades_long_new_backtest


os.environ['QI_API_KEY'] = ''
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = ''
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))


def find_account_value_and_update_account_value(date):
    df = pd.read_csv("current_trades_backtest_long.csv")
    dict_model_order_size_current_percent_return = []
    for index, row in df.iterrows():
        current_dict = {}
        current_dict['order_size'] = row['order_size']
        current_day_model_data = Qi_wrapper.get_model_data(model=row['model'], start=date, end=date, term='Long term')
        current_real_value = current_day_model_data['Model Value'][0] + current_day_model_data['Absolute Gap'][0]
        current_dict['percent_return'] = ((current_real_value - row['real_value']) / row['real_value']) * 100
        dict_model_order_size_current_percent_return.append(current_dict)
# at this point I have a dictionary of dictionaries where the keys are the model name and the values are a dictionary
# containing the order_size and percent_return for the trade it is currently in with that model
# to find the account value I need to store the cash in the account in the account_stats CSV file
    df_account_stats = pd.read_csv("account_stats.csv")
    for index, row in df_account_stats[len(df_account_stats) - 1:].iterrows():
        latest_day_cash = row['account_value']
    illiquid_value = 0
    for dic in dict_model_order_size_current_percent_return:
        illiquid_value += dic['order_size'] * ((100 + dic['percent_return']) / 100)

    value = 100000
    df_completed_trades = pd.read_csv("completed_trades_backtest_long.csv")
    for index, row in df_completed_trades.iterrows():
        buy = row['trade_entry_price']
        sell = row['trade_exit_price']
        order_size = row['order_size']
        percent_return = ((sell - buy) / buy) * 100
        value_added = order_size * (percent_return / 100)
        value += value_added
    # this code above iterates over the completed trades and finds the absolute profit on each tarde and adds that to 100000

    current_account_value = value

    # this chunk of code appends the date and account value to the end of CSV file
    df_account_tracker = pd.read_csv("account_stats.csv")
    new_row = pd.DataFrame({'date': [date], 'account_value': [current_account_value]})
    df_account_tracker = pd.concat([df_account_tracker, new_row], ignore_index=True)
    df_account_tracker.to_csv("account_stats.csv", index=False)

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
    dates_for_backtest = weekdays_between_dates('2013-05-22', '2013-06-01')
    for date in dates_for_backtest:
        find_models_to_buy_long_new_backtest(date)
        check_current_trades_long_new_backtest(date)
        find_account_value_and_update_account_value(date)
