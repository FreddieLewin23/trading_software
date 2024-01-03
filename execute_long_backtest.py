from backtest_on_one_stock import check_current_trades_long_new_backtest_bfo, find_models_to_buy_long_new_backtest_bfo, filter_csv_by_date_bfo
import os
import qi_client
import warnings
from datetime import datetime, timedelta
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['QI_API_KEY'] = 'aHUylOC5yM9xSRLpZs8Z45vHsxXClZNE4IW6rJ4n'
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = 'aHUylOC5yM9xSRLpZs8Z45vHsxXClZNE4IW6rJ4n'
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))

models_USD = [x.name
              for x in api_instance.get_models(tags='USD, Stock')
              if x.model_parameter == 'long term' and '_' not in x.name
              ][:3400]

def find_account_value_and_update_account_value(date):
    df = pd.read_csv("current_trade_bfo_backtest.csv")
    dict_model_order_size_current_percent_return = []
    for index, row in df.iterrows():
        current_dict = {}
        print(row['model'])
        current_dict['order_size'] = row['order_size']
        current_day_model_data = filter_csv_by_date_bfo(model=row['model'], start_date=date, end_date=date)
        if len(current_day_model_data) == 0:
            continue
        current_real_value = current_day_model_data['Model Value'][0] + current_day_model_data['Absolute Gap'][0]
        current_dict['percent_return'] = ((current_real_value - row['real_value']) / row['real_value']) * 100
        dict_model_order_size_current_percent_return.append(current_dict)
# at this point I have a dictionary of dictionaries where the keys are the model name and the values are a dictionary
# containing the order_size and percent_return for the trade it is currently in with that model
# to find the account value I need to store the cash in the account in the account_stats CSV file
    df_account_stats = pd.read_csv("account_stats_bfo.csv")
    for index, row in df_account_stats[len(df_account_stats) - 1:].iterrows():
        latest_day_cash = row['account_value']
    illiquid_value = 0
    for dic in dict_model_order_size_current_percent_return:
        illiquid_value += dic['order_size'] * ((100 + dic['percent_return']) / 100)

    value = 100000
    df_completed_trades = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/bfo_backtest/completed_trades_backtest_long.csv')
    for index, row in df_completed_trades.iterrows():
        buy = row['trade_entry_price']
        sell = row['trade_exit_price']
        order_size = row['order_size']
        percent_return = ((sell - buy) / buy) * 100
        value_added = order_size * (percent_return / 100)
        value += value_added
    current_account_value = value
    df_account_tracker = pd.read_csv("account_stats_bfo.csv")
    new_row = pd.DataFrame({'date': [date], 'account_value': [current_account_value]})
    df_account_tracker = pd.concat([df_account_tracker, new_row], ignore_index=True)
    df_account_tracker.to_csv("account_stats_bfo.csv", index=False)
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
    dates_for_backtest = weekdays_between_dates('2023-12-14', '2024-01-30')
    for date in dates_for_backtest:
        find_models_to_buy_long_new_backtest_bfo(date)
        check_current_trades_long_new_backtest_bfo(date)
        find_account_value_and_update_account_value(date
