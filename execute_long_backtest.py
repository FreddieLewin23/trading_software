import Qi_wrapper
from backtesting_20251001.backtest_one_day import (check_current_trades,
                                                   find_models_to_buy,
                                                   filter_csv_by_date_bfo)
import warnings
from datetime import datetime, timedelta
import pandas as pd


warnings.simplefilter(action='ignore', category=FutureWarning)


def find_account_value_and_update_account_value(date):
    df_curr = pd.read_csv("/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20251001/data/current_trades.csv")
    current_trade_illiquid = sum(df_curr['order_size'].tolist())
    value_of_investment = 0
    for index, row in df_curr.iterrows():
        buy_price = row['real_value']
        trade_data_curr = filter_csv_by_date_bfo(model=row['model'], start_date=date, end_date=date)
        # this just says, when adding unrealised gains to the account, if there is no model data
        if len(trade_data_curr) == 0:
            trade_data_curr = Qi_wrapper.get_model_data(model=row['model'], start=date, end=date, term='Long term')
        if len(trade_data_curr) == 0:
            sell_price = buy_price
        else:
            sell_price = trade_data_curr['Model Value'][0] + trade_data_curr['Absolute Gap'][0]
        value_of_investment += row['order_size'] * (1 + ((sell_price - buy_price) / buy_price))
    change_in_current_investments = value_of_investment - current_trade_illiquid

    value = 100_000
    df_completed_trades = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20251001/data/completed_trades.csv')
    for index, row in df_completed_trades.iterrows():
        buy = row['trade_entry_price']
        sell = row['trade_exit_price']
        order_size = row['order_size']
        percent_return = ((sell - buy) / buy) * 100
        value_added = order_size * (percent_return / 100)
        value += value_added
    current_account_value = value + change_in_current_investments
    df_account_tracker = pd.read_csv("/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20251001/data/account_value.csv")
    new_row = pd.DataFrame({'date': [date], 'account_value': [current_account_value]})
    df_account_tracker = pd.concat([df_account_tracker, new_row], ignore_index=True)
    df_account_tracker.to_csv("/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20251001/data/account_value.csv", index=False)
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


curr_pitc_model_data = None


if __name__ == '__main__':
    dates_for_backtest = weekdays_between_dates('2013-01-01', '2024-01-01')
    for date in dates_for_backtest:
        print(date)
        find_models_to_buy(date)
        check_current_trades(date)
        find_account_value_and_update_account_value(date)
