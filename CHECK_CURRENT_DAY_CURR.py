import os
import json
import pandas as pd
import qi_client
from datetime import datetime, timedelta
import csv
import numpy as np
import concurrent.futures
import statistics
import warnings
import math
import threading
import Qi_wrapper
import alpaca.common.exceptions
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest, OrderSide
from alpaca.trading.enums import OrderSide, TimeInForce


warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['QI_API_KEY'] = 'aHUylOC5yM9xSRLpZs8Z45vHsxXClZNE4IW6rJ4n'
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = 'aHUylOC5yM9xSRLpZs8Z45vHsxXClZNE4IW6rJ4n'
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))


API_KEY = "PKKQTM1Y95DE79L3FSLD"
SECRET_KEY = "1au0qfRaewvYWbrTR0XEZvINHnv0MkKASXhmbzfD"
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
clock = trading_client.get_clock()
account = trading_client.get_account()
positions = trading_client.get_all_positions()
orders = trading_client.get_orders()



def subtract_days_from_date(input_date_str, n):
    try:
        input_date = datetime.strptime(input_date_str, "%Y-%m-%d")
    except ValueError:
        # Handle invalid date format here
        return input_date_str  # or raise an exception
    weekdays_to_subtract = n + 1
    while weekdays_to_subtract > 0:
        input_date -= timedelta(days=1)
        if input_date.weekday() < 5:
            weekdays_to_subtract -= 1
    if input_date.month == 2 and input_date.day == 29:
        if not (input_date.year % 4 == 0 and (input_date.year % 100 != 0 or input_date.year % 400 == 0)):
            input_date = input_date.replace(day=27)
    new_date_str = input_date.strftime("%Y-%m-%d")
    return new_date_str


def qi_vol_indicator(date):
    indexes = ['S&P500', 'Euro Stoxx 600', 'USDJPY', 'EURUSD', 'USD 10Y Swap', 'EUR 10Y Swap']
    st_rsqs = []
    for model in indexes:
        data = Qi_wrapper.get_model_data(model=model, start=date, end=date, term='Short term')
        if len(data) == 0:
            continue
        st_rsqs.append(data.iloc[-1]['Rsq'])
    if len(st_rsqs) == 0:
        return 3
    return 100 - np.mean(st_rsqs)


def vol_individual_new(model, date):
    df = Qi_wrapper.get_model_data(model=model, start=subtract_days_from_date(date, 365), end=date, term='Long term')
    if len(df) == 0:
        return 0
    df['Asset Price'] = df['Model Value'] + df['Absolute Gap']
    df['Daily Percentage Change'] = df['Asset Price'].pct_change() * 100
    std_dev = df['Daily Percentage Change'].std()
    return std_dev


def find_sd_from_model_data(model, date):
    # this code extracts the data from the JSON file that is needed
    end_date = str(date).split()[0]
    year, month, day = end_date.split('-')
    if month == '02' and day == '29':
        day = '28'
    year = str(int(year) - 1)
    start_date = f"{year}-{month}-{day}"
    # increments year back one to find start date for monthly rolling average
    df = Qi_wrapper.get_model_data(model=model, start=start_date, end=end_date, term='Long term')
    monthly_return = []
    if len(df) == 0:
        return 10
    # split rows in to indexes
    groups = df.groupby(df.index.to_period('M'))
    # group-by month
    for group_name, group_df in groups:
        first_row = group_df.iloc[0]
        last_row = group_df.iloc[-1]
        # first last date of each month
        start_real_value = first_row[2] + first_row[4]
        end_real_value = last_row[2] + last_row[4]
        # find real_value from model value and absolute gap
        monthly_return.append(((end_real_value - start_real_value) / start_real_value) * 100)
    if len(monthly_return) < 2:
        return 10  # Handle the case where there are not enough data points for variance calculation
    return statistics.stdev(monthly_return)


def fvg_backtest_long(model, start, end, threshold_buy, threshold_sell, Rsq):
    df = Qi_wrapper.get_model_data(model=model, start=start, end=end, term='Long term')
    # df = Qi_wrapper.get_model_data(model=model, start=start, end=end, term='Long term')
    if len(df) == 0:
        return []
    trades_dict = []
    trades = []
    fvg_at_buy = None
    buy = float('inf')
    day_count = 0
    std = find_sd_from_model_data(model, start)
    if not std:
        return []
    stop_loss_change = 2 * std
    stop_loss_count = 0
    rsq_value_at_buy = 0
    buy_date = None
    for index, row in df.iterrows():
        if day_count == 60:
            stop_loss_change = 2 * find_sd_from_model_data(model, str(index).split()[0])
            day_count = 0
        real_value = row.iloc[2] + row.iloc[4]
        # row.iloc[2] + row.iloc[4] this is model value + absolute gap to find the real_value at this day
        fvg_value = row['FVG']
        rsq_value = row['Rsq']
        # take the column of the fvg_value for that stock
        # need to add stop-loss trades into the trades array
        if real_value < buy - stop_loss_change and buy != float("inf"):
            stop_loss_date = index
            days_between = (stop_loss_date - buy_date).days
            # days between sell date and buy date
            trades.append([buy, real_value, days_between, rsq_value, rsq_value_at_buy, stop_loss_date, buy_date])
            trades_dict.append({'Real_value_at_buy': buy, 'Real_value_at_sell': real_value,
                                'Days between trades': days_between, 'Rsq at sell': rsq_value,
                                'Rsq at buy': rsq_value_at_buy,
                                'Sell_date': stop_loss_date, 'FVG value at buy': fvg_at_buy, 'FVG value at sell': fvg_value})
            buy = float("inf")
            # reset buy to initialised value
            stop_loss_count += 1
            buy_date = None
        if fvg_value > threshold_sell and buy != float("inf"):
            sell_date = index
            # print(buy_date, sell_date, ((real_value - buy) / buy) * 100)
            days_between = (sell_date - buy_date).days
            trades_dict.append({'Real_value_at_buy': buy, 'Real_value_at_sell': real_value,
                                'Days between trades': days_between, 'Rsq at sell': rsq_value,
                                'Rsq at buy': rsq_value_at_buy,
                                'Sell_date': sell_date, 'FVG value at buy': fvg_at_buy, 'FVG value at sell': fvg_value})
            trades.append([buy, real_value, days_between, rsq_value, rsq_value_at_buy, sell_date, buy_date])
            buy = float('inf')
            buy_date = None
            rsq_value_at_buy = 0
        if fvg_value < threshold_buy and buy == float('inf') and rsq_value > Rsq:
            buy = real_value
            fvg_at_buy = fvg_value
            buy_date = index
            rsq_value_at_buy = rsq_value
        day_count += 1
    if len(trades) == 0:
        return []
    trades_profit_percentage = [((x[1] - x[0]) / x[0]) * 100 for x in trades]
    average_percentage_return = np.mean(trades_profit_percentage)
    percentage_profitable = (len([num for num in trades_profit_percentage
                                  if num > 0]) / len(trades_profit_percentage)) * 100
    days_for_trades = [x[2] for x in trades]
    # rsq_value_at_buy_average = sum([x[4] for x in trades]) / len(trades)
    # rsq_value_at_sell_average = sum([x[3] for x in trades]) / len(trades)
    results = [round(average_percentage_return, 3), len(trades_profit_percentage), round(percentage_profitable, 3),
               stop_loss_count, round(np.mean(days_for_trades), 3),
               trades, stop_loss_count, trades_dict]
    # Convert the results list into a 2D array-like structure with one row and multiple columns
    data = [results]
    # Create the DataFrame
    df_results = pd.DataFrame(data,
                              columns=['Avg. Rtrn', 'No. of Trades', 'Hit rate', 'Stop-loss trigger', 'Holding time',
                                       'Trades', 'Stop Loss Count', 'Trades as dic'])
    return results


def backtest_score_long(results_from_backtest):
    if not results_from_backtest:
        return 10
    average_returns = results_from_backtest[0]
    hit_rate = results_from_backtest[2]
    return_score = 5 + math.ceil((average_returns - 2.3) / 0.5)
    if return_score < 0:
        return_score = 0
    if return_score > 10:
        return_score = 10
    hit_rate_score = 5 + math.ceil((hit_rate - 60) / 4)
    if hit_rate_score < 0:
        hit_rate_score = 0
    if hit_rate_score > 10:
        hit_rate_score = 10
    return hit_rate_score + return_score


def mvg_current_n_day_diff(model, date, look_back):
    '''
    no longer used
    :param model:
    :param date:
    :param look_back:
    :return:
    '''
    date = str(date).split()[0]
    start_date = subtract_days_from_date(date, look_back)
    data_from_look_back = Qi_wrapper.get_model_data(model=model, start=start_date, end=start_date, term='Long term')
    data_current = Qi_wrapper.get_model_data(model=model, start=date, end=date, term='Long term')
    if len(data_current) == 0 or len(data_from_look_back) == 0:
        return 0.01
    model_value_look_back = data_from_look_back['Model Value'][0]
    absolute_gap_look_back = data_from_look_back['Absolute Gap'][0]
    real_value_look_back = model_value_look_back + absolute_gap_look_back

    if len(data_current) == 0:
        return 0.01
    model_value_look_back = data_current['Model Value'][0]
    absolute_gap_look_back = data_current['Absolute Gap'][0]
    real_value_current = model_value_look_back + absolute_gap_look_back

    return (real_value_current - real_value_look_back) / look_back


def mvg_current_linear_regression(model, date, look_back):
    date = str(date).split()[0]
    start_date = subtract_days_from_date(date, look_back)
    data_df = Qi_wrapper.get_model_data(model=model, start=start_date, end=date, term='Long term')
    if len(data_df) == 0:
        return 0.01
    model_values = []
    for index, row in data_df.iterrows():
        model_values.append(row['Model Value'])
    x_values = np.array(list(range(len(model_values))))
    y_values = np.array(model_values)
    # polyfit x and y values and deg is the number of degrees of the power series you want. UNREAL ENDPOINT
    slope, _ = np.polyfit(x_values, y_values, 1)
    return slope


def price_trend_score(model, date_of_trade_entry):
    three_day = mvg_current_linear_regression(model=model, date=date_of_trade_entry, look_back=3)
    ten_day = mvg_current_linear_regression(model=model, date=date_of_trade_entry, look_back=10)
    thirty_day = mvg_current_linear_regression(model=model, date=date_of_trade_entry, look_back=30)
    mvgs = [three_day, ten_day, thirty_day]
    return (len([num for num in mvgs if num > 0]) / len(mvgs)) * 100


def find_base_order_size():
    account_value = float(account.non_marginable_buying_power)
    return int(account_value) / 254  # over the 11 year backtest the average number of active trades was 127.1. this can be improved upon


def current_max_leverage_new(date):
    current_volatility = qi_vol_indicator(date)
    base_leverage = 1
    average_volatility = 29.44
    adjustment_factor = max(0.5, 1 - (current_volatility - average_volatility) / average_volatility)
    adjusted_max_leverage = base_leverage * adjustment_factor
    return min(2, adjusted_max_leverage)





def quantify_order_size_vol_adjusted_new(model, date_of_trade_entry):
    start_date = subtract_days_from_date(input_date_str=date_of_trade_entry, n=1800)
    res = fvg_backtest_long(model=model, start=start_date, end=date_of_trade_entry,
                            threshold_buy=-1, threshold_sell=-0.25, Rsq=65)
    bt_score = backtest_score_long(res)
    pt_score = price_trend_score(model=model, date_of_trade_entry=date_of_trade_entry)
    price_data = Qi_wrapper.get_model_data(model=model, start=date_of_trade_entry, end=date_of_trade_entry,
                                           term='Long term')
    if len(price_data) == 0:
        date_new = subtract_days_from_date(input_date_str=date_of_trade_entry, n=1)
        price_data = Qi_wrapper.get_model_data(model=model, start=date_new, end=date_new, term='Long term')
    model_value = price_data['Model Value'][0]
    absolute_gap = price_data['Absolute Gap'][0]
    real_value = model_value + absolute_gap
    if 0 <= bt_score < 2.5: bt_coefficient = 0.75
    elif 2.5 <= bt_score < 5: bt_coefficient = 1
    elif 5 <= bt_score < 7.5: bt_coefficient = 1.25
    else: bt_coefficient = 1.5
    if 0 <= pt_score < 25: pt_coefficient = 0.75
    elif 25 <= pt_score < 50: pt_coefficient = 1
    elif 50 <= pt_score < 75: pt_coefficient = 1.25
    else: pt_coefficient = 1.5
    base_order_size = float(account.equity) * 1.5 / 127
    individual_vol = vol_individual_new(model=model, date=date_of_trade_entry)
    amount_to_buy = base_order_size * bt_coefficient * pt_coefficient * (1 - ((individual_vol - 3.26) / individual_vol))
    return [round(amount_to_buy / real_value, 4), amount_to_buy]


def find_models_to_buy_long():
    models_USD = [x.name
                  for x in api_instance.get_models(tags='USD, Stock')
                  if x.model_parameter == 'long term' and '_' not in x.name
                  ][:3400]
    new_trade_models = []
    currently_open_trades = []
    i = 0
    df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/current_trades_alpaca.csv')
    for index, row in df.iterrows():
        currently_open_trades.append(row['model'])
    today_date = str(pd.Timestamp.today(tz='America/New_York').date().isoformat()).split()[0]
    curr_max_leverage = current_max_leverage_new(date=today_date)
    illiquid = float(account.long_market_value)
    for model in models_USD:
        print(f'{i} / {len(models_USD)} : {str(round(i / len(models_USD), 5) * 100)[:5]}%')
        i += 1

        if model == "NLTX":
            continue
        current_model_data = Qi_wrapper.get_model_data(model=model, start=today_date, end=today_date, term='Long term')
        if model in currently_open_trades:
            continue
        if len(current_model_data) == 0:
            continue
        today_fvg = current_model_data['FVG'][0]
        today_rsq = current_model_data['Rsq'][0]
        if float(today_rsq) >= 65 and float(today_fvg) <= -1:
            print('__________________________')
        model_value = current_model_data['Model Value'][0]
        absolute_gap = current_model_data['Absolute Gap'][0]
        real_value = model_value + absolute_gap
        if today_rsq > 65 and today_fvg < -1:
            backtest_data = fvg_backtest_long(model=model, start='2017-03-03', end='2023-12-29',
                                              threshold_buy=-1, threshold_sell=-0.25, Rsq=65)
            if len(backtest_data) == 0:
                print(f'No similar trades in last 5 years, too risky')
                continue
            backtest_score_today = backtest_score_long(backtest_data)
            print(f'Backtest Score: {backtest_score_today}')
            if illiquid <= curr_max_leverage * float(account.equity):
                # if the amount invested is greater than the max leverage * equity, do not enter trade
                print(f'Invested: {illiquid}, Max Invested Current: {curr_max_leverage * float(account.equity)}')
                if backtest_score_today > 5:
                    price_trend_score_curr = price_trend_score(model=model, date_of_trade_entry=today_date)
                    print(f'Price Trend Score: {price_trend_score_curr}')
                    if price_trend_score_curr > 25:
                        print(f'Individual Volatility: {vol_individual_new(model=model, date=today_date)}')
                        order_size = quantify_order_size_vol_adjusted_new(model=model, date_of_trade_entry=today_date)[1]
                        new_trade_models.append([model, real_value, today_date, order_size])
                        print(f'Market Volatility: {qi_vol_indicator(date=today_date)}')
                        print(f'START TRADE WITH {model} FVG:{today_fvg}, Rsq:{today_rsq}')
                        illiquid += order_size

    # SANM, CXT, SHYF, CPRT,
    csv_file = '/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/current_trades_alpaca.csv'
    file_exists = os.path.isfile(csv_file)
    file_is_empty = not file_exists or os.stat(csv_file).st_size == 0
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['model', 'real_value', 'today_date', 'order_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if file_is_empty:
            writer.writeheader()
        for model_data in new_trade_models:
            writer.writerow({'model': model_data[0], 'real_value': model_data[1],
                             'today_date': model_data[2], 'order_size': model_data[3]})

    model_amount_dict = {}
    for data in new_trade_models:
        model_amount_dict[data[0]] = data[3]

    return [new_trade_models, model_amount_dict]


def check_current_trades_long():
    currently_open_trades = []
    df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/current_trades_alpaca.csv')
    data_count = 0
    for index, row in df.iterrows():
        currently_open_trades.append([row['model'], row['real_value']])
    #     now i have an array of model names of model with trades open on the demo account
    models_to_exit = []
    i = 0
    for model in currently_open_trades:
        i += 1

        today_date = str(pd.Timestamp.today(tz='America/New_York').date().isoformat()).split()[0]
        current_model_data = Qi_wrapper.get_model_data(model=model[0], start=today_date, end=today_date, term='Long term')
        if len(current_model_data) == 1:
            data_count += 1
        # print(f'{currently_open_trades.index(model[0])} / {len(currently_open_trades)}: {(currently_open_trades.index(model[0]) / len(currently_open_trades)) * 100} ')

        if len(current_model_data) == 0:
            continue
        model_value = current_model_data['Model Value'][0]
        absolute_gap = current_model_data['Absolute Gap'][0]
        today_fvg = current_model_data['FVG'][0]
        real_value = model_value + absolute_gap
        print(f'{i} / {len(currently_open_trades)} FVG: {today_fvg}')
        stop_loss_change = 2 * find_sd_from_model_data(model=model[0], date=today_date)
        if real_value < model[1] - stop_loss_change:
            print(f'END TRADE WITH {model}: STOP-LOSS')
            currently_open_trades = [subarray for subarray in currently_open_trades if subarray[0] != model[0]]
            models_to_exit.append([model, real_value, today_date])
# this removes a model from the currently open models if a stop loss is triggered
        if today_fvg > -0.25:
            print(f'END TRADE WITH {model}: FVG')
            currently_open_trades = [subarray for subarray in currently_open_trades if subarray[0] != model[0]]
            models_to_exit.append([model, real_value, today_date])

    # models to exit is an array of arrays with the trade exit data like this [model, real_value_at_exit, date_at_exit]

    data_for_completed_trades = []
    # i want this to look like this [model, trade_entry_price, trade_exit_price, trade_entry_date, trade_exit_date]
    # i will the niterate through the currently open_trades and grab bhuy_date_date = [row['real_value], row['today_date']
    # to do this i will need to iterate through models_to_exit and append to data array [models_to_exit[0], ,models_to_exit[1], , models_to_exit[2]]

    curr_df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/current_trades_alpaca.csv')
    # this gets the order_sizes
    model_order_size_dict = {}
    for index, row in curr_df.iterrows():
        model_order_size_dict[row['model']] = row['order_size']

    # this gathers the data from the currently open trades for trades that need to be ended
    df_open_trades = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/current_trades_alpaca.csv')
    data_from_currently_open_trades = []
# models_to_exit contains all the model names for the trades that need to be ended
    for model in models_to_exit:
        model_name = model[0][0]
        for index, row in df_open_trades.iterrows():
            if row['model'] == model_name:
                model_symbol = row['model']
                trade_entry_value = row['real_value']
                trade_entry_date = row['today_date']
                data_from_currently_open_trades.append([model_symbol, trade_entry_value, trade_entry_date])
    # data_from_currently_open_trades contains data for the trades that need to be ended

    # this adds all the data into array for the completed trades csv file
    for model in models_to_exit:
        for data_point in data_from_currently_open_trades:
            if data_point[0] == model[0][0]:
                data_for_completed_trades.append([model[0][0], data_point[1],
                                                  model[1], data_point[2], model[2]])
    #data for completed trades is an array of array with model name, buy, sell, buy date, sell date

    # this writes ot the completed trades to add the complete trades
    print(data_for_completed_trades)
    existing_data = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/completed_trades_alpaca_local.csv')
    new_data = pd.DataFrame(data_for_completed_trades, columns=existing_data.columns)
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    combined_data.to_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/completed_trades_alpaca_local.csv', index=False)
    # this adds the completed trades to the completed trades array

    current_trades_df = pd.read_csv("/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/current_trades_alpaca.csv")
    model_names_to_exit = [subarray[0][0] for subarray in models_to_exit]
    print(current_trades_df)
    trade_data_to_keep = []
    for index, row in current_trades_df.iterrows():
        if row['model'] not in model_names_to_exit:
            trade_data_to_keep.append([row['model'], row['real_value'], row['today_date'], row['order_size']])

    df_new = pd.DataFrame(trade_data_to_keep, columns=['model', 'real_value', 'today_date', 'order_size'])
    print(df_new)
    df_new.to_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/current_trades_alpaca.csv', index=False)

    print(f'{data_count} / {len(currently_open_trades)} had model data')

    return [models_to_exit, model_order_size_dict]




