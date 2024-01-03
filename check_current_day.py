import os
import json
import pandas as pd
import qi_client
import datetime
import csv
import numpy as np
import concurrent.futures
import statistics
import warnings
import math
import threading
import Qi_wrapper
from backtest_json_redo_MVG import grab_data2, find_sd_from_model_data, subtract_days_from_date

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['QI_API_KEY'] = ''
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = ''
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))


def fvg_backtest_long(model, start, end, threshold_buy, threshold_sell, Rsq):
    df = grab_data2(model=model, start=start, end=end)
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


def backtest_score(results_from_backtest):
    if not results_from_backtest:
        return 10
    average_returns = results_from_backtest[0]
    hit_rate = results_from_backtest[2]
    return_score = 5 + math.ceil((2.3 - average_returns) / 0.5)
    if return_score < 0:
        return_score = 0
    if return_score > 10:
        return_score = 10
    hit_rate_score = 5 + math.ceil((60 - hit_rate) / 4)
    if hit_rate_score < 0:
        hit_rate_score = 0
    if hit_rate_score > 10:
        hit_rate_score = 10
    return hit_rate_score + average_returns


def mvg_current(model, date, look_back):
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


def price_trend_score(model, date_of_trade_entry):
    three_day = mvg_current(model=model, date=date_of_trade_entry, look_back=3)
    ten_day = mvg_current(model=model, date=date_of_trade_entry, look_back=10)
    thirty_day = mvg_current(model=model, date=date_of_trade_entry, look_back=30)
    mvgs = [three_day, ten_day, thirty_day]
    return (len([num for num in mvgs if num > 0]) / len(mvgs)) * 100


def quantify_buy_amount(model, date_of_trade_entry):
    res = fvg_backtest_long(model=model, start='2019-03-05', end='2023-01-01',
                            threshold_buy=-1, threshold_sell=-0.25, Rsq=65)
    bt_score = backtest_score(res) #from 0 to 10, 5 being average
    pt_score = price_trend_score(model=model, date_of_trade_entry=date_of_trade_entry)
#     i want the real value of the stock on the date and find (to the nearest integer) the number of stocks i want to buy
#     OKAY THE ERROR IS HAPPENING BECAUSE I AM USING THE DATA FROM CSV WHICH ISNT CURRENT.
#     NEED TO CREATE NEW QI API BASED MVG FUNCTION
    price_data = Qi_wrapper.get_model_data(model=model, start=date_of_trade_entry, end=date_of_trade_entry, term='Long term')
    if len(price_data) == 0:
        date_new = subtract_days_from_date(input_date_str=date_of_trade_entry, n=1)
        price_data = Qi_wrapper.get_model_data(model=model, start=date_new, end=date_new, term='Long term')
    model_value = price_data['Model Value'][0]
    absolute_gap = price_data['Absolute Gap'][0]
    real_value = model_value + absolute_gap

    if 0 <= bt_score < 2.5:
        bt_coefficient = 0.75
    elif 2.5 <= bt_score < 5:
        bt_coefficient = 1
    elif 5 <= bt_score < 7.5:
        bt_coefficient = 1.25
    else:
        bt_coefficient = 1.5
        
    if 0 <= pt_score < 25:
        pt_coefficient = 0.75
    elif 25 <= pt_score < 50:
        pt_coefficient = 1
    elif 50 <= pt_score < 75:
        pt_coefficient = 1.25
    else:
        pt_coefficient = 1.5

    amount_to_buy = 2000 * bt_coefficient * pt_coefficient
    return [math.ceil(amount_to_buy / real_value), amount_to_buy]


models_USD = [x.name
          for x in api_instance.get_models(tags='USD, Stock')
          if x.model_parameter == 'long term' and '_' not in x.name
          ][:3400]

def find_models_to_buy():
    new_trade_models = []
    currently_open_trades = []
    i = 0
    df = pd.read_csv('currently_open_trades.csv')
    for index, row in df.iterrows():
        currently_open_trades.append(row['model'])
    for model in models_USD:
        print(f'{i} / {len(models_USD)} : {str(round(i / len(models_USD), 5) * 100)[:5]}%')
        i += 1
        today_date = str(pd.Timestamp.today(tz='America/New_York').date().isoformat()).split()[0]
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

        # backtest_data = fvg_backtest_long(model=model, start='2019-03-03', end='2023-01-01',
        #                                   threshold_buy=-1, threshold_sell=-0.25, Rsq=65)
        # if len(backtest_data) == 0:
        #     continue

        if today_rsq > 65 and today_fvg < -1:

            backtest_data = fvg_backtest_long(model=model, start='2019-03-03', end='2023-01-01',
                                              threshold_buy=-1, threshold_sell=-0.25, Rsq=65)
            if len(backtest_data) == 0:
                continue

            backtest_score_today = backtest_score(backtest_data)
            # remove next 3 lines of code when using price trends again
            if backtest_score_today > 2:
                new_trade_models.append([model, real_value, today_date, quantify_buy_amount(model=model, date_of_trade_entry=today_date)[1]])
                print(f'START TRADE WITH {model} FVG:{today_fvg}, Rsq:{today_rsq}')

    csv_file = 'currently_open_trades.csv'
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


def check_current_trades():
    currently_open_trades = []
    df = pd.read_csv('currently_open_trades.csv')
    data_count = 0
    for index, row in df.iterrows():
        currently_open_trades.append([row['model'], row['real_value']])
    #     now i have an array of model names of model with trades open on the demo account
    models_to_exit = []
    i = 0
    for model in currently_open_trades:
        i += 1
        print(f'{i} / {len(currently_open_trades)}')
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

    curr_df = pd.read_csv('currently_open_trades.csv')
    # this gets the order_sizes
    model_order_size_dict = {}
    for index, row in curr_df.iterrows():
        model_order_size_dict[row['model']] = row['order_size']

    # this gathers the data from the currently open trades for trades that need to be ended
    df_open_trades = pd.read_csv('currently_open_trades.csv')
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
    existing_data = pd.read_csv('completed_trades_alpaca.csv')
    new_data = pd.DataFrame(data_for_completed_trades, columns=existing_data.columns)
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    combined_data.to_csv('completed_trades_alpaca.csv', index=False)
    # this adds the completed trades to the completed trades array

    current_trades_df = pd.read_csv("currently_open_trades.csv")
    model_names_to_exit = [subarray[0][0] for subarray in models_to_exit]
    print(current_trades_df)
    trade_data_to_keep = []
    for index, row in current_trades_df.iterrows():
        if row['model'] not in model_names_to_exit:
            trade_data_to_keep.append([row['model'], row['real_value'], row['today_date'], row['order_size']])

    df_new = pd.DataFrame(trade_data_to_keep, columns=['model', 'real_value', 'today_date', 'order_size'])
    print(df_new)
    df_new.to_csv('currently_open_trades.csv', index=False)

    print(f'{data_count} / {len(currently_open_trades)} had model data')

    return [models_to_exit, model_order_size_dict]








