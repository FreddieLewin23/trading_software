import os
import pandas as pd
import qi_client
import random
from dateutil.relativedelta import relativedelta
from datetime import datetime
import csv
import numpy as np
import warnings
import Qi_wrapper
from alpaca.trading.client import TradingClient


warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['QI_API_KEY'] = ''
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = ''
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))


# _________LONG TERM PAPER ACCOUNT________________________________________
API_KEY = ""
SECRET_KEY = ""
# ________________________________________________________________________

# MESS AROUND PAPER ACCOUNT________________________________________
# API_KEY = ""
# SECRET_KEY = ""
# ________________________________________________________________________


trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
clock = trading_client.get_clock()
account = trading_client.get_account()
positions = trading_client.get_all_positions()
orders = trading_client.get_orders()


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
    end_date_dt = pd.to_datetime(date, format='%Y-%m-%d')
    start_date_dt = end_date_dt - relativedelta(years=1)
    df = Qi_wrapper.get_model_data(model=model, start=str(start_date_dt)[:10], end=date, term='Long term')
    if len(df) == 0:
        return 0
    df['Asset Price'] = df['Model Value'] + df['Absolute Gap']
    df['Daily Percentage Change'] = df['Asset Price'].pct_change() * 100
    std_dev = df['Daily Percentage Change'].std()
    return std_dev


def stddev_model_data(model, date):
    end_date_str = str(date).split()[0]
    start_date_dt = pd.to_datetime(end_date_str, format='%Y-%m-%d') - relativedelta(years=1)
    start_date_str = str(start_date_dt)[:10]
    df = Qi_wrapper.get_model_data(model=model, start=start_date_str, end=end_date_str, term='Long term')
    df['Real Value'] = df['Model Value'] + df['Absolute Gap']
    df.reset_index(inplace=True)
    df['Date'] = df['index']
    df = df[['Date', 'Real Value']]
    df['Log Return'] = np.log(df['Real Value'] / df['Real Value'].shift(1))
    df = df.dropna()
    std_dev = df['Log Return'].std()
    return std_dev * 440  # normalising


def fvg_backtest_long(model, start, end, threshold_buy, threshold_sell, Rsq):
    start_str = str(start)[:10]
    end_str = str(end)[:10]
    df = Qi_wrapper.get_model_data(model=model, start=start_str, end=end_str, term='Long term')
    df['Real Value'] = df['Model Value'] + df['Absolute Gap']
    if len(df) == 0:
        return []
    trades = []
    fvg_at_buy = None
    buy = float('inf')
    day_count = 0
    std = stddev_model_data(model, start)
    if not std:
        return []
    stop_loss_change = 2 * std
    stop_loss_count = 0
    rsq_value_at_buy = 0
    buy_date = None

    for index, row in df.iterrows():
        if day_count == 60:
            stop_loss_change = 2 * stddev_model_data(model, str(index).split()[0])
            day_count = 0
        real_value = row['Real Value']
        fvg_value = row['FVG']
        rsq_value = row['Rsq']
        if real_value < buy - stop_loss_change and buy != float("inf"):
            stop_loss_date = index
            days_between = (stop_loss_date - buy_date).days
            trades.append([buy, real_value, days_between, rsq_value, rsq_value_at_buy, stop_loss_date, buy_date])
            buy = float("inf")
            stop_loss_count += 1
            buy_date = None
        if fvg_value > threshold_sell and buy != float("inf"):
            sell_date = index
            days_between = (sell_date - buy_date).days
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

    trades_dict = []
    results = [round(average_percentage_return, 3), len(trades_profit_percentage), round(percentage_profitable, 3),
               stop_loss_count, round(np.mean(days_for_trades), 3),
               trades, stop_loss_count, trades_dict]
    return results


def backtest_score_long_continuous(results_from_backtest):
    if not results_from_backtest:
        return 4.99
    average_returns = results_from_backtest[0]
    hit_rate = results_from_backtest[2]
    return_score = 5 + (average_returns - 2.3) / 0.5
    return_score = np.clip(return_score, 0, 10)
    hit_rate_score = 5 + (hit_rate - 60) / 4
    hit_rate_score = np.clip(hit_rate_score, 0, 10)
    overall = hit_rate_score + return_score
    overall_floored = max(0, overall)
    overall_capped = min(10, overall_floored)
    return overall_capped


def mvg_current_linear_regression(model, date, look_back):
    date = str(date).split()[0]
    end_date_dt = pd.to_datetime(date, format='%Y-%m-%d')
    start_date = end_date_dt - relativedelta(days=look_back)
    start_date_str = str(start_date)[:10]
    end_date_str = str(end_date_dt)[:10]
    data_df = Qi_wrapper.get_model_data(model=model, start=start_date_str, end=end_date_str, term='Long term')
    if len(data_df) == 0:
        return 0.01
    model_values = []
    for index, row in data_df.iterrows():
        model_values.append(row['Model Value'])
    x_values = np.array(list(range(len(model_values))))
    y_values = np.array(model_values)
    slope, _ = np.polyfit(x_values, y_values, 1)
    return slope


def price_trend_score(model, date_of_trade_entry):
    three_day = mvg_current_linear_regression(model=model, date=date_of_trade_entry, look_back=3)
    ten_day = mvg_current_linear_regression(model=model, date=date_of_trade_entry, look_back=10)
    thirty_day = mvg_current_linear_regression(model=model, date=date_of_trade_entry, look_back=30)
    mvgs = [three_day, ten_day, thirty_day]
    return (len([num for num in mvgs if num > 0]) / len(mvgs)) * 100


def current_max_leverage_new(date):
    current_volatility = qi_vol_indicator(date)
    base_leverage = 1
    average_volatility = 29.44
    adjustment_factor = max(0.5, 1 - (current_volatility - average_volatility) / average_volatility)
    adjusted_max_leverage = base_leverage * adjustment_factor
    return min(2, adjusted_max_leverage)


def quantify_order_size(model, date_of_trade_entry):
    date_of_trade_entry_dt = pd.to_datetime(date_of_trade_entry, format='%Y-%m-%d')
    start_date_dt = date_of_trade_entry_dt - relativedelta(days=1800)
    start_date = str(start_date_dt)[:10]
    res = fvg_backtest_long(model=model, start=start_date, end=date_of_trade_entry, threshold_buy=-1,
                            threshold_sell=-0.25, Rsq=75)
    bt_score = backtest_score_long_continuous(res)
    pt_score = price_trend_score(model=model, date_of_trade_entry=date_of_trade_entry)
    price_data = Qi_wrapper.get_model_data(model=model, start=date_of_trade_entry, end=date_of_trade_entry,
                                           term='Long term')

    if len(price_data) == 0:
        date_of_trade_entry_dt = pd.to_datetime(date_of_trade_entry, format='%Y-%m-%d')
        date_new_dt = date_of_trade_entry_dt - relativedelta(days=1)
        date_new_str = str(date_new_dt)[:10]
        price_data = Qi_wrapper.get_model_data(model=model, start=date_new_str, end=date_new_str, term='Long term')
    model_value = price_data['Model Value'][0]
    absolute_gap = price_data['Absolute Gap'][0]
    real_value = model_value + absolute_gap

    bt_coefficient = 0.75 + (bt_score / 10) * (1.5 - 0.75)
    pt_coefficient = 0.75 + (pt_score / 100) * (1.5 - 0.75)

    base_order_size = float(account.equity) / 100

    individual_vol = vol_individual_new(model=model, date=date_of_trade_entry)
    amount_to_buy = base_order_size * bt_coefficient * pt_coefficient * (1 - ((individual_vol - 3.26) / individual_vol))
    return [round(amount_to_buy / real_value, 4), amount_to_buy]


def output_point_in_time_constituents(date):
    df_r3k = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/post_qi_meeting/R3K_monthly_historical_constituents.csv')
    curr = df_r3k[date]
    model_IDs = []
    for i in range(1, len(curr)):
        model_IDs.append(curr.iloc[i])
    model_bloomberg_tickers = []
    for model in model_IDs:
        if type(model).__name__ == 'float':
            continue
        name = model.split()[0]
        if name.isalpha():
            model_bloomberg_tickers.append(name)
    return model_bloomberg_tickers


def find_models_to_buy_long():
    models_USD = [x.name
                  for x in api_instance.get_models(tags='USD, Stock')
                  if x.model_parameter == 'long term' and '_' not in x.name]
    today = str(datetime.today())[:10]
    models_r3k = output_point_in_time_constituents('2024-01-01')
    models_USD = [model for model in models_USD if model in models_r3k]
    random.shuffle(models_USD)

    new_trade_models = []
    currently_open_trades = []
    i = 0
    df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/current_trades_refactor.csv')
    for index, row in df.iterrows():
        currently_open_trades.append(row['model'])

    models_USD = [model for model in models_USD if model not in currently_open_trades]

    today_date = str(pd.Timestamp.today(tz='America/New_York').date().isoformat()).split()[0]
    curr_max_leverage = current_max_leverage_new(date=today_date)
    max_investment = curr_max_leverage * float(account.equity)
    illiquid = float(account.long_market_value)
    for model in models_USD:
        print(f'{i} / {len(models_USD)} : {str(round(i / len(models_USD), 5) * 100)[:5]}%')
        i += 1
        # if max_investment (curr_max_leverage * float(account.equity)) <= illiquid (ie invested) do not iterate through
        if max_investment <= illiquid:
            print(f'Max Invested: {max_investment}. Current Invested: {illiquid}. Over leveraged')
            break

        current_model_data = Qi_wrapper.get_model_data(model=model, start=today_date, end=today_date, term='Long term')

        if len(current_model_data) == 0:
            print(f'Missing data for {i} / {len(currently_open_trades)}. Model: {model} ')
            continue

        today_fvg = current_model_data['FVG'][0]
        today_rsq = current_model_data['Rsq'][0]

        if float(today_rsq) >= 75 and float(today_fvg) <= -1:
            print('________________FVG<-1__RSq>75________________')

        if today_fvg > -1:
            continue
        if today_rsq < 75:
            continue
        model_value = current_model_data['Model Value'][0]
        absolute_gap = current_model_data['Absolute Gap'][0]
        real_value = model_value + absolute_gap

        # MVG CHECKS
        price_trend_score_curr = price_trend_score(model=model, date_of_trade_entry=today_date)
        print(f'Price Trend Score: {price_trend_score_curr}')
        if price_trend_score_curr < 50:
            print('FAILED PRICE TREND')
            continue

        # MICRO BACKTEST CHECKS
        backtest_data = fvg_backtest_long(model=model, start=datetime.today() - relativedelta(years=5),
                                          end=str(datetime.today()), threshold_buy=-1, threshold_sell=-0.25, Rsq=75)
        backtest_score_today = backtest_score_long_continuous(backtest_data)
        if len(backtest_data) == 0:
            print(f'No similar trades in last 5 years, too risky')
            continue
        print(f'Backtest Score: {backtest_score_today}')
        if backtest_score_today < 3:
            print('FAILED MICRO BACKTEST')
            continue

        order_size = quantify_order_size(model=model, date_of_trade_entry=today_date)[1]
        new_trade_models.append([model, real_value, today_date, order_size])
        print(f'START TRADE WITH {model} FVG:{today_fvg}, Rsq:{today_rsq}')
        illiquid += order_size

    csv_file = '/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/current_trades_refactor.csv'
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
    df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/current_trades_refactor.csv')
    data_count = 0
    for index, row in df.iterrows():
        currently_open_trades.append([row['model'], row['real_value']])
    #     now i have an array of model names of model with trades open on the demo account
    models_to_exit = []
    i = 0
    today_date = str(pd.Timestamp.today(tz='America/New_York').date().isoformat()).split()[0]
    for model in currently_open_trades:
        i += 1
        current_model_data = Qi_wrapper.get_model_data(model=model[0], start=today_date, end=today_date, term='Long term')
        if len(current_model_data) == 1:
            data_count += 1

        if len(current_model_data) == 0:
            print(f'Missing data for Model: {model[0]} ')
            continue
        model_value = current_model_data['Model Value'][0]
        absolute_gap = current_model_data['Absolute Gap'][0]
        today_fvg = current_model_data['FVG'][0]
        real_value = model_value + absolute_gap
        print(f'{i} / {len(currently_open_trades)} FVG: {today_fvg}')
        stop_loss_change = 2 * stddev_model_data(model=model[0], date=today_date)
        if real_value < model[1] - stop_loss_change:
            print(f'END TRADE WITH {model}: STOP-LOSS')
            currently_open_trades = [subarray for subarray in currently_open_trades if subarray[0] != model[0]]
            models_to_exit.append([model, real_value, today_date])

        if today_fvg > -0.25:
            print(f'END TRADE WITH {model}: FVG')
            currently_open_trades = [subarray for subarray in currently_open_trades if subarray[0] != model[0]]
            models_to_exit.append([model, real_value, today_date])



    data_for_completed_trades = []

    curr_df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/current_trades_refactor.csv')
    model_order_size_dict = {}
    for index, row in curr_df.iterrows():
        model_order_size_dict[row['model']] = row['order_size']

    df_open_trades = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/current_trades_refactor.csv')
    data_from_currently_open_trades = []
    for model in models_to_exit:
        model_name = model[0][0]
        for index, row in df_open_trades.iterrows():
            if row['model'] == model_name:
                model_symbol = row['model']
                trade_entry_value = row['real_value']
                trade_entry_date = row['today_date']
                data_from_currently_open_trades.append([model_symbol, trade_entry_value, trade_entry_date])

    for model in models_to_exit:
        for data_point in data_from_currently_open_trades:
            if data_point[0] == model[0][0]:
                order_size = None
                for position in positions:
                    if position.symbol == data_point[0]:
                        order_size = position.market_value
                data_for_completed_trades.append([model[0][0], data_point[1],
                                                  model[1], data_point[2], model[2], order_size])

    existing_data = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/completed_trades_refactor.csv')
    new_data = pd.DataFrame(data_for_completed_trades, columns=existing_data.columns)
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    combined_data.to_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/completed_trades_refactor.csv', index=False)

    current_trades_df = pd.read_csv("/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/current_trades_refactor.csv")
    model_names_to_exit = [subarray[0][0] for subarray in models_to_exit]
    trade_data_to_keep = []
    for index, row in current_trades_df.iterrows():
        if row['model'] not in model_names_to_exit:
            trade_data_to_keep.append([row['model'], row['real_value'], row['today_date'], row['order_size']])

    df_new = pd.DataFrame(trade_data_to_keep, columns=['model', 'real_value', 'today_date', 'order_size'])
    df_new.to_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/current_trades_refactor.csv', index=False)

    print(f'{data_count} / {len(currently_open_trades)} had model data')

    return [models_to_exit, model_order_size_dict]
