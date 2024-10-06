import os
import pandas as pd
import qi_client
import datetime
import csv
import numpy as np
from dateutil.relativedelta import relativedelta
import warnings
import math
import Qi_wrapper
from datetime import timedelta, datetime
from post_qi_meeting.mapping_pitc_R3K_through_time import output_point_in_time_constituents
from qi_client.rest import ApiException

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['QI_API_KEY'] = 'aHUylOC5yM9xSRLpZs8Z45vHsxXClZNE4IW6rJ4n'
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = 'aHUylOC5yM9xSRLpZs8Z45vHsxXClZNE4IW6rJ4n'
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))


def filter_csv_by_date_bfo(model: str, start_date: str, end_date: str) -> pd.DataFrame:
    path = '/Users/FreddieLewin/PycharmProjects/new_dl_token/bfo_backtest/model_data/'
    path = path + f'{model}.csv'
    try:
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')  # Set 'Date' as index
    except Exception as e:
        columns = ['FVG', 'RSq', 'Model Value', 'Percentage Gap', 'Absolute Gap']
        empty_dataframe = pd.DataFrame(columns=columns)
        return empty_dataframe
    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
    return filtered_df


def subtract_days_from_date_backtest_bfo(input_date_str: str, n: int) -> str:
    input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
    new_date = input_date - timedelta(days=n)
    while new_date.weekday() >= 5:
        new_date -= timedelta(days=1)
    return new_date.strftime('%Y-%m-%d')


def output_model_data_dictionary(date: str) -> dict:
    # THIS DATE IS THE FIRST OF SOME MONTH, END DATE NEEDS TO BE THE DAY BEFORE THE END OF THIS MONTH
    year, month, day = date.split('-')
    start_of_month_str = f'{year}-{month}-01'
    models = output_point_in_time_constituents(start_of_month_str)
    res_dic = {}
    for model in models:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        if date_obj.month == 12:
            first_day_of_next_month = datetime(date_obj.year + 1, 1, 1)
        else:
            first_day_of_next_month = datetime(date_obj.year, date_obj.month + 1, 1)
        last_day_of_month = first_day_of_next_month - timedelta(days=1)
        curr_df = filter_csv_by_date_bfo(model=model, start_date=date, end_date=last_day_of_month)
        res_dic[model] = curr_df
    return res_dic


def stddev_model_data(model: str, date: str) -> float:
    if isinstance(model, list):
        model = model[0]
    end_date_str = str(date).split()[0]
    start_date_dt = pd.to_datetime(end_date_str, format='%Y-%m-%d') - relativedelta(years=1)
    start_date_str = str(start_date_dt)[:10]
    df = filter_csv_by_date_bfo(model=model, start_date=start_date_str, end_date=end_date_str)
    df['Real Value'] = df['Model Value'] + df['Absolute Gap']
    df.reset_index(inplace=True)
    try:
        df = df[['Date', 'Real Value']]
    except Exception as e:
        print(9)
    df['Log Return'] = np.log(df['Real Value'] / df['Real Value'].shift(1))
    df = df.dropna()
    std_dev = df['Log Return'].std()
    return std_dev * 440


def fvg_backtest_long(model: str, start: str, end: str, threshold_buy: float,
                      threshold_sell: float, Rsq: float) -> list[int]:

    df = filter_csv_by_date_bfo(model=model, start_date=start, end_date=end)
    if len(df) <= 300:
        df = Qi_wrapper.get_model_data(model=model, start=start, end=end, term="long term")

    if len(df) == 0:
        return []
    trades_dict = []
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
        real_value = row.iloc[2] + row.iloc[4]

        if day_count == 60:
            stop_loss_change = 2 * real_value * (stddev_model_data(model, str(index).split()[0])/100)
            day_count = 0
        fvg_value = row['FVG']
        rsq_value = row['Rsq']
        if real_value < buy - stop_loss_change and buy != float("inf"):
            stop_loss_date = index
            days_between = (stop_loss_date - buy_date).days
            trades.append([buy, real_value, days_between, rsq_value, rsq_value_at_buy, stop_loss_date, buy_date])
            trades_dict.append({'Real_value_at_buy': buy, 'Real_value_at_sell': real_value,
                                'Days between trades': days_between, 'Rsq at sell': rsq_value,
                                'Rsq at buy': rsq_value_at_buy,
                                'Sell_date': stop_loss_date, 'FVG value at buy': fvg_at_buy,
                                'FVG value at sell': fvg_value})
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
    results = [round(average_percentage_return, 3), len(trades_profit_percentage), round(percentage_profitable, 3),
               stop_loss_count, round(np.mean(days_for_trades), 3),
               trades, stop_loss_count, trades_dict]
    return results


def backtest_score_long_continuous(results_from_backtest: list[int]) -> float:
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
    start_date = subtract_days_from_date_backtest_bfo(date, look_back)
    data_df = filter_csv_by_date_bfo(model=model, start_date=start_date, end_date=date)
    if len(data_df) == 0:
        data_df = Qi_wrapper.get_model_data(model=model, start=start_date, end=date, term='Long term')
    if len(data_df) == 0:
        return 0.01
    model_values = []
    for index, row in data_df.iterrows():
        model_values.append(row['Model Value'])
    x_values = np.array(list(range(len(model_values))))
    y_values = np.array(model_values)
    # polyfit x and y values and deg is the number of degrees of the power series you want. UNREAL ENDPOINT
    try:
        slope, _ = np.polyfit(x_values, y_values, 1)
    except np.linalg.LinAlgError:
        return -0.1
    return slope


def price_trend_score(model, date_of_trade_entry):
    three_day = mvg_current_linear_regression(model=model, date=date_of_trade_entry, look_back=3)
    ten_day = mvg_current_linear_regression(model=model, date=date_of_trade_entry, look_back=10)
    thirty_day = mvg_current_linear_regression(model=model, date=date_of_trade_entry, look_back=30)
    mvgs = [three_day, ten_day, thirty_day]
    return (len([num for num in mvgs if num > 0]) / len(mvgs)) * 100


def qi_vol_indicator_bfo(date):
    indexes = ['S&P500', 'Euro Stoxx 600', 'USDJPY', 'EURUSD', 'USD 10Y Swap', 'EUR 10Y Swap']
    st_rsqs = []
    for model in indexes:
        df = pd.read_csv(f'/Users/FreddieLewin/PycharmProjects/new_dl_token/bfo_backtest/model_data/{model}.csv')
        rsq_value = df[df['Date'] == date]['Rsq'].values[0]
        st_rsqs.append(rsq_value)
    if len(st_rsqs) == 0:
        return 29.4
    return 100 - np.mean(st_rsqs)


def vol_individual(model, date):
    df = filter_csv_by_date_bfo(model=model, start_date=subtract_days_from_date_backtest_bfo(date, 365), end_date=date)
    if len(df) == 0:
        df = Qi_wrapper.get_model_data(model=model, start=subtract_days_from_date_backtest_bfo(date, 365), end=date, term='Long term')
    if len(df) == 0:
        return 0
    df['Asset Price'] = df['Model Value'] + df['Absolute Gap']
    df['Daily Percentage Change'] = df['Asset Price'].pct_change() * 100
    std_dev = df['Daily Percentage Change'].std()
    return std_dev


def quantify_order_size(model, date_of_trade_entry):
    date_of_trade_entry_dt = pd.to_datetime(date_of_trade_entry, format='%Y-%m-%d')
    start_date_dt = date_of_trade_entry_dt - relativedelta(days=1800)
    start_date = str(start_date_dt)[:10]
    res = fvg_backtest_long(model=model, start=start_date, end=date_of_trade_entry, threshold_buy=-1,
                            threshold_sell=-0.25, Rsq=75)
    bt_score = backtest_score_long_continuous(res)
    pt_score = price_trend_score(model=model, date_of_trade_entry=date_of_trade_entry)
    price_data = filter_csv_by_date_bfo(model=model, start_date=date_of_trade_entry, end_date=date_of_trade_entry)
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

    account_value_df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/account_value.csv')
    current_equity = account_value_df['account_value'].tolist()[-1]
    base_order_size = float(current_equity) / 100

    individual_vol = vol_individual(model=model, date=date_of_trade_entry)
    amount_to_buy = base_order_size * bt_coefficient * pt_coefficient * (1 - ((individual_vol - 3.26) / individual_vol))
    return [round(amount_to_buy / real_value, 4), amount_to_buy]


def current_max_leverage(date):
    current_volatility = qi_vol_indicator_bfo(date)
    max_leverage = 2.0
    average_volatility = 29.44
    adjustment_factor = max(0.5, 1 - (current_volatility - average_volatility) / average_volatility)
    adjusted_max_leverage = max_leverage * adjustment_factor
    return min(2, adjusted_max_leverage)


if __name__ == '__main__':
    models_USD = [x.name
                  for x in api_instance.get_models(tags='USD, Stock')
                  if x.model_parameter == 'long term' and '_' not in x.name
                  ]


def has_whitespace(input_string):
    return any(char.isspace() or char == '/' for char in input_string)


curr_pitc_model_data = None


def find_models_to_buy(date_of_current):
    global curr_pitc_model_data
    new_trade_models = []
    currently_open_trades = []


    year, month, day = date_of_current.split('-')
    start_of_month_str = f'{year}-{month}-01'
    bad_models = ['NLTX', 'LTHM', 'POST', 'PRET', 'CDAY', 'SFE', 'ADES', 'ACOR', 'A']
    models_USD_PITC = output_point_in_time_constituents(start_of_month_str)
    models_USD_PITC = [model for model in models_USD_PITC if model not in bad_models]

    i = 0
    df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/current.csv')
    for index, row in df.iterrows():
        currently_open_trades.append(row['model'])
    count = 0
    current_leverage = current_max_leverage(date_of_current)

    illiquid = 0
    for index, row in df.iterrows():
        illiquid += row['order_size']
    account_stats = pd.read_csv(
        '/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/account_value.csv')
    latest_account_value = account_stats.iloc[-1]['account_value']
    print(f'Account Val: {latest_account_value}, Illiquid: {illiquid}')
    today_date = date_of_current

    if today_date.split('-')[-1] == '01' or today_date.split('-')[-1] == '23' or today_date.split('-')[-1] == '26':
        print(f'UPDATED R3K MODEL DATA DICTIONARY')
        curr_pitc_model_data = output_model_data_dictionary(today_date)

    illiquid = 0
    for index, row in df.iterrows():
        illiquid += row['order_size']

    account_stats = pd.read_csv(
        '/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/account_value.csv')
    latest_account_value = 100_000
    if len(account_stats) != 0:
        latest_account_value = account_stats.iloc[-1]['account_value']

    for model in models_USD_PITC:
        if illiquid >= current_leverage * latest_account_value:
            break
        count += 1
        if has_whitespace(model):
            continue

        i += 1

        try:
            current_model_data = curr_pitc_model_data[model].loc[str(today_date).split()[0]]
        except KeyError:
            continue

        if len(current_model_data) == 0:
            try:
                # if there is missing data in my locally saved data, check the QI DB instead
                current_model_data = Qi_wrapper.get_model_data(model=model, start=today_date, end=today_date, term='Long term')
            except Exception as e:
                print(f'Exception {e} for {model}')
                continue

        if model in currently_open_trades:
            # skip the model if already in a trade
            continue
        if len(current_model_data) == 0:
            continue

        today_fvg = current_model_data['FVG']
        today_rsq = current_model_data['Rsq']

        if float(today_rsq) >= 65 and float(today_fvg) <= -1:
            print('__________________________')
        model_value = current_model_data['Model Value']
        absolute_gap = current_model_data['Absolute Gap']
        real_value = model_value + absolute_gap

        if count >= 30:
            current_leverage = current_max_leverage(date_of_current)
            count = 0

        try:
            if today_rsq > 65 and today_fvg < -1:
                # Check if illiquid is within allowable leverage first, and break if not
                if illiquid > current_leverage * latest_account_value:
                    continue

                # Calculate the price trend score
                price_trend_score_curr = price_trend_score(model=model, date_of_trade_entry=date_of_current)

                # Break if price trend score is insufficient
                if price_trend_score_curr < 20:
                    continue

                # Set the start date of the backtest
                start_date_of_backtest = subtract_days_from_date_backtest_bfo(today_date, 1000)

                # Run the backtest
                backtest_data = fvg_backtest_long(model=model, start=start_date_of_backtest, end=today_date,
                                                  threshold_buy=-1, threshold_sell=-0.25, Rsq=65)

                # Check if there are any trades from the backtest
                if len(backtest_data) == 0:
                    continue

                # Calculate backtest score
                backtest_score_today = backtest_score_long_continuous(backtest_data)

                # Break if backtest score is insufficient
                if backtest_score_today <= 3.5:
                    continue

                order_size = quantify_order_size(model=model, date_of_trade_entry=today_date)[1]

                # Check order size validity and execute trade
                if order_size <= latest_account_value / 15 and not math.isnan(order_size):
                    new_trade_models.append([model, real_value, today_date, order_size, today_fvg])
                    illiquid += order_size
                    print(f'Invested {illiquid}, Max invested: {current_leverage * latest_account_value}')


        except ApiException as e:
            print(f'ApiException error (model not found on QI database).')
            continue

    new_trade_models = sorted(new_trade_models, key=lambda x: x[-1])
    data_added = [data[:-1] for data in new_trade_models ]

    csv_file = '/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/current.csv'
    print(pd.read_csv(csv_file))
    columns = ['model', 'real_value', 'today_date', 'order_size']
    new_trade_df = pd.DataFrame(data_added, columns=columns)
    existing_data = pd.read_csv(csv_file)
    updated_data = pd.concat([existing_data, new_trade_df], ignore_index=True)
    print(updated_data)
    updated_data.to_csv(csv_file, index=False)

    model_amount_dict = {}
    for data in new_trade_models:
        model_amount_dict[data[0]] = data[3]

    return [new_trade_models, model_amount_dict]


def check_current_trades(date_of_current):
    currently_open_trades = []
    df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/current.csv')
    print(df, 'current tardes at tstart of checking function')
    data_count = 0
    for index, row in df.iterrows():
        currently_open_trades.append([row['model'], row['real_value']])
    models_to_exit = []
    i = 0
    for model in currently_open_trades:
        i += 1
        print(f'{i} / {len(currently_open_trades)}')
        today_date = date_of_current
        current_model_data = filter_csv_by_date_bfo(model=model[0], start_date=today_date, end_date=today_date)
        if len(current_model_data) == 0:
            try:
                current_model_data = Qi_wrapper.get_model_data(model=model, start=today_date, end=today_date,
                                                               term='Long term')
            except Exception as e:
                continue
        if len(current_model_data) == 1:
            data_count += 1
        if len(current_model_data) == 0:
            continue
        model_value = current_model_data['Model Value'][0]
        absolute_gap = current_model_data['Absolute Gap'][0]
        today_fvg = current_model_data['FVG'][0]
        real_value = model_value + absolute_gap

        if today_fvg > -0.25:
            print(f'END TRADE WITH {model}: FVG')
            currently_open_trades = [subarray for subarray in currently_open_trades if subarray[0] != model[0]]
            models_to_exit.append([model, real_value, today_date])
            continue

        stop_loss_change = 2 * real_value * (stddev_model_data(model, str(date_of_current).split()[0]) / 100)

        if real_value < model[1] - stop_loss_change:
            print(f'END TRADE WITH {model}: STOP-LOSS')
            currently_open_trades = [subarray for subarray in currently_open_trades if subarray[0] != model[0]]
            models_to_exit.append([model, real_value, today_date])

    data_for_completed_trades = []
    # i want this to look like this [model, trade_entry_price, trade_exit_price, trade_entry_date, trade_exit_date]
    # i will the niterate through the currently open_trades and grab bhuy_date_date = [row['real_value], row['today_date']
    # to do this i will need to iterate through models_to_exit and append to data array [models_to_exit[0], ,models_to_exit[1], models_to_exit[2]]

    curr_df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/current.csv')
    model_order_size_dict = {}
    for index, row in curr_df.iterrows():
        model_order_size_dict[row['model']] = row['order_size']

    df_open_trades = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/current.csv')
    data_from_currently_open_trades = []
    for model in models_to_exit:
        model_name = model[0][0]
        for index, row in df_open_trades.iterrows():
            if row['model'] == model_name:
                model_symbol = row['model']
                trade_entry_value = row['real_value']
                trade_entry_date = row['today_date']
                order_size_current_trade = row['order_size']
                data_from_currently_open_trades.append([model_symbol, trade_entry_value, trade_entry_date, order_size_current_trade])

    for model in models_to_exit:
        for data_point in data_from_currently_open_trades:
            if data_point[0] == model[0][0] and not math.isnan(data_point[-1]):
                data_for_completed_trades.append([model[0][0], data_point[1],
                                                  model[1], data_point[2], model[2], data_point[-1]])
    existing_data = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/completed.csv')
    new_data = pd.DataFrame(data_for_completed_trades, columns=existing_data.columns)
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    combined_data.to_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/completed.csv', index=False)
    print("Completed trades added successfully.")

    current_trades_df = pd.read_csv("/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/current.csv")
    model_names_to_exit = [subarray[0][0] for subarray in models_to_exit]
    trade_data_to_keep = []
    for index, row in current_trades_df.iterrows():
        if row['model'] not in model_names_to_exit:
            trade_data_to_keep.append([row['model'], row['real_value'], row['today_date'], row['order_size']])
    df_new = pd.DataFrame(trade_data_to_keep, columns=['model', 'real_value', 'today_date', 'order_size'])
    df_new.to_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/current.csv', index=False)
    print(f'{data_count} / {len(currently_open_trades)} had model data')
    return [models_to_exit, model_order_size_dict]
