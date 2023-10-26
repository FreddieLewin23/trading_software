import os
import json
import pandas as pd
import qi_client
from datetime import datetime, timedelta
import numpy as np
import concurrent.futures
import statistics
import warnings
import threading
import Qi_wrapper

warnings.simplefilter(action='ignore', category=FutureWarning)

# i gave up on trying the different target_stdev pull, the taget std was also far too high

os.environ['QI_API_KEY'] = 'aHUylOC5yM9xSRLpZs8Z45vHsxXClZNE4IW6rJ4n'
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = 'aHUylOC5yM9xSRLpZs8Z45vHsxXClZNE4IW6rJ4n'
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))

# opens JSON file
with open('model_data_all_US_stocks(DONT DELETE).json', 'r') as file:
    data = json.load(file)

# when inputting a date and look back, it finds the first week day on or before n days before the input date
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

# pulls data from JSON between the dates you want
def grab_data2(model, start, end):
    try:
        model_data = data[model]  # Assuming 'data' is a dictionary containing your model data
    except KeyError:
        return []
    end_date = pd.Timestamp(end)
    start_date = pd.Timestamp(start)
    df = pd.DataFrame(model_data)
    df.index = pd.to_datetime(df.index, unit='ms')
    df.index = df.index.round('D')
    indexes_new = []
    for index in list(df.index):
        indexes_new.append(str(index).split()[0])
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    # Drop the 'timestamp' column
    df = df.drop(columns=['timestamp'])
    return df



# first model value gradient using regression
def model_value_gradient_regression(model, date, look_back):
    date = str(date).split()[0]
    start_date = subtract_days_from_date(date, look_back)
    df = grab_data2(model=model, start=start_date, end=date)
    dates = np.arange(1, len(df) + 1)
    model_values = df['Model Value'].values
    mean_dates = np.mean(dates)
    mean_model_values = np.mean(model_values)
    numerator = np.sum((dates - mean_dates) * (model_values - mean_model_values))
    denominator = np.sum((dates - mean_dates) ** 2)
    if numerator == 0 or denominator == 0:
        return 0
    gradient = numerator / denominator
    return round(gradient, 5)


# model value gradient using n day difference
def model_value_gradient_n_day_diff(model, date, look_back):
    date = str(date).split()[0]
    start_date = subtract_days_from_date(date, look_back)
    df = grab_data2(model=model, start=start_date, end=date)
    first_day_model_value = df.iloc[0]['Model Value']
    last_day_model_value = df.iloc[-1]['Model Value']
    return (last_day_model_value - first_day_model_value) / look_back




# find sd from an array of change is price data between start and end of week for the 52 weeks before the date
def find_sd_from_model_data(model, date):
    # this code extracts the data from the JSON file that is needed
    end_date = str(date).split()[0]
    year, month, day = end_date.split('-')
    if month == '02' and day == '29':
        day = '28'
    year = str(int(year) - 1)
    start_date = f"{year}-{month}-{day}"
    # increments year back one to find start date for monthly rolling average
    df = grab_data2(model=model, start=start_date, end=end_date)
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
    df = grab_data2(model=model, start=start, end=end)
    trades_dict = []
    trades = []
    fvg_at_buy = None
    buy = float('inf')
    day_count = 0
    std = find_sd_from_model_data(model, start)
    if not std:
        return None
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
        if fvg_value > threshold_sell and buy != float("inf") and rsq_value > Rsq:
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
        return None
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



# i version of the backtest the only outputs the data i need for the mvg investigation
def backtest_data_output_trades(model, start, end, threshold_buy_long, threshold_sell_long, term, threshold_buy_short, threshold_sell_short, Rsq, l_s):
    results_long = None
    results_short = None
    trade_data_short = None
    trade_data_long = None
    if 'Long' in l_s:
        results_long = fvg_backtest_long(model=model, start=start, end=end, threshold_buy=threshold_buy_long,
                                         threshold_sell=threshold_sell_long, Rsq=Rsq)
    if 'Short' in l_s:
        results_short = fvg_backtest_long(model=model, start=start, end=end,
                                           threshold_buy=threshold_buy_short, threshold_sell=threshold_sell_short,
                                           Rsq=Rsq)
        # trade = [buy, real_value, days_between, rsq_value, rsq_value_at_buy, sell_date, buy_date]
    if 'Long' not in l_s and 'Short' not in l_s:
        print(f"Please select a position (Long or Short) for the backtest!")
    if results_long is not None:
        trades_long = results_long[5]
        trade_data_long = []
        for trade in trades_long:
            trade_data_long.append([trade[0], trade[1], trade[-1], model])
    #         buy_date, sell_date, days_between_buy_and_sell, percentage return
    if results_short is not None:
        trades_short = results_short[5]
        trade_data_short = []
        for trade in trades_short:
            trade_data_short.append([trade[-1], trade[-2], trade[2], ((trade[1] - trade[0]) / trade[0]) * 100])
    return [trade_data_long, trade_data_short]

# Buy Price,Sell Price,Buy Date,Model

# use threading to find the data on mvg
def find_average_mvg_threading(models, start, end):
    mvg = []
    def compute_gradient(model):
        # Check if the input date is valid
        try:
            start_date = datetime.strptime('2019-02-01', "%Y-%m-%d")
            end_date = datetime.strptime('2023-01-01', "%Y-%m-%d")
        except ValueError as e:
            print(f"Invalid date format: {e}")
            return []
        get = backtest_data_output_trades(model=model, start=start, end=end, threshold_buy_long=-1,
                                          threshold_sell_long=-0.25, term='Long term', threshold_buy_short=1,
                                          threshold_sell_short=0.25, Rsq=65, l_s=['Long'])
        data = get[0]
        if not data:
            return []
        gradients = []
        for trade in data:
            # Check if the trade date is within the valid date range
            trade_date = str(trade[0]).split(' ')[0]
            trade_date_obj = datetime.strptime(trade_date, "%Y-%m-%d")
            if trade_date_obj < start_date or trade_date_obj > end_date:
                continue  # Skip trades outside the date range
            gradient = model_value_gradient_n_day_diff(model=model, date=trade_date, look_back=10)
            gradients.append([gradient, trade[3], trade[0], trade[1], model])
        #     mvg and percentage returns, buy_date, sell_date, model
        print(model)
        return gradients
    with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
        results = executor.map(compute_gradient, models)
    for gradients in results:
        mvg.extend(gradients)
    return mvg



# models_sandp = [x.name
#           for x in api_instance.get_models(tags='S&P 500')
#           if x.model_parameter == 'long term' and '_' not in x.name
#           ][:20]

models = [x.name
          for x in api_instance.get_models(tags='USD, Stock')
          if x.model_parameter == 'long term' and '_' not in x.name
          ]


# for each trade i want percentage return, MVG 10 look back, MVG 30 look back, buy date, sell date,

# groups the percentage returns on mvg data
def grouping_FVG_backtested_trades_based_on_MVG():
    data = find_average_mvg_threading(models, '2019-02-01', '2022-01-01')
    group0, group1, group2, group3, group4, group5, group6, group7, group8, group9, group10, group11, group12, group13, group14, group15 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for trade in data:
        if trade[0] < -3 and trade not in group0:
            group0.append(trade)
        elif -3 < trade[0] < -2.5 and trade not in group1:
            group1.append(trade)
        elif -2.5 < trade[0] < -2 and trade not in group2:
            group2.append(trade)
        elif -2 < trade[0] < -1.5 and trade not in group3:
            group3.append(trade)
        elif -1.5 < trade[0] < -1 and trade not in group4:
            group4.append(trade)
        elif -1 < trade[0] < -0.65 and trade not in group5:
            group5.append(trade)
        elif -0.65 < trade[0] < -0.3 and trade not in group6:
            group6.append(trade)
        elif -0.3 < trade[0] < 0 and trade not in group7:
            group7.append(trade)
        elif 0 < trade[0] < 0.3 and trade not in group8:
            group8.append(trade)
        elif 0.3 < trade[0] < 0.65 and trade not in group9:
            group9.append(trade)
        elif 0.65 < trade[0] < 1 and trade not in group10:
            group10.append(trade)
        elif 1 < trade[0] < 1.5 and trade not in group11:
            group11.append(trade)
        elif 1.5 < trade[0] < 2 and trade not in group12:
            group12.append(trade)
        elif 2 < trade[0] < 2.5 and trade not in group13:
            group13.append(trade)
        elif 2.5 < trade[0] < 3 and trade not in group14:
            group14.append(trade)
        elif trade[0] > 3 and trade not in group15:
            group15.append(trade)
    groups = [group0, group1, group2, group3, group4, group5, group6,
              group7, group8, group9, group10, group11, group12, group13, group14, group15]
    # each data point is [model_gradient, percentage_return]
    results = []
    for group in groups:
        percentage_returns = [data[1] for data in group]
        avg_percentage_return = np.mean(percentage_returns)
        results.append([avg_percentage_return, len(group)])
    reshaped_data = np.array(results).T
    columns = ['MVG < -3', '3 < MVG < -2.5', '-2.5 < MVG < -2', '-2 < MVG <-1.5', '-1.5 < MVG <-1.0',
               '-1.0 < MVG < -0.65', '-0.65 < MVG <-0.3',
               '-0.3 < MVG < 0', '0 < MVG < 0.3', '0.3 < MVG < 0.65', '0.65 < MVG < 1.0', '1.0 < MVG < 1.5',
               '1.5 < MVG < 2', '2 < MVG < 2.5', '2.5 < MVG < 3', 'MVG > 3']
    df = pd.DataFrame(reshaped_data, columns=columns)
    df.rename({0: 'Number of Trades', 1: 'Percentage Return'}, inplace=True)
    df = df.T
    df.columns = ['Percentage Return', 'Number of Trades']
    return df


# trades.append([buy, real_value, days_between, rsq_value, rsq_value_at_buy, sell_date, buy_date])
def trade_df_from_fvg_backtest():
    all_trades = []
    for model in models:
        res = fvg_backtest_long(model=model, start='2019-02-01', end='2022-12-15', threshold_buy=-1, threshold_sell=-0.25, Rsq=65)
        if res is None:
            continue
        trades_for_current = res[5]
        for trade in trades_for_current:
            mvg_ten_day = model_value_gradient_n_day_diff(model=model, date=str(trade[-1]).split()[0], look_back=10)
            mvg_thirty_day = model_value_gradient_n_day_diff(model=model, date=str(trade[-1]).split()[0], look_back=30)
            percentage_return = ((trade[1] - trade[0]) / trade[0]) * 100
            all_trades.append([model, round(percentage_return, 3), round(mvg_ten_day, 3), round(mvg_thirty_day, 3), str(trade[-1]).split()[0], str(trade[-2]).split()[0]])
    df = pd.DataFrame(all_trades,
                      columns=['model', 'percentage_return', 'mvg_ten_day', 'mvg_thirty_day', 'buy_date', 'sell_date'])
    df.index = range(len(df))
    pd.set_option('display.max_columns', None)

    group1, group2 = [], []
    for trade in all_trades:
        if trade[2] > 0:
            group1.append(trade)
        else:
            group2.append(trade)
    group1_returns = np.mean([trade[1] for trade in group1])
    group2_returns = np.mean([trade[1] for trade in group2])

    group3, group4 = [], []
    for trade in all_trades:
        if trade[3] > 0:
            group3.append(trade)
        else:
            group4.append(trade)
    group3_returns = np.mean([trade[1] for trade in group3])
    group4_returns = np.mean([trade[1] for trade in group4])

    print(f'Avg Percentage Return for positive MVGs: {group1_returns} on a 10 day look back')
    print(f'Avg Percentage Return for negative MVGs: {group2_returns} on a 10 day look back')
    print(f'Avg Percentage Return for positive MVGs: {group3_returns} on a 30 day look back')
    print(f'Avg Percentage Return for negative MVGs: {group4_returns} on a 30 day look back')
    # return df


#
def calculate_trade_data(model, all_trades):
    res = fvg_backtest_long(model=model, start='2019-02-01', end='2022-12-15', threshold_buy=-1, threshold_sell=-0.25, Rsq=65)
    if res is None:
        return
    trades_for_current_model = res[5]
    # res[5] is an array of array for each trade found on that model that looks like this
    # [buy, real_value, days_between, rsq_value, rsq_value_at_buy, stop_loss_date, buy_date]
    for trade in trades_for_current_model:
        mvg_three_day = model_value_gradient_n_day_diff(model=model, date=str(trade[-1]).split()[0], look_back=3)
        mvg_ten_day = model_value_gradient_n_day_diff(model=model, date=str(trade[-1]).split()[0], look_back=10)
        mvg_thirty_day = model_value_gradient_n_day_diff(model=model, date=str(trade[-1]).split()[0], look_back=30)
        percentage_return = ((trade[1] - trade[0]) / trade[0]) * 100
        all_trades.append([model, round(percentage_return, 3), round(mvg_three_day, 3), round(mvg_ten_day, 3), round(mvg_thirty_day, 3), str(trade[-1]).split()[0], str(trade[-2]).split()[0]])

def main():
    all_trades = []
    thread_list = []

    for model in models:
        print(model, models.index(model))
        thread = threading.Thread(target=calculate_trade_data, args=(model, all_trades))
        thread.start()
        thread_list.append(thread)

    # Wait for all threads to finish
    for thread in thread_list:
        thread.join()

    df = pd.DataFrame(all_trades,
                      columns=['model', 'percentage_return', 'mvg_three_day', 'mvg_ten_day', 'mvg_thirty_day', 'buy_date', 'sell_date'])
    df.index = range(len(df))
    pd.set_option('display.max_columns', None)

    hit_rate = len([trade for trade in all_trades if trade[1] > 0]) / len(all_trades)
    print(hit_rate)
    print(len(all_trades))
    group1, group2 = [], []
    for trade in all_trades:
        if trade[2] > 0:
            group1.append(trade)
        else:
            group2.append(trade)
    group1_returns = np.mean([trade[1] for trade in group1])
    group2_returns = np.mean([trade[1] for trade in group2])

    group3, group4 = [], []
    for trade in all_trades:
        if trade[3] > 0:
            group3.append(trade)
        else:
            group4.append(trade)
    group3_returns = np.mean([trade[1] for trade in group3])
    group4_returns = np.mean([trade[1] for trade in group4])

    group5, group6 = [], []
    for trade in all_trades:
        if trade[4] > 0:
            group5.append(trade)
        else:
            group6.append(trade)
    group5_returns = np.mean([trade[1] for trade in group5])
    group6_returns = np.mean([trade[1] for trade in group6])

    returns = []
    for trade in all_trades:
        returns.append(trade[1])
    print(np.mean(returns))

    print(f'Avg Percentage Return for positive MVGs (30 day look back): {group5_returns}')
    print(f'Avg Percentage Return for negative MVGs (30 day look back): {group6_returns}')
    print(f'Avg Percentage Return for positive MVGs (3 day look back): {group1_returns}')
    print(f'Avg Percentage Return for negative MVGs (3 day look back): {group2_returns}')
    print(f'Avg Percentage Return for positive MVGs (10 day look back): {group3_returns}')
    print(f'Avg Percentage Return for negative MVGs (10 day look back): {group4_returns}')




if __name__ == "__main__":
    # main()
    pass
