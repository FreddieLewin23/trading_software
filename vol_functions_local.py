from find_current_day_trades_long import (subtract_days_from_date, find_sd_from_model_data, fvg_backtest_long,
                                          backtest_score_long, mvg_current_linear_regression, price_trend_score,
                                          check_current_trades_long, find_models_to_buy_long)
import Qi_wrapper
import qi_client
import os
import math
from alpaca.trading.client import TradingClient
import warnings
from statistics import stdev
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
pd.options.mode.chained_assignment = None




os.environ['QI_API_KEY'] = ''
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = ''
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))


API_KEY = ""
SECRET_KEY = ""
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
clock = trading_client.get_clock()
print(f"Market is {'open' if clock.is_open else 'closed'}")
account = trading_client.get_account()
positions = trading_client.get_all_positions()
orders = trading_client.get_orders()



# idea for the vol indicator. look at the change in the R squared for 6 main indexes short term models
# hopefully plot of my indicator lines up with VIX or the QI vol indicator

# factors affecting volatility value:
# 3, 10, 30 day change in R-squared in 10 ETFs, S&P500 (top end general exposure), NASDAQ (top end mega tech cap)
# IWV (general market), XLE (energy exposure), XLF (financial sector exposure), XLV (health exposure), BBH (biotech)
# SHY (1-3 year treasury), IEF (7-10 year treasury), IGF (infrastructure)
#
# how to quantify exposure to market sector.
# IWV has macro contributions. on the lead up to a given date, this index has 5 top contributers
# (change in driver z score * standardised change in price), associate each macro factor with some index and place bias
# on those macro factors has the main changes causing a shift of volatility (basically the factor to be weary to
# have exposure from)
#
# S&P500, NASDAQ, IWV, XLE, XLF, XLV, BBH, SHY, IEF, IGF
# pull model data from these 10 indexes, look at an R-squared regression fitting, standard deviation of the percentage
# change in the R squared for each index. The further the regression line (linear) gradients are from 0, more volatile,
# the large the standard deviation of the percentage change on the R-squared, the more volatile.
# step function quantifying this change

# why short term model data:
# i use long term for the trades since R2 higher on average, and fundamental shifts in the market are better tracked in
# the long term. however, for volatility I want a shorter term look at the current market so ST better suited.

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


def vol_indicator_one_index(index, date):
    st_data_month = Qi_wrapper.get_model_data(model=index, start=subtract_days_from_date(date, 30), end=date, term='Short term')

    x_vals_30 = np.arange(len(st_data_month))
    y_vals_30 = st_data_month['Rsq'].values
    x_vals_10 = np.arange(len(st_data_month[13:]))
    y_vals_10 = st_data_month[13:]['Rsq'].values
    x_vals_3 = np.arange(len(st_data_month[20:]))
    y_vals_3 = st_data_month[20:]['Rsq'].values
    slope_30, _ = np.polyfit(x_vals_30, y_vals_30, 1)
    slope_10, _ = np.polyfit(x_vals_10, y_vals_10, 1)
    slope_3, _ = np.polyfit(x_vals_3, y_vals_3, 1)

    st_data_month['Rsq_daily_change'] = st_data_month['Rsq'].pct_change() * 100
    std_deviation_30 = st_data_month['Rsq_daily_change'].std()

    st_data_month[13:]['Rsq_daily_change'] = st_data_month[13:]['Rsq'].pct_change() * 100
    std_deviation_10 = st_data_month[13:]['Rsq_daily_change'].std()

    st_data_month[20:]['Rsq_daily_change'] = st_data_month[20:]['Rsq'].pct_change() * 100
    std_deviation_3 = st_data_month[20:]['Rsq_daily_change'].std()

    return {'slope':[slope_30, slope_10, slope_3], 'standard_devs':[std_deviation_30, std_deviation_10, std_deviation_3]}

def vol_indicator(date):
    indexes = ['S&P500', 'NASDAQ', 'IWV', 'XLE', 'XLF', 'XLV', 'BBH', 'SHY', 'IEF', 'IGF']
    slope_30s, slope_10s, slope_3s = [], [], []
    std30, std10, std3 = [], [], []
    for index in indexes:
        res_temp = vol_indicator_one_index(index, date)
        slope_30s.append(res_temp['slope'][0])
        slope_10s.append(res_temp['slope'][1])
        slope_3s.append(res_temp['slope'][2])
        std30.append(res_temp['standard_devs'][0])
        std10.append(res_temp['standard_devs'][1])
        std3.append(res_temp['standard_devs'][2])
    avg_30slope, avg_10slope, avg_3slope = np.mean(slope_30s), np.mean(slope_10s), np.mean(slope_3s)
    avg_30std, avg_10std, avg_3std = np.mean(std30), np.mean(std10), np.mean(std3)
    regression_score = ((avg_30slope / 0.15) * 5) + ((avg_10slope / 0.10) * 5) + (avg_3slope / 0.2 * 5)
    std_score = ((avg_30std / 0.3) * 5) + ((avg_10std / 0.25) * 5) + (avg_3std / 0.25 * 5)
    return std_score + regression_score

#
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     vols = []
#     for date in weekdays_between_dates('2023-01-01', '2023-12-29'):
#         print(date)
#         vols.append(vol_indicator(date))
#     x_vals = list(range(len(vols)))
#     plt.plot(x_vals, vols, color='red', label='Own Vol Indicator in 2023')
#     plt.legend()
#     plt.savefig('/Users/FreddieLewin/Desktop/latex_file_iages/vol_indicator')
#     plt.show()








def quantify_order_size_vol_adjusted(model, date_of_trade_entry):
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

    vol_indicator_value = vol_indicator(date_of_trade_entry)

    amount_to_buy = base_order_size * bt_coefficient * pt_coefficient * (vol_indicator_value / 250)
    return [round(amount_to_buy / real_value, 4), amount_to_buy]

