import alpaca.common.exceptions
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest, OrderSide
from alpaca.trading.enums import OrderSide, TimeInForce
from CHECK_CURRENT_DAY_TRADES.py import find_models_to_buy_long, check_current_trades_long, quantify_order_size_vol_adjusted_new
import qi_client
import os
import Qi_wrapper
import numpy as np

# TO RUN THIS CODE, PLEASE HAVE 2 CSV FILES FOR CURRENT TRADES AND COMPLETED TRADES. EACH FILE SHOULD LOOK LIKE THIS:

# model,real_value,today_date,order_size
# HTLD,13.41644,2023-10-26,2812.5
# ACLS,124.09936,2023-10-30,843.75
# BWA,36.58234,2023-10-30,703.125

# model,trade_entry_price,trade_exit_price,trade_entry_date,trade_exit_date
# TSHA,2.0878,2.5815300000000003,2023-10-23,2023-10-25
# CMC,40.62148,42.08648,2023-10-24,2023-10-26
# INGR,90.53907,93.82217,2023-10-24,2023-10-26

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
df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/current_trades_alpaca.csv')


def qi_vol_indicator(date):
    indexes = ['S&P500', 'Euro Stoxx 600', 'USDJPY', 'EURUSD', 'USD 10Y Swap', 'EUR 10Y Swap']
    st_rsqs = [row['Rsq'] for model in indexes
               for _, row in Qi_wrapper.get_model_data(model=model, start=date, end=date, term='Short term').iterrows()]
    return 100 - np.mean(st_rsqs)


def current_max_leverage(date):
    current_volatility = qi_vol_indicator(date)
    base_leverage = 1
    average_volatility = 29.44
    adjustment_factor = max(0.5, 1 - (current_volatility - average_volatility) / average_volatility)
    adjusted_max_leverage = base_leverage * adjustment_factor
    return min(2, adjusted_max_leverage)



def main_long_vol_adjusted():

    curr_equity = account.equity
    df_account_tracker = pd.read_csv(
        "/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/account_value.csv")
    new_row = pd.DataFrame({'run': [len(df_account_tracker) + 1], 'account_value': [curr_equity]})
    df_account_tracker = pd.concat([df_account_tracker, new_row], ignore_index=True)
    df_account_tracker.to_csv(
        "/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/account_value.csv",
        index=False)

    models_currently_open = []
    df_open = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/current_trades_alpaca.csv')
    for index, row in df_open.iterrows():
        models_currently_open.append(row['model'])
    models_to_exit = [item[0] for item in check_current_trades_long()[0]]
    today_date = str(pd.Timestamp.today(tz='America/New_York').date().isoformat()).split()[0]
    for model in models_to_exit:
        for position in positions:
            if position.symbol == model[0]:
                order_size = position.qty_available
                market_order_data = MarketOrderRequest(
                    symbol=model[0],
                    qty=order_size,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY)
                account_request = trading_client.submit_order(market_order_data)
                print(f'{model} SELL')
    models_to_buy = [item[0] for item in find_models_to_buy_long()[0]]
    models_to_buy = [model for model in models_to_buy if model not in models_currently_open]
    current_trades_df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/current_trades_alpaca.csv')
    illiquid = 0
    for index, row in current_trades_df.iterrows():
        illiquid += row['order_size']
    illiquid = float(account.long_market_value)
    for model in models_to_buy:
        order_size = quantify_order_size_vol_adjusted_new(model=model, date_of_trade_entry=today_date)
        if order_size[0] >= float(account.equity) / 15:
            continue
        curr_max_lev = current_max_leverage(today_date)
        print(f"Order Size: {order_size[1]}, Buying Power: {curr_max_lev * float(account.equity) - illiquid}")
        if illiquid >= curr_max_lev * float(account.equity):
            print('Maximum Leverage limit reached. No more trades may be placed.')
            print(f'Invested: {illiquid}, 2 * Equity: {2 * float(account.equity)}')
            continue
        market_order_data = MarketOrderRequest(
                              symbol=model,
                              qty=order_size[0],
                              side=OrderSide.BUY,
                              time_in_force=TimeInForce.DAY)
        try:
            account_request = trading_client.submit_order(market_order_data)
            print(f'{model} BUY, order size: {order_size[1]}')
            illiquid += order_size[1]
        except alpaca.common.exceptions.APIError:
            print(f'there was a alpaca.common.exceptions.APIError for {model}. Removed from trades')

            # this removes trades from the csv file if the model is not on the alpaca DB (since that trade cannot be executed)
            csv_path = '/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/current_trades_alpaca.csv'
            df_current_trades = pd.read_csv(csv_path)
            df_current_trades = df_current_trades[df_current_trades['model'] != model]
            df_current_trades.to_csv(csv_path, index=False)
            continue


if __name__ == '__main__':
    main_long_vol_adjusted()

