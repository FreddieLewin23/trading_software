import pandas as pd
from algo_new.find_and_check_current_trades import (
    find_models_to_buy_long,
    check_current_trades_long,
    quantify_order_size_vol_adjusted_new
)
import qi_client
import time
import os
import Qi_wrapper
import numpy as np
from datetime import datetime, timedelta
import sys

# due to messed up local interpreter needing to use a different version of alpaca for dependency issues
sys.path.append('/Users/FreddieLewin/miniconda3/envs/new_dl_token/lib/python3.11/site-packages/alpaca')
import alpaca.common.exceptions
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetAssetsRequest,
    MarketOrderRequest,
    OrderSide
)
from alpaca.trading.enums import (OrderSide, TimeInForce, AssetStatus, OrderType)

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
print(f"Market is {'open' if clock.is_open else 'closed'}")
account = trading_client.get_account()
positions = trading_client.get_all_positions()
orders = trading_client.get_orders()
df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/current_trades_alpaca.csv')


def calculate_total_delta():
    total_delta = 0.0
    positions = trading_client.get_all_positions()
    for position in positions:
        qty = float(position.qty)
        current_price = float(position.current_price)

        if position.asset_class == 'option':
            # check this works properly
            option_delta = float(position.delta)
            delta = qty * option_delta * 100
        else:
            delta = qty * current_price
        total_delta += delta
    return total_delta


def return_r3k_option_ticker():
    today = datetime.today()
    r3k_data = Qi_wrapper.get_model_data(model='IWV', start=str(today)[:10], end=str(today)[:10], term='Long term')
    r3k_value = r3k_data['Model Value'].iloc[0] + r3k_data['Absolute Gap'].iloc[0]
    expiry = (today.replace(day=1) + timedelta(days=32)).replace(day=1) + timedelta(days=14)
    strike_price = round(r3k_value / 5) * 5
    ticker = f"IWV{expiry.strftime('%y%m%d')}P{str(int(strike_price * 1000)).zfill(8)}"
    return ticker


def save_hedging_transaction(transaction):
    df = pd.DataFrame([transaction])
    try:
        df.to_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/hedging_transcations.csv',
                  mode='a', header=False, index=False)
    except Exception as e:
        print(f"Error saving transaction: {e}")


def execute_delta_hedge():
    total_delta = calculate_total_delta()
    hedge_ticker = return_r3k_option_ticker()

    # change this to be the optimum hedging ratio, instead of just 100%,
    # this also assumes that each contract controls 100 shares, this needs to be confirmed before it is run live
    hedge_qty = abs(total_delta) // 100

    if hedge_qty > 1000:
        if total_delta > 0:
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL

        market_order_data = MarketOrderRequest(
            symbol=hedge_ticker,
            qty=hedge_qty,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY)

        account_request = trading_client.submit_order(market_order_data)
        print(f"Executed {side} order for {hedge_qty} contracts of {hedge_ticker}.")
        transaction = {
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "ticker": hedge_ticker,
            "side": side.value,
            "quantity": hedge_qty,
            "delta": total_delta
        }
        save_hedging_transaction(transaction)
    else:
        print("No significant delta to hedge.")


def qi_vol_indicator(date):
    date = str(date).split()[0]
    indexes = ['S&P500', 'Euro Stoxx 600', 'USDJPY', 'EURUSD', 'USD 10Y Swap', 'EUR 10Y Swap']
    st_rsqs = [row['Rsq'] for model in indexes
               for _, row in Qi_wrapper.get_model_data(model=model, start=date, end=date, term='Short term').iterrows()]
    return 100 - np.mean(st_rsqs)


def current_max_leverage(date):
    current_volatility = qi_vol_indicator(date)
    base_leverage = 1
    average_volatility = 29.44  # this is the average qi_vol the last 5 years for the PITC of the R3K, so i am using it as a benchmark to compare current vol to
    adjustment_factor = max(0.5, 1 - 0.65 * (current_volatility - average_volatility) / average_volatility) # this puts a lower bound of 0.5 of leverage
    adjusted_max_leverage = base_leverage * adjustment_factor
    return min(2, adjusted_max_leverage)  # this places an upper bound of 2 on leverage of the account


def current_max_leverage_moving_average(date, volatility_window=5):
    current_volatility = qi_vol_indicator(date)
    base_leverage = 1
    average_volatility = 29.44
    date = datetime.strptime(date, '%Y-%m-%d')
    # Calculate smoothed volatility using a moving average
    smoothed_volatility = np.mean([qi_vol_indicator(date - timedelta(days=i)) for i in range(volatility_window)])
    adjustment_factor = max(0.5, 1 - (average_volatility - smoothed_volatility) / smoothed_volatility)
    adjusted_max_leverage = base_leverage * adjustment_factor
    return min(2, adjusted_max_leverage)


def main_long_vol_adjusted():
    curr_equity = account.equity
    print(f'Account Value is at {curr_equity}')
    df_account_tracker = pd.read_csv(
        "/Users/FreddieLewin/PycharmProjects/new_dl_token/algo_new/account_value.csv")

    now_date = datetime.now()
    formatted_date = now_date.strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame({'datetime': [formatted_date], 'run': [len(df_account_tracker) + 1], 'account_value': [curr_equity]})
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
    time.sleep(5)
    # wait 5 seconds to ensure orders have gone through ensuring hedging calculations are done correctly
    # res = execute_delta_hed

